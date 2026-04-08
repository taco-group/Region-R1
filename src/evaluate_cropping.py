"""
Evaluation script to measure whether VLM cropping improves MRR.

This script:
1. Loads different model checkpoints (or just the final model)
2. For each validation example:
   - Generates cropping bbox using the model
   - Calculates MRR with original image (baseline)
   - Calculates MRR with cropped image (using model's bbox)
   - Compares the improvement
3. Reports overall statistics and optionally logs to wandb
"""
import os
import argparse
import logging
import torch
import wandb
from tqdm import tqdm
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image, ImageFile
# Enable loading of truncated images to handle slightly corrupted files
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import (
    BASE_DIR, DATA_DIR, SYSTEM_PROMPT, INFERENCE_CONFIG,
    FINAL_MODEL_DIR, MODEL_OUTPUT_DIR, MODEL_ID, EVAL_OUTPUT_DIR, EVACLIP_MODEL_ID,
    FEATURE_CACHE_PATH
)
from utils import (
    parse_bbox_from_completion, 
    crop_image_with_bbox, 
    calculate_mrr, 
    calculate_ndcg,
    set_clip_model,
    calculate_clip_scores,
    try_load_image,
    batch_encode_candidates,
    score_candidates,
    load_feature_cache,
    save_feature_cache
)
from model import load_evaclip_model
from peft import PeftModel

logger = logging.getLogger(__name__)


class CroppingEvaluator:
    """Evaluates whether cropping improves retrieval MRR."""

    def __init__(self, model_path, use_wandb=False, run_name=None, load_as_lora=False, base_model_path=None, dataset_path=None, split='val', viz_dir=None):
        """
        Initialize the evaluator.

        Args:
            model_path: Path to the model checkpoint to evaluate
            use_wandb: Whether to log results to wandb
            run_name: Custom run name for wandb
            load_as_lora: If True, load model_path as LoRA adapter on top of base_model_path
            base_model_path: Path to base model (only used if load_as_lora=True)
            dataset_path: Path to the validation dataset (parquet file)
            viz_dir: Directory to save improvement visualizations
        """
        self.model_path = model_path
        self.use_wandb = use_wandb
        self.load_as_lora = load_as_lora
        self.base_model_path = base_model_path or MODEL_ID
        self.dataset_path = dataset_path
        self.split = split
        self.viz_dir = viz_dir or os.path.join(BASE_DIR, 'improvement_viz')

        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="justifiable-cropping-eval",
                name=run_name or f"eval-{os.path.basename(model_path)}",
                tags=["evaluation", "mrr", "cropping", "eva-clip"],
                config={
                    "model_path": model_path,
                    "clip_model": EVACLIP_MODEL_ID,  # Using EVA-CLIP
                }
            )

        print(f"\n{'='*80}")
        print(f"Initializing Cropping Evaluator")
        print(f"Model: {model_path}")
        print(f"{'='*80}\n")

        # Load models
        self.vlm_model, self.vlm_processor = self._load_vlm_model()
        self.clip_model, self.clip_processor = self._load_clip_model()

        # Load validation dataset
        self.val_dataset = self._load_validation_data()

        # Load persistent cache
        load_feature_cache(FEATURE_CACHE_PATH)

        # Statistics
        self.results = []

    def _load_vlm_model(self):
        """Load the VLM model and processor."""
        print("Loading VLM model...")

        if self.load_as_lora:
            # Load base model first
            print(f"  Loading base model from {self.base_model_path}...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.base_model_path,
                torch_dtype="auto",
                device_map="auto",
            )

            # Load LoRA adapter
            print(f"  Loading LoRA adapter from {self.model_path}...")
            model = PeftModel.from_pretrained(model, self.model_path)

            # Merge LoRA weights for faster inference
            print("  Merging LoRA weights...")
            model = model.merge_and_unload()

            processor = AutoProcessor.from_pretrained(
                self.model_path,
                use_fast=True,
                padding_side="left"
            )
            print(f"✓ VLM model loaded: base ({self.base_model_path}) + LoRA ({self.model_path})")
        else:
            # Load full model directly
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                device_map="auto",
            )
            processor = AutoProcessor.from_pretrained(
                self.model_path,
                use_fast=True,
                padding_side="left"
            )
            print(f"✓ VLM model loaded from {self.model_path}")

        return model, processor

    def _load_clip_model(self):
        """Load EVA-CLIP model for ranking."""
        print("Loading EVA-CLIP model for ranking...")
        # Use EVA-CLIP instead of regular CLIP for stronger performance
        evaclip_model, evaclip_processor = load_evaclip_model()
        print("✓ EVA-CLIP model loaded")
        
        # Set global model in utils
        set_clip_model(evaclip_model, evaclip_processor)
        
        return evaclip_model, evaclip_processor

    def _load_validation_data(self):
        """Load validation or test dataset."""
        print(f"Loading {self.split} dataset...")
        
        # Use provided dataset path or default based on split
        if self.dataset_path:
            dataset_path = self.dataset_path
        else:
            dataset_path = os.path.join(DATA_DIR, f'{self.split}.parquet')
        print(f"  Dataset path: {dataset_path}")
        
        dataset = load_dataset(
            'parquet',
            data_files=dataset_path,
            split='train'
        )
        print(f"✓ Loaded {len(dataset)} {self.split} examples")
        return dataset

    def generate_bbox(self, query_image, query_text=""):
        """
        Generate bounding box for cropping using the VLM model.

        Args:
            query_image: PIL Image (query image)
            query_text: Optional query text

        Returns:
            Tuple of (bbox, generated_text) where bbox is [x1, y1, x2, y2] or None
        """
        # Create conversation
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": query_image},
                    {"type": "text", "text": query_text},
                ],
            },
        ]

        # Apply chat template
        prompt = self.vlm_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )

        # Process images
        image_inputs, video_inputs = process_vision_info(conversation)

        # Prepare inputs
        inputs = self.vlm_processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.vlm_model.device)

        # Generate
        with torch.no_grad():
            output_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=INFERENCE_CONFIG["max_new_tokens"]
            )

        # Decode
        input_length = inputs['input_ids'].shape[1]
        generated_text = self.vlm_processor.decode(output_ids[0, input_length:], skip_special_tokens=True)

        # Parse bbox
        bbox = parse_bbox_from_completion(generated_text)

        return bbox, generated_text

    def evaluate_example(self, example):
        """
        Evaluate a single example.

        Args:
            example: Dataset example with image, candidate_images, relevance_labels

        Returns:
            Dict with evaluation results
        """
        # Load images
        query_image = example['query_image_path']
        if isinstance(query_image, str):
            query_image = Image.open(query_image).convert('RGB')

        candidate_images = example['candidate_image_paths']
        if candidate_images and isinstance(candidate_images[0], str):
            candidate_images = [
                Image.open(img_path).convert('RGB')
                for img_path in candidate_images
            ]

        relevance_labels = example['relevance_labels']

        # Get query text if available
        query_text = f"User's question: {example['question']}"

        # Step 1: Generate bbox using VLM
        bbox, generated_text = self.generate_bbox(query_image, query_text)

        # Step 2: Calculate MRR with original image (baseline)
        original_scores = self.calculate_clip_ranking(query_image, candidate_images, candidate_texts)
        original_ranking = sorted(
            range(len(original_scores)),
            key=lambda x: original_scores[x],
            reverse=True
        )
        original_mrr = calculate_mrr(original_ranking, relevance_labels)
        original_ndcg = calculate_ndcg(original_ranking, relevance_labels)

        # Step 3: Calculate MRR with cropped image (if bbox available)
        if bbox is not None:
            cropped_image = crop_image_with_bbox(query_image, bbox, resize=True)
            cropped_scores = self.calculate_clip_ranking(cropped_image, candidate_images, candidate_texts)
            cropped_ranking = sorted(
                range(len(cropped_scores)),
                key=lambda x: cropped_scores[x],
                reverse=True
            )
            cropped_mrr = calculate_mrr(cropped_ranking, relevance_labels)
            cropped_ndcg = calculate_ndcg(cropped_ranking, relevance_labels)
            has_bbox = True
        else:
            # If no bbox, use original image results
            cropped_mrr = original_mrr
            cropped_ndcg = original_ndcg
            has_bbox = False

        # Calculate improvement
        mrr_improvement = cropped_mrr - original_mrr
        ndcg_improvement = cropped_ndcg - original_ndcg

        return {
            'has_bbox': has_bbox,
            'bbox': bbox,
            'original_mrr': original_mrr,
            'cropped_mrr': cropped_mrr,
            'mrr_improvement': mrr_improvement,
            'original_ndcg': original_ndcg,
            'cropped_ndcg': cropped_ndcg,
            'ndcg_improvement': ndcg_improvement,
            'relevance_labels': relevance_labels,
            'generated_text': generated_text,
        }

    def generate_bboxes_batch(self, query_images, query_texts):
        """
        Generate bounding boxes for a batch of images.
        """
        conversations = []
        for img, text in zip(query_images, query_texts):
            conversations.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": text},
                ]},
            ])

        texts = [
            self.vlm_processor.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
            for c in conversations
        ]
        
        image_inputs, video_inputs = process_vision_info(conversations)
        
        inputs = self.vlm_processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.vlm_model.device)

        with torch.no_grad():
            output_ids = self.vlm_model.generate(
                **inputs,
                max_new_tokens=INFERENCE_CONFIG["max_new_tokens"]
            )

        input_lengths = inputs.input_ids.shape[1]
        generated_texts = self.vlm_processor.batch_decode(
            output_ids[:, input_lengths:], 
            skip_special_tokens=True
        )

        bboxes = [parse_bbox_from_completion(text) for text in generated_texts]
        return bboxes, generated_texts

    def evaluate_batch(self, batch_examples):
        """Evaluate a batch of examples."""
        query_images = []
        query_texts = []
        
        # Pre-load images to avoid opening multiple times
        valid_indices = []
        for idx, ex in enumerate(batch_examples):
            q_img_path = ex['query_image_path']
            # If it's a string path, try to load it
            if isinstance(q_img_path, str):
                q_img = try_load_image(q_img_path)
            else:
                # Assume it's already a PIL image
                q_img = q_img_path
            
            if q_img is None:
                logger.warning(f"Skipping example {idx}: Failed to load query image {q_img_path}")
                continue
                
            query_images.append(q_img)
            query_texts.append(f"User's question: {ex['question']}")
            valid_indices.append(idx)
        
        # Filter batch_examples to match valid indices
        batch_examples = [batch_examples[i] for i in valid_indices]
        
        if not batch_examples:
            return []

        # 1. Batch Generate
        bboxes, generated_texts = self.generate_bboxes_batch(query_images, query_texts)
        
        batch_results = []
        
        for i, example in enumerate(batch_examples):
            # Load candidates
            # Optimization: Just use paths directly! batch_encode_candidates handles caching/loading
            candidate_images = example['candidate_image_paths']
            
            relevance_labels = example['relevance_labels']
            candidate_texts = example.get('candidate_texts', [])
            query_image = query_images[i]
            bbox = bboxes[i]
            
            # 2. Encode Candidates ONCE (with strict checking)
            candidate_feats = batch_encode_candidates(
                candidate_images, 
                candidate_texts, 
                self.clip_model, 
                self.clip_processor
            )
            
            if candidate_feats is None:
                print(f"Skipping example {i}: One or more candidate images failed to load.")
                continue
            
            # 3. Score Original (Baseline)
            # We need query features. Let's just encode query using batch_encode too (list of 1)
            query_feat_orig = batch_encode_candidates([query_image], None, self.clip_model, self.clip_processor)
            
            if query_feat_orig is None:
                # Should detect this earlier properly, but query is usually safe if pre-checked
                continue

            original_scores = score_candidates(query_feat_orig, candidate_feats)
            
            original_ranking = sorted(range(len(original_scores)), key=lambda x: original_scores[x], reverse=True)
            original_mrr = calculate_mrr(original_ranking, relevance_labels)
            original_ndcg = calculate_ndcg(original_ranking, relevance_labels)
            
            # Find first positive rank for original
            original_first_pos_rank = None
            for rank, idx in enumerate(original_ranking, 1):
                if relevance_labels[idx] > 0:
                    original_first_pos_rank = rank
                    break
            
            # 4. Score Cropped
            if bbox is not None:
                try:
                    cropped_image = crop_image_with_bbox(query_image, bbox, resize=True)
                except Exception as e:
                    print(f"    Warning: crop_image_with_bbox failed: {e}, skipping this sample")
                    continue  # Skip entire sample
                
                if cropped_image is None:
                    print(f"    Warning: cropped_image is None, skipping this sample")
                    continue  # Skip entire sample
                
                # Encode Cropped Query
                query_feat_crop = batch_encode_candidates([cropped_image], None, self.clip_model, self.clip_processor)
                if query_feat_crop is None:
                    print(f"    Warning: Failed to encode cropped image, skipping this sample")
                    continue  # Skip entire sample
                
                cropped_scores = score_candidates(query_feat_crop, candidate_feats)
                cropped_ranking = sorted(range(len(cropped_scores)), key=lambda x: cropped_scores[x], reverse=True)
                cropped_mrr = calculate_mrr(cropped_ranking, relevance_labels)
                cropped_ndcg = calculate_ndcg(cropped_ranking, relevance_labels)
                has_bbox = True
                
                # Find first positive rank for cropped
                cropped_first_pos_rank = None
                for rank, idx in enumerate(cropped_ranking, 1):
                    if relevance_labels[idx] > 0:
                        cropped_first_pos_rank = rank
                        break
            else:
                cropped_mrr = original_mrr
                cropped_ndcg = original_ndcg
                has_bbox = False
                cropped_first_pos_rank = original_first_pos_rank
                cropped_ranking = original_ranking
                
            mrr_imp = cropped_mrr - original_mrr
            ndcg_imp = cropped_ndcg - original_ndcg
            
            # Extract query info
            query_id = example.get('data_id', example.get('query_id', ''))
            question = example.get('question', '')
            
            batch_results.append({
                'has_bbox': has_bbox,
                'bbox': bbox,
                'query_id': query_id,
                'question': question,
                'num_candidates': len(candidate_images),
                'original_rank': original_first_pos_rank,
                'cropped_rank': cropped_first_pos_rank,
                'rank_improvement': (original_first_pos_rank - cropped_first_pos_rank) if (original_first_pos_rank and cropped_first_pos_rank) else 0,
                'original_mrr': original_mrr,
                'cropped_mrr': cropped_mrr,
                'mrr_improvement': mrr_imp,
                'original_ndcg': original_ndcg,
                'cropped_ndcg': cropped_ndcg,
                'ndcg_improvement': ndcg_imp,
                'relevance_labels': relevance_labels,
                'generated_text': generated_texts[i],
                'query_image': query_image,  # Keep for visualization
                'candidate_image_paths': candidate_images,  # Keep for visualization
            })
            
            # Visualize significant improvements (rank 5+ -> rank 1)
            if (original_first_pos_rank is not None and original_first_pos_rank >= 5 
                and cropped_first_pos_rank is not None and cropped_first_pos_rank == 1
                and has_bbox):
                self.visualize_improvement(
                    query_image=query_image,
                    bbox=bbox,
                    candidate_image_paths=candidate_images,
                    relevance_labels=relevance_labels,
                    original_ranking=original_ranking,
                    cropped_ranking=cropped_ranking,
                    original_rank=original_first_pos_rank,
                    cropped_rank=cropped_first_pos_rank,
                    query_id=query_id,
                    question=question,
                    viz_dir=self.viz_dir
                )
            
        return batch_results

    def visualize_improvement(self, query_image, bbox, candidate_image_paths, 
                               relevance_labels, original_ranking, cropped_ranking,
                               original_rank, cropped_rank, query_id, question, 
                               viz_dir, top_k=5):
        """
        Visualize a significant improvement case (rank 5+ -> rank 1).
        
        Creates a 2-row figure:
        - Top row: Original query image + top-5 candidates in ORIGINAL ranking order
        - Bottom row: Cropped query image + top-5 candidates in CROPPED ranking order
        
        Args:
            viz_dir: Directory to save visualization images
        """
        os.makedirs(viz_dir, exist_ok=True)
        
        # Crop the image
        cropped_image = crop_image_with_bbox(query_image, bbox, resize=True)
        
        # Create figure: 2 rows, (1 query + top_k candidates) columns
        fig, axes = plt.subplots(2, top_k + 1, figsize=(3 * (top_k + 1), 8))
        
        # ===== ROW 1: ORIGINAL QUERY + ORIGINAL RANKING =====
        # Show original query with row label
        axes[0, 0].imshow(query_image)
        axes[0, 0].set_title(f"BEFORE CROPPING\nOriginal Query", 
                            fontsize=10, fontweight='bold', color='red')
        axes[0, 0].axis('off')
        rect = patches.Rectangle((0, 0), 1, 1, transform=axes[0, 0].transAxes,
                                   linewidth=4, edgecolor='red', facecolor='none')
        axes[0, 0].add_patch(rect)
        
        # Show top-K candidates in ORIGINAL ranking order
        for i in range(min(top_k, len(original_ranking))):
            ax = axes[0, i + 1]
            cand_idx = original_ranking[i]  # Get candidate index from ranking
            cand_path = candidate_image_paths[cand_idx]
            is_correct = relevance_labels[cand_idx] > 0
            
            try:
                cand_img = try_load_image(cand_path)
                if cand_img:
                    ax.imshow(cand_img)
                else:
                    ax.text(0.5, 0.5, "Load Error", ha='center', va='center', fontsize=8)
            except:
                ax.text(0.5, 0.5, "Error", ha='center', va='center', fontsize=8)
            
            if is_correct:
                # Positive candidate - make it very visible
                ax.set_title(f"★ POSITIVE ★\nRank {i+1}", fontsize=10, color='green', fontweight='bold')
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                           linewidth=5, edgecolor='green', facecolor='none')
            else:
                ax.set_title(f"Rank {i+1}", fontsize=9, color='gray')
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                           linewidth=1, edgecolor='gray', facecolor='none')
            ax.axis('off')
            ax.add_patch(rect)
        
        # If positive is beyond top-k in original, add note
        if original_rank > top_k:
            # Add text annotation on the last subplot area
            fig.text(0.92, 0.7, f"Positive at\nRank {original_rank}\n(not in top {top_k})", 
                    fontsize=9, color='orange', ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        # ===== ROW 2: CROPPED QUERY + CROPPED RANKING =====
        # Show cropped query with row label
        axes[1, 0].imshow(cropped_image)
        axes[1, 0].set_title(f"AFTER CROPPING\nCropped Query", 
                            fontsize=10, fontweight='bold', color='green')
        axes[1, 0].axis('off')
        rect = patches.Rectangle((0, 0), 1, 1, transform=axes[1, 0].transAxes,
                                   linewidth=4, edgecolor='green', facecolor='none')
        axes[1, 0].add_patch(rect)
        
        # Show top-K candidates in CROPPED ranking order
        for i in range(min(top_k, len(cropped_ranking))):
            ax = axes[1, i + 1]
            cand_idx = cropped_ranking[i]  # Get candidate index from ranking
            cand_path = candidate_image_paths[cand_idx]
            is_correct = relevance_labels[cand_idx] > 0
            
            try:
                cand_img = try_load_image(cand_path)
                if cand_img:
                    ax.imshow(cand_img)
                else:
                    ax.text(0.5, 0.5, "Load Error", ha='center', va='center', fontsize=8)
            except:
                ax.text(0.5, 0.5, "Error", ha='center', va='center', fontsize=8)
            
            if is_correct:
                # Positive candidate - make it very visible
                ax.set_title(f"★ POSITIVE ★\nRank {i+1}", fontsize=10, color='green', fontweight='bold')
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                           linewidth=5, edgecolor='green', facecolor='none')
            else:
                ax.set_title(f"Rank {i+1}", fontsize=9, color='gray')
                rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                           linewidth=1, edgecolor='gray', facecolor='none')
            ax.axis('off')
            ax.add_patch(rect)
        
        # Overall title
        q_short = question[:45] + "..." if len(question) > 45 else question
        fig.suptitle(f"Rank Improvement: {original_rank} → {cropped_rank}\n{q_short}", 
                     fontsize=12, fontweight='bold', color='darkgreen', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # Save
        safe_id = str(query_id).replace('/', '_').replace('\\', '_')[:50]
        output_path = os.path.join(viz_dir, f"improvement_{safe_id}_{original_rank}to{cropped_rank}.png")
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"    ✓ Visualized: Rank {original_rank} → {cropped_rank} saved to {output_path}")

    def evaluate_all(self, num_samples=None, batch_size=2, output_path=None, save_every=200, resume=True):
        """
        Evaluate all validation examples.

        Args:
            num_samples: Number of samples to evaluate (None for all)
            batch_size: Batch size for VLM inference
            output_path: Path to save results CSV (enables checkpointing)
            save_every: Save checkpoint every N samples
            resume: If True and output_path exists, resume from where we left off

        Returns:
            Dict with aggregated statistics
        """
        print(f"\n{'='*80}")
        print(f"Starting Evaluation (Batch Size: {batch_size})")
        print(f"{'='*80}\n")

        # Determine number of samples
        total_samples = len(self.val_dataset) if num_samples is None else min(num_samples, len(self.val_dataset))
        
        # Load existing results if resuming
        processed_indices = set()
        self.results = []
        
        if resume and output_path and os.path.exists(output_path):
            print(f"Resuming from existing results: {output_path}")
            import csv
            with open(output_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Convert types back
                    result = {
                        'example_idx': int(row['example_idx']),
                        'query_id': row.get('query_id', ''),
                        'question': row.get('question', ''),
                        'has_bbox': row['has_bbox'] == 'True',
                        'bbox': row.get('bbox', ''),
                        'num_candidates': int(row['num_candidates']) if row.get('num_candidates') else 0,
                        'original_rank': int(row['original_rank']) if row.get('original_rank') else None,
                        'cropped_rank': int(row['cropped_rank']) if row.get('cropped_rank') else None,
                        'rank_improvement': int(row.get('rank_improvement', 0)) if row.get('rank_improvement') else 0,
                        'original_mrr': float(row['original_mrr']),
                        'cropped_mrr': float(row['cropped_mrr']),
                        'mrr_improvement': float(row['mrr_improvement']),
                        'original_ndcg': float(row['original_ndcg']),
                        'cropped_ndcg': float(row['cropped_ndcg']),
                        'ndcg_improvement': float(row.get('ndcg_improvement', 0)),
                        'generated_text': row.get('generated_text', ''),
                    }
                    self.results.append(result)
                    processed_indices.add(result['example_idx'])
            print(f"  Loaded {len(self.results)} existing results, {len(processed_indices)} indices processed")
        
        # Create list of indices to process (skip already processed)
        indices = [i for i in range(total_samples) if i not in processed_indices]
        if not indices:
            print("All samples already processed!")
            return self.calculate_statistics()
        
        print(f"Processing {len(indices)} remaining samples (out of {total_samples} total)")
        
        # Create batches
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        last_logged = (len(self.results) // save_every) * save_every
        last_saved = len(self.results)
        
        with tqdm(total=len(indices), desc="Evaluating") as pbar:
            for batch_indices in batches:
                batch_examples = [self.val_dataset[i] for i in batch_indices]
                batch_results = self.evaluate_batch(batch_examples)
                
                for idx, result in zip(batch_indices, batch_results):
                    result['example_idx'] = idx
                    self.results.append(result)

                    if self.use_wandb:
                        wandb.log({
                            'example_idx': idx,
                            'original_mrr': result['original_mrr'],
                            'cropped_mrr': result['cropped_mrr'],
                            'mrr_improvement': result['mrr_improvement'],
                            'has_bbox': 1 if result['has_bbox'] else 0,
                        })
                
                # Print progress every save_every samples
                current_milestone = (len(self.results) // save_every) * save_every
                if current_milestone > last_logged and len(self.results) > 0:
                    last_logged = current_milestone
                    running_orig_mrr = sum(r['original_mrr'] for r in self.results) / len(self.results)
                    running_crop_mrr = sum(r['cropped_mrr'] for r in self.results) / len(self.results)
                    running_imp = running_crop_mrr - running_orig_mrr
                    num_improved = sum(1 for r in self.results if r['mrr_improvement'] > 0)
                    print(f"\n  [{len(self.results)} examples] Orig MRR: {running_orig_mrr:.4f}, Cropped MRR: {running_crop_mrr:.4f}, Improvement: {running_imp:+.4f}, Improved: {num_improved}")
                    
                    # Save checkpoint
                    if output_path and len(self.results) - last_saved >= save_every:
                        self.save_detailed_results(output_path)
                        save_feature_cache(FEATURE_CACHE_PATH)
                        last_saved = len(self.results)
                        print(f"  Checkpoint saved to {output_path}")
                        print(f"  Feature cache saved to {FEATURE_CACHE_PATH}")
                
                pbar.update(len(batch_indices))

        # Final save
        if output_path:
            self.save_detailed_results(output_path)
            print(f"Final results saved to {output_path}")
        
        # Save feature cache
        save_feature_cache(FEATURE_CACHE_PATH)
        print(f"Feature cache saved to {FEATURE_CACHE_PATH}")

        # Calculate statistics
        stats = self.calculate_statistics()

        # Log summary to wandb
        if self.use_wandb:
            wandb.log({
                'summary/avg_original_mrr': stats['avg_original_mrr'],
                'summary/avg_cropped_mrr': stats['avg_cropped_mrr'],
                'summary/avg_mrr_improvement': stats['avg_mrr_improvement'],
                'summary/bbox_generation_rate': stats['bbox_generation_rate'],
                'summary/num_improved': stats['num_improved'],
            })

        return stats

    def calculate_statistics(self):
        """Calculate aggregated statistics from results."""
        if not self.results:
            return {}

        # Filter results with bbox
        results_with_bbox = [r for r in self.results if r['has_bbox']]

        # Overall statistics
        stats = {
            'total_examples': len(self.results),
            'num_with_bbox': len(results_with_bbox),
            'bbox_generation_rate': len(results_with_bbox) / len(self.results),

            # MRR statistics
            'avg_original_mrr': sum(r['original_mrr'] for r in self.results) / len(self.results),
            'avg_cropped_mrr': sum(r['cropped_mrr'] for r in self.results) / len(self.results),
            'avg_mrr_improvement': sum(r['mrr_improvement'] for r in self.results) / len(self.results),

            # NDCG statistics
            'avg_original_ndcg': sum(r['original_ndcg'] for r in self.results) / len(self.results),
            'avg_cropped_ndcg': sum(r['cropped_ndcg'] for r in self.results) / len(self.results),
            'avg_ndcg_improvement': sum(r['ndcg_improvement'] for r in self.results) / len(self.results),

            # Improvement breakdown
            'num_improved': sum(1 for r in self.results if r['mrr_improvement'] > 0.01),
            'num_degraded': sum(1 for r in self.results if r['mrr_improvement'] < -0.01),
            'num_unchanged': sum(1 for r in self.results if abs(r['mrr_improvement']) <= 0.01),
        }

        # Statistics for examples with bbox only
        if results_with_bbox:
            stats['avg_mrr_improvement_with_bbox'] = sum(
                r['mrr_improvement'] for r in results_with_bbox
            ) / len(results_with_bbox)
            stats['avg_ndcg_improvement_with_bbox'] = sum(
                r['ndcg_improvement'] for r in results_with_bbox
            ) / len(results_with_bbox)

        return stats

    def print_summary(self, stats):
        """Print evaluation summary."""
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total examples evaluated: {stats['total_examples']}")
        print(f"Examples with bbox generated: {stats['num_with_bbox']} ({stats['bbox_generation_rate']*100:.1f}%)")
        print()

        print("MRR Results:")
        print(f"  Original MRR (baseline):     {stats['avg_original_mrr']:.4f}")
        print(f"  Cropped MRR:                 {stats['avg_cropped_mrr']:.4f}")
        print(f"  Average improvement:         {stats['avg_mrr_improvement']:+.4f}")
        if 'avg_mrr_improvement_with_bbox' in stats:
            print(f"  Improvement (bbox only):     {stats['avg_mrr_improvement_with_bbox']:+.4f}")
        print()

        print("NDCG Results:")
        print(f"  Original NDCG (baseline):    {stats['avg_original_ndcg']:.4f}")
        print(f"  Cropped NDCG:                {stats['avg_cropped_ndcg']:.4f}")
        print(f"  Average improvement:         {stats['avg_ndcg_improvement']:+.4f}")
        if 'avg_ndcg_improvement_with_bbox' in stats:
            print(f"  Improvement (bbox only):     {stats['avg_ndcg_improvement_with_bbox']:+.4f}")
        print()

        print("Improvement Breakdown:")
        print(f"  Improved (MRR +0.01+):       {stats['num_improved']} ({stats['num_improved']/stats['total_examples']*100:.1f}%)")
        print(f"  Degraded (MRR -0.01-):       {stats['num_degraded']} ({stats['num_degraded']/stats['total_examples']*100:.1f}%)")
        print(f"  Unchanged:                   {stats['num_unchanged']} ({stats['num_unchanged']/stats['total_examples']*100:.1f}%)")

        print(f"\n{'='*80}\n")

    def save_detailed_results(self, output_path):
        """Save detailed results to a CSV file."""
        import pandas as pd

        df = pd.DataFrame([
            {
                'example_idx': r['example_idx'],
                'query_id': r.get('query_id', ''),
                'question': r.get('question', ''),
                'has_bbox': r['has_bbox'],
                'bbox': str(r['bbox']),
                'num_candidates': r.get('num_candidates', ''),
                'original_rank': r.get('original_rank', ''),
                'cropped_rank': r.get('cropped_rank', ''),
                'rank_improvement': r.get('rank_improvement', 0),
                'original_mrr': r['original_mrr'],
                'cropped_mrr': r['cropped_mrr'],
                'mrr_improvement': r['mrr_improvement'],
                'original_ndcg': r['original_ndcg'],
                'cropped_ndcg': r['cropped_ndcg'],
                'ndcg_improvement': r.get('ndcg_improvement', 0),
                'generated_text': r.get('generated_text', ''),
            }
            for r in self.results
        ])

        df.to_csv(output_path, index=False)
        print(f"Detailed results saved to {output_path}")

    def finish(self):
        """Clean up and finish evaluation."""
        # Save cache
        save_feature_cache(FEATURE_CACHE_PATH)
        
        if self.use_wandb:
            wandb.finish()


def evaluate_checkpoint(checkpoint_path, num_samples=None, use_wandb=False, run_name=None, dataset_path=None, split='val', output_path=None, viz_dir=None):
    """
    Evaluate a single checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of samples to evaluate (None for all)
        use_wandb: Whether to log to wandb
        run_name: Custom run name for wandb
        dataset_path: Optional path to specific dataset file
        split: Dataset split to use ('val' or 'test')
        output_path: Path to save results CSV (enables checkpointing/resume)
        viz_dir: Directory to save improvement visualizations

    Returns:
        Dict with evaluation statistics
    """
    evaluator = CroppingEvaluator(
        model_path=checkpoint_path,
        use_wandb=use_wandb,
        run_name=run_name,
        dataset_path=dataset_path,
        split=split,
        viz_dir=viz_dir
    )

    # Use provided output_path or generate default
    if output_path is None:
        checkpoint_name = os.path.basename(checkpoint_path)
        output_path = os.path.join(BASE_DIR, f"eval_results_{checkpoint_name}_{split}.csv")

    stats = evaluator.evaluate_all(num_samples=num_samples, output_path=output_path)
    evaluator.print_summary(stats)

    evaluator.finish()

    return stats


def compare_checkpoints(checkpoint_paths, num_samples=None, use_wandb=False, dataset_path=None, split='val'):
    """
    Compare multiple checkpoints.

    Args:
        checkpoint_paths: List of paths to checkpoints
        num_samples: Number of samples to evaluate per checkpoint
        use_wandb: Whether to log to wandb
        dataset_path: Optional path to specific dataset file
        split: Dataset split to use ('val' or 'test')
    """
    print(f"\n{'='*80}")
    print("COMPARING CHECKPOINTS")
    print(f"{'='*80}\n")

    all_stats = []

    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path)
        print(f"\n>>> Evaluating {checkpoint_name}...")

        stats = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            num_samples=num_samples,
            use_wandb=use_wandb,
            run_name=f"eval-{checkpoint_name}",
            dataset_path=dataset_path,
            split=split
        )

        stats['checkpoint_name'] = checkpoint_name
        stats['checkpoint_path'] = checkpoint_path
        all_stats.append(stats)

    # Print comparison summary
    print(f"\n{'='*80}")
    print("CHECKPOINT COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Checkpoint':<30} {'Original MRR':<15} {'Cropped MRR':<15} {'Improvement':<15} {'Bbox Rate':<12}")
    print("-" * 90)

    for stats in all_stats:
        print(f"{stats['checkpoint_name']:<30} "
              f"{stats['avg_original_mrr']:<15.4f} "
              f"{stats['avg_cropped_mrr']:<15.4f} "
              f"{stats['avg_mrr_improvement']:<+15.4f} "
              f"{stats['bbox_generation_rate']*100:<12.1f}%")

    print(f"\n{'='*80}\n")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM cropping impact on MRR")
    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to single checkpoint to evaluate'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all available checkpoints'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of validation samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log results to Weights & Biases'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Custom run name for wandb'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to specific parquet dataset to use for evaluation'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='val',
        choices=['val', 'test'],
        help="Dataset split to evaluate ('val' or 'test')"
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV path for results (enables checkpointing/resume)'
    )
    parser.add_argument(
        '--viz_dir',
        type=str,
        default=os.path.join(BASE_DIR, 'improvement_viz'),
        help='Directory to save improvement visualizations'
    )

    args = parser.parse_args()

    if args.compare:
        # Compare all checkpoints
        checkpoint_paths = []

        # Add training checkpoints if they exist
        checkpoint_dirs = [
            os.path.join(MODEL_OUTPUT_DIR, "checkpoint-50"),
            os.path.join(MODEL_OUTPUT_DIR, "checkpoint-80"),
        ]

        for ckpt_dir in checkpoint_dirs:
            if os.path.exists(ckpt_dir):
                checkpoint_paths.append(ckpt_dir)

        # Add final model
        if os.path.exists(FINAL_MODEL_DIR):
            checkpoint_paths.append(FINAL_MODEL_DIR)

        if not checkpoint_paths:
            print("No checkpoints found to compare!")
            return

        compare_checkpoints(
            checkpoint_paths=checkpoint_paths,
            num_samples=args.num_samples,
            use_wandb=args.wandb,
            dataset_path=args.dataset,
            split=args.split
        )

    elif args.checkpoint:
        # Evaluate single checkpoint
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            use_wandb=args.wandb,
            run_name=args.run_name,
            dataset_path=args.dataset,
            split=args.split,
            output_path=args.output,
            viz_dir=args.viz_dir
        )

    else:
        # Default: evaluate final model
        print("No checkpoint specified. Evaluating final model...")
        evaluate_checkpoint(
            checkpoint_path=FINAL_MODEL_DIR,
            num_samples=args.num_samples,
            use_wandb=args.wandb,
            run_name=args.run_name or "eval-final",
            dataset_path=args.dataset,
            split=args.split,
            output_path=args.output,
            viz_dir=args.viz_dir
        )


if __name__ == "__main__":
    main()
