"""
Dataset loading and processing utilities with lazy image loading.
Images are loaded on-demand in __getitem__ to avoid slow upfront loading.
"""
import os
import logging
from PIL import Image
from datasets import load_dataset
from config import DATA_DIR, SYSTEM_PROMPT
from torch.utils.data import Dataset as TorchDataset

logger = logging.getLogger(__name__)


class ImageDataset(TorchDataset):
    """
    Lazy-loading PyTorch Dataset that loads images on-demand.
    Only stores paths/metadata during init, loads images in __getitem__.
    
    This is much faster for large datasets since we skip upfront image loading.
    """
    def __init__(self, examples, processor):
        """
        Args:
            examples: List of dicts with paths and metadata (no PIL images)
            processor: AutoProcessor for applying chat template
        """
        self.examples = examples
        self.processor = processor
        # Column names for compatibility with HuggingFace Dataset interface
        self.column_names = ["prompt", "image", "relevance_labels", "candidate_images", "candidate_texts"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Load images on-demand and build the full example."""
        example = self.examples[idx]
        
        # Load query image on-demand
        query_image_path = example['query_image_path']
        try:
            image = Image.open(query_image_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load query image {query_image_path}: {e}")
            # Return a placeholder - the reward function will handle errors
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Load candidate images on-demand
        candidate_images = []
        for img_path in example['candidate_image_paths']:
            try:
                candidate_images.append(Image.open(img_path).convert('RGB'))
            except Exception as e:
                logger.warning(f"Failed to load candidate image {img_path}: {e}")
                candidate_images.append(Image.new('RGB', (224, 224), color='gray'))
        
        return {
            "prompt": example["prompt"],  # Pre-computed prompt
            "image": image,  # Lazily loaded PIL Image
            "relevance_labels": example["relevance_labels"],
            "candidate_images": candidate_images,  # Lazily loaded PIL Images
            "candidate_texts": example["candidate_texts"],
        }


def load_datasets():
    """Load train and validation datasets from parquet files."""
    logger.info("Loading dataset from parquet files...")

    train_dataset = load_dataset(
        'parquet',
        data_files=os.path.join(DATA_DIR, 'train.parquet'),
        split='train'
    )
    val_dataset = load_dataset(
        'parquet',
        data_files=os.path.join(DATA_DIR, 'val.parquet'),
        split='train'
    )

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    logger.info(f"Dataset columns: {train_dataset.column_names}")

    return train_dataset, val_dataset


def precompute_prompt(example, processor):
    """
    Pre-compute the prompt for an example (without loading images).
    
    Args:
        example: Dataset example with paths and metadata
        processor: AutoProcessor for the model
        
    Returns:
        Dict with prompt, paths, and metadata (no PIL images)
    """
    # Build user query
    user_query = f"User's question: {example['question']}"
    
    # Build conversation structure (image placeholder, not actual image)
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},  # Placeholder - actual image loaded in __getitem__
                {"type": "text", "text": user_query},
            ],
        },
    ]
    
    # Apply chat template to get the prompt string
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    
    return {
        "prompt": prompt,  # Pre-computed prompt string
        "query_image_path": example['query_image_path'],  # Keep path for lazy loading
        "candidate_image_paths": example['candidate_image_paths'],  # Keep paths
        "candidate_texts": example['candidate_texts'],
        "relevance_labels": example["relevance_labels"],
    }


def validate_image_path(path):
    """Check if image path exists and is readable."""
    if not isinstance(path, str):
        return False
    return os.path.exists(path)


def process_datasets(train_dataset, val_dataset, processor):
    """
    Process datasets by pre-computing prompts (lazy image loading).
    
    This is FAST because we only compute prompt strings, not load images.
    Images are loaded on-demand in ImageDataset.__getitem__().
    """
    logger.info("Processing datasets (pre-computing prompts, lazy image loading)...")

    # Process train dataset - just compute prompts, don't load images
    logger.info(f"Pre-computing prompts for train dataset ({len(train_dataset)} examples)...")
    train_processed = []
    skipped_train = 0
    
    for i, example in enumerate(train_dataset):
        if i % 100 == 0:
            logger.info(f"  Processed {i}/{len(train_dataset)} train examples...")
        
        # Validate that image paths exist
        if not validate_image_path(example.get('query_image_path')):
            skipped_train += 1
            continue
            
        # Check candidate paths (just validate, don't load)
        candidate_paths = example.get('candidate_image_paths', [])
        if not candidate_paths or not all(validate_image_path(p) for p in candidate_paths):
            skipped_train += 1
            continue
        
        # Pre-compute prompt (no image loading!)
        processed = precompute_prompt(example, processor)
        train_processed.append(processed)
    
    logger.info(f"  Pre-computed {len(train_processed)}/{len(train_dataset)} train prompts! "
                f"(Skipped {skipped_train} due to missing image paths)")
    # Process val dataset
    logger.info(f"Pre-computing prompts for val dataset ({len(val_dataset)} examples)...")
    val_processed = []
    skipped_val = 0
    
    for i, example in enumerate(val_dataset):
        if i % 50 == 0:
            logger.info(f"  Processed {i}/{len(val_dataset)} val examples...")
        
        if not validate_image_path(example.get('query_image_path')):
            skipped_val += 1
            continue
            
        candidate_paths = example.get('candidate_image_paths', [])
        if not candidate_paths or not all(validate_image_path(p) for p in candidate_paths):
            skipped_val += 1
            continue
        
        processed = precompute_prompt(example, processor)
        val_processed.append(processed)
    
    logger.info(f"  Pre-computed {len(val_processed)}/{len(val_dataset)} val prompts! "
                f"(Skipped {skipped_val} due to missing image paths)")

    # Create lazy-loading dataset wrappers
    logger.info("Creating lazy-loading dataset wrappers...")
    train_dataset = ImageDataset(train_processed, processor)
    val_dataset = ImageDataset(val_processed, processor)

    logger.info(f"Dataset columns: {train_dataset.column_names}")
    logger.info("Images will be loaded on-demand during training!")

    return train_dataset, val_dataset
