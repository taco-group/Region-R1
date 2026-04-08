"""
Script to create a re-ranking dataset from InfoSeek or E-VQA knowledge base.
Selects entries with at least 2 images, creating query-candidate pairs.
Uses EVA-CLIP + FAISS for hard negative sampling.
"""
import json
import os
import sys
import random
import logging
import argparse
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset

from config import DATA_ROOT, BASE_DIR, DATASET_NAME
from logging_config import setup_logging

logger = logging.getLogger(__name__)

# Add OMGM model path to sys.path (sibling directory to project)
OMGM_MODEL_PATH = BASE_DIR.parent / "OMGM"
if str(OMGM_MODEL_PATH) not in sys.path:
    sys.path.insert(0, str(OMGM_MODEL_PATH))

from model.retriever import ClipRetriever
from infoseek_data.infoseek_dataset import (
    WikipediaKnowledgeBase,
    _load_url_mapping,
    ReRankingQuery,
    CandidateDocument,
    ReRankingInstance
)


def load_evaclip_retriever(
    knowledge_base_path: str,
    faiss_index_path: str,
    device: str = 'cuda:0'
):
    """Load EVA-CLIP retriever with FAISS index."""
    logger.info(f"Loading EVA-CLIP retriever on {device}...")
    retriever = ClipRetriever(model="eva-clip", device=device)

    logger.info(f"Loading knowledge base from {knowledge_base_path}...")
    retriever.load_knowledge_base(knowledge_base_path)

    logger.info(f"Loading FAISS index from {faiss_index_path}...")
    retriever.load_entity_faiss_index(faiss_index_path, load_index_only=False)

    logger.info("EVA-CLIP retriever loaded with FAISS index")
    return retriever


@torch.no_grad()
def compute_evaclip_embedding(image_path: str, retriever: ClipRetriever):
    """Compute EVA-CLIP embedding for a single image."""
    try:
        image = Image.open(image_path).convert('RGB')
        image_features = retriever._get_image_features(image)
        image_features = F.normalize(image_features.float(), p=2, dim=-1)
        return image_features.cpu().numpy().flatten()
    except Exception as e:
        logger.warning(f"Failed to compute EVA-CLIP embedding for {image_path}: {e}")
        return None


@torch.no_grad()
def find_top5_candidates_with_faiss(
    query_image_path: str,
    query_image_url: str,
    retriever: ClipRetriever,
    url_mapping: Dict[str, str],
    current_entry_url: str,
    current_entry_image_urls: set,
    similarity_tolerance: float = 0.05,
    search_k: int = 2000
) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Find top-5 candidates using FAISS: 1 correct (same entry) + 4 hard negatives.
    Hard negatives are selected to have similarity scores within tolerance of the correct candidate.
    """
    try:
        query_image = Image.open(query_image_path).convert('RGB')
        query_embedding = retriever._get_image_features(query_image)
        query_norm = F.normalize(query_embedding.float(), p=2, dim=-1)

        D, I = retriever.entity_faiss_index.search(query_norm, search_k)

        correct_candidate = None
        all_negatives = []
        seen_negative_entries = set()
        seen_image_urls = {query_image_url}

        for i in range(search_k):
            similarity = D[0][i].item()
            faiss_idx = I[0][i].item()

            kb_index = retriever.faiss_index_ids[faiss_idx]
            kb_entry = retriever.knowledge_base[kb_index]

            entry_faiss_indices = [idx for idx, kid in enumerate(retriever.faiss_index_ids) if kid == kb_index]
            if not entry_faiss_indices:
                continue

            start_id = entry_faiss_indices[0]
            offset = faiss_idx - start_id

            if offset >= len(kb_entry.image_urls):
                continue

            image_url = kb_entry.image_urls[offset]

            if image_url in seen_image_urls:
                continue

            image_path = url_mapping.get(image_url)
            if not image_path or not os.path.exists(image_path):
                continue

            candidate_dict = {
                'url': image_url,
                'path': image_path,
                'entry_title': kb_entry.title,
                'entry_url': kb_entry.url,
                'similarity': similarity
            }

            if kb_entry.url == current_entry_url:
                if correct_candidate is None:
                    correct_candidate = candidate_dict
                    seen_image_urls.add(image_url)
            else:
                if kb_entry.url not in seen_negative_entries:
                    all_negatives.append(candidate_dict)
                    seen_negative_entries.add(kb_entry.url)
                    seen_image_urls.add(image_url)

        if correct_candidate is None:
            logger.warning("Could not find a correct candidate from the same entry")
            return None, []

        correct_similarity = correct_candidate['similarity']
        hard_negatives = []

        for neg_candidate in all_negatives:
            similarity_diff = abs(neg_candidate['similarity'] - correct_similarity)
            if similarity_diff <= similarity_tolerance:
                neg_candidate['similarity_diff'] = similarity_diff
                hard_negatives.append(neg_candidate)

        hard_negatives.sort(key=lambda x: x['similarity_diff'])

        if len(hard_negatives) < 4:
            logger.warning(f"Only found {len(hard_negatives)} hard negatives within tolerance {similarity_tolerance}")
            return None, []

        return correct_candidate, hard_negatives[:4]

    except Exception as e:
        logger.warning(f"FAISS search failed: {e}")
        return None, []


def compute_clip_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Compute cosine similarity between two CLIP embeddings."""
    return float(np.dot(embedding1, embedding2))


def visualize_reranking_instance(
    instance: ReRankingInstance,
    output_path: str,
    query_embedding: Optional[np.ndarray] = None,
    retriever: Optional[ClipRetriever] = None,
    title: str = "Re-ranking Instance"
):
    """Visualize a re-ranking instance with query, correct answer, and candidates."""
    num_candidates = len(instance.candidates)

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, max(3, num_candidates), hspace=0.3, wspace=0.3)

    query_img = Image.open(instance.query.query_image_path).convert('RGB')

    correct_idx = None
    correct_candidate = None
    for idx, cand in enumerate(instance.candidates):
        if cand.relevance_score == 1.0:
            correct_idx = idx
            correct_candidate = cand
            break

    ax_query = fig.add_subplot(gs[0, 0:2])
    ax_query.imshow(query_img)
    ax_query.set_title(f"Query Image\n{instance.query.query_text}",
                       fontsize=12, fontweight='bold', color='blue')
    ax_query.axis('off')

    if correct_candidate:
        correct_img = Image.open(correct_candidate.image_path).convert('RGB')
        ax_correct = fig.add_subplot(gs[0, 2:4])
        ax_correct.imshow(correct_img)

        similarity_text = ""
        if query_embedding is not None and retriever is not None:
            correct_embedding = compute_evaclip_embedding(correct_candidate.image_path, retriever)
            if correct_embedding is not None:
                similarity = compute_clip_similarity(query_embedding, correct_embedding)
                similarity_text = f"\nSimilarity: {similarity:.4f}"

        ax_correct.set_title(f"Correct Answer (GT)\n{correct_candidate.text[:50]}...{similarity_text}",
                            fontsize=12, fontweight='bold', color='green')
        ax_correct.axis('off')

    num_cols = min(5, num_candidates)

    for idx, candidate in enumerate(instance.candidates):
        row = 1 + (idx // num_cols)
        col = idx % num_cols

        ax = fig.add_subplot(gs[row, col])

        try:
            cand_img = Image.open(candidate.image_path).convert('RGB')
            ax.imshow(cand_img)

            similarity_text = ""
            if query_embedding is not None and retriever is not None:
                cand_embedding = compute_evaclip_embedding(candidate.image_path, retriever)
                if cand_embedding is not None:
                    similarity = compute_clip_similarity(query_embedding, cand_embedding)
                    similarity_text = f"Sim: {similarity:.4f}"

            is_correct = candidate.relevance_score == 1.0
            color = 'green' if is_correct else 'red'
            label = 'CORRECT' if is_correct else 'HARD NEG'

            ax.set_title(f"{label}\n{similarity_text}\n{candidate.text[:30]}...",
                        fontsize=9, color=color)

        except Exception as e:
            ax.text(0.5, 0.5, "Error loading\nimage", ha='center', va='center', fontsize=10, color='red')

        ax.axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.debug(f"Saved visualization to {output_path}")


def create_rerank_dataset(
    knowledge_base_path: str,
    url_mapping_path: str,
    image_root: str,
    output_path: str,
    faiss_index_path: str,
    num_entries: int = 100,
    num_candidates: int = 5,
    seed: int = 42,
    similarity_tolerance: float = 0.1,
    visualize_samples: int = 20,
    visualization_dir: Optional[str] = None,
    device: str = 'cuda:0'
) -> List[ReRankingInstance]:
    """
    Create a re-ranking dataset from InfoSeek or E-VQA knowledge base using EVA-CLIP with FAISS.
    """
    random.seed(seed)

    if visualization_dir is None and visualize_samples > 0:
        output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
        visualization_dir = os.path.join(output_dir, 'visualizations')

    if visualize_samples > 0:
        os.makedirs(visualization_dir, exist_ok=True)
        logger.info(f"Will save {visualize_samples} visualizations to {visualization_dir}")

    retriever = load_evaclip_retriever(knowledge_base_path, faiss_index_path, device)

    logger.info(f"Loading URL mapping from {url_mapping_path}...")
    url_mapping = _load_url_mapping(url_mapping_path, image_root)
    logger.info(f"Loaded {len(url_mapping)} URL mappings")

    logger.info(f"Loading knowledge base from {knowledge_base_path}...")
    kb = WikipediaKnowledgeBase(knowledge_base_path)
    kb.load_knowledge_base()
    logger.info(f"Loaded {len(kb)} knowledge base entries")

    logger.info("Filtering entries with at least 2 images...")
    valid_entries = []
    for entry in kb.knowledge_base:
        valid_image_urls = [url for url in entry.image_urls if url in url_mapping]
        if len(valid_image_urls) >= 2:
            valid_entries.append((entry, valid_image_urls))

    logger.info(f"Found {len(valid_entries)} entries with at least 2 images")

    if len(valid_entries) < num_entries:
        logger.warning(f"Only {len(valid_entries)} valid entries available")
        num_entries = len(valid_entries)

    logger.info(f"Selecting {num_entries} entries...")
    selected_entries = random.sample(valid_entries, num_entries)

    logger.info("Creating re-ranking instances...")
    instances = []

    entry_url_to_entry = {entry.url: entry for entry in kb.knowledge_base}

    entry_iterator = iter(selected_entries)
    attempted_entries = 0
    max_attempts = len(selected_entries) * 2

    while len(instances) < num_entries and attempted_entries < max_attempts:
        try:
            entry, valid_urls = next(entry_iterator)
        except StopIteration:
            remaining_entries = [e for e in valid_entries if e not in selected_entries]
            if not remaining_entries:
                logger.warning(f"No more entries. Created {len(instances)} instances")
                break
            entry, valid_urls = random.choice(remaining_entries)
            selected_entries.append((entry, valid_urls))

        attempted_entries += 1

        query_image_path = None
        query_image_url = None
        remaining_urls = valid_urls.copy()

        while remaining_urls and not query_image_path:
            query_url = remaining_urls[0]
            query_path = url_mapping[query_url]
            if os.path.exists(query_path):
                query_image_path = query_path
                query_image_url = query_url
                remaining_urls.pop(0)
                break
            else:
                remaining_urls.pop(0)

        if not query_image_path:
            continue

        idx = len(instances)
        query = ReRankingQuery(
            query_image_path=query_image_path,
            query_text=f"Find images related to: {entry.title}",
            query_id=f"query_{idx}_{entry.url}"
        )

        candidates = []

        correct_candidate, hard_negatives = find_top5_candidates_with_faiss(
            query_image_path=query_image_path,
            query_image_url=query_image_url,
            retriever=retriever,
            url_mapping=url_mapping,
            current_entry_url=entry.url,
            current_entry_image_urls=set(valid_urls),
            similarity_tolerance=similarity_tolerance,
            search_k=2000
        )

        if correct_candidate is None or len(hard_negatives) < 4:
            continue

        entry_text = f"{entry.title}\n" + "\n".join(entry.section_texts[:3])

        candidates.append(CandidateDocument(
            doc_id=f"doc_{idx}_correct",
            image_path=correct_candidate['path'],
            text=entry_text[:500],
            relevance_score=1.0
        ))

        for neg_idx, neg_data in enumerate(hard_negatives):
            neg_entry = entry_url_to_entry[neg_data['entry_url']]
            neg_entry_text = f"{neg_entry.title}\n" + "\n".join(neg_entry.section_texts[:3])

            candidates.append(CandidateDocument(
                doc_id=f"doc_{idx}_neg_{neg_idx}",
                image_path=neg_data['path'],
                text=neg_entry_text[:500],
                relevance_score=0.0
            ))

        query_embedding = compute_evaclip_embedding(query_image_path, retriever)

        random.shuffle(candidates)

        instance = ReRankingInstance(query=query, candidates=candidates)
        instances.append(instance)

        if visualize_samples > 0 and len(instances) <= visualize_samples:
            vis_path = os.path.join(visualization_dir, f"instance_{len(instances)-1:03d}.png")
            visualize_reranking_instance(
                instance=instance,
                output_path=vis_path,
                query_embedding=query_embedding,
                retriever=retriever,
                title=f"Instance {len(instances)-1}: {entry.title}"
            )

        if len(instances) % 50 == 0:
            logger.info(f"Created {len(instances)}/{num_entries} instances")

    logger.info(f"Created {len(instances)} valid instances")

    # Save dataset as JSON
    logger.info(f"Saving dataset to {output_path}...")

    dataset_dict = {
        'metadata': {
            'num_entries': len(instances),
            'num_candidates': num_candidates,
            'seed': seed,
            'similarity_tolerance': similarity_tolerance,
            'model': 'EVA-CLIP-8B'
        },
        'instances': []
    }

    for instance in instances:
        instance_dict = {
            'query': {
                'query_image_path': instance.query.query_image_path,
                'query_text': instance.query.query_text,
                'query_id': instance.query.query_id
            },
            'candidates': [
                {
                    'doc_id': cand.doc_id,
                    'image_path': cand.image_path,
                    'text': cand.text,
                    'relevance_score': cand.relevance_score
                }
                for cand in instance.candidates
            ]
        }
        dataset_dict['instances'].append(instance_dict)

    with open(output_path, 'w') as f:
        json.dump(dataset_dict, f, indent=2)

    logger.info(f"Saved dataset with {len(instances)} instances")

    return instances


def convert_to_parquet(
    json_dataset_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
):
    """Convert JSON dataset to train/val parquet files."""
    random.seed(seed)

    logger.info("Converting JSON dataset to parquet format...")

    with open(json_dataset_path, 'r') as f:
        dataset_dict = json.load(f)

    instances = dataset_dict['instances']
    logger.info(f"Loaded {len(instances)} instances")

    random.shuffle(instances)
    split_idx = int(len(instances) * train_ratio)
    train_instances = instances[:split_idx]
    val_instances = instances[split_idx:]

    logger.info(f"Split: {len(train_instances)} train, {len(val_instances)} val")

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_instances in [('train', train_instances), ('val', val_instances)]:
        logger.info(f"Converting {split_name} split...")

        data_list = []
        for idx, instance_dict in enumerate(split_instances):
            query_image_path = instance_dict['query']['query_image_path']
            query_text = instance_dict['query']['query_text']
            query_id = instance_dict['query']['query_id']

            try:
                if not os.path.exists(query_image_path):
                    continue

                candidate_image_paths = []
                relevance_labels = []
                candidate_texts = []

                for cand in instance_dict['candidates']:
                    cand_image_path = cand['image_path']
                    if not os.path.exists(cand_image_path):
                        continue
                    candidate_image_paths.append(cand_image_path)
                    relevance_labels.append(cand['relevance_score'])
                    candidate_texts.append(cand['text'])

                if len(candidate_image_paths) < 2:
                    continue

                messages = [{"role": "user", "content": query_text}]

                data_item = {
                    'query_id': query_id,
                    'data': messages,
                    'image': query_image_path,
                    'candidate_images': candidate_image_paths,
                    'relevance_labels': relevance_labels,
                    'candidate_texts': candidate_texts,
                }

                data_list.append(data_item)

            except Exception as e:
                logger.warning(f"Failed to process instance {idx}: {e}")
                continue

        hf_dataset = Dataset.from_list(data_list)

        parquet_path = os.path.join(output_dir, f'{split_name}.parquet')
        disk_path = os.path.join(output_dir, split_name)

        hf_dataset.save_to_disk(disk_path)
        hf_dataset.to_parquet(parquet_path)

        logger.info(f"Saved {len(hf_dataset)} instances to {parquet_path}")

    logger.info("Conversion complete!")


def main():
    parser = argparse.ArgumentParser(description="Create re-ranking dataset for VLM training")
    parser.add_argument("--num_entries", type=int, default=1000, help="Number of dataset entries")
    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidates per query")
    parser.add_argument("--seed", type=int, default=8565, help="Random seed")
    parser.add_argument("--similarity_tolerance", type=float, default=0.05, help="Similarity tolerance for hard negatives")
    parser.add_argument("--visualize_samples", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for EVA-CLIP")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    setup_logging(level=args.log_level)

    # Paths
    url_mapping_path = DATA_ROOT / "url_to_path_mapping.json"
    image_root = DATA_ROOT
    knowledge_base_path = DATA_ROOT / "wiki_100_dict_v4.json"
    faiss_index_path = DATA_ROOT / "eva-clip" / "faiss_index_I"

    output_folder = BASE_DIR / "dataset_outputs" / DATASET_NAME
    output_path = output_folder / f"{DATASET_NAME}_rerank_dataset_evaclip_hard_neg.json"
    parquet_output_dir = output_folder / "parquet_data_evaclip_hard_neg"
    visualization_dir = output_folder / "visualizations"

    # Create dataset
    instances = create_rerank_dataset(
        knowledge_base_path=str(knowledge_base_path),
        url_mapping_path=str(url_mapping_path),
        image_root=str(image_root),
        output_path=str(output_path),
        faiss_index_path=str(faiss_index_path),
        num_entries=args.num_entries,
        num_candidates=args.num_candidates,
        seed=args.seed,
        similarity_tolerance=args.similarity_tolerance,
        visualize_samples=args.visualize_samples,
        visualization_dir=str(visualization_dir),
        device=args.device
    )

    # Convert to parquet
    if instances:
        convert_to_parquet(
            json_dataset_path=str(output_path),
            output_dir=str(parquet_output_dir),
            train_ratio=0.8,
            seed=42
        )

        logger.info("="*60)
        logger.info("All done!")
        logger.info(f"  Dataset JSON: {output_path}")
        logger.info(f"  Parquet data: {parquet_output_dir}/")
        logger.info(f"  Visualizations: {visualization_dir}/")
        logger.info("="*60)


if __name__ == "__main__":
    main()
