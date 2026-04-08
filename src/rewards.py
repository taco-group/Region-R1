"""
Reward functions for GRPO training using EVA-CLIP for re-ranking.

Supported reward types:
- absolute: Measures absolute quality using CLIP similarity
- mixture: Weighted deltas (crop vs. baseline) across MRR, NDCG, rank, and margin
"""
import os
import logging
import traceback
import numpy as np
import torch
from typing import List, Tuple

from utils import (
    calculate_mrr,
    calculate_ndcg,
    parse_bbox_from_completion,
    crop_image_with_bbox,
    set_clip_model,
    calculate_clip_scores
)

logger = logging.getLogger(__name__)

# Global counter for saved images
_saved_image_count = 0
OUTPUT_DIR = "tmp_output"

# Global step counter for decaying encouragement reward
_global_training_step = 0


def save_debug_images(query_img, cropped_img, bbox, example_idx, completion_idx):
    """
    Save original and cropped images to tmp_output/ for debugging.
    Only saves for the first 5 completions globally.
    """
    global _saved_image_count

    if _saved_image_count >= 5:
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    original_path = os.path.join(
        OUTPUT_DIR,
        f"completion_{completion_idx:03d}_example_{example_idx}_original.png"
    )
    query_img.save(original_path)

    bbox_str = f"bbox_{int(bbox[0])}_{int(bbox[1])}_{int(bbox[2])}_{int(bbox[3])}" if bbox else "no_bbox"
    cropped_path = os.path.join(
        OUTPUT_DIR,
        f"completion_{completion_idx:03d}_example_{example_idx}_cropped_{bbox_str}.png"
    )
    cropped_img.save(cropped_path)

    _saved_image_count += 1
    logger.debug(f"Saved debug images for completion {completion_idx}, example {example_idx}")


def calculate_first_positive_rank(clip_ranking: List[int], ground_truth_labels: List[float]) -> int:
    """
    Calculate the rank (1-indexed) of the first positive item in the ranking.

    Args:
        clip_ranking: List of indices sorted by CLIP similarity (descending)
        ground_truth_labels: Relevance labels for each candidate

    Returns:
        Rank of first positive item (1-indexed), or len(ranking)+1 if not found
    """
    max_relevance = max(ground_truth_labels) if ground_truth_labels else 0
    if max_relevance == 0:
        return len(clip_ranking) + 1

    for position, idx in enumerate(clip_ranking):
        if idx < len(ground_truth_labels) and ground_truth_labels[idx] == max_relevance:
            return position + 1  # 1-indexed rank

    return len(clip_ranking) + 1


def calculate_margin(clip_scores: List[float], ground_truth_labels: List[float]) -> float:
    """
    Calculate margin between positive similarity and max negative similarity.

    Margin = pos_sim - max_neg_sim

    Args:
        clip_scores: CLIP similarity scores for each candidate
        ground_truth_labels: Relevance labels for each candidate

    Returns:
        Margin value (positive means good separation)
    """
    if not clip_scores or not ground_truth_labels:
        return 0.0

    max_relevance = max(ground_truth_labels)
    if max_relevance == 0:
        return 0.0

    pos_scores = []
    neg_scores = []

    for i, (score, label) in enumerate(zip(clip_scores, ground_truth_labels)):
        if label == max_relevance:
            pos_scores.append(score)
        else:
            neg_scores.append(score)

    if not pos_scores:
        return 0.0
    if not neg_scores:
        return 1.0  # Perfect separation (no negatives)

    # Use mean of positive scores and max of negative scores
    pos_sim = np.mean(pos_scores)
    max_neg_sim = max(neg_scores)
    return pos_sim - max_neg_sim


def compute_full_metrics(query_img, candidates, ground_truth_labels, candidate_texts=None) -> dict:
    """
    Compute all metrics needed for the mixture reward.

    Returns:
        Dictionary with mrr, ndcg, rank, margin, and clip_scores
    """
    clip_scores = calculate_clip_scores(query_img, candidates, candidate_texts)
    clip_ranking = sorted(
        range(len(clip_scores)),
        key=lambda x: clip_scores[x],
        reverse=True
    )

    mrr = calculate_mrr(clip_ranking, ground_truth_labels)
    ndcg = calculate_ndcg(clip_ranking, ground_truth_labels)
    rank = calculate_first_positive_rank(clip_ranking, ground_truth_labels)
    margin = calculate_margin(clip_scores, ground_truth_labels)

    return {
        'mrr': mrr,
        'ndcg': ndcg,
        'rank': rank,
        'margin': margin,
        'clip_scores': clip_scores,
        'clip_ranking': clip_ranking
    }


def clip_reranking_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Absolute reward function using CLIP to re-rank candidates based on visual similarity.
    """
    candidate_images = kwargs.get('candidate_images', [])
    candidate_texts = kwargs.get('candidate_texts', [])
    images = kwargs.get('image', [])
    relevance_labels = kwargs.get('relevance_labels', [])

    if not candidate_images or not images:
        logger.warning(f"Missing required fields. Available keys: {kwargs.keys()}")
        return [0.5] * len(completions)

    if not relevance_labels:
        logger.warning("relevance_labels not found. Using default rewards.")
        return [0.5] * len(completions)

    rewards = []

    if not hasattr(clip_reranking_reward, 'completion_counter'):
        clip_reranking_reward.completion_counter = 0

    for i, (completion, candidates, query_img, gt_labels, texts) in enumerate(
        zip(completions, candidate_images, images, relevance_labels, candidate_texts)
    ):
        try:
            bbox = parse_bbox_from_completion(completion)

            if bbox is not None:
                cropped_query_img = crop_image_with_bbox(query_img, bbox, resize=True)
                logger.debug(f"Example {i}: Using cropped image with bbox {bbox}")
            else:
                cropped_query_img = query_img
                logger.debug(f"Example {i}: No bbox found, using full query image")

            save_debug_images(
                query_img,
                cropped_query_img,
                bbox,
                example_idx=i,
                completion_idx=clip_reranking_reward.completion_counter
            )
            clip_reranking_reward.completion_counter += 1

            clip_scores = calculate_clip_scores(cropped_query_img, candidates, texts)
            clip_ranking = sorted(range(len(clip_scores)), key=lambda x: clip_scores[x], reverse=True)

            if isinstance(gt_labels, (list, tuple)):
                ground_truth_labels = list(gt_labels)
            else:
                logger.warning(f"Example {i}: relevance_labels not a list, using default")
                ground_truth_labels = [1.0 / len(candidates)] * len(candidates)

            mrr_score = calculate_mrr(clip_ranking, ground_truth_labels)
            ndcg_score = calculate_ndcg(clip_ranking, ground_truth_labels)
            reward = 0.5 * mrr_score + 0.5 * ndcg_score

            rewards.append(reward)
            logger.debug(f"Example {i}: MRR={mrr_score:.3f}, NDCG={ndcg_score:.3f}, Combined={reward:.3f}")

        except Exception as e:
            logger.error(f"Error calculating reward for example {i}: {e}")
            logger.debug(traceback.format_exc())
            rewards.append(0.3)

    return rewards


def update_training_step(step: int):
    """Update the global training step counter."""
    global _global_training_step
    _global_training_step = step


def calculate_encouragement_bonus(
    initial_bonus: float,
    decay_steps: int,
    min_bonus: float = 0.0
) -> float:
    """
    Calculate the decaying encouragement bonus for cropping.

    Uses linear decay: bonus = initial_bonus * max(0, 1 - step / decay_steps)

    Args:
        initial_bonus: Starting bonus value (default: 0.1)
        decay_steps: Number of steps for bonus to reach zero (default: 1000)
        min_bonus: Minimum bonus floor (default: 0.0)

    Returns:
        Current encouragement bonus
    """
    step = _global_training_step
    decay_factor = max(0.0, 1.0 - step / decay_steps)
    bonus = initial_bonus * decay_factor
    return max(min_bonus, bonus)


def clip_mixture_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Mixture reward function combining four delta metrics with decaying encouragement.

    Reward components when model crops:
    - Δmrr   = mrr_crop - mrr_base
    - Δndcg  = ndcg_crop - ndcg_base
    - Δrank  = log(rank_base + 1) - log(rank_crop + 1)
    - Δmargin = (pos_sim_crop - max_neg_sim_crop) - (pos_sim_base - max_neg_sim_base)

    Plus a decaying encouragement bonus for cropping.

    Args:
        completions: List of model completions
        **kwargs: Must include candidate_images, image, relevance_labels

    Configurable via kwargs:
        weight_mrr: Weight for Δmrr (default: 0.3)
        weight_ndcg: Weight for Δndcg (default: 0.3)
        weight_rank: Weight for Δrank (default: 0.2)
        weight_margin: Weight for Δmargin (default: 0.2)
        initial_encouragement: Initial encouragement bonus (default: 0.1)
        encouragement_decay_steps: Steps for bonus decay (default: 1000)

    Returns:
        List of reward values
    """
    candidate_images = kwargs.get('candidate_images', [])
    images = kwargs.get('image', [])
    candidate_texts = kwargs.get('candidate_texts', [])
    relevance_labels = kwargs.get('relevance_labels', [])

    # Configurable weights
    weight_mrr = kwargs.get('weight_mrr', 0.5)
    weight_ndcg = kwargs.get('weight_ndcg', 0.5)
    weight_rank = kwargs.get('weight_rank', 0.0)
    weight_margin = kwargs.get('weight_margin', 0.1)

    # Encouragement parameters
    initial_encouragement = kwargs.get('initial_encouragement', 0.3)
    encouragement_decay_steps = kwargs.get('encouragement_decay_steps', 5000)

    if not candidate_images or not images:
        logger.warning(f"Missing required fields. Available keys: {kwargs.keys()}")
        raise ValueError("Missing required fields. Please provide candidate_images and image.")

    if not relevance_labels:
        logger.warning("relevance_labels not found. Using default rewards.")
        raise ValueError("relevance_labels not found. Please provide relevance_labels.")

    rewards = []

    if not hasattr(clip_mixture_reward, 'completion_counter'):
        clip_mixture_reward.completion_counter = 0

    # Calculate current encouragement bonus
    encouragement_bonus = calculate_encouragement_bonus(
        initial_bonus=initial_encouragement,
        decay_steps=encouragement_decay_steps
    )

    for i, (completion, candidates, query_img, gt_labels, texts) in enumerate(
        zip(completions, candidate_images, images, relevance_labels, candidate_texts)
    ):
        try:
            if isinstance(gt_labels, (list, tuple)):
                ground_truth_labels = list(gt_labels)
            else:
                logger.warning(f"Example {i}: relevance_labels not a list, using default")
                ground_truth_labels = [1.0 / len(candidates)] * len(candidates)

            # Compute baseline metrics
            base_metrics = compute_full_metrics(query_img, candidates, ground_truth_labels, texts)

            bbox = parse_bbox_from_completion(completion)
            model_chose_to_crop = (bbox is not None)

            if model_chose_to_crop:
                # Validate bbox before cropping
                img_width, img_height = query_img.size
                x1, y1, x2, y2 = bbox

                # Check for invalid coordinates relative to image size
                x1_c = max(0, min(x1, img_width))
                x2_c = max(0, min(x2, img_width))
                y1_c = max(0, min(y1, img_height))
                y2_c = max(0, min(y2, img_height))

                if x1_c >= x2_c or y1_c >= y2_c:
                    # Invalid BBox (zero area or inverted) -> No Reward!
                    reward = -1.0
                    decision_quality = "INVALID BBOX"
                    logger.debug(f"Example {i}: {decision_quality} Reward={reward:.3f}")
                    rewards.append(reward)
                    clip_mixture_reward.completion_counter += 1
                    continue

                cropped_query_img = crop_image_with_bbox(query_img, bbox, resize=True)

                save_debug_images(
                    query_img,
                    cropped_query_img,
                    bbox,
                    example_idx=i,
                    completion_idx=clip_mixture_reward.completion_counter
                )

                # Compute cropped metrics
                crop_metrics = compute_full_metrics(cropped_query_img, candidates, ground_truth_labels, texts)

                # Calculate deltas
                delta_mrr = crop_metrics['mrr'] - base_metrics['mrr']
                delta_ndcg = crop_metrics['ndcg'] - base_metrics['ndcg']
                delta_rank = np.log(base_metrics['rank'] + 1) - np.log(crop_metrics['rank'] + 1)
                delta_margin = crop_metrics['margin'] - base_metrics['margin']

                # Weighted mixture of deltas
                weighted_delta = (
                    weight_mrr * delta_mrr +
                    weight_ndcg * delta_ndcg +
                    weight_rank * delta_rank +
                    weight_margin * delta_margin
                )

                # Zero-Floor Strategy: Clip negative deltas to -0.01
                # This makes "trying and failing" virtually free cost (-0.01),
                # so the expected value of exploration is positive.
                weighted_delta = max(weighted_delta, -0.05)

                reward = weighted_delta * 5

                decision_quality = (
                    f"CROP Δmrr={delta_mrr:+.3f} Δndcg={delta_ndcg:+.3f} "
                    f"Δrank={delta_rank:+.3f} Δmargin={delta_margin:+.3f} "
                    f"bonus={encouragement_bonus:.3f}"
                )

                logger.info(f"Example {i}: CROPPED bbox={bbox}")
                logger.info(f"  Base: mrr={base_metrics['mrr']:.3f} ndcg={base_metrics['ndcg']:.3f} "
                           f"rank={base_metrics['rank']} margin={base_metrics['margin']:.3f}")
                logger.info(f"  Crop: mrr={crop_metrics['mrr']:.3f} ndcg={crop_metrics['ndcg']:.3f} "
                           f"rank={crop_metrics['rank']} margin={crop_metrics['margin']:.3f}")
                logger.info(f"  {decision_quality} -> Reward={reward:.3f}")

            else:
                # No Crop Logic
                if base_metrics['rank'] == 1:
                    # Correct decision: Positive is already at Rank 1, so not cropping is optimal.
                    # Reward efficient behavior.
                    reward = 1.0
                    decision_quality = "NO-CROP (Correct: Rank already 1)"
                else:
                    # Positive is NOT at Rank 1, so we probably should have tried to crop.
                    # Neutral reward matching "failed crop" floor (0.0)
                    reward = 0.0
                    decision_quality = "NO-CROP"

                logger.debug(f"Example {i}: {decision_quality} Reward={reward:.3f}")

            rewards.append(reward)
            clip_mixture_reward.completion_counter += 1

        except Exception as e:
            logger.error(f"Error calculating reward for example {i}: {e}")
            logger.debug(traceback.format_exc())
            rewards.append(0)

    return rewards


def combined_reward(
    completions: List[str],
    use_mixture: bool = False,
    **kwargs
) -> List[float]:
    """
    Combined reward function wrapper.

    Args:
        completions: List of model completions
        use_mixture: Use mixture reward with four deltas
        **kwargs: Additional arguments passed to reward functions
    """
    if use_mixture:
        reranking_rewards = clip_mixture_reward(completions, **kwargs)
    else:
        reranking_rewards = clip_reranking_reward(completions, **kwargs)

    avg_reward = sum(reranking_rewards) / len(reranking_rewards)
    logger.info(f"Rewards: {[f'{r:.3f}' for r in reranking_rewards]} (avg: {avg_reward:.3f})")

    return reranking_rewards


def get_reward_functions(
    use_mixture: bool = False,
    **kwargs
):
    """
    Return list of reward functions to use in training.

    Args:
        use_mixture: Use mixture reward with four deltas
        **kwargs: Additional arguments (weights, encouragement params) for mixture reward
    """
    def reward_wrapper(completions, **kw):
        # Merge kwargs from outer scope with runtime kwargs
        merged_kwargs = {**kwargs, **kw}
        return combined_reward(
            completions,
            use_mixture=use_mixture,
            **merged_kwargs
        )

    return [reward_wrapper]
