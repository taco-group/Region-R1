"""
Utility functions for ranking evaluation and image cropping
"""
import re
import json
import math
import logging
import numpy as np
from typing import List, Tuple, Optional, Union
from PIL import Image, UnidentifiedImageError
import torch
import os

logger = logging.getLogger(__name__)

# Global variables to store CLIP/EVA-CLIP model and processor
_clip_model = None
_clip_processor = None

# Global Feature Cache: path -> tensor
_feature_cache = {}

def load_feature_cache(path: str):
    """Load feature cache from disk."""
    global _feature_cache
    if os.path.exists(path):
        try:
            logger.info(f"Loading feature cache from {path}...")
            _feature_cache = torch.load(path, map_location='cpu')
            logger.info(f"Loaded {len(_feature_cache)} cached features from disk.")
        except Exception as e:
            logger.warning(f"Failed to load feature cache: {e}")
            _feature_cache = {}
    else:
        logger.info(f"No existing feature cache found at {path}")
        _feature_cache = {}

def save_feature_cache(path: str):
    """Save feature cache to disk."""
    global _feature_cache
    try:
        logger.info(f"Saving feature cache with {len(_feature_cache)} entries to {path}...")
        torch.save(_feature_cache, path)
    except Exception as e:
        logger.warning(f"Failed to save feature cache: {e}")


def try_load_image(image_path: str) -> Optional[Image.Image]:
    """
    Attempt to load an image from path.
    Returns None if loading fails or image is truncated/corrupt/too large.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        return img
    except (IOError, OSError, UnidentifiedImageError, ValueError) as e:
        logger.warning(f"Failed to load image {image_path}: {e}")
        return None
    except Image.DecompressionBombError as e:
        logger.warning(f"Image too large (skipping) {image_path}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Unexpected error loading image {image_path}: {e}")
        return None


def set_clip_model(model, processor):
    """Set the global CLIP/EVA-CLIP model and processor."""
    global _clip_model, _clip_processor
    _clip_model = model
    _clip_processor = processor
    logger.info("CLIP model set for calculation")


def calculate_clip_scores(query_image, candidate_images, candidate_texts=None, model=None, processor=None):
    """
    Use CLIP/EVA-CLIP to score candidate images based on visual similarity to query image.
    If candidate_texts are provided, they are added to candidate image features before normalization.
    Uses global model/processor if not provided.

    Args:
        query_image: PIL Image (query image)
        candidate_images: List of PIL Images (candidates to rank)
        candidate_texts: List of strings (candidate texts) - OPTIONAL
        model: Optional specific model to use (defaults to global _clip_model)
        processor: Optional specific processor to use (defaults to global _clip_processor)

    Returns:
        List of similarity scores for each candidate image
    """
    # Use provided model/processor or fall back to globals
    clip_model = model if model is not None else _clip_model
    clip_processor = processor if processor is not None else _clip_processor

    if clip_model is None or clip_processor is None:
        raise ValueError("CLIP/EVA-CLIP model not set. Call set_clip_model() or pass model/processor.")

    device = next(clip_model.parameters()).device

    # Encode query image
    query_image_rgb = query_image.convert('RGB')
    with torch.no_grad():
        query_inputs = clip_processor(images=query_image_rgb, return_tensors="pt").to(device)
        is_evaclip = hasattr(clip_model, 'encode_image')

        if is_evaclip:
            if hasattr(query_inputs, 'pixel_values'):
                query_inputs.pixel_values = query_inputs.pixel_values.half()
            query_features = clip_model.encode_image(query_inputs.pixel_values)
        else:
            query_features = clip_model.get_image_features(**query_inputs)

        query_features = query_features / query_features.norm(dim=-1, keepdim=True)

    # Encode candidates (batched)
    candidate_features = batch_encode_candidates(candidate_images, candidate_texts, clip_model, clip_processor)

    return score_candidates(query_features, candidate_features)

    
def batch_encode_candidates(candidate_images: List[Union[Image.Image, str]], candidate_texts: Optional[List[str]] = None, model=None, processor=None) -> Optional[torch.Tensor]:
    """
    Encode a list of candidate images (and optional texts) into normalized features.
    Processes in batches to improve speed. 
    Supports input as PIL Images OR file paths (strings).
    If file paths are provided, checks the global cache first.
    """
    clip_model = model if model is not None else _clip_model
    clip_processor = processor if processor is not None else _clip_processor
    
    if clip_model is None or clip_processor is None:
        raise ValueError("CLIP/EVA-CLIP model not set.")

    device = next(clip_model.parameters()).device
    is_evaclip = hasattr(clip_model, 'encode_image')
    
    indices_to_compute = []
    final_features = [None] * len(candidate_images)

    # Check cache
    for i, item in enumerate(candidate_images):
        if isinstance(item, str) and item in _feature_cache:
            feat = _feature_cache[item].to(device)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            final_features[i] = feat
        else:
            indices_to_compute.append(i)

    if not indices_to_compute:
        return torch.cat(final_features, dim=0)
        
    # Compute missing features
    to_encode_imgs = []
    to_encode_texts = []
    
    for idx in indices_to_compute:
        item = candidate_images[idx]
        if isinstance(item, str):
            # Load from path
            img = try_load_image(item)
            if img is None:
                logger.warning(f"Failed to load candidate image {item}. Invalidating batch.")
                return None
        else:
            img = item
        to_encode_imgs.append(img)
        
        if candidate_texts:
            to_encode_texts.append(candidate_texts[idx])
            
    all_computed_feats = []
    batch_size = 32
    num_batches = math.ceil(len(to_encode_imgs) / batch_size)
    
    with torch.no_grad():
        for b in range(num_batches):
            start_b = b * batch_size
            end_b = min((b + 1) * batch_size, len(to_encode_imgs))
            
            batch_imgs = to_encode_imgs[start_b:end_b]
            
            batch_imgs_rgb = [img.convert('RGB') for img in batch_imgs]
            img_inputs = clip_processor(images=batch_imgs_rgb, return_tensors="pt", padding=True).to(device)
            
            if is_evaclip:
                if hasattr(img_inputs, 'pixel_values'):
                    img_inputs.pixel_values = img_inputs.pixel_values.half()
                img_features = clip_model.encode_image(img_inputs.pixel_values)
            else:
                img_features = clip_model.get_image_features(**img_inputs)
            
            if candidate_texts and any(to_encode_texts[start_b:end_b]):
                batch_txts = to_encode_texts[start_b:end_b]
                safe_texts = [t if t else "" for t in batch_txts]
                txt_inputs = clip_processor(text=safe_texts, return_tensors="pt", padding=True, truncation=True).to(device)
                
                if is_evaclip:
                    txt_features = clip_model.encode_text(txt_inputs.input_ids)
                else:
                    txt_features = clip_model.get_text_features(**txt_inputs)
                
                img_features = img_features + txt_features
            
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            all_computed_feats.append(img_features)

    if all_computed_feats:
        computed_tensor = torch.cat(all_computed_feats, dim=0)
        
        for local_idx, global_idx in enumerate(indices_to_compute):
            feat = computed_tensor[local_idx:local_idx+1]
            final_features[global_idx] = feat

            item = candidate_images[global_idx]
            if isinstance(item, str):
                _feature_cache[item] = feat.cpu()

    # Safety check
    final_features = [f if f is not None else torch.zeros(1, 1024).to(device) for f in final_features]
    
    return torch.cat(final_features, dim=0)


def score_candidates(query_features: torch.Tensor, candidate_features: torch.Tensor) -> List[float]:
    """
    Compute cosine similarity between query features and candidate features.
    
    Args:
        query_features: shape (1, dim)
        candidate_features: shape (N, dim)
        
    Returns:
        List of N float scores
    """
    with torch.no_grad():
        # Cosine similarity
        # query: (1, D), candidates: (N, D) -> (1, N)
        scores = (query_features @ candidate_features.T).squeeze(0)
        return scores.cpu().tolist()


def calculate_mrr(predicted_ranking: List[int], relevance_labels: List[float]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR).

    MRR measures how high the first relevant item appears in the ranking.
    For multiple relevant items, we use the highest ranked one.

    Args:
        predicted_ranking: List of indices in predicted order (0-indexed)
        relevance_labels: List of relevance scores (higher is more relevant)

    Returns:
        MRR score (0 to 1, higher is better)
    """
    if not predicted_ranking or not relevance_labels:
        return 0.0

    max_relevance = max(relevance_labels)
    if max_relevance == 0:
        return 0.0

    for position, idx in enumerate(predicted_ranking):
        if idx < len(relevance_labels) and relevance_labels[idx] == max_relevance:
            return 1.0 / (position + 1)

    return 0.0


def calculate_ndcg(predicted_ranking: List[int], relevance_labels: List[float], k: int = None) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@k).

    NDCG measures the quality of ranking considering the relevance of items
    and their positions, with higher-ranked items weighted more heavily.

    Args:
        predicted_ranking: List of indices in predicted order (0-indexed)
        relevance_labels: List of relevance scores (higher is more relevant)
        k: Only consider top k items (None for all items)

    Returns:
        NDCG score (0 to 1, higher is better)
    """
    if not predicted_ranking or not relevance_labels:
        return 0.0

    if k is None:
        k = len(predicted_ranking)

    dcg = 0.0
    for i, idx in enumerate(predicted_ranking[:k]):
        if idx < len(relevance_labels):
            dcg += relevance_labels[idx] / np.log2(i + 2)

    ideal_ranking = sorted(range(len(relevance_labels)), key=lambda x: relevance_labels[x], reverse=True)
    idcg = 0.0
    for i, idx in enumerate(ideal_ranking[:k]):
        idcg += relevance_labels[idx] / np.log2(i + 2)

    if idcg == 0:
        return 0.0

    return dcg / idcg


# =============================================================================
# Image Cropping Utilities
# =============================================================================

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def parse_bbox_from_completion(completion: str) -> Optional[List[float]]:
    """
    Parse bounding box from model completion.

    Expected format:
    <tool_call>
    {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [x1, y1, x2, y2], "label": "..."}}
    </tool_call>

    Args:
        completion: Generated text from the model

    Returns:
        List of [x1, y1, x2, y2] coordinates if found, None otherwise
    """
    start_token = "<tool_call>"
    end_token = "</tool_call>"

    if start_token not in completion or end_token not in completion:
        return None

    try:
        tool_call_content = completion.split(start_token)[1].split(end_token)[0].strip()
        tool_call = json.loads(tool_call_content)

        if "arguments" in tool_call and "bbox_2d" in tool_call["arguments"]:
            bbox = tool_call["arguments"]["bbox_2d"]
            if isinstance(bbox, list) and len(bbox) == 4:
                return [float(x) for x in bbox]

    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as e:
        logger.debug(f"JSON parsing failed: {e}, trying regex fallback")
        try:
            pattern = r'"bbox_2d"\s*:\s*\[([^\]]+)\]'
            match = re.search(pattern, completion)
            if match:
                coords = match.group(1).split(',')
                if len(coords) == 4:
                    return [float(x.strip()) for x in coords]
        except (ValueError, AttributeError) as e:
            logger.debug(f"Regex fallback failed: {e}")

    return None


def round_by_factor(number: int, factor: int) -> int:
    """Round number to nearest multiple of factor."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Ceil number to nearest multiple of factor."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Floor number to nearest multiple of factor."""
    return math.floor(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """
    Intelligently resize image dimensions to be multiples of factor,
    while respecting min/max pixel constraints.

    Args:
        height: Original height
        width: Original width
        factor: Factor to round dimensions to (default: 28)
        min_pixels: Minimum total pixels
        max_pixels: Maximum total pixels

    Returns:
        Tuple of (new_height, new_width)
    """
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)

    return h_bar, w_bar


def crop_image_with_bbox(
    image: Image.Image,
    bbox: Optional[List[float]] = None,
    resize: bool = True
) -> Image.Image:
    """
    Crop image using bounding box coordinates and optionally resize.

    Args:
        image: PIL Image to crop
        bbox: Bounding box as [x1, y1, x2, y2] where (x1, y1) is top-left
              and (x2, y2) is bottom-right. If None, returns original image.
        resize: Whether to apply smart_resize after cropping

    Returns:
        Cropped (and optionally resized) PIL Image
    """
    if bbox is None or len(bbox) != 4:
        return image

    if image.mode != "RGB":
        image = image.convert("RGB")

    left, top, right, bottom = bbox

    img_width, img_height = image.size
    left = max(0, min(left, img_width))
    right = max(0, min(right, img_width))
    top = max(0, min(top, img_height))
    bottom = max(0, min(bottom, img_height))

    if left >= right or top >= bottom:
        logger.warning(f"Invalid bbox coordinates [{left}, {top}, {right}, {bottom}]. Using original image.")
        return image

    cropped_image = image.crop((left, top, right, bottom))

    if resize:
        crop_width = int(right - left)
        crop_height = int(bottom - top)
        new_height, new_width = smart_resize(crop_height, crop_width, factor=IMAGE_FACTOR)
        cropped_image = cropped_image.resize((new_width, new_height), resample=Image.BICUBIC)

    return cropped_image
