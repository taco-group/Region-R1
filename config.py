"""
Configuration file for VLM training
"""
import os
from pathlib import Path

# =============================================================================
# Path Configuration
# =============================================================================

# Base directory is relative to this config file's location
BASE_DIR = Path(__file__).parent.resolve()

# Run configuration
RUN_NAME = "mixture_queryOnly_1222_2100"

# Output directories (relative to BASE_DIR)
OUTPUT_DIR = BASE_DIR / "outputs"
RUN_OUTPUT_DIR = BASE_DIR / "runs" / RUN_NAME
EVAL_OUTPUT_DIR = RUN_OUTPUT_DIR / "evaluations"
MODEL_OUTPUT_DIR = RUN_OUTPUT_DIR / "checkpoints"
FINAL_MODEL_DIR = RUN_OUTPUT_DIR / "final_model"

# Dataset directory - supports InfoSeek or E-VQA datasets
# Override via DATASET_NAME env var: "InfoSeek" (default) or "EVQA"
DATASET_NAME = os.environ.get("DATASET_NAME", "InfoSeek")
DATA_DIR = BASE_DIR / "Reranker_Dataset" / f"{DATASET_NAME}_top10_parquet"
FEATURE_CACHE_PATH = DATA_DIR / "test_candidate_features.pt"

# External data paths - can be overridden by environment variables
# Supports both InfoSeek and E-VQA datasets
DATA_ROOT = Path(os.environ.get(
    "DATA_ROOT",
    Path.home() / "Desktop" / f"{DATASET_NAME}-data"
))

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

# EVA-CLIP configuration for reward model
EVACLIP_MODEL_ID = "BAAI/EVA-CLIP-8B"
EVACLIP_PROCESSOR_ID = "openai/clip-vit-large-patch14"
EVACLIP_DEVICE = "cuda:0"

# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """
You are an intelligent image cropping assistant for image retrieval tasks.

Given an image and the user's question, your task is to decide whether cropping a specific region would help remove the redundant information and retrieve more relevant information from a database.

# Instructions
1. Carefully analyze the image in the context of the user's question.
2. Based on the user's question, decide whether cropping would improve re-ranking accuracy among many candidates, and push the most relevant candidate to the rank 1. You can decide:
   - If you think a specific region is most relevant to answering the question, crop to focus on it.
   - If you think the full image is already optimal for the question, do NOT crop.
3. Explain your reasoning, then either call the tool or output NO_CROP_NEEDED.

# Tools
<tools>
{"type":"function","function":{"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it.","parameters":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"Bounding box as [x1, y1, x2, y2], where (x1, y1) is top-left and (x2, y2) is bottom-right."},"label":{"type":"string","description":"Label of the cropped region (OPTIONAL)."}},"required":["bbox_2d"]}}}
</tools>

# How to call a tool
Return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

Again, if you call the tool, you MUST follow the format exactly as specified. Otherwise, I will be unable to parse your response.
"""

# =============================================================================
# LoRA Configuration
# =============================================================================

LORA_CONFIG = {
    "task_type": "CAUSAL_LM",
    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": "all-linear",
}

# =============================================================================
# Training Configuration
# =============================================================================

TRAINING_CONFIG = {
    "learning_rate": 5e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 50,
    "num_train_epochs": 2,
    "per_device_train_batch_size": 4,
    "max_completion_length": 256,
    "num_generations": 4,
    "max_prompt_length": 8192,
    "logging_steps": 10,
    "save_steps": 100,
    "bf16": True,
}

# =============================================================================
# Inference Configuration
# =============================================================================

INFERENCE_CONFIG = {
    "max_new_tokens": 256,
}

# =============================================================================
# Weights & Biases Configuration
# =============================================================================

WANDB_CONFIG = {
    "project": "Mixture",
    "entity": None,
    "name": RUN_NAME,
    "tags": ["grpo", "qwen2.5-vl-3B", "3B", "mixture"],
    "notes": "Training Qwen2.5-VL-3B with mixture reward",
}
