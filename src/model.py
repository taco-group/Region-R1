"""
Model setup and configuration
"""
import logging
import torch
from transformers import AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model
from config import MODEL_ID, LORA_CONFIG, EVACLIP_MODEL_ID, EVACLIP_PROCESSOR_ID, EVACLIP_DEVICE

logger = logging.getLogger(__name__)


def load_processor():
    """Load and return the AutoProcessor."""
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, use_fast=True, padding_side="left")
    return processor


def load_model():
    """Load the base Qwen2.5-VL model."""
    logger.info("Loading Qwen2.5-VL-3B model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        pretrained_model_name_or_path=MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return model


def apply_lora(model):
    """Apply LoRA configuration to the model."""
    logger.info("Applying LoRA configuration...")

    lora_config = LoraConfig(
        task_type=LORA_CONFIG["task_type"],
        r=LORA_CONFIG["r"],
        lora_alpha=LORA_CONFIG["lora_alpha"],
        lora_dropout=LORA_CONFIG["lora_dropout"],
        target_modules=LORA_CONFIG["target_modules"],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


def setup_model():
    """Load model and apply LoRA. Returns the configured model."""
    model = load_model()
    model = apply_lora(model)
    return model


def load_evaclip_model():
    """
    Load EVA-CLIP model and processor for reward function.
    EVA-CLIP is a stronger vision-language model used in the dataset creation.
    This model will be used to calculate image similarity scores.

    Returns:
        Tuple of (evaclip_model, evaclip_processor)
    """
    logger.info("Loading EVA-CLIP model for reward calculation...")
    logger.info(f"  Model: {EVACLIP_MODEL_ID}")
    logger.info(f"  Processor: {EVACLIP_PROCESSOR_ID}")

    evaclip_processor = AutoProcessor.from_pretrained(EVACLIP_PROCESSOR_ID)

    device = EVACLIP_DEVICE if torch.cuda.is_available() else "cpu"

    evaclip_model = AutoModel.from_pretrained(
        EVACLIP_MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    evaclip_model = evaclip_model.to(device).eval()

    logger.info(f"EVA-CLIP model loaded on device: {device}")

    return evaclip_model, evaclip_processor
