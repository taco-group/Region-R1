"""
Inference and testing utilities
"""
import os
import logging
import time
import torch
from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from config import FINAL_MODEL_DIR, SYSTEM_PROMPT, DATA_DIR, INFERENCE_CONFIG

logger = logging.getLogger(__name__)


def load_trained_model():
    """Load the trained model and processor."""
    logger.info("Loading trained model...")

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        FINAL_MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(FINAL_MODEL_DIR, use_fast=True, padding_side="left")

    return model, processor


def generate_with_reasoning(query_text, image, model, processor):
    """
    Generate ranking with reasoning for custom dataset.

    Args:
        query_text: The user query text (from data field)
        image: PIL Image object (query image)
        model: The trained model
        processor: The processor

    Returns:
        Tuple of (generated_text, inference_duration, num_generated_tokens)
    """
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query_text},
            ],
        },
    ]
    prompt = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=False
    )

    image_inputs, video_inputs = process_vision_info(conversation)

    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    start_time = time.time()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=INFERENCE_CONFIG["max_new_tokens"])
    end_time = time.time()

    generated_text = processor.decode(output_ids[0], skip_special_tokens=True)
    inference_duration = end_time - start_time

    num_input_tokens = inputs["input_ids"].shape[1]
    num_generated_tokens = output_ids.shape[1] - num_input_tokens

    return generated_text, inference_duration, num_generated_tokens


def test_model():
    """Test the trained model on validation data."""
    logger.info("Testing the trained model...")

    model, processor = load_trained_model()

    logger.info("Loading validation data for inference test...")
    val_dataset_test = load_dataset(
        'parquet',
        data_files=os.path.join(DATA_DIR, 'val.parquet'),
        split='train'
    )

    test_example = val_dataset_test[0]
    test_query = test_example['data'][0]['content'] if test_example['data'] else "Rank these images by relevance."
    test_image = test_example['image']

    logger.info(f"Test query: {test_query}")
    logger.info(f"Query image type: {type(test_image)}")
    logger.info(f"Number of candidate images: {len(test_example['candidate_images'])}")
    logger.info(f"Relevance labels: {test_example['relevance_labels']}")

    logger.info("Generating response...")
    generated_text, inference_duration, num_generated_tokens = generate_with_reasoning(
        test_query, test_image, model, processor
    )

    logger.info("=" * 80)
    logger.info("GENERATED RESPONSE:")
    logger.info("=" * 80)
    logger.info(generated_text)
    logger.info("=" * 80)
    logger.info(f"Inference time: {inference_duration:.2f}s")
    logger.info(f"Generated tokens: {num_generated_tokens}")
    logger.info("=" * 80)
