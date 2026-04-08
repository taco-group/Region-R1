"""
Main entry point for VLM training pipeline
Adapted for Justifiable Cropping dataset with PIL images
"""
import argparse
import logging
import wandb

from logging_config import setup_logging
from data import load_datasets, process_datasets
from model import load_processor, setup_model
from train import train_model, save_model
from inference import test_model
from config import WANDB_CONFIG, RUN_OUTPUT_DIR

logger = logging.getLogger(__name__)


def main(use_mixture=False, resume_from_checkpoint=None, mixture_kwargs=None):
    """
    Main training and inference pipeline.

    Args:
        use_mixture: If True, use mixture reward with 4 deltas + encouragement
        resume_from_checkpoint: Path to checkpoint directory to resume training from (optional)
        mixture_kwargs: Optional dict with mixture reward parameters
    """
    from config import TRAINING_CONFIG, LORA_CONFIG, MODEL_ID

    if use_mixture:
        reward_type_str = "Mixture-4delta"
    else:
        reward_type_str = "Absolute"

    # Generate a unique run ID to prevent accidental resumption of old runs
    # unless we are explicitly resuming training
    run_id = wandb.util.generate_id()
    
    wandb.init(
        project=WANDB_CONFIG["project"],
        entity=WANDB_CONFIG["entity"],
        name=WANDB_CONFIG["name"],
        id=run_id,  # Force unique ID
        resume="allow",
        tags=WANDB_CONFIG["tags"],
        notes=WANDB_CONFIG["notes"],
        config={
            "reward_model": "CLIP",
            "reward_type": reward_type_str,
            "model_id": MODEL_ID,
            "learning_rate": TRAINING_CONFIG["learning_rate"],
            "num_train_epochs": TRAINING_CONFIG["num_train_epochs"],
            "per_device_train_batch_size": TRAINING_CONFIG["per_device_train_batch_size"],
            "max_completion_length": TRAINING_CONFIG["max_completion_length"],
            "num_generations": TRAINING_CONFIG["num_generations"],
            "logging_steps": TRAINING_CONFIG["logging_steps"],
            "save_steps": TRAINING_CONFIG["save_steps"],
            "lora_r": LORA_CONFIG["r"],
            "lora_alpha": LORA_CONFIG["lora_alpha"],
            "lora_dropout": LORA_CONFIG["lora_dropout"],
            "lora_target_modules": LORA_CONFIG["target_modules"],
        }
    )

    logger.info("=" * 60)
    logger.info("REWARD MODEL: CLIP")
    if use_mixture:
        logger.info("REWARD TYPE: Mixture (4-delta + decaying encouragement)")
        if mixture_kwargs:
            logger.info(f"  Weights: mrr={mixture_kwargs.get('weight_mrr', 0.3)}, "
                       f"ndcg={mixture_kwargs.get('weight_ndcg', 0.3)}, "
                       f"rank={mixture_kwargs.get('weight_rank', 0.2)}, "
                       f"margin={mixture_kwargs.get('weight_margin', 0.2)}")
            logger.info(f"  Encouragement: initial={mixture_kwargs.get('initial_encouragement', 0.1)}, "
                       f"decay_steps={mixture_kwargs.get('encouragement_decay_steps', 1000)}")
    else:
        logger.info("REWARD TYPE: Absolute (measures quality)")
    if resume_from_checkpoint:
        logger.info(f"RESUMING FROM CHECKPOINT: {resume_from_checkpoint}")
    logger.info("=" * 60)

    # Step 1: Load datasets
    logger.info("STEP 1: Loading Datasets")
    train_dataset, val_dataset = load_datasets()

    # Step 2: Load processor
    logger.info("STEP 2: Loading Processor")
    processor = load_processor()

    # Step 3: Process datasets
    logger.info("STEP 3: Processing Datasets")
    train_dataset, val_dataset = process_datasets(train_dataset, val_dataset, processor)

    # Step 4: Setup model with LoRA
    logger.info("STEP 4: Setting up Model")
    model = setup_model()

    # Step 5: Train model
    logger.info("STEP 5: Training Model")
    model = train_model(
        model, processor, train_dataset, val_dataset,
        use_mixture=use_mixture,
        resume_from_checkpoint=resume_from_checkpoint,
        mixture_kwargs=mixture_kwargs
    )

    # Step 6: Save model
    logger.info("STEP 6: Saving Model")
    save_model(model, processor)

    # Step 7: Test model
    logger.info("STEP 7: Testing Model")
    test_model()

    logger.info("PIPELINE COMPLETE!")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VLM for justifiable cropping")
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    parser.add_argument(
        "--reward_type",
        type=str,
        default="mixture",
        choices=["mixture", "absolute"],
        help="Reward function type: mixture (4-delta + encouragement) or absolute"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    # Mixture reward parameters
    parser.add_argument(
        "--weight_mrr",
        type=float,
        default=0.2,
        help="Weight for delta MRR in mixture reward"
    )
    parser.add_argument(
        "--weight_ndcg",
        type=float,
        default=0.2,
        help="Weight for delta NDCG in mixture reward"
    )
    parser.add_argument(
        "--weight_rank",
        type=float,
        default=0.2,
        help="Weight for delta rank in mixture reward"
    )
    parser.add_argument(
        "--weight_margin",
        type=float,
        default=0.4,
        help="Weight for delta margin in mixture reward"
    )
    parser.add_argument(
        "--initial_encouragement",
        type=float,
        default=0.0,
        help="Initial encouragement bonus for cropping"
    )
    parser.add_argument(
        "--encouragement_decay_steps",
        type=int,
        default=5000,
        help="Number of steps for encouragement to decay to zero (linear decay)"
    )

    args = parser.parse_args()

    # Setup logging
    log_file = RUN_OUTPUT_DIR / "training.log"
    setup_logging(level=args.log_level, log_file=log_file)

    use_mixture = args.reward_type == "mixture"

    # Build mixture kwargs
    mixture_kwargs = {
        "weight_mrr": args.weight_mrr,
        "weight_ndcg": args.weight_ndcg,
        "weight_rank": args.weight_rank,
        "weight_margin": args.weight_margin,
        "initial_encouragement": args.initial_encouragement,
        "encouragement_decay_steps": args.encouragement_decay_steps,
    }

    main(
        use_mixture=use_mixture,
        resume_from_checkpoint=args.resume_from_checkpoint,
        mixture_kwargs=mixture_kwargs
    )
