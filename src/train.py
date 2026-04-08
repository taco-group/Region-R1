"""
Training logic using GRPO
"""
import logging
import wandb
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from config import MODEL_OUTPUT_DIR, TRAINING_CONFIG, FINAL_MODEL_DIR, WANDB_CONFIG
from rewards import get_reward_functions, update_training_step, calculate_encouragement_bonus
from utils import set_clip_model
from model import load_evaclip_model

logger = logging.getLogger(__name__)


class TrainingStepCallback(TrainerCallback):
    """Callback to update the global training step for decaying encouragement reward."""

    def __init__(self, initial_encouragement=0.1, decay_steps=1000):
        super().__init__()
        self.initial_encouragement = initial_encouragement
        self.decay_steps = decay_steps

    def on_step_begin(self, args, state, control, **kwargs):
        """Update global step at the beginning of each training step."""
        update_training_step(state.global_step)

        # Log encouragement bonus to wandb
        encouragement_bonus = calculate_encouragement_bonus(
            initial_bonus=self.initial_encouragement,
            decay_steps=self.decay_steps
        )

        if wandb.run is not None:
            wandb.log({
                "reward/encouragement_bonus": encouragement_bonus,
                "reward/encouragement_decay_pct": encouragement_bonus / self.initial_encouragement * 100 if self.initial_encouragement > 0 else 0,
            }, commit=False)

        return control


def create_training_config():
    """Create and return GRPOConfig for training."""
    logger.info("Configuring training arguments...")

    training_args = GRPOConfig(
        output_dir=MODEL_OUTPUT_DIR,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        lr_scheduler_type=TRAINING_CONFIG.get("lr_scheduler_type", "linear"),
        warmup_steps=TRAINING_CONFIG.get("warmup_steps", 0),
        remove_unused_columns=False,
        num_train_epochs=TRAINING_CONFIG["num_train_epochs"],
        bf16=TRAINING_CONFIG["bf16"],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        per_device_train_batch_size=TRAINING_CONFIG["per_device_train_batch_size"],
        max_completion_length=TRAINING_CONFIG["max_completion_length"],
        num_generations=TRAINING_CONFIG["num_generations"],
        max_prompt_length=None,
        scale_rewards=False,
        beta=TRAINING_CONFIG.get("beta", 0.1),
        report_to=["tensorboard", "wandb"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        logging_first_step=True,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=TRAINING_CONFIG["save_steps"],
        run_name=WANDB_CONFIG["name"],
        logging_nan_inf_filter=False,
        log_level="info",
    )

    return training_args


def setup_clip_for_rewards():
    """
    Load EVA-CLIP model and set it in the rewards module.
    This must be called before training starts.
    """
    logger.info("Setting up EVA-CLIP for reward calculation")

    evaclip_model, evaclip_processor = load_evaclip_model()
    set_clip_model(evaclip_model, evaclip_processor)

    logger.info("EVA-CLIP ready for reward calculation")


def train_model(model, processor, train_dataset, val_dataset,
                use_mixture=False, enable_eval_callback=True, resume_from_checkpoint=None,
                mixture_kwargs=None):
    """
    Train the model using GRPO.

    Args:
        model: The model to train
        processor: The processor for tokenization
        train_dataset: Training dataset
        val_dataset: Validation dataset
        use_mixture: If True, use mixture reward with 4 deltas + decaying encouragement
        enable_eval_callback: If True, run MRR evaluation after each checkpoint
        resume_from_checkpoint: Path to checkpoint directory to resume from
        mixture_kwargs: Optional dict with mixture reward parameters:
            - weight_mrr (default: 0.3)
            - weight_ndcg (default: 0.3)
            - weight_rank (default: 0.2)
            - weight_margin (default: 0.2)
            - initial_encouragement (default: 0.1)
            - encouragement_decay_steps (default: 1000)

    Returns:
        Trained model
    """
    setup_clip_for_rewards()

    logger.info("Initializing GRPOTrainer...")
    if use_mixture:
        reward_type = "Mixture (4-delta + decaying encouragement)"
    else:
        reward_type = "Absolute"
    logger.info(f"Reward Type: {reward_type}")

    training_args = create_training_config()

    # Prepare kwargs for mixture reward
    reward_kwargs = mixture_kwargs or {}
    reward_funcs = get_reward_functions(
        use_mixture=use_mixture,
        **reward_kwargs
    )

    callbacks = []

    # Add training step callback for mixture reward (updates global step and logs to wandb)
    if use_mixture:
        step_callback = TrainingStepCallback(
            initial_encouragement=reward_kwargs.get('initial_encouragement', 0.3),
            decay_steps=reward_kwargs.get('encouragement_decay_steps', 5000)
        )
        callbacks.append(step_callback)
        logger.info("Training step callback enabled for encouragement decay (logging to wandb)")

    from training_logger_callback import TrainingMetricsCallback
    metrics_callback = TrainingMetricsCallback(
        log_rewards=True,
        log_advantages=True
    )
    callbacks.append(metrics_callback)
    logger.info("Training metrics callback enabled")

    if enable_eval_callback:
        from evaluation_callback import MRREvaluationCallback
        eval_callback = MRREvaluationCallback(
            num_eval_samples=None,
            log_to_wandb=True
        )
        callbacks.append(eval_callback)
        logger.info("MRR evaluation callback enabled")

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    if resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    logger.info("Training complete!")

    return model


def save_model(model, processor):
    """Save the trained model and processor."""
    logger.info(f"Saving final model to {FINAL_MODEL_DIR}...")
    model.save_pretrained(FINAL_MODEL_DIR)
    processor.save_pretrained(FINAL_MODEL_DIR)
    logger.info("Model saved successfully!")
