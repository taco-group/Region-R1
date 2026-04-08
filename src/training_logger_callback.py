"""
Custom callback to ensure all training metrics (rewards, losses) are logged to wandb.

GRPO-specific metrics that should be logged:
- rewards/mean: Average reward across all generations
- rewards/max: Maximum reward
- rewards/min: Minimum reward
- loss: Training loss
- policy_loss: Policy gradient loss
- advantages/mean: Average advantages
- learning_rate: Current learning rate
"""
import logging
import wandb
from transformers import TrainerCallback

logger = logging.getLogger(__name__)


class TrainingMetricsCallback(TrainerCallback):
    """
    Callback to ensure all GRPO training metrics are logged to wandb.

    This callback captures metrics from the trainer's log_history and
    ensures they're properly sent to wandb, including GRPO-specific
    reward metrics.
    """

    def __init__(self, log_rewards=True, log_advantages=True):
        """
        Initialize the callback.

        Args:
            log_rewards: Whether to log reward statistics (default: True)
            log_advantages: Whether to log advantage statistics (default: True)
        """
        self.log_rewards = log_rewards
        self.log_advantages = log_advantages
        self.last_logged_step = -1
        self._metrics_defined = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when the trainer logs metrics.

        Args:
            args: Training arguments
            state: Trainer state
            control: Trainer control
            logs: Dictionary of logged metrics
            **kwargs: Additional arguments
        """
        if logs is None or not wandb.run:
            return

        # Define metrics on first use
        if not self._metrics_defined:
            wandb.define_metric("train/*", step_metric="trainer/global_step")
            wandb.define_metric("eval/*", step_metric="trainer/global_step")
            self._metrics_defined = True

        # Avoid duplicate logging
        current_step = state.global_step
        if current_step == self.last_logged_step:
            return

        self.last_logged_step = current_step

        # Prepare metrics to log
        metrics_to_log = {"trainer/global_step": current_step}

        # === Core Training Metrics ===
        if "loss" in logs:
            metrics_to_log["train/loss"] = logs["loss"]

        if "learning_rate" in logs:
            metrics_to_log["train/learning_rate"] = logs["learning_rate"]

        if "epoch" in logs:
            metrics_to_log["train/epoch"] = logs["epoch"]

        # === GRPO-Specific Metrics ===

        # Reward metrics
        if self.log_rewards:
            # Average reward
            if "rewards/mean" in logs:
                metrics_to_log["train/rewards/mean"] = logs["rewards/mean"]
            elif "reward" in logs:
                metrics_to_log["train/rewards/mean"] = logs["reward"]

            # Reward statistics
            if "rewards/max" in logs:
                metrics_to_log["train/rewards/max"] = logs["rewards/max"]
            if "rewards/min" in logs:
                metrics_to_log["train/rewards/min"] = logs["rewards/min"]
            if "rewards/std" in logs:
                metrics_to_log["train/rewards/std"] = logs["rewards/std"]

            # Individual reward functions (if using multiple)
            for key in logs:
                if key.startswith("rewards/"):
                    metrics_to_log[f"train/{key}"] = logs[key]

        # Advantage metrics
        if self.log_advantages:
            if "advantages/mean" in logs:
                metrics_to_log["train/advantages/mean"] = logs["advantages/mean"]
            if "advantages/std" in logs:
                metrics_to_log["train/advantages/std"] = logs["advantages/std"]

        # Policy loss
        if "policy_loss" in logs:
            metrics_to_log["train/policy_loss"] = logs["policy_loss"]

        # Value loss (if using value function)
        if "value_loss" in logs:
            metrics_to_log["train/value_loss"] = logs["value_loss"]

        # KL divergence (if tracked)
        if "kl" in logs:
            metrics_to_log["train/kl_divergence"] = logs["kl"]
        if "kl/mean" in logs:
            metrics_to_log["train/kl/mean"] = logs["kl/mean"]

        # Entropy (policy exploration measure)
        if "entropy" in logs:
            metrics_to_log["train/entropy"] = logs["entropy"]

        # Gradient norm (useful for debugging)
        if "grad_norm" in logs:
            metrics_to_log["train/grad_norm"] = logs["grad_norm"]

        # === Generation Metrics ===
        if "generation/length_mean" in logs:
            metrics_to_log["train/generation/length_mean"] = logs["generation/length_mean"]
        if "generation/length_max" in logs:
            metrics_to_log["train/generation/length_max"] = logs["generation/length_max"]

        # === Prompt/Input Token Length Metrics ===
        if "prompt/length_mean" in logs:
            metrics_to_log["train/prompt/length_mean"] = logs["prompt/length_mean"]
        if "prompt/length_max" in logs:
            metrics_to_log["train/prompt/length_max"] = logs["prompt/length_max"]
        if "prompt_length" in logs:
            metrics_to_log["train/prompt_length"] = logs["prompt_length"]
        # Also check for alternative naming conventions
        if "input_length" in logs:
            metrics_to_log["train/input_length"] = logs["input_length"]
        if "sequence_length" in logs:
            metrics_to_log["train/sequence_length"] = logs["sequence_length"]

        # === Log everything we found ===
        if metrics_to_log:
            wandb.log(metrics_to_log, step=current_step)

            # Log key metrics to console for monitoring
            parts = []
            if "train/rewards/mean" in metrics_to_log:
                parts.append(f"Reward: {metrics_to_log['train/rewards/mean']:.4f}")
            if "train/loss" in metrics_to_log:
                parts.append(f"Loss: {metrics_to_log['train/loss']:.4f}")
            if "train/learning_rate" in metrics_to_log:
                parts.append(f"LR: {metrics_to_log['train/learning_rate']:.2e}")
            if parts:
                logger.info(f"[Step {current_step}] {' | '.join(parts)}")

        # Also log all other metrics that might be in logs
        # (in case there are additional metrics we didn't explicitly handle)
        other_metrics = {}
        for key, value in logs.items():
            if key not in ["loss", "learning_rate", "epoch", "grad_norm"] and \
               not key.startswith("rewards/") and \
               not key.startswith("advantages/") and \
               not key.startswith("generation/") and \
               not key.startswith("prompt/") and \
               key not in ["policy_loss", "value_loss", "kl", "entropy",
                           "prompt_length", "input_length", "sequence_length"]:
                # Log with train/ prefix if not already there
                log_key = f"train/{key}" if not key.startswith("train/") else key
                other_metrics[log_key] = value

        if other_metrics and wandb.run:
            wandb.log(other_metrics, step=current_step)

    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        if wandb.run:
            logged = ["Training loss", "Reward statistics (mean, max, min, std)",
                      "Learning rate", "Policy/value losses", "Generation metrics",
                      "Input token length metrics"]
            if self.log_advantages:
                logged.append("Advantage statistics")
            logger.info(f"Training Metrics Callback enabled. Logging: {', '.join(logged)}")
