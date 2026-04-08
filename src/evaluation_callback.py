"""
Training callback for evaluating MRR improvement during training.

This callback runs evaluation after each checkpoint is saved to track
whether cropping improves MRR as training progresses.
"""
import os
import logging
import wandb
from transformers import TrainerCallback
from evaluate_cropping import CroppingEvaluator
from config import EVAL_OUTPUT_DIR, MODEL_ID

logger = logging.getLogger(__name__)


class MRREvaluationCallback(TrainerCallback):
    """
    Callback to evaluate MRR improvement during training.

    This callback:
    1. Runs after each checkpoint is saved
    2. Evaluates the checkpoint on validation data
    3. Logs MRR metrics to wandb and console
    4. Tracks improvement over training
    """

    def __init__(self, num_eval_samples=50, log_to_wandb=True):
        self.num_eval_samples = num_eval_samples
        self.log_to_wandb = log_to_wandb
        self.evaluation_history = []
        self._wandb_metrics_defined = False

    def on_save(self, args, state, control, **kwargs):
        """Called after a checkpoint is saved."""
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")

        if not os.path.exists(checkpoint_dir):
            logger.warning(f"Checkpoint directory not found: {checkpoint_dir}")
            return

        logger.info(f"Running MRR evaluation for checkpoint-{state.global_step}")

        try:
            os.makedirs(EVAL_OUTPUT_DIR, exist_ok=True)

            evaluator = CroppingEvaluator(
                model_path=checkpoint_dir,
                use_wandb=False,
                run_name=None,
                load_as_lora=True,
                base_model_path=MODEL_ID
            )

            stats = evaluator.evaluate_all(num_samples=self.num_eval_samples)
            evaluator.print_summary(stats)

            output_path = os.path.join(
                EVAL_OUTPUT_DIR,
                f"checkpoint_{state.global_step}.csv"
            )
            evaluator.save_detailed_results(output_path)
            evaluator.finish()

            eval_result = {
                'global_step': state.global_step,
                'epoch': state.epoch,
                **stats
            }
            self.evaluation_history.append(eval_result)

            if self.log_to_wandb and wandb.run is not None:
                if not self._wandb_metrics_defined:
                    wandb.define_metric("eval/*", step_metric="trainer/global_step")
                    self._wandb_metrics_defined = True

                eval_metrics = {
                    'trainer/global_step': state.global_step,
                    'eval/avg_original_mrr': stats['avg_original_mrr'],
                    'eval/avg_cropped_mrr': stats['avg_cropped_mrr'],
                    'eval/avg_mrr_improvement': stats['avg_mrr_improvement'],
                    'eval/avg_original_ndcg': stats['avg_original_ndcg'],
                    'eval/avg_cropped_ndcg': stats['avg_cropped_ndcg'],
                    'eval/avg_ndcg_improvement': stats['avg_ndcg_improvement'],
                    'eval/bbox_generation_rate': stats['bbox_generation_rate'],
                    'eval/num_improved': stats['num_improved'],
                    'eval/num_degraded': stats['num_degraded'],
                }

                if 'avg_mrr_improvement_with_bbox' in stats:
                    eval_metrics['eval/avg_mrr_improvement_with_bbox'] = stats['avg_mrr_improvement_with_bbox']
                    eval_metrics['eval/avg_ndcg_improvement_with_bbox'] = stats['avg_ndcg_improvement_with_bbox']

                wandb.log(eval_metrics, commit=True)

                wandb.run.summary["eval/latest_mrr_improvement"] = stats['avg_mrr_improvement']
                wandb.run.summary["eval/latest_cropped_mrr"] = stats['avg_cropped_mrr']
                wandb.run.summary["eval/latest_bbox_rate"] = stats['bbox_generation_rate']
                wandb.run.summary["eval/latest_step"] = state.global_step

                logger.info(f"Logged evaluation metrics to wandb at step {state.global_step}")

            logger.info(f"Evaluation complete for checkpoint-{state.global_step} | "
                       f"MRR Improvement: {stats['avg_mrr_improvement']:+.4f}")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}", exc_info=True)

    def on_train_end(self, args, state, control, **kwargs):
        """Print a summary of all evaluations at end of training."""
        if not self.evaluation_history:
            return

        logger.info("TRAINING EVALUATION HISTORY")
        logger.info(f"{'Step':<10} {'Epoch':<10} {'Original MRR':<15} {'Cropped MRR':<15} "
                    f"{'Improvement':<15} {'Bbox Rate':<12}")
        logger.info("-" * 90)

        for result in self.evaluation_history:
            logger.info(f"{result['global_step']:<10} "
                       f"{result['epoch']:<10.2f} "
                       f"{result['avg_original_mrr']:<15.4f} "
                       f"{result['avg_cropped_mrr']:<15.4f} "
                       f"{result['avg_mrr_improvement']:<+15.4f} "
                       f"{result['bbox_generation_rate']*100:<12.1f}%")

        if self.log_to_wandb and wandb.run is not None:
            import pandas as pd
            df = pd.DataFrame(self.evaluation_history)
            table = wandb.Table(dataframe=df)
            wandb.log({"eval/history_table": table})
