#!/usr/bin/env python3
"""
Simple script to run evaluation on the trained model with validation data.
This is a convenience wrapper around evaluate_cropping.py
"""
import os
import argparse
from evaluate_cropping import CroppingEvaluator
from config import FINAL_MODEL_DIR, MODEL_ID, RUN_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained cropping model on validation data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate final model on all validation data:
  python run_eval.py

  # Evaluate final model on first 50 samples:
  python run_eval.py --num_samples 50

  # Evaluate specific checkpoint:
  python run_eval.py --checkpoint runs/exp_improvement_reward/checkpoints/checkpoint-380

  # Evaluate with wandb logging:
  python run_eval.py --wandb

  # Evaluate and save detailed results:
  python run_eval.py --save_results eval_results.csv
        """
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help=f'Path to checkpoint to evaluate (default: {FINAL_MODEL_DIR})'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=None,
        help='Number of validation samples to evaluate (default: all)'
    )
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Log results to Weights & Biases'
    )
    parser.add_argument(
        '--run_name',
        type=str,
        default=None,
        help='Custom run name for wandb logging'
    )
    parser.add_argument(
        '--save_results',
        type=str,
        default=None,
        help='Path to save detailed results CSV (default: auto-generated in run output dir)'
    )

    args = parser.parse_args()

    # Determine which checkpoint to evaluate
    checkpoint_path = args.checkpoint or FINAL_MODEL_DIR

    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")

        # List available checkpoints
        run_checkpoint_dir = os.path.join(RUN_OUTPUT_DIR, "checkpoints")
        if os.path.exists(run_checkpoint_dir):
            checkpoints = sorted([
                d for d in os.listdir(run_checkpoint_dir)
                if d.startswith('checkpoint-')
            ])
            for ckpt in checkpoints:
                print(f"  - {os.path.join(run_checkpoint_dir, ckpt)}")

        if os.path.exists(FINAL_MODEL_DIR):
            print(f"  - {FINAL_MODEL_DIR} (final model)")

        return

    print("="*80)
    print("Running Evaluation on Validation Data")
    print("="*80)
    print(f"\nModel checkpoint: {checkpoint_path}")
    print(f"Num samples: {args.num_samples or 'all'}")
    print(f"Wandb logging: {args.wandb}")
    print()

    # Initialize evaluator
    # Note: The final_model is a LoRA adapter, so we need to load it with load_as_lora=True
    is_lora = os.path.exists(os.path.join(checkpoint_path, "adapter_config.json"))

    evaluator = CroppingEvaluator(
        model_path=checkpoint_path,
        use_wandb=args.wandb,
        run_name=args.run_name or f"eval-{os.path.basename(checkpoint_path)}",
        load_as_lora=is_lora,  # Auto-detect if it's a LoRA adapter
        base_model_path=MODEL_ID if is_lora else None
    )

    # Run evaluation
    stats = evaluator.evaluate_all(num_samples=args.num_samples)

    # Print summary
    evaluator.print_summary(stats)

    # Save detailed results
    if args.save_results:
        output_path = args.save_results
    else:
        checkpoint_name = os.path.basename(checkpoint_path)
        eval_output_dir = os.path.join(RUN_OUTPUT_DIR, "evaluations")
        os.makedirs(eval_output_dir, exist_ok=True)
        output_path = os.path.join(eval_output_dir, f"eval_results_{checkpoint_name}.csv")

    evaluator.save_detailed_results(output_path)

    # Finish
    evaluator.finish()

    print("\nEvaluation complete!")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
