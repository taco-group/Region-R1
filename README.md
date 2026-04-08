# Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking (ACL 2026 Findings)

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2604.05268-b31b1b.svg)](https://arxiv.org/abs/2604.05268)

</div>

<div align="center">
  <img src="assets/Overview.png" alt="overview" width="800"/>
  <p><em>Figure 1. Overview of Region-R1: Given an image-question pair, Region-R1 adaptively decides whether to retain the full image or crop a question-relevant region before scoring retrieved candidates, improving multi-modal re-ranking.</em></p>
</div>

## News
- **[2026/04/07]** We released **Region-R1**, a reinforcement learning framework for query-side region cropping in multi-modal re-ranking. Check out our [paper](https://arxiv.org/abs/2604.05268) for more details.
- **[2026/04/06]** 🎊 Our paper has been accepted to **ACL 2026 (Findings)**!!!


## Key Highlights

- **Query-Side Region Cropping:** Formulates region selection as a decision-making problem — the model learns whether to retain the full image or focus on a question-relevant region to remove visual distractors before re-ranking.

- **r-GRPO:** Proposes region-aware group relative policy optimization with decision-balanced group sampling, handling the structured action space where the crop/no-crop decision is discrete while bounding boxes are continuous.

- **Candidate-Agnostic RL Framework:** Optimized directly for re-ranking objectives (MRR, NDCG, Rank, Margin) without requiring access to candidate images at decision time.

- **State-of-the-Art Results:** Achieves conditional Recall@1 improvements of up to **20% on E-VQA** and **8% on InfoSeek**, outperforming EchoSight and OMGM baselines.

## Installation

```bash
pip install torch torchvision
pip install transformers peft trl
pip install qwen-vl-utils
pip install datasets
pip install wandb
pip install Pillow matplotlib pandas numpy
pip install faiss-gpu  # or faiss-cpu
```

### Requirements

- Python 3.10+
- CUDA-capable GPU (tested on A100/H100)

## Data Preparation

Create a re-ranking dataset from InfoSeek or E-VQA knowledge base with EVA-CLIP hard negatives:

```bash
# Set dataset name: "InfoSeek" (default) or "EVQA"
export DATASET_NAME=InfoSeek

# Set external data path (optional, defaults to ~/Desktop/<DATASET_NAME>-data)
export DATA_ROOT=/path/to/dataset-data

python src/prepare_data.py --num_entries 1000 --visualize_samples 20
```

| Argument | Default | Description |
|---|---|---|
| `--num_entries` | 1000 | Number of dataset entries to create |
| `--similarity_tolerance` | 0.05 | Hard negative similarity tolerance |
| `--visualize_samples` | 20 | Number of sample visualizations to generate |
| `--device` | `cuda:0` | CUDA device |

**Output:** `Reranker_Dataset/<DATASET_NAME>_top10_parquet/` (train.parquet, val.parquet)

## Model

Region-R1 uses the following models:

| Component | Model | Role |
|---|---|---|
| Policy Network | [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) | Decides crop/no-crop and predicts bounding boxes (fine-tuned with LoRA) |
| Scoring Model | [EVA-CLIP-8B](https://huggingface.co/BAAI/EVA-CLIP-8B) | Computes cosine similarity for re-ranking reward (frozen) |

## Training

```bash
python main.py --reward_type mixture
```

| Argument | Default | Description |
|---|---|---|
| `--reward_type` | `mixture` | Reward function: `mixture` (recommended) or `absolute` |
| `--resume_from_checkpoint` | None | Path to checkpoint directory to resume from |
| `--log_level` | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |
| `--weight_mrr` | 0.2 | Weight for delta-MRR in mixture reward |
| `--weight_ndcg` | 0.2 | Weight for delta-NDCG |
| `--weight_rank` | 0.2 | Weight for delta-rank |
| `--weight_margin` | 0.4 | Weight for delta-margin |
| `--initial_encouragement` | 0.0 | Initial encouragement bonus for cropping |
| `--encouragement_decay_steps` | 5000 | Steps for encouragement to decay to zero |

Training logs to both TensorBoard and Weights & Biases. Checkpoints are saved to `runs/<RUN_NAME>/checkpoints/`.

### Reward Functions

**Absolute** (`--reward_type absolute`): Direct MRR/NDCG score of the cropped image against candidates.

**Mixture** (`--reward_type mixture`): Weighted combination of four delta metrics (crop vs. baseline):
- **Delta-MRR:** Change in Mean Reciprocal Rank
- **Delta-NDCG:** Change in Normalized Discounted Cumulative Gain
- **Delta-Rank:** Improvement in rank of the correct candidate (log scale)
- **Delta-Margin:** Change in similarity gap between positive and hardest negative

## Evaluation

Evaluate a single checkpoint:

```bash
python src/evaluate_cropping.py --checkpoint runs/<run_name>/checkpoints/checkpoint-XXX
```

Or use the convenience wrapper:

```bash
python src/run_eval.py --checkpoint runs/<run_name>/checkpoints/checkpoint-XXX --num_samples 100
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint` | Final model | Path to checkpoint to evaluate |
| `--num_samples` | All | Number of validation samples |
| `--split` | `val` | Dataset split (`val` or `test`) |
| `--output` | Auto | Output CSV path |
| `--viz_dir` | Auto | Directory for improvement visualizations |
| `--wandb` | Off | Log results to W&B |

Example with shell script:

```bash
export CUDA_VISIBLE_DEVICES=7
bash src/run_eval.sh
```

## Results

### Re-Ranking Performance (Top-20 Candidates)

| Method | E-VQA MRR | E-VQA R@1 | InfoSeek MRR | InfoSeek R@1 |
|--------|-----------|-----------|--------------|--------------|
| EVA-CLIP | 0.224 | 14.2% | 0.553 | 46.3% |
| EchoSight | 0.402 | 36.5% | 0.586 | 53.2% |
| OMGM | 0.473 | 42.8% | 0.681 | 64.0% |
| **Region-R1** | **0.473** | **44.7%** | **0.706** | **66.5%** |

### Conditional Recall@1

| Method | E-VQA CondR@1 | InfoSeek CondR@1 |
|--------|---------------|------------------|
| EchoSight | 0.75 | 0.68 |
| OMGM | 0.73 | 0.75 |
| **Region-R1** | **0.90** | **0.81** |

## Project Structure

```
Region-R1/
├── main.py                          # Training entry point
├── config.py                        # Configuration (paths, hyperparameters, prompts)
├── utils.py                         # CLIP scoring, bbox parsing, MRR/NDCG metrics
└── src/
    ├── model.py                     # Model loading (Qwen2.5-VL-3B + LoRA, EVA-CLIP-8B)
    ├── data.py                      # Lazy-loading dataset from parquet
    ├── train.py                     # r-GRPO training loop
    ├── rewards.py                   # Reward functions (absolute, mixture)
    ├── inference.py                 # Post-training inference/testing
    ├── logging_config.py            # Logging setup
    ├── prepare_data.py              # Dataset preparation (InfoSeek or E-VQA)
    ├── evaluate_cropping.py         # Checkpoint evaluation (MRR improvement)
    ├── evaluation_callback.py       # Training callback for eval during training
    ├── training_logger_callback.py  # Training callback for metrics logging
    ├── run_eval.py                  # Convenience evaluation wrapper
    └── run_eval.sh                  # Shell script for evaluation
```

## Configuration

All configuration is centralized in `config.py`:

| Setting | Description |
|---|---|
| `RUN_NAME` | Experiment name (determines output directory) |
| `MODEL_ID` | Base VLM model (`Qwen/Qwen2.5-VL-3B-Instruct`) |
| `EVACLIP_MODEL_ID` | Reward model (`BAAI/EVA-CLIP-8B`) |
| `TRAINING_CONFIG` | Learning rate, epochs, batch size, etc. |
| `LORA_CONFIG` | LoRA rank, alpha, dropout, target modules |
| `WANDB_CONFIG` | W&B project, entity, tags |
| `SYSTEM_PROMPT` | Prompt template for the cropping assistant |

## Citation

If you find this work helpful, please consider citing our paper:

```bibtex
@article{hu2026region,
  title={Region-R1: Reinforcing Query-Side Region Cropping for Multi-Modal Re-Ranking},
  author={Hu, Chan-Wei and Tu, Zhengzhong},
  journal={arXiv preprint arXiv:2604.05268},
  year={2026}
}
```
