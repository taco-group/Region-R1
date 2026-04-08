export CUDA_VISIBLE_DEVICES=7

python evaluate_cropping.py \
    --checkpoint ./runs/mixture_queryOnly_1222_2100/checkpoints/checkpoint-6600 \
    --split test \
    --viz_dir ./test_eval_results/ckpt-6600-viz \
    --output ./test_eval_results/infoseek_test_ckpt-6600.csv