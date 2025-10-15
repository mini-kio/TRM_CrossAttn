#!/bin/bash
# Train TRM-CrossAttn on Maze-Hard dataset
# Using 4 GPUs (V100)

echo "=========================================="
echo "TRM-CrossAttn Training on Maze-Hard"
echo "=========================================="

# Set run name
run_name="trm_crossattn_maze_${RANDOM}"

echo "Run name: $run_name"
echo "Starting training..."

# Train with 4 GPUs
torchrun --nproc-per-node 4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --nnodes=1 \
  pretrain.py \
  arch=trm_crossattn \
  data_paths="[data/maze-30x30-hard-1k]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=4 \
  +run_name=${run_name} \
  ema=True

echo "=========================================="
echo "Training completed!"
echo "Results saved in: checkpoints/Maze-30x30-hard-1k-ACT-torch/${run_name}"
echo "=========================================="
