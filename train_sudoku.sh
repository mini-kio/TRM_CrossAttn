#!/bin/bash
# Train TRM-CrossAttn on Sudoku-Extreme dataset
# Using 1 GPU for quick testing

echo "=========================================="
echo "TRM-CrossAttn Training on Sudoku-Extreme"
echo "=========================================="

# Set run name
run_name="trm_crossattn_sudoku_${RANDOM}"

echo "Run name: $run_name"
echo "Starting training..."

# Train with 1 GPU (quick test)
python pretrain.py \
  arch=trm_crossattn \
  data_paths="[data/sudoku-extreme-1k-aug-1000]" \
  evaluators="[]" \
  epochs=50000 \
  eval_interval=5000 \
  lr=1e-4 \
  puzzle_emb_lr=1e-4 \
  weight_decay=1.0 \
  puzzle_emb_weight_decay=1.0 \
  arch.L_layers=2 \
  arch.H_cycles=3 \
  arch.L_cycles=6 \
  +run_name=${run_name} \
  ema=True

echo "=========================================="
echo "Training completed!"
echo "Results saved in: checkpoints/Sudoku-extreme-1k-aug-1000-ACT-torch/${run_name}"
echo "=========================================="
