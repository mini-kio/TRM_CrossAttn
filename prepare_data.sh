#!/bin/bash
# Prepare datasets for TRM-CrossAttn training

echo "=========================================="
echo "TRM-CrossAttn Dataset Preparation"
echo "=========================================="

# Create data directory
mkdir -p data

# 1. Maze-Hard dataset
echo ""
echo "[1/2] Building Maze-Hard dataset..."
echo "  Output: data/maze-30x30-hard-1k"
python3 dataset/build_maze_dataset.py

if [ $? -eq 0 ]; then
  echo "  ✓ Maze-Hard dataset created successfully!"
else
  echo "  ✗ Failed to create Maze-Hard dataset"
  exit 1
fi

# 2. Sudoku-Extreme dataset
echo ""
echo "[2/2] Building Sudoku-Extreme dataset..."
echo "  Output: data/sudoku-extreme-1k-aug-1000"
python3 dataset/build_sudoku_dataset.py \
  --output-dir data/sudoku-extreme-1k-aug-1000 \
  --subsample-size 1000 \
  --num-aug 1000

if [ $? -eq 0 ]; then
  echo "  ✓ Sudoku-Extreme dataset created successfully!"
else
  echo "  ✗ Failed to create Sudoku-Extreme dataset"
  exit 1
fi

echo ""
echo "=========================================="
echo "Dataset preparation completed!"
echo "=========================================="
echo ""
echo "Available datasets:"
echo "  - data/maze-30x30-hard-1k (for Maze-Hard)"
echo "  - data/sudoku-extreme-1k-aug-1000 (for Sudoku-Extreme)"
echo ""
echo "Next steps:"
echo "  1. Train on Maze-Hard: bash train_maze.sh"
echo "  2. Train on Sudoku: bash train_sudoku.sh"
echo "=========================================="
