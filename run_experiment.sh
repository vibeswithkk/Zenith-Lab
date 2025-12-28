#!/bin/bash
set -e

echo "=== ZENITH Real-World Test Setup ==="

# 1. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# 2. Install Dependencies (Retry loop for robustness)
echo "Installing dependencies (High Bandwidth Required)..."
pip install --upgrade pip
# Check if torch is installed
if ! python -c "import torch" &> /dev/null; then
    echo "Installing PyTorch and friends..."
    pip install torch transformers peft trl datasets accelerate psutil bitsandbytes
fi

# 3. Install Zenith from local source
echo "Installing Zenith..."
if [ -d "ZENITH_Main" ]; then
    pip install -e ./ZENITH_Main
else
    echo "Cloning Zenith..."
    git clone https://github.com/vibeswithkk/ZENITH ZENITH_Main
    pip install -e ./ZENITH_Main
fi

# 4. Run Experiment
echo "=== Starting Fine-Tuning Experiment ==="
echo "Model: google/gemma-2b"
echo "Note: Ensure you are logged into Hugging Face if using gated models (huggingface-cli login)"

# Run without Zenith first (Baseline) - Optional
# echo "Running Baseline (No Zenith)..."
# python fine_tune_gemma.py --use_zenith  # Wait, logic is: flag enables it.

# Run WITH Zenith
echo "Running WITH Zenith Optimization..."
python fine_tune_gemma.py --use_zenith --model_name google/gemma-2b --epochs 1 --batch_size 1

echo "Experiment Complete."
