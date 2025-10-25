#!/bin/bash
# DeepSeek-OCR Server Installation Script
# Run this on your desktop with RTX 5090

set -e  # Exit on error

echo "=== DeepSeek-OCR Server Installation ==="
echo ""

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo "  CUDA Version: $CUDA_VERSION"
else
    echo "✗ No NVIDIA GPU detected"
    exit 1
fi

# Determine PyTorch index URL based on CUDA version
if [[ "$CUDA_VERSION" == "13"* ]] || [[ "$CUDA_VERSION" == "12"* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
    echo "  Using PyTorch cu124 build (compatible with CUDA 12.x and 13.x)"
elif [[ "$CUDA_VERSION" == "11.8"* ]]; then
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
    echo "  Using PyTorch for CUDA 11.8"
else
    echo "⚠ CUDA version $CUDA_VERSION detected, defaulting to cu124 build"
    TORCH_INDEX="https://download.pytorch.org/whl/cu124"
fi

echo ""
echo "Step 1/4: Installing PyTorch..."
pip install torch==2.9.0 torchvision --index-url $TORCH_INDEX

echo ""
echo "Step 2/4: Installing base requirements..."
pip install -r requirements.txt

echo ""
echo "Step 3/4: Installing build dependencies..."
pip install wheel packaging ninja

echo ""
echo "Step 4/5: Installing flash-attention (this may take a few minutes to compile)..."
pip install flash-attn==2.7.3 --no-build-isolation
pip install accelerate>-0.26.0

echo ""
echo "Step 5/5: Downloading DeepSeek-OCR model (~5GB)..."
python -u download_model.py

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "Start the server with:"
echo "  python server.py --host 0.0.0.0 --port 5555 --device cuda"
echo ""
