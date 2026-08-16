#!/usr/bin/env bash
# =============================================================================
# install_env.sh — Environment Setup for TransDANN LIHC Pipeline
# =============================================================================
# Installs all dependencies needed for DANN model training + visualization.
# Run once on the Tencent Cloud machine before the main pipeline.
#
# Usage:
#   bash scripts/install_env.sh
# =============================================================================

set -e

echo "================================================================"
echo "  Environment Setup — TransDANN LIHC Pipeline"
echo "  Date: $(date)"
echo "================================================================"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

# ----- 1. Check Python -----
echo ""
echo "[1/4] Checking Python ..."
python3 --version || { echo "ERROR: Python3 not found"; exit 1; }

# ----- 2. Install PyTorch (with CUDA if available) -----
echo ""
echo "[2/4] Installing PyTorch ..."
pip3 install --upgrade pip

# Check CUDA
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo "")
    echo "  NVIDIA driver found: ${CUDA_VERSION:-unknown}"
    # Install PyTorch with CUDA 12.x support (compatible with CUDA 12.8)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
else
    echo "  No CUDA detected, installing CPU-only PyTorch"
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Verify PyTorch + CUDA
python3 -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"

# ----- 3. Install ML/data packages -----
echo ""
echo "[3/4] Installing ML & data packages ..."
cat requirements.txt | grep -v "^#" | grep -v "^$" | while read pkg; do
    pip3 install "$pkg" -q 2>&1 | tail -1 || true
done

# Additional packages needed
pip3 install scikit-survival>=0.22 -q 2>&1 | tail -1 || true
pip3 install lifelines>=0.27 -q 2>&1 | tail -1 || true

echo "  Installed packages:"
pip3 list 2>/dev/null | grep -iE "torch|numpy|pandas|sklearn|lifelines|matplotlib|seaborn|survival"

# ----- 4. Verify GPU access -----
echo ""
echo "[4/4] GPU verification ..."
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name}, {props.total_mem / 1e9:.1f}GB')
else:
    print('  ⚠ WARNING: CUDA not available — training will be slow on CPU!')
"

echo ""
echo "================================================================"
echo "  Environment setup complete!"
echo "================================================================"
