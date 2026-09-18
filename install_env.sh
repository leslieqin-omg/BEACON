#!/usr/bin/env bash
# ==============================================================================
# BEACON Environment Installation Script for RTX 4090 / CUDA 12.4 ~ 13.x
# Uses uv for ultra-fast, reproducible dependency management.
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Starting BEACON environment setup with uv"
echo " Target Hardware: NVIDIA RTX 4090"
echo " CUDA Driver compatibility: CUDA 12.4+ / 12.6+ / 13.x"
echo "=========================================================="

# 1. Check for uv installation
if ! command -v uv &> /dev/null; then
    echo "[!] uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "[+] Using uv: $(uv --version)"

# 2. Create isolated venv using Python 3.11 (optimal for PyTorch 2.5 + scientific stack)
echo "[+] Creating virtual environment (.venv) with Python 3.11..."
uv venv .venv --python 3.11

# 3. Activate venv
source .venv/bin/activate

# 4. Install PyTorch matching CUDA 12.4/12.6 wheels (fully compatible with Driver CUDA 13.x)
echo "[+] Installing PyTorch (cu124/cu126 wheel) via uv..."
# PyTorch 2.5.1+cu124 runs with native sm_89 (RTX 4090 Ada Lovelace) architecture support
uv pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Install project dependencies and develop editable package
echo "[+] Installing scientific and language model dependencies..."
uv pip install     "numpy>=1.24,<2.0"     "pandas>=2.0,<3.0"     "scikit-learn>=1.3"     "scipy>=1.10"     "matplotlib>=3.7"     "pyyaml>=6.0"     "tqdm>=4.65"     "biopython>=1.81"     "transformers>=4.40"     "fair-esm>=2.0"     "antiberty>=0.1.3"

echo "[+] Installing beacon in editable mode..."
uv pip install -e .

# 6. Verify GPU and PyTorch setup
echo "=========================================================="
echo " Verifying Environment and RTX 4090 CUDA acceleration..."
echo "=========================================================="
python -c "
import torch
print(f'PyTorch Version : {torch.__version__}')
print(f'CUDA Available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version    : {torch.version.cuda}')
    print(f'Device Count    : {torch.cuda.device_count()}')
    print(f'Device Name     : {torch.cuda.get_device_name(0)}')
    print(f'Device Compute  : {torch.cuda.get_device_capability(0)}')
    # Quick tensor test on device
    x = torch.randn(1024, 1024, device='cuda')
    y = x @ x
    print('CUDA Matrix Multiplication Test: SUCCESS (RTX 4090 ready)')
else:
    print('[!] WARNING: CUDA not detected. Ensure NVIDIA drivers are loaded.')
"

echo "=========================================================="
echo " BEACON Environment Installation Complete!"
echo " To activate run: source .venv/bin/activate"
echo "=========================================================="

