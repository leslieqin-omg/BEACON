#!/usr/bin/env bash
# ==============================================================================
# BEACON Environment Installation Script for NVIDIA RTX 4090 / CUDA 12.4 ~ 13.x
# Uses uv for ultra-fast, reproducible dependency management with Aliyun mirror.
# ==============================================================================

set -euo pipefail

echo "=========================================================="
echo " Starting BEACON environment setup with uv"
echo " Target Hardware: NVIDIA RTX 4090"
echo " CUDA Driver compatibility: CUDA 12.4+ / 12.6+ / 13.x"
echo " Mirror: Aliyun PyPI (https://mirrors.aliyun.com/pypi/simple)"
echo "=========================================================="

# 1. Setup high-speed Aliyun mirror and extended network timeout
export UV_INDEX_URL="https://mirrors.aliyun.com/pypi/simple"
export UV_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu124"
export UV_HTTP_TIMEOUT=120

# 2. Check for uv installation
if ! command -v uv &> /dev/null; then
    echo "[!] uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

echo "[+] Using uv: $(uv --version)"

# 3. Create isolated venv using Python 3.11
echo "[+] Creating virtual environment (.venv) with Python 3.11..."
uv venv .venv --python 3.11

# 4. Activate venv
source .venv/bin/activate

# 5. Install PyTorch matching CUDA 12.4 wheels (native sm_89 RTX 4090 Ada Lovelace)
echo "[+] Installing PyTorch (cu124 wheel) via uv..."
# Note: torchvision/torchaudio are omitted as BEACON is purely sequence/embedding based
uv pip install torch==2.5.1

# 6. Install project dependencies via Aliyun mirror
echo "[+] Installing scientific and language model dependencies..."
uv pip install     "numpy>=1.24,<2.0"     "pandas>=2.0,<3.0"     "scikit-learn>=1.3"     "scipy>=1.10"     "matplotlib>=3.7"     "pyyaml>=6.0"     "tqdm>=4.65"     "biopython>=1.81"     "transformers>=4.40"     "fair-esm>=2.0"     "antiberty>=0.1.3"

echo "[+] Installing beacon in editable mode..."
uv pip install -e .

# 7. Verify GPU and PyTorch setup
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
