#!/bin/bash
# Build Pie CUDA Docker images (runtime + dev).

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ "$EUID" -eq 0 ]; then
    SUDO=""
else
    SUDO="sudo"
fi

echo "Building pie:cuda12.9 (runtime)..."
$SUDO docker build -f Dockerfile.cuda --target runtime -t pie:cuda12.9 -t pie:latest .

echo "Building pie:cuda12.9-dev (development)..."
$SUDO docker build -f Dockerfile.cuda --target development -t pie:cuda12.9-dev -t pie:dev .

echo ""
echo "Available images:"
$SUDO docker images | grep -E "^pie" || echo "(no pie images found)"
echo ""
echo "Run requires NVIDIA Container Toolkit, --gpus device=N (maps to cuda:0)"
echo "and --shm-size=2g (default /dev/shm SIGBUSes the engine↔driver buffer)."
echo ""
echo "  $SUDO docker run --gpus device=0 --shm-size=2g -d -p 8080:8080 \\"
echo "    -v ~/.cache:/root/.cache pie:latest"
echo ""
echo "With auth (pass an SSH public key):"
echo "  $SUDO docker run --gpus device=0 --shm-size=2g -d -p 8080:8080 \\"
echo "    -e PIE_AUTH_USER=\"myuser\" \\"
echo "    -e PIE_AUTH_KEY=\"\$(cat ~/.ssh/id_ed25519.pub)\" \\"
echo "    -v ~/.cache:/root/.cache pie:latest"
echo ""
echo "Pre-download a model:"
echo "  $SUDO docker run --rm --gpus device=0 --shm-size=2g \\"
echo "    -v ~/.cache:/root/.cache pie:latest pie model download \"Qwen/Qwen3-0.6B\""
