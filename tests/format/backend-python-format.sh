#!/bin/bash
set -e

ROOT="$(dirname "$(dirname "$(dirname "${BASH_SOURCE[0]}")")")"

# Set PYTHONPATH so imports work
export PYTHONPATH="${ROOT}:${PYTHONPATH}"

uvx black --check \
    ${ROOT}/backend/backend-python/__init__.py \
    ${ROOT}/backend/backend-python/adapter.py \
    ${ROOT}/backend/backend-python/adapter_utils.py \
    ${ROOT}/backend/backend-python/debug_utils.py \
    ${ROOT}/backend/backend-python/handler.py \
    ${ROOT}/backend/backend-python/message.py \
    ${ROOT}/backend/backend-python/model_factory.py \
    ${ROOT}/backend/backend-python/model_loader.py \
    ${ROOT}/backend/backend-python/platform_detection.py \
    ${ROOT}/backend/backend-python/profiler/*.py \
    ${ROOT}/backend/backend-python/server.py \
    ${ROOT}/backend/backend-python/model/*.py \
    ${ROOT}/backend/backend-python/metal_kernels/*.py \
    ${ROOT}/backend/backend-python/metal_kernels/_internal/*.py
