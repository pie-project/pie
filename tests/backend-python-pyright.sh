#! /bin/bash

set -e

ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

# Run pyright check.
# See `pyrightconfig.json` for the configuration of the check.

PYTHONPATH=${ROOT}:${ROOT}/backend/backend-python:${ROOT}/backend/common_python uv run \
    --project ${ROOT}/backend/backend-python \
    --with pyright \
    pyright \
    ${ROOT}/backend/backend-python/__init__.py \
    ${ROOT}/backend/backend-python/adapter.py \
    ${ROOT}/backend/backend-python/backend_ops.py \
    ${ROOT}/backend/backend-python/common.py \
    ${ROOT}/backend/backend-python/handler.py \
    ${ROOT}/backend/backend-python/model_factory.py \
    ${ROOT}/backend/backend-python/profiler.py \
    ${ROOT}/backend/backend-python/server.py \
    ${ROOT}/backend/backend-python/model/*.py \
    ${ROOT}/backend/common_python/__init__.py \
    ${ROOT}/backend/common_python/adapter_import_utils.py \
    ${ROOT}/backend/common_python/debug_utils.py \
    ${ROOT}/backend/common_python/handler_common.py \
    ${ROOT}/backend/common_python/message.py \
    ${ROOT}/backend/common_python/model_loader.py \
    ${ROOT}/backend/common_python/server_common.py \
    ${ROOT}/backend/common_python/config/*.py \
    ${ROOT}/backend/common_python/common_model/*.py
