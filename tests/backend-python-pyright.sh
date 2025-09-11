#! /bin/bash

set -e

ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

# Run pylint check.
# See `pyrightconfig.json` for the configuration of the check.

PYTHONPATH=${ROOT}/backend/backend-python uv run \
    --project ${ROOT}/backend/backend-python \
    --with pyright \
    pyright \
    ${ROOT}/backend/backend-python/config/*.py \
    ${ROOT}/backend/backend-python/model/*.py
