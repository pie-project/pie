#! /bin/bash

set -e

ROOT="$(dirname "$(dirname "${BASH_SOURCE[0]}")")"

# Run pylint check.
# See `.pylintrc` for the configuration of the check.

PYTHONPATH=${ROOT}/backend/backend-python uv run \
    --project ${ROOT}/backend/backend-python \
    --with pylint \
    pylint --disable=R \
    ${ROOT}/backend/backend-python/config/*.py \
    ${ROOT}/backend/backend-python/model/*.py
