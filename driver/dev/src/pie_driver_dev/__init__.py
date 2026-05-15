# Pie reference Python driver — `dev` driver.
#
# The flashinfer-based PyTorch driver new features iterate against
# before being ported to C++. Shared scaffolding (worker loop, Batch,
# shmem IPC, capabilities, launcher) lives in `._bridge`;
# this wheel layers its `Engine` and per-arch model code on top.
#
# Discovered by the standalone via `python -m pie_driver_dev` (see
# `__main__.py`); no in-process registry.
from __future__ import annotations
