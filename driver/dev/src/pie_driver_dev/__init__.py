# Pie reference Python driver — `dev` driver.
#
# After Phase 8, `pie-server`'s in-process driver registry is gone.
# Drivers are now discovered by the standalone via `python -m
# pie_driver_<flavor>` directly (see `driver/dev/src/pie_driver_dev/__main__.py`),
# so this module no longer auto-registers anything on import.
from __future__ import annotations
