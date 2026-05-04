"""Shared launcher infrastructure for pie's Python drivers.

Each driver's `__main__.py` is a thin shim around `launch()` below:

    # driver/<flavor>/src/pie_driver_<flavor>/__main__.py
    from pie_driver_dev._launcher import launch
    from . import worker
    from .config import VllmDriverConfig

    if __name__ == "__main__":
        raise SystemExit(launch(
            prog="pie_driver_vllm",
            config_cls=VllmDriverConfig,
            worker=worker,
        ))

This file lives in `pie-driver-dev` rather than `pie-rpc` because vllm
and sglang already depend on `pie-driver-dev` for `worker.run_worker`,
`batching`, etc. — shipping the launcher helper from the same wheel is
honest about that existing edge instead of inventing a fourth package.

## Standalone-emitted TOML schema

The standalone's `crate::subprocess_driver::write_subprocess_startup_toml`
emits this shape:

    [model]
    name = "..."
    hf_repo = "..."
    snapshot_dir = "..."

    [driver]
    device = ["cuda:0"]
    tensor_parallel_size = 1
    activation_dtype = "bfloat16"
    random_seed = 42
    master_port = 29500
    ready_timeout_s = 1200.0

    [driver.options]
    # driver-specific config_cls fields

    [telemetry]
    enabled = false
    endpoint = ""
    service_name = ""

## Handshake JSON shape

One line per DP-replica leader, terminated by a `ready` sentinel:

    {"group_id": 0, "server_name": "/tmp/.ipc-...",
     "shmem_name": "/pie_shmem_g0", "caps": {<DriverCapabilities flat dict>}}
    {"ready": true, "num_groups": 1}

The standalone's parser is `crate::subprocess_driver::read_handshake_for_group`.
**If you change the JSON shape, change both sides** — the docstring at
the top of `subprocess_driver.rs` calls out this contract too.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import queue
import signal
import sys
import time
from dataclasses import asdict
from typing import Any


def _read_toml(path: str) -> dict:
    """Read a TOML file. Python 3.11+ has tomllib; fall back to `toml`
    for older interpreters since the launcher may run inside a venv
    whose pinned Python version we don't control.
    """
    try:
        import tomllib
        with open(path, "rb") as f:
            return tomllib.load(f)
    except ImportError:
        import toml
        with open(path, "r") as f:
            return toml.load(f)


def _validate_devices(devices: list[str], available_gpus: int) -> None:
    for dev in devices:
        if dev and dev.startswith("cuda:"):
            idx = int(dev.split(":")[1])
            if idx >= available_gpus:
                raise RuntimeError(
                    f"device {dev!r} not accessible — only {available_gpus} GPU(s) "
                    f"visible (cuda:0..cuda:{available_gpus - 1}). "
                    f"Check CUDA_VISIBLE_DEVICES."
                )


def _write_handshake(fd: int, payload: dict) -> None:
    """Write one JSON line to the parent's pipe."""
    line = (json.dumps(payload) + "\n").encode("utf-8")
    os.write(fd, line)


def launch(*, prog: str, config_cls, worker) -> int:
    """Run the standalone-side launcher lifecycle for one driver.

    Args:
        prog: argparse program name + log prefix on worker-died messages
            (e.g. `"pie_driver_vllm"`).
        config_cls: the driver's options dataclass (`NativeDriverConfig`,
            `VllmDriverConfig`, `SGLangDriverConfig`). Constructed from
            `[driver.options]` in the launcher TOML.
        worker: the driver's `worker` module. Must export
            `calculate_topology(world_size, tp_degree)` and
            `worker_main(local_rank, world_size, devices, master_port,
            model_config, driver_config, group_topology, ready_queue)`.

    Returns:
        Exit code (0 on graceful shutdown). Raises SystemExit for hard
        config errors (bad options, no devices, worker startup failure).
    """
    parser = argparse.ArgumentParser(prog=prog)
    parser.add_argument(
        "--config",
        required=True,
        help="path to standalone-emitted driver TOML",
    )
    parser.add_argument(
        "--handshake-fd",
        type=int,
        required=True,
        help="parent-owned pipe fd; launcher writes one JSON line per "
             "ready group, then a sentinel",
    )
    args = parser.parse_args()

    cfg = _read_toml(args.config)

    model_section = cfg.get("model", {})
    driver_section = cfg.get("driver", {})
    driver_options_section = driver_section.get("options", {}) or {}
    telemetry_section = cfg.get("telemetry", {}) or {}

    try:
        driver_options = config_cls(**driver_options_section)
    except TypeError as e:
        allowed = [f.name for f in dataclasses.fields(config_cls)]
        raise SystemExit(
            f"invalid keys in [driver.options] — {e}. Allowed fields: {allowed}"
        )

    devices: list[str] = list(driver_section.get("device", []))
    if not devices:
        raise SystemExit("[driver].device must be a non-empty list")
    world_size = len(devices)

    tp_degree = int(driver_section.get("tensor_parallel_size", 0)) or world_size
    master_port = int(driver_section.get("master_port", 29500))
    activation_dtype = driver_section.get("activation_dtype", "bfloat16")
    random_seed = int(driver_section.get("random_seed", 42))

    import torch
    available_gpus = torch.cuda.device_count()
    _validate_devices(devices, available_gpus)

    group_topology = worker.calculate_topology(world_size, tp_degree)
    num_groups = len(group_topology)

    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    spawn_ctx = mp.get_context("spawn")

    model_config_dict: dict[str, Any] = {
        "hf_repo": model_section.get("hf_repo", ""),
        "activation_dtype": activation_dtype,
        "random_seed": random_seed,
        "telemetry_enabled": bool(telemetry_section.get("enabled", False)),
        "telemetry_endpoint": telemetry_section.get("endpoint", ""),
        "telemetry_service_name": telemetry_section.get("service_name", ""),
    }
    driver_options_dict = asdict(driver_options)

    ready_queue = spawn_ctx.Queue()
    ctx = mp.spawn(
        worker.worker_main,
        args=(
            world_size,
            devices,
            master_port,
            model_config_dict,
            driver_options_dict,
            group_topology,
            ready_queue,
        ),
        nprocs=world_size,
        join=False,
        start_method="spawn",
        daemon=True,
    )

    _shutdown_requested = {"flag": False}

    def _on_sig(signo, _frame):
        _shutdown_requested["flag"] = True
        for p in ctx.processes:
            if p.is_alive():
                p.terminate()
    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    timeout_s = float(driver_section.get("ready_timeout_s", 1200.0))
    start = time.time()
    connected_ranks: set[int] = set()
    server_names_by_group: dict[int, str] = {}
    caps_by_group: dict[int, Any] = {}

    while len(connected_ranks) < world_size:
        for p in ctx.processes:
            if not p.is_alive() and p.exitcode != 0:
                raise SystemExit(
                    f"worker pid={p.pid} died with exit code {p.exitcode} before ready"
                )

        if time.time() - start > timeout_s:
            ready_queue.close()
            ready_queue.join_thread()
            raise SystemExit(
                f"timed out after {timeout_s:.0f}s waiting for {world_size} workers to become ready"
            )

        try:
            rank, server_name, payload = ready_queue.get(timeout=0.2)
        except queue.Empty:
            continue

        connected_ranks.add(rank)
        if server_name is not None:
            for gid, group in enumerate(group_topology):
                if rank in group:
                    server_names_by_group[gid] = server_name
                    caps_by_group[gid] = payload  # DriverCapabilities
                    break

    ready_queue.close()
    ready_queue.join_thread()

    missing = [gid for gid in range(num_groups) if gid not in caps_by_group]
    if missing:
        raise SystemExit(f"no DriverCapabilities from groups {missing}")

    # Emit per-group handshake lines, then a sentinel. `caps` no longer
    # needs to echo `shmem_name` separately — the wrapper line carries
    # it, and `embedded_driver::DriverCapabilities` accepts the field
    # missing via `#[serde(default)]`.
    for gid in range(num_groups):
        caps = caps_by_group[gid]
        caps_dict = asdict(caps) if dataclasses.is_dataclass(caps) else dict(caps)
        _write_handshake(args.handshake_fd, {
            "group_id": gid,
            "server_name": server_names_by_group[gid],
            "shmem_name": f"/pie_shmem_g{gid}",
            "caps": caps_dict,
        })
    _write_handshake(args.handshake_fd, {"ready": True, "num_groups": num_groups})
    try:
        os.close(args.handshake_fd)
    except OSError:
        pass

    # Track the first dead worker's exit code so the launcher's exit
    # status reflects the actual failure. Returning 0 here used to mask
    # worker crashes — pie-server's watchdog would log "driver exited
    # unexpectedly" but with rc=0, hiding which worker died and how.
    first_failure_code = 0

    while not _shutdown_requested["flag"]:
        any_dead = False
        for p in ctx.processes:
            if not p.is_alive() and p.exitcode != 0:
                print(
                    f"[{prog}] worker pid={p.pid} exited with code {p.exitcode}",
                    file=sys.stderr,
                )
                if first_failure_code == 0:
                    # exitcode is negative for signal-killed workers (-N
                    # for SIGN); +N for `os._exit(N)`. Map both to a
                    # non-zero positive code for the parent's view.
                    first_failure_code = abs(p.exitcode) or 1
                any_dead = True
        if any_dead:
            for p in ctx.processes:
                if p.is_alive():
                    p.terminate()
            break
        time.sleep(1.0)

    for p in ctx.processes:
        p.join(timeout=2)
        if p.is_alive():
            p.kill()
    ctx.join(timeout=1)
    return first_failure_code
