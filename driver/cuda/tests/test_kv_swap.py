"""Direct unit test for the cuda_native KV-swap control channel.

Spawns the C++ binary with `swap_pool_size > 0`, opens the SOCK_SEQPACKET
control fd, sends a copy_d2h then a copy_h2d, asserts both succeed. Doesn't
actually validate KV correctness (we'd need to seed the device pool first);
just proves the wire and copy kernels are reachable.

Usage:
    cd pie && uv run python ../driver/cuda/tests/test_kv_swap.py
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BINARY = REPO / "driver" / "cuda" / "build" / "bin" / "pie_driver_cuda"

# Build a minimal startup TOML in /tmp that points at a cached HF snapshot.
def find_snapshot() -> str:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    name = "models--Qwen--Qwen3-0.6B"
    snap = sorted((cache / name / "snapshots").iterdir())[-1]
    return str(snap)


def main() -> int:
    if not BINARY.is_file():
        print(f"binary not found at {BINARY}", file=sys.stderr)
        return 1

    snap = find_snapshot()
    toml = (
        '[shmem]\nname = "/pie_shmem_kvswap_test"\n'
        '[model]\n'
        f'hf_repo = "Qwen/Qwen3-0.6B"\nsnapshot_dir = "{snap}"\n'
        'device = "cuda:0"\ndtype = "bfloat16"\n'
        '[batching]\n'
        'kv_page_size = 32\n'
        'max_num_kv_pages = 32\n'
        'max_batch_tokens = 1024\n'
        'max_batch_size = 8\n'
        'swap_pool_size = 16\n'
    )
    toml_path = Path("/tmp/kvswap_test.toml")
    toml_path.write_text(toml)

    parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    os.set_inheritable(child_sock.fileno(), True)

    proc = subprocess.Popen(
        [str(BINARY), "--config", str(toml_path),
         "--control-fd", str(child_sock.fileno())],
        pass_fds=(child_sock.fileno(),),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )
    child_sock.close()

    # Wait for READY.
    deadline = time.monotonic() + 60.0
    ready_caps = None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        line = line.rstrip()
        if line.startswith("READY "):
            import json
            ready_caps = json.loads(line[len("READY "):])
            break
        print(f"[binary] {line}")
    if not ready_caps:
        print("never saw READY", file=sys.stderr)
        proc.kill()
        return 2
    print(f"READY caps: swap_pool_size={ready_caps['swap_pool_size']}")

    HEADER = struct.Struct("<IIII")

    def call(method: int, srcs: list[int], dsts: list[int]) -> int:
        n = len(srcs)
        # `layer` field is reserved/ignored — copies span all layers.
        msg = (HEADER.pack(method, 0, n, 0)
               + struct.pack(f"<{n}I", *srcs)
               + struct.pack(f"<{n}I", *dsts))
        parent_sock.send(msg)
        resp = parent_sock.recv(4)
        return struct.unpack("<I", resp)[0]

    rc = 0
    s = call(1, [0], [0]); print(f"copy_d2h status={s}")
    if s != 0: rc = 3
    s = call(2, [0], [0]); print(f"copy_h2d status={s}")
    if s != 0: rc = 4
    s = call(3, [0], [1]); print(f"copy_d2d status={s}")
    if s != 0: rc = 5
    s = call(4, [0], [1]); print(f"copy_h2h status={s}")
    if s != 0: rc = 6
    s = call(1, [0, 1, 2, 3], [4, 5, 6, 7]); print(f"copy_d2h bulk status={s}")
    if s != 0: rc = 7

    parent_sock.close()
    proc.send_signal(15)  # SIGTERM
    proc.wait(timeout=5)
    return rc


if __name__ == "__main__":
    sys.exit(main())
