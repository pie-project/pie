"""Shared-memory IPC fast path for fire_batch.

Layout of the shmem region:

    [HEADER_SIZE bytes header]
      0:  u32 magic (0x50494533 = 'PIE3')
      4:  u32 schema_version
      8:  u32 num_slots
      12: u32 slot_stride
      ... rest reserved ...

    [N slots, each slot_stride bytes]
      slot[i]:
        0:  u64 req_seq    (atomic; bumped by Rust on send)
        8:  u64 resp_seq   (atomic; bumped by Python on respond)
        16: u32 req_id     (correlation id, mirrored from IpcRequest.request_id)
        20: u32 method_tag (0=fire_batch, 255=other)
        24: u32 req_payload_len   (bytes consumed by the layout)
        28: u32 resp_payload_len
        32: u64 send_walltime_us
        40: u64 respond_walltime_us
        48: ...padding to 64...
        64: request payload bytes
        ...:  response payload bytes

For one in-flight at a time per slot, both sides spin on req_seq/resp_seq.
"""

from __future__ import annotations

import ctypes
import mmap
import os
import time
from typing import Optional, Tuple

# Shmem layout constants — must match Rust side (runtime/src/shmem_ipc.rs).
MAGIC = 0x50494533  # 'PIE3'
SCHEMA_VERSION = 1
HEADER_SIZE = 64
SLOT_HEADER_SIZE = 64
DEFAULT_SLOTS = 8
# Each slot holds REQUEST_BUF + RESPONSE_BUF after slot header.
DEFAULT_REQ_BUF = 4 * 1024 * 1024   # 4 MB
DEFAULT_RESP_BUF = 1 * 1024 * 1024  # 1 MB
DEFAULT_SLOT_STRIDE = SLOT_HEADER_SIZE + DEFAULT_REQ_BUF + DEFAULT_RESP_BUF

METHOD_TAG_FIRE_BATCH = 0
METHOD_TAG_COPY_D2D = 1
# h2d / d2h via shmem fast path. Mirrors METHOD_TAG_COPY_H2D /
# METHOD_TAG_COPY_D2H in runtime/src/shmem_ipc.rs. Routes the
# eviction/restore copies through the same Python thread that
# dispatches fire_batch so CUDA stream FIFO orders the kernels.
# See `project_pie_second_bleed_path_h2d_race.md`.
METHOD_TAG_COPY_H2D = 2
METHOD_TAG_COPY_D2H = 3
# Mamba copy-on-fork notification (#108 phase 5b). Mirrors
# METHOD_TAG_MAMBA_FORK in runtime/src/shmem_ipc.rs. Carries
# `(parent_ctx_id: u64, child_ctx_id: u64)` little-endian so the
# Python worker can copy mamba recurrent state from parent slot to
# child slot before any later fire_batch reaches the child. Same
# single-threaded dispatch ordering as METHOD_TAG_COPY_D2D.
METHOD_TAG_MAMBA_FORK = 4
METHOD_TAG_NONE = 255

_librt = ctypes.CDLL("librt.so.1", use_errno=True)
_librt.shm_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
_librt.shm_open.restype = ctypes.c_int
_librt.shm_unlink.argtypes = [ctypes.c_char_p]


def _shm_open_create(name: str, size: int) -> int:
    # Best-effort cleanup of stale region.
    _librt.shm_unlink(name.encode())
    fd = _librt.shm_open(name.encode(), os.O_CREAT | os.O_RDWR, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), f"shm_open({name}) failed")
    os.ftruncate(fd, size)
    return fd


def _shm_open_existing(name: str) -> int:
    fd = _librt.shm_open(name.encode(), os.O_RDWR, 0o600)
    if fd < 0:
        raise OSError(ctypes.get_errno(), f"shm_open({name}) failed")
    return fd


class ShmemServer:
    """Python-side server. Creates the shmem region and waits for requests."""

    def __init__(
        self,
        name: str,
        num_slots: int = DEFAULT_SLOTS,
        req_buf: int = DEFAULT_REQ_BUF,
        resp_buf: int = DEFAULT_RESP_BUF,
    ) -> None:
        self.name = name
        self.num_slots = num_slots
        self.req_buf_size = req_buf
        self.resp_buf_size = resp_buf
        self.slot_stride = SLOT_HEADER_SIZE + req_buf + resp_buf
        self.total_size = HEADER_SIZE + num_slots * self.slot_stride

        self._fd = _shm_open_create(name, self.total_size)
        self._mm = mmap.mmap(
            self._fd, self.total_size,
            prot=mmap.PROT_READ | mmap.PROT_WRITE,
        )
        # Ensure region is unlinked even if close() isn't called (daemon exit).
        import atexit as _atexit
        _atexit.register(self._unlink_only)
        # Get raw address. We hold a reference to _mm so the mapping stays alive.
        self._buf = (ctypes.c_uint8 * self.total_size).from_buffer(self._mm)
        self._addr = ctypes.addressof(self._buf)

        # Initialize header
        ctypes.memset(self._addr, 0, HEADER_SIZE)
        self._u32_at(0, MAGIC)
        self._u32_at(4, SCHEMA_VERSION)
        self._u32_at(8, num_slots)
        self._u32_at(12, self.slot_stride)

        # Zero out slot headers
        for i in range(num_slots):
            ctypes.memset(self._slot_addr(i), 0, SLOT_HEADER_SIZE)

        # Per-slot last-seen seq; we busy-spin on each slot.
        self._last_seen = [0] * num_slots

    # -- Layout helpers --------------------------------------------------------

    def _slot_addr(self, i: int) -> int:
        return self._addr + HEADER_SIZE + i * self.slot_stride

    def _slot_req_payload_addr(self, i: int) -> int:
        return self._slot_addr(i) + SLOT_HEADER_SIZE

    def _slot_resp_payload_addr(self, i: int) -> int:
        return self._slot_addr(i) + SLOT_HEADER_SIZE + self.req_buf_size

    def _u32_at(self, offset: int, value: int) -> None:
        ctypes.cast(self._addr + offset, ctypes.POINTER(ctypes.c_uint32))[0] = value

    def _u64_at_slot(self, slot: int, offset: int) -> int:
        return ctypes.cast(self._slot_addr(slot) + offset, ctypes.POINTER(ctypes.c_uint64))[0]

    def _set_u64_at_slot(self, slot: int, offset: int, value: int) -> None:
        ctypes.cast(self._slot_addr(slot) + offset, ctypes.POINTER(ctypes.c_uint64))[0] = value

    def _u32_at_slot(self, slot: int, offset: int) -> int:
        return ctypes.cast(self._slot_addr(slot) + offset, ctypes.POINTER(ctypes.c_uint32))[0]

    def _set_u32_at_slot(self, slot: int, offset: int, value: int) -> None:
        ctypes.cast(self._slot_addr(slot) + offset, ctypes.POINTER(ctypes.c_uint32))[0] = value

    # -- Public API ------------------------------------------------------------

    def poll_slot(self, slot: int) -> Optional[Tuple[int, int, int, int, int]]:
        """If slot has a pending request, return (req_id, method_tag, payload_len, slot_idx, send_walltime_us).
        Otherwise return None.
        """
        seq = self._u64_at_slot(slot, 0)
        if seq == self._last_seen[slot]:
            return None
        self._last_seen[slot] = seq
        req_id = self._u32_at_slot(slot, 16)
        method_tag = self._u32_at_slot(slot, 20)
        req_len = self._u32_at_slot(slot, 24)
        send_walltime_us = self._u64_at_slot(slot, 32)
        return (req_id, method_tag, req_len, slot, send_walltime_us)

    def request_payload_view(self, slot: int, length: int) -> memoryview:
        """Zero-copy view of the request payload bytes for `slot`."""
        if length > self.req_buf_size:
            raise ValueError(f"length {length} exceeds req_buf_size {self.req_buf_size}")
        addr = self._slot_req_payload_addr(slot)
        return (ctypes.c_uint8 * length).from_address(addr)

    def respond(self, slot: int, payload: bytes, respond_walltime_us: int = 0) -> None:
        """Write response bytes into slot's response buffer and bump resp_seq.

        Memory order: write payload + len + walltime FIRST, then bump resp_seq
        with release semantics so reader sees a complete response.
        """
        n = len(payload)
        if n > self.resp_buf_size:
            raise ValueError(f"response {n} exceeds resp_buf_size {self.resp_buf_size}")
        ctypes.memmove(self._slot_resp_payload_addr(slot), payload, n)
        self._set_u32_at_slot(slot, 28, n)
        self._set_u64_at_slot(slot, 40, respond_walltime_us)
        # Bump resp_seq last (matches the req_seq just received)
        seq = self._u64_at_slot(slot, 0)
        self._set_u64_at_slot(slot, 8, seq)

    def respond_view(self, slot: int) -> memoryview:
        """Mutable view of the response buffer; caller must call commit_respond."""
        addr = self._slot_resp_payload_addr(slot)
        return (ctypes.c_uint8 * self.resp_buf_size).from_address(addr)

    def commit_respond(self, slot: int, length: int, respond_walltime_us: int = 0) -> None:
        if length > self.resp_buf_size:
            raise ValueError(f"length {length} exceeds resp_buf_size {self.resp_buf_size}")
        self._set_u32_at_slot(slot, 28, length)
        self._set_u64_at_slot(slot, 40, respond_walltime_us)
        seq = self._u64_at_slot(slot, 0)
        self._set_u64_at_slot(slot, 8, seq)

    def busy_poll_any_slot(self, spin_us: int) -> Optional[Tuple[int, int, int, int, int]]:
        """Spin across all slots; return first slot with a request, or None after spin_us."""
        deadline = time.monotonic_ns() + spin_us * 1000
        while True:
            for s in range(self.num_slots):
                req = self.poll_slot(s)
                if req is not None:
                    return req
            if time.monotonic_ns() >= deadline:
                return None

    def close(self) -> None:
        # Order matters: drop ctypes views first, then mmap, then fd, then unlink.
        self._buf = None
        try:
            self._mm.close()
        except Exception:
            pass
        try:
            os.close(self._fd)
        except Exception:
            pass
        _librt.shm_unlink(self.name.encode())

    def _unlink_only(self) -> None:
        """Best-effort unlink for atexit. The fd may already be closed; that's fine."""
        try:
            _librt.shm_unlink(self.name.encode())
        except Exception:
            pass
