"""Metrics provider that connects to a live Pie server."""

from __future__ import annotations

import asyncio
import json
import sys
import traceback

import logging

logger = logging.getLogger(__name__)


import threading
import time
from typing import TYPE_CHECKING, List

from .data import (
    GPUMetrics,
    Inferlet,
    MetricsProvider,
    SystemMetrics,
    TPGroupMetrics,
)

if TYPE_CHECKING:
    from pie_client import PieClient


class PieMetricsProvider:
    """Metrics provider that polls a live Pie server.

    Connects to the Pie server via WebSocket and polls runtime stats.
    Optionally collects GPU metrics via pynvml.
    """

    def __init__(
        self,
        host: str,
        port: int,
        internal_token: str,
        config: dict | None = None,
        poll_interval: float = 1.0,
        max_history: int = 500,
    ):
        self._host = host
        self._port = port
        self._token = internal_token
        self._poll_interval = poll_interval
        self._max_history = max_history

        # Configuration for the TUI ConfigPanel
        self.config = config or {}

        # Thread-safe data
        self._lock = threading.Lock()
        self._latest_stats: dict | None = None
        self._latest_instances: list = []
        self._connected = False

        # Throughput/latency tracking
        self._last_poll_time: float = time.time()
        self._prev_active_batches: int = 0
        self._estimated_tput: float = 0.0
        self._estimated_latency: float = 0.0

        # Background polling thread
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # History arrays for TUI graphs (required by MetricsProvider protocol)
        self.kv_cache_history: List[float] = []
        self.token_tput_history: List[float] = []
        self.latency_history: List[float] = []
        self.batch_history: List[float] = []

        # GPU metrics via pynvml (optional)
        self._nvml_available = False
        self._gpu_handles = []
        self._pynvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml_available = True
            self._pynvml = pynvml
            device_count = pynvml.nvmlDeviceGetCount()
            self._gpu_handles = [
                pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(device_count)
            ]
        except Exception:
            pass  # pynvml not available or initialization failed

    def start(self) -> None:
        """Start background polling thread."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background polling thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _poll_loop(self) -> None:
        """Background loop that polls the server."""
        asyncio.run(self._async_poll_loop())

    async def _async_poll_loop(self) -> None:
        """Async poll loop."""
        from pie_client import PieClient

        server_uri = f"ws://{self._host}:{self._port}"

        while not self._stop_event.is_set():
            try:
                async with PieClient(server_uri) as client:
                    await client.auth_by_token(self._token)
                    with self._lock:
                        self._connected = True

                    while not self._stop_event.is_set():
                        try:
                            # Poll model stats
                            success, result = await client.query("model_status", "")
                            if success:
                                stats = json.loads(result)
                                with self._lock:
                                    self._latest_stats = stats

                            # Poll instances list
                            try:
                                instances = await client.list_processes()
                                with self._lock:
                                    self._latest_instances = instances
                            except Exception as e:
                                logger.debug("list_processes error: %s", e)

                        except Exception as e:
                            logger.debug("poll error: %s", e)

                        await asyncio.sleep(self._poll_interval)

            except Exception as e:
                logger.debug("connection error: %s", e)
                with self._lock:
                    self._connected = False
                await asyncio.sleep(1.0)

    def _get_gpu_metrics(self) -> list[GPUMetrics]:
        """Get GPU metrics via pynvml."""
        if not self._nvml_available or self._pynvml is None:
            return []

        try:
            pynvml = self._pynvml
            gpus = []
            for i, handle in enumerate(self._gpu_handles):
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append(
                    GPUMetrics(
                        gpu_id=i,
                        tp_group=0,
                        utilization=float(util.gpu),
                        memory_used_gb=mem.used / (1024**3),
                        memory_total_gb=mem.total / (1024**3),
                    )
                )
            return gpus
        except Exception:
            return []

    def get_metrics(self) -> SystemMetrics:
        """Get current metrics snapshot."""
        with self._lock:
            stats = self._latest_stats.copy() if self._latest_stats else {}
            instances = list(self._latest_instances)

        gpu_metrics = self._get_gpu_metrics()

        # Parse server stats (new flat key format: "model_name.kv_pages_used", etc.)
        kv_pages_used = 0
        kv_pages_total = 0
        in_flight_batches = 0
        total_tokens = 0
        last_latency_us = 0
        avg_latency_us = 0

        if stats:
            for key, value in stats.items():
                try:
                    if key.endswith(".kv_pages_used"):
                        kv_pages_used += int(value)
                    elif key.endswith(".kv_pages_total"):
                        kv_pages_total += int(value)
                    elif key.endswith(".in_flight_batches"):
                        in_flight_batches += int(value)
                    elif key.endswith(".total_tokens_processed"):
                        total_tokens += int(value)
                    elif key.endswith(".last_batch_latency_us"):
                        last_latency_us = max(last_latency_us, int(value))
                    elif key.endswith(".avg_batch_latency_us"):
                        avg_latency_us = max(avg_latency_us, int(value))
                except (ValueError, TypeError):
                    pass

        # Throughput: compute from total_tokens and elapsed time
        now = time.monotonic()
        dt = now - self._last_poll_time if hasattr(self, "_last_poll_time") else 0
        prev_tokens = getattr(self, "_prev_total_tokens", 0)

        if dt > 0 and total_tokens > prev_tokens:
            self._estimated_tput = (total_tokens - prev_tokens) / dt
        elif in_flight_batches == 0:
            self._estimated_tput = 0.0

        self._last_poll_time = now
        self._prev_total_tokens = total_tokens

        # Latency: use avg_batch_latency_us converted to ms
        if avg_latency_us > 0:
            self._estimated_latency = avg_latency_us / 1000.0
        elif in_flight_batches == 0:
            self._estimated_latency = 0.0

        # KV cache usage
        kv_cache_usage = (
            (kv_pages_used / kv_pages_total) * 100 if kv_pages_total > 0 else 0.0
        )

        # TP groups from GPU metrics
        tp_groups = [
            TPGroupMetrics(tp_id=g.gpu_id, utilization=g.utilization, gpus=[g])
            for g in gpu_metrics
        ]

        # Convert instances (now list[dict]) to Inferlet objects
        inferlets = []
        for inst in instances:
            try:
                if isinstance(inst, dict):
                    inst_id = inst.get("id", "")
                    username = inst.get("username", "user")
                    program = inst.get("program", "inferlet")
                    elapsed_secs = inst.get("elapsed_secs", 0)
                else:
                    # Fallback for old format (bare UUID strings)
                    inst_id = str(inst)
                    username = "user"
                    program = "inferlet"
                    elapsed_secs = 0

                elapsed_str = (
                    f"{elapsed_secs // 60}m{elapsed_secs % 60}s"
                    if elapsed_secs > 0
                    else "-"
                )

                inferlets.append(
                    Inferlet(
                        id=inst_id[:12] if len(inst_id) > 12 else inst_id,
                        program=program,
                        user=username,
                        status="running",
                        elapsed=elapsed_str,
                        kv_cache=0.0,
                    )
                )
            except Exception:
                pass

        # Sort by elapsed (descending) and limit to top 50
        inferlets = inferlets[:50]

        self.kv_cache_history.append(kv_cache_usage)
        self.token_tput_history.append(self._estimated_tput)
        self.latency_history.append(self._estimated_latency)
        self.batch_history.append(float(in_flight_batches))

        # Trim history
        if len(self.kv_cache_history) > self._max_history:
            self.kv_cache_history.pop(0)
            self.token_tput_history.pop(0)
            self.latency_history.pop(0)
            self.batch_history.pop(0)

        return SystemMetrics(
            kv_cache_usage=kv_cache_usage,
            kv_pages_used=kv_pages_used,
            kv_pages_total=kv_pages_total if kv_pages_total > 0 else 600,
            token_throughput=self._estimated_tput,
            latency_ms=self._estimated_latency,
            active_batches=in_flight_batches,
            tp_groups=tp_groups,
            inferlets=inferlets,
        )

    @property
    def is_connected(self) -> bool:
        """Check if connected to the server."""
        with self._lock:
            return self._connected
