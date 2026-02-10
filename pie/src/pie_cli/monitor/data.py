"""Data classes, providers, and simulation for the LLM Monitor."""

import math
import queue
import random
from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


# Default configuration display
DEFAULT_CONFIG = {
    "host": "127.0.0.1",
    "port": 8080,
    "auth": {"enabled": False},
    "hf_repo": "qwen3-32b",
    "device": [0, 1],
    "tensor_parallel_size": 2,
    "activation_dtype": "bfloat16",
    "weight_dtype": "bfloat16",
    "kv_page_size": 16,
    "max_batch_tokens": 10240,
    "max_dist_size": 32,
    "max_num_embeds": 128,
    "max_num_adapters": 32,
    "max_adapter_rank": 8,
    "gpu_mem_utilization": 0.8,
    "use_cuda_graphs": False,
    "telemetry_enabled": False,
}


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(slots=True)
class GPUMetrics:
    """Metrics for a single GPU."""

    gpu_id: int
    tp_group: int
    utilization: float
    memory_used_gb: float
    memory_total_gb: float
    history: list[float] = field(default_factory=list)

    @property
    def memory_percent(self) -> float:
        return (self.memory_used_gb / self.memory_total_gb) * 100


@dataclass(slots=True)
class TPGroupMetrics:
    """Metrics for a TP group with its GPUs."""

    tp_id: int
    utilization: float
    gpus: list[GPUMetrics]


@dataclass(slots=True)
class Inferlet:
    """Represents a running inferlet."""

    id: str
    program: str
    user: str
    status: str
    elapsed: str
    kv_cache: float


@dataclass(slots=True)
class SystemMetrics:
    """All system metrics at a point in time."""

    kv_cache_usage: float
    kv_pages_used: int
    kv_pages_total: int
    token_throughput: float
    latency_ms: float
    active_batches: int
    tp_groups: list[TPGroupMetrics]
    inferlets: list[Inferlet]


# =============================================================================
# Metrics Provider Protocol
# =============================================================================


@runtime_checkable
class MetricsProvider(Protocol):
    """Interface for metrics data sources.

    Implement this protocol to provide custom data sources for the TUI.
    """

    # History arrays for graphs (must be mutable lists)
    kv_cache_history: List[float]
    token_tput_history: List[float]
    latency_history: List[float]
    batch_history: List[float]

    def get_metrics(self) -> SystemMetrics:
        """Get the current system metrics snapshot."""
        ...


# =============================================================================
# Queue-based Provider (for external integration)
# =============================================================================


class QueueProvider:
    """Receives metrics from an external queue.

    Use this provider when integrating with a real serving system.
    Push SystemMetrics objects to the queue from your serving thread.

    Example:
        metrics_queue = queue.Queue()
        provider = QueueProvider(metrics_queue)
        app = LLMMonitorApp(provider=provider)

        # In your serving thread:
        metrics_queue.put(system_metrics)
    """

    def __init__(self, data_queue: queue.Queue, max_history: int = 500):
        self._queue = data_queue
        self._max_history = max_history
        self._last_metrics: SystemMetrics | None = None

        # History arrays for graphs
        self.kv_cache_history: List[float] = []
        self.token_tput_history: List[float] = []
        self.latency_history: List[float] = []
        self.batch_history: List[float] = []

    def get_metrics(self) -> SystemMetrics:
        """Get metrics from the queue (non-blocking).

        Returns the latest metrics if available, otherwise returns
        the last known metrics or a default empty state.
        """
        # Drain queue and keep the latest
        while True:
            try:
                self._last_metrics = self._queue.get_nowait()
            except queue.Empty:
                break

        if self._last_metrics is None:
            # Return empty state if no data yet
            return SystemMetrics(
                kv_cache_usage=0,
                kv_pages_used=0,
                kv_pages_total=600,
                token_throughput=0,
                latency_ms=0,
                active_batches=0,
                tp_groups=[],
                inferlets=[],
            )

        # Update history
        self.kv_cache_history.append(self._last_metrics.kv_cache_usage)
        self.token_tput_history.append(self._last_metrics.token_throughput)
        self.latency_history.append(self._last_metrics.latency_ms)
        self.batch_history.append(float(self._last_metrics.active_batches))

        # Trim history
        if len(self.kv_cache_history) > self._max_history:
            self.kv_cache_history.pop(0)
            self.token_tput_history.pop(0)
            self.latency_history.pop(0)
            self.batch_history.pop(0)

        return self._last_metrics


# =============================================================================
# Simulated Provider (for demo/testing)
# =============================================================================


class SimulatedProvider:
    """Generates dynamic simulated metrics for demo and testing."""

    def __init__(self, num_gpus: int = 8, num_tp_groups: int = 4):
        self.num_gpus = num_gpus
        self.num_tp_groups = num_tp_groups
        self._tick = 0

        # Assign GPUs to TP groups (distribute evenly)
        self._gpu_tp_mapping: List[int] = []
        gpus_per_tp = num_gpus // num_tp_groups
        remainder = num_gpus % num_tp_groups
        for tp in range(num_tp_groups):
            count = gpus_per_tp + (1 if tp < remainder else 0)
            self._gpu_tp_mapping.extend([tp] * count)

        # Initialize values
        self._kv_cache = random.uniform(30, 70)
        self._kv_pages_total = 600
        self._token_tput = random.uniform(600, 1200)
        self._active_batches = random.randint(3, 10)
        self._tp_utils = [random.uniform(30, 70) for _ in range(num_tp_groups)]
        self._gpu_utils = [random.uniform(40, 80) for _ in range(num_gpus)]
        self._gpu_mems = [random.uniform(8, 18) for _ in range(num_gpus)]

        # GPU utilization history
        self._gpu_histories: List[List[float]] = [[] for _ in range(num_gpus)]
        self.max_history = 500

        # Targets for smooth animation
        self._kv_target = self._kv_cache
        self._tput_target = self._token_tput
        self._batch_target = self._active_batches

        self._inferlets = [
            Inferlet("inf-001", "llama3-70b", "alice", "running", "2h 14m", 45.2),
            Inferlet("inf-002", "qwen3-32b", "bob", "running", "1h 32m", 32.1),
            Inferlet("inf-003", "mistral-7b", "carol", "idle", "45m", 0.0),
            Inferlet("inf-004", "gemma2-27b", "dave", "running", "3h 05m", 28.7),
        ]

        # Initialize latency
        self._latency = random.uniform(30, 50)
        self._latency_target = self._latency

        # History arrays (required by MetricsProvider protocol)
        self.kv_cache_history: List[float] = []
        self.token_tput_history: List[float] = []
        self.latency_history: List[float] = []
        self.batch_history: List[float] = []

    def _ease_toward(self, current: float, target: float, speed: float = 0.15) -> float:
        return current + (target - current) * speed

    def _add_noise(self, value: float, noise_level: float = 2.0) -> float:
        return value + random.uniform(-noise_level, noise_level)

    def get_metrics(self) -> SystemMetrics:
        self._tick += 1

        # Randomly update targets
        if random.random() < 0.08:
            self._kv_target = random.uniform(20, 95)
        if random.random() < 0.1:
            self._tput_target = random.uniform(300, 2200)
        if random.random() < 0.12:
            self._batch_target = random.randint(1, 16)

        # Apply wave animations
        wave1 = math.sin(self._tick * 0.1) * 15
        wave2 = math.sin(self._tick * 0.07 + 1) * 200
        wave3 = math.sin(self._tick * 0.15 + 2) * 3

        self._kv_cache = self._ease_toward(
            self._kv_cache, self._kv_target + wave1, 0.12
        )
        self._kv_cache = max(5, min(98, self._add_noise(self._kv_cache, 3)))

        self._token_tput = self._ease_toward(
            self._token_tput, self._tput_target + wave2, 0.1
        )
        self._token_tput = max(100, min(2500, self._add_noise(self._token_tput, 40)))

        self._active_batches = int(
            self._ease_toward(self._active_batches, self._batch_target + wave3, 0.15)
        )
        self._active_batches = max(
            1, min(16, self._active_batches + random.randint(-1, 1))
        )

        # Random spikes
        if random.random() < 0.03:
            self._kv_cache = min(98, self._kv_cache + random.uniform(15, 30))
        if random.random() < 0.04:
            self._token_tput = min(2400, self._token_tput + random.uniform(300, 600))
        if random.random() < 0.02:
            self._token_tput = max(200, self._token_tput - random.uniform(400, 800))

        # Update latency
        if random.random() < 0.1:
            self._latency_target = random.uniform(15, 80)
        wave4 = math.sin(self._tick * 0.12 + 3) * 8
        self._latency = self._ease_toward(
            self._latency, self._latency_target + wave4, 0.1
        )
        self._latency = max(10, min(120, self._add_noise(self._latency, 3)))

        # Update TP groups
        base_tp_wave = math.sin(self._tick * 0.08) * 10
        for i in range(self.num_tp_groups):
            if random.random() < 0.1:
                self._tp_utils[i] = random.uniform(30, 90)
            offset = math.sin(self._tick * 0.1 + i * 0.5) * 8
            self._tp_utils[i] = self._ease_toward(
                self._tp_utils[i], 60 + base_tp_wave + offset, 0.1
            )
            self._tp_utils[i] = max(10, min(98, self._add_noise(self._tp_utils[i], 2)))

        # Update GPUs
        for i in range(self.num_gpus):
            gpu_wave = math.sin(self._tick * 0.09 + i * 0.8) * 12
            if random.random() < 0.06:
                self._gpu_utils[i] = random.uniform(40, 95)
            self._gpu_utils[i] = self._ease_toward(
                self._gpu_utils[i], 65 + gpu_wave, 0.12
            )
            self._gpu_utils[i] = max(
                15, min(99, self._add_noise(self._gpu_utils[i], 3))
            )

            self._gpu_histories[i].append(self._gpu_utils[i])
            if len(self._gpu_histories[i]) > self.max_history:
                self._gpu_histories[i].pop(0)

            if random.random() < 0.05:
                self._gpu_mems[i] = random.uniform(6, 22)
            self._gpu_mems[i] = self._ease_toward(
                self._gpu_mems[i], 14 + gpu_wave * 0.3, 0.05
            )
            self._gpu_mems[i] = max(4, min(23, self._add_noise(self._gpu_mems[i], 0.3)))

        # Build TP groups
        tp_groups = []
        for tp_id in range(self.num_tp_groups):
            gpus = []
            for gpu_id in range(self.num_gpus):
                if self._gpu_tp_mapping[gpu_id] == tp_id:
                    gpus.append(
                        GPUMetrics(
                            gpu_id=gpu_id,
                            tp_group=tp_id,
                            utilization=self._gpu_utils[gpu_id],
                            memory_used_gb=self._gpu_mems[gpu_id],
                            memory_total_gb=24.0,
                            history=self._gpu_histories[gpu_id].copy(),
                        )
                    )
            tp_groups.append(
                TPGroupMetrics(
                    tp_id=tp_id, utilization=self._tp_utils[tp_id], gpus=gpus
                )
            )

        # Update history
        self.kv_cache_history.append(self._kv_cache)
        self.token_tput_history.append(self._token_tput)
        self.latency_history.append(self._latency)
        self.batch_history.append(float(self._active_batches))

        if len(self.kv_cache_history) > self.max_history * 2:
            self.kv_cache_history.pop(0)
            self.token_tput_history.pop(0)
            self.latency_history.pop(0)
            self.batch_history.pop(0)

        # Update inferlets
        for inf in self._inferlets:
            if inf.status == "running":
                inf.kv_cache = max(5, min(95, inf.kv_cache + random.uniform(-3, 3)))

        if random.random() < 0.08:
            inf = random.choice(self._inferlets)
            if inf.status == "running":
                inf.status = "idle"
                inf.kv_cache = 0.0
            else:
                inf.status = "running"
                inf.kv_cache = random.uniform(20, 60)

        kv_pages_used = int((self._kv_cache / 100) * self._kv_pages_total)

        return SystemMetrics(
            kv_cache_usage=self._kv_cache,
            kv_pages_used=kv_pages_used,
            kv_pages_total=self._kv_pages_total,
            token_throughput=self._token_tput,
            latency_ms=self._latency,
            active_batches=self._active_batches,
            tp_groups=tp_groups,
            inferlets=self._inferlets.copy(),
        )


# Backwards compatibility alias
SimulatedMetrics = SimulatedProvider
