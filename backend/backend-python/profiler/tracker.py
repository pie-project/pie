"""
Memory profiling tracker for PyTorch tensors with operation context tracking.

This module provides automatic memory tracking for all allocated tensors
(CPU, MPS, CUDA) and exports snapshots to JSON files for analysis.
It orchestrates multiple profiling components for unified profiling.
"""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .export_utils import (
    export_operation_graph,
    export_tensor_lifecycle,
    export_timing_tree,
    generate_operation_summary,
)
from .operation_tracker import OperationTracker
from .pytorch_profiler import PyTorchProfiler
from .snapshot_capture import TensorScanner
from .timing_profiler import TimingProfiler
from .types import MemorySnapshot


class MemoryTracker:
    """
    Main orchestrator for unified memory and timing profiling.

    This tracker composes multiple profiling components:
    - TensorScanner: GC-based tensor scanning and snapshot creation
    - OperationTracker: Operation graph and tensor lifecycle tracking
    - TimingProfiler: Hierarchical timing tree with GPU synchronization
    - PyTorchProfiler: PyTorch profiler integration

    It provides a unified API and handles persistence.
    """

    def __init__(
        self,
        output_dir: str = ".",
        enabled: bool = False,
        interval: float = 5.0,
        enable_timing: bool = False,
    ):
        """
        Initialize the memory tracker.

        Args:
            output_dir: Directory where memory snapshots will be saved
            enabled: Whether memory tracking is enabled
            interval: Interval in seconds between automatic snapshots (default: 5.0)
            enable_timing: Whether to enable timing profiling (unified profiler mode)
        """
        self.enabled = enabled
        self.enable_timing = enable_timing
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.snapshots: list[MemorySnapshot] = []
        self._lock = threading.Lock()
        self._interval = interval
        self._stop_event = threading.Event()
        self._tracker_thread: threading.Thread | None = None

        # Compose profiling components
        self._operation_tracker = OperationTracker()
        self._tensor_scanner = TensorScanner(
            self._operation_tracker.get_registry(),
            self._operation_tracker._tensor_last_seen,
        )
        self._timing_profiler = TimingProfiler(enable_timing)
        self._pytorch_profiler = PyTorchProfiler()

        # Operation log for hook-based tracking (added by hook tracker if enabled)
        self._operation_log: list[dict[str, Any]] = []
        self._hook_tracker: Any = None  # Lazy initialized by profile_attention

        if self.enabled:
            print("✅ Memory tracking enabled")
            print(f"   Snapshots will be saved to: {self.output_dir.absolute()}")
            print(f"   Snapshot interval: {interval}s")
            print("   Operation tracking: ON (use track_operation() context manager)")
            if self.enable_timing:
                print("   Timing profiling: ON (unified profiler mode)")
            self._start_periodic_tracking()

    @contextmanager
    def track_pytorch_ops(self):
        """
        Context manager to track low-level PyTorch operations.

        Usage:
            with tracker.track_pytorch_ops():
                output = model.forward(input)
        """
        if not self.enabled:
            yield
            return

        with self._pytorch_profiler.track_pytorch_ops(self._tensor_scanner):
            yield

    @contextmanager
    def track_operation(self, operation_name: str):
        """
        Context manager to track memory allocations and tensor access within an operation.

        Usage:
            with tracker.track_operation("model_forward"):
                output = model.forward(input)

        Args:
            operation_name: Name of the operation being tracked
        """
        if not self.enabled:
            yield
            return

        with self._operation_tracker.track_operation(
            operation_name, self._tensor_scanner
        ):
            yield

    @staticmethod
    def synchronize_device():
        """Synchronize GPU operations to ensure accurate timing measurements."""
        TimingProfiler.synchronize_device()

    @contextmanager
    def start_profile(self, name: str):
        """
        Profile an operation with timing and optional tensor tracking (unified profiler).

        Usage:
            with tracker.start_profile("operation_name"):
                # code to profile

        Args:
            name: Name of the operation (e.g., "attn_metal_kernel", "forward_pass")
        """
        # If profiling is completely disabled, use nullcontext
        if not self.enabled and not self.enable_timing:
            yield
            return

        tensor_registry = (
            self._operation_tracker.get_registry() if self.enabled else None
        )

        with self._timing_profiler.start_profile(name, tensor_registry):
            yield

    def checkpoint(self, name: str) -> None:
        """
        Record a memory checkpoint with the given name.

        Args:
            name: Descriptive name for this checkpoint (e.g., "after_model_load")
        """
        if not self.enabled:
            return

        with self._lock:
            snapshot = self._tensor_scanner.capture_snapshot(
                name,
                self._operation_tracker.get_current_operation(),
                self._operation_tracker.record_lifecycle_event,
            )
            self.snapshots.append(snapshot)

            # Print summary to console
            print(f"\n📊 Memory Checkpoint: {name}")
            print(
                f"   Total: {snapshot.total_mb:.2f} MB "
                f"across {snapshot.tensor_count} tensors"
            )
            for device, stats in snapshot.device_breakdown.items():
                print(
                    f"   {device}: {stats['total_mb']:.2f} MB "
                    f"({stats['count']} tensors)"
                )

    def save_snapshot(self, filename: str | None = None) -> str:
        """
        Save all recorded snapshots to a JSON file.

        Args:
            filename: Optional custom filename. If None, generates timestamped filename.

        Returns:
            Path to the saved file
        """
        if not self.enabled and not self.enable_timing:
            return ""

        with self._lock:
            # Check if we have any data to save
            has_snapshots = bool(self.snapshots)
            has_timing = bool(self._timing_profiler.get_timing_tree())

            if not has_snapshots and not has_timing:
                print("⚠️  No profiling data to save")
                return ""

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"unified_profile_{timestamp}.json"

            output_path = self.output_dir / filename

            # Always use unified_profiler format
            format_type = "unified_profiler"

            # Prepare data for export
            export_data = {
                "format": format_type,
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "num_snapshots": len(self.snapshots) if has_snapshots else 0,
                    "operation_tracking_enabled": bool(
                        self._operation_tracker.get_operation_allocations()
                    ),
                    "pytorch_ops_tracking_enabled": bool(
                        self._pytorch_profiler.get_ops()
                    ),
                    "tensor_lifecycle_tracking_enabled": bool(
                        self._operation_tracker.get_lifecycle()
                    ),
                    "timing_profiling_enabled": self.enable_timing,
                    "num_timing_nodes": (
                        len(self._timing_profiler.get_timing_tree())
                        if has_timing
                        else 0
                    ),
                },
            }

            # Add snapshots if available
            if has_snapshots:
                export_data["snapshots"] = [
                    asdict(snapshot) for snapshot in self.snapshots
                ]

            # Add timing tree if available (unified profiler mode)
            if has_timing:
                export_data["profiling_tree"] = export_timing_tree(
                    self._timing_profiler.get_timing_tree()
                )

            # Add operation allocations if any were tracked
            operation_allocations = self._operation_tracker.get_operation_allocations()
            if operation_allocations:
                export_data["operation_allocations"] = dict(operation_allocations)
                export_data["operation_summary"] = generate_operation_summary(
                    operation_allocations
                )

            # Add operation graph if tracked
            operation_graph = self._operation_tracker.get_operation_graph()
            if operation_graph:
                export_data["operation_graph"] = export_operation_graph(
                    operation_graph, self._operation_tracker.get_registry()
                )

            # Add tensor lifecycle if tracked
            tensor_lifecycle = self._operation_tracker.get_lifecycle()
            if tensor_lifecycle:
                export_data["tensor_lifecycle"] = export_tensor_lifecycle(
                    tensor_lifecycle, self._operation_tracker.get_registry()
                )

            # Add PyTorch operations if tracked
            pytorch_ops = self._pytorch_profiler.get_ops()
            if pytorch_ops:
                export_data["pytorch_operations"] = pytorch_ops

            # Add operation log if tracked (from hook-based tracker)
            if self._operation_log:
                export_data["operation_log"] = self._operation_log
                export_data["metadata"]["operation_log_enabled"] = True

            # Write to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)

            # Print summary
            print(f"\n💾 Profile saved to: {output_path.absolute()}")
            if has_timing:
                print(
                    f"   • Timing tree with {len(self._timing_profiler.get_timing_tree())} nodes"
                )
            if operation_allocations:
                print(f"   • {len(operation_allocations)} operations tracked")
            if operation_graph:
                print(f"   • Operation graph with {len(operation_graph)} nodes")
            if tensor_lifecycle:
                print(f"   • Lifecycle tracked for {len(tensor_lifecycle)} tensors")
            if pytorch_ops:
                print(f"   • {len(pytorch_ops)} PyTorch operations captured")
            if self._operation_log:
                print(f"   • Operation log with {len(self._operation_log)} entries")
            return str(output_path.absolute())

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all recorded snapshots.

        Returns:
            Dictionary containing summary statistics
        """
        if not self.enabled or not self.snapshots:
            return {}

        with self._lock:
            summary = {
                "num_snapshots": len(self.snapshots),
                "checkpoints": [s.checkpoint_name for s in self.snapshots],
                "peak_memory_mb": max(s.total_mb for s in self.snapshots),
                "peak_memory_checkpoint": max(
                    self.snapshots, key=lambda s: s.total_mb
                ).checkpoint_name,
                "final_memory_mb": (
                    self.snapshots[-1].total_mb if self.snapshots else 0
                ),
            }

            # Add operation tracking summary if available
            operation_allocations = self._operation_tracker.get_operation_allocations()
            if operation_allocations:
                summary["operations"] = generate_operation_summary(
                    operation_allocations
                )

            return summary

    def reset(self) -> None:
        """Clear all recorded snapshots."""
        with self._lock:
            self.snapshots.clear()

    def _start_periodic_tracking(self) -> None:
        """Start background thread for periodic memory snapshots."""
        self._tracker_thread = threading.Thread(
            target=self._periodic_snapshot_loop, daemon=True, name="MemoryTracker"
        )
        self._tracker_thread.start()

    def _periodic_snapshot_loop(self) -> None:
        """Background loop that captures memory snapshots periodically."""
        snapshot_count = 0
        while not self._stop_event.is_set():
            # Wait for the interval, or until stop is signaled
            if self._stop_event.wait(timeout=self._interval):
                break

            # Capture snapshot
            snapshot_count += 1
            snapshot_name = f"snapshot_{snapshot_count:04d}"
            with self._lock:
                snapshot = self._tensor_scanner.capture_snapshot(
                    snapshot_name,
                    self._operation_tracker.get_current_operation(),
                    self._operation_tracker.record_lifecycle_event,
                )
                self.snapshots.append(snapshot)

    def stop(self) -> None:
        """Stop periodic tracking and save final snapshot."""
        if not self.enabled:
            return

        self._stop_event.set()
        if self._tracker_thread and self._tracker_thread.is_alive():
            self._tracker_thread.join(timeout=2.0)

        # Save final snapshot
        self.save_snapshot()

    # Public API for hook-based tracker integration
    def capture_snapshot(self, checkpoint_name: str):
        """Public method to capture a memory snapshot."""
        return self._tensor_scanner.capture_snapshot(
            checkpoint_name,
            self._operation_tracker.get_current_operation(),
            self._operation_tracker.record_lifecycle_event,
        )

    def add_operation_log_entry(self, entry: dict):
        """Public method to add an entry to the operation log."""
        self._operation_log.append(entry)

    def get_operation_log_count(self) -> int:
        """Get the current count of entries in the operation log."""
        return len(self._operation_log)

    def update_tensor_metadata(self, tensor_id: int, **kwargs):
        """Public method to update tensor metadata."""
        self._operation_tracker.update_tensor_metadata(tensor_id, **kwargs)

    def get_hook_tracker(self):
        """Get or create the hook tracker for custom operation logging."""
        if self._hook_tracker is None:
            # Lazy import to avoid circular dependency
            from .hook_based_tracker import (  # pylint: disable=import-outside-toplevel
                create_hook_tracker,
            )

            self._hook_tracker = create_hook_tracker(self)
        return self._hook_tracker


# Global singleton instance
_GLOBAL_TRACKER: MemoryTracker | None = None


def initialize_memory_tracker(
    output_dir: str = ".", enabled: bool = False, enable_timing: bool = False
) -> None:
    """
    Initialize the global memory tracker (unified profiler).

    Args:
        output_dir: Directory where memory snapshots will be saved
        enabled: Whether memory tracking is enabled
        enable_timing: Whether timing profiling is enabled (unified profiler mode)
    """
    global _GLOBAL_TRACKER  # pylint: disable=global-statement
    _GLOBAL_TRACKER = MemoryTracker(
        output_dir=output_dir, enabled=enabled, enable_timing=enable_timing
    )


def get_memory_tracker() -> MemoryTracker:
    """Get the global memory tracker instance."""
    global _GLOBAL_TRACKER  # pylint: disable=global-statement
    if _GLOBAL_TRACKER is None:
        # Initialize with disabled tracker if not initialized
        _GLOBAL_TRACKER = MemoryTracker(enabled=False)
    return _GLOBAL_TRACKER


def memory_checkpoint(name: str) -> nullcontext[None]:
    """
    Context manager or direct call to record a memory checkpoint.

    Can be used as:
        memory_checkpoint("checkpoint_name")
    or:
        with memory_checkpoint("checkpoint_name"):
            # code here

    Args:
        name: Descriptive name for this checkpoint
    """
    tracker = get_memory_tracker()
    tracker.checkpoint(name)
    return nullcontext()


def save_memory_profile(filename: str | None = None) -> str:
    """
    Save recorded memory snapshots to JSON file.

    Args:
        filename: Optional custom filename

    Returns:
        Path to saved file
    """
    tracker = get_memory_tracker()
    return tracker.save_snapshot(filename)


def get_memory_summary() -> dict[str, Any]:
    """Get summary of recorded memory snapshots."""
    tracker = get_memory_tracker()
    return tracker.get_summary()


def stop_memory_tracker() -> None:
    """Stop periodic tracking and save final snapshot."""
    tracker = get_memory_tracker()
    tracker.stop()
