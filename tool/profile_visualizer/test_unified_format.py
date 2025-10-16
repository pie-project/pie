"""
Test that profile visualizer correctly handles unified profiler format.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to import server
sys.path.insert(0, str(Path(__file__).parent))

from server import normalize_profiling_data

# Test data representing unified profiler format
unified_data = {
    "format": "unified_profiler",
    "metadata": {
        "generated_at": "2025-10-14T20:00:00",
        "timing_profiling_enabled": True,
        "num_timing_nodes": 3,
    },
    "profiling_tree": [
        {
            "name": "forward_pass",
            "full_path": "forward_pass",
            "avg_latency_ms": 100.5,
            "self_time_ms": 55.3,
            "min_ms": 95.0,
            "max_ms": 105.0,
            "std_dev_ms": 3.5,
            "samples": 10,
            "children": [
                {
                    "name": "layer1",
                    "full_path": "forward_pass.layer1",
                    "avg_latency_ms": 45.2,
                    "self_time_ms": 45.2,
                    "min_ms": 44.0,
                    "max_ms": 46.0,
                    "std_dev_ms": 0.8,
                    "samples": 10,
                    "module_type": "Linear",
                    "typical_input_tensors": [
                        {
                            "id": 12345,
                            "shape": [1, 512],
                            "dtype": "float16",
                            "device": "mps:0",
                            "size_mb": 0.001,
                        }
                    ],
                    "typical_output_tensors": [
                        {
                            "id": 12346,
                            "shape": [1, 512],
                            "dtype": "float16",
                            "device": "mps:0",
                            "size_mb": 0.001,
                        }
                    ],
                }
            ],
        }
    ],
    "tensor_lifecycle": {},
    "snapshots": [],
}

# Invalid old format data (should fail)
old_format_data = {
    "metadata": {"generated_at": "2025-10-14T20:00:00"},
    "operation_log": [
        {
            "operation_name": "Linear.layer1",
            "timestamp": "2025-10-14T20:00:00.100",
            "module_type": "Linear",
        }
    ],
    "snapshots": [],
}

print("Testing Profile Visualizer - Unified Profiler Only")
print("=" * 60)

# Test 1: Normalization of valid data
print("\n1. Testing data normalization:")
normalized = normalize_profiling_data(unified_data.copy())

print(f"   Normalized format: {normalized['format']}")
print(f"   Has profiling_tree: {'profiling_tree' in normalized}")
print(f"   Tree nodes: {len(normalized['profiling_tree'])}")

assert normalized["format"] == "unified_profiler"
assert "profiling_tree" in normalized
assert len(normalized["profiling_tree"]) > 0

# Verify tree structure is preserved
root = normalized["profiling_tree"][0]
assert root["name"] == "forward_pass"
assert "children" in root
assert len(root["children"]) == 1
assert root["children"][0]["name"] == "layer1"

# Verify timing data is preserved
assert root["avg_latency_ms"] == 100.5
assert root["self_time_ms"] == 55.3
assert root["samples"] == 10

# Verify tensor data is preserved in children
child = root["children"][0]
assert "typical_input_tensors" in child
assert len(child["typical_input_tensors"]) == 1
assert child["typical_input_tensors"][0]["shape"] == [1, 512]

print("   ✅ Normalization preserves all data correctly")

# Test 2: Missing required fields should be rejected
print("\n2. Testing validation of required fields:")
try:
    normalize_profiling_data(old_format_data.copy())
    print("   ❌ Should have raised ValueError for missing profiling_tree field")
    sys.exit(1)
except ValueError as e:
    print(f"   ✅ Missing field correctly rejected: {str(e)}")
    assert "profiling_tree" in str(e), "Error message should mention missing profiling_tree"

print("\n" + "=" * 60)
print("✅ All visualizer tests passed!")
print("✅ Profile visualizer only supports unified_profiler format")
print("=" * 60)
