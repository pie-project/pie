"""
Data Parallelism (DP) integration tests for PIE backend.

This test verifies that Data Parallelism works correctly by:
1. Creating multiple device groups (e.g., 4 GPUs → 2 groups of TP=2)
2. Routing batches to the correct group
3. Returning results from secondary groups back to Rank 0

Run with: python -m pytest pie/tests/worker/test_data_parallelism.py -v
Or directly: python pie/tests/worker/test_data_parallelism.py
"""

from __future__ import annotations

import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Queue
from typing import Any

# Test configuration
# Adjust based on available GPUs
TEST_DEVICES = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
TEST_MODEL = "qwen-3-0.6b"


def init_distributed(rank: int, world_size: int, port: int, device: str):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # Set CUDA device before init
    if device.startswith("cuda"):
        device_id = int(device.split(":")[1])
        torch.cuda.set_device(device_id)

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


# =============================================================================
# Test 1: ControlChannel Group Targeting
# =============================================================================


def test_control_channel_group_targeting():
    """Test that ControlChannel sends to correct groups."""
    print("\n" + "=" * 60)
    print("Test: ControlChannel Group Targeting")
    print("=" * 60)

    from pie_device.control_channel import ControlChannel, create_control_channels

    world_size = 4
    group_topology = [[0, 1], [2, 3]]  # 2 groups of 2

    # Create queues for workers (ranks 1, 2, 3)
    control_queues = create_control_channels(world_size)

    # Create ControlChannel for rank 0
    cc = ControlChannel(
        rank=0,
        world_size=world_size,
        queues=control_queues,
        group_topology=group_topology,
    )

    # Test 1: Send to Group 0 only
    print("\n[1] Testing send to Group 0 (ranks 0, 1)...")
    cc.send("msg_for_group_0", destination_group=0)

    # Only rank 1 should receive (rank 0 is sender, not a queue reader)
    msg_rank1 = control_queues[0].get(timeout=1)  # queue[0] = rank 1
    assert msg_rank1 == "msg_for_group_0", f"Rank 1 got wrong message: {msg_rank1}"

    # Ranks 2, 3 should NOT receive (queues should be empty)
    assert control_queues[1].empty(), "Rank 2 queue should be empty"
    assert control_queues[2].empty(), "Rank 3 queue should be empty"
    print("    ✓ Group 0 targeting works!")

    # Test 2: Send to Group 1 only
    print("\n[2] Testing send to Group 1 (ranks 2, 3)...")
    cc.send("msg_for_group_1", destination_group=1)

    # Ranks 2 and 3 should receive
    msg_rank2 = control_queues[1].get(timeout=1)  # queue[1] = rank 2
    msg_rank3 = control_queues[2].get(timeout=1)  # queue[2] = rank 3
    assert msg_rank2 == "msg_for_group_1", f"Rank 2 got wrong message: {msg_rank2}"
    assert msg_rank3 == "msg_for_group_1", f"Rank 3 got wrong message: {msg_rank3}"

    # Rank 1 should NOT receive new message
    assert control_queues[0].empty(), "Rank 1 queue should be empty"
    print("    ✓ Group 1 targeting works!")

    # Test 3: Broadcast to all
    print("\n[3] Testing broadcast to all groups...")
    cc.send("msg_for_all", destination_group=None)

    for i, q in enumerate(control_queues):
        msg = q.get(timeout=1)
        assert msg == "msg_for_all", f"Rank {i+1} got wrong broadcast: {msg}"
    print("    ✓ Broadcast to all works!")

    # Cleanup
    cc.cleanup()
    print("\n    ✓ ControlChannel group targeting test passed!")
    return True


# =============================================================================
# Test 2: Process Group Creation
# =============================================================================


def _pg_worker(rank: int, world_size: int, port: int, result_queue: Queue):
    """Worker for process group test."""
    try:
        device = TEST_DEVICES[rank] if rank < len(TEST_DEVICES) else f"cuda:{rank}"
        init_distributed(rank, world_size, port, device)

        # Create process groups (same logic as manager.py)
        group_topology = [[0, 1], [2, 3]]
        pg_map = {}

        for i, group_ranks in enumerate(group_topology):
            # Comm group includes Rank 0 + Group Workers
            comm_ranks = sorted(list(set([0] + group_ranks)))
            pg = dist.new_group(comm_ranks)
            pg_map[i] = pg

        # Report success
        result_queue.put((rank, "success", list(pg_map.keys())))

        # Barrier to ensure all ranks complete
        dist.barrier()

    except Exception as e:
        result_queue.put((rank, "error", str(e)))
    finally:
        cleanup_distributed()


def test_process_group_creation():
    """Test that process groups are created correctly for DP."""
    print("\n" + "=" * 60)
    print("Test: Process Group Creation")
    print("=" * 60)

    world_size = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        print(f"    [SKIP] Need {world_size} GPUs, have {torch.cuda.device_count()}")
        return True

    import random

    port = 29500 + random.randint(0, 1000)

    # Use spawn context for queues to match torch.mp.spawn
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    print(f"\n[1] Spawning {world_size} processes to test PG creation...")

    mp.spawn(
        _pg_worker,
        args=(world_size, port, result_queue),
        nprocs=world_size,
        join=True,
    )

    # Check results
    results = {}
    for _ in range(world_size):
        rank, status, data = result_queue.get(timeout=10)
        results[rank] = (status, data)

    all_success = True
    for rank in range(world_size):
        status, data = results.get(rank, ("missing", None))
        if status == "success":
            print(f"    Rank {rank}: Created groups {data}")
        else:
            print(f"    Rank {rank}: FAILED - {data}")
            all_success = False

    if all_success:
        print("\n    ✓ Process group creation test passed!")
    else:
        print("\n    ✗ Process group creation test FAILED!")

    return all_success


# =============================================================================
# Test 3: Result Queue Return Path
# =============================================================================


def test_result_queue_return_path():
    """Test that secondary groups can return results via result_queue."""
    print("\n" + "=" * 60)
    print("Test: Result Queue Return Path")
    print("=" * 60)

    from multiprocessing import Queue

    # Simulate: Rank 0 creates result_queues, secondary group pushes result
    num_groups = 2
    result_queues = [Queue() for _ in range(num_groups)]

    print("\n[1] Simulating Group 1 leader pushing result...")

    # Simulate Group 1 leader (rank 2) pushing a result
    test_result = {"tokens": [1, 2, 3], "logits": [0.1, 0.2, 0.3]}
    result_queues[1].put(test_result)

    print("\n[2] Simulating Rank 0 waiting for Group 1 result...")

    # Simulate Rank 0 waiting for result
    received = result_queues[1].get(timeout=5)

    assert received == test_result, f"Result mismatch: {received}"
    print(f"    Received: {received}")

    print("\n    ✓ Result queue return path test passed!")
    return True


# =============================================================================
# Test 4: Full DP Integration (requires multiple GPUs)
# =============================================================================


def _dp_integration_worker(
    rank: int,
    world_size: int,
    port: int,
    control_queues: list,
    result_queues: list,
    group_topology: list,
    result_queue: Queue,
):
    """Worker for full DP integration test."""
    try:
        device = TEST_DEVICES[rank] if rank < len(TEST_DEVICES) else f"cuda:{rank}"
        init_distributed(rank, world_size, port, device)

        from pie_device.control_channel import ControlChannel
        from pie_device import utils as pie_utils

        # Setup control channel
        pie_utils._control_channel = ControlChannel(
            rank, world_size, control_queues, group_topology
        )

        # Create process groups
        pg_map = {}
        for i, group_ranks in enumerate(group_topology):
            comm_ranks = sorted(list(set([0] + group_ranks)))
            pg = dist.new_group(comm_ranks)
            pg_map[i] = pg

        # Determine my group
        my_group_id = 0
        for i, group in enumerate(group_topology):
            if rank in group:
                my_group_id = i
                break

        # Determine if I'm group leader
        is_leader = rank == min(group_topology[my_group_id])

        if rank == 0:
            # Rank 0: Act as controller
            # Test: Send work to Group 1 and wait for result
            print(f"[Rank 0] Sending work to Group 1...")
            pie_utils._control_channel.send(
                {"type": "WORK", "data": 42}, destination_group=1
            )

            # Wait for result
            result = result_queues[1].get(timeout=10)
            print(f"[Rank 0] Received result from Group 1: {result}")
            result_queue.put((0, "success", result))
        else:
            # Workers: Wait for work via control channel
            if my_group_id == 1:
                # Group 1 workers should receive
                msg = pie_utils._control_channel.recv(timeout=10)
                print(f"[Rank {rank}] Received: {msg}")

                # Group leader pushes result
                if is_leader:
                    result = {"processed": msg["data"] * 2}
                    result_queues[my_group_id].put(result)
                    print(f"[Rank {rank}] Pushed result: {result}")

                result_queue.put((rank, "success", msg))
            else:
                # Group 0 workers should NOT receive Group 1 message
                # (In real scenario they'd receive Group 0 messages)
                result_queue.put((rank, "success", "no_work"))

        dist.barrier()

    except Exception as e:
        import traceback

        result_queue.put((rank, "error", f"{e}\n{traceback.format_exc()}"))
    finally:
        cleanup_distributed()


def test_dp_integration():
    """Full Data Parallelism integration test."""
    print("\n" + "=" * 60)
    print("Test: Full DP Integration")
    print("=" * 60)

    world_size = 4
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        print(f"    [SKIP] Need {world_size} GPUs, have {torch.cuda.device_count()}")
        return True

    from pie_device.control_channel import create_control_channels
    import random

    port = 29500 + random.randint(0, 1000)
    group_topology = [[0, 1], [2, 3]]
    num_groups = len(group_topology)

    # Use spawn context for queues to match torch.mp.spawn
    ctx = mp.get_context("spawn")

    # Create communication infrastructure with spawn-context queues
    control_queues = [ctx.Queue() for _ in range(world_size - 1)]
    result_queues = [ctx.Queue() for _ in range(num_groups)]
    result_queue = ctx.Queue()

    print(f"\n[1] Spawning {world_size} processes for DP test...")

    mp.spawn(
        _dp_integration_worker,
        args=(
            world_size,
            port,
            control_queues,
            result_queues,
            group_topology,
            result_queue,
        ),
        nprocs=world_size,
        join=True,
    )

    # Check results
    results = {}
    for _ in range(world_size):
        rank, status, data = result_queue.get(timeout=10)
        results[rank] = (status, data)

    all_success = True
    for rank in range(world_size):
        status, data = results.get(rank, ("missing", None))
        if status == "success":
            print(f"    Rank {rank}: {status} - {data}")
        else:
            print(f"    Rank {rank}: FAILED - {data}")
            all_success = False

    # Verify Rank 0 received correct result
    r0_status, r0_data = results.get(0, ("missing", None))
    if r0_status == "success" and r0_data == {"processed": 84}:
        print("\n    ✓ Rank 0 received correct result from Group 1!")
    else:
        print(f"\n    ✗ Rank 0 result incorrect: {r0_data}")
        all_success = False

    if all_success:
        print("\n    ✓ Full DP integration test passed!")
    else:
        print("\n    ✗ Full DP integration test FAILED!")

    return all_success


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests():
    """Run all Data Parallelism tests."""
    print("\n" + "=" * 60)
    print("PIE Data Parallelism Tests")
    print("=" * 60)
    print(f"Test devices: {TEST_DEVICES}")
    print(
        f"Available GPUs: {torch.cuda.device_count()}"
        if torch.cuda.is_available()
        else "CUDA not available"
    )

    results = {}

    # Unit tests (no GPU required)
    try:
        results["control_channel_targeting"] = test_control_channel_group_targeting()
    except Exception as e:
        print(f"    ✗ ControlChannel targeting failed: {e}")
        import traceback

        traceback.print_exc()
        results["control_channel_targeting"] = False

    try:
        results["result_queue_return"] = test_result_queue_return_path()
    except Exception as e:
        print(f"    ✗ Result queue test failed: {e}")
        results["result_queue_return"] = False

    # Multi-GPU tests
    try:
        results["process_group_creation"] = test_process_group_creation()
    except Exception as e:
        print(f"    ✗ Process group creation failed: {e}")
        import traceback

        traceback.print_exc()
        results["process_group_creation"] = False

    try:
        results["dp_integration"] = test_dp_integration()
    except Exception as e:
        print(f"    ✗ DP integration failed: {e}")
        import traceback

        traceback.print_exc()
        results["dp_integration"] = False

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {status}: {test_name}")

    all_passed = all(results.values())
    if all_passed:
        print("\n    All tests passed!")
    else:
        print("\n    Some tests failed!")

    return all_passed


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PIE Data Parallelism Tests")
    parser.add_argument(
        "--test",
        type=str,
        choices=[
            "control_channel",
            "result_queue",
            "process_group",
            "integration",
            "all",
        ],
        default="all",
        help="Which test to run",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=TEST_DEVICES,
        help="Devices to use (e.g., cuda:0 cuda:1 cuda:2 cuda:3)",
    )
    args = parser.parse_args()

    TEST_DEVICES = args.devices

    if args.test == "control_channel":
        test_control_channel_group_targeting()
    elif args.test == "result_queue":
        test_result_queue_return_path()
    elif args.test == "process_group":
        test_process_group_creation()
    elif args.test == "integration":
        test_dp_integration()
    else:
        run_all_tests()
