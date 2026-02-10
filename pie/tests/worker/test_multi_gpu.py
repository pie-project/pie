"""
Multi-GPU inference test for PIE backend.

This test verifies that multi-GPU tensor parallel inference works correctly
by directly using the Runtime class with multiple devices.

Run with: python -m pytest test/test_multi_gpu.py -v
Or directly: python test/test_multi_gpu.py
"""

from __future__ import annotations

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pie_backend.backend import Backend, RuntimeConfig


# Default test model - use a small model for faster testing
TEST_MODEL = "qwen-3-0.6b"

# Test devices - adjust based on your setup
TEST_DEVICES = ["cuda:2", "cuda:3"]


def init_distributed(rank: int, world_size: int, port: int, backend: str = "nccl"):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def test_single_gpu_embed_tokens():
    """Test embed_tokens on single GPU (baseline test)."""
    print("=" * 60)
    print("Single GPU Embed Tokens Test")
    print("=" * 60)

    # Create single-GPU config
    config = RuntimeConfig.from_args(
        hf_repo=TEST_MODEL,
        device=TEST_DEVICES[0],
    )
    # Create mock log_queue that discards messages
    from multiprocessing import Queue

    mock_log_queue = Queue()
    runtime = Backend(config, log_queue=mock_log_queue)
    device = config.device

    print(f"\n[1] Testing embed_tokens on {device}...")
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
    embeddings = runtime.engine.embed_tokens(token_ids)

    print(f"    Input shape: {token_ids.shape}")
    print(f"    Output shape: {embeddings.shape}")
    print(f"    Expected hidden_size: {runtime.model_config.dim_hidden}")

    # Verify hidden size is correct
    assert embeddings.shape[0] == 5, f"Expected 5 tokens, got {embeddings.shape[0]}"
    assert embeddings.shape[1] == runtime.model_config.dim_hidden, (
        f"Hidden size mismatch: expected {runtime.model_config.dim_hidden}, "
        f"got {embeddings.shape[1]}"
    )

    print("    ✓ Single GPU embed_tokens passed!")
    return True


def test_single_gpu_embed_inputs():
    """Test embed_inputs method exists and works on single GPU."""
    print("\n" + "=" * 60)
    print("Single GPU Embed Inputs Test")
    print("=" * 60)

    config = RuntimeConfig.from_args(
        hf_repo=TEST_MODEL,
        device=TEST_DEVICES[0],
    )
    # Create mock log_queue that discards messages
    from multiprocessing import Queue

    mock_log_queue = Queue()
    runtime = Backend(config, log_queue=mock_log_queue)
    device = config.device

    print(f"\n[1] Testing embed_inputs on {device}...")

    # Create batch metadata similar to what batching.py produces
    batch_metadata = {
        "token_ids": [1, 2, 3, 4, 5],
    }

    embeddings = runtime.engine.embed_inputs(batch_metadata)

    print(f"    Input tokens: {batch_metadata['token_ids']}")
    print(f"    Output shape: {embeddings.shape}")

    assert embeddings.shape[0] == 5
    assert embeddings.shape[1] == runtime.model_config.dim_hidden

    print("    ✓ Single GPU embed_inputs passed!")
    return True


def test_single_gpu_forward_pass():
    """Test full forward pass on single GPU."""
    print("\n" + "=" * 60)
    print("Single GPU Forward Pass Test")
    print("=" * 60)

    config = RuntimeConfig.from_args(
        hf_repo=TEST_MODEL,
        device=TEST_DEVICES[0],
    )
    # Create mock log_queue that discards messages
    from multiprocessing import Queue

    mock_log_queue = Queue()
    runtime = Backend(config, log_queue=mock_log_queue)
    device = config.device

    print(f"\n[1] Testing full forward pass on {device}...")

    # Test embed_tokens
    token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
    embeddings = runtime.engine.embed_tokens(token_ids)
    print(f"    Embeddings shape: {embeddings.shape}")

    # Test lm_head (final projection)
    logits = runtime.engine.lm_head(embeddings)
    print(f"    Logits shape: {logits.shape}")

    # Verify logits have vocab_size dimension
    expected_vocab_size = runtime.model_config.num_vocabs
    assert (
        logits.shape[-1] == expected_vocab_size
    ), f"Logits vocab size mismatch: expected {expected_vocab_size}, got {logits.shape[-1]}"

    print("    ✓ Single GPU forward pass passed!")
    return True


def _multi_gpu_worker(rank: int, world_size: int, port: int, test_fn: str):
    """
    Worker function for multi-GPU tests.

    Args:
        rank: Process rank
        world_size: Total number of processes
        port: Port for distributed coordination
        test_fn: Which test to run ('embed', 'forward')
    """
    try:
        # Initialize distributed
        init_distributed(rank, world_size, port)

        # Create multi-GPU config
        config = RuntimeConfig.from_args(
            hf_repo=TEST_MODEL,
            devices=TEST_DEVICES[:world_size],
            rank=rank,
        )

        # Each rank creates its Runtime (loads model shard)
        # Create a simple mock log_queue that discards messages
        from multiprocessing import Queue

        mock_log_queue = Queue()
        runtime = Backend(config, log_queue=mock_log_queue)
        device = config.device

        if rank == 0:
            print(f"\n[Worker {rank}] Testing on {device}...")

        # Synchronize before test
        dist.barrier()

        if test_fn == "embed":
            # Test embed_tokens - should preserve hidden_size with all_reduce
            token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
            embeddings = runtime.engine.embed_tokens(token_ids)

            if rank == 0:
                print(f"    Embeddings shape: {embeddings.shape}")
                print(f"    Expected hidden_size: {runtime.model_config.dim_hidden}")

                # Debug: print embedding weight shape
                embed_weight = runtime.engine.weights.get("embed_token")
                print(f"    Embed weight shape: {embed_weight.shape}")

                # The key test: hidden_size should NOT be halved
                assert embeddings.shape[1] == runtime.model_config.dim_hidden, (
                    f"FAIL: Hidden size mismatch! Expected {runtime.model_config.dim_hidden}, "
                    f"got {embeddings.shape[1]}. Embeddings were incorrectly sharded."
                )
                print("    ✓ Multi-GPU embed_tokens preserves hidden_size!")

        elif test_fn == "forward":
            # Test forward pass: embed_tokens -> lm_head
            # This tests the critical distributed operations (all_gather in both)
            token_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32, device=device)
            embeddings = runtime.engine.embed_tokens(token_ids)

            if rank == 0:
                print(f"    Embeddings shape: {embeddings.shape}")

                # Verify embeddings shape is correct (full hidden_size)
                assert embeddings.shape[1] == runtime.model_config.dim_hidden, (
                    f"FAIL: Embeddings hidden size mismatch! Expected {runtime.model_config.dim_hidden}, "
                    f"got {embeddings.shape[1]}"
                )
                print("    ✓ Multi-GPU embed_tokens produces correct hidden_size!")

            # Test lm_head (has all_gather for column-parallel weights)
            logits = runtime.engine.lm_head(embeddings)

            if rank == 0:
                print(f"    Logits shape: {logits.shape}")

                # Verify logits shape is correct (full vocab_size)
                assert logits.shape[-1] == runtime.model_config.num_vocabs, (
                    f"FAIL: Logits vocab_size mismatch! Expected {runtime.model_config.num_vocabs}, "
                    f"got {logits.shape[-1]}"
                )
                print("    ✓ Multi-GPU lm_head produces correct vocab_size!")

        # Synchronize after test
        dist.barrier()

    except Exception as e:
        print(f"[Worker {rank}] Error: {e}")
        import traceback

        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


def test_multi_gpu_embed_tokens():
    """Test embed_tokens preserves hidden_size on multi-GPU."""
    print("\n" + "=" * 60)
    print("Multi-GPU Embed Tokens Test")
    print("=" * 60)

    world_size = len(TEST_DEVICES)
    if world_size < 2:
        print("    [SKIP] Need at least 2 GPUs for multi-GPU test")
        return True

    if not torch.cuda.is_available():
        print("    [SKIP] CUDA not available")
        return True

    print(f"\n[1] Spawning {world_size} processes for devices: {TEST_DEVICES}")

    import random

    port = 29500 + random.randint(0, 1000)

    mp.spawn(
        _multi_gpu_worker,
        args=(world_size, port, "embed"),
        nprocs=world_size,
        join=True,
    )

    print("    ✓ Multi-GPU embed_tokens test passed!")
    return True


def test_multi_gpu_forward_pass():
    """Test full forward pass on multi-GPU."""
    print("\n" + "=" * 60)
    print("Multi-GPU Forward Pass Test")
    print("=" * 60)

    world_size = len(TEST_DEVICES)
    if world_size < 2:
        print("    [SKIP] Need at least 2 GPUs for multi-GPU test")
        return True

    if not torch.cuda.is_available():
        print("    [SKIP] CUDA not available")
        return True

    print(f"\n[1] Spawning {world_size} processes for devices: {TEST_DEVICES}")

    import random

    port = 29500 + random.randint(0, 1000)

    mp.spawn(
        _multi_gpu_worker,
        args=(world_size, port, "forward"),
        nprocs=world_size,
        join=True,
    )

    print("    ✓ Multi-GPU forward pass test passed!")
    return True


def run_all_tests():
    """Run all multi-GPU tests."""
    print("\n" + "=" * 60)
    print("PIE Multi-GPU Inference Tests")
    print("=" * 60)
    print(f"Test model: {TEST_MODEL}")
    print(f"Test devices: {TEST_DEVICES}")

    results = {}

    # Single GPU tests (baseline)
    try:
        results["single_gpu_embed_tokens"] = test_single_gpu_embed_tokens()
    except Exception as e:
        print(f"    ✗ Single GPU embed_tokens failed: {e}")
        results["single_gpu_embed_tokens"] = False

    try:
        results["single_gpu_embed_inputs"] = test_single_gpu_embed_inputs()
    except Exception as e:
        print(f"    ✗ Single GPU embed_inputs failed: {e}")
        results["single_gpu_embed_inputs"] = False

    try:
        results["single_gpu_forward_pass"] = test_single_gpu_forward_pass()
    except Exception as e:
        print(f"    ✗ Single GPU forward pass failed: {e}")
        results["single_gpu_forward_pass"] = False

    # Multi-GPU tests
    try:
        results["multi_gpu_embed_tokens"] = test_multi_gpu_embed_tokens()
    except Exception as e:
        print(f"    ✗ Multi-GPU embed_tokens failed: {e}")
        import traceback

        traceback.print_exc()
        results["multi_gpu_embed_tokens"] = False

    try:
        results["multi_gpu_forward_pass"] = test_multi_gpu_forward_pass()
    except Exception as e:
        print(f"    ✗ Multi-GPU forward pass failed: {e}")
        import traceback

        traceback.print_exc()
        results["multi_gpu_forward_pass"] = False

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

    parser = argparse.ArgumentParser(description="PIE Multi-GPU Test")
    parser.add_argument(
        "--test",
        type=str,
        choices=["single", "multi", "all"],
        default="all",
        help="Which tests to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=TEST_MODEL,
        help="Model to test",
    )
    parser.add_argument(
        "--devices",
        type=str,
        nargs="+",
        default=TEST_DEVICES,
        help="Devices to use (e.g., cuda:0 cuda:1)",
    )
    args = parser.parse_args()

    # Update globals
    TEST_MODEL = args.model
    TEST_DEVICES = args.devices

    if args.test == "single":
        test_single_gpu_embed_tokens()
        test_single_gpu_embed_inputs()
        test_single_gpu_forward_pass()
    elif args.test == "multi":
        test_multi_gpu_embed_tokens()
        test_multi_gpu_forward_pass()
    else:
        run_all_tests()
