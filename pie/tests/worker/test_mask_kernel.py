import unittest
import time
import torch
import numpy as np
import triton
import triton.language as tl
from numba import njit, prange
from pie_backend.batching import _decode_brle

# =============================================================================
# Numba JIT-Compiled BRLE Decoder
# =============================================================================


@njit(parallel=True, cache=True)
def decode_brle_numba(flattened_masks, mask_indptr, position_ids, token_acc_seq_lens):
    """
    Numba-accelerated BRLE decoder.
    Parallel over tokens, no compile-time loop limits.
    """
    num_tokens = len(position_ids)
    total_bits = token_acc_seq_lens[-1]
    num_bytes = (total_bits + 7) // 8
    packed = np.zeros(num_bytes, dtype=np.uint8)

    for k in prange(num_tokens):
        rle_start = mask_indptr[k]
        rle_end = mask_indptr[k + 1]
        global_bit_start = token_acc_seq_lens[k]
        valid_len = position_ids[k] + 1

        curr_bit_pos = global_bit_start
        bits_consumed = 0
        is_true_run = True

        for run_idx in range(rle_start, rle_end):
            if bits_consumed >= valid_len:
                break

            run_len = flattened_masks[run_idx]
            remaining = valid_len - bits_consumed
            eff_len = min(run_len, remaining)

            if is_true_run and eff_len > 0:
                # Write bits [curr_bit_pos, curr_bit_pos + eff_len)
                for bit_off in range(eff_len):
                    bit_pos = curr_bit_pos + bit_off
                    byte_idx = bit_pos // 8
                    bit_in_byte = bit_pos % 8
                    # Big-endian: bit 0 is MSB (0x80)
                    mask = 1 << (7 - bit_in_byte)
                    packed[byte_idx] |= mask

            bits_consumed += eff_len
            curr_bit_pos += eff_len
            is_true_run = not is_true_run

    return packed


def decode_brle_masks_numba(
    flattened_masks, mask_indptr, position_ids, token_acc_seq_lens
):
    """Wrapper for Numba decoder."""
    return decode_brle_numba(
        flattened_masks.astype(np.int32),
        mask_indptr.astype(np.int32),
        position_ids.astype(np.int32),
        token_acc_seq_lens.astype(np.int32),
    )


# =============================================================================
# Triton Kernel
# =============================================================================


@triton.jit
def brle_decode_kernel(
    flattened_masks_ptr,
    mask_indptr_ptr,
    position_ids_ptr,
    token_acc_seq_lens_ptr,
    packed_masks_ptr,
    n_tokens,
    BLOCK_SIZE: tl.constexpr,
    MAX_RUNS: tl.constexpr,
    MAX_BITS_PER_RUN: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    is_valid = token_idx < n_tokens

    rle_start = tl.load(mask_indptr_ptr + token_idx, mask=is_valid, other=0)
    rle_end = tl.load(mask_indptr_ptr + token_idx + 1, mask=is_valid, other=0)
    global_start_bit = tl.load(
        token_acc_seq_lens_ptr + token_idx, mask=is_valid, other=0
    )
    pos_id = tl.load(position_ids_ptr + token_idx, mask=is_valid, other=0)
    valid_len = pos_id + 1

    curr_bit_pos = global_start_bit
    bits_consumed = tl.zeros([BLOCK_SIZE], dtype=tl.int32)

    for run_idx in range(MAX_RUNS):
        curr_rle_ptr = rle_start + run_idx
        active = (curr_rle_ptr < rle_end) & (bits_consumed < valid_len) & is_valid
        is_true_run = (run_idx % 2) == 0

        run_len = tl.load(flattened_masks_ptr + curr_rle_ptr, mask=active, other=0)
        remaining = valid_len - bits_consumed
        eff_len = tl.minimum(run_len, remaining)

        do_write = active & is_true_run & (eff_len > 0)

        for bit_off in range(MAX_BITS_PER_RUN):
            bit_active = do_write & (bit_off < eff_len)

            bit_pos = curr_bit_pos + bit_off
            word_idx = bit_pos // 32
            bit_in_word = bit_pos % 32
            mask_val = (1 << (31 - bit_in_word)).to(tl.uint32)

            word_ptr = packed_masks_ptr + word_idx
            tl.atomic_or(word_ptr, mask_val, mask=bit_active)

        bits_consumed = bits_consumed + eff_len
        curr_bit_pos = curr_bit_pos + eff_len


def decode_brle_masks_triton(
    flattened_masks,
    mask_indptr,
    position_ids,
    token_acc_seq_lens,
    max_runs=32,
    max_bits_per_run=128,
):
    n_tokens = position_ids.shape[0]
    total_bits = token_acc_seq_lens[-1].item()

    num_words = (total_bits + 31) // 32
    packed_u32 = torch.zeros(
        (num_words + 1,), dtype=torch.uint32, device=flattened_masks.device
    )

    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_tokens, BLOCK_SIZE),)

    brle_decode_kernel[grid](
        flattened_masks,
        mask_indptr,
        position_ids,
        token_acc_seq_lens,
        packed_u32,
        n_tokens=n_tokens,
        BLOCK_SIZE=BLOCK_SIZE,
        MAX_RUNS=max_runs,
        MAX_BITS_PER_RUN=max_bits_per_run,
    )

    packed_bytes = (
        packed_u32.view(torch.uint8).reshape(-1, 4)[:, [3, 2, 1, 0]].reshape(-1)
    )
    num_bytes = (total_bits + 7) // 8
    return packed_bytes[:num_bytes]


# =============================================================================
# CPU Baseline (Pure Python)
# =============================================================================


def decode_brle_masks_cpu(
    flattened_masks_cpu, mask_indptr_cpu, position_ids_cpu, token_acc_seq_lens_cpu
):
    all_rows = []
    num_tokens = len(position_ids_cpu)

    for k in range(num_tokens):
        start = mask_indptr_cpu[k]
        end = mask_indptr_cpu[k + 1]
        brle_data = flattened_masks_cpu[start:end].tolist()
        full_decoded = _decode_brle(brle_data)

        valid_len = position_ids_cpu[k] + 1
        seq_len = token_acc_seq_lens_cpu[k + 1] - token_acc_seq_lens_cpu[k]

        row = np.zeros(seq_len, dtype=bool)
        take_len = min(valid_len, len(full_decoded))
        row[:take_len] = full_decoded[:take_len]
        all_rows.append(row)

    if not all_rows:
        return np.array([], dtype=np.uint8)
    return np.packbits(np.concatenate(all_rows))


# =============================================================================
# Tests
# =============================================================================


class TestMaskKernel(unittest.TestCase):
    def test_all_implementations_match(self):
        """Verify all three implementations produce identical output."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n{'='*60}")
        print("CORRECTNESS TEST: All Implementations")
        print(f"{'='*60}")

        np.random.seed(42)

        # Generate test data
        num_tokens = 100
        flattened_list = []
        mask_indptr_list = [0]
        generated_lens = []

        for _ in range(num_tokens):
            num_runs = np.random.randint(1, 15)
            runs = np.random.randint(1, 80, size=num_runs).tolist()
            flattened_list.extend(runs)
            mask_indptr_list.append(len(flattened_list))
            generated_lens.append(sum(runs))

        flattened_np = np.array(flattened_list, dtype=np.int32)
        mask_indptr_np = np.array(mask_indptr_list, dtype=np.int32)
        position_ids_np = np.array(
            [max(0, l - 5) for l in generated_lens], dtype=np.int32
        )

        seq_lens = [l + 10 for l in generated_lens]
        acc_seq_lens = [0]
        for sl in seq_lens:
            acc_seq_lens.append(acc_seq_lens[-1] + sl)
        token_acc_seq_lens_np = np.array(acc_seq_lens, dtype=np.int32)

        # CPU Baseline
        packed_cpu = decode_brle_masks_cpu(
            flattened_np, mask_indptr_np, position_ids_np, token_acc_seq_lens_np
        )
        print(f"CPU Baseline: {len(packed_cpu)} bytes")

        # Numba
        # Warmup
        _ = decode_brle_masks_numba(
            flattened_np, mask_indptr_np, position_ids_np, token_acc_seq_lens_np
        )
        packed_numba = decode_brle_masks_numba(
            flattened_np, mask_indptr_np, position_ids_np, token_acc_seq_lens_np
        )
        print(f"Numba: {len(packed_numba)} bytes")

        # Verify Numba matches CPU
        np.testing.assert_array_equal(packed_cpu, packed_numba)
        print("  Numba == CPU: ✓")

        # Triton (if GPU available)
        if device == "cuda":
            flattened_gpu = torch.tensor(
                flattened_list, dtype=torch.int32, device=device
            )
            mask_indptr_gpu = torch.tensor(
                mask_indptr_list, dtype=torch.int32, device=device
            )
            position_ids_gpu = torch.tensor(
                position_ids_np, dtype=torch.int32, device=device
            )
            token_acc_seq_lens_gpu = torch.tensor(
                acc_seq_lens, dtype=torch.int32, device=device
            )

            packed_triton = decode_brle_masks_triton(
                flattened_gpu,
                mask_indptr_gpu,
                position_ids_gpu,
                token_acc_seq_lens_gpu,
                max_runs=32,
                max_bits_per_run=128,
            )
            packed_triton_cpu = packed_triton.cpu().numpy()
            print(f"Triton: {len(packed_triton_cpu)} bytes")

            np.testing.assert_array_equal(packed_cpu, packed_triton_cpu)
            print("  Triton == CPU: ✓")

        print("\nAll implementations match!")

    def test_performance_comparison(self):
        """Compare performance of CPU, Numba, and Triton."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")

        np.random.seed(123)
        token_counts = [50, 100, 500, 1000]

        print(
            f"\n{'Tokens':<8} | {'CPU (ms)':<12} | {'Numba (ms)':<12} | {'Triton (ms)':<12} | {'Best':<10}"
        )
        print("-" * 70)

        for num_tokens in token_counts:
            flattened_list = []
            mask_indptr_list = [0]
            generated_lens = []

            for _ in range(num_tokens):
                num_runs = np.random.randint(1, 10)
                runs = np.random.randint(1, 50, size=num_runs).tolist()
                flattened_list.extend(runs)
                mask_indptr_list.append(len(flattened_list))
                generated_lens.append(sum(runs))

            flattened_np = np.array(flattened_list, dtype=np.int32)
            mask_indptr_np = np.array(mask_indptr_list, dtype=np.int32)
            position_ids_np = np.array([l - 1 for l in generated_lens], dtype=np.int32)

            seq_lens = [l + 5 for l in generated_lens]
            acc_seq_lens = [0]
            for sl in seq_lens:
                acc_seq_lens.append(acc_seq_lens[-1] + sl)
            token_acc_seq_lens_np = np.array(acc_seq_lens, dtype=np.int32)

            # CPU Baseline
            n_iter = 10
            start = time.perf_counter()
            for _ in range(n_iter):
                _ = decode_brle_masks_cpu(
                    flattened_np, mask_indptr_np, position_ids_np, token_acc_seq_lens_np
                )
            cpu_time = (time.perf_counter() - start) / n_iter * 1000

            # Numba (warmup already done in previous test)
            start = time.perf_counter()
            for _ in range(n_iter):
                _ = decode_brle_masks_numba(
                    flattened_np, mask_indptr_np, position_ids_np, token_acc_seq_lens_np
                )
            numba_time = (time.perf_counter() - start) / n_iter * 1000

            # Triton
            triton_time = float("inf")
            if device == "cuda":
                flattened_gpu = torch.tensor(
                    flattened_list, dtype=torch.int32, device=device
                )
                mask_indptr_gpu = torch.tensor(
                    mask_indptr_list, dtype=torch.int32, device=device
                )
                position_ids_gpu = torch.tensor(
                    position_ids_np, dtype=torch.int32, device=device
                )
                token_acc_seq_lens_gpu = torch.tensor(
                    acc_seq_lens, dtype=torch.int32, device=device
                )

                # Warmup
                _ = decode_brle_masks_triton(
                    flattened_gpu,
                    mask_indptr_gpu,
                    position_ids_gpu,
                    token_acc_seq_lens_gpu,
                    max_runs=16,
                    max_bits_per_run=64,
                )
                torch.cuda.synchronize()

                start = time.perf_counter()
                for _ in range(n_iter):
                    _ = decode_brle_masks_triton(
                        flattened_gpu,
                        mask_indptr_gpu,
                        position_ids_gpu,
                        token_acc_seq_lens_gpu,
                        max_runs=16,
                        max_bits_per_run=64,
                    )
                torch.cuda.synchronize()
                triton_time = (time.perf_counter() - start) / n_iter * 1000

            # Determine best
            times = {"CPU": cpu_time, "Numba": numba_time, "Triton": triton_time}
            best = min(times, key=times.get)

            triton_str = f"{triton_time:.3f}" if device == "cuda" else "N/A"
            print(
                f"{num_tokens:<8} | {cpu_time:<12.3f} | {numba_time:<12.3f} | {triton_str:<12} | {best:<10}"
            )

        print(
            f"\nNote: For small batches, Numba may outperform Triton due to GPU launch overhead."
        )


if __name__ == "__main__":
    unittest.main()
