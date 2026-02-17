"""Benchmark: flashinfer_metal paged attention vs MLX scaled_dot_product_attention.

Not an apples-to-apples comparison — MLX uses dense (non-paged) KV, while
flashinfer_metal uses paged KV cache. This measures whether the Metal paged
attention kernel is in the right ballpark.

Run with: pytest tests/flashinfer_metal/test_bench_vs_mlx.py -v -s
"""

import time

import pytest
import torch

try:
    import mlx.core as mx

    HAS_MLX = True
except ImportError:
    HAS_MLX = False

requires_mlx = pytest.mark.skipif(not HAS_MLX, reason="MLX not installed")
requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_paged_kv(
    kv_cache: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    page_size: int,
):
    """Fill a paged KV cache from dense K/V tensors.

    kv_cache: [num_pages, 2, page_size, num_kv_heads, head_dim]
    keys/values: [seq_len, num_kv_heads, head_dim]
    """
    seq_len = keys.shape[0]
    for i in range(seq_len):
        kv_cache[i // page_size, 0, i % page_size] = keys[i]
        kv_cache[i // page_size, 1, i % page_size] = values[i]


def _bench_metal_prefill(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    seq_len: int,
    warmup: int,
    repeats: int,
) -> float:
    from flashinfer_metal import BatchPrefillWithPagedKVCacheWrapper

    num_pages = (seq_len + page_size - 1) // page_size
    device = q.device

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace)

    qo_indptr = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    kv_page_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
    kv_page_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
    last_len = seq_len - (num_pages - 1) * page_size
    kv_last_page_len = torch.tensor([last_len], dtype=torch.int32, device=device)

    wrapper.plan(
        qo_indptr, kv_page_indptr, kv_page_indices, kv_last_page_len,
        num_q_heads, num_kv_heads, head_dim, page_size,
    )

    for _ in range(warmup):
        wrapper.run(q, kv_cache)
    torch.mps.synchronize()

    times = []
    for _ in range(repeats):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        wrapper.run(q, kv_cache)
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def _bench_metal_decode(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    kv_len: int,
    batch_size: int,
    warmup: int,
    repeats: int,
) -> float:
    from flashinfer_metal import BatchDecodeWithPagedKVCacheWrapper

    device = q.device
    num_pages_per_seq = (kv_len + page_size - 1) // page_size

    workspace = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace)

    indptr = torch.arange(
        0, (batch_size + 1) * num_pages_per_seq, num_pages_per_seq,
        dtype=torch.int32, device=device,
    )
    indices = torch.arange(
        batch_size * num_pages_per_seq, dtype=torch.int32, device=device,
    )
    last_len = kv_len - (num_pages_per_seq - 1) * page_size
    last_page_lens = torch.full(
        (batch_size,), last_len, dtype=torch.int32, device=device,
    )

    wrapper.plan(
        indptr, indices, last_page_lens,
        num_q_heads, num_kv_heads, head_dim, page_size,
    )

    for _ in range(warmup):
        wrapper.run(q, kv_cache)
    torch.mps.synchronize()

    times = []
    for _ in range(repeats):
        torch.mps.synchronize()
        t0 = time.perf_counter()
        wrapper.run(q, kv_cache)
        torch.mps.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


def _bench_mlx_attention(
    q_mx: "mx.array",
    k_mx: "mx.array",
    v_mx: "mx.array",
    scale: float,
    mask: str | None,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        out = mx.fast.scaled_dot_product_attention(q_mx, k_mx, v_mx, scale=scale, mask=mask)
        mx.eval(out)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(q_mx, k_mx, v_mx, scale=scale, mask=mask)
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# Benchmark tests
# ---------------------------------------------------------------------------


@requires_mps
@requires_mlx
class TestBenchVsMlx:
    """Compare flashinfer_metal paged attention vs MLX dense SDPA."""

    WARMUP = 10
    REPEATS = 100

    @pytest.mark.parametrize(
        "seq_len,num_q_heads,num_kv_heads,head_dim",
        [
            (32, 32, 8, 128),
            (64, 32, 8, 128),
            (128, 32, 8, 128),
            (256, 32, 8, 128),
            (512, 32, 8, 128),
            (1024, 32, 8, 128),
            (2048, 32, 8, 128),
            (4096, 32, 8, 128),
        ],
        ids=["T=32", "T=64", "T=128", "T=256", "T=512", "T=1024", "T=2048", "T=4096"],
    )
    def test_prefill(self, mps_device, seq_len, num_q_heads, num_kv_heads, head_dim):
        """Prefill: all tokens are queries (simulates initial prompt processing)."""
        page_size = 16
        dtype = torch.float16
        num_pages = (seq_len + page_size - 1) // page_size
        scale = head_dim ** -0.5

        # --- Shared data (generate on CPU, distribute) ---
        q_data = torch.randn(seq_len, num_q_heads, head_dim, dtype=dtype)
        k_data = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        v_data = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)

        # --- Metal setup ---
        q_mps = q_data.to(mps_device)
        kv_cache = torch.zeros(
            num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=mps_device,
        )
        _fill_paged_kv(kv_cache, k_data.to(mps_device), v_data.to(mps_device), page_size)

        metal_ms = _bench_metal_prefill(
            q_mps, kv_cache, num_q_heads, num_kv_heads, head_dim, page_size, seq_len,
            self.WARMUP, self.REPEATS,
        )

        # --- MLX setup: [B=1, N_heads, T, D] ---
        q_mx = mx.array(q_data.numpy()).transpose(1, 0, 2)[None]   # [1, N_q, T, D]
        k_mx = mx.array(k_data.numpy()).transpose(1, 0, 2)[None]   # [1, N_kv, T, D]
        v_mx = mx.array(v_data.numpy()).transpose(1, 0, 2)[None]

        mlx_ms = _bench_mlx_attention(
            q_mx, k_mx, v_mx, scale, "causal", self.WARMUP, self.REPEATS,
        )

        ratio = metal_ms / mlx_ms
        print(
            f"\n  prefill T={seq_len:>4d}  {num_q_heads}Q/{num_kv_heads}KV  d={head_dim}"
            f"  |  Metal: {metal_ms:6.3f} ms  MLX: {mlx_ms:6.3f} ms"
            f"  |  ratio: {ratio:.2f}x {'(slower)' if ratio > 1 else '(faster)'}"
        )

    @pytest.mark.parametrize(
        "batch_size,kv_len,num_q_heads,num_kv_heads,head_dim",
        [
            # Vary KV length (B=1)
            (1, 128, 32, 8, 128),
            (1, 256, 32, 8, 128),
            (1, 512, 32, 8, 128),
            (1, 1024, 32, 8, 128),
            (1, 2048, 32, 8, 128),
            (1, 4096, 32, 8, 128),
            (1, 8192, 32, 8, 128),
            # Vary batch size (KV=512)
            (2, 512, 32, 8, 128),
            (4, 512, 32, 8, 128),
            (8, 512, 32, 8, 128),
            (16, 512, 32, 8, 128),
            (32, 512, 32, 8, 128),
            (64, 512, 32, 8, 128),
            # Large batch x long KV
            (8, 2048, 32, 8, 128),
            (16, 2048, 32, 8, 128),
            (32, 2048, 32, 8, 128),
        ],
        ids=[
            "B=1/KV=128", "B=1/KV=256", "B=1/KV=512", "B=1/KV=1024",
            "B=1/KV=2048", "B=1/KV=4096", "B=1/KV=8192",
            "B=2/KV=512", "B=4/KV=512", "B=8/KV=512",
            "B=16/KV=512", "B=32/KV=512", "B=64/KV=512",
            "B=8/KV=2048", "B=16/KV=2048", "B=32/KV=2048",
        ],
    )
    def test_decode(self, mps_device, batch_size, kv_len, num_q_heads, num_kv_heads, head_dim):
        """Decode: 1 query token per batch item, variable KV length."""
        page_size = 16
        dtype = torch.float16
        num_pages_per_seq = (kv_len + page_size - 1) // page_size
        total_pages = num_pages_per_seq * batch_size
        scale = head_dim ** -0.5

        # --- Shared data ---
        q_data = torch.randn(batch_size, num_q_heads, head_dim, dtype=dtype)
        k_data = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype)
        v_data = torch.randn(batch_size, kv_len, num_kv_heads, head_dim, dtype=dtype)

        # --- Metal setup ---
        q_mps = q_data.to(mps_device)  # [batch_size, num_q_heads, head_dim] = 1 token each
        kv_cache = torch.zeros(
            total_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=mps_device,
        )
        for b in range(batch_size):
            for i in range(kv_len):
                page_idx = b * num_pages_per_seq + i // page_size
                slot_idx = i % page_size
                kv_cache[page_idx, 0, slot_idx] = k_data[b, i]
                kv_cache[page_idx, 1, slot_idx] = v_data[b, i]

        metal_ms = _bench_metal_decode(
            q_mps, kv_cache, num_q_heads, num_kv_heads, head_dim, page_size,
            kv_len, batch_size, self.WARMUP, self.REPEATS,
        )

        # --- MLX setup: [B, N_heads, T=1, D] for query, [B, N_kv, KV_len, D] for KV ---
        q_mx = mx.array(q_data.numpy())[:, :, None, :]  # [B, N_q, 1, D]
        k_mx = mx.array(k_data.numpy()).transpose(0, 2, 1, 3)  # [B, N_kv, KV_len, D]
        v_mx = mx.array(v_data.numpy()).transpose(0, 2, 1, 3)

        mlx_ms = _bench_mlx_attention(
            q_mx, k_mx, v_mx, scale, None, self.WARMUP, self.REPEATS,
        )

        ratio = metal_ms / mlx_ms
        print(
            f"\n  decode B={batch_size:>2d} KV={kv_len:>4d}  {num_q_heads}Q/{num_kv_heads}KV  d={head_dim}"
            f"  |  Metal: {metal_ms:6.3f} ms  MLX: {mlx_ms:6.3f} ms"
            f"  |  ratio: {ratio:.2f}x {'(slower)' if ratio > 1 else '(faster)'}"
        )

    @pytest.mark.parametrize(
        "seq_len,num_q_heads,num_kv_heads,head_dim,dtype",
        [
            (256, 32, 8, 128, torch.float16),
            (256, 32, 8, 128, torch.bfloat16),
        ],
        ids=["fp16", "bf16"],
    )
    def test_prefill_dtypes(self, mps_device, seq_len, num_q_heads, num_kv_heads, head_dim, dtype):
        """Compare across dtypes."""
        page_size = 16
        num_pages = (seq_len + page_size - 1) // page_size
        scale = head_dim ** -0.5

        q_data = torch.randn(seq_len, num_q_heads, head_dim, dtype=dtype)
        k_data = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)
        v_data = torch.randn(seq_len, num_kv_heads, head_dim, dtype=dtype)

        # Metal
        q_mps = q_data.to(mps_device)
        kv_cache = torch.zeros(
            num_pages, 2, page_size, num_kv_heads, head_dim,
            dtype=dtype, device=mps_device,
        )
        _fill_paged_kv(kv_cache, k_data.to(mps_device), v_data.to(mps_device), page_size)

        metal_ms = _bench_metal_prefill(
            q_mps, kv_cache, num_q_heads, num_kv_heads, head_dim, page_size, seq_len,
            self.WARMUP, self.REPEATS,
        )

        # MLX (always uses float16 for numpy conversion, then cast)
        np_dtype_map = {torch.float16: "float16", torch.bfloat16: "float16", torch.float32: "float32"}
        np_data = q_data.float().numpy() if dtype == torch.bfloat16 else q_data.numpy()
        mlx_dtype = {torch.float16: mx.float16, torch.bfloat16: mx.bfloat16, torch.float32: mx.float32}[dtype]

        q_mx = mx.array(q_data.float().numpy()).transpose(1, 0, 2)[None].astype(mlx_dtype)
        k_mx = mx.array(k_data.float().numpy()).transpose(1, 0, 2)[None].astype(mlx_dtype)
        v_mx = mx.array(v_data.float().numpy()).transpose(1, 0, 2)[None].astype(mlx_dtype)

        mlx_ms = _bench_mlx_attention(
            q_mx, k_mx, v_mx, scale, "causal", self.WARMUP, self.REPEATS,
        )

        ratio = metal_ms / mlx_ms
        print(
            f"\n  prefill T={seq_len} {dtype}  "
            f"|  Metal: {metal_ms:6.3f} ms  MLX: {mlx_ms:6.3f} ms"
            f"  |  ratio: {ratio:.2f}x {'(slower)' if ratio > 1 else '(faster)'}"
        )
