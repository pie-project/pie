from __future__ import annotations
import io
import os
import torch
import torch.nn as nn
import math

# rand_mv (the noise-injection / random-matmul kernel under
# `pie_kernels.cuda.rand_mv_new`) is not connected for this release. The
# code is preserved for the CMA-ES adapter path but is gated off at the
# module boundary so that:
#   * adapter.py — and therefore everything that does
#     `from ..adapter import AdapterSubpass` from `pie_driver_dev/model/*.py`
#     — has zero import-time coupling to the rand_mv module;
#   * the JIT-compiled CUDA kernel never has to load, so the server boot
#     does not need ninja, nvcc, or any CUDA toolchain on PATH.
#
# To re-enable the adapter math: import RAND_MV_AVAILABLE / rand_mv inside
# the methods that use them and drop the False overrides below. See
# `_execute_fast`, `_execute_legacy`, and the `init_adapter` paths.
RAND_MV_AVAILABLE = False
rand_mv = None
_HAS_FUSED_RAND_MV = False
_HAS_MULTI_INPUT_SECTIONED = False
_BM10 = None
_EXT_SECTIONED = None
_EXT_MULTI_INPUT_SECTIONED = None


def _resolve_fast_ext():
    """No-op while rand_mv is disconnected. Returns `(None, None)`."""
    return None, None


def run_length_encode(data: list[int]) -> list[tuple[int, int]]:
    """
    Perform run-length encoding on a list of integers.

    Args:
        data (List[int]): The input list of integers.

    Returns:
        List[Tuple[int, int]]: A list of (value, count) pairs.
    """
    if not data:
        return []

    encoded = []
    current_value = data[0]
    count = 1

    for num in data[1:]:
        if num == current_value:
            count += 1
        else:
            encoded.append((current_value, count))
            current_value = num
            count = 1

    # Append the last run
    encoded.append((current_value, count))

    return encoded


class _FastPlan:
    """Pre-computed per-(rle-group) state for the CUDA + CmaesAdapter path.

    Built once at AdapterSubpass.__init__; consumed in the per-layer hot path
    so that execute() does no attribute-chain lookups, no isinstance checks,
    no tuple constructions, and no geometry arithmetic.
    """
    __slots__ = (
        "x_start", "x_end", "B",
        "rand_seeds", "rank_lora", "scaling",
        "expected_stride0", "sum_out",
        "qkv_down_scratch",  # pre-sliced scratch view (None if B too large)
        "rank_x3",            # 3 * rank_lora
        # Per-layer arrays:
        "Wd_layers", "Wu_layers", "Sd_layers", "Su_layers",
        # Pre-formatted list args for direct ext.* calls (no tuple→list
        # conversion or float() on the hot path):
        "section_widths_d_list",       # [rank, rank, rank]
        "section_offsets_d_lists",     # [[li, li+100, li+200], ...] per layer
        "x_starts_u_list",             # [0, rank, 2*rank]
        "section_widths_u_list",       # [local_d_q, local_d_k, local_d_v]
        "section_offsets_u_lists",     # [[-li, ...], ...] per layer
        "global_cols_d",               # = 3 * rank_lora (Sd.shape[1])
        # Fallback (3-UP) state for world_size != 1 OR non-contiguous q_state:
        "Wu_q_layers", "Wu_k_layers", "Wu_v_layers",
        "Su_q_layers", "Su_k_layers", "Su_v_layers",
        "local_d_q", "local_d_k", "local_d_v",
    )


class AdapterSubpass:

    def __init__(
        self,
        adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]],
        adapter_indices: list[int],
        adapter_extras: dict[int, Adapter],
        rand_seeds: torch.Tensor,
        qo_indptr: list[int],
    ):
        self.adapter_at_layer = adapter_at_layer
        self.adapter_indices = adapter_indices
        self.adapter_extras = adapter_extras
        self.rand_seeds = rand_seeds
        self.adapter_indices_rle = run_length_encode(self.adapter_indices)
        self.qo_indptr = qo_indptr
        # Try to pre-build fast-path plans. None if any group disqualifies
        # (non-CUDA, non-CmaesAdapter); execute() falls back to legacy.
        self._fast_plans = self._build_fast_plans() if _HAS_FUSED_RAND_MV else None

    def _build_fast_plans(self) -> list[_FastPlan] | None:
        plans: list[_FastPlan] = []
        i = 0
        for adapter_index, count in self.adapter_indices_rle:
            x_start = self.qo_indptr[i]
            x_end = self.qo_indptr[i + count]
            i += count
            B = x_end - x_start
            if B == 0:
                continue  # nothing to do for this group; drop it from the plan list
            adapter_info = self.adapter_extras[adapter_index]
            if not isinstance(adapter_info, CmaesAdapter):
                return None  # legacy adapter present; can't build a fast plan

            num_layers = adapter_info.num_layers
            rank_lora = adapter_info.rank
            out_indptr = adapter_info.out_features_indptr
            local_d_q = out_indptr[1] - out_indptr[0]
            local_d_k = out_indptr[2] - out_indptr[1]
            local_d_v = out_indptr[3] - out_indptr[2]
            sum_out = local_d_q + local_d_k + local_d_v

            Wd_layers = [self.adapter_at_layer[li][0][adapter_index] for li in range(num_layers)]
            Wu_layers = [self.adapter_at_layer[li][1][adapter_index] for li in range(num_layers)]

            p = _FastPlan()
            p.x_start = x_start
            p.x_end = x_end
            p.B = B
            # Ensure the seeds slice is GPU int64; the raw ext call needs no
            # further coercion. The slicing returns a view of the same dtype,
            # so this is normally cheap or a no-op.
            rseeds = self.rand_seeds[x_start:x_end]
            if rseeds.dtype != torch.int64 or not rseeds.is_cuda:
                rseeds = rseeds.to(device="cuda", dtype=torch.int64)
            p.rand_seeds = rseeds
            p.rank_lora = rank_lora
            p.rank_x3 = 3 * rank_lora
            p.scaling = adapter_info.alpha / float(rank_lora)
            p.expected_stride0 = sum_out
            p.sum_out = sum_out
            p.local_d_q = local_d_q
            p.local_d_k = local_d_k
            p.local_d_v = local_d_v
            p.global_cols_d = 3 * rank_lora
            p.qkv_down_scratch = (
                adapter_info._cast_qkv_down_scratch[:B, : 3 * rank_lora]
                if B <= adapter_info._scratch_max_B else None
            )
            p.Wd_layers = Wd_layers
            p.Wu_layers = Wu_layers
            p.Sd_layers = adapter_info.qkv_down_sigma  # already a list, kept by reference
            p.Su_layers = adapter_info.qkv_up_sigma
            # Pre-format constant args as plain Python lists (pybind expects
            # std::vector<int64_t>). Done once at __init__ so the hot path
            # doesn't construct lists per kernel call.
            p.section_widths_d_list = [rank_lora, rank_lora, rank_lora]
            p.section_offsets_d_lists = [
                [li, li + 100, li + 200] for li in range(num_layers)
            ]
            p.x_starts_u_list = [0, rank_lora, 2 * rank_lora]
            p.section_widths_u_list = [local_d_q, local_d_k, local_d_v]
            p.section_offsets_u_lists = [
                [-li, -(li + 100), -(li + 200)] for li in range(num_layers)
            ]
            # 3-UP fallback views (only consumed when world_size != 1 or
            # qkv_proj is non-contiguous):
            p.Wu_q_layers = [Wu[:, out_indptr[0]:out_indptr[1]] for Wu in Wu_layers]
            p.Wu_k_layers = [Wu[:, out_indptr[1]:out_indptr[2]] for Wu in Wu_layers]
            p.Wu_v_layers = [Wu[:, out_indptr[2]:out_indptr[3]] for Wu in Wu_layers]
            p.Su_q_layers = [Su[:, out_indptr[0]:out_indptr[1]] for Su in p.Su_layers]
            p.Su_k_layers = [Su[:, out_indptr[1]:out_indptr[2]] for Su in p.Su_layers]
            p.Su_v_layers = [Su[:, out_indptr[2]:out_indptr[3]] for Su in p.Su_layers]
            plans.append(p)
        return plans

    def execute(
        self,
        layer_idx: int,
        xs: torch.Tensor,
        q_state: torch.Tensor,
        k_state: torch.Tensor,
        v_state: torch.Tensor,
        rank: int = 0,
        world_size: int = 1,
    ):
        if self._fast_plans is not None:
            self._execute_fast(layer_idx, xs, q_state, k_state, v_state, rank, world_size)
            return
        self._execute_legacy(layer_idx, xs, q_state, k_state, v_state, rank, world_size)

    def _execute_fast(
        self, layer_idx, xs, q_state, k_state, v_state, rank, world_size,
    ):
        # Direct ext.* calls — bypass the Python wrappers (no asserts, no
        # seeds.to(), no list(tuple), no float() — all preformatted in plan).
        # First call resolves (and JIT-compiles) the BM10 sectioned kernels.
        ext_s, ext_mi = _resolve_fast_ext()
        # Wrapper kept for the rare 3-UP fallback path.
        _matmul = rand_mv.batched_randn_matmul
        for p in self._fast_plans:
            x = xs[p.x_start:p.x_end]
            qkv_down = p.qkv_down_scratch
            if qkv_down is None:
                qkv_down = torch.empty(
                    p.B, p.rank_x3, device=x.device, dtype=x.dtype
                )
            # ext signature (positional):
            #   x, seeds, S, section_widths, section_offsets,
            #   col_offset, global_cols, out, alpha, beta, W_mean
            ext_s(
                x, p.rand_seeds, p.Sd_layers[layer_idx],
                p.section_widths_d_list, p.section_offsets_d_lists[layer_idx],
                0, p.global_cols_d,
                qkv_down, 1.0, 0.0,
                p.Wd_layers[layer_idx],
            )
            if (
                world_size == 1 and ext_mi is not None
                and q_state.stride(0) == p.expected_stride0
                and q_state.stride(1) == 1
            ):
                qkv_full = q_state[p.x_start:p.x_end].as_strided(
                    (p.B, p.sum_out), q_state.stride(),
                )
                # ext signature (positional):
                #   x, seeds, S, x_starts, section_widths, section_offsets,
                #   out, alpha, beta, W_mean
                ext_mi(
                    qkv_down, p.rand_seeds, p.Su_layers[layer_idx],
                    p.x_starts_u_list, p.section_widths_u_list,
                    p.section_offsets_u_lists[layer_idx],
                    qkv_full, p.scaling, 1.0,
                    p.Wu_layers[layer_idx],
                )
                continue
            # 3-UP fallback (TP world_size != 1 or non-contiguous qkv_proj).
            d_q, d_k, d_v = torch.split(
                qkv_down, [p.rank_lora, p.rank_lora, p.rank_lora], dim=-1,
            )
            global_d_q = p.local_d_q * world_size
            global_d_k = p.local_d_k * world_size
            global_d_v = p.local_d_v * world_size
            _matmul(
                d_q, p.rand_seeds, p.Su_q_layers[layer_idx],
                W_mean=p.Wu_q_layers[layer_idx],
                out=q_state[p.x_start:p.x_end],
                alpha=p.scaling, beta=1.0,
                seed_offset=-layer_idx,
                col_offset=rank * p.local_d_q, global_cols=global_d_q,
            )
            _matmul(
                d_k, p.rand_seeds, p.Su_k_layers[layer_idx],
                W_mean=p.Wu_k_layers[layer_idx],
                out=k_state[p.x_start:p.x_end],
                alpha=p.scaling, beta=1.0,
                seed_offset=-(layer_idx + 100),
                col_offset=rank * p.local_d_k, global_cols=global_d_k,
            )
            _matmul(
                d_v, p.rand_seeds, p.Su_v_layers[layer_idx],
                W_mean=p.Wu_v_layers[layer_idx],
                out=v_state[p.x_start:p.x_end],
                alpha=p.scaling, beta=1.0,
                seed_offset=-(layer_idx + 200),
                col_offset=rank * p.local_d_v, global_cols=global_d_v,
            )

    def _execute_legacy(
        self, layer_idx, xs, q_state, k_state, v_state, rank, world_size,
    ):
        # Reached when _fast_plans is None — i.e., either no CUDA backend
        # (Metal) or at least one group is a non-CmaesAdapter. Always uses
        # the eager bmm path with optional CUDA noise injection.
        i = 0
        for adapter_index, count in self.adapter_indices_rle:

            x_start = self.qo_indptr[i]
            x_end = self.qo_indptr[i + count]
            x = xs[x_start:x_end]

            # In multi-GPU, xs is replicated (full hidden state), so inputs are consistent.
            # q_state, k_state, v_state are SHARDED (local output).

            rand_seeds = self.rand_seeds[x_start:x_end]
            assert x.shape[0] == rand_seeds.shape[0], "Batch size must match seeds."

            Wd = self.adapter_at_layer[layer_idx][0][adapter_index]
            Wu = self.adapter_at_layer[layer_idx][1][
                adapter_index
            ]  # (rank, LOCAL_d_q+LOCAL_d_k+LOCAL_d_k)
            adapter_info = self.adapter_extras[adapter_index]

            rank_lora = adapter_info.rank  # LoRA rank, not GPU rank
            out_indptr = adapter_info.out_features_indptr
            scaling = adapter_info.alpha / float(rank_lora)
            is_cmaes = isinstance(adapter_info, CmaesAdapter)

            # UP-projection geometry (LOCAL out_features per rank)
            Wu_q_local = Wu[:, out_indptr[0] : out_indptr[1]]
            Wu_k_local = Wu[:, out_indptr[1] : out_indptr[2]]
            Wu_v_local = Wu[:, out_indptr[2] : out_indptr[3]]
            local_d_q = out_indptr[1] - out_indptr[0]
            local_d_k = out_indptr[2] - out_indptr[1]
            local_d_v = out_indptr[3] - out_indptr[2]
            global_d_q = local_d_q * world_size
            global_d_k = local_d_k * world_size
            global_d_v = local_d_v * world_size

            qkv_down = x @ Wd
            d_q, d_k, d_v = torch.split(
                qkv_down, [rank_lora, rank_lora, rank_lora], dim=-1
            )

            inject_noise = RAND_MV_AVAILABLE
            if inject_noise:
                if not is_cmaes:
                    continue

                Sd = adapter_info.qkv_down_sigma[layer_idx]
                Sd_q, Sd_k, Sd_v = torch.split(
                    Sd, [rank_lora, rank_lora, rank_lora], dim=-1
                )
                d_q = d_q + rand_mv.batched_randn_matmul(
                    x, seeds=rand_seeds + layer_idx, S=Sd_q, out_dtype=x.dtype
                )
                d_k = d_k + rand_mv.batched_randn_matmul(
                    x,
                    seeds=rand_seeds + (layer_idx + 100),
                    S=Sd_k,
                    out_dtype=x.dtype,
                )
                d_v = d_v + rand_mv.batched_randn_matmul(
                    x,
                    seeds=rand_seeds + (layer_idx + 200),
                    S=Sd_v,
                    out_dtype=x.dtype,
                )

            u_q_local = d_q @ Wu_q_local
            u_k_local = d_k @ Wu_k_local
            u_v_local = d_v @ Wu_v_local

            if inject_noise:
                Su = adapter_info.qkv_up_sigma[layer_idx]
                Su_q_local = Su[:, out_indptr[0] : out_indptr[1]]
                Su_k_local = Su[:, out_indptr[1] : out_indptr[2]]
                Su_v_local = Su[:, out_indptr[2] : out_indptr[3]]

                u_q_local = u_q_local + rand_mv.batched_randn_matmul(
                    d_q,
                    seeds=rand_seeds - layer_idx,
                    S=Su_q_local,
                    out_dtype=x.dtype,
                    col_offset=rank * local_d_q,
                    global_cols=global_d_q,
                )
                u_k_local = u_k_local + rand_mv.batched_randn_matmul(
                    d_k,
                    seeds=rand_seeds - (layer_idx + 100),
                    S=Su_k_local,
                    out_dtype=x.dtype,
                    col_offset=rank * local_d_k,
                    global_cols=global_d_k,
                )
                u_v_local = u_v_local + rand_mv.batched_randn_matmul(
                    d_v,
                    seeds=rand_seeds - (layer_idx + 200),
                    S=Su_v_local,
                    out_dtype=x.dtype,
                    col_offset=rank * local_d_v,
                    global_cols=global_d_v,
                )

            q_state[x_start:x_end].add_(scaling * u_q_local)
            k_state[x_start:x_end].add_(scaling * u_k_local)
            v_state[x_start:x_end].add_(scaling * u_v_local)

            i += count


class Adapter:
    """
    Abstract base class defining the interface for a LoRA-style adapter.
    """

    adapter_id: int
    rank: int
    alpha: float
    out_features: list[int]
    out_features_indptr: list[int]

    def __init__(
        self, adapter_id: int, rank: int, alpha: float, out_features: list[int]
    ):
        self.adapter_id = adapter_id
        self.rank = rank
        self.alpha = alpha
        self.out_features = out_features

        self.out_features_indptr = [0]
        for i in range(len(out_features)):
            self.out_features_indptr.append(
                self.out_features_indptr[-1] + out_features[i]
            )


class CmaesAdapter(Adapter):
    adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]]

    num_layers: int

    population_size: int
    mu_fraction: float
    initial_sigma: float
    min_sigma: float
    min_var: float
    max_var: float

    qkv_down_sigma: list[torch.Tensor]
    qkv_up_sigma: list[torch.Tensor]

    @torch.inference_mode()
    def __init__(
        self,
        adapter_id: int,
        adapter_at_layer: list[tuple[torch.Tensor, torch.Tensor]],
        rank: int,
        alpha: float,
        in_features: int,
        out_features: list[int],
        num_layers: int,
        population_size: int,
        mu_fraction: float,
        initial_sigma: float,
        min_sigma: float,
        min_var: float,
        max_var: float,
        device: torch.device,
        dtype: torch.dtype,
        gpu_rank: int = 0,
        world_size: int = 1,
        adapter_path: str = "~/.pie/adapters/",
    ):
        super().__init__(adapter_id, rank, alpha, out_features)

        self.adapter_at_layer = adapter_at_layer

        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.in_features = in_features
        self.sum_out = int(sum(out_features))

        self.gpu_rank = gpu_rank
        self.world_size = world_size
        self.adapter_path = os.path.expanduser(adapter_path)

        # CMA-ES default knobs (can be overridden from outside if desired)
        self.population_size = population_size
        self.mu_fraction = mu_fraction

        self.initial_sigma = initial_sigma
        self.min_sigma = min_sigma
        self.min_var = min_var
        self.max_var = max_var

        # Parameter tensors (means) + effective per-weight stdevs used by kernels
        self.qkv_down_weight = []
        self.qkv_up_weight = []
        self.qkv_down_sigma = []
        self.qkv_up_sigma = []

        for i in range(num_layers):
            # DOWN: (in_features, rank * 3)
            down_cols = rank * len(out_features)
            qkv_down_weight = torch.empty(
                (in_features, down_cols), dtype=dtype, device=device
            )
            # UP: (rank, d_q + d_k + d_v)
            qkv_up_weight = torch.empty(
                (rank, self.sum_out), dtype=dtype, device=device
            )

            # Effective elementwise stddev S used by kernels; start at initial_sigma
            qkv_down_sigma = torch.full_like(
                qkv_down_weight, self.initial_sigma, dtype=dtype, device=device
            )
            qkv_up_sigma = torch.full_like(
                qkv_up_weight, self.initial_sigma, dtype=dtype, device=device
            )

            nn.init.kaiming_uniform_(qkv_down_weight, a=math.sqrt(5))
            nn.init.zeros_(qkv_up_weight)

            self.adapter_at_layer[i][0][self.adapter_id].copy_(qkv_down_weight)
            self.adapter_at_layer[i][1][self.adapter_id].copy_(qkv_up_weight)

            self.qkv_down_sigma.append(qkv_down_sigma)
            self.qkv_up_sigma.append(qkv_up_sigma)

        # ===== Diagonal CMA-ES state (float32 for stability) =====
        f32 = torch.float32

        # Calculate d_per_layer using GLOBAL parameter count
        global_sum_out = self.sum_out * self.world_size
        self.d_per_layer = float(
            in_features * rank * len(out_features) + rank * global_sum_out
        )

        # Recombination weights
        self.mu = max(1, int(self.population_size * self.mu_fraction))
        ranks = torch.arange(1, self.mu + 1, dtype=f32, device=device)
        logw = torch.log(
            torch.tensor(self.mu + 0.5, dtype=f32, device=device)
        ) - torch.log(ranks)
        self.weights = (logw / logw.sum()).to(f32)  # (mu,)
        self.weights_reshaped = self.weights.view(-1, 1, 1)
        self.mueff = (self.weights.sum() ** 2) / (self.weights.pow(2).sum())

        # Strategy parameters
        d = self.d_per_layer
        self.cc = (4.0 + self.mueff / d) / (d + 4.0 + 2.0 * self.mueff / d)
        self.cs = (self.mueff + 2.0) / (d + self.mueff + 5.0)
        self.c1 = 2.0 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((d + 2.0) ** 2 + self.mueff),
        )
        damps_term = torch.sqrt((self.mueff - 1.0) / (d + 1.0)) - 1.0
        self.damps = 1.0 + 2.0 * torch.clamp(damps_term, min=0.0) + self.cs
        self.chiN = (d**0.5) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))

        # Per-layer scalar step-size (sigma) and paths / diag covariance (var=1 initially)
        self.layer_sigma = [
            torch.tensor(self.initial_sigma, dtype=f32, device=device)
            for _ in range(num_layers)
        ]

        self.ps_down = [
            torch.zeros(
                (in_features, rank * len(out_features)), dtype=f32, device=device
            )
            for _ in range(num_layers)
        ]
        self.ps_up = [
            torch.zeros((rank, self.sum_out), dtype=f32, device=device)
            for _ in range(num_layers)
        ]
        self.pc_down = [
            torch.zeros(
                (in_features, rank * len(out_features)), dtype=f32, device=device
            )
            for _ in range(num_layers)
        ]
        self.pc_up = [
            torch.zeros((rank, self.sum_out), dtype=f32, device=device)
            for _ in range(num_layers)
        ]
        self.diag_C_down = [
            torch.ones(
                (in_features, rank * len(out_features)), dtype=f32, device=device
            )
            for _ in range(num_layers)
        ]
        self.diag_C_up = [
            torch.ones((rank, self.sum_out), dtype=f32, device=device)
            for _ in range(num_layers)
        ]

        # Pre-allocated model-dtype scratch for the DOWN-projection result
        # the eager fast path slices into. Sized for the typical decode
        # max-batch; AdapterSubpass falls back to a one-shot torch.empty
        # when a request exceeds this size. UP-projection writes directly
        # into q/k/v_state via alpha/beta, so no fp32 scratch is needed.
        self._scratch_max_B = 512
        self._cast_qkv_down_scratch = torch.empty(
            self._scratch_max_B, 3 * rank, device=device, dtype=dtype
        )

    def upload(self, name: str, data: bytes) -> None:
        """Load the adapter's state from ``adapter_{name}.pt``."""
        filename = os.path.join(self.adapter_path, f"adapter_{name}.pt")
        state_dict = torch.load(filename, map_location=self.device)

        # --- Verification ---
        # Ensure the loaded checkpoint is compatible with this adapter instance.
        assert (
            self.adapter_id == state_dict["adapter_id"]
        ), "Adapter ID mismatch during upload."
        assert self.rank == state_dict["rank"], "Rank mismatch during upload."
        assert (
            self.num_layers == state_dict["num_layers"]
        ), "Layer count mismatch during upload."

        # --- Load Hyperparameters & CMA-ES State ---
        self.alpha = state_dict["alpha"]
        self.out_features = state_dict["out_features"]
        self.population_size = state_dict["population_size"]
        self.mu_fraction = state_dict["mu_fraction"]
        self.initial_sigma = state_dict["initial_sigma"]
        self.min_sigma = state_dict["min_sigma"]
        self.min_var = state_dict["min_var"]
        self.max_var = state_dict["max_var"]
        self.in_features = state_dict["in_features"]

        # Restore CMA-ES optimizer state tensors
        self.qkv_down_sigma = state_dict["qkv_down_sigma"]
        self.qkv_up_sigma = state_dict["qkv_up_sigma"]
        self.layer_sigma = state_dict["layer_sigma"]
        self.ps_down = state_dict["ps_down"]
        self.ps_up = state_dict["ps_up"]
        self.pc_down = state_dict["pc_down"]
        self.pc_up = state_dict["pc_up"]
        self.diag_C_down = state_dict["diag_C_down"]
        self.diag_C_up = state_dict["diag_C_up"]

        # --- Recalculate Derived CMA-ES Strategy Parameters ---
        # This ensures the optimizer's internal constants are correct after loading
        # potentially different hyperparameters (e.g., population_size).
        f32 = torch.float32

        global_sum_out = self.sum_out * self.world_size
        self.d_per_layer = float(
            self.in_features * self.rank * len(self.out_features)
            + self.rank * global_sum_out
        )
        self.mu = max(1, int(self.population_size * self.mu_fraction))
        ranks = torch.arange(1, self.mu + 1, dtype=f32, device=self.device)
        logw = torch.log(
            torch.tensor(self.mu + 0.5, dtype=f32, device=self.device)
        ) - torch.log(ranks)
        self.weights = (logw / logw.sum()).to(f32)
        self.weights_reshaped = self.weights.view(-1, 1, 1)
        self.mueff = (self.weights.sum() ** 2) / (self.weights.pow(2).sum())

        d = self.d_per_layer
        self.cc = (4.0 + self.mueff / d) / (d + 4.0 + 2.0 * self.mueff / d)
        self.cs = (self.mueff + 2.0) / (d + self.mueff + 5.0)
        self.c1 = 2.0 / ((d + 1.3) ** 2 + self.mueff)
        self.cmu = min(
            1.0 - self.c1,
            2.0 * (self.mueff - 2.0 + 1.0 / self.mueff) / ((d + 2.0) ** 2 + self.mueff),
        )
        damps_term = torch.sqrt((self.mueff - 1.0) / (d + 1.0)) - 1.0
        self.damps = 1.0 + 2.0 * torch.clamp(damps_term, min=0.0) + self.cs
        self.chiN = (d**0.5) * (1.0 - 1.0 / (4.0 * d) + 1.0 / (21.0 * d**2))

        # --- Load Parameters (Weights) ---
        # Copy the loaded weights into the correct slices of the shared tensor.
        loaded_down_weights = state_dict["qkv_down_weight"]
        loaded_up_weights = state_dict["qkv_up_weight"]
        for i in range(self.num_layers):
            self.adapter_at_layer[i][0][self.adapter_id].copy_(loaded_down_weights[i])
            self.adapter_at_layer[i][1][self.adapter_id].copy_(loaded_up_weights[i])

    def download(self, name: str) -> bytes:
        """
        Snapshots the adapter's current parameters and state into a file.

        This function saves a dictionary containing the adapter's weights (extracted
        from `adapter_at_layer`), hyperparameters, and the CMA-ES optimizer state
        to f"adapter_{self.adapter_id}.pt". It returns an empty bytes object as
        per the instructions.
        """

        filename = os.path.join(self.adapter_path, f"adapter_{name}.pt")
        os.makedirs(self.adapter_path, exist_ok=True)

        # Extract the weight tensors for this specific adapter.
        # We use .clone().cpu() for safe, device-independent saving.
        qkv_down_weight = [
            self.adapter_at_layer[i][0][self.adapter_id].clone().cpu()
            for i in range(self.num_layers)
        ]
        qkv_up_weight = [
            self.adapter_at_layer[i][1][self.adapter_id].clone().cpu()
            for i in range(self.num_layers)
        ]

        # Assemble the state dictionary with all necessary data.
        state_dict = {
            # Identification & Configuration
            "adapter_id": self.adapter_id,
            "rank": self.rank,
            "alpha": self.alpha,
            "out_features": self.out_features,
            "in_features": self.in_features,
            "num_layers": self.num_layers,
            # CMA-ES Hyperparameters
            "population_size": self.population_size,
            "mu_fraction": self.mu_fraction,
            "initial_sigma": self.initial_sigma,
            "min_sigma": self.min_sigma,
            "min_var": self.min_var,
            "max_var": self.max_var,
            # Parameters (Weights)
            "qkv_down_weight": qkv_down_weight,
            "qkv_up_weight": qkv_up_weight,
            # CMA-ES State Tensors
            "qkv_down_sigma": [t.clone().cpu() for t in self.qkv_down_sigma],
            "qkv_up_sigma": [t.clone().cpu() for t in self.qkv_up_sigma],
            "layer_sigma": [t.clone().cpu() for t in self.layer_sigma],
            "ps_down": [t.clone().cpu() for t in self.ps_down],
            "ps_up": [t.clone().cpu() for t in self.ps_up],
            "pc_down": [t.clone().cpu() for t in self.pc_down],
            "pc_up": [t.clone().cpu() for t in self.pc_up],
            "diag_C_down": [t.clone().cpu() for t in self.diag_C_down],
            "diag_C_up": [t.clone().cpu() for t in self.diag_C_up],
        }

        torch.save(state_dict, filename)

        # Also save to memory to return
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        return buffer.getvalue()

    @torch.inference_mode()
    def update(self, scores: list[float], seeds: list[int], max_sigma: float) -> None:

        if not RAND_MV_AVAILABLE:
            raise RuntimeError(
                "CmaesAdapter.update() requires CUDA or MPS for noise generation. "
                "This function is not available on this platform."
            )

        if len(scores) != self.population_size or len(seeds) != self.population_size:
            raise ValueError(f"Expected {self.population_size} scores and seeds.")

        device = self.device
        f32 = torch.float32
        eps = 1e-12

        # Select top-µ by score (maximize)
        scores_t = torch.tensor(scores, device=device, dtype=f32)
        seeds_t = torch.tensor(seeds, device=device, dtype=torch.long)
        _, best_idx = torch.sort(scores_t, descending=True)
        top_idx = best_idx[: self.mu]
        top_seeds = seeds_t[top_idx]  # (µ,)

        W = self.weights_reshaped  # (µ,1,1) f32 on device

        # Convenience
        d_q, d_k, d_v = self.out_features
        rank = self.rank

        for layer_idx in range(self.num_layers):
            # ===== Recreate parents' noise matrices (bit-identical to runtime sampling) =====
            # DOWN: split columns into [rank, rank, rank]
            S_down = self.qkv_down_sigma[layer_idx].to(f32)  # (I, 3*rank)
            S_down_q, S_down_k, S_down_v = torch.split(
                S_down, [rank, rank, rank], dim=-1
            )

            # Seeds for Q/K/V (match AdapterBuffer)
            Wd_q = rand_mv.batched_randn_generate(
                seeds=top_seeds + layer_idx,
                S=S_down_q,
                device=device,
                dtype=torch.float32,
            )
            Wd_k = rand_mv.batched_randn_generate(
                seeds=top_seeds + (layer_idx + 100),
                S=S_down_k,
                device=device,
                dtype=torch.float32,
            )
            Wd_v = rand_mv.batched_randn_generate(
                seeds=top_seeds + (layer_idx + 200),
                S=S_down_v,
                device=device,
                dtype=torch.float32,
            )
            # (µ, I, rank) each -> concat on O
            Wd = torch.cat([Wd_q, Wd_k, Wd_v], dim=-1)  # (µ, I, 3*rank)

            # UP: split columns into [d_q, d_k, d_v]
            S_up = self.qkv_up_sigma[layer_idx].to(f32)  # (rank, d_q+d_k+d_v)
            S_up_q, S_up_k, S_up_v = torch.split(S_up, [d_q, d_k, d_v], dim=-1)

            # Sharding logic for UP generation
            # Global dimensions
            global_d_q = d_q * self.world_size
            global_d_k = d_k * self.world_size
            global_d_v = d_v * self.world_size

            # Local offsets
            offset_q = self.gpu_rank * d_q
            offset_k = self.gpu_rank * d_k
            offset_v = self.gpu_rank * d_v

            Wu_q = rand_mv.batched_randn_generate(
                seeds=top_seeds - layer_idx,
                S=S_up_q,
                device=device,
                dtype=torch.float32,
                col_offset=offset_q,
                global_cols=global_d_q,
            )
            Wu_k = rand_mv.batched_randn_generate(
                seeds=top_seeds - (layer_idx + 100),
                S=S_up_k,
                device=device,
                dtype=torch.float32,
                col_offset=offset_k,
                global_cols=global_d_k,
            )
            Wu_v = rand_mv.batched_randn_generate(
                seeds=top_seeds - (layer_idx + 200),
                S=S_up_v,
                device=device,
                dtype=torch.float32,
                col_offset=offset_v,
                global_cols=global_d_v,
            )
            Wu = torch.cat([Wu_q, Wu_k, Wu_v], dim=-1)  # (µ, rank, d_sum)

            # ===== Recombine means (parents = mean + noise) =====
            mean_down_old = self.adapter_at_layer[layer_idx][0][self.adapter_id].to(
                f32
            )  # (I, 3*rank)
            mean_up_old = self.adapter_at_layer[layer_idx][1][self.adapter_id].to(
                f32
            )  # (rank, d_sum)

            parents_down = mean_down_old.unsqueeze(0) + Wd  # (µ, I, 3*rank)
            parents_up = mean_up_old.unsqueeze(0) + Wu  # (µ, rank, d_sum)

            mean_down_new = torch.sum(parents_down * W, dim=0)  # (I, 3*rank)
            mean_up_new = torch.sum(parents_up * W, dim=0)  # (rank, d_sum)

            # Write back (cast to model dtype)
            self.adapter_at_layer[layer_idx][0][self.adapter_id].copy_(
                mean_down_new.to(self.dtype)
            )
            self.adapter_at_layer[layer_idx][1][self.adapter_id].copy_(
                mean_up_new.to(self.dtype)
            )

            # ===== Step-size path (ps) and covariance path (pc) updates =====
            sigma_l = self.layer_sigma[layer_idx].to(f32)  # scalar
            std_down = torch.sqrt(self.diag_C_down[layer_idx]).clamp_min(
                self.min_var**0.5
            )  # (I, 3*rank)
            std_up = torch.sqrt(self.diag_C_up[layer_idx]).clamp_min(
                self.min_var**0.5
            )  # (rank, d_sum)

            # Normalized steps
            step_down = (mean_down_new - mean_down_old) / (sigma_l + eps)  # (I, 3*rank)
            step_up = (mean_up_new - mean_up_old) / (sigma_l + eps)  # (rank, d_sum)

            Cinv_sqrt_step_down = step_down / (std_down + eps)
            Cinv_sqrt_step_up = step_up / (std_up + eps)

            c_ps = torch.sqrt(self.cs * (2.0 - self.cs) * self.mueff).to(f32)
            ps_down_new = (1.0 - self.cs) * self.ps_down[
                layer_idx
            ] + c_ps * Cinv_sqrt_step_down
            ps_up_new = (1.0 - self.cs) * self.ps_up[
                layer_idx
            ] + c_ps * Cinv_sqrt_step_up

            # Combined norm for step-size adaptation (Must be global!)
            norm_sq_down = ps_down_new.pow(2).sum()
            norm_sq_up_local = ps_up_new.pow(2).sum()

            # Aggregate UP norm across ranks (DOWN is replicated, so it's consistent)
            if self.world_size > 1:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.all_reduce(norm_sq_up_local, op=dist.ReduceOp.SUM)

            norm_ps = torch.sqrt(norm_sq_down + norm_sq_up_local)

            sigma_new = sigma_l * torch.exp(
                (self.cs / self.damps) * (norm_ps / self.chiN - 1.0)
            )
            if max_sigma and max_sigma > 0:
                sigma_new = torch.clamp(
                    sigma_new, min=self.min_sigma, max=float(max_sigma)
                )
            else:
                sigma_new = torch.clamp(sigma_new, min=self.min_sigma)

            # Heaviside for covariance path (no iteration; steady-state denom ~ 1)
            h_sigma = (norm_ps < (1.4 + 2.0 / (self.d_per_layer + 1.0)) * self.chiN).to(
                f32
            )

            c_pc = torch.sqrt(self.cc * (2.0 - self.cc) * self.mueff).to(f32)
            pc_down_new = (1.0 - self.cc) * self.pc_down[
                layer_idx
            ] + h_sigma * c_pc * step_down
            pc_up_new = (1.0 - self.cc) * self.pc_up[
                layer_idx
            ] + h_sigma * c_pc * step_up

            # ===== Diagonal covariance update (rank-1 + rank-µ) =====
            # y_parent = (parent - mean_old) / sigma_l  (shape: like params)
            y_down_parents = (parents_down - mean_down_old.unsqueeze(0)) / (
                sigma_l + eps
            )
            y_up_parents = (parents_up - mean_up_old.unsqueeze(0)) / (sigma_l + eps)

            rank1_down = self.c1 * pc_down_new.pow(2)
            rank1_up = self.c1 * pc_up_new.pow(2)
            rankmu_down = self.cmu * torch.sum((y_down_parents.pow(2)) * W, dim=0)
            rankmu_up = self.cmu * torch.sum((y_up_parents.pow(2)) * W, dim=0)

            diag_C_down_new = (
                (1.0 - self.c1 - self.cmu) * self.diag_C_down[layer_idx]
                + rank1_down
                + rankmu_down
            ).clamp(self.min_var, self.max_var)
            diag_C_up_new = (
                (1.0 - self.c1 - self.cmu) * self.diag_C_up[layer_idx]
                + rank1_up
                + rankmu_up
            ).clamp(self.min_var, self.max_var)

            # ===== Persist state and refresh runtime S tensors =====
            self.ps_down[layer_idx].copy_(ps_down_new)
            self.ps_up[layer_idx].copy_(ps_up_new)
            self.pc_down[layer_idx].copy_(pc_down_new)
            self.pc_up[layer_idx].copy_(pc_up_new)
            self.diag_C_down[layer_idx].copy_(diag_C_down_new)
            self.diag_C_up[layer_idx].copy_(diag_C_up_new)
            self.layer_sigma[layer_idx] = sigma_new

            # Effective stdevs used by kernels: S = sigma * sqrt(diag_C)
            Sd_new = (sigma_new * torch.sqrt(diag_C_down_new)).to(self.dtype)
            Su_new = (sigma_new * torch.sqrt(diag_C_up_new)).to(self.dtype)
            self.qkv_down_sigma[layer_idx].copy_(Sd_new)
            self.qkv_up_sigma[layer_idx].copy_(Su_new)
