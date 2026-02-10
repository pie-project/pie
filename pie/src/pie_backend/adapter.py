from __future__ import annotations
import io
import os
import time
import torch
import torch.nn as nn
import math
from .rand_mv import RAND_MV_AVAILABLE

if RAND_MV_AVAILABLE:
    from . import rand_mv


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
        i = 0
        for adapter_index, count in self.adapter_indices_rle:

            x_start = self.qo_indptr[i]
            x_end = self.qo_indptr[i + count]
            x = xs[x_start:x_end]

            # In multi-GPU, xs is replicated (full hidden state), so inputs are consistent.
            # q_state, k_state, v_state are SHARDED (local output).

            rand_seeds = self.rand_seeds[x_start:x_end]
            inject_noise = rand_seeds.any().item()
            assert x.shape[0] == rand_seeds.shape[0], "Batch size must match seeds."

            Wd = self.adapter_at_layer[layer_idx][0][adapter_index]
            Wu = self.adapter_at_layer[layer_idx][1][
                adapter_index
            ]  # (rank, LOCAL_d_q+LOCAL_d_k+LOCAL_d_k)
            adapter_info = self.adapter_extras[adapter_index]

            rank_lora = adapter_info.rank  # LoRA rank, not GPU rank

            # We assume CmaesAdapter was initialized with LOCAL out_features.
            # So adapter_info.out_features is [local_d_q, local_d_k, local_d_v]
            out_indptr = adapter_info.out_features_indptr  # [0, local_d_q, ...]

            # Down-projection is replicated on input (x), produces replicated output (low-rank).
            # This part executes identically on all ranks if X is identical.
            qkv_down = x @ Wd
            d_q, d_k, d_v = torch.split(
                qkv_down, [rank_lora, rank_lora, rank_lora], dim=-1
            )

            # Determine if we should inject noise for this request slice
            layer_seeds = rand_seeds - layer_idx
            # inject_noise = (layer_seeds != 0).any().item()
            # Only inject noise if CUDA/Triton is available (rand_mv requires CUDA)
            inject_noise = RAND_MV_AVAILABLE

            if inject_noise:
                if not isinstance(adapter_info, CmaesAdapter):
                    continue

                Sd = adapter_info.qkv_down_sigma[layer_idx]  # (in_features, 3*rank)
                Sd_q, Sd_k, Sd_v = torch.split(
                    Sd, [rank_lora, rank_lora, rank_lora], dim=-1
                )

                # Noise generation for DOWN projection (input x is replicated, output d_q is replicated)
                # We want the SAME noise on all ranks to keep d_q consistent.

                q_noise_down = rand_mv.batched_randn_matmul(
                    x,
                    seeds=rand_seeds + layer_idx,
                    S=Sd_q,
                    out_dtype=x.dtype,
                )
                k_noise_down = rand_mv.batched_randn_matmul(
                    x,
                    seeds=rand_seeds + (layer_idx + 100),
                    S=Sd_k,
                    out_dtype=x.dtype,
                )
                v_noise_down = rand_mv.batched_randn_matmul(
                    x,
                    seeds=rand_seeds + (layer_idx + 200),
                    S=Sd_v,
                    out_dtype=x.dtype,
                )

                d_q = d_q + q_noise_down
                d_k = d_k + k_noise_down
                d_v = d_v + v_noise_down

            # Up-projection: Input (d_q/k/v) is replicated. Output (u_q/k/v) needs to be SHARDED.
            # Wu is already LOCAL.

            Wu_q_local = Wu[:, out_indptr[0] : out_indptr[1]]
            Wu_k_local = Wu[:, out_indptr[1] : out_indptr[2]]
            Wu_v_local = Wu[:, out_indptr[2] : out_indptr[3]]

            # Compute local up-projection
            u_q_local = d_q @ Wu_q_local
            u_k_local = d_k @ Wu_k_local
            u_v_local = d_v @ Wu_v_local

            if inject_noise:
                Su = adapter_info.qkv_up_sigma[layer_idx]  # (rank, LOCAL_d_sum)

                # Slicing LOCAL Sigmas
                Su_q_local = Su[:, out_indptr[0] : out_indptr[1]]
                Su_k_local = Su[:, out_indptr[1] : out_indptr[2]]
                Su_v_local = Su[:, out_indptr[2] : out_indptr[3]]

                # Offsets for noise generation
                local_d_q = out_indptr[1] - out_indptr[0]
                local_d_k = out_indptr[2] - out_indptr[1]
                local_d_v = out_indptr[3] - out_indptr[2]

                global_d_q = local_d_q * world_size
                global_d_k = local_d_k * world_size
                global_d_v = local_d_v * world_size

                q_noise_up = rand_mv.batched_randn_matmul(
                    d_q,
                    seeds=rand_seeds - layer_idx,
                    S=Su_q_local,
                    out_dtype=x.dtype,
                    col_offset=rank * local_d_q,
                    global_cols=global_d_q,
                )
                k_noise_up = rand_mv.batched_randn_matmul(
                    d_k,
                    seeds=rand_seeds - (layer_idx + 100),
                    S=Su_k_local,
                    out_dtype=x.dtype,
                    col_offset=rank * local_d_k,
                    global_cols=global_d_k,
                )
                v_noise_up = rand_mv.batched_randn_matmul(
                    d_v,
                    seeds=rand_seeds - (layer_idx + 200),
                    S=Su_v_local,
                    out_dtype=x.dtype,
                    col_offset=rank * local_d_v,
                    global_cols=global_d_v,
                )
                u_q_local = u_q_local + q_noise_up
                u_k_local = u_k_local + k_noise_up
                u_v_local = u_v_local + v_noise_up

            # ===== 3) Combine mean + noise =====
            scaling = adapter_info.alpha / float(rank_lora)

            q_final = scaling * u_q_local
            k_final = scaling * u_k_local
            v_final = scaling * u_v_local

            q_state[x_start:x_end].add_(q_final)
            k_state[x_start:x_end].add_(k_final)
            v_state[x_start:x_end].add_(v_final)

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

    # qkv_down_weight: list[torch.Tensor]
    # qkv_up_weight: list[torch.Tensor]
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

            # self.qkv_down_weight.append(qkv_down_weight)
            # self.qkv_up_weight.append(qkv_up_weight)
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

    def upload(self, name: str, data: bytes) -> None:
        """
        Loads the adapter's state from a file and populates its parameters and state.

        The 'data' parameter, if provided, is treated as a bytes buffer containing the checkpoint.
        If 'data' is empty, it falls back to loading from f"adapter_{name}.pt".
        """
        state_dict = None
        # if data:
        #     try:
        #         # print(f"Loading adapter {name} from memory ({len(data)} bytes)...")
        #         buffer = io.BytesIO(data)
        #         state_dict = torch.load(buffer, map_location=self.device)
        #     except Exception as e:
        #         print(f"Failed to load adapter from memory: {e}. Falling back to file.")

        if state_dict is None:
            filename = os.path.join(self.adapter_path, f"adapter_{name}.pt")
            # print("Loading adapter from", filename)
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
        # print("Saving adapter to", filename)
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
                "CmaesAdapter.update() requires CUDA for noise generation. "
                "This function is not available on non-CUDA platforms (e.g., Apple Metal)."
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
            )  # self.qkv_down_weight[layer_idx].to(f32)  # (I, 3*rank)
            mean_up_old = self.adapter_at_layer[layer_idx][1][self.adapter_id].to(
                f32
            )  # self.qkv_up_weight[layer_idx].to(f32)  # (rank, d_sum)

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
