//! The Qwen3.5 hybrid family's per-fire driver objects —
//! gate-linear-attn-ws.
//!
//! Ports the host side of `model/qwen3_5/qwen3_5_forward.hpp`: the
//! linear-attention workspace (twenty-seven device buffers, allocated once
//! and reused across the fire's linear layers), and the family's two small
//! knob/state structs. The forward body itself is emitter territory.
//!
//! The allocation ORDER is part of the contract — a suballocator's
//! addresses depend on it — and it is not the declaration order:
//! `mixed_qkv_post` lands before `z`/`a`/`b`, exactly as the C++ source
//! writes it. The parity transcript pins both the order and every size.

use std::ffi::c_void;

use super::sideband_arena::DeviceMemory;

/// The seven dims the workspace is sized from.
///
/// A struct rather than seven positional ints for the reason the header
/// warns about: `k_h`/`v_h` and `k_d`/`v_d` transpose silently, and the
/// result is wrong in a way that only shows up as corrupted state later.
#[derive(Debug, Clone, Copy)]
pub struct LinearAttnDims {
    /// Token capacity the buffers are sized for.
    pub max_tokens: i32,
    /// Channels in the fused conv input.
    pub conv_dim: i32,
    /// Value heads.
    pub v_h: i32,
    /// Key heads.
    pub k_h: i32,
    /// Key dimension per head.
    pub k_d: i32,
    /// Value dimension per head.
    pub v_d: i32,
    /// Full-attention query heads (the q+gate packed buffers).
    pub hq: i32,
}

/// Per-linear-attention-layer extra workspace. Ports
/// `Qwen3_5LinearAttnWorkspace`; every field mirrors a `DeviceBuffer`
/// member as a raw device pointer (null = empty, the C++'s zero-count
/// buffer).
#[derive(Debug)]
#[allow(missing_docs)]  // the field names ARE the C++ member names; the
                        // header carries their shape comments.
pub struct LinearAttnWorkspace {
    pub mixed_qkv: *mut c_void,
    pub mixed_qkvz: *mut c_void,
    pub ba: *mut c_void,
    pub z: *mut c_void,
    pub a: *mut c_void,
    pub b: *mut c_void,
    pub mixed_qkv_post: *mut c_void,
    pub q_norm: *mut c_void,
    pub k_norm: *mut c_void,
    pub v_fp32: *mut c_void,
    pub g_log: *mut c_void,
    pub beta: *mut c_void,
    pub core_out: *mut c_void,
    pub core_out_bf16: *mut c_void,
    pub q_raw: *mut c_void,
    pub k_raw: *mut c_void,
    pub v_raw: *mut c_void,
    pub q_pre: *mut c_void,
    pub k_pre: *mut c_void,
    pub fa_qg_packed: *mut c_void,
    pub fa_gate: *mut c_void,
    pub qo_ext: *mut c_void,
    pub rs_write_state_mask: *mut c_void,
    pub qo_split: *mut c_void,
    pub split_slot_head: *mut c_void,
    pub split_slot_tail: *mut c_void,
    pub split_mask_head: *mut c_void,
    /// The capacity the N-scaled buffers were sized for.
    pub max_tokens: i32,
    released: bool,
}

impl LinearAttnWorkspace {
    /// Allocate all twenty-seven buffers, in the C++ SOURCE order. A
    /// zero-element buffer allocates nothing and stays null, as
    /// `DeviceBuffer`'s zero-count constructor does.
    pub fn allocate<M: DeviceMemory>(ops: &mut M, dims: &LinearAttnDims) -> Self {
        let n = u64::try_from(dims.max_tokens.max(0)).unwrap_or(0);
        let conv = u64::try_from(dims.conv_dim.max(0)).unwrap_or(0);
        let v_h = u64::try_from(dims.v_h.max(0)).unwrap_or(0);
        let k_h = u64::try_from(dims.k_h.max(0)).unwrap_or(0);
        let k_d = u64::try_from(dims.k_d.max(0)).unwrap_or(0);
        let v_d = u64::try_from(dims.v_d.max(0)).unwrap_or(0);
        let hq = u64::try_from(dims.hq.max(0)).unwrap_or(0);
        let v_dim = v_h * v_d;
        let k_dim = k_h * k_d;

        let mut alloc = |elems: u64, elem_size: u64| -> *mut c_void {
            if elems == 0 {
                return std::ptr::null_mut();
            }
            let bytes = usize::try_from(elems * elem_size).unwrap_or(usize::MAX);
            ops.alloc(bytes).unwrap_or(std::ptr::null_mut())
        };

        let mixed_qkv = alloc(n * conv, 2);
        let mixed_qkvz = alloc(n * (conv + v_dim), 2);
        let ba = alloc(n * 2 * v_h, 2);
        let mixed_qkv_post = alloc(n * conv, 2);
        let z = alloc(n * v_dim, 2);
        let a = alloc(n * v_h, 2);
        let b = alloc(n * v_h, 2);
        let q_norm = alloc(n * v_h * k_d, 4);
        let k_norm = alloc(n * v_h * k_d, 4);
        let v_fp32 = alloc(n * v_dim, 4);
        let g_log = alloc(n * v_h, 4);
        let beta = alloc(n * v_h, 4);
        let core_out = alloc(n * v_dim, 4);
        let core_out_bf16 = alloc(n * v_dim, 2);
        let q_raw = alloc(n * k_dim, 2);
        let k_raw = alloc(n * k_dim, 2);
        let v_raw = alloc(n * v_dim, 2);
        let q_pre = alloc(n * k_h * k_d, 4);
        let k_pre = alloc(n * k_h * k_d, 4);
        let fa_qg_packed = alloc(n * 2 * hq, 2);
        let fa_gate = alloc(n * hq, 2);
        let qo_ext = alloc(n + 1, 4);
        let rs_write_state_mask = alloc(n + 1, 1);
        let qo_split = alloc(2 * n + 1, 4);
        let split_slot_head = alloc(2 * n, 4);
        let split_slot_tail = alloc(2 * n, 4);
        let split_mask_head = alloc(2 * n, 1);

        Self {
            mixed_qkv,
            mixed_qkvz,
            ba,
            z,
            a,
            b,
            mixed_qkv_post,
            q_norm,
            k_norm,
            v_fp32,
            g_log,
            beta,
            core_out,
            core_out_bf16,
            q_raw,
            k_raw,
            v_raw,
            q_pre,
            k_pre,
            fa_qg_packed,
            fa_gate,
            qo_ext,
            rs_write_state_mask,
            qo_split,
            split_slot_head,
            split_slot_tail,
            split_mask_head,
            max_tokens: dims.max_tokens,
            released: false,
        }
    }

    /// The C++ destructor's frees, explicit for the usual reason: the
    /// memory seam is not owned by the workspace.
    pub fn release<M: DeviceMemory>(&mut self, ops: &mut M) {
        if self.released {
            return;
        }
        for p in [
            self.mixed_qkv,
            self.mixed_qkvz,
            self.ba,
            self.z,
            self.a,
            self.b,
            self.mixed_qkv_post,
            self.q_norm,
            self.k_norm,
            self.v_fp32,
            self.g_log,
            self.beta,
            self.core_out,
            self.core_out_bf16,
            self.q_raw,
            self.k_raw,
            self.v_raw,
            self.q_pre,
            self.k_pre,
            self.fa_qg_packed,
            self.fa_gate,
            self.qo_ext,
            self.rs_write_state_mask,
            self.qo_split,
            self.split_slot_head,
            self.split_slot_tail,
            self.split_mask_head,
        ] {
            if !p.is_null() {
                ops.free(p);
            }
        }
        self.released = true;
    }
}

impl Drop for LinearAttnWorkspace {
    fn drop(&mut self) {
        debug_assert!(self.released, "LinearAttnWorkspace dropped without release()");
    }
}

/// Per-fire knobs for the Qwen3.5 forward. Ports `Qwen3_5ForwardCfg`;
/// the parity transcript pins every default.
#[derive(Debug)]
pub struct Qwen35ForwardCfg {
    /// Keep every fire on the prefill kernel, even pure decodes.
    pub force_prefill_path: bool,
    /// Which short prefill-like batches use graph-friendly planning.
    pub small_prefill_naive_attention_max_tokens: i32,
    /// Tensor-parallel world size.
    pub tp_size: i32,
    /// The TP communicator; an opaque pointer until the distributed layer
    /// is ported.
    pub tp_comm: *mut c_void,
    /// Qwen MTP: pin the paged cache lookup to the verified source prefix
    /// while draft positions advance.
    pub mtp_global_cache_uses_prefix_position: bool,
}

impl Default for Qwen35ForwardCfg {
    fn default() -> Self {
        Self {
            force_prefill_path: false,
            small_prefill_naive_attention_max_tokens: 0,
            tp_size: 1,
            tp_comm: std::ptr::null_mut(),
            mtp_global_cache_uses_prefix_position: false,
        }
    }
}

/// Persistent decode-plan cache for the family. Ports `Qwen3_5PlanState`;
/// the plan handles stay opaque behind the same seam the llama-like state
/// uses.
#[derive(Debug, Default)]
pub struct Qwen35PlanState<D, P> {
    /// The pure-decode flashinfer plan.
    pub decode_plan: Option<D>,
    /// The prefill plan.
    pub prefill_plan: Option<P>,
    /// The body dispatches through `prefill_plan`.
    pub use_prefill_plan: bool,
}
