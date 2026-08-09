use kernels::{KernelSig, kernel, operands};

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The head dims `DISPATCH_HEAD_DIM` instantiates, in its order.
pub const HEAD_DIMS: &[u32] = &[64, 128, 256, 512];

/// `num_smem_stages`, as the removed `MergeStates` and `VariableLengthMergeStates`
/// host launchers set it.
pub const NUM_SMEM_STAGES: u32 = 4;

/// `num_threads`, as the removed `MergeStates` and `VariableLengthMergeStates`
/// host launchers set it.
pub const NUM_THREADS: u32 = 128;

/// `(vec_size, bdx, bdy)` for a head dim, as the removed `MergeStates` host
/// launcher derived it.
#[must_use]
pub const fn geometry(head_dim: u32) -> Option<(u32, u32, u32)> {
    let vec_size = match head_dim {
        64 | 128 | 256 => 8,
        512 => 16,
        _ => return None,
    };
    let bdx = head_dim / vec_size;
    Some((vec_size, bdx, NUM_THREADS / bdx))
}

/// The staged arms' dynamic shared memory, as the removed `MergeStates` and
/// `VariableLengthMergeStates` host launchers computed it.
#[must_use]
pub const fn smem_bytes(head_dim: u32) -> Option<u32> {
    let Some((_, _, bdy)) = geometry(head_dim) else {
        return None;
    };
    Some(NUM_SMEM_STAGES * bdy * head_dim * 2 + NUM_THREADS * 4)
}

/// The one unit: `csrc/src/cascade/merge_states.cuh`, ten rows.
pub static UNITS: &[Unit] = &[MERGE_STATES];

/// `cascade.cuh`'s three merge kernels at the four head dims
pub const MERGE_STATES: Unit = Unit {
    name: "cascade/merge_states",
    root: ROOT,
    rows: MERGE_STATES_ROWS,
    options: OPTIONS,
};

/// The root, bound once.
const ROOT: &str = include_str!("../../csrc/src/cascade/merge_states.cuh");

/// `--device-as-default-execution-space`, and it is load-bearing here for the
const OPTIONS: &[&str] = &["--device-as-default-execution-space"];

/// The `__global__` this unit's first six rows instantiate.
const MERGE_PATH: &str = "::flashinfer::MergeStatesKernel";

/// `cascade.cuh:275-281`.
const LARGE_PATH: &str = "::flashinfer::MergeStatesLargeNumIndexSetsKernel";

/// `cascade.cuh:366-371`.
const VARLEN_PATH: &str = "::flashinfer::PersistentVariableLengthMergeStatesKernel";

/// The ten contracts.
#[rustfmt::skip]
static MERGE_STATES_SIGS: [KernelSig; 10] = [
    kernel!(merge_states_v8 "attn::cascade::merge_states_v8",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf,
            s: F32s,
            v_merged: BufMut,
            s_merged: F32sMut | null,
            num_index_sets: U32,
            num_heads: U32,
            head_dim: U32,
        ]),
    kernel!(merge_states_v16 "attn::cascade::merge_states_v16",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf,
            s: F32s,
            v_merged: BufMut,
            s_merged: F32sMut | null,
            num_index_sets: U32,
            num_heads: U32,
            head_dim: U32,
        ]),
    kernel!(merge_states_large_hd64 "attn::cascade::merge_states_large_hd64",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    kernel!(merge_states_large_hd128 "attn::cascade::merge_states_large_hd128",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    kernel!(merge_states_large_hd256 "attn::cascade::merge_states_large_hd256",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    kernel!(merge_states_large_hd512 "attn::cascade::merge_states_large_hd512",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, v_merged: BufMut, s_merged: F32sMut | null,
            num_index_sets: U32, num_heads: U32,
        ]),
    kernel!(merge_states_varlen_hd64 "attn::cascade::merge_states_varlen_hd64",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
    kernel!(merge_states_varlen_hd128 "attn::cascade::merge_states_varlen_hd128",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
    kernel!(merge_states_varlen_hd256 "attn::cascade::merge_states_varlen_hd256",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
    kernel!(merge_states_varlen_hd512 "attn::cascade::merge_states_varlen_hd512",
        file = Some("cascade/merge_states.cuh"),
        operands = operands![
            v: Buf, s: F32s, indptr: I32s,
            v_merged: BufMut, s_merged: F32sMut | null,
            max_seq_len: U32, seq_len: U32s | null, num_heads: U32,
        ]),
];

/// The ten instantiations, in [`MERGE_STATES_SIGS`]' order.
#[rustfmt::skip]
static MERGE_STATES_ROWS: &[DeviceKernel] = &[
    DeviceKernel { sig: &MERGE_STATES_SIGS[0], template_path: MERGE_PATH, elem: concat!(
        "8, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[1], template_path: MERGE_PATH, elem: concat!(
        "16, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },

    DeviceKernel { sig: &MERGE_STATES_SIGS[2], template_path: LARGE_PATH, elem: concat!(
        "8, 8, 16, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[3], template_path: LARGE_PATH, elem: concat!(
        "8, 16, 8, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[4], template_path: LARGE_PATH, elem: concat!(
        "8, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[5], template_path: LARGE_PATH, elem: concat!(
        "16, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO") },

    DeviceKernel { sig: &MERGE_STATES_SIGS[6], template_path: VARLEN_PATH, elem: concat!(
        "8, 8, 16, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[7], template_path: VARLEN_PATH, elem: concat!(
        "8, 16, 8, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[8], template_path: VARLEN_PATH, elem: concat!(
        "8, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
    DeviceKernel { sig: &MERGE_STATES_SIGS[9], template_path: VARLEN_PATH, elem: concat!(
        "16, 32, 4, 4, ::pie_cuda_driver::kernels::cascade::DTypeIn, ",
        "::pie_cuda_driver::kernels::cascade::DTypeO, ",
        "::pie_cuda_driver::kernels::cascade::IdType") },
];

/// The symbol that merges a uniform-chunk-count batch at `head_dim`.
#[must_use]
pub fn merge_states_symbol(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 | 128 | 256 => Some(MERGE_STATES_SIGS[0].symbol),
        512 => Some(MERGE_STATES_SIGS[1].symbol),
        _ => None,
    }
}

/// The symbol for the staged arm at `head_dim` — `cascade.cuh`'s
/// `MergeStatesLargeNumIndexSetsKernel`.
#[must_use]
pub fn merge_states_large_symbol(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(MERGE_STATES_SIGS[2].symbol),
        128 => Some(MERGE_STATES_SIGS[3].symbol),
        256 => Some(MERGE_STATES_SIGS[4].symbol),
        512 => Some(MERGE_STATES_SIGS[5].symbol),
        _ => None,
    }
}

/// The symbol for the variable-length arm at `head_dim` — `cascade.cuh`'s
/// `PersistentVariableLengthMergeStatesKernel`.
#[must_use]
pub fn merge_states_varlen_symbol(head_dim: u32) -> Option<&'static str> {
    match head_dim {
        64 => Some(MERGE_STATES_SIGS[6].symbol),
        128 => Some(MERGE_STATES_SIGS[7].symbol),
        256 => Some(MERGE_STATES_SIGS[8].symbol),
        512 => Some(MERGE_STATES_SIGS[9].symbol),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        HEAD_DIMS, MERGE_STATES, NUM_SMEM_STAGES, NUM_THREADS, geometry,
        merge_states_large_symbol, merge_states_symbol, merge_states_varlen_symbol, smem_bytes,
    };

    /// The four head dims this lattice covers are FA2's four.
    #[test]
    fn the_lattice_covers_flashinfer_fa2s_head_dims() {
        assert_eq!(HEAD_DIMS, crate::families::fa2::HEAD_DIMS);
    }

    /// Every head dim resolves all three symbols, and nothing else resolves
    #[test]
    fn every_head_dim_has_all_three_arms() {
        for &hd in HEAD_DIMS {
            assert!(merge_states_symbol(hd).is_some(), "{hd}");
            assert!(merge_states_large_symbol(hd).is_some(), "{hd}");
            assert!(merge_states_varlen_symbol(hd).is_some(), "{hd}");
            assert!(geometry(hd).is_some(), "{hd}");
            assert!(smem_bytes(hd).is_some(), "{hd}");
        }
        for hd in [0u32, 32, 96, 120, 1024] {
            assert!(merge_states_symbol(hd).is_none(), "{hd}");
            assert!(merge_states_large_symbol(hd).is_none(), "{hd}");
            assert!(merge_states_varlen_symbol(hd).is_none(), "{hd}");
            assert!(geometry(hd).is_none(), "{hd}");
            assert!(smem_bytes(hd).is_none(), "{hd}");
        }
    }

    /// Every symbol this family answers is a row of its unit.
    #[test]
    fn every_symbol_is_a_row() {
        for &hd in HEAD_DIMS {
            for symbol in [
                merge_states_symbol(hd).unwrap(),
                merge_states_large_symbol(hd).unwrap(),
                merge_states_varlen_symbol(hd).unwrap(),
            ] {
                assert!(MERGE_STATES.hosts(symbol), "{symbol}");
            }
        }
    }

    /// [`geometry`] and the rows' `elem` strings agree.
    #[test]
    fn the_rows_match_the_derivation() {
        for &hd in HEAD_DIMS {
            let (vec_size, bdx, bdy) = geometry(hd).unwrap();
            assert_eq!(bdx * vec_size, hd, "bdx * vec_size is the head dim at {hd}");
            assert_eq!(bdx * bdy, NUM_THREADS, "the staged block is 128 threads at {hd}");

            let want = format!("{vec_size}, {bdx}, {bdy}, {NUM_SMEM_STAGES}, ");
            for symbol in
                [merge_states_large_symbol(hd).unwrap(), merge_states_varlen_symbol(hd).unwrap()]
            {
                let row = MERGE_STATES.row(symbol).expect("the symbol is a row");
                assert!(row.elem.starts_with(&want), "{symbol}: {:?} vs {want:?}", row.elem);
            }

            let row = MERGE_STATES.row(merge_states_symbol(hd).unwrap()).unwrap();
            assert!(row.elem.starts_with(&format!("{vec_size}, ")), "{}", row.elem);
        }
    }

    /// The two shared-memory figures, as the removed `MergeStates` host launcher
    /// computed them.
    #[test]
    fn the_shared_memory_is_the_figure_the_record_carries() {
        assert_eq!(smem_bytes(64), Some(8_704));
        assert_eq!(smem_bytes(128), Some(8_704));
        assert_eq!(smem_bytes(256), Some(8_704));
        assert_eq!(smem_bytes(512), Some(16_896));
        for &hd in HEAD_DIMS {
            assert!(smem_bytes(hd).unwrap() < 48 * 1024, "{hd}");
        }
    }
}
