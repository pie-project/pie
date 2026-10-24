//! Expert-parallel wiring of one process's ranks: each rank's EP signal
//! block, landing (the rows its peers dispatch to it) and return (the
//! outputs its peers combine back to it), mapped into every rank. Peers
//! write only into these fixed buffers: each rank's arena is laid out per
//! program and bucket, so no peer knows where a rank's IR values live. The
//! dispatch and combine arms copy from them into the IR's values. Nothing
//! here is taken while the ranks fire: an allocation between two ranks'
//! launches can wait on a peer spinning for the rank that waits.
#![allow(unsafe_code)]

use kernels_cuda::Tensor;
use kernels_cuda::collective::{EP_MAX_RANKS, EP_SIGNAL_BYTES, Ep, Landing, ep_scratch_bytes};

use crate::error::{Fault, Result};
use poem_ir::Dtype;

/// The widest step any dispatch or combine of the group fires: sizes the
/// fixed buffers once, at load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Widest {
    /// The largest token bucket a rank fires.
    pub tokens: u32,
    /// Bytes of a dispatched token row.
    pub row_bytes: u64,
    /// Bytes of an expert output row.
    pub out_bytes: u64,
    /// The widest fan-out.
    pub top_k: u32,
}

/// One rank's part of the wiring.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wire {
    pub ep: Ep,
    widest: Widest,
    rows: u64,
    local: u64,
    row_weights: u64,
    origin: u64,
    routed: u64,
}

impl Wire {
    /// This rank's landing at a step of `tokens` (every rank's bucket) and
    /// `k` routes a token, viewed as the kernels take it: `world * tokens`
    /// rows of `width` elements of `dtype`, dense from each buffer's start.
    pub fn landing(&self, tokens: u32, k: u32, width: u32, dtype: Dtype, weights: bool) -> Landing {
        let cap = self.ep.world * tokens;
        Landing {
            rows: Tensor::new(self.rows, cap, width, dtype),
            local: Tensor::new(self.local, cap, k, Dtype::I32),
            row_weights: weights.then(|| Tensor::new(self.row_weights, cap, k, Dtype::F32)),
            origin: Tensor::new(self.origin, cap, 1, Dtype::I32),
        }
    }

    /// This rank's return at a step of `tokens` and `k` routes a token:
    /// `tokens * k` rows of `width` elements of `dtype`.
    pub fn routed(&self, tokens: u32, k: u32, width: u32, dtype: Dtype) -> Tensor {
        Tensor::new(self.routed, tokens * k, width, dtype)
    }

    /// Whether a step of `tokens` rows `row_bytes` wide (outputs `out_bytes`)
    /// fits the buffers sized at load.
    #[must_use]
    pub fn holds(&self, tokens: u32, row_bytes: u64, out_bytes: u64, top_k: u32) -> bool {
        let w = self.widest;
        tokens <= w.tokens
            && row_bytes <= w.row_bytes
            && out_bytes <= w.out_bytes
            && top_k == w.top_k
    }
}

/// Wires `ordinals` (one rank each, in rank order) for expert parallelism
/// at `widest`. The ranks must reach one another as the peer collectives
/// do ([`crate::comm::open_peers`] checks the path first); there is no
/// fallback for a dispatch.
pub fn wire(ordinals: &[i32], widest: Widest) -> Result<Vec<Wire>> {
    let off = |why: String| Fault::program("ep::wire", why);
    let world = u32::try_from(ordinals.len()).unwrap_or(u32::MAX);
    if !(2..=EP_MAX_RANKS).contains(&world) {
        return Err(off(format!(
            "expert parallelism runs 2 to 8 ranks, not {world}"
        )));
    }
    if crate::comm::open_peers(ordinals).is_none() {
        return Err(off(
            "expert parallelism needs every rank to map every other's memory (NVLink peers)".into(),
        ));
    }
    let cap = u64::from(world) * u64::from(widest.tokens);
    let k = u64::from(widest.top_k);
    let parts = [
        EP_SIGNAL_BYTES,
        cap * widest.row_bytes,
        cap * k * 4,
        cap * k * 4,
        cap * 4,
        u64::from(widest.tokens) * k * widest.out_bytes,
        ep_scratch_bytes(widest.tokens, world),
    ];
    let mut bases = Vec::with_capacity(ordinals.len());
    for &a in ordinals {
        bases.push(carve(a, ordinals, &parts)?);
        crate::device::ctx::bind_thread(a)?;
        kernels_cuda::collective::load_ep().map_err(crate::error::kernel)?;
    }
    let mut signals = [0u64; EP_MAX_RANKS as usize];
    for (s, b) in signals.iter_mut().zip(&bases) {
        *s = b[0];
    }
    Ok(bases
        .iter()
        .enumerate()
        .map(|(rank, b)| Wire {
            ep: Ep {
                rank: rank as u32,
                world,
                signals,
                scratch: b[6],
                scratch_tokens: widest.tokens,
            },
            widest,
            rows: b[1],
            local: b[2],
            row_weights: b[3],
            origin: b[4],
            routed: b[5],
        })
        .collect())
}

/// A rank's mapping: the signal block, the landing's rows, local slots,
/// weights and origins, the return, and the dispatch scratch.
const PARTS: usize = 7;

/// One mapping on `device` that `readers` reach, cut into `parts` 256-byte
/// aligned, the first (the signal block) zeroed.
#[cfg(feature = "cuda")]
fn carve(device: i32, readers: &[i32], parts: &[u64; PARTS]) -> Result<[u64; PARTS]> {
    use crate::device::ctx::check;
    use crate::device::rt;

    let at = |p: &[u64]| p.iter().map(|b| b.next_multiple_of(256)).sum::<u64>();
    let base = crate::device::elastic::shared(device, readers, at(parts))?;
    let mut cut = [0u64; PARTS];
    for (i, c) in cut.iter_mut().enumerate() {
        *c = base + at(&parts[..i]);
    }
    // SAFETY: the signal block is the first `EP_SIGNAL_BYTES` of the mapping
    // just made on `device`.
    unsafe {
        check("cudaSetDevice", rt::cudaSetDevice(device))?;
        check(
            "cudaMemset",
            rt::cudaMemset(
                cut[0] as *mut core::ffi::c_void,
                0,
                EP_SIGNAL_BYTES as usize,
            ),
        )?;
    }
    Ok(cut)
}

#[cfg(not(feature = "cuda"))]
fn carve(_: i32, _: &[i32], _: &[u64; PARTS]) -> Result<[u64; PARTS]> {
    Err(Fault::Runtimeless)
}
