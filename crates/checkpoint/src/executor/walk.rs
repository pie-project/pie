//! Walking a finished plan: one instruction at a time, against the real
//! checkpoint bytes.
//!
//! [`super::Execution`] is how this is reached, and holds the reasons for the
//! choices a caller makes. What is here is the walk itself.
//!
//! # It is not the "host" executor
//!
//! It was called that for as long as the only arena was a `Vec<u8>`, and the
//! name became wrong the moment [`ArenaBacking`] existed: hand this walker a
//! `CudaArena` and the reads are still host reads, but the writes land in
//! device memory and the transforms run on a GPU. What is fixed is not WHERE
//! the work happens — it is that the *decisions* happen here, once, for every
//! backing. A backing that ran a different plan would be a second compiler.
//!
//! # It opens files
//!
//! The one module below `lib.rs` that does, which is why it is named for what
//! it is rather than sharing a name with the backend whose plans it accepts:
//! `crate::plan::passes::tile` decides how a plan is lowered, and this
//! executes the result. The compiler is on the other side of that line, and
//! `tests/standalone.rs` pins it — with this file as the one exemption.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::bf16;

use std::collections::HashSet;

use super::iq_grid;
use super::{Progress, Residency};
use crate::codec::cast::{cast_elements, decode_values, encode_values};
use crate::codec::fp8::{decode_fp8_e4m3_elements, f32_to_fp8_e4m3};
use crate::codec::int4::decode_int4b8_elements;
use crate::codec::mlx::decode_mlx_affine_codes;
use crate::codec::mlx::mlx_affine_group_params;
use crate::codec::mxfp4::{decode_mxfp4_elements, encode_mxfp4_group};
use crate::codec::rows::{EncodeOperand, encode_rows};
use crate::error::Error;
use crate::executor::arena::{ArenaBacking, ArenaSpan, TileMapOp};
use crate::executor::sink::TensorSink;
use crate::plan::index::{PlanIndex, instr_by_id};
use crate::plan::{
    CONVERT_TILE_MAP_MASK, DestExtent, Extent, GatherSpec, LoadPlan, SourceExtent, StorageInstr,
    TileMapKind, TransformSpec,
};
use crate::types::{BufferId, DType, Encoding, QuantScheme, RepackLayout, TILED_BAND, TILED_STEP};

#[derive(Debug, Clone)]
enum BufferLoc {
    Arena {
        offset: usize,
        len: usize,
    },
    Owned(Vec<u8>),
    View {
        input: BufferId,
        offset: usize,
        len: usize,
    },
}

#[derive(Debug, Clone, Copy)]
enum Root {
    Arena,
    Owned(BufferId),
}

/// What a freshly allocated buffer holds before anything writes to it.
///
/// Not zero, and deliberately: `cudaMalloc` does not zero, so an executor that
/// handed out zeroed memory would silently satisfy any tensor with a region no
/// source covers -- which is exactly the region [`StorageInstr::Fill`] exists
/// to cover. With zeroed allocation a missing fill and a working one produce
/// identical bytes, so no test could tell them apart; this makes the
/// difference visible.
const POISON: u8 = 0xAB;

pub(super) fn run(
    plan: &LoadPlan,
    snapshot_dir: &Path,
    residency: Residency<'_>,
    sink: &mut dyn TensorSink,
    progress: &mut dyn FnMut(Progress<'_>),
) -> Result<(), Error> {
    // The gate is what this executor *implements*, and nothing else. It used
    // to be widened by the backing's capability mask, which bought nothing:
    // every mask a target can carry is already a subset of this one, so the
    // widening never admitted a plan and the second reader was a second thing
    // to keep in step. `Repack` is the standing example of what stays refused.
    if plan.target.tile_map_mask & !CONVERT_TILE_MAP_MASK != 0 {
        return Err(invalid(
            "executor received a plan advertising TileMap transforms the host \
             does not implement",
        ));
    }
    // `Streaming` has no arena, and used to be spelled by handing the walker
    // `&mut &mut [][..]` — a zero-length backing standing in for the ABSENCE
    // of one — beside a `stream: bool` saying to ignore it. Two values for
    // one fact, and the placeholder was the thing every arena operation below
    // had to be careful not to touch.
    let mut nothing: &mut [u8] = &mut [];
    let (arena, stream): (&mut dyn ArenaBacking, bool) = match residency {
        Residency::Arena(arena) => (arena, false),
        Residency::Streaming => (&mut nothing, true),
    };
    // Read once, here, rather than per instruction: a backing whose answer
    // changed mid-plan would leave half a load on each path, and this is a
    // property of the backing rather than of a moment.
    let arena_runs_kernels = arena.runs_named_kernels();
    let files = plan
        .files
        .iter()
        .map(|file| {
            let path = PathBuf::from(&file.path);
            let path = if path.is_absolute() {
                path
            } else {
                snapshot_dir.join(path)
            };
            (file.id.0, path)
        })
        .collect::<HashMap<_, _>>();
    // Streaming has no arena: every buffer is owned and freed at its last
    // use, which is the entire point of the mode.
    let arena_len = if stream {
        0
    } else {
        usize::try_from(plan.memory.arena_bytes())
            .map_err(|_| invalid("persistent arena does not fit host address space"))?
    };
    if ArenaBacking::len(arena) < arena_len {
        return Err(invalid(format!(
            "arena is {} bytes and the plan needs {arena_len}",
            ArenaBacking::len(arena)
        )));
    }
    // The poison the executor relies on to tell "written" from "never
    // touched". A caller-supplied arena arrives holding whatever it held --
    // for a fresh device allocation that is zeros, which is a legal tensor
    // and therefore the worst possible disguise for a buffer nothing wrote.
    arena.fill(0, arena_len, POISON)?;
    let last_use = if stream {
        last_uses(plan)?
    } else {
        HashMap::new()
    };
    let mut executor = Walk {
        plan,
        index: PlanIndex::new(plan),
        files,
        arena,
        buffers: HashMap::new(),
        sink,
        finalized: HashSet::new(),
        stream,
        last_use,
        progress,
        read_bytes: 0,
        arena_runs_kernels,
    };
    executor.execute()?;
    // The last writes may still be in flight — `CudaArena` leaves them there
    // on purpose. Draining is the backing's own verb precisely so that no
    // caller has to know whether the one it handed over is such a backing.
    executor.arena.finish()?;
    Ok(())
}

/// The schedule position of each buffer's last reference, views chased.
///
/// A buffer read through a view is a use of the view's root, and the chain is
/// static — [`StorageInstr::CreateView`] names both ends — so the whole
/// analysis is one walk over the schedule. Freeing on this map is safe by
/// construction: a position after a buffer's last recorded use cannot touch
/// it, because touching it would have been recorded.
fn last_uses(plan: &LoadPlan) -> Result<HashMap<BufferId, usize>, Error> {
    let mut roots: HashMap<BufferId, BufferId> = HashMap::new();
    let mut last: HashMap<BufferId, usize> = HashMap::new();
    for (position, id) in plan.schedule.iter().enumerate() {
        let instr = instr_by_id(&plan.instrs, *id)?;
        let mut touch = |buffer: BufferId| {
            let mut buffer = buffer;
            loop {
                last.insert(buffer, position);
                match roots.get(&buffer) {
                    Some(root) => buffer = *root,
                    None => break,
                }
            }
        };
        match instr {
            StorageInstr::Allocate { buffer, .. } | StorageInstr::Fill { buffer, .. } => {
                touch(*buffer);
            }
            StorageInstr::ExtentWrite { dest, .. }
            | StorageInstr::GatherWrite { dest, .. } => touch(dest.buffer),
            StorageInstr::BulkExtentWrite { .. } => {}
            StorageInstr::TileMap {
                inputs,
                outputs,
                dest,
                ..
            } => {
                for buffer in inputs.iter().chain(outputs) {
                    touch(*buffer);
                }
                if let Some(dest) = dest {
                    touch(dest.buffer);
                }
            }
            StorageInstr::CreateView { input, output, .. } => {
                touch(*input);
                touch(*output);
                roots.insert(*output, *input);
            }
            StorageInstr::Finalize { tensor, .. } => touch(*tensor),
        }
    }
    Ok(last)
}

struct Walk<'a, 'p> {
    plan: &'a LoadPlan,
    /// The sparse half of plan lookup. Buffers and instructions are dense, so
    /// they go through [`LoadPlan::buffer`] and [`instr_by_id`] directly; tensor
    /// ids interleave two allocators and need the map.
    index: PlanIndex,
    files: HashMap<u32, PathBuf>,
    arena: &'p mut dyn ArenaBacking,
    buffers: HashMap<BufferId, BufferLoc>,
    sink: &'p mut dyn TensorSink,
    /// Names already published, because finalizing one twice is a plan bug
    /// the executor reports rather than a sink's problem to detect.
    finalized: HashSet<String>,
    /// Streaming: no arena, owned buffers, freed at last use.
    stream: bool,
    /// Filled only when streaming; see [`last_uses`].
    last_use: HashMap<BufferId, usize>,
    progress: &'p mut dyn FnMut(Progress<'_>),
    read_bytes: u64,
    /// Which [`TileMapKind`]s the arena backing runs itself, read once at
    /// entry. Zero is host mode and is every backing that says nothing.
    arena_runs_kernels: bool,
}

/// Decode one GGUF `Q4_0` block: one F16 scale, then sixteen packed bytes whose
/// low nibbles are elements 0..16 and high nibbles 16..32, each
/// `(nibble − 8) × scale`.
///
/// The only block decoder the loader carries, and it is here rather than beside
/// a reader because decoding is not reading: the runtime materialization is the
/// engine's job, and this exists so the offline executor can check the engine's
/// answer against an independent one.
///
/// Checked against a from-scratch reimplementation on
/// `qwen2.5-0.5b-instruct-q4_0.gguf`: for `blk.0.attn_q.weight` both decoders
/// land at cosine 0.89175 against the same model's HuggingFace weights, equal
/// to five decimals. That number is *also* the trap, so it is recorded here.
///
/// Comparing a GGUF against its HuggingFace twin does not give ~0.996 (the
/// error Q4_0 alone costs, which is what `o_proj` and `embed_tokens` show).
/// Every matrix an RMSNorm feeds — q, k, v, gate, up — comes out between 0.80
/// and 0.99, and the norm vectors themselves disagree despite being F32 on
/// both sides. Nothing is wrong: the two published checkpoints simply split
/// the norm scale differently between the norm vector and the columns it
/// multiplies. Only the product is observable, and folding it back
/// (`norm[i] × W[.., i]`) lifts those 120 matrices from a median of 0.977 to
/// 0.995. Tensors no norm feeds are bit-identical — `output_norm.weight` and
/// every attention bias compare exactly equal.
///
/// So a low cosine against an HF twin is evidence about the checkpoints, not
/// about this function. Bit-equality on the unnormalized tensors is the signal
/// worth watching for a regression here.
fn decode_gguf_q4_0_block_into(block: &[u8; 18], values: &mut [f32; 32]) {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for i in 0..16 {
        let packed = block[2 + i];
        let lo = (packed & 0x0f) as i32 - 8;
        let hi = ((packed >> 4) & 0x0f) as i32 - 8;
        values[i] = scale * lo as f32;
        values[i + 16] = scale * hi as f32;
    }
}

/// Decode one GGUF `Q5_0` block: an F16 scale, a 32-bit plane of fifth bits,
/// then sixteen packed bytes. Same halves as `Q4_0`, but each element's high
/// bit comes from the plane, so the range is `(nibble | bit⁴) − 16`.
///
/// The plane is indexed by element, not by byte: bit `i` belongs to the low
/// nibble of byte `i` and bit `i + 16` to its high nibble. Reading it as if it
/// followed the packing order instead is the one way to get this wrong, and it
/// produces plausible numbers rather than an error.
fn decode_gguf_q5_0_block_into(block: &[u8; 22], values: &mut [f32; 32]) {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let plane = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    for i in 0..16 {
        let packed = block[6 + i];
        let lo_bit = ((plane >> i) & 1) << 4;
        let hi_bit = ((plane >> (i + 16)) & 1) << 4;
        let lo = ((packed & 0x0f) as i32 | lo_bit as i32) - 16;
        let hi = ((packed >> 4) as i32 | hi_bit as i32) - 16;
        values[i] = scale * lo as f32;
        values[i + 16] = scale * hi as f32;
    }
}

/// Decode one GGUF `Q8_0` block: an F16 scale and thirty-two signed bytes,
/// each element `byte × scale`. The only GGUF block with no packing at all.
fn decode_gguf_q8_0_block_into(block: &[u8; 34], values: &mut [f32; 32]) {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for i in 0..32 {
        values[i] = scale * f32::from(block[2 + i] as i8);
    }
}

/// Decode one GGUF `Q4_1` block: an F16 scale, an F16 offset, then sixteen
/// packed bytes. Affine rather than symmetric — `nibble × d + m`, with no
/// bias subtracted from the nibble, because the offset already places the
/// range wherever it belongs.
///
/// The offset is *added* here and *subtracted* in the K-quants. Sharing an
/// arm between the two families would be a sign error that survives every
/// shape and size check.
fn decode_gguf_q4_1_block_into(block: &[u8; 20], values: &mut [f32; 32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    for i in 0..16 {
        let packed = block[4 + i];
        values[i] = f32::from(packed & 0x0f) * d + m;
        values[i + 16] = f32::from(packed >> 4) * d + m;
    }
}

/// Decode one GGUF `Q5_1` block: [`decode_gguf_q4_1_block_into`] plus the
/// 32-bit fifth-bit plane [`decode_gguf_q5_0_block_into`] carries, indexed the
/// same way — bit `i` for the low nibble of byte `i`, bit `i + 16` for its
/// high one.
fn decode_gguf_q5_1_block_into(block: &[u8; 24], values: &mut [f32; 32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let plane = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    for i in 0..16 {
        let packed = block[8 + i];
        let lo = (packed & 0x0f) as u32 | (((plane >> i) & 1) << 4);
        let hi = (packed >> 4) as u32 | (((plane >> (i + 16)) & 1) << 4);
        values[i] = lo as f32 * d + m;
        values[i + 16] = hi as f32 * d + m;
    }
}

/// Decode one GGUF `Q2_K` super-block: 256 elements as sixteen sub-blocks of
/// sixteen, each with a four-bit scale and a four-bit minimum, the whole
/// block sharing one F16 scale and one F16 minimum.
///
/// Affine like `Q4_K` -- an element is `d × scaleᵢ × q − dmin × minᵢ` -- but
/// laid out the other way round. `Q4_K` opens with its two F16s; here they
/// close the block, at bytes 80 and 82, after the sixteen sub-block bytes and
/// the sixty-four quant bytes. Reading this one like its neighbour finds the
/// scales where the payload is.
///
/// The payload is read in two 32-byte windows, each visited four times at
/// shifts 0, 2, 4 and 6, and each visit taking sixteen elements from the low
/// half of the window and sixteen from the high half. So a sub-block is
/// sixteen elements that share one byte offset and one shift -- not sixteen
/// consecutive bytes -- and a decoder that walked the quants linearly would
/// produce the right count of plausible numbers in the wrong places.
///
/// Checked against llama.cpp's own dequantizer rather than only against the
/// unit tests below. `Llama-3.2-1B-Instruct-Q2_K.gguf`, tensor
/// `blk.0.ffn_gate.weight`: all 16,777,216 elements are BIT-IDENTICAL to the
/// `gguf` package's `dequantize`, once the reference is rounded to the BF16
/// this writes. Against the model's own BF16 release the same artifact holds
/// a mean cosine of 0.992 over 47 tensors, which is Q2_K's error and not
/// this function's.
fn decode_gguf_q2_k_block_into(block: &[u8; 84], values: &mut [f32; 256]) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = half::f16::from_le_bytes([block[80], block[81]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[82], block[83]]).to_f32();
    let mut out = 0;
    let mut sub = 0;
    for window in 0..2 {
        let q = &qs[window * 32..window * 32 + 32];
        for step in 0..4 {
            let shift = 2 * step;
            for half in 0..2 {
                let packed = scales[sub];
                sub += 1;
                let dl = d * f32::from(packed & 0x0f);
                let ml = dmin * f32::from(packed >> 4);
                for l in 0..16 {
                    values[out] = dl * f32::from((q[half * 16 + l] >> shift) & 3) - ml;
                    out += 1;
                }
            }
        }
    }
}

/// The sixteen six-bit scales of a `Q3_K` block, unpacked from twelve bytes.
///
/// A different splice from [`gguf_k_scale_min`], which serves `Q4_K` and
/// `Q5_K`: there the twelve bytes hold eight scales AND eight minimums, here
/// they hold sixteen scales and no minimums, because `Q3_K` is symmetric.
///
/// `ggml` writes it as four `u32` words -- the low four bits of each scale
/// come from the first eight bytes, and the top two bits from the last four,
/// two bits at a time. Kept in that form deliberately: the byte-at-a-time
/// spelling is four nested index expressions that look like an off-by-one in
/// every one of them, and this is the shape the reference can be read against.
///
/// The result is biased by 32, which the caller subtracts. Returning the raw
/// six bits would make every scale positive and the whole block wrong by a
/// factor that varies per sub-block.
fn gguf_q3_k_scales(raw: &[u8; 12]) -> [i8; 16] {
    const LOW_NIBBLES: u32 = 0x0f0f_0f0f;
    const BIT_PAIRS: u32 = 0x0303_0303;
    let word =
        |i: usize| u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]]);
    let (a, b, top) = (word(0), word(1), word(2));
    let aux = [
        (a & LOW_NIBBLES) | ((top & BIT_PAIRS) << 4),
        (b & LOW_NIBBLES) | (((top >> 2) & BIT_PAIRS) << 4),
        ((a >> 4) & LOW_NIBBLES) | (((top >> 4) & BIT_PAIRS) << 4),
        ((b >> 4) & LOW_NIBBLES) | (((top >> 6) & BIT_PAIRS) << 4),
    ];
    let mut scales = [0i8; 16];
    for (i, slot) in scales.iter_mut().enumerate() {
        *slot = aux[i / 4].to_le_bytes()[i % 4] as i8;
    }
    scales
}

/// Decode one GGUF `Q3_K` super-block: 256 elements as sixteen sub-blocks of
/// sixteen, symmetric, with each element's third bit in a separate mask.
///
/// The mask reads INVERTED, and that is the whole difficulty of this block.
/// `ggml` stores the two low bits of `q + 4` and sets the mask bit when the
/// value needed no borrow, so a SET bit subtracts nothing and a CLEAR bit
/// subtracts four. Reading it the intuitive way -- set means add -- shifts
/// every element by four and still decodes, which is why it is stated here
/// rather than left to the shape of the expression.
///
/// The mask is not advanced with the quants, either. Its 32 bytes are read
/// eight times, once per `(window, shift)` pair, taking one bit each time, so
/// the bit selector runs 1, 2, 4 … 128 ACROSS both windows while the quant
/// pointer moves. Restarting it at the second window is the mistake this
/// layout invites, and it corrupts only the upper half of the block.
///
/// Checked the same way as [`decode_gguf_q2_k_block_into`]:
/// `Llama-3.2-1B-Instruct-Q3_K_M.gguf`, `blk.0.ffn_gate.weight`, all
/// 16,777,216 elements bit-identical to the `gguf` package's `dequantize`
/// after BF16 rounding. Mean cosine 0.998 against the BF16 release -- above
/// Q2_K's 0.992, which is the ordering the two widths should produce and a
/// second reason to believe both.
fn decode_gguf_q3_k_block_into(block: &[u8; 110], values: &mut [f32; 256]) {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let raw: &[u8; 12] = block[96..108].try_into().expect("twelve scale bytes");
    let d = half::f16::from_le_bytes([block[108], block[109]]).to_f32();
    let scales = gguf_q3_k_scales(raw);
    let mut out = 0;
    let mut sub = 0;
    let mut selector = 1u8;
    for window in 0..2 {
        let q = &qs[window * 32..window * 32 + 32];
        for step in 0..4 {
            let shift = 2 * step;
            for half in 0..2 {
                let dl = d * f32::from(scales[sub] - 32);
                sub += 1;
                for l in 0..16 {
                    let at = half * 16 + l;
                    let borrow = if hmask[at] & selector == 0 { 4 } else { 0 };
                    values[out] = dl * f32::from(i16::from((q[at] >> shift) & 3) - borrow);
                    out += 1;
                }
            }
            selector <<= 1;
        }
    }
}

/// The six-bit scale and six-bit minimum for one of the eight sub-blocks of a
/// `Q4_K` or `Q5_K` block, unpacked from the twelve bytes they share.
///
/// `ggml`'s `get_scale_min_k4`. The first four sub-blocks read a whole byte
/// each from the first two groups of four; the last four are spliced, taking
/// their low four bits from the third group and their high two from the bits
/// the first four sub-blocks left unused at the top of the first two groups.
/// Twelve bytes for sixteen six-bit fields, with nothing wasted — which is
/// also why it cannot be written as a shift and a mask.
fn gguf_k_scale_min(index: usize, scales: &[u8; 12]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        let scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        let min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        (scale, min)
    }
}

/// Decode one GGUF `Q4_K` super-block: 256 elements as eight 32-element
/// sub-blocks, each with its own six-bit scale and six-bit minimum, and the
/// whole block sharing one F16 scale and one F16 minimum.
///
/// Unlike the `_0` family this is affine, not symmetric: an element is
/// `d × scaleᵢ × nibble − dmin × minᵢ`, so the minimum is *subtracted* rather
/// than folded into the quantized value. The 128 payload bytes are read in
/// pairs of sub-blocks — low nibbles for the even one, high nibbles for the
/// odd — which is why the loop steps by 64 elements and not 32.
fn decode_gguf_q4_k_block_into(block: &[u8; 144], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales: &[u8; 12] = block[4..16].try_into().expect("twelve scale bytes");
    let qs = &block[16..144];
    for pair in 0..4 {
        let (sc_lo, m_lo) = gguf_k_scale_min(pair * 2, scales);
        let (sc_hi, m_hi) = gguf_k_scale_min(pair * 2 + 1, scales);
        let (d_lo, min_lo) = (d * f32::from(sc_lo), dmin * f32::from(m_lo));
        let (d_hi, min_hi) = (d * f32::from(sc_hi), dmin * f32::from(m_hi));
        let packed = &qs[pair * 32..pair * 32 + 32];
        let out = pair * 64;
        for i in 0..32 {
            values[out + i] = d_lo * f32::from(packed[i] & 0x0f) - min_lo;
            values[out + 32 + i] = d_hi * f32::from(packed[i] >> 4) - min_hi;
        }
    }
}

/// Decode one GGUF `Q5_K` super-block: `Q4_K` plus a 32-byte plane carrying
/// each element's fifth bit.
///
/// The plane is read by sub-block pair rather than by position: pair `p` uses
/// bit `2p` of `plane[i]` for the low nibble and bit `2p + 1` for the high one.
/// So one plane byte serves all eight sub-blocks at the same offset, and the
/// fifth bit adds sixteen *before* the affine minimum is subtracted.
fn decode_gguf_q5_k_block_into(block: &[u8; 176], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales: &[u8; 12] = block[4..16].try_into().expect("twelve scale bytes");
    let plane = &block[16..48];
    let qs = &block[48..176];
    for pair in 0..4 {
        let (sc_lo, m_lo) = gguf_k_scale_min(pair * 2, scales);
        let (sc_hi, m_hi) = gguf_k_scale_min(pair * 2 + 1, scales);
        let (d_lo, min_lo) = (d * f32::from(sc_lo), dmin * f32::from(m_lo));
        let (d_hi, min_hi) = (d * f32::from(sc_hi), dmin * f32::from(m_hi));
        let packed = &qs[pair * 32..pair * 32 + 32];
        let (bit_lo, bit_hi) = (1u8 << (pair * 2), 1u8 << (pair * 2 + 1));
        let out = pair * 64;
        for i in 0..32 {
            let fifth_lo = u8::from(plane[i] & bit_lo != 0) << 4;
            let fifth_hi = u8::from(plane[i] & bit_hi != 0) << 4;
            values[out + i] = d_lo * f32::from((packed[i] & 0x0f) + fifth_lo) - min_lo;
            values[out + 32 + i] = d_hi * f32::from((packed[i] >> 4) + fifth_hi) - min_hi;
        }
    }
}

/// Decode one GGUF `Q6_K` super-block: 256 elements as sixteen 16-element
/// sub-blocks, each with a signed eight-bit scale, over one F16 scale.
///
/// Symmetric like the `_0` family — `d × scaleᵢ × (six bits − 32)`, no
/// minimum — but laid out in two halves of 128 elements, and within a half the
/// four quarters are strided rather than contiguous: quarter `q` of the half
/// takes its low four bits from `ql[l + 32·(q & 1)]` (low nibble for `q < 2`,
/// high nibble above) and its top two from bits `2q..2q+2` of `qh[l]`. The
/// sub-block scale index advances by two per quarter, so the sixteen scales
/// are consumed eight per half.
fn decode_gguf_q6_k_block_into(block: &[u8; 210], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[208], block[209]]).to_f32();
    for half_index in 0..2 {
        let ql = &block[half_index * 64..half_index * 64 + 64];
        let qh = &block[128 + half_index * 32..128 + half_index * 32 + 32];
        let scales = &block[192 + half_index * 8..192 + half_index * 8 + 8];
        let out = half_index * 128;
        for i in 0..32 {
            let sub = i / 16;
            for quarter in 0..4 {
                let nibble = if quarter < 2 {
                    ql[i + 32 * quarter] & 0x0f
                } else {
                    ql[i + 32 * (quarter - 2)] >> 4
                };
                let top = (qh[i] >> (2 * quarter)) & 3;
                let q = i32::from(nibble | (top << 4)) - 32;
                let scale = f32::from(scales[sub + 2 * quarter] as i8);
                values[out + quarter * 32 + i] = d * scale * q as f32;
            }
        }
    }
}

/// The sixteen levels an `IQ4_NL` or `IQ4_XS` code indexes, from llama.cpp's
/// `kvalues_iq4nl`.
///
/// Non-uniform on purpose, and the reason these schemes cost four bits and
/// beat four-bit uniform quantization: weights cluster near zero, so the
/// levels do too — 1, 13, 25 on the way out, then 38, 53, 69, 89, 113 as the
/// gaps widen. A uniform grid spends half its codes on a range almost nothing
/// occupies.
///
/// Compiled in rather than read from the file, exactly as llama.cpp does.
/// That is what separates IQ4 from IQ2/IQ3: those index a lattice too large
/// to write down here, and a GGUF does not ship it either, so naming them
/// would not make them decodable.
const IQ4_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Decode one GGUF `IQ4_NL` block: 32 elements as sixteen bytes of paired
/// 4-bit indices over one F16 scale.
///
/// The pairing is by half, not by neighbour: byte `j` holds element `j` in its
/// low nibble and element `j + 16` in its high one. Reading it as adjacent
/// pairs would interleave the two halves of every block.
fn decode_gguf_iq4_nl_block_into(block: &[u8; 18], values: &mut [f32; 32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..18];
    for j in 0..16 {
        values[j] = d * f32::from(IQ4_LEVELS[usize::from(qs[j] & 0x0f)]);
        values[j + 16] = d * f32::from(IQ4_LEVELS[usize::from(qs[j] >> 4)]);
    }
}

/// Decode one GGUF `IQ4_XS` super-block: 256 elements as eight 32-element
/// sub-blocks over [`decode_gguf_iq4_nl_block_into`]'s levels.
///
/// Each sub-block's scale is six bits assembled from two planes — four low
/// bits from `scales_l`, two sub-blocks to a byte, and two high bits from the
/// `scales_h` u16, eight sub-blocks to it — then read as `ls - 32`, so it is
/// signed and a scale of exactly 32 means zero. The quants are laid out as
/// eight independent `IQ4_NL` payloads of sixteen bytes each, halves and all.
fn decode_gguf_iq4_xs_block_into(block: &[u8; 136], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let qs = &block[8..136];
    for sub in 0..8 {
        let low = (scales_l[sub / 2] >> (4 * (sub % 2))) & 0x0f;
        let high = ((scales_h >> (2 * sub)) & 3) as u8;
        let ls = i32::from(low | (high << 4)) - 32;
        let dl = d * ls as f32;
        let packed = &qs[sub * 16..sub * 16 + 16];
        let out = sub * 32;
        for j in 0..16 {
            values[out + j] = dl * f32::from(IQ4_LEVELS[usize::from(packed[j] & 0x0f)]);
            values[out + j + 16] = dl * f32::from(IQ4_LEVELS[usize::from(packed[j] >> 4)]);
        }
    }
}

/// The sixteen values an E2M1 nibble stands for, doubled, from llama.cpp's
/// `kvalues_mxfp4`.
///
/// E2M1 itself codes 0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives — every
/// one a half-integer. llama.cpp stores them as the integers `0, 1, 2, 3, 4,
/// 6, 8, 12` and halves the scale instead, which keeps the table exact in
/// `i8`. [`decode_gguf_mxfp4_block_into`] applies the matching half, so the
/// product is the value E2M1 names and not twice it.
///
/// The sign is the top bit of the nibble, so the negative half repeats the
/// positive one in order rather than mirroring it.
const MXFP4_LEVELS: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// Decode one GGUF `MXFP4` block: 32 elements as one E8M0 scale byte followed
/// by sixteen bytes of paired E2M1 nibbles.
///
/// **This is not the OCP Microscaling byte layout**, which is why it has a
/// scheme of its own. OCP splits a tensor into a plane of codes and a separate
/// scale tensor; ggml interleaves one scale byte into each 32-element block,
/// making the block 17 bytes rather than 16. The numbers the two describe are
/// the same; the addresses are not.
///
/// The scale is E8M0 — a bare exponent, biased by 127 — and it is applied
/// halved, at `2^(e - 128)`, to cancel the doubling in [`MXFP4_LEVELS`]. That
/// is exact for every `e`, including the two llama.cpp writes as subnormal
/// bit patterns: `e = 0` gives `2^-128` and `e = 1` gives `2^-127`, which is
/// what `ggml_e8m0_to_fp32_half`'s `0x00200000 << x` branch computes.
///
/// The nibbles pair by half, as in [`decode_gguf_iq4_nl_block_into`]: byte `j`
/// carries element `j` low and element `j + 16` high.
fn decode_gguf_mxfp4_block_into(block: &[u8; 17], values: &mut [f32; 32]) {
    let d = mxfp4_scale(block[0]);
    let qs = &block[1..17];
    for j in 0..16 {
        values[j] = d * f32::from(MXFP4_LEVELS[usize::from(qs[j] & 0x0f)]);
        values[j + 16] = d * f32::from(MXFP4_LEVELS[usize::from(qs[j] >> 4)]);
    }
}

/// `2^(e - 128)` for an E8M0 exponent byte, formed exactly.///
/// `e - 128` reaches -128, one below the smallest normal float's exponent, so
/// two of the 256 inputs are subnormal and cannot be written as an exponent
/// field. llama.cpp spells them as the literal bit patterns `0x00200000 << e`;
/// this says the same thing as a shift of the subnormal unit, which is what
/// those patterns are.
fn mxfp4_scale(e: u8) -> f32 {
    if e < 2 {
        // 0x00200000 and 0x00400000 as floats: 2^-128 and 2^-127.
        f32::from_bits(0x0020_0000 << e)
    } else {
        f32::from_bits(u32::from(e - 1) << 23)
    }
}

/// The sign byte an IQ2/IQ3 seven-bit sign index stands for.
///
/// llama.cpp ships this as a 128-byte table, `ksigns_iq2xs`, but it is a rule
/// rather than data: the low seven bits are the index itself and the eighth is
/// whatever makes the byte's population count even. The formats spend seven
/// bits to say eight signs and recover the last one from that parity, which is
/// where a good part of IQ2's advantage over a plain two-bit quantization
/// comes from.
///
/// Stated rather than transcribed because a rule can be read, and 128 magic
/// bytes cannot.
fn iq_sign_byte(index: u8) -> u8 {
    let index = index & 0x7f;
    index | (((index.count_ones() & 1) as u8) << 7)
}

/// Applies sign bit `bit` of `signs` to `value`: set means negate.
fn iq_signed(value: f32, signs: u8, bit: usize) -> f32 {
    if (signs >> bit) & 1 == 1 {
        -value
    } else {
        value
    }
}

/// Decode one GGUF `IQ2_XXS` block: 256 elements in 66 bytes.
///
/// `d: f16` then sixteen `u32`, read in pairs. Each pair covers 32 elements:
/// the first holds four bytes, each a point in the 256-entry
/// [`IQ2XXS_GRID`](iq_grid::IQ2XXS_GRID), and the second packs four seven-bit
/// sign indices at bit offsets 0, 7, 14 and 21 with a four-bit scale in its top
/// nibble.
///
/// The scale is `(0.5 + s) * 0.25`, so it is never zero — a sub-block cannot
/// be switched off, only made small. That half-step is why the four bits reach
/// 3.875 rather than 3.75, and dropping it decodes every weight about 12% too
/// small at the low end.
fn decode_gguf_iq2_xxs_block_into(block: &[u8; 66], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for group in 0..8 {
        let at = 2 + group * 8;
        let points = u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
        let aux = u32::from_le_bytes([block[at + 4], block[at + 5], block[at + 6], block[at + 7]]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.25;
        for sub in 0..4 {
            let point = ((points >> (8 * sub)) & 0xff) as usize;
            let signs = iq_sign_byte(((aux >> (7 * sub)) & 0x7f) as u8);
            let out = group * 32 + sub * 8;
            for bit in 0..8 {
                let g = f32::from(iq_grid::IQ2XXS_GRID[point * 8 + bit]);
                values[out + bit] = iq_signed(db * g, signs, bit);
            }
        }
    }
}

/// Decode one GGUF `IQ2_XS` block: 256 elements in 74 bytes.
///
/// `d: f16`, then 32 `u16`, then eight scale bytes. Each `u16` is one grid
/// point of eight elements: its low **nine** bits address the 512-entry
/// [`IQ2XS_GRID`](iq_grid::IQ2XS_GRID) and its top seven are the sign index.
/// The scale bytes hold two four-bit scales each, one per 16 elements, applied
/// as `(0.5 + s) * 0.25`.
///
/// The nine-bit split is the difference from `IQ2_XXS`, whose grid is 256
/// points and whose sign index therefore starts a bit earlier. Masking with
/// 0xFF here would address the right grid a quarter of the time.
fn decode_gguf_iq2_xs_block_into(block: &[u8; 74], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scales = &block[66..74];
    for k in 0..32 {
        let q = u16::from_le_bytes([block[2 + k * 2], block[3 + k * 2]]);
        let sub = k / 2;
        let s = (scales[sub / 2] >> (4 * (sub % 2))) & 0x0f;
        let db = d * (0.5 + f32::from(s)) * 0.25;
        let point = usize::from(q & 511);
        let signs = iq_sign_byte((q >> 9) as u8);
        for bit in 0..8 {
            let g = f32::from(iq_grid::IQ2XS_GRID[point * 8 + bit]);
            values[k * 8 + bit] = iq_signed(db * g, signs, bit);
        }
    }
}

/// Decode one GGUF `IQ2_S` block: 256 elements in 82 bytes.
///
/// `d: f16`, 32 quant bytes, 32 sign bytes, eight high-bit bytes, eight scale
/// bytes. The grid is 1024 points, so a point needs ten bits: eight from `qs`
/// and two more from `qh`, four points to a `qh` byte.
///
/// Unlike `IQ2_XXS` and `IQ2_XS` the signs are **stored outright**, one byte of
/// eight bits per grid point, not as a seven-bit index through
/// [`iq_sign_byte`]. That is what the extra eight bytes over `IQ2_XS` buy,
/// along with the wider grid: the parity trick is dropped once there is room
/// for the eighth bit.
fn decode_gguf_iq2_s_block_into(block: &[u8; 82], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..34];
    let signs = &block[34..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    for k in 0..32 {
        let high = (qh[k / 4] >> (2 * (k % 4))) & 3;
        let point = usize::from(qs[k]) | (usize::from(high) << 8);
        let sub = k / 2;
        let s = (scales[sub / 2] >> (4 * (sub % 2))) & 0x0f;
        let db = d * (0.5 + f32::from(s)) * 0.25;
        for bit in 0..8 {
            let g = f32::from(iq_grid::IQ2S_GRID[point * 8 + bit]);
            values[k * 8 + bit] = iq_signed(db * g, signs[k], bit);
        }
    }
}

/// Decode one GGUF `IQ3_XXS` block: 256 elements in 98 bytes.
///
/// `d: f16`, 64 quant bytes, then eight `u32`. The grid points are **four**
/// components, not eight, so a byte of `qs` covers four elements and the
/// 64 bytes cover all 256.
///
/// The eight trailing `u32` are shaped exactly like `IQ2_XXS`'s odd words —
/// four seven-bit sign indices and a four-bit scale on top — and each covers 32
/// elements. The scale is `(0.5 + s) * 0.5`, twice `IQ2_XXS`'s factor, because
/// the grid values are larger: `IQ3_XXS`'s components run to 62 where
/// `IQ2_XXS`'s stop at 43.
fn decode_gguf_iq3_xxs_block_into(block: &[u8; 98], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..66];
    for group in 0..8 {
        let at = 66 + group * 4;
        let aux = u32::from_le_bytes([block[at], block[at + 1], block[at + 2], block[at + 3]]);
        let db = d * (0.5 + (aux >> 28) as f32) * 0.5;
        for sub in 0..4 {
            let signs = iq_sign_byte(((aux >> (7 * sub)) & 0x7f) as u8);
            let out = group * 32 + sub * 8;
            // Eight elements is two grid points here, where the IQ2 schemes
            // get eight from one.
            for bit in 0..8 {
                let point = usize::from(qs[(out + bit) / 4]);
                let g = f32::from(iq_grid::IQ3XXS_GRID[point * 4 + (out + bit) % 4]);
                values[out + bit] = iq_signed(db * g, signs, bit);
            }
        }
    }
}

/// Decode one GGUF `IQ3_S` block: 256 elements in 110 bytes.
///
/// `d: f16`, 64 quant bytes, eight high-bit bytes, 32 sign bytes, four scale
/// bytes. The grid is 512 four-component points, so a point takes nine bits:
/// eight from `qs` and one from `qh`, **eight points to a `qh` byte** — one bit
/// each, not the two-bit fields `IQ2_S` uses.
///
/// The scale is `1 + 2s`, not `(0.5 + s) * k`: it is an odd integer, and the
/// grid components are the odd numbers 1 through 15. `IQ3_S` is the one scheme
/// here whose scale has no fractional part, which is why sharing a scale
/// helper with the others would be wrong rather than merely awkward.
fn decode_gguf_iq3_s_block_into(block: &[u8; 110], values: &mut [f32; 256]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    for i in 0..256 {
        let sub = i / 32;
        let s = (scales[sub / 2] >> (4 * (sub % 2))) & 0x0f;
        let db = d * (1.0 + 2.0 * f32::from(s));
        let p = i / 4;
        let point = usize::from(qs[p]) | (usize::from((qh[p / 8] >> (p % 8)) & 1) << 8);
        let g = f32::from(iq_grid::IQ3S_GRID[point * 4 + i % 4]);
        values[i] = iq_signed(db * g, signs[i / 8], i % 8);
    }
}

/// Decode one block of any GGUF scheme the loader knows, into the `f32` values
/// it stands for.
///
/// `block` and `values` are exactly the lengths `scheme.block_layout()` names;
/// the conversions below are that promise restated, and a caller that breaks it
/// panics here rather than reading a neighbouring block.
fn decode_gguf_block_into(scheme: QuantScheme, block: &[u8], values: &mut [f32]) {
    let bad = "block and value lengths must match the scheme's layout";
    match scheme {
        QuantScheme::GgufQ4_0 => {
            decode_gguf_q4_0_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ4_1 => {
            decode_gguf_q4_1_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ5_0 => {
            decode_gguf_q5_0_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ5_1 => {
            decode_gguf_q5_1_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ8_0 => {
            decode_gguf_q8_0_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ2K => {
            decode_gguf_q2_k_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ3K => {
            decode_gguf_q3_k_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ4K => {
            decode_gguf_q4_k_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ5K => {
            decode_gguf_q5_k_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufQ6K => {
            decode_gguf_q6_k_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufIq4Nl => {
            decode_gguf_iq4_nl_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufIq4Xs => {
            decode_gguf_iq4_xs_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufMxfp4 => {
            decode_gguf_mxfp4_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufIq2Xxs => {
            decode_gguf_iq2_xxs_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufIq2Xs => {
            decode_gguf_iq2_xs_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufIq2S => {
            decode_gguf_iq2_s_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufIq3Xxs => {
            decode_gguf_iq3_xxs_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        QuantScheme::GgufIq3S => {
            decode_gguf_iq3_s_block_into(
                block.try_into().expect(bad),
                values.try_into().expect(bad),
            );
        }
        other => unreachable!("{other:?} reports no GGUF block layout"),
    }
}

impl Walk<'_, '_> {
    fn execute(&mut self) -> Result<(), Error> {
        for (position, id) in self.plan.schedule.iter().enumerate() {
            let instr = instr_by_id(&self.plan.instrs, *id)?.clone();
            // Accounted before the match consumes the instruction, reported
            // after its work is done.
            let consumed = match &instr {
                StorageInstr::ExtentWrite { source, .. }
                | StorageInstr::BulkExtentWrite { source, .. }
                | StorageInstr::GatherWrite { source, .. } => source.span_bytes,
                StorageInstr::TileMap { source, .. } => {
                    source.as_ref().map_or(0, |source| source.span_bytes)
                }
                _ => 0,
            };
            let finalized = match &instr {
                StorageInstr::Finalize { name, .. } => Some(name.clone()),
                _ => None,
            };
            match instr {
                StorageInstr::Allocate { buffer, .. } => self.allocate(buffer)?,
                StorageInstr::Fill { buffer, .. } => self.fill(buffer)?,
                StorageInstr::ExtentWrite { source, dest, .. } => {
                    let bytes = self.read_extent(&source)?;
                    self.write_extent(&dest, &bytes, &source.stride)?;
                }
                StorageInstr::BulkExtentWrite {
                    source,
                    dest_offset,
                    ..
                } => {
                    if self.stream {
                        return Err(invalid(
                            "streaming execution has no persistent arena for this \
                             BulkExtentWrite to target; compile the plan with \
                             plan::compile_streaming, or give the execution an \
                             arena",
                        ));
                    }
                    let bytes = self.read_extent(&source)?;
                    self.write_arena(dest_offset, &bytes)?;
                }
                StorageInstr::GatherWrite {
                    source,
                    dest,
                    gather,
                    ..
                } => {
                    let bytes = self.read_extent(&source)?;
                    let permuted = permute(&bytes, &gather)?;
                    self.write_extent(&dest, &permuted, &Extent::byte_run(permuted.len() as u64))?;
                }
                StorageInstr::TileMap { .. } => self.tile_map(&instr)?,
                StorageInstr::CreateView {
                    input,
                    output,
                    view,
                    ..
                } => {
                    let len = extent_bytes(&view.stride)?;
                    let offset = checked_usize(view.offset)?
                        .checked_add(checked_usize(view.stride.base_offset)?)
                        .ok_or_else(|| invalid("view offset overflow"))?;
                    self.resolve(input, offset, len)?;
                    self.buffers
                        .insert(output, BufferLoc::View { input, offset, len });
                }
                StorageInstr::Finalize { tensor, name, .. } => {
                    let bytes = self.buffer_bytes(tensor)?.to_vec();
                    if !self.finalized.insert(name.clone()) {
                        return Err(invalid(format!("tensor '{name}' was finalized twice")));
                    }
                    self.sink.publish(&name, &bytes)?;
                }
            }
            if self.stream {
                // Everything whose last reference just executed is dead;
                // dropping it here is what makes peak memory the working set.
                self.buffers
                    .retain(|buffer, _| self.last_use.get(buffer) != Some(&position));
            }
            self.read_bytes += consumed;
            (self.progress)(Progress {
                read_bytes: self.read_bytes,
                total_read_bytes: self.plan.memory.checkpoint_read_bytes,
                finalized: finalized.as_deref(),
            });
        }
        Ok(())
    }

    fn allocate(&mut self, id: BufferId) -> Result<(), Error> {
        let decl = self.plan.buffer(id)?;
        let len = checked_usize(decl.bytes)?;
        let loc = if !self.stream
            && let Some(offset) = decl.arena_offset()
        {
            let offset = checked_usize(offset)?;
            let end = offset
                .checked_add(len)
                .ok_or_else(|| invalid("persistent buffer range overflow"))?;
            if end > ArenaBacking::len(self.arena) {
                return Err(invalid(format!("persistent buffer {} exceeds arena", id.0)));
            }
            BufferLoc::Arena { offset, len }
        } else {
            BufferLoc::Owned(vec![POISON; len])
        };
        if self.buffers.insert(id, loc).is_some() {
            return Err(invalid(format!("buffer {} was allocated twice", id.0)));
        }
        Ok(())
    }

    /// Zero a buffer. An `Owned` buffer is already zero at allocation, but a
    /// persistent one is a window into an arena that may hold anything.
    fn fill(&mut self, id: BufferId) -> Result<(), Error> {
        let (root, offset, len) = self.resolve(id, 0, usize::MAX)?;
        match root {
            Root::Arena => self.arena.fill(offset, len, 0)?,
            Root::Owned(owner) => match self.buffers.get_mut(&owner) {
                Some(BufferLoc::Owned(bytes)) => bytes[offset..offset + len].fill(0),
                _ => return Err(invalid(format!("buffer {} is not writable", id.0))),
            },
        }
        Ok(())
    }

    fn read_extent(&self, source: &SourceExtent) -> Result<Vec<u8>, Error> {
        let mut normalized = source.stride.clone();
        let base_offset = normalized.base_offset;
        normalized.base_offset = 0;
        let physical = physical_source_bytes(&normalized)?;
        let raw = self.read_file(
            source.file_id.0,
            source
                .file_offset
                .checked_add(base_offset)
                .ok_or_else(|| invalid("source base offset overflow"))?,
            physical,
            self.plan.target.max_tile_bytes,
        )?;
        gather_strided(raw, &normalized)
    }

    fn read_file(
        &self,
        file_id: u32,
        offset: u64,
        len: u64,
        tile_bound: u64,
    ) -> Result<Vec<u8>, Error> {
        let path = self
            .files
            .get(&file_id)
            .ok_or_else(|| invalid(format!("plan references unknown file id {file_id}")))?;
        let len = checked_usize(len)?;
        let mut out = vec![0u8; len];
        let mut file =
            File::open(path).map_err(|err| invalid(format!("open {}: {err}", path.display())))?;

        // A large read is split across threads on positioned reads — the file
        // handle is shared, only the offsets differ, and byte `i` lands at
        // `out[i]` either way, so the result is the serial read's. What it
        // buys is page-cache bandwidth: one thread memcpying out of the cache
        // was the second-largest serial cost of a conversion after the encode
        // itself.
        #[cfg(unix)]
        {
            const PARALLEL_READ_MIN: usize = 64 << 20;
            let workers = std::thread::available_parallelism()
                .map_or(1, std::num::NonZero::get)
                .min(16);
            if len >= PARALLEL_READ_MIN && workers > 1 {
                use std::os::unix::fs::FileExt;
                let chunk = len.div_ceil(workers);
                let failures: Vec<std::io::Result<()>> = std::thread::scope(|scope| {
                    out.chunks_mut(chunk)
                        .enumerate()
                        .map(|(at, buf)| {
                            let file = &file;
                            scope.spawn(move || {
                                file.read_exact_at(buf, offset + (at * chunk) as u64)
                            })
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                        .map(|worker| worker.join().expect("a read worker does not panic"))
                        .collect()
                });
                for failure in failures {
                    failure.map_err(|err| invalid(format!("read {}: {err}", path.display())))?;
                }
                return Ok(out);
            }
        }

        file.seek(SeekFrom::Start(offset))
            .map_err(|err| invalid(format!("seek {}: {err}", path.display())))?;
        let tile = if tile_bound == 0 {
            len.max(1)
        } else {
            checked_usize(tile_bound)?.max(1)
        };
        for chunk in out.chunks_mut(tile) {
            file.read_exact(chunk)
                .map_err(|err| invalid(format!("read {}: {err}", path.display())))?;
        }
        Ok(out)
    }

    fn write_extent(
        &mut self,
        dest: &DestExtent,
        compact: &[u8],
        source_stride: &Extent,
    ) -> Result<(), Error> {
        require_same_byte_count(source_stride, &dest.stride)?;
        if !dest.stride.has_dense_destination() {
            return Err(invalid(
                "non-compact ExtentWrite destinations are unsupported",
            ));
        }
        let base = checked_usize(dest.offset)?
            .checked_add(checked_usize(dest.stride.base_offset)?)
            .ok_or_else(|| invalid("destination offset overflow"))?;
        self.write_buffer(dest.buffer, base, compact)
    }

    fn write_arena(&mut self, offset: u64, bytes: &[u8]) -> Result<(), Error> {
        let offset = checked_usize(offset)?;
        let end = offset
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("arena write range overflow"))?;
        debug_assert!(end <= ArenaBacking::len(self.arena));
        self.arena.write(offset, bytes)
    }

    fn tile_map(&mut self, instr: &StorageInstr) -> Result<(), Error> {
        let StorageInstr::TileMap {
            kind,
            source,
            dest,
            inputs,
            outputs,
            tile,
            transform,
            ..
        } = instr
        else {
            return Err(invalid("tile_map was handed something else"));
        };
        let (kind, max_tile_bytes) = (*kind, tile.max_tile_bytes);
        let (source, dest) = (source.as_ref(), dest.as_ref());
        let transform = transform.clone();
        // THE DEVICE PATH, tried before a byte is staged to the host.
        //
        // Only when the backing claimed this kind AND every operand is
        // already a span of the arena — which is exactly the case that costs
        // a round trip on the host path, because `buffer_bytes` on an arena
        // buffer is a device read that synchronizes. An operand the executor
        // would have had to read from a FILE is not eligible: those bytes are
        // on the host either way, so transforming them here saves nothing and
        // the host path is the honest one.
        //
        // One gate, not two. The plan named a kernel per instruction or named
        // none, and `arena_tile_map_op` reads that answer; a second per-KIND
        // mask beside it could only ever be wider than the plan's own reply.
        // What a backing may NOT do is fail quietly: an op it is offered is
        // one the compiler decided it can run, so it runs it or it errors.
        if self.arena_runs_kernels
            && let Some(op) =
                self.arena_tile_map_op(kind, source, dest, inputs, outputs, &transform)?
        {
            self.arena.run_tile_map(&op)?;
            return Ok(());
        }
        let input = if let Some(source) = source {
            self.read_extent(source)?
        } else {
            let input = inputs
                .first()
                .ok_or_else(|| invalid("TileMap has no source or input buffer"))?;
            self.buffer_bytes(*input)?.to_vec()
        };
        let output = match kind {
            TileMapKind::Reblock => input,
            TileMapKind::Cast => self.cast_bytes(
                source,
                inputs.first().copied(),
                outputs.first().copied(),
                &input,
            )?,
            TileMapKind::Scale => {
                self.scale_bytes(source, inputs, outputs.first().copied(), &input, &transform)?
            }
            TileMapKind::Bias => {
                self.bias_bytes(source, inputs, outputs.first().copied(), &input, &transform)?
            }
            TileMapKind::Decode => {
                self.decode_bytes(outputs.first().copied(), &input, &transform)?
            }
            // **THE ONLY EXECUTOR A REPACK HAS.** No device mask carries the
            // bit, so this arm is not the host fallback the ones above it
            // are — it is where the transform runs, at `pie model import`,
            // once per plane.
            TileMapKind::Repack => self.repack_bytes(&input, &transform)?,
            // Encode is the one transform with more than one output — the
            // payload and the scale (and zero-point) tensors it cannot be
            // read without — so both arms write their own buffers and return
            // rather than flowing into the single-output plumbing below. The
            // MLX affine scheme keeps its own arm: it publishes three tensors
            // where every other scheme publishes two.
            TileMapKind::Encode if transform.to == Some(QuantScheme::MlxAffineU4) => {
                let written =
                    self.encode_mlx_affine_u4(&input, source, inputs, outputs, &transform)?;
                for (buffer, bytes) in written {
                    self.write_buffer(buffer, 0, &bytes)?;
                }
                return Ok(());
            }
            TileMapKind::Encode => {
                return self.encode_bytes(
                    source,
                    inputs,
                    outputs,
                    &input,
                    &transform,
                    max_tile_bytes,
                );
            }
            other => {
                return Err(invalid(format!(
                    "host storage executor does not implement {other:?} transforms"
                )));
            }
        };
        let tile = if max_tile_bytes == 0 {
            output.len().max(1)
        } else {
            checked_usize(max_tile_bytes)?.max(1)
        };
        if let Some(dest) = dest {
            let source_stride = source.map(|source| &source.stride).unwrap_or(&dest.stride);
            // A per-group `Scale` and a `Decode` are the transforms whose
            // output is a different width from their input by design:
            // unpacking four-bit codes into `BF16` quadruples the bytes, and a
            // GGUF block trades 18 bytes for 64. A `Cast` preserves the
            // *element count* while the width follows the representation —
            // F16→BF16 happened to keep the bytes equal, which is how the
            // byte check survived until an F32→BF16 plan first ran here.
            // Every other kind moves the same bytes it read, and the
            // mismatch is a bug worth catching.
            if kind == TileMapKind::Cast {
                require_same_element_count(source_stride, &dest.stride)?;
            } else if transform.scale_blocks.is_empty()
                && kind != TileMapKind::Decode
                && kind != TileMapKind::Repack
            {
                require_same_byte_count(source_stride, &dest.stride)?;
            }
            if !dest.stride.has_dense_destination() {
                return Err(invalid("non-compact TileMap destinations are unsupported"));
            }
            let base = checked_usize(dest.offset)?
                .checked_add(checked_usize(dest.stride.base_offset)?)
                .ok_or_else(|| invalid("destination offset overflow"))?;
            for (offset, chunk) in output.chunks(tile).enumerate() {
                self.write_buffer(dest.buffer, base + offset * tile, chunk)?;
            }
            return Ok(());
        }
        let output_id = outputs
            .first()
            .copied()
            .ok_or_else(|| invalid("TileMap has no output buffer"))?;
        if output.len() != self.buffer_bytes(output_id)?.len() {
            return Err(invalid(format!(
                "TileMap produced {} bytes for {}-byte output buffer",
                output.len(),
                self.buffer_bytes(output_id)?.len()
            )));
        }
        for (offset, chunk) in output.chunks(tile).enumerate() {
            self.write_buffer(output_id, offset * tile, chunk)?;
        }
        Ok(())
    }

    /// The operands of a `TileMap`, as arena spans — or `None` when this one
    /// is not the device path's business.
    ///
    /// `None` is the ordinary answer and is never an error: the plan named no
    /// kernel for this instruction, or a transform reading a checkpoint
    /// extent has its bytes on the host already, or one writing an owned
    /// buffer has no arena address to write to. All run on the host path
    /// exactly as before. Only a transform the compiler chose a row for, whose
    /// input AND every output are resident in the arena, is handed over —
    /// which is the case the host path pays a round trip for.
    ///
    /// The kernel question is asked HERE rather than inside the backing,
    /// because it is the plan's answer and reading it is not a decision. A
    /// backing that had to unwrap an `Option<&str>` would be a backing that
    /// could reply "the plan named nothing", which the executor is holding
    /// the plan and can see.
    fn arena_tile_map_op<'t>(
        &self,
        kind: TileMapKind,
        source: Option<&SourceExtent>,
        dest: Option<&DestExtent>,
        inputs: &[BufferId],
        outputs: &[BufferId],
        transform: &'t TransformSpec,
    ) -> Result<Option<TileMapOp<'t>>, Error> {
        // No row, no delegation: the plan is saying the host runs this one.
        let Some(kernel) = transform.kernel.as_deref() else {
            return Ok(None);
        };
        // A checkpoint source means host bytes; nothing to delegate.
        if source.is_some() {
            return Ok(None);
        }
        let Some(&input) = inputs.first() else {
            return Ok(None);
        };
        let Some(src) = self.arena_span(input)? else {
            return Ok(None);
        };
        // `Encode` publishes a payload AND the scales it cannot be read
        // without — the same `&[payload, scales]` the host path destructures.
        // Both have to be in the arena or neither is delegable.
        let dst_scales = if kind == TileMapKind::Encode {
            match outputs.get(1) {
                Some(&scales) => match self.arena_span(scales)? {
                    Some(span) => Some(span),
                    None => return Ok(None),
                },
                // Two outputs is what `Encode` means; one is a plan the host
                // path rejects with a better message than this one could.
                None => return Ok(None),
            }
        } else {
            None
        };
        // The destination is either an extent naming a buffer, or the first
        // output buffer whole. Both must land in the arena.
        let (dst, dst_buffer) = match dest {
            Some(dest) => {
                if !dest.stride.has_dense_destination() {
                    return Ok(None);
                }
                let Some(whole) = self.arena_span(dest.buffer)? else {
                    return Ok(None);
                };
                let base = checked_usize(dest.offset)?
                    .checked_add(checked_usize(dest.stride.base_offset)?)
                    .ok_or_else(|| invalid("destination offset overflow"))?;
                let len = extent_bytes(&dest.stride)?;
                let end = base
                    .checked_add(len)
                    .ok_or_else(|| invalid("destination range overflow"))?;
                if end > whole.len {
                    return Err(invalid("TileMap destination leaves its buffer"));
                }
                let offset = whole
                    .offset
                    .checked_add(base)
                    .ok_or_else(|| invalid("destination offset overflow"))?;
                (ArenaSpan { offset, len }, dest.buffer)
            }
            None => {
                let Some(&output) = outputs.first() else {
                    return Ok(None);
                };
                let Some(span) = self.arena_span(output)? else {
                    return Ok(None);
                };
                (span, output)
            }
        };
        // A blocked `Scale` reads its factors from a second input; a uniform
        // one carries them in `scale_factor_bits`. An unfindable factor
        // operand is not an error here — it is a plan the host path knows how
        // to run and this one does not.
        let factors = if transform.scale_blocks.is_empty() {
            None
        } else {
            match inputs.get(1) {
                Some(&factor_buffer) => match self.arena_span(factor_buffer)? {
                    Some(span) => Some(span),
                    None => return Ok(None),
                },
                None => return Ok(None),
            }
        };
        Ok(Some(TileMapOp {
            kernel,
            src,
            dst,
            dst_scales,
            factors,
            shape: self.buffer_rectangle(dst_buffer),
        }))
    }

    /// Where `id` lives in the arena, or `None` if it is a host-owned buffer.
    fn arena_span(&self, id: BufferId) -> Result<Option<ArenaSpan>, Error> {
        let (root, offset, len) = self.resolve(id, 0, usize::MAX)?;
        Ok(match root {
            Root::Arena => Some(ArenaSpan { offset, len }),
            Root::Owned(_) => None,
        })
    }

    /// A buffer's declared shape as the rectangle a transform walks.
    ///
    /// The same source `host::encode_bytes` reads — the buffer's own type, not
    /// an extent's dims — and folded by the same rule
    /// ([`crate::types::rectangle`]), so a backing and the host path launch
    /// over one number rather than two that can disagree. It answered `None`
    /// for any rank but 2, which sent a rank-3 bank's transform to a device
    /// backing with no extent at all.
    fn buffer_rectangle(&self, id: BufferId) -> Option<(u32, u32)> {
        let (rows, cols) = crate::types::rectangle(&self.plan.buffer(id).ok()?.ty.shape)?;
        Some((u32::try_from(rows).ok()?, u32::try_from(cols).ok()?))
    }

    /// Add one constant to every element, in the operand's own dtype.
    ///
    /// The arithmetic is done in `f32` and written back at the source width,
    /// which for a BF16 operand means the sum is rounded once -- the same
    /// single rounding a scale takes, and the reason both are stated as one
    /// kernel rather than as a host loop somebody writes twice.
    ///
    /// The operand is always a plain float buffer — a quantized operand was
    /// refused before a plan existed, at both ranks: the per-block `Scale`
    /// before a per-block `Bias` is what turned codes into numbers.
    /// **THE REPACK, ON THE HOST** — the permutation
    /// `kernels_cuda::linear::tiled`'s two relabelling passes write, computed
    /// here because this is where a repack runs.
    ///
    /// The invariant every lowerable transform is held to is that it has a
    /// host implementation (`plan::passes::tile`'s
    /// `every_transform_a_backend_may_lower_has_a_host_implementation`), and
    /// for this one the host is not a fallback: no device mask carries
    /// `Repack`, so this is the ONLY executor it has. `pie model import`
    /// compiles against `CONVERT_TILE_MAP_MASK`, reaches here once per plane,
    /// and the artifact it writes holds the repacked bytes.
    ///
    /// **IT IS A GATHER AND NOTHING ELSE.** No arithmetic touches a code or a
    /// factor: an output word is eight input nibbles shifted into place, and
    /// an output factor is one input factor moved. That is what makes it
    /// legal under serve-as-stored — the served row is the stored row — and
    /// it is what lets the golden be an exact round trip rather than a
    /// tolerance.
    ///
    /// The two marlin layouts are NOT implemented here and say so: they are
    /// the `native_mxfp4_moe` device path's, no plan in this tree lowers one
    /// to the host, and a permutation written from a banner nobody runs is a
    /// second answer waiting to disagree with the first.
    fn repack_bytes(&self, bytes: &[u8], transform: &TransformSpec) -> Result<Vec<u8>, Error> {
        let spec = transform
            .repack
            .ok_or_else(|| invalid("a Repack instruction carries no layout"))?;
        let (rows, cols) = (spec.source_rows as usize, spec.source_cols as usize);
        let (target_rows, target_cols) = (spec.target_rows as usize, spec.target_cols as usize);
        match spec.layout {
            RepackLayout::TiledAffineU4Weight => {
                if target_cols != cols {
                    return Err(invalid(format!(
                        "TiledAffineU4Weight Repack pads columns ({cols} -> {target_cols}), \
                         and the fragment map has no column padding"
                    )));
                }
                let row_bytes = cols / 2;
                if bytes.len() < rows * row_bytes {
                    return Err(invalid(format!(
                        "TiledAffineU4Weight Repack reads {rows} rows of {row_bytes} bytes \
                         from a {}-byte operand",
                        bytes.len()
                    )));
                }
                // Word `lane` of tile `(band, k tile)` holds, at nibble
                // `s + 4h`, the code at `k = 16*kt + 2*(lane%4) + 8*(s&1) + h`
                // and `n = 16*band + lane/4 + 8*(s>=2)`; four k tiles are
                // grouped as one lane's `uint4`, so the word order is
                // `[band][k quad][lane][4]`. This is `tiled.cuh`'s
                // `repack_affine_tiled` read off its banner, which is where
                // the golden's un-repack is read from too.
                let band = TILED_BAND as usize;
                let quad = (TILED_STEP / TILED_BAND) as usize;
                let bands = target_rows / band;
                let k_tiles = cols / band;
                let quads = k_tiles / quad;
                let mut out = vec![0u8; target_rows * row_bytes];
                let mut at = 0usize;
                for b in 0..bands {
                    for kq in 0..quads {
                        for lane in 0..32usize {
                            for word in 0..quad {
                                let kt = kq * quad + word;
                                let col_of = lane / 4;
                                let k_base = kt * band + 2 * (lane % 4);
                                let mut res = 0u32;
                                for s in 0..4usize {
                                    let col = b * band + col_of + usize::from(s >= 2) * 8;
                                    let k_off = usize::from(s % 2 == 1) * 8;
                                    for h in 0..2usize {
                                        if col >= rows {
                                            continue;
                                        }
                                        let kk = k_base + k_off + h;
                                        let byte = bytes[col * row_bytes + kk / 2];
                                        let code = if kk % 2 == 0 {
                                            u32::from(byte & 0xF)
                                        } else {
                                            u32::from(byte >> 4)
                                        };
                                        res |= code << (4 * (s + 4 * h));
                                    }
                                }
                                out[at * 4..at * 4 + 4].copy_from_slice(&res.to_le_bytes());
                                at += 1;
                            }
                        }
                    }
                }
                Ok(out)
            }
            RepackLayout::TiledAffineFactor => {
                if target_cols != cols {
                    return Err(invalid(format!(
                        "TiledAffineFactor Repack pads groups ({cols} -> {target_cols}), \
                         and a band's padding is rows and not groups"
                    )));
                }
                // `[n][group]` becomes `[n band][group][16]`, which is a
                // transpose of the (column, group) rectangle inside each band
                // — the sixteen columns of a band adjacent, so a lane's two
                // (eight apart) are one short run. A band's columns past `n`
                // are a zero factor, which with the zero code above makes the
                // padded weight exactly zero.
                const FACTOR: usize = 2;
                if bytes.len() < rows * cols * FACTOR {
                    return Err(invalid(format!(
                        "TiledAffineFactor Repack reads {rows} rows of {cols} two-byte \
                         factors from a {}-byte operand",
                        bytes.len()
                    )));
                }
                let band = TILED_BAND as usize;
                let mut out = vec![0u8; target_rows * cols * FACTOR];
                for (at, slot) in out.chunks_exact_mut(FACTOR).enumerate() {
                    let j = at % band;
                    let rest = at / band;
                    let g = rest % cols;
                    let row = rest / cols * band + j;
                    if row < rows {
                        let from = (row * cols + g) * FACTOR;
                        slot.copy_from_slice(&bytes[from..from + FACTOR]);
                    }
                }
                Ok(out)
            }
            other => Err(invalid(format!(
                "host storage executor does not implement the {other:?} Repack layout; it \
                 is the mxfp4 MoE device path's, and no plan in this tree lowers one here"
            ))),
        }
    }

    fn bias_bytes(
        &self,
        source: Option<&SourceExtent>,
        inputs: &[BufferId],
        output: Option<BufferId>,
        bytes: &[u8],
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        let dtype = if let Some(source) = source {
            self.source_dtype(source.tensor_id)?
        } else if let Some(input) = inputs.first() {
            self.buffer_dtype(*input)?
        } else {
            return Err(invalid("host Bias requires a source or input buffer"));
        };
        if !transform.scale_blocks.is_empty() {
            let output =
                output.ok_or_else(|| invalid("per-block Bias requires an output buffer"))?;
            let values = decode_values(bytes, dtype)?;
            return self.fold_per_block(values, inputs, output, transform, "Bias", |v, f| v + f);
        }
        let by = f32::from_bits(transform.bias_bits);
        let mut values = decode_values(bytes, dtype)?;
        for value in &mut values {
            *value = f64::from(*value as f32 + by);
        }
        encode_values(&values, dtype)
    }

    /// `Scale` multiplies, and what it multiplies by is the only thing that
    /// varies: a constant every element shares, or one factor per group of
    /// elements read from a second operand.
    ///
    /// The uniform form keeps the type it was handed — `infer` gives it back
    /// unchanged — so there is no output dtype to look up. The per-group form
    /// is the one that also *decodes*, because a quantized tensor's elements
    /// are only numbers once their factors are applied; its output dtype is
    /// therefore the one the output buffer declares.
    ///
    /// The multiply happens in `f32`, not in the `f64` `decode_values` yields,
    /// because that is what the CUDA kernel does and the two executors are
    /// compared bit for bit. Every input dtype either form accepts is exactly
    /// representable in `f32`, so the narrowing before the multiply cannot
    /// lose anything the device would have kept.
    fn scale_bytes(
        &self,
        source: Option<&SourceExtent>,
        inputs: &[BufferId],
        output: Option<BufferId>,
        bytes: &[u8],
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        let payload = || -> Result<DType, Error> {
            if let Some(source) = source {
                self.source_dtype(source.tensor_id)
            } else if let Some(input) = inputs.first() {
                self.buffer_dtype(*input)
            } else {
                Err(invalid("host Scale requires a source or input buffer"))
            }
        };
        if !transform.scale_blocks.is_empty() {
            let output =
                output.ok_or_else(|| invalid("per-block Scale requires an output buffer"))?;
            let elements = match transform.from {
                None => decode_values(bytes, payload()?)?,
                Some(QuantScheme::Mxfp4E2M1E8M0) => decode_mxfp4_elements(bytes),
                Some(QuantScheme::Int4B8) => decode_int4b8_elements(bytes),
                Some(QuantScheme::Fp8E4M3) => decode_fp8_e4m3_elements(bytes),
                // Both affine widths land here, and the payload does not say
                // which this is: the element count does. A whole number of
                // codes fills the bytes at exactly one of the two widths.
                Some(QuantScheme::MlxAffineU4) => {
                    let total: i64 = self.buffer_shape(output)?.iter().product();
                    let total = usize::try_from(total)
                        .map_err(|_| invalid("per-block Scale output has a negative extent"))?;
                    let bits = match (bytes.len() * 2 == total, bytes.len() == total) {
                        (true, false) => 4,
                        (false, true) => 8,
                        _ => {
                            return Err(invalid(format!(
                                "{} bytes of MLX affine codes fill {total} elements at \
                                 neither four nor eight bits",
                                bytes.len()
                            )));
                        }
                    };
                    decode_mlx_affine_codes(bytes, bits)
                }
                Some(other) => {
                    return Err(invalid(format!(
                        "host Scale does not implement {other:?} elements"
                    )));
                }
            };
            return self.scale_per_block(elements, inputs, output, transform);
        }
        let dtype = payload()?;
        let factor = f32::from_bits(transform.scale_factor_bits);
        let mut values = decode_values(bytes, dtype)?;
        for value in &mut values {
            *value = f64::from(*value as f32 * factor);
        }
        encode_values(&values, dtype)
    }

    /// The MLX affine-U4 encode: the one scheme that publishes *three*
    /// tensors — a quantized weight is unreadable without the metadata the
    /// same pass computes, and an affine scheme's metadata is scales *and*
    /// zero points. Every two-output scheme lives in [`Self::encode_bytes`].
    ///
    /// `outputs` is positional and the loader fixed the order: the weight, then
    /// the metadata in the order `quant_metadata_outputs` declared it.
    fn encode_mlx_affine_u4(
        &self,
        bytes: &[u8],
        source: Option<&SourceExtent>,
        inputs: &[BufferId],
        outputs: &[BufferId],
        transform: &TransformSpec,
    ) -> Result<Vec<(BufferId, Vec<u8>)>, Error> {
        let Some(scheme) = transform.to else {
            return Err(invalid("Encode carries no destination scheme"));
        };
        if scheme != QuantScheme::MlxAffineU4 {
            return Err(invalid(format!(
                "host Encode does not implement {scheme:?}"
            )));
        }
        let [weight, scales, biases] = outputs else {
            return Err(invalid(format!(
                "encoding to {scheme:?} writes a weight, its scales and its zero \
                 points, but the instruction has {} outputs",
                outputs.len()
            )));
        };
        let shape = self.buffer_shape(*weight)?.to_vec();
        // Folded, for the reason `encode_bytes` states: the affine pass reads
        // groups of `group` consecutive elements out of a row-major buffer, so
        // a leading axis is rows and nothing else.
        let Some((rows, cols)) = crate::types::rectangle(&shape) else {
            return Err(invalid(format!(
                "encoding to {scheme:?} scales a [rows, cols] rectangle, not \
                 {shape:?}"
            )));
        };
        let group = i64::from(scheme.default_group_size());
        if group <= 0 || cols % group != 0 {
            return Err(invalid(format!(
                "encoding to {scheme:?} groups {group} columns, which does not \
                 divide the {cols} of {shape:?}"
            )));
        }
        // The operand is whatever the chain decoded to; `decode_values` gives
        // f64 and the quantizer works in f32, which is what MLX's own encoder
        // does and therefore what the codes have to be rounded from.
        //
        // Read off the OPERAND, which knows its own dtype. It used to be
        // recovered by dividing the byte count by the destination's element
        // count and matching 2 against BF16 and 4 against F32 — a guess of
        // exactly the kind this crate refuses everywhere else, and one only
        // needed because a buffer's type lived on a tensor an intermediate
        // does not have.
        let operand_dtype = if let Some(input) = inputs.first() {
            self.buffer_dtype(*input)?
        } else if let Some(source) = source {
            self.source_dtype(source.tensor_id)?
        } else {
            return Err(invalid("host Encode requires a source or input buffer"));
        };
        let values = decode_values(bytes, operand_dtype)?;
        if values.len() as i64 != rows * cols {
            return Err(invalid(format!(
                "Encode was handed {} elements for the {shape:?} it must quantize",
                values.len()
            )));
        }

        let n_groups = (rows * cols / group) as usize;
        let mut packed = Vec::with_capacity(values.len() / 8 * 4);
        let mut scale_values = Vec::with_capacity(n_groups);
        let mut bias_values = Vec::with_capacity(n_groups);
        for chunk in values.chunks(group as usize) {
            let (scale, bias) = mlx_affine_group_params(chunk);
            for word in chunk.chunks(8) {
                let mut out: u32 = 0;
                for (k, &value) in word.iter().enumerate() {
                    let code = (((value as f32) - bias) / scale).round().clamp(0.0, 15.0) as u32;
                    out |= code << (k * 4);
                }
                packed.extend_from_slice(&out.to_le_bytes());
            }
            scale_values.push(f64::from(scale));
            bias_values.push(f64::from(bias));
        }
        Ok(vec![
            (*weight, packed),
            (*scales, encode_values(&scale_values, DType::Bf16)?),
            (*biases, encode_values(&bias_values, DType::Bf16)?),
        ])
    }

    /// The multiply, as one spelling of [`Self::fold_per_block`].
    fn scale_per_block(
        &self,
        values: Vec<f64>,
        inputs: &[BufferId],
        output: BufferId,
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        self.fold_per_block(values, inputs, output, transform, "Scale", |v, f| v * f)
    }

    fn fold_per_block(
        &self,
        mut values: Vec<f64>,
        inputs: &[BufferId],
        output: BufferId,
        transform: &TransformSpec,
        what: &'static str,
        fold: impl Fn(f32, f32) -> f32,
    ) -> Result<Vec<u8>, Error> {
        let factors = *inputs
            .last()
            .ok_or_else(|| invalid(format!("per-block {what} has no factor operand")))?;
        let factors = decode_values(&self.buffer_bytes(factors)?, self.buffer_dtype(factors)?)?;
        let shape = self.buffer_shape(output)?.to_vec();
        let blocks = &transform.scale_blocks;
        if shape.len() != blocks.len() {
            return Err(invalid(format!(
                "per-block {what} blocks {blocks:?} do not match output shape {shape:?}"
            )));
        }

        // Extents of the factor tensor, and the row-major strides to index it
        // by. Both are folded in one reverse pass so the two can never be
        // computed from different shapes.
        let mut counts = vec![0i64; shape.len()];
        let mut strides = vec![0i64; shape.len()];
        let mut running = 1i64;
        for axis in (0..shape.len()).rev() {
            let block = blocks[axis];
            if block <= 0 || shape[axis] % block != 0 {
                return Err(invalid(format!(
                    "per-block {what} block {block} does not divide axis {axis} of {shape:?}"
                )));
            }
            counts[axis] = shape[axis] / block;
            strides[axis] = running;
            running *= counts[axis];
        }
        let total: i64 = shape.iter().product();
        if values.len() as i64 != total {
            return Err(invalid(format!(
                "per-block {what} has {} elements but shape {shape:?} needs {total}",
                values.len()
            )));
        }
        if factors.len() as i64 != running {
            return Err(invalid(format!(
                "per-block {what} has {} factors but blocking {blocks:?} of {shape:?} \
                 needs {running}",
                factors.len()
            )));
        }

        // One odometer over the logical shape. `index` is the factor's flat
        // position, carried alongside so the division happens once per axis
        // step instead of once per element.
        let mut coord = vec![0i64; shape.len()];
        for value in &mut values {
            let mut index = 0i64;
            for axis in 0..shape.len() {
                index += (coord[axis] / blocks[axis]) * strides[axis];
            }
            let factor = factors[index as usize] as f32;
            *value = f64::from(fold(*value as f32, factor));
            for axis in (0..shape.len()).rev() {
                coord[axis] += 1;
                if coord[axis] < shape[axis] {
                    break;
                }
                coord[axis] = 0;
            }
        }
        encode_values(&values, self.buffer_dtype(output)?)
    }

    fn cast_bytes(
        &self,
        source: Option<&SourceExtent>,
        input: Option<BufferId>,
        output: Option<BufferId>,
        bytes: &[u8],
    ) -> Result<Vec<u8>, Error> {
        let output = output.ok_or_else(|| invalid("host Cast requires an output buffer"))?;
        let from = if let Some(input) = input {
            self.buffer_dtype(input)?
        } else if let Some(source) = source {
            self.source_dtype(source.tensor_id)?
        } else {
            return Err(invalid("host Cast requires a source or input buffer"));
        };
        let to = self.buffer_dtype(output)?;
        if from == to {
            return Ok(bytes.to_vec());
        }
        cast_elements(bytes, from, to)
    }

    /// Decode a self-contained blocked payload to its logical dtype: the
    /// `Cast(Quant → Raw)` direction, which the builder lowers to `Decode`.
    ///
    /// Only schemes whose scales live inside the block reach here — the
    /// validate pass admits exactly those — so the payload bytes are the whole
    /// story and there is no factor operand to fetch. Which schemes those are
    /// is not a list kept here: it is exactly the set that answers
    /// `block_layout()`, so a scheme becomes decodable by declaring its layout
    /// and gaining an arm in [`decode_gguf_block_into`], and cannot be half
    /// admitted. The `f32` values it yields narrow to the scheme's logical
    /// `BF16` by round to nearest even.
    fn decode_bytes(
        &self,
        output: Option<BufferId>,
        bytes: &[u8],
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        let scheme = transform
            .from
            .ok_or_else(|| invalid("host Decode names no source scheme"))?;
        let Some((elements, block_bytes)) = scheme.block_layout() else {
            return Err(invalid(format!(
                "host Decode implements the blocked GGUF schemes, not {scheme:?}"
            )));
        };
        let (elements, block_bytes) = (elements as usize, block_bytes as usize);
        let output = output.ok_or_else(|| invalid("host Decode requires an output buffer"))?;
        let dtype = self.buffer_dtype(output)?;
        if dtype != DType::Bf16 {
            return Err(invalid(format!(
                "GGUF {scheme:?} decodes to its logical BF16, not {dtype:?}"
            )));
        }
        if !bytes.len().is_multiple_of(block_bytes) {
            return Err(invalid(format!(
                "host Decode read {} bytes, not whole {block_bytes}-byte {scheme:?} blocks",
                bytes.len()
            )));
        }
        let blocks = bytes.len() / block_bytes;
        let out_bytes = elements * 2;
        let mut out = vec![0u8; blocks * out_bytes];
        // Blocks are self-contained, so they decode in parallel the same way
        // encode rows do: disjoint output slices, any worker count, the same
        // bytes.
        let workers = if blocks < (1 << 15) {
            1
        } else {
            std::thread::available_parallelism()
                .map_or(1, std::num::NonZero::get)
                .min(blocks)
        };
        let per_worker = blocks.div_ceil(workers);
        std::thread::scope(|scope| {
            let mut out_rest = &mut out[..];
            let mut start = 0usize;
            while start < blocks {
                let count = per_worker.min(blocks - start);
                let (chunk, rest) = std::mem::take(&mut out_rest).split_at_mut(count * out_bytes);
                out_rest = rest;
                let source = &bytes[start * block_bytes..(start + count) * block_bytes];
                scope.spawn(move || {
                    let mut values = vec![0.0f32; elements];
                    for (block, out) in source
                        .chunks_exact(block_bytes)
                        .zip(chunk.chunks_exact_mut(out_bytes))
                    {
                        decode_gguf_block_into(scheme, block, &mut values);
                        for (value, le) in values.iter().zip(out.chunks_exact_mut(2)) {
                            le.copy_from_slice(&bf16::from_f32(*value).to_bits().to_le_bytes());
                        }
                    }
                });
                start += count;
            }
        });
        Ok(out)
    }

    /// Quantize on the host: the ports of the CUDA encode kernels, arithmetic
    /// first, one arm per scheme `validate-target-support` admits.
    ///
    /// * `Mxfp4E2M1E8M0` — `quant_bf16_to_mxfp4.cu`: per 32-element group
    ///   along the last axis, absmax → one E8M0 byte scale (the smallest
    ///   power of two whose ×6 covers the absmax), each element divided by it
    ///   and rounded through the kernel's midpoint table.
    /// * `Fp8E4M3` — `quant_bf16_to_fp8.cu::quant_per_channel_kernel`: per
    ///   row, absmax → one `F32` factor `absmax/448` (`1.0` for a dead row),
    ///   elements saturate-cast to E4M3, round to nearest even.
    /// * `Int8Symmetric` — the same shape with `127` for `448` and
    ///   `rintf`'s round-half-even for the cast.
    ///
    /// The operand reaches every scheme as `BF16` rows because that is what
    /// the device encodes from (`materialize_encode_input_bf16_rows`): a
    /// wider weight is narrowed first — encoding straight from `f32` would
    /// keep mantissa bits the device never saw — and an FP8 block-scaled
    /// source, the one the instruction names by `metadata_source`, is
    /// dequantized with its block factors the way
    /// `transcode_engine.hpp::fp8_tile_scale` resolves them.
    ///
    /// Every row's output depends on that row alone, in all three schemes, so
    /// rows are encoded in parallel; the worker count changes wall-clock and
    /// nothing else.
    fn encode_bytes(
        &mut self,
        source: Option<&SourceExtent>,
        inputs: &[BufferId],
        outputs: &[BufferId],
        bytes: &[u8],
        transform: &TransformSpec,
        max_tile_bytes: u64,
    ) -> Result<(), Error> {
        let Some(scheme) = transform.to else {
            return Err(invalid("host Encode names no target scheme"));
        };
        let &[payload, scales] = outputs else {
            return Err(invalid("host Encode expects weight and scale outputs"));
        };
        let shape = self.buffer_shape(payload)?.to_vec();
        // THE RECTANGLE, FOLDED. `[experts, rows, cols]` is `[experts * rows,
        // cols]` in the same bytes in the same order, and every line below
        // indexes `row * cols + c`, so the fold is the layout rather than a
        // reinterpretation of it -- see `types::rectangle`, which is where the
        // rule is stated once. This read `&[rows, cols]` and refused any other
        // rank, which is what stopped an expert bank being quantized at load
        // time.
        let Some((rows, cols)) = crate::types::rectangle(&shape) else {
            return Err(invalid(format!(
                "host Encode scales a [rows, cols] rectangle, and {shape:?} has \
                 no axis left over to hold one scale per row"
            )));
        };
        let (rows, cols) = (checked_usize_i64(rows)?, checked_usize_i64(cols)?);
        let scale_shape = self.buffer_shape(scales)?.to_vec();
        // What the plan's `ScaleLayout` built: the payload's leading axes,
        // then the scheme's own last axis. Compared whole rather than folded,
        // because a scales plane the engine BINDS is bound at its declared
        // rank and a fold here would accept a plan that published the wrong
        // one.
        let lead = &shape[..shape.len() - 1];

        let dtype = if let Some(source) = source {
            self.source_dtype(source.tensor_id)?
        } else if let Some(input) = inputs.first() {
            self.buffer_dtype(*input)?
        } else {
            return Err(invalid("host Encode requires a source or input buffer"));
        };
        if bytes.len() != rows * cols * dtype.bytes_ceil() as usize {
            return Err(invalid(format!(
                "host Encode read {} bytes for a [{rows}, {cols}] {dtype:?} weight",
                bytes.len()
            )));
        }
        let operand = if dtype == DType::E4m3 {
            let source = source.ok_or_else(|| {
                invalid("host Encode of an FP8 operand requires a checkpoint source")
            })?;
            self.fp8_block_operand(source, transform, bytes, rows, cols)?
        } else {
            EncodeOperand::Widened { bytes, dtype }
        };
        if let EncodeOperand::Widened { dtype, .. } = &operand
            && !matches!(dtype, DType::Bf16 | DType::F16 | DType::F32)
        {
            return Err(invalid(format!(
                "host Encode reads BF16 (or F16/F32 narrowed to it, or \
                 block-scaled F8E4M3), not {dtype:?}"
            )));
        }

        let (packed, scale_out) = match scheme {
            QuantScheme::Mxfp4E2M1E8M0 => {
                if cols == 0 || !cols.is_multiple_of(32) {
                    return Err(invalid(format!(
                        "host MXFP4 Encode cols must be a multiple of 32, got {cols}"
                    )));
                }
                let groups = cols / 32;
                let want = crate::types::grouped_shape(lead, groups as i64);
                if scale_shape != want {
                    return Err(invalid(format!(
                        "host MXFP4 Encode scale must be {want:?}, got {scale_shape:?}"
                    )));
                }
                let mut packed = vec![0u8; rows * cols / 2];
                let mut scale_out = vec![0u8; rows * groups];
                let job = |row: usize, buf: &mut [f32], out: &mut [u8], scale: &mut [u8]| {
                    operand.row_bf16(row, cols, buf);
                    for g in 0..groups {
                        scale[g] = encode_mxfp4_group(
                            &buf[g * 32..g * 32 + 32],
                            &mut out[g * 16..g * 16 + 16],
                        );
                    }
                };
                encode_rows(
                    rows,
                    cols,
                    cols / 2,
                    groups,
                    &mut packed,
                    &mut scale_out,
                    &job,
                );
                (packed, scale_out)
            }
            QuantScheme::Fp8E4M3 | QuantScheme::Int8Symmetric => {
                if scale_shape != lead {
                    return Err(invalid(format!(
                        "host per-channel Encode scale must be {lead:?}, got {scale_shape:?}"
                    )));
                }
                let int8 = scheme == QuantScheme::Int8Symmetric;
                let code_max = if int8 { 127.0f32 } else { 448.0 };
                let mut packed = vec![0u8; rows * cols];
                let mut scale_out = vec![0u8; rows * 4];
                let job = |row: usize, buf: &mut [f32], out: &mut [u8], scale: &mut [u8]| {
                    operand.row_bf16(row, cols, buf);
                    let mut absmax = 0.0f32;
                    for &v in buf.iter() {
                        let a = v.abs();
                        if a > absmax {
                            absmax = a;
                        }
                    }
                    // A dead row quantizes by 1.0 to all zeros, exactly as
                    // the kernel's degenerate arm says.
                    let (recip, factor) = if absmax > 0.0 {
                        (code_max / absmax, absmax / code_max)
                    } else {
                        (1.0, 1.0)
                    };
                    scale.copy_from_slice(&factor.to_le_bytes());
                    if int8 {
                        for (value, out) in buf.iter().zip(out.iter_mut()) {
                            let q = (value * recip).round_ties_even() as i32;
                            *out = q.clamp(-128, 127) as i8 as u8;
                        }
                    } else {
                        for (value, out) in buf.iter().zip(out.iter_mut()) {
                            *out = f32_to_fp8_e4m3(value * recip);
                        }
                    }
                };
                encode_rows(rows, cols, cols, 4, &mut packed, &mut scale_out, &job);
                (packed, scale_out)
            }
            other => {
                return Err(invalid(format!(
                    "no encode kernel writes {other:?}, so the host executor \
                     does not either"
                )));
            }
        };

        if self.buffer_bytes(payload)?.len() != packed.len() {
            return Err(invalid("host Encode payload buffer size mismatch"));
        }
        if self.buffer_bytes(scales)?.len() != scale_out.len() {
            return Err(invalid("host Encode scale buffer size mismatch"));
        }
        let tile = if max_tile_bytes == 0 {
            packed.len().max(1)
        } else {
            checked_usize(max_tile_bytes)?.max(1)
        };
        for (offset, chunk) in packed.chunks(tile).enumerate() {
            self.write_buffer(payload, offset * tile, chunk)?;
        }
        self.write_buffer(scales, 0, &scale_out)?;
        Ok(())
    }

    /// Resolve an FP8 block-scaled Encode operand the way
    /// `transcode_engine.hpp::fp8_tile_scale` does.
    ///
    /// The factor tensor is the one the instruction names — the loader read
    /// the tensor table, so the loader answered — the group size is the ratio
    /// of the weight's *on-disk* shape to the factor tensor's (square, or the
    /// checkpoint is not one either kernel indexes), and a TP shard's offset
    /// within the full weight is decoded from the extent's base offset, which
    /// is in elements because FP8 is one byte each.
    fn fp8_block_operand<'b>(
        &self,
        source: &SourceExtent,
        transform: &TransformSpec,
        bytes: &'b [u8],
        rows: usize,
        cols: usize,
    ) -> Result<EncodeOperand<'b>, Error> {
        let Some(metadata_source) = transform.metadata_source else {
            return Err(invalid(
                "host Encode of an FP8 source needs the block-scale tensor its \
                 instruction names, and this instruction names none",
            ));
        };
        let weight = self
            .index
            .source(self.plan, source.tensor_id)
            .ok_or_else(|| invalid("FP8 Encode source is not in the plan"))?;
        let scale = self
            .index
            .source(self.plan, metadata_source)
            .ok_or_else(|| invalid("FP8 Encode scale tensor is not in the plan"))?;
        let (&[true_rows, true_cols], &[scale_rows, scale_cols]) =
            (weight.shape.as_slice(), scale.shape.as_slice())
        else {
            return Err(invalid(format!(
                "FP8 Encode weight and scale must be 2-D, got {:?} and {:?}",
                weight.shape, scale.shape
            )));
        };
        if !matches!(scale.encoding, Encoding::Raw(DType::F32)) {
            return Err(invalid(format!(
                "FP8 Encode scale '{}' must be F32",
                scale.name
            )));
        }
        let (true_rows, true_cols) = (checked_usize_i64(true_rows)?, checked_usize_i64(true_cols)?);
        let (scale_rows, scale_cols) = (
            checked_usize_i64(scale_rows)?,
            checked_usize_i64(scale_cols)?,
        );
        let group_rows = true_rows.checked_div(scale_rows).unwrap_or(0);
        let group_cols = true_cols.checked_div(scale_cols).unwrap_or(0);
        if group_rows == 0 || group_rows != group_cols {
            return Err(invalid(format!(
                "FP8 Encode source '{}' has unsupported scale shape \
                 [{scale_rows}, {scale_cols}] for weight [{true_rows}, {true_cols}]",
                weight.name
            )));
        }
        let group = group_rows;

        // The rank's corner within the full weight, then within the factors.
        let base = checked_usize(source.stride.base_offset)?;
        let scale_row_offset = (base / true_cols) / group;
        let scale_col_offset = (base % true_cols) / group;
        if scale_row_offset + rows.div_ceil(group) > scale_rows + 1
            || scale_col_offset + cols.div_ceil(group) > scale_cols + 1
        {
            return Err(invalid(format!(
                "FP8 Encode shard [{rows}, {cols}] at offset {base} reaches past \
                 the [{scale_rows}, {scale_cols}] factors of '{}'",
                scale.name
            )));
        }

        let factor_bytes = self.read_extent(&SourceExtent {
            file_id: scale.file_id,
            tensor_id: scale.id,
            file_offset: scale.file_offset,
            span_bytes: scale.span_bytes,
            stride: Extent {
                base_offset: 0,
                element_bytes: 4,
                dims: vec![crate::extent::Dim {
                    count: (scale_rows * scale_cols) as i64,
                    src_stride: 4,
                    dst_stride: 4,
                }],
            },
            dtype: DType::F32,
        })?;
        let factors: Vec<f32> = factor_bytes
            .chunks_exact(4)
            .map(|le| f32::from_le_bytes(le.try_into().unwrap()))
            .collect();
        // The last block may be short, so the strict check is on the first
        // element each row reads; the loop below indexes the same way the
        // blocked dequant kernel does.
        let last_index = (scale_row_offset + (rows.saturating_sub(1)) / group) * scale_cols
            + scale_col_offset
            + (cols.saturating_sub(1)) / group;
        if rows > 0 && cols > 0 && last_index >= factors.len() {
            return Err(invalid(format!(
                "FP8 Encode factors for '{}' end at {} but the shard reads {}",
                scale.name,
                factors.len(),
                last_index
            )));
        }
        Ok(EncodeOperand::BlockScaledFp8 {
            bytes,
            factors,
            scale_cols,
            group,
            scale_row_offset,
            scale_col_offset,
        })
    }

    fn buffer_dtype(&self, id: BufferId) -> Result<DType, Error> {
        match &self.plan.buffer(id)?.ty.encoding {
            Encoding::Raw(dtype) => Ok(*dtype),
            Encoding::Quant(_) => Err(invalid("host Cast does not accept quantized buffers")),
        }
    }

    /// A buffer's declared shape.
    fn buffer_shape(&self, id: BufferId) -> Result<&[i64], Error> {
        Ok(self.plan.buffer(id)?.ty.shape.as_slice())
    }

    fn source_dtype(&self, id: crate::types::TensorId) -> Result<DType, Error> {
        let source = self
            .index
            .source(self.plan, id)
            .ok_or_else(|| invalid(format!("source tensor {} is missing", id.0)))?;
        match source.encoding {
            Encoding::Raw(dtype) => Ok(dtype),
            Encoding::Quant(_) => Err(invalid("host Cast does not accept quantized sources")),
        }
    }

    fn buffer_bytes(&self, id: BufferId) -> Result<Cow<'_, [u8]>, Error> {
        let (root, offset, len) = self.resolve(id, 0, usize::MAX)?;
        match root {
            Root::Arena => self.arena.read(offset, len),
            Root::Owned(root) => match self.buffers.get(&root) {
                Some(BufferLoc::Owned(bytes)) => bytes
                    .get(offset..offset + len)
                    .map(Cow::Borrowed)
                    .ok_or_else(|| invalid("owned buffer range is out of bounds")),
                _ => Err(invalid("resolved owned buffer is missing")),
            },
        }
    }

    fn write_buffer(&mut self, id: BufferId, offset: usize, bytes: &[u8]) -> Result<(), Error> {
        let (root, base, _) = self.resolve(id, offset, bytes.len())?;
        let end = base
            .checked_add(bytes.len())
            .ok_or_else(|| invalid("buffer write range overflow"))?;
        match root {
            Root::Arena => {
                debug_assert!(end <= ArenaBacking::len(self.arena));
                self.arena.write(base, bytes)?;
            }
            Root::Owned(root) => match self.buffers.get_mut(&root) {
                Some(BufferLoc::Owned(dest)) => dest
                    .get_mut(base..end)
                    .ok_or_else(|| invalid("owned buffer write is out of bounds"))?
                    .copy_from_slice(bytes),
                _ => return Err(invalid("resolved owned buffer is missing")),
            },
        }
        Ok(())
    }

    fn resolve(
        &self,
        id: BufferId,
        extra_offset: usize,
        requested_len: usize,
    ) -> Result<(Root, usize, usize), Error> {
        let loc = self
            .buffers
            .get(&id)
            .ok_or_else(|| invalid(format!("buffer {} is not allocated", id.0)))?;
        match loc {
            BufferLoc::Arena { offset, len } => {
                resolve_range(Root::Arena, *offset, *len, extra_offset, requested_len)
            }
            BufferLoc::Owned(bytes) => {
                resolve_range(Root::Owned(id), 0, bytes.len(), extra_offset, requested_len)
            }
            BufferLoc::View { input, offset, len } => {
                let requested = if requested_len == usize::MAX {
                    len.saturating_sub(extra_offset)
                } else {
                    requested_len
                };
                if extra_offset > *len || requested > len.saturating_sub(extra_offset) {
                    return Err(invalid("view range is out of bounds"));
                }
                self.resolve(*input, offset + extra_offset, requested)
            }
        }
    }
}

fn resolve_range(
    root: Root,
    base: usize,
    available: usize,
    extra_offset: usize,
    requested_len: usize,
) -> Result<(Root, usize, usize), Error> {
    if extra_offset > available {
        return Err(invalid("buffer offset is out of bounds"));
    }
    let len = if requested_len == usize::MAX {
        available - extra_offset
    } else {
        requested_len
    };
    if len > available - extra_offset {
        return Err(invalid("buffer range is out of bounds"));
    }
    Ok((root, base + extra_offset, len))
}

/// Walk a [`GatherSpec`]'s table over `bytes`, one block at a time.
///
/// This is the whole of the gather lowering's execution, and it is deliberately
/// the dullest function in the file. The table was compiled by
/// `contract::compile`, priced by `contract::compile::Lowering::cost`, and
/// checked by `plan::verify`; all that is left is the copy, on the host, while
/// the bytes are between the file and the device they are staged into. A
/// `[768, 1536]` bf16 bank is 2.3 MB moved once against a load measured in
/// gigabytes.
///
/// Rows are the axes outside the gathered one, and they share one table — which
/// is why a permutation of 1.18M elements is 1536 numbers and a loop.
fn permute(bytes: &[u8], gather: &GatherSpec) -> Result<Vec<u8>, Error> {
    let block = checked_usize(gather.block_bytes)?;
    let rows = checked_usize(gather.rows)?;
    let src_row = checked_usize(gather.src_row_bytes)?;
    let dst_row = checked_usize(gather.dst_row_bytes())?;
    let total = rows
        .checked_mul(dst_row)
        .ok_or_else(|| invalid("gather output size overflow"))?;
    if rows.saturating_mul(src_row) > bytes.len() {
        return Err(invalid(format!(
            "gather reads {} bytes of a {}-byte source",
            rows.saturating_mul(src_row),
            bytes.len()
        )));
    }
    let mut out = vec![0u8; total];
    for (at, index) in gather.indices.iter().enumerate() {
        let index = usize::try_from(*index)
            .map_err(|_| invalid(format!("gather index {index} is negative")))?;
        let src = index
            .checked_mul(block)
            .ok_or_else(|| invalid("gather source offset overflow"))?;
        if src + block > src_row {
            return Err(invalid(format!(
                "gather index {index} reads past the end of a {src_row}-byte row"
            )));
        }
        let dst = at * block;
        for row in 0..rows {
            let from = row * src_row + src;
            let to = row * dst_row + dst;
            out[to..to + block].copy_from_slice(&bytes[from..from + block]);
        }
    }
    Ok(out)
}

fn gather_strided(raw: Vec<u8>, extent: &Extent) -> Result<Vec<u8>, Error> {
    let shape = extent
        .dims
        .iter()
        .map(|dim| checked_usize_i64(dim.count))
        .collect::<Result<Vec<_>, _>>()?;
    let elements = shape.iter().try_fold(1usize, |n, dim| {
        n.checked_mul(*dim)
            .ok_or_else(|| invalid("extent element count overflow"))
    })?;
    let elem = extent.element_bytes as usize;
    let total = elements.saturating_mul(elem);

    // Fold every trailing source-dense dimension into one contiguous run, so
    // the loop below moves a run per iteration instead of an element. The
    // common extents collapse entirely — a whole tensor, or a row shard whose
    // rows are contiguous — and what is left iterates only the axes that
    // actually stride. Walking element-wise here was measured at ~2.8 s of a
    // 3 s conversion, on bounds checks and a `Vec` per element.
    let mut run = elem;
    let mut outer = extent.dims.len();
    while outer > 0 {
        let dim = &extent.dims[outer - 1];
        if dim.src_stride != run as i64 {
            break;
        }
        match run.checked_mul(checked_usize_i64(dim.count)?) {
            Some(next) => run = next,
            None => return Err(invalid("extent run length overflow")),
        }
        outer -= 1;
    }
    if outer == 0 {
        // One dense run: the physical bytes *are* the compact bytes, and the
        // read already owns them — a copy here would double the largest
        // allocation of a conversion to move nothing.
        if raw.len() < total {
            return Err(invalid("source extent is out of bounds"));
        }
        let mut raw = raw;
        raw.truncate(total);
        return Ok(raw);
    }

    let mut out = vec![0u8; total];
    let mut coord = vec![0usize; outer];
    let mut dst = 0usize;
    while dst < total {
        let src = extent_offset(&coord, &extent.dims[..outer], true)?;
        out.get_mut(dst..dst + run)
            .ok_or_else(|| invalid("compact extent range overflow"))?
            .copy_from_slice(
                raw.get(src..src + run)
                    .ok_or_else(|| invalid("source extent is out of bounds"))?,
            );
        dst += run;
        for axis in (0..outer).rev() {
            coord[axis] += 1;
            if coord[axis] < shape[axis] {
                break;
            }
            coord[axis] = 0;
        }
    }
    Ok(out)
}

/// A copy is well-formed when the two sides span the same number of bytes.
///
/// Not the same *dimensions*: the two extents describe different things. The
/// source's dims say how to walk the checkpoint, which the compiler states as
/// coarsely as it can — a contiguous row shard becomes one dim of wide
/// elements. The destination's dims describe a compact block, which is checked
/// separately. Requiring the two decompositions to match rejected every sharded
/// plan, because `[2048] x 6144 bytes` and `[2048, 3072] x 2 bytes` are the same
/// 12 MiB written the same way. Neither the CUDA nor the Metal executor makes
/// that comparison; this one should not either.
fn require_same_byte_count(source: &Extent, dest: &Extent) -> Result<(), Error> {
    let bytes = |extent: &Extent| -> Option<i64> {
        extent
            .dims
            .iter()
            .try_fold(i64::from(extent.element_bytes), |acc, dim| {
                acc.checked_mul(dim.count)
            })
    };
    let (source_bytes, dest_bytes) = (bytes(source), bytes(dest));
    if source_bytes.is_none() || source_bytes != dest_bytes {
        return Err(invalid(format!(
            "source extent spans {source_bytes:?} bytes but the destination spans \
             {dest_bytes:?}"
        )));
    }
    Ok(())
}

/// A cast is well-formed when the two sides hold the same number of
/// *elements*; the widths differ by exactly the representations' ratio.
fn require_same_element_count(source: &Extent, dest: &Extent) -> Result<(), Error> {
    let count = |extent: &Extent| -> Option<i64> {
        extent
            .dims
            .iter()
            .try_fold(1i64, |acc, dim| acc.checked_mul(dim.count))
    };
    let (source_count, dest_count) = (count(source), count(dest));
    if source_count.is_none() || source_count != dest_count {
        return Err(invalid(format!(
            "cast source holds {source_count:?} elements but the destination holds \
             {dest_count:?}"
        )));
    }
    Ok(())
}

fn extent_offset(index: &[usize], dims: &[crate::plan::Dim], source: bool) -> Result<usize, Error> {
    index
        .iter()
        .zip(dims)
        .try_fold(0usize, |offset, (index, dim)| {
            let stride = if source {
                checked_usize_i64(dim.src_stride)?
            } else {
                checked_usize_i64(dim.dst_stride)?
            };
            offset
                .checked_add(
                    index
                        .checked_mul(stride)
                        .ok_or_else(|| invalid("extent index overflow"))?,
                )
                .ok_or_else(|| invalid("extent offset overflow"))
        })
}

fn extent_bytes(extent: &Extent) -> Result<usize, Error> {
    let elements = extent.dims.iter().try_fold(1usize, |n, dim| {
        n.checked_mul(checked_usize_i64(dim.count)?)
            .ok_or_else(|| invalid("extent byte count overflow"))
    })?;
    elements
        .checked_mul(extent.element_bytes as usize)
        .ok_or_else(|| invalid("extent byte count overflow"))
}

fn physical_source_bytes(extent: &Extent) -> Result<u64, Error> {
    physical_bytes(extent, true)
}

fn physical_bytes(extent: &Extent, source: bool) -> Result<u64, Error> {
    let mut end = 0u64;
    for dim in &extent.dims {
        if dim.count == 0 {
            return Ok(0);
        }
        let count =
            u64::try_from(dim.count - 1).map_err(|_| invalid("negative extent dimension"))?;
        let stride = u64::try_from(if source {
            dim.src_stride
        } else {
            dim.dst_stride
        })
        .map_err(|_| invalid("negative extent stride"))?;
        end = end
            .checked_add(
                count
                    .checked_mul(stride)
                    .ok_or_else(|| invalid("extent range overflow"))?,
            )
            .ok_or_else(|| invalid("extent range overflow"))?;
    }
    end.checked_add(u64::from(extent.element_bytes))
        .ok_or_else(|| invalid("extent range overflow"))
}

fn checked_usize(value: u64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| invalid("value does not fit usize"))
}

fn checked_usize_i64(value: i64) -> Result<usize, Error> {
    usize::try_from(value).map_err(|_| invalid("negative or oversized extent value"))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Contract(message.into())
}

#[cfg(test)]
mod tests {
    use super::super::sink::MemorySink;
    use super::super::{Execution, HostStorage};
    use super::*;
    use crate::plan::{
        BufferDecl, DestExtent, Dim, MemoryPlan, SourceTensorDecl, StorageTarget, TileSpec,
        TransformSpec,
    };
    use crate::types::{FileId, InstrId, TensorDecl, TensorId};

    fn extent(base_offset: u64, element_bytes: u32, dims: &[(i64, i64, i64)]) -> Extent {
        Extent {
            base_offset,
            element_bytes,
            dims: dims
                .iter()
                .map(|&(count, src_stride, dst_stride)| Dim {
                    count,
                    src_stride,
                    dst_stride,
                })
                .collect(),
        }
    }

    /// A padded tensor, compiled and then actually run.
    ///
    /// Every other test here hands the executor a plan built by hand, which
    /// cannot catch a compiler that emits the fill in the wrong place. This one
    /// goes contract -> `compile` -> `execute_plan` and reads the bytes back,
    /// so the padded columns being zero is a fact about the whole path.
    #[test]
    fn a_padded_tensor_comes_back_with_zeros_where_no_source_reaches() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};

        let dir = std::env::temp_dir().join(format!("pie_host_pad_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"raw":{"dtype":"U8","shape":[2,3],"data_offsets":[0,6]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&[1, 2, 3, 4, 5, 6]);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 6,
                shape: vec![2, 3],
                encoding: Encoding::Raw(DType::U8),
            }],
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "padded",
                Expr::concat(
                    1,
                    vec![
                        Expr::fill(0.0, TensorType::raw(vec![2, 1], DType::U8)),
                        Expr::src("raw"),
                        Expr::fill(0.0, TensorType::raw(vec![2, 1], DType::U8)),
                    ],
                ),
                vec![2, 5],
                Encoding::Raw(DType::U8),
            )],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        let storage = Execution::new(&plan, &dir).run().unwrap();
        assert_eq!(
            storage.tensors["padded"],
            vec![0, 1, 2, 3, 0, 0, 4, 5, 6, 0]
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// The one node that changes a value, compiled and then actually run.
    #[test]
    fn a_scaled_tensor_comes_back_multiplied() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};

        let dir = std::env::temp_dir().join(format!("pie_host_scale_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let values: [f32; 4] = [1.0, 2.0, 3.0, 5.0];
        let header = r#"{"raw":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        for value in values {
            file.extend_from_slice(&value.to_le_bytes());
        }
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 16,
                shape: vec![4],
                encoding: Encoding::Raw(DType::F32),
            }],
        };
        // Not a round number: a factor the executor merely copied instead of
        // multiplying by would still pass with 1.0, and one it truncated to
        // `f64` would still pass with 0.25.
        let factor = 1.0f32 / 3.0;
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "scaled",
                Expr::src("raw").scale(factor),
                vec![4],
                Encoding::Raw(DType::F32),
            )],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        assert!(
            plan.instrs.iter().any(|instr| matches!(
                instr,
                StorageInstr::TileMap {
                    kind: TileMapKind::Scale,
                    transform,
                    ..
                } if transform.scale_factor_bits == factor.to_bits()
            )),
            "the factor did not survive lowering: {:?}",
            plan.instrs
        );

        let storage = Execution::new(&plan, &dir).run().unwrap();
        let expected: Vec<u8> = values
            .iter()
            .flat_map(|value| (value * factor).to_le_bytes())
            .collect();
        assert_eq!(storage.tensors["scaled"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Runtime quantization, compiled and then actually run.
    ///
    /// The first row visits every E2M1 codepoint from both sides of each
    /// midpoint boundary; the second forces a different E8M0 scale. An
    /// executor that mis-rounds a boundary, swaps a nibble pair, drops a sign,
    /// or applies one row's scale to the other cannot pass.
    #[test]
    fn a_bf16_tensor_is_encoded_to_mxfp4_with_its_scales() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_encode_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Scale 2^0: the row's absmax is exactly 6.0, so every value divides
        // by one and the codepoint is read straight off the midpoint table.
        #[rustfmt::skip]
        let row0: [f32; 32] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            0.2, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
            0.4, 0.6, 1.1, 1.6, 2.2, 3.2, 4.5, 5.5,
        ];
        let mut data = Vec::new();
        for value in row0 {
            data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        // Scale 2^1: absmax 12.0, and 12/2 = 6 encodes as the top magnitude.
        for _ in 0..32 {
            data.extend_from_slice(&bf16::from_f32(12.0).to_bits().to_le_bytes());
        }
        let header = r#"{"raw":{"dtype":"BF16","shape":[2,32],"data_offsets":[0,128]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&data);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 128,
                shape: vec![2, 32],
                encoding: Encoding::Raw(DType::Bf16),
            }],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Quant(spec.clone())),
                vec![2, 32],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = Execution::new(&plan, &dir).run().unwrap();

        #[rustfmt::skip]
        let expected_row0: [u8; 16] = [
            0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE,
            0x10, 0x32, 0x54, 0x76, 0x11, 0x32, 0x54, 0x76,
        ];
        let mut expected = expected_row0.to_vec();
        expected.extend(std::iter::repeat_n(0x77u8, 16));
        assert_eq!(storage.tensors["w"], expected);
        assert_eq!(storage.tensors["w.scales"], vec![127, 128]);
        std::fs::remove_dir_all(dir).ok();
    }

    /// The same bytes, declared as an expert bank: rank 3 is rank 2, folded.
    ///
    /// This is the differential the rank-3 encode is worth having. The file is
    /// byte-identical to the test above and so is the answer; only the
    /// DECLARATION changes, from `[2, 32]` to `[2, 1, 32]`. A dense tensor is
    /// row-major, so the two describe the same bytes in the same order, and
    /// every quantizer in the tree indexes `row * cols + c` — which is why
    /// folding the leading axes (`types::rectangle`) is reading the layout
    /// rather than reinterpreting it.
    ///
    /// The scales plane does NOT fold: it comes back `[2, 1, 1]`, the payload's
    /// leading axes with one entry per 32-column block, because that is the
    /// rank an engine binds it at and the shape `model_dsl`'s `Weight::planes`
    /// interns for it.
    ///
    /// Before this, `ScaleLayout::for_encode` refused any rank but 2 and kimi's
    /// mxfp4 expert banks over a bf16 checkpoint could not be compiled at all
    /// (menlo M18's open item).
    #[test]
    fn an_expert_bank_encodes_to_the_same_bytes_as_the_rows_it_stacks() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_bank_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        #[rustfmt::skip]
        let row0: [f32; 32] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
            0.2, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0,
            0.4, 0.6, 1.1, 1.6, 2.2, 3.2, 4.5, 5.5,
        ];
        let mut data = Vec::new();
        for value in row0 {
            data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        for _ in 0..32 {
            data.extend_from_slice(&bf16::from_f32(12.0).to_bits().to_le_bytes());
        }
        let header = r#"{"raw":{"dtype":"BF16","shape":[2,1,32],"data_offsets":[0,128]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&data);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 128,
                shape: vec![2, 1, 32],
                encoding: Encoding::Raw(DType::Bf16),
            }],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(2)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "experts",
                Expr::src("raw").cast(Encoding::Quant(spec.clone())),
                vec![2, 1, 32],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        // The plane the engine binds, at the rank it binds it.
        let scales = plan
            .tensors
            .iter()
            .find(|decl| decl.name == "experts.scales")
            .expect("the encode publishes its scales under the name the plan binds");
        assert_eq!(scales.shape, vec![2, 1, 1]);

        let storage = Execution::new(&plan, &dir).run().unwrap();
        #[rustfmt::skip]
        let expected_row0: [u8; 16] = [
            0x10, 0x32, 0x54, 0x76, 0x90, 0xBA, 0xDC, 0xFE,
            0x10, 0x32, 0x54, 0x76, 0x11, 0x32, 0x54, 0x76,
        ];
        let mut expected = expected_row0.to_vec();
        expected.extend(std::iter::repeat_n(0x77u8, 16));
        assert_eq!(storage.tensors["experts"], expected);
        assert_eq!(storage.tensors["experts.scales"], vec![127, 128]);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Progress reports bytes monotonically against the plan's own total and
    /// names each tensor as it is published.
    #[test]
    fn progress_counts_bytes_and_names_finalized_tensors() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};

        let dir = std::env::temp_dir().join(format!("pie_host_progress_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"raw":{"dtype":"U8","shape":[16],"data_offsets":[0,16]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&[7u8; 16]);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 16,
                shape: vec![16],
                encoding: Encoding::Raw(DType::U8),
            }],
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw"),
                vec![16],
                Encoding::Raw(DType::U8),
            )],
            groups: Vec::new(),
        };
        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();

        let mut seen = Vec::new();
        let storage = Execution::new(&plan, &dir)
            .progress(&mut |progress| {
                seen.push((
                    progress.read_bytes,
                    progress.total_read_bytes,
                    progress.finalized.map(str::to_string),
                ));
            })
            .run()
            .unwrap();
        assert_eq!(storage.tensors["w"], vec![7u8; 16]);

        assert_eq!(seen.len(), plan.schedule.len());
        assert!(seen.windows(2).all(|pair| pair[0].0 <= pair[1].0));
        let last = seen.last().unwrap();
        assert_eq!(last.0, plan.memory.checkpoint_read_bytes);
        assert_eq!(last.1, plan.memory.checkpoint_read_bytes);
        assert!(
            seen.iter().any(|(.., name)| name.as_deref() == Some("w")),
            "no event named the published tensor: {seen:?}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A GGUF Q4_0 tensor decoded to BF16 through a `Cast` contract: the
    /// `Decode` path, compiled and actually run — and refused for any target
    /// that is not the conversion one.
    ///
    /// The two blocks carry different scales and visit both nibble halves
    /// (Q4_0 splits a block low-nibbles-first, not interleaved), so a decoder
    /// that swapped halves, dropped a scale, or forgot the −8 bias cannot
    /// pass.
    #[test]
    fn a_gguf_q4_0_tensor_is_decoded_to_bf16() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_gguf_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Block 0: scale 0.5, every byte 0x9E → low nibbles 14 (+6), high 9
        // (+1): elements 0..16 are 3.0, 16..32 are 0.5. Block 1: scale 2.0,
        // every byte 0x08 → low 8 (0), high 0 (−8): 0.0 then −16.0.
        let mut blocks = Vec::new();
        blocks.extend_from_slice(&[0x00, 0x38]); // f16 0.5
        blocks.extend_from_slice(&[0x9E; 16]);
        blocks.extend_from_slice(&[0x00, 0x40]); // f16 2.0
        blocks.extend_from_slice(&[0x08; 16]);
        std::fs::write(dir.join("model.gguf"), &blocks).unwrap();

        let spec = QuantSpec {
            scheme: QuantScheme::GgufQ4_0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(0)),
        };
        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.gguf".to_string(),
                size_bytes: 36,
                format: crate::types::CheckpointFormat::Gguf,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: 0,
                span_bytes: 36,
                shape: vec![64],
                encoding: Encoding::Quant(spec),
            }],
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Raw(DType::Bf16)),
                vec![64],
                Encoding::Raw(DType::Bf16),
            )],
            groups: Vec::new(),
        };

        // Every non-conversion target refuses the plan at validation — a
        // device has no kernel for a self-scaled block.
        let refused = crate::plan::compile(&metadata, &contract, StorageTarget::default());
        assert!(refused.is_err(), "the host target accepted a Decode");

        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = Execution::new(&plan, &dir).run().unwrap();

        let mut expected = Vec::new();
        for value in std::iter::repeat_n(3.0f32, 16)
            .chain(std::iter::repeat_n(0.5, 16))
            .chain(std::iter::repeat_n(0.0, 16))
            .chain(std::iter::repeat_n(-16.0, 16))
        {
            expected.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Per-channel FP8: `quant_per_channel_kernel`'s row scale and cast,
    /// compiled and actually run.
    #[test]
    fn a_bf16_tensor_is_encoded_to_fp8_per_channel() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_fp8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Row 0's absmax is exactly the E4M3 maximum, so its factor is 1.0
        // and the codes read straight off the format; row 1 halves it, so a
        // factor applied to the wrong row cannot pass.
        let values: [[f32; 4]; 2] = [[448.0, -224.0, 1.0, 0.0], [224.0, 112.0, -56.0, 28.0]];
        let mut data = Vec::new();
        for row in values {
            for value in row {
                data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
            }
        }
        let header = r#"{"raw":{"dtype":"BF16","shape":[2,4],"data_offsets":[0,16]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&data);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 16,
                shape: vec![2, 4],
                encoding: Encoding::Raw(DType::Bf16),
            }],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 0,
            channel_axis: Some(Axis(0)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Quant(spec.clone())),
                vec![2, 4],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = Execution::new(&plan, &dir).run().unwrap();
        assert_eq!(
            storage.tensors["w"],
            vec![0x7E, 0xF6, 0x38, 0x00, 0x7E, 0x76, 0xEE, 0x66]
        );
        let mut scales = Vec::new();
        scales.extend_from_slice(&1.0f32.to_le_bytes());
        scales.extend_from_slice(&0.5f32.to_le_bytes());
        assert_eq!(storage.tensors["w_scale_inv"], scales);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Per-channel INT8: same shape as FP8 with `rintf`'s half-to-even.
    #[test]
    fn a_bf16_tensor_is_encoded_to_int8_per_channel() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_int8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Absmax exactly 127 makes the factor 1.0, so the two half-way values
        // pin the rounding rule: 2.5 → 2 and 3.5 → 4 is ties-to-even, and
        // either a truncation or a half-up rounding fails one of them.
        let values: [f32; 4] = [127.0, 2.5, 3.5, -4.0];
        let mut data = Vec::new();
        for value in values {
            data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        let header = r#"{"raw":{"dtype":"BF16","shape":[1,4],"data_offsets":[0,8]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&data);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![RawTensor {
                id: TensorId(0),
                name: "raw".to_string(),
                file_id: FileId(0),
                file_offset: data_offset,
                span_bytes: 8,
                shape: vec![1, 4],
                encoding: Encoding::Raw(DType::Bf16),
            }],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Int8Symmetric,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 0,
            channel_axis: Some(Axis(0)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "w",
                Expr::src("raw").cast(Encoding::Quant(spec.clone())),
                vec![1, 4],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        let storage = Execution::new(&plan, &dir).run().unwrap();
        assert_eq!(storage.tensors["w"], vec![127, 2, 4, 0xFC]);
        assert_eq!(storage.tensors["w_scale_inv"], 1.0f32.to_le_bytes());
        std::fs::remove_dir_all(dir).ok();
    }

    /// An FP8 block-scaled checkpoint encoded to MXFP4 in one `Cast`: the
    /// instruction names its `_scale_inv` sibling and the executor dequantizes
    /// with it before encoding, the way the device's fused path does.
    #[test]
    fn an_fp8_block_scaled_source_is_dequantized_then_encoded() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::plan::CONVERT_TILE_MAP_MASK;
        use crate::types::{Axis, QuantScheme, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_fp8mx_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // A [64, 64] FP8 weight of ones under a single [1, 1] block factor of
        // 2.0: every dequantized element is 2.0, every MXFP4 group absmax is
        // 2.0 → scale byte 126 (2^-1), and 2.0 / 2^-1 = 4.0 encodes as
        // magnitude 6 in both nibbles. A factor that was dropped — or applied
        // as its reciprocal — produces different bytes everywhere.
        let weight = vec![0x38u8; 64 * 64];
        let factor = 2.0f32.to_le_bytes();
        let header = r#"{"w":{"dtype":"F8_E4M3","shape":[64,64],"data_offsets":[0,4096]},"#
            .to_string()
            + r#""w_scale_inv":{"dtype":"F32","shape":[1,1],"data_offsets":[4096,4100]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&weight);
        file.extend_from_slice(&factor);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 4096,
                    shape: vec![64, 64],
                    encoding: Encoding::Raw(DType::E4m3),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "w_scale_inv".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 4096,
                    span_bytes: 4,
                    shape: vec![1, 1],
                    encoding: Encoding::Raw(DType::F32),
                },
            ],
        };
        let spec = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![TensorContract::new(
                "q",
                Expr::src("w").cast(Encoding::Quant(spec.clone())),
                vec![64, 64],
                Encoding::Quant(spec),
            )],
            groups: Vec::new(),
        };
        let target = StorageTarget {
            tile_map_mask: CONVERT_TILE_MAP_MASK,
            ..StorageTarget::default()
        };

        let plan = crate::plan::compile(&metadata, &contract, target).unwrap();
        assert!(
            plan.instrs.iter().any(|instr| matches!(
                instr,
                StorageInstr::TileMap {
                    kind: TileMapKind::Encode,
                    transform,
                    ..
                } if transform.metadata_source == Some(TensorId(1))
            )),
            "the block-scale tensor did not land on the instruction: {:?}",
            plan.instrs
        );
        let storage = Execution::new(&plan, &dir).run().unwrap();
        assert_eq!(storage.tensors["q"], vec![0x66u8; 64 * 32]);
        assert_eq!(storage.tensors["q.scales"], vec![126u8; 64 * 2]);
        std::fs::remove_dir_all(dir).ok();
    }

    /// `Int4B8` is a different element format under the same `Scale` node.
    ///
    /// Nothing about the node changes -- one operand, one factor tensor, the
    /// same grouping rule -- so what this pins is that the *scheme* is what
    /// says how a code becomes a number: `nibble - 8` here against a lookup
    /// table for MXFP4. Reading one as the other is silent, because both are
    /// four bits packed low nibble first.
    #[test]
    fn an_int4b8_source_is_dequantized_by_its_factors() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_int4b8_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Two rows of 32 codes. Every byte is 0x9E, so the codes alternate
        // 14 (+6) and 9 (+1) -- an asymmetric pair straddling the bias, so
        // neither a swapped nibble order nor a missing `- 8` still passes.
        let packed = [0x9Eu8; 32];
        // bf16 2.0 and -0.5, one per row: different enough that a factor
        // applied to the wrong row cannot go unnoticed, and signed so that a
        // factor read as unsigned cannot either.
        let mut factors = Vec::new();
        for value in [2.0f32, -0.5] {
            factors.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }

        let header = r#"{"w":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},"#.to_string()
            + r#""s":{"dtype":"BF16","shape":[2,1],"data_offsets":[32,36]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&packed);
        file.extend_from_slice(&factors);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 32,
                    shape: vec![2, 16],
                    encoding: Encoding::Raw(DType::U8),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 32,
                    span_bytes: 4,
                    shape: vec![2, 1],
                    encoding: Encoding::Raw(DType::Bf16),
                },
            ],
        };

        let int4b8 = QuantSpec {
            scheme: QuantScheme::Int4B8,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s"),
                    vec![2, 1],
                    Encoding::Raw(DType::Bf16),
                ),
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![2, 32],
                            encoding: Encoding::Quant(int4b8),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![2, 32],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        let storage = Execution::new(&plan, &dir).run().unwrap();

        let mut expected = Vec::new();
        for row_scale in [2.0f32, -0.5] {
            for _ in 0..16 {
                expected
                    .extend_from_slice(&bf16::from_f32(6.0 * row_scale).to_bits().to_le_bytes());
                expected
                    .extend_from_slice(&bf16::from_f32(1.0 * row_scale).to_bits().to_le_bytes());
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// A block that spans rows *and* columns -- the shape GLM-5.1's FP8
    /// `kv_b_proj` ships and the one thing the old `group`/`axis` pair could
    /// not name.
    ///
    /// A single group size along a single axis can only ever describe a
    /// one-dimensional blocking. Deriving it from the ratio of the two shapes
    /// makes the two-dimensional case fall out with no new field, so this is
    /// the test that the generalization is real rather than a rename: the
    /// factors are `[2, 2]` over a `[4, 4]` payload, i.e. 2x2 tiles, and each
    /// of the four is distinct so a wrong index cannot land on the right
    /// number.
    ///
    /// FP8 elements on purpose. It is the format the 2-D blocking arrives in,
    /// and the values (1, 2, 3, ... 16) are exactly representable in E4M3, so
    /// a decode that is off by an exponent shows up as a factor of two rather
    /// than a rounding difference.
    #[test]
    fn a_two_dimensional_block_indexes_its_factors_by_row_and_column() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_block2d_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // E4M3 for 1..=16: exponent = floor(log2(v)) + 7, mantissa = the three
        // bits below the leading one.
        let encode = |value: f64| -> u8 {
            let exponent = value.log2().floor() as i32;
            let mantissa = (value / f64::from(exponent).exp2() - 1.0) * 8.0;
            (((exponent + 7) as u8) << 3) | (mantissa.round() as u8)
        };
        let payload: Vec<u8> = (1..=16).map(|v| encode(f64::from(v))).collect();
        let factors: Vec<f32> = vec![1.0, 10.0, 100.0, 1000.0];
        let mut factor_bytes = Vec::new();
        for factor in &factors {
            factor_bytes.extend_from_slice(&factor.to_le_bytes());
        }

        let header = r#"{"w":{"dtype":"F8_E4M3","shape":[4,4],"data_offsets":[0,16]},"#.to_string()
            + r#""s":{"dtype":"F32","shape":[2,2],"data_offsets":[16,32]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&payload);
        file.extend_from_slice(&factor_bytes);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 16,
                    shape: vec![4, 4],
                    encoding: Encoding::Raw(DType::E4m3),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 16,
                    span_bytes: 16,
                    shape: vec![2, 2],
                    encoding: Encoding::Raw(DType::F32),
                },
            ],
        };

        let fp8 = QuantSpec {
            scheme: QuantScheme::Fp8E4M3,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 2,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s"),
                    vec![2, 2],
                    Encoding::Raw(DType::F32),
                )
                .internal(),
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![4, 4],
                            encoding: Encoding::Quant(fp8),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![4, 4],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        assert!(
            plan.instrs.iter().any(|instr| matches!(
                instr,
                StorageInstr::TileMap { transform, .. } if transform.scale_blocks == vec![2, 2]
            )),
            "the blocking is derived at plan time: {:?}",
            plan.instrs
        );
        let storage = Execution::new(&plan, &dir).run().unwrap();

        let mut expected = Vec::new();
        for row in 0..4i64 {
            for col in 0..4i64 {
                let value = f64::from((row * 4 + col + 1) as i32);
                let factor = factors[((row / 2) * 2 + col / 2) as usize];
                expected.extend_from_slice(
                    &bf16::from_f32(value as f32 * factor)
                        .to_bits()
                        .to_le_bytes(),
                );
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// Factors a contract needed but the engine does not: declared, used,
    /// never bound.
    ///
    /// The algebra has no `let`, so scaling by a tensor means publishing that
    /// tensor. Published as a runtime weight, a slab of dequantization factors
    /// lands in the persistent arena and stays there for the life of the
    /// process -- an arena view reclaims nothing when erased -- and the engine
    /// gets a name in its bind table that no kernel will ever ask for.
    ///
    /// `Visibility::Internal` is that name without either consequence. What
    /// this pins is that both go away together and the arithmetic does not
    /// change: the same bytes come out of `w`, `scales` is absent from the
    /// bind table, and the plan's persistent footprint drops by exactly the
    /// factors it no longer keeps.
    #[test]
    fn internal_factors_are_used_but_never_bound() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_internal_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let packed = [0x9Eu8; 32];
        let mut factors = Vec::new();
        for value in [2.0f32, -0.5] {
            factors.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
        }
        let header = r#"{"w":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},"#.to_string()
            + r#""s":{"dtype":"BF16","shape":[2,1],"data_offsets":[32,36]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&packed);
        file.extend_from_slice(&factors);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 32,
                    shape: vec![2, 16],
                    encoding: Encoding::Raw(DType::U8),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 32,
                    span_bytes: 4,
                    shape: vec![2, 1],
                    encoding: Encoding::Raw(DType::Bf16),
                },
            ],
        };

        let int4b8 = QuantSpec {
            scheme: QuantScheme::Int4B8,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = |scales: TensorContract| ModelContract {
            alignment: 1,
            tensors: vec![
                scales,
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![2, 32],
                            encoding: Encoding::Quant(int4b8.clone()),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![2, 32],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };
        let declare_scales = || {
            TensorContract::new(
                "scales",
                Expr::src("s"),
                vec![2, 1],
                Encoding::Raw(DType::Bf16),
            )
        };

        let public = crate::plan::compile(
            &metadata,
            &contract(declare_scales()),
            StorageTarget::default(),
        )
        .unwrap();
        let internal = crate::plan::compile(
            &metadata,
            &contract(declare_scales().internal()),
            StorageTarget::default(),
        )
        .unwrap();

        let storage = Execution::new(&internal, &dir).run().unwrap();
        let mut expected = Vec::new();
        for row_scale in [2.0f32, -0.5] {
            for _ in 0..16 {
                expected
                    .extend_from_slice(&bf16::from_f32(6.0 * row_scale).to_bits().to_le_bytes());
                expected
                    .extend_from_slice(&bf16::from_f32(1.0 * row_scale).to_bits().to_le_bytes());
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        assert!(
            !storage.tensors.contains_key("scales"),
            "an internal declaration must not reach the engine's bind table"
        );
        // 2 rows x 1 bf16 factor: the whole of what the public declaration was
        // keeping. Stated exactly, because "smaller" would also pass if the
        // arena had merely stopped aligning it.
        assert_eq!(
            public.memory.persistent_bytes - internal.memory.persistent_bytes,
            4
        );
        std::fs::remove_dir_all(dir).ok();
    }

    /// A block-scaled MXFP4 tensor and its factors, dequantized by the plan.
    ///
    /// This is the shape the DeepSeek families' expert weights arrive in, and
    /// the reason `Scale` takes a tensor factor at all: before it did, an engine
    /// loaded both halves, ran its own kernel, and left the packed originals
    /// resident because a view into the arena cannot be freed.
    #[test]
    fn a_block_scaled_source_is_dequantized_by_its_factors() {
        use crate::file::{File, Metadata, RawTensor};
        use crate::contract::{Expr, ModelContract, TensorContract, TensorType};
        use crate::types::{Axis, QuantSpec};

        let dir = std::env::temp_dir().join(format!("pie_host_mxfp4_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        // Two rows of 32 codes. Every byte is 0x21, so the codes alternate
        // 1 (0.5) and 2 (1.0) — an asymmetric pair, so a nibble order that
        // swapped low for high would not still pass.
        let packed = [0x21u8; 32];
        // 2^(128-127) = 2 on row 0, 2^(126-127) = 0.5 on row 1: different
        // enough that a factor applied to the wrong row cannot go unnoticed.
        let exponents = [128u8, 126];

        let header = r#"{"w":{"dtype":"U8","shape":[2,16],"data_offsets":[0,32]},"#.to_string()
            + r#""s":{"dtype":"U8","shape":[2,1],"data_offsets":[32,34]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&packed);
        file.extend_from_slice(&exponents);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                RawTensor {
                    id: TensorId(0),
                    name: "w".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset,
                    span_bytes: 32,
                    shape: vec![2, 16],
                    encoding: Encoding::Raw(DType::U8),
                },
                RawTensor {
                    id: TensorId(1),
                    name: "s".to_string(),
                    file_id: FileId(0),
                    file_offset: data_offset + 32,
                    span_bytes: 2,
                    shape: vec![2, 1],
                    encoding: Encoding::Raw(DType::U8),
                },
            ],
        };

        let mxfp4 = QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: Some(Axis(1)),
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![
                TensorContract::new(
                    "scales",
                    Expr::src("s").transmute(TensorType {
                        shape: vec![2, 1],
                        encoding: Encoding::Raw(DType::E8m0),
                    }),
                    vec![2, 1],
                    Encoding::Raw(DType::E8m0),
                ),
                TensorContract::new(
                    "w",
                    Expr::src("w")
                        .transmute(TensorType {
                            shape: vec![2, 32],
                            encoding: Encoding::Quant(mxfp4),
                        })
                        .scale_per_block(Expr::out("scales")),
                    vec![2, 32],
                    Encoding::Raw(DType::Bf16),
                ),
            ],
            groups: Vec::new(),
        };

        let plan = crate::plan::compile(&metadata, &contract, StorageTarget::default()).unwrap();
        let storage = Execution::new(&plan, &dir).run().unwrap();

        let mut expected = Vec::new();
        for row_scale in [2.0f32, 0.5] {
            for _ in 0..16 {
                expected
                    .extend_from_slice(&bf16::from_f32(0.5 * row_scale).to_bits().to_le_bytes());
                expected
                    .extend_from_slice(&bf16::from_f32(1.0 * row_scale).to_bits().to_le_bytes());
            }
        }
        assert_eq!(storage.tensors["w"], expected);
        std::fs::remove_dir_all(dir).ok();
    }

    /// The three shapes `Execution` offers still build, and the live one is the
    /// third.
    ///
    /// THE LIVE SHAPE IS `streaming().sink().progress()`, and `pie model
    /// import` is the only production caller of any of them — twice, once for
    /// the spool and once for the merge. Its progress closure captures locals
    /// mutably WHILE a `&mut` sink is held by the same builder, which is the
    /// one thing about the chain that can fail to borrow-check.
    ///
    /// It said "four shapes", named `pie model build` as the second caller, and
    /// added that the binary writing them "cannot be compiled from here (it does
    /// not build on this branch)". None of the three was true. There are three
    /// shapes, not four. `pie model build` is deleted, not unbuildable — R3
    /// retired it with the load contract its transforms authored. And the
    /// binary that does write the live shape is `pie`, a workspace member that
    /// `cargo check --workspace` compiles like any other.
    ///
    /// The first two shapes are pinned here because pinning them costs three
    /// lines and NOTHING IN PRODUCTION EXERCISES THEM: bare `run()` and
    /// `arena().sink()` are read by this crate's own tests and by
    /// `engine-metal`, which is out of the workspace. They are public API, so
    /// the borrow-check property is worth a caller somewhere, and here is the
    /// only place it currently has one that runs.
    #[test]
    fn every_caller_shape_of_the_builder_compiles() {
        let (dir, plan) = fixture();

        // `execute_plan`: allocate the arena, keep every tensor.
        let storage = Execution::new(&plan, &dir).run().unwrap();
        assert!(!storage.arena.is_empty(), "the caller took no arena");
        assert!(!storage.tensors.is_empty(), "and no sink");

        // `execute_plan_into_arena`: the caller's host arena, the caller's sink.
        let mut arena = vec![0u8; plan.memory.persistent_bytes as usize];
        let mut sink = MemorySink::default();
        let taken = Execution::new(&plan, &dir)
            .arena(&mut &mut arena[..])
            .sink(&mut sink)
            .run()
            .unwrap();
        assert_eq!(
            taken,
            HostStorage::default(),
            "a caller who supplied both holds the results already"
        );
        assert_eq!(arena, storage.arena, "and gets the same bytes");

        // `execute_plan_into`: streaming, with a progress closure that mutates
        // two captured locals while `sink` is borrowed by the same builder.
        let mut streamed = MemorySink::default();
        let mut seen = 0usize;
        let mut last = 0u64;
        Execution::new(&plan, &dir)
            .streaming()
            .sink(&mut streamed)
            .progress(&mut |progress| {
                seen += 1;
                last = progress.read_bytes;
            })
            .run()
            .unwrap();
        assert_eq!(seen, plan.schedule.len());
        assert_eq!(last, plan.memory.checkpoint_read_bytes);
        assert_eq!(
            streamed.tensors, storage.tensors,
            "streaming publishes the same tensors as the resident path"
        );

        std::fs::remove_dir_all(dir).ok();
    }

    /// **A DEVICE-TARGETED PLAN CAN BE STREAMED — BY BEING COMPILED FOR IT.**
    ///
    /// The refusal above this one is real and stays: `BulkExtentWrite`
    /// addresses an arena by offset and a streaming execution has none. What
    /// changes is that a caller who wants the device's plan and the host's
    /// memory shape no longer has to choose between them. It compiles the same
    /// contract against the same backend through
    /// [`crate::plan::compile_streaming`], which runs the pipeline without the
    /// two passes that exist to serve the arena, and gets a schedule the
    /// streaming residency can run.
    ///
    /// ```text
    ///  1. shaped    -> the ordinary CUDA plan carries `BulkExtentWrite`s and
    ///                  the streaming one carries none
    ///  2. refused   -> streaming the ordinary one still fails, and the
    ///                  sentence says what to do about it
    ///  3. identical -> the streaming run of the streaming plan publishes the
    ///                  same tensors, byte for byte, as the arena run of the
    ///                  ordinary one — which is the whole correctness claim,
    ///                  because a load's artifact is a function of what its
    ///                  sink was handed
    ///  4. freed     -> and it is not the same plan with the instruction
    ///                  renamed: the hoist that emits those writes also pulls
    ///                  every `Allocate` to the head of the schedule, and
    ///                  under that arrangement "freed at its last use" frees
    ///                  nothing until the whole model is live at once. The
    ///                  streaming plan finalizes its first tensor before it
    ///                  allocates its last buffer.
    /// ```
    #[test]
    fn a_device_plan_compiled_for_streaming_runs_without_an_arena() {
        use crate::contract::{Expr, ModelContract, TensorContract};
        use crate::file::{File, Metadata, RawTensor};
        use crate::types::BackendKind;

        let dir = std::env::temp_dir().join(format!(
            "pie_streaming_device_plan_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // Three dense passthroughs and one cast, so the plan carries both the
        // writes the coalescer rewrites and a transform it must leave alone.
        let header = r#"{"a":{"dtype":"U8","shape":[64],"data_offsets":[0,64]},"#.to_string()
            + r#""b":{"dtype":"U8","shape":[64],"data_offsets":[64,128]},"#
            + r#""c":{"dtype":"F16","shape":[8],"data_offsets":[128,144]},"#
            + r#""d":{"dtype":"U8","shape":[64],"data_offsets":[144,208]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend((0..64u8).map(|byte| byte.wrapping_mul(3).wrapping_add(1)));
        file.extend((0..64u8).map(|byte| byte.wrapping_mul(5).wrapping_add(2)));
        for value in 0..8u16 {
            let element = half::f16::from_f32(f32::from(value) + 0.5);
            file.extend_from_slice(&element.to_bits().to_le_bytes());
        }
        file.extend((0..64u8).map(|byte| byte.wrapping_mul(7).wrapping_add(3)));
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), &file).unwrap();

        let raw = |id: u32, name: &str, at: u64, span: u64, shape: Vec<i64>, encoding| RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: data_offset + at,
            span_bytes: span,
            shape,
            encoding,
        };
        let metadata = Metadata {
            files: vec![File {
                id: FileId(0),
                path: "model.safetensors".to_string(),
                size_bytes,
                format: crate::types::CheckpointFormat::Safetensors,
            }],
            tensors: vec![
                raw(0, "a", 0, 64, vec![64], Encoding::Raw(DType::U8)),
                raw(1, "b", 64, 64, vec![64], Encoding::Raw(DType::U8)),
                raw(2, "c", 128, 16, vec![8], Encoding::Raw(DType::F16)),
                raw(3, "d", 144, 64, vec![64], Encoding::Raw(DType::U8)),
            ],
        };
        let contract = ModelContract {
            alignment: 1,
            tensors: vec![
                TensorContract::new("w_a", Expr::src("a"), vec![64], Encoding::Raw(DType::U8)),
                TensorContract::new("w_b", Expr::src("b"), vec![64], Encoding::Raw(DType::U8)),
                TensorContract::new(
                    "w_c",
                    Expr::src("c").cast(Encoding::Raw(DType::Bf16)),
                    vec![8],
                    Encoding::Raw(DType::Bf16),
                ),
                TensorContract::new("w_d", Expr::src("d"), vec![64], Encoding::Raw(DType::U8)),
            ],
            groups: Vec::new(),
        };

        let target = StorageTarget::for_backend(BackendKind::Cuda, 0, 1);
        let arena_plan = crate::plan::compile(&metadata, &contract, target.clone()).unwrap();
        let stream_plan = crate::plan::compile_streaming(&metadata, &contract, target).unwrap();

        // ── (1) THE TWO SHAPES.
        let bulk = |plan: &LoadPlan| {
            plan.instrs
                .iter()
                .filter(|instr| matches!(instr, StorageInstr::BulkExtentWrite { .. }))
                .count()
        };
        assert!(
            bulk(&arena_plan) > 0,
            "a CUDA-targeted plan coalesces its dense persistent writes, or this \
             test is not testing the thing it names"
        );
        assert_eq!(
            bulk(&stream_plan),
            0,
            "the streaming pipeline leaves the coalescer out, so nothing addresses \
             an arena by offset"
        );
        assert!(
            !stream_plan
                .passes
                .iter()
                .any(|pass| pass.pass == "coalesce-persistent-arena-writes"
                    || pass.pass == "hoist-bulk-arena-writes"),
            "and it says so in the passes it ran: {:?}",
            stream_plan.passes.iter().map(|pass| &pass.pass).collect::<Vec<_>>()
        );
        assert_eq!(
            arena_plan.passes.len(),
            stream_plan.passes.len() + 2,
            "and it leaves out exactly those two"
        );

        // ── (2) THE REFUSAL THE OTHER PLAN STILL EARNS.
        let mut refused = MemorySink::default();
        let error = Execution::new(&arena_plan, &dir)
            .streaming()
            .sink(&mut refused)
            .run()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("no persistent arena") && error.contains("compile_streaming"),
            "{error}"
        );

        // ── (3) AND THE SAME BYTES OUT OF BOTH.
        let mut arena = vec![0u8; arena_plan.memory.arena_bytes() as usize];
        let mut landed = MemorySink::default();
        Execution::new(&arena_plan, &dir)
            .arena(&mut &mut arena[..])
            .sink(&mut landed)
            .run()
            .unwrap();
        let mut streamed = MemorySink::default();
        Execution::new(&stream_plan, &dir)
            .streaming()
            .sink(&mut streamed)
            .run()
            .unwrap();
        assert_eq!(
            streamed.tensors, landed.tensors,
            "the arena is where the bytes waited, never what they were"
        );
        assert_eq!(streamed.tensors.len(), 4);
        assert_eq!(
            streamed.tensors["w_a"],
            (0..64u8).map(|byte| byte.wrapping_mul(3).wrapping_add(1)).collect::<Vec<_>>()
        );

        // ── (4) AND A SCHEDULE THAT CAN FREE AS IT GOES.
        let positions = |plan: &LoadPlan, want: fn(&StorageInstr) -> bool| -> Vec<usize> {
            plan.schedule
                .iter()
                .enumerate()
                .filter(|(_, id)| want(instr_by_id(&plan.instrs, **id).unwrap()))
                .map(|(at, _)| at)
                .collect()
        };
        let allocates = |instr: &StorageInstr| matches!(instr, StorageInstr::Allocate { .. });
        let finalizes = |instr: &StorageInstr| matches!(instr, StorageInstr::Finalize { .. });
        assert!(
            positions(&arena_plan, allocates).iter().max().unwrap()
                < positions(&arena_plan, finalizes).iter().min().unwrap(),
            "the hoist puts every Allocate ahead of every Finalize — which is what \
             makes streaming this plan hold the whole model at once even where the \
             instruction it refuses is not the problem"
        );
        assert!(
            positions(&stream_plan, finalizes).iter().min().unwrap()
                < positions(&stream_plan, allocates).iter().max().unwrap(),
            "and the streaming plan publishes its first tensor before it allocates \
             its last buffer, which is the property the freeing rests on"
        );

        std::fs::remove_dir_all(dir).ok();
    }

    fn fixture() -> (PathBuf, LoadPlan) {
        let dir = std::env::temp_dir().join(format!(
            "pie_host_storage_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let header = r#"{"raw":{"dtype":"U8","shape":[6],"data_offsets":[0,6]}}"#;
        let mut file = (header.len() as u64).to_le_bytes().to_vec();
        file.extend_from_slice(header.as_bytes());
        let data_offset = file.len() as u64;
        file.extend_from_slice(&[99, 1, 2, 99, 3, 4]);
        let size_bytes = file.len() as u64;
        std::fs::write(dir.join("model.safetensors"), file).unwrap();

        let mut program = LoadPlan::empty(StorageTarget {
            max_tile_bytes: 2,
            preferred_alignment: 8,
            ..StorageTarget::default()
        });
        // The plan names its own files; the executor does not go looking.
        // Relative, so `snapshot_dir` still has a job.
        program.files.push(crate::plan::CheckpointFileDecl {
            id: FileId(0),
            path: "model.safetensors".to_string(),
            size_bytes,
            format: crate::types::CheckpointFormat::Safetensors,
        });
        program.sources.push(SourceTensorDecl {
            id: TensorId(0),
            name: "raw".to_string(),
            file_id: FileId(0),
            file_offset: data_offset,
            span_bytes: 6,
            shape: vec![2, 3],
            encoding: Encoding::Raw(DType::U8),
        });
        program.tensors = vec![
            TensorDecl {
                id: TensorId(0),
                name: "selected".to_string(),
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::U8),
                alignment: 8,
                visibility: crate::types::Visibility::Public,
            },
            TensorDecl {
                id: TensorId(1),
                name: "cast".to_string(),
                shape: vec![2, 2],
                encoding: Encoding::Raw(DType::U16),
                alignment: 8,
                visibility: crate::types::Visibility::Public,
            },
        ];
        program.buffers = vec![
            BufferDecl {
                id: BufferId(0),
                tensor: Some(TensorId(0)),
                ty: crate::contract::TensorType::raw(vec![2, 2], DType::U8),
                bytes: 4,
                alignment: 8,
                temporary: false,
                persistent_offset: Some(0),
                scratch_offset: None,
            },
            BufferDecl {
                id: BufferId(1),
                tensor: Some(TensorId(1)),
                ty: crate::contract::TensorType::raw(vec![2, 2], DType::U16),
                bytes: 8,
                alignment: 8,
                temporary: false,
                persistent_offset: Some(8),
                scratch_offset: None,
            },
        ];
        program.instrs = vec![
            StorageInstr::Allocate {
                id: InstrId(0),
                buffer: BufferId(0),
            },
            StorageInstr::Allocate {
                id: InstrId(1),
                buffer: BufferId(1),
            },
            StorageInstr::ExtentWrite {
                id: InstrId(2),
                source: SourceExtent {
                    file_id: FileId(0),
                    tensor_id: TensorId(0),
                    file_offset: data_offset,
                    span_bytes: 4,
                    stride: extent(1, 1, &[(2, 3, 2), (2, 1, 1)]),
                    dtype: DType::U16,
                },
                dest: DestExtent {
                    buffer: BufferId(0),
                    offset: 0,
                    stride: extent(0, 1, &[(2, 2, 2), (2, 1, 1)]),
                },
            },
            StorageInstr::TileMap {
                id: InstrId(3),
                kind: TileMapKind::Cast,
                source: None,
                dest: None,
                inputs: vec![BufferId(0)],
                outputs: vec![BufferId(1)],
                tile: TileSpec {
                    max_tile_bytes: 2,
                },
                transform: TransformSpec::default(),
            },
            StorageInstr::Finalize {
                id: InstrId(4),
                tensor: BufferId(1),
                name: "cast".to_string(),
            },
        ];
        program.schedule = (0..5).map(InstrId).collect();
        program.memory = MemoryPlan {
            persistent_bytes: 16,
            checkpoint_read_bytes: 4,
            device_write_bytes: 12,
            ..MemoryPlan::default()
        };
        (dir, program)
    }

    #[test]
    fn executes_place_strided_cast_and_tiled_writes() {
        let (dir, program) = fixture();
        let storage = Execution::new(&program, &dir).run().unwrap();
        let values = storage.tensors["cast"]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1, 2, 3, 4]);
        assert_eq!(&storage.arena[..4], &[1, 2, 3, 4]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_noncompact_destination_extents() {
        let (dir, mut program) = fixture();
        let StorageInstr::ExtentWrite { dest, .. } = &mut program.instrs[2] else {
            panic!("fixture instruction changed");
        };
        dest.stride = extent(0, 1, &[(2, 2, 3), (2, 1, 1)]);
        let error = Execution::new(&program, &dir)
            .run()
            .unwrap_err()
            .to_string();
        assert!(error.contains("non-compact ExtentWrite destination"));
        std::fs::remove_dir_all(dir).ok();
    }

    /// The compiler states the source as coarsely as it can, so a contiguous
    /// read arrives as one dim of wide elements while the destination it lands
    /// in is described element by element. Requiring the two decompositions to
    /// agree rejected every sharded plan — `pie-loader replay` on a real
    /// checkpoint at tp 1/2 failed outright — even though both sides named the
    /// same bytes.
    #[test]
    fn accepts_a_source_factored_differently_from_its_destination() {
        let (dir, mut program) = fixture();
        let StorageInstr::ExtentWrite { source, dest, .. } = &mut program.instrs[2] else {
            panic!("fixture instruction changed");
        };
        // Same four bytes, described as one 4-byte element instead of four.
        source.stride = extent(0, 4, &[(1, 1, 1)]);
        assert_eq!(dest.stride.element_bytes, 1);
        assert_eq!(dest.stride.dims.iter().map(|d| d.count).sum::<i64>(), 4);
        Execution::new(&program, &dir)
            .run()
            .expect("equal byte counts should be accepted");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_a_source_and_destination_of_different_byte_counts() {
        let (dir, mut program) = fixture();
        let StorageInstr::ExtentWrite { source, .. } = &mut program.instrs[2] else {
            panic!("fixture instruction changed");
        };
        source.stride = extent(0, 1, &[(3, 1, 1)]);
        let error = Execution::new(&program, &dir)
            .run()
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("bytes but the destination spans"),
            "unexpected error: {error}"
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn casts_direct_checkpoint_extents() {
        let (dir, mut plan) = fixture();
        let file_offset = plan.sources[0].file_offset;
        let StorageInstr::TileMap { source, inputs, .. } = &mut plan.instrs[3] else {
            panic!("fixture instruction changed");
        };
        *source = Some(SourceExtent {
            file_id: FileId(0),
            tensor_id: TensorId(0),
            file_offset,
            span_bytes: 4,
            stride: extent(1, 1, &[(2, 3, 2), (2, 1, 1)]),
            dtype: DType::U16,
        });
        inputs.clear();
        let storage = Execution::new(&plan, &dir).run().unwrap();
        let values = storage.tensors["cast"]
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(values, vec![1, 2, 3, 4]);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn rejects_unsupported_advertised_transforms() {
        // The gate is the host's own implementation, and nothing else. It used
        // to be widened by the arena backing's capability mask; the widening
        // never admitted a plan, because every mask a target can carry is a
        // subset of `CONVERT_TILE_MAP_MASK`. `Transcode` is outside it.
        let (dir, mut plan) = fixture();
        plan.target.tile_map_mask |= crate::plan::TILE_MAP_TRANSCODE;
        let error = Execution::new(&plan, &dir).run().unwrap_err().to_string();
        assert!(
            error.contains("TileMap transforms the host does not implement"),
            "{error}"
        );
        std::fs::remove_dir_all(dir).ok();
    }
}

/// The GGUF block decoders, pinned by layout rather than by value.
///
/// Every one of these is a bit-gather, and a bit-gather's failure mode is not
/// a wrong number but a right number in the wrong place — which no shape, size
/// or sum check can see. So the payloads below are chosen so that each decoded
/// element reports *where its bits came from*, and the assertions are about
/// position.
///
/// The values themselves were checked separately and are not re-derived here:
/// every decoder was run against a from-scratch reimplementation over the
/// tensors of `qwen2.5-0.5b-instruct-q4_k_m.gguf` and `-q5_k_m.gguf`, and
/// agreed bit for bit on 8.9M elements of Q4_K, Q5_K, Q6_K, Q5_0, Q5_1 and
/// Q8_0. Q4_1 is the one scheme no file at hand uses, so it is held only by
/// the layout tests here.
#[cfg(test)]
mod gguf_block_tests {
    use super::*;

    fn f16(value: f32) -> [u8; 2] {
        half::f16::from_f32(value).to_le_bytes()
    }

    /// A 32-element block splits its sixteen payload bytes into low nibbles
    /// first and high nibbles second — not byte order, not interleaved.
    ///
    /// The payload is a ramp, so byte `i` carries `i` in its low nibble and
    /// zero in its high one. An implementation that emitted the two nibbles of
    /// a byte adjacently would produce the same multiset of values and fail
    /// only here.
    #[test]
    fn a_thirty_two_element_block_puts_every_low_nibble_before_any_high_one() {
        let ramp: Vec<u8> = (0..16u8).collect();
        let mut values = [0.0f32; 32];

        let mut q4_0 = [0u8; 18];
        q4_0[..2].copy_from_slice(&f16(1.0));
        q4_0[2..].copy_from_slice(&ramp);
        decode_gguf_q4_0_block_into(&q4_0, &mut values);
        for i in 0..16 {
            assert_eq!(values[i], i as f32 - 8.0, "q4_0 low nibble {i}");
            assert_eq!(values[i + 16], -8.0, "q4_0 high nibble {i}");
        }

        // The `_1` schemes add their offset; a sign error here is invisible to
        // every other check because the block still decodes to plausible
        // weights, just mirrored about the offset.
        let mut q4_1 = [0u8; 20];
        q4_1[..2].copy_from_slice(&f16(1.0));
        q4_1[2..4].copy_from_slice(&f16(2.0));
        q4_1[4..].copy_from_slice(&ramp);
        decode_gguf_q4_1_block_into(&q4_1, &mut values);
        for i in 0..16 {
            assert_eq!(values[i], i as f32 + 2.0, "q4_1 low nibble {i}");
            assert_eq!(values[i + 16], 2.0, "q4_1 high nibble {i}");
        }

        // Q8_0 has no nibbles at all, so this is the control: if the ordering
        // assertions above ever pass for the wrong reason, this one still
        // holds and tells them apart. Its ramp straddles zero because Q8_0's
        // codes are SIGNED, and a ramp of small positive bytes reads the same
        // either way.
        let mut q8_0 = [0u8; 34];
        q8_0[..2].copy_from_slice(&f16(1.0));
        for (i, byte) in q8_0[2..].iter_mut().enumerate() {
            *byte = (i as i8 - 16) as u8;
        }
        decode_gguf_q8_0_block_into(&q8_0, &mut values);
        for (i, value) in values.iter().enumerate() {
            assert_eq!(*value, i as f32 - 16.0, "q8_0 element {i}");
        }
    }

    /// The fifth-bit plane of a `_0`/`_1` block is indexed by ELEMENT — bit
    /// `i` for the low nibble of byte `i`, bit `i + 16` for its high one.
    ///
    /// Two planes, each all-ones in one half. Reading the plane in packing
    /// order instead (bits `2i` and `2i + 1`, which is what the K-quants do)
    /// splits both halves down the middle rather than choosing one, so the
    /// two conventions disagree on every element here.
    #[test]
    fn a_fifth_bit_plane_is_indexed_by_element_and_not_by_packing_order() {
        let ramp: Vec<u8> = (0..16u8).collect();
        let mut values = [0.0f32; 32];

        for (plane, low_carries, high_carries) in
            [(0x0000_ffffu32, true, false), (0xffff_0000u32, false, true)]
        {
            let mut q5_0 = [0u8; 22];
            q5_0[..2].copy_from_slice(&f16(1.0));
            q5_0[2..6].copy_from_slice(&plane.to_le_bytes());
            q5_0[6..].copy_from_slice(&ramp);
            decode_gguf_q5_0_block_into(&q5_0, &mut values);
            for i in 0..16 {
                let low = i as f32 + if low_carries { 16.0 } else { 0.0 } - 16.0;
                let high = if high_carries { 16.0 } else { 0.0 } - 16.0;
                assert_eq!(values[i], low, "q5_0 plane {plane:#x} low {i}");
                assert_eq!(values[i + 16], high, "q5_0 plane {plane:#x} high {i}");
            }

            let mut q5_1 = [0u8; 24];
            q5_1[..2].copy_from_slice(&f16(1.0));
            q5_1[2..4].copy_from_slice(&f16(2.0));
            q5_1[4..8].copy_from_slice(&plane.to_le_bytes());
            q5_1[8..].copy_from_slice(&ramp);
            decode_gguf_q5_1_block_into(&q5_1, &mut values);
            for i in 0..16 {
                let low = i as f32 + if low_carries { 16.0 } else { 0.0 } + 2.0;
                let high = if high_carries { 16.0 } else { 0.0 } + 2.0;
                assert_eq!(values[i], low, "q5_1 plane {plane:#x} low {i}");
                assert_eq!(values[i + 16], high, "q5_1 plane {plane:#x} high {i}");
            }
        }
    }

    /// A K-quant's last four sub-blocks read their scale and minimum from the
    /// spliced bytes, and the splice does not give the same answer as the
    /// straight read.
    ///
    /// Twelve scale bytes of `0x01` are the cheapest payload that separates
    /// the two branches of `gguf_k_scale_min`: sub-blocks 0..4 come out
    /// `(scale 1, min 1)` and sub-blocks 4..8 come out `(scale 1, min 0)`,
    /// because the spliced minimum takes its low four bits from the high
    /// nibble of a byte that holds `0x01`. So the decoded block steps from
    /// `nibble − 1` to `nibble` exactly at element 128, and an implementation
    /// that ran the `j < 4` branch for all eight sub-blocks would hold `−1`
    /// throughout.
    #[test]
    fn a_k_quant_splices_the_scales_of_its_last_four_sub_blocks() {
        let mut values = [0.0f32; 256];

        let mut q4_k = [0u8; 144];
        q4_k[..2].copy_from_slice(&f16(1.0));
        q4_k[2..4].copy_from_slice(&f16(1.0));
        q4_k[4..16].copy_from_slice(&[0x01; 12]);
        for (i, byte) in q4_k[16..].iter_mut().enumerate() {
            *byte = (i & 0x0f) as u8;
        }
        decode_gguf_q4_k_block_into(&q4_k, &mut values);
        for pair in 0..4 {
            // Sub-block 2·pair carries the low nibbles, 2·pair + 1 the high
            // ones, which the ramp leaves at zero.
            let minimum = if pair < 2 { 1.0 } else { 0.0 };
            for i in 0..32 {
                assert_eq!(
                    values[pair * 64 + i],
                    (i & 0x0f) as f32 - minimum,
                    "q4_k pair {pair} low {i}"
                );
                assert_eq!(
                    values[pair * 64 + 32 + i],
                    -minimum,
                    "q4_k pair {pair} high {i}"
                );
            }
        }

        // Q5_K adds the plane, and reads it by PAIR rather than by element:
        // bit 2·pair for the low nibbles, 2·pair + 1 for the high ones. A
        // plane of `0b11` therefore lifts pair 0 alone.
        let mut q5_k = [0u8; 176];
        q5_k[..2].copy_from_slice(&f16(1.0));
        q5_k[2..4].copy_from_slice(&f16(1.0));
        q5_k[4..16].copy_from_slice(&[0x01; 12]);
        q5_k[16..48].copy_from_slice(&[0b11; 32]);
        for (i, byte) in q5_k[48..].iter_mut().enumerate() {
            *byte = (i & 0x0f) as u8;
        }
        decode_gguf_q5_k_block_into(&q5_k, &mut values);
        for pair in 0..4 {
            let minimum = if pair < 2 { 1.0 } else { 0.0 };
            let fifth = if pair == 0 { 16.0 } else { 0.0 };
            for i in 0..32 {
                assert_eq!(
                    values[pair * 64 + i],
                    (i & 0x0f) as f32 + fifth - minimum,
                    "q5_k pair {pair} low {i}"
                );
                assert_eq!(
                    values[pair * 64 + 32 + i],
                    fifth - minimum,
                    "q5_k pair {pair} high {i}"
                );
            }
        }
    }

    /// `Q2_K` keeps its super-block scales at the END of the payload.
    ///
    /// The one thing about this block that is not like `Q4_K`, which opens
    /// with the same two F16s. A decoder that read them at offset zero would
    /// take two sub-block scale bytes for a scale, which is a number rather
    /// than a crash.
    ///
    /// So the two F16s are given values no scale byte can spell -- `d` is 1.0
    /// and `dmin` is 0.0 -- and the sub-block bytes are filled with a pattern
    /// that would read as an absurd scale if it were mistaken for one. With
    /// `dmin` at zero the minimum drops out and each element is just its
    /// sub-block scale times its two-bit quant, which is what makes the
    /// striding visible on its own.
    ///
    /// The quants are `0b11_10_01_00` everywhere, so shift `s` yields quant
    /// `s` for every element: the four shifts of a window separate cleanly,
    /// and sub-block `n` reports `scale(n) * (n / 2 % 4)`.
    #[test]
    fn q2_k_reads_its_super_block_scales_after_the_payload() {
        let mut block = [0u8; 84];
        // Sub-block n has scale 1 + n mod 15 and minimum 0. Modulo fifteen
        // and not n + 1: the scale is a NIBBLE, so a sixteenth sub-block
        // numbered straight through would store 16 and read back as 0 -- and
        // as a scale of zero it would agree with an undecoded block.
        for (n, byte) in block[..16].iter_mut().enumerate() {
            *byte = 1 + (n as u8) % 15;
        }
        block[16..80].fill(0b11_10_01_00);
        block[80..82].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[82..84].copy_from_slice(&half::f16::from_f32(0.0).to_le_bytes());

        let mut values = [0f32; 256];
        decode_gguf_q2_k_block_into(&block, &mut values);

        for n in 0..16 {
            let scale = (1 + n % 15) as f32;
            // Shift 2·step of `0b11_10_01_00`, and step is the pair index
            // within a window.
            let quant = (n % 8 / 2) as f32;
            for l in 0..16 {
                assert_eq!(
                    values[n * 16 + l],
                    scale * quant,
                    "sub-block {n}, element {l}"
                );
            }
        }
        // Sub-block 0 is quant 0, so it is the one place a misread scale
        // would hide. Pinned from the other side: give it a non-zero quant.
        block[16] = 0b11_11_11_01;
        decode_gguf_q2_k_block_into(&block, &mut values);
        assert_eq!(values[0], 1.0, "d is 1.0 and sub-block 0's scale is 1");
    }

    /// `Q3_K`'s third bit reads inverted, and its selector spans both windows.
    ///
    /// Two mistakes this layout invites, and neither produces a wrong shape
    /// or an out-of-range value -- only wrong numbers. `ggml` sets the mask
    /// bit when an element needed no borrow, so a SET bit subtracts nothing
    /// and a CLEAR one subtracts four; and the bit selector advances across
    /// the two 32-byte quant windows rather than restarting at the second, so
    /// eight `(window, shift)` pairs consume the eight bits of each mask byte.
    ///
    /// Both are pinned at once with an all-zero mask against an all-ones one,
    /// on a block whose quants are zero and whose scale is one. Every element
    /// is then `-4` or `0`, and any half-block the selector got wrong reports
    /// the other value.
    #[test]
    fn q3_k_borrows_where_its_mask_is_clear_and_carries_the_selector_across() {
        let mut block = [0u8; 110];
        // Twelve scale bytes that decode to 33 everywhere: the low nibble is
        // 1 in the first eight bytes and the top two bits are 0b10, giving
        // 0b100001 = 33, which is 1 after the bias of 32.
        for i in 0..8 {
            block[96 + i] = 0x11;
        }
        for i in 8..12 {
            block[96 + i] = 0xAA;
        }
        block[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        assert_eq!(
            gguf_q3_k_scales(block[96..108].try_into().unwrap()),
            [33i8; 16],
            "the splice, before it is used"
        );

        let mut values = [0f32; 256];
        decode_gguf_q3_k_block_into(&block, &mut values);
        assert!(
            values.iter().all(|v| *v == -4.0),
            "a clear mask borrows four everywhere"
        );

        block[0..32].fill(0xFF);
        decode_gguf_q3_k_block_into(&block, &mut values);
        assert!(
            values.iter().all(|v| *v == 0.0),
            "a set mask borrows nothing -- the bit is not an addend"
        );

        // The selector, isolated: only bit 4 set means only the FIFTH of the
        // eight (window, shift) pairs keeps its value. That pair is in the
        // second window, so a selector restarted at the window boundary would
        // light the first pair instead.
        block[0..32].fill(1 << 4);
        decode_gguf_q3_k_block_into(&block, &mut values);
        let lit: Vec<usize> = (0..256).filter(|i| values[*i] == 0.0).collect();
        assert_eq!(lit.len(), 32, "one (window, shift) pair is 32 elements");
        assert_eq!(
            lit[0], 128,
            "and it is the first pair of the SECOND window, not of the first"
        );
    }

    /// Q6_K's four quarters are strided, not contiguous: quarters 0 and 1 take
    /// the low nibbles of the two halves of `ql`, quarters 2 and 3 the high
    /// ones, and each quarter's top two bits come from its own bit pair of
    /// `qh`.
    ///
    /// The two halves of each `ql` run differ (`0x30` against `0x51`) so a
    /// quarter reports which BYTE it read as well as which nibble, and
    /// `qh = 0xE4` gives quarter `q` the value `q` in its bit pair. The four
    /// quarters therefore decode to four distinct numbers, and any
    /// transposition of them — the one mistake this layout invites — swaps
    /// two. Filling `ql` uniformly instead would pin the nibble halves and
    /// silently miss a swapped byte offset.
    #[test]
    fn q6_k_strides_its_quarters_across_the_nibble_halves_and_the_bit_pairs() {
        let mut block = [0u8; 210];
        for half in 0..2 {
            block[half * 64..half * 64 + 32].fill(0x30);
            block[half * 64 + 32..half * 64 + 64].fill(0x51);
        }
        block[128..192].fill(0b1110_0100);
        block[192..208].fill(1);
        block[208..210].copy_from_slice(&f16(1.0));

        let mut values = [0.0f32; 256];
        decode_gguf_q6_k_block_into(&block, &mut values);

        // 0 | 0<<4, 1 | 1<<4, 3 | 2<<4, 5 | 3<<4 — each less the 32 bias.
        let expected = [-32.0f32, -15.0, 3.0, 21.0];
        for half in 0..2 {
            for (quarter, want) in expected.iter().enumerate() {
                for i in 0..32 {
                    assert_eq!(
                        values[half * 128 + quarter * 32 + i],
                        *want,
                        "q6_k half {half} quarter {quarter} element {i}"
                    );
                }
            }
        }

        // With the codes held flat, the sub-block scales report which of the
        // sixteen was used: `scales[8·half + i/16 + 2·quarter]`. Setting them
        // to their own index makes every one of those readings distinct.
        block[..128].fill(0);
        block[128..192].fill(0);
        for (i, byte) in block[192..208].iter_mut().enumerate() {
            *byte = i as u8;
        }
        decode_gguf_q6_k_block_into(&block, &mut values);
        for half in 0..2 {
            for quarter in 0..4 {
                for i in 0..32 {
                    let scale = (half * 8 + i / 16 + 2 * quarter) as f32;
                    assert_eq!(
                        values[half * 128 + quarter * 32 + i],
                        scale * -32.0,
                        "q6_k scale index, half {half} quarter {quarter} element {i}"
                    );
                }
            }
        }
    }

    /// `IQ4_NL` pairs its nibbles by HALF, not by neighbour, and its codes
    /// are indices rather than numbers.
    ///
    /// Two mistakes this layout invites, and each survives every size check.
    /// Reading byte `j` as elements `2j` and `2j+1` interleaves the two
    /// halves of the block — the values are all still present, in the wrong
    /// places. Reading the nibble as a magnitude instead of an index into
    /// `IQ4_LEVELS` gives numbers of the right sign and roughly the right
    /// spread. So the codes here are laid out to separate the halves (`0x10`
    /// for the first sixteen bytes, `0x32` for nothing, since there are only
    /// sixteen) and the expectation is the table, not arithmetic on the code.
    #[test]
    fn iq4_nl_pairs_its_nibbles_by_half_not_by_neighbour() {
        let mut block = [0u8; 18];
        block[0..2].copy_from_slice(&f16(1.0));
        // Low nibble counts up, high nibble counts down, so no element of the
        // first half can be confused with one of the second.
        for (j, byte) in block[2..18].iter_mut().enumerate() {
            *byte = (j as u8) | (((15 - j) as u8) << 4);
        }
        let mut values = [0.0f32; 32];
        decode_gguf_iq4_nl_block_into(&block, &mut values);
        for j in 0..16 {
            assert_eq!(
                values[j],
                f32::from(IQ4_LEVELS[j]),
                "iq4_nl first half element {j} reads the low nibble"
            );
            assert_eq!(
                values[j + 16],
                f32::from(IQ4_LEVELS[15 - j]),
                "iq4_nl second half element {j} reads the high nibble"
            );
        }
        // The levels are not a line. If they were, this scheme would be Q4_0
        // with extra steps and the table would not be worth carrying.
        let first_gap = f32::from(IQ4_LEVELS[8]) - f32::from(IQ4_LEVELS[7]);
        let last_gap = f32::from(IQ4_LEVELS[15]) - f32::from(IQ4_LEVELS[14]);
        assert!(
            last_gap > first_gap * 2.0,
            "the levels must crowd near zero: {first_gap} against {last_gap}"
        );
    }

    /// `IQ4_XS` builds each sub-block scale from six bits split across two
    /// planes, and reads the result SIGNED.
    ///
    /// Four low bits come from `scales_l`, two sub-blocks to a byte; two high
    /// bits come from the `scales_h` u16, eight sub-blocks to it; and the
    /// assembled value is biased by 32. Dropping the high plane caps every
    /// scale at 15 — which still decodes, to weights uniformly too small.
    /// Dropping the bias flips the sign of every sub-block whose scale is
    /// under 32. Both are pinned here by giving each of the eight sub-blocks
    /// a different scale and holding the codes flat.
    #[test]
    fn iq4_xs_assembles_a_six_bit_scale_from_two_planes() {
        let mut block = [0u8; 136];
        block[0..2].copy_from_slice(&f16(1.0));
        // Six bits biased by 32, so the scale runs `-32..=31` and both signs
        // have to appear. Each is chosen so its low nibble and its high pair
        // both carry information: no sub-block is decodable from one plane.
        let want: [i32; 8] = [-32, -20, -8, -1, 0, 12, 20, 31];
        let mut scales_h = 0u16;
        for (sub, ls) in want.iter().enumerate() {
            let raw = (ls + 32) as u8;
            block[4 + sub / 2] |= (raw & 0x0f) << (4 * (sub % 2));
            scales_h |= u16::from((raw >> 4) & 3) << (2 * sub);
        }
        block[2..4].copy_from_slice(&scales_h.to_le_bytes());
        // Every code index 8, whose level is 1, so each element reports its
        // sub-block's scale and nothing else.
        block[8..136].fill(0x88);

        let mut values = [0.0f32; 256];
        decode_gguf_iq4_xs_block_into(&block, &mut values);
        for (sub, ls) in want.iter().enumerate() {
            let expected = *ls as f32 * f32::from(IQ4_LEVELS[8]);
            for i in 0..32 {
                assert_eq!(
                    values[sub * 32 + i],
                    expected,
                    "iq4_xs sub-block {sub} element {i}"
                );
            }
        }
    }

    /// One real `IQ4_XS` block, decoded to values a second quantization of
    /// the same weights agrees with.
    ///
    /// Structure tests cannot catch a wrong level table: any monotone table
    /// decodes to plausible weights. So the table was checked against data.
    /// `Llama-3.2-1B-Instruct-UD-Q2_K_XL.gguf` stores five `ffn_down`
    /// tensors as IQ4_XS, and `Llama-3.2-1B-Instruct-Q3_K_M.gguf` stores four
    /// of the same five as Q4_K — the same weights, quantized twice,
    /// independently. Decoded both ways they agree at r = 0.994 with a
    /// magnitude ratio of 0.998, and the residual is what discriminates:
    /// NRMSE against the Q4_K decode is 0.106 for this table, 0.147 with two
    /// adjacent levels transposed, and 0.207 for any linear table. The
    /// correct one is the minimum.
    ///
    /// This block is the first of `blk.13.ffn_down.weight` in that file. It
    /// is here so the agreement survives without the 700 MB it was measured
    /// on.
    #[test]
    fn iq4_xs_decodes_a_real_block_the_way_a_second_quantization_reads_it() {
        const BLOCK: [u8; 136] = [
            0xd1, 0x00, 0x0c, 0xcf, 0x62, 0x90, 0x28, 0x9b, 0xcb, 0x72, 0x1a, 0xc5, 0xb7, 0x96,
            0x49, 0x3d, 0xf0, 0x54, 0x18, 0x6d, 0xba, 0xdb, 0x8a, 0xc5, 0xbe, 0x70, 0x03, 0x8f,
            0xb3, 0x9a, 0x94, 0x48, 0xbc, 0xec, 0x55, 0x99, 0x34, 0x7b, 0xc7, 0x13, 0xd8, 0x59,
            0x98, 0x86, 0x61, 0xa9, 0xe7, 0x95, 0xa8, 0x4c, 0x0d, 0x84, 0x6c, 0xa7, 0xa4, 0x96,
            0x52, 0xb7, 0x49, 0x5a, 0x31, 0x4a, 0x79, 0x92, 0x6e, 0x1d, 0x08, 0xc3, 0x5d, 0x9a,
            0xcf, 0x96, 0xbc, 0xa1, 0xa9, 0x7e, 0x48, 0x55, 0xa3, 0x2d, 0x01, 0xb2, 0xb8, 0x8e,
            0x88, 0x76, 0x7f, 0x86, 0x7c, 0xdd, 0xf7, 0xae, 0x2e, 0xb6, 0xd5, 0xd0, 0x40, 0x81,
            0x29, 0xd6, 0xdb, 0xd2, 0x7a, 0xfd, 0x64, 0xf2, 0x12, 0xb3, 0xbb, 0x17, 0x39, 0xf1,
            0xda, 0x0c, 0x8f, 0xaa, 0xf6, 0xdd, 0x78, 0xd9, 0x43, 0xdb, 0x63, 0x45, 0xd3, 0xd2,
            0x89, 0xb7, 0xa0, 0x88, 0x55, 0x2b, 0x58, 0x9c, 0x9a, 0x34,
        ];
        let mut values = [0.0f32; 256];
        decode_gguf_iq4_xs_block_into(&BLOCK, &mut values);

        let first: [f32; 8] = [
            -1.420_140_3e-2,
            3.101_885_3e-2,
            -9.343_028e-3,
            1.308_023_9e-2,
            3.737_211_2e-3,
            8.221_865e-3,
            -4.858_374_6e-3,
            -2.578_675_7e-2,
        ];
        for (i, want) in first.iter().enumerate() {
            assert!(
                (values[i] - want).abs() < 1e-9,
                "element {i}: {} against {want}",
                values[i]
            );
        }
        // The whole block, so a fault past element eight is not missed. Sums
        // rather than 256 literals, and both signs of one: the plain sum
        // would survive a sign flip that the absolute sum catches, and the
        // absolute sum would survive a permutation that neither catches but
        // the structure tests above do.
        let sum: f32 = values.iter().sum();
        let abs: f32 = values.iter().map(|v| v.abs()).sum();
        assert!((sum - -6.562_543e-2).abs() < 1e-6, "block sum {sum}");
        assert!((abs - 3.578_529_1).abs() < 1e-5, "block absolute sum {abs}");
    }

    /// The E8M0 scale is exactly `2^(e - 128)` for every one of its 256 codes.
    ///
    /// The halving is the whole reason this is `- 128` and not `- 127`:
    /// [`MXFP4_LEVELS`] stores E2M1's half-integers doubled so the table can be
    /// `i8`, and the scale carries the matching half. Two codes then land below
    /// the smallest normal float, which is why llama.cpp writes them as
    /// subnormal bit patterns and why this cannot be `2f32.powi(e - 128)`
    /// naively formed from an exponent field.
    ///
    /// Checked against `f64` arithmetic, which represents every one of the 256
    /// powers exactly, and compared for equality rather than tolerance: these
    /// are powers of two, so a correct implementation is exact and a tolerance
    /// would hide an off-by-one in the bias.
    #[test]
    fn the_mxfp4_scale_is_a_power_of_two_at_every_code() {
        for e in 0u16..=255 {
            let want = 2f64.powi(i32::from(e) - 128);
            let got = f64::from(mxfp4_scale(e as u8));
            assert_eq!(got, want, "e={e}");
        }
    }

    /// A real `MXFP4` block, decoded to what the same weights say in OpenAI's
    /// own planar MXFP4.
    ///
    /// The first block of `blk.0.ffn_down_exps.weight` from
    /// `ggml-org/gpt-oss-20b-GGUF`. Its reference is not another
    /// implementation of this decoder but the *other layout* of the same
    /// tensor: `openai/gpt-oss-20b`'s
    /// `model.layers.0.mlp.experts.down_proj_blocks`/`_scales`, which is OCP
    /// planar MXFP4 and pairs its nibbles by neighbour where ggml pairs them
    /// by half. Decoding both and comparing gave **exact equality over 4000
    /// blocks**, with the scale bytes byte-identical between the two files and
    /// the quant bytes not — llama.cpp repacks the nibbles and copies the
    /// scales.
    ///
    /// That is what makes this a test of the pairing rather than of a
    /// transcription: reading these codes as adjacent pairs correlates with
    /// the truth at **0.066**, so the mistake is not subtle once it is
    /// measured, and is invisible until it is.
    #[test]
    fn mxfp4_decodes_a_real_block_the_way_the_planar_layout_reads_it() {
        const BLOCK: [u8; 17] = [
            0x79, 0x1b, 0x07, 0xac, 0x4d, 0xa2, 0xe2, 0x2a, 0x25, 0x41, 0x43, 0x9a, 0xeb, 0x12,
            0x2c, 0x28, 0x81,
        ];
        let mut values = [0.0f32; 32];
        decode_gguf_mxfp4_block_into(&BLOCK, &mut values);

        // e = 121, so the scale is 2^-7 and every value is a small multiple of
        // it. Exact comparisons: each is a half-integer times a power of two.
        assert_eq!(mxfp4_scale(BLOCK[0]), 0.0078125);
        let want: [f32; 32] = [
            -0.0234375, 0.09375, -0.03125, -0.046875, 0.015625, 0.015625, -0.015625, 0.046875,
            0.0078125, 0.0234375, -0.015625, -0.0234375, 0.015625, -0.03125, 0.0, 0.0078125,
            0.0078125, 0.0, -0.015625, 0.03125, -0.015625, -0.0625, 0.015625, 0.015625, 0.03125,
            0.03125, -0.0078125, -0.0625, 0.0078125, 0.015625, 0.015625, 0.0,
        ];
        assert_eq!(values, want);
    }

    /// The nibbles pair by half, and the levels are E2M1's.
    ///
    /// Byte `j` carries element `j` and element `j + 16`, so a synthetic block
    /// whose bytes are `j | ((15 - j) << 4)` gives the levels in order down
    /// the first half and in reverse down the second. Adjacent pairing would
    /// interleave them, which no sum or magnitude check would notice.
    ///
    /// The scale is `e = 127`, which the halving makes `2^-1`, so the decoded
    /// values are the E2M1 value set itself: 0, 0.5, 1, 1.5, 2, 3, 4, 6 and
    /// their negatives. That anchors [`MXFP4_LEVELS`] absolutely rather than
    /// only relative to itself — the table is those values doubled, and this
    /// is the code at which the halving gives them back. `e = 128` would give
    /// the doubled table, which is what makes the bias worth pinning.
    #[test]
    fn mxfp4_pairs_its_nibbles_by_half_over_the_e2m1_value_set() {
        let mut block = [0u8; 17];
        block[0] = 127;
        for j in 0..16u8 {
            block[1 + usize::from(j)] = j | ((15 - j) << 4);
        }
        let mut values = [0.0f32; 32];
        decode_gguf_mxfp4_block_into(&block, &mut values);

        let e2m1: [f32; 16] = [
            0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ];
        for j in 0..16 {
            assert_eq!(values[j], e2m1[j], "first half element {j}");
            assert_eq!(values[j + 16], e2m1[15 - j], "second half element {j}");
        }
    }

    /// The affine blocks place their range with an offset, not with a bias.
    ///
    /// Q4_1 and Q5_1 had no test at all. They are the two arms whose doc
    /// warns about a sign -- the offset is ADDED here and SUBTRACTED in the
    /// K-quants -- and a sign error there survives every shape and size
    /// check, because the block is still the right length and the numbers are
    /// still plausible.
    ///
    /// So each is pinned against its symmetric twin through an identity
    /// rather than against a table of expected floats. `Q4_0` spells its
    /// range as `(n - 8) * d`; `Q4_1` has no bias, so the same mapping is
    /// exactly `m = -8d`. If the two agree for all 32 elements then the
    /// offset is added with the right sign AND no bias is hiding underneath
    /// it -- one wrong sign moves every element by `16d`, and a stray bias by
    /// `8d`. `Q5_1` against `Q5_0` at `m = -16d` says the same, and says it
    /// about the fifth-bit plane too, since both read the same plane bytes.
    ///
    /// The identity alone would pass if BOTH decoders were wrong the same
    /// way, so absolute values are anchored beside it. The nibbles are laid
    /// out so that the low nibble of byte `i` is `i` and its high nibble is
    /// `15 - i`: every element differs from its neighbour, and the two halves
    /// run in opposite directions, so a decoder that swapped them reports the
    /// mirror rather than something that still looks sorted.
    #[test]
    fn the_affine_blocks_offset_their_range_where_the_symmetric_ones_bias_it() {
        let d = 0.5f32;
        let le = |v: f32| half::f16::from_f32(v).to_le_bytes();
        let nibbles: [u8; 16] = std::array::from_fn(|i| ((15 - i as u8) << 4) | i as u8);

        let mut q4_0 = [0u8; 18];
        q4_0[..2].copy_from_slice(&le(d));
        q4_0[2..].copy_from_slice(&nibbles);
        let mut symmetric = [0f32; 32];
        decode_gguf_q4_0_block_into(&q4_0, &mut symmetric);

        let mut q4_1 = [0u8; 20];
        q4_1[..2].copy_from_slice(&le(d));
        q4_1[2..4].copy_from_slice(&le(-8.0 * d));
        q4_1[4..].copy_from_slice(&nibbles);
        let mut affine = [0f32; 32];
        decode_gguf_q4_1_block_into(&q4_1, &mut affine);

        assert_eq!(symmetric, affine, "Q4_1 at m = -8d is Q4_0");
        assert_eq!(affine[0], -4.0, "element 0 is the LOW nibble of byte 0");
        assert_eq!(affine[15], 3.5);
        assert_eq!(affine[16], 3.5, "element 16 is the HIGH nibble of byte 0");
        assert_eq!(affine[31], -4.0);

        // Bit 3 lifts the low nibble of byte 3; bit 20 lifts the HIGH nibble
        // of byte 4. Two bits far apart in the word and adjacent in the
        // block, so a plane read in packing order lands on neither.
        let plane = ((1u32 << 3) | (1 << 20)).to_le_bytes();

        let mut q5_0 = [0u8; 22];
        q5_0[..2].copy_from_slice(&le(d));
        q5_0[2..6].copy_from_slice(&plane);
        q5_0[6..].copy_from_slice(&nibbles);
        let mut symmetric = [0f32; 32];
        decode_gguf_q5_0_block_into(&q5_0, &mut symmetric);

        let mut q5_1 = [0u8; 24];
        q5_1[..2].copy_from_slice(&le(d));
        q5_1[2..4].copy_from_slice(&le(-16.0 * d));
        q5_1[4..8].copy_from_slice(&plane);
        q5_1[8..].copy_from_slice(&nibbles);
        let mut affine = [0f32; 32];
        decode_gguf_q5_1_block_into(&q5_1, &mut affine);

        assert_eq!(symmetric, affine, "Q5_1 at m = -16d is Q5_0");
        // 3 | 16 = 19, against the 3 its neighbours read.
        assert_eq!(affine[3], 19.0 * d - 8.0);
        assert_eq!(affine[4], 4.0 * d - 8.0, "the bit did not leak sideways");
        // High nibble of byte 4 is 11, lifted to 27.
        assert_eq!(affine[20], 27.0 * d - 8.0);
        assert_eq!(affine[21], 10.0 * d - 8.0);
    }

    /// `ksigns_iq2xs` is a parity rule, so state the rule and check it holds.
    ///
    /// llama.cpp ships 128 bytes; [`iq_sign_byte`] ships one line. The two are
    /// the same thing only if every result has even population count and the
    /// low seven bits come back unchanged, which is what makes seven bits
    /// enough to say eight signs. If the rule were ever wrong the failure
    /// would be a sign flip on one element in eight — visible in a perplexity
    /// number and in nothing else — so it is checked here over the whole
    /// domain rather than sampled.
    #[test]
    fn the_iq_sign_byte_is_the_index_completed_to_even_parity() {
        for index in 0u16..=255 {
            let index = index as u8;
            let got = iq_sign_byte(index);
            assert_eq!(got & 0x7f, index & 0x7f, "index {index} was not preserved");
            assert_eq!(got.count_ones() % 2, 0, "index {index} gave odd parity");
        }
        // The eighth bit is genuinely used: half the domain sets it.
        let set = (0u16..128)
            .filter(|i| iq_sign_byte(*i as u8) & 0x80 != 0)
            .count();
        assert_eq!(set, 64, "the parity bit is not carrying information");
    }

    /// `IQ2_XS` splits its `u16` at bit **nine**, not bit eight.
    ///
    /// The grid is 512 points, so the point index does not fit in a byte. A
    /// decoder that masked with 0xFF — the obvious thing, and what `IQ2_XXS`
    /// correctly does over its 256-point grid — would read the right point for
    /// the half of the domain below 256 and the wrong one above, and would
    /// also shift the sign index by one bit. This block puts the point at 511
    /// so both halves of a wrong split are wrong.
    #[test]
    fn iq2_xs_takes_nine_bits_of_point_and_seven_of_sign() {
        let mut block = [0u8; 74];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[2..4].copy_from_slice(&511u16.to_le_bytes());
        let mut values = [0.0f32; 256];
        decode_gguf_iq2_xs_block_into(&block, &mut values);

        // Scale nibble zero is still (0.5 + 0) * 0.25, never zero.
        let db = 0.125f32;
        for (bit, got) in values[..8].iter().enumerate() {
            let want = db * f32::from(iq_grid::IQ2XS_GRID[511 * 8 + bit]);
            assert_eq!(*got, want, "element {bit} of point 511");
        }
        // And the test discriminates: an eight-bit mask would land on 255.
        let truncated = &iq_grid::IQ2XS_GRID[255 * 8..255 * 8 + 8];
        let correct = &iq_grid::IQ2XS_GRID[511 * 8..511 * 8 + 8];
        assert_ne!(
            truncated, correct,
            "point 255 and 511 must differ to test this"
        );
    }

    /// `IQ2_S` packs four points per `qh` byte; `IQ3_S` packs eight.
    ///
    /// Both extend an eight-bit `qs` with bits from `qh`, and the two schemes
    /// sit next to each other in this file, so the field widths are easy to
    /// cross. They are not interchangeable: `IQ2_S` takes **two** bits per
    /// point to reach a 1024-point grid, `IQ3_S` takes **one** to reach 512.
    /// Using the wrong one addresses the right grid for the first few points
    /// and drifts after, which is exactly the kind of fault a whole-block sum
    /// would hide behind plausible numbers.
    #[test]
    fn the_high_bits_of_iq2_s_and_iq3_s_are_different_widths() {
        // IQ2_S: qh byte 0 = 0b11 in field k=0 lifts point 0 to point 768.
        let mut block = [0u8; 82];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[66] = 0b11;
        let mut values = [0.0f32; 256];
        decode_gguf_iq2_s_block_into(&block, &mut values);
        for (bit, got) in values[..8].iter().enumerate() {
            let want = 0.125 * f32::from(iq_grid::IQ2S_GRID[768 * 8 + bit]);
            assert_eq!(*got, want, "IQ2_S element {bit} of point 768");
        }
        // Field k=1 is the *next two bits*, so it is untouched and reads point 0.
        for bit in 0..8 {
            let want = 0.125 * f32::from(iq_grid::IQ2S_GRID[bit]);
            assert_eq!(values[8 + bit], want, "IQ2_S point 1 was disturbed");
        }

        // IQ3_S: qh byte 0 bit 0 lifts point 0 to point 256, and bit 1 is the
        // *next point*, not the second bit of this one.
        let mut block = [0u8; 110];
        block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
        block[66] = 0b11;
        let mut values = [0.0f32; 256];
        decode_gguf_iq3_s_block_into(&block, &mut values);
        for (i, got) in values[..8].iter().enumerate() {
            let want = f32::from(iq_grid::IQ3S_GRID[256 * 4 + i % 4]);
            assert_eq!(*got, want, "IQ3_S element {i} of point 256");
        }
    }

    /// `IQ3_S`'s scale is `1 + 2s` — an odd integer, with no half-step.
    ///
    /// Every other IQ scheme here scales by `(0.5 + s) * k`. `IQ3_S` does not,
    /// because its grid components are the odd numbers 1..15 and the odd
    /// multiplier is what keeps the product on the intended lattice. Sharing a
    /// scale helper across the schemes would therefore be wrong rather than
    /// merely awkward, and this pins that down: with `d = 1` and scale nibble
    /// `s`, element zero is exactly `(1 + 2s)` times a grid entry.
    #[test]
    fn the_iq3_s_scale_is_an_odd_integer() {
        for s in 0u8..16 {
            let mut block = [0u8; 110];
            block[0..2].copy_from_slice(&half::f16::from_f32(1.0).to_le_bytes());
            block[106] = s;
            let mut values = [0.0f32; 256];
            decode_gguf_iq3_s_block_into(&block, &mut values);
            let want = f32::from(1 + 2 * u16::from(s)) * f32::from(iq_grid::IQ3S_GRID[0]);
            assert_eq!(values[0], want, "IQ3_S scale nibble {s}");
        }
    }

    /// The five lattice schemes, each on a real block, against `gguf-py`.
    ///
    /// These grids are not in the file: llama.cpp compiles them in, and
    /// [`iq_grid`] compiles in the same ones. So the only correctness argument
    /// available is agreement with the reference implementation, and it was
    /// taken over whole tensors — 110M elements across
    /// `Llama-3.2-1B-Instruct-UD-Q2_K_XL.gguf` and
    /// `Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf`, decoded by `gguf-py`, rounded
    /// to BF16 and compared to the imported artifact. Every element matched
    /// exactly; `IQ4_XS`, already known good, rode along as the control.
    ///
    /// The blocks below are the first block of one tensor per scheme from
    /// those files, kept here so the agreement survives without the 3 GB it
    /// was measured on. Values come from `gguf-py`, not from this decoder.
    ///
    /// Each block is checked at its first eight elements and then by two sums
    /// over all 256: the plain sum would survive a sign flip that the absolute
    /// sum catches, and the absolute sum would survive a permutation that
    /// neither catches but the structure tests above do.
    #[test]
    fn the_iq_lattice_schemes_decode_real_blocks_the_way_gguf_py_does() {
        fn check(name: &str, values: &[f32; 256], first: [f32; 8], sum: f32, abs: f32) {
            for (i, want) in first.iter().enumerate() {
                assert!(
                    (values[i] - want).abs() < 1e-9,
                    "{name} element {i}: {} against {want}",
                    values[i]
                );
            }
            let got_sum: f32 = values.iter().sum();
            let got_abs: f32 = values.iter().map(|v| v.abs()).sum();
            assert!((got_sum - sum).abs() < 1e-5, "{name} block sum {got_sum}");
            assert!(
                (got_abs - abs).abs() < 1e-4,
                "{name} absolute sum {got_abs}"
            );
        }

        // blk.0.attn_k.weight of Llama-3.2-1B-Instruct-UD-IQ2_XXS.gguf
        const IQ2_XXS: [u8; 66] = [
            0xf9, 0x19, 0x00, 0x00, 0x00, 0x00, 0x70, 0xc7, 0x96, 0xf0, 0x85, 0x4b, 0x52, 0x04,
            0x2f, 0x34, 0x68, 0x47, 0x12, 0x07, 0x01, 0x0d, 0x85, 0x5f, 0x5e, 0x78, 0x35, 0xfb,
            0xc4, 0xe2, 0x5b, 0x93, 0xf8, 0x38, 0x36, 0x21, 0x15, 0xc9, 0x12, 0xdc, 0x4b, 0x47,
            0x45, 0xdb, 0x7b, 0x47, 0xb7, 0xd4, 0x95, 0x36, 0xb5, 0x70, 0x59, 0x0d, 0x9e, 0x1b,
            0xe1, 0x5a, 0xae, 0x54, 0x2d, 0x2f, 0x07, 0x38, 0x9f, 0x51,
        ];
        let mut values = [0.0f32; 256];
        decode_gguf_iq2_xxs_block_into(&IQ2_XXS, &mut values);
        check(
            "IQ2_XXS",
            &values,
            [
                9.040_642e-2,
                9.040_642e-2,
                9.040_642e-2,
                9.040_642e-2,
                -9.040_642e-2,
                -9.040_642e-2,
                -9.040_642e-2,
                -9.040_642e-2,
            ],
            -1.180_022_5,
            1.587_908_5e1,
        );

        // blk.2.attn_k.weight of Llama-3.2-1B-Instruct-UD-Q2_K_XL.gguf
        const IQ2_XS: [u8; 74] = [
            0x4e, 0x11, 0x7d, 0xa1, 0x9d, 0x56, 0xcc, 0x1b, 0x2c, 0x50, 0x57, 0xa0, 0x6a, 0x7a,
            0x7e, 0x93, 0xdb, 0xe4, 0xbb, 0x19, 0xee, 0x42, 0x90, 0x43, 0x9c, 0x5c, 0xd5, 0xaa,
            0x06, 0x87, 0x34, 0x22, 0x8c, 0xd4, 0x90, 0xfb, 0x4c, 0x53, 0xa3, 0xd1, 0x50, 0x94,
            0x4b, 0xd9, 0x4f, 0x81, 0x79, 0x19, 0x37, 0xef, 0x4b, 0xbd, 0xb9, 0x4b, 0x25, 0xfa,
            0xe9, 0x34, 0x49, 0xb6, 0xa7, 0x0c, 0xa8, 0x1b, 0x7a, 0xeb, 0xa7, 0x5e, 0xb4, 0x9c,
            0x99, 0x8f, 0x9a, 0x79,
        ];
        let mut values = [0.0f32; 256];
        decode_gguf_iq2_xs_block_into(&IQ2_XS, &mut values);
        check(
            "IQ2_XS",
            &values,
            [
                5.220_830_4e-2,
                9.713_173e-3,
                5.220_830_4e-2,
                3.035_366_5e-2,
                -9.713_173e-3,
                9.713_173e-3,
                -5.220_830_4e-2,
                3.035_366_5e-2,
            ],
            1.295_089_7e-3,
            6.650_609_5,
        );

        // blk.2.ffn_gate.weight of the same file.
        const IQ2_S: [u8; 82] = [
            0x03, 0x0d, 0x76, 0x19, 0xaf, 0x0e, 0x6c, 0xa5, 0x86, 0xad, 0x8b, 0xdb, 0xf2, 0x1d,
            0x54, 0x00, 0x57, 0x01, 0x10, 0x43, 0x6c, 0x52, 0x21, 0xc7, 0xd1, 0xec, 0x5f, 0x87,
            0x53, 0x7b, 0x14, 0x64, 0xce, 0x45, 0x14, 0x58, 0x2d, 0x69, 0x55, 0xc9, 0x1b, 0x91,
            0xa8, 0x1c, 0xdc, 0x2d, 0x29, 0xa9, 0x70, 0xbd, 0xa3, 0xe7, 0x0e, 0xc2, 0x1f, 0xce,
            0x54, 0x57, 0x2d, 0x5e, 0x0f, 0x15, 0xdd, 0x5e, 0xbf, 0x4d, 0xf4, 0x35, 0xc5, 0xc1,
            0x28, 0xa9, 0x4a, 0x86, 0xb8, 0x88, 0xba, 0x8f, 0x6f, 0x8f, 0x79, 0xa9,
        ];
        let mut values = [0.0f32; 256];
        decode_gguf_iq2_s_block_into(&IQ2_S, &mut values);
        check(
            "IQ2_S",
            &values,
            [
                2.795_079_4e-2,
                2.795_079_4e-2,
                -5.200_147_6e-3,
                5.200_147_6e-3,
                -1.625_046_1e-2,
                1.625_046_1e-2,
                5.200_147_6e-3,
                5.200_147_6e-3,
            ],
            -2.665_075_7e-2,
            3.598_693_3,
        );

        // blk.0.attn_k.weight of the same file.
        const IQ3_XXS: [u8; 98] = [
            0x4e, 0x10, 0x20, 0x9e, 0x3e, 0x6d, 0x88, 0x50, 0x7e, 0x1d, 0x7b, 0x95, 0xa6, 0xa0,
            0x93, 0x58, 0x21, 0x46, 0xcb, 0x1a, 0x2f, 0x47, 0x05, 0x75, 0x8d, 0x00, 0x42, 0x27,
            0x01, 0xda, 0x70, 0xa7, 0x77, 0xc7, 0x79, 0x67, 0x08, 0x49, 0x3b, 0x3f, 0xbc, 0x84,
            0x9a, 0x98, 0x1c, 0xce, 0xb6, 0x5b, 0x54, 0x22, 0x06, 0x79, 0x29, 0x2f, 0x45, 0x4c,
            0x7e, 0x0c, 0x63, 0x26, 0xa0, 0x19, 0x1a, 0x13, 0x73, 0xa1, 0x70, 0xc7, 0x96, 0x90,
            0x2f, 0x34, 0x68, 0x87, 0x85, 0x5f, 0x5e, 0xe8, 0x5b, 0x93, 0xf8, 0x78, 0x12, 0xdc,
            0x4b, 0x97, 0xb7, 0xd4, 0x95, 0x76, 0x9e, 0x1b, 0xe1, 0xfa, 0x07, 0x38, 0x9f, 0xb1,
        ];
        let mut values = [0.0f32; 256];
        decode_gguf_iq3_xxs_block_into(&IQ3_XXS, &mut values);
        check(
            "IQ3_XXS",
            &values,
            [
                2.995_205e-2,
                1.098_241_8e-1,
                4.992_008_2e-2,
                9.984_016e-3,
                -4.992_008_2e-2,
                -1.098_241_8e-1,
                -4.992_008_2e-2,
                -6.988_811_5e-2,
            ],
            -8.318_262e-1,
            1.324_616_2e1,
        );

        // blk.0.ffn_gate.weight of the same file.
        const IQ3_S: [u8; 110] = [
            0x2c, 0x08, 0x31, 0x1f, 0x91, 0xa6, 0xe2, 0x67, 0xa3, 0xbf, 0x44, 0xeb, 0xcf, 0xe5,
            0x1b, 0x2c, 0x16, 0x51, 0x23, 0xd8, 0x20, 0xc5, 0x3f, 0xc2, 0x32, 0x7a, 0x81, 0xf9,
            0x01, 0x9c, 0x71, 0xfb, 0x9a, 0x52, 0x7e, 0x12, 0x8f, 0x3e, 0xd5, 0x3a, 0x0b, 0x6a,
            0xe1, 0x77, 0x05, 0xf2, 0x33, 0x95, 0x9d, 0xff, 0xdc, 0x5f, 0x91, 0x8a, 0xe1, 0x3c,
            0x28, 0xca, 0x1c, 0x69, 0x9d, 0x1a, 0x5d, 0xc9, 0x3e, 0xe6, 0xee, 0x68, 0x74, 0xa8,
            0x89, 0x58, 0x50, 0x4a, 0x16, 0xa3, 0x26, 0x3d, 0xe4, 0xbf, 0x38, 0x0f, 0x06, 0xdc,
            0x73, 0x30, 0x33, 0x61, 0x87, 0x98, 0x78, 0xa7, 0x78, 0x73, 0x69, 0x35, 0x46, 0xd8,
            0xe7, 0xa4, 0x29, 0x84, 0xb2, 0x66, 0x4b, 0x96, 0x8b, 0xa8, 0xcb, 0xaf,
        ];
        let mut values = [0.0f32; 256];
        decode_gguf_iq3_s_block_into(&IQ3_S, &mut values);
        check(
            "IQ3_S",
            &values,
            [
                3.221_082_7e-2,
                -1.464_128_5e-2,
                -1.464_128_5e-2,
                2.928_257e-3,
                -8.784_771e-3,
                2.049_779_9e-2,
                2.928_257e-3,
                2.049_779_9e-2,
            ],
            -1.858_806_6e-1,
            3.636_131_3,
        );
    }
}
