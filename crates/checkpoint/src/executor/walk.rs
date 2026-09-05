//! Walks a finished plan against checkpoint bytes; the only module below `lib.rs` that opens files.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use half::bf16;

use std::collections::HashSet;

use super::iq_grid;
use super::{Progress, Residency};
use crate::consume::SourceLedger;
use crate::codec::cast::{cast_elements, decode_values, encode_values};
use crate::codec::fp8::{decode_fp8_e4m3_elements, f32_to_fp8_e4m3};
use crate::codec::int4::decode_int4b8_elements;
use crate::codec::mlx::decode_mlx_affine_codes;
use crate::codec::mlx::mlx_affine_group_params_bits;
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

/// Fill value for freshly allocated buffers. Non-zero so a missing
/// [`StorageInstr::Fill`] doesn't look like valid zeroed data.
const POISON: u8 = 0xAB;

pub(super) fn run(
    plan: &LoadPlan,
    snapshot_dir: &Path,
    residency: Residency<'_>,
    sink: &mut dyn TensorSink,
    progress: &mut dyn FnMut(Progress<'_>),
    consume: Option<&SourceLedger>,
) -> Result<(), Error> {
    if plan.target.tile_map_mask & !CONVERT_TILE_MAP_MASK != 0 {
        return Err(invalid(
            "executor received a plan advertising TileMap transforms the host \
             does not implement",
        ));
    }
    let mut nothing: &mut [u8] = &mut [];
    let (arena, stream): (&mut dyn ArenaBacking, bool) = match residency {
        Residency::Arena(arena) => (arena, false),
        Residency::Streaming => (&mut nothing, true),
    };
    // Cached once: a property of the backing, not of a moment.
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
    // Streaming has no arena: every buffer is owned and freed at its last use.
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
    // Poison marks "never touched"; a legal tensor value can itself be zero.
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
        consume,
    };
    executor.execute()?;
    // finish() drains in-flight writes; CudaArena leaves them in flight until then.
    executor.arena.finish()?;
    Ok(())
}

/// Last schedule position each buffer is referenced, following views to their root.
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
    /// Tensor-id lookup; ids interleave two allocators so a map is needed
    /// (buffers/instructions are dense).
    index: PlanIndex,
    files: HashMap<u32, PathBuf>,
    arena: &'p mut dyn ArenaBacking,
    buffers: HashMap<BufferId, BufferLoc>,
    sink: &'p mut dyn TensorSink,
    /// Names already published; finalizing one twice is a plan bug.
    finalized: HashSet<String>,
    /// Streaming: no arena, owned buffers, freed at last use.
    stream: bool,
    /// Filled only when streaming; see [`last_uses`].
    last_use: HashMap<BufferId, usize>,
    progress: &'p mut dyn FnMut(Progress<'_>),
    read_bytes: u64,
    /// Whether the arena backing runs TileMap kernels itself; cached once at entry.
    arena_runs_kernels: bool,
    /// **`--consume-source`, reaching the DECODE.** Present when the caller is
    /// consuming the checkpoint it is converting: each source range this walk
    /// reads for the last time is handed back to the filesystem as the read
    /// returns, so the source shrinks while the output grows.
    ///
    /// `Some` also makes the source handles WRITABLE, which is the only reason
    /// the ledger has to reach this far down —
    /// [`consume::release`](crate::consume::release) needs a descriptor it can
    /// punch a hole in.
    ///
    /// The ledger is what says "for the last time"; see its module note for the
    /// proof, which names this file's two read sites — [`Walk::read_file`] and
    /// [`Walk::fp8_block_operand`] — by hand.
    consume: Option<&'p SourceLedger>,
}

/// GGUF `Q4_0` block: F16 scale + 16 packed bytes; low nibble = element
/// 0..16, high nibble = element 16..32, value = (nibble − 8) × scale.
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

/// GGUF `Q5_0` block: F16 scale + 32-bit fifth-bit plane + 16 packed bytes;
/// plane bit `i`/`i+16` is the fifth bit of byte `i`'s low/high nibble.
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

/// GGUF `Q8_0` block: F16 scale + 32 signed bytes, value = byte × scale
/// (no packing).
fn decode_gguf_q8_0_block_into(block: &[u8; 34], values: &mut [f32; 32]) {
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for i in 0..32 {
        values[i] = scale * f32::from(block[2 + i] as i8);
    }
}

/// GGUF `Q4_1` block: F16 scale `d`, F16 offset `m`, 16 packed bytes;
/// value = nibble × d + m (affine, not symmetric).
fn decode_gguf_q4_1_block_into(block: &[u8; 20], values: &mut [f32; 32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let m = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    for i in 0..16 {
        let packed = block[4 + i];
        values[i] = f32::from(packed & 0x0f) * d + m;
        values[i + 16] = f32::from(packed >> 4) * d + m;
    }
}

/// GGUF `Q5_1` block: [`decode_gguf_q4_1_block_into`] plus the fifth-bit
/// plane [`decode_gguf_q5_0_block_into`] carries, indexed the same way.
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

/// GGUF `Q2_K` super-block: 256 elements as 16 sub-blocks with 4-bit
/// scale+min each, over one F16 scale/min (closing the block, bytes 80/82).
/// Payload is two 32-byte windows, each visited 4 times at shifts 0/2/4/6.
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

/// `Q3_K`'s sixteen scales, unpacked from 12 bytes: low 4 bits from the
/// first 8 bytes, top 2 bits from the last 4 (two bits at a time); result
/// is biased by 32, which the caller subtracts.
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

/// GGUF `Q3_K` super-block: 256 elements, 16 sub-blocks, symmetric, each
/// element's third bit in a separate mask (inverted: set bit = no borrow).
/// The mask's 32 bytes are read 8 times, one bit per (window, shift) pair,
/// and must not restart at the second window.
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

/// Six-bit scale + six-bit minimum for one of `Q4_K`/`Q5_K`'s eight
/// sub-blocks, unpacked from the shared 12 bytes (ggml's `get_scale_min_k4`).
/// Sub-blocks 4-7 splice their bits from the high bits sub-blocks 0-3 leave unused.
fn gguf_k_scale_min(index: usize, scales: &[u8; 12]) -> (u8, u8) {
    if index < 4 {
        (scales[index] & 63, scales[index + 4] & 63)
    } else {
        let scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        let min = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
        (scale, min)
    }
}

/// GGUF `Q4_K` super-block: 256 elements as 8 sub-blocks with own 6-bit
/// scale/min, over one F16 scale/min. Affine: d×scaleᵢ×nibble − dmin×minᵢ.
/// Payload read in sub-block pairs (low nibble = even, high = odd), so the
/// loop steps by 64 elements.
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

/// GGUF `Q5_K` super-block: `Q4_K` plus a 32-byte fifth-bit plane, read per
/// sub-block pair `p` (bit `2p` for the low nibble, `2p+1` for the high);
/// the fifth bit adds 16 before the affine minimum is subtracted.
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

/// GGUF `Q6_K` super-block: 256 elements as 16 sub-blocks with signed 8-bit
/// scale, over one F16 scale. Two 128-element halves, each with 4 strided
/// quarters (low 4 bits from `ql`, top 2 from `qh`); scale index advances
/// by 2 per quarter.
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

/// The sixteen levels an `IQ4_NL`/`IQ4_XS` code indexes (llama.cpp's
/// `kvalues_iq4nl`), non-uniform and compiled in rather than read from the file.
const IQ4_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// GGUF `IQ4_NL` block: 32 elements as 16 bytes of paired 4-bit indices
/// over one F16 scale. Byte `j` holds element `j` (low nibble) and
/// element `j + 16` (high nibble), not adjacent elements.
fn decode_gguf_iq4_nl_block_into(block: &[u8; 18], values: &mut [f32; 32]) {
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qs = &block[2..18];
    for j in 0..16 {
        values[j] = d * f32::from(IQ4_LEVELS[usize::from(qs[j] & 0x0f)]);
        values[j + 16] = d * f32::from(IQ4_LEVELS[usize::from(qs[j] >> 4)]);
    }
}

/// GGUF `IQ4_XS` super-block: 256 elements as 8 sub-blocks over
/// [`decode_gguf_iq4_nl_block_into`]'s levels. Each sub-block's 6-bit scale
/// is 4 bits from `scales_l` + 2 from `scales_h`, read as `ls - 32`.
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

/// The sixteen values an E2M1 nibble stands for, doubled so the table is
/// exact in `i8` (llama.cpp's `kvalues_mxfp4`); sign is the top nibble bit.
const MXFP4_LEVELS: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// GGUF `MXFP4` block: 32 elements as one E8M0 scale byte + 16 bytes of
/// paired E2M1 nibbles (17 bytes total, scale interleaved per block rather
/// than a separate tensor). Scale is applied as `2^(e-128)`, halved to
/// cancel [`MXFP4_LEVELS`]'s doubling.
fn decode_gguf_mxfp4_block_into(block: &[u8; 17], values: &mut [f32; 32]) {
    let d = mxfp4_scale(block[0]);
    let qs = &block[1..17];
    for j in 0..16 {
        values[j] = d * f32::from(MXFP4_LEVELS[usize::from(qs[j] & 0x0f)]);
        values[j + 16] = d * f32::from(MXFP4_LEVELS[usize::from(qs[j] >> 4)]);
    }
}

/// `2^(e - 128)` for an E8M0 exponent byte, formed exactly. The two lowest
/// inputs are subnormal (exponent field can't reach that low).
fn mxfp4_scale(e: u8) -> f32 {
    if e < 2 {
        f32::from_bits(0x0020_0000 << e)
    } else {
        f32::from_bits(u32::from(e - 1) << 23)
    }
}

/// The sign byte a IQ2/IQ3 seven-bit sign index stands for (llama.cpp's
/// `ksigns_iq2xs`): low 7 bits are the index, the 8th makes popcount even.
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

/// GGUF `IQ2_XXS` block: 256 elements in 66 bytes. F16 `d` + 16 `u32` in
/// pairs (32 elements each): four grid-point bytes into
/// [`IQ2XXS_GRID`](iq_grid::IQ2XXS_GRID), then four 7-bit sign indices at
/// bit offsets 0/7/14/21 plus a 4-bit scale in the top nibble, applied as
/// `(0.5 + s) * 0.25`.
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

/// GGUF `IQ2_XS` block: 256 elements in 74 bytes. F16 `d` + 32 `u16` + 8
/// scale bytes. Each `u16`: low 9 bits address the 512-entry
/// [`IQ2XS_GRID`](iq_grid::IQ2XS_GRID), top 7 are the sign index. Scale
/// bytes hold two 4-bit scales each, `(0.5 + s) * 0.25`.
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

/// GGUF `IQ2_S` block: 256 elements in 82 bytes. F16 `d`, 32 quant bytes,
/// 32 sign bytes, 8 high-bit bytes, 8 scale bytes. Grid is 1024 points (10
/// bits: 8 from `qs`, 2 from `qh`); signs are stored outright, one byte per point.
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

/// GGUF `IQ3_XXS` block: 256 elements in 98 bytes. F16 `d`, 64 quant
/// bytes, 8 `u32`. Grid points are 4 components, so a `qs` byte covers 4
/// elements; scale is `(0.5 + s) * 0.5` (twice `IQ2_XXS`'s factor).
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

/// GGUF `IQ3_S` block: 256 elements in 110 bytes. F16 `d`, 64 quant bytes,
/// 8 high-bit bytes, 32 sign bytes, 4 scale bytes. Grid is 512 4-component
/// points (9 bits: 8 from `qs`, 1 from `qh`); scale is `1 + 2s`.
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

/// Decode one block of any GGUF scheme the loader knows into its `f32`
/// values. `block`/`values` must match `scheme.block_layout()`'s lengths;
/// a caller that breaks that panics here rather than reading a neighbouring block.
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
            // Counted before the match consumes the instruction.
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
                // Drops buffers at their last use, bounding peak memory to the working set.
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
        let span = len;
        let len = checked_usize(len)?;
        let mut out = vec![0u8; len];
        // **WRITABLE WHEN THE CALLER IS CONSUMING THE SOURCE**, and read-only
        // otherwise. `F_PUNCHHOLE` and `fallocate` both need a descriptor they
        // can write, and an execution that is not releasing anything should not
        // be able to: a checkpoint the operator asked to KEEP is opened by this
        // executor with no way to change it.
        let mut file = File::options()
            .read(true)
            .write(self.consume.is_some())
            .open(path)
            .map_err(|err| invalid(format!("open {}: {err}", path.display())))?;

        // Splits a large read across threads on positioned reads (shared file
        // handle, different offsets); result matches a serial read.
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
                self.release_if_spent(&file, path, offset, span);
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
        self.release_if_spent(&file, path, offset, span);
        Ok(out)
    }

    /// **THE DECODE'S HALF OF `--consume-source`.** The bytes are in `out`; if
    /// the ledger can prove no later read of this import wants them, the
    /// filesystem gets them back now rather than when the file is deleted at
    /// the end.
    ///
    /// AFTER the reads and never before: a punch is not undoable, and a read
    /// that failed leaves an import that may still be retried against a source
    /// that is whole where it was not read.
    ///
    /// Called at both returns of [`Walk::read_file`] rather than once at a
    /// funnel, because the parallel arm returns early — and `read_file` is
    /// itself the funnel every source read in this file passes through, which
    /// is what makes the ledger's account of the read sites checkable by
    /// reading one function.
    fn release_if_spent(&self, file: &File, path: &Path, offset: u64, len: u64) {
        let Some(ledger) = self.consume else { return };
        if ledger.last_read(path, offset, len) {
            crate::consume::release(file, offset, len);
        }
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
        // Tried before staging to the host: only when the backing claims this
        // kind and every operand is already an arena span. A backing offered
        // an op must run it or error, never fail silently.
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
            TileMapKind::Unary => self.unary_bytes(source, inputs, &input, &transform)?,
            TileMapKind::Decode => {
                self.decode_bytes(outputs.first().copied(), &input, &transform)?
            }
            // Repack has no device implementation; this is its only
            // executor, run once per plane during import.
            TileMapKind::Repack => self.repack_bytes(&input, &transform)?,
            // Encode publishes more than one output buffer (payload + scale
            // metadata), so it writes directly and returns instead of using
            // the single-output path below.
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
            // Per-group `Scale`/`Decode` change width by design (unpacking,
            // GGUF blocks); `Cast` preserves element count, not byte count.
            // Every other kind must move the same byte count it read.
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

    /// The operands of a `TileMap`, as arena spans, or `None` when this
    /// instruction isn't eligible for the device path (no kernel named, a
    /// host-resident operand, or an owned output with no arena address).
    /// `None` is not an error; it just runs on the host path.
    fn arena_tile_map_op<'t>(
        &self,
        kind: TileMapKind,
        source: Option<&SourceExtent>,
        dest: Option<&DestExtent>,
        inputs: &[BufferId],
        outputs: &[BufferId],
        transform: &'t TransformSpec,
    ) -> Result<Option<TileMapOp<'t>>, Error> {
        // No row, no delegation: the plan says the host runs this one.
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
        // Encode's payload and scales must both be in the arena, or neither
        // is delegated.
        let dst_scales = if kind == TileMapKind::Encode {
            match outputs.get(1) {
                Some(&scales) => match self.arena_span(scales)? {
                    Some(span) => Some(span),
                    None => return Ok(None),
                },
                // A single-output Encode plan is invalid; the host path gives
                // the clearer error.
                None => return Ok(None),
            }
        } else {
            None
        };
        // Destination is either an extent naming a buffer, or the whole
        // first output buffer; both must land in the arena.
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
        // Per-block Scale reads factors from a second input; uniform Scale
        // carries them in `scale_factor_bits`. A missing factor operand just
        // means the host path runs this plan instead.
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

    /// A buffer's declared shape as the (rows, cols) rectangle a transform
    /// walks ([`crate::types::rectangle`]); `None` for any rank but 2.
    fn buffer_rectangle(&self, id: BufferId) -> Option<(u32, u32)> {
        let (rows, cols) = crate::types::rectangle(&self.plan.buffer(id).ok()?.ty.shape)?;
        Some((u32::try_from(rows).ok()?, u32::try_from(cols).ok()?))
    }

    /// Repack: the host-only tiled-layout permutation `pie model import`
    /// applies once per plane (no device implementation exists). Pure
    /// gather of codes/factors, no arithmetic, so it is an exact round trip.
    /// The marlin layouts are the device MoE path's and unimplemented here.
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
                // Word `lane` of tile (band, k tile) holds at nibble s+4h
                // the code at k=16*kt+2*(lane%4)+8*(s&1)+h, n=16*band+lane/4+8*(s>=2);
                // four k tiles form one lane's uint4: word order [band][k quad][lane][4].
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
                // [n][group] -> [n band][group][16]: transpose of the
                // (column, group) rectangle within each band. Columns past
                // `n` get a zero factor, zeroing the padded weight.
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

    /// `Unary` applies one elementwise function, in `f64` and rounded
    /// through `f32` the way `Bias` is. A value outside the function's
    /// domain is refused rather than answered with a `NaN`: it means the
    /// checkpoint does not hold what the contract said it holds, and a
    /// silent `NaN` reaches the weights as fluent nonsense.
    fn unary_bytes(
        &self,
        source: Option<&SourceExtent>,
        inputs: &[BufferId],
        bytes: &[u8],
        transform: &TransformSpec,
    ) -> Result<Vec<u8>, Error> {
        let dtype = if let Some(source) = source {
            self.source_dtype(source.tensor_id)?
        } else if let Some(input) = inputs.first() {
            self.buffer_dtype(*input)?
        } else {
            return Err(invalid("host Unary requires a source or input buffer"));
        };
        let op = transform
            .unary
            .ok_or_else(|| invalid("a Unary transform names no function"))?;
        let mut values = decode_values(bytes, dtype)?;
        for (at, value) in values.iter_mut().enumerate() {
            if !op.defined_at(*value) {
                return Err(invalid(format!(
                    "{op:?} is defined for {} elements and element {at} is {value}; \
                     the checkpoint does not hold what this contract reads",
                    op.domain()
                )));
            }
            *value = f64::from(op.apply(*value) as f32);
        }
        encode_values(&values, dtype)
    }

    /// `Scale` multiplies by a per-tensor constant, or by per-group factors
    /// from a second operand (which also decodes quantized codes). The
    /// multiply happens in `f32` to match the CUDA kernel bit-for-bit.
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
                // Element count, not the payload, distinguishes MLX affine
                // 4-bit vs 8-bit codes.
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

    /// MLX affine-U4 encode: publishes weight + scales + zero-points
    /// (affine metadata needs both). Every two-output scheme lives in
    /// [`Self::encode_bytes`] instead. `outputs` order: weight, then
    /// metadata as `quant_metadata_outputs` declared it.
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
        // Groups are `group` consecutive elements of a row-major buffer, so
        // the leading axis must be rows.
        let Some((rows, cols)) = crate::types::rectangle(&shape) else {
            return Err(invalid(format!(
                "encoding to {scheme:?} scales a [rows, cols] rectangle, not \
                 {shape:?}"
            )));
        };
        // The destination's own width and group: `MlxAffineU4` names the
        // codec, and the buffer's spec says whether it is 2, 4 or 8 bits wide.
        let (bits, group) = match &self.plan.buffer(*weight)?.ty.encoding {
            Encoding::Quant(spec) => (u32::from(spec.bits_per_element), spec.group_size),
            Encoding::Raw(_) => (4, scheme.default_group_size()),
        };
        if !matches!(bits, 2 | 4 | 8) {
            return Err(invalid(format!(
                "encoding to {scheme:?} at {bits} bits: the codec packs 2, 4 or 8"
            )));
        }
        let codes_per_word = (32 / bits) as usize;
        #[allow(clippy::cast_precision_loss)]
        let top = ((1u32 << bits) - 1) as f32;
        let group = i64::from(group);
        if group <= 0 || cols % group != 0 {
            return Err(invalid(format!(
                "encoding to {scheme:?} groups {group} columns, which does not \
                 divide the {cols} of {shape:?}"
            )));
        }
        // `decode_values` gives f64; the quantizer rounds in f32, matching
        // MLX's own encoder. dtype comes from the operand, not a byte-count guess.
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
        let mut packed = Vec::with_capacity(values.len() / codes_per_word * 4);
        let mut scale_values = Vec::with_capacity(n_groups);
        let mut bias_values = Vec::with_capacity(n_groups);
        for chunk in values.chunks(group as usize) {
            let (scale, bias) = mlx_affine_group_params_bits(chunk, bits);
            for word in chunk.chunks(codes_per_word) {
                let mut out: u32 = 0;
                for (k, &value) in word.iter().enumerate() {
                    let code = (((value as f32) - bias) / scale).round().clamp(0.0, top) as u32;
                    out |= code << (k as u32 * bits);
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

        // Factor-tensor extents and row-major strides, folded in one pass
        // so they can't be computed from different shapes.
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

        // One odometer over the logical shape; `index` (the factor's flat
        // position) is carried alongside so division happens once per axis step.
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

    /// Decode a self-contained blocked GGUF payload to logical BF16 (the
    /// `Cast(Quant -> Raw)` direction, lowered to `Decode`). Only schemes
    /// with in-block scales are admitted — exactly those with a
    /// `block_layout()` and an arm in [`decode_gguf_block_into`].
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
        // Blocks are independent, so decoding splits across workers into
        // disjoint output slices.
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

    /// Quantize on the host, matching the CUDA encode kernels arm for arm:
    /// `Mxfp4E2M1E8M0` groups 32 elements by absmax -> E8M0 scale;
    /// `Fp8E4M3`/`Int8Symmetric` scale per row by absmax/448 or /127 (dead
    /// row -> factor 1.0). Operand is always BF16 rows; an FP8 block-scaled
    /// source is dequantized first via its own block factors.
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
        // [experts, rows, cols] folds to [experts*rows, cols] in the same
        // byte order (see `types::rectangle`); every line below indexes row*cols+c.
        let Some((rows, cols)) = crate::types::rectangle(&shape) else {
            return Err(invalid(format!(
                "host Encode scales a [rows, cols] rectangle, and {shape:?} has \
                 no axis left over to hold one scale per row"
            )));
        };
        let (rows, cols) = (checked_usize_i64(rows)?, checked_usize_i64(cols)?);
        let scale_shape = self.buffer_shape(scales)?.to_vec();
        // Compared whole rather than folded: the engine binds a scales plane
        // at its declared rank.
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
                    // A dead row (absmax 0) quantizes to all zeros via factor 1.0.
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

    /// Resolves an FP8 block-scaled Encode operand (mirrors
    /// `fp8_tile_scale`). Group size = on-disk weight shape / factor-tensor
    /// shape (must be square); a shard's factor offset comes from the
    /// extent's base offset, in elements (FP8 is one byte each).
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

        // **THE ONE READ THE PLAN'S INSTRUCTION LIST DOES NOT DESCRIBE**, and
        // the reason `consume::SourceLedger` keeps a blocked list beside its
        // read counts. This runs once per SHARD of the payload, over the whole
        // of the factor tensor each time, so a ledger that only counted the
        // extents instructions name would see one read of this span and release
        // it — and the next shard would encode against zeros. The ledger blocks
        // every `metadata_source` span for exactly this call. If a second
        // out-of-band read is ever added to this file, it goes in that list too.
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
        // Checks only the first element each row reads, since the last block may be short.
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

/// Walks a [`GatherSpec`]'s table over `bytes`, copying one block at a time.
/// Rows share one table, so a large permutation is just the row count times
/// a loop over it.
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

    // Folds trailing dense dimensions into one contiguous run, so the loop
    // below moves a run per iteration, not an element; a fully dense extent
    // collapses to one run.
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
        // One dense run: the physical bytes are already the compact bytes.
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

/// A copy is well-formed when both sides span the same byte count, not the
/// same dimensions: source dims describe the checkpoint walk, dest dims a
/// compact block, and the two decompositions need not match.
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

/// The bytes a source extent physically spans, base offset excluded.
///
/// `pub(crate)` for [`consume::SourceLedger`](crate::consume::SourceLedger),
/// which has to predict the ranges [`Walk::read_extent`] will ask for and would
/// be predicting a different checkpoint if it measured them its own way.
pub(crate) fn physical_source_bytes(extent: &Extent) -> Result<u64, Error> {
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
