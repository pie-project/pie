//! Env-gated per-kernel activation dump for the accuracy gate.
//!
//! The MLX reference writes `<dir>/<layer>.<kernel>.npy` for every tapped
//! intermediate. This is the raw-Metal counterpart: identical file names,
//! identical shapes, so the two trees can be diffed tap by tap and the
//! FIRST diverging (kernel, layer) named.
//!
//! It is off unless `PIE_METAL_GOLDEN_DIR` names a directory. When on, the
//! scratch schedule switches to `no_recycle` (every activation value gets
//! its own pool buffer, so nothing is overwritten before the dump) and the
//! pool is allocated CPU-visible. Both are diagnostic-only; the shipped
//! path is untouched.
//!
//! The mixture's taps exist for a reason worth keeping: the sorted tensors
//! are deliberately untapped — their row order is the driver's own, so a
//! dump of them would diff against nothing — but everything a reference can
//! also produce is tapped in TOKEN order, which until this table existed
//! was nothing at all. The routed FFN was the one block of the family no
//! parity run could see, and it is where Qwen3.6-35B-A3B went wrong.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::region::Region;
use crate::tuning::Tuning;

use super::abi::Kernel;
use super::color::ScratchSchedule;
use super::geometry::DecodeGeometry;
use super::logits::bf16_to_f32;
use super::sizing::{RoutedProjection, moe_sorted_rows};

/// One kernel's tap: what the reference calls it, which scratch bind
/// carries its output, and how wide one row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tap {
    /// The reference's name for this intermediate.
    pub name: &'static str,
    /// The scratch bind index the kernel writes its output at.
    pub out_bind: u8,
    /// Elements per row.
    pub width: u32,
}

/// The tap for `kind`, or `None` for a kernel the reference has no
/// counterpart to (QSplit and KvAppend are layout moves, Argmax is
/// host-side).
///
/// Mirrors the reference's dump call sites exactly; widths come from the
/// geometry because they are the geometry's numbers.
#[must_use]
#[allow(clippy::match_same_arms)]
pub fn tap_for(kind: Kernel, g: &DecodeGeometry) -> Option<Tap> {
    let q_dim = g.n_q_heads * g.head_dim;
    let kv_dim = g.n_kv_heads * g.head_dim;
    let tap = |name, out_bind, width| {
        Some(Tap {
            name,
            out_bind,
            width,
        })
    };
    match kind {
        // Tied or untied, the reference's tap is one `embed`: the kinds
        // differ in which tensor is asked for, not in what comes out.
        Kernel::EmbedGather | Kernel::EmbedUntied => tap("embed", 4, g.hidden),
        Kernel::Rms => tap("attn_norm", 2, g.hidden),
        Kernel::FfnRms => tap("ffn_norm", 2, g.hidden),
        Kernel::FinalRms => tap("final_norm", 2, g.hidden),

        Kernel::QmvIn => tap("gdn_in_qkv", 4, g.gdn_conv_dim),
        Kernel::QmvInZ => tap("gdn_in_z", 4, g.gdn_v_total),
        Kernel::GdnInA => tap("gdn_in_a", 4, g.gdn_v_heads),
        Kernel::GdnInB => tap("gdn_in_b", 4, g.gdn_v_heads),
        // The reference's `gdn_core` tap is the output of gated_delta_net,
        // which already includes the gate RMSNorm — so it lines up with
        // GatedRms here, not with the bare recurrence.
        Kernel::GatedRms => tap("gdn_core", 3, g.gdn_v_heads * g.gdn_v_dim),
        Kernel::QmvOut => tap("gdn_out", 4, g.hidden),

        Kernel::QmvQ => tap("q_proj", 4, 2 * q_dim),
        Kernel::QmvK => tap("k_proj", 4, kv_dim),
        Kernel::QmvV => tap("v_proj", 4, kv_dim),
        Kernel::QNorm => tap("q_norm", 2, q_dim),
        Kernel::KNorm => tap("k_norm", 2, kv_dim),
        Kernel::Rope => tap("rope_q", 0, q_dim),
        Kernel::RopeK => tap("rope_k", 0, kv_dim),
        Kernel::Sdpa | Kernel::SdpaPaged => tap("sdpa", 3, q_dim),
        Kernel::AttnGate => tap("gated", 0, q_dim),
        Kernel::QmvO => tap("o_proj", 4, g.hidden),
        Kernel::Residual => tap("attn_resid", 2, g.hidden),

        Kernel::QmvGate => tap("gate_proj", 4, g.intermediate),
        Kernel::QmvUp => tap("up_proj", 4, g.intermediate),
        // Routed, the dense SwiGLU that remains is the SHARED expert's, at
        // its own width. Named `swiglu` at `g.intermediate` for both, this
        // tap was zero elements wide on every routed checkpoint — present,
        // empty, and silently skipped by anything comparing it.
        Kernel::SiluMul => {
            if g.is_moe() {
                tap("shared_act", 2, g.shared_intermediate)
            } else {
                tap("swiglu", 2, g.intermediate)
            }
        }
        Kernel::QmvDown => tap("down_proj", 4, g.hidden),
        Kernel::LayerOut => tap("layer_out", 2, g.hidden),

        Kernel::LlRouter => tap("router", 4, g.n_experts),
        Kernel::LlMoeCombine => tap("moe_out", 2, g.hidden),

        // ── gpt-oss, through the shared-geometry view: the sorted stack is
        // `experts_per_token` rows at decode (tile 1), so the routed taps
        // carry the whole stack and a comparer reads it row-wise. ──
        Kernel::GoQmvQ => tap("q_proj", 4, q_dim),
        Kernel::GoQmvK => tap("k_proj", 4, kv_dim),
        Kernel::GoQmvV => tap("v_proj", 4, kv_dim),
        Kernel::GoSdpaSink => tap("sdpa", 3, q_dim),
        Kernel::GoQmvO => tap("o_proj", 4, g.hidden),
        Kernel::GoRouter => tap("router", 4, g.n_experts),
        Kernel::GoExpertGate => tap("expert_gate", 4, g.experts_per_token * g.moe_intermediate),
        Kernel::GoExpertUp => tap("expert_up", 4, g.experts_per_token * g.moe_intermediate),
        Kernel::GoSwiGlu => tap("expert_act", 2, g.experts_per_token * g.moe_intermediate),
        Kernel::GoExpertDown => tap("expert_down", 4, g.experts_per_token * g.hidden),
        Kernel::GoExpertCombine => tap("moe_out", 2, g.hidden),
        Kernel::LlSharedGate => tap("shared_gate", 4, g.shared_intermediate),
        Kernel::LlSharedUp => tap("shared_up", 4, g.shared_intermediate),
        Kernel::LlSharedDown => tap("shared_down", 4, g.hidden),
        Kernel::LlSharedGateProj => tap("shared_g", 4, 1),
        Kernel::LlSharedCombine => tap("ffn_out", 3, g.hidden),
        _ => None,
    }
}

/// The dump directory, or `None` when the dump is off.
///
/// Created here, once, rather than left to the caller: the npy writer used
/// to open its file and RETURN SILENTLY on failure, so a directory that did
/// not exist produced a run that looked like a successful dump and left
/// nothing behind — and the diff that was supposed to bisect a wrong answer
/// had no files to compare. Failing to create is still not fatal (the run
/// is a valid benchmark without its dump), but it is no longer silent: the
/// error is returned for the caller to log.
///
/// # Errors
///
/// The `create_dir_all` failure, when the variable is set but the
/// directory cannot exist.
pub fn dir_from_env() -> Result<Option<PathBuf>, std::io::Error> {
    match std::env::var_os("PIE_METAL_GOLDEN_DIR") {
        None => Ok(None),
        Some(dir) if dir.is_empty() => Ok(None),
        Some(dir) => {
            let dir = PathBuf::from(dir);
            std::fs::create_dir_all(&dir)?;
            Ok(Some(dir))
        }
    }
}

/// Whether a tap dump keeps the ordinary scratch recycling
/// (`PIE_METAL_TAPS_RECYCLE=1`).
///
/// A dump normally gives every value its own pool buffer, because a
/// recycled producer's buffer holds someone else's value by the time the
/// fire retires and there would be nothing to read. The cost is that the
/// taps then describe an allocation no fire outside the dump ever runs
/// against — so a colouring that reuses a buffer while its value is still
/// live is the ONE class of defect that dump structurally cannot see.
/// Recycling dumps against the real allocation instead: values whose
/// buffer was reused read as whatever overwrote them, which is exactly
/// what makes the two dumps comparable and the first diverging tap the
/// answer.
#[must_use]
pub fn taps_recycle() -> bool {
    std::env::var_os("PIE_METAL_TAPS_RECYCLE").is_some()
}

/// Write `[rows, width]` f32 data as a v1 `.npy`.
///
/// # Errors
///
/// The file open or write failure — surfaced, not swallowed; see
/// [`dir_from_env`] for the run this cost once.
pub fn write_npy(path: &Path, data: &[f32], rows: u32, width: u32) -> Result<(), std::io::Error> {
    let mut header =
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {width}), }}");
    // Magic + version + length + text + '\n' must be 64-byte aligned.
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut out = std::fs::File::create(path)?;
    out.write_all(b"\x93NUMPY\x01\x00")?;
    out.write_all(
        &u16::try_from(header.len())
            .expect("an npy header is short")
            .to_le_bytes(),
    )?;
    out.write_all(header.as_bytes())?;
    for value in data {
        out.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// One dispatch of the DAG, as much of it as tapping needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TapSite {
    /// The kernel dispatched.
    pub kind: Kernel,
    /// Its layer, or `None` for the pre/post-stack dispatches, whose taps
    /// are named without a layer prefix.
    pub layer: Option<u32>,
}

/// Write every tapped activation of the DAG as `<dir>/<layer>.<name>.npy`,
/// f32, shape `[n_rows, width]`. Row `t` is read at `t * row_stride_bytes`
/// inside its pool slot, which is how the binder lays per-token prefill
/// rows out.
///
/// q_norm/k_norm, both ropes and attn_gate rewrite their input in place,
/// so under `no_recycle` they share one buffer with the tap before them
/// and that buffer only ever holds the LAST writer's value. Dumping the
/// earlier name too would publish the later tensor under it and read as a
/// divergence that is really the dump lying. Only the final writer of a
/// colour is named.
///
/// # Errors
///
/// The first file that fails to write.
///
/// # Safety
///
/// The GPU must be done with the pool: the fire whose activations are
/// being dumped has completed and nothing has been encoded against the
/// pool since.
pub unsafe fn dump_taps<R: Region>(
    dir: &Path,
    sites: &[TapSite],
    schedule: &ScratchSchedule,
    pool: &[R],
    geometry: &DecodeGeometry,
    n_rows: u32,
    row_stride_bytes: u64,
) -> Result<(), std::io::Error> {
    if n_rows == 0 {
        return Ok(());
    }
    let n = sites.len().min(schedule.per_dispatch.len());

    let color_of = |ordinal: usize, out_bind: u8| {
        schedule.per_dispatch[ordinal]
            .iter()
            .find(|bind| bind.bind_index == out_bind)
            .map(|bind| bind.color as usize)
    };

    let mut last_writer: Vec<Option<usize>> = vec![None; pool.len()];
    for (ordinal, site) in sites.iter().enumerate().take(n) {
        let Some(tap) = tap_for(site.kind, geometry) else {
            continue;
        };
        if let Some(color) = color_of(ordinal, tap.out_bind)
            && color < pool.len()
        {
            last_writer[color] = Some(ordinal);
        }
    }

    for (ordinal, site) in sites.iter().enumerate().take(n) {
        let Some(tap) = tap_for(site.kind, geometry) else {
            continue;
        };
        let Some(color) = color_of(ordinal, tap.out_bind) else {
            continue;
        };
        if color >= pool.len() || last_writer[color] != Some(ordinal) {
            continue;
        }
        let region = &pool[color];
        let mut rows = Vec::with_capacity(n_rows as usize * tap.width as usize);
        for t in 0..u64::from(n_rows) {
            let offset = t * row_stride_bytes;
            if region
                .check("tap slot", offset, u64::from(tap.width) * 2)
                .is_err()
            {
                rows.resize(rows.len() + tap.width as usize, 0.0);
                continue;
            }
            // SAFETY: `check` bounded the span inside the region, and the
            // caller's contract covers the GPU.
            let src = unsafe {
                std::slice::from_raw_parts(
                    region.contents().cast::<u8>().as_ptr().add(offset as usize),
                    tap.width as usize * 2,
                )
            };
            rows.extend(
                src.chunks_exact(2)
                    .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))),
            );
        }
        let name = match site.layer {
            None => tap.name.to_string(),
            Some(layer) => format!("{layer}.{}", tap.name),
        };
        write_npy(&dir.join(format!("{name}.npy")), &rows, n_rows, tap.width)?;
    }
    Ok(())
}

/// Write one already-materialized bf16 tensor (the lm_head logits live in
/// IO, not scratch, so they never pass through the schedule).
///
/// # Errors
///
/// The file write failure.
pub fn dump_bf16(
    dir: &Path,
    name: &str,
    bf16: &[u16],
    rows: u32,
    width: u32,
    row_stride_elems: usize,
) -> Result<(), std::io::Error> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    let mut out = Vec::with_capacity(rows as usize * width as usize);
    for r in 0..rows as usize {
        let row = &bf16[r * row_stride_elems..r * row_stride_elems + width as usize];
        out.extend(row.iter().map(|&bits| bf16_to_f32(bits)));
    }
    write_npy(&dir.join(format!("{name}.npy")), &out, rows, width)
}

/// Write one bf16 tensor that the expert sort REORDERED, restoring the
/// layout it had before the sort: `[rows, slots * width]`, slot-major,
/// which is what every reference and every bisect already speaks.
///
/// `perm` is the sort's own output — its within-expert order is decided by
/// atomics and is not reproducible on the host, so it is read rather than
/// recomputed. `perm[p]` is the `(token, slot)` pair stored at row `p`, or
/// `-1` for a padding row. Pairs the sort never wrote stay zero.
///
/// # Errors
///
/// The file write failure.
pub fn dump_bf16_sorted(
    dir: &Path,
    name: &str,
    bf16: &[u16],
    perm: &[i32],
    rows: u32,
    slots: u32,
    width: u32,
) -> Result<(), std::io::Error> {
    if rows == 0 || slots == 0 || width == 0 {
        return Ok(());
    }
    let pairs = rows as usize * slots as usize;
    let mut out = vec![0.0f32; pairs * width as usize];
    for (p, &sel) in perm.iter().enumerate() {
        let Ok(sel) = usize::try_from(sel) else {
            continue; // a padding row
        };
        if sel >= pairs {
            continue;
        }
        let stored = &bf16[p * width as usize..(p + 1) * width as usize];
        let dst = &mut out[sel * width as usize..(sel + 1) * width as usize];
        for (slot, &bits) in dst.iter_mut().zip(stored) {
            *slot = bf16_to_f32(bits);
        }
    }
    write_npy(&dir.join(format!("{name}.npy")), &out, rows, slots * width)
}

/// The exact token ids this pass ran, so the reference can be regenerated
/// on them.
///
/// # Errors
///
/// The file write failure.
pub fn dump_tokens(dir: &Path, ids: &[u32]) -> Result<(), std::io::Error> {
    if ids.is_empty() {
        return Ok(());
    }
    let line = ids.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    std::fs::write(dir.join("tokens.txt"), line + "\n")
}

/// How many rows the sorted dump must expect its stored tensor to hold —
/// the sort's own bound, asked from the one place that owns it.
#[must_use]
pub fn sorted_dump_rows(geometry: &DecodeGeometry, tuning: &Tuning, n_tokens: u32) -> u64 {
    moe_sorted_rows(geometry, tuning, n_tokens, RoutedProjection::Matmul)
}

#[cfg(test)]
mod tests {
    use core::ffi::c_void;
    use core::ptr::NonNull;

    use super::super::color::Use;
    use super::super::color::schedule_scratch;
    use super::*;

    #[derive(Debug)]
    struct Host(Vec<u8>);

    // SAFETY: the pointer is the Vec's allocation and `len` its length.
    unsafe impl Region for Host {
        fn contents(&self) -> NonNull<c_void> {
            NonNull::new(self.0.as_ptr().cast_mut().cast()).expect("vec allocates")
        }
        fn len(&self) -> u64 {
            self.0.len() as u64
        }
    }

    fn read_npy(path: &Path) -> (String, Vec<f32>) {
        let bytes = std::fs::read(path).expect("the dump wrote");
        assert_eq!(&bytes[0..8], b"\x93NUMPY\x01\x00");
        let len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + len) % 64, 0, "the header is 64-byte aligned");
        let header = String::from_utf8(bytes[10..10 + len].to_vec()).unwrap();
        let data = bytes[10 + len..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        (header, data)
    }

    fn bf16_bits(value: f32) -> u16 {
        (value.to_bits() >> 16) as u16
    }

    #[test]
    fn the_tap_table_speaks_the_references_names_and_widths() {
        let dense = DecodeGeometry::default();
        assert_eq!(
            tap_for(Kernel::EmbedGather, &dense),
            Some(Tap {
                name: "embed",
                out_bind: 4,
                width: 1024
            })
        );
        assert_eq!(
            tap_for(Kernel::GatedRms, &dense).unwrap().name,
            "gdn_core",
            "the reference taps gated_delta_net AFTER its gate RMSNorm"
        );
        assert_eq!(tap_for(Kernel::QSplit, &dense), None);

        // The swiglu tap that was present-but-empty on routed checkpoints.
        assert_eq!(tap_for(Kernel::SiluMul, &dense).unwrap().name, "swiglu");
        let routed = DecodeGeometry {
            n_experts: 64,
            experts_per_token: 4,
            moe_intermediate: 768,
            shared_intermediate: 512,
            ..DecodeGeometry::default()
        };
        let shared = tap_for(Kernel::SiluMul, &routed).unwrap();
        assert_eq!((shared.name, shared.width), ("shared_act", 512));
    }

    #[test]
    fn only_the_final_writer_of_a_colour_is_named() {
        let dir = std::env::temp_dir().join("golden-tap-last-writer");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let geometry = DecodeGeometry {
            hidden: 4,
            n_q_heads: 1,
            head_dim: 4,
            ..DecodeGeometry::default()
        };
        // Rope rewrites QNorm's buffer in place: same value, same colour.
        let sites = [
            TapSite {
                kind: Kernel::QNorm,
                layer: Some(0),
            },
            TapSite {
                kind: Kernel::Rope,
                layer: Some(0),
            },
        ];
        let uses = [
            Use {
                ordinal: 0,
                bind_index: 2,
                value: 0,
                is_write: true,
            },
            Use {
                ordinal: 1,
                bind_index: 0,
                value: 0,
                is_write: true,
            },
        ];
        let schedule = schedule_scratch(2, &uses, &[0, 1], 1, false).unwrap();
        let pool = [Host(
            [1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|&v| bf16_bits(v).to_le_bytes())
                .collect(),
        )];
        unsafe { dump_taps(&dir, &sites, &schedule, &pool, &geometry, 1, 8) }.unwrap();
        assert!(
            !dir.join("0.q_norm.npy").exists(),
            "the earlier name would publish the later tensor"
        );
        let (header, data) = read_npy(&dir.join("0.rope_q.npy"));
        assert!(header.contains("(1, 4)"));
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn the_sorted_dump_restores_token_order_and_zeroes_padding() {
        let dir = std::env::temp_dir().join("golden-tap-sorted");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Two (token, slot) pairs stored in sort order [pair 1, padding,
        // pair 0]; width 2.
        let stored: Vec<u16> = [10.0f32, 11.0, 0.0, 0.0, 20.0, 21.0]
            .iter()
            .map(|&v| bf16_bits(v))
            .collect();
        let perm = [1, -1, 0];
        dump_bf16_sorted(&dir, "gathered", &stored, &perm, 1, 2, 2).unwrap();
        let (header, data) = read_npy(&dir.join("gathered.npy"));
        assert!(header.contains("(1, 4)"), "[rows, slots * width]");
        assert_eq!(data, [20.0, 21.0, 10.0, 11.0]);
    }

    #[test]
    fn tokens_are_one_comma_line_the_reference_reruns() {
        let dir = std::env::temp_dir().join("golden-tap-tokens");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dump_tokens(&dir, &[5, 7, 11]).unwrap();
        assert_eq!(
            std::fs::read_to_string(dir.join("tokens.txt")).unwrap(),
            "5,7,11\n"
        );
    }
}
