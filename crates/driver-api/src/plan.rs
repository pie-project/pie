//! Owned driver verb plans shared by local and remote backends.
//!
//! These are process-independent values. Borrowed pointers and completion cells
//! stay in the runtime's local submission layer.

use serde::{Deserialize, Serialize};

use crate::{
    PIE_MEMORY_DOMAIN_HOST_PINNED, PieKvMoveCell, PieMemoryDomain, PiePoolRange, PieStateCopyRange,
};

pub const CHANNEL_TICKET_NONE: u64 = u64::MAX;

/// Binary run-length encoded attention-mask row.
///
/// Even run indices are false and odd run indices are true. A row beginning
/// with true therefore starts with a zero-length false run.
#[derive(Default, Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EncodedMask {
    pub runs: Vec<u32>,
    pub total_size: u64,
}

impl EncodedMask {
    pub fn new(runs: Vec<u32>, total_size: u64) -> Self {
        Self { runs, total_size }
    }

    pub fn len(&self) -> usize {
        self.total_size as usize
    }

    pub fn is_empty(&self) -> bool {
        self.total_size == 0
    }
}

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlan {
    pub token_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub kv_page_indices: Vec<u32>,
    pub kv_page_indptr: Vec<u32>,
    pub kv_last_page_lens: Vec<u32>,
    pub qo_indptr: Vec<u32>,
    pub rs_slot_ids: Vec<u32>,
    pub rs_slot_flags: Vec<u8>,
    pub rs_fold_lens: Vec<u32>,
    pub rs_buffer_slot_ids: Vec<u32>,
    pub rs_buffer_slot_indptr: Vec<u32>,
    /// The buffered prefix each row REPLAYS ahead of its own tokens: the slabs
    /// (`rs_buffer_read_slot_ids`), the row CSR over them
    /// (`rs_buffer_read_indptr`, `R + 1`), and how many tokens of them to
    /// replay (`rs_buffer_read_lens`, `R`). Empty when nothing is buffered.
    pub rs_buffer_read_slot_ids: Vec<u32>,
    pub rs_buffer_read_indptr: Vec<u32>,
    pub rs_buffer_read_lens: Vec<u32>,
    /// Physical offset of each row's logical buffer token 0 (`R`). A fold that
    /// lands mid-page cannot release the page it half-consumed, so the
    /// survivors keep their offsets and every buffer span is `head + logical`.
    pub rs_buffer_heads: Vec<u32>,
    /// WorkingSet-relative buffer page -> physical slot, for channel-resolved
    /// `rs-geometry`. Per REQUEST ROW, unlike [`Self::kv_translation`]: a pass
    /// binds one RS working set per request, so there is no single table for
    /// the fire. `rs_translation_indptr` is the row CSR, `R + 1` entries.
    pub rs_translation: Vec<u32>,
    pub rs_translation_indptr: Vec<u32>,
    pub masks: Vec<EncodedMask>,
    pub mask_indptr: Vec<u32>,
    pub sampling_indices: Vec<u32>,
    pub sampling_indptr: Vec<u32>,
    pub context_ids: Vec<u64>,
    pub single_token_mode: bool,
    pub device_resolved_geometry: bool,
    pub has_user_mask: bool,
    /// tart STRUCTURAL v0/S-2 (0.3 re-port): the pass's layer truncation
    /// (`set-max-layers`). Engine-internal — it crosses the driver ABI as
    /// the region table's per-region k, never as a scalar word.
    pub max_layers: Option<u32>,
    /// The program's hook stages write the `attn_page_mask` sink (Track B
    /// page substitution). Engine-internal, same precedent as
    /// `max_layers`: the admission gate reads it — a page-mask hook needs
    /// the full-R paged decode path, so it cannot ride the banded walk
    /// and its group stays depth-homogeneous.
    #[serde(default)]
    pub hook_page_mask: bool,
    /// The program binds an `AttnMask` descriptor port to a CHANNEL, so the
    /// driver resolves a dense per-cell mask pre-forward. Such a fire must be
    /// submitted SOLO: a multi-program batch cannot merge one program's dense
    /// mask with another's geometry, and the driver fails loud rather than
    /// execute a wrong one.
    #[serde(default)]
    pub dense_device_mask: bool,
    /// Exclusive physical KV page high-water required before this launch.
    #[serde(default)]
    pub required_kv_pages: u32,
    pub image_indptr: Vec<u32>,
    pub image_grids: Vec<u32>,
    pub image_anchor_positions: Vec<u32>,
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    pub image_mrope_positions: Vec<u32>,
    pub image_mrope_indptr: Vec<u32>,
    pub image_patch_positions: Vec<u32>,
    pub image_anchor_rows: Vec<u32>,
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    pub audio_anchor_rows: Vec<u32>,
    pub audio_indptr: Vec<u32>,
    pub embed_rows: Vec<u8>,
    pub embed_indptr: Vec<u32>,
    pub embed_shapes: Vec<u32>,
    pub embed_dtypes: Vec<u8>,
    pub embed_anchor_rows: Vec<u32>,
    pub embed_block_indptr: Vec<u32>,
    pub kv_len: Vec<u32>,
    pub kv_len_device: Vec<u64>,
    pub kv_translation: Vec<u32>,
    pub kv_write_lower_bounds: Vec<u64>,
    pub kv_write_upper_bounds: Vec<u64>,
    pub kv_translation_version: u64,
    pub channel_expected_head: Vec<u64>,
    pub channel_expected_tail: Vec<u64>,
}

/// Why a plan cannot be fired.
///
/// Names the member and both numbers that disagree, because a refusal that
/// says only "invalid" leaves the caller to re-derive what this already knows.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Malformed(pub String);

impl core::fmt::Display for Malformed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Malformed {}

impl LaunchPlan {
    /// Every CSR invariant a fire depends on, checked before anything is
    /// staged or written.
    ///
    /// # Why this exists
    ///
    /// The Metal seam derived each token's physical KV page with
    /// `kv_page_indices.get(virt).copied().unwrap_or(0)`. A request whose CSR
    /// is short — truncated, mis-sized, or simply longer in positions than in
    /// pages — therefore resolved to physical page **0**, which belongs to
    /// some other request, and the fire wrote this request's keys and values
    /// over a stranger's cache. Nothing faults, nothing logs, and the damage
    /// lands on a request that did nothing wrong.
    ///
    /// `unwrap_or(0)` is the whole defect and it cannot be fixed at the call
    /// site, because there is no safe fallback page. The only correct answer
    /// is to refuse the frame — before the pool is touched, which is the
    /// *decide, then move* rule the driver's `store/control.rs` records the
    /// cost of breaking.
    ///
    /// It lives here rather than in a backend because `driver-api` is where
    /// the family's other eleven validators live, and because a check a
    /// backend can skip is one a backend will skip.
    ///
    /// # Errors
    ///
    /// [`Malformed`], naming the member and the numbers that disagree.
    pub fn validate_geometry(&self) -> Result<(), Malformed> {
        let bad = |why: String| Err(Malformed(why));
        let tokens = self.token_ids.len();

        if !self.position_ids.is_empty() && self.position_ids.len() != tokens {
            return bad(format!(
                "position_ids has {} entries for {tokens} tokens",
                self.position_ids.len()
            ));
        }

        // The QO CSR: starts at zero, never decreases, ends at the token
        // count. An empty one is the documented default — one request over
        // every token — and not a defect.
        if !self.qo_indptr.is_empty() {
            if self.qo_indptr[0] != 0 {
                return bad(format!("qo_indptr starts at {}, not 0", self.qo_indptr[0]));
            }
            if let Some(w) = self.qo_indptr.windows(2).find(|w| w[0] > w[1]) {
                return bad(format!("qo_indptr decreases: {} then {}", w[0], w[1]));
            }
            let last = *self.qo_indptr.last().unwrap_or(&0) as usize;
            if last != tokens {
                return bad(format!("qo_indptr ends at {last}, not the {tokens} tokens"));
            }
        }

        // The KV page CSR, checked the same way and then against the array it
        // indexes.
        if !self.kv_page_indptr.is_empty() {
            if self.kv_page_indptr[0] != 0 {
                return bad(format!(
                    "kv_page_indptr starts at {}, not 0",
                    self.kv_page_indptr[0]
                ));
            }
            if let Some(w) = self.kv_page_indptr.windows(2).find(|w| w[0] > w[1]) {
                return bad(format!("kv_page_indptr decreases: {} then {}", w[0], w[1]));
            }
            let last = *self.kv_page_indptr.last().unwrap_or(&0) as usize;
            if last > self.kv_page_indices.len() {
                return bad(format!(
                    "kv_page_indptr ends at {last}, past the {} entries \
                     kv_page_indices holds",
                    self.kv_page_indices.len()
                ));
            }
            if !self.qo_indptr.is_empty() && self.kv_page_indptr.len() != self.qo_indptr.len() {
                return bad(format!(
                    "kv_page_indptr has {} rows and qo_indptr has {}; one of \
                     them is not one-per-request",
                    self.kv_page_indptr.len(),
                    self.qo_indptr.len()
                ));
            }
        }

        // A read-out row that is not a token row reads whatever follows the
        // logits — the same shape of defect, one buffer over.
        if let Some(&row) = self
            .sampling_indices
            .iter()
            .find(|&&r| r as usize >= tokens)
        {
            return bad(format!(
                "sampling_indices names row {row}, past the {tokens} rows this fire has"
            ));
        }

        Ok(())
    }

    /// Whether every token's KV write lands inside its OWN request's pages.
    ///
    /// Separate from [`Self::validate_geometry`] because it needs the pool's
    /// page size, which the plan does not carry. This is the check that
    /// stands between a short CSR and a write into physical page 0.
    ///
    /// # Errors
    ///
    /// [`Malformed`], naming the token, its request, and the span it left.
    pub fn validate_kv_writes(&self, page_size: u32) -> Result<(), Malformed> {
        if self.kv_page_indptr.is_empty() || self.position_ids.is_empty() {
            return Ok(());
        }
        let page = page_size.max(1);
        let req_of_token = self.req_of_token();
        for (t, &pos) in self.position_ids.iter().enumerate() {
            let Some(&r) = req_of_token.get(t) else {
                return Err(Malformed(format!(
                    "token {t} belongs to no request; qo_indptr covers {}",
                    req_of_token.len()
                )));
            };
            let r = r as usize;
            let (Some(&base), Some(&end)) =
                (self.kv_page_indptr.get(r), self.kv_page_indptr.get(r + 1))
            else {
                return Err(Malformed(format!(
                    "request {r} has no kv_page_indptr span; the CSR holds {} rows",
                    self.kv_page_indptr.len()
                )));
            };
            let virt = base as usize + (pos / page) as usize;
            if virt >= end as usize {
                return Err(Malformed(format!(
                    "token {t} of request {r} sits at position {pos}, which wants \
                     virtual page {} of a CSR span holding {} — resolving it \
                     would write KV into another request's pages",
                    pos / page,
                    end - base
                )));
            }
            if virt >= self.kv_page_indices.len() {
                return Err(Malformed(format!(
                    "token {t} of request {r} indexes kv_page_indices[{virt}] of {}",
                    self.kv_page_indices.len()
                )));
            }
        }
        Ok(())
    }

    /// Which request each token belongs to, from the QO CSR.
    ///
    /// An empty `qo_indptr` is the default the field documents: one request
    /// over every token.
    #[must_use]
    pub fn req_of_token(&self) -> Vec<u32> {
        let tokens = self.token_ids.len();
        if self.qo_indptr.len() < 2 {
            return vec![0; tokens];
        }
        let mut out = vec![0u32; tokens];
        for (r, w) in self.qo_indptr.windows(2).enumerate() {
            for slot in out
                .iter_mut()
                .take((w[1] as usize).min(tokens))
                .skip(w[0] as usize)
            {
                *slot = u32::try_from(r).unwrap_or(0);
            }
        }
        out
    }
}

pub const RS_FLAG_RESET: u8 = 1;
pub const RS_FLAG_FOLD: u8 = 2;
pub const RS_FLAG_BUFFER_WRITE: u8 = 4;
/// This row's fold length is not host-known; it comes from the `rs_fold_len`
/// descriptor port and `rs_fold_lens[r]` is a placeholder.
pub const RS_FLAG_FOLD_LEN_DEVICE: u8 = 8;

#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProgramRegistration {
    pub program_hash: u64,
    /// Kernels the host generated for this driver's backend, empty unless the
    /// driver advertised a
    /// [`codegen_backend`](crate::capabilities::DriverCapabilities::codegen_backend).
    #[serde(default)]
    pub emitted_kernels: Vec<EmittedKernel>,
    /// The emitter version `emitted_kernels` was built with; part of the
    /// driver's compile-cache key, so a bump must miss rather than reuse.
    #[serde(default)]
    pub emitter_version: u32,
    /// Per-region bind verdicts and intrinsic side-table analysis, joined to
    /// `emitted_kernels` on `(stage_index, region_index)`.
    #[serde(default)]
    pub region_analysis: Vec<RegionAnalysis>,
    /// The program itself, in the shape a driver executes it.
    #[serde(default)]
    pub launch: LaunchPackage,
    /// The canonical PTIR container, for the in-workspace reference driver
    /// only.
    ///
    /// `driver-dummy` is a Rust crate that links the compiler's own IR and
    /// interpreter, so it cannot drift from them the way a hand-written C++
    /// mirror can. It is the one consumer that still wants PTIR.
    ///
    /// This is deliberately **not** part of the C ABI: [`PieProgramDesc`] has
    /// no counterpart field, so a native driver cannot see PTIR even by
    /// accident.
    ///
    /// [`PieProgramDesc`]: crate::local::PieProgramDesc
    #[serde(default)]
    pub reference_ptir: Vec<u8>,
}

/// **The launch package** — the owned counterpart of
/// [`PieLaunchPackage`](crate::local::PieLaunchPackage).
///
/// This is what a driver receives instead of PTIR. `stages` and `plans` are
/// parallel arrays in attachment order.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPackage {
    pub values: Vec<LaunchValue>,
    pub channels: Vec<LaunchChannel>,
    pub ports: Vec<LaunchPort>,
    /// Program-wide name table for second-party kernels and sinks.
    pub names: Vec<String>,
    pub stages: Vec<LaunchStage>,
    pub plans: Vec<LaunchStagePlan>,
}

/// One declared SSA value. Owned counterpart of
/// [`PieLaunchValue`](crate::local::PieLaunchValue).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchValue {
    pub id: u32,
    /// `PIE_VALUE_*`.
    pub source: u8,
    pub dtype: u8,
    /// `PTIR_INTR_*` when `source` is `PIE_VALUE_INTRINSIC`.
    pub intrinsic: u8,
    pub channel: u32,
    pub literal_bits: u32,
    pub shape: Vec<u32>,
}

/// One op in a stage DAG. Owned counterpart of
/// [`PieLaunchOp`](crate::local::PieLaunchOp).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchOp {
    /// `PTIR_OP_*`.
    pub code: u16,
    pub result_count: u16,
    pub result_id: u32,
    /// `PTIR_INTR_*`, for `intrinsic_val`.
    pub intrinsic: u16,
    pub lit_dtype: u8,
    pub dtype: u8,
    pub pred_tag: u8,
    pub rng_kind: u8,
    pub lit_bits: u32,
    pub pred_payload: u32,
    /// Channel slot, or `u32::MAX` when the op touches no channel.
    pub channel: u32,
    pub name_index: u32,
    pub imm: u32,
    pub imm2: u32,
    pub imm3: u32,
    pub args: Vec<u32>,
    pub shape: Vec<u32>,
}

/// One channel declaration. Owned counterpart of
/// [`PieLaunchChannel`](crate::local::PieLaunchChannel).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannel {
    pub id: u32,
    pub capacity: u32,
    pub dtype: u8,
    /// `PIE_CHANNEL_*` bits.
    pub flags: u8,
    /// -1 private, 0 import, 1 export.
    pub extern_dir: i8,
    /// `PIE_READINESS_*` — the direction this channel's first op requires.
    pub readiness: u8,
    pub shape: Vec<u32>,
    pub extern_name: Vec<u8>,
}

/// One descriptor-port binding. Owned counterpart of
/// [`PieLaunchPort`](crate::local::PieLaunchPort).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPort {
    /// `PTIR_PORT_*`.
    pub port: u8,
    pub is_const: bool,
    pub const_dtype: u8,
    pub channel: u32,
    pub const_shape: Vec<u32>,
    pub const_data: Vec<u8>,
}

/// A `(channel, value)` pair. Owned counterpart of
/// [`PieLaunchPut`](crate::local::PieLaunchPut).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPut {
    pub channel: u32,
    pub value: u32,
}

/// One stage program. Owned counterpart of
/// [`PieLaunchStage`](crate::local::PieLaunchStage).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStage {
    /// Prologue 0, OnAttnProj 1, OnAttn 2, Epilogue 3.
    pub kind: u8,
    pub ops: Vec<LaunchOp>,
    pub puts: Vec<LaunchPut>,
    pub takes: Vec<u32>,
    pub reads: Vec<u32>,
}

/// One region. Owned counterpart of
/// [`PieLaunchRegion`](crate::local::PieLaunchRegion).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRegion {
    /// `PIE_REGION_GENERATED` or `PIE_REGION_LIBRARY`.
    pub kind: u8,
    pub library: u8,
    pub schedule: u8,
    pub nodes: Vec<u32>,
    pub inputs: Vec<u32>,
    pub outputs: Vec<u32>,
    pub sinks: Vec<LaunchPut>,
}

/// One normalized value type. Owned counterpart of
/// [`PieLaunchPlanValue`](crate::local::PieLaunchPlanValue).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlanValue {
    pub dtype: u8,
    /// Per-dimension extent kind: `PIE_EXTENT_STATIC` or a runtime extent.
    pub extents: Vec<u8>,
    /// Per-dimension literal extent, meaningful where `extents` is static.
    pub dims: Vec<u32>,
}

/// One lane-binding rule. Owned counterpart of
/// [`PieLaunchChannelRule`](crate::local::PieLaunchChannelRule).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannelRule {
    pub value: u32,
    pub local: u32,
}

/// The per-program launch plan for one stage. Owned counterpart of
/// [`PieLaunchStagePlan`](crate::local::PieLaunchStagePlan).
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStagePlan {
    pub signature_hash: u64,
    /// Graph-cache identity (`tensor_compiler::plan::stage_identity`).
    pub identity: u64,
    /// `PIE_STAGE_REQUIRES_*` bits.
    pub flags: u32,
    pub mtp_rows: u32,
    pub ops: Vec<LaunchOp>,
    /// Source op positions each normalized op covers.
    pub source_ops: Vec<Vec<u32>>,
    pub value_types: Vec<LaunchPlanValue>,
    /// Local channel slot → program-global dense channel index.
    pub channel_bindings: Vec<u32>,
    /// Local name slot → canonical second-party kernel name.
    pub names: Vec<String>,
    pub singleton: Vec<LaunchRegion>,
    pub fused: Vec<LaunchRegion>,
    /// Runtime extents any value in the stage depends on, ascending.
    pub used_extents: Vec<u8>,
    pub channel_rules: Vec<LaunchChannelRule>,
    /// Why `PIE_STAGE_GROUPED_VALID` is clear. Empty when it is set.
    pub error: String,
}

/// Every per-region decision the host made. The owned counterpart of
/// [`PieRegionAnalysis`](crate::local::PieRegionAnalysis).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegionAnalysis {
    pub stage_index: u32,
    pub region_index: u32,
    /// `PIE_REGION_*` bits.
    pub flags: u32,
    pub direct_argmax: Vec<DirectArgmax>,
    /// Nodes the rewrites make redundant, ascending.
    pub skipped: Vec<u32>,
}

/// One `argmax` that reads a logits intrinsic's buffer directly. The owned
/// counterpart of [`PieDirectArgmax`](crate::local::PieDirectArgmax).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DirectArgmax {
    pub node: u32,
    pub source_value: u32,
    pub intrinsic: u16,
    pub requires_single_row: u8,
}

/// One host-emitted kernel. The owned counterpart of
/// [`PieEmittedKernel`](crate::local::PieEmittedKernel).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmittedKernel {
    /// `PIE_KERNEL_*`.
    pub kind: u32,
    pub stage_index: u32,
    pub region_index: u32,
    /// Entry-point symbol; empty when emission failed.
    pub entry_name: String,
    /// Backend source; empty when emission failed.
    pub source: String,
    /// Why emission failed. Empty on success.
    ///
    /// A failure is not necessarily fatal: a driver may have a slower path for
    /// the same region, and recording *why* is what lets it tell a deliberate
    /// fallback from a bug.
    pub error: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChannelRegistrationPlan {
    pub driver_id: usize,
    pub channel_id: u64,
    pub shape: Vec<u32>,
    pub dtype: u8,
    pub host_role: u8,
    pub seeded: bool,
    pub extern_dir: u8,
    pub capacity: u32,
    pub reader_wait_id: u64,
    pub writer_wait_id: u64,
    pub extern_name: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCopyPlan {
    pub src_domain: PieMemoryDomain,
    pub src_device_ordinal: u32,
    pub dst_domain: PieMemoryDomain,
    pub dst_device_ordinal: u32,
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
    pub cells: Vec<PieKvMoveCell>,
}

impl Default for KvCopyPlan {
    fn default() -> Self {
        Self {
            src_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            src_device_ordinal: 0,
            dst_domain: PIE_MEMORY_DOMAIN_HOST_PINNED,
            dst_device_ordinal: 0,
            src_page_ids: Vec::new(),
            dst_page_ids: Vec::new(),
            cells: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StateCopyPlan {
    pub slot_ranges: Vec<PieStateCopyRange>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MediaEncodePlan {
    pub image_grids: Vec<u32>,
    pub image_pixels: Vec<u8>,
    pub image_pixel_indptr: Vec<u32>,
    pub image_patch_positions: Vec<u32>,
    pub image_anchor_rows: Vec<u32>,
    pub audio_features: Vec<u8>,
    pub audio_feature_indptr: Vec<u32>,
    pub audio_anchor_rows: Vec<u32>,
    pub output_rows: Vec<u8>,
    pub output_row_indptr: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PoolResizePlan {
    pub pool_id: u64,
    pub target_pages: u64,
    pub map_ranges: Vec<PiePoolRange>,
    pub unmap_ranges: Vec<PiePoolRange>,
}

#[cfg(test)]
mod geometry_tests {
    use super::*;

    /// Two requests, two pages each, four tokens — a well-formed frame.
    fn sound() -> LaunchPlan {
        LaunchPlan {
            token_ids: vec![10, 11, 12, 13],
            position_ids: vec![0, 1, 0, 1],
            qo_indptr: vec![0, 2, 4],
            kv_page_indptr: vec![0, 2, 4],
            // Request 0 owns physical pages 7 and 8; request 1 owns 3 and 4.
            kv_page_indices: vec![7, 8, 3, 4],
            ..LaunchPlan::default()
        }
    }

    #[test]
    fn a_sound_frame_passes_both_halves() {
        let p = sound();
        p.validate_geometry().expect("a sound frame");
        p.validate_kv_writes(16)
            .expect("every write is in its own span");
        assert_eq!(p.req_of_token(), vec![0, 0, 1, 1]);
    }

    /// THE defect this exists for.
    ///
    /// Request 1's positions run past the pages its CSR span holds. The seam
    /// resolved that with `kv_page_indices.get(virt).copied().unwrap_or(0)`,
    /// which is physical page **0** — a page belonging to some other request
    /// entirely. The fire then wrote request 1's keys over that request's
    /// cache, and nothing faulted.
    #[test]
    fn a_token_past_its_own_csr_span_is_refused_not_folded_to_page_zero() {
        let mut p = sound();
        // One page each now, but request 1 still has two tokens at positions
        // 0 and 1 — and with a page size of 1, position 1 wants a second page
        // it does not own.
        p.kv_page_indptr = vec![0, 1, 2];
        p.kv_page_indices = vec![7, 3];
        let why = p
            .validate_kv_writes(1)
            .expect_err("a token past its span must be refused")
            .0;
        assert!(
            why.contains("another request's pages"),
            "the refusal must say what the fold would have cost: {why}"
        );
        // And the geometry half is happy with it, which is why the two are
        // separate checks: the CSR is internally consistent and still wrong
        // for these positions.
        p.validate_geometry()
            .expect("the CSR itself is well-formed");
    }

    #[test]
    fn a_csr_that_disagrees_with_itself_is_refused_by_member() {
        let mut p = sound();
        p.qo_indptr = vec![0, 2, 3];
        assert!(
            p.validate_geometry()
                .unwrap_err()
                .0
                .contains("qo_indptr ends at 3"),
            "the count that disagrees has to be in the message"
        );

        let mut p = sound();
        p.qo_indptr = vec![0, 3, 2];
        assert!(p.validate_geometry().unwrap_err().0.contains("decreases"));

        let mut p = sound();
        p.kv_page_indptr = vec![0, 2, 9];
        assert!(
            p.validate_geometry()
                .unwrap_err()
                .0
                .contains("past the 4 entries"),
            "a CSR ending past its own index array is the short-CSR case"
        );

        let mut p = sound();
        p.position_ids = vec![0, 1];
        assert!(
            p.validate_geometry()
                .unwrap_err()
                .0
                .contains("position_ids")
        );

        // A read-out row that is not a token row reads whatever follows the
        // logits.
        let mut p = sound();
        p.sampling_indices = vec![0, 9];
        assert!(
            p.validate_geometry()
                .unwrap_err()
                .0
                .contains("sampling_indices names row 9")
        );
    }

    /// An empty CSR is the documented DEFAULT — one request over every token
    /// — and must not be refused as malformed.
    #[test]
    fn the_documented_defaults_are_not_refused() {
        let p = LaunchPlan {
            token_ids: vec![1, 2, 3],
            ..LaunchPlan::default()
        };
        p.validate_geometry().expect("an empty CSR is a default");
        p.validate_kv_writes(16)
            .expect("no KV family, nothing to check");
        assert_eq!(p.req_of_token(), vec![0, 0, 0]);
    }
}
