//! Owned driver verb plans shared by local and remote backends.
//!
//! These are process-independent values. Borrowed pointers and completion cells
//! stay in the runtime's local submission layer.

use serde::{Deserialize, Serialize};

use crate::{DeviceDomain, KvMoveCell, PIE_MEMORY_DOMAIN_HOST_PINNED, PoolRange, StateCopyRange};

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

    /// Expand the runs into `dst`, setting one bit per true element.
    ///
    /// `dst` must be at least [`Self::words`] long; bits past [`Self::len`]
    /// are never written, so a longer buffer keeps whatever it held.
    fn expand_into(&self, dst: &mut [u32]) {
        let total = self.len();
        let mut at = 0usize;
        for (index, &run) in self.runs.iter().enumerate() {
            let end = at.saturating_add(run as usize).min(total);
            if index % 2 == 1 {
                for bit in at..end {
                    dst[bit / 32] |= 1 << (bit % 32);
                }
            }
            if end == total {
                break;
            }
            at = end;
        }
    }

    /// How many `u32` words [`Self::expand_into`] needs.
    #[must_use]
    pub fn words(&self) -> usize {
        self.len().div_ceil(32)
    }
}

/// [`LaunchPlan::masks`] expanded into the dense per-row bitmask a driver
/// stages, with the two CSRs that index it.
///
/// # Why this is a type and not three returns
///
/// The three arrays are only meaningful together — `request_indptr` indexes
/// `word_indptr`, which indexes `words` — and the mask consumer takes them as
/// a triple. Naming the triple is what stops a caller from pairing one plan's
/// words with another's CSR.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
pub struct MaskWords {
    /// Per request row, into `word_indptr`. Always `requests + 1` entries.
    pub request_indptr: Vec<u32>,
    /// Per mask, into `words`. Always `masks + 1` entries.
    pub word_indptr: Vec<u32>,
    /// The bitmask words themselves, one run of [`EncodedMask::words`] each.
    pub words: Vec<u32>,
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
    /// [`Self::masks`] as dense bitmask words, with the CSRs that index them.
    ///
    /// # Why this lives here
    ///
    /// [`EncodedMask`] is defined in this module, so its expansion is this
    /// module's business — the same rule that put `adopt` beside the two
    /// representations it converts between. It was in the engine
    /// (`driver/abi.rs`, `MaskWordsStorage`) only because the C view was
    /// built there and the CUDA driver read the result through a
    /// `#[repr(C)]` descriptor. A driver that takes the owned plan needs the
    /// expansion itself, and two drivers needing it would have meant two
    /// copies of a bit loop.
    ///
    /// An empty [`Self::mask_indptr`] yields the all-zero request CSR, which
    /// is the documented "no row names a mask" default rather than a defect.
    #[must_use]
    pub fn bitmask_words(&self) -> MaskWords {
        let requests = self.qo_indptr.len().saturating_sub(1);
        let request_indptr = if self.mask_indptr.is_empty() {
            vec![0; requests + 1]
        } else {
            self.mask_indptr.clone()
        };

        let mut word_indptr = Vec::with_capacity(self.masks.len() + 1);
        let mut words = Vec::new();
        word_indptr.push(0);
        for mask in &self.masks {
            let start = words.len();
            words.resize(start + mask.words(), 0);
            mask.expand_into(&mut words[start..]);
            word_indptr.push(u32::try_from(words.len()).unwrap_or(u32::MAX));
        }
        MaskWords {
            request_indptr,
            word_indptr,
            words,
        }
    }

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
    /// The canonical PTIR container. **Written, and no longer read.**
    ///
    /// It existed for the in-workspace reference driver: `driver-dummy` was
    /// a Rust crate that linked the compiler's own IR and interpreter, so it
    /// could not drift from them the way a hand-written C++ mirror can, and
    /// it was the one consumer that still wanted PTIR. That crate is
    /// deleted, and nothing else asks for this field — `engine` fills it at
    /// two sites and every remaining backend takes [`LaunchPackage`]
    /// instead.
    ///
    /// It is kept rather than removed in the same change that removed its
    /// consumer, because the two are separable and only one of them was
    /// asked for: a field nothing reads costs a clone per program
    /// registration, which is worth measuring before deciding, and PTIR is
    /// the shape a future in-workspace interpreter would want back.
    ///
    /// This is deliberately **not** part of the C ABI: [`PieProgramDesc`] has
    /// no counterpart field, so a native driver cannot see PTIR even by
    /// accident.
    ///
    /// [`PieProgramDesc`]: crate::local::PieProgramDesc
    #[serde(default)]
    pub reference_ptir: Vec<u8>,
}

/// **The launch package** — what a driver registers a program AS.
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

/// One declared SSA value.
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

/// One op in a stage DAG.
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

/// One channel declaration.
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

/// One descriptor-port binding.
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

/// A `(channel, value)` pair.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPut {
    pub channel: u32,
    pub value: u32,
}

/// One stage program.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchStage {
    /// Prologue 0, OnAttnProj 1, OnAttn 2, Epilogue 3.
    pub kind: u8,
    pub ops: Vec<LaunchOp>,
    pub puts: Vec<LaunchPut>,
    pub takes: Vec<u32>,
    pub reads: Vec<u32>,
}

/// One region.
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

/// One normalized value type.
#[derive(Default, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchPlanValue {
    pub dtype: u8,
    /// Per-dimension extent kind: `PIE_EXTENT_STATIC` or a runtime extent.
    pub extents: Vec<u8>,
    /// Per-dimension literal extent, meaningful where `extents` is static.
    pub dims: Vec<u32>,
}

/// One lane-binding rule.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchChannelRule {
    pub value: u32,
    pub local: u32,
}

/// The per-program launch plan for one stage.
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
    pub src_domain: DeviceDomain,
    pub src_device_ordinal: u32,
    pub dst_domain: DeviceDomain,
    pub dst_device_ordinal: u32,
    pub src_page_ids: Vec<u32>,
    pub dst_page_ids: Vec<u32>,
    pub cells: Vec<KvMoveCell>,
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

impl KvCopyPlan {
    /// The two rules `validate_kv_copy_desc` stated that a `Vec` does not:
    /// both domains name a real one, and the page lists are parallel.
    ///
    /// The rest of that validator was `ptr/len mismatch` and an
    /// `abi_version`/`reserved0` check — a representation this plan does not
    /// have.
    ///
    /// # Errors
    ///
    /// [`Malformed`], naming the member and the numbers that disagree.
    pub fn validate(&self) -> Result<(), Malformed> {
        if !crate::local::pie_memory_domain_is_valid(self.src_domain) {
            return Err(Malformed(format!(
                "src_domain names no memory domain: {}",
                self.src_domain
            )));
        }
        if !crate::local::pie_memory_domain_is_valid(self.dst_domain) {
            return Err(Malformed(format!(
                "dst_domain names no memory domain: {}",
                self.dst_domain
            )));
        }
        if self.src_page_ids.len() != self.dst_page_ids.len() {
            return Err(Malformed(format!(
                "src_page_ids has {} entries and dst_page_ids {}",
                self.src_page_ids.len(),
                self.dst_page_ids.len()
            )));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct StateCopyPlan {
    pub slot_ranges: Vec<StateCopyRange>,
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

impl MediaEncodePlan {
    /// Every rule `validate_encode_desc` stated about a media payload.
    ///
    /// Unlike the transfer verbs, most of this validator was NOT about the C
    /// shape: it checks that the image and audio planes describe the same
    /// counts, that each byte payload is `f32`-aligned and exactly
    /// partitioned by its CSR, and that a plane with no anchors carries no
    /// payload either. All of it comes across.
    ///
    /// The two `output_*` members are the driver's out-params; only their
    /// SHAPE is checked here, because their contents are what the call
    /// produces.
    ///
    /// # Errors
    ///
    /// [`Malformed`], naming the member and the numbers that disagree.
    pub fn validate(&self) -> Result<(), Malformed> {
        const F32: usize = std::mem::size_of::<f32>();
        const U16: usize = std::mem::size_of::<u16>();
        let bad = |why: String| Err(Malformed(why));

        let images = self.image_anchor_rows.len();
        let clips = self.audio_anchor_rows.len();
        if images + clips == 0 {
            return bad("an encode carries no image and no audio anchor".into());
        }
        if self.output_row_indptr.len() != images + clips + 1 {
            return bad(format!(
                "output_row_indptr has {} entries for {images} images and {clips} clips",
                self.output_row_indptr.len()
            ));
        }
        if self.output_rows.is_empty() || !self.output_rows.len().is_multiple_of(U16) {
            return bad(format!(
                "output_rows is {} bytes, which is empty or not a whole number of u16",
                self.output_rows.len()
            ));
        }

        if images == 0 {
            if !self.image_grids.is_empty()
                || !self.image_pixels.is_empty()
                || !self.image_pixel_indptr.is_empty()
                || !self.image_patch_positions.is_empty()
            {
                return bad("an image payload arrived with no image anchor to attach it to".into());
            }
        } else {
            if self.image_grids.len() != images.saturating_mul(3) {
                return bad(format!(
                    "image_grids has {} entries for {images} images",
                    self.image_grids.len()
                ));
            }
            if self.image_pixel_indptr.len() != images + 1 {
                return bad(format!(
                    "image_pixel_indptr has {} entries for {images} images",
                    self.image_pixel_indptr.len()
                ));
            }
            if self.image_pixels.is_empty() || !self.image_pixels.len().is_multiple_of(F32) {
                return bad(format!(
                    "image_pixels is {} bytes, which is empty or not a whole number of f32",
                    self.image_pixels.len()
                ));
            }
            if self.image_patch_positions.is_empty()
                || !self.image_patch_positions.len().is_multiple_of(2)
            {
                return bad(format!(
                    "image_patch_positions has {} entries, which is empty or not a whole number of pairs",
                    self.image_patch_positions.len()
                ));
            }
            partition(
                &self.image_pixel_indptr,
                "image_pixel_indptr",
                self.image_pixels.len(),
                F32,
                false,
            )?;
        }

        if clips == 0 {
            if !self.audio_features.is_empty() || !self.audio_feature_indptr.is_empty() {
                return bad("an audio payload arrived with no audio anchor to attach it to".into());
            }
        } else {
            if self.audio_feature_indptr.len() != clips + 1 {
                return bad(format!(
                    "audio_feature_indptr has {} entries for {clips} clips",
                    self.audio_feature_indptr.len()
                ));
            }
            if self.audio_features.is_empty() || !self.audio_features.len().is_multiple_of(F32) {
                return bad(format!(
                    "audio_features is {} bytes, which is empty or not a whole number of f32",
                    self.audio_features.len()
                ));
            }
            // STRICT: an empty clip is not a clip, and the encoder would
            // produce no row for it.
            partition(
                &self.audio_feature_indptr,
                "audio_feature_indptr",
                self.audio_features.len(),
                F32,
                true,
            )?;
        }
        Ok(())
    }
}

/// A CSR that must EXACTLY partition `bytes`, on `align`-byte bounds.
///
/// `strict` makes every segment nonempty, which is what the audio plane
/// requires and the image plane does not.
fn partition(
    indptr: &[u32],
    name: &str,
    bytes: usize,
    align: usize,
    strict: bool,
) -> Result<(), Malformed> {
    if indptr.first().copied() != Some(0) {
        return Err(Malformed(format!(
            "{name} starts at {:?}, not 0",
            indptr.first()
        )));
    }
    if indptr.last().copied() != Some(bytes as u32) {
        return Err(Malformed(format!(
            "{name} ends at {:?}, not the {bytes} bytes it partitions",
            indptr.last()
        )));
    }
    for w in indptr.windows(2) {
        let ordered = if strict { w[0] < w[1] } else { w[0] <= w[1] };
        if !ordered {
            return Err(Malformed(format!(
                "{name} segment {}..{} is empty or inverted",
                w[0], w[1]
            )));
        }
        if w[0] as usize % align != 0 || w[1] as usize % align != 0 {
            return Err(Malformed(format!(
                "{name} segment {}..{} is not {align}-byte aligned",
                w[0], w[1]
            )));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct PoolResizePlan {
    pub pool_id: u64,
    pub target_pages: u64,
    pub map_ranges: Vec<PoolRange>,
    pub unmap_ranges: Vec<PoolRange>,
}

#[cfg(test)]
mod encode_tests {
    use super::*;

    /// One image, one anchor, one soft token — the shape the rest perturb.
    /// Ported from `local`'s `encode_layout_and_validation_are_stable`.
    fn sound() -> MediaEncodePlan {
        MediaEncodePlan {
            image_grids: vec![1, 1, 1],
            image_pixels: vec![0; 4],
            image_pixel_indptr: vec![0, 4],
            image_patch_positions: vec![0, 0],
            image_anchor_rows: vec![0],
            output_rows: vec![0; 2],
            output_row_indptr: vec![0; 2],
            ..MediaEncodePlan::default()
        }
    }

    fn why(plan: &MediaEncodePlan) -> String {
        plan.validate().expect_err("must be refused").to_string()
    }

    #[test]
    fn the_sound_plan_is_not_refused() {
        sound().validate().unwrap();
    }

    /// The case the original named: a pixel CSR whose last bound is not the
    /// byte count it claims to partition.
    #[test]
    fn a_pixel_csr_that_does_not_reach_its_bytes_is_refused() {
        let mut p = sound();
        p.image_pixel_indptr = vec![0, 3];
        assert!(why(&p).contains("image_pixel_indptr"), "{}", why(&p));
    }

    #[test]
    fn an_unaligned_pixel_bound_is_refused() {
        let mut p = sound();
        p.image_pixels = vec![0; 8];
        p.image_pixel_indptr = vec![0, 8];
        p.image_anchor_rows = vec![0];
        p.validate().expect("aligned is fine");
        p.image_anchor_rows = vec![0, 1];
        p.image_grids = vec![1, 1, 1, 1, 1, 1];
        p.image_pixel_indptr = vec![0, 2, 8];
        p.output_row_indptr = vec![0; 3];
        assert!(why(&p).contains("4-byte aligned"), "{}", why(&p));
    }

    #[test]
    fn a_payload_with_no_anchor_is_refused() {
        let mut p = sound();
        p.audio_features = vec![0; 4];
        assert!(why(&p).contains("no audio anchor"), "{}", why(&p));
    }

    #[test]
    fn an_encode_with_no_medium_at_all_is_refused() {
        let p = MediaEncodePlan::default();
        assert!(
            why(&p).contains("no image and no audio anchor"),
            "{}",
            why(&p)
        );
    }

    #[test]
    fn an_output_csr_that_does_not_cover_the_media_is_refused() {
        let mut p = sound();
        p.output_row_indptr = vec![0; 3];
        assert!(
            why(&p).contains("output_row_indptr has 3 entries"),
            "{}",
            why(&p)
        );
    }

    /// The audio plane's partition is STRICT: an empty clip is not a clip.
    #[test]
    fn an_empty_audio_segment_is_refused() {
        let mut p = MediaEncodePlan {
            audio_anchor_rows: vec![0, 1],
            audio_features: vec![0; 8],
            audio_feature_indptr: vec![0, 0, 8],
            output_rows: vec![0; 2],
            output_row_indptr: vec![0; 3],
            ..MediaEncodePlan::default()
        };
        assert!(why(&p).contains("empty or inverted"), "{}", why(&p));
        p.audio_feature_indptr = vec![0, 4, 8];
        p.validate().expect("two nonempty clips are fine");
    }
}

#[cfg(test)]
mod kv_copy_tests {
    use super::*;

    /// Ported from `local`'s `kv_copy_validator_rejects_invalid_memory_domain`.
    #[test]
    fn a_domain_that_names_nothing_is_refused() {
        let plan = KvCopyPlan {
            src_domain: 99,
            ..KvCopyPlan::default()
        };
        let why = plan.validate().expect_err("must be refused").to_string();
        assert!(why.contains("src_domain"), "{why}");
    }

    #[test]
    fn page_lists_that_are_not_parallel_are_refused() {
        let plan = KvCopyPlan {
            src_page_ids: vec![0, 1],
            dst_page_ids: vec![2],
            ..KvCopyPlan::default()
        };
        let why = plan.validate().expect_err("must be refused").to_string();
        assert!(
            why.contains("src_page_ids has 2 entries and dst_page_ids 1"),
            "{why}"
        );
    }

    #[test]
    fn a_parallel_device_to_device_move_is_accepted() {
        KvCopyPlan {
            src_domain: crate::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
            dst_domain: crate::local::PIE_MEMORY_DOMAIN_CUDA_DEVICE,
            src_page_ids: vec![0],
            dst_page_ids: vec![2],
            ..KvCopyPlan::default()
        }
        .validate()
        .unwrap();
    }
}

#[cfg(test)]
mod mask_tests {
    use super::*;

    /// The reference expansion: the loop this replaced, in the engine's
    /// `driver/abi.rs`, written against `grammar::bitmask`.
    fn reference(plan: &LaunchPlan) -> MaskWords {
        let requests = plan.qo_indptr.len().saturating_sub(1);
        let request_indptr = if plan.mask_indptr.is_empty() {
            vec![0; requests + 1]
        } else {
            plan.mask_indptr.clone()
        };
        let mut word_indptr = vec![0u32];
        let mut words: Vec<u32> = Vec::new();
        for mask in &plan.masks {
            let word_count = mask.len().div_ceil(32);
            let start = words.len();
            words.resize(start + word_count, 0);
            let mut run_start = 0usize;
            for (index, &run_len) in mask.runs.iter().enumerate() {
                let run_end = run_start.saturating_add(run_len as usize);
                if index % 2 == 1 {
                    for bit in run_start..run_end.min(mask.len()) {
                        words[start + bit / 32] |= 1 << (bit % 32);
                    }
                }
                run_start = run_end;
            }
            word_indptr.push(words.len() as u32);
        }
        MaskWords {
            request_indptr,
            word_indptr,
            words,
        }
    }

    fn plan_with(masks: Vec<EncodedMask>, mask_indptr: Vec<u32>, qo: Vec<u32>) -> LaunchPlan {
        LaunchPlan {
            qo_indptr: qo,
            masks,
            mask_indptr,
            ..LaunchPlan::default()
        }
    }

    #[test]
    fn expansion_agrees_with_the_loop_it_replaced() {
        let cases = vec![
            // Empty: no mask, no row.
            plan_with(Vec::new(), Vec::new(), vec![0, 2]),
            // One row that starts false, then true, then false.
            plan_with(
                vec![EncodedMask::new(vec![3, 5, 2], 10)],
                vec![0, 1],
                vec![0, 1],
            ),
            // A row beginning with TRUE — the documented zero-length false run.
            plan_with(
                vec![EncodedMask::new(vec![0, 7], 7)],
                vec![0, 1],
                vec![0, 1],
            ),
            // Crossing a word boundary, and a run that overruns `total_size`.
            plan_with(
                vec![EncodedMask::new(vec![30, 40, 100], 64)],
                vec![0, 1],
                vec![0, 1],
            ),
            // Two masks over two requests, exercising both CSRs.
            plan_with(
                vec![
                    EncodedMask::new(vec![1, 31], 32),
                    EncodedMask::new(vec![0, 1, 62, 1], 64),
                ],
                vec![0, 1, 2],
                vec![0, 1, 2],
            ),
            // Masks present but no `mask_indptr` — the all-zero default.
            plan_with(
                vec![EncodedMask::new(vec![2, 2], 4)],
                Vec::new(),
                vec![0, 1, 2, 3],
            ),
        ];
        for (n, plan) in cases.iter().enumerate() {
            assert_eq!(plan.bitmask_words(), reference(plan), "case {n} disagrees");
        }
    }

    #[test]
    fn a_row_that_begins_true_sets_its_first_bit() {
        let plan = plan_with(
            vec![EncodedMask::new(vec![0, 3], 3)],
            vec![0, 1],
            vec![0, 1],
        );
        assert_eq!(plan.bitmask_words().words, vec![0b111]);
    }

    /// Ported from `engine/src/driver/abi.rs`, which expanded these runs on
    /// the way into the C descriptor.
    #[test]
    fn two_rows_expand_to_the_words_the_c_view_used_to_pack() {
        let plan = plan_with(
            vec![
                EncodedMask::new(vec![0, 3, 1], 4),
                EncodedMask::new(vec![1, 2, 1], 4),
            ],
            vec![0, 1, 2],
            vec![0, 1, 2],
        );
        let got = plan.bitmask_words();
        assert_eq!(got.request_indptr, vec![0, 1, 2]);
        assert_eq!(got.word_indptr, vec![0, 1, 2]);
        assert_eq!(got.words, vec![0b0111, 0b0110]);
    }

    /// Likewise: an omitted mask table is empty rows, not a refusal.
    #[test]
    fn omitted_mask_expands_as_empty_rows() {
        let plan = plan_with(Vec::new(), Vec::new(), vec![0, 1, 2]);
        let got = plan.bitmask_words();
        assert_eq!(got.request_indptr, vec![0, 0, 0]);
        assert_eq!(got.word_indptr, vec![0]);
        assert!(got.words.is_empty());
    }

    #[test]
    fn a_run_past_the_row_end_writes_no_bit_past_it() {
        // 40 true elements declared over a 32-element row: the tail is not a
        // reason to touch the next mask's words.
        let plan = plan_with(
            vec![
                EncodedMask::new(vec![0, 40], 32),
                EncodedMask::new(vec![32], 32),
            ],
            vec![0, 1, 2],
            vec![0, 1, 2],
        );
        let got = plan.bitmask_words();
        assert_eq!(got.words, vec![u32::MAX, 0]);
        assert_eq!(got.word_indptr, vec![0, 1, 2]);
    }
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
