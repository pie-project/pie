//! pie:core/inference - ForwardPass + sampler programs; pie:core/tensor -
//! Tensor + Program resources.

use crate::api::adapter::Adapter;
use crate::api::pie;
use crate::inference::ForwardOutput;
use crate::inference::forward_prepare;
use crate::inference::structured::compiled_grammar::CompiledGrammar;
use crate::inference::structured::grammar::Grammar as InternalGrammar;
use crate::inference::structured::json_schema::{
    JsonSchemaOptions, builtin_json_grammar, json_schema_to_grammar,
};
use crate::inference::structured::matcher::GrammarMatcher;
use crate::inference::structured::regex::regex_to_grammar;
use crate::instance::InstanceState;
use crate::inference;
use anyhow::Result;
use pie_driver_abi::Brle;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use wasmtime::component::Resource;
use wasmtime::component::{Accessor, HasSelf};
use wasmtime_wasi::WasiView;

#[derive(Debug, Clone, serde::Serialize)]
pub struct ExecuteProfileSnapshot {
    pub calls: u64,
    pub hits: u64,
    pub misses: u64,
    pub total_us: u64,
    pub prepare_us: u64,
    pub hit_wait_us: u64,
    pub cold_prepare_us: u64,
    pub pin_us: u64,
    pub submit_wait_us: u64,
    pub postprocess_us: u64,
}

struct ExecuteProfileStats {
    calls: AtomicU64,
    hits: AtomicU64,
    misses: AtomicU64,
    total_us: AtomicU64,
    prepare_us: AtomicU64,
    hit_wait_us: AtomicU64,
    cold_prepare_us: AtomicU64,
    pin_us: AtomicU64,
    submit_wait_us: AtomicU64,
    postprocess_us: AtomicU64,
}

static EXECUTE_PROFILE: ExecuteProfileStats = ExecuteProfileStats {
    calls: AtomicU64::new(0),
    hits: AtomicU64::new(0),
    misses: AtomicU64::new(0),
    total_us: AtomicU64::new(0),
    prepare_us: AtomicU64::new(0),
    hit_wait_us: AtomicU64::new(0),
    cold_prepare_us: AtomicU64::new(0),
    pin_us: AtomicU64::new(0),
    submit_wait_us: AtomicU64::new(0),
    postprocess_us: AtomicU64::new(0),
};

fn execute_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("PIE_PROFILE_EXECUTE").is_some())
}

fn elapsed_us(duration: Duration) -> u64 {
    duration.as_micros() as u64
}

pub fn execute_profile_snapshot() -> Option<ExecuteProfileSnapshot> {
    if !execute_profile_enabled() {
        return None;
    }
    Some(ExecuteProfileSnapshot {
        calls: EXECUTE_PROFILE.calls.load(Ordering::Relaxed),
        hits: EXECUTE_PROFILE.hits.load(Ordering::Relaxed),
        misses: EXECUTE_PROFILE.misses.load(Ordering::Relaxed),
        total_us: EXECUTE_PROFILE.total_us.load(Ordering::Relaxed),
        prepare_us: EXECUTE_PROFILE.prepare_us.load(Ordering::Relaxed),
        hit_wait_us: EXECUTE_PROFILE.hit_wait_us.load(Ordering::Relaxed),
        cold_prepare_us: EXECUTE_PROFILE.cold_prepare_us.load(Ordering::Relaxed),
        pin_us: EXECUTE_PROFILE.pin_us.load(Ordering::Relaxed),
        submit_wait_us: EXECUTE_PROFILE.submit_wait_us.load(Ordering::Relaxed),
        postprocess_us: EXECUTE_PROFILE.postprocess_us.load(Ordering::Relaxed),
    })
}

#[derive(Default)]
struct ExecuteProfileSample {
    hit: bool,
    prepare_us: u64,
    hit_wait_us: u64,
    cold_prepare_us: u64,
    pin_us: u64,
    submit_wait_us: u64,
    postprocess_us: u64,
}

fn record_execute_profile(sample: ExecuteProfileSample, total_us: u64) {
    if !execute_profile_enabled() {
        return;
    }
    EXECUTE_PROFILE.calls.fetch_add(1, Ordering::Relaxed);
    if sample.hit {
        EXECUTE_PROFILE.hits.fetch_add(1, Ordering::Relaxed);
    } else {
        EXECUTE_PROFILE.misses.fetch_add(1, Ordering::Relaxed);
    }
    EXECUTE_PROFILE
        .total_us
        .fetch_add(total_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .prepare_us
        .fetch_add(sample.prepare_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .hit_wait_us
        .fetch_add(sample.hit_wait_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .cold_prepare_us
        .fetch_add(sample.cold_prepare_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .pin_us
        .fetch_add(sample.pin_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .submit_wait_us
        .fetch_add(sample.submit_wait_us, Ordering::Relaxed);
    EXECUTE_PROFILE
        .postprocess_us
        .fetch_add(sample.postprocess_us, Ordering::Relaxed);
}

/// WASM-facing forward-pass accumulator. The WIT methods append/set
/// into `req: pie_driver_abi::ForwardRequest` directly — at `execute()`
/// we just finalize the per-request indptrs and submit. `model_id`
/// is WASM-side routing info (not on the wire) and `adapter_seed`
/// is stored separately because it doesn't have its own WIT setter
/// but is used at execute-time to populate the adapter binding.
#[derive(Debug)]
/// Host backing for the WIT `tensor` resource (`pie:core/tensor.tensor`): a
/// typed, host-resident byte buffer. Under the MVP sync interface every tensor
/// binding is *submit* — the inferlet writes the value before `execute()` and
/// the host gathers `data` into the sampling carrier at attach time.
pub struct Tensor {
    shape: Vec<u32>,
    dtype: pie::core::tensor::Dtype,
    data: Vec<u8>,
}

/// Host backing for the WIT `program` resource (`pie:core/tensor.program`): a
/// compiled sampling program. Decoded once at construction (`tensor::Program →
/// SamplingProgram`, via [`program_decode::decode_program`]) and encoded to L0
/// bytecode for the bridge carrier. Keeps the declared output kinds (for
/// response marshaling) and the input arity (for attach-time binding
/// validation).
pub struct Program {
    /// The interned, shared `#9` cache artifact: canonical bytecode, its hash
    /// (== the driver `ProgramHandle`), per-output marshaling kinds, and input
    /// arity. Identical programs across requests/inferlets share one `Arc`.
    cached: Arc<crate::api::program_cache::CachedProgram>,
}

pub struct ForwardPass {
    /// Set when the ctx is bound via `context()` (the new WIT `forward-pass`
    /// constructor takes no model — the context carries the model identity).
    pub model_id: usize,
    /// Merged kv-working-set descriptor (#21 WIT `kv-working-set`): read context
    /// `[inp_start, inp_start+inp_len)` with `valid_tokens` prefix, write the
    /// CONTIGUOUS slot range `[output_start, output_start+output_len)` whose first
    /// written row sits at in-page token `offset`. Resolved to physical pages in
    /// the atomic arena transaction at `execute()`. Replaces the split
    /// kv-context/kv-output records (no generation/indices — the inferlet owns
    /// working-set correctness).
    kv_ws: Option<KvWorkingSetDesc>,
    /// Merged rs-working-set descriptor (#21 WIT `rs-working-set`): the buffered
    /// recurrent-state token range `[start_token, start_token+len_tokens)`.
    rs_ws: Option<RsWorkingSetDesc>,
    /// `rs-fold-buffered(n)` (W9 piggyback): fold the first `n` buffered RS tokens
    /// of this pass's RS working set into its folded state as part of this
    /// forward. Lowered to `rs_fold_lens` + `RS_FLAG_FOLD` over the buffered
    /// slabs (`rs_buffer_slot_ids`); the driver gathers + replays them.
    fold_buffered_tokens: Option<u32>,
    pub adapter_seed: Option<i64>,
    req: pie_driver_abi::ForwardRequest,
    /// `next-inputs(positions)` (#21 run-ahead carrier, delta's L-map): the NEXT
    /// pass's input slots the carrier fills with THIS pass's sampled token. The
    /// host assigns the link L (`next_input_map::apply_next_input_carrier`); the
    /// guest threads no link-ids.
    next_input_positions: Vec<u32>,
    /// `next-attention-mask(mask)` (#21 run-ahead): the attention-mask carrier
    /// for the next pass (parallel to `next-inputs`). Recorded here; the carrier
    /// wiring rides delta's next-input L-map. Empty on the greedy-decode path.
    next_attention_mask: Vec<Brle>,
    /// Sampling programs attached via `sampler(...)` / `samplers(...)`, each with
    /// its attach-time binding-map and gathered submit inputs. Empty for a pass
    /// that does no sampling (pure prefill); one entry for the single-sampler
    /// path; many for `samplers`. Flattened into the bridge carrier at `execute()`.
    programs: Vec<AttachedProgram>,
    /// #21 1c run-ahead: the in-flight forward state stored by the sync
    /// `execute()` (eager-submit) and consumed by the async `output()`/`outputs()`
    /// (await→finalize→tensor). `None` until `execute()`; the forward-pass IS the
    /// in-flight handle (Option A).
    pending: Option<PendingForward>,
    /// A prepare/submit error deferred from the SYNC `execute()` (the WIT
    /// `execute: func()` returns nothing, so a recoverable failure can't surface
    /// there). The async `output()`/`outputs()` reports it as the WIT `error`.
    exec_error: Option<String>,
    /// `fresh-generate()` (#26): this pass is the FIRST forward of a new
    /// `generate()` on its context. The run-ahead next-input carry
    /// (`pending_next_input`) lives per-instance, so a prior generate's terminal
    /// producer can leave a dangling carry; this flag tells `execute()` to drop
    /// (and free) any dangling carry for THIS context before the carrier's
    /// consumer-inject, so the new generate's prime never injects a stale token.
    /// The guest's `Generator` sets it once per generate.
    fresh_generate: bool,
}

/// Merged kv-working-set descriptor — see [`ForwardPass::kv_ws`].
struct KvWorkingSetDesc {
    set: Resource<crate::working_set::kv::KvWorkingSet>,
    inp_start: u32,
    /// Read-context page count `[inp_start, inp_start+inp_len)`. Retained for WIT
    /// descriptor completeness; the host read derives the pinned page count from
    /// `valid_tokens` (the valid attention prefix) — `inp_len`'s trailing
    /// reserved slots may be unwritten and must not be resolved. In the disjoint
    /// (Option-B) convention `inp_len == valid_tokens.div_ceil(page_size)`.
    #[allow(dead_code)]
    inp_len: u32,
    valid_tokens: u32,
    output_start: u32,
    output_len: u32,
    offset: u32,
}

/// Merged rs-working-set descriptor — see [`ForwardPass::rs_ws`].
struct RsWorkingSetDesc {
    set: Resource<crate::working_set::rs::RsWorkingSet>,
    start_token: u32,
    len_tokens: u32,
}

/// One sampling program staged on a [`ForwardPass`] before submit: its L0
/// bytecode, declared output kinds (for response marshaling), the attach-time
/// [`Binding`](pie_sampling_ir::Binding) per input slot (the binding-map ridden
/// to the carrier so the driver can wire each `Op::Input(i)`), and the
/// submit-bound tensor values gathered from the bound `tensor` resources.
#[derive(Debug)]
struct AttachedProgram {
    bytecode: Vec<u8>,
    output_kinds: Vec<pie_sampling_ir::OutputKind>,
    /// Per-output element count (`ValueType.shape.numel()`), declared order. `1`
    /// for scalar/single-`Token`, `k` for a `[k]`-Token / `[k]` vector. Drives the
    /// shape-aware marshal + the single-`[1]`-Token fast-path gate.
    output_elem_counts: Vec<u32>,
    bindings: Vec<pie_sampling_ir::Binding>,
    submit_inputs: Vec<pie_driver_abi::SamplingInput>,
    /// #27 cut #2 device-alias late inputs: `(late-key, DeviceLateInput)` for
    /// each `host{key, late-bound}` tensor the host uploaded directly to device
    /// (`upload_late_input`). At execute the keys ride `sampling_late_keys` (len-0
    /// staged sentinel) + the device ptr/flag ride `sampling_late_device_*`; the
    /// `DeviceLateInput` handles move to the `PendingForward` (kept resident until
    /// the fire finalizes, then freed on drop). Empty when no late device inputs.
    late_device_inputs: Vec<(u32, crate::api::tensor_io::DeviceLateInput)>,
    /// Sampling positions carried by this program's `logits` binding (the WIT
    /// `input-binding::logits(positions)`); flattened into the request's
    /// `sampling_indices` at `execute()`. Empty if the program reads no logits.
    logits_positions: Vec<u32>,
    /// #10: the program-identity hash (alpha's distinct-count key) — computed once
    /// at attach via `program_identity_hash(bytecode, bindings)`; threaded into the
    /// scheduler accumulation policy so identical grammars dedup to one compile.
    identity_hash: u64,
}

fn empty_forward_request() -> pie_driver_abi::ForwardRequest {
    pie_driver_abi::ForwardRequest {
        adapter_bindings: vec![pie_driver_abi::AdapterBinding {
            adapter_id: -1,
            seed: -1,
        }],
        output_spec_flags: vec![false],
        // Image side-channel CSR roots: the per-image pixel/mrope indptrs and
        // the per-request image_indptr all begin with a leading 0 so the
        // batch-merge in `inference::request` can offset and append cleanly.
        image_indptr: vec![0],
        image_pixel_indptr: vec![0],
        image_mrope_indptr: vec![0],
        // Audio side-channel CSR roots (leading 0, like the image roots).
        audio_feature_indptr: vec![0],
        audio_indptr: vec![0],
        ..Default::default()
    }
}

/// Number of bytes in one element of a `tensor` dtype.
fn dtype_byte_size(dtype: pie::core::tensor::Dtype) -> usize {
    use pie::core::tensor::Dtype;
    match dtype {
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::Bool => 1,
    }
}

/// One host `tensor` output: the program's declared output value (#21 —
/// `output()`/`outputs()` return raw tensors directly; the `slot-output` enum
/// dissolved). dtype/shape come from the program's declared `OutputKind` (the
/// host already knows them from the program).
struct OutputTensor {
    shape: Vec<u32>,
    dtype: pie::core::tensor::Dtype,
    data: Vec<u8>,
}

/// #37: true iff `bytecode` is one of the recognized de-hardwired STANDARD samplers
/// (the driver's #8 recognizer set) — its `program_hash` (== driver `ProgramHandle`,
/// bytecode-only, NOT the binding-XOR'd `#10 identity_hash`) is in
/// `standard_program_hashes(vocab)`. A recognized-STANDARD attached program writes
/// `pi.sampled` (eager-D2H-fillable → the fast-path pinned is correct); a CUSTOM
/// program marshals `per_req` (rich, the pinned never fills → #36 carrier class).
/// The hash set is memoized per vocab (process-stable; built once, not per-forward);
/// on a build error → empty set → nothing recognized → conservative rich (#36 behavior).
fn is_recognized_standard(bytecode: &[u8], vocab: u32) -> bool {
    use std::collections::{HashMap, HashSet};
    use std::sync::{LazyLock, Mutex};
    static STD_HASHES: LazyLock<Mutex<HashMap<u32, HashSet<u64>>>> =
        LazyLock::new(|| Mutex::new(HashMap::new()));
    let hash = pie_sampling_ir::program_hash(bytecode);
    let mut cache = STD_HASHES.lock().expect("std-hash cache poisoned");
    let set = cache.entry(vocab).or_insert_with(|| {
        sampling_edsl::standard_program_hashes(vocab)
            .map(|v| v.into_iter().map(|(h, _)| h).collect())
            .unwrap_or_default()
    });
    set.contains(&hash)
}

/// Reconstruct each attached program's declared outputs as host `tensor`s, in
/// attach order then per-program output-slot order. Token → `[1] i32`;
/// Scalar/Entropy → `[1] f32` (mirostat S/μ); Logits/Logprobs → `[k] f32`. The
/// bare-token fast paths are the common greedy/decode shape (one `i32` token).
fn build_output_tensors(
    output: ForwardOutput,
    programs_output_kinds: &[Vec<pie_sampling_ir::OutputKind>],
    programs_output_elem_counts: &[Vec<u32>],
) -> Vec<OutputTensor> {
    use pie::core::tensor::Dtype;
    match output {
        ForwardOutput::Token(token) => vec![OutputTensor {
            shape: vec![1],
            dtype: Dtype::I32,
            data: token.to_le_bytes().to_vec(),
        }],
        ForwardOutput::Tokens(tokens) => tokens
            .into_iter()
            .map(|t| OutputTensor {
                shape: vec![1],
                dtype: Dtype::I32,
                data: t.to_le_bytes().to_vec(),
            })
            .collect(),
        ForwardOutput::Response(resp) => {
            build_output_tensors_from_response(resp, programs_output_kinds, programs_output_elem_counts)
        }
    }
}

/// Per-program tensor reconstruction over a rich [`ForwardResponse`] (#21:
/// program outputs → host `tensor`s, in attach then output-slot order, pulled
/// from the response channels). Token → `[1] i32`; Scalar/Entropy → `[1] f32`;
/// Logits/Logprobs → `[k] f32`. Distribution (ids+probs pair) + Embedding are
/// #18/WS5 follow-ups (not on the greedy-decode path).
fn build_output_tensors_from_response(
    resp: pie_driver_abi::ForwardResponse,
    programs_output_kinds: &[Vec<pie_sampling_ir::OutputKind>],
    programs_output_elem_counts: &[Vec<u32>],
) -> Vec<OutputTensor> {
    use pie::core::tensor::Dtype;
    use pie_sampling_ir::OutputKind;

    let mut tok_iter = resp.tokens.into_iter();
    let mut logit_iter: Box<dyn Iterator<Item = Vec<u8>>> = if resp.logits_byte_indptr.len() >= 2 {
        let blobs: Vec<_> = (0..resp.logits_byte_indptr.len() - 1)
            .map(|b| {
                let lo = resp.logits_byte_indptr[b] as usize;
                let hi = resp.logits_byte_indptr[b + 1] as usize;
                resp.logits_bytes[lo..hi].to_vec()
            })
            .collect();
        Box::new(blobs.into_iter())
    } else {
        Box::new(std::iter::empty())
    };
    let mut lp_iter: Box<dyn Iterator<Item = Vec<f32>>> = if resp.logprobs_val_indptr.len() >= 2 {
        let lps: Vec<_> = (0..resp.logprobs_val_indptr.len() - 1)
            .map(|s| {
                let lo = resp.logprobs_val_indptr[s] as usize;
                let hi = resp.logprobs_val_indptr[s + 1] as usize;
                resp.logprobs_values[lo..hi].to_vec()
            })
            .collect();
        Box::new(lps.into_iter())
    } else {
        Box::new(std::iter::empty())
    };
    let mut ent_iter = resp.entropies.into_iter();

    // #32: a `[k]`-Token output (elem_count > 1) reads its tokens from the
    // per-(request,output) `program_tokens` CSR — this per-request `resp` carries
    // one segment per declared output slot (empty for non-`[k]` outputs); a single
    // Token (elem_count == 1) stays the dense `tokens` channel above.
    let program_tokens = resp.program_tokens;
    let program_tokens_indptr = resp.program_tokens_indptr;
    let mut prog_tok_seg = 0usize;

    let mut tensors: Vec<OutputTensor> = Vec::new();
    for (p, kinds) in programs_output_kinds.iter().enumerate() {
        let elem_counts = programs_output_elem_counts.get(p);
        for (o, kind) in kinds.iter().enumerate() {
            let elem_count = elem_counts.and_then(|e| e.get(o)).copied().unwrap_or(1);
            let t = match kind {
                // `[k]`-Token: this output slot's segment of `program_tokens`.
                OutputKind::Token if elem_count > 1 => {
                    let lo = program_tokens_indptr.get(prog_tok_seg).copied().unwrap_or(0) as usize;
                    let hi = program_tokens_indptr
                        .get(prog_tok_seg + 1)
                        .copied()
                        .unwrap_or(lo as u32) as usize;
                    let slice = program_tokens.get(lo..hi).unwrap_or(&[]);
                    let data: Vec<u8> = slice.iter().flat_map(|&t| t.to_le_bytes()).collect();
                    Some(OutputTensor {
                        shape: vec![slice.len() as u32],
                        dtype: Dtype::I32,
                        data,
                    })
                }
                OutputKind::Token => tok_iter.next().map(|tok| OutputTensor {
                    shape: vec![1],
                    dtype: Dtype::I32,
                    data: tok.to_le_bytes().to_vec(),
                }),
                // Entropy + Scalar (mirostat μ / S = −log p) share the per-slot
                // scalar `entropies` channel.
                OutputKind::Entropy | OutputKind::Scalar => {
                    ent_iter.next().map(|e| OutputTensor {
                        shape: vec![1],
                        dtype: Dtype::F32,
                        data: e.to_le_bytes().to_vec(),
                    })
                }
                OutputKind::Logits => logit_iter.next().map(|bytes| OutputTensor {
                    shape: vec![(bytes.len() / 4) as u32],
                    dtype: Dtype::F32,
                    data: bytes,
                }),
                OutputKind::Logprobs => lp_iter.next().map(|lps| OutputTensor {
                    shape: vec![lps.len() as u32],
                    dtype: Dtype::F32,
                    data: bytemuck::cast_slice(&lps).to_vec(),
                }),
                // Distribution (ids+probs pair) + Embedding: rich-measurement
                // tensors, #18/WS5 follow-up.
                OutputKind::Distribution | OutputKind::Embedding => None,
            };
            // Every declared output occupies one `program_tokens` slot (empty for
            // non-`[k]`-Token outputs), so advance the segment cursor per output.
            prog_tok_seg += 1;
            if let Some(t) = t {
                tensors.push(t);
            }
        }
    }
    tensors
}

impl pie::core::inference::Host for InstanceState {}

/// Aggregate interface-level `Host` for `pie:core/working-set`, required by
/// the generated `HostKvWorkingSet` (charlie) + `HostRsWorkingSet` (delta)
/// resource impls. echo owns this (central bindgen) since it spans both lanes.
impl pie::core::working_set::Host for InstanceState {}

impl pie::core::inference::HostForwardPass for InstanceState {
    async fn new(&mut self) -> Result<Resource<ForwardPass>> {
        // Initialize the accumulator with the per-request invariants:
        // single adapter binding (-1 sentinels = unbound). Single-model: the
        // bound model is index 0. P3 explicit working-set descriptors
        // (kv/rs-context, kv/rs-output) are bound by their setters; there is no
        // ambient context handle.
        let pass = ForwardPass {
            model_id: 0,
            kv_ws: None,
            rs_ws: None,
            fold_buffered_tokens: None,
            adapter_seed: None,
            req: empty_forward_request(),
            next_input_positions: Vec::new(),
            next_attention_mask: Vec::new(),
            programs: Vec::new(),
            pending: None,
            exec_error: None,
            fresh_generate: false,
        };
        Ok(self.ctx().table.push(pass)?)
    }

    /// Merged kv-working-set descriptor (#21 WIT `kv-working-set`): read context
    /// = prior FULL pages `[inp_start, inp_start+inp_len)` with `valid_tokens`
    /// prefix; write = the contiguous new-KV range `[output_start,
    /// output_start+output_len)`; `offset` = in-page token offset of the first
    /// written row (the first write page's partial-prior prefix → last-page
    /// valid_len). Read ⊎ write disjoint (Option B); resolved to physical pages
    /// in the txn prepare. The inferlet owns working-set correctness — no
    /// generation/indices guard.
    #[allow(clippy::too_many_arguments)]
    async fn kv_working_set(
        &mut self,
        this: Resource<ForwardPass>,
        set: Resource<crate::working_set::kv::KvWorkingSet>,
        inp_start: u32,
        inp_len: u32,
        valid_tokens: u32,
        output_start: u32,
        output_len: u32,
        offset: u32,
    ) -> Result<()> {
        let set = Resource::new_borrow(set.rep());
        self.ctx().table.get_mut(&this)?.kv_ws = Some(KvWorkingSetDesc {
            set,
            inp_start,
            inp_len,
            valid_tokens,
            output_start,
            output_len,
            offset,
        });
        Ok(())
    }

    // ── #21 run-ahead next-input carrier (delta's host L-map) ───────────
    /// `next-inputs(positions)`: the NEXT pass's input slots the carrier fills
    /// with THIS pass's sampled token. Host owns the link L (assigned at
    /// `execute()` via [`crate::api::next_input_map::apply_next_input_carrier`]);
    /// the guest threads no link-ids (replaces the 3 granular link setters).
    async fn next_inputs(
        &mut self,
        this: Resource<ForwardPass>,
        positions: Vec<u32>,
    ) -> Result<()> {
        self.ctx().table.get_mut(&this)?.next_input_positions = positions;
        Ok(())
    }

    /// `next-attention-mask(mask)`: the attention-mask carrier for the next pass
    /// (run-ahead mask carryover, parallel to `next-inputs`). Recorded; the
    /// carrier wiring rides delta's L-map. Inert on the greedy-decode path.
    async fn next_attention_mask(
        &mut self,
        this: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> Result<()> {
        let brle_masks: Vec<Brle> = mask.into_iter().map(Brle::from_vec).collect();
        self.ctx().table.get_mut(&this)?.next_attention_mask = brle_masks;
        Ok(())
    }

    /// #26 `fresh-generate()`: mark this pass as the first forward of a new
    /// `generate()` so `execute()` drops any dangling next-input carry left on
    /// this context by a prior generate's terminal producer (the stop-terminal /
    /// explicit-restart path; golf's loop omits the carry on the predictable
    /// max-boundary terminal). No-arg flag; the context is the pass's
    /// kv-working-set.
    async fn fresh_generate(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        self.ctx().table.get_mut(&this)?.fresh_generate = true;
        Ok(())
    }

    /// Merged rs-working-set descriptor (#21 WIT `rs-working-set`): the buffered
    /// recurrent-state token range `[start_token, start_token+len_tokens)`
    /// (read+write). Materialised + pinned in the txn prepare.
    async fn rs_working_set(
        &mut self,
        this: Resource<ForwardPass>,
        set: Resource<crate::working_set::rs::RsWorkingSet>,
        start_token: u32,
        len_tokens: u32,
    ) -> Result<()> {
        let set = Resource::new_borrow(set.rep());
        self.ctx().table.get_mut(&this)?.rs_ws = Some(RsWorkingSetDesc {
            set,
            start_token,
            len_tokens,
        });
        Ok(())
    }

    /// Fold the first `tokens` buffered RS tokens of this pass's RS working set
    /// into its folded recurrent state as part of this forward (W9 piggyback).
    /// Recorded here; `execute()` lowers it to `rs_fold_lens` + `RS_FLAG_FOLD`
    /// over the buffered slabs so the driver gathers + replays them in-forward.
    async fn rs_fold_buffered(&mut self, this: Resource<ForwardPass>, tokens: u32) -> Result<()> {
        self.ctx().table.get_mut(&this)?.fold_buffered_tokens = Some(tokens);
        Ok(())
    }

    async fn input_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        tokens: Vec<u32>,
        positions: Vec<u32>,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.token_ids.extend(tokens);
        pass.req.position_ids.extend(positions);
        Ok(())
    }

    /// Splice an encoded visual span into the pending request. Appends
    /// `token_count` placeholder rows to `token_ids`/`position_ids` (so the
    /// forward has KV slots for the soft tokens; the driver overwrites their
    /// embeddings with the vision-encoder output) and stages the pixel tensor +
    /// per-patch positions + the batch row offset on the image side-channel.
    /// See MULTIMODAL.md §2.5.
    async fn input_image(
        &mut self,
        this: Resource<ForwardPass>,
        image: Resource<crate::api::media::Image>,
        anchor: u32,
    ) -> Result<()> {
        let (token_count, grid, patch_grid, uses_mrope, pixels, positions) = {
            let img = self.ctx().table.get(&image)?;
            (
                img.span.token_count,
                img.span.grid,    // merged LLM grid (for M-RoPE positions)
                img.patch_grid,   // pre-merge patch grid (for the driver encoder)
                img.uses_mrope,
                img.pixels.clone(),
                img.positions.clone(),
            )
        };

        let pass = self.ctx().table.get_mut(&this)?;
        let req = &mut pass.req;

        // Row offset (within this request) where the soft-token rows begin.
        let anchor_row = req.token_ids.len() as u32;
        req.image_anchor_rows.push(anchor_row);

        // Placeholder rows: valid token id 0 (overwritten by the encoder
        // scatter), positions sequential from `anchor` (Gemma 1-D RoPE).
        for i in 0..token_count {
            req.token_ids.push(0);
            req.position_ids.push(anchor + i);
        }

        // Pixel tensor (f32 → little-endian bytes) + per-patch positions.
        req.image_pixels
            .extend_from_slice(bytemuck::cast_slice(&pixels));
        req.image_pixel_indptr.push(req.image_pixels.len() as u32);
        req.image_patch_positions.extend_from_slice(&positions);

        // Wire `image_grids` carries the PRE-MERGE patch grid: the driver's
        // vision encoder needs patch units (t*h*w == n_patch). The merged grid
        // is used below only for the M-RoPE position side-channel.
        req.image_grids
            .extend_from_slice(&[patch_grid.t, patch_grid.h, patch_grid.w]);
        req.image_anchor_positions.push(anchor);
        if uses_mrope {
            for p in crate::multimodal::qwen_mrope_positions(grid, anchor) {
                req.image_mrope_positions.extend_from_slice(&p);
            }
        }
        req.image_mrope_indptr
            .push(req.image_mrope_positions.len() as u32);

        Ok(())
    }

    /// Splice an encoded audio clip into the pending request. Direct analog of
    /// `input_image`: appends `token_count` placeholder rows (KV slots for the
    /// audio soft tokens; the driver overwrites their embeddings with the
    /// audio-encoder output) and stages the log-mel features + the batch row
    /// offset on the audio side-channel. See audio_frontend.md.
    async fn input_audio(
        &mut self,
        this: Resource<ForwardPass>,
        audio: Resource<crate::api::media::Audio>,
        anchor: u32,
    ) -> Result<()> {
        let (token_count, mel) = {
            let a = self.ctx().table.get(&audio)?;
            (a.token_count, a.mel.clone())
        };

        let pass = self.ctx().table.get_mut(&this)?;
        let req = &mut pass.req;

        // Row offset (within this request) where the soft-token rows begin.
        let anchor_row = req.token_ids.len() as u32;
        req.audio_anchor_rows.push(anchor_row);

        // Placeholder rows: valid token id 0 (overwritten by the encoder
        // scatter), positions sequential from `anchor` (Gemma 1-D RoPE).
        for i in 0..token_count {
            req.token_ids.push(0);
            req.position_ids.push(anchor + i);
        }

        // Log-mel features (f32 → little-endian bytes).
        req.audio_features
            .extend_from_slice(bytemuck::cast_slice(&mel));
        req.audio_feature_indptr.push(req.audio_features.len() as u32);

        Ok(())
    }

    async fn output_speculative_tokens(
        &mut self,
        this: Resource<ForwardPass>,
        flag: bool,
    ) -> Result<()> {
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.output_spec_flags = vec![flag];
        Ok(())
    }

    async fn attention_mask(
        &mut self,
        this: Resource<ForwardPass>,
        mask: Vec<Vec<u32>>,
    ) -> Result<()> {
        let brle_masks: Vec<Brle> = mask.into_iter().map(Brle::from_vec).collect();

        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.masks = brle_masks;
        Ok(())
    }

    // ── Programmable sampler (Sampling IR) — L1/bravo host carrier ──────
    //
    // The binding-free front door: the inferlet builds a `program` resource
    // (decoded + compiled at construction, see `HostProgram::new`) and attaches
    // it with one `input-binding` per program input slot. `sampler`/`batch-sampler`
    // walk the bindings positionally (`inputs[i]` ↔ `Op::Input(i)`): a `logits`
    // binding marks the LM-head intrinsic and carries the sampling positions; a
    // `tensor` binding is a submit-bound device value the host gathers now. The
    // per-slot binding-map + gathered submit values ride the bridge carrier at
    // `execute()`; the driver wires each `Op::Input(i)` from the binding-map.
    async fn sampler(
        &mut self,
        this: Resource<ForwardPass>,
        program: Resource<Program>,
        inputs: Vec<pie::core::inference::InputBinding>,
    ) -> Result<()> {
        let attached = self.attach_program(&program, inputs)?;
        let pass = self.ctx().table.get_mut(&this)?;
        pass.programs.push(attached);
        Ok(())
    }

    async fn samplers(
        &mut self,
        this: Resource<ForwardPass>,
        programs: Vec<Resource<Program>>,
        inputs: Vec<Vec<pie::core::inference::InputBinding>>,
    ) -> Result<()> {
        if programs.len() != inputs.len() {
            return Err(anyhow::anyhow!(
                "samplers: {} programs but {} binding lists",
                programs.len(),
                inputs.len()
            ));
        }
        let mut attached = Vec::with_capacity(programs.len());
        for (program, binds) in programs.into_iter().zip(inputs) {
            attached.push(self.attach_program(&program, binds)?);
        }
        let pass = self.ctx().table.get_mut(&this)?;
        pass.programs.extend(attached);
        Ok(())
    }

    async fn adapter(
        &mut self,
        this: Resource<ForwardPass>,
        adapter: Resource<Adapter>,
    ) -> Result<()> {
        let adapter_id = self.ctx().table.get(&adapter)?.adapter_id;
        let pass = self.ctx().table.get_mut(&this)?;
        pass.req.adapter_bindings[0].adapter_id = adapter_id as i64;
        Ok(())
    }

    async fn drop(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        // P3 `execute()` is inline (prepare→submit→await→finalize in one async
        // fn), so a dropped pass holds no pending pin to release — just free the
        // resource-table entry.
        self.ctx().table.delete(this)?;
        Ok(())
    }

    /// #21: SYNC eager-submit (`execute: func()` — no return). Prepares + submits
    /// the forward and stores the in-flight [`PendingForward`] on the forward-pass
    /// (Option A — the pass IS the in-flight handle); a recoverable prepare/submit
    /// failure is deferred to `output()`/`outputs()` (stored as `exec_error`),
    /// since `execute` has no `error` channel. Delegates to the free
    /// [`execute_impl`] (the sync `HostForwardPass` trait gives `&mut self`, so no
    /// `accessor` is threaded).
    async fn execute(&mut self, this: Resource<ForwardPass>) -> Result<()> {
        execute_impl(self, this).await
    }
}

impl InstanceState {
    /// Walk one `sampler(...)`/`batch-sampler(...)` attachment: validate the
    /// binding arity against the program's input slots, build the per-slot
    /// [`Binding`](pie_sampling_ir::Binding) map, and gather each submit-bound
    /// `tensor` value into the carrier (keyed by its slot index so the driver
    /// can wire `Op::Input(i)`). The `logits` binding's positions are captured
    /// as the program's sampling positions.
    fn attach_program(
        &mut self,
        program: &Resource<Program>,
        inputs: Vec<pie::core::inference::InputBinding>,
    ) -> Result<AttachedProgram> {
        use pie::core::inference::InputBinding;
        let (bytecode, output_kinds, output_elem_counts, num_inputs, input_readiness) = {
            let p = self.ctx().table.get(program)?;
            (
                p.cached.bytecode.clone(),
                p.cached.output_kinds.clone(),
                p.cached.output_elem_counts.clone(),
                p.cached.num_inputs,
                p.cached.input_readiness.clone(),
            )
        };
        if inputs.len() != num_inputs {
            return Err(anyhow::anyhow!(
                "sampler: program declares {num_inputs} input slot(s) but {} binding(s) supplied",
                inputs.len()
            ));
        }
        let mut bindings = Vec::with_capacity(inputs.len());
        let mut submit_inputs = Vec::new();
        let mut late_device_inputs: Vec<(u32, crate::api::tensor_io::DeviceLateInput)> = Vec::new();
        let mut logits_positions: Vec<u32> = Vec::new();
        let mut saw_logits = false;
        for (i, binding) in inputs.into_iter().enumerate() {
            match binding {
                InputBinding::Logits(positions) => {
                    if saw_logits {
                        return Err(anyhow::anyhow!(
                            "sampler: multiple logits bindings are not supported (MVP)"
                        ));
                    }
                    saw_logits = true;
                    logits_positions = positions;
                    bindings.push(pie_sampling_ir::Binding::Logits);
                }
                InputBinding::MtpLogits => {
                    // #21 mtp-logits (de-hardwired speculation): the speculator's
                    // DRAFT logits intrinsic — source-selects the draft rows of
                    // `ws.logits` (M=1 ⇒ row 0) rather than a separate buffer,
                    // resolved driver-side by echo's `IntrinsicKind`. A distinct
                    // MANIFEST binding (`Binding::MtpLogits`, not a flag on
                    // `Logits`), carried to the driver via `SamplingBinding::
                    // MtpLogits` (kind 2). Payload-less: the draft-row offset is
                    // implicit (M=1), so no host-supplied positions.
                    bindings.push(pie_sampling_ir::Binding::MtpLogits);
                }
                InputBinding::Tensor(tensor) => {
                    // The slot index is the TensorKey wired to `Op::Input(i)`. The
                    // readiness comes from the program's `InputDecl` (alpha's #27
                    // `InputDecl.ready`): `Submit` gathers the value now into
                    // `sampling_input_*`; `Late` uploads it DIRECTLY to a device
                    // buffer (`upload_late_input` → `pie_tensor_write_async`, no IPC
                    // staging) and rides the `sampling_late_device_*` carrier, for
                    // the in-program mask-apply (#27 cut #2).
                    let key = i as u32;
                    let data = self.ctx().table.get(&tensor)?.data.clone();
                    // The owned tensor handle is consumed by the binding.
                    self.ctx().table.delete(tensor)?;
                    let ready = input_readiness
                        .get(i)
                        .copied()
                        .unwrap_or(pie_sampling_ir::Readiness::Submit);
                    match ready {
                        pie_sampling_ir::Readiness::Late => {
                            match crate::api::tensor_io::upload_late_input(&data) {
                                Some(device) => {
                                    late_device_inputs.push((key, device));
                                    bindings.push(pie_sampling_ir::Binding::Tensor {
                                        key,
                                        ready: pie_sampling_ir::Readiness::Late,
                                    });
                                }
                                None => {
                                    // Host-only build / device-alloc unavailable:
                                    // fall back to the submit-staged path (the Late
                                    // device channel needs `driver-cuda`; never
                                    // exercised without a GPU fire).
                                    bindings.push(pie_sampling_ir::Binding::Tensor {
                                        key,
                                        ready: pie_sampling_ir::Readiness::Submit,
                                    });
                                    submit_inputs
                                        .push(pie_driver_abi::SamplingInput { key, bytes: data });
                                }
                            }
                        }
                        pie_sampling_ir::Readiness::Submit => {
                            bindings.push(pie_sampling_ir::Binding::Tensor {
                                key,
                                ready: pie_sampling_ir::Readiness::Submit,
                            });
                            submit_inputs
                                .push(pie_driver_abi::SamplingInput { key, bytes: data });
                        }
                    }
                }
            }
        }
        // #10: program-identity hash (alpha's distinct-count key) — computed ONCE
        // here at attach, where bytecode + bindings are structured, before any
        // carrier encoding. Intrinsic binds (Logits/MtpLogits) dedup to one.
        let identity_hash = pie_sampling_ir::program_identity_hash(&bytecode, &bindings);
        Ok(AttachedProgram {
            bytecode,
            output_kinds,
            output_elem_counts,
            bindings,
            submit_inputs,
            late_device_inputs,
            logits_positions,
            identity_hash,
        })
    }
}

// =============================================================================
// tensor resources (pie:core/tensor) — Tensor (typed buffer) + Program
// =============================================================================

/// Expected packed byte length of a tensor with the given shape + dtype.
fn expected_byte_len(shape: &[u32], dtype: pie::core::tensor::Dtype) -> usize {
    let elems: usize = shape.iter().map(|&d| d as usize).product();
    elems * dtype_byte_size(dtype)
}

impl pie::core::tensor::Host for InstanceState {}

impl pie::core::tensor::HostTensor for InstanceState {
    async fn new(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::core::tensor::Dtype,
    ) -> Result<Resource<Tensor>> {
        // Allocate a zero-filled buffer sized to shape×dtype; `write` /
        // `from-data` populate it before `execute()`.
        let data = vec![0u8; expected_byte_len(&shape, dtype)];
        Ok(self.ctx().table.push(Tensor { shape, dtype, data })?)
    }

    async fn from_data(
        &mut self,
        shape: Vec<u32>,
        dtype: pie::core::tensor::Dtype,
        data: Vec<u8>,
    ) -> Result<Result<Resource<Tensor>, String>> {
        let expected = expected_byte_len(&shape, dtype);
        if data.len() != expected {
            return Ok(Err(format!(
                "tensor data is {} bytes but shape {shape:?}/{dtype:?} needs {expected}",
                data.len()
            )));
        }
        Ok(Ok(self.ctx().table.push(Tensor { shape, dtype, data })?))
    }

    async fn shape(&mut self, this: Resource<Tensor>) -> Result<Vec<u32>> {
        Ok(self.ctx().table.get(&this)?.shape.clone())
    }

    async fn dtype(&mut self, this: Resource<Tensor>) -> Result<pie::core::tensor::Dtype> {
        Ok(self.ctx().table.get(&this)?.dtype)
    }

    async fn write(
        &mut self,
        this: Resource<Tensor>,
        data: Vec<u8>,
    ) -> Result<Result<(), String>> {
        let t = self.ctx().table.get_mut(&this)?;
        let expected = expected_byte_len(&t.shape, t.dtype);
        if data.len() != expected {
            return Ok(Err(format!(
                "tensor write is {} bytes but shape needs {expected}",
                data.len()
            )));
        }
        t.data = data;
        Ok(Ok(()))
    }

    async fn read(&mut self, this: Resource<Tensor>) -> Result<Result<Vec<u8>, String>> {
        Ok(Ok(self.ctx().table.get(&this)?.data.clone()))
    }

    async fn drop(&mut self, this: Resource<Tensor>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

impl pie::core::tensor::HostProgram for InstanceState {
    async fn new(
        &mut self,
        inputs: Vec<pie::core::tensor::Input>,
        ops: Vec<pie::core::tensor::Op>,
        outputs: Vec<pie::core::tensor::Output>,
    ) -> Result<Resource<Program>> {
        // Decode + validate at construction so a malformed program is rejected
        // here (the constructor traps) rather than at attach/submit. Untrusted
        // inferlet input — `decode_program` is fully fallible (never panics).
        let program = crate::api::program_decode::decode_program(inputs, ops, outputs)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let bytecode = pie_sampling_ir::encode(&program);
        // #9: intern by the canonical bytecode hash (== the driver `ProgramHandle`).
        // Identical programs across requests/inferlets share one cached artifact —
        // `output_kinds` is derived once (skipped on a cache hit) — and the cache
        // is the #10 group / #8 hash-match fast-path registry. Host-side, no GPU.
        let cached = crate::api::program_cache::intern(&program, bytecode)
            .map_err(|e| anyhow::anyhow!("sampling program output kinds: {e:?}"))?;
        Ok(self.ctx().table.push(Program { cached })?)
    }

    async fn drop(&mut self, this: Resource<Program>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

/// Owns the forward's arena transaction (the kv/rs working-set page pins from
/// `resolve_read`/`cow_write_slot`) until [`await_and_finalize`] consumes it via
/// [`take`](Self::take) for the normal commit/abort. If the [`PendingForward`] is
/// instead dropped UN-consumed — an `output()`/`outputs()` error-return, or a
/// proc-terminate before finalize — the guard's `Drop` ABORTS the txn, releasing
/// the pins, rather than letting a bare `ArenaTxn` drop uncommitted and leak them
/// (the merged/concurrent-path abort symptom: `ArenaTxn dropped without
/// commit/abort — pins=1`). `finalize_forward_txn` (the normal path) ALWAYS
/// commits/aborts, so this only fires when finalize is never reached. A
/// down-payment on #23 abort-isolation: an aborted/terminated forward must
/// release its pins.
struct ForwardTxnGuard {
    txn: Option<crate::arena::ArenaTxn>,
    model_id: usize,
    driver_idx: usize,
}

impl ForwardTxnGuard {
    fn new(txn: crate::arena::ArenaTxn, model_id: usize, driver_idx: usize) -> Self {
        Self {
            txn: Some(txn),
            model_id,
            driver_idx,
        }
    }

    /// Hand the txn to the normal `finalize_forward_txn` commit/abort path; the
    /// guard is left empty so its `Drop` is a no-op.
    fn take(&mut self) -> Option<crate::arena::ArenaTxn> {
        self.txn.take()
    }
}

impl Drop for ForwardTxnGuard {
    fn drop(&mut self) {
        if let Some(txn) = self.txn.take() {
            // Un-finalized drop (error-return / proc-terminate before finalize):
            // abort the txn to release the kv/rs working-set pins. The normal
            // path already `take()`-d it, so this never double-handles (a
            // committed/aborted txn is consumed by value). `crate::arena::get` is
            // self-contained (no store/accessor needed), so it is safe in `Drop`;
            // the lock is only taken here when finalize did NOT run, so it cannot
            // deadlock against `finalize_forward_txn`.
            let arena_arc = crate::arena::get(self.model_id, self.driver_idx);
            if let Ok(mut arena) = arena_arc.lock() {
                arena.txn_abort(txn);
            }
        }
    }
}

/// In-flight forward state carried from the eager submit across the async
/// driver round-trip to the await→finalize. Phase-2
/// ([`finalize_forward_output`]) consumes it. The owned `txn` keeps the pins /
/// CoW copies alive until commit/abort; `rx` is the driver completion handle.
///
/// Today `execute` builds this then finalizes inline (single WIT method). The
/// 1c run-ahead surface (Option A — the forward-pass IS the in-flight handle)
/// stores it on the forward-pass so the async `output()`/`outputs()` awaits it;
/// the eager `execute` releases its `&mut ctx` borrow at that point so the loop
/// can hold two passes in flight (the `submit_async`→scheduler boundary).
struct PendingForward {
    rx: tokio::sync::oneshot::Receiver<Result<ForwardOutput>>,
    txn: ForwardTxnGuard,
    kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>>,
    seal_hashes: Vec<(u32, u64)>,
    model_id: usize,
    driver_idx: usize,
    rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>>,
    fold_buffered_tokens: Option<u32>,
    programs_output_kinds: Vec<Vec<pie_sampling_ir::OutputKind>>,
    /// #32 per-output element counts (declared order, parallel to
    /// `programs_output_kinds`); a `[k]`-Token (`elem_count > 1`) reads its tokens
    /// from the `program_tokens` CSR instead of the dense `tokens` channel.
    programs_output_elem_counts: Vec<Vec<u32>>,
    /// #27 cut #1 fast-path: pinned host buffers the driver eager-D2H's each
    /// declared output VALUE into (empty ⇒ legacy `ForwardResponse` marshal).
    /// Filled by the time `output()` reads them — the driver defers the
    /// forward-done response until the D2H lands (the `(a2)` seam).
    pinned_outputs: Vec<crate::api::tensor_io::PinnedOutput>,
    /// #27 cut #2: device-alias late-input upload handles, kept resident until the
    /// fire finalizes (the mask-apply kernel reads the device buffer during the
    /// fire). Freed on drop here, after `await_and_finalize`. Empty when no late
    /// device inputs.
    late_device_inputs: Vec<crate::api::tensor_io::DeviceLateInput>,
    profile_start: Option<Instant>,
    /// #23 overlap abort-isolation: the producer link this pass produced and the
    /// prior producer link it consumed (injected from). At finalize the write-log
    /// uses these to cascade-abort the consumer if its producer aborted, and to
    /// publish this pass's own outcome for its consumer.
    next_input_deps: crate::api::next_input_map::NextInputDeps,
}

/// #23 verify (TEST-ONLY, env-gated): force a designated *producer* pass to report
/// failure. Evaluated at the finalize success-determination — AFTER `rx.await`
/// resolved `Some` (the producer's forward device-succeeded and **retained** its
/// sampled token, and in the run-ahead overlap the consumer's inject is already
/// enqueued from that valid retained copy) — so flipping it to failure reproduces
/// the **retain-FOUND-then-host-abort** path: the producer's drain-gated
/// deferred-free races the in-flight inject (compute-sanitizer "free
/// strictly-after-drain"), and the consumer cascade-aborts fail-closed
/// (token-for-token). One mid-chain knob exercises both #23 teeth.
///
/// Keyed on the producer's monotonic link via `PIE_TEST_ABORT_PRODUCER_LINK`
/// (read once). **UNSET ⇒ always `false` ⇒ ZERO production behavior** — the #19
/// `PIE_MIROSTAT_DUMP` env-instrument pattern (test-only; flagged for the land
/// guard). Non-producer passes (no `produced` link) are never targeted.
fn test_force_producer_abort(deps: &crate::api::next_input_map::NextInputDeps) -> bool {
    static ABORT_LINK: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    let target = *ABORT_LINK.get_or_init(|| {
        std::env::var("PIE_TEST_ABORT_PRODUCER_LINK")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
    });
    abort_target_matches(deps.produced, target)
}

/// Pure targeting predicate for [`test_force_producer_abort`] (env-free, so it is
/// unit-testable): abort iff a target link is configured AND this pass is the
/// producer for it. An unset target (`None`) never matches ⇒ zero production
/// behavior; a non-producer pass (`produced = None`) is never targeted.
fn abort_target_matches(produced: Option<u32>, target: Option<u32>) -> bool {
    target.is_some() && produced == target
}

/// Phase-2 of a forward pass: await the driver round-trip, finalize the forward
/// txn (commit/abort + seal/fold via [`InstanceState::finalize_forward_txn`],
/// store reachable through `accessor.with`), and reconstruct the program's
/// declared output tensors. Consumes the [`PendingForward`] stored on the
/// forward-pass by the eager `execute`. The async `output()`/`outputs()` surface
/// calls this after taking the stored `PendingForward`.
async fn await_and_finalize(
    accessor: &Accessor<InstanceState, HasSelf<InstanceState>>,
    pending: PendingForward,
) -> Result<Result<Vec<OutputTensor>, String>> {
    let PendingForward {
        rx,
        mut txn,
        kv_set,
        seal_hashes,
        model_id,
        driver_idx,
        rs_fold_set,
        fold_buffered_tokens,
        programs_output_kinds,
        programs_output_elem_counts,
        pinned_outputs,
        late_device_inputs,
        profile_start,
        next_input_deps,
    } = pending;

    // Await the driver result (P3 native async), then finalize the forward txn
    // (store reachable via `accessor.with`).
    let forward_result: Option<ForwardOutput> = match rx.await {
        Ok(Ok(resp)) => Some(resp),
        Ok(Err(e)) => {
            tracing::warn!("future output failed: {e:#}");
            None
        }
        Err(_) => None,
    };
    // #23 verify (A-scoped, env-gated): force a designated producer's forward to
    // report failure AFTER it device-succeeded + retained, reproducing the
    // retain-FOUND-then-abort UAF race for the compute-sanitizer harness. UNSET ⇒
    // no-op (zero production behavior); see `test_force_producer_abort`.
    let success = forward_result.is_some() && !test_force_producer_abort(&next_input_deps);

    // Take the txn out of its guard for the normal commit/abort. The now-empty
    // guard's `Drop` is a no-op; the leak-abort only fires if this finalize is
    // never reached (error-return / proc-terminate).
    let forward_txn = txn
        .take()
        .expect("forward txn consumed exactly once at finalize");
    // #23: finalize resolves the overlap cascade and returns the EFFECTIVE success
    // — a consumer whose producer aborted (or is unresolved, fail-closed) is forced
    // to abort even on driver success, so its poisoned output never surfaces.
    let effective_success = accessor.with(|mut access| {
        access.get().finalize_forward_txn(
            success,
            forward_txn,
            kv_set,
            seal_hashes,
            model_id,
            driver_idx,
            rs_fold_set,
            fold_buffered_tokens,
            next_input_deps,
        )
    })?;

    // #27 cut #2: the forward has completed (the device-resident late inputs, e.g.
    // the packed mask, were consumed by the mask-apply during the fire). Free the
    // device buffers now (drop runs `pie_device_free`).
    drop(late_device_inputs);

    // Reconstruct the declared output tensors. Fast-path (#27 cut #1): the driver
    // eager-D2H'd each output VALUE into its pinned buffer; `rx` resolved only
    // after the D2H landed (the `(a2)` deferred forward-done), so the buffers are
    // filled — copy them out (the `ForwardResponse` output channels are
    // success-only / empty for a fast-path pass). Legacy: marshal the response
    // channels. Both paths are gated on the EFFECTIVE success (#23): a cascade-
    // aborted consumer (driver-ok but its producer aborted) yields NO output, so
    // `output()` surfaces the abort instead of a poisoned tensor.
    let tensors = if effective_success && !pinned_outputs.is_empty() {
        pinned_outputs
            .into_iter()
            .map(|out| {
                let data = crate::api::tensor_io::read(&out);
                OutputTensor {
                    shape: out.shape.clone(),
                    dtype: out.dtype,
                    data,
                }
            })
            .collect()
    } else if effective_success {
        match forward_result {
            Some(resp) => build_output_tensors(resp, &programs_output_kinds, &programs_output_elem_counts),
            None => Vec::new(),
        }
    } else {
        // Driver failure OR overlap cascade-abort ⇒ no output tensor.
        Vec::new()
    };
    if let Some(start) = profile_start {
        record_execute_profile(ExecuteProfileSample::default(), elapsed_us(start.elapsed()));
    }
    Ok(Ok(tensors))
}

impl pie::core::inference::HostForwardPassWithStore<InstanceState> for HasSelf<InstanceState> {
    /// #21 `output: async func() -> result<tensor, error>`. Takes the
    /// [`PendingForward`] stored on the forward-pass by the SYNC `execute`,
    /// awaits + finalizes it ([`await_and_finalize`]), and returns the program's
    /// FIRST declared output as a host `tensor`. Because the commit/abort runs in
    /// THIS async fn (store reachable via `accessor.with`), the forward txn is
    /// finalized atomically with the read — the lost-KV-commit failure mode of a
    /// pollable/get split is structurally impossible.
    async fn output(
        accessor: &Accessor<InstanceState, Self>,
        this: Resource<ForwardPass>,
    ) -> Result<Result<Resource<Tensor>, String>> {
        let (pending, exec_error) = accessor.with(|mut access| -> Result<_> {
            let pass = access.get().ctx().table.get_mut(&this)?;
            Ok((pass.pending.take(), pass.exec_error.take()))
        })?;
        if let Some(e) = exec_error {
            return Ok(Err(e));
        }
        let Some(pending) = pending else {
            return Ok(Err("output() called before execute() (or output already taken)".into()));
        };
        let tensors = match await_and_finalize(accessor, pending).await? {
            Ok(t) => t,
            Err(e) => return Ok(Err(e)),
        };
        let Some(OutputTensor { shape, dtype, data }) = tensors.into_iter().next() else {
            return Ok(Err("forward produced no output tensor".into()));
        };
        let handle = accessor.with(|mut access| -> Result<_> {
            Ok(access.get().ctx().table.push(Tensor { shape, dtype, data })?)
        })?;
        Ok(Ok(handle))
    }

    /// #21 `outputs: async func() -> result<list<tensor>, error>`. As [`output`]
    /// but returns ALL of the attached programs' declared outputs (attach order
    /// then per-program output-slot order), one host `tensor` each.
    async fn outputs(
        accessor: &Accessor<InstanceState, Self>,
        this: Resource<ForwardPass>,
    ) -> Result<Result<Vec<Resource<Tensor>>, String>> {
        let (pending, exec_error) = accessor.with(|mut access| -> Result<_> {
            let pass = access.get().ctx().table.get_mut(&this)?;
            Ok((pass.pending.take(), pass.exec_error.take()))
        })?;
        if let Some(e) = exec_error {
            return Ok(Err(e));
        }
        let Some(pending) = pending else {
            return Ok(Err("outputs() called before execute() (or output already taken)".into()));
        };
        let tensors = match await_and_finalize(accessor, pending).await? {
            Ok(t) => t,
            Err(e) => return Ok(Err(e)),
        };
        let mut handles = Vec::with_capacity(tensors.len());
        for OutputTensor { shape, dtype, data } in tensors {
            let handle = accessor.with(|mut access| -> Result<_> {
                Ok(access.get().ctx().table.push(Tensor { shape, dtype, data })?)
            })?;
            handles.push(handle);
        }
        Ok(Ok(handles))
    }
}

/// #21 eager-submit (free fn so the SYNC `HostForwardPass::execute` can call it
/// with `&mut InstanceState` — the sync trait has no `accessor`). Prepares +
/// submits the forward and stores the in-flight [`PendingForward`] on the
/// forward-pass; the async `output()`/`outputs()` await + finalize it. A
/// recoverable prepare/submit failure is stored as the pass's `exec_error`
/// (surfaced by `output()`), NOT returned here — `execute: func()` has no error
/// channel. The outer `Result` is reserved for unrecoverable host traps.
async fn execute_impl(
    state: &mut InstanceState,
    this: Resource<ForwardPass>,
) -> Result<()> {
        let profile_start = execute_profile_enabled().then(Instant::now);
        // Drain the accumulator: the explicit memory descriptors + the staged
        // ForwardRequest. There is no ambient context handle (W5). Every store /
        // resource-table touch in this async func goes through `accessor.with`
        // (P3: the host async fn has no `&mut self`).
        let (
            model_id,
            adapter_seed,
            kv_ws,
            rs_ws,
            fold_buffered_tokens,
            mut req,
            attached_programs,
            next_input_positions,
            fresh_generate,
        ) = {
            let pass = state.ctx().table.get_mut(&this)?;
            (
                pass.model_id,
                pass.adapter_seed,
                pass.kv_ws.take(),
                pass.rs_ws.take(),
                pass.fold_buffered_tokens.take(),
                std::mem::replace(&mut pass.req, empty_forward_request()),
                std::mem::take(&mut pass.programs),
                std::mem::take(&mut pass.next_input_positions),
                pass.fresh_generate,
            )
        };
        // v1: single-driver. Multi-driver binds the working set's device on
        // first materialization (`bind_driver`), wired at consolidation.
        let driver_idx = 0usize;

        // Flatten every attached sampling program into the bridge carrier and
        // collect their declared output kinds (attach order) for the slot-output
        // marshaling. Each program's `logits` binding positions become the
        // pass's sampling positions; its submit-bound tensor values + per-slot
        // binding-map ride the carrier so the driver wires each `Op::Input(i)`.
        let has_programs = !attached_programs.is_empty();
        // #37: a request takes the rich path iff it carries an attached CUSTOM
        // program — one that marshals to `per_req` (so the eager-D2H pinned dst is
        // never filled: the #19/#36 carrier class). A recognized de-hardwired
        // STANDARD sampler writes `pi.sampled` → the pinned fast-path IS correct, so
        // it stays eligible. Custom iff ANY attached program isn't recognized-standard.
        let has_custom_program = has_programs && {
            let vocab = crate::model::model().vocab_size();
            attached_programs
                .iter()
                .any(|p| !is_recognized_standard(&p.bytecode, vocab))
        };
        let mut programs_output_kinds: Vec<Vec<pie_sampling_ir::OutputKind>> =
            Vec::with_capacity(attached_programs.len());
        let mut programs_output_elem_counts: Vec<Vec<u32>> =
            Vec::with_capacity(attached_programs.len());
        // #10: per-program identity hashes (distinct-count key), attach order →
        // threaded into the scheduler accumulation policy via submit_async. Empty
        // for plain decode (no attached programs) ⇒ policy's free-to-batch path.
        let mut program_identity_hashes: Vec<u64> = Vec::with_capacity(attached_programs.len());
        let mut logits_positions: Vec<u32> = Vec::new();
        // #27 cut #2: the late device-alias upload handles, kept resident on the
        // PendingForward until the fire finalizes (then freed on drop).
        let mut late_device_handles: Vec<crate::api::tensor_io::DeviceLateInput> = Vec::new();
        for program in attached_programs {
            // #11 prefetch-seam: fire-and-forget warm of the JIT compile-cache at
            // admission — best-effort + non-blocking (no-op until a JIT sampling
            // backend registers; correctness never depends on it landing). The NVRTC
            // compile overlaps the in-flight run-ahead steps so it's off TTFT, and the
            // later submit-fire's `get_or_compile` finds it Ready ⇒ cache hit. Keyed by
            // `program_identity_hash(bytecode, manifest)` — the SAME (kind,key) manifest
            // the submit path reconstructs, so prefetch-hash ≡ submit-hash ≡ the #10
            // dedup key. Merged R≥2 prefetches each program. Fires before `bytecode`/
            // `bindings` are moved into the carrier below.
            crate::driver::prefetch_compile(driver_idx, &program.bytecode, &program.bindings);
            program_identity_hashes.push(program.identity_hash);
            programs_output_kinds.push(program.output_kinds);
            programs_output_elem_counts.push(program.output_elem_counts);
            logits_positions.extend(program.logits_positions);
            let bindings = program
                .bindings
                .iter()
                .map(|b| match b {
                    pie_sampling_ir::Binding::Logits => pie_driver_abi::SamplingBinding::Logits,
                    pie_sampling_ir::Binding::MtpLogits => {
                        pie_driver_abi::SamplingBinding::MtpLogits
                    }
                    pie_sampling_ir::Binding::Tensor { key, .. } => {
                        pie_driver_abi::SamplingBinding::Tensor { key: *key }
                    }
                })
                .collect();
            // The device-alias late inputs become declared late keys (staged value
            // len 0 → the driver resolves the device-alias instead); their device
            // ptr + R12 flag ride `sampling_late_device_*`, parallel to the keys.
            let late_keys: Vec<u32> =
                program.late_device_inputs.iter().map(|(k, _)| *k).collect();
            req.push_sampling_program(&pie_driver_abi::SamplingProgramSubmission {
                bytecode: program.bytecode,
                inputs: program.submit_inputs,
                bindings,
                late_keys,
                late_inputs: Vec::new(),
            });
            for (_key, device) in program.late_device_inputs {
                req.sampling_late_device_ptrs.push(device.device_ptr());
                req.sampling_late_device_flags.push(device.flag_ptr());
                req.sampling_late_device_lens.push(device.byte_len());
                late_device_handles.push(device);
            }
        }
        if has_programs {
            // The `logits` binding(s) carry the sampling positions as ABSOLUTE
            // sequence positions; map each to its RELATIVE row in this fire's
            // input feed via `position_ids` (absolute==relative only at prefill).
            // Fall back to the last input row when none were supplied so the LM
            // head still runs and the program samples there.
            if !logits_positions.is_empty() {
                req.sampling_indices = logits_positions
                    .iter()
                    .map(|&p| {
                        req.position_ids
                            .iter()
                            .position(|&pos| pos == p)
                            .map(|i| i as u32)
                            .unwrap_or_else(|| (req.token_ids.len() as u32).saturating_sub(1))
                    })
                    .collect();
            } else if req.sampling_indices.is_empty() && !req.token_ids.is_empty() {
                req.sampling_indices = vec![req.token_ids.len() as u32 - 1];
            }
        }

        // Empty-input guard: a forward must compute at least one query row.
        // Without input rows `qo_indptr` collapses to `[0, 0]` and the pass is a
        // no-op submit; the old context API rejected this. Image/audio spans
        // push placeholder rows into `token_ids`, so this covers all input kinds.
        if let Err(e) = forward_prepare::check_input_nonempty(req.token_ids.len()) {
            // Defer to `output()` — `execute: func()` has no error channel.
            state.ctx().table.get_mut(&this)?.exec_error = Some(format!("{e:?}"));
            return Ok(());
        }

        // WIT spec: "if not provided, fallback to causal mask". Then stamp the
        // per-request indptr shape ([0, N]).
        let has_user_mask = !req.masks.is_empty();
        if req.masks.is_empty() && !req.position_ids.is_empty() {
            req.masks = req
                .position_ids
                .iter()
                .map(|&pos| Brle::all_true((pos + 1) as usize))
                .collect();
        }
        req.has_user_mask = has_user_mask;
        req.single_token_mode = !has_user_mask && req.token_ids.len() <= 1;
        req.adapter_bindings[0].seed = adapter_seed.unwrap_or(-1);
        req.qo_indptr = vec![0, req.token_ids.len() as u32];
        req.mask_indptr = vec![0, req.masks.len() as u32];
        req.logit_mask_indptr = vec![0, req.logit_masks.len() as u32];
        req.sampling_indptr = vec![0, req.sampling_indices.len() as u32];
        req.sampler_indptr = vec![0, req.n_samplers() as u32];
        // Sampling-program carrier: per-request count CSR, mirroring
        // `sampler_indptr` (the nested per-program CSRs are rooted by
        // `push_sampling_program`). `[0, 0]` when no program is attached.
        req.sampling_program_indptr = vec![0, req.n_sampling_programs() as u32];
        req.spec_indptr = vec![0, req.spec_token_ids.len() as u32];
        req.kv_page_indptr = vec![0];
        // Batch-affinity id: the KV working set's resource handle replaces the
        // old context id (used by the scheduler for request grouping).
        let affinity = kv_ws.as_ref().map(|d| d.set.rep()).unwrap_or(0);
        req.context_ids = vec![affinity as u64];

        let page_size = crate::page_size::tokens_per_page(model_id);

        // ── prepare: validate descriptors; alloc/CoW + pin write targets; pin
        //    read pages; resolve to driver physical ids — all under one txn.
        //    The whole prepare is synchronous, so it runs inside one
        //    `accessor.with` closure (store + arena both reachable); the owned
        //    `txn` + projection cross back out for the async submit. ──
        type PrepOut = (
            forward_prepare::KvProjection,
            crate::arena::ArenaTxn,
            Vec<crate::arena::MovePlan>,
        );
        let prepared: std::result::Result<PrepOut, String> = 'prepare: {
            let arena_arc = crate::arena::get(model_id, driver_idx);
            let mut arena = arena_arc.lock().unwrap();
            let mut txn = arena.txn_begin();
            let mut move_plans: Vec<crate::arena::MovePlan> = Vec::new();

            type InnerOut = (
                forward_prepare::KvProjection,
                Vec<u32>,
                Vec<u8>,
                Vec<u32>,
                Vec<u32>,
                Vec<u32>,
            );
            let inner: std::result::Result<InnerOut, String> = 'prep: {
                // KV read context → pinned physical pages. Read only the written
                // valid-token prefix: kv-context `len` may include trailing
                // reserved slots (the WIT permits "trailing reserved slots may be
                // empty") that are not part of attention and may be unwritten;
                // resolving the full `len` would reject them. `valid_pages` is the
                // ceil of valid-tokens. Pure prefill (valid_tokens==0) reads none.
                let (context_pages, valid_tokens) = if let Some(d) = &kv_ws {
                    let valid_pages = d.valid_tokens.div_ceil(page_size);
                    let objs = if valid_pages == 0 {
                        Vec::new()
                    } else {
                        match state.ctx().table.get(&d.set) {
                            Ok(ws) => match ws.resolve_read(d.inp_start, valid_pages) {
                                Ok(o) => o,
                                Err(e) => break 'prep Err(e.to_string()),
                            },
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    };
                    let mut pages = Vec::with_capacity(objs.len());
                    for obj in &objs {
                        if let Err(e) = arena.txn_pin(&mut txn, *obj) {
                            break 'prep Err(e.to_string());
                        }
                        match arena.blocks(*obj) {
                            Ok(b) => pages.push(b[0]),
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    }
                    (pages, d.valid_tokens)
                } else {
                    (Vec::new(), 0)
                };

                // KV write outputs → CoW'd + pinned physical pages. #21 Option-B
                // adapter: the write is the CONTIGUOUS slot range `[output_start,
                // output_start+output_len)`. Per-page valid-len is reconstructed
                // from `offset` (the in-page token offset of the first written
                // row) + `n` (the input-token count): page `i` holds
                // `clamp((offset+n) − i·page_size, 0, page_size)` valid tokens.
                // No indices array, no generation/range guard (dropped under #21
                // — the inferlet owns working-set correctness). `cow_write_slot`
                // CoW-chains from the slot's existing object, preserving any
                // in-flight producer prefix (alpha review check #1).
                let mut writes: Vec<forward_prepare::KvWrite> = Vec::new();
                if let Some(d) = &kv_ws {
                    let n = req.token_ids.len() as u32;
                    for i in 0..d.output_len {
                        let idx = d.output_start + i;
                        let valid_len = (d.offset + n)
                            .saturating_sub(i * page_size)
                            .min(page_size);
                        let cow = {
                            let ws = match state.ctx().table.get_mut(&d.set) {
                                Ok(w) => w,
                                Err(e) => break 'prep Err(e.to_string()),
                            };
                            ws.cow_write_slot(idx, &mut txn, &mut arena)
                        };
                        let (obj, move_plan) = match cow {
                            Ok(v) => v,
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        if let Some(mp) = move_plan {
                            move_plans.push(mp);
                        }
                        if let Err(e) = arena.txn_pin(&mut txn, obj) {
                            break 'prep Err(e.to_string());
                        }
                        let page = match arena.blocks(obj) {
                            Ok(b) => b[0],
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        writes.push(forward_prepare::KvWrite {
                            slot_index: idx,
                            page,
                            valid_len,
                        });
                    }
                }

                // Project onto the contiguous driver page run + last-page length.
                let proj = match forward_prepare::project_kv(
                    &context_pages,
                    valid_tokens,
                    &writes,
                    page_size,
                ) {
                    Ok(p) => p,
                    Err(e) => break 'prep Err(format!("{e:?}")),
                };

                // RS v1 (write + in-forward fold only; `rs-context` read deferred
                // to v2 — the hybrid read+write case needs per-request token-range
                // ABI). Buffered slabs ride `rs_buffer_slot_ids` (page-major CSR:
                // slab s = tokens [s·page, (s+1)·page)); the FOLD bit disambiguates
                // — clear = rs-output write-target (W10, write_state=false), set =
                // fold read-source. `rs_slot_ids[r]` is the folded state slot.
                let mut rs_slot_ids: Vec<u32> = Vec::new();
                let mut rs_slot_flags: Vec<u8> = Vec::new();
                let mut rs_buffer_slot_ids: Vec<u32> = Vec::new();
                let mut rs_buffer_slot_indptr: Vec<u32> = vec![0];
                let mut rs_fold_lens: Vec<u32> = Vec::new();
                if let Some(rs) = &rs_ws {
                    // Materialise + pin the buffered write slabs, page-major.
                    let cow = {
                        let ws = match state.ctx().table.get_mut(&rs.set) {
                            Ok(w) => w,
                            Err(e) => break 'prep Err(e.to_string()),
                        };
                        ws.cow_write_buffer(rs.start_token, rs.len_tokens, &mut txn, &mut arena)
                    };
                    let (objs, move_plan) = match cow {
                        Ok(v) => v,
                        Err(e) => break 'prep Err(e.to_string()),
                    };
                    if let Some(mp) = move_plan {
                        move_plans.push(mp);
                    }
                    for obj in &objs {
                        if let Err(e) = arena.txn_pin(&mut txn, *obj) {
                            break 'prep Err(e.to_string());
                        }
                        match arena.blocks(*obj) {
                            Ok(b) => rs_buffer_slot_ids.push(b[0]),
                            Err(e) => break 'prep Err(e.to_string()),
                        }
                    }
                    rs_buffer_slot_indptr.push(rs_buffer_slot_ids.len() as u32);

                    // The folded recurrent_state slot (driver reads/writes it).
                    let folded = match state.ctx().table.get(&rs.set) {
                        Ok(ws) => ws.folded_object(),
                        Err(e) => break 'prep Err(e.to_string()),
                    };
                    let folded_block = match folded {
                        Some(obj) => match arena.blocks(obj) {
                            Ok(b) => b[0],
                            Err(e) => break 'prep Err(e.to_string()),
                        },
                        None => 0,
                    };
                    rs_slot_ids.push(folded_block);

                    // FOLD bit + `rs_fold_lens` iff this pass folds the buffered
                    // suffix into the folded state in-forward (W9 piggyback).
                    let mut flag = 0u8;
                    if let Some(n) = fold_buffered_tokens {
                        flag |= pie_driver_abi::RS_FLAG_FOLD;
                        rs_fold_lens.push(n);
                    } else {
                        rs_fold_lens.push(0);
                    }
                    rs_slot_flags.push(flag);
                } else if fold_buffered_tokens.is_some() {
                    // v1: the fold rides the rs-output set; a fold-only pass with
                    // no rs-output to carry the RS set isn't expressible yet (v2: a
                    // fold descriptor names its own set).
                    break 'prep Err(
                        "fold-buffered without rs-output is unsupported in v1 (rs-output carries the RS set)"
                            .to_string(),
                    );
                }

                Ok((
                    proj,
                    rs_slot_ids,
                    rs_slot_flags,
                    rs_buffer_slot_ids,
                    rs_buffer_slot_indptr,
                    rs_fold_lens,
                ))
            };

            // On any prepare failure: abort the txn (discard staged allocs/CoW
            // copies, release pins) and revert any repointed KV slots; the prior
            // mappings stay visible (W13). No driver submission happened.
            let (proj, rs_slot_ids, rs_slot_flags, rs_buffer_slot_ids, rs_buffer_slot_indptr, rs_fold_lens) =
                match inner {
                    Ok(v) => v,
                    Err(e) => {
                        arena.txn_abort(txn);
                        drop(arena);
                        if let Some(d) = &kv_ws {
                            if let Ok(ws) = state.ctx().table.get_mut(&d.set) {
                                ws.abort_writes();
                            }
                        }
                        break 'prepare Err(e);
                    }
                };

            if !rs_slot_ids.is_empty() {
                req.rs_slot_ids = rs_slot_ids;
                req.rs_slot_flags = rs_slot_flags;
                req.rs_fold_lens = rs_fold_lens;
            }
            if !rs_buffer_slot_ids.is_empty() {
                req.rs_buffer_slot_ids = rs_buffer_slot_ids;
                req.rs_buffer_slot_indptr = rs_buffer_slot_indptr;
            }

            // Release the arena lock BEFORE the async submit; the owned `txn`
            // keeps the pins/CoW copies alive until commit/abort.
            drop(arena);
            Ok((proj, txn, move_plans))
        };

        let (proj, txn, move_plans) = match prepared {
            Ok(v) => v,
            Err(e) => {
                // Recoverable prepare failure — defer it to `output()` (the WIT
                // `execute: func()` has no error channel). The txn / KV slots were
                // already aborted/reverted in the prepare block.
                state.ctx().table.get_mut(&this)?.exec_error = Some(e);
                return Ok(());
            }
        };

        // Issue the device d2d for every CoW'd write target: copy the original
        // page content into the private copy before the driver writes into it.
        for mp in &move_plans {
            if let Err(e) = crate::driver::copy_d2d(driver_idx, &mp.from, &mp.to) {
                tracing::warn!("forward CoW d2d copy failed: {e:#}");
            }
        }

        // CAS-seal eligibility (W6/W7): host-hash the full pages this forward
        // fills from an EMPTY context (`valid_tokens == 0`), so the forward's
        // tokens start at a page boundary and each page's whole content is known
        // here. Chained from prev_hash 0. Sealing onto a non-empty context needs
        // the context tip's sealed hash (follow-up); until then those pages stay
        // private-dirty (W7) — correct, just no dedup.
        let seal_eligible = !proj.full_page_writes.is_empty()
            && kv_ws.as_ref().map(|d| d.valid_tokens).unwrap_or(0) == 0
            && req.position_ids.len() == req.token_ids.len()
            && req.masks.len() == req.token_ids.len();
        let seal_hashes: Vec<(u32, u64)> = if seal_eligible {
            let page_hashes = crate::page_hash::compute_page_hashes(
                page_size as usize,
                &req.token_ids,
                &req.position_ids,
                &req.masks,
                0,
                adapter_seed,
            );
            proj
                .full_page_writes
                .iter()
                .filter_map(|&slot| page_hashes.get(slot as usize).map(|&h| (slot, h)))
                .collect()
        } else {
            Vec::new()
        };

        // Carry the KV working-set handle + the txn across the async boundary;
        // finalize (after the driver round-trip) commits (seal full pages) /
        // aborts on them.
        let kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>> = kv_ws
            .as_ref()
            .map(|d| Resource::new_borrow(d.set.rep()));

        // The RS working set whose folded boundary advances on a committed
        // fold-buffered (W9) — v1 rides the rs-working-set.
        let rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>> =
            if fold_buffered_tokens.is_some() {
                rs_ws.as_ref().map(|d| Resource::new_borrow(d.set.rep()))
            } else {
                None
            };

        // #21 next-inputs carrier: register this pass as a pipeline source (if it
        // declared `next-inputs`) + inject the prior producer's retained sample
        // into this pass. The host owns the global link id; the guest threads
        // none. No-op when neither role applies. Must run AFTER `input-tokens` is
        // staged (it reads `req.token_ids.len()`) and BEFORE submit. The context id
        // (KV working-set rep) scopes the carryover to consecutive same-context
        // passes so a terminal producer's dangling carry can't leak into the next
        // context (0 = no-KV pass ⇒ never a carrier consumer/producer).
        let next_input_context_id = kv_ws.as_ref().map(|d| d.set.rep()).unwrap_or(0);
        // #26 fresh-generate: BEFORE the carrier's consumer-inject, drop any
        // dangling carry left on THIS context by a prior generate's terminal
        // producer (stop-terminal / explicit-restart). Free the stale retained
        // device buffer on this prime's request so it doesn't leak; the prime
        // then does NOT inject the stale token. A different context's pending is
        // left untouched for `apply_next_input_carrier`'s own mismatch branch.
        if fresh_generate {
            if let Some(link) = crate::api::next_input_map::clear_pending_for_context(
                &mut state.pending_next_input,
                next_input_context_id,
            ) {
                req.push_next_input_free_link(link);
            }
            // #23: a fresh generate starts a clean dependency chain — drop any
            // lingering terminal-producer link from the prior generation.
            state.overlap_links.clear();
        }
        let next_input_deps = crate::api::next_input_map::apply_next_input_carrier(
            &mut state.pending_next_input,
            &mut state.next_input_link_counter,
            &mut req,
            &next_input_positions,
            next_input_context_id,
        );

        // #27 cut #1 fast-path: pre-allocate a pinned host buffer per declared
        // output value + thread the dst pointers into `req.sampling_output_*`, so
        // the driver eager-D2H's the sampled outputs straight into them (skipping
        // the `ForwardResponse` marshal). Empty when not fast-path-eligible or
        // without the `driver-cuda` feature ⇒ legacy path. Must run BEFORE submit
        // (mutates `req`); the buffers are carried on the `PendingForward`.
        let pinned_outputs = crate::api::tensor_io::populate_output_fastpath(
            &mut req,
            &programs_output_kinds,
            &programs_output_elem_counts,
            // #36/#37: a request takes the rich path iff it has an attached CUSTOM
            // program (marshals to `per_req` → the eager-D2H pinned never fills). A
            // recognized de-hardwired STANDARD sampler writes `pi.sampled`, so the
            // pinned fast-path is correct and it keeps it (#37 restores that perf).
            has_custom_program,
        );

        // Single-model: the SERVICE routes to the bound model; no model_id arg.
        let submit_result = inference::submit_async(
            req,
            driver_idx,
            proj.physical_page_ids,
            proj.last_page_len,
            program_identity_hashes,
        );

        let rx = match submit_result {
            Ok(rx) => rx,
            Err(e) => {
                // Submit never reached the driver — abort the txn + revert the
                // repointed KV slots (W13).
                let arena_arc = crate::arena::get(model_id, driver_idx);
                arena_arc.lock().unwrap().txn_abort(txn);
                if let Some(ko) = &kv_set {
                    if let Ok(ws) = state.ctx().table.get_mut(ko) {
                        ws.abort_writes();
                    }
                }
                tracing::warn!("inference::submit failed: {e:#}");
                // Defer to `output()` — `execute: func()` has no error channel.
                state.ctx().table.get_mut(&this)?.exec_error = Some(e.to_string());
                return Ok(());
            }
        };

        // Eager-submit done; store the in-flight state on the forward-pass
        // (Option A). Phase-2 (await → finalize → build output tensors) runs in
        // the async `output()`/`outputs()`, which take this `PendingForward`. The
        // `&mut state` borrow releases here so the run-ahead loop can hold two
        // passes in flight.
        let pending = PendingForward {
            rx,
            txn: ForwardTxnGuard::new(txn, model_id, driver_idx),
            kv_set,
            seal_hashes,
            model_id,
            driver_idx,
            rs_fold_set,
            fold_buffered_tokens,
            programs_output_kinds,
            programs_output_elem_counts,
            pinned_outputs,
            late_device_inputs: late_device_handles,
            profile_start,
            next_input_deps,
        };
        state.ctx().table.get_mut(&this)?.pending = Some(pending);
        Ok(())
    }

impl InstanceState {
    /// Commit (on driver success) or abort (on failure) the forward transaction
    /// from `execute()`. Commit releases pins, publishes the CoW'd write targets
    /// (`commit_writes`), and CAS-seals eligible full pages; abort discards
    /// staged objects and reverts repointed slots.
    #[allow(clippy::too_many_arguments)]
    fn finalize_forward_txn(
        &mut self,
        success: bool,
        txn: crate::arena::ArenaTxn,
        kv_set: Option<Resource<crate::working_set::kv::KvWorkingSet>>,
        seal_hashes: Vec<(u32, u64)>,
        model_id: usize,
        driver_idx: usize,
        rs_fold_set: Option<Resource<crate::working_set::rs::RsWorkingSet>>,
        fold_tokens: Option<u32>,
        next_input_deps: crate::api::next_input_map::NextInputDeps,
    ) -> Result<bool> {
        // #23 overlap abort-isolation: resolve the cascade. A consumer that
        // injected from a producer link that did NOT explicitly commit (aborted OR
        // unresolved — fail-closed) is forced to abort even on driver success, so a
        // poisoned generation never commits its txn/KV. The write-log also records
        // THIS pass's (effective) outcome under its produced link, chaining the
        // poison downstream. This is device-drain-neutral (host txn/KV only — it
        // never touches the device `retained_next_input` consumer count).
        let success = self.overlap_links.finalize(success, next_input_deps);

        // Carry rollback: a terminal producer that aborted with its carry still
        // pending (no consumer took it in the overlap) leaves a dangling carry —
        // clear it so a later same-context pass doesn't inject the aborted sample.
        // (The consumed case is already covered: the consumer emitted the
        // drain-gated free-link at inject; here we only catch the un-consumed
        // terminal.) The device retained buffer is freed by the next pass's
        // `clear_pending_for_context` (no leak).
        let terminal_dangling = !success
            && next_input_deps.produced.is_some_and(|prod| {
                self.pending_next_input
                    .as_ref()
                    .is_some_and(|p| p.link == prod)
            });
        if terminal_dangling {
            self.pending_next_input = None;
        }

        let arena_arc = crate::arena::get(model_id, driver_idx);
        if success {
            // Commit, publish the repointed slots, then CAS-seal eligible full
            // pages. Lock order is arena → kv_cas, both held sync (no await).
            let mut arena = arena_arc.lock().unwrap();
            arena
                .txn_commit(txn)
                .map_err(|e| anyhow::anyhow!("forward txn_commit failed: {e}"))?;
            if let Some(kv_set) = &kv_set {
                let cas_arc = crate::working_set::kv_cas::get(model_id, driver_idx);
                let mut cas = cas_arc.lock().unwrap();
                if let Ok(ws) = self.ctx().table.get_mut(kv_set) {
                    ws.commit_writes();
                    for (slot, hash) in &seal_hashes {
                        if let Err(e) = ws.seal(*slot, *hash, &mut arena, &mut cas) {
                            tracing::warn!("CAS seal of slot {slot} failed: {e}");
                        }
                    }
                }
            }
            // Advance the RS folded boundary on a committed in-forward fold (W9):
            // consume the first `n` buffered tokens into the folded state. Only
            // on success — a fold never advances across an aborted forward.
            if let (Some(n), Some(rs_set)) = (fold_tokens, &rs_fold_set) {
                if let Ok(ws) = self.ctx().table.get_mut(rs_set) {
                    if let Err(e) = ws.advance_fold(n, &mut arena) {
                        tracing::warn!("advance_fold({n}) failed: {e:?}");
                    }
                }
            }
        } else {
            {
                let mut arena = arena_arc.lock().unwrap();
                arena.txn_abort(txn);
            }
            if let Some(kv_set) = &kv_set {
                if let Ok(ws) = self.ctx().table.get_mut(kv_set) {
                    ws.abort_writes();
                }
            }
        }
        Ok(success)
    }
}

// =============================================================================
// Grammar resource
// =============================================================================

/// A compiled grammar that describes valid output structure.
#[derive(Debug)]
pub struct Grammar {
    /// The original source string (for compiled grammar cache keying).
    pub source: String,
    /// The parsed grammar AST.
    pub inner: Arc<InternalGrammar>,
}

impl pie::core::inference::HostGrammar for InstanceState {
    async fn from_json_schema(
        &mut self,
        schema: String,
    ) -> Result<Result<Resource<Grammar>, String>> {
        match json_schema_to_grammar(&schema, &JsonSchemaOptions::default()) {
            Ok(g) => {
                let grammar = Grammar {
                    source: schema,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn json(&mut self) -> Result<Resource<Grammar>> {
        let g = builtin_json_grammar()?;
        let grammar = Grammar {
            source: "__builtin_json__".into(),
            inner: Arc::new(g),
        };
        Ok(self.ctx().table.push(grammar)?)
    }

    async fn from_regex(&mut self, pattern: String) -> Result<Result<Resource<Grammar>, String>> {
        match regex_to_grammar(&pattern) {
            Ok(g) => {
                let grammar = Grammar {
                    source: pattern,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn from_ebnf(&mut self, ebnf: String) -> Result<Result<Resource<Grammar>, String>> {
        match InternalGrammar::from_ebnf(&ebnf, "root") {
            Ok(g) => {
                let grammar = Grammar {
                    source: ebnf,
                    inner: Arc::new(g),
                };
                Ok(Ok(self.ctx().table.push(grammar)?))
            }
            Err(e) => Ok(Err(e.to_string())),
        }
    }

    async fn to_string(&mut self, this: Resource<Grammar>) -> Result<String> {
        let g = self.ctx().table.get(&this)?;
        Ok(g.inner.to_string())
    }

    async fn drop(&mut self, this: Resource<Grammar>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}

// =============================================================================
// Matcher resource
// =============================================================================

/// Stateful matcher that walks the grammar automaton, producing token masks.
pub struct Matcher {
    pub(crate) inner: GrammarMatcher,
}

impl std::fmt::Debug for Matcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Matcher").finish()
    }
}

impl pie::core::inference::HostMatcher for InstanceState {
    async fn new(&mut self, grammar: Resource<Grammar>) -> Result<Resource<Matcher>> {
        let grammar_res = self.ctx().table.get(&grammar)?;
        let source = grammar_res.source.clone();
        let grammar_inner = grammar_res.inner.clone();

        // Single-model: the tokenizer comes from the global bound model.
        let model = crate::model::model();
        let tok = model.tokenizer().clone();
        let stop_tokens = model.instruct().seal();

        let compiled = CompiledGrammar::get_or_compile(&source, &grammar_inner, &tok);
        let inner = GrammarMatcher::with_compiled(compiled, tok, stop_tokens, 10);

        let matcher = Matcher { inner };
        Ok(self.ctx().table.push(matcher)?)
    }

    async fn accept_tokens(
        &mut self,
        this: Resource<Matcher>,
        token_ids: Vec<u32>,
    ) -> Result<Result<(), String>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        for &id in &token_ids {
            if !matcher.inner.accept_token(id) {
                return Ok(Err(format!("token {} rejected by grammar", id)));
            }
        }
        Ok(Ok(()))
    }

    async fn mask(&mut self, this: Resource<Matcher>) -> Result<Vec<u32>> {
        let matcher = self.ctx().table.get_mut(&this)?;
        // The packed allowed-token bitmask (`[ceil(vocab/32)]` u32, bit 1 =
        // allowed) — the `mask-apply` (0x65) mask operand. Returned directly,
        // no BRLE round-trip.
        Ok(matcher.inner.fill_next_token_mask())
    }

    async fn is_terminated(&mut self, this: Resource<Matcher>) -> Result<bool> {
        let matcher = self.ctx().table.get(&this)?;
        Ok(matcher.inner.is_terminated())
    }

    async fn reset(&mut self, this: Resource<Matcher>) -> Result<()> {
        let matcher = self.ctx().table.get_mut(&this)?;
        matcher.inner.reset();
        Ok(())
    }

    async fn drop(&mut self, this: Resource<Matcher>) -> Result<()> {
        self.ctx().table.delete(this)?;
        Ok(())
    }
}


#[cfg(test)]
mod sampling_program_tests {
    use super::*;
    use crate::api::pie::core::tensor as wit;
    use crate::api::program_decode::decode_program;

    // #37: the host-side recognizer-mirror distinguishes recognized de-hardwired
    // STANDARD samplers (→ fast-path/pinned) from custom programs (→ rich).
    #[test]
    fn is_recognized_standard_matches_standard_rejects_other() {
        let vocab = 128u32;
        let std_progs = sampling_edsl::standard_programs(vocab).expect("standard programs");
        assert!(!std_progs.is_empty());
        // Every driver-recognized STANDARD sampler bytecode is recognized host-side.
        for (bytecode, _) in &std_progs {
            assert!(
                is_recognized_standard(bytecode, vocab),
                "standard sampler must be recognized"
            );
        }
        // A standard bytecode for a DIFFERENT vocab is NOT recognized for `vocab`
        // (the hash is vocab-specific) — proves it's a real membership check, not
        // a trivially-true gate that would let custom programs through.
        let other = sampling_edsl::standard_programs(256).expect("standard programs (other vocab)");
        assert!(!is_recognized_standard(&other[0].0, vocab));
        // Garbage bytecode → not recognized → conservative rich (fail-safe class).
        assert!(!is_recognized_standard(b"not-a-program", vocab));
    }

    // #23 verify seam targeting predicate (env-free core of `test_force_producer_abort`).
    #[test]
    fn abort_target_matches_only_configured_producer() {
        // Unset target ⇒ never matches (ZERO production behavior when env unset).
        assert!(!abort_target_matches(Some(2), None));
        assert!(!abort_target_matches(None, None));
        // Configured target ⇒ matches ONLY the producer for that exact link.
        assert!(abort_target_matches(Some(2), Some(2)));
        assert!(!abort_target_matches(Some(3), Some(2))); // a different producer
        assert!(!abort_target_matches(None, Some(2))); // a non-producer pass
    }

    /// WIT-shaped parts for a greedy-argmax program over a `[vocab]` logits
    /// input: `Input(0)` then `ReduceArgmax` → one `i32` (Token) output.
    fn argmax_parts(vocab: u32) -> (Vec<wit::Input>, Vec<wit::Op>, Vec<wit::Output>) {
        let inputs = vec![wit::Input {
            shape: vec![vocab],
            dtype: wit::Dtype::F32,
            ready: wit::Readiness::Submit,
        }];
        let ops = vec![
            wit::Op {
                outputs: vec![wit::Value {
                    id: 0,
                    shape: vec![vocab],
                    dtype: wit::Dtype::F32,
                }],
                kind: wit::OpKind::Input(0),
            },
            wit::Op {
                outputs: vec![wit::Value {
                    id: 1,
                    shape: vec![],
                    dtype: wit::Dtype::I32,
                }],
                kind: wit::OpKind::ReduceArgmax(0),
            },
        ];
        (inputs, ops, vec![wit::Output { id: 1, kind: wit::OutputKind::Token }])
    }

    #[test]
    fn decode_argmax_program_infers_token_output() {
        let (inputs, ops, outs) = argmax_parts(32);
        let program = decode_program(inputs, ops, outs).expect("valid argmax program");
        assert_eq!(program.inputs.len(), 1);
        let kinds = pie_sampling_ir::output_kinds(&program).expect("kinds");
        // i32 argmax output ⇒ Token (the dtype⇒kind convention (a)).
        assert_eq!(kinds, vec![pie_sampling_ir::OutputKind::Token]);
        // Bytecode round-trips through the L0 codec the carrier ships.
        let bytecode = pie_sampling_ir::encode(&program);
        let back = pie_sampling_ir::decode(&bytecode).expect("re-decode");
        assert_eq!(back, program);
    }

    #[test]
    fn decode_rejects_out_of_range_output() {
        // Output id 9 is never defined → fail loud (not a silent miscompile).
        let (inputs, ops, _) = argmax_parts(8);
        assert!(
            decode_program(inputs, ops, vec![wit::Output { id: 9, kind: wit::OutputKind::Token }])
                .is_err()
        );
    }

    #[test]
    fn decode_rejects_oversized_shape() {
        // A rank-5 input shape exceeds MAX_RANK(4) → fallible decode, never panic.
        let inputs = vec![wit::Input {
            shape: vec![1, 1, 1, 1, 1],
            dtype: wit::Dtype::F32,
            ready: wit::Readiness::Submit,
        }];
        assert!(decode_program(inputs, Vec::new(), Vec::new()).is_err());
    }

    #[test]
    fn build_tensors_token_fast_path() {
        // ForwardOutput::Token + a single Token-output program ⇒ one `[1] i32`
        // tensor carrying the token (the #21 output()→tensor shape).
        use pie::core::tensor::Dtype;
        let out = build_output_tensors(
            ForwardOutput::Token(42),
            &[vec![pie_sampling_ir::OutputKind::Token]],
            &[vec![1]],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![1]);
        assert!(matches!(out[0].dtype, Dtype::I32));
        assert_eq!(out[0].data, 42i32.to_le_bytes().to_vec());
    }

    #[test]
    fn build_tensors_mirostat_token_entropy() {
        // The 4090 mirostat shape: a `[Token, Scalar]` program. The token rides
        // `resp.tokens` → `[1] i32`; the scalar surprise S rides the `entropies`
        // channel → `[1] f32` (SEAM-A). Tensors are the program's declared
        // outputs in order.
        use pie::core::tensor::Dtype;
        use pie_sampling_ir::OutputKind;
        let resp = pie_driver_abi::ForwardResponse {
            tokens: vec![137],
            entropies: vec![2.7],
            ..Default::default()
        };
        let out = build_output_tensors(
            ForwardOutput::Response(resp),
            &[vec![OutputKind::Token, OutputKind::Scalar]],
            &[vec![1, 1]],
        );
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0].dtype, Dtype::I32));
        assert_eq!(out[0].data, 137i32.to_le_bytes().to_vec());
        assert!(matches!(out[1].dtype, Dtype::F32));
        assert_eq!(out[1].data, 2.7f32.to_le_bytes().to_vec());
    }

    #[test]
    fn build_tensors_k_token_reads_program_tokens() {
        // #32: a `[k]`-Token output (elem_count k>1) reads its k tokens from the
        // per-(request,output) `program_tokens` CSR (off the dense `tokens` / spec
        // channels) → a `[k] i32` tensor. The per-request `resp` carries one
        // segment per output slot.
        use pie::core::tensor::Dtype;
        use pie_sampling_ir::OutputKind;
        let resp = pie_driver_abi::ForwardResponse {
            // one output slot, segment [0,3) = tokens [11, 22, 33].
            program_tokens_indptr: vec![0, 3],
            program_tokens: vec![11, 22, 33],
            ..Default::default()
        };
        let out = build_output_tensors(
            ForwardOutput::Response(resp),
            &[vec![OutputKind::Token]],
            &[vec![3]],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape, vec![3]);
        assert!(matches!(out[0].dtype, Dtype::I32));
        let want: Vec<u8> = [11u32, 22, 33].iter().flat_map(|t| t.to_le_bytes()).collect();
        assert_eq!(out[0].data, want);
    }

    #[test]
    fn program_bytecode_roundtrips_through_carrier() {
        // The host accumulator → bridge carrier → readback path: a program's
        // opaque bytecode + a submit-bound input survive untouched.
        let (inputs, ops, outs) = argmax_parts(32);
        let bytecode = pie_sampling_ir::encode(&decode_program(inputs, ops, outs).unwrap());
        let submission = pie_driver_abi::SamplingProgramSubmission {
            bytecode: bytecode.clone(),
            inputs: vec![pie_driver_abi::SamplingInput {
                key: 0,
                bytes: vec![1, 2, 3, 4],
            }],
            bindings: vec![
                pie_driver_abi::SamplingBinding::Logits,
                pie_driver_abi::SamplingBinding::Tensor { key: 0 },
            ],
            late_keys: vec![],
            late_inputs: vec![],
        };
        let mut req = pie_driver_abi::ForwardRequest::default();
        req.push_sampling_program(&submission);
        assert_eq!(req.n_sampling_programs(), 1);
        let back = req.sampling_program_at(0).unwrap();
        assert_eq!(back.bytecode, bytecode);
        // The per-slot binding-map survives the carrier round-trip.
        assert_eq!(
            back.bindings,
            vec![
                pie_driver_abi::SamplingBinding::Logits,
                pie_driver_abi::SamplingBinding::Tensor { key: 0 },
            ]
        );
    }
}
