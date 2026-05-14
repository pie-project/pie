//! Request and response types for inference.
//!
//! Defines the wire format for forward pass batching.

use serde::{Deserialize, Serialize};
use smallvec::{smallvec, SmallVec};

use crate::adapter::AdapterId;
use crate::context::ContextId;
use crate::context::pagestore::PhysicalPageId;
use crate::inference::brle::{self, Brle};
use crate::device::DeviceId;

/// Inline storage for the page-trim bitmap. Sized to cover up to 1024 pages
/// (16 u64 words = 128 bytes, fits in one cache line per word) without ever
/// touching the heap. Larger contexts spill to the heap transparently.
const TRIM_INLINE_WORDS: usize = 16;
type TrimBits = SmallVec<[u64; TRIM_INLINE_WORDS]>;

// =============================================================================
// Page-trim plan
// =============================================================================
//
// When every query row of a request's attention mask agrees that an entire
// page's worth of KV positions is False, that physical page can be excluded
// from the wire-format `kv_page_indices` — the kernel reads fewer KV slots
// and the BRLE rows get sliced down to match. This is a pure performance
// optimization with no semantic change: position IDs of input tokens are
// unaffected (RoPE is independent of page list shape) and the page-hash
// chain used by the radix-trie dedup operates on `req.masks` upstream of
// this point, so trimming the wire copy doesn't perturb caching.
//
// The eligibility window stops at `first_writeable_page` — pages that the
// kernel will write new K/V into this pass cannot be dropped even if the
// mask says all-False, because the kernel's write target is determined by
// position-in-`kv_page_indices`.
//
// Eligibility math:
//   total_kv = (num_pages - 1) * page_size + last_page_len   (post-pass)
//   kv_before = total_kv - tokens.len()                       (pre-pass)
//   first_writeable_page = kv_before / page_size

/// A computed trim plan for a single request: which pages to drop and the
/// corresponding bit ranges to slice out of every BRLE row.
struct TrimPlan {
    /// Bitmask over `[0, num_pages)`: bit p set ⇒ page p is dropped.
    dropped_bits: TrimBits,
    /// Sorted disjoint `[s, e)` ranges in original-coord space, one per
    /// dropped page: `[p*page_size, (p+1)*page_size)`. Passed to
    /// `Brle::write_skipping` for each row.
    skip_ranges: Vec<(u32, u32)>,
}

impl TrimPlan {
    /// Compute the trim plan, or return `None` if no pages can be dropped.
    /// Returning `None` means the caller should take the fast path with
    /// zero extra allocations.
    fn compute(
        masks: &[Brle],
        num_pages: u32,
        last_page_len: u32,
        page_size: u32,
        num_input_tokens: u32,
    ) -> Option<Self> {
        if num_pages == 0 || page_size == 0 || masks.is_empty() {
            return None;
        }

        // Eligibility window: only pages strictly before the first page that
        // receives new K/V writes are candidates. last_page_len reflects the
        // post-pass state for non-spec input tokens; subtracting num_input_tokens
        // yields the pre-pass kv length. Speculative tokens write past
        // last_page_len into reserved pages, which are also writeable, but
        // they live in pages >= first_writeable_page either way so they don't
        // affect the cutoff.
        let total_kv = (num_pages - 1) * page_size + last_page_len;
        let kv_before = total_kv.saturating_sub(num_input_tokens);
        let first_writeable_page = kv_before / page_size;
        if first_writeable_page == 0 {
            return None;
        }

        let total_seq_len = total_kv;
        let num_words = ((num_pages as usize) + 63) / 64;

        // Running eligibility: AND-reduction across rows, seeded with the
        // writeable-window mask. SmallVec keeps both bitmaps inline on the
        // stack for typical `num_pages <= TRIM_INLINE_WORDS * 64` (1024).
        let mut eligible: TrimBits = smallvec![0u64; num_words];
        brle::set_bits(&mut eligible, 0, first_writeable_page);

        let mut row_bits: TrimBits = smallvec![0u64; num_words];
        for mask in masks {
            for w in row_bits.iter_mut() {
                *w = 0;
            }
            mask.droppable_page_bits(page_size, num_pages, total_seq_len, &mut row_bits);
            for (e, r) in eligible.iter_mut().zip(row_bits.iter()) {
                *e &= *r;
            }
            // Early exit: once eligibility hits zero, no further rows can
            // bring it back. Common case for non-causal masks where rows
            // disagree on which pages are reachable.
            if eligible.iter().all(|&w| w == 0) {
                return None;
            }
        }

        // Materialize skip_ranges in page order. Walk set bits LSB-first per
        // word; each set bit p contributes [p*page_size, (p+1)*page_size).
        let mut skip_ranges: Vec<(u32, u32)> = Vec::new();
        for (w_idx, &word) in eligible.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let lsb = bits.trailing_zeros();
                let p = (w_idx as u32) * 64 + lsb;
                if p >= num_pages {
                    break;
                }
                skip_ranges.push((p * page_size, (p + 1) * page_size));
                bits &= bits.wrapping_sub(1);
            }
        }

        Some(TrimPlan {
            dropped_bits: eligible,
            skip_ranges,
        })
    }

    #[inline]
    fn is_page_dropped(&self, p: u32) -> bool {
        let w = (p / 64) as usize;
        let b = p % 64;
        self.dropped_bits.get(w).map(|word| (word >> b) & 1 != 0).unwrap_or(false)
    }
}

/// Sampler configuration for token generation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Sampler {
    Multinomial { temperature: f32, seed: Option<u32> },
    TopK { temperature: f32, k: u32 },
    TopP { temperature: f32, p: f32 },
    MinP { temperature: f32, p: f32 },
    TopKTopP { temperature: f32, k: u32, p: f32 },
    Embedding,
    Dist { temperature: f32, num_tokens: u32 },
    /// Return raw (pre-softmax, untemperatured) logits for each requested
    /// position. The host packs them as native-endian f32 bytes; length =
    /// vocab_size * 4 per request slot.
    RawLogits,
    /// Return log p(token | context) at this position — one f32 per slot.
    /// Computed via log_softmax(logits) with no temperature scaling.
    Logprob { token_id: u32 },
    /// Multi-candidate variant of Logprob — returns log p(t | context) for
    /// each `t` in `token_ids` at this position. One inner list per slot.
    Logprobs { token_ids: Vec<u32> },
    /// Return Shannon entropy H(p) of the unscaled next-token distribution.
    Entropy,
}

impl Sampler {
    /// Get the sampler type ID for serialization.
    pub fn type_id(&self) -> u32 {
        match self {
            Sampler::Multinomial { .. } => 1,
            Sampler::TopK { .. } => 2,
            Sampler::TopP { .. } => 3,
            Sampler::MinP { .. } => 4,
            Sampler::TopKTopP { .. } => 5,
            Sampler::Embedding => 6,
            Sampler::Dist { .. } => 0,
            Sampler::RawLogits => 7,
            Sampler::Logprob { .. } => 8,
            Sampler::Logprobs { .. } => 9,
            Sampler::Entropy => 10,
        }
    }

    /// Get the temperature value.
    pub fn temperature(&self) -> f32 {
        match self {
            Sampler::Multinomial { temperature, .. } => *temperature,
            Sampler::TopK { temperature, .. } => *temperature,
            Sampler::TopP { temperature, .. } => *temperature,
            Sampler::MinP { temperature, .. } => *temperature,
            Sampler::TopKTopP { temperature, .. } => *temperature,
            Sampler::Embedding => 0.0,
            Sampler::Dist { temperature, .. } => *temperature,
            Sampler::RawLogits => 0.0,
            Sampler::Logprob { .. } => 0.0,
            Sampler::Logprobs { .. } => 0.0,
            Sampler::Entropy => 0.0,
        }
    }

    /// Get top_k value (0 if not applicable).
    pub fn top_k(&self) -> u32 {
        match self {
            Sampler::TopK { k, .. } => *k,
            Sampler::TopKTopP { k, .. } => *k,
            _ => 0,
        }
    }

    /// Get top_p value (1.0 if not applicable).
    pub fn top_p(&self) -> f32 {
        match self {
            Sampler::TopP { p, .. } => *p,
            Sampler::TopKTopP { p, .. } => *p,
            _ => 1.0,
        }
    }

    /// Get min_p value (0.0 if not applicable).
    pub fn min_p(&self) -> f32 {
        match self {
            Sampler::MinP { p, .. } => *p,
            _ => 0.0,
        }
    }

    /// Get seed value (0 if not applicable or unset).
    pub fn seed(&self) -> u32 {
        match self {
            Sampler::Multinomial { seed, .. } => seed.unwrap_or(0),
            _ => 0,
        }
    }
}

/// Forward pass request for a single sequence.
#[derive(Debug, Clone)]
pub struct ForwardPassRequest {
    /// Context ID for KV cache page resolution.
    pub context_id: ContextId,
    /// Input token IDs.
    pub tokens: Vec<u32>,
    /// Token positions.
    pub positions: Vec<u32>,
    /// Speculative token IDs.
    pub speculative_tokens: Vec<u32>,
    /// Speculative token positions.
    pub speculative_positions: Vec<u32>,
    /// Whether to include speculative tokens in the output.
    pub output_speculative_tokens: bool,
    /// Attention masks (BRLE encoded, one per token).
    pub masks: Vec<Brle>,
    /// Whether the user supplied custom masks via `attention_mask()`. False
    /// means the runtime synthesized causal masks as the default. Drives
    /// kernel dispatch: a single-token request with a user-supplied mask must
    /// route to the prefill kernel (which honors `custom_mask`); a single-
    /// token request with synthesized causal can use the cuda-graph decode
    /// kernel which has no `custom_mask` argument.
    pub has_user_mask: bool,
    /// Logit mask (BRLE encoded, applied to vocabulary).
    pub logit_mask: Option<Brle>,
    /// Indices of tokens to sample from.
    pub sampling_indices: Vec<u32>,
    /// Sampler configurations for each sampling index.
    pub samplers: Vec<Sampler>,
    /// Optional adapter ID.
    pub adapter_id: Option<AdapterId>,
    /// Optional adapter seed (for ZO optimization).
    pub adapter_seed: Option<i64>,
}

/// One typed result per `forward_pass.sampler(...)` slot, in slot order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SlotOutput {
    Token(u32),
    Distribution(Vec<u32>, Vec<f32>),
    /// Native-endian f32 bytes for `Sampler::RawLogits`.
    Logits(Vec<u8>),
    /// Length 1 for `Sampler::Logprob`, length K for `Sampler::Logprobs`.
    Logprobs(Vec<f32>),
    Entropy(f32),
    /// Hidden-state embedding bytes (placeholder; not wired yet).
    Embedding(Vec<u8>),
}

/// Output from a forward pass: per-slot results plus per-request spec channel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ForwardPassOutput {
    pub slots: Vec<SlotOutput>,
    /// Next iteration's speculative draft tokens (empty if non-spec).
    pub spec_tokens: Vec<u32>,
    pub spec_positions: Vec<u32>,
}

/// Response for a single forward pass request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForwardPassResponse {
    pub tokens: Vec<u32>,
    pub dists: Vec<(Vec<u32>, Vec<f32>)>,
    /// Raw logits as native-endian f32 bytes, one buffer per RawLogits slot.
    /// Empty unless the request used `Sampler::RawLogits` for some position.
    #[serde(default)]
    pub logits: Vec<Vec<u8>>,
    /// Per-slot logprobs: length 1 for `Sampler::Logprob`, length K for
    /// `Sampler::Logprobs`. Empty unless the request used either variant.
    #[serde(default)]
    pub logprobs: Vec<Vec<f32>>,
    /// Per-slot entropies. Empty unless the request used `Sampler::Entropy`.
    #[serde(default)]
    pub entropies: Vec<f32>,
    /// Next speculative tokens (empty if non-speculative).
    pub spec_tokens: Vec<u32>,
    /// Next speculative positions (empty if non-speculative).
    pub spec_positions: Vec<u32>,
}

// =============================================================================
// Batched Request (for Python RPC)
// =============================================================================

/// Wrapper for Vec<u32> that serializes as raw bytes.
#[derive(Debug, Clone, Default)]
pub struct ByteVec(pub Vec<u32>);

impl Serialize for ByteVec {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes: &[u8] = bytemuck::cast_slice(&self.0);
        serializer.serialize_bytes(bytes)
    }
}

impl<'de> Deserialize<'de> for ByteVec {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        let values: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(ByteVec(values))
    }
}

/// Wrapper for Vec<f32> that serializes as raw bytes.
#[derive(Debug, Clone, Default)]
pub struct ByteVecF32(pub Vec<f32>);

impl Serialize for ByteVecF32 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let bytes: &[u8] = bytemuck::cast_slice(&self.0);
        serializer.serialize_bytes(bytes)
    }
}

impl<'de> Deserialize<'de> for ByteVecF32 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bytes: Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        let values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
            .collect();
        Ok(ByteVecF32(values))
    }
}

/// Batched forward pass request sent to Python.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchedForwardPassRequest {
    // === Core token data (concatenated, ByteVec for zero-copy) ===
    pub token_ids: ByteVec,
    pub position_ids: ByteVec,

    // === KV cache layout ===
    pub kv_page_indices: ByteVec,
    pub kv_page_indptr: ByteVec,
    pub kv_last_page_lens: ByteVec,

    // === Query/output indirection ===
    pub qo_indptr: ByteVec,

    // === Attention masks (BRLE, flattened) ===
    pub flattened_masks: ByteVec,
    pub mask_indptr: ByteVec,

    // === Logit masks (BRLE, per request) ===
    pub logit_masks: ByteVec,
    pub logit_mask_indptr: ByteVec,

    // === Sampling indices ===
    pub sampling_indices: ByteVec,
    pub sampling_indptr: ByteVec,

    // === Sampler parameters (SoA, one per sampler) ===
    pub sampler_temperatures: ByteVecF32,
    pub sampler_top_k: ByteVec,
    pub sampler_top_p: ByteVecF32,
    pub sampler_min_p: ByteVecF32,
    pub sampler_types: ByteVec,
    pub sampler_seeds: ByteVec,
    pub request_num_samplers: ByteVec,

    // === Per-sampler ragged label lists for Logprob/Logprobs ===
    // For Logprob: 1 label per sampler. For Logprobs: K labels per sampler.
    // For all other sampler types: empty range (0 labels).
    #[serde(default)]
    pub sampler_label_ids: ByteVec,
    #[serde(default)]
    pub sampler_label_indptr: ByteVec,

    // === Adapter (per request) ===
    pub adapter_indices: Vec<Option<AdapterId>>,
    pub adapter_seeds: Vec<Option<i64>>,

    // === Speculative decoding (concatenated with indptr) ===
    pub spec_token_ids: ByteVec,
    pub spec_position_ids: ByteVec,
    pub spec_indptr: ByteVec,
    pub output_spec_flags: Vec<bool>,

    // === Pass-level speculation (per request) ===
    /// True iff the runtime wants the driver to build a speculative
    /// next-pass for this ctx after sampling. The driver is free to
    /// decline (e.g., if it would cross a KV-page boundary). See
    /// SPECULATIVE_EXECUTION_DESIGN.md §"Driver-runtime protocol".
    /// Default false until step 6d wires the predicate.
    #[serde(default)]
    pub predict_flags: Vec<bool>,

    // === Context (per request) ===
    // Stable per-context identifier — preferred session key for backends
    // that maintain per-context state (e.g. n-gram drafter token history).
    pub context_ids: Vec<ContextId>,

    // === Inference hints ===
    pub single_token_mode: bool,
    /// True iff at least one request in this batch supplied a user attention
    /// mask. Distinct from `flattened_masks.is_empty()` because the runtime
    /// synthesizes a causal-default mask for any inferlet that didn't provide
    /// one. Used internally by `single_token_mode` (a real user mask forces
    /// the prefill path) and exposed for future routing: drivers advertise
    /// `DriverCapabilities.supports_user_attention_mask`, and the runtime
    /// routes mask-dependent requests to a driver that returns True.
    pub has_user_mask: bool,

    // === Routing ===
    pub device_id: DeviceId,
}

impl BatchedForwardPassRequest {
    pub fn new(device_id: DeviceId) -> Self {
        Self {
            token_ids: ByteVec(Vec::new()),
            position_ids: ByteVec(Vec::new()),
            kv_page_indices: ByteVec(Vec::new()),
            kv_page_indptr: ByteVec(vec![0]),
            kv_last_page_lens: ByteVec(Vec::new()),
            qo_indptr: ByteVec(vec![0]),
            flattened_masks: ByteVec(Vec::new()),
            mask_indptr: ByteVec(vec![0]),
            logit_masks: ByteVec(Vec::new()),
            logit_mask_indptr: ByteVec(vec![0]),
            sampling_indices: ByteVec(Vec::new()),
            sampling_indptr: ByteVec(vec![0]),
            sampler_temperatures: ByteVecF32(Vec::new()),
            sampler_top_k: ByteVec(Vec::new()),
            sampler_top_p: ByteVecF32(Vec::new()),
            sampler_min_p: ByteVecF32(Vec::new()),
            sampler_types: ByteVec(Vec::new()),
            sampler_seeds: ByteVec(Vec::new()),
            request_num_samplers: ByteVec(Vec::new()),
            sampler_label_ids: ByteVec(Vec::new()),
            sampler_label_indptr: ByteVec(vec![0]),
            adapter_indices: Vec::new(),
            adapter_seeds: Vec::new(),
            spec_token_ids: ByteVec(Vec::new()),
            spec_position_ids: ByteVec(Vec::new()),
            spec_indptr: ByteVec(vec![0]),
            output_spec_flags: Vec::new(),
            predict_flags: Vec::new(),
            context_ids: Vec::new(),
            single_token_mode: true,
            has_user_mask: false,
            device_id,
        }
    }

    /// Append the request's physical page IDs to `kv_page_indices`,
    /// honoring the trim plan if present.
    fn emit_kv_pages(
        &mut self,
        physical_page_ids: &[PhysicalPageId],
        trim: Option<&TrimPlan>,
    ) {
        match trim {
            None => self.kv_page_indices.0.extend(physical_page_ids),
            Some(plan) => {
                for (idx, &pid) in physical_page_ids.iter().enumerate() {
                    if !plan.is_page_dropped(idx as u32) {
                        self.kv_page_indices.0.push(pid);
                    }
                }
            }
        }
    }

    /// Flatten one BRLE buffer per row into `flattened_masks`, applying the
    /// trim plan's skip ranges if present. Updates `mask_indptr` per row.
    fn emit_attention_masks(&mut self, masks: &[Brle], trim: Option<&TrimPlan>) {
        match trim {
            None => {
                for mask in masks {
                    self.flattened_masks.0.extend(&mask.buffer);
                    self.mask_indptr.0.push(self.flattened_masks.0.len() as u32);
                }
            }
            Some(plan) => {
                for mask in masks {
                    mask.write_skipping(&plan.skip_ranges, &mut self.flattened_masks.0);
                    self.mask_indptr.0.push(self.flattened_masks.0.len() as u32);
                }
            }
        }
    }

    /// Add a request to the batch.
    ///
    /// `page_size` is the model's KV page size in tokens. It's used by the
    /// page-trim optimization: when every row of the request's attention mask
    /// marks an entire page-sized range as False, that physical page is
    /// excluded from `kv_page_indices` and the corresponding bits are sliced
    /// out of every BRLE row before flattening. Pages that will receive new
    /// K/V writes this pass are protected from trimming.
    pub fn add_request(
        &mut self,
        req: &ForwardPassRequest,
        physical_page_ids: &[PhysicalPageId],
        last_page_len: u32,
        page_size: u32,
        predict: bool,
    ) {
        // Tokens and positions
        self.token_ids.0.extend(&req.tokens);
        self.position_ids.0.extend(&req.positions);

        // Compute the page-trim plan. Returns None for the common case where
        // no pages can be dropped (causal masks, decode steps, etc.) — the
        // fast path below uses zero allocations.
        let trim = TrimPlan::compute(
            &req.masks,
            physical_page_ids.len() as u32,
            last_page_len,
            page_size,
            req.tokens.len() as u32,
        );

        // KV cache layout (page list + indptr) and query/output indirection.
        self.emit_kv_pages(physical_page_ids, trim.as_ref());
        self.kv_page_indptr.0.push(self.kv_page_indices.0.len() as u32);
        self.kv_last_page_lens.0.push(last_page_len);
        self.qo_indptr.0.push(self.token_ids.0.len() as u32);

        // Attention masks (flatten BRLE), trimmed if any pages were dropped.
        self.emit_attention_masks(&req.masks, trim.as_ref());

        // Logit mask (flatten BRLE, per request)
        if let Some(ref mask) = req.logit_mask {
            self.logit_masks.0.extend(&mask.buffer);
        }
        self.logit_mask_indptr.0.push(self.logit_masks.0.len() as u32);

        // Sampling indices
        self.sampling_indices.0.extend(&req.sampling_indices);
        self.sampling_indptr.0.push(self.sampling_indices.0.len() as u32);

        // Samplers (SoA)
        self.request_num_samplers.0.push(req.samplers.len() as u32);
        for sampler in &req.samplers {
            self.sampler_types.0.push(sampler.type_id());
            self.sampler_temperatures.0.push(sampler.temperature());
            self.sampler_top_k.0.push(sampler.top_k());
            self.sampler_top_p.0.push(sampler.top_p());
            self.sampler_min_p.0.push(sampler.min_p());
            self.sampler_seeds.0.push(sampler.seed());
            // Logprob/Logprobs label payload (empty range for other types).
            match sampler {
                Sampler::Logprob { token_id } => {
                    self.sampler_label_ids.0.push(*token_id);
                }
                Sampler::Logprobs { token_ids } => {
                    self.sampler_label_ids.0.extend_from_slice(token_ids);
                }
                _ => {}
            }
            self.sampler_label_indptr.0.push(self.sampler_label_ids.0.len() as u32);
        }

        // Adapter
        self.adapter_indices.push(req.adapter_id);
        self.adapter_seeds.push(req.adapter_seed);

        // Speculative decoding
        self.spec_token_ids.0.extend(&req.speculative_tokens);
        self.spec_position_ids.0.extend(&req.speculative_positions);
        self.spec_indptr.0.push(self.spec_token_ids.0.len() as u32);
        self.output_spec_flags.push(req.output_speculative_tokens);

        // Pass-level speculation flag (per request).
        self.predict_flags.push(predict);

        // Context (stable id for per-context backend state)
        self.context_ids.push(req.context_id);

        // Inference hint: route to the prefill kernel (`single_token_mode = false`)
        // when ANY request needs `custom_mask` honored. Two cases qualify:
        //   - multi-token requests (the cuda-graph decode path doesn't accept
        //     more than one query token per request);
        //   - single-token requests with user-supplied masks (the decode kernel
        //     drops `custom_mask`, so honoring the mask requires prefill).
        // Synthesized causal masks do NOT trigger this — the decode kernel's
        // built-in causal already covers them.
        if req.tokens.len() > 1 || req.has_user_mask {
            self.single_token_mode = false;
        }
        if req.has_user_mask {
            self.has_user_mask = true;
        }
    }

    pub fn num_requests(&self) -> usize {
        self.adapter_indices.len()
    }

    pub fn total_tokens(&self) -> usize {
        self.token_ids.0.len()
    }
}

/// Batched forward pass response from Python.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchedForwardPassResponse {
    pub results: Vec<ForwardPassResponse>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytevec_u32_serde_roundtrip() {
        let original = ByteVec(vec![0, 1, u32::MAX, 42, 0xDEAD_BEEF]);
        let packed = rmp_serde::to_vec(&original).expect("serialize");
        let decoded: ByteVec = rmp_serde::from_slice(&packed).expect("deserialize");
        assert_eq!(original.0, decoded.0);
    }

    #[test]
    fn bytevec_f32_serde_roundtrip() {
        let original = ByteVecF32(vec![0.0, 1.0, -1.5, f32::INFINITY, f32::NAN]);
        let packed = rmp_serde::to_vec(&original).expect("serialize");
        let decoded: ByteVecF32 = rmp_serde::from_slice(&packed).expect("deserialize");
        // NaN != NaN, so compare element-by-element
        for (a, b) in original.0.iter().zip(decoded.0.iter()) {
            if a.is_nan() {
                assert!(b.is_nan(), "expected NaN, got {b}");
            } else {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn bytevec_empty_roundtrip() {
        let original = ByteVec(vec![]);
        let packed = rmp_serde::to_vec(&original).expect("serialize");
        let decoded: ByteVec = rmp_serde::from_slice(&packed).expect("deserialize");
        assert!(decoded.0.is_empty());
    }

    // -- Page-trim integration tests -----------------------------------------

    fn make_request(
        tokens: Vec<u32>,
        positions: Vec<u32>,
        masks: Vec<Brle>,
    ) -> ForwardPassRequest {
        let has_user_mask = !masks.is_empty();
        ForwardPassRequest {
            context_id: 0,
            tokens,
            positions,
            speculative_tokens: vec![],
            speculative_positions: vec![],
            output_speculative_tokens: false,
            masks,
            has_user_mask,
            logit_mask: None,
            sampling_indices: vec![],
            samplers: vec![],
            adapter_id: None,
            adapter_seed: None,
        }
    }

    #[test]
    fn add_request_causal_decode_no_trim() {
        // Single-token decode at position 47, page_size=16, num_pages=3,
        // last_page_len=16. Causal mask (all-true [0,48]) → no false runs,
        // no pages can be dropped. Wire format must match the pre-optimization
        // layout exactly: all 3 pages present, mask buffer untouched.
        let causal = Brle::all_true(48);
        let req = make_request(vec![999], vec![47], vec![causal.clone()]);
        let pages: Vec<PhysicalPageId> = vec![100, 101, 102];

        let mut batch = BatchedForwardPassRequest::new(0);
        batch.add_request(&req, &pages, 16, 16, false);

        assert_eq!(batch.kv_page_indices.0, vec![100, 101, 102]);
        assert_eq!(batch.kv_page_indptr.0, vec![0, 3]);
        assert_eq!(batch.kv_last_page_lens.0, vec![16]);
        // Mask buffer is the original BRLE (no rewrite).
        assert_eq!(batch.flattened_masks.0, causal.buffer);
    }

    #[test]
    fn add_request_attention_sink_trims_middle_pages() {
        // Decode at position 319 with sink+window mask: sink=4, gap=252,
        // window=64, total seq_len=320. page_size=16, num_pages=20,
        // last_page_len=16. The single new token writes to page 19 (the last
        // page), so first_writeable_page = 319/16 = 19 → eligible window is
        // pages 0..=18.
        //
        // Per-row droppable: false run [4, 256) covers pages 1..=15 fully.
        // After AND with eligibility window {0..=18}: pages 1..=15 dropped.
        let mask = Brle::from_vec(vec![0, 4, 252, 64]); // sink+window
        assert_eq!(mask.len(), 320);

        let req = make_request(vec![999], vec![319], vec![mask]);
        let pages: Vec<PhysicalPageId> =
            (0..20).map(|i| 1000 + i as PhysicalPageId).collect();

        let mut batch = BatchedForwardPassRequest::new(0);
        batch.add_request(&req, &pages, 16, 16, false);

        // Surviving pages: 0, 16, 17, 18, 19 (5 pages).
        let expected_pages: Vec<u32> = vec![1000, 1016, 1017, 1018, 1019];
        assert_eq!(batch.kv_page_indices.0, expected_pages);
        assert_eq!(batch.kv_page_indptr.0, vec![0, 5]);
        // last_page_len unchanged — last page is never dropped.
        assert_eq!(batch.kv_last_page_lens.0, vec![16]);

        // Trimmed BRLE: original false run [4, 256) shrinks by 15*16 = 240
        // bits (15 dropped pages). Layout becomes:
        //   sink(4) | gap'(12) | window(64)
        // i.e., BRLE buffer = [0, 4, 12, 64], total_size = 80 = 5*16.
        assert_eq!(batch.flattened_masks.0, vec![0, 4, 12, 64]);
        assert_eq!(batch.mask_indptr.0, vec![0, 4]);
    }

    #[test]
    fn add_request_window_only_trims_leading_pages() {
        // Sliding-window mask: gap=240 (false), window=80 (true), seq_len=320.
        // page_size=16, num_pages=20, last_page_len=16. Decode at position 319.
        //   eligible window: pages 0..=18 (writeable = page 19).
        //   row droppable: pages 0..=14 (false run [0, 240) covers them fully).
        // Drop pages 0..=14 (15 pages); pages 15..=19 remain.
        let mask = Brle::from_vec(vec![240, 80]);
        assert_eq!(mask.len(), 320);

        let req = make_request(vec![999], vec![319], vec![mask]);
        let pages: Vec<PhysicalPageId> =
            (0..20).map(|i| 2000 + i as PhysicalPageId).collect();

        let mut batch = BatchedForwardPassRequest::new(0);
        batch.add_request(&req, &pages, 16, 16, false);

        let expected_pages: Vec<u32> = (15..20).map(|i| 2000 + i).collect();
        assert_eq!(batch.kv_page_indices.0, expected_pages);
        // After dropping 15 leading false pages, remaining mask is:
        //   false: 240 - 15*16 = 0  →  zero-length false prefix preserved
        //   true: 80
        // Buffer: [0, 80], total_size = 80.
        assert_eq!(batch.flattened_masks.0, vec![0, 80]);
    }

    #[test]
    fn add_request_writeable_pages_are_protected() {
        // Pathological: a request whose mask is all-False, but kv_before is
        // entirely contained in a single non-final page. The writeable-window
        // guard must protect that page even though the mask agrees it's
        // droppable.
        //
        // page_size=16, kv_before=10 (one partial page), tokens.len()=6
        // (filling the page). num_pages=1, last_page_len=16, total_kv=16.
        // first_writeable_page = 10/16 = 0 → no eligible pages.
        let mask = Brle::from_vec(vec![16]); // all false, total 16
        let req = make_request(vec![1, 2, 3, 4, 5, 6], vec![10, 11, 12, 13, 14, 15], vec![mask.clone()]);
        let pages: Vec<PhysicalPageId> = vec![777];

        let mut batch = BatchedForwardPassRequest::new(0);
        batch.add_request(&req, &pages, 16, 16, false);

        // No pages dropped; original mask buffer preserved.
        assert_eq!(batch.kv_page_indices.0, vec![777]);
        assert_eq!(batch.flattened_masks.0, mask.buffer);
    }

    #[test]
    fn add_request_rows_disagree_no_drops() {
        // Two-token prefill, page_size=16, num_pages=4 (seq_len=64),
        // last_page_len=16, kv_before=62 → first_writeable_page=3, eligible
        // window {0,1,2}. Two rows whose droppable sets are disjoint within
        // that window:
        //   row 0: false [0, 32)  → pages 0,1 droppable
        //   row 1: false [32, 48) → page 2 droppable
        // AND-reduction collapses to ∅, so the trim path bails. Verify the
        // wire format is byte-identical to the no-trim layout.
        let row0 = Brle::from_vec(vec![32, 32]); // false 32, true 32
        let row1 = Brle::from_vec(vec![0, 32, 16, 16]); // true 32, false 16, true 16
        assert_eq!(row0.len(), 64);
        assert_eq!(row1.len(), 64);

        let req = make_request(vec![1, 2], vec![62, 63], vec![row0.clone(), row1.clone()]);
        let pages: Vec<PhysicalPageId> = vec![10, 11, 12, 13];

        let mut batch = BatchedForwardPassRequest::new(0);
        batch.add_request(&req, &pages, 16, 16, false);

        // Fast path: original pages and mask buffers byte-for-byte.
        assert_eq!(batch.kv_page_indices.0, vec![10, 11, 12, 13]);
        let mut expected_masks = row0.buffer.clone();
        expected_masks.extend(&row1.buffer);
        assert_eq!(batch.flattened_masks.0, expected_masks);
        assert_eq!(batch.mask_indptr.0, vec![0, 2, 6]);
    }

    #[test]
    fn add_request_multi_row_identical_sink_pattern() {
        // Prefill with multiple input tokens, every row has the same
        // sink+window mask (a common inferlet pattern). Verify that the trim
        // applies uniformly across rows and the per-row mask offsets in
        // `mask_indptr` track the trimmed buffer correctly.
        let mask = Brle::from_vec(vec![0, 4, 252, 64]); // seq_len 320
        let req = make_request(
            vec![10, 20, 30],
            vec![317, 318, 319],
            vec![mask.clone(), mask.clone(), mask.clone()],
        );
        let pages: Vec<PhysicalPageId> =
            (0..20).map(|i| 5000 + i as PhysicalPageId).collect();

        let mut batch = BatchedForwardPassRequest::new(0);
        // Three new tokens at positions 317..319, kv_before=317 →
        // first_writeable_page=19. Pages 1..=15 still dropped by the mask.
        batch.add_request(&req, &pages, 16, 16, false);

        let expected_pages: Vec<u32> = vec![5000, 5016, 5017, 5018, 5019];
        assert_eq!(batch.kv_page_indices.0, expected_pages);

        // Three identical rows trimmed identically: each row's BRLE shrinks
        // to [0, 4, 12, 64] (4 entries). Total flattened length = 12.
        let trimmed_row: Vec<u32> = vec![0, 4, 12, 64];
        let mut expected_flat: Vec<u32> = Vec::new();
        for _ in 0..3 {
            expected_flat.extend_from_slice(&trimmed_row);
        }
        assert_eq!(batch.flattened_masks.0, expected_flat);
        assert_eq!(batch.mask_indptr.0, vec![0, 4, 8, 12]);
    }
}
