//! Minimal chunked prefill support for a single oversized pending request.
//!
//! This module intentionally does not water-fill normal batches or split
//! already-scheduled requests. It only converts one request that exceeds the
//! driver's `max_forward_tokens` into sequential prefix chunks, then merges the
//! per-chunk sampler outputs back into the original single-request response.

use std::collections::BTreeMap;

use anyhow::Result;
use tokio::sync::{mpsc, oneshot};

use crate::context::pagestore::{PhysicalPageId, compute_last_page_len};
use crate::driver::SchedulerLimits;
use crate::inference::ForwardOutput;

use super::{Completion, PendingRequest, RequestCapacityUsage, request_capacity_usage};

pub(super) struct ChunkContinuation {
    original_request: pie_bridge::ForwardRequest,
    response_tx: oneshot::Sender<Result<ForwardOutput>>,
    physical_page_ids: Vec<PhysicalPageId>,
    final_last_page_len: u32,
    chunk_end: usize,
    chunk_size: usize,
    sampler_slots_by_chunk: BTreeMap<usize, Vec<usize>>,
    response_accumulator: ChunkResponseAccumulator,
}

struct ChunkResponseAccumulator {
    slots: Vec<Option<ChunkSlotOutput>>,
}

enum ChunkSlotOutput {
    Token(u32),
    Dist { ids: Vec<u32>, probs: Vec<f32> },
    Logits(Vec<u8>),
    Logprobs(Vec<f32>),
    Entropy(f32),
}

struct BuiltChunk {
    request: pie_bridge::ForwardRequest,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
}

impl PendingRequest {
    pub(super) fn maybe_start_chunking(
        self,
        limits: SchedulerLimits,
        page_size: u32,
    ) -> std::result::Result<Self, (Self, String)> {
        if matches!(&self.completion, Completion::Chunk { .. }) {
            return Ok(self);
        }

        let usage = request_capacity_usage(&self, page_size);
        if usage.forward_tokens <= limits.max_forward_tokens {
            return Ok(self);
        }

        if let Err(msg) = validate_chunkable_request(&self.request, limits.max_forward_tokens) {
            return Err((self, msg));
        }

        let chunk_size = limits.max_forward_tokens;
        let chunk_end = chunk_size.min(self.request.token_ids.len());
        let sampler_slots_by_chunk = chunk_sampler_slots_by_chunk(&self.request, chunk_size);
        if let Err(msg) = validate_chunk_capacity(
            &self,
            limits,
            page_size,
            chunk_size,
            &sampler_slots_by_chunk,
        ) {
            return Err((self, msg));
        }
        let chunk_sampler_slots = sampler_slots_by_chunk.get(&0).cloned().unwrap_or_default();
        let chunk = match build_chunk_request_for_slots(
            &self.request,
            &self.physical_page_ids,
            self.last_page_len,
            0,
            chunk_end,
            page_size,
            &chunk_sampler_slots,
        ) {
            Ok(chunk) => chunk,
            Err(msg) => return Err((self, msg)),
        };

        let PendingRequest {
            request,
            completion,
            physical_page_ids,
            last_page_len,
        } = self;
        let Completion::Direct(response_tx) = completion else {
            unreachable!("chunk continuations returned above");
        };
        let response_accumulator = ChunkResponseAccumulator::new(request.samplers.len());

        Ok(PendingRequest {
            request: chunk.request,
            completion: Completion::Chunk {
                continuation: ChunkContinuation {
                    original_request: request,
                    response_tx,
                    physical_page_ids,
                    final_last_page_len: last_page_len,
                    chunk_end,
                    chunk_size,
                    sampler_slots_by_chunk,
                    response_accumulator,
                },
                sampler_slots: chunk_sampler_slots,
            },
            physical_page_ids: chunk.physical_page_ids,
            last_page_len: chunk.last_page_len,
        })
    }

    pub(super) fn send_error(self, msg: String) {
        self.send_result::<ForwardOutput>(Err(anyhow::anyhow!(msg)), None, 0);
    }

    pub(super) fn send_result<T>(
        self,
        result: Result<T>,
        submit_tx: Option<&crossbeam::channel::Sender<PendingRequest>>,
        page_size: u32,
    ) where
        T: Into<ForwardOutput>,
    {
        let PendingRequest {
            request,
            completion,
            physical_page_ids: _,
            last_page_len: _,
        } = self;

        let result = result.map(Into::into);
        match completion {
            Completion::Direct(tx) => {
                tx.send(result).ok();
            }
            Completion::Chunk {
                continuation,
                sampler_slots,
            } => match result {
                Ok(ForwardOutput::Response(resp)) => continuation.complete_chunk(
                    resp,
                    request.samplers,
                    sampler_slots,
                    submit_tx,
                    page_size,
                ),
                Ok(_) => {
                    continuation
                        .response_tx
                        .send(Err(anyhow::anyhow!(
                            "chunked prefill expected a structured response"
                        )))
                        .ok();
                }
                Err(e) => {
                    continuation.response_tx.send(Err(e)).ok();
                }
            },
            Completion::Chain { state } => {
                // Error path: the chain never had a healthy reply, so just
                // forward the error to the inferlet / staged-entry holder
                // and let the chain terminate (no next stage submitted).
                let _ = state.response.send(result);
            }
        }
    }
}

impl ChunkContinuation {
    fn complete_chunk(
        mut self,
        resp: pie_bridge::ForwardResponse,
        chunk_samplers: Vec<pie_bridge::Sampler>,
        chunk_sampler_slots: Vec<usize>,
        submit_tx: Option<&crossbeam::channel::Sender<PendingRequest>>,
        page_size: u32,
    ) {
        if let Err(msg) =
            self.response_accumulator
                .record_response(&chunk_samplers, &chunk_sampler_slots, resp)
        {
            self.response_tx.send(Err(anyhow::anyhow!(msg))).ok();
            return;
        }

        if self.chunk_end >= self.original_request.token_ids.len() {
            let ChunkContinuation {
                original_request,
                response_tx,
                response_accumulator,
                ..
            } = self;
            match response_accumulator.into_response(&original_request.samplers) {
                Ok(resp) => {
                    response_tx.send(Ok(ForwardOutput::Response(resp))).ok();
                }
                Err(msg) => {
                    response_tx.send(Err(anyhow::anyhow!(msg))).ok();
                }
            }
            return;
        }

        let Some(submit_tx) = submit_tx else {
            self.response_tx
                .send(Err(anyhow::anyhow!(
                    "chunked prefill continuation could not be requeued: scheduler shutting down"
                )))
                .ok();
            return;
        };

        match self.into_next_pending(page_size) {
            Ok(next) => {
                if let Err(err) = submit_tx.send(next) {
                    err.0.send_error(
                        "chunked prefill continuation could not be requeued: scheduler channel closed"
                            .to_string(),
                    );
                }
            }
            Err((cont, msg)) => {
                cont.response_tx.send(Err(anyhow::anyhow!(msg))).ok();
            }
        }
    }

    fn into_next_pending(
        self,
        page_size: u32,
    ) -> std::result::Result<PendingRequest, (Self, String)> {
        let start = self.chunk_end;
        let end = start
            .saturating_add(self.chunk_size)
            .min(self.original_request.token_ids.len());
        let chunk_idx = start / self.chunk_size;
        let chunk_sampler_slots = self
            .sampler_slots_by_chunk
            .get(&chunk_idx)
            .cloned()
            .unwrap_or_default();
        let chunk = match build_chunk_request_for_slots(
            &self.original_request,
            &self.physical_page_ids,
            self.final_last_page_len,
            start,
            end,
            page_size,
            &chunk_sampler_slots,
        ) {
            Ok(chunk) => chunk,
            Err(msg) => return Err((self, msg)),
        };

        Ok(PendingRequest {
            request: chunk.request,
            completion: Completion::Chunk {
                continuation: ChunkContinuation {
                    chunk_end: end,
                    ..self
                },
                sampler_slots: chunk_sampler_slots,
            },
            physical_page_ids: chunk.physical_page_ids,
            last_page_len: chunk.last_page_len,
        })
    }
}

impl ChunkResponseAccumulator {
    fn new(num_slots: usize) -> Self {
        Self {
            slots: (0..num_slots).map(|_| None).collect(),
        }
    }

    fn record_response(
        &mut self,
        samplers: &[pie_bridge::Sampler],
        sampler_slots: &[usize],
        resp: pie_bridge::ForwardResponse,
    ) -> std::result::Result<(), String> {
        use pie_bridge::Sampler;

        if samplers.len() != sampler_slots.len() {
            return Err(format!(
                "chunked prefill sampler slot map mismatch: {} samplers, {} slots",
                samplers.len(),
                sampler_slots.len()
            ));
        }

        let mut tokens = response_tokens(&resp)?.into_iter();
        let mut dists = response_dists(&resp)?.into_iter();
        let mut logits = response_logits(&resp)?.into_iter();
        let mut logprobs = response_logprobs(&resp)?.into_iter();
        let mut entropies = response_entropies(&resp)?.into_iter();

        for (sampler, &slot) in samplers.iter().zip(sampler_slots.iter()) {
            match sampler {
                Sampler::Multinomial { .. }
                | Sampler::TopK { .. }
                | Sampler::TopP { .. }
                | Sampler::MinP { .. }
                | Sampler::TopKTopP { .. } => {
                    let token = tokens.next().ok_or_else(|| {
                        format!("chunked prefill missing token output for sampler slot {slot}")
                    })?;
                    self.store(slot, ChunkSlotOutput::Token(token))?;
                }
                Sampler::Dist { .. } => {
                    let (ids, probs) = dists.next().ok_or_else(|| {
                        format!(
                            "chunked prefill missing distribution output for sampler slot {slot}"
                        )
                    })?;
                    self.store(slot, ChunkSlotOutput::Dist { ids, probs })?;
                }
                Sampler::RawLogits => {
                    let bytes = logits.next().ok_or_else(|| {
                        format!("chunked prefill missing logits output for sampler slot {slot}")
                    })?;
                    self.store(slot, ChunkSlotOutput::Logits(bytes))?;
                }
                Sampler::Logprob { .. } | Sampler::Logprobs { .. } => {
                    let values = logprobs.next().ok_or_else(|| {
                        format!("chunked prefill missing logprobs output for sampler slot {slot}")
                    })?;
                    self.store(slot, ChunkSlotOutput::Logprobs(values))?;
                }
                Sampler::Entropy => {
                    let entropy = entropies.next().ok_or_else(|| {
                        format!("chunked prefill missing entropy output for sampler slot {slot}")
                    })?;
                    self.store(slot, ChunkSlotOutput::Entropy(entropy))?;
                }
                Sampler::Embedding => {}
            }
        }

        if tokens.len() > 0 {
            return Err(format!(
                "chunked prefill received {} extra token outputs",
                tokens.len()
            ));
        }
        if dists.len() > 0 {
            return Err(format!(
                "chunked prefill received {} extra distribution outputs",
                dists.len()
            ));
        }
        if logits.len() > 0 {
            return Err(format!(
                "chunked prefill received {} extra logits outputs",
                logits.len()
            ));
        }
        if logprobs.len() > 0 {
            return Err(format!(
                "chunked prefill received {} extra logprobs outputs",
                logprobs.len()
            ));
        }
        if entropies.len() > 0 {
            return Err(format!(
                "chunked prefill received {} extra entropy outputs",
                entropies.len()
            ));
        }

        Ok(())
    }

    fn store(&mut self, slot: usize, output: ChunkSlotOutput) -> std::result::Result<(), String> {
        let Some(existing) = self.slots.get_mut(slot) else {
            return Err(format!(
                "chunked prefill sampler slot {slot} is outside {} original samplers",
                self.slots.len()
            ));
        };
        if existing.is_some() {
            return Err(format!(
                "chunked prefill received duplicate output for sampler slot {slot}"
            ));
        }
        *existing = Some(output);
        Ok(())
    }

    fn into_response(
        mut self,
        samplers: &[pie_bridge::Sampler],
    ) -> std::result::Result<pie_bridge::ForwardResponse, String> {
        use pie_bridge::Sampler;

        let mut out = pie_bridge::ForwardResponse {
            num_requests: 1,
            tokens_indptr: vec![0],
            dists_req_indptr: vec![0],
            dists_kv_indptr: vec![0],
            logits_req_indptr: vec![0],
            logits_byte_indptr: vec![0],
            logprobs_req_indptr: vec![0],
            logprobs_val_indptr: vec![0],
            entropies_indptr: vec![0],
            ..Default::default()
        };

        for (slot, sampler) in samplers.iter().enumerate() {
            match sampler {
                Sampler::Multinomial { .. }
                | Sampler::TopK { .. }
                | Sampler::TopP { .. }
                | Sampler::MinP { .. }
                | Sampler::TopKTopP { .. } => match self.take(slot)? {
                    ChunkSlotOutput::Token(token) => out.tokens.push(token),
                    other => return Err(slot_type_error(slot, "token", &other)),
                },
                Sampler::Dist { .. } => match self.take(slot)? {
                    ChunkSlotOutput::Dist { ids, probs } => {
                        out.dists_ids.extend(ids);
                        out.dists_probs.extend(probs);
                        out.dists_kv_indptr.push(out.dists_ids.len() as u32);
                    }
                    other => return Err(slot_type_error(slot, "distribution", &other)),
                },
                Sampler::RawLogits => match self.take(slot)? {
                    ChunkSlotOutput::Logits(bytes) => {
                        out.logits_bytes.extend(bytes);
                        out.logits_byte_indptr.push(out.logits_bytes.len() as u32);
                    }
                    other => return Err(slot_type_error(slot, "logits", &other)),
                },
                Sampler::Logprob { .. } | Sampler::Logprobs { .. } => match self.take(slot)? {
                    ChunkSlotOutput::Logprobs(values) => {
                        out.logprobs_values.extend(values);
                        out.logprobs_val_indptr
                            .push(out.logprobs_values.len() as u32);
                    }
                    other => return Err(slot_type_error(slot, "logprobs", &other)),
                },
                Sampler::Entropy => match self.take(slot)? {
                    ChunkSlotOutput::Entropy(entropy) => out.entropies.push(entropy),
                    other => return Err(slot_type_error(slot, "entropy", &other)),
                },
                Sampler::Embedding => {}
            }
        }

        out.tokens_indptr.push(out.tokens.len() as u32);
        out.dists_req_indptr
            .push((out.dists_kv_indptr.len() - 1) as u32);
        out.logits_req_indptr
            .push((out.logits_byte_indptr.len() - 1) as u32);
        out.logprobs_req_indptr
            .push((out.logprobs_val_indptr.len() - 1) as u32);
        out.entropies_indptr.push(out.entropies.len() as u32);

        Ok(out)
    }

    fn take(&mut self, slot: usize) -> std::result::Result<ChunkSlotOutput, String> {
        self.slots
            .get_mut(slot)
            .and_then(Option::take)
            .ok_or_else(|| format!("chunked prefill missing output for sampler slot {slot}"))
    }
}

fn slot_type_error(slot: usize, expected: &str, got: &ChunkSlotOutput) -> String {
    format!(
        "chunked prefill expected {expected} output for sampler slot {slot}, got {}",
        slot_output_name(got)
    )
}

fn slot_output_name(output: &ChunkSlotOutput) -> &'static str {
    match output {
        ChunkSlotOutput::Token(_) => "token",
        ChunkSlotOutput::Dist { .. } => "distribution",
        ChunkSlotOutput::Logits(_) => "logits",
        ChunkSlotOutput::Logprobs(_) => "logprobs",
        ChunkSlotOutput::Entropy(_) => "entropy",
    }
}

fn response_tokens(resp: &pie_bridge::ForwardResponse) -> std::result::Result<Vec<u32>, String> {
    if resp.tokens_indptr.len() >= 2 {
        if resp.tokens_indptr[0] != 0 {
            return Err(format!(
                "chunked prefill token response starts at {}, expected 0",
                resp.tokens_indptr[0]
            ));
        }
        let end = resp.tokens_indptr[1] as usize;
        if end > resp.tokens.len() {
            return Err(format!(
                "chunked prefill token response range ends at {end}, but only {} tokens returned",
                resp.tokens.len()
            ));
        }
        if end != resp.tokens.len() {
            return Err(format!(
                "chunked prefill token response has {} trailing tokens outside request range",
                resp.tokens.len() - end
            ));
        }
    }
    Ok(resp.tokens.clone())
}

fn response_dists(
    resp: &pie_bridge::ForwardResponse,
) -> std::result::Result<Vec<(Vec<u32>, Vec<f32>)>, String> {
    if resp.dists_ids.len() != resp.dists_probs.len() {
        return Err(format!(
            "chunked prefill distribution response has {} ids and {} probabilities",
            resp.dists_ids.len(),
            resp.dists_probs.len()
        ));
    }
    let count = response_nested_slot_count(
        &resp.dists_req_indptr,
        &resp.dists_kv_indptr,
        "distribution",
    )?;
    let mut out = Vec::with_capacity(count);
    for k in 0..count {
        let lo = resp.dists_kv_indptr[k] as usize;
        let hi = resp.dists_kv_indptr[k + 1] as usize;
        if hi < lo || hi > resp.dists_ids.len() {
            return Err(format!(
                "chunked prefill distribution response has invalid range {lo}..{hi}"
            ));
        }
        out.push((
            resp.dists_ids[lo..hi].to_vec(),
            resp.dists_probs[lo..hi].to_vec(),
        ));
    }
    Ok(out)
}

fn response_logits(
    resp: &pie_bridge::ForwardResponse,
) -> std::result::Result<Vec<Vec<u8>>, String> {
    let count =
        response_nested_slot_count(&resp.logits_req_indptr, &resp.logits_byte_indptr, "logits")?;
    let mut out = Vec::with_capacity(count);
    for b in 0..count {
        let lo = resp.logits_byte_indptr[b] as usize;
        let hi = resp.logits_byte_indptr[b + 1] as usize;
        if hi < lo || hi > resp.logits_bytes.len() {
            return Err(format!(
                "chunked prefill logits response has invalid range {lo}..{hi}"
            ));
        }
        out.push(resp.logits_bytes[lo..hi].to_vec());
    }
    Ok(out)
}

fn response_logprobs(
    resp: &pie_bridge::ForwardResponse,
) -> std::result::Result<Vec<Vec<f32>>, String> {
    let count = response_nested_slot_count(
        &resp.logprobs_req_indptr,
        &resp.logprobs_val_indptr,
        "logprobs",
    )?;
    let mut out = Vec::with_capacity(count);
    for s in 0..count {
        let lo = resp.logprobs_val_indptr[s] as usize;
        let hi = resp.logprobs_val_indptr[s + 1] as usize;
        if hi < lo || hi > resp.logprobs_values.len() {
            return Err(format!(
                "chunked prefill logprobs response has invalid range {lo}..{hi}"
            ));
        }
        out.push(resp.logprobs_values[lo..hi].to_vec());
    }
    Ok(out)
}

fn response_entropies(resp: &pie_bridge::ForwardResponse) -> std::result::Result<Vec<f32>, String> {
    if resp.entropies_indptr.len() >= 2 {
        if resp.entropies_indptr[0] != 0 {
            return Err(format!(
                "chunked prefill entropy response starts at {}, expected 0",
                resp.entropies_indptr[0]
            ));
        }
        let end = resp.entropies_indptr[1] as usize;
        if end > resp.entropies.len() {
            return Err(format!(
                "chunked prefill entropy response range ends at {end}, but only {} entropies returned",
                resp.entropies.len()
            ));
        }
        if end != resp.entropies.len() {
            return Err(format!(
                "chunked prefill entropy response has {} trailing values outside request range",
                resp.entropies.len() - end
            ));
        }
    }
    Ok(resp.entropies.clone())
}

fn response_nested_slot_count(
    req_indptr: &[u32],
    slot_indptr: &[u32],
    label: &str,
) -> std::result::Result<usize, String> {
    let slots_from_indptr = slot_indptr.len().saturating_sub(1);
    if req_indptr.len() < 2 {
        return Ok(slots_from_indptr);
    }
    if req_indptr[0] != 0 {
        return Err(format!(
            "chunked prefill {label} response starts at {}, expected 0",
            req_indptr[0]
        ));
    }
    let count = req_indptr[1] as usize;
    if count != slots_from_indptr {
        return Err(format!(
            "chunked prefill {label} response has request count {count}, but {} slot ranges",
            slots_from_indptr
        ));
    }
    Ok(count)
}

fn validate_chunkable_request(
    req: &pie_bridge::ForwardRequest,
    max_forward_tokens: usize,
) -> std::result::Result<(), String> {
    if max_forward_tokens == 0 {
        return Err("driver max forward tokens is zero".to_string());
    }
    if !req.spec_token_ids.is_empty() {
        return Err(format!(
            "forward request has {} input tokens and {} speculative tokens, exceeding driver limit {}; chunked prefill does not yet support speculative drafts",
            req.token_ids.len(),
            req.spec_token_ids.len(),
            max_forward_tokens
        ));
    }
    validate_chunk_request_shape(req)?;

    let total = req.token_ids.len();
    if total <= max_forward_tokens {
        return Ok(());
    }

    for &idx in &req.sampling_indices {
        let idx = idx as usize;
        if idx >= total {
            return Err(format!(
                "chunked prefill sampler index {idx} is outside the input window of {total} tokens"
            ));
        }
    }

    Ok(())
}

fn validate_chunk_request_shape(
    req: &pie_bridge::ForwardRequest,
) -> std::result::Result<(), String> {
    if req.position_ids.len() != req.token_ids.len() {
        return Err(format!(
            "chunked prefill requires one position per token (got {} tokens, {} positions)",
            req.token_ids.len(),
            req.position_ids.len()
        ));
    }
    if !req.masks.is_empty() && req.masks.len() != req.token_ids.len() {
        return Err(format!(
            "chunked prefill requires either zero masks or one mask per token (got {} tokens, {} masks)",
            req.token_ids.len(),
            req.masks.len()
        ));
    }
    if req.sampling_indices.len() != req.samplers.len() {
        return Err(format!(
            "chunked prefill requires one sampler per sampling index (got {} indices, {} samplers)",
            req.sampling_indices.len(),
            req.samplers.len()
        ));
    }
    Ok(())
}

fn validate_chunk_capacity(
    pending: &PendingRequest,
    limits: SchedulerLimits,
    page_size: u32,
    chunk_size: usize,
    sampler_slots_by_chunk: &BTreeMap<usize, Vec<usize>>,
) -> std::result::Result<(), String> {
    if limits.max_forward_requests == 0 {
        return Err("driver max forward requests is zero".to_string());
    }

    let total = pending.request.token_ids.len();
    for start in (0..total).step_by(chunk_size) {
        let end = start.saturating_add(chunk_size).min(total);
        let chunk_idx = start / chunk_size;
        let sampler_slots = sampler_slots_by_chunk
            .get(&chunk_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let usage = chunk_capacity_usage(
            &pending.request,
            &pending.physical_page_ids,
            pending.last_page_len,
            start,
            end,
            page_size,
            sampler_slots,
        )?;
        if let Some(msg) = chunk_limit_error(usage, limits) {
            return Err(msg);
        }
    }
    Ok(())
}

fn chunk_limit_error(usage: RequestCapacityUsage, limits: SchedulerLimits) -> Option<String> {
    if usage.forward_tokens > limits.max_forward_tokens {
        return Some(format!(
            "forward request chunk has {} forward tokens, exceeding driver limit {}",
            usage.forward_tokens, limits.max_forward_tokens
        ));
    }

    if usage.page_refs > limits.max_page_refs {
        return Some(format!(
            "forward request chunk has {} page refs, exceeding driver limit {}",
            usage.page_refs, limits.max_page_refs
        ));
    }

    if usage.sampler_rows > limits.max_sampler_rows {
        return Some(format!(
            "forward request chunk has {} sampler rows, exceeding driver limit {}",
            usage.sampler_rows, limits.max_sampler_rows
        ));
    }

    if usage.logprob_labels > limits.max_logprob_labels {
        return Some(format!(
            "forward request chunk has {} logprob labels, exceeding driver limit {}",
            usage.logprob_labels, limits.max_logprob_labels
        ));
    }

    let custom_mask_bytes = if usage.has_spec_drafts {
        usage.spec_custom_mask_bytes
    } else {
        usage.user_custom_mask_bytes
    };
    if custom_mask_bytes > limits.max_custom_mask_bytes {
        return Some(format!(
            "forward request chunk needs {custom_mask_bytes} custom mask bytes, exceeding driver limit {}",
            limits.max_custom_mask_bytes
        ));
    }

    None
}

#[cfg(test)]
fn build_chunk_request(
    original: &pie_bridge::ForwardRequest,
    full_physical_page_ids: &[PhysicalPageId],
    final_last_page_len: u32,
    start: usize,
    end: usize,
    page_size: u32,
) -> std::result::Result<
    (
        pie_bridge::ForwardRequest,
        Vec<PhysicalPageId>,
        u32,
        Vec<usize>,
    ),
    String,
> {
    let sampler_slots = collect_chunk_sampler_slots(original, start, end);
    let chunk = build_chunk_request_for_slots(
        original,
        full_physical_page_ids,
        final_last_page_len,
        start,
        end,
        page_size,
        &sampler_slots,
    )?;
    Ok((
        chunk.request,
        chunk.physical_page_ids,
        chunk.last_page_len,
        sampler_slots,
    ))
}

fn build_chunk_request_for_slots(
    original: &pie_bridge::ForwardRequest,
    full_physical_page_ids: &[PhysicalPageId],
    final_last_page_len: u32,
    start: usize,
    end: usize,
    page_size: u32,
    sampler_slots: &[usize],
) -> std::result::Result<BuiltChunk, String> {
    if page_size == 0 {
        return Err("chunked prefill requires non-zero page size".to_string());
    }
    if start >= end || end > original.token_ids.len() {
        return Err(format!(
            "invalid chunk range {start}..{end} for {} tokens",
            original.token_ids.len()
        ));
    }
    validate_chunk_request_shape(original)?;

    let (chunk_pages, chunk_last_page_len) = chunk_page_shape(
        original,
        full_physical_page_ids,
        final_last_page_len,
        end,
        page_size,
    )?;

    let mut sampling_indices = Vec::new();
    let mut samplers = Vec::new();
    for &slot in sampler_slots {
        let Some(&idx) = original.sampling_indices.get(slot) else {
            return Err(format!(
                "chunked prefill sampler slot {slot} is outside {} sampling indices",
                original.sampling_indices.len()
            ));
        };
        let Some(sampler) = original.samplers.get(slot) else {
            return Err(format!(
                "chunked prefill sampler slot {slot} is outside {} samplers",
                original.samplers.len()
            ));
        };
        let idx_usize = idx as usize;
        if (start..end).contains(&idx_usize) {
            sampling_indices.push((idx_usize - start) as u32);
            samplers.push(sampler.clone());
        } else {
            return Err(format!(
                "chunked prefill sampler slot {slot} index {idx_usize} is outside chunk range {start}..{end}"
            ));
        }
    }

    let chunk_len = end - start;
    let (masks, mask_indptr) = if original.masks.is_empty() {
        (Vec::new(), vec![0, 0])
    } else {
        (
            original.masks[start..end].to_vec(),
            vec![0, chunk_len as u32],
        )
    };
    let (logit_masks, logit_mask_indptr) = if samplers.is_empty() {
        (Vec::new(), vec![0, 0])
    } else {
        (
            original.logit_masks.clone(),
            vec![0, original.logit_masks.len() as u32],
        )
    };
    let mut output_spec_flags = if end == original.token_ids.len() {
        original.output_spec_flags.clone()
    } else {
        vec![false]
    };
    if output_spec_flags.is_empty() {
        output_spec_flags.push(false);
    }
    let rs_slot_flags = if start == 0 {
        original.rs_slot_flags.clone()
    } else {
        vec![0; original.rs_slot_flags.len()]
    };
    let sampling_len = sampling_indices.len() as u32;
    let sampler_len = samplers.len() as u32;
    let chunk = pie_bridge::ForwardRequest {
        token_ids: original.token_ids[start..end].to_vec(),
        position_ids: original.position_ids[start..end].to_vec(),
        kv_page_indices: Vec::new(),
        kv_page_indptr: vec![0],
        kv_last_page_lens: Vec::new(),
        qo_indptr: vec![0, chunk_len as u32],
        rs_slot_ids: original.rs_slot_ids.clone(),
        rs_slot_flags,
        masks,
        mask_indptr,
        logit_masks,
        logit_mask_indptr,
        sampling_indices,
        sampling_indptr: vec![0, sampling_len],
        samplers,
        sampler_indptr: vec![0, sampler_len],
        adapter_bindings: original.adapter_bindings.clone(),
        spec_token_ids: Vec::new(),
        spec_position_ids: Vec::new(),
        spec_indptr: vec![0, 0],
        output_spec_flags,
        context_ids: original.context_ids.clone(),
        single_token_mode: !original.has_user_mask && chunk_len <= 1,
        has_user_mask: original.has_user_mask,
    };

    Ok(BuiltChunk {
        request: chunk,
        physical_page_ids: full_physical_page_ids[..chunk_pages].to_vec(),
        last_page_len: chunk_last_page_len,
    })
}

fn chunk_capacity_usage(
    original: &pie_bridge::ForwardRequest,
    full_physical_page_ids: &[PhysicalPageId],
    final_last_page_len: u32,
    start: usize,
    end: usize,
    page_size: u32,
    sampler_slots: &[usize],
) -> std::result::Result<RequestCapacityUsage, String> {
    if start >= end || end > original.token_ids.len() {
        return Err(format!(
            "invalid chunk range {start}..{end} for {} tokens",
            original.token_ids.len()
        ));
    }
    let (chunk_pages, chunk_last_page_len) = chunk_page_shape(
        original,
        full_physical_page_ids,
        final_last_page_len,
        end,
        page_size,
    )?;
    let chunk_len = end - start;
    let logprob_labels = sampler_slots
        .iter()
        .filter_map(|&slot| original.samplers.get(slot))
        .map(|sampler| match sampler {
            pie_bridge::Sampler::Logprob { .. } => 1,
            pie_bridge::Sampler::Logprobs { token_ids } => token_ids.len(),
            _ => 0,
        })
        .sum();
    let mut all_samplers_token = true;
    let mut has_prob_sampling = false;
    let mut has_output_spec = false;
    for &slot in sampler_slots {
        if let Some(sampler) = original.samplers.get(slot) {
            if !super::is_token_sampler(sampler) {
                all_samplers_token = false;
            }
            if super::sampler_needs_prob_rows(sampler) {
                has_prob_sampling = true;
            }
        }
        has_output_spec |= original
            .output_spec_flags
            .get(slot)
            .copied()
            .unwrap_or(false);
    }
    let has_dense_logit_requirement = original.has_user_mask
        || !original.logit_masks.is_empty()
        || has_output_spec
        || !all_samplers_token;
    let user_custom_mask_bytes = if original.has_user_mask && chunk_len > 1 {
        super::packed_mask_bytes(chunk_len, chunk_pages, chunk_last_page_len, page_size)
    } else {
        0
    };
    let spec_custom_mask_bytes =
        super::packed_mask_bytes(chunk_len, chunk_pages, chunk_last_page_len, page_size);

    Ok(RequestCapacityUsage {
        forward_tokens: chunk_len,
        page_refs: chunk_pages,
        logit_rows: if has_dense_logit_requirement {
            sampler_slots.len()
        } else {
            0
        },
        prob_rows: if has_prob_sampling {
            sampler_slots.len()
        } else {
            0
        },
        sampler_rows: sampler_slots.len(),
        logprob_labels,
        user_custom_mask_bytes,
        spec_custom_mask_bytes,
        has_spec_drafts: false,
        has_rs_spec_drafts: false,
        has_dense_logit_requirement,
        has_prob_sampling,
        is_single_token_decode: chunk_len == 1
            && original.single_token_mode
            && !original.has_user_mask,
        all_samplers_token,
    })
}

fn chunk_page_shape(
    original: &pie_bridge::ForwardRequest,
    full_physical_page_ids: &[PhysicalPageId],
    final_last_page_len: u32,
    end: usize,
    page_size: u32,
) -> std::result::Result<(usize, u32), String> {
    if page_size == 0 {
        return Err("chunked prefill requires non-zero page size".to_string());
    }

    let final_pages = full_physical_page_ids.len() as u32;
    let final_total_kv = total_kv_for_pages(final_pages, final_last_page_len, page_size);
    let kv_before = final_total_kv
        .checked_sub(original.token_ids.len() as u32)
        .ok_or_else(|| {
            format!(
                "chunked prefill cannot infer kv prefix: final_total_kv={final_total_kv}, input_tokens={}",
                original.token_ids.len()
            )
        })?;
    let chunk_total_kv = kv_before.saturating_add(end as u32);
    let chunk_pages = chunk_total_kv.div_ceil(page_size);
    if chunk_pages == 0 || chunk_pages > final_pages {
        return Err(format!(
            "chunked prefill page prefix out of range: need {chunk_pages}, have {final_pages}"
        ));
    }
    let chunk_last_page_len = compute_last_page_len(chunk_total_kv, chunk_pages, page_size);
    Ok((chunk_pages as usize, chunk_last_page_len))
}

fn chunk_sampler_slots_by_chunk(
    original: &pie_bridge::ForwardRequest,
    chunk_size: usize,
) -> BTreeMap<usize, Vec<usize>> {
    let mut by_chunk = BTreeMap::new();
    if chunk_size == 0 {
        return by_chunk;
    }
    for (slot, &idx) in original.sampling_indices.iter().enumerate() {
        by_chunk
            .entry(idx as usize / chunk_size)
            .or_insert_with(Vec::new)
            .push(slot);
    }
    by_chunk
}

#[cfg(test)]
fn collect_chunk_sampler_slots(
    original: &pie_bridge::ForwardRequest,
    start: usize,
    end: usize,
) -> Vec<usize> {
    original
        .sampling_indices
        .iter()
        .enumerate()
        .filter_map(|(slot, &idx)| {
            let idx = idx as usize;
            (start..end).contains(&idx).then_some(slot)
        })
        .collect()
}

fn total_kv_for_pages(num_pages: u32, last_page_len: u32, page_size: u32) -> u32 {
    if num_pages == 0 {
        0
    } else {
        (num_pages - 1)
            .saturating_mul(page_size)
            .saturating_add(last_page_len)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use anyhow::Result;
    use tokio::sync::{mpsc, oneshot};

    use crate::context::pagestore::compute_last_page_len;
    use crate::driver::SchedulerLimits;

    use super::super::{Completion, PendingRequest};
    use super::*;

    fn limits(max_requests: usize, max_tokens: usize, max_pages: usize) -> SchedulerLimits {
        SchedulerLimits {
            max_forward_requests: max_requests,
            max_forward_tokens: max_tokens,
            max_page_refs: max_pages,
            max_logit_rows: usize::MAX,
            max_prob_rows: usize::MAX,
            max_sampler_rows: usize::MAX,
            max_custom_mask_bytes: usize::MAX,
            max_logprob_labels: usize::MAX,
        }
    }

    fn positioned_pending_with_receiver(
        tokens: usize,
        page_size: u32,
    ) -> (PendingRequest, oneshot::Receiver<Result<ForwardOutput>>) {
        let (tx, rx) = oneshot::channel();
        let pages = (tokens as u32).div_ceil(page_size);
        let last_page_len = compute_last_page_len(tokens as u32, pages, page_size);
        (
            PendingRequest::direct(
                pie_bridge::ForwardRequest {
                    token_ids: (0..tokens as u32).collect(),
                    position_ids: (0..tokens as u32).collect(),
                    qo_indptr: vec![0, tokens as u32],
                    sampling_indptr: vec![0, 0],
                    sampler_indptr: vec![0, 0],
                    adapter_bindings: vec![pie_bridge::AdapterBinding {
                        adapter_id: -1,
                        seed: -1,
                    }],
                    output_spec_flags: vec![true],
                    context_ids: vec![7],
                    ..Default::default()
                },
                tx,
                (0..pages).map(|p| 100 + p).collect(),
                last_page_len,
            ),
            rx,
        )
    }

    fn positioned_pending(tokens: usize, page_size: u32) -> PendingRequest {
        positioned_pending_with_receiver(tokens, page_size).0
    }

    fn positioned_pending_with_prefix(
        tokens: usize,
        page_size: u32,
        kv_before: u32,
    ) -> PendingRequest {
        let (tx, _rx) = oneshot::channel();
        let total_kv = kv_before + tokens as u32;
        let pages = total_kv.div_ceil(page_size);
        let last_page_len = compute_last_page_len(total_kv, pages, page_size);
        PendingRequest::direct(
            pie_bridge::ForwardRequest {
                token_ids: (0..tokens as u32).collect(),
                position_ids: (kv_before..kv_before + tokens as u32).collect(),
                qo_indptr: vec![0, tokens as u32],
                sampling_indptr: vec![0, 0],
                sampler_indptr: vec![0, 0],
                adapter_bindings: vec![pie_bridge::AdapterBinding {
                    adapter_id: -1,
                    seed: -1,
                }],
                output_spec_flags: vec![true],
                context_ids: vec![7],
                ..Default::default()
            },
            tx,
            (0..pages).map(|p| 100 + p).collect(),
            last_page_len,
        )
    }

    fn token_response(token: u32) -> pie_bridge::ForwardResponse {
        pie_bridge::ForwardResponse {
            num_requests: 1,
            tokens_indptr: vec![0, 1],
            tokens: vec![token],
            ..Default::default()
        }
    }

    fn token_response_many(tokens: Vec<u32>) -> pie_bridge::ForwardResponse {
        pie_bridge::ForwardResponse {
            num_requests: 1,
            tokens_indptr: vec![0, tokens.len() as u32],
            tokens,
            ..Default::default()
        }
    }

    fn entropy_response(entropy: f32) -> pie_bridge::ForwardResponse {
        pie_bridge::ForwardResponse {
            num_requests: 1,
            entropies_indptr: vec![0, 1],
            entropies: vec![entropy],
            ..Default::default()
        }
    }

    fn expect_forward_response(result: Result<ForwardOutput>) -> pie_bridge::ForwardResponse {
        match result.expect("chunked response ok") {
            ForwardOutput::Response(resp) => resp,
            other => panic!("expected ForwardOutput::Response, got {other:?}"),
        }
    }

    fn with_spec(mut req: PendingRequest, spec_tokens: usize) -> PendingRequest {
        req.request.spec_token_ids = vec![1; spec_tokens];
        req.request.spec_position_ids = vec![1; spec_tokens];
        req.request.spec_indptr = vec![0, spec_tokens as u32];
        req
    }

    fn true_suffix_masks(tokens: usize, total_kv: u32) -> Vec<pie_bridge::Brle> {
        (0..tokens)
            .map(|i| {
                let false_prefix = (i as u32).min(total_kv);
                pie_bridge::Brle::from_vec(vec![false_prefix, total_kv - false_prefix])
            })
            .collect()
    }

    fn chunk_sampler_slots(req: &PendingRequest) -> &[usize] {
        match &req.completion {
            Completion::Chunk { sampler_slots, .. } => sampler_slots,
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected chunk continuation"),
        }
    }

    #[test]
    fn oversized_prefill_starts_chunking() {
        let pending = positioned_pending(10, 4);
        let chunked = match pending.maybe_start_chunking(limits(8, 4, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };

        assert_eq!(chunked.request.token_ids, vec![0, 1, 2, 3]);
        assert_eq!(chunked.request.position_ids, vec![0, 1, 2, 3]);
        assert_eq!(chunked.request.qo_indptr, vec![0, 4]);
        assert_eq!(chunked.physical_page_ids, vec![100]);
        assert_eq!(chunked.last_page_len, 4);

        match chunked.completion {
            Completion::Chunk {
                continuation: cont, ..
            } => {
                assert_eq!(cont.chunk_end, 4);
                assert_eq!(cont.chunk_size, 4);
                assert_eq!(cont.physical_page_ids, vec![100, 101, 102]);
                assert_eq!(cont.final_last_page_len, 2);
            }
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected chunk continuation"),
        }
    }

    #[test]
    fn chunk_continuation_builds_next_prefix() {
        let pending = positioned_pending(10, 4);
        let chunked = match pending.maybe_start_chunking(limits(8, 4, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        let cont = match chunked.completion {
            Completion::Chunk {
                continuation: cont, ..
            } => cont,
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected chunk continuation"),
        };

        let next = match cont.into_next_pending(4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(next.request.token_ids, vec![4, 5, 6, 7]);
        assert_eq!(next.request.position_ids, vec![4, 5, 6, 7]);
        assert_eq!(next.physical_page_ids, vec![100, 101]);
        assert_eq!(next.last_page_len, 4);
        match next.completion {
            Completion::Chunk {
                continuation: cont, ..
            } => assert_eq!(cont.chunk_end, 8),
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected chunk continuation"),
        }
    }

    #[test]
    fn final_chunk_remaps_sampler_index() {
        let mut pending = positioned_pending(10, 4);
        pending.request.sampling_indices = vec![9];
        pending.request.sampling_indptr = vec![0, 1];
        pending.request.samplers = vec![pie_bridge::Sampler::TopK {
            temperature: 0.0,
            k: 1,
        }];
        pending.request.sampler_indptr = vec![0, 1];

        let (chunk, pages, last_page_len, sampler_slots) = build_chunk_request(
            &pending.request,
            &pending.physical_page_ids,
            pending.last_page_len,
            8,
            10,
            4,
        )
        .expect("final chunk");

        assert_eq!(chunk.token_ids, vec![8, 9]);
        assert_eq!(chunk.sampling_indices, vec![1]);
        assert_eq!(chunk.samplers.len(), 1);
        assert_eq!(chunk.sampling_indptr, vec![0, 1]);
        assert_eq!(chunk.sampler_indptr, vec![0, 1]);
        assert_eq!(chunk.output_spec_flags, vec![true]);
        assert_eq!(sampler_slots, vec![0]);
        assert_eq!(pages, vec![100, 101, 102]);
        assert_eq!(last_page_len, 2);
    }

    #[test]
    fn non_final_chunk_remaps_sampler_index() {
        let mut pending = positioned_pending(10, 4);
        pending.request.sampling_indices = vec![3];
        pending.request.sampling_indptr = vec![0, 1];
        pending.request.samplers = vec![pie_bridge::Sampler::TopK {
            temperature: 0.0,
            k: 1,
        }];
        pending.request.sampler_indptr = vec![0, 1];

        let chunked = match pending.maybe_start_chunking(limits(8, 4, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };

        assert_eq!(chunked.request.token_ids, vec![0, 1, 2, 3]);
        assert_eq!(chunked.request.sampling_indices, vec![3]);
        assert_eq!(chunk_sampler_slots(&chunked), &[0]);
    }

    #[test]
    fn chunked_merges_arbitrary_sampler_outputs_in_original_order() {
        let (mut pending, mut response_rx) = positioned_pending_with_receiver(10, 4);
        pending.request.sampling_indices = vec![9, 1, 5];
        pending.request.sampling_indptr = vec![0, 3];
        pending.request.samplers = vec![
            pie_bridge::Sampler::TopK {
                temperature: 0.0,
                k: 1,
            },
            pie_bridge::Sampler::TopK {
                temperature: 0.0,
                k: 1,
            },
            pie_bridge::Sampler::Entropy,
        ];
        pending.request.sampler_indptr = vec![0, 3];

        let first = match pending.maybe_start_chunking(limits(8, 4, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(first.request.token_ids, vec![0, 1, 2, 3]);
        assert_eq!(first.request.sampling_indices, vec![1]);
        assert_eq!(chunk_sampler_slots(&first), &[1]);

        let (submit_tx, submit_rx) = crossbeam::channel::unbounded();
        let weak_submit_tx = submit_tx.clone();
        first.send_result(Ok(token_response(11)), Some(&weak_submit_tx), 4);

        let second = submit_rx.try_recv().expect("second chunk");
        assert_eq!(second.request.token_ids, vec![4, 5, 6, 7]);
        assert_eq!(second.request.sampling_indices, vec![1]);
        assert_eq!(chunk_sampler_slots(&second), &[2]);
        second.send_result(Ok(entropy_response(0.5)), Some(&weak_submit_tx), 4);

        let final_chunk = submit_rx.try_recv().expect("final chunk");
        assert_eq!(final_chunk.request.token_ids, vec![8, 9]);
        assert_eq!(final_chunk.request.sampling_indices, vec![1]);
        assert_eq!(chunk_sampler_slots(&final_chunk), &[0]);
        final_chunk.send_result(Ok(token_response(99)), Some(&weak_submit_tx), 4);

        let merged = expect_forward_response(response_rx.try_recv().expect("merged response"));
        assert_eq!(merged.num_requests, 1);
        assert_eq!(merged.tokens, vec![99, 11]);
        assert_eq!(merged.tokens_indptr, vec![0, 2]);
        assert_eq!(merged.entropies, vec![0.5]);
        assert_eq!(merged.entropies_indptr, vec![0, 1]);
    }

    #[test]
    fn chunked_validation_rejects_bad_shapes() {
        fn assert_reject(req: PendingRequest, max_tokens: usize, needle: &str) {
            match req.maybe_start_chunking(limits(8, max_tokens, 100), 4) {
                Ok(_) => panic!("expected rejection containing {needle:?}"),
                Err((_, msg)) => assert!(
                    msg.contains(needle),
                    "expected {msg:?} to contain {needle:?}"
                ),
            }
        }

        assert_reject(positioned_pending(10, 4), 0, "zero");

        assert_reject(with_spec(positioned_pending(10, 4), 1), 4, "speculative");

        let mut missing_position = positioned_pending(10, 4);
        missing_position.request.position_ids.pop();
        assert_reject(missing_position, 4, "one position per token");

        let mut bad_masks = positioned_pending(10, 4);
        bad_masks.request.masks = vec![pie_bridge::Brle::all_true(1)];
        assert_reject(bad_masks, 4, "zero masks or one mask per token");

        let mut bad_sampler_count = positioned_pending(10, 4);
        bad_sampler_count.request.sampling_indices = vec![0];
        assert_reject(bad_sampler_count, 4, "one sampler per sampling index");

        let mut out_of_range_sampler = positioned_pending(10, 4);
        out_of_range_sampler.request.sampling_indices = vec![10];
        out_of_range_sampler.request.samplers = vec![pie_bridge::Sampler::TopK {
            temperature: 0.0,
            k: 1,
        }];
        assert_reject(out_of_range_sampler, 4, "outside the input window");
    }

    #[test]
    fn chunked_validation_rejects_later_chunk_page_limit_before_first_submit() {
        let pending = positioned_pending(10, 4);
        let err = match pending.maybe_start_chunking(limits(8, 4, 2), 4) {
            Ok(_) => panic!("expected page-ref rejection"),
            Err((_, msg)) => msg,
        };

        assert!(err.contains("page refs"), "{err}");
    }

    #[test]
    fn chunked_validation_rejects_later_chunk_sampler_limit_before_first_submit() {
        let mut capped = limits(8, 4, 100);
        capped.max_sampler_rows = 2;
        let mut pending = positioned_pending(10, 4);
        pending.request.sampling_indices = vec![4, 5, 6];
        pending.request.sampling_indptr = vec![0, 3];
        pending.request.samplers = (0..3)
            .map(|_| pie_bridge::Sampler::TopK {
                temperature: 0.0,
                k: 1,
            })
            .collect();
        pending.request.sampler_indptr = vec![0, 3];

        let err = match pending.maybe_start_chunking(capped, 4) {
            Ok(_) => panic!("expected sampler-row rejection"),
            Err((_, msg)) => msg,
        };

        assert!(err.contains("sampler rows"), "{err}");
    }

    #[test]
    fn chunked_validation_rejects_later_chunk_mask_limit_before_first_submit() {
        let mut capped = limits(8, 4, 100);
        capped.max_custom_mask_bytes = 3;
        let tokens = 10;
        let page_size = 4;
        let mut pending = positioned_pending(tokens, page_size);
        pending.request.masks = true_suffix_masks(tokens, tokens as u32);
        pending.request.mask_indptr = vec![0, tokens as u32];
        pending.request.has_user_mask = true;
        pending.request.single_token_mode = false;

        let err = match pending.maybe_start_chunking(capped, page_size) {
            Ok(_) => panic!("expected custom-mask rejection"),
            Err((_, msg)) => msg,
        };

        assert!(err.contains("custom mask bytes"), "{err}");
    }

    #[test]
    fn chunk_ranges_cover_request_across_lengths_limits_and_page_sizes() {
        for page_size in [1, 2, 3, 4, 7, 16] {
            for tokens in 1..=80 {
                for max_tokens in 1..=16 {
                    let pending = positioned_pending(tokens, page_size);
                    let mut current = match pending
                        .maybe_start_chunking(limits(8, max_tokens, usize::MAX), page_size)
                    {
                        Ok(p) => p,
                        Err((_, msg)) => panic!("{msg}"),
                    };
                    let mut seen = Vec::new();

                    loop {
                        assert!(current.request.token_ids.len() <= max_tokens.max(tokens));
                        if tokens > max_tokens {
                            assert!(current.request.token_ids.len() <= max_tokens);
                        }
                        assert_eq!(
                            current.request.qo_indptr,
                            vec![0, current.request.token_ids.len() as u32]
                        );
                        seen.extend_from_slice(&current.request.token_ids);

                        match current.completion {
                            Completion::Direct(_) | Completion::Chain { .. } => {
                                assert!(tokens <= max_tokens);
                                break;
                            }
                            Completion::Chunk {
                                continuation: cont, ..
                            } => {
                                if cont.chunk_end >= tokens {
                                    break;
                                }
                                current = match cont.into_next_pending(page_size) {
                                    Ok(p) => p,
                                    Err((_, msg)) => panic!("{msg}"),
                                };
                            }
                        }
                    }

                    let expected: Vec<u32> = (0..tokens as u32).collect();
                    assert_eq!(seen, expected);
                }
            }
        }
    }

    #[test]
    fn chunk_ranges_account_for_existing_kv_prefix() {
        let pending = positioned_pending_with_prefix(10, 4, 3);
        let first = match pending.maybe_start_chunking(limits(8, 4, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(first.request.token_ids, vec![0, 1, 2, 3]);
        assert_eq!(first.physical_page_ids, vec![100, 101]);
        assert_eq!(first.last_page_len, 3);

        let cont = match first.completion {
            Completion::Chunk {
                continuation: cont, ..
            } => cont,
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected continuation"),
        };
        let second = match cont.into_next_pending(4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(second.request.token_ids, vec![4, 5, 6, 7]);
        assert_eq!(second.physical_page_ids, vec![100, 101, 102]);
        assert_eq!(second.last_page_len, 3);

        let cont = match second.completion {
            Completion::Chunk {
                continuation: cont, ..
            } => cont,
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected continuation"),
        };
        let final_chunk = match cont.into_next_pending(4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(final_chunk.request.token_ids, vec![8, 9]);
        assert_eq!(final_chunk.physical_page_ids, vec![100, 101, 102, 103]);
        assert_eq!(final_chunk.last_page_len, 1);
    }

    #[test]
    fn chunked_custom_masks_keep_full_rows_and_prefix_pages() {
        let kv_before = 3;
        let tokens = 10;
        let page_size = 4;
        let total_kv = kv_before + tokens as u32;
        let mut pending = positioned_pending_with_prefix(tokens, page_size, kv_before);
        let masks = true_suffix_masks(tokens, total_kv);
        pending.request.masks = masks.clone();
        pending.request.mask_indptr = vec![0, tokens as u32];
        pending.request.has_user_mask = true;
        pending.request.single_token_mode = false;

        let first = match pending.maybe_start_chunking(limits(8, 4, 100), page_size) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(first.request.token_ids, vec![0, 1, 2, 3]);
        assert_eq!(first.request.masks, masks[0..4]);
        assert_eq!(first.request.mask_indptr, vec![0, 4]);
        assert!(first.request.has_user_mask);
        assert!(!first.request.single_token_mode);
        assert_eq!(first.physical_page_ids, vec![100, 101]);
        assert_eq!(first.last_page_len, 3);

        let cont = match first.completion {
            Completion::Chunk {
                continuation: cont, ..
            } => cont,
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected continuation"),
        };
        let second = match cont.into_next_pending(page_size) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(second.request.token_ids, vec![4, 5, 6, 7]);
        assert_eq!(second.request.masks, masks[4..8]);
        assert_eq!(second.request.mask_indptr, vec![0, 4]);
        assert!(second.request.has_user_mask);
        assert!(!second.request.single_token_mode);
        assert_eq!(second.physical_page_ids, vec![100, 101, 102]);
        assert_eq!(second.last_page_len, 3);

        let cont = match second.completion {
            Completion::Chunk {
                continuation: cont, ..
            } => cont,
            Completion::Direct(_) | Completion::Chain { .. } => panic!("expected continuation"),
        };
        let final_chunk = match cont.into_next_pending(page_size) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        assert_eq!(final_chunk.request.token_ids, vec![8, 9]);
        assert_eq!(final_chunk.request.masks, masks[8..10]);
        assert_eq!(final_chunk.request.mask_indptr, vec![0, 2]);
        assert!(final_chunk.request.has_user_mask);
        assert!(!final_chunk.request.single_token_mode);
        assert_eq!(final_chunk.physical_page_ids, vec![100, 101, 102, 103]);
        assert_eq!(final_chunk.last_page_len, 1);
    }

    #[test]
    fn chunk_ranges_preserve_custom_masks_across_prefixes() {
        for page_size in [1, 2, 3, 4, 7, 16] {
            for kv_before in 0..=9 {
                for tokens in 1..=48 {
                    for max_tokens in 1..=12 {
                        if tokens <= max_tokens {
                            continue;
                        }

                        let total_kv = kv_before + tokens as u32;
                        let mut pending =
                            positioned_pending_with_prefix(tokens, page_size, kv_before);
                        let masks = true_suffix_masks(tokens, total_kv);
                        pending.request.masks = masks.clone();
                        pending.request.mask_indptr = vec![0, tokens as u32];
                        pending.request.has_user_mask = true;
                        pending.request.single_token_mode = false;

                        let mut current = match pending
                            .maybe_start_chunking(limits(8, max_tokens, usize::MAX), page_size)
                        {
                            Ok(p) => p,
                            Err((_, msg)) => panic!("{msg}"),
                        };
                        let mut offset = 0usize;

                        loop {
                            let chunk_len = current.request.token_ids.len();
                            assert!(chunk_len <= max_tokens);
                            assert_eq!(
                                current.request.token_ids,
                                (offset as u32..(offset + chunk_len) as u32).collect::<Vec<_>>()
                            );
                            assert_eq!(current.request.masks, masks[offset..offset + chunk_len]);
                            assert_eq!(current.request.mask_indptr, vec![0, chunk_len as u32]);
                            assert!(current.request.has_user_mask);
                            assert!(!current.request.single_token_mode);

                            offset += chunk_len;
                            match current.completion {
                                Completion::Direct(_) | Completion::Chain { .. } => panic!("expected chunk continuation"),
                                Completion::Chunk {
                                    continuation: cont, ..
                                } => {
                                    if cont.chunk_end >= tokens {
                                        break;
                                    }
                                    current = match cont.into_next_pending(page_size) {
                                        Ok(p) => p,
                                        Err((_, msg)) => panic!("{msg}"),
                                    };
                                }
                            }
                        }
                        assert_eq!(offset, tokens);
                    }
                }
            }
        }
    }

    #[test]
    fn chunked_no_sampler_request_returns_empty_response() {
        let (pending, mut response_rx) = positioned_pending_with_receiver(5, 4);
        let first = match pending.maybe_start_chunking(limits(8, 2, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        let (submit_tx, submit_rx) = crossbeam::channel::unbounded();
        let weak_submit_tx = submit_tx.clone();

        let mut current = first;
        loop {
            current.send_result(
                Ok(pie_bridge::ForwardResponse::default()),
                Some(&weak_submit_tx),
                4,
            );
            if let Ok(result) = response_rx.try_recv() {
                let merged = expect_forward_response(result);
                assert_eq!(merged.num_requests, 1);
                assert_eq!(merged.tokens_indptr, vec![0, 0]);
                assert_eq!(merged.dists_req_indptr, vec![0, 0]);
                assert_eq!(merged.dists_kv_indptr, vec![0]);
                assert_eq!(merged.logits_req_indptr, vec![0, 0]);
                assert_eq!(merged.logits_byte_indptr, vec![0]);
                assert_eq!(merged.logprobs_req_indptr, vec![0, 0]);
                assert_eq!(merged.logprobs_val_indptr, vec![0]);
                assert_eq!(merged.entropies_indptr, vec![0, 0]);
                break;
            }
            current = submit_rx.try_recv().expect("next continuation");
        }
    }

    #[test]
    fn chunked_reports_requeue_failure_when_scheduler_is_gone() {
        let (pending, mut response_rx) = positioned_pending_with_receiver(5, 4);
        let chunked = match pending.maybe_start_chunking(limits(8, 2, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        // Simulate "scheduler gone" by dropping the receiver — the
        // clone held below will then fail to send. With the prior tokio
        // mpsc + Weak design the test simulated this by dropping the
        // strong sender; crossbeam doesn't have weak senders so we now
        // close the channel from the receiver side instead.
        let (submit_tx, submit_rx) = crossbeam::channel::unbounded();
        drop(submit_rx);

        chunked.send_result(
            Ok(pie_bridge::ForwardResponse::default()),
            Some(&submit_tx),
            4,
        );

        let err = response_rx
            .try_recv()
            .expect("response error")
            .expect_err("expected requeue error");
        assert!(
            err.to_string().contains("scheduler channel closed")
                || err.to_string().contains("scheduler shutting down")
        );
    }

    #[test]
    fn chunk_response_accumulator_rejects_malformed_responses() {
        let token_sampler = pie_bridge::Sampler::TopK {
            temperature: 0.0,
            k: 1,
        };

        let mut missing = ChunkResponseAccumulator::new(1);
        let err = missing
            .record_response(
                &[token_sampler.clone()],
                &[0],
                pie_bridge::ForwardResponse::default(),
            )
            .expect_err("missing token should fail");
        assert!(err.contains("missing token output"));

        let mut extra = ChunkResponseAccumulator::new(0);
        let err = extra
            .record_response(&[], &[], token_response(7))
            .expect_err("extra token should fail");
        assert!(err.contains("extra token outputs"));

        let mut duplicate = ChunkResponseAccumulator::new(1);
        duplicate
            .record_response(&[token_sampler.clone()], &[0], token_response(7))
            .expect("first token");
        let err = duplicate
            .record_response(&[token_sampler], &[0], token_response(8))
            .expect_err("duplicate slot should fail");
        assert!(err.contains("duplicate output"));

        let mut bad_nested = ChunkResponseAccumulator::new(1);
        let err = bad_nested
            .record_response(
                &[pie_bridge::Sampler::Dist {
                    temperature: 1.0,
                    num_tokens: 2,
                }],
                &[0],
                pie_bridge::ForwardResponse {
                    num_requests: 1,
                    dists_req_indptr: vec![0, 1],
                    dists_kv_indptr: vec![0, 2],
                    dists_ids: vec![1],
                    dists_probs: vec![1.0],
                    ..Default::default()
                },
            )
            .expect_err("invalid nested range should fail");
        assert!(err.contains("invalid range"));
    }

    #[test]
    fn chunked_dense_sampler_stress_preserves_original_order() {
        let tokens = 4099usize;
        let max_tokens = 127usize;
        let sample_count = 1024usize;
        let (mut pending, mut response_rx) = positioned_pending_with_receiver(tokens, 64);
        pending.request.sampling_indices = (0..sample_count)
            .map(|i| ((i * 257 + 13) % tokens) as u32)
            .collect();
        pending.request.sampling_indptr = vec![0, sample_count as u32];
        pending.request.samplers = (0..sample_count)
            .map(|_| pie_bridge::Sampler::TopK {
                temperature: 0.0,
                k: 1,
            })
            .collect();
        pending.request.sampler_indptr = vec![0, sample_count as u32];

        let first = match pending.maybe_start_chunking(limits(8, max_tokens, usize::MAX), 64) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        let (submit_tx, submit_rx) = crossbeam::channel::unbounded();
        let weak_submit_tx = submit_tx.clone();
        let mut current = first;
        let mut chunks = 0usize;

        let merged = loop {
            chunks += 1;
            let tokens_for_chunk: Vec<u32> = chunk_sampler_slots(&current)
                .iter()
                .map(|&slot| 50_000 + slot as u32)
                .collect();
            current.send_result(
                Ok(token_response_many(tokens_for_chunk)),
                Some(&weak_submit_tx),
                64,
            );
            if let Ok(result) = response_rx.try_recv() {
                break expect_forward_response(result);
            }
            current = submit_rx.try_recv().expect("next continuation");
        };

        let expected: Vec<u32> = (0..sample_count).map(|slot| 50_000 + slot as u32).collect();
        assert_eq!(chunks, tokens.div_ceil(max_tokens));
        assert_eq!(merged.tokens, expected);
        assert_eq!(merged.tokens_indptr, vec![0, sample_count as u32]);
    }

    #[test]
    fn chunk_response_accumulator_preserves_original_probe_sampler_order() {
        let original_samplers = vec![
            pie_bridge::Sampler::RawLogits,
            pie_bridge::Sampler::Dist {
                temperature: 1.0,
                num_tokens: 2,
            },
            pie_bridge::Sampler::Logprobs {
                token_ids: vec![7, 8],
            },
            pie_bridge::Sampler::Logprob { token_id: 9 },
            pie_bridge::Sampler::Entropy,
        ];
        let mut acc = ChunkResponseAccumulator::new(original_samplers.len());

        acc.record_response(
            &[original_samplers[1].clone(), original_samplers[3].clone()],
            &[1, 3],
            pie_bridge::ForwardResponse {
                num_requests: 1,
                dists_req_indptr: vec![0, 1],
                dists_kv_indptr: vec![0, 2],
                dists_ids: vec![10, 11],
                dists_probs: vec![0.7, 0.3],
                logprobs_req_indptr: vec![0, 1],
                logprobs_val_indptr: vec![0, 1],
                logprobs_values: vec![-2.0],
                ..Default::default()
            },
        )
        .expect("first chunk response");

        acc.record_response(
            &[
                original_samplers[0].clone(),
                original_samplers[2].clone(),
                original_samplers[4].clone(),
            ],
            &[0, 2, 4],
            pie_bridge::ForwardResponse {
                num_requests: 1,
                logits_req_indptr: vec![0, 1],
                logits_byte_indptr: vec![0, 4],
                logits_bytes: vec![1, 2, 3, 4],
                logprobs_req_indptr: vec![0, 1],
                logprobs_val_indptr: vec![0, 2],
                logprobs_values: vec![0.1, 0.2],
                entropies_indptr: vec![0, 1],
                entropies: vec![0.9],
                ..Default::default()
            },
        )
        .expect("second chunk response");

        let merged = acc.into_response(&original_samplers).expect("merged");
        assert_eq!(merged.logits_byte_indptr, vec![0, 4]);
        assert_eq!(merged.logits_bytes, vec![1, 2, 3, 4]);
        assert_eq!(merged.dists_kv_indptr, vec![0, 2]);
        assert_eq!(merged.dists_ids, vec![10, 11]);
        assert_eq!(merged.dists_probs, vec![0.7, 0.3]);
        assert_eq!(merged.logprobs_val_indptr, vec![0, 2, 3]);
        assert_eq!(merged.logprobs_values, vec![0.1, 0.2, -2.0]);
        assert_eq!(merged.entropies, vec![0.9]);
    }

    #[test]
    fn intermediate_chunk_requeues_continuation() {
        let pending = positioned_pending(10, 4);
        let chunked = match pending.maybe_start_chunking(limits(8, 4, 100), 4) {
            Ok(p) => p,
            Err((_, msg)) => panic!("{msg}"),
        };
        let (tx, rx) = crossbeam::channel::unbounded();
        let weak_tx = tx.clone();
        chunked.send_result(
            Ok(pie_bridge::ForwardResponse::default()),
            Some(&weak_tx),
            4,
        );

        let next = rx.try_recv().expect("next continuation");
        assert_eq!(next.request.token_ids, vec![4, 5, 6, 7]);
        assert_eq!(next.physical_page_ids, vec![100, 101]);
    }

    #[test]
    #[ignore = "large allocation/performance smoke for chunk construction"]
    fn chunk_request_builder_large_prompt_perf_smoke() {
        let tokens = 1_000_003usize;
        let max_tokens = 1024usize;
        let page_size = 128u32;
        let pending = positioned_pending(tokens, page_size);
        let start = Instant::now();
        let mut current =
            match pending.maybe_start_chunking(limits(8, max_tokens, usize::MAX), page_size) {
                Ok(p) => p,
                Err((_, msg)) => panic!("{msg}"),
            };
        let mut chunks = 0usize;
        let mut total_seen = 0usize;

        loop {
            chunks += 1;
            total_seen += current.request.token_ids.len();
            match current.completion {
                Completion::Direct(_) | Completion::Chain { .. } => break,
                Completion::Chunk {
                    continuation: cont, ..
                } => {
                    if cont.chunk_end >= tokens {
                        break;
                    }
                    current = match cont.into_next_pending(page_size) {
                        Ok(p) => p,
                        Err((_, msg)) => panic!("{msg}"),
                    };
                }
            }
        }

        let elapsed = start.elapsed();
        eprintln!(
            "chunk_request_builder_large_prompt_perf_smoke: tokens={tokens} chunks={chunks} elapsed={elapsed:?}"
        );
        assert_eq!(chunks, tokens.div_ceil(max_tokens));
        assert_eq!(total_seen, tokens);
    }
}
