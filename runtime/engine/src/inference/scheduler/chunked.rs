//! Minimal chunked prefill support for a single oversized pending request.
//!
//! This module intentionally does not water-fill normal batches or split
//! already-scheduled requests. It only converts one request that exceeds the
//! driver's `max_forward_tokens` into sequential prefix chunks, then merges the
//! per-chunk sampler outputs back into the original single-request response.

use std::collections::BTreeMap;

use anyhow::Result;
use tokio::sync::oneshot;

use crate::arena::PhysicalPageId;
use crate::working_set::page_size::compute_last_page_len;
use crate::driver::SchedulerLimits;
use crate::inference::ForwardOutput;

use super::{Completion, PendingRequest};
use crate::inference::batch::{
    RequestCapacityUsage, is_token_sampler, packed_mask_bytes, request_capacity_usage,
    sampler_needs_prob_rows,
};

pub(super) struct ChunkContinuation {
    original_request: pie_driver_abi::ForwardRequest,
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
    request: pie_driver_abi::ForwardRequest,
    physical_page_ids: Vec<PhysicalPageId>,
    last_page_len: u32,
}

impl PendingRequest {
    pub(crate) fn maybe_start_chunking(
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
            program_identity_hashes,
            pipeline_id,
            ..
        } = self;
        let Completion::Direct(response_tx) = completion else {
            unreachable!("chunk continuations returned above");
        };
        let response_accumulator = ChunkResponseAccumulator::new(request.n_samplers());

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
            // Carry the original request's identity onto the first chunk.
            program_identity_hashes,
            pipeline_id,
            submitted_at_us: 0,
            prebuilt: false,
        })
    }

    pub(crate) fn send_error(self, msg: String) {
        self.send_result::<ForwardOutput>(Err(anyhow::anyhow!(msg)), None, 0);
    }

    pub(crate) fn send_result<T>(
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
            program_identity_hashes: _,
            ..
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
                    request.samplers(),
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
        }
    }
}

impl ChunkContinuation {
    fn complete_chunk(
        mut self,
        resp: pie_driver_abi::ForwardResponse,
        chunk_samplers: Vec<pie_driver_abi::Sampler>,
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
            match response_accumulator.into_response(&original_request.samplers()) {
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
            // Prefill-chunk continuation: re-enters the scheduler via `submit_tx`
            // and re-announces its identity on arrival; the per-program hashes are
            // recomputed at the decode/sampling step, not carried on the chunk.
            program_identity_hashes: Vec::new(),
            // M-A1: prefill-chunk continuations rejoin the wave on re-arrival;
            // ChunkContinuation doesn't carry the pipeline_id (guru: thread it if
            // chunked prefill must hold wave membership across chunks).
            pipeline_id: None,
            submitted_at_us: 0,
            prebuilt: false,
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
        samplers: &[pie_driver_abi::Sampler],
        sampler_slots: &[usize],
        resp: pie_driver_abi::ForwardResponse,
    ) -> std::result::Result<(), String> {
        use pie_driver_abi::Sampler;

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
        samplers: &[pie_driver_abi::Sampler],
    ) -> std::result::Result<pie_driver_abi::ForwardResponse, String> {
        use pie_driver_abi::Sampler;

        let mut out = pie_driver_abi::ForwardResponse {
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

fn response_tokens(resp: &pie_driver_abi::ForwardResponse) -> std::result::Result<Vec<u32>, String> {
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
    resp: &pie_driver_abi::ForwardResponse,
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
    resp: &pie_driver_abi::ForwardResponse,
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
    resp: &pie_driver_abi::ForwardResponse,
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

fn response_entropies(resp: &pie_driver_abi::ForwardResponse) -> std::result::Result<Vec<f32>, String> {
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
    req: &pie_driver_abi::ForwardRequest,
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
    req: &pie_driver_abi::ForwardRequest,
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
    if req.sampling_indices.len() != req.n_samplers() {
        return Err(format!(
            "chunked prefill requires one sampler per sampling index (got {} indices, {} samplers)",
            req.sampling_indices.len(),
            req.n_samplers()
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

fn build_chunk_request_for_slots(
    original: &pie_driver_abi::ForwardRequest,
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
        let Some(sampler) = original.sampler_at(slot) else {
            return Err(format!(
                "chunked prefill sampler slot {slot} is outside {} samplers",
                original.n_samplers()
            ));
        };
        let idx_usize = idx as usize;
        if (start..end).contains(&idx_usize) {
            sampling_indices.push((idx_usize - start) as u32);
            samplers.push(sampler);
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
    let mut chunk = pie_driver_abi::ForwardRequest {
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
        // Sampler SoA filled by `set_samplers` below from the per-slot list.
        sampler_indptr: vec![0, sampler_len],
        adapter_bindings: original.adapter_bindings.clone(),
        spec_token_ids: Vec::new(),
        spec_position_ids: Vec::new(),
        spec_indptr: vec![0, 0],
        output_spec_flags,
        context_ids: original.context_ids.clone(),
        single_token_mode: !original.has_user_mask && chunk_len <= 1,
        has_user_mask: original.has_user_mask,
        // TODO(multimodal): chunked prefill does not yet split visual spans
        // across chunks. Until the vision encoder consumes these fields, chunks
        // carry no images (empty CSR roots). See MULTIMODAL.md.
        image_indptr: vec![0, 0],
        image_grids: Vec::new(),
        image_anchor_positions: Vec::new(),
        image_pixels: Vec::new(),
        image_pixel_indptr: vec![0],
        image_mrope_positions: Vec::new(),
        image_mrope_indptr: vec![0],
        image_patch_positions: Vec::new(),
        image_anchor_rows: Vec::new(),
        audio_features: Vec::new(),
        audio_feature_indptr: vec![0],
        audio_anchor_rows: Vec::new(),
        audio_indptr: vec![0, 0],
        ..Default::default()
    };
    chunk.set_samplers(&samplers);

    Ok(BuiltChunk {
        request: chunk,
        physical_page_ids: full_physical_page_ids[..chunk_pages].to_vec(),
        last_page_len: chunk_last_page_len,
    })
}

fn chunk_capacity_usage(
    original: &pie_driver_abi::ForwardRequest,
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
        .filter_map(|&slot| original.sampler_at(slot))
        .map(|sampler| match sampler {
            pie_driver_abi::Sampler::Logprob { .. } => 1,
            pie_driver_abi::Sampler::Logprobs { token_ids } => token_ids.len(),
            _ => 0,
        })
        .sum();
    let mut all_samplers_token = true;
    let mut has_prob_sampling = false;
    let mut has_output_spec = false;
    for &slot in sampler_slots {
        if let Some(sampler) = original.sampler_at(slot) {
            if !is_token_sampler(&sampler) {
                all_samplers_token = false;
            }
            if sampler_needs_prob_rows(&sampler) {
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
        packed_mask_bytes(chunk_len, chunk_pages, chunk_last_page_len, page_size)
    } else {
        0
    };
    let spec_custom_mask_bytes =
        packed_mask_bytes(chunk_len, chunk_pages, chunk_last_page_len, page_size);

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
        has_dense_logit_requirement,
        has_prob_sampling,
        is_single_token_decode: chunk_len == 1
            && original.single_token_mode
            && !original.has_user_mask,
        all_samplers_token,
    })
}

fn chunk_page_shape(
    original: &pie_driver_abi::ForwardRequest,
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
    original: &pie_driver_abi::ForwardRequest,
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

fn total_kv_for_pages(num_pages: u32, last_page_len: u32, page_size: u32) -> u32 {
    if num_pages == 0 {
        0
    } else {
        (num_pages - 1)
            .saturating_mul(page_size)
            .saturating_add(last_page_len)
    }
}

