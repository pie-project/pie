//! Runtime driver-channel benchmarks.
//!
//! These measure the real runtime-side channel implementations instead
//! of the lower-level bridge primitives:
//!
//! - `inproc_channel/*` uses `pie::driver::InProcChannel::submit`,
//!   the real `ffi_vtable()` callbacks, the real pending map/inbox,
//!   and the real Rust-side `PieResponseFrameDesc` -> `DriverResponse`
//!   conversion. The fake C++ driver thread mirrors `InProcServer` just
//!   enough to convert the request descriptor to legacy scratch and send
//!   a small `ForwardResponse`.
//! - `inproc_polling_channel/*` uses
//!   `pie::driver::InProcPollingChannel::submit`, the same FFI callbacks,
//!   and fixed preallocated slots with polling/yield waits.
//! - `shmem_channel/*` uses `pie::driver::ShmemChannel::submit` and a
//!   real `pie_bridge::ipc::ShmemServer`. The fake Python worker copies
//!   the shmem payload to bytes, parses/touches it, builds a small
//!   `ForwardResponse`, and commits it.
//!
//! Run:
//!   cargo bench -p pie --bench driver_channel

use std::collections::{HashMap, VecDeque};
use std::ffi::c_void;
use std::hint::black_box;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, mpsc};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};

use pie::driver::{
    DriverChannel, DriverRequest, InProcChannel, InProcPollingChannel, InProcVTable, ShmemChannel,
};
use pie_bridge::ipc::ShmemServer;
use pie_bridge::schema::{
    __pie_response_frame_from_desc, PIE_REQUEST_PAYLOAD_ADAPTER, PIE_REQUEST_PAYLOAD_COPY,
    PIE_REQUEST_PAYLOAD_FORWARD, PIE_REQUEST_PAYLOAD_HEALTH, PIE_RESPONSE_PAYLOAD_FORWARD,
    PIE_RESPONSE_PAYLOAD_STATUS, PieAdapterBindingDesc, PieForwardRequestDesc,
    PieForwardResponseDesc, PieFrameDesc, PieResponseFrameDesc, PieResponsePayloadDesc,
    PieSamplerDesc, PieStatusResponseDesc, pie_frame_view,
};
use pie_bridge::wire::{encode_response, parse_request};
use pie_bridge::{
    AdapterBinding, ForwardRequest, ForwardResponse, Frame, RequestPayload, ResponseFrame,
    ResponsePayload, SCHEMA_HASH, Sampler,
};

const BALANCED_SPIN_BUDGET_US: u64 = 1_000;
const LOW_LATENCY_SPIN_BUDGET_US: u64 = u64::MAX;
const DRIVER_ID: usize = 0;

fn make_request(n_tokens: usize) -> DriverRequest {
    DriverRequest {
        driver_id: DRIVER_ID,
        payload: RequestPayload::Forward(make_forward_request(n_tokens)),
    }
}

fn make_frame(n_tokens: usize) -> Frame {
    Frame {
        driver_id: DRIVER_ID as u32,
        payload: RequestPayload::Forward(make_forward_request(n_tokens)),
    }
}

fn make_forward_request(n_tokens: usize) -> ForwardRequest {
    ForwardRequest {
        token_ids: (0..n_tokens).map(|i| i as u32).collect(),
        position_ids: (0..n_tokens).map(|i| i as u32).collect(),
        qo_indptr: vec![0, n_tokens as u32],
        kv_page_indptr: vec![0, 1],
        kv_page_indices: vec![0],
        kv_last_page_lens: vec![n_tokens.min(128) as u32],
        samplers: vec![Sampler::TopKTopP {
            temperature: 0.7,
            k: 50,
            p: 0.9,
        }],
        sampler_indptr: vec![0, 1],
        adapter_bindings: vec![AdapterBinding {
            adapter_id: -1,
            seed: -1,
        }],
        single_token_mode: false,
        has_user_mask: false,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// In-proc fake C++ driver.
// ---------------------------------------------------------------------------

#[derive(Default)]
struct LegacyScratch {
    flattened_masks: Vec<u32>,
    mask_byte_indptr: Vec<u32>,
    logit_masks_flat: Vec<u32>,
    logit_mask_byte_indptr: Vec<u32>,
    sampler_types: Vec<u32>,
    sampler_temperatures: Vec<f32>,
    sampler_top_k: Vec<u32>,
    sampler_top_p: Vec<f32>,
    sampler_min_p: Vec<f32>,
    sampler_seeds: Vec<u32>,
    sampler_label_ids: Vec<u32>,
    sampler_label_indptr: Vec<u32>,
    request_num_samplers: Vec<u32>,
    adapter_indices: Vec<i64>,
    adapter_seeds: Vec<i64>,
}

impl LegacyScratch {
    fn clear(&mut self) {
        self.flattened_masks.clear();
        self.mask_byte_indptr.clear();
        self.logit_masks_flat.clear();
        self.logit_mask_byte_indptr.clear();
        self.sampler_types.clear();
        self.sampler_temperatures.clear();
        self.sampler_top_k.clear();
        self.sampler_top_p.clear();
        self.sampler_min_p.clear();
        self.sampler_seeds.clear();
        self.sampler_label_ids.clear();
        self.sampler_label_indptr.clear();
        self.request_num_samplers.clear();
        self.adapter_indices.clear();
        self.adapter_seeds.clear();
    }
}

fn new_to_old_sampler_kind(kind: u8) -> u32 {
    match kind {
        0 => 1,   // Multinomial
        1 => 2,   // TopK
        2 => 3,   // TopP
        3 => 4,   // MinP
        4 => 5,   // TopKTopP
        5 => 6,   // Embedding
        6 => 0,   // Dist
        7 => 7,   // RawLogits
        8 => 8,   // Logprob
        9 => 9,   // Logprobs
        10 => 10, // Entropy
        _ => 0,
    }
}

fn slice_from_raw<'a, T>(ptr: *const T, len: usize) -> &'a [T] {
    if ptr.is_null() || len == 0 {
        &[]
    } else {
        // SAFETY: all desc pointers come from the real generated
        // `pie_frame_view` and remain valid until `send_response`.
        unsafe { std::slice::from_raw_parts(ptr, len) }
    }
}

fn touch_adapter_bindings(items: &[PieAdapterBindingDesc], scratch: &mut LegacyScratch) {
    scratch.adapter_indices.resize(items.len(), 0);
    scratch.adapter_seeds.resize(items.len(), 0);
    for (i, item) in items.iter().enumerate() {
        scratch.adapter_indices[i] = item.adapter_id;
        scratch.adapter_seeds[i] = item.seed;
    }
}

fn touch_samplers(items: &[PieSamplerDesc], sampler_indptr: &[u32], scratch: &mut LegacyScratch) {
    scratch.sampler_types.resize(items.len(), 0);
    scratch.sampler_temperatures.resize(items.len(), 0.0);
    scratch.sampler_top_k.resize(items.len(), 0);
    scratch.sampler_top_p.resize(items.len(), 0.0);
    scratch.sampler_min_p.resize(items.len(), 0.0);
    scratch.sampler_seeds.resize(items.len(), 0);
    scratch.sampler_label_indptr.reserve(items.len() + 1);
    scratch.sampler_label_indptr.push(0);

    for (i, s) in items.iter().enumerate() {
        scratch.sampler_types[i] = new_to_old_sampler_kind(s.kind);
        scratch.sampler_temperatures[i] = s.temperature;
        scratch.sampler_top_p[i] = 1.0;
        match s.kind {
            1 => scratch.sampler_top_k[i] = s.k,
            2 => scratch.sampler_top_p[i] = s.p,
            3 => scratch.sampler_min_p[i] = s.p,
            4 => {
                scratch.sampler_top_k[i] = s.k;
                scratch.sampler_top_p[i] = s.p;
            }
            6 => scratch.sampler_top_k[i] = s.num_tokens,
            _ => {}
        }
        scratch.sampler_seeds[i] = s.seed;
        if s.kind == 8 {
            scratch.sampler_label_ids.push(s.token_id);
        } else if s.kind == 9 {
            scratch
                .sampler_label_ids
                .extend_from_slice(slice_from_raw(s.token_ids_ptr, s.token_ids_len));
        }
        scratch
            .sampler_label_indptr
            .push(scratch.sampler_label_ids.len() as u32);
    }

    if sampler_indptr.len() > 1 {
        scratch
            .request_num_samplers
            .extend(sampler_indptr.windows(2).map(|w| w[1] - w[0]));
    }
}

fn touch_forward_desc(f: &PieForwardRequestDesc, scratch: &mut LegacyScratch) {
    scratch.clear();

    let token_ids = slice_from_raw::<u32>(f.token_ids_ptr, f.token_ids_len);
    let position_ids = slice_from_raw::<u32>(f.position_ids_ptr, f.position_ids_len);
    let kv_page_indices = slice_from_raw::<u32>(f.kv_page_indices_ptr, f.kv_page_indices_len);
    let kv_page_indptr = slice_from_raw::<u32>(f.kv_page_indptr_ptr, f.kv_page_indptr_len);
    let kv_last_page_lens = slice_from_raw::<u32>(f.kv_last_page_lens_ptr, f.kv_last_page_lens_len);
    let qo_indptr = slice_from_raw::<u32>(f.qo_indptr_ptr, f.qo_indptr_len);
    let sampling_indices = slice_from_raw::<u32>(f.sampling_indices_ptr, f.sampling_indices_len);
    let sampling_indptr = slice_from_raw::<u32>(f.sampling_indptr_ptr, f.sampling_indptr_len);
    let spec_token_ids = slice_from_raw::<u32>(f.spec_token_ids_ptr, f.spec_token_ids_len);
    let spec_position_ids = slice_from_raw::<u32>(f.spec_position_ids_ptr, f.spec_position_ids_len);
    let spec_indptr = slice_from_raw::<u32>(f.spec_indptr_ptr, f.spec_indptr_len);
    let output_spec_flags = slice_from_raw::<u8>(f.output_spec_flags_ptr, f.output_spec_flags_len);
    let context_ids = slice_from_raw::<u64>(f.context_ids_ptr, f.context_ids_len);

    scratch.mask_byte_indptr.push(0);
    for b in slice_from_raw(f.masks_ptr, f.masks_len) {
        scratch
            .flattened_masks
            .extend_from_slice(slice_from_raw(b.buffer_ptr, b.buffer_len));
        scratch
            .mask_byte_indptr
            .push(scratch.flattened_masks.len() as u32);
    }

    scratch.logit_mask_byte_indptr.push(0);
    let logit_mask_indptr = slice_from_raw::<u32>(f.logit_mask_indptr_ptr, f.logit_mask_indptr_len);
    let logit_masks = slice_from_raw(f.logit_masks_ptr, f.logit_masks_len);
    for w in logit_mask_indptr.windows(2) {
        for i in w[0]..w[1] {
            if let Some(b) = logit_masks.get(i as usize) {
                scratch
                    .logit_masks_flat
                    .extend_from_slice(slice_from_raw(b.buffer_ptr, b.buffer_len));
            }
        }
        scratch
            .logit_mask_byte_indptr
            .push(scratch.logit_masks_flat.len() as u32);
    }

    let samplers = slice_from_raw(f.samplers_ptr, f.samplers_len);
    let sampler_indptr = slice_from_raw::<u32>(f.sampler_indptr_ptr, f.sampler_indptr_len);
    touch_samplers(samplers, sampler_indptr, scratch);
    touch_adapter_bindings(
        slice_from_raw(f.adapter_bindings_ptr, f.adapter_bindings_len),
        scratch,
    );

    black_box((
        token_ids.len(),
        position_ids.len(),
        kv_page_indices.len(),
        kv_page_indptr.len(),
        kv_last_page_lens.len(),
        qo_indptr.len(),
        sampling_indices.len(),
        sampling_indptr.len(),
        spec_token_ids.len(),
        spec_position_ids.len(),
        spec_indptr.len(),
        output_spec_flags.len(),
        context_ids.len(),
        f.single_token_mode,
        f.has_user_mask,
        scratch.sampler_types.len(),
        scratch.adapter_indices.len(),
    ));
}

fn touch_inproc_frame_desc(frame: &PieFrameDesc, scratch: &mut LegacyScratch) {
    black_box(frame.driver_id);
    match frame.payload.kind {
        PIE_REQUEST_PAYLOAD_FORWARD => touch_forward_desc(&frame.payload.forward, scratch),
        PIE_REQUEST_PAYLOAD_COPY => {
            black_box((
                frame.payload.copy.dir,
                frame.payload.copy.srcs_len,
                frame.payload.copy.dsts_len,
            ));
        }
        PIE_REQUEST_PAYLOAD_ADAPTER => {
            black_box((
                frame.payload.adapter.op,
                frame.payload.adapter.adapter_id,
                frame.payload.adapter.path_len,
            ));
        }
        PIE_REQUEST_PAYLOAD_HEALTH => {}
        _ => {}
    }
}

fn send_forward_response(
    send_response: unsafe extern "C" fn(*mut c_void, u32, *const PieResponseFrameDesc),
    ctx: *mut c_void,
    req_id: u32,
    driver_id: u32,
) {
    // Stack arrays mimic the C++ response view scratch. The vtable
    // consumes descriptors synchronously before this function returns.
    let tokens_indptr = [0u32, 1];
    let tokens = [1u32];
    let fwd = PieForwardResponseDesc {
        num_requests: 1,
        tokens_indptr_ptr: tokens_indptr.as_ptr(),
        tokens_indptr_len: tokens_indptr.len(),
        tokens_ptr: tokens.as_ptr(),
        tokens_len: tokens.len(),
        ..Default::default()
    };
    let resp = PieResponseFrameDesc {
        driver_id,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_FORWARD,
            forward: fwd,
            status: PieStatusResponseDesc { status: 0 },
        },
    };
    // SAFETY: response desc and stack arrays live through the
    // synchronous callback.
    unsafe { send_response(ctx, req_id, &resp) };
}

fn send_status_response(
    send_response: unsafe extern "C" fn(*mut c_void, u32, *const PieResponseFrameDesc),
    ctx: *mut c_void,
    req_id: u32,
    driver_id: u32,
) {
    let resp = PieResponseFrameDesc {
        driver_id,
        aborted: 0,
        payload: PieResponsePayloadDesc {
            kind: PIE_RESPONSE_PAYLOAD_STATUS,
            forward: PieForwardResponseDesc::default(),
            status: PieStatusResponseDesc { status: 0 },
        },
    };
    // SAFETY: response desc lives through the synchronous callback.
    unsafe { send_response(ctx, req_id, &resp) };
}

#[derive(Clone, Copy)]
enum InProcDriverMode {
    StatusAckOnly,
    ForwardAckOnly,
    ForwardWithLegacyDemux,
}

fn spawn_inproc_driver(vt: InProcVTable, mode: InProcDriverMode) -> thread::JoinHandle<()> {
    let recv = vt.recv;
    let send_response = vt.send_response;
    let ctx = vt.ctx as usize;
    thread::spawn(move || {
        let ctx = ctx as *mut c_void;
        let mut scratch = LegacyScratch::default();
        loop {
            let mut request_ptr: *const PieFrameDesc = ptr::null();
            let mut req_id: u32 = 0;
            // SAFETY: vtable was produced by the real InProcChannel.
            let rc = unsafe { recv(ctx, &mut request_ptr, &mut req_id) };
            if rc != 0 || request_ptr.is_null() {
                break;
            }
            // SAFETY: InProcChannel keeps the descriptor valid until
            // the matching send_response call below.
            let frame = unsafe { &*request_ptr };
            match mode {
                InProcDriverMode::StatusAckOnly => {
                    send_status_response(send_response, ctx, req_id, frame.driver_id);
                }
                InProcDriverMode::ForwardAckOnly => {
                    send_forward_response(send_response, ctx, req_id, frame.driver_id);
                }
                InProcDriverMode::ForwardWithLegacyDemux => {
                    touch_inproc_frame_desc(frame, &mut scratch);
                    send_forward_response(send_response, ctx, req_id, frame.driver_id);
                }
            }
        }
    })
}

fn bench_inproc_channel(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("inproc_channel");
    for &n in &[16usize, 256, 4096, 16_384] {
        let channel = Arc::new(InProcChannel::with_spin_budget(BALANCED_SPIN_BUDGET_US));
        let vt = channel.ffi_vtable();
        let ctx = vt.ctx;
        let driver = spawn_inproc_driver(vt, InProcDriverMode::ForwardWithLegacyDemux);

        group.bench_function(format!("tokens={n}"), |b| {
            b.iter_batched(
                || make_request(n),
                |req| {
                    let resp = rt
                        .block_on(async { channel.submit(req).await })
                        .expect("inproc submit");
                    black_box(resp);
                },
                BatchSize::SmallInput,
            );
        });

        channel.abort();
        driver.join().expect("inproc driver join");
        // SAFETY: ctx was produced by ffi_vtable and the driver thread is
        // no longer using it.
        unsafe { InProcChannel::release(ctx as *mut c_void) };
    }
    group.finish();
}

fn bench_inproc_polling_channel(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("inproc_polling_channel");
    for &n in &[16usize, 256, 4096, 16_384] {
        let channel = Arc::new(
            InProcPollingChannel::with_capacity_and_spin_budget(1024, LOW_LATENCY_SPIN_BUDGET_US)
                .expect("polling channel"),
        );
        let vt = channel.ffi_vtable();
        let ctx = vt.ctx;
        let driver = spawn_inproc_driver(vt, InProcDriverMode::ForwardWithLegacyDemux);

        group.bench_function(format!("tokens={n}"), |b| {
            b.iter_batched(
                || make_request(n),
                |req| {
                    let resp = rt
                        .block_on(async { channel.submit(req).await })
                        .expect("inproc polling submit");
                    black_box(resp);
                },
                BatchSize::SmallInput,
            );
        });

        channel.abort();
        driver.join().expect("inproc polling driver join");
        // SAFETY: ctx was produced by ffi_vtable and the driver thread is
        // no longer using it.
        unsafe { InProcPollingChannel::release(ctx as *mut c_void) };
    }
    group.finish();
}

fn bench_inproc_channel_breakdown(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("inproc_channel_breakdown");

    for (label, mode) in [
        ("status_ack_only", InProcDriverMode::StatusAckOnly),
        ("forward_ack_only", InProcDriverMode::ForwardAckOnly),
        (
            "forward_with_legacy_demux",
            InProcDriverMode::ForwardWithLegacyDemux,
        ),
    ] {
        for &n in &[16usize, 16_384] {
            let channel = Arc::new(InProcChannel::with_spin_budget(BALANCED_SPIN_BUDGET_US));
            let vt = channel.ffi_vtable();
            let ctx = vt.ctx;
            let driver = spawn_inproc_driver(vt, mode);

            group.bench_function(format!("{label}/tokens={n}"), |b| {
                b.iter_batched(
                    || make_request(n),
                    |req| {
                        let resp = rt
                            .block_on(async { channel.submit(req).await })
                            .expect("inproc submit");
                        black_box(resp);
                    },
                    BatchSize::SmallInput,
                );
            });

            channel.abort();
            driver.join().expect("inproc driver join");
            // SAFETY: ctx was produced by ffi_vtable and the driver thread is
            // no longer using it.
            unsafe { InProcChannel::release(ctx as *mut c_void) };
        }
    }

    group.finish();
}

fn bench_inproc_local_costs(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("inproc_local_costs");

    group.bench_function("tokio_block_on_ready", |b| {
        b.iter(|| {
            rt.block_on(async {
                black_box(());
            });
        });
    });

    group.bench_function("tokio_oneshot_ready_same_thread", |b| {
        b.iter(|| {
            let (tx, rx) = tokio::sync::oneshot::channel::<u32>();
            tx.send(1).expect("oneshot send");
            let value = rt.block_on(rx).expect("oneshot receive");
            black_box(value);
        });
    });

    let (oneshot_req_tx, oneshot_req_rx) =
        mpsc::sync_channel::<tokio::sync::oneshot::Sender<u32>>(1024);
    let oneshot_worker = thread::spawn(move || {
        while let Ok(reply_tx) = oneshot_req_rx.recv() {
            let _ = reply_tx.send(1);
        }
    });
    group.bench_function("tokio_oneshot_cross_thread", |b| {
        b.iter(|| {
            let (reply_tx, reply_rx) = tokio::sync::oneshot::channel::<u32>();
            oneshot_req_tx.send(reply_tx).expect("worker request send");
            let value = rt.block_on(reply_rx).expect("worker response receive");
            black_box(value);
        });
    });
    group.finish();
    drop(oneshot_req_tx);
    oneshot_worker.join().expect("oneshot worker join");

    let mut group = c.benchmark_group("inproc_local_costs");
    group.bench_function("box_frame_from_request/tokens=16", |b| {
        b.iter_batched(
            || make_request(16),
            |req| {
                let frame = Box::new(Frame {
                    driver_id: req.driver_id as u32,
                    payload: req.payload,
                });
                black_box(frame);
            },
            BatchSize::SmallInput,
        );
    });

    let pending = Mutex::new(HashMap::<u32, usize>::new());
    group.bench_function("mutex_hashmap_insert_remove", |b| {
        let mut req_id = 0u32;
        b.iter(|| {
            req_id = req_id.wrapping_add(1);
            let mut guard = pending.lock().expect("pending lock");
            guard.insert(req_id, black_box(1));
            let value = guard.remove(&req_id);
            black_box(value);
        });
    });

    let inbox = Mutex::new(VecDeque::<u32>::new());
    let inbox_cv = Condvar::new();
    group.bench_function("mutex_vecdeque_push_pop_notify", |b| {
        let mut req_id = 0u32;
        b.iter(|| {
            req_id = req_id.wrapping_add(1);
            let mut guard = inbox.lock().expect("inbox lock");
            guard.push_back(req_id);
            inbox_cv.notify_one();
            let value = guard.pop_front();
            black_box(value);
        });
    });

    group.bench_function("response_desc_to_owned_status", |b| {
        b.iter(|| {
            let desc = PieResponseFrameDesc {
                driver_id: DRIVER_ID as u32,
                aborted: 0,
                payload: PieResponsePayloadDesc {
                    kind: PIE_RESPONSE_PAYLOAD_STATUS,
                    forward: PieForwardResponseDesc::default(),
                    status: PieStatusResponseDesc { status: 0 },
                },
            };
            let owned = __pie_response_frame_from_desc(black_box(&desc));
            black_box(owned);
        });
    });

    group.bench_function("response_desc_to_owned_forward", |b| {
        b.iter(|| {
            let tokens_indptr = [0u32, 1];
            let tokens = [1u32];
            let desc = PieResponseFrameDesc {
                driver_id: DRIVER_ID as u32,
                aborted: 0,
                payload: PieResponsePayloadDesc {
                    kind: PIE_RESPONSE_PAYLOAD_FORWARD,
                    forward: PieForwardResponseDesc {
                        num_requests: 1,
                        tokens_indptr_ptr: tokens_indptr.as_ptr(),
                        tokens_indptr_len: tokens_indptr.len(),
                        tokens_ptr: tokens.as_ptr(),
                        tokens_len: tokens.len(),
                        ..Default::default()
                    },
                    status: PieStatusResponseDesc { status: 0 },
                },
            };
            let owned = __pie_response_frame_from_desc(black_box(&desc));
            black_box(owned);
        });
    });

    for &n in &[16usize, 16_384] {
        let frame = make_frame(n);
        group.bench_function(format!("pie_frame_view/tokens={n}"), |b| {
            b.iter(|| {
                let view = pie_frame_view(black_box(&frame));
                black_box(view);
            });
        });

        let view = pie_frame_view(&frame);
        group.bench_function(format!("legacy_demux_only/tokens={n}"), |b| {
            let mut scratch = LegacyScratch::default();
            b.iter(|| {
                touch_inproc_frame_desc(black_box(&view.desc), &mut scratch);
            });
        });

        group.bench_function(
            format!("pie_frame_view_plus_legacy_demux/tokens={n}"),
            |b| {
                let mut scratch = LegacyScratch::default();
                b.iter(|| {
                    let view = pie_frame_view(black_box(&frame));
                    touch_inproc_frame_desc(black_box(&view.desc), &mut scratch);
                    black_box(view);
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Shmem fake Python worker.
// ---------------------------------------------------------------------------

fn unique_name(suffix: &str) -> String {
    format!(
        "/pie_driver_channel_bench_{}_{}_{}",
        suffix,
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time")
            .as_nanos()
    )
}

fn touch_archived_frame(bytes: &[u8]) -> u32 {
    let frame = parse_request(bytes).expect("parse request");
    let mut acc = u32::from(frame.driver_id);
    if let pie_bridge::ArchivedRequestPayload::Forward(fr) = &frame.payload {
        acc = acc.wrapping_add(fr.token_ids.len() as u32);
        acc = acc.wrapping_add(fr.position_ids.len() as u32);
        acc = acc.wrapping_add(fr.qo_indptr.len() as u32);
        acc = acc.wrapping_add(fr.samplers.len() as u32);
        acc = acc.wrapping_add(fr.adapter_bindings.len() as u32);
        for sampler in fr.samplers.iter() {
            acc = acc.wrapping_add(match sampler {
                pie_bridge::ArchivedSampler::Multinomial { .. } => 1,
                pie_bridge::ArchivedSampler::TopK { .. } => 2,
                pie_bridge::ArchivedSampler::TopP { .. } => 3,
                pie_bridge::ArchivedSampler::MinP { .. } => 4,
                pie_bridge::ArchivedSampler::TopKTopP { .. } => 5,
                pie_bridge::ArchivedSampler::Embedding => 6,
                pie_bridge::ArchivedSampler::Dist { .. } => 7,
                pie_bridge::ArchivedSampler::RawLogits => 8,
                pie_bridge::ArchivedSampler::Logprob { .. } => 9,
                pie_bridge::ArchivedSampler::Logprobs { token_ids } => 10 + token_ids.len() as u32,
                pie_bridge::ArchivedSampler::Entropy => 11,
            });
        }
    }
    black_box(acc)
}

fn response_bytes(driver_id: u32) -> Vec<u8> {
    encode_response(&ResponseFrame {
        driver_id,
        aborted: false,
        payload: ResponsePayload::Forward(ForwardResponse {
            num_requests: 1,
            tokens_indptr: vec![0, 1],
            tokens: vec![1],
            ..Default::default()
        }),
    })
    .expect("encode response")
}

fn spawn_shmem_driver(name: &str) -> (Arc<ShmemServer>, thread::JoinHandle<()>, Arc<AtomicBool>) {
    let server = Arc::new(
        ShmemServer::create(
            name,
            8,
            4 * 1024 * 1024,
            8 * 1024 * 1024,
            BALANCED_SPIN_BUDGET_US,
            SCHEMA_HASH,
        )
        .expect("shmem server create"),
    );
    let stop = Arc::new(AtomicBool::new(false));
    let server_for_thread = server.clone();
    let stop_for_thread = stop.clone();
    let handle = thread::spawn(move || {
        while !stop_for_thread.load(Ordering::Relaxed) {
            let Some(lease) = server_for_thread.poll_blocking(Duration::from_millis(50)) else {
                continue;
            };
            // PyLease.payload returns Python bytes, so production pays a
            // copy out of the shmem slot. Keep that cost here.
            let payload = lease.payload().to_vec();
            let driver_id = touch_archived_frame(&payload);
            let resp = response_bytes(driver_id);
            lease.commit(&resp).expect("commit response");
        }
    });
    (server, handle, stop)
}

fn bench_shmem_channel(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("shmem_channel");
    for &n in &[16usize, 256, 4096, 16_384] {
        let name = unique_name(&format!("n{n}"));
        let (server, driver, stop) = spawn_shmem_driver(&name);
        let channel =
            ShmemChannel::open(&name, BALANCED_SPIN_BUDGET_US).expect("shmem channel open");

        group.bench_function(format!("tokens={n}"), |b| {
            b.iter_batched(
                || make_request(n),
                |req| {
                    let resp = rt
                        .block_on(async { channel.submit(req).await })
                        .expect("shmem submit");
                    black_box(resp);
                },
                BatchSize::SmallInput,
            );
        });

        stop.store(true, Ordering::Relaxed);
        server.stop();
        driver.join().expect("shmem driver join");
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_inproc_channel, bench_inproc_polling_channel, bench_inproc_channel_breakdown, bench_inproc_local_costs, bench_shmem_channel
}
criterion_main!(benches);
