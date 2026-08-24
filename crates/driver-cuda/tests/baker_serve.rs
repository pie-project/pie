//! The baker lane through the REAL serve path.
//!
//! `baker-smoke` proved the chain outside the driver: text → `Plan` → lane
//! → `Program` → GPU → argmax, one request, one fire, no scheduler. This
//! binary asks the question that smoke cannot: does the same `Program`
//! answer a completion when it is fired by the thing that actually serves —
//! through `Shell::open` / `load_model` / `launch`, on the driver's own KV
//! pools, recurrent slabs and runtime planes, with a request lifecycle
//! around it.
//!
//! # The BANKED oracle
//!
//! This file used to A/B against the live legacy path: two legs in one
//! process, `PIE_BAKER=1` for one and unset for the other, asserted equal.
//! The legacy leg is GONE — `R2` deleted the legacy fire path, so there is
//! no second leg left to compare against and no knob to select one.
//!
//! What survived is the ANSWER. The three chains below were taken from
//! that A/B while both legs still ran, at commit `49fa5c588`, on an L40S,
//! against `Qwen/Qwen3.5-0.8B-Base`: the legacy driver's own greedy output,
//! which `real_hybrid.rs` had in turn held to the *transformers* oracle
//! under its calibrated bar. They are committed here as constants
//! ([`BANKED`], [`BANKED_APART_R1`]) with that provenance, and the baker
//! leg now runs ALONE against them.
//!
//! Equality and not a tolerance, unchanged: a token is an argmax, and an
//! argmax that moves means a real disagreement, not a reduction order.
//!
//! What the banking costs, stated: an oracle in the tree cannot notice that
//! the WEIGHTS moved. If the checkpoint is ever republished, these numbers
//! are stale and the failure will read as a baker regression. The
//! provenance line above is what an investigator checks first.
//!
//! # Why the prompt is ingested one token at a time
//!
//! The lane this fires is the DECODE lane and nothing else: `qo_one` is the
//! fact that selected it, so every fire carries exactly one row. A prompt
//! of seven is therefore seven fires that share the caches — which is what
//! autoregressive decoding IS, and what makes the last fire's logits
//! comparable to a seven-token prefill's last row. The prefill lane needs
//! `ssm.gated_delta_chunked`, which no cuda routine claims yet (W2), and
//! `program::bound` says so by refusing that lane rather than binding it
//! wrong.
//!
//! Run it:
//!
//! ```text
//! cargo test -p driver-cuda --features cuda-13,abi --test baker_serve -- --nocapture
//! ```

#![allow(clippy::print_stderr, clippy::print_stdout)]

mod common;

use common::{device_or_skip, gpu_guard};

/// The checkpoint this gate is written against.
///
/// `-Base` and not the instruct release, matching `serve.rs:2141` and
/// `real_hybrid.rs:290`: the committed oracle
/// (`tests/oracle/real_decode/qwen3_5_0_8b.json`) states its provenance as
/// `Qwen/Qwen3.5-0.8B-Base`, and a test that compared against it from the
/// other checkpoint would be measuring the wrong weights. This test's own
/// gate is leg-A-against-leg-B and would pass on either — the point of
/// pinning it is that BOTH legs read the same bytes.
const CACHE_DIR: &str = "models--Qwen--Qwen3.5-0.8B-Base";

/// The new-catalog row that checkpoint bridges to.
const SKU: &str = "qwen35-d0.8b-bf16-kv-bf16";

/// The newest cached snapshot directory, or `None`.
///
/// The `.index.json` arm is not optional politeness: neither Qwen3.5-0.8B
/// snapshot in this cache carries a plain `model.safetensors` — the shard
/// is `model.safetensors-00001-of-00001.safetensors` and is only reachable
/// through the index. A gate that probed for `model.safetensors` alone
/// would skip forever and print `test result: ok` while measuring nothing.
fn snapshot() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let snaps = std::path::PathBuf::from(home)
        .join(format!(".cache/huggingface/hub/{CACHE_DIR}/snapshots"));
    std::fs::read_dir(&snaps).ok()?.filter_map(Result::ok).find_map(|e| {
        let d = e.path();
        (d.join("model.safetensors").is_file()
            || d.join("model.safetensors.index.json").is_file())
        .then_some(d)
    })
}

/// The boot document both legs open with.
///
/// `[model] config` points at the snapshot's OWN `config.json` — the read
/// `load_impl` needs for the one field a catalog row does not hold, the
/// declared quantization. Not an agent-scratchpad descriptor: `serve.rs`
/// records what that pattern costs, a gate that skipped forever on every
/// machine but one and passed while doing it.
fn boot(snap: &std::path::Path) -> String {
    format!(
        "[model]\nconfig = \"{}\"\n",
        snap.join("config.json").display()
    )
}

/// Stage 1 + 2: the lane builds, and every call in it resolves at LOAD.
///
/// This is the `PlanSource` gate, and it is separate from the fire gate on
/// purpose: a load that traces, produces, uploads, joins and resolves is a
/// complete claim on its own, and when the fire gate fails this one says
/// whether the failure is upstream of the fire.
///
/// NO KNOB. `PIE_BAKER` is retired: the baker lane is the only lane, so a
/// load that cannot build one REFUSES rather than falling back. `armed`
/// after a successful `load_model` is therefore a tautology — asserted
/// anyway, because it is the tautology this whole file rests on and a
/// `Shell` that returned `Ok` with no lane would be the one bug this
/// binary could not otherwise see.
#[test]
fn the_baker_lane_builds_and_resolves_at_load() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("baker load") else {
        return;
    };
    let Some(snap) = snapshot() else {
        eprintln!("skipped: no cached {CACHE_DIR}");
        return;
    };

    let broker = driver_api::completion::CompletionBroker::new();
    let mut shell =
        driver_cuda::serve::Shell::open(boot(&snap).as_bytes(), broker).expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    shell.load_model(&load).expect("the snapshot loads");
    assert!(
        shell.baker_is_armed(),
        "the load answered Ok for `{SKU}` and armed no lane; the load's own \
         stderr names every unresolved call",
    );
    drop(shell);
}

// `an_unset_knob_builds_no_lane_at_all` STOOD HERE and is deleted with the
// knob it tested. It asserted that `PIE_BAKER` unset left `state.baker`
// `None` — the cheap half of "the legacy path is untouched". There is no
// legacy path to leave untouched and no knob to unset: `Boot::baker` is
// gone, the lane is built on every load, and a load that cannot build one
// answers `Err` instead of serving something else.

// ── The gate ────────────────────────────────────────────────────────────

use driver_api::completion::CompletionBroker;
use driver_api::local::{
    ChannelBinding, InstanceBinding, PIE_CHANNEL_DTYPE_F32, PIE_CHANNEL_EXTERN_NONE,
    PIE_CHANNEL_HOST_ROLE_READER, PIE_RS_FLAG_RESET, PIE_STATUS_OK,
};
use driver_api::{ChannelRegistrationPlan, EncodedMask, FrameSubmission, InstanceBindingPlan,
    LaunchPlan, StepSubmission};
use driver_cuda::serve::Shell;

/// Qwen3.5-0.8B's vocabulary. Restated rather than read off the caps
/// because the ring's cell size is what it sizes, and a cell that
/// disagreed with the driver would be found by `deliver_logits` refusing to
/// match a reader channel rather than by anything here.
const VOCAB: usize = 248_320;
/// Tokens generated greedily, per leg.
const STEPS: usize = 8;
/// The KV page size. Not a knob (`boot::KV_PAGE_SIZE`).
const PAGE: u32 = 16;
/// Ring cells. Each fire publishes one and the loop consumes it before the
/// next, so this never wraps — the margin is so a wrap would be a bug
/// rather than the design.
const CAPACITY: u32 = 7;
const CELLS: u64 = CAPACITY as u64 + 1;

/// **The banked chain.** What the LEGACY driver generated from [`PROMPT`],
/// greedily, eight tokens, on this checkpoint.
///
/// # Provenance
///
/// Taken at commit `49fa5c588` on an NVIDIA L40S, from the A/B this file
/// used to run live: the legacy fire path's own argmax chain, printed as
/// `legacy [15, 16, 17, 18, 19, 20, 21, 22]` while both legs still
/// existed. That path's answers were in turn held to the *transformers*
/// oracle by `real_hybrid.rs` under its calibrated bar (top-5 membership,
/// logits within 1.25), so the chain below is two hops from HF and one
/// from a driver that no longer exists.
///
/// The legacy leg was deleted with the path (R2, "the baker path is THE
/// path"). This constant is what it left behind.
const BANKED: [u32; STEPS] = [15, 16, 17, 18, 19, 20, 21, 22];

/// The second prompt of the DIFFERENT-prompts batch, `[2, 3, 5, 7, 11, 13,
/// 17]`, banked the same way and from the same run:
/// `apart legacy r1 [18, 19, 20, 21, 22, 23, 24, 15]`.
///
/// Its row 0 is [`BANKED`] — the batch's first request is [`PROMPT`], and
/// a batched row must answer what the same prompt answers alone.
const BANKED_APART_R1: [u32; STEPS] = [18, 19, 20, 21, 22, 23, 24, 15];

/// The prompt, ingested ONE TOKEN AT A TIME.
///
/// `oracle/real_decode/qwen3_5_0_8b.json`'s `prompt_ids`, so the tokens
/// this drives with are the ones the family's other real-weight gates use.
/// Nothing here compares against that oracle's argmax — that is a
/// this-tree-versus-transformers claim and needs the calibrated bar
/// `real_hybrid.rs:19-41` argues for. This gate's claim is narrower and
/// sharper: two paths through ONE tree agree exactly.
const PROMPT: &[u32] = &[1, 2, 3, 5, 7, 11, 13];

/// Run one leg: open a shell, load, and generate `STEPS` tokens greedily,
/// one fire per token.
///
/// `tag` is the program hash, distinct per leg — it is the dedup key for
/// `register_program`, and two legs sharing it would have the second bind
/// the first's registration. There is only one KIND of leg now (the baker
/// lane is the only lane), but a run still opens more than one shell, so
/// the distinctness rule stands.
fn leg(snap: &std::path::Path, tag: u64) -> (Vec<u32>, Vec<f32>) {
    let broker = CompletionBroker::new();
    let mut d = Shell::open(boot(snap).as_bytes(), broker.clone()).expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.to_path_buf(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    d.load_model(&load).expect("the snapshot loads");
    assert!(
        d.baker_is_armed(),
        "the load answered Ok and armed no lane; there is no other path for \
         this leg to have quietly taken, so this can only be a Shell that \
         served with nothing to fire",
    );

    // The reader channel the raw-logits fallback publishes into. The
    // program is registered with NO bytecode on purpose: `plan.executable`
    // is false, nothing is compiled, and the fire falls back to publishing
    // the vocabulary rather than a sampled token — which is what lets this
    // test do its own greedy argmax and compare tokens.
    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 1,
        shape: vec![VOCAB as u32],
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: CAPACITY,
        reader_wait_id: 3,
        writer_wait_id: 4,
        extern_name: Vec::new(),
    };
    let chb: ChannelBinding = d.register_channel(&ch).expect("the channel registers");
    let prog = driver_api::ProgramRegistration {
        program_hash: tag,
        ..Default::default()
    };
    // THE ID `register_program` ANSWERED, not a literal `0`.
    // `bind_instance` refuses a `program_id` its map does not hold, and
    // `next_id` starts at 1 — so a hardcoded zero binds nothing, leaves
    // `instance_id` at its default, and `deliver_logits` then matches no
    // instance and publishes nothing. The symptom is a ring of zeros and an
    // argmax of 0 every step, which is a wrong answer rather than an error.
    let program_id = d.register_program(&prog).expect("the program registers");
    let inst = InstanceBindingPlan {
        driver_id: 0,
        program_id,
        requested_instance_id: 0,
        pacing_wait_id: 0,
        channel_ids: vec![1],
        seed_values: Vec::new(),
        geometry_class: driver_api::GeometryClass::Host,
    };
    let binding: InstanceBinding = d.bind_instance(&inst).expect("the instance binds");
    let instance_ids = vec![binding.instance_id];

    // Sized ONCE, for the whole run. `kv_pools_for` grows the pool when a
    // fire asks for more pages than it holds, and growth REPLACES pages
    // without migrating them (`fire/launch.rs:1193-1195`) — so a chain that
    // let the pool grow mid-run would silently lose its own history.
    let total_pages = ((PROMPT.len() + STEPS) as u32).div_ceil(PAGE).max(1);
    let all_pages: Vec<u32> = (0..total_pages).collect();

    let argmax_of = |i: u64| -> u32 {
        // SAFETY: the mirror is `cell_bytes * (capacity + 1)` of host
        // memory the channel registration allocated and the binding
        // published; `i % CELLS` keeps the read inside it.
        let cell = unsafe {
            std::slice::from_raw_parts(
                (chb.mirror_base as *const f32).add((i % CELLS) as usize * VOCAB),
                VOCAB,
            )
        };
        cell.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(t, _)| t as u32)
            .expect("a non-empty vocabulary")
    };

    // One fire, one token. `first` carries the recurrent-state RESET: a
    // fresh sequence zeroes its gdn slabs, and every step after CONTINUES
    // them. Getting this wrong is invisible — no error, just wrong tokens —
    // and it is the one field a decode chain must not default.
    let fire = |d: &mut Shell, kv_len: u32, token: u32, position: u32, first: bool| {
        let pages_used = kv_len.div_ceil(PAGE).max(1);
        let mut cell = driver_api::local::TerminalCell::pending();
        let cell_ptr: *mut driver_api::local::TerminalCell = &mut cell;
        let step = StepSubmission {
            plan: LaunchPlan {
                token_ids: vec![token],
                position_ids: vec![position],
                kv_page_indices: all_pages[..pages_used as usize].to_vec(),
                kv_page_indptr: vec![0, pages_used],
                kv_last_page_lens: vec![kv_len - (pages_used - 1) * PAGE],
                qo_indptr: vec![0, 1],
                rs_slot_ids: vec![0],
                rs_slot_flags: vec![if first { PIE_RS_FLAG_RESET } else { 0 }],
                ..Default::default()
            },
            // One request: one roster row, one CSR entry, one cell. The
            // token count belongs to `qo_indptr`, which partitions TOKENS;
            // these three partition REQUESTS.
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::local::PIE_GEOMETRY_CLASS_HOST],
            terminal_cells: vec![cell_ptr],
            ..Default::default()
        };
        let frame = FrameSubmission {
            instance_ids: instance_ids.clone(),
            required_kv_pages: total_pages,
            steps: vec![step],
            ..Default::default()
        };
        // A completion PER FIRE: a settled one stays settled, so one reused
        // across a chain fences only the first.
        let (target, completion) = broker.launch_completion(1);
        assert_eq!(
            d.launch(&frame, target).map_or_else(|s| s, |()| PIE_STATUS_OK),
            PIE_STATUS_OK,
            "the fire is accepted",
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            if let Some(settled) = completion.check() {
                settled.expect("the fire completed");
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the fire never completed",
            );
            std::thread::yield_now();
        }
    };

    // The ring's head is the engine's half of the protocol: word 0. The
    // driver owns the tail.
    let words = chb.word_base as *mut u64;
    let mut consumed = 0u64;
    // The whole vocabulary of the row the oracle describes — kept, not just
    // its argmax, because `the_last_prompt_row_matches_the_transformers_
    // oracle` needs five logits and five probes out of it. Filled by the
    // last PROMPT fire and never overwritten.
    let mut oracle_row: Vec<f32> = Vec::new();
    let mut advance = |keep: bool| -> u32 {
        if keep {
            // SAFETY: as `argmax_of` — the mirror is `cell_bytes *
            // (capacity + 1)` of host memory the registration allocated, and
            // `% CELLS` keeps the read inside it.
            oracle_row = unsafe {
                std::slice::from_raw_parts(
                    (chb.mirror_base as *const f32).add((consumed % CELLS) as usize * VOCAB),
                    VOCAB,
                )
            }
            .to_vec();
        }
        let got = argmax_of(consumed);
        consumed += 1;
        // SAFETY: `word_base` is the channel's four-word control block,
        // published by the registration and alive for the channel's life.
        unsafe { words.write_volatile(consumed) };
        got
    };

    // The prompt, one fire each. THIS IS THE DECODE LANE and nothing else:
    // seven fires that share the caches, which is what autoregressive
    // ingest IS. It costs seven forward passes where a prefill costs one,
    // and it buys a gate that does not wait on W2's
    // `ssm.gated_delta_chunked`.
    let mut next = 0u32;
    for (i, &token) in PROMPT.iter().enumerate() {
        let position = i as u32;
        fire(&mut d, position + 1, token, position, i == 0);
        next = advance(i == PROMPT.len() - 1);
    }

    // Then the generation: feed each argmax back in.
    let mut generated = vec![next];
    for s in 0..STEPS - 1 {
        let position = PROMPT.len() as u32 + s as u32;
        fire(&mut d, position + 1, next, position, false);
        next = advance(false);
        generated.push(next);
    }
    drop(d);
    (generated, oracle_row)
}

/// **The gate**: the baker lane answers a completion, and it answers the
/// chain the legacy path answered before it was deleted.
#[test]
fn the_baker_lane_generates_the_banked_eight_tokens() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("baker chain") else {
        return;
    };
    let Some(snap) = snapshot() else {
        eprintln!("skipped: no cached {CACHE_DIR}");
        return;
    };

    let (baker, _) = leg(&snap, 0xBA_5E_00);
    eprintln!("baker  {baker:?}");
    eprintln!("banked {BANKED:?}");

    // EQUALITY, not a tolerance. The bar was argued for when the other side
    // of this comparison was a live second leg, and banking the leg does not
    // weaken it: a token is an argmax, and an argmax that moves is a real
    // disagreement rather than a reduction order.
    assert_eq!(
        baker.as_slice(),
        BANKED.as_slice(),
        "the baker lane disagrees with the chain banked from the legacy \
         path at 49fa5c588",
    );
    // A decoder stuck on one token passes a naive equality A/B. This is
    // `serve.rs:1856-1859`'s guard, and it is worth stealing verbatim.
    assert!(
        baker.iter().skip(1).any(|&t| t != baker[0]),
        "eight steps that repeat one token would be a broken chain: {baker:?}",
    );
}

// ── The stretch: two requests in one fire ───────────────────────────────

/// Run one leg with TWO requests batched into every fire.
///
/// The single-request leg above proves the lane computes; this proves it
/// computes per ROW. Everything that could quietly collapse a batch to its
/// first row is exercised: the arena's row cut, the KV page CSR, the
/// recurrent slot table, `qo_indptr`'s request count, and the logits
/// repitch.
///
/// The prompts are the caller's because the two shapes measure different
/// failures, and both are real:
///
/// * DIFFERENT prompts catch a batch that collapsed to row 0 — two rows
///   that generated one continuation. Identical prompts would agree even
///   then.
/// * IDENTICAL prompts catch the opposite, and it is the failure W1
///   actually measured: row 0 correct and row 1 garbage
///   (`[220, 16, 96382, ...]` against `[15, 16, 17, ...]`). Identical
///   inputs MUST produce identical outputs, so the row-1 answer is checkable
///   against the row-0 answer with no oracle at all.
fn leg_pair(snap: &std::path::Path, tag: u64, prompts: [&[u32]; 2]) -> Vec<Vec<u32>> {
    const R: usize = 2;

    let broker = CompletionBroker::new();
    let mut d = Shell::open(boot(snap).as_bytes(), broker.clone()).expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.to_path_buf(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    d.load_model(&load).expect("the snapshot loads");
    assert!(d.baker_is_armed(), "the load armed no lane");

    // One reader channel and one instance per request: `deliver_logits`
    // matches a request to its channel through its INSTANCE, so two
    // requests sharing one instance would publish both answers into one
    // ring and the second would overwrite the first.
    let mut mirrors = Vec::new();
    let mut instance_ids = Vec::new();
    for r in 0..R {
        let cid = r as u64 + 1;
        let ch = ChannelRegistrationPlan {
            driver_id: 0,
            channel_id: cid,
            shape: vec![VOCAB as u32],
            dtype: PIE_CHANNEL_DTYPE_F32,
            host_role: PIE_CHANNEL_HOST_ROLE_READER,
            seeded: false,
            extern_dir: PIE_CHANNEL_EXTERN_NONE,
            capacity: CAPACITY,
            // Nonzero and DISTINCT across channels, or the shared validator
            // refuses the registration.
            reader_wait_id: 10 + r as u64 * 2,
            writer_wait_id: 11 + r as u64 * 2,
            extern_name: Vec::new(),
        };
        let chb = d.register_channel(&ch).expect("the channel registers");
        let prog = driver_api::ProgramRegistration {
            program_hash: tag + r as u64,
            ..Default::default()
        };
        let program_id = d.register_program(&prog).expect("the program registers");
        let inst = InstanceBindingPlan {
            driver_id: 0,
            program_id,
            requested_instance_id: 0,
            pacing_wait_id: 0,
            channel_ids: vec![cid],
            seed_values: Vec::new(),
            geometry_class: driver_api::GeometryClass::Host,
        };
        let b = d.bind_instance(&inst).expect("the instance binds");
        mirrors.push((chb.mirror_base, chb.word_base));
        instance_ids.push(b.instance_id);
    }

    // DISJOINT PAGES PER REQUEST. The frame states one flat page list and a
    // CSR that partitions it; two requests pointing at one page would share
    // a KV history and the answers would drift together rather than apart.
    let per = ((prompts[0].len() + STEPS) as u32).div_ceil(PAGE).max(1);
    let total_pages = per * R as u32;

    let fire = |d: &mut Shell, kv_len: u32, tokens: &[u32], positions: &[u32], first: bool| {
        let used = kv_len.div_ceil(PAGE).max(1);
        let mut indices = Vec::new();
        let mut indptr = vec![0u32];
        let mut lens = Vec::new();
        for r in 0..R {
            indices.extend((0..used).map(|p| r as u32 * per + p));
            indptr.push(indices.len() as u32);
            lens.push(kv_len - (used - 1) * PAGE);
        }
        let mut cells = [
            driver_api::local::TerminalCell::pending(),
            driver_api::local::TerminalCell::pending(),
        ];
        // Two live `&mut`s out of one array: the cells must be DISTINCT
        // across a frame's members, and `split_at_mut` is how a frame with
        // two of them gets both.
        let (a, b) = cells.split_at_mut(1);
        let step = StepSubmission {
            plan: LaunchPlan {
                token_ids: tokens.to_vec(),
                position_ids: positions.to_vec(),
                kv_page_indices: indices,
                kv_page_indptr: indptr,
                kv_last_page_lens: lens,
                // One token row per request.
                qo_indptr: vec![0, 1, 2],
                // One recurrent slot each: row r continues slot r.
                rs_slot_ids: vec![0, 1],
                rs_slot_flags: vec![if first { PIE_RS_FLAG_RESET } else { 0 }; R],
                ..Default::default()
            },
            roster_rows: vec![0, 1],
            sub_batch_indptr: vec![0, 2],
            sub_batch_class: vec![driver_api::local::PIE_GEOMETRY_CLASS_HOST],
            terminal_cells: vec![&mut a[0], &mut b[0]],
            ..Default::default()
        };
        let frame = FrameSubmission {
            instance_ids: instance_ids.clone(),
            required_kv_pages: total_pages,
            steps: vec![step],
            ..Default::default()
        };
        let (target, completion) = broker.launch_completion(1);
        assert_eq!(
            d.launch(&frame, target).map_or_else(|s| s, |()| PIE_STATUS_OK),
            PIE_STATUS_OK,
            "the two-request fire is accepted",
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            if let Some(settled) = completion.check() {
                settled.expect("the fire completed");
                return;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the fire never completed"
            );
            std::thread::yield_now();
        }
    };

    let mut consumed = 0u64;
    let mut next = [0u32; R];
    let advance = |consumed: &mut u64| -> [u32; R] {
        let mut out = [0u32; R];
        for (r, &(mirror, words)) in mirrors.iter().enumerate() {
            // SAFETY: as in `leg` — the registration published both bases,
            // and `% CELLS` keeps the read inside the mirror.
            let cell = unsafe {
                std::slice::from_raw_parts(
                    (mirror as *const f32).add((*consumed % CELLS) as usize * VOCAB),
                    VOCAB,
                )
            };
            out[r] = cell
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(t, _)| t as u32)
                .expect("a non-empty vocabulary");
            // SAFETY: the channel's four-word control block, published by
            // the registration and alive for the channel's life.
            unsafe { (words as *mut u64).write_volatile(*consumed + 1) };
        }
        *consumed += 1;
        out
    };

    for i in 0..prompts[0].len() {
        let tokens: Vec<u32> = prompts.iter().map(|p| p[i]).collect();
        let positions = vec![i as u32; R];
        fire(&mut d, i as u32 + 1, &tokens, &positions, i == 0);
        next = advance(&mut consumed);
    }
    let mut generated: Vec<Vec<u32>> = next.iter().map(|&t| vec![t]).collect();
    for s in 0..STEPS - 1 {
        let position = prompts[0].len() as u32 + s as u32;
        fire(&mut d, position + 1, &next, &vec![position; R], false);
        next = advance(&mut consumed);
        for (r, &t) in next.iter().enumerate() {
            generated[r].push(t);
        }
    }
    drop(d);
    generated
}

/// **The gate W1 could not meet**: two requests in one fire, both rows
/// right.
///
/// # What this used to measure
///
/// The legacy path was correct at two rows: given IDENTICAL prompts it
/// answered both rows the same, as identical inputs must. The baker lane
/// answered row 0 correctly and row 1 as garbage — `[220, 16, 96382,
/// 96738, ...]` against the legacy `[15, 16, 17, 18, ...]` on both rows —
/// so `fire::launch::baker_fire` REFUSED `rows > 1` and this test was
/// `#[ignore]`d as the running repro.
///
/// # What fixed it
///
/// Not a stride on the mark. `baker::marks::Rect::column` was localised
/// correctly — a mark carries `{ptr, rows, width}` and no stride, so the cut
/// a packed operand's half needs reported the CUT's width as its row stride
/// when the bytes stride by the PACKED width — but the fix was to stop
/// cutting. `ssm.gdn_prep` and `ssm.gated_delta` are claim bodies now, each
/// taking exactly the packed rows its declaration states, and every
/// packed→compact cut happens in a kernel that is told the packing (the
/// same `qwen_gdn_v_gates` the chunked point already used, plus a
/// `qwen_gdn_ba_gates` writer so the fused decay row is packed the way its
/// readers read it). `Rect::column` is gone, and so is the refusal.
///
/// # The two shapes, and why both
///
/// Identical prompts are the repro verbatim: row 1 must equal row 0 AND
/// both must equal [`BANKED`]. Different prompts are the complement: the
/// two rows must NOT agree, or a batch that quietly read row 0 twice would
/// pass the first half. Both prompts generate MOVING chains ([`BANKED`] and
/// [`BANKED_APART_R1`]), so the second shape also says that row 1's eight
/// steps are eight real forward passes and not one answer repeated.
///
/// The legacy legs are BANKED (see the file header): they ran, they printed
/// these four rows, and then the path they ran on was deleted.
#[test]
fn two_batched_requests_match_the_banked_rows() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("baker multi-request chain") else {
        return;
    };
    let Some(snap) = snapshot() else {
        eprintln!("skipped: no cached {CACHE_DIR}");
        return;
    };

    const SAME: [&[u32]; 2] = [PROMPT, PROMPT];
    const APART: [&[u32]; 2] = [PROMPT, &[2, 3, 5, 7, 11, 13, 17]];

    let baker_same = leg_pair(&snap, 0xBA_5E_10, SAME);
    let baker_apart = leg_pair(&snap, 0xBA_5E_20, APART);

    eprintln!("same   r0 {:?}\nsame   r1 {:?}", baker_same[0], baker_same[1]);
    eprintln!("apart  r0 {:?}\napart  r1 {:?}", baker_apart[0], baker_apart[1]);

    // THE REPRO. Row 1 against row 0 first, because that assertion needs no
    // oracle at all — banked or live: identical inputs, identical outputs,
    // or the batch is wrong.
    assert_eq!(
        baker_same[0], baker_same[1],
        "two identical prompts generated different continuations; row 1 of \
         the batch is not being computed: {baker_same:?}",
    );
    assert_eq!(
        baker_same,
        vec![BANKED.to_vec(), BANKED.to_vec()],
        "a two-request batch of identical prompts disagrees with the banked \
         chain",
    );

    // THE COMPLEMENT. Two different prompts must generate two different
    // continuations, or the executor read row 0 twice — the other way a
    // batch can be silently wrong.
    assert_eq!(
        baker_apart,
        vec![BANKED.to_vec(), BANKED_APART_R1.to_vec()],
        "a two-request batch of different prompts disagrees with the banked \
         rows",
    );
    assert_ne!(
        baker_apart[0], baker_apart[1],
        "two different prompts generated the same continuation; the batch \
         collapsed to one row: {baker_apart:?}",
    );

    // A decoder stuck on one token passes every equality above. The
    // single-request gate's guard, on every row of both shapes.
    for row in baker_same.iter().chain(&baker_apart) {
        assert!(
            row.iter().skip(1).any(|t| *t != row[0]),
            "eight steps that repeat one token would be a broken chain: {row:?}",
        );
    }
}

// ── The transformers oracle, ported off `real_hybrid.rs` ────────────────

/// **The third-party gate**: this tree against *transformers*, through the
/// serve path.
///
/// # What this is a port of
///
/// `tests/real_hybrid.rs` — E-gate family #1's parity anchor. It built the
/// qwen3_5 hybrid's forward by hand (`lower()`, a `MapResolver`, a
/// hand-bound `AttnCtx`/`GdnCtx`/`DispatchCtx`), prefilled seven tokens
/// through `bind::run`, and held the last row's logits to
/// `tests/oracle/real_decode/qwen3_5_0_8b.json`. Every piece of machinery in
/// its first 850 lines is deleted, so the test could not survive as written.
///
/// The CLAIM survives, and it is the one claim `baker_serve`'s banked chains
/// cannot make: those compare this tree to ITSELF at an earlier commit, and
/// an error introduced before that commit would be banked along with
/// everything else. This compares to a third party that never ran this code.
///
/// # What moved, and what did not
///
/// * The path. The forward is fired by the SERVE path — `Shell::launch`,
///   the driver's own pools, the delivery tail — which is what the legacy
///   version could not do and what makes this a gate on the thing that
///   ships.
/// * The fire class. The oracle is the last row of a seven-token PREFILL;
///   this ingests those seven tokens one at a time through the DECODE lane.
///   That is autoregressive ingest, and the last fire's logits are the same
///   row: `qwen35-d0.8b`'s prefill lane still refuses
///   (`ssm.gated_delta_chunked`, W2's remaining half), and refusing to
///   compare would have deleted the claim instead of moving it.
/// * The bar, verbatim. `real_hybrid.rs:19-41` argued for it and the
///   argument is unchanged, so it is restated rather than re-derived:
///   transformers bf16 ≈ transformers fp32 here (top logits within 0.03),
///   so HF is a clean reference and the gap is OURS; the gap is bf16 arenas
///   between every launch plus 18 GDN layers of L2-norm/exp/softplus
///   nonlinearity, tracked to ~5% residual norm at every depth by HF
///   `hidden_states` bisection; and the C++ driver's own harness explicitly
///   refused argmax equality for this family, because "with bf16 +
///   flashinfer's R-dependent prefill tiling, the very first decoded token
///   can legitimately flip". So: our argmax is one of HF's top-5, every HF
///   top-5 id sits in our top-8, top-5 logits within 1.25, probes within
///   0.6. A structural bug (a swapped binding, a wrong state slab) blows all
///   four; bf16 accumulation passes them.
/// * The precision of the READ. `real_hybrid` read bf16 off the device and
///   widened it itself; this reads the f32 cell the delivery already
///   widened into. Same bits, one fewer copy.
#[test]
fn the_last_prompt_row_matches_the_transformers_oracle() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("transformers oracle") else {
        return;
    };
    let Some(snap) = snapshot() else {
        eprintln!("skipped: no cached {CACHE_DIR}");
        return;
    };

    let reference: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("tests/oracle/real_decode/qwen3_5_0_8b.json"),
        )
        .expect("reference file"),
    )
    .expect("reference json");

    // The oracle's own prompt, restated as a check rather than assumed: it
    // is `PROMPT`, and a drift in either would compare the wrong row.
    let oracle_prompt: Vec<u32> = reference["prompt_ids"]
        .as_array()
        .expect("prompt_ids")
        .iter()
        .map(|v| u32::try_from(v.as_u64().expect("id")).expect("id"))
        .collect();
    assert_eq!(
        oracle_prompt.as_slice(),
        PROMPT,
        "the oracle describes a different prompt than this file drives",
    );

    let (_, row) = leg(&snap, 0x0AC1_E000);
    assert_eq!(row.len(), VOCAB, "the reader channel published no row");

    let mut all: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("ours top8: {:?}", &all[..8]);

    let ids = |k: &str| -> Vec<usize> {
        reference[k]
            .as_array()
            .unwrap_or_else(|| panic!("{k}"))
            .iter()
            .map(|v| usize::try_from(v.as_u64().expect("id")).expect("id"))
            .collect()
    };
    let vals = |k: &str| -> Vec<f32> {
        reference[k]
            .as_array()
            .unwrap_or_else(|| panic!("{k}"))
            .iter()
            .map(|v| v.as_f64().expect("v") as f32)
            .collect()
    };

    let ids5 = ids("top5_ids");
    let vals5 = vals("top5_logits");
    let our_argmax = all[0].0;
    assert!(
        ids5.contains(&our_argmax),
        "our argmax {our_argmax} ({}) is not one of HF's top-5 {ids5:?}",
        all[0].1,
    );
    let our_top8: Vec<usize> = all[..8].iter().map(|(t, _)| *t).collect();
    for t in &ids5 {
        assert!(
            our_top8.contains(t),
            "HF top-5 token {t} missing from our top-8 {our_top8:?}",
        );
    }
    for (t, hf) in ids5.iter().zip(&vals5) {
        let ours = row[*t];
        assert!(
            (ours - hf).abs() < 1.25,
            "top-5 token {t}: ours {ours} vs HF {hf}",
        );
    }
    for (t, hf) in ids("probe_ids").iter().zip(&vals("probe_logits")) {
        let ours = row[*t];
        assert!(
            (ours - hf).abs() < 0.6,
            "probe token {t}: ours {ours} vs HF {hf}",
        );
    }
}

/// **The mask gate**: a user attention mask is refused BY THE TEXT, and the
/// same shell answers the causal fire on either side of the refusal.
///
/// # What this gate is, and what it deliberately is not
///
/// It is not "a masked fire answers". No catalog row can produce one, which
/// is the finding this test records rather than the one it hides:
///
/// * `masked` is a fact exactly ONE family declares (`gemma_4`'s `Facts`).
///   Every other text has one attention arm and it is causal.
/// * gemma-4 does not reach a fire on this driver at all -- `Deployment::of`
///   refuses the checkpoint at `load_model` ("more than one kv plane width
///   across its layers", the 256/512 split between its sliding and global
///   layers).
/// * and past that, gemma's masked statements carry its 512-wide window,
///   which `attention.masked`'s claim body refuses BY NAME
///   (`kernels-cuda/src/attn.rs`): the masked arm reads the mask and nothing
///   else, so serving both would lose one of the two orderings.
///
/// So what CHANGED is which answer this driver refuses with, and that is
/// what this asserts. The refusal used to be the FLAG: `has_user_mask` set
/// was refused before the lane was even picked, on the grounds that no arm
/// read a staged mask. One does now -- `attention.masked` is a claim body
/// reading the raise door, and `publish_seam_pins` stages
/// `element_mask::from_words` for a frame that carries a table. So the
/// question moved to `baker::word_of`, where the lane is picked: a text with
/// no `masked` fact cannot express the request, and the refusal says that.
///
/// # Why the sandwich
///
/// The mask refusal is per-FRAME and lands before a byte is allocated. A
/// fire on each side of it, from the SAME shell, is what says the refusal
/// left no state behind -- no half-grown pool, no consumed ring cell, no
/// stranded completion. The two argmaxes are the ones a two-token chain
/// gives, so a refusal that had quietly advanced the KV would move them.
#[test]
fn a_user_mask_is_refused_by_the_text_and_not_by_the_flag() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("baker user mask") else {
        return;
    };
    let Some(snap) = snapshot() else {
        eprintln!("skipped: no cached {CACHE_DIR}");
        return;
    };

    let broker = CompletionBroker::new();
    let mut d = Shell::open(boot(&snap).as_bytes(), broker.clone()).expect("the driver creates");
    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.clone(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    d.load_model(&load).expect("the snapshot loads");

    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 1,
        shape: vec![VOCAB as u32],
        dtype: PIE_CHANNEL_DTYPE_F32,
        host_role: PIE_CHANNEL_HOST_ROLE_READER,
        seeded: false,
        extern_dir: PIE_CHANNEL_EXTERN_NONE,
        capacity: CAPACITY,
        reader_wait_id: 3,
        writer_wait_id: 4,
        extern_name: Vec::new(),
    };
    let chb: ChannelBinding = d.register_channel(&ch).expect("the channel registers");
    let program_id = d
        .register_program(&driver_api::ProgramRegistration {
            program_hash: 0x00A5_C0DE,
            ..Default::default()
        })
        .expect("the program registers");
    let binding: InstanceBinding = d
        .bind_instance(&InstanceBindingPlan {
            driver_id: 0,
            program_id,
            requested_instance_id: 0,
            pacing_wait_id: 0,
            channel_ids: vec![1],
            seed_values: Vec::new(),
            geometry_class: driver_api::GeometryClass::Host,
        })
        .expect("the instance binds");
    let instance_ids = vec![binding.instance_id];

    // One page holds both tokens, so nothing grows mid-test and a refusal
    // that had allocated would show up as a moved argmax rather than as an
    // allocator complaint.
    let all_pages: Vec<u32> = vec![0];

    // One decode fire. `masked` decides whether the frame carries the
    // attention-mask table, and NOTHING ELSE about it changes: same token,
    // same page, same CSR -- so the only thing the driver can be answering
    // differently is the mask.
    let fire = |d: &mut Shell, kv_len: u32, token: u32, position: u32, first: bool, masked: bool| {
        let mut cell = driver_api::local::TerminalCell::pending();
        let cell_ptr: *mut driver_api::local::TerminalCell = &mut cell;
        // A decode row attends its whole context: `kv_len` true cells, which
        // is `[0, kv_len]` as runs (an empty false run, then the trues). It
        // is deliberately the CAUSAL answer -- the point is that a mask which
        // asks for nothing extra is refused all the same, because the refusal
        // is about what the TEXT can express and not about what the mask says.
        let (masks, mask_indptr) = if masked {
            (
                vec![EncodedMask::new(vec![0, kv_len], u64::from(kv_len))],
                vec![0, 1],
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let step = StepSubmission {
            plan: LaunchPlan {
                token_ids: vec![token],
                position_ids: vec![position],
                kv_page_indices: all_pages.clone(),
                kv_page_indptr: vec![0, 1],
                kv_last_page_lens: vec![kv_len],
                qo_indptr: vec![0, 1],
                rs_slot_ids: vec![0],
                rs_slot_flags: vec![if first { PIE_RS_FLAG_RESET } else { 0 }],
                has_user_mask: masked,
                masks,
                mask_indptr,
                ..Default::default()
            },
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![driver_api::local::PIE_GEOMETRY_CLASS_HOST],
            terminal_cells: vec![cell_ptr],
            ..Default::default()
        };
        let frame = FrameSubmission {
            instance_ids: instance_ids.clone(),
            required_kv_pages: 1,
            steps: vec![step],
            ..Default::default()
        };
        let (target, completion) = broker.launch_completion(1);
        let status = d
            .launch(&frame, target)
            .map_or_else(|s| s, |()| PIE_STATUS_OK);
        if status != PIE_STATUS_OK {
            return status;
        }
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        loop {
            if let Some(settled) = completion.check() {
                settled.expect("the fire completed");
                return PIE_STATUS_OK;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "the fire never completed",
            );
            std::thread::yield_now();
        }
    };

    let words = chb.word_base as *mut u64;
    let mut consumed = 0u64;
    let mut advance = || -> u32 {
        // SAFETY: as `leg`'s `argmax_of` -- the mirror is `cell_bytes *
        // (capacity + 1)` of host memory the registration allocated, and
        // `% CELLS` keeps the read inside it.
        let cell = unsafe {
            std::slice::from_raw_parts(
                (chb.mirror_base as *const f32).add((consumed % CELLS) as usize * VOCAB),
                VOCAB,
            )
        };
        let got = cell
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(t, _)| t as u32)
            .expect("a non-empty vocabulary");
        consumed += 1;
        // SAFETY: `word_base` is the channel's four-word control block,
        // published by the registration and alive for the channel's life.
        unsafe { words.write_volatile(consumed) };
        got
    };

    // 1. The causal fire, which is the one this SKU can serve.
    assert_eq!(
        fire(&mut d, 1, PROMPT[0], 0, true, false),
        PIE_STATUS_OK,
        "the unmasked fire is accepted",
    );
    let first = advance();

    // 2. The same fire with a mask. `qwen35-d0.8b` states one attention arm
    //    and it is causal, so the lane cannot carry the request.
    assert_ne!(
        fire(&mut d, 2, PROMPT[1], 1, false, true),
        PIE_STATUS_OK,
        "a text that states no `masked` fact must refuse a masked frame \
         rather than attend causally over it, which is the one wrong answer \
         that looks right",
    );

    // 3. And the shell still serves. A refusal that had grown a pool or
    //    advanced the KV would move this argmax off the chain's second
    //    token; it is the one `the_baker_lane_generates_the_banked_eight_
    //    tokens` walks through on its way to `BANKED`.
    assert_eq!(
        fire(&mut d, 2, PROMPT[1], 1, false, false),
        PIE_STATUS_OK,
        "the shell serves the causal fire after refusing the masked one",
    );
    let second = advance();
    eprintln!("masked-gate argmaxes: {first} then {second}");
    assert_ne!(
        first, second,
        "two positions of a real chain, not one answer repeated",
    );
    drop(d);
}
