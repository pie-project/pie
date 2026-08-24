//! **The three banked answers, fired through the shell that serves.**
//!
//! These were `baker-smoke`'s gate: three checkpoints, one token in, an
//! argmax and a logit banked from the first end-to-end fire each SKU ever
//! managed. `scripts/banked-argmaxes.sh` ran that binary three times and a
//! non-zero exit was the whole check.
//!
//! The binary is gone, and it should be: it reached `kernels-cuda`
//! directly and was therefore a SECOND executor beside `driver-cuda`'s,
//! with its own pools, staging, binding and fire. Two executors over one
//! set of kernels drift, and only one of them was ever measured against a
//! checkpoint. So the gate moves to the path that actually serves — the
//! `Shell`, its load contract, its pools and its submission — which makes
//! it strictly stronger than what it replaces.
//!
//! `baker_serve.rs` already fires qwen and gemma end to end against banked
//! token chains and a transformers oracle. It does NOT cover gpt-oss-20b at
//! all, and it reads the Base qwen checkpoint rather than the instruct one
//! these answers were banked from. Both gaps close here.
#![allow(clippy::print_stderr, clippy::print_stdout)]

mod common;
use common::{device_or_skip, gpu_guard};

use driver_api::completion::CompletionBroker;
use driver_api::local::{
    ChannelBinding, InstanceBinding, PIE_CHANNEL_DTYPE_F32, PIE_CHANNEL_EXTERN_NONE,
    PIE_CHANNEL_HOST_ROLE_READER, PIE_RS_FLAG_RESET, PIE_STATUS_OK,
};
use driver_api::{
    ChannelRegistrationPlan, FrameSubmission, InstanceBindingPlan, LaunchPlan, StepSubmission,
};
use driver_cuda::serve::Shell;

/// The KV page size. Not a knob (`boot::KV_PAGE_SIZE`).
const PAGE: u32 = 16;
/// Ring cells. One fire publishes one; the margin is so a wrap is a bug.
const CAPACITY: u32 = 7;
const CELLS: u64 = CAPACITY as u64 + 1;

/// **The token every banked answer was fired from.**
///
/// `baker-smoke`'s default prompt was the single id 785 and
/// `banked-argmaxes.sh` never overrode it, so all three answers below are
/// "one fire, one row, position zero". A prompt of one is what makes them
/// comparable across three tokenizers that agree on nothing else.
const PROMPT: u32 = 785;

/// A checkpoint, and what it answered the first time it served.
struct Banked {
    /// The catalog row, which is where the vocabulary comes from.
    sku: &'static str,
    /// The HF cache directory holding the snapshot.
    cache: &'static str,
    /// The argmax token id.
    id: usize,
    /// The logit **as rendered to four decimals**.
    ///
    /// COMPARED AS RENDERED, not bit-exact, and that is deliberate rather
    /// than sloppy: gemma's logit is 7.59375 and 7.5938 is its four-decimal
    /// rendering — the form it was banked in, and every digit a bf16 logit
    /// carries. Comparing the parsed floats would fail on a number that is
    /// right.
    logit: &'static str,
}

const BANKED: [Banked; 3] = [
    Banked {
        sku: "qwen35-d0.8b-bf16-kv-bf16",
        cache: "models--Qwen--Qwen3.5-0.8B",
        id: 198,
        logit: "12.3125",
    },
    Banked {
        sku: "gptoss-20b-bf16-mxfp4-kv-bf16",
        cache: "models--openai--gpt-oss-20b",
        id: 11,
        logit: "14.4375",
    },
    Banked {
        sku: "gemma4-e4b-bf16-kv-bf16",
        cache: "models--google--gemma-4-E4B-it",
        id: 785,
        logit: "7.5938",
    },
];

/// The newest cached snapshot directory, or `None`.
///
/// The `.index.json` arm is not optional politeness: several of these
/// snapshots carry no plain `model.safetensors` — the shard is
/// `model.safetensors-00001-of-00001.safetensors` and is only reachable
/// through the index. A gate that probed for the plain name alone would
/// skip forever and print `test result: ok` while measuring nothing.
fn snapshot_of(cache_dir: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let snaps = std::path::PathBuf::from(home)
        .join(format!(".cache/huggingface/hub/{cache_dir}/snapshots"));
    std::fs::read_dir(&snaps)
        .ok()?
        .filter_map(Result::ok)
        .find_map(|e| {
            let d = e.path();
            (d.join("model.safetensors").is_file()
                || d.join("model.safetensors.index.json").is_file())
            .then_some(d)
        })
}

/// The boot document, pointing at the snapshot's own `config.json` — the
/// read `load_impl` needs for the one field a catalog row does not hold.
fn boot(snap: &std::path::Path) -> String {
    format!(
        "[model]\nconfig = \"{}\"\n",
        snap.join("config.json").display()
    )
}

/// One fire of one token, and the whole logit row it published.
fn logits_of(snap: &std::path::Path, vocab: usize, tag: u64) -> Vec<f32> {
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
         this fire to have quietly taken",
    );

    let ch = ChannelRegistrationPlan {
        driver_id: 0,
        channel_id: 1,
        shape: vec![vocab as u32],
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

    let mut cell = driver_api::local::TerminalCell::pending();
    let cell_ptr: *mut driver_api::local::TerminalCell = &mut cell;
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: vec![PROMPT],
            position_ids: vec![0],
            kv_page_indices: vec![0],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![1],
            qo_indptr: vec![0, 1],
            rs_slot_ids: vec![0],
            rs_slot_flags: vec![PIE_RS_FLAG_RESET],
            ..Default::default()
        },
        roster_rows: vec![0],
        sub_batch_indptr: vec![0, 1],
        sub_batch_class: vec![driver_api::local::PIE_GEOMETRY_CLASS_HOST],
        terminal_cells: vec![cell_ptr],
        ..Default::default()
    };
    let frame = FrameSubmission {
        instance_ids: vec![binding.instance_id],
        required_kv_pages: PAGE.div_ceil(PAGE),
        steps: vec![step],
        ..Default::default()
    };

    let (target, completion) = broker.launch_completion(1);
    assert_eq!(
        d.launch(&frame, target)
            .map_or_else(|s| s, |()| PIE_STATUS_OK),
        PIE_STATUS_OK,
        "the fire is accepted",
    );
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    loop {
        if let Some(settled) = completion.check() {
            settled.expect("the fire completed");
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the fire never completed",
        );
        std::thread::yield_now();
    }

    // SAFETY: the mirror is `CELLS` cells of `vocab` f32s, published by the
    // fire that just settled; cell 0 is this fire's and nothing else writes
    // it before the shell drops.
    let row = unsafe { std::slice::from_raw_parts(chb.mirror_base as *const f32, vocab) }.to_vec();
    unsafe { (chb.word_base as *mut u64).write_volatile(1) };
    let _ = CELLS;
    drop(d);
    row
}

/// **The gate.** Three SKUs, one fire each, the banked id at the banked
/// logit.
///
/// `#[ignore]`d because it needs a GPU and three cached checkpoints, which
/// is the same reason `baker_serve.rs` is not part of a default sweep.
/// `scripts/banked-argmaxes.sh` is what runs it.
#[test]
#[ignore = "needs a CUDA device and three cached checkpoints"]
fn the_three_banked_argmaxes() {
    let _gpu = gpu_guard();
    let Some(_dev) = device_or_skip("banked argmaxes") else {
        return;
    };

    let mut fired = 0usize;
    for (tag, b) in BANKED.iter().enumerate() {
        let Some(snap) = snapshot_of(b.cache) else {
            eprintln!("[skip] {} — {} is not cached", b.sku, b.cache);
            continue;
        };
        let vocab = model::serve::row(b.sku)
            .unwrap_or_else(|| panic!("`{}` is not a catalog row", b.sku))
            .vocab as usize;

        let row = logits_of(&snap, vocab, 0xBA_5E_00 + tag as u64);
        let (id, logit) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a non-empty vocabulary");

        let rendered = format!("{logit:.4}");
        assert_eq!(
            (id, rendered.as_str()),
            (b.id, b.logit),
            "the banked answer for `{}` is {} at {} and this fire answered \
             {id} at {rendered}",
            b.sku,
            b.id,
            b.logit,
        );
        println!("{}: {} at {} — matched", b.sku, b.id, b.logit);
        fired += 1;
    }

    assert!(
        fired > 0,
        "no checkpoint was cached, so this gate measured nothing; it must \
         not pass quietly",
    );
}
