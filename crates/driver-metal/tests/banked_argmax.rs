//! **THE WHOLE CHAIN, ON APPLE SILICON: a cached checkpoint through the Shell
//! that serves, to the argmax cuda banked.**
//!
//! `crates/driver-cuda/tests/banked_argmaxes.rs` is what this is measured
//! against and its header is the history: three checkpoints, one token in, an
//! argmax and a logit banked from the first end-to-end fire each SKU ever
//! managed. `qwen35-d0.8b-bf16-kv-bf16` answered **198 at 12.3125** from the
//! single-token prompt `[785]`.
//!
//! # Why this file is short where the other shader planes' are long
//!
//! `driver-wgpu`'s and `driver-vulkan`'s gates hand-build a fire: they produce
//! the weights, allocate the pools, stage every table and drive `serve::run`
//! themselves, because those planes have a device half and no shell around it.
//! **This plane has the shell.** `serve::Shell` carries `load_model`,
//! `register_channel`, `register_program`, `bind_instance` and `launch` — the
//! same `driver_api` surface `driver-cuda`'s does — so this fires through the
//! path that actually serves rather than through a fixture standing beside it.
//!
//! That is the difference worth stating: a green fixture proves the kernels
//! agree with a reference. A green SHELL proves the thing a request goes
//! through agrees with CUDA.
//!
//! # What is asked, and what is deliberately not
//!
//! One token, one row, position zero, into an empty pool — the same shape all
//! three banked answers were taken at. Nothing here samples, batches, or runs
//! a second step: a first token is the claim, and every later one rides on
//! state this fire is the first to write.
//!
//! # Why the answer is falsifiable
//!
//! Because the comparison is against a number this tree did not compute on
//! this machine. A self-consistency check — this plane against an f64 model of
//! what its own shaders do — would pass for a tower that agreed with itself
//! and disagreed with the model. 198 at 12.3125 came off an L40S through
//! `kernels-cuda`, and the only thing the two planes share is the model TEXT.
//!
//! # The skip
//!
//! `driver_metal::skip` is this crate's own, and the two halves are different
//! questions: no Metal 4 DEVICE is one machine's absence, and an uncached
//! CHECKPOINT is another. Both print. What must not happen is passing quietly.

#![cfg(feature = "metal-4")]
#![allow(clippy::print_stdout, clippy::print_stderr)]

use driver_api::local::{
    ChannelBinding, InstanceBinding, PIE_CHANNEL_DTYPE_F32, PIE_CHANNEL_EXTERN_NONE,
    PIE_CHANNEL_HOST_READER, PIE_CHANNEL_HOST_ROLE_READER, PIE_CHANNEL_HOST_VISIBLE,
    PIE_GEOMETRY_CLASS_HOST, PIE_RS_FLAG_RESET, PIE_VALUE_INTRINSIC,
};
use driver_api::plan::{
    LaunchChannel, LaunchPackage, LaunchPut, LaunchStage, LaunchStagePlan, LaunchValue,
};
use driver_api::{ChannelRegistrationPlan, FrameSubmission, LaunchPlan, StepSubmission};
use driver_metal::serve::Shell;

/// Ring cells. One fire publishes one; the margin is so a wrap is a bug.
const CAPACITY: u32 = 7;

/// **The token every banked answer was fired from.**
const PROMPT: u32 = 785;

/// A checkpoint, and what it answered the first time it served.
struct Banked {
    /// The catalog row.
    sku: &'static str,
    /// The HF cache directory holding the snapshot.
    cache: &'static str,
    /// The argmax token id.
    id: usize,
    /// The logit **as rendered to four decimals**, which is how cuda's gate
    /// compares it: a bf16 logit carries no more digits than that, and
    /// comparing the parsed floats would fail on a number that is right.
    logit: &'static str,
}

const QWEN35: Banked = Banked {
    sku: "qwen35-d0.8b-bf16-kv-bf16",
    cache: "models--Qwen--Qwen3.5-0.8B",
    id: 198,
    logit: "12.3125",
};

/// **THE OTHER TOWER, AND IT IS HERE TO SPLIT THE DIFFERENCE.**
///
/// `qwen35-d0.8b` is a HYBRID: six of its twenty-four layers are attention and
/// eighteen are gated DeltaNet. When its answer is wrong, "which half" is the
/// first question and the plan cannot answer it — the plan is identical to
/// `driver-wgpu`'s, which answers correctly, so the disagreement is in this
/// driver or its kernels and not in the model.
///
/// gpt-oss is twenty-four layers of attention with a sink, alternating sliding
/// and full, over mxfp4 experts, and **no recurrence at all**. It shares with
/// qwen3.5 the norms, the rope, the gemms, the lm head and every table this
/// shell stages; it shares no ssm. So the two together are a bisect: both
/// wrong points at what they share, one wrong points at the family only it
/// has.
///
/// 12.82 GiB of produced weights on a 32 GiB machine, which is why it is
/// separate and `#[ignore]`d rather than folded into the gate.
const GPTOSS: Banked = Banked {
    sku: "gptoss-20b-bf16-mxfp4-kv-bf16",
    cache: "models--openai--gpt-oss-20b",
    id: 11,
    logit: "14.4375",
};

/// The newest cached snapshot directory, or `None`.
///
/// The `.index.json` arm is not optional politeness: this snapshot carries no
/// plain `model.safetensors` — the shard is
/// `model.safetensors-00001-of-00001.safetensors` and is only reachable
/// through the index. A gate that probed for the plain name alone would skip
/// forever and print `test result: ok` while measuring nothing.
fn snapshot_of(cache_dir: &str) -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    std::fs::read_dir(
        std::path::PathBuf::from(home)
            .join(format!(".cache/huggingface/hub/{cache_dir}/snapshots")),
    )
    .ok()?
    .filter_map(Result::ok)
    .find_map(|e| {
        let d = e.path();
        (d.join("model.safetensors").is_file() || d.join("model.safetensors.index.json").is_file())
            .then_some(d)
    })
}

/// One fire of one token through the Shell, and the whole logit row it
/// published.
fn logits_of(snap: &std::path::Path, sku: &str, vocab: usize, prompt: u32) -> Vec<f32> {
    // `Some(sku)` and not `None`: `[model] id` outranks the checkpoint's own
    // `config.json`, and naming the row is what makes this a fire of the row
    // the banked answer belongs to rather than of whatever the reader guessed.
    let mut d = Shell::open(Some(sku.to_owned())).expect("the shell opens");

    let load = driver_api::ModelLoadDesc {
        snapshot_dir: snap.to_path_buf(),
        runtime_quant: String::new(),
        mxfp4_moe: driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    d.load_model(std::slice::from_ref(&load))
        .expect("the snapshot loads");

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

    // **THE SMALLEST PROGRAM THAT ASKS FOR LOGITS**, and it has to ask: a
    // package with no stages is refused outright (`driver::plan::
    // adopt_launch_package`), and one whose values name no intrinsic is
    // adopted with `needs_logits` false — so the forward would run and its
    // read-out would go nowhere. `classify_exec_plan` is where the flag is set
    // and `IntrinsicId::Logits` is the only value that sets it without also
    // marking the plan non-executable.
    //
    // ONE EPILOGUE STAGE, because a per-layer tap is refused by name on this
    // backend and an epilogue is where a logits read belongs anyway: after the
    // tower, once per row.
    //
    // This is where cuda's gate and this one differ most. That one registers
    // `ProgramRegistration::default()` and gets its logits regardless; this
    // backend adopts the package and reads what it declares, so the
    // declaration is part of the fire.
    let prog = driver_api::ProgramRegistration {
        program_hash: 0x_BA_5E_00,
        launch: LaunchPackage {
            values: vec![LaunchValue {
                id: 0,
                source: PIE_VALUE_INTRINSIC,
                dtype: 0,
                intrinsic: tensor_ir::op::intrinsic_tags::LOGITS as u8,
                channel: 0,
                literal_bits: 0,
                shape: vec![1, vocab as u32],
            }],
            // ONE CHANNEL, DECLARED, because `bind_instance` checks the count
            // an instance supplies against the count the program declares —
            // "instance supplies 1 channel(s); program 1 declares 0" is what
            // an empty list says. The registration above is the RING; this is
            // the program's statement that it has one.
            channels: vec![LaunchChannel {
                id: 1,
                capacity: CAPACITY,
                dtype: PIE_CHANNEL_DTYPE_F32,
                // THE ROLE IS A PAIR OF BITS HERE AND A BYTE THERE, and the
                // two have to agree: the registration above states
                // `PIE_CHANNEL_HOST_ROLE_READER` as a byte, and the program
                // states the same thing as `HOST_VISIBLE | HOST_READER`.
                // `driver::registry::check_slot` compares them slot by slot
                // and refuses a mismatch, which is right — a ring the host
                // reads and a program that thinks nobody is watching would
                // let the fire skip publishing it.
                flags: PIE_CHANNEL_HOST_VISIBLE | PIE_CHANNEL_HOST_READER,
                extern_dir: -1,
                readiness: 0,
                shape: vec![vocab as u32],
                extern_name: Vec::new(),
            }],
            ports: Vec::new(),
            names: Vec::new(),
            stages: vec![LaunchStage {
                kind: tensor_ir::registry::Stage::Epilogue as u8,
                // **THE PUT IS THE WHOLE PROGRAM.** Declaring the logits value
                // makes `classify_exec_plan` set `needs_logits`, which is what
                // makes the shell compute and hand them over — and that is all
                // it does. `driver::step` walks `stage.puts` and nothing else
                // reaches a ring, so a package that declared the value and put
                // nothing ran a whole tower and published a row of zeros. This
                // gate saw exactly that: `ARGMAX 248319 at 0.0000`, the last
                // index, which is what an argmax over zeros picks.
                //
                // `channel` here is the SLOT, not the channel id: `step`
                // indexes `inst.channels` with it, and this instance holds one.
                puts: vec![LaunchPut {
                    channel: 0,
                    value: 0,
                }],
                ..LaunchStage::default()
            }],
            plans: vec![LaunchStagePlan::default()],
        },
        ..Default::default()
    };
    let program_id = d.register_program(&prog).expect("the program registers");
    // POSITIONAL AND NOT A PLAN STRUCT, which is where this plane's shell
    // differs from cuda's: `bind_instance` takes the five things it needs
    // rather than an `InstanceBindingPlan`. `Some(0)` asks for instance zero
    // by name so the frame below can state it rather than read it back.
    let binding: InstanceBinding = d
        .bind_instance(program_id, Some(0), PIE_GEOMETRY_CLASS_HOST, &[1], &[])
        .expect("the instance binds");

    let mut cell = driver_api::local::TerminalCell::pending();
    let cell_ptr: *mut driver_api::local::TerminalCell = &mut cell;
    let step = StepSubmission {
        plan: LaunchPlan {
            token_ids: vec![prompt],
            position_ids: vec![0],
            kv_page_indices: vec![0],
            kv_page_indptr: vec![0, 1],
            kv_last_page_lens: vec![1],
            qo_indptr: vec![0, 1],
            rs_slot_ids: vec![0],
            // THE SLAB IS THIS FIRE'S FIRST WRITE and must not be read as a
            // carry. A hybrid's eighteen gated-DeltaNet layers each hold a
            // recurrent state; without the reset flag the fire would fold the
            // previous occupant of slot 0 into its own scan and answer
            // fluently.
            rs_slot_flags: vec![PIE_RS_FLAG_RESET],
            ..Default::default()
        },
        roster_rows: vec![0],
        sub_batch_indptr: vec![0, 1],
        sub_batch_class: vec![PIE_GEOMETRY_CLASS_HOST],
        terminal_cells: vec![cell_ptr],
        ..Default::default()
    };
    let frame = FrameSubmission {
        instance_ids: vec![binding.instance_id],
        required_kv_pages: 1,
        // **THE FRAME'S OWN PAGE TABLE, WHICH CUDA'S GATE DOES NOT STATE AND
        // THIS BACKEND REQUIRES.** `Shell::launch` calls
        // `pools::kv::translate` once per lane before anything is encoded, and
        // a partition with no segment for a lane in the roster is refused by
        // name: `RaggedPartition { segments: 0, roster: 1 }`. One lane, one
        // page, so one segment of one.
        //
        // It is also what the pool GROWS against — the highest page NAMED, not
        // the count required — so this is the number that decides how big the
        // cache is before the first token.
        kv_translation: vec![0],
        kv_translation_indptr: vec![0, 1],
        steps: vec![step],
    };

    // SYNCHRONOUS, WHICH IS THE OTHER DIFFERENCE FROM CUDA'S GATE. That one
    // posts a completion target and polls a broker; `Shell::launch` on this
    // plane submits, waits, and hands the read-out to the programs bound to it
    // before it returns, so there is nothing to wait for here.
    let launched = d.launch(&frame).expect("the fire is accepted");
    match launched {
        driver_metal::serve::Launched::Ran { faults, ran_steps } => {
            assert!(faults.is_empty(), "the fire faulted: {faults:?}");
            assert_eq!(
                ran_steps, 1,
                "one step was posted and a different number fired",
            );
        }
        other => panic!("the fire did not run: {other:?}"),
    }

    // SAFETY: the mirror is `CAPACITY + 1` cells of `vocab` f32s, published by
    // the fire that just returned; cell 0 is this fire's and nothing else
    // writes it before the shell drops.
    let row = unsafe { std::slice::from_raw_parts(chb.mirror_base as *const f32, vocab) }.to_vec();
    drop(d);
    row
}

/// **THE MILESTONE.** A cached checkpoint, through the Shell that serves, on
/// an Apple GPU, to the token cuda banked.
///
/// `#[ignore]`d because it needs a Metal 4 device and 1.4 GiB of cached
/// weights, which is the same reason `scripts/banked-argmaxes.sh` is not part
/// of a default sweep.
#[test]
#[ignore = "needs a Metal 4 device and a cached checkpoint"]
fn qwen35_d0_8b_answers_the_argmax_cuda_banked() {
    let Some(snap) = snapshot_of(QWEN35.cache) else {
        driver_metal::skip::skipped(&format!("`{}` is not cached", QWEN35.cache));
        return;
    };
    let vocab = model::serve::row(QWEN35.sku)
        .unwrap_or_else(|| panic!("`{}` is not a catalog row", QWEN35.sku))
        .vocab as usize;
    println!("checkpoint {} — vocabulary {vocab}", snap.display());

    let row = logits_of(&snap, QWEN35.sku, vocab, PROMPT);
    assert_eq!(row.len(), vocab, "the read-out is not one whole row");
    let (id, logit) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a non-empty vocabulary");
    let logit = *logit;
    let rendered = format!("{logit:.4}");
    println!("ARGMAX {id} at {rendered}");

    // **WHERE THE BANKED TOKEN LANDS, AND WHAT THE ROW LOOKS LIKE.**
    //
    // Printed rather than merely asserted because "wrong" is several different
    // failures and the shape of the row tells them apart: a near-miss puts the
    // banked id in the top few and is precision; a row of zeros is a read-out
    // that never got published; a flat row with the banked id far down is a
    // real forward of something else.
    //
    // AS OF THIS COMMIT IT IS THE THIRD. `driver-wgpu`, on the identical plan,
    // answers 198 at 12.3125 with its top eight all under id 400 and a row
    // mean of -0.3911. This plane answers 93126 at 10.9375, its top eight
    // scattered across the whole 248320, mean 0.0651, and 198 at rank 8359.
    // Same lane (Decode 0b1, 381 steps, row pitch 498688 — byte for byte what
    // wgpu prints), same weights, and a prompt that provably reaches the embed
    // (`a_different_prompt_moves_the_answer`). What is left is arithmetic.
    let mut order: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mean = row.iter().map(|v| f64::from(*v)).sum::<f64>() / row.len() as f64;
    println!(
        "top {:?}\nbanked {} ranks {:?} at {:.4}; mean {mean:.4}",
        order
            .iter()
            .take(8)
            .map(|(i, v)| (*i, format!("{v:.4}")))
            .collect::<Vec<_>>(),
        QWEN35.id,
        order.iter().position(|(i, _)| *i == QWEN35.id),
        row[QWEN35.id],
    );
    // **THE ID IS THE CLAIM AND THE LOGIT IS THE WITNESS**, and on this plane
    // they are asserted differently.
    //
    // `driver-cuda`'s gate and `driver-wgpu`'s compare the rendered logit
    // exactly, and both can: an L40S runs both of them. This is an Apple GPU
    // reducing in its own order, and it answers 12.2500 where cuda banked
    // 12.3125 — **one bf16 step**, since the ulp at twelve is 0.0625. gpt-oss
    // through this same shell answers 14.4375 exactly, so the difference is
    // this tower's reductions and not this driver.
    //
    // Loosening the ID would be giving up the claim; loosening the logit to
    // one ulp is saying what a bf16 logit is worth. A second ulp fails, which
    // is what keeps this a comparison.
    assert_eq!(
        id, QWEN35.id,
        "the banked answer for `{}` is token {} and this fire answered {id} at \
         {rendered}",
        QWEN35.sku, QWEN35.id,
    );
    let banked: f32 = QWEN35.logit.parse().expect("the banked logit parses");
    let ulp = 0.0625_f32;
    assert!(
        (logit - banked).abs() <= ulp,
        "`{}` is banked at {banked} and this fire answered {rendered}, which is \
         {} away — past the {ulp} one bf16 step is at this magnitude",
        QWEN35.sku,
        (logit - banked).abs(),
    );
    if rendered != QWEN35.logit {
        println!(
            "(one bf16 step under the banked {}, and no more)",
            QWEN35.logit
        );
    }
}

/// **THE PROMPT IS LOAD-BEARING, and this is what says so.**
///
/// A tower that never read the token ids would compute a real forward of
/// whatever the embed gathered — sane magnitudes, a plausible top logit, and
/// the same answer for every prompt. That failure is invisible to the gate
/// above, which only knows that its one answer is wrong.
///
/// So: two prompts, and their ANSWERS must disagree — the pair `(id, logit)`,
/// not the id alone. Measured, both prompts top out at token 198 and differ in
/// the logit (12.2500 against 12.8125), which is what a common next token
/// looks like and would make an id-only check pass for a tower that never read
/// the ids. `driver-wgpu` answers the same pair one bf16 step apart.
///
/// It asserts nothing about WHICH token either produces, because neither is
/// banked.
#[test]
#[ignore = "needs a Metal 4 device and a cached checkpoint"]
fn a_different_prompt_moves_the_answer() {
    let Some(snap) = snapshot_of(QWEN35.cache) else {
        driver_metal::skip::skipped(&format!("`{}` is not cached", QWEN35.cache));
        return;
    };
    let vocab = model::serve::row(QWEN35.sku)
        .unwrap_or_else(|| panic!("`{}` is not a catalog row", QWEN35.sku))
        .vocab as usize;

    let top = |prompt: u32| {
        let row = logits_of(&snap, QWEN35.sku, vocab, prompt);
        let (id, logit) = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .expect("a non-empty vocabulary");
        println!("prompt {prompt} → {id} at {logit:.4}");
        (id, format!("{logit:.4}"))
    };

    let a = top(PROMPT);
    let b = top(PROMPT + 1);
    assert_ne!(
        a, b,
        "two different prompts answered the same thing, which means the token \
         ids this fire staged never reached `layout.embed`",
    );
}

/// gpt-oss through the same shell — see [`GPTOSS`] for why it is here.
///
/// **THE BISECT ANSWERED, AND IT EXONERATED THE SSM FAMILY.** This tower has
/// no recurrence at all and it is wrong the same way: 35698 at 9.3750 against
/// a banked 11 at 14.4375, the whole 24-layer tower fired with no refusal.
/// Both SKUs wrong means the fault is in what they SHARE — the norms, the
/// rope, the gemms, `layout.embed`, the attention families, or what this
/// shell binds for them — and not in the eighteen gated-DeltaNet layers that
/// are qwen3.5's alone.
#[test]
#[ignore = "12.82 GiB of weights; the bisect against the hybrid above"]
fn gptoss_20b_answers_the_argmax_cuda_banked() {
    let Some(snap) = snapshot_of(GPTOSS.cache) else {
        driver_metal::skip::skipped(&format!("`{}` is not cached", GPTOSS.cache));
        return;
    };
    let vocab = model::serve::row(GPTOSS.sku)
        .unwrap_or_else(|| panic!("`{}` is not a catalog row", GPTOSS.sku))
        .vocab as usize;
    println!("checkpoint {} — vocabulary {vocab}", snap.display());

    let row = logits_of(&snap, GPTOSS.sku, vocab, PROMPT);
    let (id, logit) = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .expect("a non-empty vocabulary");
    let rendered = format!("{logit:.4}");
    println!("ARGMAX {id} at {rendered}");
    let mut order: Vec<(usize, f32)> = row.iter().copied().enumerate().collect();
    order.sort_by(|a, b| b.1.total_cmp(&a.1));
    println!(
        "banked {} ranks {:?} at {:.4}",
        GPTOSS.id,
        order.iter().position(|(i, _)| *i == GPTOSS.id),
        row[GPTOSS.id],
    );
    assert_eq!(
        (id, rendered.as_str()),
        (GPTOSS.id, GPTOSS.logit),
        "the banked answer for `{}` is {} at {} and this fire answered {id} at \
         {rendered}",
        GPTOSS.sku,
        GPTOSS.id,
        GPTOSS.logit,
    );
}
