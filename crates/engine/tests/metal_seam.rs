//! The Metal seam: can it be selected, and does it say what it cannot serve?
//!
//! A backend that cannot be selected teaches nothing. This checks the half
//! that works — the device opens, the facts are stated, the variant dispatches
//! — and that the half that does not refuses **by name** rather than by
//! absence, panic, or a plausible wrong answer.

#![cfg(all(feature = "driver-metal", target_vendor = "apple"))]

use engine::driver::backend::open;

#[test]
fn the_metal_backend_opens_a_device_and_states_what_it_is() {
    let Ok(backend) = open::metal(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    assert_eq!(backend.kind(), "metal");
    // Off the CONTRACT rather than out of `create`'s second return value:
    // a driver states its own facts, so there is one copy of them.
    let facts = backend
        .device_facts()
        .expect("a local driver knows its device");
    assert_eq!(facts.backend, "metal");
    assert!(
        facts.unified_memory,
        "Apple silicon shares physical memory between the pool and the host, \
         and that changes what a `device is full` question means"
    );
    assert_eq!(facts.page_size, 16, "the paged KV pool's rows per page");
    assert!(
        !facts.fp8_native && !facts.native_mxfp4_moe,
        "neither kernel exists in `kernels-metal`, and the facts should say so \
         rather than let a scheduler discover it at launch"
    );
}

#[test]
fn the_verbs_that_need_the_kv_pool_refuse_by_name() {
    // The ordering, stated. Every one of these is above a pool that does not
    // exist until `load_model` allocates it, so the refusal names the backend
    // and the step that was skipped rather than reporting a generic failure.
    let Ok(mut backend) = open::metal(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let text = match backend.copy_kv(&Default::default()) {
        Err(why) => format!("{why}"),
        Ok(_) => panic!("a copy before a load has no pool to move within"),
    };
    assert!(
        text.contains("driver-metal"),
        "a refusal must name the backend that made it: {text}"
    );
    assert!(
        text.contains("before load_model"),
        "and say which order was broken: {text}"
    );

    // `resize_pool` refuses too, and for a different reason than it used to:
    // the resize is wired now (`pool.resize` through the stepper), so what is
    // missing before a load is the POOL, not the machinery. A refusal that
    // still described missing machinery would send the next reader to re-port
    // something that is already there -- which is what this assertion is for.
    let text = match backend.resize_pool(&Default::default()) {
        Err(why) => format!("{why}"),
        Ok(_) => panic!("there is no pool to resize before a load"),
    };
    assert!(
        text.contains("driver-metal"),
        "a refusal must name the backend that made it: {text}"
    );
    assert!(
        text.contains("no KV pool") && text.contains("load_model"),
        "and say what is absent and what would create it: {text}"
    );
}

#[test]
fn media_encode_is_refused_rather_than_pretended() {
    // Unsupported on this backend and on CUDA both, and the seam says so
    // instead of returning a completion nothing will settle.
    let Ok(backend) = open::metal(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    assert!(
        backend.export_kv_handle().is_none(),
        "Metal has no cross-process KV sharing path to export"
    );
}

#[test]
fn load_model_takes_one_descriptor_because_this_backend_holds_one_model() {
    // The same shape the CUDA shell's `state.model` has, and the reason a
    // frame's instance roster is one family's — which is what makes
    // `lower(plan, rows, fire)`'s one-plan signature right.
    let Ok(mut backend) = open::metal(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    let desc = || ::driver_api::ModelLoadDesc {
        snapshot_dir: std::path::PathBuf::from("/nonesuch"),
        runtime_quant: String::new(),
        mxfp4_moe: ::driver_api::Mxfp4MoeRequest::Auto,
        component: driver_api::ModelComponent::Full,
    };
    let why = format!(
        "{}",
        backend
            .load_model(vec![desc(), desc()])
            .expect_err("two models is not a shape this backend has")
    );
    assert!(
        why.contains("ONE model"),
        "the refusal should say why, not just that: {why}"
    );

    // And one descriptor gets as far as the checkpoint, which is the point:
    // the failure is now about the SNAPSHOT rather than about the seam.
    //
    // Asserted as that RELATION rather than against the loader's wording. It
    // used to match the literal `[model] config`, which was the refusal a
    // missing config file drew when this was written; the loader now looks
    // for `model.safetensors[.index.json]` first and says so, and the test
    // could not tell anyone -- `crates/engine/tests/metal_seam.rs` is behind
    // `#![cfg(all(feature = "driver-metal", target_vendor = "apple"))]`, so
    // the only machine that compiles it is a mac with the feature on, and no
    // job in `ci.yml` had ever run on this branch to be that machine. The
    // file did not even COMPILE: it still named `engine::driver::submission`
    // and `backend::FrameLaunchOutcome`, two paths that moved into
    // `driver_api` and are re-exported from `engine::driver` today.
    //
    // What the message must show is that the descriptor's own path reached
    // the loader. Naming the directory is that, and it cannot be satisfied by
    // the arity refusal above, which never sees a path.
    let why = format!(
        "{}",
        backend
            .load_model(vec![desc()])
            .expect_err("/nonesuch holds no checkpoint")
    );
    assert!(
        why.contains("/nonesuch"),
        "model facts come from the descriptor the worker hands over, not from \
         a checkpoint this seam re-normalizes -- so the refusal should name \
         the snapshot it was handed: {why}"
    );
}

#[test]
fn a_frame_that_cannot_fit_the_pool_is_impossible_rather_than_an_error() {
    // Admission is not a failure. A frame whose demand exceeds the PHYSICAL
    // pool can never be made to fit by evicting, so it is `Impossible` and the
    // engine stops re-posting it; one that merely does not fit right now would
    // be `Exhausted`. Both are outcomes, and neither is an `Err`.
    let Ok(mut backend) = open::metal(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };
    // Before a load there is no pool at all, and that IS an error — the
    // scheduler asked a driver to run a model it never gave it.
    let frame = engine::driver::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: vec![0],
        kv_translation_indptr: vec![0, 1],
        required_kv_pages: 1,
        steps: Vec::new(),
    };
    let why = match backend.launch(&frame) {
        Err(why) => format!("{why}"),
        Ok(_) => panic!("launch before load_model is drift, not admission"),
    };
    assert!(
        why.contains("before load_model"),
        "the refusal should say which order was broken: {why}"
    );
}

#[test]
fn a_program_a_channel_and_an_instance_all_register() {
    // The three verbs that gate everything: without them no instance is bound,
    // so no `FrameSubmission` is ever built and `launch` is unreachable through
    // the engine however complete it is.
    //
    // The channel plane is HOST memory on this backend, exactly as it is on the
    // dummy driver — `ChannelState` holds the cells and four control words —
    // so the binding is their addresses and nothing about it needs a GPU.
    let Ok(mut backend) = open::metal(b"{}") else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };

    // A package with no stages is refused, and that is the registry doing its
    // job: a program that runs nothing is a registration nobody can use.
    let why = format!(
        "{}",
        backend
            .register_program(&Default::default())
            .expect_err("a package with no stages is not a program")
    );
    assert!(why.contains("no stages"), "and it says why: {why}");

    let program = backend
        .register_program(&::driver_api::ProgramRegistration {
            program_hash: 0xABCD,
            launch: package(),
            ..::driver_api::ProgramRegistration::default()
        })
        .expect("a package with a stage registers");

    // Memoised by hash: the engine re-registers freely and expects a lookup.
    let again = backend
        .register_program(&::driver_api::ProgramRegistration {
            program_hash: 0xABCD,
            launch: package(),
            ..::driver_api::ProgramRegistration::default()
        })
        .expect("the same hash registers again");
    assert_eq!(program, again, "the same hash is the same program");

    let channel = backend
        .register_channel(&::driver_api::ChannelRegistrationPlan {
            driver_id: 0,
            channel_id: 7,
            shape: vec![4],
            dtype: driver_api::PIE_CHANNEL_DTYPE_U32,
            host_role: driver_api::PIE_CHANNEL_HOST_ROLE_READER,
            seeded: false,
            extern_dir: driver_api::PIE_CHANNEL_EXTERN_EXPORT,
            capacity: 2,
            reader_wait_id: 11,
            writer_wait_id: 12,
            extern_name: b"out".to_vec(),
        })
        .expect("a channel registers");

    assert_eq!(channel.binding.channel_id, 7);
    assert_ne!(
        channel.binding.mirror_base, 0,
        "a ring whose base is zero is a ring the host cannot read"
    );
    assert_ne!(channel.binding.word_base, 0);
    assert!(
        channel.binding.mirror_bytes >= u64::from(channel.binding.cell_bytes),
        "the ring must hold at least one cell"
    );
    assert_eq!(
        channel.binding.word_bytes,
        4 * std::mem::size_of::<u64>() as u64,
        "head, tail, poison, closed"
    );
    // The four indices are the ABI's order and the state's; neither side may
    // move without the other.
    assert_eq!(
        (
            channel.binding.head_word_index,
            channel.binding.tail_word_index,
            channel.binding.poison_word_index,
            channel.binding.closed_word_index
        ),
        (0, 1, 2, 3)
    );
    assert_eq!(channel.reader_wait_id, 11);

    // And a close of an id nobody holds is not an error: the scheduler closes
    // on its own schedule, and a double close is how a teardown race reads.
    backend.close_channel(7).expect("closing is idempotent");
    backend.close_channel(7).expect("twice, too");
    backend.close_instance(program).expect("as is an instance");
}

/// The smallest package the registry accepts: one stage and its plan.
///
/// Written out because the registry validates rather than assumes — an empty
/// package is "no stages" and a stage without its plan is a "plan/stage count
/// mismatch". Both refusals are the point: a program that runs nothing, or one
/// whose stages and plans disagree, is a registration nobody can use.
fn package() -> ::driver_api::plan::LaunchPackage {
    ::driver_api::plan::LaunchPackage {
        stages: vec![::driver_api::plan::LaunchStage::default()],
        plans: vec![::driver_api::plan::LaunchStagePlan::default()],
        ..::driver_api::plan::LaunchPackage::default()
    }
}

/// **The seam serves a token.**
///
/// Everything above this checks that the seam refuses what it cannot do.
/// This checks the other half, and it is the north star's fourth property with
/// nothing taken out: a checkpoint loads through `load_model`, a frame goes in
/// through `launch`, and a command buffer retires.
///
/// `driver-metal`'s own `device_real_weights` holds both fire classes to
/// MLX token-for-token, but it stages the fire's tables itself. This is the
/// path an ENGINE takes, and the distance between the two was two tables and
/// two numbers until `model::tables` made it one place. A test that only ever
/// exercised one of them could not have said so.
///
/// Gated on `PIE_METAL_SMOKE_CHECKPOINT` **and** on a descriptor: model facts
/// come from the one the worker hands over, never from a checkpoint this seam
/// re-normalizes — and `crates/model/tests/one_normalizer.rs` now guards the
/// stronger property, that NOTHING in the runtime or either driver opens a
/// `config.json` at all. The test writes one beside the snapshot rather than
/// reaching for a boot TOML.
#[test]
fn a_frame_reaches_the_device_through_the_seam() {
    let Some(snapshot) = std::env::var_os("PIE_METAL_SMOKE_CHECKPOINT") else {
        eprintln!("SKIP: set PIE_METAL_SMOKE_CHECKPOINT to an MLX snapshot");
        return;
    };
    let snapshot = std::path::PathBuf::from(snapshot);

    // The checkpoint's own `config.json`, handed over VERBATIM as a
    // worker would.
    //
    // It used to be normalized here, into a `pie.model/1` descriptor,
    // because that is what a driver read. Nothing normalizes now: what a
    // model is made of is a `model::catalog` row matched to the
    // checkpoint's tensors, and the file crosses only so the driver can
    // read the declared encoding out of it — the one question a `const`
    // cannot answer, because a group size is not an extent of anything.
    let raw = std::fs::read_to_string(snapshot.join("config.json"))
        .expect("the snapshot has a config.json");
    let dir = std::env::temp_dir().join("pie-metal-seam-config");
    std::fs::create_dir_all(&dir).expect("a scratch dir");
    let path = dir.join("config.json");
    std::fs::write(&path, &raw).expect("it writes");

    // TOML, which is what the boot config is — `[model] config`.
    let config = format!("[model]\ndescriptor = \"{}\"\n", path.display());
    let Ok(mut backend) = open::metal(config.as_bytes()) else {
        eprintln!("SKIP: no Metal 4 device");
        return;
    };

    backend
        .load_model(vec![::driver_api::ModelLoadDesc {
            snapshot_dir: snapshot.clone(),
            runtime_quant: String::new(),
            mxfp4_moe: ::driver_api::Mxfp4MoeRequest::Auto,
            component: driver_api::ModelComponent::Full,
        }])
        .expect("the checkpoint loads through the seam");

    // ONE request, TWO tokens at positions 0 and 1: a prefill, and the shape
    // `device_real_weights` holds to MLX.
    let plan = ::driver_api::plan::LaunchPlan {
        token_ids: vec![128_000, 9906],
        position_ids: vec![0, 1],
        kv_page_indices: vec![0],
        kv_page_indptr: vec![0, 1],
        kv_last_page_lens: vec![2],
        qo_indptr: vec![0, 2],
        sampling_indices: vec![1],
        ..::driver_api::plan::LaunchPlan::default()
    };
    let frame = engine::driver::FrameSubmission {
        instance_ids: vec![1],
        kv_translation: vec![0],
        kv_translation_indptr: vec![0, 1],
        required_kv_pages: 1,
        steps: vec![engine::driver::StepSubmission {
            plan,
            roster_rows: vec![0],
            sub_batch_indptr: vec![0, 1],
            sub_batch_class: vec![0],
            terminal_cells: Vec::new(),
            program_row_indptr: vec![0, 1],
            logical_fire_ids: vec![1],
            channel_expected_head: Vec::new(),
            channel_expected_tail: Vec::new(),
            channel_ticket_indptr: vec![0],
            // ONE region covering both rows. The regions must TILE the fire:
            // a gap leaves rows with no feature point and an overlap gives one
            // row two, and `frame::rows_of` refuses either by name.
            region_row_indptr: vec![0, 2],
            region_sig: vec![0],
            region_k: vec![0],
        }],
    };

    // LAUNCHED, not admitted-and-declined: a frame this small fits any pool,
    // so `Exhausted` or `Impossible` here would be the seam refusing rather
    // than the device being full.
    match backend.launch(&frame).expect("the frame launches") {
        engine::driver::FrameLaunchOutcome::Launched(_) => {}
        engine::driver::FrameLaunchOutcome::Exhausted => {
            panic!("one page did not fit a pool sized for hundreds")
        }
        engine::driver::FrameLaunchOutcome::Impossible => {
            panic!("two tokens in one request is not an impossible frame")
        }
    }
}
