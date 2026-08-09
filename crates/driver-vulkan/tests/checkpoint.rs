//! Do the names a plan states exist in a checkpoint?
//!
//! The crate's one remaining structural gap is that nothing loads weights.
//! `lib.rs` says why: `Arg::Weight` carries a name and no WIDTH, so a plan does
//! not say how large a tensor is, and every whole-plan test here holds one
//! four-megabyte block under all 704 names -- which works only because
//! `TokenIds` is all zeros and every gather reads row zero.
//!
//! A checkpoint has the sizes. What it does not obviously have is the NAMES: a
//! plan says `layer.27.down` and a checkpoint says whatever the publisher's
//! safetensors said. Whether a loader is a lookup or a conversion is the
//! difference between an hour of work and a component, and nobody had
//! measured it. So this measures it. It is not a loader and does not pretend
//! to be one.
//!
//! # What it measured
//!
//! Zero of 704, on a real `Qwen/Qwen3-0.6B` snapshot. Not one weight name a
//! qwen3 plan states is a tensor name that checkpoint holds. The plan says
//! `layer.0.down`; the checkpoint says `model.layers.0.mlp.down_proj.weight`.
//! And the plan wants `embed.scales` and `embed.zeros`, which a bfloat16
//! checkpoint does not hold under ANY spelling -- they are outputs of
//! quantizing rather than tensors anyone published.
//!
//! So a weight loader for this crate is a CONVERSION, not a lookup, and it
//! belongs above a driver rather than in one: a driver that knew how to turn
//! `model.layers.0.mlp.down_proj.weight` into `layer.0.down` plus scales and
//! zeros would be a driver with opinions about checkpoint conventions.
//! `model-loader`'s `plan::compile` is where that already lives. What
//! `driver-vulkan` owes is what it already has -- `Weights::hold`, which takes
//! a name and bytes and asks nothing about where they came from.
//!
//! # Then the conversion was run, and it still disagreed
//!
//! `the_loader_states_the_names_this_driver_binds` does the second half:
//! compile the load plan the way `model-loader` would and hold the text's
//! names against what THAT publishes rather than against raw safetensors.
//! **704 of 704 still disagree**, because the two sides use different
//! conventions rather than different files -- the loader publishes
//! `layers.0.mlp.down_proj.biases` where the text binds `layer.0.down.zeros`.
//!
//! That is what `src/names.rs` translates, and running the same comparison
//! through `Naming::mlx()` leaves **nothing** over. Which means a real weight
//! load is no longer blocked on naming; it is blocked only on there being an
//! executor that stages the bytes.
//!
//! # Two facts about real artifacts, found on the way
//!
//! **A stock `Qwen/Qwen3-0.6B` cannot be identified by this build.**
//! `catalog::identify` answers `qwen3-0.6b: unexpected lm_head`. That
//! snapshot's `config.json` says `tie_word_embeddings: true` and its
//! `model.safetensors` publishes `lm_head.weight` *and*
//! `model.embed_tokens.weight` -- tied and exported anyway, which is ordinary
//! for an HF export and which the manifest treats as a contradiction. Left
//! where it was found: loosening another crate's manifest from a driver's
//! test is how a refusal that meant something becomes one nobody remembers.
//!
//! **An unquantised checkpoint cannot reach this path at all.** The contract
//! answers `Metal llama needs quantized weights: this checkpoint carries no
//! `.scales` tensors`. So the artifact these two tests want is a
//! pre-quantised one; the numbers above were measured against
//! `mlx-community/Qwen3-0.6B-4bit`, whose `model.embed_tokens.weight` is
//! `[151936, 128]` -- 1024 four-bit values packed eight to a word.
//!
//! # A second finding, about the artifacts on this machine
//!
//! `~/.cache/pie/models/{qwen-3-0.6b,llama-3.2-1b-instruct}` look like the
//! obvious inputs and are not readable here: both begin `ZTEN0001`, while the
//! `ztensor` 2.1.1 this workspace resolves opens on
//! `89 5a 54 32 0d 0a 1a 0a` (`format::MAGIC`) and answers `cannot detect the
//! format`. They are v1 artifacts under a v2 reader. The HF snapshot cache is
//! readable, which is what the number above was measured against.
//!
//! # A second model, and what it changed
//!
//! One checkpoint cannot tell a CONVERSION from a table that happens to spell
//! one model's names, so a second was added: `mlx-community/Qwen2.5-1.5B-`
//! `Instruct-4bit`, a different generation with a different role set (no
//! qk-norm), a different width, and 648 bound weights against qwen3's 704.
//! `PIE_CHECKPOINT` takes a colon-separated list, each snapshot names its own
//! fixture by its embed width, and running the same two comparisons over both
//! leaves nothing over on either. Feeding the SAME snapshot twice is refused,
//! because two runs of one model would prove once.
//!
//! **The four-megabyte block was one model's arithmetic.** For qwen3-0.6B
//! exactly three weights are larger than the block every whole-plan test in
//! `tests/device.rs` holds a name under -- `embed` and its two sidecars. For
//! qwen2.5-1.5B it is **eighty-seven**: that model's mlp is 8960 wide, so
//! `gate_proj`, `up_proj` and `down` overflow in all 28 layers. On this card
//! an undersized weight is not a fault -- an out-of-bounds storage read
//! returns zero, silently -- so the guess that was safe for one model reads
//! past the end of 84 buffers for the other, and says nothing.
//!
//! # A finding about the Metal text, not about this driver
//!
//! Qwen2.5 has attention biases: `LlamaLikeFacts::qwen2_5_1_5b` states
//! `qkv_bias: true`, the semantic text and the CUDA text both add
//! `{q,k,v}_proj.bias` to the raw projections, and the checkpoint ships all 84
//! of them. **The Metal text ignores the fact entirely** -- measured: the
//! lowered qwen2.5 decode plan binds 648 weights and not one of them is a
//! bias, and its kernel set is the same nine `affine_qmv`/`rms`/`rope`/`sdpa`
//! points a qwen3 decode uses.
//!
//! "Slightly wrong, silently" turned out to be generously put.
//! `tests/device.rs` serves this checkpoint on the card and it does not
//! continue a pattern it was shown; a numpy forward of the same weights
//! continues the pattern WITH the biases and reproduces the card's wrong
//! answer exactly without them.
//!
//! That is `crates/model`'s text and `driver-metal`'s contract as much as this
//! driver's, and it is recorded rather than worked around: this crate cannot
//! bind a weight no plan asks for, and inventing the names here would mean a
//! driver deciding what a model computes.
//!
//! # Why it skips rather than fails without a checkpoint
//!
//! `PIE_CHECKPOINT` names snapshot directories, colon-separated. With none,
//! this prints and
//! returns: a test that passed silently on a machine with no artifact would
//! be reporting the absence of the checkpoint as the presence of agreement,
//! and a test that FAILED there would be this crate reporting someone else's
//! missing download.

/// A plan's weight names and a checkpoint's tensor names overlap completely or
/// not at all -- never partly.
#[test]
fn the_names_a_plan_states_are_names_a_checkpoint_holds() {
    let mut seen = 0usize;
    for dir in snapshots() {
        seen += 1;
        raw_names_agree(&dir);
    }
    if seen == 0 {
        eprintln!("no PIE_CHECKPOINT, so the name agreement is unmeasured");
    }
}

/// One snapshot's half of the test above.
fn raw_names_agree(dir: &str) {
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::FireClass;

    // Skipping on an unreadable checkpoint rather than failing, because the
    // reason is a fact about the artifact and not about this crate -- see the
    // module doc. The error is printed in full so that a skip can never be
    // read as a pass.
    let meta = match model_loader::checkpoint::read::parse_checkpoint_metadata(
        std::path::Path::new(dir),
    ) {
        Ok(meta) => meta,
        Err(e) => {
            eprintln!("{dir} is not readable as a checkpoint ({e}), so the names are unmeasured");
            return;
        }
    };
    let Some(fixture) = fixture_of(dir, &meta) else {
        return;
    };
    let held: std::collections::BTreeSet<&str> =
        meta.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        held.len() > 100,
        "only {} tensors, so this is not a whole checkpoint",
        held.len()
    );

    let plan = llama_like_metal(
        &(fixture.facts)(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let low = lower(
        &plan,
        &[Row::default()],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the plan lowers");
    let wanted: std::collections::BTreeSet<&str> = low
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) => Some(n.as_str()),
            _ => None,
        })
        .collect();
    assert!(wanted.len() > 500, "{} weight names", wanted.len());

    // The interesting number is not "how many are missing" but "how many
    // overlap", because the two safe answers are ALL and NONE and the
    // dangerous answer is in between. None means a loader must convert; all
    // means it can look up. A partial overlap means a loader could load the
    // names that happen to agree, leave the rest at whatever the arena held,
    // and produce logits -- wrong ones, with nothing refused.
    let shared = wanted.iter().filter(|n| held.contains(*n)).count();
    assert!(
        shared == 0 || shared == wanted.len(),
        "{shared} of {} plan names are also checkpoint tensors. Neither none nor all, which is \
         the one answer a loader cannot act on: it would load the agreeing names and silently \
         leave the rest unwritten.",
        wanted.len()
    );

    // What it measured on a real `Qwen/Qwen3-0.6B` snapshot: ZERO of 704. The
    // plan says `layer.0.down` and the checkpoint says
    // `model.layers.0.mlp.down_proj.weight`, and the plan also wants
    // `embed.scales` and `embed.zeros`, which a bfloat16 checkpoint does not
    // contain in any spelling -- they are outputs of quantizing, not tensors
    // anyone published.
    //
    // So a weight loader for this crate is not a lookup and cannot be. It is
    // the conversion `model-loader`'s `plan::compile` already exists to
    // describe, and it belongs above a driver: a driver that knew how to turn
    // `model.layers.0.mlp.down_proj.weight` into `layer.0.down` plus scales
    // and zeros would be a driver that had opinions about checkpoints.
    //
    // That is the finding this file was written to get, and it settles the
    // shape of the work rather than leaving it to be guessed at.
    if shared == 0 {
        eprintln!(
            "none of {} plan names are checkpoint tensors; loading is a conversion, not a lookup",
            wanted.len()
        );
    }
}

/// A checkpoint this file knows how to recognise, and the text it belongs to.
///
/// The snapshot names itself: `model.embed_tokens.weight`'s shape is enough to
/// tell these apart, and it is a fact the file carries rather than one an
/// environment variable claims. That matters because the failure mode of
/// guessing wrong is not an error -- it is this file reporting the FIXTURE's
/// names as ones the checkpoint is missing, which reads exactly like a loader
/// defect. `driver-metal`'s own version of this gate records having made that
/// mistake.
struct Fixture {
    /// The catalog row, taken by id. See `compiled_plan_for` for why not
    /// `catalog::identify`.
    id: &'static str,
    /// The forward-facts fixture whose text states the names.
    facts: fn() -> model::shared::llama_like::forward::facts::LlamaLikeFacts,
    /// `model.embed_tokens.weight`'s shape in the 4-bit snapshot, which is the
    /// PACKED width: eight four-bit values to a word.
    embed: &'static [i64],
    /// How many weights the decode text binds, sidecars included.
    bound: usize,
    /// Their total size in bytes.
    ///
    /// Pinned rather than printed, because the point of these numbers is that
    /// they can CHANGE: a contract that started fusing projections, or a
    /// target that changed a block layout, moves them, and a test that only
    /// printed would let that happen quietly.
    total: u64,
    /// `embed`'s own size. It is the widest in both fixtures -- the tied table
    /// read at both ends of the text.
    embed_bytes: u64,
    /// How many weights are larger than the four-megabyte block every
    /// whole-plan test in `tests/device.rs` holds a name under.
    ///
    /// Three for qwen3-0.6B and **eighty-seven** for qwen2.5-1.5B, which is
    /// the finding that made this field exist. The 84 extra are that model's
    /// mlp projections: 1536 x 8960 at four bits is 6.9 MiB, so `gate_proj`,
    /// `up_proj` and `down` of all 28 layers overflow. On this card an
    /// undersized weight is not a fault -- an out-of-bounds storage read
    /// returns zero, silently -- so a junk-weight whole-plan fire against a
    /// qwen2.5 text would read past the end of 84 buffers and report nothing.
    over: usize,
}

/// Every checkpoint shape this file is written against.
///
/// Two, and the second is not a duplicate of the first: qwen2.5 is a different
/// generation with a different role set (no qk-norm) and a different width, so
/// a `Naming` table that had quietly specialised to qwen3 would leave names
/// over here and nowhere else.
const FIXTURES: &[Fixture] = &[
    Fixture {
        id: "qwen3-0.6b",
        facts: model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen3_0_6b,
        embed: &[151_936, 128],
        bound: 704,
        total: 335_372_288,
        embed_bytes: 77_791_232,
        over: 3,
    },
    Fixture {
        id: "qwen2.5-1.5b",
        facts: model::shared::llama_like::forward::facts::LlamaLikeFacts::qwen2_5_1_5b,
        embed: &[151_936, 192],
        bound: 648,
        total: 868_432_896,
        embed_bytes: 116_686_848,
        over: 87,
    },
];

/// The snapshot directories `PIE_CHECKPOINT` names, which may be several.
///
/// Colon-separated, the way a `PATH` is. One checkpoint proves the conversion
/// works; a second proves it is a CONVERSION and not a table that happens to
/// spell one model's names, and that is a different claim.
fn snapshots() -> Vec<String> {
    match std::env::var("PIE_CHECKPOINT") {
        Ok(v) => v
            .split(':')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// The fixture whose embed width this checkpoint's is, or `None` with the
/// reason printed.
fn fixture_of(
    dir: &str,
    meta: &model_loader::checkpoint::CheckpointMetadata,
) -> Option<&'static Fixture> {
    let shape = meta
        .tensors
        .iter()
        .find(|t| t.name == "model.embed_tokens.weight")
        .map(|t| t.shape.clone())
        .unwrap_or_default();
    let found = FIXTURES.iter().find(|f| shape == f.embed);
    if found.is_none() {
        eprintln!("{dir} embeds {shape:?}, which is no fixture this file states");
    }
    found
}

/// Every weight name this fixture's decode plan binds, sidecars included.
///
/// A `scale.` marker is left out: it is a constant riding the weight slot
/// rather than a tensor, so no loader publishes one and no binder looks one
/// up.
fn names_the_text_binds(fixture: &Fixture) -> std::collections::BTreeSet<String> {
    use model::shared::llama_like::forward::facts::LlamaLikeMetalFacts;
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::FireClass;

    let text = llama_like_metal(
        &(fixture.facts)(),
        &LlamaLikeMetalFacts::synthetic(),
        FireClass::Decode,
    );
    let low = lower(
        &text,
        &[Row::default()],
        Fire {
            captures_across_splits: false,
        },
    )
    .expect("the plan lowers");
    let wanted: std::collections::BTreeSet<String> = low
        .args
        .iter()
        .filter_map(|a| match a {
            Arg::Weight(n) => Some(n.clone()),
            _ => None,
        })
        .filter(|n| !n.starts_with("scale."))
        .collect();
    assert!(wanted.len() > 500, "{} weight names", wanted.len());
    wanted
}

/// The load plan `model-loader` would compile for whatever `PIE_CHECKPOINT`
/// names, or `None` with the reason printed.
///
/// Shared, because compiling it is the fiddly half and a second copy would be
/// a second place for the skip conditions to drift apart.
fn compiled_plan_for(dir: &str) -> Option<(model_loader::plan::LoadPlan, &'static Fixture)> {
    let path = std::path::Path::new(dir);
    let Ok(meta) = model_loader::checkpoint::read::parse_checkpoint_metadata(path) else {
        eprintln!("{dir} is not readable as a checkpoint, so the names are unmeasured");
        return None;
    };
    let fixture = fixture_of(dir, &meta)?;
    // THE ROW BY NAME, NOT BY `catalog::identify`, and the reason is a finding
    // rather than a shortcut.
    //
    // `identify` REFUSES this snapshot. Measured, on a stock
    // `Qwen/Qwen3-0.6B` HuggingFace snapshot: "qwen3-0.6b: unexpected
    // lm_head". Its `config.json` says `tie_word_embeddings: true` and its
    // `model.safetensors` publishes `lm_head.weight` AND
    // `model.embed_tokens.weight` -- the head is tied and exported anyway,
    // which is ordinary for an HF export and which the catalog's manifest
    // treats as a contradiction. So this build serves no row for the most
    // widely downloaded qwen3 artifact there is.
    //
    // That is a fact about `crates/model`'s catalog and not about this
    // driver, and it is left where it was found rather than worked around
    // from here: a driver crate quietly loosening another crate's manifest is
    // how a refusal that meant something becomes a refusal nobody remembers.
    // What it costs HERE is only the identification, and this test is about
    // the NAMES, so the row is taken by id and the shape is checked below.
    let Some(row) = model::catalog::find(fixture.id) else {
        eprintln!(
            "this build has no `{}` row, so the names are unmeasured",
            fixture.id
        );
        return None;
    };
    let config =
        match model_loader::checkpoint::read::read_meta(&meta, model::encoding::CONFIG_OBJECT) {
            Ok(Some(bytes)) => String::from_utf8(bytes).expect("the embedded config is utf8"),
            _ => match std::fs::read_to_string(path.join("config.json")) {
                Ok(text) => text,
                Err(e) => {
                    eprintln!("{dir}/config.json: {e}, so the names are unmeasured");
                    return None;
                }
            },
        };
    let encoding = model::encoding::Encoding::from_config_json(&config)
        .expect("the config states an encoding");

    // `BackendKind::Vulkan`, which this crate's work added to `model-loader`.
    // It was Metal's for as long as there was no Vulkan arm, and the note
    // that stood here recorded WHY that was tolerable: the plan Metal's
    // target produces is one this driver consumes correctly, measured, so
    // what a Vulkan arm would have to differ in was a question with evidence
    // behind it rather than a guess. The evidence said: nothing. The arm is
    // its own statement anyway, because "they agree today" is a fact about
    // today and an alias would make it a fact forever.
    let target = model_loader::plan::StorageTarget::for_backend(
        model_loader::types::BackendKind::Vulkan,
        0,
        1,
    );
    assert_eq!(
        target.tile_map_mask,
        model_loader::plan::StorageTarget::for_backend(
            model_loader::types::BackendKind::Metal,
            0,
            1
        )
        .tile_map_mask,
        "the Vulkan target no longer admits what Metal's does, so the plan this \
         test measures is not the one the device suite was proven against -- \
         re-measure it there before changing this"
    );
    let (plan, _) = match model::boot::compile_load_plan_for(
        path,
        &meta,
        &target,
        row,
        &encoding,
        model::boot::Binding::MLX_IN_PLACE,
    ) {
        Ok(both) => both,
        Err(e) => panic!(
            "the loader would not compile a plan for `{}`: {e}",
            row.id()
        ),
    };
    Some((plan, fixture))
}

/// The loader's own plan states the names this driver's plan wants.
///
/// The test above measured the raw checkpoint and found zero of 704, which
/// settled that loading is a CONVERSION. It did not check the other half of
/// that sentence -- that the conversion already exists and produces what this
/// driver asks for. That was reasoned from reading `model-loader` and never
/// run, which is the kind of claim this crate is supposed to measure.
///
/// So this runs it: the same snapshot, through `catalog::identify` ->
/// `contract::author` -> `plan::compile`, and compares the tensor names the
/// compiled plan publishes against the weight names the lowering states.
///
/// # What it needs from a driver that does not exist yet
///
/// Nothing here executes the plan. `plan::compile` takes a `StorageTarget`
/// whose `BackendKind` is `Cuda`, `Metal` or `Unknown` -- there is no Vulkan
/// arm -- and executing it needs a `model-loader` executor this crate has not
/// written. What a target changes is alignment, tile budget and which
/// transforms are claimed; it does not change what the tensors are CALLED.
/// So the names are measurable now and the bytes are not, and this measures
/// the half that is measurable rather than waiting for the half that is not.
#[test]
fn the_loader_states_the_names_this_driver_binds() {
    let mut measured: Vec<&str> = Vec::new();
    for dir in snapshots() {
        let Some((plan, fixture)) = compiled_plan_for(&dir) else {
            continue;
        };
        measured.push(fixture.id);
        names_agree(&plan, fixture);
    }
    if measured.is_empty() {
        eprintln!("no snapshot named a fixture this file states, so the names are unmeasured");
    }
    only_once(&measured);
}

/// Two snapshots of the SAME model would run a test twice and prove it once.
///
/// The claim these fixtures exist to make is that the conversion is a
/// conversion rather than one model's spelling table, and that claim needs the
/// runs to be distinct.
fn only_once(measured: &[&str]) {
    let distinct: std::collections::BTreeSet<&str> = measured.iter().copied().collect();
    assert_eq!(
        distinct.len(),
        measured.len(),
        "the same fixture was measured twice: {measured:?}"
    );
}

/// One snapshot's half of the test above.
///
/// Split out so the loop stays a loop and the reasoning stays in one place:
/// every claim below is about a plan and a text, not about which model they
/// belong to, and a version of this written twice would be two places for the
/// controls to drift apart.
fn names_agree(plan: &model_loader::plan::LoadPlan, fixture: &Fixture) {
    let published: std::collections::BTreeSet<&str> =
        plan.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        published.len() > 100,
        "the plan published {} tensors, so it is not a whole model",
        published.len()
    );

    let wanted = names_the_text_binds(fixture);
    let wanted: std::collections::BTreeSet<&str> = wanted.iter().map(String::as_str).collect();

    // THROUGH THE TRANSLATION, which is the whole point. Held directly, all
    // 704 disagree -- the two sides share no convention, and that measurement
    // is what `src/names.rs` was written for. What is under test here is
    // whether the table is TOTAL: a role it does not carry, or a spelling
    // this loader does not publish, shows up as a name left over.
    let naming = driver_vulkan::names::Naming::mlx();
    let missing: Vec<&str> = wanted
        .iter()
        .copied()
        .filter(|n| {
            let spellings = naming.spellings(n);
            // An empty answer is a name outside the text's shape, and it is
            // deliberately NOT treated as a match: it would turn a role the
            // table has never heard of into a silent pass.
            !spellings.is_empty() && !spellings.iter().any(|s| published.contains(s.as_str()))
        })
        .collect();
    // The negative control for the table itself. Without it every name is
    // left over, so a table that had quietly stopped translating would not
    // read as "still fine".
    let raw = wanted.iter().filter(|n| !published.contains(*n)).count();
    assert_eq!(
        raw,
        wanted.len(),
        "the raw names already agree, so the translation is not what made this pass"
    );
    // And a name the table cannot decompose must be reported rather than
    // skipped, or the filter above would hide drift.
    let undecomposed: Vec<&str> = wanted
        .iter()
        .copied()
        .filter(|n| naming.spellings(n).is_empty())
        .collect();
    assert!(
        undecomposed.is_empty(),
        "{} names this text binds are not in `Naming`'s shape at all: {:?}",
        undecomposed.len(),
        &undecomposed[..undecomposed.len().min(8)]
    );
    // The same all-or-nothing shape as the test above, and for the same
    // reason: a partial answer is the one a driver cannot act on, because it
    // would bind the agreeing names and leave the rest at whatever the arena
    // held.
    assert!(
        missing.is_empty(),
        "{} of {} names the text binds are not tensors the loader publishes; the first few \
         are {:?}. Loading is a conversion, and this is the conversion not being the one \
         this driver needs.",
        missing.len(),
        wanted.len(),
        &missing[..missing.len().min(8)]
    );
}

/// How large every weight this driver binds actually is.
///
/// # The number `lib.rs` says does not exist
///
/// "A plan does not say how large a tensor is" is true of a PLAN, and every
/// whole-plan test in `tests/device.rs` works around it by holding one
/// four-megabyte block under all 704 names. That guess is load-bearing and
/// unchecked, and on this card an undersized weight does not fail: an
/// out-of-bounds storage read returns zero, silently, with the validation
/// layer saying nothing. It cost a whole debugging session once already --
/// `embed` at 4 MiB made every logit `-0`.
///
/// A load plan has the widths, because `TensorDecl` carries a shape and an
/// encoding and `encoding_nbytes` turns those into bytes. So this asks it,
/// and the answer is what a real load would have to allocate.
#[test]
fn the_loader_states_how_large_every_weight_this_driver_binds_is() {
    let mut measured: Vec<&str> = Vec::new();
    for dir in snapshots() {
        let Some((plan, fixture)) = compiled_plan_for(&dir) else {
            continue;
        };
        measured.push(fixture.id);
        widths_are_stated(&plan, fixture);
    }
    if measured.is_empty() {
        eprintln!("no snapshot named a fixture this file states, so the widths are unmeasured");
    }
    only_once(&measured);
}

/// One snapshot's half of the test above.
fn widths_are_stated(plan: &model_loader::plan::LoadPlan, fixture: &Fixture) {
    let naming = driver_vulkan::names::Naming::mlx();
    let by_name: std::collections::HashMap<&str, &model_loader::types::TensorDecl> =
        plan.tensors.iter().map(|t| (t.name.as_str(), t)).collect();

    let mut total: u64 = 0;
    let mut widest: (String, u64) = (String::new(), 0);
    // The guess `tests/device.rs` holds every name under, and the count of
    // names it is too small for.
    const GUESS: u64 = 4 << 20;
    let mut over = 0usize;
    let mut measured = 0usize;
    for traced in names_the_text_binds(fixture) {
        let spellings = naming.spellings(&traced);
        let Some(decl) = spellings.iter().find_map(|s| by_name.get(s.as_str())) else {
            panic!("`{traced}` resolves to none of {spellings:?}");
        };
        let bytes = model_loader::types::encoding_nbytes(&decl.shape, &decl.encoding)
            .unwrap_or_else(|| panic!("`{}` states no width", decl.name));
        assert!(bytes > 0, "`{}` is zero bytes", decl.name);
        total += bytes;
        measured += 1;
        if bytes > widest.1 {
            widest = (traced.clone(), bytes);
        }
        if bytes > GUESS {
            over += 1;
        }
    }
    eprintln!(
        "{}: {measured} weights, {total} bytes, widest {} at {}, {over} over the guess",
        fixture.id, widest.0, widest.1
    );
    assert_eq!(
        measured, fixture.bound,
        "the text binds {measured} weights, not {}",
        fixture.bound
    );

    // MEASURED, on the `mlx-community/*-4bit` snapshots these fixtures name.
    assert_eq!(total, fixture.total, "the whole model's bound weights");
    assert_eq!(widest.0, "embed", "the widest weight");
    // 74.2 MiB for qwen3-0.6B, and the reason `tests/device.rs` gives `embed`
    // a real block: the tied table is read at both ends of the text and is
    // nineteen times the guess below.
    assert_eq!(widest.1, fixture.embed_bytes, "`embed` in bytes");
    // For qwen3-0.6B the sum is 78_296 bytes short of the safetensors file it
    // was measured against, which is its header plus the tensors this text
    // does not bind -- so the loader is not silently dropping half a model.
    // For qwen2.5 that gap is 195_663 bytes, and 84 of the tensors in it are
    // the attention biases the Metal text never asks for. See the module doc.

    // AND THE FINDING, which the second fixture changed. For qwen3-0.6B
    // exactly three names are larger than the block every whole-plan test in
    // `tests/device.rs` holds them under -- `embed` and its two sidecars --
    // so the guess is safe for the other 701 and now provably so rather than
    // untested. For qwen2.5-1.5B it is EIGHTY-SEVEN, because that model's mlp
    // is 8960 wide and its three projections overflow in all 28 layers.
    //
    // So "4 MiB is enough for everything except the tied table" was never a
    // rule; it was one model's arithmetic. Recorded here because the failure
    // it guards is silent on this card.
    assert_eq!(
        over, fixture.over,
        "weights larger than the 4 MiB block device.rs holds"
    );
    for sidecar in ["embed", "embed.scales", "embed.zeros"] {
        let decl = naming
            .spellings(sidecar)
            .iter()
            .find_map(|s| by_name.get(s.as_str()).copied())
            .expect("the tied table resolves");
        assert!(
            model_loader::types::encoding_nbytes(&decl.shape, &decl.encoding).unwrap() > GUESS,
            "`{sidecar}` was expected to be one of the three that overflow"
        );
    }
}
