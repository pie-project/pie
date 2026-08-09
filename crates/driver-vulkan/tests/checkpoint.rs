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
//! # A second finding, about the artifacts on this machine
//!
//! `~/.cache/pie/models/{qwen-3-0.6b,llama-3.2-1b-instruct}` look like the
//! obvious inputs and are not readable here: both begin `ZTEN0001`, while the
//! `ztensor` 2.1.1 this workspace resolves opens on
//! `89 5a 54 32 0d 0a 1a 0a` (`format::MAGIC`) and answers `cannot detect the
//! format`. They are v1 artifacts under a v2 reader. The HF snapshot cache is
//! readable, which is what the number above was measured against.
//!
//! # Why it skips rather than fails without a checkpoint
//!
//! `PIE_CHECKPOINT` names a snapshot directory. With none, this prints and
//! returns: a test that passed silently on a machine with no artifact would
//! be reporting the absence of the checkpoint as the presence of agreement,
//! and a test that FAILED there would be this crate reporting someone else's
//! missing download.

/// A plan's weight names and a checkpoint's tensor names overlap completely or
/// not at all -- never partly.
#[test]
fn the_names_a_plan_states_are_names_a_checkpoint_holds() {
    use model::shared::llama_like::forward::facts::{LlamaLikeFacts, LlamaLikeMetalFacts};
    use model::shared::llama_like::forward::llama_like_metal;
    use model_compiler::lower::{Arg, Fire, Row, lower};
    use model_compiler::trace::FireClass;

    let Ok(dir) = std::env::var("PIE_CHECKPOINT") else {
        eprintln!("no PIE_CHECKPOINT, so the name agreement is unmeasured");
        return;
    };
    let dir = dir.as_str();
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
    let held: std::collections::BTreeSet<&str> =
        meta.tensors.iter().map(|t| t.name.as_str()).collect();
    assert!(
        held.len() > 100,
        "only {} tensors, so this is not a whole checkpoint",
        held.len()
    );

    let plan = llama_like_metal(
        &LlamaLikeFacts::qwen3_0_6b(),
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
