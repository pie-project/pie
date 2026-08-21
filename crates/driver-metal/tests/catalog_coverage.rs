//! Which catalog rows this shell can SERVE, measured rather than listed.
//!
//! This file replaces `family_registry.rs`, and what it replaces is worth
//! stating because the shape of the check changed and not just its subject.
//!
//! `family_registry.rs` reconciled TWO tables that both answered "which
//! `model_type` does Metal support?" — `driver_metal::facts::ModelFamily::of`,
//! a `match` over config strings that picked this driver's geometry, against
//! `model::contract::MLX_ROWS`, a list that picked the author writing the
//! storage contract. They answered different questions (an author is a
//! storage schema; a family is a compute graph) and were forbidden to
//! disagree about the SET, because a `model_type` with a family and no author
//! got a plan-time *"no author for model_type"* while one with an author and
//! no family got a boot-time *"unsupported model_type"* — two unrelated
//! errors, one cause, a family added on one side only.
//!
//! # BOTH TABLES ARE GONE AND THE DISAGREEMENT IS NOT REPRESENTABLE
//!
//! There is one row per model now, it is a `const`, and `catalog::Variant`
//! requires `author` AND `deployment` AND `trace` AND `chat` of every row with
//! no default bodies. A row that can be authored is a row that can be deployed
//! BY CONSTRUCTION — there is no second table to fall out of, so the check
//! that file existed for cannot fail and has nothing to assert.
//!
//! What is left is the part a type still cannot hold, and it is this driver's
//! version of the gap `driver-cuda/tests/catalog_coverage.rs` names: a row may
//! compile, deploy, and still be UNREACHABLE FROM METAL, because this backend
//! has no forward text for its architecture or no kernel for the shape it
//! projects. That is a real gap and it deserves a gate.
//!
//! # The first half of that gap is a row's answer now
//!
//! "No forward text for its architecture" used to be a fact about
//! `driver-metal`: `model/text.rs` held eleven architecture STRINGS and a
//! `serves()` that tested membership. It is a fact about the ROW now —
//! `catalog::Variant::trace` takes a `Deployed` whose `backend` says which
//! driver is asking, and a row with no Metal text refuses there. So the three
//! text gates below ask the catalog rather than a list, which is why none of
//! them names a model except the two the device gates open.
//!
//! The old list's own defect is the argument for the change and it is worth
//! keeping: `LLAMA_LIKE` named `gemma4`, so `serve/load.rs` claimed every
//! gemma-4 at its pre-staging gate and a second refusal ninety lines later
//! rejected it on the sandwich norm — after the 17 GB the first gate exists
//! to avoid. It also omitted `gemma3`, which the same text models.
//!
//! # Why this one derives its answer instead of listing it
//!
//! `driver-cuda`'s equivalent keeps a `NOT_YET_SERVABLE` const and reconciles
//! it both ways, which is the right idiom when the set is small and stable —
//! its list is one entry long. The Metal gap is neither. It moves every time a
//! refusal in `batch/geometry.rs` is retired, and a hand-kept list of a dozen
//! model ids is a list that gets updated by whoever's build broke, which is
//! the failure mode this whole refactor exists to stop.
//!
//! So the assertion here is the INVARIANT rather than the census: **a row this
//! driver cannot serve must be refused for a reason the `Deployment` itself
//! shows**. Not "refused", which any bug produces — refused with a message
//! naming a fact you can point at in the projected value. That is checkable
//! without a list, cannot go stale, and catches the thing a list would miss
//! anyway: a refusal that fires for the WRONG reason, which reads as coverage
//! and is a silent hole.

use std::collections::BTreeSet;

use driver_metal::batch::{AffineFormat, geometry_from_deployment};
use model::catalog::{self, Deployed, MetalBinding};
use model::deployment::{Deployment, KvStyle, RopeScaling};
use model_ir::kernels::Backend;
use model_ir::trace::FireClass;

/// The affine point every row is measured at.
///
/// A row does not state its quantization and must not: `mlx-community`
/// publishes the same weights at 4 bits group 64 and at 8 bits group 32, and
/// the two pack to shapes that are not distinguishable from any tensor's
/// extents. `G64_B4` is the shipped body format, so it is the point at which
/// "can this build launch that shape" is the question an operator is actually
/// asking.
const AT: AffineFormat = AffineFormat::G64_B4;

/// The same point, as the binding a row is asked for a text with.
///
/// Deliberately `model::binding::ANY_ENCODING` rather than a second literal:
/// the driver's pre-staging refusal is answered with that value, and a test
/// that measured coverage at a DIFFERENT binding would be measuring a
/// question the driver never asks. The tests below that vary it say so.
const AT_METAL: MetalBinding = driver_metal::model::binding::ANY_ENCODING;

/// Every catalog row, projected once, keeping the ones that deploy.
///
/// A row that refuses `deployment()` is not this file's business —
/// `driver-cuda/tests/catalog_coverage.rs` owns that reconciliation and holds
/// a stated list for it. Skipping them here rather than failing keeps ONE
/// file authoritative about that gap; two gates on one property is how a list
/// gets updated in one place and read in the other.
fn deployable() -> Vec<(&'static dyn catalog::Variant, Deployment)> {
    catalog::catalog()
        .iter()
        // NOT THE TEST ROWS, and the reason is that this file's numbers must
        // not depend on who else is in the build. `model`'s `test-rows`
        // feature adds a row its own catalog calls "a row that is not a
        // model"; nothing in `driver-metal` asks for the feature, but cargo
        // UNIFIES features across a workspace build, so `cargo test
        // --workspace` handed this census one more row than `cargo test -p
        // driver-metal` did and the two disagreed about a constant.
        //
        // Filtering by the prefix rather than by the `cfg` is deliberate:
        // `a_shipped_catalog_has_no_test_rows` is what keeps the prefix
        // meaningful, and a `cfg` here would have to be repeated at every
        // count below.
        .filter(|row| !row.id().starts_with("test-"))
        .filter_map(|row| row.deployment(Deployed::single()).ok().map(|d| (*row, d)))
        .collect()
}

/// An ordinary deployment projects; an extraordinary one may refuse.
///
/// # The two halves, and why the second one is the interesting half
///
/// A row this driver can serve must project a `DecodeGeometry`, and the
/// numbers in it must be the ones the `Deployment` states — not a rounding of
/// them, not a default that happens to look right. That is the first half and
/// it is the cheap one.
///
/// The second half is the claim a census of unservable model ids could never
/// make. `geometry_from_facts` once refused a llama config as *"carrying no
/// decoder shape"* while the shape sat in the family-prefixed block the
/// projection had not looked in — and a list-shaped test would have recorded
/// llama as unservable, gone green, and said nothing. So instead of listing
/// which rows refuse, this asks whether the row that refused had ANYTHING
/// about it to refuse: see [`extraordinary`].
#[test]
fn an_ordinary_row_projects_a_geometry_and_an_extraordinary_one_may_refuse() {
    let mut unexplained = Vec::new();
    let mut served = 0usize;
    let mut refused = 0usize;

    for (row, d) in deployable() {
        let id = row.id();
        let arch = d.advertised.arch;
        match geometry_from_deployment(&d, row.load_shape(), AT) {
            Ok(g) => {
                served += 1;
                // The projection is not allowed to invent. Every number below
                // is one `lowering/consts.rs` binds into a kernel argument
                // buffer, and a mismatch here is a kernel reading the wrong
                // extent off the right pointer — which does not fault, it
                // returns fluent nonsense.
                assert_eq!(g.n_layers, d.layers, "{id}: layers");
                assert_eq!(g.hidden, d.shape.hidden, "{id}: hidden");
                assert_eq!(g.n_q_heads, d.shape.q_heads, "{id}: q_heads");
                assert_eq!(g.n_kv_heads, d.shape.kv_heads, "{id}: kv_heads");
                assert_eq!(g.vocab, d.shape.vocab, "{id}: vocab");
                assert_eq!(g.eps, d.norm_eps, "{id}: norm eps");
                assert!(g.head_dim > 0, "{id}: a served geometry with no head dim");
                assert_eq!(
                    g.quant, AT,
                    "{id}: the affine point is the caller's, not the row's"
                );
            }
            Err(why) => {
                refused += 1;
                assert!(
                    !why.0.trim().is_empty(),
                    "{id} (`{arch}`) is refused with an empty sentence; a \
                     refusal an operator cannot read is a crash with extra \
                     steps"
                );
                if extraordinary(&d).is_none() {
                    unexplained.push(format!("{id} (`{arch}`): {}", why.0));
                }
            }
        }
    }

    assert!(
        unexplained.is_empty(),
        "these rows project an ORDINARY deployment — dense FFN, paged KV, no \
         recurrent slab, one head dim, one window, a head count GQA groups — \
         and `geometry_from_deployment` refused them anyway. Every refusal in \
         `batch/geometry.rs` is guarded by a condition none of these meet, so \
         each line below is a refusal firing on the wrong fact:\n  {}",
        unexplained.join("\n  ")
    );

    // The audit has to be able to fail. A catalog that enumerated nothing —
    // a feature gate flipped, a `catalog()` that returns an empty slice —
    // passes every assertion above in silence, which is the exact failure
    // `mod_audit.rs` and `layering.rs` both guard their own scans against.
    // FIFTY-THREE ROWS, FORTY-TWO OF THEM SERVED.
    //
    // This was `served + refused > 10` and `served > 0`, which is five times
    // below the census and catches only the empty-slice case the comment
    // above describes. The failure it could not see is the one worth seeing:
    // a row moving from SERVED to REFUSED. That is a model this backend used
    // to project and no longer does, and it changes neither the deployment
    // count nor the fact that something is still served, so both floors held
    // while the backend lost a family.
    //
    // Both numbers move when the catalog does, which is a deliberate act --
    // adding a model is a commit that should say so here too.
    assert_eq!(
        served + refused,
        53,
        "the catalog is {} rows deployed; this gate expects 53. A model \
         added or retired moves it.",
        served + refused
    );
    assert_eq!(
        served,
        42,
        "{served} of the {} deployed rows project a Metal geometry and this \
         gate expects 42. A FALL here with the total unchanged is a row that \
         stopped projecting -- a model this backend used to serve and now \
         refuses by name -- which is the regression neither floor could see.",
        served + refused
    );
}

/// What is EXTRAORDINARY about a deployment, or `None` if nothing is.
///
/// # Why the predicate runs this way round
///
/// The first draft of this file matched refusal MESSAGES: a sentence about
/// top-k was allowed only over a mixture, one about two head dims only over a
/// stack that has two. It read well and it was the wrong shape, because it
/// pinned every refusal's wording in a file that has no other reason to know
/// it — rename a sentence and a coverage gate goes red for a change that
/// altered nothing.
///
/// The claim worth making is the contrapositive and it needs no wording at
/// all: **a deployment with nothing unusual in it MUST project a geometry.**
/// Every refusal in `batch/geometry.rs` is guarded by one of the conditions
/// below — a mixture, a recurrent slab, a latent KV plane, a sliding
/// schedule, two head dims, a head count GQA cannot group, a zero where a
/// width belongs — so a value with none of them has no reachable refusal, and
/// one that is refused anyway is a bug in the ladder rather than a limit of
/// the backend.
///
/// That is exactly the historical failure this replaces a census with. The
/// projection this file's predecessor guarded once refused a llama-3 as
/// *"carrying no decoder shape"* — llama-3 being the most ordinary stack in
/// the catalog — because the reader had put the shape in a family-prefixed
/// block the projection did not look in. A list of unservable ids would have
/// recorded llama as unservable and gone green. This function says llama is
/// ordinary, so a refusal of it is a failure, and it says so without knowing
/// one word of the message.
fn extraordinary(d: &Deployment) -> Option<String> {
    if matches!(d.rope_scaling, Some(RopeScaling::Yarn { .. })) {
        // OLMo 3 is the row this catches: dense, paged, one head dim, one
        // window — ordinary by every other clause here — and rescaled by
        // YaRN, for which this driver derives no frequency table. It is
        // named before the shape clauses because it is a refusal about the
        // LADDER and not about a width, and reading it second would report
        // the wrong reason.
        return Some("a YaRN-rescaled ladder this driver builds no table for".into());
    }
    if d.shape.moe_intermediate > 0 {
        return Some("a routed mixture: `Deployment` states no top-k".into());
    }
    if d.shape.intermediate == 0 {
        return Some("no dense FFN width".into());
    }
    if d.recurrent.is_some() {
        return Some("a recurrent slab".into());
    }
    if !matches!(d.kv, KvStyle::Paged) {
        return Some("a latent KV plane this driver does not page".into());
    }
    if d.layers == 0
        || d.shape.hidden == 0
        || d.shape.vocab == 0
        || d.shape.q_heads == 0
        || d.shape.kv_heads == 0
        || d.shape.head_dim_alloc() == 0
    {
        return Some("a zero where a width belongs".into());
    }
    if !d.shape.q_heads.is_multiple_of(d.shape.kv_heads) {
        return Some("a head count GQA cannot group".into());
    }
    if d.attention.len() != d.layers as usize {
        return Some(format!(
            "{} layers and {} per-layer attention entries",
            d.layers,
            d.attention.len()
        ));
    }
    let head_dims: BTreeSet<u32> = d.attention.iter().map(|a| a.head_dim).collect();
    if head_dims.len() > 1 || head_dims.contains(&0) {
        return Some(format!("{} distinct per-layer head dims", head_dims.len()));
    }
    let windows: BTreeSet<i32> = d.attention.iter().map(|a| a.window).collect();
    if windows != BTreeSet::from([0]) {
        return Some(format!("a sliding schedule: windows {windows:?}"));
    }
    let bases: BTreeSet<u32> = d.attention.iter().map(|a| a.rope_theta.to_bits()).collect();
    if bases.len() > 2 {
        return Some(format!("{} distinct rope bases", bases.len()));
    }
    None
}

/// The checkpoints the device gates actually open stay servable.
///
/// `device_checkpoint_names.rs` and `device_real_weights.rs` are run against
/// real snapshots of these two. If either stops projecting a geometry those
/// gates do not fail — they SKIP, printing a refusal to stderr that nobody
/// reads, and the coverage they represent evaporates without a red build.
/// This is the assertion that turns that into a failure, and it is the same
/// argument `driver-cuda`'s `the_three_live_families_are_servable` makes.
///
/// The text assertion is new and it is the sharper half. `serves` used to be
/// a membership test against `model/text.rs`'s eleven architecture strings,
/// so it could only say that a NAME was listed; it is the row's own answer
/// now, and `Backend::of_family` reads which backend that answer was written
/// for. A row that answered a Metal load with `llama_like.cuda.decode` would
/// pass every gate in this driver and fault at its first dispatch, on a
/// symbol no Metal shader exports.
#[test]
fn the_two_checkpoints_the_device_gates_open_still_project_a_geometry() {
    for want in ["llama-3.2-1b", "qwen3-0.6b"] {
        let row = catalog::find(want).unwrap_or_else(|| {
            panic!("`{want}` must stay in the catalog: the device gates open it")
        });
        let d = row
            .deployment(Deployed::single())
            .unwrap_or_else(|e| panic!("`{want}` must stay deployable: {e}"));
        driver_metal::model::binding::serves(row).unwrap_or_else(|e| {
            panic!(
                "`{want}` advertises `{}` and this build has no Metal text for \
                 it ({e}) — `load_model` would refuse it before staging and \
                 the device gates would skip it",
                d.advertised.arch
            )
        });
        let plan = driver_metal::model::binding::text(row, FireClass::Decode, &AT_METAL)
            .unwrap_or_else(|e| panic!("`{want}` refused a decode text: {e}"));
        assert_eq!(
            Backend::of_family(&plan.family),
            Some(Backend::Metal),
            "`{want}` answered a METAL load with `{}`. Every symbol in it \
             would be looked up in the Metal table at fire time, and the \
             lookup is the only thing standing between a CUDA text and a \
             dispatch",
            plan.family
        );
        let g = geometry_from_deployment(&d, row.load_shape(), AT)
            .unwrap_or_else(|e| panic!("`{want}` must stay launchable: {}", e.0));
        assert!(
            g.n_layers > 0 && g.hidden > 0 && g.vocab > 0 && g.head_dim > 0,
            "`{want}` projected a geometry with a zero in it"
        );
    }
}

/// The pre-staging question is the same question at every fire class.
///
/// `serve/load.rs` asks the row ONCE, before it stages, and it asks at
/// `FireClass::Decode` — but what it is really asking is whether the
/// checkpoint can be served at all, and a served checkpoint prefills before
/// it decodes. A row with a decode text and no prefill text would pass that
/// gate, stage its weights, admit its first request and refuse the fire.
///
/// That is not a hypothetical shape of bug: `Variant::trace`'s own doc
/// records `unbuilt_kv_store()` existing because "a family could hold a facts
/// row, load happily and die at its first fire", and the whole point of
/// moving the refusal to the row was to make the door the place that answers.
/// A door that answers for one class only is half a door.
#[test]
fn the_pre_staging_question_is_answered_the_same_at_every_fire_class() {
    let mut disagree = Vec::new();
    for (row, _) in deployable() {
        let door = driver_metal::model::binding::serves(row).is_ok();
        for class in [FireClass::Decode, FireClass::Prefill] {
            let here = driver_metal::model::binding::text(row, class, &AT_METAL).is_ok();
            if here != door {
                disagree.push(format!(
                    "{}: {class:?} says {here}, the door says {door}",
                    row.id()
                ));
            }
        }
    }
    assert!(
        disagree.is_empty(),
        "these rows answer the Metal question differently at different fire \
         classes. `serve/load.rs` asks once, before it stages 17 GB, and a \
         row that agrees at the door and refuses at the fire has spent all of \
         it to say so:\n  {}",
        disagree.join("\n  ")
    );
}

/// The text this driver runs is the text the ROW states — every row.
///
/// `model::binding::text` is four lines and this test is why it may not grow
/// a fifth. What it replaced was `text::plan_for(arch, class, &facts,
/// &metal)`: a match on an architecture STRING, over facts the driver had
/// rebuilt from nine tensor probes, which is two opportunities per fire to
/// describe a different model than the one that was identified. The claim now
/// is that the driver contributes NOTHING to the text but the binding — no
/// fallback, no adjustment, no second answer for a row it thinks it knows
/// better.
///
/// The one thing it does contribute is a refusal, and this pins its shape:
/// when the row's answer is a Metal text the driver hands that value back
/// UNCHANGED, and when it is not the driver refuses rather than editing it.
/// Those are the only two outcomes. A driver that could rewrite a text would
/// be back to describing models.
///
/// Asked of every deployable row rather than of a fixture, because the defect
/// this replaces was never in the common case. `LLAMA_LIKE` served `llama`
/// correctly for a year; what it got wrong was `gemma4`, which it listed and
/// the load path refused, and `gemma3`, which it omitted and the same text
/// models.
#[test]
fn the_text_this_driver_runs_is_the_text_the_row_states() {
    let bindings = [
        AT_METAL,
        MetalBinding {
            qmm_partial_rows: false,
            qmm_fp16_precast: true,
            qmm_tile: None,
            quant_group: 128,
            quant_bits: 8,
            router_quant_group: 0,
            router_quant_bits: 0,
            ..AT_METAL
        },
    ];
    let mut rows = 0usize;
    let mut passed_through = 0usize;
    for (row, _) in deployable() {
        rows += 1;
        for class in [FireClass::Decode, FireClass::Prefill] {
            for b in &bindings {
                let theirs = row.trace(class, Deployed::metal(b));
                match driver_metal::model::binding::text(row, class, b) {
                    Ok(mine) => {
                        assert_eq!(
                            Ok(&mine),
                            theirs.as_ref(),
                            "`{}` gets a different {class:?} text through this \
                             driver than it states for itself at g{}/b{}",
                            row.id(),
                            b.quant_group,
                            b.quant_bits
                        );
                        assert_eq!(
                            Backend::of_family(&mine.family),
                            Some(Backend::Metal),
                            "`{}` was served `{}`, which is not a Metal text",
                            row.id(),
                            mine.family
                        );
                        passed_through += 1;
                    }
                    // The two refusable shapes, and no third: the row said no,
                    // or the row answered for another backend. Anything else
                    // would be this driver having an opinion about a model.
                    Err(_) => {
                        let stated_metal = theirs
                            .as_ref()
                            .is_ok_and(|p| Backend::of_family(&p.family) == Some(Backend::Metal));
                        assert!(
                            !stated_metal,
                            "`{}` states a Metal {class:?} text and this driver \
                             refused it anyway",
                            row.id()
                        );
                    }
                }
            }
        }
    }
    assert!(
        rows > 10,
        "only {rows} rows deployed; this gate is not reading the catalog"
    );
    assert!(
        passed_through > 0,
        "no row's text reached a fire; the door is shut for everything"
    );
}
