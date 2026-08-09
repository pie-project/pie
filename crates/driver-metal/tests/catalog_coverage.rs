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
use model::catalog::{self, Deployed};
use model::deployment::{Deployment, KvStyle, RopeScaling};

/// The affine point every row is measured at.
///
/// A row does not state its quantization and must not: `mlx-community`
/// publishes the same weights at 4 bits group 64 and at 8 bits group 32, and
/// the two pack to shapes that are not distinguishable from any tensor's
/// extents. `G64_B4` is the shipped body format, so it is the point at which
/// "can this build launch that shape" is the question an operator is actually
/// asking.
const AT: AffineFormat = AffineFormat::G64_B4;

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
                assert_eq!(g.quant, AT, "{id}: the affine point is the caller's, not the row's");
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
    assert!(
        served + refused > 10,
        "only {} rows deployed at all; this gate is not reading the catalog",
        served + refused
    );
    assert!(
        served > 0,
        "no catalog row projects a Metal geometry — this backend serves \
         nothing and every device test below it is measuring an empty set"
    );
    eprintln!("{served} rows project a geometry, {refused} are refused by name");
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
    if d.shape.q_heads % d.shape.kv_heads != 0 {
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
#[test]
fn the_two_checkpoints_the_device_gates_open_still_project_a_geometry() {
    for want in ["llama-3.2-1b", "qwen3-0.6b"] {
        let row = catalog::find(want)
            .unwrap_or_else(|| panic!("`{want}` must stay in the catalog: the device gates open it"));
        let d = row
            .deployment(Deployed::single())
            .unwrap_or_else(|e| panic!("`{want}` must stay deployable: {e}"));
        assert!(
            driver_metal::model::text::serves(d.advertised.arch),
            "`{want}` advertises `{}`, which no Metal text serves — the \
             device gates would skip it",
            d.advertised.arch
        );
        let g = geometry_from_deployment(&d, row.load_shape(), AT)
            .unwrap_or_else(|e| panic!("`{want}` must stay launchable: {}", e.0));
        assert!(
            g.n_layers > 0 && g.hidden > 0 && g.vocab > 0 && g.head_dim > 0,
            "`{want}` projected a geometry with a zero in it"
        );
    }
}

/// The text's allow-list names architectures, and three of them are dead.
///
/// # This is a finding, pinned so it stays one
///
/// `model/text.rs`'s `LLAMA_LIKE` is a hand-typed list of architecture
/// strings. Three entries — `llama3`, `llama4` and `qwen3_moe` — are names NO
/// catalog row advertises, and they are dead in a specific and instructive
/// way: the llama generation advertises `llama` for all five of its rows, and
/// the Qwen-3 mixtures advertise `qwen3` exactly like its dense rows, because
/// a MIXTURE IS A FACT OF THE ROW AND NOT A SEPARATE ARCHITECTURE. Those
/// entries are the fossil of a table that keyed on `config.json`'s
/// `model_type`, where `qwen3_moe` and `qwen3` genuinely were two keys.
///
/// They are listed rather than deleted because deleting them is a behaviour
/// change to `serves()` for an operator who spelled one in a boot file, and
/// that is a separate commit from this one. What this test buys is that the
/// list cannot GROW a fourth by accident, and that if one of the three starts
/// being advertised the line comes out.
#[test]
fn the_texts_allow_list_is_advertised_by_the_catalog_or_stated_dead() {
    /// Entries of `LLAMA_LIKE` that no catalog row advertises, and why.
    const NO_ROW_ADVERTISES: &[&str] = &[
        // The llama generation advertises `llama` for 3.1, 3.2 and 3.3
        // alike; a point release is not an architecture.
        "llama3",
        "llama4",
        // A mixture is a fact of the row — `moe_intermediate` and
        // `n_experts` — not a second architecture beside the dense one.
        "qwen3_moe",
    ];

    let advertised: BTreeSet<String> =
        deployable().iter().map(|(_, d)| canonical(d.advertised.arch)).collect();
    let dead: BTreeSet<String> = NO_ROW_ADVERTISES.iter().map(|a| canonical(a)).collect();

    let mut orphans = Vec::new();
    let mut revived = Vec::new();
    for entry in driver_metal::model::text::known() {
        let c = canonical(entry);
        match (advertised.contains(&c), dead.contains(&c)) {
            (true, true) => revived.push(entry),
            (false, false) => orphans.push(entry),
            _ => {}
        }
    }

    assert!(
        orphans.is_empty(),
        "`model/text.rs`'s LLAMA_LIKE names {orphans:?}, which no catalog row \
         advertises. An entry no row can reach is a name that only ever \
         matches an operator's typo — and it reads as coverage. Either a row \
         should advertise it, or it belongs in NO_ROW_ADVERTISES here with \
         the argument for why."
    );
    assert!(
        revived.is_empty(),
        "{revived:?} are stated dead here and ARE advertised by a row now; \
         delete their lines from NO_ROW_ADVERTISES"
    );
}

/// The same reduction `model/text.rs` applies, restated for the test.
///
/// Deliberately a COPY and not a call: `canonical` is private to `text.rs`
/// and making it public so a test can borrow it would put a spelling rule on
/// the crate's API surface, which `layering.rs` is the gate against. Six
/// lines duplicated is cheaper than a public function nobody outside a test
/// calls — and if the two ever disagree, the `orphans` assertion above fires,
/// which is the copy checking itself.
fn canonical(arch: &str) -> String {
    arch.chars().filter(|c| *c != '_' && *c != '-').flat_map(char::to_lowercase).collect()
}
