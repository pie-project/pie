//! `serve::ROWS` is the catalog's serving face, and this is what makes that
//! a measurement rather than a claim.
//!
//! # The pin this replaces
//!
//! `model-legacy/tests/serve_rows.rs` held every column of `serve::ROWS`
//! equal to the LEGACY catalog's answer for the same checkpoint. That test
//! linked both crates and existed only for as long as there were two
//! catalogs; R3 deleted the second one, and the pin dies with it.
//!
//! What replaces it is the same discipline against this crate's own single
//! source: a row's numbers must equal what the row's own trace says. The
//! trace is what the driver fires, so a row that drifted would be a sampler
//! sized for one model in front of a program computing another.
//!
//! # What is NOT checked here, and why that is honest
//!
//! `arch` and `max_model_len` are deployment facts. No plan states either —
//! a trace says what a layer computes, not what a fleet calls the model or
//! how long a context an operator will admit — so they are STATED on the row
//! with the value `model-legacy`'s spec carried, and there is nothing to
//! hold them against. They are listed by name in
//! [`every_stated_fact_is_named`] so the set cannot grow silently.

use model::deployment::{Deployment, Refusal};
use model::serve;
use model_ir::kernels::Backend;

/// Every row's `layers` and `vocab` are what its own trace says.
#[test]
fn every_rows_numbers_are_its_own_traces() {
    let mut checked = 0;
    for row in serve::ROWS {
        let trace = model::trace_of(row.id)
            .unwrap_or_else(|| panic!("`{}` is a serving row of no catalog SKU", row.id));
        let plan = trace(Backend::Cuda);
        let dep = match Deployment::of(&plan, Default::default()) {
            Ok(dep) => dep,
            // A row whose pools this build cannot lay out still has a tower
            // and a logits width, and both are read before the refusal — so
            // the refusal is not an excuse to skip the check. Read them off
            // the plan directly.
            Err(_) => {
                assert_eq!(
                    plan.params
                        .iter()
                        .find(|p| p.name == "embed")
                        .expect("every plan states `embed`")
                        .shape[0],
                    u64::from(row.vocab),
                    "`{}`: the row's vocab is not its `embed` table's depth",
                    row.id,
                );
                checked += 1;
                continue;
            }
        };
        assert_eq!(
            dep.layers, row.layers,
            "`{}`: the row says {} layers and its trace says {}",
            row.id, row.layers, dep.layers,
        );
        assert_eq!(
            dep.shape.vocab, row.vocab,
            "`{}`: the row says {} logits and its trace says {}",
            row.id, row.vocab, dep.shape.vocab,
        );
        checked += 1;
    }
    assert_eq!(checked, serve::ROWS.len(), "a row went unchecked");
}

/// Every catalog SKU has a serving row, and every serving row is a SKU.
///
/// The two id spaces are ONE now. A catalog row with no serving row is a
/// model a driver could load and the engine could not name; a serving row
/// with no catalog row is a template for a model nothing traces.
#[test]
fn the_catalog_and_the_serving_table_are_the_same_ids() {
    let catalog: Vec<&str> = model::catalog().into_iter().map(|(id, _)| id).collect();
    let serving: Vec<&str> = serve::ids();
    let mut missing: Vec<&str> = catalog
        .iter()
        .copied()
        .filter(|id| !serving.contains(id))
        .collect();
    missing.sort_unstable();
    assert!(
        missing.is_empty(),
        "these SKUs trace but cannot be served (no `serve::ROWS` row): {missing:?}",
    );
    let mut extra: Vec<&str> = serving
        .iter()
        .copied()
        .filter(|id| !catalog.contains(id))
        .collect();
    extra.sort_unstable();
    assert!(
        extra.is_empty(),
        "these serving rows name no catalog SKU: {extra:?}",
    );
}

/// The facts a row STATES rather than measures, by name.
///
/// A guard on the shape of the table, not on its values: the moment a
/// column is added here it has to be either derivable from the plan (and
/// then checked above) or a deployment fact (and then listed here, with the
/// reason no trace can answer it).
#[test]
fn every_stated_fact_is_named() {
    const STATED: &[(&str, &str)] = &[
        (
            "arch",
            "the architecture label a control plane files the model under",
        ),
        ("max_model_len", "the context ceiling a deployment admits"),
        (
            "template",
            "how a turn of conversation is written and read back",
        ),
    ];
    assert_eq!(STATED.len(), 3, "a stated fact appeared or vanished");
}

/// The SKUs whose pools this build cannot lay out say so BY NAME.
///
/// Not an inventory of what is broken — every one of these traces, binds and
/// is a real model. It is where the driver's "I have no pool for this"
/// refusal is written down once, so that adding an MLA pool is a change with
/// a test that moves.
#[test]
fn the_pool_refusals_are_the_measured_ones() {
    let mut refused: Vec<(&str, String)> = Vec::new();
    for row in serve::ROWS {
        let plan = model::trace_of(row.id).expect("a serving row is a SKU")(Backend::Cuda);
        if let Err(why) = Deployment::of(&plan, Default::default()) {
            refused.push((row.id, format!("{why}")));
        }
    }
    let names: Vec<&str> = refused.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        names,
        vec![
            // MLA and the compressed planes: a single-plane latent row, and
            // this build provisions the k/v pair a pager allocates.
            //
            // THE THREE GEMMA ROWS LEFT THIS LIST. They were here for "two KV
            // plane widths across the tower and this build lays out one",
            // which was a fact about this function rather than about the
            // pool: `KvCacheLayout` has carried a per-layer head width and kv
            // head count all along, and `Deployment::of` handed it one number
            // repeated. It reads the rows now. The layer-by-layer proof is
            // `the_two_kinds_of_gemmas_tower_are_read_apart` below.
            "dsv4-base-bf16-kv-bf16",
            "dsv4-base-bf16-kv-bf16-tp2",
            "glm5-a12b-bf16-bf16-kv-bf16",
            "glm5-a12b-bf16-bf16-kv-bf16-tp2",
            "kimik3-bf16-mxfp4-kv-bf16",
            "kimik3-bf16-mxfp4-kv-bf16-tp2",
        ],
        "the set of SKUs with no pool moved: {refused:?}",
    );
}

/// gemma-4's two attention kinds are read APART, layer by layer.
///
/// The row this pins is the one the refusal above used to stand on, and it is
/// the reason a pool that laid out one width would be wrong rather than
/// merely wasteful: 35 of e4b's 42 layers read a 256-wide head and 7 read a
/// 512-wide one, so a uniform 256 would have every global layer striding a
/// quarter of the plane it wrote and a uniform 512 would double the pool and
/// still stride the sliding layers wrong.
///
/// The KV SOURCE is the other half. e4b's trailing 18 layers project no k/v
/// and attend through an earlier layer's pages — `kv.22` for the sliding
/// kind, `kv.23` for the global one — and a `kv_source` of `l` would have the
/// pager allocate 18 layers of pages nothing ever writes and every one of
/// those layers attend over zeros.
#[test]
fn the_two_kinds_of_gemmas_tower_are_read_apart() {
    let plan = model::trace_of("gemma4-e4b-bf16-kv-bf16").expect("a SKU")(Backend::Cuda);
    let dep = Deployment::of(&plan, Default::default()).expect("gemma-4's pool is layable now");
    assert_eq!(dep.layers, 42);
    assert_eq!(dep.attention.len(), 42);

    // The checkpoint's own `layer_types`: `full_attention` every sixth layer,
    // `sliding_attention` on the rest.
    for (l, at) in dep.attention.iter().enumerate() {
        let full = l % 6 == 5;
        assert_eq!(
            at.head_dim,
            if full { 512 } else { 256 },
            "layer {l} attends at the wrong head width: {at:?}",
        );
        assert_eq!(at.kv_heads, 2, "layer {l}: {at:?}");
    }
    assert_eq!(
        dep.attention.iter().filter(|a| a.head_dim == 512).count(),
        7,
        "gemma-4-e4b states 7 full-attention layers",
    );

    // The 24 owning layers read their own pages; the 18 shared ones read the
    // last owner OF THEIR OWN KIND.
    for (l, at) in dep.attention.iter().enumerate() {
        let want = if l < 24 {
            l as u32
        } else if l % 6 == 5 {
            23
        } else {
            22
        };
        assert_eq!(
            at.kv_source, want,
            "layer {l} reads the wrong pages: {at:?}"
        );
    }

    // The scalars are the WIDEST, and the GQA check reads every layer.
    assert_eq!(dep.shape.head_dim, 512);
    assert_eq!(dep.shape.kv_heads, 2);
    assert_eq!(dep.shape.q_heads, 8);
    dep.servable_by(&[4])
        .expect("8 q over 2 kv is group 4 on every layer");
    assert!(
        dep.servable_by(&[8]).is_err(),
        "a build with no group-4 decode cannot serve this stack",
    );
}

/// A refusal names what it found, in words, and is never a bare `None`.
#[test]
fn a_pool_refusal_carries_its_reason() {
    let plan = model::trace_of("kimik3-bf16-mxfp4-kv-bf16").expect("a SKU")(Backend::Cuda);
    let Err(why) = Deployment::of(&plan, Default::default()) else {
        panic!("kimi attends through a latent row and this build has no store for one");
    };
    assert!(matches!(why, Refusal::Unsupported(_)), "{why}");
    assert!(format!("{why}").contains("latent"), "{why}");
}
