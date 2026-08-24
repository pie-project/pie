//! The dispatch generator: this plane's `*_CLAIMS` × the floor's `*_POINTS`
//! → `src/points_dispatch.rs`.
//!
//! ONE ARM PER CLAIMED POINT, and every arm is the same three moves the
//! hand shim in `baker-smoke` made by hand: read the operands and scalars
//! off the statement by COLUMN INDEX, wear them as the marks the declaration
//! states, call the trait method. The indices are counted HERE, once, off
//! the slot list in declaration order — which is the order the DSL's
//! builders record in — so nothing re-derives them at the fire.
//!
//! WHAT THIS READS AND NOTHING ELSE:
//!
//! * `kernels::points::*_POINTS` — the slots, their marks, their dtypes,
//!   and how many scalar axes the method quantifies over;
//! * this crate's `*_CLAIMS` — which of those points the plane answers. A
//!   point with no claim keeps the family's default body (a backlog row),
//!   and an arm for it would turn a measured gap into a call that refuses
//!   one layer deeper.
//!
//! WHAT IT CANNOT READ. A tier-2 point is an inherent method on `Ctx` — no
//! trait, so `#[claims]` never sees it and no table names it. Those stay
//! hand-written wherever they are needed, and `Call::Tier2` is how a lane
//! says so.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use kernels::points::{
    ATTENTION_POINTS, DIST_POINTS, Dtype, GATE_POINTS, GEMM_POINTS, HC_POINTS, INDEX_POINTS,
    LAYOUT_POINTS, MLA_POINTS, MLP_POINTS, MOE_POINTS, Mark, NORM_POINTS, POOL_POINTS, Point,
    Prim, ROPE_POINTS, SSM_POINTS,
};

/// One family: its trait, its points, and the plane's claims against them.
///
/// THE ONE HAND-WRITTEN LIST IN THIS GENERATOR, and the seam is worth
/// naming: `#[claims]` states its table BESIDE the impl, so a family's
/// claims live in whichever module claimed them, and nothing in the tree
/// enumerates either the point tables or the claim tables. A family added
/// to `kernels/src/points.rs` and not added here would be generated as
/// nothing at all. `every_claim_has_an_arm` in
/// `tests/points_dispatch_is_current.rs` checks the half that IS
/// checkable — that no claim named here was dropped on the way out — and a
/// registry (`linkme`, the way `#[routine]` collects) would close the rest.
pub struct Family {
    /// The trait, as `kernels::points` spells it: what the generated file
    /// has to import for `ctx.<method>()` to resolve.
    pub trait_name: &'static str,
    pub points: &'static [Point],
    pub claims: &'static [&'static str],
}

pub fn families() -> Vec<Family> {
    vec![
        Family { trait_name: "Norm", points: NORM_POINTS, claims: kernels_cuda::norm::NORM_CLAIMS },
        Family { trait_name: "Mlp", points: MLP_POINTS, claims: kernels_cuda::mlp::MLP_CLAIMS },
        Family { trait_name: "Gemm", points: GEMM_POINTS, claims: kernels_cuda::gemm::GEMM_CLAIMS },
        Family { trait_name: "Dist", points: DIST_POINTS, claims: kernels_cuda::dist::DIST_CLAIMS },
        Family { trait_name: "Rope", points: ROPE_POINTS, claims: kernels_cuda::rope::ROPE_CLAIMS },
        Family { trait_name: "Moe", points: MOE_POINTS, claims: kernels_cuda::moe::MOE_CLAIMS },
        Family { trait_name: "Gate", points: GATE_POINTS, claims: kernels_cuda::mlp::GATE_CLAIMS },
        Family {
            trait_name: "Layout",
            points: LAYOUT_POINTS,
            claims: kernels_cuda::layout::LAYOUT_CLAIMS,
        },
        Family { trait_name: "Ssm", points: SSM_POINTS, claims: kernels_cuda::ssm::SSM_CLAIMS },
        Family {
            trait_name: "Attention",
            points: ATTENTION_POINTS,
            claims: kernels_cuda::attn::ATTENTION_CLAIMS,
        },
        Family { trait_name: "Mla", points: MLA_POINTS, claims: kernels_cuda::attn::MLA_CLAIMS },
        Family {
            trait_name: "Index",
            points: INDEX_POINTS,
            claims: kernels_cuda::attn::INDEX_CLAIMS,
        },
        Family { trait_name: "Pool", points: POOL_POINTS, claims: kernels_cuda::attn::POOL_CLAIMS },
        Family { trait_name: "Hc", points: HC_POINTS, claims: kernels_cuda::norm::HC_CLAIMS },
    ]
}

/// The scalar slots that do NOT ride the params run.
///
/// `gemm.attention_landing`'s `layer` is the statement's own TAG — the DSL
/// records it with `.layer(l)` and the driver has always read it off
/// `Op::layer`. The declaration has no way to say that, so the fact stands
/// in two places: here, and as `Except::LayerTag` in
/// `model-dsl/tests/builders_are_the_points.rs`. A `#[tag]` on the
/// declaration would close the seam and retire both rows.
const TAGGED: &[(&str, &str)] = &[("gemm.attention_landing", "layer")];

fn tagged(point: &str, slot: &str) -> bool {
    TAGGED.iter().any(|(p, s)| *p == point && *s == slot)
}

/// The elements a generic axis is instantiated at.
///
/// TWO, and mechanically both for every point: `Elem^axes` with no
/// per-point instantiation list, which is `.wiki/baker.md`'s rule ("no
/// instantiation annotations, ever"). They are the two an arena mints for a
/// generic axis today — `model_compiler::program::Dt` has three more and all
/// three are `Fixed` in every declaration that uses them, so no axis rides
/// one. A plane that grows an `f16` path adds it here and the file
/// regenerates.
const AXES: &[(&str, &str)] = &[("Axis::Bf16", "bf16"), ("Axis::F32", "f32")];

/// The reprs a BANK axis is instantiated at, and how many weight columns one
/// bank of each occupies.
///
/// The `AXES` rule one line up, for the other kind of axis: `Repr^reprs` with
/// no per-point instantiation list. The set is closed and today it has one
/// member — `kernels::points::Repr`'s own doc argues why that is the point of
/// the axis and not an argument against it — so a bank slot's match has one
/// arm plus the refusal, and a second repr is a row here and a regenerated
/// file.
///
/// THE THIRD COLUMN IS `Repr::PLANES` and it is why this is a table rather
/// than a reuse of `AXES`: an element slot reads ONE column, a bank slot
/// reads as many as its repr stores planes, so the `Const` counter advances
/// by a number that depends on which arm is being written.
const REPRS: &[(&str, &str, usize)] = &[("Form::Mxfp4", "Mxfp4", 2)];

/// The element a slot's payload rides, as the generated turbofish spells it.
/// `at` is the axis instantiation for this arm.
fn element(point: &Point, dtype: Dtype, at: &[&str]) -> String {
    match dtype {
        Dtype::Generic(a) => at[a].to_string(),
        Dtype::Fixed(Prim::F32) => "f32".into(),
        Dtype::Fixed(Prim::I32) => "i32".into(),
        Dtype::Fixed(Prim::U32) => "u32".into(),
        Dtype::Fixed(Prim::U8) => "u8".into(),
        Dtype::Fixed(Prim::Bool) => {
            panic!("`{}`: a tensor slot cannot ride `bool`", point.name)
        }
        Dtype::Opaque => panic!("`{}`: a pool row has no element to name", point.name),
        Dtype::Bank(_) => panic!("`{}`: a bank has no element to name", point.name),
    }
}

/// How many `Const` columns one slot occupies: one for a weight, and one per
/// stored plane for a bank. `at` is this arm's repr instantiation.
fn columns(dtype: Dtype, at: &[&str]) -> usize {
    match dtype {
        Dtype::Bank(r) => REPRS
            .iter()
            .find(|(_, name, _)| *name == at[r])
            .map_or(1, |(_, _, planes)| *planes),
        _ => 1,
    }
}

/// Where this arm reads repr axis `r` from: the `Const` COLUMN the bank slot
/// riding it begins at.
///
/// A plain column index and not a [`Site`], because a repr is not an
/// element and does not come off a rectangle at all — see
/// `BoundOp::form`. The count below is the column count and not the slot
/// count, which for every point that exists today is the same number: a
/// second bank slot on one point would have to know the first's plane count
/// to place itself, and nothing declares one.
fn bank_column(point: &Point, r: usize) -> usize {
    let mut consts = 0usize;
    for s in point.slots {
        if s.mark != Mark::Const {
            continue;
        }
        if s.dtype == Dtype::Bank(r) {
            return consts;
        }
        consts += 1;
    }
    panic!(
        "`{}`: repr axis {r} rides no bank slot, so nothing can answer its form",
        point.name
    )
}

/// The pool a `Cache` slot names, which `Dtype::Opaque` does not say.
///
/// The floor declares two associated types and the table records neither:
/// a pool row's element was decided when the slab was allocated, so
/// `Mark::Cache` carries `Opaque` for both. The recurrent slabs are `Ssm`'s
/// and the paged KV is every other family's — the same stand-in
/// `model-dsl/tests/builders_are_the_points.rs` makes when it picks between
/// `&State` and `&Pages`, and a `Mark::Cache(Pool)` on the floor would
/// retire both.
fn pool(point: &Point) -> &'static str {
    let family = point.name.split('.').next().unwrap_or_default();
    if family == "ssm" { "op.recurrent()?" } else { "op.pages()?" }
}

/// Where this arm reads axis `a`'s element from.
///
/// The first slot riding the axis, preferring the OUT column — an `Out` or
/// an `InOut`'s result is the rectangle this fire's arena minted, and its
/// element is what the walk settled — then an `In`, then a `Const`. Ranked
/// rather than "first slot", because a `Const` witness would answer with the
/// CHECKPOINT's repr and a bank may be quantised where the activation is
/// not.
fn witness(point: &Point, a: usize) -> String {
    let mut best: Option<(u8, String, bool)> = None;
    let (mut ins, mut outs, mut consts) = (0usize, 0usize, 0usize);
    let mut banked = false;
    for s in point.slots {
        let (rank, site, shifted) = match s.mark {
            Mark::Out => {
                let at = (0, format!("Site::Out({outs})"), false);
                outs += 1;
                at
            }
            Mark::InOut => {
                let at = (0, format!("Site::Out({outs})"), false);
                ins += 1;
                outs += 1;
                at
            }
            Mark::In => {
                let at = (1, format!("Site::In({ins})"), false);
                ins += 1;
                at
            }
            Mark::Const => {
                // A bank occupies `Repr::PLANES` columns and is never an
                // element witness itself; what it does do is move every
                // `Const` column after it by a number this ranking does not
                // know, which the flag below carries.
                if matches!(s.dtype, Dtype::Bank(_)) {
                    banked = true;
                    continue;
                }
                let at = (2, format!("Site::Const({consts})"), banked);
                consts += 1;
                at
            }
            Mark::Cache | Mark::Scalar => continue,
        };
        if s.dtype != Dtype::Generic(a) {
            continue;
        }
        if best.as_ref().is_none_or(|(r, _, _)| rank < *r) {
            best = Some((rank, site, shifted));
        }
    }
    let (_, site, shifted) = best.unwrap_or_else(|| {
        panic!("`{}`: axis {a} rides no slot, so nothing can answer its element", point.name)
    });
    // A BANK'S COLUMN COUNT IS PER-ARM and a witness is picked once per
    // point, so a `Const` site standing after a bank has no fixed index to
    // report. Nothing declares one — `moe.matmul_select_bias`'s axis is
    // witnessed by its `Out`, which is the ranking's own first preference —
    // and the day something does, the ranking needs the repr in hand and has
    // to move into `arm`.
    assert!(
        !shifted,
        "`{}`: axis {a}'s witness is a `Const` standing after a bank, whose column \
         count is not fixed at the point",
        point.name
    );
    site
}

/// The method names TWO CLAIMED FAMILIES BOTH DECLARE.
///
/// `Attention`, `Mla`, `Index` and `Pool` each declare a `kv_append`: four
/// pools, four appends, one name. Every trait a family claims from is
/// imported into the generated file, so `ctx.kv_append(..)` is ambiguous the
/// moment a second one of them claims — which is what `mla.kv_append`
/// landing did. The names are collected off the POINT tables rather than the
/// claim tables, so an arm's spelling does not change when a sibling family
/// claims or unclaims: what makes a name ambiguous is that two families
/// DECLARE it, and both traits are in scope either way.
fn ambiguous(families: &[Family]) -> BTreeSet<&'static str> {
    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    let mut twice: BTreeSet<&'static str> = BTreeSet::new();
    for f in families {
        // One family cannot collide with itself: a trait's methods are
        // distinct by definition.
        let mut mine: BTreeSet<&'static str> = BTreeSet::new();
        for p in f.points {
            mine.insert(p.name.split('.').next_back().expect("`family.method`"));
        }
        for m in mine {
            if !seen.insert(m) {
                twice.insert(m);
            }
        }
    }
    twice
}

/// One call: the trait method with every slot read off the bound statement.
///
/// `at` is this arm's ELEMENT instantiation and `reprs_at` its REPR one; the
/// turbofish is the two runs concatenated, which is the order `#[points]`
/// enforces on the declaration (`<T: Scalar, R: Repr>`).
fn call(
    point: &Point,
    at: &[&str],
    reprs_at: &[&str],
    trait_name: &str,
    shared: &BTreeSet<&'static str>,
) -> String {
    let method = point.name.split('.').next_back().expect("`family.method`");
    let (mut ins, mut outs, mut consts, mut params) = (0usize, 0usize, 0usize, 0usize);
    let mut args: Vec<String> = Vec::new();
    for s in point.slots {
        let arg = match s.mark {
            Mark::In => {
                let a = format!("op.tin::<{}>({ins})?", element(point, s.dtype, at));
                ins += 1;
                a
            }
            Mark::Out => {
                let a = format!("op.tout::<{}>({outs})?", element(point, s.dtype, at));
                outs += 1;
                a
            }
            Mark::InOut => {
                let a = format!("op.tinout::<{}>({ins}, {outs})?", element(point, s.dtype, at));
                ins += 1;
                outs += 1;
                a
            }
            Mark::Const => {
                let a = match s.dtype {
                    // A bank reads its FIRST column here and `Repr::PLANES`
                    // of them in total; the counter follows the repr.
                    Dtype::Bank(r) => format!("op.bank::<{}>({consts})?", reprs_at[r]),
                    _ => format!("op.tconst::<{}>({consts})?", element(point, s.dtype, at)),
                };
                consts += columns(s.dtype, reprs_at);
                a
            }
            Mark::Cache => pool(point).to_string(),
            Mark::Scalar => {
                if tagged(point.name, s.name) {
                    // The tag is not in the params run, so the run's index
                    // does not move.
                    "op.layer()?".to_string()
                } else {
                    let a = match s.dtype {
                        Dtype::Fixed(Prim::F32) => format!("op.f32({params})?"),
                        Dtype::Fixed(Prim::U32) => format!("op.u32({params})?"),
                        Dtype::Fixed(Prim::Bool) => format!("op.bool({params})?"),
                        Dtype::Fixed(Prim::I32) => format!("op.u32({params})? as i32"),
                        other => panic!("`{}`: scalar `{}` rides {other:?}", point.name, s.name),
                    };
                    params += 1;
                    a
                }
            }
        };
        args.push(arg);
    }
    let quantified: Vec<&str> = at.iter().chain(reprs_at.iter()).copied().collect();
    let turbofish = if quantified.is_empty() {
        String::new()
    } else {
        format!("::<{}>", quantified.join(", "))
    };
    // THE RECEIVER SPELLING IS THE DISAMBIGUATION, and nothing else
    // changes: `Mla::kv_append::<bf16>(ctx, ..)` is `ctx.kv_append::<bf16>(..)`
    // with the family said out loud. See `ambiguous`.
    if shared.contains(method) {
        let mut all = vec!["ctx".to_string()];
        all.extend(args);
        return format!("{trait_name}::{method}{turbofish}({})", all.join(", "));
    }
    format!("ctx.{method}{turbofish}({})", args.join(", "))
}

/// The arm for one point: the `Elem^axes × Repr^reprs` cartesian, or a bare
/// call when the method quantifies over nothing.
///
/// ONE MATCH OVER BOTH RUNS, and the scrutinee is the two witnesses in
/// declaration order — the elements read off rectangles, the reprs read off
/// the parameter table. They are separate questions with separate answers,
/// but they pick ONE instantiation between them, so a nested match would be
/// two places to write the same refusal.
fn arm(point: &Point, trait_name: &str, shared: &BTreeSet<&'static str>) -> String {
    let mut out = String::new();
    let quantified = point.axes + point.reprs;
    if quantified == 0 {
        let _ = writeln!(
            out,
            "        {:?} => {},",
            point.name,
            call(point, &[], &[], trait_name, shared)
        );
        return out;
    }
    let mut asks: Vec<String> = (0..point.axes)
        .map(|a| format!("op.dtype({})?", witness(point, a)))
        .collect();
    asks.extend((0..point.reprs).map(|r| format!("op.form({})?", bank_column(point, r))));
    let scrutinee = if quantified == 1 {
        asks[0].clone()
    } else {
        format!("({})", asks.join(", "))
    };
    let _ = writeln!(out, "        {:?} => match {scrutinee} {{", point.name);
    for at in cartesian(point.axes, point.reprs) {
        let pattern: Vec<&str> = at.iter().map(|(p, _)| *p).collect();
        let named: Vec<&str> = at.iter().map(|(_, e)| *e).collect();
        let (elements, reprs_at) = named.split_at(point.axes);
        let pattern = if quantified == 1 {
            pattern[0].to_string()
        } else {
            format!("({})", pattern.join(", "))
        };
        let _ = writeln!(
            out,
            "            {pattern} => {},",
            call(point, elements, reprs_at, trait_name, shared)
        );
    }
    let _ = writeln!(
        out,
        "            _ => Err(Refusal::Absent {{ what: {:?} }}),",
        format!(
            "`{}`, at an element or repr this plane does not instantiate",
            point.name
        )
    );
    let _ = writeln!(out, "        }},");
    out
}

/// `AXES^a × REPRS^r`, in odometer order — the elements first, then the
/// reprs, which is the order `#[points]` makes a declaration state them in.
fn cartesian(a: usize, r: usize) -> Vec<Vec<(&'static str, &'static str)>> {
    let mut out = vec![Vec::new()];
    for _ in 0..a {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                AXES.iter().map(move |a| {
                    let mut next = prefix.clone();
                    next.push(*a);
                    next
                })
            })
            .collect();
    }
    for _ in 0..r {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                REPRS.iter().map(move |(pattern, name, _)| {
                    let mut next = prefix.clone();
                    next.push((*pattern, *name));
                    next
                })
            })
            .collect();
    }
    out
}

/// The file.
pub fn generate() -> String {
    let families = families();
    let shared = ambiguous(&families);
    let mut traits: BTreeSet<&'static str> = BTreeSet::new();
    let mut arms = String::new();
    let mut claimed = 0usize;
    for f in &families {
        let mut wrote = false;
        for p in f.points {
            if !f.claims.contains(&p.name) {
                continue;
            }
            if !wrote {
                let _ = writeln!(arms, "        // ── {} ──", p.name.split('.').next().unwrap_or_default());
                wrote = true;
            }
            arms.push_str(&arm(p, f.trait_name, &shared));
            claimed += 1;
            traits.insert(f.trait_name);
        }
    }
    let traits = traits.into_iter().collect::<Vec<_>>().join(", ");

    let mut out = String::new();
    let _ = write!(
        out,
        r#"//! GENERATED — do not edit. One arm per point this plane CLAIMS, read off
//! `kernels::points`' slot lists and this crate's `*_CLAIMS` tables.
//!
//! The generator is `tests/points_dispatch_is_current/generator.rs`;
//! `cargo test -p kernels-cuda --test points_dispatch_is_current` refuses a
//! stale file and `UPDATE_POINTS_DISPATCH=1` rewrites it.
//!
//! Every arm is the same three moves: read the operands and the scalars off
//! the bound statement BY COLUMN INDEX — the index counted off the point's
//! slot list in declaration order, which is the order `model_dsl::kernels`
//! records in — wear them as the marks the declaration states, and call the
//! trait method. A generic axis becomes a match over the element its witness
//! slot carries, and an element with no arm is a refusal naming the point.
//! There is no default, no cast, and no per-point special case: what a claim
//! needs that a statement does not carry is STAGING, and staging is not this
//! file's job.
//!
//! {claimed} claimed point(s). An unclaimed point keeps its family's default
//! body — a measured backlog row — and gets no arm here, so a lane that
//! states one refuses with the point named rather than one call deeper.
#![cfg_attr(rustfmt, rustfmt::skip)]
// The families whose points are claimed change as the plane grows; the
// prelude does not.
#![allow(unused_imports)]

use kernels::bound::{{Axis, BoundOp, Site}};
// `Form` and the reprs beside the family traits: a bank axis's arm matches
// on the one and turbofishes the other, exactly as an element axis matches
// on `Axis` and turbofishes `bf16`.
use kernels::points::{{Form, Mxfp4}};
use kernels::points::{{{traits}}};
use kernels::routine::Refusal;

use crate::jit::Ctx;
use crate::jit::abi::bf16;

/// Fire one bound statement through this plane's claims.
pub fn dispatch<'p, B>(ctx: &Ctx<'p>, op: &B) -> Result<(), Refusal>
where
    B: BoundOp<Plane = Ctx<'p>>,
{{
    match op.point() {{
{arms}        _ => Err(Refusal::Absent {{
            // The point is the CALLER's — it handed this the statement and
            // can name it in the report. `Refusal` carries `&'static str`
            // and a generated arm has no name to leak.
            what: "a point this plane does not claim; see the family's `*_CLAIMS`",
        }}),
    }}
}}
"#
    );
    out
}
