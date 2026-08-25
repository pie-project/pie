//! The dispatch generator: a plane's `*_CLAIMS` × the floor's `*_POINTS`
//! → that plane's `src/points_dispatch.rs`.
//!
//! ONE GENERATOR, EVERY PLANE, and the file lives here because cuda's was
//! the first plane to need it. What a plane brings is [`Plane`] — the `use`
//! lines its arms resolve through, and the elements its generic axes are
//! instantiated at — and [`Surface`] rows naming its own claim tables. What
//! the generator brings is everything else: the column counting, the witness
//! ranking, the cartesian, the census. `kernels-metal/tests/` reaches this
//! file by `#[path]` for exactly that reason (the idiom
//! `grammar/tests/tokenizer_bytes.rs` already uses across crates), and the
//! shader planes that follow reach it the same way.
//!
//! THE PLANE-SHAPED HALF IS THREE FIELDS AND A LIST, which is the measured
//! answer to "what is cuda about this file": the receiver is spelled `Ctx`
//! on every plane, the marks are the floor's, the accessors are
//! `BoundOp`'s, and the pools are `op.recurrent()` / `op.pages()`. What
//! genuinely differed was the two `use` lines and the SPELLING of the
//! half-precision element (`bf16` on cuda, `bfloat` on metal) — the second
//! is why [`Plane::axes`] is a field and not a constant.
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
//! AND THE PLANE'S TIER-2 SURFACE, off the same two reads. A tier-2 point is
//! an inherent method on `Ctx` — no trait, so no floor table declares it —
//! but `#[claims]` reads that block too (`.wiki/baker.md` says it does) and
//! writes its slots into `TIER2_POINTS`. That is a point table like any
//! other, so it generates like any other: same slot reader, same column
//! counting, same `Elem^axes` match, ONE ARM IN THE SAME MATCH. What it does
//! NOT have is a claim table, because a tier-2 point is declared and claimed
//! by the same line — see [`Surface`].

use std::collections::BTreeSet;
use std::fmt::Write as _;

// The floor's vocabulary and nothing of any plane's: the point tables
// themselves are named by each plane's own `surfaces()`, which is what makes
// one generator serve two crates.
use kernels::points::{Dtype, Mark, Point, Prim};

/// One block of arms, and what it is generated from.
///
/// TWO KINDS, AND THE ASYMMETRY IS THE TIER ITSELF. A tier-1 family's points
/// are DECLARED on the floor — every plane's obligation — and CLAIMED here,
/// so it carries two tables and the gap between them IS the backlog. A tier-2
/// surface is declared and claimed by the same inherent method, so it carries
/// one table and can have no backlog row; and it has no trait, so nothing
/// ever has to name a receiver through one — an inherent method wins method
/// resolution outright.
///
/// THE ONE HAND-WRITTEN LIST is each plane's own `surfaces()`, and the seam
/// is worth naming: `#[claims]` states its table BESIDE the impl, so a
/// family's claims live in whichever module claimed them, and nothing in the
/// tree enumerates either the point tables or the claim tables. A family
/// added to `kernels/src/points.rs` and not added to a plane's list would be
/// generated as nothing at all. [`every_claim_has_an_arm`] checks the half
/// that IS checkable — that no claim named there was dropped on the way out
/// — [`every_family_the_floor_declares_has_a_surface`] closes the DECLARED
/// half against `kernels::points::declared()`, and a registry (`linkme`, the
/// way `#[routine]` collected) would close the rest.
///
/// THE LIST IS THE CALLER'S because it names the PLANE's tables
/// (`kernels_cuda::norm::NORM_CLAIMS`, `kernels_metal::norm::NORM_CLAIMS`),
/// and a generator that named one of them could only ever write for one
/// plane.
pub enum Surface {
    /// A tier-1 family: the floor declares `points`, this plane answers
    /// `claims`, and the rest are measured backlog rows.
    Family {
        /// The trait, as `kernels::points` spells it: what the generated file
        /// has to import for `ctx.<method>()` to resolve.
        trait_name: &'static str,
        points: &'static [Point],
        claims: &'static [&'static str],
    },
    /// The plane's tier-2 surface: inherent methods on its own `Ctx`, whose
    /// declaration and claim are one line.
    ///
    /// A PLANE MAY STATE NONE, and `kernels-metal` states none — every launch
    /// it holds is a tier-1 claim or a private fn. So this variant is
    /// constructed by one of the two test targets that read this file and not
    /// the other, which is a fact about the planes and not a dead line.
    #[allow(dead_code)]
    Tier2 { points: &'static [Point] },
}

impl Surface {
    /// The points this surface puts an arm in the dispatch for.
    ///
    /// An unclaimed tier-1 point keeps its family's default body and gets no
    /// arm; every tier-2 point gets one, because there is no such thing as a
    /// tier-2 point this plane declared and did not claim.
    pub fn arms(&self) -> Vec<&'static Point> {
        match self {
            Surface::Family { points, claims, .. } => {
                points.iter().filter(|p| claims.contains(&p.name)).collect()
            }
            Surface::Tier2 { points } => points.iter().collect(),
        }
    }

    /// Every point this surface declares, answered or not. `arms` is the
    /// answered half; the difference is the measured backlog.
    pub const fn declares(&self) -> &'static [Point] {
        match self {
            Surface::Family { points, .. } | Surface::Tier2 { points } => points,
        }
    }

    /// The trait this surface's arms are called through, where the method
    /// name alone is ambiguous. `None` on the tier-2 surface: there is no
    /// trait, and an inherent method needs no disambiguation.
    const fn trait_name(&self) -> Option<&'static str> {
        match self {
            Surface::Family { trait_name, .. } => Some(trait_name),
            Surface::Tier2 { .. } => None,
        }
    }
}

/// What one plane brings to the generator: how the file it writes reaches
/// that plane's receiver and elements.
///
/// EVERY FIELD IS SOMETHING A PLANE ANSWERS DIFFERENTLY, and the census that
/// produced the list is short because most of the file is not plane-shaped:
/// the marks are the floor's, the accessors are `BoundOp`'s, the pools are
/// `op.recurrent()` / `op.pages()`, and the receiver is spelled `Ctx` on
/// every plane (a struct on cuda, `dyn Encode + 'a` on the shader planes —
/// which is why `BoundOp::Plane` is `?Sized`).
pub struct Plane {
    /// The package, as cargo spells it: what the generated header tells a
    /// reader to run when the file is stale.
    pub krate: &'static str,

    /// Where this generator sits relative to that package, for the same
    /// sentence. Two planes read one file, so neither can assume it is
    /// beside its own tests.
    pub generator: &'static str,

    /// The committed file this plane's dispatch is written to.
    pub dispatch: std::path::PathBuf,

    /// The `use` lines the arms resolve through — the receiver and the
    /// plane's own half-precision element, the two names the floor cannot
    /// give. `crate::`-relative: the file is written INTO the plane crate.
    pub prelude: &'static str,

    /// The elements a generic axis is instantiated at, as (the `Axis`
    /// pattern, the turbofish).
    ///
    /// TWO, and mechanically both for every point: `Elem^axes` with no
    /// per-point instantiation list, which is `.wiki/baker.md`'s rule ("no
    /// instantiation annotations, ever"). They are the two an arena mints for
    /// a generic axis today — `model_compiler::program::Dt` has three more
    /// and all three are `Fixed` in every declaration that uses them, so no
    /// axis rides one.
    ///
    /// A FIELD AND NOT A CONSTANT because the SPELLING is the plane's: cuda's
    /// half-precision element is `jit::abi::bf16` and metal's is
    /// `plane::bfloat`, each declared in its own crate for the reason both
    /// files state — the floor cannot name a device type. A plane that grows
    /// an `f16` path adds a row to its own table and its file regenerates.
    pub axes: &'static [(&'static str, &'static str)],

    /// The `CANON` rows this plane answers by symbol rather than by point —
    /// read by [`every_canon_row_is_an_unclaimed_point`], which is the check
    /// that such a row is still a backlog row and not a dead line.
    pub canon: &'static [(&'static str, &'static str)],
}

impl Plane {
    /// The half-precision element's spelling, for the one line of the
    /// generated header that names it.
    fn half(&self) -> &'static str {
        self.axes.first().map_or("bf16", |(_, e)| *e)
    }
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

/// The reprs a BANK axis is instantiated at, and how many weight columns one
/// bank of each occupies.
///
/// [`Plane::axes`]'s rule for the other kind of axis: `Repr^reprs` with no
/// per-point instantiation list. The set is closed and today it has one
/// member — `kernels::points::Repr`'s own doc argues why that is the point of
/// the axis and not an argument against it — so a bank slot's match has one
/// arm plus the refusal, and a second repr is a row here and a regenerated
/// file.
///
/// A CONSTANT WHERE THE ELEMENTS ARE A FIELD, and the asymmetry is the
/// floor's: a repr marker is `kernels::points::Mxfp4`, declared once and
/// spelled the same in every plane's file, where a half-precision element is
/// each plane's own type. Nothing here is cuda's, so nothing here is
/// parameterised. A plane that claims no bank point never reads this table —
/// `cartesian` walks it `reprs` times and every declaration it claims states
/// zero.
///
/// THE THIRD COLUMN IS `Repr::PLANES` and it is why this is a table rather
/// than a reuse of the elements: an element slot reads ONE column, a bank
/// slot reads as many as its repr stores planes, so the `Const` counter
/// advances by a number that depends on which arm is being written.
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
///
/// A TIER-2 POINT HAS NO FAMILY TO READ, its name being the method alone, so
/// the split answers with the whole name and the paged arm is what it lands
/// on. That is right for the one that exists and it is right for the reason
/// the rule above is right — the recurrent pool is `Ssm`'s and nothing else's
/// — but it is an accident of spelling rather than a reading, and it is the
/// second thing the floor mark would close.
fn pool(point: &Point) -> &'static str {
    let family = point.name.split('.').next().unwrap_or_default();
    if family == "ssm" {
        "op.recurrent()?"
    } else {
        "op.pages()?"
    }
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
        panic!(
            "`{}`: axis {a} rides no slot, so nothing can answer its element",
            point.name
        )
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

/// The method names TWO SURFACES BOTH DECLARE.
///
/// `Attention`, `Mla`, `Index` and `Pool` each declare a `kv_append`: four
/// pools, four appends, one name. Every trait a family claims from is
/// imported into the generated file, so `ctx.kv_append(..)` is ambiguous the
/// moment a second one of them claims — which is what `mla.kv_append`
/// landing did. The names are collected off the POINT tables rather than the
/// claim tables, so an arm's spelling does not change when a sibling family
/// claims or unclaims: what makes a name ambiguous is that two surfaces
/// DECLARE it, and both traits are in scope either way.
///
/// THE TIER-2 SURFACE IS COUNTED, and it is the one that would be silent.
/// Two traits colliding is a compile error the moment the file is built; an
/// INHERENT method colliding with a trait method is not — inherent wins
/// method resolution outright, so a tier-2 `decode` would quietly steal
/// `attention.decode`'s arm and the wrong kernel would fire. Counting it here
/// makes the tier-1 arm spell its trait, which is exactly the fix, and leaves
/// the tier-2 arm as `ctx.decode(..)`, which is exactly right.
fn ambiguous(surfaces: &[Surface]) -> BTreeSet<&'static str> {
    let mut seen: BTreeSet<&'static str> = BTreeSet::new();
    let mut twice: BTreeSet<&'static str> = BTreeSet::new();
    for s in surfaces {
        // One surface cannot collide with itself: a trait's methods are
        // distinct by definition, and so are one impl block's.
        let mut mine: BTreeSet<&'static str> = BTreeSet::new();
        let points = match s {
            Surface::Family { points, .. } | Surface::Tier2 { points } => points,
        };
        for p in *points {
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
    trait_name: Option<&str>,
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
                let a = format!(
                    "op.tinout::<{}>({ins}, {outs})?",
                    element(point, s.dtype, at)
                );
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
    //
    // A TIER-2 ARM NEVER TAKES IT, shared name or not: there is no trait to
    // say out loud, and an inherent method is what a shared name resolves to
    // anyway.
    if let Some(family) = trait_name.filter(|_| shared.contains(method)) {
        let mut all = vec!["ctx".to_string()];
        all.extend(args);
        return format!("{family}::{method}{turbofish}({})", all.join(", "));
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
fn arm(
    plane: &Plane,
    point: &Point,
    trait_name: Option<&str>,
    shared: &BTreeSet<&'static str>,
) -> String {
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
    for at in cartesian(plane, point.axes, point.reprs) {
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

/// One row of [`generate`]'s claim census: the point, the slot its FIRST
/// element axis is witnessed at, and the elements that axis is instantiated
/// at.
///
/// THE TABLE IS THE MATCH, which is what closes the doubling
/// `driver-cuda/src/baker/resolve.rs` used to carry by hand. That file needs
/// to answer "does this plane claim this point, at the element this lane's
/// rectangles ride" BEFORE anything fires, and a `match` cannot be
/// enumerated — so it kept its own spelling of the arms and a note about why
/// the drift was fail-safe. Both halves are written here now, from the same
/// `witness` and the same elements the arm above is written from, so there is
/// nothing left to drift.
///
/// ONE WITNESS AND NOT ALL OF THEM. A point's arm matches on `axes + reprs`
/// scrutinees; the census reports the first ELEMENT axis, because that is the
/// one an executor can answer from the lowered plan alone — a repr comes off
/// the parameter table and a second element axis rides a slot the first one
/// already pins by construction (no declaration has two independent element
/// axes today, and `#[points]` would have to grow a second witness rule
/// before one could). `None` for a point that quantifies over nothing, which
/// resolves by name.
fn row(plane: &Plane, point: &Point) -> String {
    let witness = if point.axes == 0 {
        "None".to_string()
    } else {
        format!("Some({})", witness(point, 0))
    };
    let elements: Vec<&str> = plane.axes.iter().map(|(pattern, _)| *pattern).collect();
    format!(
        "    ({:?}, {witness}, &[{}]),\n",
        point.name,
        elements.join(", ")
    )
}

/// `axes^a × REPRS^r`, in odometer order — the elements first, then the
/// reprs, which is the order `#[points]` makes a declaration state them in.
fn cartesian(plane: &Plane, a: usize, r: usize) -> Vec<Vec<(&'static str, &'static str)>> {
    let mut out = vec![Vec::new()];
    for _ in 0..a {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                plane.axes.iter().map(move |a| {
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
pub fn generate(plane: &Plane, surfaces: &[Surface]) -> String {
    let shared = ambiguous(surfaces);
    let mut traits: BTreeSet<&'static str> = BTreeSet::new();
    let mut arms = String::new();
    let mut census = String::new();
    let mut tier2 = String::new();
    let mut claimed = 0usize;
    let mut inherent = 0usize;
    for s in surfaces {
        let mut wrote = false;
        for p in s.arms() {
            if !wrote {
                let _ = writeln!(
                    arms,
                    "        // ── {} ──",
                    // A tier-1 point's heading is its family; a tier-2 point
                    // has none, and what its arms share is the tier.
                    match s {
                        Surface::Family { .. } => p.name.split('.').next().unwrap_or_default(),
                        Surface::Tier2 { .. } => "tier-2, inherent on Ctx",
                    }
                );
                wrote = true;
            }
            arms.push_str(&arm(plane, p, s.trait_name(), &shared));
            // TWO CENSUSES, ONE PER CALL VARIANT. A lane reaches a tier-1
            // point as `Call::Point` and a tier-2 statement as `Call::Tier2`,
            // and an executor's eager pass asks its question against the one
            // its call can actually produce — a `Call::Point` naming an
            // inherent method is not a thing `call_of` can answer.
            match s {
                Surface::Family { .. } => {
                    census.push_str(&row(plane, p));
                    claimed += 1;
                }
                Surface::Tier2 { .. } => {
                    tier2.push_str(&row(plane, p));
                    inherent += 1;
                }
            }
            if let Some(t) = s.trait_name() {
                traits.insert(t);
            }
        }
    }
    let traits = traits.into_iter().collect::<Vec<_>>().join(", ");
    let (krate, generator, prelude, half) =
        (plane.krate, plane.generator, plane.prelude, plane.half());

    let mut out = String::new();
    let _ = write!(
        out,
        r#"//! GENERATED — do not edit. One arm per point this plane ANSWERS: the tier-1
//! points it CLAIMS, read off `kernels::points`' slot lists and this crate's
//! `*_CLAIMS` tables, and its TIER-2 surface, read off `TIER2_POINTS`.
//!
//! The generator is `{generator}`;
//! `cargo test -p {krate} --test points_dispatch_is_current` refuses a
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
//! {claimed} claimed point(s) and {inherent} tier-2 point(s). An unclaimed
//! tier-1 point keeps its family's default body — a measured backlog row —
//! and gets no arm here, so a lane that states one refuses with the point
//! named rather than one call deeper. A tier-2 point cannot be unclaimed: the
//! inherent method that declares it is the claim.
//!
//! ONE MATCH FOR BOTH TIERS, which is what `.wiki/baker.md` draws, and the
//! two name spaces cannot collide inside it: a tier-1 point is
//! `family.method` and carries a dot, an inherent method cannot. A lane
//! reaches the first as `Call::Point` and the second as `Call::Tier2`, having
//! stripped the `cuda::` gate the plan spells it with.
#![cfg_attr(rustfmt, rustfmt::skip)]
// The families whose points are claimed change as the plane grows; the
// prelude does not.
#![allow(unused_imports)]

use kernels::bound::{{BoundOp, Site}};
use kernels::points::ScalarKind;
// `Form` and the reprs beside the family traits: a bank axis's arm matches
// on the one and turbofishes the other, exactly as an element axis matches
// on `ScalarKind` and turbofishes `{half}`.
use kernels::points::{{Form, Mxfp4}};
use kernels::points::{{{traits}}};
use kernels::plane::Refusal;

{prelude}

/// Every point this plane claims, with the slot its element axis is read off
/// and the elements that axis is instantiated at.
///
/// THE MATCH BELOW, ENUMERABLE. An executor that wants to know before it
/// fires whether a lane's every statement will resolve cannot walk a
/// `match`; it walks this. `None` is a point that quantifies over nothing and
/// resolves by name alone.
pub const CLAIMED: &[(&str, Option<Site>, &[ScalarKind])] = &[
{census}];

/// The same, for this plane's TIER-2 surface — the points an inherent method
/// on `Ctx` both declares and claims.
///
/// A SECOND TABLE AND NOT A COLUMN ON THE FIRST, because the two answer
/// different questions asked by different call variants. `CLAIMED` answers
/// "does this plane claim the tier-1 point this lane routed as
/// `Call::Point`"; this answers the same of a `Call::Tier2`. A row in the
/// wrong table would be a row no call can reach.
pub const TIER2: &[(&str, Option<Site>, &[ScalarKind])] = &[
{tier2}];

/// Fire one bound statement through this plane's claims and its tier-2
/// surface.
pub fn dispatch<'p, B>(ctx: &Ctx<'p>, op: &B) -> Result<(), Refusal>
where
    B: BoundOp<Plane = Ctx<'p>>,
{{
    match op.point() {{
{arms}        _ => Err(Refusal::Absent {{
            // The point is the CALLER's — it handed this the statement and
            // can name it in the report. `Refusal` carries `&'static str`
            // and a generated arm has no name to leak.
            what: "a point this plane does not claim; see the family's `*_CLAIMS`, \
                   or `TIER2_POINTS` for an inherent one",
        }}),
    }}
}}
"#
    );
    // `kernels*` CARRIES NO COMMENTS, so neither may anything written into
    // it. The prose that used to head this file, and the four notes inside
    // it, were removed from every plane at `e77f3ce82` on the user's
    // instruction; a generator that put them back on the next `--write`
    // would undo that silently and for ever. What the file means is written
    // HERE, in the crate that writes it, where a comment is allowed to live.
    let mut kept = String::with_capacity(out.len());
    let mut blank = false;
    for line in out.lines() {
        if line.trim_start().starts_with("//") {
            continue;
        }
        if line.trim().is_empty() {
            if blank {
                continue;
            }
            blank = true;
        } else {
            blank = false;
        }
        kept.push_str(line);
        kept.push('\n');
    }
    kept
}

// ── the four questions, asked of every plane ───────────────────────────
//
// THE CHECKS ARE THE GENERATOR'S because they are questions about the
// GENERATOR's output, and a second plane asking them by copying them would
// be the drift they exist to catch. What a plane's test target holds is its
// `Plane` and its `surfaces()`; every `#[test]` below it is one line.

/// `src/points_dispatch.rs` is CURRENT against the tables that write it.
///
/// The `model-loader/tests/golden_plans.rs` idiom, and the one
/// `model-dsl-legacy`'s retired `wrappers_are_current.rs` used for the same
/// kind of file: regenerate into a string, diff it against what is
/// committed, refuse a stale copy. `UPDATE_POINTS_DISPATCH=1` rewrites it —
/// and, like every golden rewrite, the SAME run still tests the code
/// compiled from the old file, so run it once more to prove the new one.
///
/// CHECKED IN RATHER THAN BUILT INTO `OUT_DIR`, against each plane crate's
/// own `build.rs` note, and for one reason the note's argument does not
/// cover: the dispatch's arms are read. A carried-header list is a list;
/// this is the plane's answer to every point it claims, and a reviewer who
/// cannot see the diff cannot see a claim quietly changing which slot it
/// reads. The staleness the note warns about is what this check is.
///
/// # Panics
///
/// When the committed file is not what the generator writes.
pub fn points_dispatch_is_current(plane: &Plane, surfaces: &[Surface]) {
    let want = generate(plane, surfaces);
    let at = &plane.dispatch;
    let have = std::fs::read_to_string(at).unwrap_or_default();
    if want == have {
        return;
    }
    if std::env::var_os("UPDATE_POINTS_DISPATCH").is_some() {
        std::fs::write(at, &want).unwrap_or_else(|e| panic!("rewriting {}: {e}", at.display()));
        return;
    }
    // Point at the first diverging line rather than dumping two files.
    let line = want
        .lines()
        .zip(have.lines())
        .position(|(w, h)| w != h)
        .map_or_else(
            || want.lines().count().min(have.lines().count()) + 1,
            |i| i + 1,
        );
    panic!(
        "{} is STALE against `kernels::points` × this plane's `*_CLAIMS` \
         (first divergence at line {line}). The dispatch is generated, never \
         edited: regenerate with `cargo run -p points-dispatch -- --write` \
         followed by `cargo fmt`, and review the diff.",
        at.display(),
    );
}

/// The committed file answers EVERY point the plane answers, and answers no
/// point it does not.
///
/// The freshness check above proves the file is what the generator writes;
/// this proves the generator did not drop a claim on the way out. It is the
/// half of "the tables are the source of truth" that IS checkable — a surface
/// missing from a plane's `surfaces()` is invisible to both, and
/// [`Surface`]'s own header says so.
///
/// BOTH TIERS, because both get an arm in the one match: a tier-1 point this
/// plane claims, and every point of its tier-2 surface — which is all of
/// them, an inherent method being its own claim.
///
/// # Panics
///
/// When a claim has no arm, or an arm answers no claim.
pub fn every_claim_has_an_arm(plane: &Plane, surfaces: &[Surface]) {
    let file = std::fs::read_to_string(&plane.dispatch).expect("the committed dispatch");
    let mut missing: Vec<&str> = Vec::new();
    let mut claimed: Vec<&str> = Vec::new();
    for surface in surfaces {
        for point in surface.arms() {
            claimed.push(point.name);
            if !file.contains(&format!("        {:?} =>", point.name)) {
                missing.push(point.name);
            }
        }
    }
    assert!(
        missing.is_empty(),
        "claimed and not dispatched: {missing:?}"
    );

    // And nothing else: an arm for an UNCLAIMED point would answer a
    // measured backlog row with a call that refuses one layer deeper.
    let arms: Vec<&str> = file
        .lines()
        .filter_map(|l| l.strip_prefix("        \""))
        .filter_map(|l| l.split_once("\" =>"))
        .map(|(point, _)| point)
        .collect();
    let extra: Vec<&&str> = arms.iter().filter(|a| !claimed.contains(a)).collect();
    assert!(extra.is_empty(), "dispatched and not claimed: {extra:?}");
    assert_eq!(
        arms.len(),
        claimed.len(),
        "one arm per claim, and no arm twice"
    );
}

/// Every `CANON` row names a point this plane DOES NOT claim.
///
/// `model_compiler::sweep::resolve` asks the claim tables first and the
/// plane's `CANON` second, so a row for a point the plane claims is
/// unreachable — a line that reads as a live answer and is never consulted.
/// A row for a point the FLOOR does not declare is worse: nothing will ever
/// ask for it, and a typo in the family half looks exactly like a backlog.
///
/// This replaced `kernels::canon::ROLES`, a closed list of family prefixes
/// the `#[routine]` attribute asserted its `canon` column against at build
/// time. A prefix list could only catch a misspelled FAMILY; the point
/// tables catch a misspelled point, a point that moved, and a row that
/// stopped being a backlog because the plane grew a claim for it.
///
/// # Panics
///
/// When a `CANON` row names an undeclared point, or one the plane claims.
pub fn every_canon_row_is_an_unclaimed_point(plane: &Plane, surfaces: &[Surface]) {
    let mut declared: Vec<&str> = Vec::new();
    let mut claimed: Vec<&str> = Vec::new();
    for surface in surfaces {
        declared.extend(surface.declares().iter().map(|p| p.name));
        claimed.extend(surface.arms().into_iter().map(|p| p.name));
    }
    for (claim, symbol) in plane.canon {
        assert!(
            declared.contains(claim),
            "`CANON` answers `{claim}` with `{symbol}`, and no family declares that point"
        );
        assert!(
            !claimed.contains(claim),
            "`CANON` answers `{claim}`, which this plane CLAIMS -- the claim wins at \
             resolution and this row is never read"
        );
    }
}

/// A plane's `surfaces()` names EVERY family the floor declares.
///
/// THE HALF OF THE SEAM THAT BECAME CHECKABLE AT W5. [`Surface`]'s header
/// names the one hand-written list and says the gap plainly: a family added
/// to `kernels/src/points.rs` and not added there would be generated as
/// nothing at all — no arm, no backlog row, and no check with anything to
/// say, because the other two read the list rather than the floor.
///
/// `kernels::points::declared()` closed it. The floor now enumerates itself,
/// so a family a plane's list forgot is a family the floor holds and that
/// plane's dispatch cannot see — which is exactly what a point declared and
/// silently unanswerable would look like. What stays open is the CLAIM half:
/// nothing enumerates `*_CLAIMS`, so a claim table not named is still
/// invisible, and that is what a registry would close.
///
/// # Panics
///
/// When a declared point reaches no `Surface`.
pub fn every_family_the_floor_declares_has_a_surface(surfaces: &[Surface]) {
    let named: Vec<&str> = surfaces
        .iter()
        .flat_map(|s| s.declares().iter().map(|p| p.name))
        .collect();
    let missing: Vec<&str> = kernels::points::declared()
        .map(|p| p.name)
        .filter(|p| !named.contains(p))
        .collect();
    assert!(
        missing.is_empty(),
        "{} point(s) are declared on the floor and reach no `Surface`, so this plane's \
         dispatch is generated as if they did not exist: {missing:?}",
        missing.len()
    );
}
