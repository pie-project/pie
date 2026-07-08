//! B4 beam geometry golden vectors (host-side, ptir-dsl + CPU reference interp).
//!
//! The locked wire-form geometry contract, re-expressed as SDK trace tests: the
//! §6.2 beam epilogue authored via the neutral [`Builder`], bound, and run on
//! echo's reference interpreter — its emitted geometry (`Pages`/`KvLen`/`WSlot`/
//! `WOff`/`AttnMask` descriptor ports) must equal the vectors locked in
//! `runtime/src/ptir/ptir_beam.rs` (`golden_charlie_fork_freeze_csrs`,
//! `golden_continue_tail`, `golden_page_turn`, `golden_fork_freeze`). These
//! vectors ARE the contract and must live outside `ptir_beam.rs` before B5
//! deletes the host replay.
//!
//! The epilogue is the faithful §6.2 transcription (byte-identical to
//! `p1_overview::s6_2_beam_epilogue_binds`, GOLDEN_BEAM); here we drive it on the
//! interp with seeded geometry + crafted logits (to force a specific `parent`)
//! and read the resulting geometry off the NEXT step's descriptor.

use pie_ptir::interp::{Instance, NoKernels, PassInputs, Value};
use pie_ptir::registry::{ModelProfile, Port};
use pie_ptir::validate::bind;

use ptir_dsl::builder::Builder;
use ptir_dsl::prelude::*;
use ptir_dsl::{Channel, Traced};

const B: u32 = 2; // beams
const V: u32 = 8; // vocab
const P: u32 = 3; // page slots per beam
const PAGE_T: u32 = 4; // tokens per page

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

/// Build the §6.2 beam epilogue via the neutral Builder (channels 0..=15 in
/// declaration order, matching `common::traces::beam_trace`).
fn build_beam() -> Traced {
    ptir_dsl::model::configure(V, PAGE_T, 2);

    let pages = leak(Channel::seeded([B, P], dtype::u32).named("pages"));
    let lens = leak(Channel::seeded([B, P], dtype::u32).named("lens"));
    let klen = leak(Channel::from(vec![0u32; B as usize]).named("klen"));
    let kvm = leak(Channel::seeded([B, P * PAGE_T], dtype::bool).named("kvm"));
    let pos = leak(Channel::from(vec![0u32; B as usize]).named("pos"));
    let np = leak(Channel::from(vec![1u32; B as usize]).named("np"));
    let tslot = leak(Channel::from(vec![0u32; B as usize]).named("tslot"));
    let tfill = leak(Channel::from(vec![0u32; B as usize]).named("tfill"));
    let w_slot = leak(Channel::from(vec![0u32; B as usize]).named("w_slot"));
    let w_off = leak(Channel::from(vec![0u32; B as usize]).named("w_off"));
    let toks = leak(Channel::from(vec![1i32; B as usize]).named("toks"));
    let scores = leak(Channel::from(vec![0.0f32; B as usize]).named("scores"));
    let fresh = leak(Channel::new([B], dtype::u32).named("fresh"));
    let out = leak(Channel::new([B], dtype::i32).named("out"));
    let out_par = leak(Channel::new([B], dtype::u32).named("out_par"));
    let out_scr = leak(Channel::new([B], dtype::f32).named("out_scr"));

    fresh.put(vec![0u32; B as usize]); // mark fresh host-writer

    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>());
    let page_rows = Tensor::constant((0u32..=B).map(|i| i * P).collect::<Vec<_>>());

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes_b);
    b.bind_port(Port::Positions, pos);
    b.bind_port(Port::Pages, pages);
    b.bind_port(Port::PageIndptr, page_rows);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::AttnMask, kvm);
    b.stage(Stage::Epilogue, move || {
        let cand = add(broadcast(reshape(scores.take(), [B, 1]), [B, V]), log_softmax(intrinsics::logits()));
        let (s, i) = top_k(reshape(cand, [B * V]), B);
        let parent = div(&i, V);
        let pg = gather(pages.take(), &parent);
        let pl = gather(lens.take(), &parent);
        let n = gather(np.take(), &parent);
        let tf = gather(tfill.take(), &parent);
        let lanes = iota(B);
        let heir = scatter_set(&lanes, &parent, &lanes);
        let cont = and(eq(gather(heir, &parent), &lanes), lt(&tf, PAGE_T));
        let slot = select(&cont, gather(tslot.take(), &parent), fresh.take());
        let off = select(&cont, &tf, 0u32);
        let n2 = select(&cont, &n, add(&n, 1u32));
        let tcol = add(mul(&lanes, P), sub(&n2, 1u32));
        pages.put(reshape(scatter_set(reshape(pg, [B * P]), &tcol, &slot), [B, P]));
        let off1 = add(&off, 1u32);
        let pl2 = reshape(scatter_set(reshape(pl, [B * P]), &tcol, &off1), [B, P]);
        lens.put(&pl2);
        klen.put(add(mul(sub(&n2, 1u32), PAGE_T), &off1));
        let io = reshape(iota(PAGE_T), [1, 1, PAGE_T]);
        let iob = broadcast(io, [B, P, PAGE_T]);
        let lb = broadcast(reshape(&pl2, [B, P, 1]), [B, P, PAGE_T]);
        kvm.put(reshape(lt(iob, lb), [B, P * PAGE_T]));
        pos.put(add(pos.take(), 1u32));
        np.put(&n2);
        tslot.put(&slot);
        tfill.put(&off1);
        w_slot.put(&slot);
        w_off.put(&off);
        let tok_i = cast(rem(&i, V), ptir_dsl::DType::I32);
        toks.put(&tok_i);
        scores.put(&s);
        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
    });
    b.build().expect("beam epilogue binds")
}

fn beam_profile() -> ModelProfile {
    ModelProfile { vocab: V, page_size: PAGE_T, num_layers: 2, ..ModelProfile::dummy() }
}

fn u32s(v: &[u32]) -> Value {
    Value::U32(v.to_vec())
}

/// Craft a `[B,V]` logit block whose per-row peak tokens make `top_k` over the
/// flattened `[B*V]` cand pick both survivors from beam `parent_beam` (so the
/// epilogue's `parent = [parent_beam, parent_beam]`), with tokens `t0`,`t1`.
fn logits_forcing_parent(parent_beam: u32, t0: u32, t1: u32) -> PassInputs {
    let mut l = vec![0.0f32; (B * V) as usize];
    let row = (parent_beam * V) as usize;
    // Two dominant tokens in the parent row. log_softmax is shift-invariant, so
    // the OTHER (uniform-0) row's best cand is exactly -log(V) ≈ -2.08; both
    // parent-row winners must clear that floor. With peaks 20/19 the second
    // token's log_softmax ≈ -1.3 > -2.08, so top_k picks BOTH from this row.
    l[row + t0 as usize] = 20.0;
    l[row + t1 as usize] = 19.0;
    PassInputs { logits: Some(Value::F32(l)), ..Default::default() }
}

fn descriptor_port(r: &pie_ptir::interp::StepReport, p: Port) -> Value {
    r.descriptor.iter().find(|(q, _)| *q == p).map(|(_, v)| v.clone()).unwrap_or_else(|| {
        panic!("port {p:?} not in descriptor")
    })
}

// ───────────────────────────────────────────────────────────────────────────
// GOLDEN — charlie fork-freeze (ptir_beam.rs::golden_charlie_fork_freeze_csrs):
// mid-run state pages=[5,6|5,6], lens=[4,2|4,2], np=[2,2], tail slot 6 fill 2.
// Both survivors fork from beam 0 (parent=[0,0]); fresh=[7,8]. Expect the [B,P]
// geometry pages=[5,6,7|5,6,_], klen=[9,7], w_slot=[7,6], w_off=[0,2].
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn golden_charlie_fork_freeze() {
    let traced = build_beam();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();

    // Seed the mid-run beam state (declaration order 0..=11; 12=fresh host-fed).
    let seeds: Vec<(u32, Value)> = vec![
        (0, u32s(&[5, 6, 0, 5, 6, 0])),  // pages [B,P]
        (1, u32s(&[4, 2, 0, 4, 2, 0])),  // lens  [B,P]
        (2, u32s(&[6, 6])),              // klen (stale; recomputed)
        (3, {
            let mut m = vec![false; (B * P * PAGE_T) as usize];
            for lane in 0..B as usize {
                for j in 0..P as usize {
                    let fill = [4usize, 2, 0][j];
                    for o in 0..fill {
                        m[lane * (P * PAGE_T) as usize + j * PAGE_T as usize + o] = true;
                    }
                }
            }
            Value::Bool(m)
        }),
        (4, u32s(&[6, 6])),   // pos
        (5, u32s(&[2, 2])),   // np
        (6, u32s(&[6, 6])),   // tslot (tail page slot)
        (7, u32s(&[2, 2])),   // tfill (tail fill)
        (8, u32s(&[6, 6])),   // w_slot
        (9, u32s(&[2, 2])),   // w_off
        (10, Value::I32(vec![1, 2])), // toks
        (11, Value::F32(vec![0.0, 0.0])), // scores
    ];
    let mut inst = Instance::new(&bound, &seeds).unwrap();

    // Force parent=[0,0] (both survivors from beam 0). fresh=[7,8].
    let inputs = logits_forcing_parent(0, 2, 3);
    // First step: fresh not yet fed ⇒ dummy-run (no commit).
    let r0 = inst.step(&bound, &inputs, &mut NoKernels).unwrap();
    assert!(!r0.committed, "fresh is the late edge ⇒ dummy-run");
    inst.host_put(&bound, 12, u32s(&[7, 8])).unwrap();
    let r1 = inst.step(&bound, &inputs, &mut NoKernels).unwrap();
    assert!(r1.committed, "missed: {:?}", r1.missed);

    // parent harvested from out_par (ch 14).
    assert_eq!(inst.host_take(&bound, 14).unwrap(), u32s(&[0, 0]), "both fork from beam 0");

    // The NEXT step's descriptor reflects the updated geometry.
    inst.host_put(&bound, 12, u32s(&[9, 10])).unwrap();
    let r2 = inst.step(&bound, &inputs, &mut NoKernels).unwrap();

    assert_eq!(descriptor_port(&r2, Port::KvLen), u32s(&[9, 7]), "klen (3-1)*4+1=9 ; (2-1)*4+3=7");
    assert_eq!(descriptor_port(&r2, Port::WSlot), u32s(&[7, 6]), "lane0 fork fresh 7; lane1 heir tail 6");
    assert_eq!(descriptor_port(&r2, Port::WOff), u32s(&[0, 2]), "lane0 fresh off 0; lane1 heir off 2");
    // Pages [B,P] = [5,6,7 | 5,6,_].
    let Value::U32(pg) = descriptor_port(&r2, Port::Pages) else { panic!() };
    assert_eq!(&pg[0..3], &[5, 6, 7], "lane 0 forks a fresh page (slot 7)");
    assert_eq!(&pg[3..5], &[5, 6], "heir lane 1 continues beam 0's pages");
}

/// A fresh single-token-per-beam start (`ptir_beam.rs::seed`): slots [100,101],
/// one prompt token each. Seeds channels 0..=11 (12 = fresh, host-fed).
fn seed_fresh(tfill: u32, page0_fill: u32) -> Vec<(u32, Value)> {
    let mut kvm = vec![false; (B * P * PAGE_T) as usize];
    for lane in 0..B as usize {
        for o in 0..page0_fill as usize {
            kvm[lane * (P * PAGE_T) as usize + o] = true;
        }
    }
    vec![
        (0, u32s(&[100, 0, 0, 101, 0, 0])),   // pages
        (1, u32s(&[page0_fill, 0, 0, page0_fill, 0, 0])), // lens
        (2, u32s(&[page0_fill, page0_fill])), // klen
        (3, Value::Bool(kvm)),                // kvm
        (4, u32s(&[1, 1])),                   // pos
        (5, u32s(&[1, 1])),                   // np
        (6, u32s(&[100, 101])),               // tslot
        (7, u32s(&[tfill, tfill])),           // tfill
        (8, u32s(&[100, 101])),               // w_slot
        (9, u32s(&[tfill.saturating_sub(1), tfill.saturating_sub(1)])), // w_off
        (10, Value::I32(vec![1, 1])),         // toks
        (11, Value::F32(vec![0.0, 0.0])),     // scores
    ]
}

/// Force `parent = [0, 1]` (each beam continues from itself): one dominant token
/// per row, so `top_k` picks each row's peak — the top-B are beam0's + beam1's.
fn logits_diagonal(t0: u32, t1: u32) -> PassInputs {
    let mut l = vec![0.0f32; (B * V) as usize];
    l[t0 as usize] = 20.0; // beam 0 peak
    l[(V + t1) as usize] = 20.0; // beam 1 peak
    PassInputs { logits: Some(Value::F32(l)), ..Default::default() }
}

/// Run one committed epilogue step (dummy-run until `fresh` is fed), then read
/// the resulting geometry off the following step's descriptor.
fn step_then_descriptor(
    inst: &mut Instance,
    bound: &pie_ptir::validate::BoundTrace,
    inputs: &PassInputs,
    fresh: &[u32],
) -> pie_ptir::interp::StepReport {
    let r0 = inst.step(bound, inputs, &mut NoKernels).unwrap();
    assert!(!r0.committed, "fresh is the late edge ⇒ dummy-run first");
    inst.host_put(bound, 12, u32s(fresh)).unwrap();
    let r1 = inst.step(bound, inputs, &mut NoKernels).unwrap();
    assert!(r1.committed, "committed after fresh fed; missed {:?}", r1.missed);
    // A following step exposes the post-commit geometry in its descriptor.
    inst.step(bound, inputs, &mut NoKernels).unwrap()
}

// ───────────────────────────────────────────────────────────────────────────
// GOLDEN — continue-tail (ptir_beam.rs::golden_continue_tail): each beam
// continues from itself (heir, tail has room) → same slot, offset advances.
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn golden_continue_tail() {
    let traced = build_beam();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();
    let mut inst = Instance::new(&bound, &seed_fresh(1, 1)).unwrap();
    let d = step_then_descriptor(&mut inst, &bound, &logits_diagonal(0, 1), &[200, 201]);
    assert_eq!(descriptor_port(&d, Port::WSlot), u32s(&[100, 101]), "wrote parent's tail slot");
    assert_eq!(descriptor_port(&d, Port::WOff), u32s(&[1, 1]), "appended after the prompt token");
    assert_eq!(descriptor_port(&d, Port::KvLen), u32s(&[2, 2]), "one more valid token (np=1)");
    let Value::U32(pg) = descriptor_port(&d, Port::Pages) else { panic!() };
    assert_eq!(pg[0], 100, "beam 0 tail slot unchanged");
    assert_eq!(pg[3], 101, "beam 1 tail slot unchanged");
}

// ───────────────────────────────────────────────────────────────────────────
// GOLDEN — page-turn (ptir_beam.rs::golden_page_turn): the tail page is FULL, so
// even the heir forks a fresh page (offset 0, np+1).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn golden_page_turn() {
    let traced = build_beam();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();
    // tail full: tfill=PAGE_T, page-0 fill=PAGE_T.
    let mut inst = Instance::new(&bound, &seed_fresh(PAGE_T, PAGE_T)).unwrap();
    let d = step_then_descriptor(&mut inst, &bound, &logits_diagonal(0, 1), &[200, 201]);
    assert_eq!(descriptor_port(&d, Port::WSlot), u32s(&[200, 201]), "wrote the FRESH grant");
    assert_eq!(descriptor_port(&d, Port::WOff), u32s(&[0, 0]), "start of the fresh page");
    assert_eq!(descriptor_port(&d, Port::KvLen), u32s(&[PAGE_T + 1, PAGE_T + 1]), "full page + 1 (np=2)");
    let Value::U32(pg) = descriptor_port(&d, Port::Pages) else { panic!() };
    assert_eq!(pg[1], 200, "beam 0 page 1 = fresh slot");
    assert_eq!(pg[P as usize + 1], 201, "beam 1 page 1 = fresh slot");
}

// ───────────────────────────────────────────────────────────────────────────
// GOLDEN — fork-freeze (ptir_beam.rs::golden_fork_freeze): both survivors fork
// from beam 0. The heir (last lane) continues in-place; the non-heir forks a
// fresh page (the frozen sibling references beam 0's shared tail read-only).
// ───────────────────────────────────────────────────────────────────────────
#[test]
fn golden_fork_freeze() {
    let traced = build_beam();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();
    let mut inst = Instance::new(&bound, &seed_fresh(1, 1)).unwrap();
    // parent=[0,0]; heir[0] = last lane with parent==0 = lane 1 → lane 1 continues.
    let d = step_then_descriptor(&mut inst, &bound, &logits_forcing_parent(0, 2, 3), &[200, 201]);
    let Value::U32(ws) = descriptor_port(&d, Port::WSlot) else { panic!() };
    let Value::U32(wo) = descriptor_port(&d, Port::WOff) else { panic!() };
    assert_eq!(ws[1], 100, "heir continues beam 0's tail slot");
    assert_eq!(wo[1], 1, "heir appends in place");
    assert_eq!(ws[0], 200, "fork took the fresh slot");
    assert_eq!(wo[0], 0, "fork writes a fresh page");
    let Value::U32(pg) = descriptor_port(&d, Port::Pages) else { panic!() };
    assert_eq!(pg[0], 100, "fork keeps beam 0's shared page 0");
    assert_eq!(pg[P as usize], 100, "heir keeps beam 0's page 0");
    // klen: lane0 fork np=2 → (2-1)*4+1=5 ; lane1 heir np=1 → (1-1)*4+2=2.
    assert_eq!(descriptor_port(&d, Port::KvLen), u32s(&[5, 2]), "fork np=2 klen 5; heir np=1 klen 2");
}


// ═══════════════════════════════════════════════════════════════════════════
// B1 wire-form: the device-geometry program computes CSR page_indptr = CumSum(np)
// (leading 0) and the densely-packed live pages IN-GRAPH, bound to the
// PageIndptr / Pages ports — so the driver reads page_indptr[B] live entries and
// never sees the host replay. Channels 16 = page_indptr [B+1], 17 = packed [B*P].
// ═══════════════════════════════════════════════════════════════════════════

fn build_beam_wire() -> Traced {
    ptir_dsl::model::configure(V, PAGE_T, 2);

    let pages = leak(Channel::seeded([B, P], dtype::u32).named("pages"));
    let lens = leak(Channel::seeded([B, P], dtype::u32).named("lens"));
    let klen = leak(Channel::from(vec![0u32; B as usize]).named("klen"));
    let kvm = leak(Channel::seeded([B, P * PAGE_T], dtype::bool).named("kvm"));
    let pos = leak(Channel::from(vec![0u32; B as usize]).named("pos"));
    let np = leak(Channel::from(vec![1u32; B as usize]).named("np"));
    let tslot = leak(Channel::from(vec![0u32; B as usize]).named("tslot"));
    let tfill = leak(Channel::from(vec![0u32; B as usize]).named("tfill"));
    let w_slot = leak(Channel::from(vec![0u32; B as usize]).named("w_slot"));
    let w_off = leak(Channel::from(vec![0u32; B as usize]).named("w_off"));
    let toks = leak(Channel::from(vec![1i32; B as usize]).named("toks"));
    let scores = leak(Channel::from(vec![0.0f32; B as usize]).named("scores"));
    let fresh = leak(Channel::new([B], dtype::u32).named("fresh"));
    let out = leak(Channel::new([B], dtype::i32).named("out"));
    let out_par = leak(Channel::new([B], dtype::u32).named("out_par"));
    let out_scr = leak(Channel::new([B], dtype::f32).named("out_scr"));
    // Wire-form derivative channels: device-produced each fire but read by the
    // descriptor (Pages/PageIndptr ports), so seeded for the fire-0 read (like
    // pages/kvm). Put-only ⇒ the tracer auto-drains them.
    let page_indptr = leak(Channel::seeded([B + 1], dtype::u32).named("page_indptr"));
    let packed = leak(Channel::seeded([B * P], dtype::u32).named("packed"));
    // w_cont [B] bool (channel 18): per-lane "continued a shared tail in place"
    // (heir) vs "forked a fresh page". A host-reader terminal output — the
    // runtime reads it at finalize to reclaim the UNUSED fresh page grants of
    // continuing heirs (PageLease.reclaim_after_fire).
    let w_cont_ch = leak(Channel::new([B], dtype::bool).named("w_cont"));

    fresh.put(vec![0u32; B as usize]);

    let lanes_b = Tensor::constant((0u32..=B).collect::<Vec<_>>());

    let mut b = Builder::new();
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes_b);
    b.bind_port(Port::Positions, pos);
    // Wire form: Pages ← densely-packed live pages; PageIndptr ← CumSum(np).
    b.bind_port(Port::Pages, packed);
    b.bind_port(Port::PageIndptr, page_indptr);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::AttnMask, kvm);
    b.stage(Stage::Epilogue, move || {
        let cand = add(broadcast(reshape(scores.take(), [B, 1]), [B, V]), log_softmax(intrinsics::logits()));
        let (s, i) = top_k(reshape(cand, [B * V]), B);
        let parent = div(&i, V);
        let pg = gather(pages.take(), &parent);
        let pl = gather(lens.take(), &parent);
        let n = gather(np.take(), &parent);
        let tf = gather(tfill.take(), &parent);
        let lanes = iota(B);
        let heir = scatter_set(&lanes, &parent, &lanes);
        let cont = and(eq(gather(heir, &parent), &lanes), lt(&tf, PAGE_T));
        let slot = select(&cont, gather(tslot.take(), &parent), fresh.take());
        let off = select(&cont, &tf, 0u32);
        let n2 = select(&cont, &n, add(&n, 1u32));
        let tcol = add(mul(&lanes, P), sub(&n2, 1u32));
        let pg3 = reshape(scatter_set(reshape(pg, [B * P]), &tcol, &slot), [B, P]);
        pages.put(&pg3);
        let off1 = add(&off, 1u32);
        let pl2 = reshape(scatter_set(reshape(pl, [B * P]), &tcol, &off1), [B, P]);
        lens.put(&pl2);
        klen.put(add(mul(sub(&n2, 1u32), PAGE_T), &off1));
        let io = reshape(iota(PAGE_T), [1, 1, PAGE_T]);
        let iob = broadcast(io, [B, P, PAGE_T]);
        let lb = broadcast(reshape(&pl2, [B, P, 1]), [B, P, PAGE_T]);
        kvm.put(reshape(lt(iob, lb), [B, P * PAGE_T]));
        pos.put(add(pos.take(), 1u32));
        np.put(&n2);
        tslot.put(&slot);
        tfill.put(&off1);
        w_slot.put(&slot);
        w_off.put(&off);
        let tok_i = cast(rem(&i, V), ptir_dsl::DType::I32);
        toks.put(&tok_i);
        scores.put(&s);

        // ── wire-form geometry (B1) — emitted BEFORE the reader terminal puts
        // so the reader auto-drain drop can't shift these ops' SSA value ids ──
        // page_indptr [B+1] = [0, CumSum(np)] — scatter the inclusive prefix sum
        // into a leading-0 vector at destinations iota(B)+1. CumSum is F32-only
        // (a scan op), so route np through F32 (values are small, exact).
        let cps = cast(cumsum(cast(&n2, ptir_dsl::DType::F32)), ptir_dsl::DType::U32); // [B], e.g. [3,5]
        let zeros_b1 = Tensor::constant(vec![0u32; (B + 1) as usize]);
        let pidx = scatter_set(zeros_b1, add(iota(B), 1u32), &cps); // [0, 3, 5]
        page_indptr.put(&pidx);
        // Densely pack the live [B,P] pages: entry (b,j) → packed[page_start[b]+j].
        // page_start = page_indptr[0..B]; padding entries (j>=np[b]) land at
        // >= next beam's start and are overwritten by that beam's higher-flat-
        // index live entries (last-wins scatter); trailing padding sits at
        // >= sum(np) and the driver ignores it.
        let page_start = gather(&pidx, iota(B)); // [0, 3]
        let start_bp = broadcast(reshape(&page_start, [B, 1]), [B, P]);
        let col = broadcast(reshape(iota(P), [1, P]), [B, P]);
        let dest = reshape(add(start_bp, col), [B * P]);
        let pg3_flat = reshape(&pg3, [B * P]);
        packed.put(scatter_set(&pg3_flat, &dest, &pg3_flat));

        // reader terminal outputs LAST (their auto-drain ChanTakes are dropped
        // at build; keeping them at the tail preserves the wire ops' value ids).
        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        w_cont_ch.put(&cont); // per-lane heir(true)/fork(false) for page reclaim
    });
    b.build().expect("wire-form beam epilogue binds")
}

/// charlie fork-freeze, wire form: page_indptr = [0,3,5], packed = [5,6,7,5,6].
#[test]
fn wire_charlie_page_indptr_and_packed() {
    let traced = build_beam_wire();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();
    let seeds: Vec<(u32, Value)> = vec![
        (0, u32s(&[5, 6, 0, 5, 6, 0])),
        (1, u32s(&[4, 2, 0, 4, 2, 0])),
        (2, u32s(&[6, 6])),
        (3, {
            let mut m = vec![false; (B * P * PAGE_T) as usize];
            for lane in 0..B as usize {
                for j in 0..P as usize {
                    let fill = [4usize, 2, 0][j];
                    for o in 0..fill {
                        m[lane * (P * PAGE_T) as usize + j * PAGE_T as usize + o] = true;
                    }
                }
            }
            Value::Bool(m)
        }),
        (4, u32s(&[6, 6])),
        (5, u32s(&[2, 2])),
        (6, u32s(&[6, 6])),
        (7, u32s(&[2, 2])),
        (8, u32s(&[6, 6])),
        (9, u32s(&[2, 2])),
        (10, Value::I32(vec![1, 2])),
        (11, Value::F32(vec![0.0, 0.0])),
        (16, u32s(&[0, 2, 4])),          // page_indptr fire-0 seed (np=[2,2])
        (17, u32s(&[5, 6, 0, 5, 6, 0])), // packed fire-0 seed (shape-valid)
    ];
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    let inputs = logits_forcing_parent(0, 2, 3);
    // First step: fresh not yet fed ⇒ dummy-run.
    let r0 = inst.step(&bound, &inputs, &mut NoKernels).unwrap();
    assert!(!r0.committed);
    inst.host_put(&bound, 12, u32s(&[7, 8])).unwrap();
    let r1 = inst.step(&bound, &inputs, &mut NoKernels).unwrap();
    assert!(r1.committed, "missed: {:?}", r1.missed);
    // w_cont (ch 18): lane 0 forks (false), heir lane 1 continues in place (true).
    assert_eq!(inst.host_take(&bound, 18).unwrap(), Value::Bool(vec![false, true]), "w_cont");

    // The NEXT step's descriptor reflects the updated wire-form geometry.
    inst.host_put(&bound, 12, u32s(&[9, 10])).unwrap();
    let d = inst.step(&bound, &inputs, &mut NoKernels).unwrap();

    // CSR page_indptr = [0, np0, np0+np1] = [0, 3, 5].
    assert_eq!(descriptor_port(&d, Port::PageIndptr), u32s(&[0, 3, 5]), "page_indptr = [0, CumSum(np)]");
    // Densely-packed live pages: beam0 [5,6,7] then beam1 [5,6]; trailing padding
    // ignored (driver reads page_indptr[B]=5 entries).
    let Value::U32(pk) = descriptor_port(&d, Port::Pages) else { panic!() };
    assert_eq!(&pk[0..5], &[5, 6, 7, 5, 6], "packed live pages");
}

/// continue-tail, wire form: np=[1,1] ⇒ page_indptr=[0,1,2], packed=[100,101].
#[test]
fn wire_continue_tail_packed() {
    let traced = build_beam_wire();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();
    let mut seeds = seed_fresh(1, 1);
    seeds.push((16, u32s(&[0, 1, 2]))); // page_indptr fire-0 (np=[1,1])
    seeds.push((17, u32s(&[100, 0, 0, 101, 0, 0]))); // packed fire-0 seed
    let mut inst = Instance::new(&bound, &seeds).unwrap();
    let d = step_then_descriptor(&mut inst, &bound, &logits_diagonal(0, 1), &[200, 201]);
    assert_eq!(descriptor_port(&d, Port::PageIndptr), u32s(&[0, 1, 2]), "one live page per beam");
    let Value::U32(pk) = descriptor_port(&d, Port::Pages) else { panic!() };
    assert_eq!(&pk[0..2], &[100, 101], "packed live pages (padding gaps closed)");
}
