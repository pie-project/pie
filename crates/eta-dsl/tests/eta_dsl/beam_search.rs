use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_ir::registry::{ModelProfile, Port};
use eta_ir::validate::{BoundTrace, bind};

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Dtype, Traced};

const B: u32 = 2;
const V: u32 = 8;
const PAGE_T: u32 = 4;
const POOL_PAGES: u32 = 3;
const POOL: u32 = POOL_PAGES * PAGE_T;

fn leak<T>(v: T) -> &'static T {
    Box::leak(Box::new(v))
}

fn build_designb() -> Traced {
    let mask = leak(Channel::seeded([B, POOL], dtype::bool).named("mask"));
    let scores = leak(Channel::from(vec![0.0f32; B as usize]).named("scores"));
    let toks = leak(Channel::from(vec![1i32; B as usize]).named("toks"));
    let pos = leak(Channel::from(vec![0u32; B as usize]).named("pos"));
    let fill = leak(Channel::from(vec![0u32; 1]).named("fill"));
    let klen = leak(Channel::from(vec![0u32; B as usize]).named("klen"));
    let w_slot = leak(Channel::from(vec![0u32; B as usize]).named("w_slot"));
    let w_off = leak(Channel::from(vec![0u32; B as usize]).named("w_off"));
    let out = leak(Channel::new([B], dtype::i32).named("out"));
    let out_par = leak(Channel::new([B], dtype::u32).named("out_par"));
    let out_scr = leak(Channel::new([B], dtype::f32).named("out_scr"));
    let out_mask = leak(Channel::new([B, POOL], dtype::bool).named("out_mask"));
    let out_wslot = leak(Channel::new([B], dtype::u32).named("out_wslot"));
    let out_woff = leak(Channel::new([B], dtype::u32).named("out_woff"));

    let pool_pages: Vec<u32> = (0..B).flat_map(|_| 0..POOL_PAGES).collect();
    let page_indptr: Vec<u32> = (0..=B).map(|b| b * POOL_PAGES).collect();
    let pages_c = leak(Channel::from(pool_pages).named("pages"));
    let page_indptr_c = leak(Channel::from(page_indptr).named("page_indptr"));
    let lanes_b = leak(Channel::from((0u32..=B).collect::<Vec<_>>()).named("indptr"));

    let mut b = Builder::new(V, PAGE_T);
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes_b);
    b.bind_port(Port::Positions, pos);
    b.bind_port(Port::Pages, pages_c);
    b.bind_port(Port::PageIndptr, page_indptr_c);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::WSlot, w_slot);
    b.bind_port(Port::WOff, w_off);
    b.bind_port(Port::AttnMask, mask);
    b.stage(Stage::Epilogue, move || {
        let cand = add(
            broadcast(reshape(scores.take(), [B, 1]), [B, V]),
            log_softmax(intrinsics::logits()),
        );
        let (s, i) = top_k(reshape(cand, [B * V]), B);
        let parent = div(&i, V);
        let tok_i = cast(rem(&i, V), Dtype::I32);

        let base = fill.take();
        let lane = iota(B);
        let base_b = broadcast(reshape(&base, [1]), [B]);
        let wpos = add(&base_b, &lane);

        let inherited = gather(mask.take(), &parent);
        let col = broadcast(reshape(iota(POOL), [1, POOL]), [B, POOL]);
        let wpos_b = broadcast(reshape(&wpos, [B, 1]), [B, POOL]);
        let newpos = eq(col, wpos_b);
        let new_mask = or(inherited, &newpos);
        mask.put(&new_mask);

        let w_slot_v = div(&wpos, PAGE_T);
        let w_off_v = rem(&wpos, PAGE_T);
        w_slot.put(&w_slot_v);
        w_off.put(&w_off_v);

        let filled = add(&base, B);
        klen.take();
        klen.put(broadcast(reshape(&filled, [1]), [B]));

        pos.put(add(pos.take(), 1u32));
        fill.put(&filled);
        scores.put(&s);
        toks.put(&tok_i);

        out.put(&tok_i);
        out_par.put(&parent);
        out_scr.put(&s);
        out_mask.put(&new_mask);
        out_wslot.put(&w_slot_v);
        out_woff.put(&w_off_v);
    });
    b.build().expect("design B beam epilogue binds")
}

fn beam_profile() -> ModelProfile {
    ModelProfile {
        vocab: V,
        page_size: PAGE_T,
        num_layers: 2,
        ..ModelProfile::dummy()
    }
}

fn u32s(v: &[u32]) -> Value {
    Value::U32(v.to_vec())
}

fn mask_of(rows: &[&[u32]]) -> Value {
    let mut m = vec![false; (B * POOL) as usize];
    for (b, positions) in rows.iter().enumerate() {
        for &p in *positions {
            m[b * POOL as usize + p as usize] = true;
        }
    }
    Value::Bool(m)
}

fn logits_forcing_parent(parent_beam: u32, t0: u32, t1: u32) -> PassInputs {
    let mut l = vec![0.0f32; (B * V) as usize];
    let row = (parent_beam * V) as usize;
    l[row + t0 as usize] = 20.0;
    l[row + t1 as usize] = 19.0;
    PassInputs {
        logits: Some(Value::F32(l)),
        ..Default::default()
    }
}

struct Harvest {
    tok: Value,
    par: Value,
    mask: Value,
    wslot: Value,
    woff: Value,
}

fn harvest(inst: &mut Instance, bound: &BoundTrace) -> Harvest {
    let tok = inst.host_take(bound, 8).unwrap();
    let par = inst.host_take(bound, 9).unwrap();
    let _scr = inst.host_take(bound, 10).unwrap();
    let mask = inst.host_take(bound, 11).unwrap();
    let wslot = inst.host_take(bound, 12).unwrap();
    let woff = inst.host_take(bound, 13).unwrap();
    Harvest {
        tok,
        par,
        mask,
        wslot,
        woff,
    }
}

#[test]
fn golden_designb_fork_from_shared_prefix() {
    let traced = build_designb();
    let bound = bind(traced.container().clone(), beam_profile()).unwrap();

    let seeds: Vec<(u32, Value)> = vec![
        (0, mask_of(&[&[0, 1], &[0, 1]])),
        (1, Value::F32(vec![0.0, 0.0])),
        (2, Value::I32(vec![1, 1])),
        (3, u32s(&[2, 2])),
        (4, u32s(&[2])),
        (5, u32s(&[2, 2])),
        (6, u32s(&[0, 0])),
        (7, u32s(&[0, 0])),
        (14, u32s(&[0, 1, 2, 0, 1, 2])),
        (15, u32s(&[0, 3, 6])),
        (16, u32s(&[0, 1, 2])),
    ];
    let mut inst = Instance::new(&bound, &seeds).unwrap();

    let inputs = logits_forcing_parent(0, 2, 3);
    let r0 = inst.step(&bound, &inputs, &mut NoKernels).unwrap();
    assert!(
        r0.committed,
        "no host-writer late edge in Design B ⇒ first fire commits: {:?}",
        r0.missed
    );

    let h = harvest(&mut inst, &bound);
    assert_eq!(h.par, u32s(&[0, 0]), "out_par: both fork from beam 0");
    assert_eq!(h.tok, Value::I32(vec![2, 3]), "out: tokens [2,3]");
    assert_eq!(
        h.mask,
        mask_of(&[&[0, 1, 2], &[0, 1, 3]]),
        "mask: beam0={{0,1,2}}, beam1={{0,1,3}} (shared {{0,1}} + own append)"
    );
    assert_eq!(
        h.wslot,
        u32s(&[0, 0]),
        "w_slot: positions 2,3 in pool page 0"
    );
    assert_eq!(h.woff, u32s(&[2, 3]), "w_off: offsets 2,3 within page 0");
}
