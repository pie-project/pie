use eta_compiler::eval::interp::{Instance, NoKernels, PassInputs, Value};
use eta_ir::registry::{EngineCaps, Graph, Port, Readout, ReadoutKind};
use eta_ir::validate::BoundTrace;

use eta_dsl::builder::Builder;
use eta_dsl::prelude::*;
use eta_dsl::{Channel, Dtype, Traced};

const B: u32 = 2;
const V: u32 = 8;
const PAGE_T: u32 = 4;
const POOL: u32 = 3 * PAGE_T;

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
    let slots = leak(Channel::from(vec![0u32; B as usize]).named("slots"));
    let out = leak(Channel::new([B], dtype::i32).named("out"));
    let out_par = leak(Channel::new([B], dtype::u32).named("out_par"));
    let out_scr = leak(Channel::new([B], dtype::f32).named("out_scr"));
    let out_mask = leak(Channel::new([B, POOL], dtype::bool).named("out_mask"));
    let out_slots = leak(Channel::new([B], dtype::u32).named("out_slots"));

    let lanes_b = leak(Channel::from((0u32..=B).collect::<Vec<_>>()).named("indptr"));

    let mut b = Builder::new(V, PAGE_T);
    b.bind_port(Port::EmbedTokens, toks);
    b.bind_port(Port::EmbedIndptr, lanes_b);
    b.bind_port(Port::Positions, pos);
    b.bind_port(Port::KvLen, klen);
    b.bind_port(Port::Slots, slots);
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

        slots.put(&wpos);

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
        out_slots.put(&wpos);
    });
    b.build().expect("design B beam epilogue binds")
}

fn served(
    container: eta_ir::container::TraceContainer,
    lanes: Vec<Option<Readout>>,
    num_layers: u32,
) -> Result<eta_ir::validate::BoundTrace, eta_ir::validate::ValidateError> {
    eta_ir::validate::bind(
        container,
        lanes,
        Graph {
            num_layers,
            ..Graph::default()
        },
        EngineCaps {
            lora: true,
            attn_page_mask: true,
        },
    )
}

fn logits_lane(vocab: u32) -> Vec<Option<Readout>> {
    vec![Some(Readout {
        kind: ReadoutKind::Logits,
        width: vocab,
    })]
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
    slots: Value,
}

fn harvest(inst: &mut Instance, bound: &BoundTrace) -> Harvest {
    let tok = inst.host_take(bound, 7).unwrap();
    let par = inst.host_take(bound, 8).unwrap();
    let _scr = inst.host_take(bound, 9).unwrap();
    let mask = inst.host_take(bound, 10).unwrap();
    let slots = inst.host_take(bound, 11).unwrap();
    Harvest {
        tok,
        par,
        mask,
        slots,
    }
}

#[test]
fn golden_designb_fork_from_shared_prefix() {
    let traced = build_designb();
    let bound = served(traced.container().clone(), logits_lane(V), 2).unwrap();

    let seeds: Vec<(u32, Value)> = vec![
        (0, mask_of(&[&[0, 1], &[0, 1]])),
        (1, Value::F32(vec![0.0, 0.0])),
        (2, Value::I32(vec![1, 1])),
        (3, u32s(&[2, 2])),
        (4, u32s(&[2])),
        (5, u32s(&[2, 2])),
        (6, u32s(&[0, 0])),
        (12, u32s(&[0, 1, 2])),
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
        h.slots,
        u32s(&[2, 3]),
        "slots: each beam appends at 2 and 3"
    );
}
