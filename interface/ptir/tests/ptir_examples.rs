//! Thrust-3 P0/P4 exit tests: the overview **§3 example** (grammar-masked
//! gumbel-greedy decode) and the **§6.2 beam epilogue** serialize, validate,
//! and hash stably; and both run end-to-end on the tier-0 reference
//! interpreter — dummy-run on a late host edge observed and recovered, replay
//! determinism (T8), beam reorder/freeze geometry exact.
#![cfg(feature = "eval")]

use pie_ptir::container::{decode, encode};
use pie_ptir::container_hash;
use pie_ptir::interp::Value;
use pie_ptir::interp::{Instance, NoKernels, PassInputs};
use pie_ptir::registry::{ModelProfile, Port};
use pie_ptir::validate::bind;

#[path = "common/traces.rs"]
mod traces;
use traces::*;

#[test]
fn section3_serializes_validates_hashes_stably() {
    let c = section3_trace();
    let bytes = encode(&c);
    let h = container_hash(&bytes);
    // decode → re-encode is byte-identical (canonical), so the hash is stable.
    let c2 = decode(&bytes).expect("decode");
    assert_eq!(c2, c);
    assert_eq!(container_hash(&encode(&c2)), h);
    let bound = bind(c, ModelProfile::dummy()).expect("bind");
    assert_eq!(bound.hash, h);
}

#[test]
fn section3_end_to_end_with_late_mask_and_recovery() {
    let c = section3_trace();
    let b = bind(c, ModelProfile::dummy()).unwrap();
    let seeds = [
        (0u32, i32s(&[1])),       // BOS
        (3u32, u32s(&[1])),       // len
        (4u32, u32s(&[1234, 0])), // rng [key, ctr]
    ];
    let mut inst = Instance::new(&b, &seeds).unwrap();

    // Prime mask_0 (allow everything), fire step 0: strong logit on 7 wins.
    inst.host_put(&b, 2, allow_all()).unwrap();
    let inputs = PassInputs {
        logits: Some(flat_logits(7, 100.0)),
        ..Default::default()
    };
    let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(r.committed);
    // Descriptor saw the seeded BOS token.
    assert_eq!(r.descriptor[0], (Port::EmbedTokens, i32s(&[1])));
    assert_eq!(inst.host_take(&b, 1).unwrap(), i32s(&[7]));

    // Step 1 submitted before mask_1 lands: sample parks (dummy-run, no
    // commit) — §1's software-pipelining contract.
    let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(!r.committed);
    assert_eq!(r.missed.unwrap().0, 2);
    assert_eq!(
        inst.host_read(&b, 1),
        Err(pie_ptir::interp::HostError::WouldBlock)
    );

    // mask_1 excludes 7, allows only 3: resubmission commits with 3 even
    // though the raw logit still favors 7 — the grammar constrains exactly.
    inst.host_put(&b, 2, allow_only(&[3])).unwrap();
    let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(r.committed);
    assert_eq!(inst.host_take(&b, 1).unwrap(), i32s(&[3]));
}

#[test]
fn section3_replay_determinism_t8() {
    // Same seeds + same feeds ⇒ token-identical streams, across instances.
    let c = section3_trace();
    let b = bind(c, ModelProfile::dummy()).unwrap();
    let run = || {
        let seeds = [
            (0u32, i32s(&[1])),
            (3u32, u32s(&[9])),
            (4u32, u32s(&[77, 5])),
        ];
        let mut inst = Instance::new(&b, &seeds).unwrap();
        // Flat logits: the gumbel noise alone decides — pure (key, ctr).
        let inputs = PassInputs {
            logits: Some(f32s(&[0.0; VOCAB as usize])),
            ..Default::default()
        };
        let mut toks = Vec::new();
        for _ in 0..4 {
            inst.host_put(&b, 2, allow_all()).unwrap();
            let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
            assert!(r.committed);
            toks.push(inst.host_take(&b, 1).unwrap());
        }
        toks
    };
    let (a, c2) = (run(), run());
    assert_eq!(a, c2);
    // And the counter advanced ⇒ successive steps differ (noise is fresh).
    assert_ne!(a[0], a[1]);
}

#[test]
fn beam_epilogue_serializes_validates_hashes_stably() {
    let c = beam_trace();
    let bytes = encode(&c);
    let h = container_hash(&bytes);
    assert_eq!(decode(&bytes).expect("decode"), c);
    let bound = bind(c.clone(), beam_profile()).expect("bind");
    assert_eq!(bound.hash, h);
    // Identity is deterministic across processes for the same trace.
    assert_eq!(container_hash(&encode(&decode(&bytes).unwrap())), h);
}

#[test]
fn beam_step_reorder_freeze_geometry_exact() {
    let c = beam_trace();
    let b = bind(c, beam_profile()).unwrap();
    // Prompt shared: both rows read pages [5,6,_]; page 5 full (4), page 6
    // partial (2); physical span 6; two live entries; tail slot 6, fill 2.
    let seeds: Vec<(u32, Value)> = vec![
        (0, u32s(&[5, 6, 0, 5, 6, 0])),
        (1, u32s(&[4, 2, 0, 4, 2, 0])),
        (2, u32s(&[6, 6])),
        (3, {
            let mut m = vec![false; (BB * P * PAGE) as usize];
            for lane in 0..BB as usize {
                for j in 0..P as usize {
                    let lens = [4, 2, 0][j];
                    for o in 0..lens {
                        m[lane * (P * PAGE) as usize + j * PAGE as usize + o] = true;
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
        (10, i32s(&[1, 2])),
        (11, f32s(&[0.0, 0.0])),
    ];
    let mut inst = Instance::new(&b, &seeds).unwrap();

    // No headroom grant yet ⇒ dummy-run, no commit (fresh is the late edge).
    let mut logits = vec![0.0f32; (BB * V) as usize];
    logits[3] = 8.0; // row 0 → token 3
    logits[(V + 5) as usize] = 7.0; // row 1 → token 5
    let inputs = PassInputs {
        logits: Some(Value::F32(logits)),
        ..Default::default()
    };
    let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(!r.committed);
    assert_eq!(r.missed.unwrap().0, 12);

    // fresh granted ⇒ commit. Both winners survive (parent = [0, 1]), both
    // are their parent's designated child with tail room ⇒ freeze/continue
    // path: no fresh slot consumed into geometry, tails advance to fill 3.
    inst.host_put(&b, 12, u32s(&[7, 8])).unwrap();
    let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(r.committed, "missed: {:?}", r.missed);

    assert_eq!(inst.host_take(&b, 13).unwrap(), i32s(&[3, 5])); // tokens
    assert_eq!(inst.host_take(&b, 14).unwrap(), u32s(&[0, 1])); // parents
    let Value::F32(scr) = inst.host_take(&b, 15).unwrap() else {
        panic!()
    };
    assert!(scr[0] > scr[1], "row-0 winner scored higher: {scr:?}");

    // Descriptor for the NEXT pass sees the updated geometry.
    let r2 = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(!r2.committed); // out not harvested → back-pressure parks it
    let get = |p: Port| {
        r2.descriptor
            .iter()
            .find(|(q, _)| *q == p)
            .unwrap()
            .1
            .clone()
    };
    assert_eq!(get(Port::EmbedTokens), i32s(&[3, 5]));
    assert_eq!(get(Port::KvLen), u32s(&[7, 7])); // span grew by one
    assert_eq!(get(Port::Pages), u32s(&[5, 6, 0, 5, 6, 0])); // no page turn
    assert_eq!(get(Port::WSlot), u32s(&[6, 6]));
    assert_eq!(get(Port::WOff), u32s(&[2, 2])); // the chosen token lands at
    // the old fill; tfill/lens
    // already cover it (off+1)
    assert_eq!(get(Port::Positions), u32s(&[7, 7]));
    // kvm: per lane, page lens now [4, 3, 0].
    let Value::Bool(m) = get(Port::AttnMask) else {
        panic!()
    };
    let lane0: Vec<bool> = m[..(P * PAGE) as usize].to_vec();
    let expect: Vec<bool> = (0..(P * PAGE) as usize)
        .map(|o| {
            let (page, off) = (o / PAGE as usize, o % PAGE as usize);
            off < [4usize, 3, 0][page]
        })
        .collect();
    assert_eq!(lane0, expect);
}

#[test]
fn beam_page_turn_takes_fresh_slot() {
    // Same setup but the tail page is FULL (fill 4): every heir takes the
    // fresh-slot path at offset 0 — the page boundary is just divergence's
    // fresh path taken by every heir at once (§6.2).
    let c = beam_trace();
    let b = bind(c, beam_profile()).unwrap();
    let seeds: Vec<(u32, Value)> = vec![
        (0, u32s(&[5, 6, 0, 5, 6, 0])),
        (1, u32s(&[4, 4, 0, 4, 4, 0])),
        (2, u32s(&[8, 8])),
        (3, {
            let mut m = vec![false; (BB * P * PAGE) as usize];
            for lane in 0..BB as usize {
                for o in 0..8 {
                    m[lane * (P * PAGE) as usize + o] = true;
                }
            }
            Value::Bool(m)
        }),
        (4, u32s(&[8, 8])),
        (5, u32s(&[2, 2])),
        (6, u32s(&[6, 6])),
        (7, u32s(&[4, 4])), // tail full
        (8, u32s(&[6, 6])),
        (9, u32s(&[4, 4])),
        (10, i32s(&[1, 2])),
        (11, f32s(&[0.0, 0.0])),
    ];
    let mut inst = Instance::new(&b, &seeds).unwrap();
    inst.host_put(&b, 12, u32s(&[7, 8])).unwrap();
    let mut logits = vec![0.0f32; (BB * V) as usize];
    logits[3] = 8.0;
    logits[(V + 5) as usize] = 7.0;
    let inputs = PassInputs {
        logits: Some(Value::F32(logits)),
        ..Default::default()
    };
    let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(r.committed, "missed: {:?}", r.missed);
    inst.host_take(&b, 13).unwrap();
    inst.host_take(&b, 14).unwrap();
    inst.host_take(&b, 15).unwrap();
    let r2 = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    let get = |p: Port| {
        r2.descriptor
            .iter()
            .find(|(q, _)| *q == p)
            .unwrap()
            .1
            .clone()
    };
    // Fresh slots 7/8 entered each row's third entry; writes land there at 0.
    assert_eq!(get(Port::Pages), u32s(&[5, 6, 7, 5, 6, 8]));
    assert_eq!(get(Port::KvLen), u32s(&[9, 9])); // (3-1)*4 + 1
    assert_eq!(get(Port::WSlot), u32s(&[7, 8]));
    assert_eq!(get(Port::WOff), u32s(&[0, 0]));
}

#[test]
fn beam_reorder_both_from_one_parent_freezes_sibling() {
    // Row 0 dominates: both winners come from parent 0 (a fork). The
    // designated child (last in lane order — lane 1) continues parent 0's
    // tail; lane 0 opens a fresh slot at offset 0 (freeze, not copy).
    let c = beam_trace();
    let b = bind(c, beam_profile()).unwrap();
    let seeds: Vec<(u32, Value)> = vec![
        (0, u32s(&[5, 6, 0, 9, 9, 0])), // row 1 differs so reorder is visible
        (1, u32s(&[4, 2, 0, 4, 2, 0])),
        (2, u32s(&[6, 6])),
        (3, Value::Bool(vec![true; (BB * P * PAGE) as usize])),
        (4, u32s(&[6, 6])),
        (5, u32s(&[2, 2])),
        (6, u32s(&[6, 6])),
        (7, u32s(&[2, 2])),
        (8, u32s(&[6, 6])),
        (9, u32s(&[2, 2])),
        (10, i32s(&[1, 2])),
        (11, f32s(&[5.0, -100.0])), // row 0 vastly better
    ];
    let mut inst = Instance::new(&b, &seeds).unwrap();
    inst.host_put(&b, 12, u32s(&[7, 8])).unwrap();
    let mut logits = vec![0.0f32; (BB * V) as usize];
    logits[3] = 8.0;
    logits[4] = 7.5;
    let inputs = PassInputs {
        logits: Some(Value::F32(logits)),
        ..Default::default()
    };
    let r = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    assert!(r.committed, "missed: {:?}", r.missed);
    assert_eq!(inst.host_take(&b, 14).unwrap(), u32s(&[0, 0])); // both from row 0
    assert_eq!(inst.host_take(&b, 13).unwrap(), i32s(&[3, 4]));
    inst.host_take(&b, 15).unwrap();
    let r2 = inst.step(&b, &inputs, &mut NoKernels).unwrap();
    let get = |p: Port| {
        r2.descriptor
            .iter()
            .find(|(q, _)| *q == p)
            .unwrap()
            .1
            .clone()
    };
    // heir[0] = 1 (last child in lane order): lane 1 continues tail slot 6 at
    // offset 2; lane 0 froze and opened fresh slot 7 at offset 0.
    assert_eq!(get(Port::Pages), u32s(&[5, 6, 7, 5, 6, 0]));
    assert_eq!(get(Port::WSlot), u32s(&[7, 6]));
    assert_eq!(get(Port::WOff), u32s(&[0, 2]));
    // lens: lane 0's inherited entry for slot 6 stays FROZEN at 2 (its kvm
    // row masks the sibling's new token); lane 1's advanced to 3.
    let Value::Bool(m) = get(Port::AttnMask) else {
        panic!()
    };
    let at = |lane: usize, page: usize, off: usize| {
        m[lane * (P * PAGE) as usize + page * PAGE as usize + off]
    };
    assert!(at(0, 1, 1)); // shared prefix visible
    assert!(
        !at(0, 1, 2),
        "lane 0 must NOT see the sibling's tail token (freeze)"
    );
    assert!(at(1, 1, 2), "lane 1 (designated child) sees its own append");
    // klen presents frozen pages FULL? No — physical span counts each lane's
    // last page partial; lane 0's new tail is entry 3 at fill 1: (3-1)*4+1=9;
    // lane 1 continues: (2-1)*4+3=7.
    assert_eq!(get(Port::KvLen), u32s(&[9, 7]));
}
