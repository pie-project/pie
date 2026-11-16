//! A keep's token runs on the rs list (CONTRACT-BC 18 left the buffer's
//! start off a slot): rows of an rs list slot written with known bytes move
//! at an offset into another slot, row for row across every rs plane; the
//! rows no run names keep what they held. A run that reads the rows it
//! writes, or reaches past its slot, is refused by name and copies nothing;
//! whole slots still copy whole. Skipped without a PJRT plugin.

use engine::Engine;
use engine::chunk::{ShapeId, SlotCopy, SlotRun};

use crate::common_llm::{Model, Refs};

const MODEL: &str = "qwen35-tiny";

fn slot(model: &Model, id: ShapeId, at: u32) -> Vec<u8> {
    let shell = model.engine.shell().expect("loaded");
    shell.slot_bytes(&[(id, at)]).expect("the slot reads")
}

fn runs(
    model: &mut Model,
    id: ShapeId,
    slots: Vec<(u32, u32)>,
    runs: Vec<SlotRun>,
) -> Result<(), String> {
    let device = model.loaded.caps.device.domain;
    model
        .engine
        .copy_slots(&SlotCopy {
            shape: id,
            src: device,
            dst: device,
            slots,
            runs,
        })
        .map_err(|e| e.to_string())
}

#[test]
fn an_rs_token_run_lands_row_for_row() {
    let Some(_device) = crate::common_dit::device() else {
        return;
    };
    let mut model = Model::load_with(MODEL, None, 256, 7, None);
    let (id, tokens, slot_bytes, planes) = {
        let shell = model.engine.shell().expect("loaded");
        let rs = shell
            .pools()
            .layout()
            .rs()
            .expect("a hybrid model lays an rs list")
            .clone();
        let planes: Vec<(u64, u64)> = rs
            .planes
            .iter()
            .map(|p| (p.offset, p.row_bytes()))
            .collect();
        (rs.id, rs.tokens, rs.slot_bytes as usize, planes)
    };
    assert!(tokens >= 4, "an rs list slot of {tokens} tokens");
    assert!(!planes.is_empty(), "the rs list lays its planes");
    let mut refs = Refs::of(&model);
    let (src, dst, whole) = (refs.take(id), refs.take(id), refs.take(id));
    let seed: Vec<u8> = (0..slot_bytes)
        .map(|at| u8::try_from((at as u64 * 7 + 1) % 251).expect("a byte"))
        .collect();
    {
        let shell = model.engine.shell_mut().expect("loaded");
        shell
            .write_slot_bytes(&[(id, src)], &seed)
            .expect("the source lands");
        shell
            .write_slot_bytes(&[(id, dst)], &vec![0xEE; slot_bytes])
            .expect("the destination lands");
    }
    assert_eq!(
        slot(&model, id, src),
        seed,
        "the source holds what was written"
    );

    // Two runs, neither on a slot boundary nor a slot long: src's tokens
    // 2.. onto dst's 1.., then src's first two after them.
    let long = tokens - 3;
    runs(
        &mut model,
        id,
        Vec::new(),
        vec![
            SlotRun {
                src,
                src_token: 2,
                dst,
                dst_token: 1,
                tokens: long,
            },
            SlotRun {
                src,
                src_token: 0,
                dst,
                dst_token: 1 + long,
                tokens: 2,
            },
        ],
    )
    .expect("the rs list's token runs copy");
    let landed = slot(&model, id, dst);
    let mut want = vec![0xEE; slot_bytes];
    for &(offset, row) in &planes {
        let (offset, row) = (offset as usize, row as usize);
        for t in 0..tokens as usize {
            let from = match t as u32 {
                t if (1..1 + long).contains(&t) => Some(t - 1 + 2),
                t if (1 + long..3 + long).contains(&t) => Some(t - 1 - long),
                _ => None,
            };
            if let Some(s) = from {
                let s = s as usize;
                want[offset + t * row..offset + (t + 1) * row]
                    .copy_from_slice(&seed[offset + s * row..offset + (s + 1) * row]);
            }
        }
    }
    for (k, &(offset, row)) in planes.iter().enumerate() {
        let (offset, row) = (offset as usize, row as usize);
        for t in 0..tokens as usize {
            let at = offset + t * row..offset + (t + 1) * row;
            assert!(
                landed[at.clone()] == want[at],
                "rs plane {k}, token {t} of the destination slot"
            );
        }
    }
    assert!(
        landed == want,
        "bytes no plane's run names keep what they held"
    );

    // Refused by name, nothing copied: a run reading the rows it writes,
    // and one past its slot.
    let refused = runs(
        &mut model,
        id,
        Vec::new(),
        vec![SlotRun {
            src,
            src_token: 0,
            dst: src,
            dst_token: 1,
            tokens: 2,
        }],
    )
    .expect_err("an overlapping run is refused");
    assert!(
        refused.contains("reads the rows it writes"),
        "refused as: {refused}"
    );
    let refused = runs(
        &mut model,
        id,
        Vec::new(),
        vec![SlotRun {
            src,
            src_token: 0,
            dst,
            dst_token: tokens - 1,
            tokens: 2,
        }],
    )
    .expect_err("a run past its slot is refused");
    assert!(
        refused.contains("rows of one list slot"),
        "refused as: {refused}"
    );
    assert_eq!(slot(&model, id, src), seed, "a refused run copies nothing");
    assert!(
        slot(&model, id, dst) == landed,
        "a refused run copies nothing"
    );

    // Whole slots still copy whole, beside no run.
    runs(&mut model, id, vec![(src, whole)], Vec::new()).expect("a whole rs list slot copies");
    assert_eq!(slot(&model, id, whole), seed);
}
