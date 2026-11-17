#![cfg(target_vendor = "apple")]

//! A draft head scatters the trunk's samples over a lane's rows at the
//! readout rows (`gather.scatter_rows`, Qwen3.5's MTP head). On CUDA a fire
//! padded its readout rows past the live ones with row 0 (to a recorded
//! body's ceiling), and every pad scattered a stale sample over row 0
//! (sc/l40s-mtp-9b 5ca184dc2). Metal records no body and pads nothing: a
//! fire's readout table and every readout-axis value are cut to its live
//! rows. This guards that: should a Metal fire ever pad its readouts, row 0
//! would take a pad's sample here.
//!
//! Each readout row's sample is the table row its own index names, so a
//! pad (row 0) would land table row 0 at row 0. The one readout row, 5,
//! reads row 0 back through the lane's tokens (token 5 is 0): row 0 must
//! still hold its own token's row, 3. The control reads row 0 out live:
//! its sample lands there and row 5 reads table row 0, so the test sees a
//! scatter at row 0 when one happens.

use engine_metal::serve::Seated;
use engine_metal::{Lane, Shell};
use poem::{Dtype, ForwardHybrid, Input, Platform, Request, StateRows, Weight, ops};

use super::a_scattered_pool_serves_the_same_logits::{fixture_from, load_with, seeded};

const VOCAB: u32 = 64;
const WIDTH: u64 = 128;

struct Placed;

impl ForwardHybrid for Placed {
    fn states(&self) -> StateRows {
        StateRows::new()
    }

    fn forward(&self, inputs: Input) {
        let table = Weight::sym("embed", [u64::from(VOCAB), WIDTH], Dtype::Bf16);
        let own = ops::gather::embed(&inputs.tokens(), &table, VOCAB);
        let sampled = ops::gather::embed(&inputs.readout_rows(), &table, VOCAB);
        let placed = ops::gather::scatter_rows(&sampled, &inputs.readout_rows(), &own);
        let by_token = ops::gather::gather_rows(&placed, &inputs.tokens());
        let read = ops::gather::gather_rows(&by_token, &inputs.readout_rows());
        poem::output("logits", &read, None);
    }
}

#[test]
fn a_padded_readout_scatters_nothing() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let trace = poem::trace_hybrid("padded-readout-micro", &Placed, Platform::Metal);
    let table: Vec<f32> = seeded(&trace)
        .into_iter()
        .find(|(name, ..)| name.ends_with("embed"))
        .map(|(.., bytes)| {
            bytes
                .chunks_exact(2)
                .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
                .collect()
        })
        .expect("the embedding table is seeded");
    let fixture = fixture_from(trace.clone(), "padded-readout-micro");
    let mut shell: Shell = load_with(&fixture, 256);
    let w = WIDTH as usize;
    let nearest = |row: &[f32]| -> usize {
        (0..VOCAB as usize)
            .min_by(|&a, &b| {
                let off = |id: usize| -> f32 {
                    table[id * w..][..w]
                        .iter()
                        .zip(row)
                        .map(|(t, g)| (t - g).abs())
                        .sum()
                };
                off(a).total_cmp(&off(b))
            })
            .expect("a vocabulary")
    };
    let tokens: Vec<u32> = vec![3, 1, 4, 2, 6, 0, 5];
    let read = [5u32];
    shell.open(0).expect("the slot opens");
    for fire in 0..2 {
        let got = shell
            .fire_seated(&[Seated {
                kv_less: true,
                readout: Some(read.as_slice()),
                ..Seated::of(Lane {
                    slot: 0,
                    word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                    tokens: &tokens,
                })
            }])
            .expect("the fire runs");
        let got = &got[0];
        assert_eq!(got.len(), w, "the one row read out");
        assert_eq!(
            nearest(got),
            3,
            "fire {fire}: row 0 holds its own token's row, 3; a padded readout's sample is \
             table row 0"
        );
    }

    // The control: a live readout at row 0 does scatter its sample there,
    // and row 5 then reads table row 0 back, as a pad would leave it.
    let both = [5u32, 0];
    let got = shell
        .fire_seated(&[Seated {
            kv_less: true,
            readout: Some(both.as_slice()),
            ..Seated::of(Lane {
                slot: 0,
                word: trace.facts.word(&Request::new(tokens.len() as u32, false)),
                tokens: &tokens,
            })
        }])
        .expect("the control fire runs");
    let row5 = &got[0][..w];
    assert_eq!(
        nearest(row5),
        0,
        "a sample scattered at row 0 is what row 5 reads"
    );
}
