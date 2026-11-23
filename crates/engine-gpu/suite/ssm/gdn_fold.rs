//! A group's fold (`BEAM-PAGE-LISTS.md` §11.3): the gated delta rule's fold
//! twin (`ssm_gdn_carried_fold_*`) folds a set's `n` buffered rows into its
//! state exactly as the carried decode step does when it commits those `n`
//! rows ahead of its own, which it then neither folds nor lands: the state
//! after each is the same, bit for bit, and the twin reads no input and
//! writes no output (its `y` and the rs pages stay as they were).
//! Qwen3.5-0.8B's shape (16 key heads, 16 value heads, 128 wide), rows
//! across two pages and a chunk boundary.

use kernels_slang::Tensor;
use kernels_slang::ssm::gdn_recurrent::{Gated, Tactic};
use kernels_slang::ssm::{CARRY_HEADER, Carry, CarryPlane, RowSpan, StatePool};
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

const KH: u32 = 16;
const VH: u32 = 16;
const D: u32 = 128;
const PAGE: u32 = 16;

#[test]
fn a_groups_fold_is_the_carried_steps_commit() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let t = |buf: u32, r: u32, w: u32, d: Dtype| Tensor::new(buf, r, w, d);
    let slab = VH * D * D;
    let pitch = 2 * KH * D + VH * D;
    let (ba_row, k_row, u_row) = (VH, KH * D / 2, VH * D);
    let planes = [
        CarryPlane { at: 0, row: ba_row },
        CarryPlane { at: PAGE * ba_row, row: k_row },
        CarryPlane { at: PAGE * (ba_row + k_row), row: u_row },
    ];
    // The state's slab, then two rs pages; small values, a word apart.
    let bank_values: Vec<f32> = noise(3, (3 * slab) as usize).iter().map(|v| v * 0.01).collect();
    let (qkv, _) = gpu.up(&noise(5, pitch as usize));
    let (ba, _) = gpu.up(&noise(6, (2 * VH) as usize));
    let (dt, _) = gpu.up(&noise(7, VH as usize));
    let a_log = gpu.up_f32(&vec![-1.0; VH as usize]);
    let gated = Gated {
        ba: t(ba, 1, 2 * VH, Dtype::Bf16),
        dt_bias: t(dt, VH, 1, Dtype::Bf16),
        a_log: t(a_log, VH, 1, Dtype::F32),
    };
    let slots = t(gpu.up_i32(&[0]), 1, 1, Dtype::I32);
    for &tier in crate::gpu::tiers() {
        let tactic = Tactic {
            tier,
            register_min_rows: 0,
        };
        for n in [1u32, 7, 20] {
            // The lane: no own row landed (-1), `n` carried rows over pages 1
            // and 2, all `n` committed.
            let header = CARRY_HEADER as usize;
            let mut table = vec![0i32; header];
            let tok = table.len() as i32;
            table.extend(0..n as i32);
            let pages = table.len() as i32;
            table.extend([1, 2]);
            table[..header].copy_from_slice(&[-1, n as i32, tok, pages, n as i32]);
            let mut state_after = Vec::new();
            for fold in [false, true] {
                let bank = gpu.up_f32(&bank_values);
                let pool = StatePool::bank(t(bank, 3, slab, Dtype::F32), slots);
                let y_values = vec![7.0f32; (VH * D) as usize];
                let y = gpu.up_f32(&y_values);
                let carry = Carry {
                    table: t(gpu.up_i32(&table), table.len() as u32, 1, Dtype::I32),
                    slots,
                    lane0: 0,
                    page_tokens: PAGE,
                    stride_words: slab,
                };
                let y_t = t(y, 1, VH * D, Dtype::F32);
                gpu.run(|c| {
                    if fold {
                        kernels_slang::ssm::gdn_recurrent::gated_delta_carried_fold(
                            c,
                            tactic,
                            t(qkv, 1, pitch, Dtype::Bf16),
                            gated,
                            &pool,
                            &carry,
                            planes,
                            KH,
                            VH,
                            D,
                            D,
                            y_t,
                            1,
                        )
                    } else {
                        kernels_slang::ssm::gdn_recurrent::gated_delta_carried(
                            c,
                            tactic,
                            t(qkv, 1, pitch, Dtype::Bf16),
                            gated,
                            &pool,
                            &carry,
                            planes,
                            KH,
                            VH,
                            D,
                            D,
                            y_t,
                            RowSpan { row0: 0, rows: 1 },
                        )
                    }
                });
                let got = gpu.down_u32(bank, (3 * slab) as usize);
                let want_pages: Vec<u32> =
                    bank_values[slab as usize..].iter().map(|v| v.to_bits()).collect();
                assert_eq!(&got[slab as usize..], want_pages.as_slice(), "{tier:?} n={n} fold={fold}: the rs pages stay");
                if fold {
                    assert!(
                        gpu.down_f32(y, (VH * D) as usize).iter().all(|&v| v == 7.0),
                        "{tier:?} n={n}: the fold writes no output"
                    );
                }
                state_after.push(got[..slab as usize].to_vec());
            }
            assert_ne!(
                state_after[0],
                bank_values[..slab as usize].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "{tier:?} n={n}: the commit moved the state"
            );
            assert_eq!(state_after[0], state_after[1], "{tier:?} n={n}: the fold is the commit, bit for bit");
            eprintln!("{tier:?} n={n}: the fold folds as the carried step commits");
        }
    }
}
