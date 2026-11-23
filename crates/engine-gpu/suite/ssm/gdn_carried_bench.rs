//! The carried decode step's cost beside the plain one, on the device's
//! clock: Qwen3.5-0.8B's gated-delta shape (16 key heads, 16 value heads,
//! 128-wide), one and sixteen lanes, each carrying `n` buffered rows and
//! committing one of them, as a run-ahead decode does. A probe run by hand:
//! `-- --ignored gdn_carried_costs --nocapture`.

use kernels_slang::Tensor;
use kernels_slang::ssm::gdn_recurrent::{Gated, Tactic};
use kernels_slang::ssm::{CARRY_HEADER, Carry, CarryPlane, RowSpan, StatePool};
use poem_ir::Dtype;

use crate::gpu::{Gpu, noise};

const KH: u32 = 16;
const VH: u32 = 16;
const D: u32 = 128;
const PAGE: u32 = 16;
const REPS: u32 = 200;

#[test]
#[ignore = "a probe run by hand"]
fn gdn_carried_costs() {
    let Some(mut gpu) = Gpu::open() else {
        return;
    };
    let t = |buf: u32, r: u32, w: u32, d: Dtype| Tensor::new(buf, r, w, d);
    let slab = VH * D * D;
    let pitch = 2 * KH * D + VH * D;
    // A page's planes: [b | a] (VH words), keys (KH * D / 2), u (VH * D).
    let (ba_row, k_row, u_row) = (VH, KH * D / 2, VH * D);
    let planes = [
        CarryPlane { at: 0, row: ba_row },
        CarryPlane { at: PAGE * ba_row, row: k_row },
        CarryPlane { at: PAGE * (ba_row + k_row), row: u_row },
    ];
    for &tier in crate::gpu::tiers() {
        let tactic = Tactic {
            tier,
            register_min_rows: 0,
        };
        for lanes in [1u32, 16] {
            // A slab a lane's state, then a slab a lane's two rs pages.
            let rows = 3 * lanes;
            let bank = gpu.up_f32(&noise(3, (rows * slab) as usize).iter().map(|v| v * 0.01).collect::<Vec<_>>());
            let pool = StatePool::bank(
                t(bank, rows, slab, Dtype::F32),
                t(gpu.up_u32(&(0..lanes).collect::<Vec<_>>()), lanes, 1, Dtype::U32),
            );
            let (qkv, _) = gpu.up(&noise(5, (lanes * pitch) as usize));
            let qkv = t(qkv, lanes, pitch, Dtype::Bf16);
            let (ba, _) = gpu.up(&noise(6, (lanes * 2 * VH) as usize));
            let (dt, _) = gpu.up(&noise(7, VH as usize));
            let a_log = gpu.up_f32(&vec![-1.0; VH as usize]);
            let gates = gpu.up_f32(&vec![-0.1; (lanes * 2 * VH) as usize]);
            let y = t(gpu.up_f32(&vec![0.0; (lanes * VH * D) as usize]), lanes, VH * D, Dtype::F32);
            let slots = t(
                gpu.up_i32(&(0..lanes as i32).collect::<Vec<_>>()),
                lanes,
                1,
                Dtype::I32,
            );
            let gated = Gated {
                ba: t(ba, lanes, 2 * VH, Dtype::Bf16),
                dt_bias: t(dt, VH, 1, Dtype::Bf16),
                a_log: t(a_log, VH, 1, Dtype::F32),
            };
            let span = RowSpan { row0: 0, rows: lanes };
            let plain = gpu.timed(|c| {
                for _ in 0..REPS {
                    kernels_slang::ssm::gdn_recurrent::gated_delta(
                        c,
                        tactic,
                        qkv,
                        t(gates, lanes, 2 * VH, Dtype::F32),
                        &pool,
                        KH,
                        VH,
                        D,
                        D,
                        y,
                        span,
                    )?;
                }
                Ok(())
            }) / f64::from(REPS);
            let mut line = format!("{tier:?} {lanes} lane(s): plain {:.2} us", plain / 1e3);
            for (n, commit) in [(0u32, 1i32), (1, 1), (2, 1), (4, 1), (10, 1), (10, 0), (10, 11)] {
                let header = CARRY_HEADER as usize;
                let mut table = vec![0i32; lanes as usize * header];
                for lane in 0..lanes as usize {
                    let at = table.len() as i32;
                    table.extend((0..n as i32).collect::<Vec<_>>());
                    let pages = table.len() as i32;
                    table.extend([(lanes as usize + 2 * lane) as i32, (lanes as usize + 2 * lane + 1) as i32]);
                    table[lane * header..][..header]
                        .copy_from_slice(&[n as i32, n as i32, at, pages, commit]);
                }
                let carry = Carry {
                    table: t(gpu.up_i32(&table), table.len() as u32, 1, Dtype::I32),
                    slots,
                    lane0: 0,
                    page_tokens: PAGE,
                    stride_words: slab,
                };
                let carried = gpu.timed(|c| {
                    for _ in 0..REPS {
                        kernels_slang::ssm::gdn_recurrent::gated_delta_carried(
                            c, tactic, qkv, gated, &pool, &carry, planes, KH, VH, D, D, y, span,
                        )?;
                    }
                    Ok(())
                }) / f64::from(REPS);
                line += &format!(", n={n}/c{commit} {:.2} us", carried / 1e3);
            }
            println!("{line}");
        }
    }
}
