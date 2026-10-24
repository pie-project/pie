//! The fused gated-delta decode step (`ssm::gdn_step_fused`) against its
//! primitives on the device (split, conv update, gate prep, recurrence):
//! bit for bit on every output and both states, windowed and masked, and
//! the same again when replayed (its tickets left at zero).

#![cfg(feature = "rocm")]

mod common;

use common::{Gpu, STATE, bf16s, values};
use dtype::Dtype;
use kernels_rocm::gather;
use kernels_rocm::ssm;
use kernels_rocm::{RecurrentPool, Tensor};

struct Geo {
    k_heads: u32,
    v_heads: u32,
    conv: u32,
}

const D: u32 = 128;

struct Case {
    qkvz: Vec<f32>,
    conv_w: Vec<f32>,
    ba: Vec<f32>,
    dt_bias: Vec<f32>,
    a_log: Vec<f32>,
    slab: Vec<f32>,
    conv_slab: Vec<f32>,
}

#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn run(
    gpu: &Gpu,
    g: &Geo,
    case: &Case,
    stated: u32,
    win: Option<[i32; 4]>,
    slots: &[i32],
    mask: Option<&[u8]>,
    fused: bool,
) -> Vec<Vec<f32>> {
    let c = 2 * g.k_heads * D + g.v_heads * D;
    let zw = g.v_heads * D;
    let se = g.v_heads * D * D;
    let n_slots = (case.slab.len() / se as usize) as u32;
    let slab = gpu.tensor(&case.slab, n_slots, se, Dtype::Bf16);
    let conv_slab = gpu.tensor(&case.conv_slab, n_slots, g.conv * c, Dtype::Bf16);
    let mask_t = mask.map(|mask| {
        let mut words = vec![0i32; mask.len().div_ceil(4)];
        for (i, &b) in mask.iter().enumerate() {
            words[i / 4] |= i32::from(b) << (8 * (i % 4));
        }
        let m = gpu.ints(&words);
        Tensor::new(m.ptr, mask.len() as u32, 1, Dtype::U8)
    });
    let slot_ids = gpu.column(slots);
    let pool = |conv: bool| RecurrentPool {
        slab: if conv { Tensor::ABSENT } else { slab },
        slot_ids,
        slot_stride_elems: i64::from(se),
        conv_slab: if conv { conv_slab } else { Tensor::ABSENT },
        conv_stride: i64::from(g.conv * c),
        write_state: mask_t.is_none(),
        write_state_mask: mask_t.unwrap_or(Tensor::ABSENT),
        ..STATE
    };
    let (conv_pool, delta_pool) = (pool(true), pool(false));
    let qkvz = gpu.tensor(&case.qkvz, stated, c + zw, Dtype::Bf16);
    let conv_w = gpu.tensor(&case.conv_w, c, g.conv, Dtype::Bf16);
    let ba = gpu.tensor(&case.ba, stated, 2 * g.v_heads, Dtype::Bf16);
    let dt_bias = gpu.tensor(&case.dt_bias, 1, g.v_heads, Dtype::Bf16);
    let a_log = gpu.tensor(&case.a_log, 1, g.v_heads, Dtype::F32);
    let mut qkv = gpu.zeros(stated, c, Dtype::Bf16);
    let mut z = gpu.zeros(stated, zw, Dtype::Bf16);
    let mut y = gpu.zeros(stated, zw, Dtype::F32);
    let ctx = gpu.ctx();
    let win_t = win.map(|w| gpu.ints(&w));
    if let Some(w) = win_t {
        ctx.arm_stage(w.ptr);
    }
    if fused {
        assert!(
            ssm::gdn_step_fused(
                &ctx,
                qkvz,
                conv_w,
                &conv_pool,
                g.conv,
                1,
                ba,
                dt_bias,
                a_log,
                &delta_pool,
                g.k_heads,
                g.v_heads,
                D,
                D,
                &mut qkv,
                &mut z,
                &mut y,
            )
            .unwrap(),
            "the fused step covers this shape"
        );
    } else {
        let mut conv = gpu.zeros(stated, c, Dtype::Bf16);
        let mut gates = gpu.zeros(stated, 2 * g.v_heads, Dtype::F32);
        gather::split_rows(&ctx, qkvz, c, &mut qkv, &mut z).unwrap();
        ssm::causal_conv1d(&ctx, qkv, conv_w, None, &conv_pool, g.conv, 1, &mut conv).unwrap();
        ssm::gdn_gates(&ctx, ba, dt_bias, a_log, &mut gates).unwrap();
        ssm::gated_delta(
            &ctx,
            conv,
            z,
            gates,
            &delta_pool,
            g.k_heads,
            g.v_heads,
            D,
            D,
            &mut y,
        )
        .unwrap();
    }
    vec![
        gpu.read(&qkv),
        gpu.read(&z),
        gpu.read(&y),
        gpu.read(&slab),
        gpu.read(&conv_slab),
    ]
}

fn same(what: &str, have: &[Vec<f32>], want: &[Vec<f32>], live: &dyn Fn(usize, usize) -> bool) {
    let names = ["qkv", "z", "y", "state", "conv state"];
    for (i, (h, w)) in have.iter().zip(want).enumerate() {
        assert_eq!(h.len(), w.len());
        for (e, (a, b)) in h.iter().zip(w).enumerate() {
            if !live(i, e) {
                continue;
            }
            assert!(
                a.to_bits() == b.to_bits(),
                "{what}: {} element {e}: fused {a} vs primitives {b}",
                names[i]
            );
        }
    }
}

#[test]
fn the_fused_step_lands_what_its_primitives_land() {
    let Some(gpu) = Gpu::open() else { return };
    for g in [
        Geo {
            k_heads: 16,
            v_heads: 48,
            conv: 4,
        },
        Geo {
            k_heads: 4,
            v_heads: 4,
            conv: 4,
        },
        Geo {
            k_heads: 2,
            v_heads: 8,
            conv: 3,
        },
    ] {
        let c = (2 * g.k_heads * D + g.v_heads * D) as usize;
        let zw = (g.v_heads * D) as usize;
        let se = zw * D as usize;
        let n_slots = 8usize;
        for (stated, win, slots, mask) in [
            (1u32, None, vec![3i32], None),
            (5, None, vec![0, 4, 2, 7, 1], None),
            // Seven stated rows, the live four rows 2..6 on lanes 1..5, one
            // lane unseated and one masked.
            (
                7,
                Some([4, 2, 4, 1]),
                vec![6, 0, -1, 4, 2, 5, 3],
                Some(vec![1u8, 1, 1, 0, 1, 1, 1]),
            ),
        ] {
            let case = Case {
                qkvz: bf16s(stated as usize * (c + zw), 41, 1.0),
                conv_w: bf16s(c * g.conv as usize, 42, 0.5),
                ba: bf16s(stated as usize * 2 * g.v_heads as usize, 43, 1.5),
                dt_bias: bf16s(g.v_heads as usize, 44, 0.5),
                a_log: values(g.v_heads as usize, 45, 0.5),
                slab: bf16s(n_slots * se, 46, 0.2),
                conv_slab: bf16s(n_slots * g.conv as usize * c, 47, 1.0),
            };
            let what = format!(
                "{}x{} heads, conv {}, {stated} rows, window {win:?}",
                g.k_heads, g.v_heads, g.conv
            );
            let want = run(&gpu, &g, &case, stated, win, &slots, mask.as_deref(), false);
            let have = run(&gpu, &g, &case, stated, win, &slots, mask.as_deref(), true);
            // Rows outside the window are left as found by both; the split's
            // copies of an unseated lane's row are the fused step's alone.
            let (rows0, rows_n, lane0) = win.map_or((0, stated as usize, 0), |w| {
                (w[1] as usize, w[0] as usize, w[3] as usize)
            });
            let seated = |row: usize| {
                row >= rows0 && row < rows0 + rows_n && slots[row - rows0 + lane0] >= 0
            };
            let live = |out: usize, e: usize| match out {
                0 => seated(e / c),
                1 | 2 => seated(e / zw),
                _ => true,
            };
            same(&what, &have, &want, &live);
            let again = run(&gpu, &g, &case, stated, win, &slots, mask.as_deref(), true);
            same(&format!("{what}, replayed"), &again, &have, &|_, _| true);
        }
    }
}
