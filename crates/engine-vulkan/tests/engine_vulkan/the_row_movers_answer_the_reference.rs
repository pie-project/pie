//! Row movement on the first Vulkan device, beside the hand-written kernels: gather,
//! scatter (live rows skipping a negative route), permutation packing, the
//! pool and merge folds, a layer's slice, the splits, and a flat word copy.
//! The movers are exact, so the reference is the moved values themselves.

use crate::gpu::{Gpu, both, noise};
use kernels_vulkan::{Tensor, layout};
use poem_ir::Dtype;

const W: usize = 48;

fn st(t: Tensor) -> kernels_slang::Tensor {
    kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype)
}

#[test]
fn the_row_movers_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    let rows = 6usize;
    let (xb, xr) = gpu.up(&noise(1, rows * W));
    let xv = noise(2, rows * W);
    let xf = gpu.up_f32(&xv);
    let xr: Vec<f64> = xr.iter().map(|v| f64::from(*v)).collect();
    let xf64: Vec<f64> = xv.iter().map(|v| f64::from(*v)).collect();
    let t = |buf: u32, r: usize, w: usize, d: Dtype| Tensor::new(buf, r as u32, w as u32, d);
    let pick = [4i32, 0, 5, 2];
    let idx = gpu.up_i32(&pick);
    let it = t(idx, pick.len(), 1, Dtype::I32);
    let k = pick.len();

    for (dtype, src, vals, f32) in [(Dtype::Bf16, xb, &xr, false), (Dtype::F32, xf, &xf64, true)] {
        let words = if f32 { 2 } else { 1 };
        // gather: tight[i] = wide[pick[i]]
        let want: Vec<f64> = pick
            .iter()
            .flat_map(|&r| vals[r as usize * W..][..W].to_vec())
            .collect();
        both(
            &mut gpu,
            &format!("gather rows {dtype:?}"),
            &|g| vec![(g.zeros(words * k * W), k * W, f32)],
            &[want],
            &|g, o| {
                g.run(|c| {
                    layout::gather_rows(c, t(src, rows, W, dtype), it, t(o[0].0, k, W, dtype))
                })
            },
            &|g, o| {
                g.run_slang(|c| {
                    kernels_slang::gather::rows::gather_rows(
                        c,
                        st(t(src, rows, W, dtype)),
                        st(it),
                        st(t(o[0].0, k, W, dtype)),
                    )
                })
            },
        );
        // scatter: wide[pick[i]] = tight[i], the other rows left as zeros
        let mut want = vec![0.0; rows * W];
        for (i, &r) in pick.iter().enumerate() {
            want[r as usize * W..][..W].copy_from_slice(&vals[i * W..][..W]);
        }
        both(
            &mut gpu,
            &format!("scatter rows {dtype:?}"),
            &|g| vec![(g.zeros(words * rows * W), rows * W, f32)],
            &[want],
            &|g, o| {
                g.run(|c| {
                    layout::scatter_rows(c, t(src, k, W, dtype), it, t(o[0].0, rows, W, dtype))
                })
            },
            &|g, o| {
                g.run_slang(|c| {
                    kernels_slang::gather::rows::scatter_rows(
                        c,
                        st(t(src, k, W, dtype)),
                        st(it),
                        st(t(o[0].0, rows, W, dtype)),
                    )
                })
            },
        );
    }

    // live scatter: a negative route drops its row
    let routes = [3i32, -1, 0];
    let rt = t(gpu.up_i32(&routes), 3, 1, Dtype::I32);
    let mut want = vec![0.0; rows * W];
    for (i, &r) in routes.iter().enumerate() {
        if r >= 0 {
            want[r as usize * W..][..W].copy_from_slice(&xr[i * W..][..W]);
        }
    }
    both(
        &mut gpu,
        "scatter live rows",
        &|g| vec![(g.zeros(rows * W), rows * W, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                layout::scatter_live_rows(
                    c,
                    t(xb, 3, W, Dtype::Bf16),
                    rt,
                    t(o[0].0, rows, W, Dtype::Bf16),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::gather::rows::scatter_live_rows(
                    c,
                    st(t(xb, 3, W, Dtype::Bf16)),
                    st(rt),
                    st(t(o[0].0, rows, W, Dtype::Bf16)),
                )
            })
        },
    );

    // pack, then unpack back
    let perm = [5i32, 3, 1, 0, 2, 4];
    let pt = t(gpu.up_i32(&perm), rows, 1, Dtype::I32);
    let packed_want: Vec<f64> = perm
        .iter()
        .flat_map(|&r| xr[r as usize * W..][..W].to_vec())
        .collect();
    let packed = gpu.zeros(rows * W);
    gpu.run_slang(|c| {
        kernels_slang::gather::rows::pack_rows(
            c,
            st(t(xb, rows, W, Dtype::Bf16)),
            st(pt),
            st(t(packed, rows, W, Dtype::Bf16)),
        )
    });
    crate::gpu::close(&gpu.down(packed, rows * W), &packed_want, "slang pack rows");
    let back = gpu.zeros(rows * W);
    gpu.run_slang(|c| {
        kernels_slang::gather::rows::unpack_rows(
            c,
            st(t(packed, rows, W, Dtype::Bf16)),
            st(pt),
            st(t(back, rows, W, Dtype::Bf16)),
        )
    });
    crate::gpu::close(&gpu.down(back, rows * W), &xr, "slang unpack rows");

    // pool and merge: 4 rows of a 2 x 2 fold (an 8-row source, 2 blocks)
    let (sb, sr) = gpu.up(&noise(3, 8 * W));
    let sr: Vec<f64> = sr.iter().map(|v| f64::from(*v)).collect();
    let pooled: Vec<f64> = (0..2 * W)
        .map(|i| {
            (0..4)
                .map(|r| sr[((i / W) * 4 + r) * W + i % W])
                .sum::<f64>()
                / 4.0
        })
        .collect();
    both(
        &mut gpu,
        "pool rows",
        &|g| vec![(g.zeros(2 * W), 2 * W, false)],
        &[pooled],
        &|g, o| {
            g.run(|c| {
                layout::pool_rows(c, t(sb, 8, W, Dtype::Bf16), 2, t(o[0].0, 2, W, Dtype::Bf16))
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::gather::rows::pool_rows(
                    c,
                    st(t(sb, 8, W, Dtype::Bf16)),
                    2,
                    st(t(o[0].0, 2, W, Dtype::Bf16)),
                )
            })
        },
    );
    both(
        &mut gpu,
        "merge rows",
        &|g| vec![(g.zeros(8 * W), 8 * W, false)],
        std::slice::from_ref(&sr),
        &|g, o| {
            g.run(|c| {
                layout::merge_rows(
                    c,
                    t(sb, 8, W, Dtype::Bf16),
                    2,
                    t(o[0].0, 2, 4 * W, Dtype::Bf16),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::gather::rows::merge_rows(
                    c,
                    st(t(sb, 8, W, Dtype::Bf16)),
                    2,
                    st(t(o[0].0, 2, 4 * W, Dtype::Bf16)),
                )
            })
        },
    );

    // select layer 1's 16-wide slice of each 48-wide row
    let want: Vec<f64> = (0..rows * 16)
        .map(|i| xr[(i / 16) * W + 16 + i % 16])
        .collect();
    both(
        &mut gpu,
        "select",
        &|g| vec![(g.zeros(rows * 16), rows * 16, false)],
        &[want],
        &|g, o| {
            g.run(|c| {
                layout::select(
                    c,
                    t(xb, rows, W, Dtype::Bf16),
                    1,
                    16,
                    t(o[0].0, rows, 16, Dtype::Bf16),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::gather::rows::select(
                    c,
                    st(t(xb, rows, W, Dtype::Bf16)),
                    1,
                    16,
                    st(t(o[0].0, rows, 16, Dtype::Bf16)),
                )
            })
        },
    );

    // split 48 into 20 + 28; into q 16, k 16, v 16; heads of 8 into q and gate
    let cut = |a: usize, w: usize| -> Vec<f64> {
        (0..rows * w).map(|i| xr[(i / w) * W + a + i % w]).collect()
    };
    both(
        &mut gpu,
        "split rows",
        &|g| {
            vec![
                (g.zeros(rows * 20), rows * 20, false),
                (g.zeros(rows * 28), rows * 28, false),
            ]
        },
        &[cut(0, 20), cut(20, 28)],
        &|g, o| {
            g.run(|c| {
                layout::split_rows(
                    c,
                    t(xb, rows, W, Dtype::Bf16),
                    20,
                    t(o[0].0, rows, 20, Dtype::Bf16),
                    t(o[1].0, rows, 28, Dtype::Bf16),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::gather::rows::split_rows(
                    c,
                    st(t(xb, rows, W, Dtype::Bf16)),
                    20,
                    st(t(o[0].0, rows, 20, Dtype::Bf16)),
                    st(t(o[1].0, rows, 28, Dtype::Bf16)),
                )
            })
        },
    );
    let three = |g: &mut Gpu| {
        (0..3)
            .map(|_| (g.zeros(rows * 16), rows * 16, false))
            .collect::<Vec<_>>()
    };
    both(
        &mut gpu,
        "split qkv",
        &three,
        &[cut(0, 16), cut(16, 16), cut(32, 16)],
        &|g, o| {
            let p = |i: usize| t(o[i].0, rows, 16, Dtype::Bf16);
            g.run(|c| layout::split_qkv(c, t(xb, rows, W, Dtype::Bf16), 16, 16, p(0), p(1), p(2)))
        },
        &|g, o| {
            let p = |i: usize| st(t(o[i].0, rows, 16, Dtype::Bf16));
            g.run_slang(|c| {
                kernels_slang::gather::rows::split_qkv(
                    c,
                    st(t(xb, rows, W, Dtype::Bf16)),
                    16,
                    16,
                    p(0),
                    p(1),
                    p(2),
                )
            })
        },
    );
    // three heads of 8 query + 8 gate columns
    let (q_want, g_want): (Vec<f64>, Vec<f64>) = (
        (0..rows * 24)
            .map(|i| xr[(i / 24) * W + ((i % 24) / 8) * 16 + i % 8])
            .collect(),
        (0..rows * 24)
            .map(|i| xr[(i / 24) * W + ((i % 24) / 8) * 16 + 8 + i % 8])
            .collect(),
    );
    both(
        &mut gpu,
        "split q gate",
        &|g| {
            vec![
                (g.zeros(rows * 24), rows * 24, false),
                (g.zeros(rows * 24), rows * 24, false),
            ]
        },
        &[q_want, g_want],
        &|g, o| {
            g.run(|c| {
                layout::split_q_gate(
                    c,
                    t(xb, rows, W, Dtype::Bf16),
                    8,
                    t(o[0].0, rows, 24, Dtype::Bf16),
                    t(o[1].0, rows, 24, Dtype::Bf16),
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::gather::rows::split_q_gate(
                    c,
                    st(t(xb, rows, W, Dtype::Bf16)),
                    8,
                    st(t(o[0].0, rows, 24, Dtype::Bf16)),
                    st(t(o[1].0, rows, 24, Dtype::Bf16)),
                )
            })
        },
    );

    // a flat copy of the f32 rows
    both(
        &mut gpu,
        "copy words",
        &|g| vec![(g.zeros(2 * rows * W), rows * W, true)],
        std::slice::from_ref(&xf64),
        &|g, o| {
            g.run(|c| {
                layout::copy_words(
                    c,
                    t(xf, 1, 1, Dtype::U32),
                    t(o[0].0, 1, 1, Dtype::U32),
                    (rows * W * 4) as u64,
                )
            })
        },
        &|g, o| {
            g.run_slang(|c| {
                kernels_slang::gather::rows::copy_words(
                    c,
                    st(t(xf, 1, 1, Dtype::U32)),
                    st(t(o[0].0, 1, 1, Dtype::U32)),
                    (rows * W * 4) as u64,
                )
            })
        },
    );
}
