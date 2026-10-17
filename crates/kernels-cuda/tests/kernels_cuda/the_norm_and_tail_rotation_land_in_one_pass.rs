#![cfg(feature = "cuda")]

//! DeepSeek-V4's norm-then-rotate pairs (`norm_rope_last`): q's scale-free
//! norm of 64 heads of 512, the kv plane's and compressed latent's scaled
//! norm of one 512-wide head, the index keys' of one 128-wide head, each
//! with its 64-wide tail turned forward or back, plain or YaRN-scaled. Under
//! `kernel.elemwise.rows_vec` the one pass lands what the two launches land
//! within a bf16 ulp of the norm, on the staged window's rows and nowhere
//! else; off, it is the two launches.

use crate::common::tc5::e4m3;
use crate::common::{Gpu, Lcg, from_bf16, to_bf16};
use dtype::Dtype;
use kernels_cuda::attn::dsv4;
use kernels_cuda::elemwise::rope::{self, Yarn};
use kernels_cuda::tensor::{KvPool, Tensor};
use kernels_cuda::tuning::Tuning;

const PAGE: usize = 16;

const YARN: Yarn = Yarn {
    factor: 16.0,
    beta_fast: 32.0,
    beta_slow: 1.0,
    original_max_position: 65536,
};

const WINDOWS: [Option<(u32, u32, u32)>; 2] = [None, Some((24, 17, 5))];

/// DeepSeek's fp8 pool of `cells` rows on `plane`, its pages in order.
fn ds_pool(gpu: &mut Gpu, plane: u64, cells: usize) -> KvPool {
    let pages_n = cells / PAGE;
    let page_ids: Vec<u32> = (0..pages_n as u32).collect();
    KvPool {
        keys: Tensor::new(plane, cells as u32, dsv4::DS_ROW, Dtype::U8),
        values: Tensor::new(plane, cells as u32, dsv4::DS_ROW, Dtype::U8),
        bf16_keys: Tensor::ABSENT,
        bf16_values: Tensor::ABSENT,
        key_scales: Tensor::ABSENT,
        value_scales: Tensor::ABSENT,
        page_indices: Tensor::new(gpu.up(&page_ids), pages_n as u32, 1, Dtype::U32),
        page_indptr: Tensor::new(gpu.up(&[0u32, pages_n as u32]), 2, 1, Dtype::U32),
        last_page_lens: Tensor::ABSENT,
        row_valid: Tensor::ABSENT,
        env_min: Tensor::ABSENT,
        env_max: Tensor::ABSENT,
        has_envelopes: false,
        page_size: PAGE as i32,
        seq_stride: i64::from(dsv4::DS_ROW),
        head_stride: i64::from(dsv4::DS_ROW),
        layout: 0,
        scheme_byte: 0,
        block_size: 0,
        max_pages_per_request: pages_n as i32,
        pages_in_batch: pages_n as i32,
    }
}

/// Each plane row's write page (`reversed`: the pages back to front) and
/// offset.
fn write_slots(gpu: &mut Gpu, planes: usize, rows: u32, reversed: bool) -> (Tensor, Tensor) {
    let pages_n = planes.next_multiple_of(PAGE) / PAGE;
    let page: Vec<u32> = (0..planes)
        .map(|r| {
            if reversed {
                (pages_n - 1 - r / PAGE) as u32
            } else {
                (r / PAGE) as u32
            }
        })
        .collect();
    let off: Vec<u32> = (0..planes).map(|r| (r % PAGE) as u32).collect();
    (
        Tensor::new(gpu.up(&page), rows, 1, Dtype::U32),
        Tensor::new(gpu.up(&off), rows, 1, Dtype::U32),
    )
}

/// A context with `rows_vec` set, the window armed where there is one.
fn ctx_of(gpu: &mut Gpu, vec: bool, window: Option<(u32, u32, u32)>) -> kernels_cuda::Ctx {
    let ctx = gpu.ctx().with_tuning(Tuning {
        rows_vec: vec,
        ..Tuning::DEFAULT
    });
    if let Some((_, live, base)) = window {
        ctx.arm_stage(gpu.up(&[live, base, 0u32, 0u32]));
    }
    ctx
}

struct Case {
    heads: u32,
    head_dim: u32,
    weighted: bool,
    inverse: bool,
    yarn: bool,
}

fn pass(vec: bool, c: &Case, window: Option<(u32, u32, u32)>) -> Vec<u16> {
    let (rows, _, base) = window.unwrap_or((37, 37, 0));
    let planes = (base + rows) as usize;
    let width = (c.heads * c.head_dim) as usize;
    let mut lcg = Lcg::seeded(0x40e + u64::from(c.head_dim) + u64::from(c.heads));
    let x: Vec<u16> = (0..planes * width)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w: Vec<u16> = (0..c.head_dim).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let positions: Vec<i32> = (0..planes as i32).map(|r| r * 997 + 5).collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let w_at = gpu.up(&w);
    let pos_at = gpu.up(&positions);
    let y_at = gpu.up(&vec![0x3f80u16; planes * width]);
    let ctx = ctx_of(&mut gpu, vec, window);
    rope::norm_rope_last(
        &ctx,
        Tensor::new(x_at, rows, width as u32, Dtype::Bf16),
        c.weighted
            .then(|| Tensor::new(w_at, c.head_dim, 1, Dtype::Bf16)),
        c.head_dim,
        1e-6,
        Tensor::new(pos_at, rows, 1, Dtype::I32),
        64,
        10000.0,
        true,
        c.inverse,
        c.yarn.then_some(YARN),
        &mut Tensor::new(y_at, rows, width as u32, Dtype::Bf16),
    )
    .expect("the norm and rotation fire");
    gpu.sync();
    gpu.down(y_at, planes * width)
}

#[test]
fn the_norm_and_tail_rotation_land_in_one_pass() {
    // q's 64 heads turned back under YaRN, the kv plane's weighted head,
    // the index keys' 128-wide one, and a 256-wide head.
    let cases = [
        Case {
            heads: 64,
            head_dim: 512,
            weighted: false,
            inverse: true,
            yarn: true,
        },
        Case {
            heads: 1,
            head_dim: 512,
            weighted: true,
            inverse: false,
            yarn: true,
        },
        Case {
            heads: 1,
            head_dim: 128,
            weighted: true,
            inverse: false,
            yarn: false,
        },
        Case {
            heads: 4,
            head_dim: 256,
            weighted: false,
            inverse: false,
            yarn: true,
        },
    ];
    for c in &cases {
        for window in WINDOWS {
            let old = pass(false, c, window);
            let new = pass(true, c, window);
            for (i, (&g, &w)) in new.iter().zip(&old).enumerate() {
                let (gf, wf) = (from_bf16(g), from_bf16(w));
                assert!(
                    (gf - wf).abs() <= wf.abs() / 64.0 + 2e-2,
                    "heads {} d {} weighted {} {window:?}: element {i} is {gf} against {wf}",
                    c.heads,
                    c.head_dim,
                    c.weighted
                );
            }
        }
    }
}

/// The kv plane normed, rotated and written as DeepSeek's fp8 rows: the
/// pool's rows as they read, `one` choosing the one pass (else the norm and
/// rotation, then the fp8 append).
fn kv_rows(one: bool, window: Option<(u32, u32, u32)>) -> Vec<f32> {
    let (rows, _, base) = window.unwrap_or((40, 40, 0));
    let planes = (base + rows) as usize;
    let cells = planes.next_multiple_of(PAGE);
    let row = dsv4::DS_ROW as usize;
    let data = dsv4::DS_DATA as usize;
    let nope = dsv4::DS_NOPE as usize;
    let mut lcg = Lcg::seeded(0x4a11);
    let x: Vec<u16> = (0..planes * 512)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w: Vec<u16> = (0..512).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let positions: Vec<i32> = (0..planes as i32).map(|r| r * 31 + 2).collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let w_at = gpu.up(&w);
    let pos_at = gpu.up(&positions);
    let y_at = gpu.zeros(planes * 512 * 2);
    let plane = gpu.zeros(cells * row);
    let pool = ds_pool(&mut gpu, plane, cells);
    let (wp, wo) = write_slots(&mut gpu, planes, rows, true);
    let ctx = ctx_of(&mut gpu, one, window);
    let xt = Tensor::new(x_at, rows, 512, Dtype::Bf16);
    let wt = Tensor::new(w_at, 512, 1, Dtype::Bf16);
    let pt = Tensor::new(pos_at, rows, 1, Dtype::I32);
    let mut yt = Tensor::new(y_at, rows, 512, Dtype::Bf16);
    let landed = rope::norm_rope_last_append(
        &ctx,
        xt,
        wt,
        512,
        1e-6,
        pt,
        64,
        10000.0,
        true,
        false,
        Some(YARN),
        &mut yt,
        Some(rope::Append {
            pool: &pool,
            write_page: wp,
            write_offset: wo,
        }),
    )
    .expect("the one pass fires");
    assert_eq!(landed, one, "the one pass lands where the profile says");
    if !landed {
        rope::norm_rope_last(
            &ctx,
            xt,
            Some(wt),
            512,
            1e-6,
            pt,
            64,
            10000.0,
            true,
            false,
            Some(YARN),
            &mut yt,
        )
        .expect("the norm and rotation fire");
        dsv4::append(&ctx, yt, &pool, wp, wo).expect("the fp8 append fires");
    }
    gpu.sync();
    let raw: Vec<u8> = gpu.down(plane, cells * row);
    let mut read = vec![0f32; cells * 512];
    for c in 0..cells {
        let base = c / PAGE * PAGE * row;
        let off = c % PAGE;
        let b = &raw[base + off * data..base + (off + 1) * data];
        let sc = &raw[base + PAGE * data + off * 8..base + PAGE * data + off * 8 + 8];
        for d in 0..512 {
            read[c * 512 + d] = if d < nope {
                e4m3(b[d]) * 2f32.powi(i32::from(sc[d / 64]) - 127)
            } else {
                let at = nope + (d - nope) * 2;
                from_bf16(u16::from_le_bytes([b[at], b[at + 1]]))
            };
        }
    }
    read
}

/// The one pass's fp8 rows are, byte for byte, what the separate fp8 write
/// (`dsv4::append`) lands from the same pass's bf16 rows.
fn check_append_bytes(window: Option<(u32, u32, u32)>) {
    let (rows, _, base) = window.unwrap_or((40, 40, 0));
    let planes = (base + rows) as usize;
    let cells = planes.next_multiple_of(PAGE);
    let row = dsv4::DS_ROW as usize;
    let mut lcg = Lcg::seeded(0xb17e5);
    let x: Vec<u16> = (0..planes * 512)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w: Vec<u16> = (0..512).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let positions: Vec<i32> = (0..planes as i32).map(|r| r * 13 + 7).collect();
    let mut gpu = Gpu::open();
    let x_at = gpu.up(&x);
    let w_at = gpu.up(&w);
    let pos_at = gpu.up(&positions);
    let y_at = gpu.zeros(planes * 512 * 2);
    let planes_at = [gpu.zeros(cells * row), gpu.zeros(cells * row)];
    let one = ds_pool(&mut gpu, planes_at[0], cells);
    let two = ds_pool(&mut gpu, planes_at[1], cells);
    let (wp, wo) = write_slots(&mut gpu, planes, rows, true);
    let ctx = ctx_of(&mut gpu, true, window);
    let mut yt = Tensor::new(y_at, rows, 512, Dtype::Bf16);
    let landed = rope::norm_rope_last_append(
        &ctx,
        Tensor::new(x_at, rows, 512, Dtype::Bf16),
        Tensor::new(w_at, 512, 1, Dtype::Bf16),
        512,
        1e-6,
        Tensor::new(pos_at, rows, 1, Dtype::I32),
        64,
        10000.0,
        true,
        false,
        None,
        &mut yt,
        Some(rope::Append {
            pool: &one,
            write_page: wp,
            write_offset: wo,
        }),
    )
    .expect("the one pass fires");
    assert!(landed, "the one pass lands");
    dsv4::append(&ctx, yt, &two, wp, wo).expect("the separate write fires");
    gpu.sync();
    let a: Vec<u8> = gpu.down(planes_at[0], cells * row);
    let b: Vec<u8> = gpu.down(planes_at[1], cells * row);
    let first = a.iter().zip(&b).position(|(x, y)| x != y);
    assert!(
        first.is_none(),
        "{window:?}: byte {first:?} differs (cell {:?})",
        first.map(|i| (i / (PAGE * row)) * PAGE)
    );
    assert!(a.iter().any(|&v| v != 0), "{window:?}: rows are written");
}

#[test]
fn the_kv_plane_lands_its_fp8_rows_in_one_pass() {
    for window in WINDOWS {
        let old = kv_rows(false, window);
        let new = kv_rows(true, window);
        let mut written = 0;
        for (i, (&g, &w)) in new.iter().zip(&old).enumerate() {
            assert!(
                (g - w).abs() <= w.abs() / 8.0 + 2e-2,
                "{window:?}: element {i} reads {g} against {w}"
            );
            written += usize::from(w != 0.0);
        }
        assert!(written > 0, "rows are written");
        check_append_bytes(window);
    }
}

/// q and the kv plane, `pair` choosing the one launch (else q's one pass,
/// then the plane's): q's rows, the plane's rows, and the fp8 pool's bytes.
fn pair_pass(pair: bool, window: Option<(u32, u32, u32)>) -> (Vec<u16>, Vec<u16>, Vec<u8>) {
    let (rows, _, base) = window.unwrap_or((40, 40, 0));
    let planes = (base + rows) as usize;
    let cells = planes.next_multiple_of(PAGE);
    let row = dsv4::DS_ROW as usize;
    let mut lcg = Lcg::seeded(0x9a12);
    let qx: Vec<u16> = (0..planes * 64 * 512)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let kx: Vec<u16> = (0..planes * 512)
        .map(|_| to_bf16(lcg.unit() * 3.0))
        .collect();
    let w: Vec<u16> = (0..512).map(|_| to_bf16(0.5 + lcg.unit())).collect();
    let positions: Vec<i32> = (0..planes as i32).map(|r| r * 29 + 1).collect();
    let mut gpu = Gpu::open();
    let qx_at = gpu.up(&qx);
    let kx_at = gpu.up(&kx);
    let w_at = gpu.up(&w);
    let pos_at = gpu.up(&positions);
    let qy_at = gpu.zeros(planes * 64 * 512 * 2);
    let ky_at = gpu.zeros(planes * 512 * 2);
    let plane = gpu.zeros(cells * row);
    let pool = ds_pool(&mut gpu, plane, cells);
    let (wp, wo) = write_slots(&mut gpu, planes, rows, false);
    let ctx = ctx_of(&mut gpu, true, window);
    let pt = Tensor::new(pos_at, rows, 1, Dtype::I32);
    let wt = Tensor::new(w_at, 512, 1, Dtype::Bf16);
    let append = || {
        Some(rope::Append {
            pool: &pool,
            write_page: wp,
            write_offset: wo,
        })
    };
    if pair {
        let landed = rope::norm_rope_pair(
            &ctx,
            rope::Side {
                x: Tensor::new(qx_at, rows, 64 * 512, Dtype::Bf16),
                head_dim: 512,
                eps: 1e-6,
                y: &mut Tensor::new(qy_at, rows, 64 * 512, Dtype::Bf16),
            },
            rope::Side {
                x: Tensor::new(kx_at, rows, 512, Dtype::Bf16),
                head_dim: 512,
                eps: 1e-5,
                y: &mut Tensor::new(ky_at, rows, 512, Dtype::Bf16),
            },
            wt,
            pt,
            64,
            10000.0,
            Some(YARN),
            append(),
        )
        .expect("the pair fires");
        assert!(landed, "the pair lands in one launch");
    } else {
        rope::norm_rope_last(
            &ctx,
            Tensor::new(qx_at, rows, 64 * 512, Dtype::Bf16),
            None,
            512,
            1e-6,
            pt,
            64,
            10000.0,
            true,
            false,
            Some(YARN),
            &mut Tensor::new(qy_at, rows, 64 * 512, Dtype::Bf16),
        )
        .expect("q fires");
        let landed = rope::norm_rope_last_append(
            &ctx,
            Tensor::new(kx_at, rows, 512, Dtype::Bf16),
            wt,
            512,
            1e-5,
            pt,
            64,
            10000.0,
            true,
            false,
            Some(YARN),
            &mut Tensor::new(ky_at, rows, 512, Dtype::Bf16),
            append(),
        )
        .expect("the plane fires");
        assert!(landed, "the plane lands in one pass");
    }
    gpu.sync();
    (
        gpu.down(qy_at, planes * 64 * 512),
        gpu.down(ky_at, planes * 512),
        gpu.down(plane, cells * row),
    )
}

#[test]
fn q_and_the_kv_plane_land_in_one_launch() {
    for window in WINDOWS {
        let one = pair_pass(true, window);
        let two = pair_pass(false, window);
        assert!(one.0 == two.0, "{window:?}: q's rows differ");
        assert!(one.1 == two.1, "{window:?}: the plane's rows differ");
        assert!(one.2 == two.2, "{window:?}: the fp8 pool's bytes differ");
        assert!(
            one.1.iter().any(|&b| b != 0),
            "{window:?}: the plane is written"
        );
    }
}
