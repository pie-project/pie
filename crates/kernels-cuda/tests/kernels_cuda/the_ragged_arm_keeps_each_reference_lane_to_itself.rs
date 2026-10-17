#![cfg(feature = "cuda")]

use crate::common::host::{assert_near, ragged_attention};
use crate::common::{Gpu, Lcg};
use dtype::Dtype;
use kernels_cuda::attn_ragged::{self, RaggedMask};
use kernels_cuda::tensor::Tensor;

const TOLERANCE: f32 = 1.0e-2;

struct Group {
    plain: u32,
    references: Vec<u32>,
}

fn tables(groups: &[Group]) -> (Vec<i32>, Vec<i32>) {
    let mut indptr = vec![0i32];
    let mut tags = Vec::new();
    let mut lane = 0i32;
    for group in groups {
        tags.extend(std::iter::repeat_n(-1, group.plain as usize));
        lane += 1;
        for &rows in &group.references {
            tags.extend(std::iter::repeat_n(lane, rows as usize));
            lane += 1;
        }
        indptr.push(tags.len() as i32);
    }
    (indptr, tags)
}

fn allowed(tags: &[i32], qo: usize, kv: usize) -> bool {
    tags[qo] < 0 || tags[kv] == tags[qo]
}

fn check(hd: u32, q_heads: u32, kv_heads: u32, seed: u64, edge: bool) {
    let groups = [
        Group {
            plain: 7,
            references: Vec::new(),
        },
        Group {
            plain: 5,
            references: vec![6],
        },
        Group {
            plain: 9,
            references: vec![4, 3, 5],
        },
    ];
    let (table, tags) = tables(&groups);
    let rows = *table.last().unwrap() as usize;
    let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
    let mut lcg = Lcg::seeded(seed);
    let (q_raw, q_f) = lcg.row(rows * qw);
    let (k_raw, k_f) = lcg.row(rows * kw);
    let (v_raw, v_f) = lcg.row(rows * kw);
    let sm_scale = 1.0 / (hd as f32).sqrt();
    let heads = [q_heads as usize, kv_heads as usize, hd as usize];
    let want = ragged_attention(
        [&q_f, &k_f, &v_f],
        &table,
        &table,
        heads,
        sm_scale,
        |_, _, r, j| allowed(&tags, r, j).then_some(0.0),
    );

    let mut gpu = Gpu::open();
    let q_at = gpu.up(&q_raw);
    let k_at = gpu.up(&k_raw);
    let v_at = gpu.up(&v_raw);
    let table_at = gpu.up(&table);
    let tags_at = if edge {
        gpu.up_at_edge(&tags)
    } else {
        gpu.up(&tags)
    };
    let o_at = gpu.zeros(rows * qw * 2);
    let ctx = gpu.ctx();
    let csr = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
    let tag_table = Tensor::new(tags_at, rows as u32, 1, Dtype::I32);
    attn_ragged::ragged(
        &ctx,
        Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16),
        Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16),
        Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16),
        csr,
        csr,
        hd,
        sm_scale,
        RaggedMask::ReferenceTags {
            q_tags: tag_table,
            kv_tags: tag_table,
        },
        &mut Tensor::new(o_at, rows as u32, qw as u32, Dtype::Bf16),
    )
    .expect("the tag-masked arm fires");
    gpu.sync();
    let got: Vec<u16> = gpu.down(o_at, rows * qw);
    assert_near(
        &got,
        &want,
        TOLERANCE,
        &format!("head width {hd}, edge {edge}"),
    );
}

/// With `edge`, the tag tables end where the device's mapped memory ends,
/// so a read of a tag past the last row (a query tile's padding rows) faults
/// rather than landing on a neighbour's bytes, as it once did under a
/// parallel suite.
#[test]
fn the_ragged_arm_keeps_each_reference_lane_to_itself_every_case() {
    for edge in [false, true] {
        for (hd, q_heads, kv_heads, seed) in
            [(64, 4, 2, 0x71), (128, 2, 2, 0x72), (256, 2, 1, 0x73)]
        {
            check(hd, q_heads, kv_heads, seed, edge);
        }
    }
    a_short_tag_table_is_refused();
    the_class_tables_are_read_no_further_than_the_rows_they_class();
}

fn a_short_tag_table_is_refused() {
    let mut gpu = Gpu::open();
    let q_at = gpu.zeros(8 * 128);
    let table_at = gpu.up(&[0i32, 8]);
    let one = gpu.up(&[-1i32]);
    let ctx = gpu.ctx();
    let plane = Tensor::new(q_at, 8, 64, Dtype::Bf16);
    let csr = Tensor::new(table_at, 2, 1, Dtype::I32);
    let short = Tensor::new(one, 1, 1, Dtype::I32);
    let refused = attn_ragged::ragged(
        &ctx,
        plane,
        plane,
        plane,
        csr,
        csr,
        64,
        0.125,
        RaggedMask::ReferenceTags {
            q_tags: short,
            kv_tags: short,
        },
        &mut Tensor::new(q_at, 8, 64, Dtype::Bf16),
    )
    .expect_err("a one-entry tag table over eight rows is refused");
    assert!(
        refused.to_string().contains("tag table"),
        "the refusal names the table: {refused}"
    );
}

/// The class arm over class tables that end where mapped memory ends, every
/// pair of classes allowed: it lands what the unmasked arm lands, and reads
/// no class past the rows it classes.
fn the_class_tables_are_read_no_further_than_the_rows_they_class() {
    let table = [0i32, 7, 18, 39];
    let rows = 39usize;
    for (hd, q_heads, kv_heads) in [(64u32, 4u32, 2u32), (128, 2, 2), (256, 2, 1)] {
        let (qw, kw) = ((q_heads * hd) as usize, (kv_heads * hd) as usize);
        let mut lcg = Lcg::seeded(0x77 + u64::from(hd));
        let (q_raw, _) = lcg.row(rows * qw);
        let (k_raw, _) = lcg.row(rows * kw);
        let (v_raw, _) = lcg.row(rows * kw);
        let classes: Vec<i32> = (0..rows as i32).map(|r| r % 3).collect();
        let mut gpu = Gpu::open();
        let q_at = gpu.up(&q_raw);
        let k_at = gpu.up(&k_raw);
        let v_at = gpu.up(&v_raw);
        let table_at = gpu.up(&table);
        let classes_at = gpu.up_at_edge(&classes);
        let allowed_at = gpu.up_at_edge(&[1u8; 9]);
        let full_at = gpu.zeros(rows * qw * 2);
        let classed_at = gpu.zeros(rows * qw * 2);
        let ctx = gpu.ctx();
        let csr = Tensor::new(table_at, table.len() as u32, 1, Dtype::I32);
        let class_table = Tensor::new(classes_at, rows as u32, 1, Dtype::I32);
        for (mask, out) in [
            (RaggedMask::None, full_at),
            (
                RaggedMask::ClassTable {
                    q_classes: class_table,
                    kv_classes: class_table,
                    table: Tensor::new(allowed_at, 9, 1, Dtype::U8),
                    count: 3,
                },
                classed_at,
            ),
        ] {
            attn_ragged::ragged(
                &ctx,
                Tensor::new(q_at, rows as u32, qw as u32, Dtype::Bf16),
                Tensor::new(k_at, rows as u32, kw as u32, Dtype::Bf16),
                Tensor::new(v_at, rows as u32, kw as u32, Dtype::Bf16),
                csr,
                csr,
                hd,
                0.0,
                mask,
                &mut Tensor::new(out, rows as u32, qw as u32, Dtype::Bf16),
            )
            .expect("the arm fires");
        }
        gpu.sync();
        let full: Vec<u16> = gpu.down(full_at, rows * qw);
        let classed: Vec<u16> = gpu.down(classed_at, rows * qw);
        assert_eq!(
            full, classed,
            "every class allowed is no mask (head width {hd})"
        );
    }
}
