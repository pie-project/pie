#![cfg(target_vendor = "apple")]

//! The ternary qmv kernels (`ptq1_0_qmv`, `pq2_0_qmv`) compute each output row
//! independently of the activation-batch width `m`: the grid is one threadgroup
//! per (activation vector, 8 output rows), so the weight matrix is read once per
//! vector with NO cross-row reuse. It follows that row 0's logits must read back
//! bit-for-bit the same whether the launch carried 1 activation row or 2048. This
//! test fires each ternary codec across a batch ladder on model-representative
//! shapes and asserts that invariance — a real defect (e.g. a grid/stride bug
//! that bleeds another row's state into row 0's accumulator) breaks it.
//!
//! (This was once a us/row prefill-pricing bench; the timing scaffolding was
//! perf-only and has been stripped — only the row-0 invariance correctness check
//! survives as a real assertion.)
//!
//! Env knobs: PIE_TERN_SHAPES ("KxN,KxN"), PIE_TERN_ROWS ("1,128,512,2048").

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::encode::{Arg, Encode, Fire, Grid};
use kernels_metal::linear::quant;
use model_ir::Dtype;

const PTQ1_0_FILE: &str = "linear/quant_ptq1_0.metal";
const PQ2_0_FILE: &str = "linear/quant_pq2_0.metal";
const QMV_GROUP: [u32; 3] = [32, 2, 1];

const PTQ1_0_BYTES: u32 = 28;
const PQ2_0_BYTES: u32 = 34;
const BLOCK: u32 = 128;

fn noise(at: u64) -> u8 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 40) as u8
}

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

fn to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits((u32::from(hi) << 24) | (u32::from(lo) << 16))
}

fn list(name: &str, fallback: &str) -> Vec<u32> {
    std::env::var(name)
        .unwrap_or_else(|_| fallback.to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

fn shapes() -> Vec<(u32, u32)> {
    std::env::var("PIE_TERN_SHAPES")
        .unwrap_or_else(|_| "5120x17408,17408x5120,5120x6144,2048x5120".to_string())
        .split(',')
        .filter_map(|s| {
            let (k, n) = s.trim().split_once('x')?;
            Some((k.trim().parse().ok()?, n.trim().parse().ok()?))
        })
        .collect()
}

#[derive(Clone, Copy)]
enum Codec {
    Ptq1_0,
    Pq2_0,
}

impl Codec {
    fn file(self) -> &'static str {
        match self {
            Codec::Ptq1_0 => PTQ1_0_FILE,
            Codec::Pq2_0 => PQ2_0_FILE,
        }
    }
    fn entry(self) -> &'static str {
        match self {
            Codec::Ptq1_0 => "ptq1_0_qmv_bfloat16",
            Codec::Pq2_0 => "pq2_0_qmv_bfloat16",
        }
    }
    fn block_bytes(self) -> u32 {
        match self {
            Codec::Ptq1_0 => PTQ1_0_BYTES,
            Codec::Pq2_0 => PQ2_0_BYTES,
        }
    }
    fn dtype(self) -> Dtype {
        match self {
            Codec::Ptq1_0 => Dtype::Ptq1_0,
            Codec::Pq2_0 => Dtype::Pq2_0,
        }
    }
    fn label(self) -> &'static str {
        match self {
            Codec::Ptq1_0 => "PTQ1_0 (1.75bpw)",
            Codec::Pq2_0 => "PQ2_0 (2.125bpw)",
        }
    }
    /// Lay one row of `num_blocks` blocks: a sane fp16 scale per block (so the
    /// decoded weights and the dot are finite) plus random code bytes. We are
    /// checking invariance, not validating a codec, so any byte pattern decodes
    /// to a valid trit/code and the per-row result only has to be finite and
    /// stable across `m`.
    fn write_block(self, blk: &mut [u8], seed: u64) {
        let scale = bf16(0.012 + 0.001 * f32::from(noise(seed) % 8));
        for (at, byte) in blk.iter_mut().enumerate() {
            *byte = noise(seed ^ (at as u64).wrapping_mul(0x100));
        }
        match self {
            // Trailing fp16 scale at bytes 26-27.
            Codec::Ptq1_0 => blk[26..28].copy_from_slice(&scale),
            // Leading fp16 scale at bytes 0-1.
            Codec::Pq2_0 => blk[0..2].copy_from_slice(&scale),
        }
    }
}

#[test]
fn the_ternary_qmv_row_0_is_invariant_to_batch() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    eprintln!("device: {}", device.name());

    let rows = list("PIE_TERN_ROWS", "1,128,512,2048");
    assert!(
        !rows.is_empty(),
        "PIE_TERN_ROWS names at least one batch width"
    );
    for (k, n) in shapes() {
        assert!(
            k.is_multiple_of(BLOCK),
            "K={k} must be a whole number of 128-blocks"
        );
        eprintln!("\n================  K={k}  N={n}  ================");
        for codec in [Codec::Ptq1_0, Codec::Pq2_0] {
            check_row_0_invariance(&device, &handles, &pipelines, codec, k, n, &rows);
        }
    }
}

/// Fire `codec`'s qmv at each batch width in `rows` and assert row 0 reads back
/// identically every time — the invariance guaranteed by the per-row grid.
fn check_row_0_invariance(
    device: &Context,
    handles: &Handles,
    pipelines: &Pipelines,
    codec: Codec,
    k: u32,
    n: u32,
    rows: &[u32],
) {
    let num_blocks = k / BLOCK;
    let row_bytes = num_blocks * codec.block_bytes();
    let codes_bytes = u64::from(n) * u64::from(row_bytes);
    let widest = *rows.iter().max().expect("a row");

    let mut codes_b = Buffer::zeroed(device, codes_bytes).expect("codes");
    {
        let mut codes = vec![0u8; usize::try_from(codes_bytes).expect("codes fit")];
        for r in 0..n {
            for blk in 0..num_blocks {
                let at = (u64::from(r) * u64::from(num_blocks) + u64::from(blk))
                    * u64::from(codec.block_bytes());
                let at = usize::try_from(at).expect("offset fits");
                let end = at + codec.block_bytes() as usize;
                codec.write_block(&mut codes[at..end], u64::from(r) ^ (u64::from(blk) << 20));
            }
        }
        codes_b.write(0, &codes).expect("write codes");
    }
    let mut act_b = Buffer::zeroed(device, u64::from(widest) * u64::from(k) * 2).expect("act");
    {
        let mut act =
            vec![0u8; usize::try_from(u64::from(widest) * u64::from(k) * 2).expect("act fits")];
        for (at, pair) in act.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            pair.copy_from_slice(&bf16(0.02 * (f32::from(noise(at as u64) % 16) - 8.0)));
        }
        act_b.write(0, &act).expect("write act");
    }
    let out_b = Buffer::zeroed(device, u64::from(widest) * u64::from(n) * 2).expect("out");

    let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("a handle");
    let (hc, ha, ho) = (bind(&codes_b), bind(&act_b), bind(&out_b));

    let (ki, ni) = (
        i32::try_from(k).expect("k fits"),
        i32::try_from(n).expect("n fits"),
    );

    eprintln!("  {} qmv  ({row_bytes} B/row)", codec.label());

    let mut reference: Option<Vec<f32>> = None;
    for &m in rows {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(device, &frame, pipelines, handles);
        sink.fire(
            Fire::at(codec.file(), codec.entry()).apply(Grid::of(
                quant::qmv_grid("row0", i32::try_from(m).expect("m"), ni).expect("grid"),
                QMV_GROUP,
            )),
            &[
                Tensor::new(hc, n, k, codec.dtype()).arg(),
                Tensor::new(ha, m, k, Dtype::Bf16).arg(),
                Tensor::new(ho, m, n, Dtype::Bf16).arg_mut(),
                ki.arg(),
                ni.arg(),
            ],
        )
        .expect("the qmv launch");
        frame.commit().expect("the commit");

        // qmv computes each output row independently of m, so row 0 must read
        // back identically for every m (a real defect breaks it).
        let raw = handles.read(ho, u64::from(n) * 2).expect("read row 0");
        let got: Vec<f32> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|p| to_f32(p[0], p[1]))
            .collect();
        match &reference {
            None => reference = Some(got),
            Some(want) => {
                let mut worst = 0.0f64;
                for (a, b) in want.iter().zip(&got) {
                    let scale = f64::from(a.abs()).max(f64::from(b.abs())).max(1e-3);
                    worst = worst.max(f64::from((a - b).abs()) / scale);
                }
                assert!(
                    worst <= 1e-6,
                    "{} rows {m} answers row 0 differently from rows {}: worst relative {worst:.2e}",
                    codec.label(),
                    rows[0]
                );
                eprintln!(
                    "    rows {m:>4}: row 0 matches rows {} (worst {worst:.2e})",
                    rows[0]
                );
            }
        }
    }
}
