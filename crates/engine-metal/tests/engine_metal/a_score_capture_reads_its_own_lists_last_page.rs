#![cfg(target_vendor = "apple")]

//! A score capture finds a lane's keys from its pool's page table and the
//! plan's last-page lengths together: `(pages - 1) * page_size +
//! last_page_len`. A plan reads the kv list's geometry, whose last page holds
//! `(n - 1) % T + 1` of a lane's `n` rows; a window list of `T_w < T`-token
//! slots holds `(n - 1) % T_w + 1` on its last. Over the window list's pages
//! its own length lands the host's masses and no key past the lane's rows
//! holds any; the kv list's length runs past them, which is why the engine
//! hands a capture its cache row's own table's lengths.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::attn::score;
use kernels_metal::{KvPool, PrefillPlan, RaggedTensor, Tensor};
use poem_ir::Dtype;

const HEAD_DIM: usize = 64;
const Q_HEADS: usize = 2;
const KV_HEADS: usize = 1;
const SLOT: usize = 16;
const KV_PAGE: usize = 32;
const POOL_PAGES: usize = 8;
const OBSERVE: usize = 8;
const KV_MAX: usize = 64;
const SCALE: f32 = 0.125;

fn unit(at: u64) -> f32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    ((x >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16(v: f32) -> f32 {
    let bits = v.to_bits();
    f32::from_bits(((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) << 16)
}

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    keep: std::cell::RefCell<Vec<Buffer>>,
}

impl Rig {
    fn upload(&self, bytes: &[u8]) -> u32 {
        let mut b = Buffer::zeroed(&self.device, (bytes.len() as u64).max(16)).expect("a plane");
        b.write(0, bytes).expect("write");
        let h = self.handles.bind(&b, 0, b.bytes()).expect("a handle");
        self.keep.borrow_mut().push(b);
        h
    }

    fn bf16s(&self, v: &[f32], rows: usize, width: usize) -> Tensor {
        let bytes: Vec<u8> = v
            .iter()
            .flat_map(|f| ((bf16(*f).to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        Tensor::new(self.upload(&bytes), rows as u32, width as u32, Dtype::Bf16)
    }

    fn ints(&self, v: &[i32], dtype: Dtype) -> Tensor {
        let bytes: Vec<u8> = v.iter().flat_map(|i| i.to_le_bytes()).collect();
        Tensor::new(self.upload(&bytes), v.len() as u32, 1, dtype)
    }

    fn bytes(&self, v: &[u8]) -> Tensor {
        Tensor::new(self.upload(v), v.len() as u32, 1, Dtype::U8)
    }
}

/// The lane's mass over its first `KV_MAX` keys, `[Q_HEADS * KV_MAX]`: each
/// of its last `OBSERVE` rows' causal softmax, averaged.
fn reference(q: &[f32], keys: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; Q_HEADS * KV_MAX];
    let rows = OBSERVE.min(n);
    for head in 0..Q_HEADS {
        let kv_head = head / (Q_HEADS / KV_HEADS);
        for w in 0..rows {
            let row = n - rows + w;
            let scores: Vec<f32> = (0..=row)
                .map(|j| {
                    (0..HEAD_DIM)
                        .map(|d| {
                            bf16(q[(row * Q_HEADS + head) * HEAD_DIM + d])
                                * bf16(keys[(j * KV_HEADS + kv_head) * HEAD_DIM + d])
                        })
                        .sum::<f32>()
                        * SCALE
                })
                .collect();
            let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = scores.iter().map(|s| (s - top).exp()).sum();
            for (j, s) in scores.iter().enumerate() {
                out[head * KV_MAX + j] += (s - top).exp() / total / rows as f32;
            }
        }
    }
    out
}

#[test]
fn a_score_capture_reads_its_own_lists_last_page() {
    let Ok(device) = Context::bind() else {
        eprintln!("skipping: no Metal device");
        return;
    };
    let rig = Rig {
        device,
        handles: Handles::new(),
        pipelines: Pipelines::new(),
        keep: std::cell::RefCell::new(Vec::new()),
    };
    let width = KV_HEADS * HEAD_DIM;
    // Every row of the pool holds a key, so a capture that reads past a
    // lane's rows finds mass there.
    let pool_keys: Vec<f32> = (0..POOL_PAGES * SLOT * width)
        .map(|at| unit(at as u64 + 7))
        .collect();
    let keys = rig.bf16s(&pool_keys, POOL_PAGES * SLOT, width);
    // The lane's window-list slots, scattered; the table runs on past them,
    // as a fire's tables run on into the next lane's.
    let slots = [5i32, 2, 7, 0, 3, 4];
    for n in [20usize, 50, 61] {
        let own = (n - 1) % SLOT + 1;
        let listed = (n - 1) % KV_PAGE + 1;
        assert_ne!(own, listed, "{n} rows end the two lists' last pages apart");
        let pages = n.div_ceil(SLOT);
        // The lane's keys in token order, as its slots hold them.
        let lane_keys: Vec<f32> = (0..n)
            .flat_map(|j| {
                let at = (slots[j / SLOT] as usize * SLOT + j % SLOT) * width;
                pool_keys[at..at + width].to_vec()
            })
            .collect();
        let q_host: Vec<f32> = (0..n * Q_HEADS * HEAD_DIM)
            .map(|at| unit(at as u64 + 1000 * n as u64))
            .collect();
        let want = reference(&q_host, &lane_keys, n);
        let q = RaggedTensor {
            data: rig.bf16s(&q_host, n, Q_HEADS * HEAD_DIM),
            indptr: rig.ints(&[0, n as i32], Dtype::I32),
        };
        let pool = KvPool {
            keys,
            values: keys,
            page_indices: rig.ints(&slots, Dtype::U32),
            page_indptr: rig.ints(&[0, pages as i32], Dtype::U32),
            page_size: SLOT as i32,
            seq_stride: width as u64,
            head_stride: HEAD_DIM as u64,
            q8: false,
            page_stride: [0; 2],
        }
        .paged_densely();
        let capture = |last_page_len: usize| -> Vec<f32> {
            let plan = PrefillPlan {
                positions: rig.ints(&(0..n as i32).collect::<Vec<_>>(), Dtype::I32),
                last_page_len: rig.ints(&[last_page_len as i32], Dtype::I32),
                request_of_token: rig.ints(&vec![0; n], Dtype::I32),
                mask: rig.bytes(&[0]),
                mask_enabled: rig.bytes(&vec![0; n]),
                mask_stride: 1,
            };
            let out = Buffer::zeroed(&rig.device, (Q_HEADS * KV_MAX * 4) as u64).expect("a plane");
            let handle = rig.handles.bind(&out, 0, out.bytes()).expect("a handle");
            let scores = Tensor::new(handle, 1, (Q_HEADS * KV_MAX) as u32, Dtype::F32);
            let frame = rig.device.frame().expect("a frame");
            let sink = Sink::new(&rig.device, &frame, &rig.pipelines, &rig.handles);
            score::capture(
                &sink,
                q,
                &plan,
                &pool,
                HEAD_DIM as u32,
                KV_HEADS as u32,
                SCALE,
                OBSERVE as u32,
                KV_MAX as u32,
                1,
                scores,
            )
            .expect("the capture encodes");
            frame.commit().expect("the commit");
            rig.handles
                .read(handle, (Q_HEADS * KV_MAX * 4) as u64)
                .expect("read")
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect()
        };
        let past = |got: &[f32]| -> f32 {
            got.chunks(KV_MAX)
                .flat_map(|head| head[n..].iter().copied())
                .fold(0.0, f32::max)
        };

        let got = capture(own);
        let worst = want
            .iter()
            .zip(&got)
            .map(|(w, g)| (w - g).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "{n} rows: its own last page ({own} keys): worst {worst:.2e} from the host, \
             mass past the lane {:.2e}",
            past(&got)
        );
        assert_eq!(past(&got), 0.0, "{n} rows: no key past the lane holds mass");
        assert!(
            worst < 2e-3,
            "{n} rows: the masses are the host's (worst {worst:.2e})"
        );

        let misread = capture(listed);
        eprintln!(
            "{n} rows: the kv list's last page ({listed} keys): mass past the lane {:.2e}",
            past(&misread)
        );
        assert!(
            past(&misread) > 0.0,
            "{n} rows: the kv list's length over the window list's pages reads past the lane"
        );
    }
}
