//! The collectives on the first Vulkan device, beside the hand-written
//! kernels, against a host reference: one rank of three, the other ranks'
//! slots of the band filled beforehand, bf16 and f32, odd counts where the
//! collective takes them. The rendezvous is the frame's own ordering.

use engine_vulkan::{Sink, Slang};
use kernels_vulkan::{ArgValue, Encode, Error, Fire, Tensor, collective};
use poem_ir::Dtype;

use crate::gpu::{Gpu, close, noise};

const WORLD: u32 = 3;
const RANK: u32 = 1;
const SLOT_BYTES: u32 = 8192 * 4;

/// A sink that is rank `RANK` of a `WORLD`-rank communicator over `band`.
struct Ranked<'a> {
    sink: &'a Sink<'a>,
    band: u32,
}

impl Encode for Ranked<'_> {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        self.sink.fire(fire, args)
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        self.sink.absent()
    }

    fn comm(&self, _op: &'static str) -> Result<kernels_vulkan::Comm, Error> {
        Ok(kernels_vulkan::Comm {
            rank: RANK,
            world: WORLD,
            band: Tensor::new(self.band, 1, 0, Dtype::Bf16),
            slot_bytes: SLOT_BYTES,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    Reduce,
    Gather,
    Scatter,
}

fn width(dtype: Dtype) -> usize {
    if dtype == Dtype::F32 { 4 } else { 2 }
}

/// Values up as `dtype`, and the values the device holds.
fn up(gpu: &mut Gpu, dtype: Dtype, v: &[f32]) -> (u32, Vec<f32>) {
    if dtype == Dtype::F32 {
        (gpu.up_f32(v), v.to_vec())
    } else {
        gpu.up(v)
    }
}

fn down(gpu: &Gpu, dtype: Dtype, h: u32, n: usize) -> Vec<f32> {
    if dtype == Dtype::F32 {
        gpu.down_f32(h, n)
    } else {
        gpu.down(h, n)
    }
}

fn case(gpu: &mut Gpu, op: Op, dtype: Dtype, n: usize, slang: bool) {
    let slot = SLOT_BYTES as usize / width(dtype);
    let (band, held) = up(gpu, dtype, &noise(7 + n as u64, slot * WORLD as usize));
    let (x, xs) = up(gpu, dtype, &noise(n as u64, n));
    // The band once this rank has published.
    let mut seen = held.clone();
    seen[RANK as usize * slot..][..n].copy_from_slice(&xs);
    let at = |r: usize, i: usize| f64::from(seen[r * slot + i]);
    let ranks = 0..WORLD as usize;
    let (y, out_n, want): (u32, usize, Vec<f64>) = match op {
        Op::Reduce => (
            x,
            n,
            (0..n)
                .map(|i| ranks.clone().map(|r| at(r, i)).sum())
                .collect(),
        ),
        Op::Gather => {
            let out_n = n * WORLD as usize;
            let (y, _) = up(gpu, dtype, &vec![0.0; out_n]);
            let want = (0..out_n).map(|e| at(e / n, e % n)).collect();
            (y, out_n, want)
        }
        Op::Scatter => {
            let shard = n / WORLD as usize;
            let (y, _) = up(gpu, dtype, &vec![0.0; shard]);
            let base = RANK as usize * shard;
            let want = (0..shard)
                .map(|i| ranks.clone().map(|r| at(r, base + i)).sum())
                .collect();
            (y, shard, want)
        }
    };
    let xt = Tensor::new(x, 1, n as u32, dtype);
    let yt = Tensor::new(y, 1, out_n as u32, dtype);
    gpu.run(|sink| {
        let ranked = Ranked { sink, band };
        if slang {
            let st = |t: Tensor| kernels_slang::Tensor::new(t.buf, t.rows, t.width, t.dtype);
            let c = &Slang(&ranked);
            match op {
                Op::Reduce => kernels_slang::collective::band::all_reduce(c, st(xt)),
                Op::Gather => kernels_slang::collective::band::all_gather(c, st(xt), st(yt)),
                Op::Scatter => kernels_slang::collective::band::reduce_scatter(c, st(xt), st(yt)),
            }
            .map_err(engine_vulkan::encode::from_slang)
        } else {
            match op {
                Op::Reduce => collective::all_reduce(&ranked, xt),
                Op::Gather => collective::all_gather(&ranked, xt, yt),
                Op::Scatter => collective::reduce_scatter(&ranked, xt, yt),
            }
        }
    });
    let who = if slang { "slang " } else { "" };
    let got = down(gpu, dtype, y, out_n);
    if matches!(op, Op::Gather) {
        let exact: Vec<f32> = want.iter().map(|w| *w as f32).collect();
        assert_eq!(got, exact, "{who}{op:?} {dtype:?} {n}");
        eprintln!("{who}{op:?} {dtype:?} {n}: exact");
    } else {
        close(&got, &want, &format!("{who}{op:?} {dtype:?} {n}"));
    }
    // The band's other slots are the other ranks': the collective leaves them.
    let after = down(gpu, dtype, band, slot * WORLD as usize);
    for r in ranks.filter(|&r| r != RANK as usize) {
        assert_eq!(
            after[r * slot..][..slot],
            held[r * slot..][..slot],
            "{who}{op:?} {dtype:?} {n}: rank {r}'s slot"
        );
    }
}

#[test]
fn the_collectives_answer_the_reference() {
    let Some(mut gpu) = Gpu::open() else {
        eprintln!("not asked: no device");
        return;
    };
    for dtype in [Dtype::Bf16, Dtype::F32] {
        for slang in [false, true] {
            for n in [6usize, 1001, 8192] {
                case(&mut gpu, Op::Reduce, dtype, n, slang);
            }
            // A bf16 shard of odd length would land its ranks mid-word.
            for n in [6usize, 1000, 8192] {
                case(&mut gpu, Op::Gather, dtype, n, slang);
            }
            for n in [6usize, 999, 8190] {
                case(&mut gpu, Op::Scatter, dtype, n, slang);
            }
        }
    }
}
