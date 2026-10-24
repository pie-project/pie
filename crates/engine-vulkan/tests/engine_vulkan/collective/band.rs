//! The collectives on the first Vulkan device against a host reference: one rank of three, the other ranks'
//! slots of the band filled beforehand, bf16 and f32, odd counts where the
//! collective takes them. The rendezvous is the frame's own ordering.

use engine_vulkan::Sink;
use kernels_slang::{ArgValue, Encode, Error, Fire, Target, Tensor};
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

    fn target(&self) -> Target {
        self.sink.target()
    }

    fn comm(&self, _op: &'static str) -> Result<kernels_slang::Comm, Error> {
        Ok(kernels_slang::Comm {
            rank: RANK,
            world: WORLD,
            band: self.band,
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

fn case(gpu: &mut Gpu, op: Op, dtype: Dtype, n: usize) {
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
        let c = &Ranked { sink, band };
        match op {
            Op::Reduce => kernels_slang::collective::band::all_reduce(c, xt),
            Op::Gather => kernels_slang::collective::band::all_gather(c, xt, yt),
            Op::Scatter => kernels_slang::collective::band::reduce_scatter(c, xt, yt),
        }
    });
    let got = down(gpu, dtype, y, out_n);
    if matches!(op, Op::Gather) {
        let exact: Vec<f32> = want.iter().map(|w| *w as f32).collect();
        assert_eq!(got, exact, "{op:?} {dtype:?} {n}");
        eprintln!("{op:?} {dtype:?} {n}: exact");
    } else {
        close(&got, &want, &format!("{op:?} {dtype:?} {n}"));
    }
    // The band's other slots are the other ranks': the collective leaves them.
    let after = down(gpu, dtype, band, slot * WORLD as usize);
    for r in ranks.filter(|&r| r != RANK as usize) {
        assert_eq!(
            after[r * slot..][..slot],
            held[r * slot..][..slot],
            "{op:?} {dtype:?} {n}: rank {r}'s slot"
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
        for n in [6usize, 1001, 8192] {
            case(&mut gpu, Op::Reduce, dtype, n);
        }
        // A bf16 shard of odd length would land its ranks mid-word.
        for n in [6usize, 1000, 8192] {
            case(&mut gpu, Op::Gather, dtype, n);
        }
        for n in [6usize, 999, 8190] {
            case(&mut gpu, Op::Scatter, dtype, n);
        }
    }
}
