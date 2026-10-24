#![cfg(feature = "cuda")]
#![allow(unsafe_code)]

//! ROCm has no peer kernels (its collectives are RCCL's), so this is CUDA's alone.

use engine_cuda::device::{Buffer, Context};
use kernels_cuda::{Ctx, Tensor, collective};
use poem_ir::Dtype;

use crate::micro::{ordinals, ranks};

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

/// Small integers, every partial sum of eight ranks' exact in bf16.
fn plane(rows: u32, width: u32, rank: u32) -> Vec<u8> {
    (0..rows * width)
        .flat_map(|i| bf16(((i % 29) + rank) as f32))
        .collect()
}

fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect()
}

/// Rank `r` on device `r % count`: the kernels read the other ranks'
/// stages through the peer mappings (a rank on its own device, through the
/// same pointers). One message spans more than one stage, so the chunking
/// is exercised.
#[test]
fn peer_collectives_answer_what_every_rank_holds() {
    let _alone = crate::alone();
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let world = ranks();
    let ordinals = ordinals(world);
    let peers = engine_cuda::comm::open_peers(&ordinals).expect("every rank reaches every other");
    let devices: Vec<Context> = (ordinals.iter())
        .map(|&ordinal| Context::bind(ordinal, None).unwrap())
        .collect();
    // SAFETY: the stages outlive the test; each context's stream is live.
    let ctxs: Vec<Ctx> = devices
        .iter()
        .zip(&peers)
        .map(|(device, peers)| unsafe { Ctx::on(device.stream()).with_peers(*peers) })
        .collect();
    let on = |rank: u32, bytes: usize, plane: Option<Vec<u8>>| {
        devices[rank as usize].bind_thread().unwrap();
        let mut buf = Buffer::zeroed(bytes).unwrap();
        if let Some(plane) = plane {
            buf.write(0, &plane).unwrap();
        }
        buf
    };

    let (rows, width) = (3000u32, 4096u32);
    let mut bufs: Vec<Buffer> = (0..world)
        .map(|rank| {
            on(
                rank,
                (rows * width * 2) as usize,
                Some(plane(rows, width, rank)),
            )
        })
        .collect();
    for ((device, ctx), buf) in devices.iter().zip(&ctxs).zip(&mut bufs) {
        device.bind_thread().unwrap();
        let mut t = Tensor::new(buf.ptr(), rows, width, Dtype::Bf16);
        collective::all_reduce(ctx, &mut t).unwrap();
    }
    for device in &devices {
        device.bind_thread().unwrap();
        device.synchronize().unwrap();
    }
    let lift = (world * (world - 1) / 2) as f32;
    let want: Vec<f32> = (0..rows * width)
        .map(|i| world as f32 * (i % 29) as f32 + lift)
        .collect();
    for (rank, buf) in bufs.iter().enumerate() {
        let mut got = vec![0u8; (rows * width * 2) as usize];
        buf.read(0, &mut got).unwrap();
        assert!(values(&got) == want, "rank {rank}'s all-reduce");
    }

    let (rows, width) = (5u32, 1024u32);
    let xs: Vec<Buffer> = (0..world)
        .map(|rank| {
            on(
                rank,
                (rows * width * 2) as usize,
                Some(plane(rows, width, rank)),
            )
        })
        .collect();
    let ys: Vec<Buffer> = (0..world)
        .map(|rank| on(rank, (rows * width * 2 * world) as usize, None))
        .collect();
    for (((device, ctx), x), y) in devices.iter().zip(&ctxs).zip(&xs).zip(&ys) {
        device.bind_thread().unwrap();
        let mut out = Tensor::new(y.ptr(), rows, world * width, Dtype::Bf16);
        collective::all_gather(
            ctx,
            Tensor::new(x.ptr(), rows, width, Dtype::Bf16),
            &mut out,
        )
        .unwrap();
    }
    for device in &devices {
        device.bind_thread().unwrap();
        device.synchronize().unwrap();
    }
    let shards: Vec<Vec<f32>> = (0..world)
        .map(|rank| values(&plane(rows, width, rank)))
        .collect();
    let want: Vec<f32> = (0..rows as usize)
        .flat_map(|r| {
            shards
                .iter()
                .flat_map(move |s| s[r * width as usize..(r + 1) * width as usize].to_vec())
        })
        .collect();
    for (rank, y) in ys.iter().enumerate() {
        let mut got = vec![0u8; (rows * width * 2 * world) as usize];
        y.read(0, &mut got).unwrap();
        assert!(values(&got) == want, "rank {rank}'s all-gather");
    }
}
