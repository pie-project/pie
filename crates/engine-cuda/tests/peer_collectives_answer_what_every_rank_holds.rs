#![cfg(feature = "cuda")]

use engine_cuda::device::{Buffer, Context};
use kernels_cuda::{Ctx, Tensor, collective};
use model_ir::Dtype;

fn bf16(v: f32) -> [u8; 2] {
    let bits = v.to_bits();
    [(bits >> 16) as u8, (bits >> 24) as u8]
}

fn plane(rows: u32, width: u32, rank: u32) -> Vec<u8> {
    (0..rows * width)
        .flat_map(|i| bf16(((i % 61) as f32) + rank as f32 * 100.0))
        .collect()
}

fn values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks(2)
        .map(|b| f32::from_bits(u32::from(u16::from_le_bytes([b[0], b[1]])) << 16))
        .collect()
}

/// Two ranks on one device stand in for a pair of peers: the kernels read
/// the other rank's stage through the same pointers a peer mapping hands
/// out. One message spans more than one stage, so the chunking is exercised.
#[test]
fn peer_collectives_answer_what_every_rank_holds() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let peers = engine_cuda::comm::open_peers(&[0, 0]).expect("one device reaches itself");
    let devices: Vec<Context> = (0..2).map(|_| Context::bind(0, None).unwrap()).collect();
    // SAFETY: the stages outlive the test; each context's stream is live.
    let ctxs: Vec<Ctx> = devices
        .iter()
        .zip(&peers)
        .map(|(device, peers)| unsafe { Ctx::on(device.stream()).with_peers(*peers) })
        .collect();

    let (rows, width) = (3000u32, 4096u32);
    let mut bufs: Vec<Buffer> = (0..2)
        .map(|rank| {
            let mut buf = Buffer::zeroed((rows * width * 2) as usize).unwrap();
            buf.write(0, &plane(rows, width, rank)).unwrap();
            buf
        })
        .collect();
    for (ctx, buf) in ctxs.iter().zip(&mut bufs) {
        let mut t = Tensor::new(buf.ptr(), rows, width, Dtype::Bf16);
        collective::all_reduce(ctx, &mut t).unwrap();
    }
    for device in &devices {
        device.synchronize().unwrap();
    }
    let want: Vec<f32> = (0..rows * width)
        .map(|i| 2.0 * (i % 61) as f32 + 100.0)
        .collect();
    for buf in &bufs {
        let mut got = vec![0u8; (rows * width * 2) as usize];
        buf.read(0, &mut got).unwrap();
        assert_eq!(values(&got), want);
    }

    let (rows, width) = (5u32, 1024u32);
    let xs: Vec<Buffer> = (0..2)
        .map(|rank| {
            let mut buf = Buffer::zeroed((rows * width * 2) as usize).unwrap();
            buf.write(0, &plane(rows, width, rank)).unwrap();
            buf
        })
        .collect();
    let ys: Vec<Buffer> = (0..2)
        .map(|_| Buffer::zeroed((rows * width * 4) as usize).unwrap())
        .collect();
    for ((ctx, x), y) in ctxs.iter().zip(&xs).zip(&ys) {
        let mut out = Tensor::new(y.ptr(), rows, 2 * width, Dtype::Bf16);
        collective::all_gather(
            ctx,
            Tensor::new(x.ptr(), rows, width, Dtype::Bf16),
            &mut out,
        )
        .unwrap();
    }
    for device in &devices {
        device.synchronize().unwrap();
    }
    let shards: Vec<Vec<f32>> = (0..2)
        .map(|rank| values(&plane(rows, width, rank)))
        .collect();
    let want: Vec<f32> = (0..rows as usize)
        .flat_map(|r| {
            shards
                .iter()
                .flat_map(move |s| s[r * width as usize..(r + 1) * width as usize].to_vec())
        })
        .collect();
    for y in &ys {
        let mut got = vec![0u8; (rows * width * 4) as usize];
        y.read(0, &mut got).unwrap();
        assert_eq!(values(&got), want);
    }
}
