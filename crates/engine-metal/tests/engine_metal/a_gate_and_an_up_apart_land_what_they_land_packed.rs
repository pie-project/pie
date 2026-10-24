#![cfg(target_vendor = "apple")]

//! Legs the pack left apart: each gated unit over a gate and an up plane
//! lands what it lands over the `[gate | up]` rows of the same values, bit
//! for bit, an unclamped one clamped at the largest finite f32.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::linear::mlp;
use poem_ir::Dtype;

fn unit(at: u64) -> f32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    ((x >> 32) as u32 as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .map(|f| ((f.to_bits() + 0x7fff + ((f.to_bits() >> 16) & 1)) >> 16) as u16)
        .flat_map(u16::to_le_bytes)
        .collect()
}

struct Rig {
    device: Context,
    handles: Handles,
    pipelines: Pipelines,
    held: Vec<Buffer>,
}

impl Rig {
    fn plane(&mut self, data: &[f32]) -> u32 {
        let bytes = bf16_bytes(data);
        let mut buf = Buffer::zeroed(&self.device, bytes.len() as u64).expect("a plane");
        buf.write(0, &bytes).expect("write");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("a handle");
        self.held.push(buf);
        handle
    }

    fn empty(&mut self, elements: usize) -> u32 {
        let buf = Buffer::zeroed(&self.device, (elements * 2) as u64).expect("a plane");
        let handle = self.handles.bind(&buf, 0, buf.bytes()).expect("a handle");
        self.held.push(buf);
        handle
    }

    fn read(&self, handle: u32, elements: usize) -> Vec<u8> {
        self.handles
            .read(handle, (elements * 2) as u64)
            .expect("read")
    }

    fn fire(&self, f: impl FnOnce(&Sink<'_>)) {
        let frame = self.device.frame().expect("a frame");
        let sink = Sink::new(&self.device, &frame, &self.pipelines, &self.handles);
        f(&sink);
        frame.commit().expect("the commit");
    }
}

#[test]
fn a_gate_and_an_up_apart_land_what_they_land_packed() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let mut rig = Rig {
        device,
        handles: Handles::new(),
        pipelines: Pipelines::new(),
        held: Vec::new(),
    };
    let (rows, inter) = (3usize, 1000usize);
    let n = rows * inter;
    let gv: Vec<f32> = (0..n as u64).map(|i| 4.0 * unit(i)).collect();
    let uv: Vec<f32> = (0..n as u64).map(|i| 4.0 * unit(i + (1 << 20))).collect();
    let pv: Vec<f32> = gv
        .chunks(inter)
        .zip(uv.chunks(inter))
        .flat_map(|(g, u)| g.iter().chain(u).copied().collect::<Vec<_>>())
        .collect();
    let b = |h: u32, r: usize, w: usize| Tensor::new(h, r as u32, w as u32, Dtype::Bf16);
    let (gate, up, packed) = (rig.plane(&gv), rig.plane(&uv), rig.plane(&pv));
    let (gt, ut, pt) = (
        b(gate, rows, inter),
        b(up, rows, inter),
        b(packed, rows, 2 * inter),
    );
    let iu = inter as u32;
    let (limit, alpha, beta, cap) = (2.5f32, 1.702f32, 3.0f32, 4.0f32);
    let max = f32::MAX;
    let forms = [
        "swiglu",
        "swiglu clamp",
        "swiglu clamp alpha",
        "swiglu alpha",
        "situ",
        "situ capped",
        "geglu",
    ];
    for (k, what) in forms.iter().enumerate() {
        let (apart, joined) = (rig.empty(n), rig.empty(n));
        let (ya, yj) = (b(apart, rows, inter), b(joined, rows, inter));
        rig.fire(|s| {
            match k {
                0 => mlp::swiglu_clamp_split(s, gt, ut, max, ya),
                1 => mlp::swiglu_clamp_split(s, gt, ut, limit, ya),
                2 => mlp::swiglu_clamp_alpha_split(s, gt, ut, limit, alpha, ya),
                3 => mlp::swiglu_clamp_alpha_split(s, gt, ut, max, alpha, ya),
                4 => mlp::situ_split(s, gt, ut, beta, None, ya),
                5 => mlp::situ_split(s, gt, ut, beta, Some(cap), ya),
                _ => mlp::geglu_tanh(s, gt, ut, ya),
            }
            .expect("apart");
            match k {
                0 => mlp::swiglu(s, pt, iu, yj),
                1 => mlp::swiglu_clamp(s, pt, iu, limit, yj),
                2 => mlp::swiglu_clamp_alpha(s, pt, iu, limit, alpha, yj),
                3 => mlp::swiglu_clamp_alpha(s, pt, iu, max, alpha, yj),
                4 => mlp::situ(s, pt, iu, beta, None, yj),
                5 => mlp::situ(s, pt, iu, beta, Some(cap), yj),
                _ => mlp::geglu_tanh_packed(s, pt, iu, yj),
            }
            .expect("joined");
        });
        let (a, j) = (rig.read(apart, n), rig.read(joined, n));
        let off = a
            .as_chunks::<2>()
            .0
            .iter()
            .zip(j.as_chunks::<2>().0)
            .filter(|(a, j)| a != j)
            .count();
        assert_eq!(
            off, 0,
            "{what}: {off} of {n} values differ apart and joined"
        );
    }
}
