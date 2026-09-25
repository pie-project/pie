#![cfg(target_vendor = "apple")]
use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::encode::{Arg, ArgValue, Encode, Fire, Grid};
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use kernels_metal::{Tensor, attn::ssm};
use model_ir::Dtype;

fn data(n: usize, salt: u64, bf16: bool) -> Vec<u8> {
    (0..n)
        .flat_map(|i| {
            let mut x = (i as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ salt;
            x ^= x >> 33;
            let v = ((x >> 40) as f32 / 16777216. - 0.5) * 0.2;
            if bf16 {
                ((v.to_bits() >> 16) as u16).to_le_bytes().to_vec()
            } else {
                v.to_le_bytes().to_vec()
            }
        })
        .collect()
}

#[test]
fn the_parallel_conv_preserves_outputs_and_state() {
    let dev = Context::bind().unwrap();
    let handles = Handles::new();
    let pipes = Pipelines::new();
    for (lengths, channels, width, dilation) in [
        (vec![1, 0, 2, 8, 1366], 128u32, 4u32, 1u32),
        (vec![32, 0, 65, 7], 128, 4, 1),
        (vec![1024, 1, 63], 128, 8, 4),
        (vec![1024, 1], 128, 9, 4),
        (vec![1024, 0, 33], 128, 1, 1),
        (vec![1366], 10240, 4, 1),
        (vec![512; 4], 10240, 4, 1),
        (vec![2048], 10240, 4, 1),
        (vec![128; 16], 10240, 4, 1),
    ] {
        let hist = (width - 1) * dilation + 1;
        let mut ptr = vec![0i32];
        for len in &lengths {
            ptr.push(ptr.last().unwrap() + len);
        }
        let rows = *ptr.last().unwrap() as u32;
        let slots: Vec<u32> = lengths
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| std::iter::repeat_n((lengths.len() - i) as u32, n as usize))
            .collect();
        let state_bytes = u64::from(hist * channels) * (lengths.len() as u64 + 2) * 4;
        let initial = data(state_bytes as usize / 4, 99, false);
        let upload = |bytes: &[u8]| {
            let mut buf = Buffer::zeroed(&dev, bytes.len() as u64).unwrap();
            buf.write(0, bytes).unwrap();
            buf
        };
        let x = upload(&data((rows * channels) as usize, 17, true));
        let weight = upload(&data((channels * width) as usize, 123, true));
        let indptr = upload(&ptr.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<_>>());
        let slots = upload(
            &slots
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).unwrap();
        let (hx, hw, hi, hs) = (bind(&x), bind(&weight), bind(&indptr), bind(&slots));
        for alias in [false, true] {
            let mut reference = None;
            for candidate in [false, true] {
                let old = upload(&initial);
                let new = upload(&initial);
                let out = Buffer::zeroed(&dev, u64::from(rows * channels) * 2).unwrap();
                let (ho, hn, hy) = (
                    bind(&old),
                    bind(if alias { &old } else { &new }),
                    bind(&out),
                );
                let launch = |sink: &Sink<'_>| {
                    if candidate {
                        ssm::causal_conv1d_chunked(
                            sink,
                            RaggedTensor {
                                data: Tensor::new(hx, rows, channels, Dtype::Bf16),
                                indptr: Tensor::new(hi, ptr.len() as u32, 1, Dtype::I32),
                            },
                            Tensor::new(hw, channels, width, Dtype::Bf16),
                            &RecurrentPool {
                                conv_state: Tensor::new(
                                    ho,
                                    lengths.len() as u32 + 2,
                                    hist * channels,
                                    Dtype::F32,
                                ),
                                new_conv_state: Tensor::new(
                                    hn,
                                    lengths.len() as u32 + 2,
                                    hist * channels,
                                    Dtype::F32,
                                ),
                                slots: Tensor::new(hs, rows, 1, Dtype::U32),
                                state: Tensor::new(ho, 1, 1, Dtype::F32),
                            },
                            width,
                            dilation,
                            Tensor::new(hy, rows, channels, Dtype::Bf16),
                        )
                        .unwrap();
                    } else {
                        sink.fire(
                            Fire::at(
                                "attn/ssm_causal_conv1d.metal",
                                "causal_conv1d_chunked_bfloat16",
                            )
                            .apply(Grid::of(
                                [channels, lengths.len() as u32, 1],
                                [256.min(channels), 1, 1],
                            )),
                            &[
                                ArgValue::Buffer(hx),
                                ArgValue::Buffer(hi),
                                ArgValue::Buffer(hw),
                                ArgValue::Buffer(ho),
                                ArgValue::BufferMut(hn),
                                ArgValue::Buffer(hs),
                                ArgValue::BufferMut(hy),
                                (channels as i32).arg(),
                                (width as i32).arg(),
                                (dilation as i32).arg(),
                            ],
                        )
                        .unwrap();
                    }
                };
                let frame = dev.frame().unwrap();
                launch(&Sink::new(&dev, &frame, &pipes, &handles));
                frame.commit().unwrap();
                let result = (
                    handles.read(hy, u64::from(rows * channels) * 2).unwrap(),
                    handles.read(hn, state_bytes).unwrap(),
                );
                if let Some(want) = &reference {
                    assert_eq!(
                        &result, want,
                        "rows={rows} width={width} dilation={dilation} alias={alias}"
                    );
                } else {
                    reference = Some(result);
                }
            }
        }
    }
}
