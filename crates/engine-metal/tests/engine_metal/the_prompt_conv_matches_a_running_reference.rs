#![cfg(target_vendor = "apple")]

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::ssm;
use kernels_metal::tensor::{RaggedTensor, RecurrentPool};
use poem_ir::Dtype;

const WIDTH: u32 = 4;

fn noise(at: u64) -> u32 {
    let mut x = at.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678_9ABC_DEF0;
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    (x >> 32) as u32
}

fn unit(at: u64) -> f32 {
    (noise(at) as f32 / u32::MAX as f32) * 2.0 - 1.0
}

fn to_bf16(v: f32) -> u16 {
    let bits = v.to_bits();
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(v: u16) -> f32 {
    f32::from_bits(u32::from(v) << 16)
}

/// Each lane's rows convolved one after another from its carried state, as
/// a decode runs them, and the state each lane carries out.
fn reference(
    x: &[f32],
    weight: &[f32],
    state: &[f32],
    spans: &[u32],
    channels: usize,
    dilation: usize,
) -> (Vec<f32>, Vec<f32>) {
    let width = WIDTH as usize;
    let hist = (width - 1) * dilation + 1;
    let mut y = vec![0.0f32; x.len()];
    let mut out = state.to_vec();
    let mut begin = 0usize;
    for (slot, &span) in spans.iter().enumerate() {
        let span = span as usize;
        let slab = slot * hist * channels;
        for c in 0..channels {
            let mut window: Vec<f32> = (0..hist).map(|s| state[slab + s * channels + c]).collect();
            for t in 0..span {
                window.remove(0);
                window.push(x[(begin + t) * channels + c]);
                let mut acc = 0.0f32;
                for k in 0..width {
                    acc += window[hist - 1 - (width - 1 - k) * dilation] * weight[c * width + k];
                }
                y[(begin + t) * channels + c] = acc / (1.0 + (-acc).exp());
            }
            for (s, v) in window.iter().enumerate() {
                out[slab + s * channels + c] = *v;
            }
        }
        begin += span;
    }
    (y, out)
}

#[test]
fn every_prompt_row_and_carried_state_answers_a_running_reference() {
    let Ok(device) = Context::bind() else {
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let upload = |bytes: &[u8]| {
        let mut buffer = Buffer::zeroed(&device, bytes.len() as u64).expect("a plane");
        buffer.write(0, bytes).expect("upload");
        buffer
    };
    let mut checked = 0usize;
    for (channels, dilation, spans) in [
        (10240u32, 1u32, vec![557u32]),
        (8192, 1, vec![3, 600, 1, 17, 16]),
        (96, 2, vec![2, 33, 7]),
        (96, 7, vec![40, 5]),
    ] {
        let hist = (WIDTH - 1) * dilation + 1;
        let rows: u32 = spans.iter().sum();
        let lanes = spans.len() as u32;
        let cells = (rows * channels) as usize;
        let x: Vec<f32> = (0..cells)
            .map(|at| from_bf16(to_bf16(unit(at as u64) * 2.0)))
            .collect();
        let weight: Vec<f32> = (0..(channels * WIDTH) as usize)
            .map(|at| from_bf16(to_bf16(unit(at as u64 ^ 0x77) * 0.6)))
            .collect();
        let state: Vec<f32> = (0..(lanes * hist * channels) as usize)
            .map(|at| unit(at as u64 ^ 0x5151))
            .collect();
        let (want_y, want_state) = reference(
            &x,
            &weight,
            &state,
            &spans,
            channels as usize,
            dilation as usize,
        );

        let bf =
            |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|f| to_bf16(*f).to_le_bytes()).collect() };
        let mut indptr = vec![0i32];
        let mut slots = Vec::new();
        for (lane, &span) in spans.iter().enumerate() {
            indptr.push(indptr[lane] + span as i32);
            slots.extend(std::iter::repeat_n(lane as u32, span as usize));
        }
        let x_b = upload(&bf(&x));
        let w_b = upload(&bf(&weight));
        let s_b = upload(
            &state
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let i_b = upload(
            &indptr
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let slot_b = upload(
            &slots
                .iter()
                .flat_map(|i| i.to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        let y_bytes = cells as u64 * 2;
        let y_b = upload(&vec![0xa5; y_bytes as usize + 64]);
        let bind = |b: &Buffer| handles.bind(b, 0, b.bytes()).expect("bind");
        let bank = Tensor::new(bind(&s_b), lanes, hist * channels, Dtype::F32);
        // The pool's state and new state name one bank, as the engine binds them.
        let pool = RecurrentPool {
            state: bank,
            slots: Tensor::new(bind(&slot_b), rows, 1, Dtype::U32),
            conv_state: bank,
            new_conv_state: bank,
        };
        let yh = bind(&y_b);
        let frame = device.frame().expect("frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles);
        ssm::causal_conv1d_chunked(
            &sink,
            RaggedTensor {
                data: Tensor::new(bind(&x_b), rows, channels, Dtype::Bf16),
                indptr: Tensor::new(bind(&i_b), lanes + 1, 1, Dtype::I32),
            },
            Tensor::new(bind(&w_b), channels, WIDTH, Dtype::Bf16),
            &pool,
            WIDTH,
            dilation,
            Tensor::new(yh, rows, channels, Dtype::Bf16),
        )
        .expect("conv");
        frame.commit().expect("commit");

        let got = handles.read(yh, y_bytes + 64).expect("read");
        assert!(
            got[y_bytes as usize..].iter().all(|&b| b == 0xa5),
            "write beyond the output at {spans:?}"
        );
        for (at, want) in want_y.iter().enumerate() {
            let actual = from_bf16(u16::from_le_bytes([got[2 * at], got[2 * at + 1]]));
            assert!(
                (actual - want).abs() <= 0.008 * want.abs() + 1e-3,
                "{channels} channels, dilation {dilation}, {spans:?}: row {} channel {}: {actual} vs {want}",
                at / channels as usize,
                at % channels as usize
            );
        }
        let carried = handles
            .read(bank.buf, u64::from(lanes * hist * channels) * 4)
            .expect("read state");
        for (at, want) in want_state.iter().enumerate() {
            let actual =
                f32::from_le_bytes(carried[4 * at..4 * at + 4].try_into().expect("four bytes"));
            assert_eq!(actual, *want, "{spans:?}: carried state at {at}");
        }
        checked += want_y.len() + want_state.len();
    }
    eprintln!("checked {checked} conv outputs and carried states against a running reference");
}
