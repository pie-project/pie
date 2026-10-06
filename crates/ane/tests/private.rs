#![cfg(target_vendor = "apple")]

use ane::private::{Program, Surface, available, constant_blob};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLSharedEvent};

fn half(v: f32) -> u16 {
    half_bits(v)
}

fn half_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = (b >> 13) & 0x3ff;
    if v == 0.0 {
        return sign;
    }
    sign | ((exp as u16) << 10) | man as u16
}

fn float(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let man = f32::from(h & 0x3ff);
    if exp == 0 {
        return sign * man * 2f32.powi(-24);
    }
    sign * (1.0 + man / 1024.0) * 2f32.powi(exp - 15)
}

#[test]
fn a_private_program_runs_between_two_event_values() {
    if let Err(why) = available() {
        eprintln!("skipping: {why}");
        return;
    }
    let (channels, tokens) = (128u32, 64u32);
    let x = Surface::new(channels, tokens, false).unwrap();
    let y = Surface::new(channels, tokens, false).unwrap();
    let mil = format!(
        "program(1.3)\n{{\n    func main<ios18>({xt} x) {{\n        tensor<fp16, [1, 1, {c}, {t}]> x_t = tensor_buffer_to_tensor<ios17>(input = x);\n        tensor<fp16, [1, 1, {c}, 1]> w = const()[name = string(\"w\"), val = tensor<fp16, [1, 1, {c}, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), offset = uint64(64)))];\n        tensor<fp16, [1, 1, {c}, {t}]> y_t = mul(x = x_t, y = w);\n        {yt} y = tensor_to_tensor_buffer<ios17>(input = y_t, interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({ys}));\n    }} -> (y);\n}}\n",
        xt = x.buffer_type(channels, tokens),
        yt = y.buffer_type(channels, tokens),
        ys = y.strides(channels),
        c = channels,
        t = tokens,
    );
    let weights = constant_blob(&vec![half(2.0); channels as usize]);
    let program =
        Program::compile(&mil, &weights, &std::env::temp_dir().join("pie-ane-test")).unwrap();
    let procedure = program.procedure("main").unwrap();
    assert_eq!(program.inputs(procedure), vec!["x".to_string()]);
    let binding = program.bind(procedure, &[&x], &y).unwrap();
    let stride = x.stride() as usize / 2;
    let xs = unsafe {
        std::slice::from_raw_parts_mut(x.base().cast::<u16>(), channels as usize * stride)
    };
    for c in 0..channels as usize {
        for t in 0..tokens as usize {
            xs[c * stride + t] = half((c as f32 - 64.0) * 0.25 + t as f32 * 0.01);
        }
    }
    let device = MTLCreateSystemDefaultDevice().unwrap();
    let event: Retained<ProtocolObject<dyn MTLSharedEvent>> = device.newSharedEvent().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    unsafe {
        program
            .enqueue(
                &binding,
                Retained::as_ptr(&event) as *mut _,
                1,
                2,
                Box::new(move |ok| tx.send(ok).unwrap()),
            )
            .unwrap();
    }
    assert_eq!(event.signaledValue(), 0);
    event.setSignaledValue(1);
    assert!(event.waitUntilSignaledValue_timeoutMS(2, 5000));
    assert!(rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap());
    let ys =
        unsafe { std::slice::from_raw_parts(y.base().cast::<u16>(), channels as usize * stride) };
    for c in 0..channels as usize {
        for t in 0..tokens as usize {
            let want = 2.0 * float(xs[c * stride + t]);
            let got = float(ys[c * stride + t]);
            assert!(
                (got - want).abs() <= 1e-2 * want.abs().max(1.0),
                "c {c} t {t}: {got} vs {want}"
            );
        }
    }
}
