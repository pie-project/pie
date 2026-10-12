#![cfg(target_vendor = "apple")]

//! A live encode can hand work to another engine mid-frame: `signal` ends
//! the command buffer so far and has it raise the hand-off event, `wait`
//! starts the next buffer behind that value. Here the GPU is both sides:
//! one Hadamard, a signal, a wait on the same value, a second Hadamard.
//! `H·H = I`, so the data comes back as it went in only if both launches
//! ran, in order, across the fence — and the event reads the signaled value
//! once the frame has landed.

use engine_metal::device::{Buffer, Context, Handles, Pipelines};
use engine_metal::encode::Sink;
use kernels_metal::Tensor;
use kernels_metal::ane::Handoff;
use kernels_metal::elemwise::pointwise;
use kernels_metal::{Encode, Error};
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLSharedEvent};
use poem_ir::Dtype;

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn f32_floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn a_fenced_frame_lands_both_sides() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    // A shared event is shareable across devices, so the default device's
    // serves the context's.
    let event = || {
        MTLCreateSystemDefaultDevice()
            .expect("a device")
            .newSharedEvent()
            .expect("a shared event")
    };
    let handoff = Handoff::new(event(), event());

    let (rows, width, block) = (4u32, 256u32, 256u32);
    let data: Vec<f32> = (0..rows * width)
        .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
        .collect();
    let bytes = u64::from(rows) * u64::from(width) * 4;
    let mut buf = Buffer::zeroed(&device, bytes).expect("a buffer");
    buf.write(0, &f32_bytes(&data)).expect("write x");
    let handle = handles.bind(&buf, 0, buf.bytes()).expect("a handle");
    let x = Tensor::new(handle, rows, width, Dtype::F32);

    {
        let frame = device.frame().expect("a frame");
        let sink = Sink::new(&device, &frame, &pipelines, &handles).with_handoff(Some(&handoff));
        let at = handoff.allot();
        pointwise::hadamard(&sink, x, block, None).expect("the first launch");
        sink.signal(at.ready).expect("the signal fence");
        sink.wait(at.done).expect("the wait fence");
        pointwise::hadamard(&sink, x, block, None).expect("the second launch");
        // The other engine's side: answer `done` once the GPU raises `ready`.
        let (ready, done) = (handoff.ready_event().clone(), handoff.done_event().clone());
        let other = std::thread::spawn(move || {
            assert!(ready.waitUntilSignaledValue_timeoutMS(at.ready, 10_000));
            done.setSignaledValue(at.done);
        });
        frame.commit().expect("the commit");
        other.join().expect("the other engine");
        assert_eq!(
            handoff.ready_event().signaledValue(),
            at.ready,
            "the GPU raised the event"
        );
    }
    let got = f32_floats(&handles.read(handle, bytes).expect("read x"));
    for (i, (&g, &w)) in got.iter().zip(&data).enumerate() {
        assert!((g - w).abs() <= 1e-4, "element {i}: got {g}, want {w}");
    }
}

#[test]
fn an_encode_without_a_handoff_refuses_to_fence() {
    let Ok(device) = Context::bind() else {
        eprintln!("not asked: no Metal device");
        return;
    };
    let handles = Handles::new();
    let pipelines = Pipelines::new();
    let frame = device.frame().expect("a frame");
    let sink = Sink::new(&device, &frame, &pipelines, &handles);
    assert!(matches!(
        sink.signal(1),
        Err(Error::Backend {
            op: "handoff.signal",
            ..
        })
    ));
    assert!(matches!(
        sink.wait(1),
        Err(Error::Backend {
            op: "handoff.wait",
            ..
        })
    ));
}
