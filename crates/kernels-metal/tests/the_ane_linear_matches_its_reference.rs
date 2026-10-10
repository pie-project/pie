#![cfg(target_vendor = "apple")]

use kernels_metal::ane::linear::{Linear, Memory, Shape};
use kernels_metal::ane::{Surface, available};
use objc2::rc::Retained;
use objc2_metal::{MTLCreateSystemDefaultDevice, MTLDevice, MTLSharedEvent};

fn half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    if value == 0.0 || exponent <= 0 {
        return sign;
    }
    let rounded = ((bits & 0x7f_ffff) + 0x1000) >> 13;
    let (exponent, mantissa) = if rounded == 0x400 {
        (exponent + 1, 0)
    } else {
        (exponent, rounded)
    };
    sign | ((exponent as u16) << 10) | mantissa as u16
}

fn float(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = f64::from(h & 0x3ff);
    if exponent == 0 {
        return sign * mantissa * 2f64.powi(-24);
    }
    sign * (1.0 + mantissa / 1024.0) * 2f64.powi(exponent - 15)
}

struct Random(u64);
impl Random {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn code(&mut self) -> i8 {
        (self.next() * 254.0 - 127.0).round() as i8
    }
}

fn put_i8(surface: &Surface, row: u32, column: u32, value: i8) {
    unsafe {
        *surface
            .base()
            .add((row * surface.stride() + column) as usize)
            .cast::<i8>() = value
    };
}
fn put_f16(surface: &Surface, row: u32, column: u32, value: f32) {
    unsafe {
        *surface
            .base()
            .add((row * surface.stride() + column * 2) as usize)
            .cast::<u16>() = half(value)
    };
}
fn get_f16(surface: &Surface, row: u32, column: u32) -> f64 {
    float(unsafe {
        *surface
            .base()
            .add((row * surface.stride() + column * 2) as usize)
            .cast::<u16>()
    })
}

/// A 3072-wide contraction (two 1536-segments), 1024 output columns with
/// one unit of 512 to the Neural Engine, one 512-row chunk, checked against
/// an f64 reference of the same int8/fp16 arithmetic.
#[test]
fn the_ane_linear_matches_its_reference_on_one_chunk() {
    if let Err(why) = available() {
        eprintln!("skipping: {why}");
        return;
    }
    let shape = Shape::new(3072, 1024, 1).unwrap();
    assert_eq!(
        (shape.segment, shape.segments(), shape.keep, shape.ane),
        (1536, 2, 512, 512)
    );
    let memory = Memory::new(&shape).unwrap();
    let started = std::time::Instant::now();
    let linear =
        Linear::compile(&shape, &memory, &std::env::temp_dir().join("pie-ane-test")).unwrap();
    eprintln!("compiled in {:.1}s", started.elapsed().as_secs_f64());
    let rows = 512u32;
    let (k, columns, segment) = (shape.k, shape.ane, shape.segment);
    let mut random = Random(11);
    let x: Vec<Vec<i8>> = (0..k)
        .map(|_| (0..rows).map(|_| random.code()).collect())
        .collect();
    let w: Vec<Vec<i8>> = (0..columns)
        .map(|_| (0..k).map(|_| random.code()).collect())
        .collect();
    let s: Vec<f32> = (0..columns)
        .map(|_| 0.05 + random.next() as f32 * 0.1)
        .collect();
    for c in 0..k {
        for t in 0..rows {
            put_i8(
                &memory.inputs[(c / segment) as usize],
                c % segment,
                t,
                x[c as usize][t as usize],
            );
        }
    }
    let set = &memory.sets[1];
    for r in 0..columns {
        for c in 0..k {
            put_i8(
                &set.w[(c / segment) as usize],
                r,
                c % segment,
                w[r as usize][c as usize],
            );
        }
        put_f16(&set.scale, r, 0, s[r as usize]);
    }
    let device = MTLCreateSystemDefaultDevice().unwrap();
    let event: Retained<_> = device.newSharedEvent().unwrap();
    let evaluation = linear.evaluation(rows).unwrap();
    let (tx_done, rx_done) = std::sync::mpsc::channel();
    unsafe {
        linear
            .program
            .enqueue(
                &linear.evaluations[evaluation].1[1],
                &event,
                1,
                2,
                Box::new(move |ok| tx_done.send(ok).unwrap()),
            )
            .unwrap();
    }
    let started = std::time::Instant::now();
    event.setSignaledValue(1);
    assert!(event.waitUntilSignaledValue_timeoutMS(2, 10_000));
    eprintln!(
        "evaluated in {:.2} ms",
        started.elapsed().as_secs_f64() * 1e3
    );
    assert!(
        rx_done
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap()
    );
    let mut worst = 0.0f64;
    let (mut diff, mut total) = (0.0f64, 0.0f64);
    for t in (0..rows as usize).step_by(29) {
        for c in 0..columns as usize {
            let want: f64 = (0..k as usize)
                .map(|j| f64::from(w[c][j]) / 128.0 * f64::from(x[j][t]) / 128.0)
                .sum::<f64>()
                * float(half(s[c]));
            let got = get_f16(&memory.partial, c as u32, t as u32);
            diff += (got - want) * (got - want);
            total += want * want;
            worst = worst.max((got - want).abs());
        }
    }
    let error = (diff / total).sqrt();
    eprintln!("relative RMS error {error:.5}, worst {worst:.3e}");
    assert!(error < 0.01, "relative RMS error {error}");
}
