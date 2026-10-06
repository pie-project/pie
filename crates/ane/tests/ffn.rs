#![cfg(target_vendor = "apple")]

use ane::ffn::{Ffn, INTERMEDIATE_BLOCK, Memory, SEGMENT, Shape, signs};
use ane::private::{Surface, available};
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

#[test]
fn the_ffn_program_matches_its_reference_on_one_chunk() {
    if let Err(why) = available() {
        eprintln!("skipping: {why}");
        return;
    }
    let shape = Shape::new(SEGMENT, 2 * INTERMEDIATE_BLOCK, 1).unwrap();
    let memory = Memory::new(&shape).unwrap();
    let started = std::time::Instant::now();
    let ffn = Ffn::compile(&shape, &memory, &std::env::temp_dir().join("pie-ane-test")).unwrap();
    eprintln!("compiled in {:.1}s", started.elapsed().as_secs_f64());
    let rows = 512u32;
    let (hidden, channels) = (shape.hidden, shape.ane);
    let mut random = Random(7);
    let x: Vec<Vec<i8>> = (0..hidden)
        .map(|_| (0..rows).map(|_| random.code()).collect())
        .collect();
    let tx: Vec<f32> = (0..rows)
        .map(|_| 2.0 + random.next() as f32 * 4.0)
        .collect();
    let wg: Vec<Vec<i8>> = (0..channels)
        .map(|_| (0..hidden).map(|_| random.code()).collect())
        .collect();
    let wu: Vec<Vec<i8>> = (0..channels)
        .map(|_| (0..hidden).map(|_| random.code()).collect())
        .collect();
    let wd: Vec<Vec<i8>> = (0..hidden)
        .map(|_| (0..channels).map(|_| random.code()).collect())
        .collect();
    let sg: Vec<f32> = (0..channels)
        .map(|_| 0.05 + random.next() as f32 * 0.1)
        .collect();
    let su: Vec<f32> = (0..channels)
        .map(|_| 0.05 + random.next() as f32 * 0.1)
        .collect();
    let sd: Vec<f32> = (0..hidden)
        .map(|_| 0.05 + random.next() as f32 * 0.1)
        .collect();
    for c in 0..hidden {
        for t in 0..rows {
            put_i8(&memory.inputs[0], c, t, x[c as usize][t as usize]);
        }
    }
    for t in 0..rows {
        put_f16(&memory.token_scale, 0, t, tx[t as usize]);
    }
    let set = &memory.sets[0];
    for r in 0..channels {
        for c in 0..hidden {
            put_i8(&set.gate[0], r, c, wg[r as usize][c as usize]);
            put_i8(&set.up[0], r, c, wu[r as usize][c as usize]);
        }
        put_f16(&set.gate_scale, r, 0, sg[r as usize]);
        put_f16(&set.up_scale, r, 0, su[r as usize]);
    }
    for r in 0..hidden {
        for c in 0..channels {
            put_i8(&set.down[0], r, c, wd[r as usize][c as usize]);
        }
        put_f16(&set.down_scale, r, 0, sd[r as usize]);
    }
    let device = MTLCreateSystemDefaultDevice().unwrap();
    let event: Retained<_> = device.newSharedEvent().unwrap();
    let evaluation = ffn.evaluation(rows).unwrap();
    let (tx_done, rx_done) = std::sync::mpsc::channel();
    unsafe {
        ffn.program
            .enqueue(
                &ffn.evaluations[evaluation].1[0],
                Retained::as_ptr(&event) as *mut _,
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

    let sign = signs();
    let block = INTERMEDIATE_BLOCK as usize;
    let norm = 1.0 / (block as f64).sqrt();
    let mut worst = 0.0f64;
    let (mut diff, mut total) = (0.0f64, 0.0f64);
    let (mut cross, mut got2) = (0.0f64, 0.0f64);
    for t in (0..rows as usize).step_by(37) {
        let xt: Vec<f64> = (0..hidden as usize)
            .map(|c| f64::from(x[c][t]) / 128.0)
            .collect();
        let mut h = vec![0.0f64; channels as usize];
        for r in 0..channels as usize {
            let g: f64 = (0..hidden as usize)
                .map(|c| f64::from(wg[r][c]) / 128.0 * xt[c])
                .sum::<f64>()
                * f64::from(half_round(sg[r]))
                * f64::from(half_round(tx[t]));
            let u: f64 = (0..hidden as usize)
                .map(|c| f64::from(wu[r][c]) / 128.0 * xt[c])
                .sum::<f64>()
                * f64::from(half_round(su[r]));
            h[r] = g / (1.0 + (-g).exp()) * u;
        }
        let mut hr = vec![0.0f64; channels as usize];
        for ch in 0..channels as usize {
            let group = ch / block * block;
            let o = ch % block;
            hr[ch] = (0..block)
                .map(|i| {
                    let had = if (o & i).count_ones() & 1 == 1 {
                        -1.0
                    } else {
                        1.0
                    };
                    f64::from(sign[o]) * had * norm * h[group + i]
                })
                .sum();
        }
        let scale = get_f16(&memory.partial, hidden, t as u32) * f64::from(half_round(tx[t]));
        for c in 0..hidden as usize {
            let want: f64 = (0..channels as usize)
                .map(|j| f64::from(wd[c][j]) / 128.0 * hr[j])
                .sum::<f64>()
                * f64::from(half_round(sd[c]))
                * f64::from(half_round(tx[t]));
            let got = get_f16(&memory.partial, c as u32, t as u32) * scale;
            cross += got * want;
            got2 += got * got;
            diff += (got - want) * (got - want);
            total += want * want;
            worst = worst.max((got - want).abs());
        }
    }
    let error = (diff / total).sqrt();
    eprintln!(
        "relative RMS error {error:.4}, worst {worst:.3e}, ratio {:.4}, correlation {:.4}",
        cross / total,
        cross / (total * got2).sqrt()
    );
    assert!(error < 0.05, "relative RMS error {error}");
}

fn half_round(v: f32) -> f32 {
    float(half(v)) as f32
}
