use core::ffi::c_void;

use cudarc::runtime::sys as rt;

use crate::jit::Ctx;
use crate::jit::abi::{Tensor, bf16};
use kernels::plane::{Const, In, InOut, Out};
use kernels::points::{Layout, Norm};

fn to_bf16(v: f32) -> u16 {
    let b = v.to_bits();
    let rounding = 0x7fff + ((b >> 16) & 1);
    ((b + rounding) >> 16) as u16
}

fn of_bf16(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

struct Device {
    stream: *mut c_void,
    slabs: Vec<*mut c_void>,
}

impl Device {
    fn open() -> Option<Device> {
        if unsafe { rt::cudaSetDevice(0) } != rt::cudaError::cudaSuccess {
            return None;
        }

        if unsafe { rt::cudaFree(core::ptr::null_mut()) } != rt::cudaError::cudaSuccess {
            return None;
        }
        let mut raw: rt::cudaStream_t = core::ptr::null_mut();
        let code =
            unsafe { rt::cudaStreamCreateWithPriority(&raw mut raw, rt::cudaStreamNonBlocking, 0) };
        (code == rt::cudaError::cudaSuccess).then(|| Device {
            stream: raw.cast(),
            slabs: Vec::new(),
        })
    }

    fn up(&mut self, host: &[u16]) -> *mut bf16 {
        let bytes = std::mem::size_of_val(host);
        let mut ptr: *mut c_void = core::ptr::null_mut();
        assert_eq!(
            unsafe { rt::cudaMalloc(&raw mut ptr, bytes) },
            rt::cudaError::cudaSuccess
        );
        self.slabs.push(ptr);
        assert_eq!(
            unsafe {
                rt::cudaMemcpyAsync(
                    ptr,
                    host.as_ptr().cast(),
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                    self.stream.cast(),
                )
            },
            rt::cudaError::cudaSuccess
        );
        ptr.cast()
    }

    fn down(&self, src: *const bf16, len: usize) -> Vec<u16> {
        let mut host = vec![0u16; len];
        assert_eq!(
            unsafe {
                rt::cudaMemcpyAsync(
                    host.as_mut_ptr().cast(),
                    src.cast(),
                    len * 2,
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                    self.stream.cast(),
                )
            },
            rt::cudaError::cudaSuccess
        );
        assert_eq!(
            unsafe { rt::cudaStreamSynchronize(self.stream.cast()) },
            rt::cudaError::cudaSuccess,
            "the fire itself failed, not the copy"
        );
        host
    }

    fn ctx(&self) -> Ctx<'_> {
        unsafe { Ctx::on(self.stream) }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        for slab in self.slabs.drain(..) {
            let _ = unsafe { rt::cudaFree(slab) };
        }
        let _ = unsafe { rt::cudaStreamDestroy(self.stream.cast()) };
    }
}

#[test]
fn scale_multiplies_by_a_device_scalar() {
    let Some(mut dev) = Device::open() else {
        eprintln!("no cuda device: skipping");
        return;
    };
    const ROWS: i32 = 3;
    const WIDTH: i32 = 37;
    let n = (ROWS * WIDTH) as usize;

    let host: Vec<u16> = (0..n).map(|i| to_bf16((i as f32) * 0.5 - 7.0)).collect();
    let s = to_bf16(0.1);

    let x = dev.up(&host);
    let scalar = dev.up(&[s]);

    dev.ctx()
        .scale::<bf16>(
            Const::<Tensor<bf16>>::new(scalar.cast_const()),
            InOut::<Tensor<bf16>> {
                ptr: x,
                rows: ROWS,
                width: WIDTH,
            },
        )
        .expect("the point is claimed and the rectangle is well formed");

    let got = dev.down(x.cast_const(), n);
    let want: Vec<u16> = host
        .iter()
        .map(|&v| to_bf16(of_bf16(v) * of_bf16(s)))
        .collect();
    assert_eq!(got, want, "s = {}", of_bf16(s));
}

#[test]
fn select_takes_one_layers_slice() {
    let Some(mut dev) = Device::open() else {
        eprintln!("no cuda device: skipping");
        return;
    };
    const ROWS: i32 = 4;
    const LAYERS: u32 = 5;
    const WIDTH: u32 = 7;
    const LAYER: u32 = 3;
    let stride = (LAYERS * WIDTH) as i32;

    let table: Vec<u16> = (0..ROWS)
        .flat_map(|r| (0..stride).map(move |c| to_bf16((r * 1000 + c) as f32)))
        .collect();
    let src = dev.up(&table);
    let dst = dev.up(&vec![to_bf16(-1.0); (ROWS * WIDTH as i32) as usize]);

    dev.ctx()
        .select::<bf16>(
            In::<Tensor<bf16>> {
                ptr: src.cast_const(),
                rows: ROWS,
                width: stride,
            },
            LAYER,
            WIDTH,
            Out::<Tensor<bf16>> {
                ptr: dst,
                rows: ROWS,
                width: WIDTH as i32,
            },
        )
        .expect("the point is claimed and the slice is inside the row");

    let got = dev.down(dst.cast_const(), (ROWS * WIDTH as i32) as usize);
    let want: Vec<u16> = (0..ROWS)
        .flat_map(|r| {
            (0..WIDTH as i32).map(move |i| to_bf16((r * 1000 + (LAYER * WIDTH) as i32 + i) as f32))
        })
        .collect();
    assert_eq!(got, want);
}

#[test]
fn select_refuses_a_layer_the_relay_does_not_reach() {
    let ctx = unsafe { Ctx::on(core::ptr::null_mut()) };
    let table = In::<Tensor<bf16>> {
        ptr: core::ptr::null(),
        rows: 2,
        width: 20,
    };
    let out = |width: i32| Out::<Tensor<bf16>> {
        ptr: core::ptr::null_mut(),
        rows: 2,
        width,
    };

    let past = ctx.select::<bf16>(table, 4, 7, out(7));
    assert!(
        matches!(past, Err(kernels::Refusal::Narrow { .. })),
        "{past:?}"
    );

    let edge = ctx.select::<bf16>(table, 2, 7, out(7));
    assert!(
        matches!(edge, Err(kernels::Refusal::Narrow { .. })),
        "{edge:?}"
    );

    let mismatch = ctx.select::<bf16>(table, 1, 7, out(9));
    assert!(
        matches!(mismatch, Err(kernels::Refusal::Narrow { .. })),
        "{mismatch:?}"
    );
}
