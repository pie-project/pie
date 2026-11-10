//! A generated row kernel lands a result where its store says: a window of
//! the row's columns in a plane of its own (a split's piece), or a paged
//! cache's row its page and slot name (`page · page_size + slot`, none where
//! the slot lies past the page or the row is not live), bf16 or e4m3 under a
//! scale, each element as kernels-ir's `CacheDtype::stored` declares it. A
//! cache whose pages lie a stride apart (rows of other planes between them)
//! takes the same rows a stride apart, and nothing between.

#![cfg(feature = "rocm")]

mod common;
#[path = "common/within.rs"]
mod within;

use dtype::Dtype::Bf16;
use kernels_ir::anchor::CacheDtype;
use kernels_ir::{Binary, Op, Shape};
use kernels_rocm::elemwise::skeleton::Cache;
use kernels_rocm::synth::{self, In, Plane, Region};
use kernels_rocm::tensor::Tensor;
use within::{Gpu, Lcg, from_bf16};

/// An e4m3 byte's value.
fn e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = i32::from((byte >> 3) & 0xf);
    let mantissa = f32::from(byte & 7);
    sign * if exponent == 0 {
        mantissa * 2f32.powi(-9)
    } else {
        (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }
}

#[test]
fn a_result_lands_its_window_or_its_cache_row() {
    if !within::present() {
        return;
    }
    lands(0);
}

#[test]
fn a_result_lands_in_pages_a_stride_apart() {
    if !within::present() {
        return;
    }
    lands(3);
}

/// `gap` rows of the plane lie between one page's rows and the next page's.
fn lands(gap: usize) {
    let (rows, width, page_size, pages) = (7usize, 256usize, 4u32, 4usize);
    let page_rows = page_size as usize + gap;
    let page_stride = if gap == 0 {
        0
    } else {
        (page_rows * width) as u32
    };
    // x · s, landed three ways
    let region = Region {
        width: width as u32,
        params: vec![Shape::Row, Shape::Row],
        ops: vec![Op::Binary(Binary::Mul, 0, 1)],
        results: vec![2, 2, 2],
        rounds: vec![(2, Bf16)],
    };
    let mut gpu = Gpu::open();
    let mut lcg = Lcg::seeded(3);
    let ((xr, xv), (sr, sv)) = (lcg.row(rows * width), lcg.row(rows * width));
    let (x, s) = (gpu.up(&xr), gpu.up(&sr));
    let page: Vec<i32> = vec![2, 0, 3, 1, 1, 0, 2];
    let slot: Vec<i32> = vec![1, 0, 3, 5, 2, 3, 0];
    let valid: Vec<u8> = vec![1, 1, 1, 1, 1, 0, 1];
    let (pg, sl, va) = (gpu.up(&page), gpu.up(&slot), gpu.up(&valid));
    let cells = page_rows * pages;
    let window = gpu.zeros(rows * 128 * 2);
    let bf16_cache = gpu.zeros(cells * width * 2);
    let fp8_cache = gpu.zeros(cells * width);
    let kernel = synth::synth(&region, &[In::Rows(Bf16), In::Rows(Bf16)], 3).expect("synthesized");
    kernel.load().expect("loads");
    let t = |base: u64, r: usize, w: usize| Tensor::new(base, r as u32, w as u32, Bf16);
    let cache = |base: u64, e4m3: Option<f32>| Cache {
        base,
        width: width as u32,
        page: pg,
        slot: sl,
        valid: va,
        page_size,
        e4m3,
        page_stride,
    };
    let scale = 0.5f32;
    let ins = [
        Plane::rows(&t(x, rows, width)),
        Plane::rows(&t(s, rows, width)),
    ];
    let outs = [
        Plane::window(&t(window, rows, 128), 64, 128),
        Plane::paged(cache(bf16_cache, None), 0, width as u32),
        Plane::paged(cache(fp8_cache, Some(scale)), 0, width as u32),
    ];
    kernel
        .fire(&gpu.ctx(), &ins, &outs, rows as u32)
        .unwrap_or_else(|e| panic!("fires: {e:?}"));
    gpu.sync();
    let product: Vec<f32> = (0..rows * width)
        .map(|i| kernels_ir::skeleton::round_bf16(xv[i] * sv[i]))
        .collect();
    let windowed: Vec<u16> = gpu.down(window, rows * 128);
    for r in 0..rows {
        for c in 0..128 {
            assert_eq!(
                from_bf16(windowed[r * 128 + c]),
                product[r * width + 64 + c],
                "window"
            );
        }
    }
    let (bf16_rows, fp8_rows): (Vec<u16>, Vec<u8>) = (
        gpu.down(bf16_cache, cells * width),
        gpu.down(fp8_cache, cells * width),
    );
    let fp8 = CacheDtype::E4m3 { scale };
    for cell in 0..cells {
        // the row landing at this cell, if one does
        let landed = (0..rows).find(|&r| {
            valid[r] != 0
                && slot[r] >= 0
                && (slot[r] as u32) < page_size
                && page[r] as usize * page_rows + slot[r] as usize == cell
        });
        for c in 0..width {
            let (b, q) = (
                from_bf16(bf16_rows[cell * width + c]),
                e4m3(fp8_rows[cell * width + c]),
            );
            match landed {
                Some(r) => {
                    assert_eq!(b, product[r * width + c], "cell {cell} bf16");
                    assert_eq!(q, fp8.stored(product[r * width + c]), "cell {cell} e4m3");
                }
                None => assert!(b == 0.0 && q == 0.0, "cell {cell} landed nothing"),
            }
        }
    }
}
