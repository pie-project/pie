//! Lazy views of the chunk pool (`kernels_xla::hlo` `pool_view`): every op
//! the builder rewrites onto pool rows answers bit for bit what the same op
//! answers on the view materialized whole (`bitcast` -> `transpose` ->
//! `reshape`, the packed rows order) and packed back. Skipped without a
//! PJRT plugin.

use std::sync::Arc;

use engine_xla::pjrt::{Api, Arg, Client, ElementType};
use kernels_xla::hlo::{Built, Elem, Func, GatherDims, Ty, Val};

const ROWS: i64 = 64;

/// A case: builds outputs from the view and the i32 inputs.
type Case<'a> = &'a dyn Fn(&mut Func, Val, &[Val]) -> Built<Vec<Val>>;

fn client() -> Option<Client> {
    let api: Arc<Api> = match Api::discover(None) {
        Ok(api) => api,
        Err(e) => {
            eprintln!("skipped: {e}");
            return None;
        }
    };
    Some(Client::create(api).expect("client"))
}

fn per_word(elem: Elem) -> i64 {
    i64::from(32 / elem.bits())
}

/// The whole view, as plain ops.
fn whole(f: &mut Func, pool: Val, elem: Elem, dims: &[i64]) -> Built<Val> {
    let k = per_word(elem);
    let x = f.bitcast(pool, elem)?;
    let x = if k == 1 {
        x
    } else {
        f.transpose(x, &[0, 2, 1])?
    };
    f.reshape(x, dims)
}

/// A whole view back as the pool, as plain ops.
fn pack(f: &mut Func, x: Val, elem: Elem) -> Built<Val> {
    let k = per_word(elem);
    let x = if k == 1 {
        f.reshape(x, &[ROWS, 128])?
    } else {
        let x = f.reshape(x, &[ROWS, k, 128])?;
        f.transpose(x, &[0, 2, 1])?
    };
    f.bitcast(x, Elem::U32)
}

/// Bits of a value as u32 words, so results compare exactly.
fn bits(f: &mut Func, x: Val) -> Built<Val> {
    let n = f.ty(x).elements();
    let e = f.elem(x);
    let k = per_word(e);
    let x = f.reshape(x, &[n])?;
    if e == Elem::U32 {
        return Ok(x);
    }
    let x = f.bitcast(x, if k == 1 { Elem::U32 } else { e })?;
    if k == 1 {
        return Ok(x);
    }
    let pad = (k - n % k) % k;
    let x = if pad > 0 {
        let z = f.const_i(Elem::U8, 0, &[]);
        let z = f.bitcast(z, e)?;
        f.pad(x, z, &[0], &[pad], &[0])?
    } else {
        x
    };
    let x = f.reshape(x, &[(n + pad) / k, k])?;
    f.bitcast(x, Elem::U32)
}

/// Runs `case` twice over the same pool: once on a lazy view, once on the
/// view built whole; the results (read values, and the pool after writes)
/// must agree bit for bit.
fn check(name: &str, elem: Elem, dims: &[i64], ints: &[Vec<i32>], case: Case<'_>) {
    let Some(client) = client() else { return };
    let dev = client.device(0).expect("a device");
    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let pool: Vec<u8> = (0..ROWS * 128)
        .flat_map(|_| {
            seed = seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // normal, finite floats at every width (a subnormal may flush
            // in one fusion and not another)
            let w = ((seed >> 32) as u32 & 0x3F3F_3F3F) | 0x2020_2020;
            w.to_le_bytes()
        })
        .collect();
    let run = |lazy: bool| -> Vec<Vec<u8>> {
        let mut f = Func::new("main");
        let p = f.param(Ty::new(Elem::U32, &[ROWS, 128]), None);
        let args: Vec<Val> = ints
            .iter()
            .map(|v| f.param(Ty::new(Elem::I32, &[v.len() as i64]), None))
            .collect();
        let v = if lazy {
            f.pool_view(p, elem, dims).unwrap()
        } else {
            whole(&mut f, p, elem, dims).unwrap()
        };
        let outs = case(&mut f, v, &args).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let outs: Vec<Val> = outs
            .into_iter()
            .map(|o| {
                if f.ty(o).dims == dims && f.elem(o) == elem && (lazy || !f.is_view(o)) {
                    let pool = if lazy {
                        f.pool_of(o, elem).unwrap()
                    } else {
                        pack(&mut f, o, elem).unwrap()
                    };
                    bits(&mut f, pool).unwrap()
                } else {
                    bits(&mut f, o).unwrap()
                }
            })
            .collect();
        if lazy {
            assert!(
                f.materialized().is_empty(),
                "{name}: materialized {:?}",
                f.materialized()
            );
        }
        let text = f.module("view", &outs);
        let exe = client
            .compile(&text)
            .unwrap_or_else(|e| panic!("{name}: {e}\n{text}"));
        let pb = client
            .upload(dev, &pool, ElementType::U32, &[ROWS, 128])
            .unwrap();
        let ibs: Vec<_> = ints
            .iter()
            .map(|v| {
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                client
                    .upload(dev, &b, ElementType::S32, &[v.len() as i64])
                    .unwrap()
            })
            .collect();
        let mut a = vec![Arg::Keep(&pb)];
        a.extend(ibs.iter().map(Arg::Keep));
        let (outs, done) = exe.execute(dev, a).unwrap();
        done.wait().unwrap();
        outs.iter().map(|o| o.download().unwrap()).collect()
    };
    let lazy = run(true);
    let plain = run(false);
    for (i, (a, b)) in lazy.iter().zip(&plain).enumerate() {
        let words = |x: &[u8]| -> Vec<u32> {
            x.chunks(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        };
        let (wa, wb) = (words(a), words(b));
        let bad: Vec<usize> = (0..wa.len()).filter(|&j| wa[j] != wb[j]).collect();
        assert!(
            bad.is_empty(),
            "{name}: output {i}: {} of {} words differ, first {:?}; lazy {:08x?} plain {:08x?}",
            bad.len(),
            wa.len(),
            &bad[..bad.len().min(8)],
            bad.iter().take(4).map(|&j| wa[j]).collect::<Vec<_>>(),
            bad.iter().take(4).map(|&j| wb[j]).collect::<Vec<_>>()
        );
    }
    eprintln!("{name}: {} outputs bit-identical", lazy.len());
}

fn scalar(f: &mut Func, v: Val, i: i64) -> Built<Val> {
    let s = f.slice(v, &[i], &[i + 1], &[1])?;
    f.reshape(s, &[])
}

#[test]
fn rows_read_by_ids() {
    let ids = vec![vec![0, 1, 5, 6, 127, 64, -3, 100_000, 33]];
    for (elem, w) in [
        (Elem::Bf16, 128),
        (Elem::Bf16, 64),
        (Elem::Bf16, 512),
        (Elem::F32, 128),
        (Elem::F32, 32),
        (Elem::F8E4m3fn, 128),
        (Elem::F8E4m3fn, 256),
    ] {
        let n = ROWS * 128 * per_word(elem) / w;
        check(
            &format!("take_rows {elem:?} w{w}"),
            elem,
            &[n, w],
            &ids,
            &|f, v, a| Ok(vec![f.take_rows(v, a[0])?]),
        );
    }
}

#[test]
fn pages_read_by_ids() {
    // [pages, T, w] with T·w a page; pages through a gather of [1, T, w].
    for (elem, t, w) in [
        (Elem::Bf16, 16, 128),
        (Elem::Bf16, 4, 64),
        (Elem::F32, 2, 64),
    ] {
        let pages = ROWS * 128 * per_word(elem) / (t * w);
        let ids = vec![vec![0, 3, 1, (pages - 1) as i32, 99_999, -1]];
        check(
            &format!("pages {elem:?} {t}x{w}"),
            elem,
            &[pages, t, w],
            &ids,
            &|f, v, a| {
                let n = f.dims(a[0])[0];
                let idx = f.reshape(a[0], &[n, 1])?;
                Ok(vec![f.gather(
                    v,
                    idx,
                    &GatherDims {
                        offset_dims: vec![1, 2],
                        collapsed_slice_dims: vec![0],
                        start_index_map: vec![0],
                        index_vector_dim: 1,
                        ..GatherDims::default()
                    },
                    &[1, t, w],
                )?])
            },
        );
    }
}

#[test]
fn row_ranges_at_any_start() {
    for (elem, w, n) in [
        (Elem::Bf16, 128, 9),
        (Elem::Bf16, 128, 16),
        (Elem::Bf16, 64, 7),
        (Elem::F32, 128, 5),
    ] {
        let rows = ROWS * 128 * per_word(elem) / w;
        let starts = vec![vec![0, 1, 3, (rows - 2) as i32]];
        check(
            &format!("dynamic_slice {elem:?} w{w} n{n}"),
            elem,
            &[rows, w],
            &starts,
            &|f, v, a| {
                let z = f.const_i(Elem::I32, 0, &[]);
                let mut out = Vec::new();
                for i in 0..4 {
                    let s = scalar(f, a[0], i)?;
                    out.push(f.dynamic_slice(v, &[s, z], &[n, w])?);
                }
                out.push(f.slice(v, &[3, 0], &[3 + n, w], &[1, 1])?);
                Ok(out)
            },
        );
    }
}

#[test]
fn rows_written_by_ids() {
    // Distinct ids, pairs sharing a pool row, one out of range (dropped).
    let ids = vec![vec![4, 5, 0, 9, 126, 127, 100_000, 31]];
    for (elem, w) in [
        (Elem::Bf16, 128),
        (Elem::Bf16, 64),
        (Elem::Bf16, 256),
        (Elem::Bf16, 512),
        (Elem::F32, 128),
        (Elem::F32, 64),
        (Elem::F8E4m3fn, 128),
    ] {
        let n = ROWS * 128 * per_word(elem) / w;
        check(
            &format!("put_rows {elem:?} w{w}"),
            elem,
            &[n, w],
            &ids,
            &|f, v, a| {
                let k = f.dims(a[0])[0];
                let src = f.slice(v, &[n - k, 0], &[n, w], &[1, 1])?;
                // fresh values: the rows at the end, negated
                let src = f.neg(src);
                Ok(vec![f.put_rows(
                    v,
                    a[0],
                    src,
                    kernels_xla::hlo::Combine::Set,
                )?])
            },
        );
    }
}

#[test]
fn row_ranges_written_at_any_start() {
    for (elem, w, n) in [
        (Elem::Bf16, 128, 9),
        (Elem::Bf16, 128, 16),
        (Elem::Bf16, 128, 1),
        (Elem::Bf16, 64, 7),
        (Elem::F32, 128, 3),
    ] {
        let rows = ROWS * 128 * per_word(elem) / w;
        for start in [0, 1, 3, (rows - n) as i32, rows as i32] {
            let starts = vec![vec![start]];
            check(
                &format!("dus {elem:?} w{w} n{n} at {start}"),
                elem,
                &[rows, w],
                &starts,
                &|f, v, a| {
                    let z = f.const_i(Elem::I32, 0, &[]);
                    let s = scalar(f, a[0], 0)?;
                    let src = f.slice(v, &[0, 0], &[n, w], &[1, 1])?;
                    let src = f.neg(src);
                    Ok(vec![f.dynamic_update_slice(v, src, &[s, z])?])
                },
            );
        }
    }
}

#[test]
fn a_loop_and_a_case_carry_a_view() {
    let ids = vec![vec![2, 7, 11]];
    let (elem, w) = (Elem::Bf16, 128);
    let n = ROWS * 256 / w;
    check("loop", elem, &[n, w], &ids, &|f, v, a| {
        let ids = a[0];
        let out = f.for_loop(3, &[v], |f, i, c| {
            let one = f.reshape(i, &[1])?;
            let id = f.dynamic_slice(ids, &[i], &[1])?;
            let _ = one;
            let row = f.take_rows(c[0], id)?;
            let row = f.neg(row);
            let plus = f.const_i(Elem::I32, 1, &[1]);
            let id = f.add(id, plus)?;
            Ok(vec![f.put_rows(
                c[0],
                id,
                row,
                kernels_xla::hlo::Combine::Set,
            )?])
        })?;
        let sel = f.const_i(Elem::I32, 1, &[]);
        let ty = f.ty(out[0]).clone();
        let base = out[0];
        let picked = f.case(
            sel,
            &[ty],
            vec![
                Box::new(move |_f: &mut Func| Ok(vec![base])),
                Box::new(move |f: &mut Func| {
                    let z = f.const_i(Elem::I32, 0, &[]);
                    let at = f.const_i(Elem::I32, 5, &[]);
                    let src = f.slice(base, &[0, 0], &[3, w], &[1, 1])?;
                    Ok(vec![f.dynamic_update_slice(base, src, &[at, z])?])
                }),
            ],
        )?;
        Ok(vec![picked[0]])
    });
}

#[test]
fn rows_landed_as_the_lanes_land_them() {
    // `lanes::land_rows`: dropped lanes at `i32::MAX - lane`, the rest
    // distinct, hinted unique.
    let max = i32::MAX;
    let ids = vec![vec![6, max - 1, 7, 2, max - 4, 3, 40, 41]];
    for (elem, w) in [(Elem::Bf16, 128), (Elem::Bf16, 64), (Elem::Bf16, 512)] {
        let n = ROWS * 128 * per_word(elem) / w;
        check(
            &format!("land_rows {elem:?} w{w}"),
            elem,
            &[n, w],
            &ids,
            &|f, v, a| {
                let k = f.dims(a[0])[0];
                let src = f.slice(v, &[n - k, 0], &[n, w], &[1, 1])?;
                let src = f.neg(src);
                Ok(vec![f.put_rows_hinted(
                    v,
                    a[0],
                    src,
                    kernels_xla::hlo::Combine::Set,
                    true,
                )?])
            },
        );
    }
}

#[test]
fn rows_landed_twice_keep_the_last() {
    // Padding lanes of a prompt all land on one (null) page: a set scatter
    // lets the last win, and so must a block narrower than a pool row.
    let ids = vec![vec![0, 0, 5, 0, 9, 5, 0, 3]];
    for (elem, w) in [(Elem::Bf16, 128), (Elem::Bf16, 64), (Elem::F8E4m3fn, 128)] {
        let n = ROWS * 128 * per_word(elem) / w;
        check(
            &format!("land twice {elem:?} w{w}"),
            elem,
            &[n, w],
            &ids,
            &|f, v, a| {
                let k = f.dims(a[0])[0];
                let src = f.slice(v, &[n - k, 0], &[n, w], &[1, 1])?;
                let src = f.neg(src);
                Ok(vec![f.put_rows(
                    v,
                    a[0],
                    src,
                    kernels_xla::hlo::Combine::Set,
                )?])
            },
        );
    }
}
