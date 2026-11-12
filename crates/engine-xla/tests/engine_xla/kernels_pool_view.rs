//! Lazy views of the chunk pool (`kernels_xla::hlo` `pool_view`): every op
//! the builder rewrites onto pool rows answers bit for bit what the same op
//! answers on the view materialized whole (`bitcast` -> `transpose` ->
//! `reshape`, the packed rows order) and packed back. Skipped without a
//! PJRT plugin.

use std::sync::Arc;

use engine_xla::pjrt::{Api, Arg, Client, ElementType};
use kernels_xla::hlo::{Built, Elem, Func, GatherDims, PoolOrder, Ty, Val};

const ROWS: i64 = 64;

thread_local! {
    /// Whether [`check`] lays its views in the kv row-pair order where their
    /// shape allows one ([`order_for`]).
    static ORDERED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// The row-pair order of 16-token blocks a view of `dims` takes under
/// [`ORDERED`]: bf16 rows at least two pool rows wide.
fn order_for(elem: Elem, dims: &[i64]) -> Option<PoolOrder> {
    if !ORDERED.with(std::cell::Cell::get) || elem != Elem::Bf16 {
        return None;
    }
    let w = *dims.last()?;
    let rpt = w * 2 / 512;
    (w * 2 % 512 == 0 && rpt >= 2 && ROWS % (16 * rpt) == 0)
        .then_some(PoolOrder { tokens: 16, rpt })
}

/// Pool rows `[ROWS, 128]` of whole blocks in view order (`to_view`) or
/// back, as plain ops.
fn reorder(f: &mut Func, rows: Val, o: PoolOrder, to_view: bool) -> Built<Val> {
    let (a, b) = if to_view {
        (o.rpt, o.tokens)
    } else {
        (o.tokens, o.rpt)
    };
    let x = f.reshape(rows, &[ROWS / (o.tokens * o.rpt), a, b, 128])?;
    let x = f.transpose(x, &[0, 2, 1, 3])?;
    f.reshape(x, &[ROWS, 128])
}

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
    let pool = match order_for(elem, dims) {
        Some(o) => reorder(f, pool, o, true)?,
        None => pool,
    };
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
    let order = order_for(elem, f.dims(x));
    let x = if k == 1 {
        f.reshape(x, &[ROWS, 128])?
    } else {
        let x = f.reshape(x, &[ROWS, k, 128])?;
        f.transpose(x, &[0, 2, 1])?
    };
    let x = f.bitcast(x, Elem::U32)?;
    match order {
        Some(o) => reorder(f, x, o, false),
        None => Ok(x),
    }
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
            f.pool_view_ordered(p, elem, dims, order_for(elem, dims))
                .unwrap()
        } else {
            whole(&mut f, p, elem, dims).unwrap()
        };
        let outs = case(&mut f, v, &args).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let outs: Vec<Val> = outs
            .into_iter()
            .map(|o| {
                if f.ty(o).dims == dims && f.elem(o) == elem && (lazy || !f.is_view(o)) {
                    let pool = if lazy {
                        f.pool_of_ordered(o, elem, order_for(elem, dims)).unwrap()
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
        (Elem::Bf16, 16, 512),
        (Elem::Bf16, 16, 1024),
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

/// A word of the large pool, by place.
fn word(r: i64, l: i64) -> u32 {
    let x = (r as u64)
        .wrapping_mul(0x9E37_79B9)
        .wrapping_add(l as u64 * 0x85EB_CA6B);
    // normal bf16 halves
    ((x as u32) & 0x3F3F_3F3F) | 0x2020_2020
}

fn read_pages_of(rows: i64, w: i64, ids: &[i32]) {
    // 3 GiB: block reads past 2 GiB (the DMA path on a TPU) land the bits
    // the packed rows order puts there.
    let Some(client) = client() else { return };
    let dev = client.device(0).expect("a device");
    let t = 16i64;
    let per = t * w * 2 / 512; // pool rows a page
    let pages = rows / per;
    let mut f = Func::new("main");
    let p = f.param(Ty::new(Elem::U32, &[rows, 128]), None);
    let at = f.param(Ty::new(Elem::I32, &[ids.len() as i64]), None);
    let v = f.pool_view(p, Elem::Bf16, &[pages, t, w]).unwrap();
    let idx = f.reshape(at, &[ids.len() as i64, 1]).unwrap();
    let got = f
        .gather(
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
        )
        .unwrap();
    let text = f.module("far", &[got]);
    let exe = client.compile(&text).unwrap_or_else(|e| panic!("{e}"));
    let mut host = Vec::with_capacity((rows * 512) as usize);
    for r in 0..rows {
        for l in 0..128 {
            host.extend_from_slice(&word(r, l).to_le_bytes());
        }
    }
    let pb = client
        .upload(dev, &host, ElementType::U32, &[rows, 128])
        .unwrap();
    drop(host);
    let ib: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ib = client
        .upload(dev, &ib, ElementType::S32, &[ids.len() as i64])
        .unwrap();
    let (outs, done) = exe
        .execute(dev, vec![Arg::Keep(&pb), Arg::Keep(&ib)])
        .unwrap();
    done.wait().unwrap();
    let out = outs[0].download().unwrap();
    // element e of page p: pool row p·per + e / 256, half (e / 128) % 2,
    // lane e % 128
    let mut bad = 0usize;
    for (b, &p) in ids.iter().enumerate() {
        // a gather clamps its start into the view
        let p = i64::from(p).clamp(0, pages - 1);
        for e in 0..t * w {
            let r = p * per + e / 256;
            let half = (e / 128) % 2;
            let want = (word(r, e % 128) >> (16 * half)) as u16;
            let at = ((b as i64 * t * w + e) * 2) as usize;
            let got = u16::from_le_bytes([out[at], out[at + 1]]);
            if got != want {
                bad += 1;
            }
        }
    }
    assert_eq!(bad, 0, "elements read wrong ({rows} rows, {w} wide)");
}

#[test]
fn pages_read_far_into_a_large_pool() {
    let rows: i64 = 6 << 20;
    let pages = rows / 64;
    read_pages_of(
        rows,
        1024,
        &[
            0,
            1,
            (pages / 2) as i32,
            (pages * 3 / 4) as i32,
            (pages - 1) as i32,
        ],
    );
}

#[test]
fn pages_read_as_a_routed_model_reads_them() {
    // the routed mini's kv plane at a 1 GiB pool: pages of 16 x 512, padded
    // ids clamped
    let rows: i64 = 2_138_112;
    let pages = rows / 32;
    let mut ids: Vec<i32> = (0..24).map(|i| i * 7).collect();
    ids.extend([
        i32::MAX,
        i32::MAX - 1,
        -1,
        -5,
        (pages - 1) as i32,
        pages as i32,
    ]);
    read_pages_of(rows, 512, &ids);
}

#[test]
fn pages_written_far_into_a_large_pool() {
    // 3 GiB: block writes past 2 GiB (the DMA path on a TPU) land where the
    // packed rows order puts them, and nowhere else.
    let Some(client) = client() else { return };
    let dev = client.device(0).expect("a device");
    let rows: i64 = 6 << 20;
    let (t, w) = (16i64, 1024i64);
    let per = t * w * 2 / 512;
    let pages = rows / per;
    let ids: Vec<i32> = vec![
        1,
        (pages / 2) as i32,
        (pages * 3 / 4) as i32,
        (pages - 1) as i32,
    ];
    let n = ids.len() as i64;
    let mut f = Func::new("main");
    let p = f.param(Ty::new(Elem::U32, &[rows, 128]), Some(0));
    let at = f.param(Ty::new(Elem::I32, &[n]), None);
    let v = f.pool_view(p, Elem::Bf16, &[pages, t, w]).unwrap();
    // fresh rows: page b all `0x3F81 + b` as bf16 bits (normal floats)
    let fill = f.iota(Elem::I32, &[n, t, w], 0);
    let one = f.like_i(fill, 0x3F81);
    let fill = f.add(fill, one).unwrap();
    let fill = f.convert(fill, Elem::U16);
    let fill = f.bitcast(fill, Elem::Bf16).unwrap();
    let idx = f.reshape(at, &[n, 1]).unwrap();
    let out = f
        .scatter_hinted(
            v,
            idx,
            fill,
            &kernels_xla::hlo::ScatterDims {
                update_window_dims: vec![1, 2],
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..kernels_xla::hlo::ScatterDims::default()
            },
            kernels_xla::hlo::Combine::Set,
            true,
        )
        .unwrap();
    let pool = f.pool_of(out, Elem::Bf16).unwrap();
    let text = f.module("far", &[pool]);
    let exe = client.compile(&text).unwrap_or_else(|e| panic!("{e}"));
    let mut host = Vec::with_capacity((rows * 512) as usize);
    for r in 0..rows {
        for l in 0..128 {
            host.extend_from_slice(&word(r, l).to_le_bytes());
        }
    }
    let pb = client
        .upload(dev, &host, ElementType::U32, &[rows, 128])
        .unwrap();
    let ib: Vec<u8> = ids.iter().flat_map(|x| x.to_le_bytes()).collect();
    let ib = client.upload(dev, &ib, ElementType::S32, &[n]).unwrap();
    let (outs, done) = exe
        .execute(dev, vec![Arg::Donate(pb), Arg::Keep(&ib)])
        .unwrap();
    done.wait().unwrap();
    let got = outs[0].download().unwrap();
    let mut written = vec![None; pages as usize];
    for (b, &p) in ids.iter().enumerate() {
        written[p as usize] = Some(b as u32 + 0x3F81);
    }
    let mut bad = Vec::new();
    for r in 0..rows {
        let page = (r / per) as usize;
        for l in 0..128 {
            let want = match written[page] {
                Some(x) => x | (x << 16),
                None => word(r, l),
            };
            let at = ((r * 128 + l) * 4) as usize;
            let g = u32::from_le_bytes([got[at], got[at + 1], got[at + 2], got[at + 3]]);
            if g != want && bad.len() < 4 {
                bad.push((r, l, g, want));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "pool words wrong (row, lane, got, want): {bad:x?}"
    );
}

#[test]
fn rows_landed_then_their_pages_read_in_one_fire() {
    // A decode on a 1 GiB pool: token rows (2 pool rows each, 512 wide)
    // landed by block, then their pages read back by block, in one program.
    let Some(client) = client() else { return };
    let dev = client.device(0).expect("a device");
    let rows: i64 = 2_138_112;
    let (t, w) = (16i64, 512i64);
    let per = t * w * 2 / 512;
    let pages = rows / per;
    let page_ids: Vec<i32> = vec![3, 70, (pages - 2) as i32];
    // one token at row 5 of each page
    let tok: Vec<i32> = page_ids.iter().map(|&p| p * t as i32 + 5).collect();
    let n = tok.len() as i64;
    let mut f = Func::new("main");
    let p = f.param(Ty::new(Elem::U32, &[rows, 128]), Some(0));
    let at = f.param(Ty::new(Elem::I32, &[n]), None);
    let pg = f.param(Ty::new(Elem::I32, &[n]), None);
    let flat = f.pool_view(p, Elem::Bf16, &[pages * t, w]).unwrap();
    let fill = f.iota(Elem::I32, &[n, w], 0);
    let base = f.like_i(fill, 0x3F81);
    let fill = f.add(fill, base).unwrap();
    let fill = f.convert(fill, Elem::U16);
    let fill = f.bitcast(fill, Elem::Bf16).unwrap();
    let flat = f
        .put_rows_hinted(flat, at, fill, kernels_xla::hlo::Combine::Set, true)
        .unwrap();
    let paged = f.reshape(flat, &[pages, t, w]).unwrap();
    let idx = f.reshape(pg, &[n, 1]).unwrap();
    let got = f
        .gather(
            paged,
            idx,
            &GatherDims {
                offset_dims: vec![1, 2],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                ..GatherDims::default()
            },
            &[1, t, w],
        )
        .unwrap();
    let pool = f.pool_of(paged, Elem::Bf16).unwrap();
    let text = f.module("land_read", &[pool, got]);
    let exe = client.compile(&text).unwrap_or_else(|e| panic!("{e}"));
    let host: Vec<u8> = (0..rows * 128)
        .flat_map(|i| word(i / 128, i % 128).to_le_bytes())
        .collect();
    let pb = client
        .upload(dev, &host, ElementType::U32, &[rows, 128])
        .unwrap();
    let up = |v: &[i32]| {
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        client
            .upload(dev, &b, ElementType::S32, &[v.len() as i64])
            .unwrap()
    };
    let (ab, gb) = (up(&tok), up(&page_ids));
    let (outs, done) = exe
        .execute(dev, vec![Arg::Donate(pb), Arg::Keep(&ab), Arg::Keep(&gb)])
        .unwrap();
    done.wait().unwrap();
    let out = outs[1].download().unwrap();
    let mut bad = Vec::new();
    for (b, &p) in page_ids.iter().enumerate() {
        for e in 0..t * w {
            let (row, j) = (e / w, e % w);
            let want = if row == 5 {
                0x3F81 + b as u16
            } else {
                let r = i64::from(p) * per + e / 256;
                (word(r, e % 128) >> (16 * ((e / 128) % 2))) as u16
            };
            let at = ((b as i64 * t * w + e) * 2) as usize;
            let g = u16::from_le_bytes([out[at], out[at + 1]]);
            if g != want && bad.len() < 6 {
                bad.push((b, row, j, g, want));
            }
        }
    }
    assert!(bad.is_empty(), "(page, row, col, got, want): {bad:x?}");
}

#[test]
fn a_decode_conv_rolls_its_windows_in_the_pool() {
    // ssm::causal_conv1d on a pool view (the Mosaic conv step on a TPU):
    // each lane's window rolled by its row lands where the lane writes, the
    // rest of the pool stands; the output agrees with a host reference.
    use dtype::Dtype;
    use kernels_xla::{Landing, Operand};
    let Some(client) = client() else { return };
    let dev = client.device(0).expect("a device");
    let rows: i64 = 4096;
    let (c, taps, t) = (512i64, 4i64, 8i64);
    let hist = taps; // dilation 1
    let width = c;
    let windows = rows * 2 * 128 / (hist * c);
    let slots: Vec<i32> = vec![3, 9, 0, 17, 5, 40, 41, 2];
    let mut seed = 7u64;
    let mut next = || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        ((seed >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let bf = |x: f32| (x.to_bits() >> 16) as u16;
    let unbf = |h: u16| f32::from_bits(u32::from(h) << 16);
    // the pool as bf16 elements in flat order, packed into rows
    let flat: Vec<u16> = (0..rows * 256).map(|_| bf(next())).collect();
    let mut pool = vec![0u8; (rows * 512) as usize];
    for r in 0..rows as usize {
        for l in 0..128 {
            let lo = flat[r * 256 + l];
            let hi = flat[r * 256 + 128 + l];
            pool[(r * 128 + l) * 4..(r * 128 + l) * 4 + 2].copy_from_slice(&lo.to_le_bytes());
            pool[(r * 128 + l) * 4 + 2..(r * 128 + l) * 4 + 4].copy_from_slice(&hi.to_le_bytes());
        }
    }
    let x: Vec<u16> = (0..t * c).map(|_| bf(next())).collect();
    let w: Vec<f32> = (0..c * taps).map(|_| next()).collect();
    let mut f = Func::new("main");
    let p = f.param(Ty::new(Elem::U32, &[rows, 128]), None);
    let xv = f.param(Ty::new(Elem::Bf16, &[t, c]), None);
    let wv = f.param(Ty::new(Elem::F32, &[c, taps]), None);
    let sv = f.param(Ty::new(Elem::I32, &[t]), None);
    let slab = f
        .pool_view(p, Elem::Bf16, &[rows * 256 / width, width])
        .unwrap();
    let op = |val, rows: i64, width: i64, dtype| Operand {
        val,
        rows: rows as u32,
        width: width as u32,
        dtype,
    };
    let (y, slab) = kernels_xla::ssm::causal_conv1d(
        &mut f,
        op(xv, t, c, Dtype::Bf16),
        op(wv, c, taps, Dtype::F32),
        None,
        op(slab, rows * 256 / width, width, Dtype::Bf16),
        op(sv, t, 1, Dtype::I32),
        taps as u32,
        1,
        Landing {
            rows: t as u32,
            width: c as u32,
            dtype: Dtype::F32,
        },
    )
    .unwrap();
    let pool_out = f.pool_of(slab, Elem::Bf16).unwrap();
    let y = f.convert(y, Elem::F32);
    // The XLA form's own arithmetic over the same windows (read by XLA's
    // gather off the pool as the plain view), for a bitwise comparison.
    let whole = {
        let x = f.bitcast(p, Elem::Bf16).unwrap();
        let x = f.transpose(x, &[0, 2, 1]).unwrap();
        f.reshape(x, &[rows * 256 / (hist * c), hist * c]).unwrap()
    };
    let past = f.take_rows(whole, sv).unwrap();
    let x32 = f.convert(xv, Elem::F32);
    let mut acc: Option<kernels_xla::hlo::Val> = None;
    for k in 0..taps {
        let pos = hist - 1 - (taps - 1 - k);
        let tap = if pos + 1 == hist {
            x32
        } else {
            let s = f.slice_axis(past, 1, (pos + 1) * c, (pos + 2) * c).unwrap();
            f.convert(s, Elem::F32)
        };
        let wk = f.slice_axis(wv, 1, k, k + 1).unwrap();
        let wk = f.reshape(wk, &[c]).unwrap();
        let wk = f.broadcast(wk, &[t, c], &[1]).unwrap();
        let term = f.mul(tap, wk).unwrap();
        acc = Some(match acc {
            None => term,
            Some(a) => f.add(a, term).unwrap(),
        });
    }
    let yx = f.silu(acc.unwrap()).unwrap();
    let text = f.module("conv", &[y, pool_out, yx]);
    let exe = client.compile(&text).unwrap_or_else(|e| panic!("{e}"));
    let up = |b: &[u8], ty, d: &[i64]| client.upload(dev, b, ty, d).unwrap();
    let pb = up(&pool, ElementType::U32, &[rows, 128]);
    let xb: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
    let xb = up(&xb, ElementType::Bf16, &[t, c]);
    let wb: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    let wb = up(&wb, ElementType::F32, &[c, taps]);
    let sb: Vec<u8> = slots.iter().flat_map(|v| v.to_le_bytes()).collect();
    let sb = up(&sb, ElementType::S32, &[t]);
    let (outs, done) = exe
        .execute(
            dev,
            vec![
                Arg::Keep(&pb),
                Arg::Keep(&xb),
                Arg::Keep(&wb),
                Arg::Keep(&sb),
            ],
        )
        .unwrap();
    done.wait().unwrap();
    let yo = outs[0].download().unwrap();
    let po = outs[1].download().unwrap();
    let yx = outs[2].download().unwrap();
    let same = yo == yx;
    let ndiff = yo
        .chunks(4)
        .zip(yx.chunks(4))
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "conv: the kernel's output = the XLA form's over the same windows: {same} ({ndiff} differ)"
    );
    let elem = |bytes: &[u8], e: usize| -> u16 {
        let r = e / 256;
        let (l, h) = ((e % 256) % 128, (e % 256) / 128);
        let at = (r * 128 + l) * 4 + 2 * h;
        u16::from_le_bytes([bytes[at], bytes[at + 1]])
    };
    let win = (hist * c) as usize;
    let mut worst = 0f32;
    let mut bad_state = Vec::new();
    for (lane, &s) in slots.iter().enumerate() {
        let base = s as usize * win;
        let past: Vec<f32> = (0..win).map(|e| unbf(flat[base + e])).collect();
        for ch in 0..c as usize {
            let mut acc = 0f32;
            for k in 0..taps as usize {
                let pos = hist as usize - 1 - (taps as usize - 1 - k);
                let tap = if pos + 1 == hist as usize {
                    unbf(x[lane * c as usize + ch])
                } else {
                    past[(pos + 1) * c as usize + ch]
                };
                let term = tap * w[ch * taps as usize + k];
                acc = if k == 0 { term } else { acc + term };
            }
            let want = acc / (1.0 + (-acc).exp());
            let at = (lane * c as usize + ch) * 4;
            let got = f32::from_le_bytes([yo[at], yo[at + 1], yo[at + 2], yo[at + 3]]);
            worst = worst.max((got - want).abs());
        }
        // the window rolled: rows 1.. then x
        for e in 0..win {
            let want = if e < (hist as usize - 1) * c as usize {
                flat[base + c as usize + e]
            } else {
                x[lane * c as usize + e - (hist as usize - 1) * c as usize]
            };
            let got = elem(&po, base + e);
            if got != want && bad_state.len() < 4 {
                bad_state.push((lane, e, got, want));
            }
        }
    }
    let untouched = (0..(rows * 256) as usize)
        .filter(|e| !slots.iter().any(|&s| (e / win) as i32 == s))
        .all(|e| elem(&po, e) == flat[e]);
    eprintln!(
        "conv: output within {worst} of the host's; windows {}",
        windows
    );
    assert!(
        bad_state.is_empty(),
        "windows rolled wrong (lane, elem, got, want): {bad_state:x?}"
    );
    assert!(
        untouched,
        "the conv touched the pool outside its lanes' windows"
    );
    assert!(worst < 1e-3, "output off by {worst}");
}

/// The view tests again with every view that can take it in the kv
/// row-pair order (`kernels_xla::hlo::PoolOrder`): each op answers what it
/// answers on the view materialized in that order.
#[test]
fn views_in_the_row_pair_order_answer_alike() {
    ORDERED.with(|o| o.set(true));
    rows_read_by_ids();
    pages_read_by_ids();
    row_ranges_at_any_start();
    rows_written_by_ids();
    row_ranges_written_at_any_start();
    a_loop_and_a_case_carry_a_view();
    rows_landed_as_the_lanes_land_them();
    rows_landed_twice_keep_the_last();
    pages_and_tokens_written();
    ORDERED.with(|o| o.set(false));
}

/// Whole pages and single tokens of `[pages, 16, w]` landed by scatter, then
/// read back by gather: a kv plane's writes and reads.
#[test]
fn pages_and_tokens_written() {
    for w in [128, 512, 1024] {
        let pages = ROWS * 256 / (16 * w);
        let at = vec![
            vec![(pages - 1) as i32, 0],
            vec![0, 5, 15, 9],
            vec![(pages - 1) as i32, 0, 0, (pages - 1) as i32],
        ];
        check(
            &format!("pages and tokens written w{w}"),
            Elem::Bf16,
            &[pages, 16, w],
            &at,
            &|f, v, a| {
                let page_dims = GatherDims {
                    offset_dims: vec![1, 2],
                    collapsed_slice_dims: vec![0],
                    start_index_map: vec![0],
                    index_vector_dim: 1,
                    ..GatherDims::default()
                };
                // read page 0 and the last page whole, write them swapped
                let idx = f.reshape(a[0], &[2, 1])?;
                let got = f.gather(v, idx, &page_dims, &[1, 16, w])?;
                let rev = f.reverse(a[0], &[0]);
                let ridx = f.reshape(rev, &[2, 1])?;
                let v2 = f.scatter(
                    v,
                    ridx,
                    got,
                    &kernels_xla::hlo::ScatterDims {
                        update_window_dims: vec![1, 2],
                        inserted_window_dims: vec![0],
                        scatter_dims_to_operand_dims: vec![0],
                        index_vector_dim: 1,
                        ..Default::default()
                    },
                    kernels_xla::hlo::Combine::Set,
                )?;
                // tokens (page, slot): rows of the first page written over
                let tok = f.gather(v2, idx, &page_dims, &[1, 16, w])?;
                let rows = f.slice(tok, &[0, 0, 0], &[1, 4, w], &[1, 1, 1])?;
                let rows = f.reshape(rows, &[4, w])?;
                let pt = f.reshape(a[2], &[4, 1])?;
                let st = f.reshape(a[1], &[4, 1])?;
                let ix = f.concat(&[pt, st], 1)?;
                let v3 = f.scatter(
                    v2,
                    ix,
                    rows,
                    &kernels_xla::hlo::ScatterDims {
                        update_window_dims: vec![1],
                        inserted_window_dims: vec![0, 1],
                        scatter_dims_to_operand_dims: vec![0, 1],
                        index_vector_dim: 1,
                        ..Default::default()
                    },
                    kernels_xla::hlo::Combine::Set,
                )?;
                let back = f.gather(v3, idx, &page_dims, &[1, 16, w])?;
                let got = f.reshape(got, &[32, w])?;
                let back = f.reshape(back, &[32, w])?;
                Ok(vec![got, back, v3])
            },
        );
    }
}
