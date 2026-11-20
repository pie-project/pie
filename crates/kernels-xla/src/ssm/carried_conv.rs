//! A causal conv1d decode step over a lane's committed window and its
//! carried (buffered, not yet folded) input rows: the conv half of the
//! state + buffer step (`new-memory/designs/lagged-fold.md`), as engine-cuda
//! runs it (`conv1d_carried`).
//!
//! For one lane, over positions `t` in `-span..=n` (`span = (K - 1) ·
//! dilation`, `n` its carried rows): `at(t)` is the committed window's cell
//! `span + t` for `t < 0`, carried row `t` for `t < n`, and the own row at
//! `t = n`. The own output is `silu(sum_k w[c, k] · at(n - (K - 1 - k) ·
//! dilation) + bias[c])`, and the window stored is `at(commit - span + s)`
//! for each `s` in `0..span`, the window at the lane's commit (never past a
//! row the guest has not accepted).

use crate::hlo::{Built, Cmp, Elem, Fold, Func, GatherDims, Val};

/// A step's inputs, f32. `lanes` lanes, `channels` channels, `rows` the
/// carried rows laid (at least each lane's).
#[derive(Clone, Copy, Debug)]
pub struct CarriedConv {
    /// The committed window `[lanes, span, channels]`.
    pub window: Val,
    /// The carried input rows `[lanes, rows, channels]`.
    pub carried: Val,
    /// The own row `[lanes, channels]`.
    pub own: Val,
    /// Each lane's carried rows and commit (`[lanes]` i32; `commit` in
    /// `0..=count + 1`).
    pub count: Val,
    pub commit: Val,
    /// The taps `[channels, K]` and the bias `[channels]`, if any.
    pub weight: Val,
    pub bias: Option<Val>,
}

/// The own row's output `[lanes, channels]` (after the SiLU) and the window
/// at the commit `[lanes, span, channels]`.
#[derive(Clone, Copy, Debug)]
pub struct ConvStepped {
    pub y: Val,
    pub window: Val,
}

/// `at(t)` for each lane and each position of `t` (`[lanes, m]` i32, in
/// `-span..=count`): the window, a carried row or the own row.
fn at(f: &mut Func, c: &CarriedConv, span: i64, t: Val) -> Built<Val> {
    let wd = f.dims(c.window).to_vec();
    let (lanes, channels) = (wd[0], wd[2]);
    let rows = f.dims(c.carried)[1];
    let m = f.dims(t)[1];
    // the window and the carried rows as one run: position t at span + t
    let run = f.concat(&[c.window, c.carried], 1)?; // [L, span + R, C]
    let span_v = f.const_i(Elem::I32, span, &[lanes, m]);
    let at = f.add(t, span_v)?;
    let lo = f.const_i(Elem::I32, 0, &[lanes, m]);
    let hi = f.const_i(Elem::I32, span + rows - 1, &[lanes, m]);
    let at = f.clamp(lo, at, hi)?;
    let lane = f.iota(Elem::I32, &[lanes, m], 0);
    let lane = f.reshape(lane, &[lanes, m, 1])?;
    let pos = f.reshape(at, &[lanes, m, 1])?;
    let idx = f.concat(&[lane, pos], 2)?;
    let picked = f.gather(
        run,
        idx,
        &GatherDims {
            offset_dims: vec![2],
            collapsed_slice_dims: vec![0, 1],
            start_index_map: vec![0, 1],
            index_vector_dim: 2,
            ..GatherDims::default()
        },
        &[1, 1, channels],
    )?; // [L, m, C]
    // the own row where t = count
    let n = f.broadcast(c.count, &[lanes, m], &[0])?;
    let own_at = f.compare(Cmp::Eq, t, n)?;
    let own_at = f.broadcast(own_at, &[lanes, m, channels], &[0, 1])?;
    let own = f.broadcast(c.own, &[lanes, m, channels], &[0, 2])?;
    f.select(own_at, own, picked)
}

/// Runs the step (module docs) over every lane.
pub fn conv_step_carried(f: &mut Func, c: &CarriedConv, dilation: i64) -> Built<ConvStepped> {
    let wd = f.dims(c.window).to_vec();
    let (lanes, span, channels) = (wd[0], wd[1], wd[2]);
    let taps = f.dims(c.weight)[1];
    let i = Elem::I32;
    // the own output: tap k reads at(n - (K - 1 - k) · dilation)
    let k = f.iota(i, &[lanes, taps], 1);
    let back = {
        let km = f.const_i(i, taps - 1, &[lanes, taps]);
        let lag = f.sub(km, k)?;
        let dil = f.const_i(i, dilation, &[lanes, taps]);
        f.mul(lag, dil)?
    };
    let n = f.broadcast(c.count, &[lanes, taps], &[0])?;
    let t = f.sub(n, back)?;
    let reads = at(f, c, span, t)?; // [L, K, C]
    let w = f.transpose(c.weight, &[1, 0])?; // [K, C]
    let w = f.broadcast(w, &[lanes, taps, channels], &[1, 2])?;
    let p = f.mul(reads, w)?;
    let mut y = f.reduce(p, &[1], Fold::Sum)?; // [L, C]
    if let Some(b) = c.bias {
        let b = f.broadcast(b, &[lanes, channels], &[1])?;
        y = f.add(y, b)?;
    }
    // SiLU
    let neg = f.neg(y);
    let e = f.exp(neg);
    let one = f.like_f(e, 1.0);
    let d = f.add(one, e)?;
    let y = f.div(y, d)?;
    // the window at the commit: at(commit - span + s)
    let s = f.iota(i, &[lanes, span], 1);
    let cm = f.broadcast(c.commit, &[lanes, span], &[0])?;
    let sp = f.const_i(i, span, &[lanes, span]);
    let t = f.sub(cm, sp)?;
    let t = f.add(t, s)?;
    let window = at(f, c, span, t)?;
    Ok(ConvStepped { y, window })
}
