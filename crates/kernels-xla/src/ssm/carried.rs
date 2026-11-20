//! A gated-delta decode step over a lane's committed state and its carried
//! (buffered, not yet folded) rows: the state + buffer step
//! (`new-memory/designs/lagged-fold.md`, "The decode replay kernel"), as
//! one HLO region. The carried rows are read as a linear attention with
//! their cached corrected values `u`; the state is folded only through the
//! lane's commit, never past a row the guest has not accepted.
//!
//! For one lane and value head (key head `h / group`), with the committed
//! state `S` (`[K, V]`), carried rows `0..n` (raw keys `k_s`, corrected
//! values `u_s`, log decays `g_s`) and the own row (`q`, `k`, `v`, `g_n`,
//! `beta`), keys and queries L2-normalised (`x / sqrt(sum x^2 + 1e-6)`, the
//! query times `q_scale`):
//!
//! - `G_s = exp(g_n + sum_{s<i<n} g_i)`, `g0 = exp(g_n + sum_{i<n} g_i)`
//! - `a = g0 S^T k + sum_s G_s (k_s . k) u_s`,
//!   `o = g0 S^T q + sum_s G_s (k_s . q) u_s`
//! - `u_n = beta (v - a)`, `y = o + (k . q) u_n`
//! - the state at the commit `c` (`0..=n+1`): the first `min(c, n)` carried
//!   rows folded in order (`S <- exp(g_j) S + k_j u_j^T`), then the own row
//!   where `c = n + 1`; as a closed form over the folded rows `F`,
//!   `S_c = exp(sum_F g) S + sum_{j in F} exp(sum_{i in F, i > j} g_i) k_j u_j^T`.
//!
//! The same step engine-cuda runs (`gdn_step_carried`), in f32; the state
//! is a sum in another order than its sequential fold, so the two agree
//! within rounding, not bit for bit.

use crate::hlo::{Built, Cmp, Elem, Fold, Func, Val};

/// A step's inputs, every one f32. `lanes` lanes, `heads` value heads over
/// `k_heads` key heads, `rows` the carried rows laid (at least each lane's).
#[derive(Clone, Copy, Debug)]
pub struct Carried {
    /// The committed state, `[lanes, heads, k_dim, v_dim]`.
    pub state: Val,
    /// The carried rows' raw keys `[lanes, rows, k_heads, k_dim]`, corrected
    /// values `[lanes, rows, heads, v_dim]` and log decays `[lanes, rows,
    /// heads]`.
    pub keys: Val,
    pub values: Val,
    pub decays: Val,
    /// Each lane's carried rows (`[lanes]` i32, at most `rows`) and its
    /// commit (`[lanes]` i32, `0..=count + 1`).
    pub count: Val,
    pub commit: Val,
    /// The own row: query and key `[lanes, k_heads, k_dim]` (raw), value
    /// `[lanes, heads, v_dim]`, log decay and beta `[lanes, heads]`.
    pub q: Val,
    pub k: Val,
    pub v: Val,
    pub decay: Val,
    pub beta: Val,
}

/// A step's outputs: the own row's output and corrected value (`[lanes,
/// heads, v_dim]`), and the state at the commit (`[lanes, heads, k_dim,
/// v_dim]`, f32; a lane committing nothing has its state as it was).
#[derive(Clone, Copy, Debug)]
pub struct Stepped {
    pub y: Val,
    pub u: Val,
    pub state: Val,
}

/// `x / sqrt(sum_{last} x^2 + 1e-6)`.
fn l2norm(f: &mut Func, x: Val) -> Built<Val> {
    let dims = f.dims(x).to_vec();
    let last = (dims.len() - 1) as i64;
    let sq = f.mul(x, x)?;
    let sum = f.reduce(sq, &[last], Fold::Sum)?;
    let eps = f.like_f(sum, 1e-6);
    let sum = f.add(sum, eps)?;
    let inv = f.rsqrt(sum);
    let keep: Vec<i64> = (0..last).collect();
    let inv = f.broadcast(inv, &dims, &keep)?;
    f.mul(x, inv)
}

/// The key heads of `x` (`[.., k_heads, d]`) repeated to the value heads
/// (`[.., heads, d]`, head `h` reading key head `h / group`).
fn to_value_heads(f: &mut Func, x: Val, heads: i64) -> Built<Val> {
    let dims = f.dims(x).to_vec();
    let r = dims.len();
    let k_heads = dims[r - 2];
    let group = heads / k_heads;
    if group == 1 {
        return Ok(x);
    }
    let mut wide = dims[..r - 1].to_vec();
    wide.push(group);
    wide.push(dims[r - 1]);
    let map: Vec<i64> = (0..r as i64 - 1).chain([r as i64]).collect();
    let x = f.broadcast(x, &wide, &map)?;
    let mut out = dims[..r - 2].to_vec();
    out.push(heads);
    out.push(dims[r - 1]);
    f.reshape(x, &out)
}

/// The inclusive prefix sum of `x` along `axis`: log2(n) shifted adds.
fn prefix_sum(f: &mut Func, mut x: Val, axis: usize) -> Built<Val> {
    let dims = f.dims(x).to_vec();
    let r = dims.len();
    let n = dims[axis];
    let zero = f.const_f(Elem::F32, 0.0, &[]);
    let mut k = 1;
    while k < n {
        let mut hi = dims.clone();
        hi[axis] = n - k;
        let head = f.slice(x, &vec![0; r], &hi, &vec![1; r])?;
        let mut lo = vec![0; r];
        lo[axis] = k;
        let shifted = f.pad(head, zero, &lo, &vec![0; r], &vec![0; r])?;
        x = f.add(x, shifted)?;
        k *= 2;
    }
    Ok(x)
}

/// Runs the step (module docs) over every lane and value head.
pub fn gdn_step_carried(f: &mut Func, c: &Carried, q_scale: f32) -> Built<Stepped> {
    let sd = f.dims(c.state).to_vec();
    let (lanes, heads, k_dim, v_dim) = (sd[0], sd[1], sd[2], sd[3]);
    let rows = f.dims(c.keys)[1];
    let i = Elem::I32;
    // normalised keys and queries, at the value heads
    let q = l2norm(f, c.q)?;
    let scale = f.like_f(q, f64::from(q_scale));
    let q = f.mul(q, scale)?;
    let q = to_value_heads(f, q, heads)?; // [L, H, K]
    let k = l2norm(f, c.k)?;
    let k = to_value_heads(f, k, heads)?; // [L, H, K]
    let ks = l2norm(f, c.keys)?;
    let ks = to_value_heads(f, ks, heads)?; // [L, R, H, K]
    // the carried rows each lane holds
    let row = f.iota(i, &[lanes, rows], 1);
    let count = f.broadcast(c.count, &[lanes, rows], &[0])?;
    let valid = f.compare(Cmp::Lt, row, count)?; // [L, R]
    let valid_h = f.broadcast(valid, &[lanes, rows, heads], &[0, 1])?;
    let zero_g = f.const_f(Elem::F32, 0.0, &[lanes, rows, heads]);
    let gs = f.select(valid_h, c.decays, zero_g)?; // [L, R, H]
    // decays to the own row: G_s = exp(g_n + total - incl_s), g0 = exp(g_n + total)
    let incl = prefix_sum(f, gs, 1)?;
    let total = f.reduce(gs, &[1], Fold::Sum)?; // [L, H]
    let gn = c.decay; // [L, H]
    let to_own = f.add(gn, total)?; // [L, H]
    let g0 = f.exp(to_own);
    let to_own_r = f.broadcast(to_own, &[lanes, rows, heads], &[0, 2])?;
    let after = f.sub(to_own_r, incl)?;
    let big_g = f.exp(after);
    let big_g = f.select(valid_h, big_g, zero_g)?; // [L, R, H]
    // S^T k and S^T q: [L, H, V]
    let st = |f: &mut Func, x: Val| -> Built<Val> {
        f.dot_general_at(c.state, x, &[0, 1], &[0, 1], &[2], &[2], Elem::F32, true)
    };
    let stk = st(f, k)?;
    let stq = st(f, q)?;
    let g0v = f.broadcast(g0, &[lanes, heads, v_dim], &[0, 1])?;
    // (k_s . k) and (k_s . q): [L, R, H]
    let dots = |f: &mut Func, x: Val| -> Built<Val> {
        let xb = f.broadcast(x, &[lanes, rows, heads, k_dim], &[0, 2, 3])?;
        let p = f.mul(ks, xb)?;
        f.reduce(p, &[3], Fold::Sum)
    };
    // sum_s w_s u_s: [L, H, V]
    let mix = |f: &mut Func, w: Val| -> Built<Val> {
        f.dot_general_at(w, c.values, &[0, 2], &[0, 2], &[1], &[1], Elem::F32, true)
    };
    let dk = dots(f, k)?;
    let wk = f.mul(big_g, dk)?;
    let ak = mix(f, wk)?;
    let a = f.mul(g0v, stk)?;
    let a = f.add(a, ak)?;
    let dq = dots(f, q)?;
    let wq = f.mul(big_g, dq)?;
    let oq = mix(f, wq)?;
    let o = f.mul(g0v, stq)?;
    let o = f.add(o, oq)?;
    // the own row
    let beta = f.broadcast(c.beta, &[lanes, heads, v_dim], &[0, 1])?;
    let diff = f.sub(c.v, a)?;
    let u = f.mul(beta, diff)?; // [L, H, V]
    let kq = f.mul(k, q)?;
    let kq = f.reduce(kq, &[2], Fold::Sum)?;
    let kq = f.broadcast(kq, &[lanes, heads, v_dim], &[0, 1])?;
    let ku = f.mul(kq, u)?;
    let y = f.add(o, ku)?;
    // the fold: rows 0..rows the carried, row `rows` the own; folded where
    // carried j < min(c, n), the own where c = n + 1
    let n_rows = rows + 1;
    let all_k = {
        let own = f.reshape(k, &[lanes, 1, heads, k_dim])?;
        f.concat(&[ks, own], 1)?
    };
    let all_u = {
        let own = f.reshape(u, &[lanes, 1, heads, v_dim])?;
        f.concat(&[c.values, own], 1)?
    };
    let all_g = {
        let own = f.reshape(gn, &[lanes, 1, heads])?;
        f.concat(&[c.decays, own], 1)?
    };
    let slot = f.iota(i, &[lanes, n_rows], 1);
    let cnt = f.broadcast(c.count, &[lanes, n_rows], &[0])?;
    let cmt = f.broadcast(c.commit, &[lanes, n_rows], &[0])?;
    let upto = f.min(cmt, cnt)?;
    let carried_folded = f.compare(Cmp::Lt, slot, upto)?;
    let own_slot = f.const_i(i, rows, &[lanes, n_rows]);
    let is_own = f.compare(Cmp::Eq, slot, own_slot)?;
    let one = f.const_i(i, 1, &[lanes, n_rows]);
    let n_plus = f.add(cnt, one)?;
    let own_commits = f.compare(Cmp::Eq, cmt, n_plus)?;
    let own_folded = f.and(is_own, own_commits)?;
    let carried_part = f.compare(Cmp::Lt, slot, own_slot)?;
    let carried_folded = f.and(carried_folded, carried_part)?;
    let folded = f.or(carried_folded, own_folded)?; // [L, N]
    let folded_h = f.broadcast(folded, &[lanes, n_rows, heads], &[0, 1])?;
    let zero_n = f.const_f(Elem::F32, 0.0, &[lanes, n_rows, heads]);
    let fg = f.select(folded_h, all_g, zero_n)?;
    let lam = f.reduce(fg, &[1], Fold::Sum)?; // [L, H]
    let fincl = prefix_sum(f, fg, 1)?;
    let lam_n = f.broadcast(lam, &[lanes, n_rows, heads], &[0, 2])?;
    let suffix = f.sub(lam_n, fincl)?; // sum over folded rows after j
    let coef = f.exp(suffix);
    let coef = f.select(folded_h, coef, zero_n)?;
    let coef_k = f.broadcast(coef, &[lanes, n_rows, heads, k_dim], &[0, 1, 2])?;
    let wk_all = f.mul(all_k, coef_k)?;
    let rank = f.dot_general_at(wk_all, all_u, &[0, 2], &[0, 2], &[1], &[1], Elem::F32, true)?;
    let decayed = f.exp(lam);
    let decayed = f.broadcast(decayed, &[lanes, heads, k_dim, v_dim], &[0, 1])?;
    let s0 = f.mul(decayed, c.state)?;
    let state = f.add(s0, rank)?; // [L, H, K, V]
    Ok(Stepped { y, u, state })
}
