//! A beam group's step (`new-memory/designs/BEAM-PAGE-LISTS.md` §5–6) as
//! one HLO region a fire runs ahead of its first append or read: the same
//! algorithm as `engine::beam::step` (the host replay the test checks it
//! against), over the group's resident state words and the parents its last
//! epilogue wrote, every value on the device.
//!
//! The state is `[words]` i32 (`Layout::words`): fires run, the pool's pages
//! at the last fire, `map` (`[lanes, cap]`), `free` (`[cap]`, 0 / 1) and
//! `pending` (`[frames, cap]`, 0 / 1). Lanes, columns, frames and the page
//! size are the program's; the set's frozen slots, its pool pages and the
//! slot the beams write are the fire's (scalars).

use crate::hlo::{Built, Cmp, Combine, Elem, Fold, Func, GatherDims, ScatterDims, Val};

/// What `BEAM-PAGE-LISTS.md` §7 calls each lane's fault (as
/// `engine::beam::fault`).
pub mod fault {
    pub const PAST_RESERVE: i64 = 1;
    pub const SHORT: i64 = 2;
    pub const BELOW_FLOOR: i64 = 3;
    pub const COPY_BOUNDS: i64 = 4;
    pub const PARENT: i64 = 5;
    pub const RESERVE_EXHAUSTED: i64 = 6;
}

/// A group's static shape: its beams, the columns (and pool pages) its
/// state is laid for, the engine's frames in flight and its kv page size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub lanes: i64,
    pub cap: i64,
    pub frames: i64,
    pub page: i64,
}

impl Layout {
    /// The state's i32 words: fired, pool, map, free, pending, and each
    /// pair of lanes' common depth (`lca`, `[lanes, lanes]`, §11.4).
    #[must_use]
    pub fn words(&self) -> i64 {
        2 + self.lanes * self.cap
            + self.cap
            + self.frames.max(1) * self.cap
            + self.lanes * self.lanes
    }
}

/// The fire's numbers (§3), i32 scalars: the set's frozen slots, its pool
/// pages past them (the reserve), the slot every beam writes, and the
/// group's beams (at most the layout's lanes: a group runs on the smallest
/// width rung that holds it, the beams past its own dead, holding and
/// taking no page and faulting nothing).
#[derive(Clone, Copy, Debug)]
pub struct At {
    pub folded: Val,
    pub reserve: Val,
    pub slot: Val,
    pub live: Val,
}

/// One step's results: the state after it, each beam's list (`[lanes,
/// cap]` pool pages, columns `0..=c` live), its copy (`[lanes, 3]`: src,
/// dst, tokens; tokens 0 none), its fault (`[lanes]`, 0 none), and the
/// fire's column and in-page offset (scalars).
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub state: Val,
    pub lists: Val,
    pub copies: Val,
    pub faults: Val,
    pub column: Val,
    pub offset: Val,
    pub first_page: Val,
    /// The group's common slot `c_t` (§11.3, to the row §11.4): the live
    /// lanes' most recent common ancestor's depth, capped at the slot past
    /// the fire's row and never below the folded slot (scalar).
    pub common: Val,
}

fn scalar(f: &mut Func, x: i64) -> Val {
    f.const_i(Elem::I32, x, &[])
}

fn i32_of(f: &mut Func, pred: Val) -> Val {
    f.convert(pred, Elem::I32)
}

fn any(f: &mut Func, pred: Val, axes: &[i64]) -> Built<Val> {
    let x = i32_of(f, pred);
    let m = f.reduce(x, axes, Fold::Max)?;
    let zero = f.like_i(m, 0);
    f.compare(Cmp::Gt, m, zero)
}

/// `table[idx]` elementwise (`table` `[n]`, `idx` any shape), clamped as
/// XLA's gather clamps.
fn take(f: &mut Func, table: Val, idx: Val) -> Built<Val> {
    let dims = f.dims(idx).to_vec();
    let mut at = dims.clone();
    at.push(1);
    let at = f.reshape(idx, &at)?;
    f.gather(
        table,
        at,
        &GatherDims {
            offset_dims: vec![],
            collapsed_slice_dims: vec![0],
            start_index_map: vec![0],
            index_vector_dim: dims.len() as i64,
            ..GatherDims::default()
        },
        &[1],
    )
}

/// The inclusive prefix sum of `x` (`[n]` i32): log2(n) shifted adds.
fn prefix_sum(f: &mut Func, mut x: Val) -> Built<Val> {
    let n = f.dims(x)[0];
    let zero = f.const_i(Elem::I32, 0, &[]);
    let mut k = 1;
    while k < n {
        let head = f.slice_axis(x, 0, 0, n - k)?;
        let shifted = f.pad(head, zero, &[k], &[0], &[0])?;
        x = f.add(x, shifted)?;
        k *= 2;
    }
    Ok(x)
}

/// `base` (`[n]`) with each `upd[i]` combined into `base[idx[i]]`; an index
/// out of range lands nowhere.
fn scatter_with(f: &mut Func, base: Val, idx: Val, upd: Val, combine: Combine) -> Built<Val> {
    let m = f.dims(idx)[0];
    let at = f.reshape(idx, &[m, 1])?;
    f.scatter(
        base,
        at,
        upd,
        &ScatterDims {
            update_window_dims: vec![],
            inserted_window_dims: vec![0],
            input_batching_dims: vec![],
            scatter_indices_batching_dims: vec![],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
        },
        combine,
    )
}

/// Runs one fire of the group: primes it on its first fire, else reorders
/// it by `parents` (`[lanes]` i32), then checks the lists and copies.
pub fn step(f: &mut Func, layout: Layout, state: Val, parents: Val, at: At) -> Built<Step> {
    let Layout {
        lanes: b_n,
        cap,
        frames,
        page: s,
    } = layout;
    let frames = frames.max(1);
    let s = s.max(1);
    let i = Elem::I32;
    let sv = scalar(f, s);
    let zero = scalar(f, 0);
    let one = scalar(f, 1);

    // §3's numbers.
    let f0 = f.div(at.folded, sv)?;
    let tail = f.rem(at.folded, sv)?;
    let slot_page = f.div(at.slot, sv)?;
    let c = f.sub(slot_page, f0)?;
    let c = f.max(c, zero)?;
    let o = f.rem(at.slot, sv)?;
    let before = f.sub(at.slot, one)?;
    let before = f.max(before, zero)?;
    let before = f.div(before, sv)?;
    let cp = f.sub(before, f0)?;
    let cp = f.max(cp, zero)?;
    let slot_zero = f.compare(Cmp::Eq, at.slot, zero)?;
    let cp = f.select(slot_zero, zero, cp)?;
    let capv = scalar(f, cap);
    let q_n = f.min(at.reserve, capv)?;
    let has_tail = f.compare(Cmp::Gt, tail, zero)?;

    // The state's words.
    let fired = f.slice_axis(state, 0, 0, 1)?;
    let fired = f.reshape(fired, &[])?;
    let pool = f.slice_axis(state, 0, 1, 2)?;
    let pool = f.reshape(pool, &[])?;
    let at_map = 2;
    let at_free = at_map + b_n * cap;
    let at_pending = at_free + cap;
    let map = f.slice_axis(state, 0, at_map, at_free)?;
    let map = f.reshape(map, &[b_n, cap])?;
    let free = f.slice_axis(state, 0, at_free, at_pending)?;
    let pending = f.slice_axis(state, 0, at_pending, at_pending + frames * cap)?;
    let pending = f.reshape(pending, &[frames, cap])?;
    let at_lca = at_pending + frames * cap;
    let lca = f.slice_axis(state, 0, at_lca, at_lca + b_n * b_n)?;
    let lca = f.reshape(lca, &[b_n, b_n])?;

    let q_iota = f.iota(i, &[cap], 0);
    let b_iota = f.iota(i, &[b_n], 0);
    let live_b = f.splat(at.live, &[b_n])?;
    let alive = f.compare(Cmp::Lt, b_iota, live_b)?;
    let q_zero = f.like_i(q_iota, 0);
    let is_q0 = f.compare(Cmp::Eq, q_iota, q_zero)?;
    let tail_q = f.splat(has_tail, &[cap])?;
    // page 0 is the prompt's own when the prompt ends mid-page: never free
    let frozen0 = f.and(is_q0, tail_q)?;
    let not_frozen0 = f.not(frozen0);
    let q_n_q = f.splat(q_n, &[cap])?;
    let in_pool = f.compare(Cmp::Lt, q_iota, q_n_q)?;

    // ---- §5: priming.
    let col_iota = f.iota(i, &[b_n, cap], 1);
    let row_iota = f.iota(i, &[b_n, cap], 0);
    let col0 = f.like_i(col_iota, 0);
    let is_col0 = f.compare(Cmp::Eq, col_iota, col0)?;
    let zeros_map = f.like_i(map, 0);
    let map_p = f.select(is_col0, row_iota, zeros_map)?;
    let lanes_q = f.splat(at.live, &[cap])?;
    let past_lanes = f.compare(Cmp::Ge, q_iota, lanes_q)?;
    let free_p = f.and(in_pool, past_lanes)?;
    let free_p = f.and(free_p, not_frozen0)?;
    let free_p = i32_of(f, free_p);
    let pending_p = f.like_i(pending, 0);
    let b_zero = f.like_i(b_iota, 0);
    let not_b0 = f.compare(Cmp::Gt, b_iota, b_zero)?;
    let o_pos = f.compare(Cmp::Gt, o, zero)?;
    let tail_o = f.and(has_tail, o_pos)?;
    let tail_o = f.splat(tail_o, &[b_n])?;
    let copy_p = f.and(not_b0, tail_o)?;
    let copy_p = f.and(copy_p, alive)?;
    let zero_b = f.like_i(b_iota, 0);
    let tail_b = f.splat(tail, &[b_n])?;
    let copy_p_tokens = f.select(copy_p, tail_b, zero_b)?;
    let copy_p_dst = f.select(copy_p, b_iota, zero_b)?;
    let copies_p = {
        let src = f.reshape(zero_b, &[b_n, 1])?;
        let dst = f.reshape(copy_p_dst, &[b_n, 1])?;
        let tok = f.reshape(copy_p_tokens, &[b_n, 1])?;
        f.concat(&[src, dst, tok], 1)?
    };
    let faults_p = {
        let c_off = f.compare(Cmp::Ne, c, zero)?;
        let short = f.const_i(i, fault::SHORT, &[]);
        let short = f.select(c_off, short, zero)?;
        let over = f.compare(Cmp::Gt, at.live, q_n)?;
        let ex = f.const_i(i, fault::RESERVE_EXHAUSTED, &[]);
        let ex = f.select(over, ex, zero)?;
        let worst = f.max(short, ex)?;
        f.splat(worst, &[b_n])?
    };

    // ---- §6: a later fire.
    // pages that joined the pool since the last fire are free
    let pool_q = f.splat(pool, &[cap])?;
    let joined = f.compare(Cmp::Ge, q_iota, pool_q)?;
    let joined = f.and(joined, in_pool)?;
    let joined_free = i32_of(f, not_frozen0);
    let free0 = f.select(joined, joined_free, free)?;
    // §6.1
    let ge0 = f.compare(Cmp::Ge, parents, zero_b)?;
    let lt = f.compare(Cmp::Lt, parents, live_b)?;
    let valid = f.and(ge0, lt)?;
    let valid = f.and(valid, alive)?;
    let par = f.select(valid, parents, b_iota)?;
    let bad = f.not(valid);
    let bad = f.and(bad, alive)?;
    let parent_fault = f.like_i(b_iota, fault::PARENT);
    let faults_s = f.select(bad, parent_fault, zero_b)?;
    // §6.2: first child of its parent, the need for a fresh page, its rank
    let pe = f.broadcast(par, &[b_n, b_n], &[1])?; // [b, e] = par[e]
    let pb = f.broadcast(par, &[b_n, b_n], &[0])?; // [b, e] = par[b]
    let ei = f.iota(i, &[b_n, b_n], 1);
    let bi = f.iota(i, &[b_n, b_n], 0);
    let earlier = f.compare(Cmp::Lt, ei, bi)?;
    let same = f.compare(Cmp::Eq, pe, pb)?;
    let shared = f.and(earlier, same)?;
    let not_first = any(f, shared, &[1])?;
    let o_zero = f.compare(Cmp::Eq, o, zero)?;
    let o_zero_b = f.splat(o_zero, &[b_n])?;
    let need = f.or(o_zero_b, not_first)?;
    let need = f.and(need, alive)?;
    let need_i = i32_of(f, need);
    let need_e = f.broadcast(need_i, &[b_n, b_n], &[1])?;
    let zeros_bb = f.like_i(need_e, 0);
    let ranked = f.select(earlier, need_e, zeros_bb)?;
    let rank = f.reduce(ranked, &[1], Fold::Sum)?;
    // the k-th lowest free page: each free page's place among them
    let free_b = f.compare(Cmp::Gt, free0, q_zero)?;
    let free_i = i32_of(f, free_b);
    let place = {
        let count = prefix_sum(f, free_i)?;
        let one_q = f.like_i(count, 1);
        f.sub(count, one_q)?
    };
    let (fresh, found) = {
        let pl = f.broadcast(place, &[b_n, cap], &[1])?;
        let fr = f.broadcast(free_b, &[b_n, cap], &[1])?;
        let rk = f.broadcast(rank, &[b_n, cap], &[0])?;
        let hit = f.compare(Cmp::Eq, pl, rk)?;
        let hit = f.and(hit, fr)?;
        let found = any(f, hit, &[1])?;
        let qs = f.iota(i, &[b_n, cap], 1);
        let z = f.like_i(qs, 0);
        let qs = f.select(hit, qs, z)?;
        (f.reduce(qs, &[1], Fold::Sum)?, found)
    };
    let not_found = f.not(found);
    let short_handed = f.and(need, not_found)?;
    let ex_b = f.like_i(b_iota, fault::RESERVE_EXHAUSTED);
    let ex_b = f.select(short_handed, ex_b, zero_b)?;
    let faults_s = f.max(faults_s, ex_b)?;
    let gets = f.and(need, found)?;
    // inh[b] = map[par[b]]
    let inh = {
        let at = f.reshape(par, &[b_n, 1])?;
        f.gather(
            map,
            at,
            &GatherDims {
                offset_dims: vec![1],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                ..GatherDims::default()
            },
            &[1, cap],
        )?
    };
    let c_bc = f.splat(c, &[b_n, cap])?;
    let at_c = f.compare(Cmp::Eq, col_iota, c_bc)?;
    let gets_bc = f.broadcast(gets, &[b_n, cap], &[0])?;
    let fresh_bc = f.broadcast(fresh, &[b_n, cap], &[0])?;
    let put = f.and(at_c, gets_bc)?;
    let map_s = f.select(put, fresh_bc, inh)?;
    // copies: a child mid-page takes its parent's tail
    let inh_c = {
        let z = f.like_i(inh, 0);
        let v = f.select(at_c, inh, z)?;
        f.reduce(v, &[1], Fold::Sum)?
    };
    let o_pos_bc = f.splat(o_pos, &[b_n])?;
    let copy_s = f.and(gets, o_pos_bc)?;
    let o_b = f.splat(o, &[b_n])?;
    let copies_s = {
        let src = f.select(copy_s, inh_c, zero_b)?;
        let dst = f.select(copy_s, fresh, zero_b)?;
        let tok = f.select(copy_s, o_b, zero_b)?;
        let src = f.reshape(src, &[b_n, 1])?;
        let dst = f.reshape(dst, &[b_n, 1])?;
        let tok = f.reshape(tok, &[b_n, 1])?;
        f.concat(&[src, dst, tok], 1)?
    };
    // §6.3: free and pending
    let framesv = scalar(f, frames);
    let k = f.rem(fired, framesv)?;
    let taken = {
        let fq = f.broadcast(fresh, &[b_n, cap], &[0])?;
        let qs = f.iota(i, &[b_n, cap], 1);
        let hit = f.compare(Cmp::Eq, fq, qs)?;
        let hit = f.and(hit, gets_bc)?;
        any(f, hit, &[0])?
    };
    let pend_k = f.dynamic_slice(pending, &[k, zero], &[1, cap])?;
    let pend_k = f.reshape(pend_k, &[cap])?;
    let pend_k = f.compare(Cmp::Gt, pend_k, q_zero)?;
    let not_taken = f.not(taken);
    let kept = f.and(free_b, not_taken)?;
    let free_s = f.or(kept, pend_k)?;
    let free_s = f.and(free_s, not_frozen0)?;
    let free_s = f.and(free_s, in_pool)?;
    let free_s = i32_of(f, free_s);
    // a page some beam held in a column through the last fire and no beam
    // holds there now
    // A page lies in one column of every list that names it (§6.3), and a
    // parent with any child keeps its pages through its first child, so a
    // page held through the last fire is freed when only rows no child
    // continues name it: one max over the pages of 1 + (its row continues).
    let freed = {
        let parent_of = f.broadcast(par, &[b_n, b_n], &[0])?; // [b, p] = par[b]
        let rows = f.iota(i, &[b_n, b_n], 1);
        let child = f.compare(Cmp::Eq, parent_of, rows)?;
        let child_alive = f.broadcast(alive, &[b_n, b_n], &[0])?;
        let child = f.and(child, child_alive)?;
        let continues = any(f, child, &[0])?; // [p]
        let continues = i32_of(f, continues);
        let one_b = f.like_i(continues, 1);
        let mark = f.add(continues, one_b)?;
        let mark = f.broadcast(mark, &[b_n, cap], &[0])?;
        let was_live = f.broadcast(alive, &[b_n, cap], &[0])?;
        let cp_bc = f.splat(cp, &[b_n, cap])?;
        let through = f.compare(Cmp::Le, col_iota, cp_bc)?;
        let held = f.and(through, was_live)?;
        let none = f.like_i(map, 0);
        let mark = f.select(held, mark, none)?;
        let idx = f.reshape(map, &[b_n * cap])?;
        let upd = f.reshape(mark, &[b_n * cap])?;
        let base = f.like_i(q_iota, 0);
        let worst = scatter_with(f, base, idx, upd, Combine::Max)?;
        let only_ended = f.like_i(worst, 1);
        let gone = f.compare(Cmp::Eq, worst, only_ended)?;
        // page 0 under a mid-page prompt is never freed
        let gone = f.and(gone, not_frozen0)?;
        i32_of(f, gone)
    };
    let freed = f.reshape(freed, &[1, cap])?;
    let pending_s = f.dynamic_update_slice(pending, freed, &[k, zero])?;

    // ---- the prime or the step
    let primes = f.compare(Cmp::Eq, fired, zero)?;
    let pick = |f: &mut Func, a: Val, b: Val| -> Built<Val> {
        let dims = f.dims(a).to_vec();
        let p = f.splat(primes, &dims)?;
        f.select(p, a, b)
    };
    let map2 = pick(f, map_p, map_s)?;
    let free2 = pick(f, free_p, free_s)?;
    let pending2 = pick(f, pending_p, pending_s)?;
    let copies2 = pick(f, copies_p, copies_s)?;
    let mut faults2 = pick(f, faults_p, faults_s)?;

    // ---- §6.5's bounds, over each beam's live columns and its write
    let short_c = f.compare(Cmp::Ge, c, capv)?;
    let short_c = f.splat(short_c, &[b_n])?;
    let short_f = f.like_i(b_iota, fault::SHORT);
    let short_f = f.select(short_c, short_f, zero_b)?;
    faults2 = f.max(faults2, short_f)?;
    let live = f.compare(Cmp::Le, col_iota, c_bc)?;
    let q_n_bc = f.splat(q_n, &[b_n, cap])?;
    let past = f.compare(Cmp::Ge, map2, q_n_bc)?;
    let past = f.and(past, live)?;
    let past = any(f, past, &[1])?;
    let past_f = f.like_i(b_iota, fault::PAST_RESERVE);
    let past_f = f.select(past, past_f, zero_b)?;
    let q_w = {
        let z = f.like_i(map2, 0);
        let v = f.select(at_c, map2, z)?;
        f.reduce(v, &[1], Fold::Sum)?
    };
    let f0_b = f.splat(f0, &[b_n])?;
    let s_b = f.like_i(b_iota, s);
    let abs = f.add(f0_b, q_w)?;
    let abs = f.mul(abs, s_b)?;
    let abs = f.add(abs, o_b)?;
    let folded_b = f.splat(at.folded, &[b_n])?;
    let q_n_b = f.splat(q_n, &[b_n])?;
    let w_past = f.compare(Cmp::Ge, q_w, q_n_b)?;
    let below = f.compare(Cmp::Lt, abs, folded_b)?;
    let end = f.add(f0_b, q_n_b)?;
    let end = f.mul(end, s_b)?;
    let beyond = f.compare(Cmp::Ge, abs, end)?;
    let floor = f.or(w_past, below)?;
    let floor = f.or(floor, beyond)?;
    let floor_f = f.like_i(b_iota, fault::BELOW_FLOOR);
    let floor_f = f.select(floor, floor_f, zero_b)?;
    // (§6.5 skips a lane already short)
    let not_short = f.not(short_c);
    let past_f = f.select(not_short, past_f, zero_b)?;
    let floor_f = f.select(not_short, floor_f, zero_b)?;
    faults2 = f.max(faults2, past_f)?;
    faults2 = f.max(faults2, floor_f)?;
    // §6.4's bounds
    let src = f.slice_axis(copies2, 1, 0, 1)?;
    let src = f.reshape(src, &[b_n])?;
    let dst = f.slice_axis(copies2, 1, 1, 2)?;
    let dst = f.reshape(dst, &[b_n])?;
    let tok = f.slice_axis(copies2, 1, 2, 3)?;
    let tok = f.reshape(tok, &[b_n])?;
    let some = f.compare(Cmp::Gt, tok, zero_b)?;
    let moves = f.compare(Cmp::Ne, src, dst)?;
    let real = f.and(some, moves)?;
    let src_past = f.compare(Cmp::Ge, src, q_n_b)?;
    let dst_past = f.compare(Cmp::Ge, dst, q_n_b)?;
    let long = f.compare(Cmp::Gt, tok, s_b)?;
    let dst_abs = f.add(f0_b, dst)?;
    let dst_abs = f.mul(dst_abs, s_b)?;
    let into_frozen = f.compare(Cmp::Lt, dst_abs, folded_b)?;
    let bad_copy = f.or(src_past, dst_past)?;
    let bad_copy = f.or(bad_copy, long)?;
    let bad_copy = f.or(bad_copy, into_frozen)?;
    let bad_copy = f.and(bad_copy, real)?;
    let copy_f = f.like_i(b_iota, fault::COPY_BOUNDS);
    let copy_f = f.select(bad_copy, copy_f, zero_b)?;
    faults2 = f.max(faults2, copy_f)?;
    let faults2 = f.select(alive, faults2, zero_b)?;

    // ---- §11.4: each pair's common depth, and the group's common slot.
    // Lanes forked from one parent agree below this fire's row; others
    // keep their parents' pair's depth. Priming: every pair the fire's slot.
    let p = {
        let hi = scalar(f, b_n - 1);
        f.clamp(zero, parents, hi)?
    };
    let lca_p = {
        let rows = f.take_rows(lca, p)?; // [i][j] = lca[p_i][j]
        let rows = f.transpose(rows, &[1, 0])?;
        let both = f.take_rows(rows, p)?; // [j][i] = lca[p_i][p_j]
        f.transpose(both, &[1, 0])?
    };
    let pi = f.broadcast(p, &[b_n, b_n], &[0])?;
    let pj = f.broadcast(p, &[b_n, b_n], &[1])?;
    let same = f.compare(Cmp::Eq, pi, pj)?;
    let slot_bb = f.splat(at.slot, &[b_n, b_n])?;
    let lca_s = f.select(same, slot_bb, lca_p)?;
    let lca2 = pick(f, slot_bb, lca_s)?;
    let end_t = f.add(at.slot, one)?;
    let common = {
        let ii = f.iota(i, &[b_n, b_n], 0);
        let jj = f.iota(i, &[b_n, b_n], 1);
        let upper = f.compare(Cmp::Lt, ii, jj)?;
        let live_bb = f.splat(at.live, &[b_n, b_n])?;
        let live_j = f.compare(Cmp::Lt, jj, live_bb)?;
        let pair = f.and(upper, live_j)?;
        let end_bb = f.splat(end_t, &[b_n, b_n])?;
        let depth = f.select(pair, lca2, end_bb)?;
        let least = f.reduce(depth, &[0, 1], Fold::Min)?;
        let least = f.min(least, end_t)?;
        f.max(least, at.folded)?
    };

    // ---- the state after
    let fired2 = f.add(fired, one)?;
    let fired2 = f.reshape(fired2, &[1])?;
    let pool2 = f.reshape(q_n, &[1])?;
    let map_flat = f.reshape(map2, &[b_n * cap])?;
    let pending_flat = f.reshape(pending2, &[frames * cap])?;
    let lca_flat = f.reshape(lca2, &[b_n * b_n])?;
    let state2 = f.concat(&[fired2, pool2, map_flat, free2, pending_flat, lca_flat], 0)?;
    Ok(Step {
        state: state2,
        lists: map2,
        copies: copies2,
        faults: faults2,
        column: c,
        offset: o,
        first_page: f0,
        common,
    })
}

/// Each beam's resolved entries: `lists` (`[lanes, cap]` pool pages) through
/// `translation` (`[pages]` i32, the pool's pages from the set's first
/// pool page), as `[lanes, cap]`; columns past the live ones read whatever
/// the clamped translation holds (a caller keeps its staged words there).
pub fn resolve(f: &mut Func, lists: Val, translation: Val) -> Built<Val> {
    take(f, translation, lists)
}
