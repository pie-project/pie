//! Host phases: the steps of a fire the host runs between fabric phases
//! (`HostPhase`, its `HostOp`s) over the fire's state, as the words the
//! fabric phases read and wrote.

use super::*;

/// Pastes a phase's view windows back into their roots.
pub(super) fn paste_views(manifest: &Manifest, state: &mut HashMap<String, Vec<u32>>) {
    for view in &manifest.views {
        let Some(words) = state.remove(&view.name) else {
            continue;
        };
        if let Some(root) = state.get_mut(&view.root) {
            let from = view.offset as usize;
            let to = (from + words.len()).min(root.len());
            root[from..to].copy_from_slice(&words[..to - from]);
        }
    }
}

/// The pool cell holding `pos` of request `req`: `indices[indptr[req] + pos
/// / ps] · ps + pos % ps`, or none when a table runs out.
fn cell_of(indices: &[i32], indptr: &[i32], req: i32, pos: i64, ps: i64) -> Option<usize> {
    let pos = pos.max(0);
    let first = i64::from(*indptr.get(usize::try_from(req).ok()?)?);
    let at = usize::try_from(first + pos / ps).ok()?;
    let page = i64::from(*indices.get(at)?);
    usize::try_from(page * ps + pos % ps).ok()
}

/// The first `clips` boxes `(t, h, w, off)` of the i32 clip table `name`.
fn clip_boxes(state: &HashMap<String, Vec<u32>>, name: &str, clips: u32) -> Vec<[i64; 4]> {
    let words = state.get(name).map(Vec::as_slice).unwrap_or(&[]);
    (0..clips as usize)
        .map(|l| {
            let at = |j: usize| words.get(4 * l + j).map_or(0, |w| i64::from(*w as i32));
            [at(0), at(1), at(2), at(3)]
        })
        .collect()
}

/// `boxes` rebased to a segmented-fire window whose buffers start at its own rows.
fn rebased(mut boxes: Vec<[i64; 4]>, base: u32) -> Vec<[i64; 4]> {
    for b in &mut boxes {
        b[3] -= i64::from(base);
    }
    boxes
}

fn clip_of(boxes: &[[i64; 4]], row: i64) -> Option<(usize, [i64; 4], i64)> {
    boxes.iter().enumerate().find_map(|(l, b)| {
        let local = row - b[3];
        (local >= 0 && local < b[0] * b[1] * b[2]).then_some((l, *b, local))
    })
}

fn unravel(b: [i64; 4], local: i64) -> [i64; 3] {
    let rest = local / b[2];
    [rest / b[1], rest % b[1], local % b[2]]
}

fn ravel(b: [i64; 4], t: i64, h: i64, w: i64) -> i64 {
    b[3] + (t * b[1] + h) * b[2] + w
}

/// Runs a host phase over the fire's buffers.
pub(super) fn run_host(
    host: &HostPhase,
    state: &mut HashMap<String, Vec<u32>>,
    at: usize,
) -> Result<()> {
    let take = |state: &HashMap<String, Vec<u32>>, name: &str| -> Result<Vec<f32>> {
        state
            .get(name)
            .map(|w| w.iter().map(|x| f32::from_bits(*x)).collect())
            .ok_or_else(|| Fault::Unbound {
                what: format!("phase {at}: host op names {name}, which the fire has no buffer for"),
            })
    };
    let put = |state: &mut HashMap<String, Vec<u32>>, name: &str, v: Vec<f32>| {
        state.insert(name.to_string(), v.iter().map(|x| x.to_bits()).collect());
    };
    match &host.op {
        HostOp::Embed {
            ids,
            table,
            y,
            rows,
            width,
            limit,
        } => {
            let (rows, width, limit) = (*rows as usize, *width as usize, *limit as usize);
            let id_words = state.get(ids).cloned().unwrap_or_default();
            let table = take(state, table)?;
            let mut out = take(state, y)?;
            out.resize(rows * width, 0.0);
            for r in 0..rows {
                let id = id_words.get(r).map_or(0, |w| *w as i32);
                let id = if id < 0 || id as usize >= limit {
                    0
                } else {
                    id as usize
                };
                let src = table.get(id * width..(id + 1) * width).unwrap_or(&[]);
                let dst = &mut out[r * width..(r + 1) * width];
                dst[..src.len()].copy_from_slice(src);
            }
            put(state, y, out);
        }
        HostOp::SplitRows {
            x,
            left,
            right,
            rows,
            width,
            cut,
        } => {
            let (rows, width, cut) = (*rows as usize, *width as usize, *cut as usize);
            let src = take(state, x)?;
            let mut l = take(state, left)?;
            let mut r = take(state, right)?;
            let rw = width - cut;
            for i in 0..rows {
                let row = src.get(i * width..(i + 1) * width).unwrap_or(&[]);
                if row.len() < width {
                    continue;
                }
                if let Some(d) = l.get_mut(i * cut..(i + 1) * cut) {
                    d.copy_from_slice(&row[..cut]);
                }
                if let Some(d) = r.get_mut(i * rw..(i + 1) * rw) {
                    d.copy_from_slice(&row[cut..]);
                }
            }
            put(state, left, l);
            put(state, right, r);
        }
        HostOp::SplitQGate {
            packed,
            q,
            gate,
            rows,
            heads,
            head_dim,
        } => {
            let (rows, heads, hd) = (*rows as usize, *heads as usize, *head_dim as usize);
            let src = take(state, packed)?;
            let mut qv = take(state, q)?;
            let mut gv = take(state, gate)?;
            for r in 0..rows {
                for h in 0..heads {
                    let from = r * heads * 2 * hd + h * 2 * hd;
                    let to = r * heads * hd + h * hd;
                    if let (Some(s), Some(d)) = (src.get(from..from + hd), qv.get_mut(to..to + hd))
                    {
                        d.copy_from_slice(s);
                    }
                    if let (Some(s), Some(d)) =
                        (src.get(from + hd..from + 2 * hd), gv.get_mut(to..to + hd))
                    {
                        d.copy_from_slice(s);
                    }
                }
            }
            put(state, q, qv);
            put(state, gate, gv);
        }
        HostOp::TopkSoftmax {
            logits,
            routes,
            weights,
            rows,
            width,
            experts,
            top_k,
        } => {
            let (rows, width, experts, top_k) = (
                *rows as usize,
                *width as usize,
                *experts as usize,
                *top_k as usize,
            );
            let src = take(state, logits)?;
            let mut r = vec![0u32; rows * top_k];
            let mut w = vec![0f32; rows * top_k];
            for t in 0..rows {
                let row = src
                    .get(t * width..t * width + experts.min(width))
                    .unwrap_or(&[]);
                let mut picked: Vec<usize> = Vec::with_capacity(top_k);
                for _ in 0..top_k {
                    let mut best: Option<(usize, f32)> = None;
                    for (i, v) in row.iter().enumerate() {
                        if v.is_nan() || picked.contains(&i) {
                            continue;
                        }
                        if best.is_none_or(|(_, b)| *v > b) {
                            best = Some((i, *v));
                        }
                    }
                    match best {
                        Some((i, _)) => picked.push(i),
                        None => break,
                    }
                }
                let chosen: Vec<f32> = picked.iter().map(|i| row[*i]).collect();
                let m = chosen.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let z: f32 = chosen.iter().map(|v| (v - m).exp()).sum();
                for s in 0..top_k {
                    let at = t * top_k + s;
                    match picked.get(s).zip(chosen.get(s)) {
                        Some((i, v)) => {
                            r[at] = *i as u32;
                            w[at] = (v - m).exp() / z;
                        }
                        None => {
                            r[at] = (-1i32) as u32;
                            w[at] = 0.0;
                        }
                    }
                }
            }
            state.insert(routes.to_string(), r);
            put(state, weights, w);
        }
        HostOp::SplitQkv {
            packed,
            q,
            k,
            v,
            rows,
            q_width,
            kv_width,
        } => {
            let (rows, qw, kw) = (*rows as usize, *q_width as usize, *kv_width as usize);
            let width = qw + 2 * kw;
            let src = take(state, packed)?;
            let mut qv = take(state, q)?;
            let mut kv = take(state, k)?;
            let mut vv = take(state, v)?;
            for r in 0..rows {
                let Some(row) = src.get(r * width..(r + 1) * width) else {
                    break;
                };
                if let Some(d) = qv.get_mut(r * qw..(r + 1) * qw) {
                    d.copy_from_slice(&row[..qw]);
                }
                if let Some(d) = kv.get_mut(r * kw..(r + 1) * kw) {
                    d.copy_from_slice(&row[qw..qw + kw]);
                }
                if let Some(d) = vv.get_mut(r * kw..(r + 1) * kw) {
                    d.copy_from_slice(&row[qw + kw..]);
                }
            }
            put(state, q, qv);
            put(state, k, kv);
            put(state, v, vv);
        }
        HostOp::EmbedWeighted {
            ids,
            weights,
            table,
            y,
            rows,
            taps,
            width,
            limit,
        } => {
            let (rows, taps, width, limit) = (
                *rows as usize,
                *taps as usize,
                *width as usize,
                *limit as usize,
            );
            let id_words = state.get(ids).cloned().unwrap_or_default();
            let w = take(state, weights)?;
            let t = take(state, table)?;
            let mut out = vec![0f32; rows * width];
            for r in 0..rows {
                for tap in 0..taps {
                    let id = id_words.get(r * taps + tap).map_or(0, |v| *v as i32);
                    let id = if id < 0 || id as usize >= limit {
                        0
                    } else {
                        id as usize
                    };
                    let g = w.get(r * taps + tap).copied().unwrap_or(0.0);
                    let src = t.get(id * width..(id + 1) * width).unwrap_or(&[]);
                    for (o, v) in out[r * width..(r + 1) * width].iter_mut().zip(src) {
                        *o += g * v;
                    }
                }
            }
            put(state, y, out);
        }
        HostOp::ArgmaxMerge {
            parts,
            y,
            rows,
            cg,
            block,
            column,
            y_width,
        } => {
            let (rows, cg, block, column, y_width) = (
                *rows as usize,
                *cg as usize,
                *block as usize,
                *column as usize,
                *y_width as usize,
            );
            let src = take(state, parts)?;
            let mut out = state.get(y).cloned().unwrap_or_default();
            out.resize(rows * y_width, 0);
            for r in 0..rows {
                let pairs = src.get(r * 2 * cg..(r + 1) * 2 * cg).unwrap_or(&[]);
                let mut best: Option<(usize, f32)> = None;
                for c in 0..cg {
                    let (Some(&v), Some(&i)) = (pairs.get(2 * c), pairs.get(2 * c + 1)) else {
                        continue;
                    };
                    let local = i.to_bits() as i32;
                    if local < 0 || v.is_nan() {
                        continue;
                    }
                    if best.is_none_or(|(_, b)| v > b) {
                        best = Some((c * block + local as usize, v));
                    }
                }
                out[r * y_width + column] = best.map_or(0, |(i, _)| i as u32);
            }
            state.insert(y.to_string(), out);
        }
        HostOp::Argmax {
            x,
            y,
            rows,
            width,
            column,
            y_width,
        } => {
            let (rows, width, column, y_width) = (
                *rows as usize,
                *width as usize,
                *column as usize,
                *y_width as usize,
            );
            let src = take(state, x)?;
            let mut out = state.get(y).cloned().unwrap_or_default();
            out.resize(rows * y_width, 0);
            for r in 0..rows {
                let row = src.get(r * width..(r + 1) * width).unwrap_or(&[]);
                let mut best: Option<(usize, f32)> = None;
                for (i, v) in row.iter().enumerate() {
                    if v.is_nan() {
                        continue;
                    }
                    if best.is_none_or(|(_, b)| *v > b) {
                        best = Some((i, *v));
                    }
                }
                out[r * y_width + column] = best.map_or(0, |(i, _)| i as u32);
            }
            state.insert(y.to_string(), out);
        }
        HostOp::Matmul { act, w, y, m, k, n } => {
            let (m, k, n) = (*m as usize, *k as usize, *n as usize);
            let a = take(state, act)?;
            let wt = take(state, w)?;
            let mut out = take(state, y)?;
            for i in 0..m {
                let Some(row) = a.get(i * k..(i + 1) * k) else {
                    break;
                };
                for j in 0..n {
                    let Some(wr) = wt.get(j * k..(j + 1) * k) else {
                        break;
                    };
                    let dot: f32 = row.iter().zip(wr).map(|(x, y)| x * y).sum();
                    if let Some(slot) = out.get_mut(i * n + j) {
                        *slot = dot;
                    }
                }
            }
            put(state, y, out);
        }
        HostOp::MatmulGrouped {
            x,
            w,
            routes,
            y,
            rows,
            groups,
            k,
            n,
            experts,
        } => {
            let (rows, groups, k, n, experts) = (
                *rows as usize,
                *groups as usize,
                *k as usize,
                *n as usize,
                *experts as usize,
            );
            let a = take(state, x)?;
            let wt = take(state, w)?;
            let route_words = state.get(routes).cloned().unwrap_or_default();
            let mut out = take(state, y)?;
            out.resize(rows * groups * n, 0.0);
            for r in 0..rows {
                for g in 0..groups {
                    let e = route_words.get(r * groups + g).map_or(-1, |v| *v as i32);
                    if e < 0 || e as usize >= experts {
                        continue;
                    }
                    let Some(row) = a.get((r * groups + g) * k..(r * groups + g + 1) * k) else {
                        break;
                    };
                    let block = e as usize * n * k;
                    for j in 0..n {
                        let Some(wr) = wt.get(block + j * k..block + (j + 1) * k) else {
                            break;
                        };
                        out[(r * groups + g) * n + j] =
                            row.iter().zip(wr).map(|(x, y)| x * y).sum();
                    }
                }
            }
            put(state, y, out);
        }
        HostOp::TopK {
            x,
            values,
            indices,
            rows,
            width,
            k,
        } => {
            let (rows, width, k) = (*rows as usize, *width as usize, *k as usize);
            let src = take(state, x)?;
            let mut vs = vec![0f32; rows * k];
            let mut is = vec![0u32; rows * k];
            for r in 0..rows {
                let row = src.get(r * width..(r + 1) * width).unwrap_or(&[]);
                let mut picked: Vec<usize> = Vec::with_capacity(k);
                for s in 0..k {
                    let mut best: Option<(usize, f32)> = None;
                    for (i, v) in row.iter().enumerate() {
                        if v.is_nan() || picked.contains(&i) {
                            continue;
                        }
                        if best.is_none_or(|(_, b)| *v > b) {
                            best = Some((i, *v));
                        }
                    }
                    let Some((i, v)) = best else {
                        break;
                    };
                    picked.push(i);
                    vs[r * k + s] = v;
                    is[r * k + s] = i as u32;
                }
            }
            put(state, values, vs);
            state.insert(indices.to_string(), is);
        }
        HostOp::SelectorWalk {
            cand,
            indptr,
            unary,
            hp,
            tokens,
            pred,
            succ,
            picks,
            rows,
            k,
            rank,
            vocab,
            first,
        } => {
            let (rows, k, rank, vocab, first) = (
                *rows as usize,
                *k as usize,
                *rank as usize,
                *vocab as usize,
                *first as usize,
            );
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let cands = ints(cand);
            let ip = ints(indptr);
            let toks = ints(tokens);
            let un = take(state, unary)?;
            let hv = match hp {
                Some(h) => Some(take(state, h)?),
                None => None,
            };
            let pr = take(state, pred)?;
            let sc = take(state, succ)?;
            let mut out = ints(picks);
            out.resize(rows, 0);
            let at = |v: &[i32], i: usize| v.get(i).copied().unwrap_or(0);
            for l in 0..ip.len().saturating_sub(1) {
                let (lo, hi) = (at(&ip, l), at(&ip, l + 1));
                if hi <= lo || lo < 0 || lo as usize >= rows {
                    continue;
                }
                let (lo, hi) = (lo as usize, (hi as usize).min(rows));
                if first > 0 {
                    out[lo] = at(&cands, lo * k);
                }
                let mut prev = at(&toks, lo);
                for (row, slot) in out.iter_mut().enumerate().take(hi).skip(lo + first) {
                    let mut scores = Vec::with_capacity(k);
                    for c in 0..k {
                        let id = at(&cands, row * k + c);
                        let mut s = un.get(row * k + c).copied().unwrap_or(0.0);
                        if prev >= 0 && (prev as usize) < vocab && id >= 0 && (id as usize) < vocab
                        {
                            let bilinear: f32 = (0..rank)
                                .map(|d| {
                                    let h = hv.as_ref().map_or(1.0, |h| {
                                        h.get(row * rank + d).copied().unwrap_or(0.0)
                                    });
                                    pr.get(prev as usize * rank + d).copied().unwrap_or(0.0)
                                        * h
                                        * sc.get(id as usize * rank + d).copied().unwrap_or(0.0)
                                })
                                .sum();
                            s += bilinear;
                        }
                        scores.push(s);
                    }
                    // The first strict maximum; a NaN never wins past slot 0.
                    let mut best = 0usize;
                    if !scores[0].is_nan() {
                        let clean = |v: f32| if v.is_nan() { f32::NEG_INFINITY } else { v };
                        for (c, s) in scores.iter().enumerate() {
                            if clean(*s) > clean(scores[best]) {
                                best = c;
                            }
                        }
                    }
                    let pick = at(&cands, row * k + best);
                    *slot = pick;
                    prev = pick;
                }
            }
            state.insert(picks.to_string(), out.iter().map(|v| *v as u32).collect());
        }
        HostOp::PleNgramIds {
            ids,
            indptr,
            slots,
            state: slab,
            out,
            rows,
            eos,
            mults,
            primes,
            offsets,
            heads_per_ngram,
        } => {
            let (rows, hpn) = (*rows as usize, *heads_per_ngram as usize);
            let eos = *eos as i32;
            let span = mults.len() - 1;
            let heads = primes.len();
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let id_words = ints(ids);
            let slot_words = ints(slots);
            let mut cells = ints(slab);
            let slots_rows = cells.len().checked_div(span).unwrap_or(0);
            let lanes: Vec<(usize, usize)> = match indptr {
                Some(ip) => {
                    let ip = ints(ip);
                    ip.windows(2)
                        .map(|w| (w[0].max(0) as usize, (w[1].max(w[0]).max(0)) as usize))
                        .collect()
                }
                None => (0..rows).map(|t| (t, t + 1)).collect(),
            };
            let mut result = vec![0u32; rows * heads];
            let id_at = |t: usize| id_words.get(t).copied().unwrap_or(0);
            for (b, e) in lanes {
                let (b, e) = (b.min(rows), e.min(rows));
                let n = e - b;
                let slot = slot_words.get(b).copied().unwrap_or(-1);
                let past: Vec<i32> = if slot >= 0 && (slot as usize) < slots_rows {
                    cells[slot as usize * span..(slot as usize + 1) * span].to_vec()
                } else {
                    vec![0; span]
                };
                for j in 0..n {
                    let t = b + j;
                    let mut window = vec![id_at(t)];
                    let mut crossed = false;
                    for p in 1..=span {
                        let src = j as i64 - p as i64;
                        let mut w = if src >= 0 {
                            id_at(t - p)
                        } else {
                            let cell = past[(span as i64 + src) as usize];
                            if cell == 0 { eos } else { cell - 1 }
                        };
                        if crossed {
                            w = eos;
                        }
                        crossed |= w == eos;
                        window.push(w);
                    }
                    let mut mixed = 0u64;
                    for (p, w) in window.iter().enumerate() {
                        mixed ^= (i64::from(*w) as u64).wrapping_mul(mults[p]);
                        if p == 0 {
                            continue;
                        }
                        let lo = (p - 1) * hpn;
                        for i in 0..hpn {
                            let r = (mixed % primes[lo + i]).wrapping_add(offsets[lo + i]);
                            result[t * heads + lo + i] = (r & 0xFFFF_FFFF) as u32;
                        }
                    }
                }
                if n > 0 && slot >= 0 && (slot as usize) < slots_rows {
                    let next: Vec<i32> = (0..span)
                        .map(|s| {
                            let src = n as i64 - span as i64 + s as i64;
                            if src >= 0 {
                                id_at(b + src as usize).wrapping_add(1)
                            } else {
                                past[(src + span as i64) as usize]
                            }
                        })
                        .collect();
                    cells[slot as usize * span..(slot as usize + 1) * span].copy_from_slice(&next);
                }
            }
            state.insert(out.to_string(), result);
            state.insert(slab.to_string(), cells.iter().map(|v| *v as u32).collect());
        }
        HostOp::PageWrite {
            src,
            table,
            write_page,
            write_offset,
            rows,
            width,
            page_size,
            table_rows,
        } => {
            let (rows, width, ps, table_rows) = (
                *rows as usize,
                *width as usize,
                *page_size as i64,
                *table_rows as usize,
            );
            let a = take(state, src)?;
            let mut t = take(state, table)?;
            let pages = state.get(write_page).cloned().unwrap_or_default();
            let offs = state.get(write_offset).cloned().unwrap_or_default();
            for r in 0..rows {
                let page = i64::from(pages.get(r).map_or(-1, |v| *v as i32));
                let off = i64::from(offs.get(r).map_or(-1, |v| *v as i32));
                if page < 0 || off < 0 || off >= ps {
                    continue;
                }
                let cell = (page * ps + off) as usize;
                let Some(row) = a.get(r * width..(r + 1) * width) else {
                    break;
                };
                if cell < table_rows
                    && let Some(dst) = t.get_mut(cell * width..(cell + 1) * width)
                {
                    dst.copy_from_slice(row);
                }
            }
            put(state, table, t);
        }
        HostOp::Boundary {
            positions,
            request_of_token,
            row_valid,
            boundary_pos,
            boundary_req,
            boundary_rope,
            rows,
            ratio,
        } => {
            let (rows, ratio) = (*rows as usize, *ratio as i32);
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (p, rq, v) = (ints(positions), ints(request_of_token), ints(row_valid));
            let at = |t: &[i32], i: usize| t.get(i).copied().unwrap_or(0);
            let mut bpos = vec![0u32; rows];
            let mut breq = vec![0u32; rows];
            let mut brope = vec![0u32; rows];
            for r in 0..rows {
                let pos = at(&p, r);
                let valid = at(&v, r) != 0;
                let closes = (pos.wrapping_add(1)) % ratio == 0;
                let is_b = valid && closes;
                bpos[r] = if is_b { pos } else { -1 } as u32;
                breq[r] = at(&rq, r) as u32;
                brope[r] = if is_b { (pos / ratio) * ratio } else { 0 } as u32;
            }
            state.insert(boundary_pos.to_string(), bpos);
            state.insert(boundary_req.to_string(), breq);
            state.insert(boundary_rope.to_string(), brope);
        }
        HostOp::BlockMean {
            boundary_pos,
            boundary_req,
            keys,
            indices,
            indptr,
            entries,
            rows,
            head_dim,
            ratio,
            page_size,
        } => {
            let (rows, hd, ratio, ps) = (
                *rows as usize,
                *head_dim as usize,
                *ratio as i64,
                i64::from(*page_size),
            );
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (bpos, breq, idx, ip) = (
                ints(boundary_pos),
                ints(boundary_req),
                ints(indices),
                ints(indptr),
            );
            let table = take(state, keys)?;
            // The index cache plane is `[rows][hd]`, as the op checked.
            let key_width = hd;
            let mut out = vec![0f32; rows * hd];
            for r in 0..rows {
                let bp = i64::from(bpos.get(r).copied().unwrap_or(-1));
                if bp < 0 {
                    continue;
                }
                let rq = breq.get(r).copied().unwrap_or(0);
                let mut sum = vec![0f32; hd];
                for i in 0..ratio {
                    let pos = bp + i - (ratio - 1);
                    if pos < 0 {
                        continue;
                    }
                    if let Some(cell) = cell_of(&idx, &ip, rq, pos, ps)
                        && let Some(row) = table.get(cell * key_width..cell * key_width + hd)
                    {
                        for (s, v) in sum.iter_mut().zip(row) {
                            *s += v;
                        }
                    }
                }
                for (o, s) in out[r * hd..(r + 1) * hd].iter_mut().zip(&sum) {
                    *o = s / ratio as f32;
                }
            }
            put(state, entries, out);
        }
        HostOp::PoolWrite {
            entries,
            boundary_pos,
            boundary_req,
            keys,
            indices,
            indptr,
            rows,
            width,
            page_size,
        } => {
            let (rows, width, ps) = (*rows as usize, *width as usize, i64::from(*page_size));
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (bpos, breq, idx, ip) = (
                ints(boundary_pos),
                ints(boundary_req),
                ints(indices),
                ints(indptr),
            );
            let e = take(state, entries)?;
            let mut table = take(state, keys)?;
            for r in 0..rows {
                let bp = i64::from(bpos.get(r).copied().unwrap_or(-1));
                if bp < 0 {
                    continue;
                }
                let rq = breq.get(r).copied().unwrap_or(0);
                let Some(row) = e.get(r * width..(r + 1) * width) else {
                    break;
                };
                if let Some(cell) = cell_of(&idx, &ip, rq, bp, ps)
                    && let Some(dst) = table.get_mut(cell * width..(cell + 1) * width)
                {
                    dst.copy_from_slice(row);
                }
            }
            put(state, keys, table);
        }
        HostOp::IndexTopk {
            q,
            weights,
            keys,
            indices,
            indptr,
            positions,
            request_of_token,
            selection,
            rows,
            heads,
            head_dim,
            top_k,
            ratio,
            page_size,
            max_pages,
        } => {
            let (rows, h, d, k, ratio, ps) = (
                *rows as usize,
                *heads as usize,
                *head_dim as usize,
                *top_k as usize,
                i64::from((*ratio).max(1)),
                i64::from(*page_size),
            );
            let nk = i64::from(*max_pages) * ps / ratio;
            let ints = |name: &str| -> Vec<i32> {
                state
                    .get(name)
                    .map(|w| w.iter().map(|v| *v as i32).collect())
                    .unwrap_or_default()
            };
            let (idx, ip, pos, rq) = (
                ints(indices),
                ints(indptr),
                ints(positions),
                ints(request_of_token),
            );
            let qv = take(state, q)?;
            let wv = match weights {
                Some(w) => Some(take(state, w)?),
                None => None,
            };
            let table = take(state, keys)?;
            let mut sel = vec![-1i32; rows * k];
            for r in 0..rows {
                let p = i64::from(pos.get(r).copied().unwrap_or(-1));
                let req = rq.get(r).copied().unwrap_or(0);
                let nkeys = if p + 1 > 0 {
                    ((p + 1) / ratio).clamp(0, nk)
                } else {
                    0
                } as usize;
                let mut scores = Vec::with_capacity(nkeys);
                for j in 0..nkeys {
                    let kpos = (j as i64 + 1) * ratio - 1;
                    let key: &[f32] = cell_of(&idx, &ip, req, kpos, ps)
                        .and_then(|cell| table.get(cell * d..(cell + 1) * d))
                        .unwrap_or(&[]);
                    let mut s = 0f32;
                    for hh in 0..h {
                        let qrow = &qv[(r * h + hh) * d..(r * h + hh + 1) * d];
                        let dot: f32 = qrow.iter().zip(key).map(|(a, b)| a * b).sum();
                        let w = wv
                            .as_ref()
                            .map_or(1.0, |w| w.get(r * h + hh).copied().unwrap_or(0.0));
                        s += dot.max(0.0) * w;
                    }
                    scores.push(s);
                }
                let taken: Vec<usize> = if nkeys <= k {
                    (0..nkeys).collect()
                } else {
                    let mut lo = scores.iter().copied().fold(f32::INFINITY, f32::min);
                    let mut hi = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    for _ in 0..40 {
                        let mid = (lo + hi) * 0.5;
                        let cnt = scores.iter().filter(|s| **s >= mid).count();
                        if cnt > k {
                            lo = mid;
                        } else {
                            hi = mid;
                        }
                    }
                    (0..nkeys).filter(|j| scores[*j] >= hi).take(k).collect()
                };
                for (slot, j) in taken.iter().enumerate() {
                    sel[r * k + slot] = *j as i32;
                }
            }
            state.insert(
                selection.to_string(),
                sel.iter().map(|v| *v as u32).collect(),
            );
        }
        HostOp::GroupRoutes {
            routes,
            rows,
            groups,
        } => {
            let (rows, groups) = (*rows as usize, *groups as usize);
            let out: Vec<u32> = (0..rows * groups).map(|i| (i % groups) as u32).collect();
            state.insert(routes.to_string(), out);
        }
        HostOp::EmbedConcat {
            ids,
            table,
            y,
            rows,
            heads,
            width,
            limit,
        } => {
            let (rows, heads, width, limit) = (
                *rows as usize,
                *heads as usize,
                *width as usize,
                *limit as usize,
            );
            let id_words = state.get(ids).cloned().unwrap_or_default();
            let table = take(state, table)?;
            let mut out = take(state, y)?;
            out.resize(rows * heads * width, 0.0);
            for slice in 0..rows * heads {
                let id = id_words.get(slice).map_or(-1, |w| *w as i32);
                let dst = &mut out[slice * width..(slice + 1) * width];
                if id < 0 || id as usize >= limit {
                    dst.fill(0.0);
                    continue;
                }
                let src = table
                    .get(id as usize * width..(id as usize + 1) * width)
                    .unwrap_or(&[]);
                dst[..src.len()].copy_from_slice(src);
            }
            put(state, y, out);
        }
        HostOp::Cells {
            tgrid,
            positions,
            y,
            clips,
            rows,
            axis,
            width,
            base,
        } => {
            let (rows, width, axis) = (*rows as usize, *width as usize, *axis as usize);
            let boxes = rebased(clip_boxes(state, tgrid, *clips), *base);
            let pos = state.get(positions).cloned().unwrap_or_default();
            let mut out = vec![0f32; rows * width];
            for r in 0..rows {
                let o = &mut out[r * width..(r + 1) * width];
                match clip_of(&boxes, r as i64) {
                    Some((_, b, local)) => {
                        let [t, h, w] = unravel(b, local);
                        o[0] = t as f32;
                        o[1] = h as f32;
                        o[2] = w as f32;
                        o[width - 3] = b[0] as f32;
                        o[width - 2] = b[1] as f32;
                        o[width - 1] = b[2] as f32;
                    }
                    None => o[axis] = pos.get(r).map_or(0, |w| *w as i32) as f32,
                }
            }
            put(state, y, out);
        }
        HostOp::ConcatCols {
            x,
            y,
            rows,
            width,
            stride,
            at,
        } => {
            let (rows, width, stride, at) = (
                *rows as usize,
                *width as usize,
                *stride as usize,
                *at as usize,
            );
            let src = take(state, x)?;
            let mut out = take(state, y)?;
            out.resize(rows * stride, 0.0);
            for r in 0..rows {
                let Some(row) = src.get(r * width..(r + 1) * width) else {
                    continue;
                };
                out[r * stride + at..r * stride + at + width].copy_from_slice(row);
            }
            put(state, y, out);
        }
        HostOp::GroupRows {
            csr,
            groups,
            lanes,
            y,
            csr_len,
            lane_count,
            rows,
        } => {
            let word = |name: &str, at: i64| -> Option<i64> {
                let at = usize::try_from(at).ok()?;
                state.get(name)?.get(at).map(|w| i64::from(*w as i32))
            };
            let out: Vec<f32> = (0..*rows as i64)
                .map(|r| {
                    let l = word(lanes, r).unwrap_or(-1);
                    let g = if (0..i64::from(*lane_count)).contains(&l) {
                        word(groups, l).unwrap_or(-1)
                    } else {
                        -1
                    };
                    if !(0..i64::from(*csr_len) - 1).contains(&g) {
                        return 0.0;
                    }
                    match (word(csr, g), word(csr, g + 1)) {
                        (Some(a), Some(b)) => (b - a) as f32,
                        _ => 0.0,
                    }
                })
                .collect();
            put(state, y, out);
        }
        HostOp::GroupFirst {
            x,
            csr,
            groups,
            lanes,
            y,
            csr_len,
            lane_count,
            rows,
            width,
            x_rows,
        } => {
            let word = |name: &str, at: i64| -> Option<i64> {
                let at = usize::try_from(at).ok()?;
                state.get(name)?.get(at).map(|w| i64::from(*w as i32))
            };
            let (rows, width) = (*rows as usize, *width as usize);
            let src = take(state, x)?;
            let mut out = vec![0f32; rows * width];
            for r in 0..rows {
                let l = word(lanes, r as i64).unwrap_or(-1);
                let g = if (0..i64::from(*lane_count)).contains(&l) {
                    word(groups, l).unwrap_or(-1)
                } else {
                    -1
                };
                if !(0..i64::from(*csr_len) - 1).contains(&g) {
                    continue;
                }
                let (Some(a), Some(b)) = (word(csr, g), word(csr, g + 1)) else {
                    continue;
                };
                if b <= a || !(0..i64::from(*x_rows)).contains(&a) {
                    continue;
                }
                let a = a as usize;
                if let Some(row) = src.get(a * width..(a + 1) * width) {
                    out[r * width..(r + 1) * width].copy_from_slice(row);
                }
            }
            put(state, y, out);
        }
        HostOp::Patch {
            x,
            grid,
            y,
            o_grid,
            clips,
            rows,
            channels,
            p,
            fold,
            bases,
        } => {
            let (rows, c) = (*rows as usize, *channels as usize);
            let [r1, r2, r3] = p.map(i64::from);
            let vol = (r1 * r2 * r3) as usize;
            let ins = rebased(clip_boxes(state, grid, *clips), bases[0]);
            let outs = rebased(clip_boxes(state, o_grid, *clips), bases[1]);
            let src = take(state, x)?;
            let width = if *fold { c * vol } else { c };
            let mut out = vec![0f32; rows * width];
            for row in 0..rows {
                let Some((lane, ob, local)) = clip_of(&outs, row as i64) else {
                    continue;
                };
                let Some(&ib) = ins.get(lane) else { continue };
                let [ot, oh, ow] = unravel(ob, local);
                let dst = &mut out[row * width..(row + 1) * width];
                if *fold {
                    for (col, d) in dst.iter_mut().enumerate() {
                        let (cin, blk) = ((col / vol) as i64, (col % vol) as i64);
                        let (i1, i2, i3) = (blk / (r2 * r3), (blk / r3) % r2, blk % r3);
                        let s = ravel(ib, ot * r1 + i1, oh * r2 + i2, ow * r3 + i3);
                        *d = src
                            .get(s as usize * c + cin as usize)
                            .copied()
                            .unwrap_or(0.0);
                    }
                } else {
                    let s = ravel(ib, ot / r1, oh / r2, ow / r3) as usize;
                    let blk = (((ot % r1) * r2 + oh % r2) * r3 + ow % r3) as usize;
                    for (col, d) in dst.iter_mut().enumerate() {
                        *d = src
                            .get(s * c * vol + col * vol + blk)
                            .copied()
                            .unwrap_or(0.0);
                    }
                }
            }
            put(state, y, out);
        }
    }
    for name in &host.rounds {
        if let Some(words) = state.get_mut(name) {
            for w in words.iter_mut() {
                *w = crate::bench::round_bf16(f32::from_bits(*w)).to_bits();
            }
        }
    }
    Ok(())
}
