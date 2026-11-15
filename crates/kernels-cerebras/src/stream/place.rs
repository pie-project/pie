//! Where a stream unit's values sit and how its rows are ordered: each unit's CY roots and the
//! column slices each holds ([`Layout`], chosen so every root region's chunk and every head is
//! whole at one root), the weight tiles that follows from, and the row stream's schedule
//! ([`Schedule`]: what each unit's CX roots do with each block of rows arriving, and the order
//! each K / V column stores and attends rows in).
//!
//! The row side is uniform: every unit has `py` CX roots (the weight rows), each holding
//! `width / py` columns of every row-side value in order. The column side has the attention's
//! head roots first (root `h` holds kv head `h`'s q, k, v and its output, so it alone talks to
//! the head's K / V column) and then `rest` roots, chosen as the fewest whose tiles and held rows
//! fit.

use std::collections::BTreeMap;

use poem_ir::ValueId;
use serde::Serialize;

use super::unit::{Cols, Reader, Side, Step, Unit};

type R<T> = Result<T, String>;

/// A tile's 16-bit words a weight PE holds at most.
pub const TILE_BUDGET: u32 = 12_288;
/// Column roots a unit takes at most besides its head roots (a CS-3 is 750 PEs wide).
pub const MAX_REST: u32 = 512;
/// A root's words of held rows (host rows in and out, rows held across the barrier) at most.
pub const ROOT_BUDGET: u32 = 6_144;

/// Columns `[from, from + len)` of `base`, held at one root.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Slice {
    pub base: u32,
    pub from: u32,
    pub len: u32,
}

impl Slice {
    fn of(base: ValueId, from: u32, len: u32) -> Slice {
        Slice {
            base: base.0,
            from,
            len,
        }
    }

    fn covers(&self, base: ValueId, col: u32) -> bool {
        self.base == base.0 && col >= self.from && col < self.from + self.len
    }
}

/// A unit's roots: `py` row roots, its column roots' slices (head roots first).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Layout {
    pub py: u32,
    pub heads: u32,
    pub cols: Vec<Vec<Slice>>,
    /// The row store: the rows a row root holds over the barrier are kept by its row's weight
    /// PEs (row `t` by column `t % px`), cast to them when made and summed back up the row
    /// chain after the barrier ([`store_words`]).
    pub spill: bool,
}

/// Where a root region runs: the roots `first .. first + roots` of its side, `chunk` columns each.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct RegionAt {
    pub first: u32,
    pub roots: u32,
    pub chunk: u32,
}

impl Layout {
    /// The column roots.
    #[must_use]
    pub fn px(&self) -> u32 {
        self.cols.len() as u32
    }

    /// The slices of `c`'s columns root `r` holds, in its order: (offset into `c`, slice).
    #[must_use]
    pub fn parts(&self, r: u32, c: Cols) -> Vec<(u32, Slice)> {
        let mut out = Vec::new();
        for s in &self.cols[r as usize] {
            if s.base != c.base.0 {
                continue;
            }
            let lo = s.from.max(c.from);
            let hi = (s.from + s.len).min(c.from + c.width);
            if lo < hi {
                out.push((
                    lo - c.from,
                    Slice {
                        base: s.base,
                        from: lo,
                        len: hi - lo,
                    },
                ));
            }
        }
        out
    }

    /// Words of `c` at root `r`.
    #[must_use]
    pub fn words(&self, r: u32, c: Cols) -> u32 {
        self.parts(r, c).iter().map(|(_, s)| s.len).sum()
    }

    /// Where the region of `width` columns over `cands` (its row parameters' and results'
    /// columns) runs: on the head roots where they hold one of them chunk by chunk, else on the
    /// rest.
    #[must_use]
    pub fn region_at(&self, side: Side, width: u32, cands: &[Cols]) -> RegionAt {
        if side == Side::Row {
            return RegionAt {
                first: 0,
                roots: self.py,
                chunk: width / self.py,
            };
        }
        if self.heads > 0 && width.is_multiple_of(self.heads) {
            let chunk = width / self.heads;
            if cands.iter().any(|c| {
                (0..self.heads).all(|h| {
                    self.cols[h as usize]
                        .iter()
                        .any(|s| s.covers(c.base, c.from + h * chunk))
                })
            }) {
                return RegionAt {
                    first: 0,
                    roots: self.heads,
                    chunk,
                };
            }
        }
        let rest = self.px() - self.heads;
        RegionAt {
            first: self.heads,
            roots: rest,
            chunk: width / rest.max(1),
        }
    }
}

/// A region step's row parameters' and results' columns (where its chunks are).
#[must_use]
pub fn region_cols(r: &Reader, chain: &poem_compiler::fuse::chain::Chain) -> Vec<Cols> {
    let mut v: Vec<Cols> = row_params(r, chain).into_iter().map(|(_, c)| c).collect();
    v.extend(chain.outs.iter().map(|o| Cols {
        base: o.value,
        from: 0,
        width: chain.region.width,
    }));
    v
}

/// The row parameters of a region step: (param index, columns).
#[must_use]
pub fn row_params(r: &Reader, chain: &poem_compiler::fuse::chain::Chain) -> Vec<(usize, Cols)> {
    chain
        .ins
        .iter()
        .enumerate()
        .filter(|(k, i)| {
            chain.region.params[*k] != kernels_ir::Shape::Scalar
                && !i.shared
                && i.by.is_none()
                && !r.lane(i.value)
        })
        .map(|(k, i)| {
            let c = r.cols(i.value);
            (
                k,
                Cols {
                    base: c.base,
                    from: c.from + i.from,
                    width: chain.region.width,
                },
            )
        })
        .collect()
}

fn push(cols: &mut [Vec<Slice>], r: usize, s: Slice) {
    if !cols[r].contains(&s) {
        cols[r].push(s);
    }
}

fn covered(cols: &[Vec<Slice>], base: ValueId, col: u32) -> bool {
    cols.iter().flatten().any(|s| s.covers(base, col))
}

/// The layout of `u` with `rest` column roots besides its head roots; `None` where a width does
/// not divide.
fn layout_with(r: &Reader, u: &Unit, py: u32, rest: u32) -> Option<Layout> {
    let heads = u
        .steps
        .iter()
        .find_map(|s| match s {
            Step::Store { .. } => u.joint.map(|j| j.kv_heads),
            _ => None,
        })
        .unwrap_or(0);
    let n = (heads + rest) as usize;
    let mut cols: Vec<Vec<Slice>> = vec![Vec::new(); n];
    if let Some(j) = u.joint.filter(|_| heads > 0) {
        let (qw, d) = (j.q_heads / j.kv_heads * j.head_dim, j.head_dim);
        for s in &u.steps {
            match s {
                Step::Store { q, k, v } => {
                    for h in 0..heads {
                        push(
                            &mut cols,
                            h as usize,
                            Slice::of(q.base, q.from + h * qw, qw),
                        );
                        push(&mut cols, h as usize, Slice::of(k.base, k.from + h * d, d));
                        push(&mut cols, h as usize, Slice::of(v.base, v.from + h * d, d));
                    }
                }
                Step::Attend { out, .. } => {
                    for h in 0..heads {
                        push(&mut cols, h as usize, Slice::of(*out, h * qw, qw));
                    }
                }
                _ => {}
            }
        }
    }
    let lay = |cols: Vec<Vec<Slice>>| Layout {
        py,
        heads,
        cols,
        spill: false,
    };
    // the column regions, in order
    for s in &u.steps {
        let Step::Region {
            side: Side::Col,
            chain,
        } = s
        else {
            continue;
        };
        let w = chain.region.width;
        let at = lay(cols.clone()).region_at(Side::Col, w, &region_cols(r, chain));
        if at.roots == 0 || !w.is_multiple_of(at.roots) {
            return None;
        }
        for i in 0..at.roots {
            let root = (at.first + i) as usize;
            for (_, c) in row_params(r, chain) {
                push(
                    &mut cols,
                    root,
                    Slice::of(c.base, c.from + i * at.chunk, at.chunk),
                );
            }
            for o in &chain.outs {
                push(&mut cols, root, Slice::of(o.value, i * at.chunk, at.chunk));
            }
        }
    }
    // what is left: each value's uncovered columns over the rest, evenly
    let spread = |cols: &mut Vec<Vec<Slice>>, c: Cols| -> Option<()> {
        let left: Vec<u32> = (c.from..c.from + c.width)
            .filter(|&k| !covered(cols, c.base, k))
            .collect();
        if left.is_empty() {
            return Some(());
        }
        if rest == 0 || !(left.len() as u32).is_multiple_of(rest) {
            return None;
        }
        let per = left.len() / rest as usize;
        for i in 0..rest as usize {
            let part = &left[i * per..(i + 1) * per];
            let mut a = 0;
            while a < part.len() {
                let mut b = a + 1;
                while b < part.len() && part[b] == part[b - 1] + 1 {
                    b += 1;
                }
                push(
                    cols,
                    heads as usize + i,
                    Slice::of(c.base, part[a], (b - a) as u32),
                );
                a = b;
            }
        }
        Some(())
    };
    for s in &u.steps {
        match s {
            Step::Matvec {
                from: Side::Col,
                input,
                ..
            } => spread(&mut cols, *input)?,
            Step::Matvec {
                from: Side::Row,
                output,
                n,
                ..
            } => spread(
                &mut cols,
                Cols {
                    base: *output,
                    from: 0,
                    width: *n,
                },
            )?,
            Step::Out {
                value,
                side: Side::Col,
                ..
            } => spread(&mut cols, *value)?,
            _ => {}
        }
    }
    Some(lay(cols))
}

/// Words a weight PE of column `col` holds: each matvec's tile.
#[must_use]
pub fn tile_words(u: &Unit, l: &Layout, col: u32) -> u32 {
    u.steps
        .iter()
        .map(|s| tile_dims(s, l, col).map_or(0, |(a, b)| a * b))
        .sum()
}

/// A matvec step's tile at column `col`: (its rows: the x words it reads, its columns: the
/// partials it sends); `None` where the step is no matvec or the column holds none of it.
#[must_use]
pub fn tile_dims(s: &Step, l: &Layout, col: u32) -> Option<(u32, u32)> {
    let Step::Matvec {
        from,
        input,
        output,
        k,
        n,
        ..
    } = s
    else {
        return None;
    };
    let (a, b) = match from {
        Side::Row => (
            k / l.py,
            l.words(
                col,
                Cols {
                    base: *output,
                    from: 0,
                    width: *n,
                },
            ),
        ),
        Side::Col => (l.words(col, *input), n / l.py),
    };
    (a > 0 && b > 0).then_some((a, b))
}

/// Words of held rows at column root `col` (host rows, rows held over the barrier).
fn held_words(u: &Unit, l: &Layout, col: u32) -> u32 {
    let mut w = 0;
    for s in &u.steps {
        match s {
            Step::In {
                value,
                side: Side::Col,
                host: true,
            } => {
                w += u.rows
                    * l.words(
                        col,
                        Cols {
                            base: *value,
                            from: 0,
                            width: u32::MAX / 2,
                        },
                    );
            }
            Step::Out {
                value,
                side: Side::Col,
                host: true,
                ..
            } => w += u.rows * l.words(col, *value),
            _ => {}
        }
    }
    for v in &u.held {
        if u.sides.get(v) == Some(&Side::Col) {
            w += u.rows
                * l.words(
                    col,
                    Cols {
                        base: *v,
                        from: 0,
                        width: u32::MAX / 2,
                    },
                );
        }
    }
    w
}

/// Words of held rows at a row root of `py`: (the host's rows, rows held over the barrier).
fn row_held_words(r: &Reader, u: &Unit, py: u32) -> (u32, u32) {
    let (mut host, mut over) = (0, 0);
    let mut hin = Vec::new();
    for s in &u.steps {
        if let Step::In {
            value,
            side: Side::Row,
            host: true,
        } = s
        {
            host += u.rows * (r.width(*value) / py);
            hin.push(*value);
        }
    }
    for v in &u.held {
        if u.sides.get(v) == Some(&Side::Row) && !hin.contains(v) {
            over += u.rows * (r.width(*v) / py);
        }
    }
    (host, over)
}

/// Why the rows a unit holds at a root do not fit it: the host's rows (in or out, all a step's
/// at once) or rows held over the barrier.
fn held_refusal(u: &Unit, side: &str, host: u32, over: u32) -> String {
    let what = if host >= over {
        format!(
            "the host's {} rows a step held whole at a {side} root ({host} words > \
             {ROOT_BUDGET}): the host's rows streamed a batch at a time (M2b)",
            u.rows
        )
    } else {
        format!(
            "{} rows held over the barrier at a {side} root ({over} words > {ROOT_BUDGET}): a \
             row store off the roots, in the unit's weight PEs (M2b), or row replicas",
            u.rows
        )
    };
    format!("{}: {what}", u.name)
}

/// Words of the host's rows at column root `col` (in and out).
fn col_host_words(u: &Unit, l: &Layout, col: u32) -> u32 {
    let held: u32 = u
        .held
        .iter()
        .filter(|v| u.sides.get(*v) == Some(&Side::Col))
        .map(|v| {
            u.rows
                * l.words(
                    col,
                    Cols {
                        base: *v,
                        from: 0,
                        width: u32::MAX / 2,
                    },
                )
        })
        .sum();
    held_words(u, l, col) - held
}

/// 16-bit words a weight PE holds at most: its tiles and its share of the row store.
pub const WEIGHT_WORDS: u32 = 20_480;

/// The row store's f32 words a row (a row root's held chunks, each value's in step order) and
/// the rows a weight PE of a `px`-column unit keeps.
#[must_use]
pub fn store_words(r: &Reader, u: &Unit, py: u32, px: u32) -> (u32, u32) {
    let per = row_held_words(r, u, py).1 / u.rows.max(1);
    (per, u.rows.div_ceil(px.max(1)))
}

/// `u`'s layout: the fewest rest roots whose every tile (at most `tile` words) and held rows
/// fit. Rows held over the barrier that outgrow the row roots go to the row store
/// ([`Layout::spill`]; `spill`: always, where any are held); a unit whose rows outgrow its roots
/// otherwise is refused by what it holds. With `widen` (quarters), a unit whose column regions
/// run on its rest roots takes the fewest that fit at least `widen / 4` more than the fewest.
pub fn layout(r: &Reader, u: &Unit, py: u32, tile: u32, spill: bool, widen: u32) -> R<Layout> {
    let l0 = layout_fewest(r, u, py, tile, spill, 0)?;
    let rest0 = l0.px() - l0.heads;
    let on_rest = u.steps.iter().any(|s| match s {
        Step::Region {
            side: Side::Col,
            chain,
        } => {
            l0.region_at(Side::Col, chain.region.width, &region_cols(r, chain))
                .first
                == l0.heads
        }
        _ => false,
    });
    if widen == 0 || !on_rest || rest0 == 0 {
        return Ok(l0);
    }
    let want = rest0 + (rest0 * widen).div_ceil(4);
    Ok(layout_fewest(r, u, py, tile, spill, want).unwrap_or(l0))
}

/// [`layout`]'s fewest rest roots from `from` on.
fn layout_fewest(r: &Reader, u: &Unit, py: u32, tile: u32, spill: bool, from: u32) -> R<Layout> {
    let (host, over) = row_held_words(r, u, py);
    if host > ROOT_BUDGET {
        return Err(held_refusal(u, "row", host, over));
    }
    let spill = over > 0 && (spill || host + over > ROOT_BUDGET);
    let stored = |l: &Layout| -> bool {
        let (per, n) = store_words(r, u, py, l.px());
        (0..l.px()).all(|c| tile_words(u, l, c) + 2 * per * n <= WEIGHT_WORDS)
    };
    let mut short: Option<(u32, u32)> = None;
    let mut unstored = false;
    for rest in from..=MAX_REST {
        let Some(l) = layout_with(r, u, py, rest) else {
            continue;
        };
        if l.px() == 0 || !(0..l.px()).all(|c| tile_words(u, &l, c) < tile) {
            continue;
        }
        if spill && !stored(&l) {
            unstored = true;
            continue;
        }
        if (0..l.px()).all(|c| held_words(u, &l, c) <= ROOT_BUDGET) {
            return Ok(Layout { spill, ..l });
        }
        // the tiles fit, the held rows not: what the fewest roots hold
        if short.is_none() {
            let worst = (0..l.px())
                .max_by_key(|&c| held_words(u, &l, c))
                .unwrap_or(0);
            let h = col_host_words(u, &l, worst);
            short = Some((h, held_words(u, &l, worst) - h));
        }
    }
    Err(match short {
        Some((h, o)) => held_refusal(u, "column", h, o),
        None if unstored => format!(
            "{}: its row store ({} rows of {} words) fits beside the tiles at no column roots \
             up to {MAX_REST}",
            u.name,
            u.rows,
            over / u.rows.max(1)
        ),
        None => format!(
            "{}: no column roots up to {MAX_REST} fit its tiles ({tile} words)",
            u.name
        ),
    })
}

/// A run of a row stream's rows: `n` rows of class `class`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Block {
    pub class: usize,
    pub n: u32,
}

/// What a unit's CX roots do with a run of rows, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Seg {
    /// Its own rows `t0 ..` before the barrier (all of the body without one); `host`: from the
    /// host (no row stream in).
    Early { t0: u32, n: u32, host: bool },
    /// Its own rows after the barrier.
    Late { t0: u32, n: u32 },
    /// Another unit's rows, on.
    Fwd { n: u32 },
    /// Rows no later unit runs, dropped.
    Drop { n: u32 },
}

/// One unit's place in the row stream.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct UnitSched {
    pub segs: Vec<Seg>,
    /// Its own rows in order: (class, the class's row).
    pub rows: Vec<(usize, u32)>,
    /// Rows out on the stream.
    pub out: u32,
    /// Rows in on the stream.
    pub inn: u32,
}

/// A K / V column's order: stores from each unit of the block, then the queries (rows `p`
/// numbered in query order).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct KvSched {
    /// The block's units (by path index), the first on the path first.
    pub units: Vec<usize>,
    /// (unit index into `units`, rows, first p).
    pub stores: Vec<(usize, u32, u32)>,
    /// (unit index into `units`, rows).
    pub queries: Vec<(usize, u32)>,
}

/// The stream's order through the path units (each: its classes, its rows from the host, a
/// barrier, its rows out to the host: off the stream) and each block's K / V column (`blocks`: path indices of a stage's attending units).
pub fn schedule(
    units: &[(Vec<usize>, bool, bool, bool)],
    class_rows: &[u32],
    blocks: &[Vec<usize>],
) -> R<(Vec<UnitSched>, Vec<KvSched>)> {
    let mut stream: Vec<Block> = Vec::new();
    let mut out = Vec::new();
    for (i, (classes, host, barrier, keep)) in units.iter().enumerate() {
        let later = |c: usize| units[i + 1..].iter().any(|(cl, _, _, _)| cl.contains(&c));
        let mut segs = Vec::new();
        let mut rows = Vec::new();
        let mut next: Vec<Block> = Vec::new();
        let mut own: Vec<Block> = Vec::new();
        let inn: u32 = stream.iter().map(|b| b.n).sum();
        let mut t = 0;
        if *host {
            for &c in classes {
                let n = class_rows[c];
                segs.push(Seg::Early {
                    t0: t,
                    n,
                    host: true,
                });
                rows.extend((0..n).map(|k| (c, k)));
                t += n;
                if *barrier {
                    own.push(Block { class: c, n });
                } else if !*keep {
                    next.push(Block { class: c, n });
                }
            }
        }
        for b in &stream {
            if classes.contains(&b.class) && !*host {
                segs.push(Seg::Early {
                    t0: t,
                    n: b.n,
                    host: false,
                });
                rows.extend((0..b.n).map(|k| (b.class, k)));
                t += b.n;
                if *barrier {
                    own.push(*b);
                } else if !*keep {
                    next.push(*b);
                }
            } else if later(b.class) {
                segs.push(Seg::Fwd { n: b.n });
                next.push(*b);
            } else {
                segs.push(Seg::Drop { n: b.n });
            }
        }
        if *barrier {
            let mut t0 = 0;
            for b in &own {
                segs.push(Seg::Late { t0, n: b.n });
                t0 += b.n;
            }
            if !*keep {
                next.extend(own);
            }
        }
        // the last unit sends nothing on (a later one drops what it does not run)
        let out_n = if i + 1 == units.len() {
            0
        } else {
            next.iter().map(|b| b.n).sum()
        };
        out.push(UnitSched {
            segs,
            rows,
            out: out_n,
            inn,
        });
        stream = next;
    }
    let mut kv = Vec::new();
    for blk in blocks {
        // p by query order: each unit's own rows in its order, units in path order
        let mut base = BTreeMap::new();
        let mut p = 0;
        let mut queries = Vec::new();
        for (k, &u) in blk.iter().enumerate() {
            base.insert(u, p);
            let n = out[u].rows.len() as u32;
            queries.push((k, n));
            p += n;
        }
        // stores: in the order rows reach the block's first unit
        let first = blk[0];
        let mut stores = Vec::new();
        let mut t_of: BTreeMap<usize, u32> = BTreeMap::new();
        let arriving: Vec<(usize, u32)> = {
            // the first unit's runs: own (early) runs and forwarded runs, in order
            let mut v = Vec::new();
            let mut fi = 0;
            let fwd_classes = fwd_blocks(units, class_rows, first);
            for s in &out[first].segs {
                match s {
                    Seg::Early { n, .. } => v.push((usize::MAX, *n)),
                    Seg::Fwd { n } => {
                        v.push((fwd_classes[fi], *n));
                        fi += 1;
                    }
                    _ => {}
                }
            }
            v
        };
        for (c, n) in arriving {
            let owner = if c == usize::MAX {
                Some(first)
            } else {
                blk.iter().copied().find(|&u| units[u].0.contains(&c))
            };
            let Some(u) = owner else { continue };
            let k = blk.iter().position(|&x| x == u).unwrap_or(0);
            let t = t_of.entry(u).or_insert(0);
            stores.push((k, n, base[&u] + *t));
            *t += n;
        }
        kv.push(KvSched {
            units: blk.clone(),
            stores,
            queries,
        });
    }
    Ok((out, kv))
}

/// The classes of the runs unit `at` forwards, in order.
fn fwd_blocks(
    units: &[(Vec<usize>, bool, bool, bool)],
    class_rows: &[u32],
    at: usize,
) -> Vec<usize> {
    let (s, _) = schedule_blocks(units, class_rows, at);
    s
}

/// The runs arriving at unit `at`: (classes of those it forwards, all arriving).
fn schedule_blocks(
    units: &[(Vec<usize>, bool, bool, bool)],
    class_rows: &[u32],
    at: usize,
) -> (Vec<usize>, Vec<Block>) {
    let mut stream: Vec<Block> = Vec::new();
    for (i, (classes, host, barrier, keep)) in units.iter().enumerate() {
        let later = |c: usize| units[i + 1..].iter().any(|(cl, _, _, _)| cl.contains(&c));
        if i == at {
            let f = stream
                .iter()
                .filter(|b| !(classes.contains(&b.class) && !*host) && later(b.class))
                .map(|b| b.class)
                .collect();
            return (f, stream);
        }
        let mut next = Vec::new();
        let mut own = Vec::new();
        if *host {
            for &c in classes {
                let b = Block {
                    class: c,
                    n: class_rows[c],
                };
                if *barrier {
                    own.push(b);
                } else if !*keep {
                    next.push(b);
                }
            }
        }
        for b in &stream {
            if classes.contains(&b.class) && !*host {
                if *barrier {
                    own.push(*b);
                } else if !*keep {
                    next.push(*b);
                }
            } else if later(b.class) {
                next.push(*b);
            }
        }
        if !*keep {
            next.extend(own);
        }
        stream = next;
    }
    (Vec::new(), stream)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// flux2-mini's path: two entries (from the host), two double blocks a class each, two
    /// single blocks of both, the exit of the image rows. Every unit's rows reach it, and each
    /// K / V column stores what its units send in the order they send it.
    #[test]
    fn the_row_stream_orders_flux_s_units() {
        let units = vec![
            (vec![0], true, false, false),
            (vec![1], true, false, false),
            (vec![0], false, true, false),
            (vec![1], false, true, false),
            (vec![0], false, true, false),
            (vec![1], false, true, false),
            (vec![0, 1], false, true, false),
            (vec![0, 1], false, true, false),
            (vec![1], false, false, true),
        ];
        let (s, kv) = schedule(
            &units,
            &[64, 64],
            &[vec![2, 3], vec![4, 5], vec![6], vec![7]],
        )
        .unwrap();
        assert_eq!(
            s[1].segs,
            [
                Seg::Early {
                    t0: 0,
                    n: 64,
                    host: true
                },
                Seg::Fwd { n: 64 }
            ]
        );
        // d0 txt: the image rows on first, then its own
        assert_eq!(
            s[2].segs,
            [
                Seg::Fwd { n: 64 },
                Seg::Early {
                    t0: 0,
                    n: 64,
                    host: false
                },
                Seg::Late { t0: 0, n: 64 }
            ]
        );
        assert_eq!(kv[0].stores, [(1, 64, 64), (0, 64, 0)]);
        assert_eq!(kv[0].queries, [(0, 64), (1, 64)]);
        assert_eq!(s[6].rows.len(), 128);
        assert_eq!(
            s[8].segs,
            [
                Seg::Drop { n: 64 },
                Seg::Early {
                    t0: 0,
                    n: 64,
                    host: false
                }
            ]
        );
        assert_eq!(s[8].out, 0);
    }
}
