//! Lazy views of the chunk pool.
//!
//! The pool is one `u32 [R, 128]` buffer. A plane is read as a view of the
//! whole pool at its own element and shape, but on a TPU no such view may be
//! built: a `u32 -> bf16` reinterpretation grows a minor axis of 2 that the
//! tiled layout pads to 128 (a copy of 64 pools), and any reshape of the pool
//! to another minor width relayouts the whole pool. So a view here is never
//! emitted. It is a value with a type and no text, and the ops a kernel
//! applies to it are rewritten onto the pool:
//!
//! - `reshape` of a view is a view;
//! - `gather`, `dynamic_slice`, `slice` read whole pool rows of the u32
//!   buffer and unpack only those;
//! - `scatter` (set) and `dynamic_update_slice` pack their updates into pool
//!   rows and scatter them, a block narrower than a row through an
//!   and-mask / or pair (commutative, so two blocks of one row may land in
//!   one scatter);
//! - loops carry the pool, and give their body and results views again.
//!
//! Anything else materializes the view (logged in [`Func::materialized`]):
//! correct, but a copy of the pool.
//!
//! Element order (the "packed rows" order, what a Mosaic kernel sees when it
//! bitcasts a `u32 [r, 128]` block to a narrower element in VMEM): with `k`
//! elements a word, pool row `r` holds elements `r·128k .. (r+1)·128k` of
//! the flat view; element `r·128k + h·128 + l` is part `h` of word `l`.
//! A plane's run must therefore cover whole pool rows (512-byte aligned and
//! sized) so that no row mixes planes of different elements.

use super::{Built, Cmp, Combine, Elem, Func, GatherDims, ScatterDims, Ty, Val, malformed};

/// Lanes of a pool row.
pub const LANES: i64 = 128;

/// A view's pool and element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct View {
    pub pool: Val,
    pub elem: Elem,
}

/// What each carried value is: a view's element and shape, or not a view.
pub(super) type Kinds = Vec<Option<(Elem, Vec<i64>)>>;

/// Elements one u32 word holds.
fn per_word(elem: Elem) -> Option<i64> {
    match elem.bits() {
        8 => Some(4),
        16 => Some(2),
        32 => Some(1),
        _ => None,
    }
}

/// The unsigned element of `bits`.
fn uint(bits: u32) -> Elem {
    match bits {
        8 => Elem::U8,
        16 => Elem::U16,
        _ => Elem::U32,
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a.abs() } else { gcd(b, a % b) }
}

/// The elements of the slice of `dims` taking `sizes` when they lie as one
/// flat run: past the first axis taken at more than 1, every axis is whole.
fn block_len(dims: &[i64], sizes: &[i64]) -> Option<i64> {
    let first = (0..dims.len()).find(|&i| sizes[i] != 1);
    if let Some(f) = first
        && ((f + 1)..dims.len()).any(|i| sizes[i] != dims[i])
    {
        return None;
    }
    Some(sizes.iter().product())
}

fn strides(dims: &[i64]) -> Vec<i64> {
    let mut st = vec![1i64; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        st[i] = st[i + 1] * dims[i + 1];
    }
    st
}

impl Func {
    /// A view of the pool `u32 [R, 128]` as `elem` of `dims` (every element
    /// of the pool, in the packed rows order).
    pub fn pool_view(&mut self, pool: Val, elem: Elem, dims: &[i64]) -> Built<Val> {
        let pool = match self.views.get(&pool.0) {
            Some(v) => v.pool,
            None => pool,
        };
        let pt = self.ty(pool).clone();
        let Some(k) = per_word(elem) else {
            return malformed("pool_view", format!("a {} view", elem.mlir()));
        };
        if pt.elem != Elem::U32 || pt.rank() != 2 || pt.dims[1] != LANES {
            return malformed("pool_view", format!("a view of {pt}"));
        }
        let ty = Ty::new(elem, dims);
        if ty.elements() != pt.dims[0] * LANES * k {
            return malformed("pool_view", format!("{ty} over {pt}"));
        }
        let id = self.next;
        self.next += 1;
        let v = self.fresh(format!("%view{id}"), ty);
        self.views.insert(v.0, View { pool, elem });
        Ok(v)
    }

    /// A view's pool and element (what a Mosaic kernel takes in its place,
    /// bitcasting the reference: `crate::mosaic::Body::ref_bitcast`).
    #[must_use]
    pub fn view_parts(&self, v: Val) -> Option<(Val, Elem)> {
        self.views.get(&v.0).map(|view| (view.pool, view.elem))
    }

    #[must_use]
    pub fn is_view(&self, v: Val) -> bool {
        self.views.contains_key(&v.0)
    }

    /// What was materialized from views (each a copy of the pool).
    #[must_use]
    pub fn materialized(&self) -> &[String] {
        &self.materialized
    }

    fn pool_rows(&self, view: View) -> i64 {
        self.ty(view.pool).dims[0]
    }

    /// The pool a value of a view's type lands as: a view's own pool; any
    /// other value is packed (a copy of the pool, logged).
    pub fn pool_of(&mut self, v: Val, elem: Elem) -> Built<Val> {
        if let Some(view) = self.views.get(&v.0).copied() {
            return Ok(view.pool);
        }
        let Some(k) = per_word(elem) else {
            return malformed("pool_of", format!("a {} plane", elem.mlir()));
        };
        let n = self.ty(v).elements();
        let rows = n / (LANES * k);
        self.materialized.push(format!("pack {}", self.ty(v)));
        self.pack(v, rows, k)
    }

    /// `x` (`k·128·rows` elements in packed rows order) as `u32 [rows, 128]`.
    fn pack(&mut self, x: Val, rows: i64, k: i64) -> Built<Val> {
        let bits = self.elem(x).bits();
        let x = if k == 1 {
            self.reshape(x, &[rows, LANES])?
        } else {
            let x = self.reshape(x, &[rows, k, LANES])?;
            self.transpose(x, &[0, 2, 1])?
        };
        let x = if self.elem(x).is_float() || self.elem(x) != uint(bits) {
            self.bitcast(x, uint(bits))?
        } else {
            x
        };
        if k == 1 {
            return Ok(if self.elem(x) == Elem::U32 {
                x
            } else {
                self.bitcast(x, Elem::U32)?
            });
        }
        self.bitcast(x, Elem::U32)
    }

    /// `u32 [.., n, 128]` rows as `elem [.., n·128k]` in packed rows order.
    fn unpack(&mut self, rows: Val, elem: Elem) -> Built<Val> {
        let k = per_word(elem).unwrap_or(1);
        let dims = self.dims(rows).to_vec();
        let lead = &dims[..dims.len() - 2];
        let n = dims[dims.len() - 2];
        let x = self.bitcast(rows, elem)?;
        let x = if k == 1 {
            x
        } else {
            let r = dims.len() as i64;
            let mut perm: Vec<i64> = (0..r - 1).collect();
            perm.push(r);
            perm.push(r - 1);
            self.transpose(x, &perm)?
        };
        let mut out = lead.to_vec();
        out.push(n * LANES * k);
        self.reshape(x, &out)
    }

    /// The whole view as a value (a copy of the pool, logged).
    fn materialize(&mut self, v: Val, why: &str) -> Built<Val> {
        let view = self.views[&v.0];
        let dims = self.ty(v).dims.clone();
        self.materialized
            .push(format!("{why}: {}", Ty::new(view.elem, &dims)));
        let x = self.unpack(view.pool, view.elem)?;
        self.reshape(x, &dims)
    }

    /// `vs` with every view materialized (for ops that have no rewrite).
    pub(super) fn solid(&mut self, vs: &[Val], why: &str) -> Vec<Val> {
        if !vs.iter().any(|v| self.views.contains_key(&v.0)) {
            return vs.to_vec();
        }
        vs.iter()
            .map(|&v| {
                if self.views.contains_key(&v.0) {
                    self.materialize(v, why).unwrap_or(v)
                } else {
                    v
                }
            })
            .collect()
    }

    /// `[B · n]` pool rows: `n` from each of the `[B]` starts.
    fn rows_of(&mut self, start: Val, batch: i64, n: i64) -> Built<Val> {
        if n == 1 {
            return self.reshape(start, &[batch]);
        }
        let s = self.broadcast(start, &[batch, n], &[0])?;
        let i = self.iota(Elem::I32, &[batch, n], 1);
        let r = self.add(s, i)?;
        self.reshape(r, &[batch * n])
    }

    /// The i32 scalar `x` clamped to `[0, hi]`.
    fn clamp_to(&mut self, x: Val, hi: i64) -> Built<Val> {
        let lo = self.like_i(x, 0);
        let top = self.like_i(x, hi.max(0));
        self.clamp(lo, x, top)
    }

    /// Reads `[B, len]` of a view: block `b` starts at flat element
    /// `Σ starts[j][b] · stride(axes[j])` (each start already in range).
    /// `None` where the block cannot be read by rows.
    fn read_block(
        &mut self,
        view: View,
        dims: &[i64],
        starts: &[(usize, Val)],
        batch: i64,
        len: i64,
    ) -> Built<Option<Val>> {
        let k = per_word(view.elem).unwrap_or(1);
        let e = LANES * k;
        let st = strides(dims);
        let mut g = 0i64;
        for &(axis, _) in starts {
            g = gcd(g, st[axis]);
        }
        let total = dims.iter().product::<i64>();
        if g == 0 {
            g = total;
        }
        // The unit the start is counted in: a divisor of a row or a whole
        // number of rows.
        let unit = if g % e == 0 { e } else { g };
        if e % unit != 0 || total / unit > i64::from(i32::MAX) {
            return Ok(None);
        }
        let rows_total = self.pool_rows(view);
        // q: the start in units.
        let mut q: Option<Val> = None;
        for &(axis, s) in starts {
            let s = self.reshape(s, &[batch])?;
            let m = self.like_i(s, st[axis] / unit);
            let t = self.mul(s, m)?;
            q = Some(match q {
                Some(q) => self.add(q, t)?,
                None => t,
            });
        }
        let q = match q {
            Some(q) => q,
            None => self.const_i(Elem::I32, 0, &[batch]),
        };
        let per_row = e / unit;
        let (row, off) = if per_row == 1 {
            (q, None)
        } else {
            let d = self.like_i(q, per_row);
            let row = self.div(q, d)?;
            let rem = self.rem(q, d)?;
            let u = self.like_i(rem, unit);
            (row, Some(self.mul(rem, u)?))
        };
        // Rows a block spans: aligned blocks exactly, others the most they
        // can cross.
        let aligned = len % e == 0 && off.is_none();
        let sub = e % len == 0 && unit % len == 0;
        let nr = if aligned {
            len / e
        } else if sub {
            1
        } else {
            (e - unit + len + e - 1) / e
        }
        .min(rows_total);
        // Keep the rows read inside the pool (a gather clamps its start,
        // which would shift the block).
        let (row, off) = if aligned || sub {
            (row, off)
        } else {
            let top = self.like_i(row, rows_total - nr);
            let rc = self.min(row, top)?;
            let back = self.sub(row, rc)?;
            let el = self.like_i(back, e);
            let back = self.mul(back, el)?;
            let off = match off {
                Some(o) => self.add(o, back)?,
                None => back,
            };
            (rc, Some(off))
        };
        // One index a pool row: a gather of row windows is a loop on a TPU
        // (carrying the pool), of single rows a native gather.
        let ids = self.rows_of(row, batch, nr)?;
        let got = self.take_rows(view.pool, ids)?;
        let got = self.reshape(got, &[batch, nr, LANES])?;
        let flat = self.unpack(got, view.elem)?;
        let Some(off) = off else {
            return Ok(Some(flat));
        };
        if nr * e == len {
            return Ok(Some(flat));
        }
        // Pick `len` elements at `off` from each batch row.
        let at = self.reshape(off, &[batch, 1])?;
        let picked = self.gather(
            flat,
            at,
            &GatherDims {
                offset_dims: vec![1],
                collapsed_slice_dims: vec![],
                operand_batching_dims: vec![0],
                start_indices_batching_dims: vec![0],
                start_index_map: vec![1],
                index_vector_dim: 1,
            },
            &[1, len],
        )?;
        Ok(Some(self.reshape(picked, &[batch, len])?))
    }

    /// The `[B]` start of operand axis `axis` read from `indices` (component
    /// `j` along `ivd`).
    fn component(&mut self, indices: Val, ivd: i64, j: usize, batch: i64) -> Built<Val> {
        let dims = self.dims(indices).to_vec();
        let x = if (ivd as usize) < dims.len() {
            let mut lo = vec![0; dims.len()];
            let mut hi = dims.clone();
            lo[ivd as usize] = j as i64;
            hi[ivd as usize] = j as i64 + 1;
            self.slice(indices, &lo, &hi, &vec![1; dims.len()])?
        } else {
            indices
        };
        let x = if self.elem(x) == Elem::I32 {
            x
        } else {
            self.convert(x, Elem::I32)
        };
        self.reshape(x, &[batch])
    }

    /// `gather` of a view, by pool rows. `None` where it has no rewrite.
    pub(super) fn view_gather(
        &mut self,
        operand: Val,
        indices: Val,
        dims: &GatherDims,
        sizes: &[i64],
    ) -> Built<Option<Val>> {
        let view = self.views[&operand.0];
        let od = self.ty(operand).dims.clone();
        if !dims.operand_batching_dims.is_empty() || sizes.len() != od.len() {
            return Ok(None);
        }
        let Some(len) = block_len(&od, sizes) else {
            return Ok(None);
        };
        let idims = self.dims(indices).to_vec();
        let mut bdims = idims.clone();
        if (dims.index_vector_dim as usize) < bdims.len() {
            bdims.remove(dims.index_vector_dim as usize);
        }
        let batch: i64 = bdims.iter().product();
        let mut starts = Vec::new();
        for (j, &axis) in dims.start_index_map.iter().enumerate() {
            let s = self.component(indices, dims.index_vector_dim, j, batch)?;
            let a = axis as usize;
            let s = self.clamp_to(s, od[a] - sizes[a])?;
            starts.push((a, s));
        }
        let Some(flat) = self.read_block(view, &od, &starts, batch, len)? else {
            return Ok(None);
        };
        // Canonical [batch..., offsets...], then the requested order.
        let offsets: Vec<i64> = (0..od.len())
            .filter(|&i| !dims.collapsed_slice_dims.contains(&(i as i64)))
            .map(|i| sizes[i])
            .collect();
        let mut canon = bdims.clone();
        canon.extend(&offsets);
        let x = self.reshape(flat, &canon)?;
        let rank = canon.len();
        let nb = bdims.len();
        let mut perm = Vec::with_capacity(rank);
        let (mut bi, mut oi) = (0usize, 0usize);
        for p in 0..rank as i64 {
            if dims.offset_dims.contains(&p) {
                perm.push((nb + oi) as i64);
                oi += 1;
            } else {
                perm.push(bi as i64);
                bi += 1;
            }
        }
        if perm.iter().enumerate().all(|(i, &p)| i as i64 == p) {
            return Ok(Some(x));
        }
        Ok(Some(self.transpose(x, &perm)?))
    }

    /// A block read at scalar starts (`dynamic_slice`, `slice`).
    pub(super) fn view_slice_at(
        &mut self,
        x: Val,
        starts: &[Val],
        sizes: &[i64],
    ) -> Built<Option<Val>> {
        let view = self.views[&x.0];
        let od = self.ty(x).dims.clone();
        let Some(len) = block_len(&od, sizes) else {
            return Ok(None);
        };
        let mut at = Vec::new();
        for (a, &s) in starts.iter().enumerate() {
            if sizes[a] == od[a] {
                continue;
            }
            let s = if self.elem(s) == Elem::I32 {
                s
            } else {
                self.convert(s, Elem::I32)
            };
            let s = self.clamp_to(s, od[a] - sizes[a])?;
            at.push((a, s));
        }
        let Some(flat) = self.read_block(view, &od, &at, 1, len)? else {
            return Ok(None);
        };
        Ok(Some(self.reshape(flat, sizes)?))
    }

    /// Lands `[B, len]` blocks (`ups`) in a view at `starts` (`valid[b]`
    /// false drops block b). `None` where it has no rewrite.
    #[allow(clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        view: View,
        dims: &[i64],
        starts: &[(usize, Val)],
        valid: Option<Val>,
        ups: Val,
        batch: i64,
        len: i64,
        unique: bool,
    ) -> Built<Option<Val>> {
        let k = per_word(view.elem).unwrap_or(1);
        let e = LANES * k;
        let st = strides(dims);
        let mut g = 0i64;
        for &(axis, _) in starts {
            g = gcd(g, st[axis]);
        }
        let total = dims.iter().product::<i64>();
        if g == 0 {
            g = total;
        }
        let rows_total = self.pool_rows(view);
        let whole = g % e == 0 && len % e == 0;
        let sub = e % len == 0 && g % len == 0;
        if !whole && !sub {
            return Ok(None);
        }
        let unit = if whole { e } else { len };
        if total / unit > i64::from(i32::MAX) {
            return Ok(None);
        }
        let mut q: Option<Val> = None;
        for &(axis, s) in starts {
            let s = self.reshape(s, &[batch])?;
            let m = self.like_i(s, st[axis] / unit);
            let t = self.mul(s, m)?;
            q = Some(match q {
                Some(q) => self.add(q, t)?,
                None => t,
            });
        }
        let q = match q {
            Some(q) => q,
            None => self.const_i(Elem::I32, 0, &[batch]),
        };
        let per_row = e / unit;
        let (row, slot) = if per_row == 1 {
            (q, None)
        } else {
            let d = self.like_i(q, per_row);
            (self.div(q, d)?, Some(self.rem(q, d)?))
        };
        // A block narrower than a row lands by and / or, which would merge
        // two blocks landing on the same place (a set lets one win): keep
        // only the last of each place, as a set scatter in order does.
        let valid = if whole || batch == 1 {
            valid
        } else {
            let last = self.last_of_each(q, valid)?;
            Some(match valid {
                Some(v) => self.and(v, last)?,
                None => last,
            })
        };
        let row = match valid {
            Some(ok) => {
                // Each dropped block past the pool at rows of its own: a
                // scatter hinted unique may not see one index twice.
                let away = self.like_i(row, rows_total);
                let nth = self.iota(Elem::I32, &[batch], 0);
                let span = self.like_i(nth, if whole { len / e } else { 1 });
                let nth = self.mul(nth, span)?;
                let away = self.add(away, nth)?;
                self.select(ok, row, away)?
            }
            None => row,
        };
        let bits = view.elem.bits();
        let raw = if self.elem(ups) == uint(bits) {
            ups
        } else {
            self.bitcast(ups, uint(bits))?
        };
        if whole {
            let nr = len / e;
            let packed = {
                let x = self.reshape(raw, &[batch * nr, k, LANES])?;
                let x = if k == 1 {
                    self.reshape(x, &[batch * nr, LANES])?
                } else {
                    self.transpose(x, &[0, 2, 1])?
                };
                self.bitcast(x, Elem::U32)?
            };
            // one index a pool row (see `read_block`)
            let ids = self.rows_of(row, batch, nr)?;
            let pool = self.put_rows_hinted(view.pool, ids, packed, Combine::Set, unique)?;
            return Ok(Some(pool));
        }
        // A block inside one row: the row's bits with the block in place
        // (zero elsewhere) and its mask, then and-not / or.
        let slots = e / len;
        let slot = slot.unwrap_or_else(|| self.const_i(Elem::I32, 0, &[batch]));
        let pos = self.iota(Elem::I32, &[batch, slots, len], 1);
        let want = self.broadcast(slot, &[batch, slots, len], &[0])?;
        let here = self.compare(Cmp::Eq, pos, want)?;
        let spread = self.broadcast(raw, &[batch, slots, len], &[0, 2])?;
        let zero = self.like_i(spread, 0);
        let vals = self.select(here, spread, zero)?;
        let all = if bits >= 32 {
            i64::from(u32::MAX)
        } else {
            (1i64 << bits) - 1
        };
        let ones = self.like_i(spread, all);
        let mask = self.select(here, zero, ones)?;
        let pack_row = |f: &mut Func, x: Val| -> Built<Val> {
            let x = f.reshape(x, &[batch, k, LANES])?;
            let x = if k == 1 {
                f.reshape(x, &[batch, LANES])?
            } else {
                f.transpose(x, &[0, 2, 1])?
            };
            f.bitcast(x, Elem::U32)
        };
        let vals = pack_row(self, vals)?;
        let keep = pack_row(self, mask)?;
        let idx = self.reshape(row, &[batch, 1])?;
        let sd = ScatterDims {
            update_window_dims: vec![1],
            inserted_window_dims: vec![0],
            scatter_dims_to_operand_dims: vec![0],
            index_vector_dim: 1,
            ..ScatterDims::default()
        };
        let pool = self.scatter(view.pool, idx, keep, &sd, Combine::And)?;
        let pool = self.scatter(pool, idx, vals, &sd, Combine::Or)?;
        Ok(Some(pool))
    }

    /// Whether block `b` is the last (highest index) of the valid blocks
    /// with its `key` (`[B]` i32): a stable sort by key, each run's last.
    fn last_of_each(&mut self, key: Val, valid: Option<Val>) -> Built<Val> {
        let b = self.dims(key)[0];
        let nth = self.iota(Elem::I32, &[b], 0);
        // invalid blocks get keys of their own, below every place
        let key = match valid {
            Some(ok) => {
                let neg = self.like_i(nth, -1);
                let own = self.sub(neg, nth)?;
                self.select(ok, key, own)?
            }
            None => key,
        };
        let sorted = self.sort(&[key, nth], 0, true, |f, l, r| {
            f.compare(Cmp::Lt, l[0], r[0])
        })?;
        let (sk, si) = (sorted[0], sorted[1]);
        // last of its run: the next sorted key differs (or there is none)
        let next = self.slice(sk, &[1], &[b], &[1])?;
        let big = self.const_i(Elem::I32, i64::from(i32::MIN), &[1]);
        let next = self.concat(&[next, big], 0)?;
        let last_sorted = self.compare(Cmp::Ne, sk, next)?;
        // back to block order
        let zero = self.const_i(Elem::Pred, 0, &[b]);
        let at = self.reshape(si, &[b, 1])?;
        self.scatter_hinted(
            zero,
            at,
            last_sorted,
            &ScatterDims {
                inserted_window_dims: vec![0],
                scatter_dims_to_operand_dims: vec![0],
                index_vector_dim: 1,
                ..ScatterDims::default()
            },
            Combine::Set,
            true,
        )
    }

    /// `scatter` (set) onto a view. `None` where it has no rewrite.
    pub(super) fn view_scatter(
        &mut self,
        operand: Val,
        indices: Val,
        updates: Val,
        dims: &ScatterDims,
        unique: bool,
    ) -> Built<Option<Val>> {
        let view = self.views[&operand.0];
        let od = self.ty(operand).dims.clone();
        if !dims.input_batching_dims.is_empty() {
            return Ok(None);
        }
        let ud = self.dims(updates).to_vec();
        // Window sizes per operand axis.
        let mut sizes = Vec::with_capacity(od.len());
        let mut w = dims.update_window_dims.iter();
        for i in 0..od.len() {
            if dims.inserted_window_dims.contains(&(i as i64)) {
                sizes.push(1);
            } else {
                let Some(&u) = w.next() else {
                    return Ok(None);
                };
                sizes.push(ud[u as usize]);
            }
        }
        let Some(len) = block_len(&od, &sizes) else {
            return Ok(None);
        };
        let idims = self.dims(indices).to_vec();
        let mut bdims = idims.clone();
        if (dims.index_vector_dim as usize) < bdims.len() {
            bdims.remove(dims.index_vector_dim as usize);
        }
        let batch: i64 = bdims.iter().product();
        // Updates as [batch..., window...] -> [B, len].
        let wdims: Vec<i64> = dims.update_window_dims.clone();
        let mut perm: Vec<i64> = (0..ud.len() as i64)
            .filter(|i| !wdims.contains(i))
            .collect();
        perm.extend(&wdims);
        let ups = if perm.iter().enumerate().all(|(i, &p)| i as i64 == p) {
            updates
        } else {
            self.transpose(updates, &perm)?
        };
        let ups = self.reshape(ups, &[batch, len])?;
        let mut starts = Vec::new();
        let mut valid: Option<Val> = None;
        for (j, &axis) in dims.scatter_dims_to_operand_dims.iter().enumerate() {
            let s = self.component(indices, dims.index_vector_dim, j, batch)?;
            let a = axis as usize;
            let lo = self.like_i(s, 0);
            let hi = self.like_i(s, od[a] - sizes[a]);
            let ge = self.compare(Cmp::Ge, s, lo)?;
            let le = self.compare(Cmp::Le, s, hi)?;
            let ok = self.and(ge, le)?;
            valid = Some(match valid {
                Some(v) => self.and(v, ok)?,
                None => ok,
            });
            let s = self.clamp_to(s, od[a] - sizes[a])?;
            starts.push((a, s));
        }
        let Some(pool) = self.write_block(view, &od, &starts, valid, ups, batch, len, unique)?
        else {
            return Ok(None);
        };
        let dims = od;
        Ok(Some(self.pool_view(pool, view.elem, &dims)?))
    }

    /// `dynamic_update_slice` onto a view. `None` where it has no rewrite.
    pub(super) fn view_update_at(
        &mut self,
        x: Val,
        update: Val,
        starts: &[Val],
    ) -> Built<Option<Val>> {
        let view = self.views[&x.0];
        let od = self.ty(x).dims.clone();
        let sizes = self.dims(update).to_vec();
        let Some(len) = block_len(&od, &sizes) else {
            return Ok(None);
        };
        let mut at = Vec::new();
        for (a, &s) in starts.iter().enumerate() {
            if sizes[a] == od[a] {
                continue;
            }
            let s = if self.elem(s) == Elem::I32 {
                s
            } else {
                self.convert(s, Elem::I32)
            };
            let s = self.clamp_to(s, od[a] - sizes[a])?;
            at.push((a, s));
        }
        let ups = self.reshape(update, &[1, len])?;
        if let Some(pool) = self.write_block(view, &od, &at, None, ups, 1, len, true)? {
            return Ok(Some(self.pool_view(pool, view.elem, &od)?));
        }
        // One block across rows at any start: read the rows it covers,
        // splice it in, write them back (one block, so no row is shared).
        let k = per_word(view.elem).unwrap_or(1);
        let e = LANES * k;
        let st = strides(&od);
        let mut unit = 0i64;
        for &(axis, _) in &at {
            unit = gcd(unit, st[axis]);
        }
        let total = od.iter().product::<i64>();
        if unit == 0 {
            unit = total;
        }
        if e % unit != 0 && unit % e != 0 {
            return Ok(None);
        }
        let unit = unit.min(e);
        if total / unit > i64::from(i32::MAX) {
            return Ok(None);
        }
        let rows_total = self.pool_rows(view);
        let mut q = self.const_i(Elem::I32, 0, &[]);
        for &(axis, s) in &at {
            let m = self.like_i(s, st[axis] / unit);
            let t = self.mul(s, m)?;
            q = self.add(q, t)?;
        }
        let per_row = self.like_i(q, e / unit);
        let row = self.div(q, per_row)?;
        let rem = self.rem(q, per_row)?;
        let u = self.like_i(rem, unit);
        let off = self.mul(rem, u)?;
        let nr = ((e - unit + len + e - 1) / e).min(rows_total);
        let top = self.like_i(row, rows_total - nr);
        let rc = self.min(row, top)?;
        let back = self.sub(row, rc)?;
        let el = self.like_i(back, e);
        let back = self.mul(back, el)?;
        let off = self.add(off, back)?;
        let zero = self.const_i(Elem::I32, 0, &[]);
        let rows = self.dynamic_slice(view.pool, &[rc, zero], &[nr, LANES])?;
        let flat = self.unpack(rows, view.elem)?;
        let flat = self.reshape(flat, &[nr * e])?;
        let up = self.reshape(update, &[len])?;
        let flat = self.dynamic_update_slice(flat, up, &[off])?;
        let packed = self.pack(flat, nr, k)?;
        let pool = self.dynamic_update_slice(view.pool, packed, &[rc, zero])?;
        Ok(Some(self.pool_view(pool, view.elem, &od)?))
    }

    /// Loop-carried views: the pools they ride on, and how to make them
    /// views again.
    pub(super) fn unview_carried(
        &mut self,
        inits: &[Val],
    ) -> (Vec<Val>, Kinds) {
        let mut out = Vec::with_capacity(inits.len());
        let mut kinds = Vec::with_capacity(inits.len());
        for &v in inits {
            if let Some(view) = self.views.get(&v.0).copied() {
                out.push(view.pool);
                kinds.push(Some((view.elem, self.ty(v).dims.clone())));
            } else {
                out.push(v);
                kinds.push(None);
            }
        }
        (out, kinds)
    }

    pub(super) fn review(
        &mut self,
        vs: &[Val],
        kinds: &[Option<(Elem, Vec<i64>)>],
    ) -> Built<Vec<Val>> {
        let mut out = Vec::with_capacity(vs.len());
        for (&v, kind) in vs.iter().zip(kinds) {
            out.push(match kind {
                Some((elem, dims)) => self.pool_view(v, *elem, dims)?,
                None => v,
            });
        }
        Ok(out)
    }

    pub(super) fn pools_of(
        &mut self,
        vs: &[Val],
        kinds: &[Option<(Elem, Vec<i64>)>],
    ) -> Built<Vec<Val>> {
        let mut out = Vec::with_capacity(vs.len());
        for (&v, kind) in vs.iter().zip(kinds) {
            out.push(match kind {
                Some((elem, _)) => self.pool_of(v, *elem)?,
                None => v,
            });
        }
        Ok(out)
    }
}
