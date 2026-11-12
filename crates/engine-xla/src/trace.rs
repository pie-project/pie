//! Roots, handles and the fire tracer.
//!
//! A *root* is one array the fire program may name: a weight plane, a pool
//! plane, a per-fire input, or an activation the plan computes. A *handle*
//! (the `buf` of a `crate::handle::Tensor`) is a row window of a root. The
//! engine resolves plan values to handles exactly as engine-wgpu does; the
//! tracer then turns every kernel's reads and writes of handles into SSA
//! values of one StableHLO function:
//!
//! - the first read of a root that lives on the device (weights, pools, inputs)
//!   makes it a parameter;
//! - an activation root starts as zeros; a write covering it whole replaces
//!   its value, a window write is a `dynamic_update_slice` into it;
//! - a pool root that was written is a result, aliased to its parameter so
//!   the executable updates it in place.

use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::emit::{Cx, Emit, Env};
use crate::handle::Tensor;
use dtype::Dtype;
use kernels_xla::hlo::{Elem, Func, Ty, Val};

/// Where a root's value comes from when the program first reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    /// A weight plane, landed at load; `plane` numbers the planes of one
    /// param (0 codes/dense, 1 scales, 2 biases). A `transposed` plane
    /// landed as `[width, rows]` (a gemm weight, its output axis minor: the
    /// MXU streams it at full bandwidth) and reads as `[rows, width]`.
    Weight {
        param: u32,
        plane: u8,
        transposed: bool,
    },
    /// A pool plane, persistent across fires and updated in place.
    Pool { row: u32, plane: u8 },
    /// A per-fire input the shell uploads; `input` indexes the fire's inputs.
    Input { input: u32 },
    /// A per-fire i32 input packed with the others into one upload, starting
    /// at element `offset` of the pack.
    Packed { offset: u32 },
    /// The pack itself: every packed input, one i32 vector.
    Pack,
    /// An activation computed by this fire; zeros until written.
    Temp,
}

impl Source {
    #[must_use]
    pub const fn on_device(self) -> bool {
        !matches!(self, Source::Temp | Source::Packed { .. })
    }

    #[must_use]
    pub const fn persistent(self) -> bool {
        matches!(self, Source::Pool { .. })
    }
}

/// One array: `rows × width` elements of `dtype` (a packed dtype is stored
/// as the raw bytes of its rows, see [`storage`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Root {
    pub source: Source,
    pub dtype: Dtype,
    pub rows: u32,
    pub width: u32,
}

/// The storage type of a root: plain dtypes as themselves; a quantized
/// bank's codes one per element in their native narrow type
/// (`crate::pack::storage`); other packed formats (K-quant blocks) as
/// `u8 [rows, bytes per row]`.
#[must_use]
pub fn storage(dtype: Dtype, rows: u32, width: u32) -> Option<Ty> {
    let (r, w) = (i64::from(rows), i64::from(width));
    let plain = |elem| Some(Ty::new(elem, &[r, w]));
    match dtype {
        Dtype::F32 => plain(Elem::F32),
        Dtype::F16 => plain(Elem::F16),
        Dtype::Bf16 => plain(Elem::Bf16),
        Dtype::E4m3 => plain(Elem::F8E4m3fn),
        Dtype::E5m2 => plain(Elem::F8E5m2),
        Dtype::E8m0 => plain(Elem::F8E8m0fnu),
        Dtype::I64 => plain(Elem::I64),
        Dtype::I32 => plain(Elem::I32),
        Dtype::I16 => plain(Elem::I16),
        Dtype::I8 => plain(Elem::I8),
        Dtype::U64 => plain(Elem::U64),
        Dtype::U32 => plain(Elem::U32),
        Dtype::U16 => plain(Elem::U16),
        Dtype::U8 | Dtype::Bool => plain(Elem::U8),
        packed if kernels_xla::pack::code_elem(packed).is_some() => {
            crate::pack::storage(packed, rows, width)
        }
        packed => {
            let bytes = packed_row_bytes(packed, u64::from(width))?;
            Some(Ty::new(Elem::U8, &[r, i64::try_from(bytes).ok()?]))
        }
    }
}

/// Bytes one row of `width` logical elements of a packed format occupies,
/// as the checkpoint lands it (see engine-wgpu `plane_bytes`).
#[must_use]
pub fn packed_row_bytes(dtype: Dtype, width: u64) -> Option<u64> {
    Some(match dtype {
        Dtype::Mxfp4 | Dtype::U8g64 => width,
        Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => width.div_ceil(2),
        Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => width.div_ceil(4),
        // A K-quant param is declared with its row's bytes as its width.
        Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k => width,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug)]
struct Binding {
    root: u32,
    row_offset: u32,
    /// A pool plane's row order (`kernels_xla::hlo::PoolOrder`).
    order: Option<kernels_xla::hlo::PoolOrder>,
}

/// The root and handle tables. Roots and handles minted before [`seal`](Self::seal)
/// live for the load; the rest are the fire's and go at
/// [`rewind`](Self::rewind).
#[derive(Debug, Default)]
pub struct Handles {
    roots: RefCell<Vec<Root>>,
    bindings: RefCell<Vec<Binding>>,
    sealed: std::cell::Cell<(usize, usize)>,
}

impl Handles {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a root and returns a handle over all of it.
    pub fn root(&self, root: Root) -> Tensor {
        let mut roots = self.roots.borrow_mut();
        roots.push(root);
        let at = roots.len() as u32 - 1;
        let mut bindings = self.bindings.borrow_mut();
        bindings.push(Binding {
            root: at,
            row_offset: 0,
            order: None,
        });
        Tensor::new(bindings.len() as u32 - 1, root.rows, root.width, root.dtype)
    }

    /// A handle over the whole of `t`'s root read as `width`-wide rows of
    /// `dtype`: a plane of the pool, as many rows as the root's bits hold.
    pub fn view(&self, t: Tensor, dtype: Dtype, width: u32) -> Result<Tensor, kernels_xla::Error> {
        self.view_ordered(t, dtype, width, None)
    }

    /// [`Handles::view`] whose reads and writes lay the pool's rows in
    /// `order`.
    pub fn view_ordered(
        &self,
        t: Tensor,
        dtype: Dtype,
        width: u32,
        order: Option<kernels_xla::hlo::PoolOrder>,
    ) -> Result<Tensor, kernels_xla::Error> {
        let b = self.bindings.borrow()[t.buf as usize];
        let root = self.root_of(b.root);
        let probe = Tensor::new(t.buf, 0, width, dtype);
        let rows = view_rows(root, probe)?;
        let mut bindings = self.bindings.borrow_mut();
        bindings.push(Binding {
            root: b.root,
            row_offset: 0,
            order,
        });
        Ok(Tensor::new(bindings.len() as u32 - 1, rows, width, dtype))
    }

    /// A handle over rows `[skip, skip + keep)` of `t`'s root window.
    #[must_use]
    pub fn cut(&self, t: Tensor, skip: u32, keep: u32) -> Tensor {
        if skip == 0 && keep >= t.rows {
            return t;
        }
        let b = self.bindings.borrow()[t.buf as usize];
        let rows = keep.min(t.rows.saturating_sub(skip));
        let mut bindings = self.bindings.borrow_mut();
        bindings.push(Binding {
            root: b.root,
            row_offset: b.row_offset + skip,
            order: b.order,
        });
        Tensor::new(bindings.len() as u32 - 1, rows, t.width, t.dtype)
    }

    /// The root under a handle, and where in it the handle starts.
    #[must_use]
    pub fn locate(&self, buf: u32) -> Option<(u32, u32)> {
        let b = *self.bindings.borrow().get(buf as usize)?;
        Some((b.root, b.row_offset))
    }

    /// The row order a handle reads and writes its pool plane in.
    #[must_use]
    pub fn order_of(&self, buf: u32) -> Option<kernels_xla::hlo::PoolOrder> {
        self.bindings
            .borrow()
            .get(buf as usize)
            .and_then(|b| b.order)
    }

    #[must_use]
    pub fn root_of(&self, root: u32) -> Root {
        self.roots.borrow()[root as usize]
    }

    #[must_use]
    pub fn roots(&self) -> usize {
        self.roots.borrow().len()
    }

    /// Everything minted so far lives for the load.
    pub fn seal(&self) {
        self.sealed
            .set((self.roots.borrow().len(), self.bindings.borrow().len()));
    }

    /// Drops the fire's roots and handles.
    pub fn rewind(&self) {
        let (roots, bindings) = self.sealed.get();
        self.roots.borrow_mut().truncate(roots);
        self.bindings.borrow_mut().truncate(bindings);
    }
}

/// What a traced fire needs bound to run: its parameters (roots, in order)
/// and results (roots written, then extra outputs), and which parameter each
/// pool result aliases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<Source>,
    /// For each result: the pool it updates (and the parameter it aliases),
    /// or `None` for an extra output (readouts).
    pub results: Vec<Option<(Source, usize)>>,
}

/// The tracer: one StableHLO function under construction for one fire.
pub struct Tracer<'h> {
    handles: &'h Handles,
    state: RefCell<State>,
    tuning: kernels_xla::tuning::Tuning,
}

/// Where the pack's own parameter is kept among the traced roots.
const PACK_ROOT: u32 = u32::MAX;

struct State {
    pack_len: u32,
    func: Func,
    current: BTreeMap<u32, Val>,
    params: Vec<(u32, Source)>,
    written: BTreeMap<u32, ()>,
    /// A root's per-head sums of squares, landed with its current value.
    sums: BTreeMap<u32, Val>,
    /// The pool value each pool handle was last read from or written as.
    view_reads: BTreeMap<u32, Val>,
    /// Each pool value read at another element and width, once: two planes
    /// of one view read the same value, so a kernel that scatters into both
    /// chains the second onto the first.
    views: Vec<(ViewKey, Val)>,
}

impl<'h> Tracer<'h> {
    #[must_use]
    pub fn new(handles: &'h Handles) -> Self {
        Self {
            tuning: kernels_xla::tuning::Tuning::DEFAULT,
            handles,
            state: RefCell::new(State {
                pack_len: 0,
                func: Func::new("main"),
                current: BTreeMap::new(),
                params: Vec::new(),
                written: BTreeMap::new(),
                sums: BTreeMap::new(),
                view_reads: BTreeMap::new(),
                views: Vec::new(),
            }),
        }
    }

    #[must_use]
    pub fn with_tuning(mut self, tuning: kernels_xla::tuning::Tuning) -> Self {
        self.tuning = tuning;
        self.state
            .borrow_mut()
            .func
            .set_rope_flat_rows(tuning.rope_flat_rows);
        self
    }

    /// Declares how long the fire's input pack is (see `Source::Packed`).
    #[must_use]
    pub fn with_pack(self, len: u32) -> Self {
        self.state.borrow_mut().pack_len = len;
        self
    }

    /// The current whole value of `t`'s root, materialized as a parameter or
    /// zeros if nothing has named it yet.
    pub fn value(&self, t: Tensor) -> Result<Val, kernels_xla::Error> {
        let mut state = self.state.borrow_mut();
        let State {
            func,
            current,
            params,
            pack_len,
            ..
        } = &mut *state;
        let (root, _) = self.locate(t)?;
        root_value(self.handles, func, current, params, *pack_len, root)
    }

    /// Runs `f` against the function under construction.
    pub fn with<R>(&self, f: impl FnOnce(&mut Func) -> R) -> R {
        f(&mut self.state.borrow_mut().func)
    }

    fn locate(&self, t: Tensor) -> Result<(u32, u32), kernels_xla::Error> {
        self.handles
            .locate(t.buf)
            .ok_or_else(|| kernels_xla::Error::Backend {
                op: "trace",
                detail: format!("handle {} was never minted", t.buf),
            })
    }

    /// Finishes the function: pool roots written become aliased results,
    /// then `extra` values. Returns the module text and its signature.
    pub fn finish(self, extra: &[Val]) -> (String, Signature) {
        let State {
            func,
            current,
            params,
            written,
            ..
        } = self.state.into_inner();
        let mut results: Vec<Val> = Vec::new();
        let mut sig_results = Vec::new();
        let mut aliases: Vec<(usize, u32)> = Vec::new();
        for (&root, ()) in &written {
            let source = self.handles.root_of(root).source;
            if !source.persistent() {
                continue;
            }
            let param = params
                .iter()
                .position(|(r, _)| *r == root)
                .expect("a pool root is read before it is written; every pool op scatters");
            aliases.push((param, results.len() as u32));
            sig_results.push(Some((source, param)));
            results.push(current[&root]);
        }
        for &v in extra {
            results.push(v);
            sig_results.push(None);
        }
        let mut func = func;
        for (param, result) in aliases {
            func.alias(param, result);
        }
        let text = func.module("fire", &results);
        if !func.materialized().is_empty() {
            // A view of the pool with no row rewrite: correct, but a copy of
            // the pool each fire. `PIE_XLA_VIEW_STRICT=1` refuses it (tests).
            let what = func.materialized().join("; ");
            assert!(
                std::env::var("PIE_XLA_VIEW_STRICT").as_deref() != Ok("1"),
                "a fire materializes pool views: {what}"
            );
            tracing::warn!("a fire materializes pool views (a pool copy each): {what}");
        }
        (
            text,
            Signature {
                params: params.into_iter().map(|(_, s)| s).collect(),
                results: sig_results,
            },
        )
    }
}

fn root_value(
    handles: &Handles,
    func: &mut Func,
    current: &mut BTreeMap<u32, Val>,
    params: &mut Vec<(u32, Source)>,
    pack_len: u32,
    root: u32,
) -> Result<Val, kernels_xla::Error> {
    if let Some(&v) = current.get(&root) {
        return Ok(v);
    }
    let r = handles.root_of(root);
    let ty = storage(r.dtype, r.rows, r.width).ok_or_else(|| kernels_xla::Error::Backend {
        op: "trace",
        detail: format!("root {root} is {:?}, which has no storage form", r.dtype),
    })?;
    if let Source::Packed { offset } = r.source {
        let pack = match current.get(&PACK_ROOT) {
            Some(&v) => v,
            None => {
                params.push((PACK_ROOT, Source::Pack));
                let v = func.param(Ty::new(Elem::I32, &[i64::from(pack_len)]), None);
                current.insert(PACK_ROOT, v);
                v
            }
        };
        let n = ty.elements();
        let flat = func.slice_axis(pack, 0, i64::from(offset), i64::from(offset) + n)?;
        let v = func.reshape(flat, &ty.dims)?;
        current.insert(root, v);
        return Ok(v);
    }
    let v = if let Source::Weight {
        transposed: true, ..
    } = r.source
    {
        params.push((root, r.source));
        // A sub-byte codes plane lands with its rows padded to whole
        // 128-lane tiles (`crate::weights::gemm_only`).
        let rows = if ty.elem.bits() < 8 {
            let tile = crate::weights::CODES_TILE as i64;
            (ty.dims[0] + tile - 1) / tile * tile
        } else {
            ty.dims[0]
        };
        let landed = Ty::new(ty.elem, &[ty.dims[1], rows]);
        let p = func.param(landed, None);
        let t = func.transpose(p, &[1, 0])?;
        if rows == ty.dims[0] {
            t
        } else {
            func.slice_axis(t, 0, 0, ty.dims[0])?
        }
    } else if r.source.on_device() {
        params.push((root, r.source));
        let p = func.param(ty, None);
        if r.source.persistent() {
            func.pin_row_major(p);
        }
        p
    } else {
        func.const_i(ty.elem, 0, &ty.dims)
    };
    current.insert(root, v);
    Ok(v)
}

/// The tracer's `Env`: the handle tables plus the state's maps, split off the
/// function so a `Cx` can hold both.
struct Lens<'a> {
    handles: &'a Handles,
    pack_len: u32,
    current: &'a mut BTreeMap<u32, Val>,
    params: &'a mut Vec<(u32, Source)>,
    written: &'a mut BTreeMap<u32, ()>,
    sums: &'a mut BTreeMap<u32, Val>,
    view_reads: &'a mut BTreeMap<u32, Val>,
    views: &'a mut Vec<(ViewKey, Val)>,
}

impl Lens<'_> {
    /// The root `t` is the whole of.
    fn whole(&self, t: Tensor) -> Option<u32> {
        let (root, offset) = self.handles.locate(t.buf)?;
        (offset == 0 && t.rows == self.handles.root_of(root).rows).then_some(root)
    }
}

impl Env for Lens<'_> {
    fn read(&mut self, f: &mut Func, t: Tensor) -> Result<Val, kernels_xla::Error> {
        let (root, offset) =
            self.handles
                .locate(t.buf)
                .ok_or_else(|| kernels_xla::Error::Backend {
                    op: "trace",
                    detail: format!("handle {} was never minted", t.buf),
                })?;
        let r = self.handles.root_of(root);
        let whole = root_value(
            self.handles,
            f,
            self.current,
            self.params,
            self.pack_len,
            root,
        )?;
        if r.source.persistent() {
            self.view_reads.insert(t.buf, whole);
        }
        let rows = view_rows(r, t)?;
        let order = self.handles.order_of(t.buf);
        let key = (whole, t.dtype, t.width, order);
        let whole = match self.views.iter().find(|(held, _)| *held == key) {
            Some(&(_, v)) if r.source.persistent() => v,
            _ => {
                let v = reinterpret(f, whole, r, t, order)?;
                if r.source.persistent() && (t.width != r.width || t.dtype != r.dtype) {
                    self.views.push((key, v));
                }
                v
            }
        };
        if offset == 0 && t.rows == rows {
            return Ok(whole);
        }
        Ok(f.slice_axis(whole, 0, i64::from(offset), i64::from(offset + t.rows))?)
    }

    fn write(&mut self, f: &mut Func, t: Tensor, v: Val) -> Result<(), kernels_xla::Error> {
        let (root, offset) =
            self.handles
                .locate(t.buf)
                .ok_or_else(|| kernels_xla::Error::Backend {
                    op: "trace",
                    detail: format!("handle {} was never minted", t.buf),
                })?;
        let r = self.handles.root_of(root);
        if t.width != r.width || t.dtype != r.dtype {
            // A view of a pool lands whole, as the pool's own bits.
            let rows = view_rows(r, t)?;
            if !r.source.persistent() || offset != 0 || t.rows != rows {
                return Err(kernels_xla::Error::Backend {
                    op: "trace",
                    detail: format!(
                        "a {}-wide {:?} write of {} rows lands on a {}-wide {:?} root",
                        t.width, t.dtype, t.rows, r.width, r.dtype
                    ),
                });
            }
            if is_pool(r)
                && let Some((pool, _)) = f.view_parts(v)
                && self.current.get(&root) == Some(&pool)
            {
                // The pool as it stands: nothing lands (a lowering that
                // wrote several planes of the pool in turn lands each on the
                // pool the last write left).
                self.view_reads.insert(t.buf, pool);
                return Ok(());
            }
            self.stale(root, t)?;
            let next = unview(f, v, r, self.handles.order_of(t.buf))?;
            self.current.insert(root, next);
            self.sums.remove(&root);
            self.view_reads.insert(t.buf, next);
            self.written.insert(root, ());
            return Ok(());
        }
        if r.source.persistent() {
            self.stale(root, t)?;
        }
        let next = if offset == 0 && t.rows == r.rows {
            v
        } else {
            let whole = root_value(
                self.handles,
                f,
                self.current,
                self.params,
                self.pack_len,
                root,
            )?;
            let at = f.const_i(Elem::I32, i64::from(offset), &[]);
            let zero = f.const_i(Elem::I32, 0, &[]);
            f.dynamic_update_slice(whole, v, &[at, zero])?
        };
        self.current.insert(root, next);
        if r.source.persistent() {
            self.view_reads.insert(t.buf, next);
        }
        self.written.insert(root, ());
        self.sums.remove(&root);
        Ok(())
    }

    fn sums(&mut self, t: Tensor) -> Option<Val> {
        self.sums.get(&self.whole(t)?).copied()
    }

    fn keep_sums(&mut self, t: Tensor, v: Val) {
        if let Some(root) = self.whole(t) {
            self.sums.insert(root, v);
        }
    }
}

impl Lens<'_> {
    /// Refuses a write of a pool handle built on the pool as it stood
    /// before another handle of it was written since: one pool backs every
    /// plane, and a whole-plane result read before the other plane's write
    /// would land without it. A lowering lands one plane of a pool an emit,
    /// reading it after the last write.
    fn stale(&self, root: u32, t: Tensor) -> Result<(), kernels_xla::Error> {
        match (self.view_reads.get(&t.buf), self.current.get(&root)) {
            (Some(read), Some(now)) if read != now => Err(kernels_xla::Error::Backend {
                op: "trace",
                detail: format!(
                    "pool handle {} lands a value built on the pool as it stood before \
                     another of its planes was written",
                    t.buf
                ),
            }),
            _ => Ok(()),
        }
    }
}

/// The rows `t` views its root as: the root's bits at `t`'s width and
/// element (the root's own rows when they are the same).
fn view_rows(root: Root, t: Tensor) -> Result<u32, kernels_xla::Error> {
    if t.width == root.width && t.dtype == root.dtype {
        return Ok(root.rows);
    }
    let bits = |dtype: Dtype, rows: u32, width: u32| {
        storage(dtype, rows, width).map(|ty| ty.elements() * i64::from(ty.elem.bits()))
    };
    let all = bits(root.dtype, root.rows, root.width);
    let row = bits(t.dtype, 1, t.width);
    match (all, row) {
        (Some(all), Some(row)) if row > 0 && all % row == 0 => {
            u32::try_from(all / row).map_err(|_| kernels_xla::Error::Backend {
                op: "trace",
                detail: format!("a view of {} rows", all / row),
            })
        }
        _ => Err(kernels_xla::Error::Backend {
            op: "trace",
            detail: format!(
                "a {}-wide {:?} view of a {}x{} {:?} root",
                t.width, t.dtype, root.rows, root.width, root.dtype
            ),
        }),
    }
}

/// The chunk pool: the one `u32 [R, 128]` root every plane views.
fn is_pool(root: Root) -> bool {
    root.source.persistent()
        && root.dtype == Dtype::U32
        && root.width == kernels_xla::hlo::POOL_LANES as u32
}

/// A view's whole value back as its root's: the inverse of [`reinterpret`].
fn unview(
    f: &mut Func,
    v: Val,
    root: Root,
    order: Option<kernels_xla::hlo::PoolOrder>,
) -> Result<Val, kernels_xla::Error> {
    if is_pool(root) {
        return Ok(f.pool_of_ordered(v, f.elem(v), order)?);
    }
    let want =
        storage(root.dtype, root.rows, root.width).ok_or_else(|| kernels_xla::Error::Backend {
            op: "trace",
            detail: format!("a {:?} root, which has no storage form", root.dtype),
        })?;
    let bytes = f.bitcast(v, Elem::U8)?;
    let per = i64::from(want.elem.bits() / 8).max(1);
    let n = f.ty(bytes).elements();
    if per == 1 {
        return Ok(f.reshape(bytes, &want.dims)?);
    }
    let r = f.reshape(bytes, &[n / per, per])?;
    let r = f.bitcast(r, want.elem)?;
    Ok(f.reshape(r, &want.dims)?)
}

/// A handle may view its root under another width or element: a packed
/// weight read as raw bytes, a pool plane read out of the one pool.
/// A cached view: the root value, the element and width it is read at,
/// and its row order.
type ViewKey = (Val, Dtype, u32, Option<kernels_xla::hlo::PoolOrder>);

fn reinterpret(
    f: &mut Func,
    v: Val,
    root: Root,
    t: Tensor,
    order: Option<kernels_xla::hlo::PoolOrder>,
) -> Result<Val, kernels_xla::Error> {
    if t.width == root.width && t.dtype == root.dtype {
        return Ok(v);
    }
    let want = storage(t.dtype, view_rows(root, t)?, t.width);
    if is_pool(root)
        && let Some(ty) = &want
    {
        // Never the whole pool at another element: a lazy view whose reads
        // and writes go by pool rows (`kernels_xla::hlo` view).
        return Ok(f.pool_view_ordered(v, ty.elem, &ty.dims, order)?);
    }
    match want {
        Some(ty)
            if ty.elements() * i64::from(ty.elem.bits())
                == f.ty(v).elements() * i64::from(f.ty(v).elem.bits()) =>
        {
            let bytes = f.bitcast(v, Elem::U8)?;
            let flat = f.reshape(bytes, &[f.ty(bytes).elements()])?;
            let per = i64::from(ty.elem.bits() / 8).max(1);
            let r = f.reshape(flat, &[ty.dims[0], ty.dims[1], per])?;
            let r = if per == 1 {
                f.reshape(r, &ty.dims)?
            } else {
                f.bitcast(r, ty.elem)?
            };
            Ok(r)
        }
        _ => Err(kernels_xla::Error::Backend {
            op: "trace",
            detail: format!(
                "a {}-wide {:?} view of a {}-wide {:?} root",
                t.width, t.dtype, root.width, root.dtype
            ),
        }),
    }
}

impl Emit for Tracer<'_> {
    fn scope(&self, name: &str) {
        self.state.borrow_mut().func.set_scope(Some(name));
    }

    fn emit(
        &self,
        body: &mut dyn FnMut(&mut Cx<'_>) -> Result<(), kernels_xla::Error>,
    ) -> Result<(), kernels_xla::Error> {
        let mut state = self.state.borrow_mut();
        let State {
            func,
            current,
            params,
            written,
            pack_len,
            sums,
            view_reads,
            views,
        } = &mut *state;
        let mut lens = Lens {
            handles: self.handles,
            pack_len: *pack_len,
            current,
            params,
            written,
            sums,
            view_reads,
            views,
        };
        let mut cx = Cx::new(func, &mut lens).with_tuning(self.tuning);
        body(&mut cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(handles: &Handles) -> Tensor {
        handles.root(Root {
            source: Source::Pool { row: 0, plane: 0 },
            dtype: Dtype::U32,
            rows: 64,
            width: 128,
        })
    }

    #[test]
    fn a_plane_reads_the_pool_as_its_own_rows_and_lands_back_whole() {
        let handles = Handles::new();
        let root = pool(&handles);
        let keys = handles.view(root, Dtype::Bf16, 256).expect("a view");
        assert_eq!(keys.rows, 64 * 128 * 4 / (256 * 2));
        let tracer = Tracer::new(&handles);
        tracer
            .emit(&mut |cx| {
                let v = cx.read(keys)?;
                assert_eq!(cx.dims(v), &[64, 256]);
                // a row changed (an unchanged view lands nothing)
                let f = cx.func();
                let row = f.slice(v, &[1, 0], &[2, 256], &[1, 1])?;
                let zero = f.const_i(Elem::I32, 0, &[]);
                let v = f.dynamic_update_slice(v, row, &[zero, zero])?;
                cx.write(keys, v)
            })
            .expect("a whole view lands");
        let (text, sig) = tracer.finish(&[]);
        assert_eq!(sig.results.len(), 1, "the pool is the one result");
        // A lazy view: the pool is never reinterpreted whole.
        assert!(
            !text.contains("tensor<64x128x2xbf16>") && !text.contains("tensor<64x256xbf16>"),
            "{text}"
        );
    }

    #[test]
    fn a_plane_landed_off_a_pool_another_plane_replaced_is_refused() {
        let handles = Handles::new();
        let root = pool(&handles);
        let keys = handles.view(root, Dtype::Bf16, 256).expect("a view");
        let values = handles.view(root, Dtype::Bf16, 256).expect("a view");
        let tracer = Tracer::new(&handles);
        let refused = tracer.emit(&mut |cx| {
            let k = cx.read(keys)?;
            let v = cx.read(values)?;
            assert_eq!(k, v, "one view of one pool value is read once");
            // keys change a row (a write of an unchanged view lands nothing)
            let f = cx.func();
            let row = f.slice(k, &[1, 0], &[2, 256], &[1, 1])?;
            let zero = f.const_i(Elem::I32, 0, &[]);
            let k = f.dynamic_update_slice(k, row, &[zero, zero])?;
            cx.write(keys, k)?;
            cx.write(values, v)
        });
        assert!(
            refused.is_err(),
            "the second plane's stale value lands nowhere"
        );
        // Read after the first write, it lands.
        let tracer = Tracer::new(&handles);
        tracer
            .emit(&mut |cx| {
                let k = cx.read(keys)?;
                cx.write(keys, k)?;
                let v = cx.read(values)?;
                cx.write(values, v)
            })
            .expect("a plane read after the other's write lands");
    }
}
