//! Roots, handles and the fire tracer.
//!
//! A *root* is one array the fire program may name: a weight plane, a pool
//! plane, a per-fire input, or an activation the plan computes. A *handle*
//! (the `buf` of a `kernels_xla::Tensor`) is a row window of a root. The
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

use dtype::Dtype;
use kernels_xla::hlo::{Elem, Func, Ty, Val};
use kernels_xla::{Cx, Emit, Env, Tensor};

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
/// (`kernels_xla::pack::storage`); other packed formats (K-quant blocks) as
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
            kernels_xla::pack::storage(packed, rows, width)
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
}

/// The root and handle tables. Roots and handles minted before [`seal`]
/// live for the load; the rest are the fire's and go at [`rewind`].
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
        });
        Tensor::new(bindings.len() as u32 - 1, root.rows, root.width, root.dtype)
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
        });
        Tensor::new(bindings.len() as u32 - 1, rows, t.width, t.dtype)
    }

    /// The root under a handle, and where in it the handle starts.
    #[must_use]
    pub fn locate(&self, buf: u32) -> Option<(u32, u32)> {
        let b = *self.bindings.borrow().get(buf as usize)?;
        Some((b.root, b.row_offset))
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
}

/// Where the pack's own parameter is kept among the traced roots.
const PACK_ROOT: u32 = u32::MAX;

struct State {
    pack_len: u32,
    func: Func,
    current: BTreeMap<u32, Val>,
    params: Vec<(u32, Source)>,
    written: BTreeMap<u32, ()>,
}

impl<'h> Tracer<'h> {
    #[must_use]
    pub fn new(handles: &'h Handles) -> Self {
        Self {
            handles,
            state: RefCell::new(State {
                pack_len: 0,
                func: Func::new("main"),
                current: BTreeMap::new(),
                params: Vec::new(),
                written: BTreeMap::new(),
            }),
        }
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
            (ty.dims[0] + 127) / 128 * 128
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
        func.param(ty, None)
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
        let whole = reinterpret(f, whole, r, t)?;
        if offset == 0 && t.rows == r.rows {
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
            return Err(kernels_xla::Error::Backend {
                op: "trace",
                detail: format!(
                    "a {}-wide {:?} write lands on a {}-wide {:?} root",
                    t.width, t.dtype, r.width, r.dtype
                ),
            });
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
        self.written.insert(root, ());
        Ok(())
    }
}

/// A handle may view its root under another width or element (a packed
/// weight read as raw bytes); only the identity is served for now.
fn reinterpret(f: &mut Func, v: Val, root: Root, t: Tensor) -> Result<Val, kernels_xla::Error> {
    if t.width == root.width && t.dtype == root.dtype {
        return Ok(v);
    }
    let want = storage(t.dtype, root.rows, t.width);
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
        } = &mut *state;
        let mut lens = Lens {
            handles: self.handles,
            pack_len: *pack_len,
            current,
            params,
            written,
        };
        let mut cx = Cx::new(func, &mut lens);
        body(&mut cx)
    }
}
