//! Roots, handles and the fire tracer.
//!
//! A *root* is one array the fire program may name: a weight plane, a pool
//! plane, a per-fire input, or an activation the plan computes. A *handle*
//! (the `buf` of a `kernels_cerebras::Tensor`) is a row window of a root.
//! The engine resolves plan values to handles exactly as engine-wgpu does;
//! the tracer then turns every kernel's reads and writes of handles into
//! device buffers of one CSL program:
//!
//! - every root a kernel touches becomes an exported buffer of the program,
//!   uploaded before the run when it lives on the host (weights, pools,
//!   inputs) and downloaded after when a phase wrote it;
//! - an activation root is a device array that starts as zeros;
//! - a handle that windows its root reads through a copy into a private
//!   array and writes back after its phase, so kernels only see whole
//!   buffers.

use std::cell::RefCell;
use std::collections::BTreeMap;

use dtype::Dtype;
use kernels_cerebras::phase::{BankSplit, Buf, ColPlane, HostOp, Reduce, Segment};
use kernels_cerebras::{Cx, Emit, Tensor};

use crate::fabric::{Emitting, HostPhase, Operands, Placed, Program, Rendered, elem_of};

mod text;

pub use text::{render, unrender};

/// Where a root's value comes from when the program first reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Source {
    /// A weight plane, landed at load; `plane` numbers the planes of one
    /// param (0 codes/dense, 1 scales, 2 biases).
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

/// One array: `rows × width` elements of `dtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Root {
    pub source: Source,
    pub dtype: Dtype,
    pub rows: u32,
    pub width: u32,
}

/// Bytes one row of `width` logical elements of a packed format occupies,
/// as the checkpoint lands it (see engine-wgpu `plane_bytes`).
#[must_use]
pub fn packed_row_bytes(dtype: Dtype, width: u64) -> Option<u64> {
    Some(match dtype {
        Dtype::Mxfp4 | Dtype::U8g64 => width,
        Dtype::U4g64 | Dtype::U4g32 | Dtype::U4g64tiled => width.div_ceil(2),
        Dtype::U2g32 | Dtype::U2g64 | Dtype::U2g128 => width.div_ceil(4),
        Dtype::U2g16k | Dtype::I3g16k | Dtype::U4g32k | Dtype::U5g32k | Dtype::I6g16k => width,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug)]
struct Binding {
    root: u32,
    row_offset: u32,
}

/// The root and handle tables. Roots and handles minted before
/// [`Handles::seal`] live for the load; the rest are the fire's and go at
/// [`Handles::rewind`].
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
/// pool result aliases. `names` are the parameters' exported buffer names;
/// `outputs` the results' names and shapes, in result order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Signature {
    pub params: Vec<Source>,
    /// For each result: the pool it updates (and the parameter it aliases),
    /// or `None` for an extra output (readouts).
    pub results: Vec<Option<(Source, usize)>>,
    pub names: Vec<String>,
    pub outputs: Vec<(String, (Dtype, u32, u32))>,
    /// Every buffer of the fire, with its shape and element type.
    pub buffers: Vec<crate::fabric::Export>,
    /// Buffers that are windows of the pack: `(name, word offset, words)`.
    pub packed_views: Vec<(String, u32, u32)>,
}

/// Separates the phase programs inside a fire's text.
pub const PHASE_SEPARATOR: &str = "\n=====phase\n";

/// The tracer: one CSL program under construction for one fire.
pub struct Tracer<'h> {
    handles: &'h Handles,
    plan: crate::fabric::Planner,
    state: RefCell<State>,
}

/// Where the pack's own parameter is kept among the traced roots.
const PACK_ROOT: u32 = u32::MAX;

struct State {
    pack_len: u32,
    program: Program,
    /// Root -> its exported buffer.
    current: BTreeMap<u32, Buf>,
    params: Vec<(u32, Source)>,
    /// Weights also bound packed (two bf16 a word) under their own export
    /// name (`root_name + "h"`), after `params` in the signature.
    halves: Vec<(u32, Source)>,
    written: BTreeMap<u32, ()>,
    /// Extra outputs asked for: whole roots downloaded after the run.
    extras: Vec<u32>,
    /// Pack windows: `(buffer name, word offset, words)`.
    views: Vec<(String, u32, u32)>,
}

/// The name a root's buffer exports under.
/// Whether a kv pool root (bf16, a pool) stays on the device: kept pools
/// on (the mode needs one resident server over a unified shape to be
/// right; a dry shell traced for the shape marks the pools alike, so the
/// partition it picks already counts them).
fn kept_pool(_program: &Program, r: &Root) -> bool {
    crate::fabric::keep_pools() && matches!(r.source, Source::Pool { .. }) && r.dtype == Dtype::Bf16
}

pub(crate) fn root_name(root: u32) -> String {
    if root == PACK_ROOT {
        "pack".to_string()
    } else {
        format!("r{root}")
    }
}

impl<'h> Tracer<'h> {
    #[must_use]
    pub fn new(handles: &'h Handles) -> Self {
        Self {
            handles,
            plan: crate::fabric::Planner {
                fabric: crate::fabric::terms(),
            },
            state: RefCell::new(State {
                pack_len: 0,
                program: Program::new(),
                current: BTreeMap::new(),
                params: Vec::new(),
                halves: Vec::new(),
                written: BTreeMap::new(),
                extras: Vec::new(),
                views: Vec::new(),
            }),
        }
    }

    /// Declares how long the fire's input pack is (see `Source::Packed`).
    #[must_use]
    pub fn with_pack(self, len: u32) -> Self {
        self.state.borrow_mut().pack_len = len;
        self
    }

    /// The model-wide shape a whole-fire program follows (see
    /// `crate::fabric::Unified`).
    pub fn with_unified(self, unified: Option<crate::fabric::Unified>) -> Self {
        self.state.borrow_mut().program.unified = unified;
        self
    }

    fn locate(&self, t: Tensor) -> Result<(u32, u32), kernels_cerebras::Error> {
        self.handles
            .locate(t.buf)
            .ok_or_else(|| kernels_cerebras::Error::Backend {
                op: "trace",
                detail: format!("handle {} was never minted", t.buf),
            })
    }

    /// Asks for the whole root under `t` as an extra output of the fire; the
    /// host slices what it needs. Returns the root's index among the extras.
    pub fn extra(&self, t: Tensor) -> Result<usize, kernels_cerebras::Error> {
        let (root, _) = self.locate(t)?;
        let mut state = self.state.borrow_mut();
        let State {
            program,
            current,
            params,
            pack_len,
            extras,
            views,
            ..
        } = &mut *state;
        root_buf(
            self.handles,
            program,
            current,
            params,
            views,
            *pack_len,
            root,
        )?;
        if let Some(at) = extras.iter().position(|r| *r == root) {
            return Ok(at);
        }
        extras.push(root);
        Ok(extras.len() - 1)
    }

    /// Finishes the program: pool roots written become aliased results,
    /// then the extra outputs. Returns the rendered text and its signature.
    /// Under kept pools a fire that does not render as one whole-fire
    /// program is refused: its fallback programs would hold pools of their
    /// own, apart from the resident server's.
    pub fn finish(self) -> crate::Result<(String, Signature)> {
        let State {
            program,
            current,
            params,
            halves,
            written,
            extras,
            views,
            ..
        } = self.state.into_inner();
        let mut sig_results = Vec::new();
        let mut outputs = Vec::new();
        for (&root, ()) in &written {
            let source = self.handles.root_of(root).source;
            if !source.persistent() {
                continue;
            }
            let param = params
                .iter()
                .position(|(r, _)| *r == root)
                .expect("a pool root is read before it is written; every pool op scatters");
            sig_results.push(Some((source, param)));
            let r = self.handles.root_of(root);
            outputs.push((current[&root].name.clone(), (r.dtype, r.rows, r.width)));
        }
        for root in extras {
            sig_results.push(None);
            let r = self.handles.root_of(root);
            outputs.push((current[&root].name.clone(), (r.dtype, r.rows, r.width)));
        }
        let names = params
            .iter()
            .map(|(root, _)| root_name(*root))
            .chain(
                halves
                    .iter()
                    .map(|(root, _)| format!("{}h", root_name(*root))),
            )
            .collect();
        let (phases, whole) = program.render_phases_whole();
        if crate::fabric::keep_pools() && !whole {
            return Err(crate::Fault::Unbound {
                what: "the fire does not render as one whole-fire program, which kept pools need (the refusal is logged at debug level)".to_string(),
            });
        }
        let text = phases
            .iter()
            .map(render)
            .collect::<Vec<_>>()
            .join(PHASE_SEPARATOR);
        Ok((
            text,
            Signature {
                params: params.into_iter().chain(halves).map(|(_, s)| s).collect(),
                results: sig_results,
                names,
                outputs,
                buffers: program.exports(),
                packed_views: views,
            },
        ))
    }
}

/// The exported buffer of `root`, declared on first use: an input the host
/// uploads for a root that lives on the host, zeros for an activation.
fn root_buf(
    handles: &Handles,
    program: &mut Program,
    current: &mut BTreeMap<u32, Buf>,
    params: &mut Vec<(u32, Source)>,
    views: &mut Vec<(String, u32, u32)>,
    pack_len: u32,
    root: u32,
) -> Result<Buf, kernels_cerebras::Error> {
    if let Some(b) = current.get(&root) {
        return Ok(b.clone());
    }
    let r = handles.root_of(root);
    elem_of("trace", r.dtype)?;
    let buf = if let Source::Packed { offset } = r.source {
        // A packed input is a window of the pack root; the host cuts it.
        if let std::collections::btree_map::Entry::Vacant(slot) = current.entry(PACK_ROOT) {
            params.push((PACK_ROOT, Source::Pack));
            slot.insert(program.declare("pack", Dtype::I32, pack_len.max(1), 1, true)?);
        }
        let n = r.rows * r.width;
        let view = program.declare(&root_name(root), r.dtype, r.rows, r.width, true)?;
        views.push((view.name.clone(), offset, n));
        view
    } else if r.source.on_device() {
        params.push((root, r.source));
        let b = program.declare(&root_name(root), r.dtype, r.rows, r.width, true)?;
        if matches!(r.source, Source::Weight { .. }) || kept_pool(program, &r) {
            program.keep(&b.name);
        }
        b
    } else {
        program.declare(&root_name(root), r.dtype, r.rows, r.width, false)?
    };
    current.insert(root, buf.clone());
    Ok(buf)
}

/// The tracer's `Operands`: the handle tables plus the state's maps.
struct Lens<'a> {
    handles: &'a Handles,
    fabric: crate::fabric::Fabric,
    pack_len: u32,
    current: &'a mut BTreeMap<u32, Buf>,
    params: &'a mut Vec<(u32, Source)>,
    halves: &'a mut Vec<(u32, Source)>,
    written: &'a mut BTreeMap<u32, ()>,
    views: &'a mut Vec<(String, u32, u32)>,
}

impl Lens<'_> {
    fn root(
        &mut self,
        program: &mut Program,
        t: Tensor,
    ) -> Result<(u32, u32, Buf), kernels_cerebras::Error> {
        let (root, offset) =
            self.handles
                .locate(t.buf)
                .ok_or_else(|| kernels_cerebras::Error::Backend {
                    op: "trace",
                    detail: format!("handle {} was never minted", t.buf),
                })?;
        let r = self.handles.root_of(root);
        if t.width != r.width || t.dtype != r.dtype {
            return Err(kernels_cerebras::Error::Backend {
                op: "trace",
                detail: format!(
                    "a {}-wide {:?} handle views a {}-wide {:?} root",
                    t.width, t.dtype, r.width, r.dtype
                ),
            });
        }
        let buf = root_buf(
            self.handles,
            program,
            self.current,
            self.params,
            self.views,
            self.pack_len,
            root,
        )?;
        Ok((root, offset, buf))
    }
}

impl Operands for Lens<'_> {
    fn read(&mut self, p: &mut Program, t: Tensor) -> Result<Placed, kernels_cerebras::Error> {
        let (_, offset, whole) = self.root(p, t)?;
        Ok(Placed {
            whole,
            offset,
            rows: t.rows,
        })
    }

    fn write(&mut self, p: &mut Program, t: Tensor) -> Result<Placed, kernels_cerebras::Error> {
        let (root, offset, whole) = self.root(p, t)?;
        self.written.insert(root, ());
        p.written(&whole.name);
        Ok(Placed {
            whole,
            offset,
            rows: t.rows,
        })
    }

    /// A bf16 weight read whole, packed two halves a word, under
    /// `PIE_CEREBRAS_HALF_WEIGHTS`: bound beside the plain parameters.
    fn read_half(
        &mut self,
        p: &mut Program,
        t: Tensor,
    ) -> Result<Option<Placed>, kernels_cerebras::Error> {
        if !self.fabric.half_weights {
            return Ok(None);
        }
        let Some((root, offset)) = self.handles.locate(t.buf) else {
            return Ok(None);
        };
        let r = self.handles.root_of(root);
        let whole_weight = matches!(r.source, Source::Weight { .. })
            && r.dtype == Dtype::Bf16
            && r.width.is_multiple_of(2)
            && offset == 0
            && t.rows == r.rows
            && t.width == r.width;
        if !whole_weight {
            return Ok(None);
        }
        if !self.halves.iter().any(|(rt, _)| *rt == root) {
            self.halves.push((root, r.source));
        }
        let buf = p.declare_packed(&format!("{}h", root_name(root)), r.dtype, r.rows, r.width)?;
        p.keep(&buf.name);
        Ok(Some(Placed {
            whole: buf,
            offset: 0,
            rows: r.rows,
        }))
    }

    /// A bf16 kv pool plane kept packed under its plain root name (every
    /// phase that touches the pool sees the packed form), under
    /// `PIE_CEREBRAS_HALF_POOL`.
    fn pool_half(
        &mut self,
        p: &mut Program,
        t: Tensor,
    ) -> Result<Option<Placed>, kernels_cerebras::Error> {
        if !self.fabric.half_pool {
            return Ok(None);
        }
        let Some((root, offset)) = self.handles.locate(t.buf) else {
            return Ok(None);
        };
        let r = self.handles.root_of(root);
        let whole_pool = matches!(r.source, Source::Pool { .. })
            && r.dtype == Dtype::Bf16
            && r.width.is_multiple_of(2)
            && offset == 0
            && t.rows == r.rows
            && t.width == r.width;
        if !whole_pool {
            return Ok(None);
        }
        // A pool read packed is written in place too (kv_append): the root
        // counts as written, so the fire returns it to the pool store (the
        // first packed run of pico lost every appended key here).
        self.written.insert(root, ());
        if let Some(b) = self.current.get(&root) {
            return Ok(Some(Placed {
                whole: b.clone(),
                offset: 0,
                rows: r.rows,
            }));
        }
        let buf = p.declare_packed(&root_name(root), r.dtype, r.rows, r.width)?;
        if kept_pool(p, &r) {
            p.keep(&buf.name);
        }
        self.params.push((root, r.source));
        self.current.insert(root, buf.clone());
        Ok(Some(Placed {
            whole: buf,
            offset: 0,
            rows: r.rows,
        }))
    }
}

impl Emit for Tracer<'_> {
    fn plan(&self) -> &dyn kernels_cerebras::plan::Plan {
        &self.plan
    }

    fn scope(&self, name: &str) {
        self.state.borrow_mut().program.scope(name);
    }

    fn emit(
        &self,
        body: &mut dyn FnMut(&mut Cx<'_>) -> Result<(), kernels_cerebras::Error>,
    ) -> Result<(), kernels_cerebras::Error> {
        let mut state = self.state.borrow_mut();
        let State {
            program,
            current,
            params,
            halves,
            written,
            pack_len,
            views,
            ..
        } = &mut *state;
        let mut lens = Lens {
            handles: self.handles,
            fabric: self.plan.fabric,
            pack_len: *pack_len,
            current,
            params,
            halves,
            written,
            views,
        };
        let scope = program.take_scope();
        let mut sink = Emitting::new(program, &mut lens, &self.plan);
        let mut cx = Cx::new(&mut sink, &scope);
        body(&mut cx)?;
        cx.finish();
        Ok(())
    }
}
