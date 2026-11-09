//! A typed builder for Mosaic TPU kernels, carried inside StableHLO as a
//! `stablehlo.custom_call @tpu_custom_call` (what JAX lowers a Pallas TPU
//! kernel to), with no Python in the loop.
//!
//! The payload libtpu takes (`backend_config`, a JSON object):
//! `{"custom_call_config": {"body": <base64 module>, "serialization_format":
//! 1, "needs_layout_passes": true}, "scoped_memory_configs": [{"memory_space":
//! 1, "offset": 0, "size": <vmem bytes>}]}`. The body is the kernel module
//! in its *serialized* form, which JAX writes as MLIR bytecode but libtpu
//! parses as text just as well (checked against libtpu 0.0.48): every op
//! printed generically under a `stable_mosaic.` prefix (`"stable_mosaic.
//! arith.addi"`, `"stable_mosaic.tpu.matmul"`), attributes in their own
//! dialect's syntax (`#tpu.memory_space<vmem>`), and the module attribute
//! `stable_mosaic.version` naming the op set's version (libtpu upgrades
//! older versions on the way in). This builder prints exactly that.
//!
//! A kernel is the gridded form Pallas emits: a main function over the grid
//! ids, the scalar-prefetch operands (SMEM), one VMEM block per operand and
//! result, and one "transform" function per operand/result mapping the grid
//! ids (and prefetched scalars) to the block's index. libtpu's Mosaic
//! compiler pipelines the blocks between HBM and VMEM (double buffering; a
//! block whose index repeats from one step to the next is not re-fetched,
//! a result block is written back when its index changes).
//!
//! Reference: `jax/_src/tpu_custom_call.py` (the payload) and the Mosaic
//! Pallas lowers (`jax/_src/pallas/mosaic/lowering.py`), read off JAX 0.11.2.

use std::fmt::{self, Write as _};

use crate::hlo::{Built, Elem, Malformed, Ty};

/// The op set version this builder prints (libtpu 0.0.48's current one).
const VERSION: u32 = 17;

fn bad<T>(detail: impl Into<String>) -> Built<T> {
    Err(Malformed {
        op: "mosaic",
        detail: detail.into(),
    })
}

/// Where a reference lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Vmem,
    Smem,
    /// HBM, read only by the kernel's own DMAs.
    Any,
}

impl Space {
    fn mlir(self) -> &'static str {
        match self {
            Self::Vmem => "#tpu.memory_space<vmem>",
            Self::Smem => "#tpu.memory_space<smem>",
            Self::Any => "#tpu.memory_space<any>",
        }
    }
}

/// A Mosaic value type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MTy {
    I1,
    I32,
    Index,
    Vector(Vec<i64>, Elem),
    Ref(Vec<i64>, Elem, Space),
    /// DMA semaphores (`dims` empty: one).
    Sem(Vec<i64>),
}

/// Mosaic spells integers signless.
fn elem_name(e: Elem) -> Built<&'static str> {
    Ok(match e {
        Elem::Pred => "i1",
        Elem::U4 => "i4",
        Elem::I8 | Elem::U8 => "i8",
        Elem::I16 | Elem::U16 => "i16",
        Elem::I32 | Elem::U32 => "i32",
        Elem::F16 => "f16",
        Elem::Bf16 => "bf16",
        Elem::F32 => "f32",
        Elem::F8E5m2 => "f8E5M2",
        Elem::F8E4m3fn => "f8E4M3FN",
        other => return bad(format!("no Mosaic element type for {other:?}")),
    })
}

fn shape(dims: &[i64], elem: Elem) -> String {
    let mut s = String::new();
    for d in dims {
        let _ = write!(s, "{d}x");
    }
    s.push_str(elem_name(elem).unwrap_or("?"));
    s
}

impl fmt::Display for MTy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I1 => f.write_str("i1"),
            Self::I32 => f.write_str("i32"),
            Self::Index => f.write_str("index"),
            Self::Vector(d, e) => write!(f, "vector<{}>", shape(d, *e)),
            Self::Ref(d, e, s) => write!(f, "memref<{}, {}>", shape(d, *e), s.mlir()),
            Self::Sem(d) => {
                let mut dims = String::new();
                for x in d {
                    let _ = write!(dims, "{x}x");
                }
                write!(
                    f,
                    "memref<{dims}!tpu.dma_semaphore, #tpu.memory_space<semaphore_mem>>"
                )
            }
        }
    }
}

/// A value inside one Mosaic function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct V(u32);

/// Integer comparisons (`arith.cmpi` predicates).
#[derive(Clone, Copy, Debug)]
pub enum ICmp {
    Eq = 0,
    Ne = 1,
    Slt = 2,
    Sle = 3,
    Sgt = 4,
    Sge = 5,
}

/// One function's body under construction.
pub struct Body {
    names: Vec<String>,
    tys: Vec<MTy>,
    text: String,
    depth: usize,
    next: u32,
}

impl Body {
    fn new(args: &[MTy]) -> (Self, Vec<V>) {
        let mut b = Self {
            names: Vec::new(),
            tys: Vec::new(),
            text: String::new(),
            depth: 2,
            next: 0,
        };
        let vs = args
            .iter()
            .enumerate()
            .map(|(i, t)| b.fresh(format!("%arg{i}"), t.clone()))
            .collect();
        (b, vs)
    }

    fn fresh(&mut self, name: String, ty: MTy) -> V {
        self.names.push(name);
        self.tys.push(ty);
        V(self.names.len() as u32 - 1)
    }

    #[must_use]
    pub fn ty(&self, v: V) -> &MTy {
        &self.tys[v.0 as usize]
    }

    fn line(&mut self, s: &str) {
        for _ in 0..self.depth {
            self.text.push_str("  ");
        }
        self.text.push_str(s);
        self.text.push('\n');
    }

    fn list(&self, vs: &[V], f: impl Fn(&Self, V) -> String) -> String {
        vs.iter()
            .map(|&v| f(self, v))
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Emits `stable_mosaic.<op>` generically.
    pub fn op(&mut self, op: &str, operands: &[V], attrs: &str, outs: &[MTy]) -> Vec<V> {
        let names = self.list(operands, |b, v| b.names[v.0 as usize].clone());
        let tys = self.list(operands, |b, v| b.tys[v.0 as usize].to_string());
        let attrs = if attrs.is_empty() {
            String::new()
        } else {
            format!(" {{{attrs}}}")
        };
        let results = match outs.len() {
            0 => "()".to_string(),
            1 => outs[0].to_string(),
            _ => format!(
                "({})",
                outs.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        };
        let id = self.next;
        self.next += 1;
        let head = match outs.len() {
            0 => String::new(),
            1 => format!("%{id} = "),
            n => format!("%{id}:{n} = "),
        };
        self.line(&format!(
            "{head}\"stable_mosaic.{op}\"({names}){attrs} : ({tys}) -> {results}"
        ));
        let single = outs.len() == 1;
        outs.iter()
            .enumerate()
            .map(|(i, t)| {
                let n = if single {
                    format!("%{id}")
                } else {
                    format!("%{id}#{i}")
                };
                self.fresh(n, t.clone())
            })
            .collect()
    }

    fn op1(&mut self, op: &str, operands: &[V], attrs: &str, out: MTy) -> V {
        self.op(op, operands, attrs, &[out])[0]
    }

    // ------------------------------------------------------------ scalars

    pub fn i32(&mut self, x: i64) -> V {
        self.op1(
            "arith.constant",
            &[],
            &format!("value = {x} : i32"),
            MTy::I32,
        )
    }

    pub fn index(&mut self, x: i64) -> V {
        self.op1(
            "arith.constant",
            &[],
            &format!("value = {x} : index"),
            MTy::Index,
        )
    }

    fn int_binary(&mut self, op: &str, a: V, b: V, flags: bool) -> Built<V> {
        let t = self.ty(a).clone();
        let int = match &t {
            MTy::Vector(_, e) => matches!(e, Elem::I32 | Elem::Pred),
            t => matches!(t, MTy::I32 | MTy::Index | MTy::I1),
        };
        if &t != self.ty(b) || !int {
            return bad(format!("{op} over {t} and {}", self.ty(b)));
        }
        let attrs = if flags {
            "overflowFlags = #arith.overflow<none>"
        } else {
            ""
        };
        Ok(self.op1(op, &[a, b], attrs, t))
    }

    pub fn addi(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.addi", a, b, true)
    }
    pub fn subi(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.subi", a, b, true)
    }
    pub fn muli(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.muli", a, b, true)
    }
    /// Signed division; the grid's operands are never negative here.
    pub fn divsi(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.divsi", a, b, false)
    }
    pub fn remsi(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.remsi", a, b, false)
    }
    pub fn minsi(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.minsi", a, b, false)
    }
    pub fn maxsi(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.maxsi", a, b, false)
    }
    pub fn andi(&mut self, a: V, b: V) -> Built<V> {
        self.int_binary("arith.andi", a, b, false)
    }

    pub fn cmpi(&mut self, p: ICmp, a: V, b: V) -> Built<V> {
        if self.ty(a) != self.ty(b) {
            return bad(format!("cmpi over {} and {}", self.ty(a), self.ty(b)));
        }
        let out = self.mask_of(a);
        Ok(self.op1(
            "arith.cmpi",
            &[a, b],
            &format!("predicate = {} : i64", p as i64),
            out,
        ))
    }

    /// The i1 type shaped like `v`: a mask for a vector, a scalar otherwise.
    fn mask_of(&self, v: V) -> MTy {
        match self.ty(v) {
            MTy::Vector(d, _) => MTy::Vector(d.clone(), Elem::Pred),
            _ => MTy::I1,
        }
    }

    /// `a > b` (`gt`) or `a >= b` elementwise over float vectors, ordered.
    pub fn cmpf(&mut self, gt: bool, a: V, b: V) -> Built<V> {
        if self.ty(a) != self.ty(b) {
            return bad(format!("cmpf over {} and {}", self.ty(a), self.ty(b)));
        }
        let out = self.mask_of(a);
        let p = if gt { 2 } else { 3 };
        Ok(self.op1(
            "arith.cmpf",
            &[a, b],
            &format!("fastmath = #arith.fastmath<none>, predicate = {p} : i64"),
            out,
        ))
    }

    /// `c ? a : b`; `c` is a scalar i1 or a mask shaped like `a`.
    pub fn select(&mut self, c: V, a: V, b: V) -> Built<V> {
        let mask = self.mask_of(a);
        if self.ty(a) != self.ty(b) || (self.ty(c) != &MTy::I1 && self.ty(c) != &mask) {
            return bad("select over mismatched operands");
        }
        let t = self.ty(a).clone();
        Ok(self.op1("arith.select", &[c, a, b], "", t))
    }

    /// The i32 index along `axis` of a `dims` vector (`tpu.iota`).
    pub fn iota(&mut self, dims: &[i64], axis: usize) -> V {
        self.op1(
            "tpu.iota",
            &[],
            &format!("dimensions = array<i32: {axis}>"),
            MTy::Vector(dims.to_vec(), Elem::I32),
        )
    }

    /// `e^v` elementwise.
    pub fn exp(&mut self, v: V) -> Built<V> {
        let t = self.ty(v).clone();
        if !matches!(&t, MTy::Vector(_, e) if e.is_float()) {
            return bad(format!("exp of {t}"));
        }
        Ok(self.op1("math.exp", &[v], "fastmath = #arith.fastmath<none>", t))
    }

    /// A 2-d vector transposed (`tpu.transpose`).
    pub fn transpose(&mut self, v: V) -> Built<V> {
        let (d, e) = self.vec_of(v)?;
        if d.len() != 2 {
            return bad(format!("transpose of {}", self.ty(v)));
        }
        Ok(self.op1(
            "tpu.transpose",
            &[v],
            "permutation = array<i64: 1, 0>",
            MTy::Vector(vec![d[1], d[0]], e),
        ))
    }

    /// `xs` joined along `axis` (`tpu.concatenate`).
    pub fn concat(&mut self, xs: &[V], axis: usize) -> Built<V> {
        let (mut d, e) = self.vec_of(xs[0])?;
        let mut n = 0;
        for &x in xs {
            let (dx, ex) = self.vec_of(x)?;
            let same = dx
                .iter()
                .zip(&d)
                .enumerate()
                .all(|(i, (a, b))| i == axis || a == b);
            if ex != e || dx.len() != d.len() || !same {
                return bad(format!("concat of {} with {}", self.ty(xs[0]), self.ty(x)));
            }
            n += dx[axis];
        }
        d[axis] = n;
        Ok(self.op1(
            "tpu.concatenate",
            xs,
            &format!("dimension = {axis} : i32"),
            MTy::Vector(d, e),
        ))
    }

    /// The `sizes` window of `v` at `offs` (`vector.extract_strided_slice`).
    pub fn slice(&mut self, v: V, offs: &[i64], sizes: &[i64]) -> Built<V> {
        let (d, e) = self.vec_of(v)?;
        let inside = offs
            .iter()
            .zip(sizes)
            .zip(&d)
            .all(|((o, n), m)| *o >= 0 && o + n <= *m);
        if offs.len() != d.len() || sizes.len() != d.len() || !inside {
            return bad(format!("slice {offs:?}+{sizes:?} of {}", self.ty(v)));
        }
        let list = |xs: &[i64]| {
            xs.iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        };
        let ones = vec![1; d.len()];
        Ok(self.op1(
            "vector.extract_strided_slice",
            &[v],
            &format!(
                "offsets = [{}], sizes = [{}], strides = [{}]",
                list(offs),
                list(sizes),
                list(&ones)
            ),
            MTy::Vector(sizes.to_vec(), e),
        ))
    }

    /// `v`, promised to be a multiple of `m` (`tpu.assume_multiple`): lets
    /// the compiler prove a computed window offset tile-aligned.
    pub fn assume_multiple(&mut self, v: V, m: i64) -> Built<V> {
        if self.ty(v) != &MTy::I32 {
            return bad(format!("assume_multiple of {}", self.ty(v)));
        }
        Ok(self.op1(
            "tpu.assume_multiple",
            &[v],
            &format!("multiple = {m} : i32"),
            MTy::I32,
        ))
    }

    /// An i32 as an `index` (to address a reference).
    pub fn as_index(&mut self, v: V) -> Built<V> {
        match self.ty(v) {
            MTy::Index => Ok(v),
            MTy::I32 => Ok(self.op1("arith.index_cast", &[v], "", MTy::Index)),
            t => bad(format!("index_cast of {t}")),
        }
    }

    /// The scalar `r[idx]` of an SMEM reference.
    pub fn load(&mut self, r: V, idx: &[V]) -> Built<V> {
        let MTy::Ref(dims, elem, Space::Smem) = self.ty(r).clone() else {
            return bad(format!("a scalar load from {}", self.ty(r)));
        };
        if dims.len() != idx.len() || elem != Elem::I32 {
            return bad(format!("load of {} at {} indices", self.ty(r), idx.len()));
        }
        let mut ops = vec![r];
        for &i in idx {
            ops.push(self.as_index(i)?);
        }
        Ok(self.op1("memref.load", &ops, "", MTy::I32))
    }

    // ------------------------------------------------------------ vectors

    /// `r[at .. at + dims]` of a VMEM reference as a vector (`dims` of the
    /// reference's rank).
    pub fn vload(&mut self, r: V, at: &[V], dims: &[i64]) -> Built<V> {
        let MTy::Ref(rd, elem, Space::Vmem) = self.ty(r).clone() else {
            return bad(format!("a vector load from {}", self.ty(r)));
        };
        if rd.len() != at.len() || dims.len() != rd.len() {
            return bad(format!("vector load of {dims:?} from {}", self.ty(r)));
        }
        let mut ops = vec![r];
        for &i in at {
            ops.push(self.as_index(i)?);
        }
        Ok(self.op1("vector.load", &ops, "", MTy::Vector(dims.to_vec(), elem)))
    }

    /// The whole of a VMEM reference.
    pub fn vload_all(&mut self, r: V) -> Built<V> {
        let MTy::Ref(rd, ..) = self.ty(r).clone() else {
            return bad(format!("a vector load from {}", self.ty(r)));
        };
        let zeros: Vec<V> = rd.iter().map(|_| self.index(0)).collect();
        self.vload(r, &zeros, &rd)
    }

    /// Stores `v` into a VMEM reference at `at`.
    pub fn vstore(&mut self, v: V, r: V, at: &[V]) -> Built<()> {
        let (MTy::Vector(vd, ve), MTy::Ref(rd, re, Space::Vmem)) =
            (self.ty(v).clone(), self.ty(r).clone())
        else {
            return bad(format!("store of {} into {}", self.ty(v), self.ty(r)));
        };
        if ve != re || vd.len() != rd.len() || at.len() != rd.len() {
            return bad(format!("store of {} into {}", self.ty(v), self.ty(r)));
        }
        let mut ops = vec![v, r];
        for &i in at {
            ops.push(self.as_index(i)?);
        }
        let n = at.len();
        self.op(
            "tpu.vector_store",
            &ops,
            &format!(
                "add = false, operandSegmentSizes = array<i32: 1, 1, {n}, 0>, strides = array<i32>"
            ),
            &[],
        );
        Ok(())
    }

    /// Stores `v` over the whole of a VMEM reference.
    pub fn vstore_all(&mut self, v: V, r: V) -> Built<()> {
        let MTy::Ref(rd, ..) = self.ty(r).clone() else {
            return bad(format!("a store into {}", self.ty(r)));
        };
        let zeros: Vec<V> = rd.iter().map(|_| self.index(0)).collect();
        self.vstore(v, r, &zeros)
    }

    /// `r[at ..]` as a reference of `dims` (same rank, same space).
    pub fn ref_slice(&mut self, r: V, at: &[V], dims: &[i64]) -> Built<V> {
        let out = match self.ty(r).clone() {
            MTy::Ref(rd, e, sp) if rd.len() == at.len() && dims.len() == rd.len() => {
                MTy::Ref(dims.to_vec(), e, sp)
            }
            MTy::Sem(rd) if rd.len() == at.len() && dims.len() == rd.len() => {
                MTy::Sem(dims.to_vec())
            }
            t => return bad(format!("a {dims:?} slice of {t}")),
        };
        let mut ops = vec![r];
        for &i in at {
            if self.ty(i) != &MTy::I32 {
                return bad("a slice index is i32");
            }
            ops.push(i);
        }
        let n = at.len();
        Ok(self.op1(
            "tpu.memref_slice",
            &ops,
            &format!("operandSegmentSizes = array<i32: 1, {n}, 0>"),
            out,
        ))
    }

    /// `r` without its leading unit axis.
    pub fn ref_squeeze(&mut self, r: V) -> Built<V> {
        let out = match self.ty(r).clone() {
            MTy::Ref(rd, e, sp) if rd.first() == Some(&1) => MTy::Ref(rd[1..].to_vec(), e, sp),
            MTy::Sem(rd) if rd.first() == Some(&1) => MTy::Sem(rd[1..].to_vec()),
            t => return bad(format!("a squeeze of {t}")),
        };
        Ok(self.op1("tpu.memref_squeeze", &[r], "", out))
    }

    /// Starts copying `src` into `dst`, signalling `sem`.
    pub fn dma_start(&mut self, src: V, dst: V, sem: V) -> Built<()> {
        // source, source semaphore, target, target semaphore, device, core,
        // subcore: libtpu 0.0.48 parses the seven-segment form.
        self.op(
            "tpu.enqueue_dma",
            &[src, dst, sem],
            "operandSegmentSizes = array<i32: 1, 0, 1, 1, 0, 0, 0>",
            &[],
        );
        Ok(())
    }

    /// Waits for the copy of `src` into `dst` that signals `sem`.
    pub fn dma_wait(&mut self, sem: V, src: V, dst: V) -> Built<()> {
        self.op(
            "tpu.wait_dma2",
            &[sem, src, dst],
            "operandSegmentSizes = array<i32: 1, 1, 1, 0, 0>",
            &[],
        );
        Ok(())
    }

    /// A float vector of `x` everywhere.
    pub fn splat_f(&mut self, dims: &[i64], elem: Elem, x: f64) -> Built<V> {
        let t = MTy::Vector(dims.to_vec(), elem);
        let lit = format!("dense<{}> : {t}", float_lit(x, elem)?);
        Ok(self.op1("arith.constant", &[], &format!("value = {lit}"), t))
    }

    fn vec_of(&self, v: V) -> Built<(Vec<i64>, Elem)> {
        match self.ty(v) {
            MTy::Vector(d, e) => Ok((d.clone(), *e)),
            t => bad(format!("{t} is not a vector")),
        }
    }

    /// An i32 vector of `x` everywhere.
    pub fn splat_i(&mut self, dims: &[i64], x: i64) -> Built<V> {
        let t = MTy::Vector(dims.to_vec(), Elem::I32);
        Ok(self.op1("arith.constant", &[], &format!("value = dense<{x}> : {t}"), t))
    }

    /// An i32 scalar everywhere in a `dims` vector (`vector.broadcast`).
    pub fn broadcast_i(&mut self, v: V, dims: &[i64]) -> Built<V> {
        if self.ty(v) != &MTy::I32 {
            return bad(format!("broadcast of {} as an i32", self.ty(v)));
        }
        Ok(self.op1(
            "vector.broadcast",
            &[v],
            "",
            MTy::Vector(dims.to_vec(), Elem::I32),
        ))
    }

    /// A float vector widened (`arith.extf`) or narrowed (`arith.truncf`).
    pub fn convert_f(&mut self, v: V, to: Elem) -> Built<V> {
        let (d, e) = self.vec_of(v)?;
        if e == to {
            return Ok(v);
        }
        if !e.is_float() || !to.is_float() {
            return bad(format!("float convert {e:?} -> {to:?}"));
        }
        let op = if to.bits() > e.bits() {
            "arith.extf"
        } else {
            "arith.truncf"
        };
        Ok(self.op1(op, &[v], "", MTy::Vector(d, to)))
    }

    fn float_binary(&mut self, op: &str, a: V, b: V) -> Built<V> {
        let t = self.ty(a).clone();
        if &t != self.ty(b) {
            return bad(format!("{op} over {t} and {}", self.ty(b)));
        }
        Ok(self.op1(op, &[a, b], "fastmath = #arith.fastmath<none>", t))
    }

    pub fn addf(&mut self, a: V, b: V) -> Built<V> {
        self.float_binary("arith.addf", a, b)
    }

    pub fn mulf(&mut self, a: V, b: V) -> Built<V> {
        self.float_binary("arith.mulf", a, b)
    }

    pub fn subf(&mut self, a: V, b: V) -> Built<V> {
        self.float_binary("arith.subf", a, b)
    }

    pub fn divf(&mut self, a: V, b: V) -> Built<V> {
        self.float_binary("arith.divf", a, b)
    }

    pub fn maxf(&mut self, a: V, b: V) -> Built<V> {
        self.float_binary("arith.maximumf", a, b)
    }

    /// `v` stretched along its unit axes to `dims` (`vector.broadcast`).
    pub fn broadcast(&mut self, v: V, dims: &[i64]) -> Built<V> {
        let (d, e) = self.vec_of(v)?;
        if d.len() != dims.len() || d.iter().zip(dims).any(|(&a, &b)| a != b && a != 1) {
            return bad(format!("broadcast of {} to {dims:?}", self.ty(v)));
        }
        Ok(self.op1("vector.broadcast", &[v], "", MTy::Vector(dims.to_vec(), e)))
    }

    /// `v`'s elements, in order, as a `dims` vector (`vector.shape_cast`).
    pub fn shape_cast(&mut self, v: V, dims: &[i64]) -> Built<V> {
        let (d, e) = self.vec_of(v)?;
        if d.iter().product::<i64>() != dims.iter().product::<i64>() {
            return bad(format!("shape_cast of {} to {dims:?}", self.ty(v)));
        }
        Ok(self.op1("vector.shape_cast", &[v], "", MTy::Vector(dims.to_vec(), e)))
    }

    /// Rows `lo .. lo + n` of a 2-d vector.
    pub fn rows(&mut self, v: V, lo: i64, n: i64) -> Built<V> {
        let (d, _) = self.vec_of(v)?;
        if d.len() != 2 {
            return bad(format!("rows of {}", self.ty(v)));
        }
        self.slice(v, &[lo, 0], &[n, d[1]])
    }

    /// `acc + lhs · rhs` over 2-d vectors, contracting `lhs`'s axis 1 with
    /// `rhs`'s axis 0 (`[m, k]·[k, n]`), or with its axis 1 when `rhs_t`
    /// (`[m, k]·[n, k]ᵀ`). The accumulator's type is the result's.
    pub fn matmul(&mut self, lhs: V, rhs: V, acc: V, rhs_t: bool) -> Built<V> {
        let (ld, _) = self.vec_of(lhs)?;
        let (rd, _) = self.vec_of(rhs)?;
        let (ad, ae) = self.vec_of(acc)?;
        let (k2, n) = if rhs_t {
            (rd[1], rd[0])
        } else {
            (rd[0], rd[1])
        };
        if ld.len() != 2 || rd.len() != 2 || ld[1] != k2 || ad != [ld[0], n] {
            return bad(format!(
                "matmul of {} by {} into {}",
                self.ty(lhs),
                self.ty(rhs),
                self.ty(acc)
            ));
        }
        let dims = if rhs_t {
            "#tpu.dot_dimension_numbers<[1], [1], [0], [0], [0, 0, 1, 0], [], []>"
        } else {
            "#tpu.dot_dimension_numbers<[1], [0], [0], [1], [0, 0, 1, 1], [], []>"
        };
        Ok(self.op1(
            "tpu.matmul",
            &[lhs, rhs, acc],
            &format!(
                "dimension_numbers = {dims}, transpose_lhs = false, transpose_lhs_hint = false, \
                 transpose_rhs = false"
            ),
            MTy::Vector(ad, ae),
        ))
    }

    /// `acc + lhsᵀ · rhs` over 2-d vectors (`[k, m]ᵀ·[k, n]`), the MXU
    /// reading `lhs` transposed (`transpose_lhs`).
    pub fn matmul_tl(&mut self, lhs: V, rhs: V, acc: V) -> Built<V> {
        let (ld, _) = self.vec_of(lhs)?;
        let (rd, _) = self.vec_of(rhs)?;
        let (ad, ae) = self.vec_of(acc)?;
        if ld.len() != 2 || rd.len() != 2 || ld[0] != rd[0] || ad != [ld[1], rd[1]] {
            return bad(format!(
                "matmul_tl of {} by {} into {}",
                self.ty(lhs),
                self.ty(rhs),
                self.ty(acc)
            ));
        }
        Ok(self.op1(
            "tpu.matmul",
            &[lhs, rhs, acc],
            "dimension_numbers = #tpu.dot_dimension_numbers<[0], [0], [1], [1], [0, 1, 1, 1], [], []>, \
             transpose_lhs = false, transpose_lhs_hint = false, transpose_rhs = false",
            MTy::Vector(ad, ae),
        ))
    }

    /// `acc + lhs · rhs` over 2-d or (batch-leading) 3-d vectors: `lc` and
    /// `rc` name the contracted axis of each (the minor two axes are the
    /// matrix), `fp32` asks for a full-precision product (six bf16 passes;
    /// off, one pass over bf16-rounded operands). The result's axes are the
    /// batch, then `lhs`'s free axis, then `rhs`'s.
    pub fn matmul_g(
        &mut self,
        lhs: V,
        rhs: V,
        acc: V,
        lc: usize,
        rc: usize,
        fp32: bool,
    ) -> Built<V> {
        let (ld, _) = self.vec_of(lhs)?;
        let (rd, _) = self.vec_of(rhs)?;
        let (ad, ae) = self.vec_of(acc)?;
        let r = ld.len();
        let b = r - 2;
        let (lf, rf) = (2 * b + 1 - lc, 2 * b + 1 - rc);
        let batch_ok = ld[..b] == rd[..b] && ad[..b] == ld[..b];
        if !(2..=3).contains(&r)
            || rd.len() != r
            || ad.len() != r
            || !(b..r).contains(&lc)
            || !(b..r).contains(&rc)
            || !batch_ok
            || ld[lc] != rd[rc]
            || ad[b] != ld[lf]
            || ad[b + 1] != rd[rf]
        {
            return bad(format!(
                "matmul of {} by {} into {} (contracting {lc} with {rc})",
                self.ty(lhs),
                self.ty(rhs),
                self.ty(acc)
            ));
        }
        let bl: Vec<String> = (0..b).map(|i| i.to_string()).collect();
        let mut order: Vec<String> = Vec::new();
        for i in 0..b {
            order.push(format!("0, {i}"));
        }
        order.push(format!("0, {lf}"));
        order.push(format!("1, {rf}"));
        let dims = format!(
            "#tpu.dot_dimension_numbers<[{lc}], [{rc}], [{lf}], [{rf}], [{}], [{}], [{}]>",
            order.join(", "),
            bl.join(", "),
            bl.join(", ")
        );
        let precision = if fp32 {
            ", precision = #tpu.contract_precision<fp32>"
        } else {
            ""
        };
        Ok(self.op1(
            "tpu.matmul",
            &[lhs, rhs, acc],
            &format!(
                "dimension_numbers = {dims}{precision}, transpose_lhs = false, transpose_lhs_hint = false, \
                 transpose_rhs = false"
            ),
            MTy::Vector(ad, ae),
        ))
    }

    // ------------------------------------------------------------ control

    /// `for i in lb..ub step st { body(i) }` over i32 bounds (`scf.for`, no
    /// carried values: state lives in references).
    pub fn for_range(
        &mut self,
        lb: V,
        ub: V,
        st: V,
        body: impl FnOnce(&mut Self, V) -> Built<()>,
    ) -> Built<()> {
        if [lb, ub, st].iter().any(|&v| self.ty(v) != &MTy::I32) {
            return bad("scf.for over non-i32 bounds");
        }
        let names = self.list(&[lb, ub, st], |b, v| b.names[v.0 as usize].clone());
        let iv = self.fresh(format!("%iv{}", self.next), MTy::I32);
        self.next += 1;
        let ivn = self.names[iv.0 as usize].clone();
        self.line(&format!("\"stable_mosaic.scf.for\"({names}) ({{"));
        self.line(&format!("^bb0({ivn}: i32):"));
        self.depth += 1;
        body(self, iv)?;
        self.line("\"stable_mosaic.scf.yield\"() : () -> ()");
        self.depth -= 1;
        self.line("}) : (i32, i32, i32) -> ()");
        Ok(())
    }

    /// The sum of a 2-d f32 vector over its rows, kept as `[1, cols]`
    /// (`vector.multi_reduction` over axis 0).
    pub fn sum_rows(&mut self, v: V) -> Built<V> {
        let (d, e) = self.vec_of(v)?;
        if d.len() != 2 || e != Elem::F32 {
            return bad(format!("a column sum of {}", self.ty(v)));
        }
        let init = self.splat_f(&[d[1]], e, 0.0)?;
        let r = self.op1(
            "vector.multi_reduction",
            &[v, init],
            "kind = #vector.kind<add>, reduction_dims = array<i64: 0>",
            MTy::Vector(vec![d[1]], e),
        );
        self.shape_cast(r, &[1, d[1]])
    }

    /// The largest of a 2-d f32 vector's rows, kept as `[1, cols]`
    /// (`vector.multi_reduction` over axis 0).
    pub fn max_rows(&mut self, v: V) -> Built<V> {
        let (d, e) = self.vec_of(v)?;
        if d.len() != 2 || e != Elem::F32 {
            return bad(format!("a column max of {}", self.ty(v)));
        }
        let init = self.splat_f(&[d[1]], e, f64::NEG_INFINITY)?;
        let r = self.op1(
            "vector.multi_reduction",
            &[v, init],
            "kind = #vector.kind<maximumf>, reduction_dims = array<i64: 0>",
            MTy::Vector(vec![d[1]], e),
        );
        self.shape_cast(r, &[1, d[1]])
    }

    /// `scf.if cond { then }` (no results).
    pub fn when(&mut self, cond: V, then: impl FnOnce(&mut Self) -> Built<()>) -> Built<()> {
        if self.ty(cond) != &MTy::I1 {
            return bad(format!("scf.if on {}", self.ty(cond)));
        }
        let c = self.names[cond.0 as usize].clone();
        self.line(&format!("\"stable_mosaic.scf.if\"({c}) ({{"));
        self.depth += 1;
        then(self)?;
        self.line("\"stable_mosaic.scf.yield\"() : () -> ()");
        self.depth -= 1;
        self.line("}, {");
        self.depth += 1;
        self.line("\"stable_mosaic.scf.yield\"() : () -> ()");
        self.depth -= 1;
        self.line("}) : (i1) -> ()");
        Ok(())
    }

    fn ret(&mut self, vs: &[V]) {
        let names = self.list(vs, |b, v| b.names[v.0 as usize].clone());
        let tys = self.list(vs, |b, v| b.tys[v.0 as usize].to_string());
        self.line(&format!(
            "\"stable_mosaic.func.return\"({names}) : ({tys}) -> ()"
        ));
    }
}

/// How a grid axis may be run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sem {
    Parallel,
    Arbitrary,
}

/// How a block's index addresses its array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Window {
    /// The index counts whole blocks along each axis.
    Blocked,
    /// The index is an element offset along each axis (it need not be a
    /// multiple of the block; the DMA wants it tile-aligned).
    Element,
    /// No window: the whole array stays in HBM for the kernel's own DMAs.
    Any,
}

/// The grid ids and the prefetched scalars' references, as a transform or
/// the kernel body sees them.
pub struct Grid {
    pub ids: Vec<V>,
    pub prefetch: Vec<V>,
}

/// Maps the grid to one block's index (i32 per axis).
pub type IndexMap = Box<dyn Fn(&mut Body, &Grid) -> Built<Vec<V>>>;

/// One operand or result of a kernel: its whole array and the block of it
/// a grid step sees.
pub struct Operand {
    pub array: Vec<i64>,
    pub elem: Elem,
    pub block: Vec<i64>,
    pub window: Window,
    pub index: IndexMap,
}

/// A gridded Mosaic kernel.
pub struct Kernel {
    pub name: String,
    pub grid: Vec<i64>,
    pub semantics: Vec<Sem>,
    /// i32 arrays read before the grid runs (SMEM), passed first.
    pub prefetch: Vec<Vec<i64>>,
    pub ins: Vec<Operand>,
    pub outs: Vec<Operand>,
    /// The scoped VMEM the kernel may use (the compiler's default is
    /// 32 MiB on v6e).
    pub vmem_limit: Option<u64>,
    /// VMEM buffers and semaphores the body keeps across grid steps.
    pub scratch: Vec<MTy>,
}

/// What the kernel body gets: the grid and one VMEM block per operand and
/// result.
pub struct Blocks {
    pub grid: Grid,
    pub ins: Vec<V>,
    pub outs: Vec<V>,
    pub scratch: Vec<V>,
}

/// A kernel ready to call from StableHLO.
#[derive(Clone, Debug)]
pub struct Compiled {
    /// The Mosaic module text (for tests and debugging).
    pub module: String,
    /// The custom call's `backend_config`.
    pub config: String,
    pub operands: Vec<Ty>,
    pub results: Vec<Ty>,
}

impl Kernel {
    /// Builds the module; `body` writes one grid step.
    pub fn build(&self, body: impl FnOnce(&mut Body, &Blocks) -> Built<()>) -> Built<Compiled> {
        if self.grid.len() != self.semantics.len() || self.grid.is_empty() {
            return bad("a grid needs one semantics per axis");
        }
        for o in self.ins.iter().chain(&self.outs) {
            if o.window == Window::Any {
                continue;
            }
            if o.block.len() != o.array.len() {
                return bad(format!("a {:?} block of a {:?} array", o.block, o.array));
            }
            for (i, (&b, &a)) in o.block.iter().zip(&o.array).enumerate() {
                if b > a || b <= 0 {
                    return bad(format!("block {:?} of {:?}", o.block, o.array));
                }
                // Mosaic's tiling rule for the minor two axes: the block
                // equals the array or is whole (8, 128) tiles.
                let rank = o.block.len();
                let tile = if i + 1 == rank {
                    128
                } else if i + 2 == rank {
                    8
                } else {
                    1
                };
                if b != a && b % tile != 0 {
                    return bad(format!(
                        "block {:?} of {:?} breaks the (8, 128) tiling",
                        o.block, o.array
                    ));
                }
            }
        }
        let grid_tys: Vec<MTy> = self.grid.iter().map(|_| MTy::I32).collect();
        let pre_tys: Vec<MTy> = self
            .prefetch
            .iter()
            .map(|d| MTy::Ref(d.clone(), Elem::I32, Space::Smem))
            .collect();
        let block_tys: Vec<MTy> = self
            .ins
            .iter()
            .chain(&self.outs)
            .map(|o| match o.window {
                Window::Any => MTy::Ref(o.array.clone(), o.elem, Space::Any),
                _ => MTy::Ref(o.block.clone(), o.elem, Space::Vmem),
            })
            .collect();

        let mut s = String::new();
        let _ = writeln!(
            s,
            "module attributes {{stable_mosaic.version = {VERSION} : i64}} {{"
        );

        // The kernel.
        let mut args = grid_tys.clone();
        args.extend(pre_tys.iter().cloned());
        args.extend(block_tys.iter().cloned());
        args.extend(self.scratch.iter().cloned());
        let (mut b, vs) = Body::new(&args);
        let (g, rest) = vs.split_at(grid_tys.len());
        let (p, rest) = rest.split_at(pre_tys.len());
        let (ins, rest) = rest.split_at(self.ins.len());
        let (outs, scratch) = rest.split_at(self.outs.len());
        let blocks = Blocks {
            grid: Grid {
                ids: g.to_vec(),
                prefetch: p.to_vec(),
            },
            ins: ins.to_vec(),
            outs: outs.to_vec(),
            scratch: scratch.to_vec(),
        };
        body(&mut b, &blocks)?;
        b.ret(&[]);
        let sems = self
            .semantics
            .iter()
            .map(|s| match s {
                Sem::Parallel => "#tpu.dimension_semantics<parallel>",
                Sem::Arbitrary => "#tpu.dimension_semantics<arbitrary>",
            })
            .collect::<Vec<_>>()
            .join(", ");
        let windows = self
            .ins
            .iter()
            .chain(&self.outs)
            .enumerate()
            .map(|(i, o)| {
                if o.window == Window::Any {
                    return "{}".to_string();
                }
                let bounds = o.block.iter().map(ToString::to_string).collect::<Vec<_>>().join(", ");
                let kind = match o.window {
                    Window::Blocked | Window::Any => String::new(),
                    Window::Element => {
                        let z = vec!["0"; o.block.len()].join(", ");
                        format!(", window_kind = #tpu.element_window<[{z}], [{z}]>")
                    }
                };
                format!(
                    "{{transform_indices = @transform_{i}, window_bounds = array<i64: {bounds}>{kind}}}"
                )
            })
            .collect::<Vec<_>>()
            .join(", ");
        let fty = |args: &[MTy], res: &[MTy]| {
            format!(
                "({}) -> ({})",
                args.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", "),
                res.iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let bounds = self
            .grid
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        write_func(&mut s, &args, &b.text);
        let _ = writeln!(
            s,
            "  }}) {{dimension_semantics = [{sems}], function_type = {}, iteration_bounds = \
             array<i64: {bounds}>, scalar_prefetch = {} : i64, scratch_operands = {} : i64, sym_name \
             = \"{}\", tpu.core_type = #tpu.core_type<tc>, window_params = [{windows}]}} : () -> ()",
            fty(&args, &[]),
            self.prefetch.len(),
            self.scratch.len(),
            self.name
        );

        // One transform per window.
        let mut targs = grid_tys.clone();
        targs.extend(pre_tys.iter().cloned());
        for (i, o) in self.ins.iter().chain(&self.outs).enumerate() {
            if o.window == Window::Any {
                continue;
            }
            let (mut b, vs) = Body::new(&targs);
            let (g, p) = vs.split_at(grid_tys.len());
            let grid = Grid {
                ids: g.to_vec(),
                prefetch: p.to_vec(),
            };
            let idx = (o.index)(&mut b, &grid)?;
            if idx.len() != o.block.len() || idx.iter().any(|&v| b.ty(v) != &MTy::I32) {
                return bad(format!(
                    "transform {i} returns {} indices for a rank-{} block",
                    idx.len(),
                    o.block.len()
                ));
            }
            b.ret(&idx);
            write_func(&mut s, &targs, &b.text);
            let res: Vec<MTy> = idx.iter().map(|_| MTy::I32).collect();
            let _ = writeln!(
                s,
                "  }}) {{function_type = {}, sym_name = \"transform_{i}\"}} : () -> ()",
                fty(&targs, &res)
            );
        }
        s.push_str("}\n");

        let mut config = format!(
            "{{\"custom_call_config\": {{\"body\": \"{}\", \"serialization_format\": 1, \
             \"needs_layout_passes\": true}}",
            base64(s.as_bytes())
        );
        if let Some(bytes) = self.vmem_limit {
            let _ = write!(
                config,
                ", \"scoped_memory_configs\": [{{\"memory_space\":1, \"offset\": 0, \"size\": {bytes}}}]"
            );
        }
        config.push('}');

        let mut operands: Vec<Ty> = self
            .prefetch
            .iter()
            .map(|d| Ty::new(Elem::I32, d))
            .collect();
        operands.extend(self.ins.iter().map(|o| Ty::new(o.elem, &o.array)));
        let results = self
            .outs
            .iter()
            .map(|o| Ty::new(o.elem, &o.array))
            .collect();
        Ok(Compiled {
            module: s,
            config,
            operands,
            results,
        })
    }
}

fn write_func(s: &mut String, args: &[MTy], body: &str) {
    let head = args
        .iter()
        .enumerate()
        .map(|(i, t)| format!("%arg{i}: {t}"))
        .collect::<Vec<_>>()
        .join(", ");
    let _ = writeln!(s, "  \"stable_mosaic.func.func\"() ({{");
    let _ = writeln!(s, "  ^bb0({head}):");
    s.push_str(body);
}

impl Compiled {
    /// Calls the kernel from a StableHLO function: `operands` are the
    /// prefetched i32 arrays, then the inputs, in declaration order.
    pub fn call(
        &self,
        f: &mut crate::hlo::Func,
        operands: &[crate::hlo::Val],
    ) -> Built<Vec<crate::hlo::Val>> {
        if operands.len() != self.operands.len() {
            return bad(format!(
                "{} operands for a kernel of {}",
                operands.len(),
                self.operands.len()
            ));
        }
        for (i, (&v, want)) in operands.iter().zip(&self.operands).enumerate() {
            if f.ty(v) != want {
                return bad(format!(
                    "operand {i} is {}, the kernel takes {want}",
                    f.ty(v)
                ));
            }
        }
        Ok(f.custom_call(
            "tpu_custom_call",
            operands,
            &self.config,
            self.results.clone(),
        ))
    }

    /// [`Compiled::call`] with result `out` written in place over operand
    /// `operand` (counting the prefetched arrays), for each pair of
    /// `aliases`: the kernel writes only the blocks it visits, the rest of
    /// the operand stands.
    pub fn call_aliased(
        &self,
        f: &mut crate::hlo::Func,
        operands: &[crate::hlo::Val],
        aliases: &[(usize, usize)],
    ) -> Built<Vec<crate::hlo::Val>> {
        for &(out, operand) in aliases {
            if self.results.get(out) != self.operands.get(operand) {
                return bad(format!(
                    "result {out} cannot stand in for operand {operand}"
                ));
            }
        }
        if operands.len() != self.operands.len() {
            return bad(format!(
                "{} operands for a kernel of {}",
                operands.len(),
                self.operands.len()
            ));
        }
        for (i, (&v, want)) in operands.iter().zip(&self.operands).enumerate() {
            if f.ty(v) != want {
                return bad(format!(
                    "operand {i} is {}, the kernel takes {want}",
                    f.ty(v)
                ));
            }
        }
        Ok(f.custom_call_aliased(
            "tpu_custom_call",
            operands,
            &self.config,
            self.results.clone(),
            aliases,
        ))
    }
}

/// Standard base64 with padding.
#[must_use]
/// A float literal MLIR parses: an infinity as its bits.
fn float_lit(x: f64, elem: Elem) -> Built<String> {
    if x.is_finite() {
        return Ok(format!("{x:.9e}"));
    }
    let neg = x < 0.0;
    Ok(match elem {
        Elem::F32 if neg => "0xFF800000".into(),
        Elem::F32 => "0x7F800000".into(),
        Elem::Bf16 if neg => "0xFF80".into(),
        Elem::Bf16 => "0x7F80".into(),
        e => return bad(format!("an infinity of {e:?}")),
    })
}

pub fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (u32::from(c[0]) << 16)
            | (u32::from(*c.get(1).unwrap_or(&0)) << 8)
            | u32::from(*c.get(2).unwrap_or(&0));
        for i in 0..4 {
            if i <= c.len() {
                s.push(A[((n >> (18 - 6 * i)) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_the_rfc_vectors() {
        for (x, y) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("fooba", "Zm9vYmE="),
            ("foobar", "Zm9vYmFy"),
        ] {
            assert_eq!(base64(x.as_bytes()), y);
        }
    }

    #[test]
    fn an_add_kernel_prints_the_serialized_form() {
        let k = Kernel {
            name: "add".into(),
            grid: vec![4],
            semantics: vec![Sem::Parallel],
            prefetch: vec![],
            ins: (0..2)
                .map(|_| Operand {
                    array: vec![32, 128],
                    elem: Elem::F32,
                    block: vec![8, 128],
                    window: Window::Blocked,
                    index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![g.ids[0], b.i32(0)])),
                })
                .collect(),
            outs: vec![Operand {
                array: vec![32, 128],
                elem: Elem::F32,
                block: vec![8, 128],
                window: Window::Blocked,
                index: Box::new(|b: &mut Body, g: &Grid| Ok(vec![g.ids[0], b.i32(0)])),
            }],
            vmem_limit: None,
            scratch: vec![],
        };
        let c = k
            .build(|b, blk| {
                let x = b.vload_all(blk.ins[0])?;
                let y = b.vload_all(blk.ins[1])?;
                let z = b.addf(x, y)?;
                b.vstore_all(z, blk.outs[0])
            })
            .unwrap();
        assert!(
            c.module.contains("\"stable_mosaic.arith.addf\""),
            "{}",
            c.module
        );
        assert!(
            c.module
                .contains("window_params = [{transform_indices = @transform_0")
        );
        assert!(
            c.config
                .starts_with("{\"custom_call_config\": {\"body\": \"")
        );
        assert_eq!(c.operands.len(), 2);
    }
}
