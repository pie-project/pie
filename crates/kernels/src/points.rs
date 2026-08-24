//! Baker's declaration floor (.wiki/baker.md): one trait per family, one
//! method per point. The method's path IS the point's name — `Norm::rmsnorm`
//! states `"norm.rmsnorm"` and nothing else spells it. A plane implements a
//! family on its own `Ctx`; a method it does not override answers with the
//! default body below — an unclaimed point is a measured backlog row, never
//! a compile error and never a trace-time panic.
//!
//! Declarations are semantic: statement operands and scalars only, `Out`
//! slots last. Plane staging (plan residents, host mirrors, derived widths)
//! never appears here — an implementation pulls it from `self`.

//! A point's arity IS its slot list, and both generators read that list off
//! the signature. Folding a wide one into a struct would hide the slots
//! from the table, so wide is what a wide point looks like.
#![allow(clippy::too_many_arguments)]

use kernels_macros::points;

use crate::routine::{Cache, Const, ConstRun, Elem, In, InOut, Out, Refusal};

/// A point, as `#[points]` reads it off the method that declares it. The two
/// generators — the DSL's statement builders and a plane's dispatch — write
/// from this and from nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    /// `"norm.rmsnorm"`: the family, then the method, verbatim.
    pub name: &'static str,

    /// The method's `T: Scalar` generics. A dispatch's match is `Elem^axes`,
    /// and every `Dtype::Generic` on a slot indexes into it.
    pub axes: usize,

    /// The method's `R: Repr` generics — the BANK axes, which are not
    /// elements and are counted apart for that reason. A dispatch's match is
    /// `Elem^axes × Repr^reprs` and every [`Dtype::Bank`] on a slot indexes
    /// into this run. Zero on every point but the quantised ones.
    pub reprs: usize,

    /// The operands, in declaration order.
    pub slots: &'static [Slot],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    /// The parameter's own name, which is the operand's.
    pub name: &'static str,

    pub mark: Mark,

    pub dtype: Dtype,
}

/// What the slot is to a statement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    In,

    InOut,

    Out,

    /// `Const<Self::Tensor<..>>`: a weight the statement carries, and what
    /// registers the Load contract's parameter.
    Const,

    /// `Cache<Self::Recurrent>` or `Cache<Self::Pages>`: a row of a POOL the
    /// driver keeps across fires, named by the statement's cache reference.
    ///
    /// A mark says WHO BINDS THE SLOT, and this is the third binder. `In`,
    /// `InOut` and `Out` are the arena's; `Const` is the load-time
    /// parameter table's; `Cache` is the cache pool's, and the future
    /// binder has to know the difference before it can answer one — an
    /// arena region is minted for this fire, a pool row outlives it and is
    /// addressed by the request's slot.
    Cache,

    /// A bare host scalar, which is what wearing no mark means.
    Scalar,
}

/// The dtype the slot's payload rides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// `Self::Tensor<T>` for the method's n-th axis: slots sharing an axis
    /// are the same dtype at every instantiation, which is the declaration's
    /// whole claim about them.
    Generic(usize),

    /// Spelled in the declaration: `Self::Tensor<f32>`, or a bare scalar.
    Fixed(Prim),

    /// `Self::Bank<R>` for the method's n-th REPR axis: a quantised bank,
    /// which is not a rectangle of elements and has no dtype to name. What
    /// the slot carries is the plane's own view of however many BYTE PLANES
    /// the repr stores one bank as — mxfp4 stores two, the E2M1 codes and
    /// the E8M0 block scales — so the slot occupies [`Repr::PLANES`] weight
    /// columns rather than one, and a dispatch's `Elem^axes` match never
    /// indexes it.
    Bank(usize),

    /// The slot carries a plane-side VIEW, not a rectangle of elements: the
    /// only dtype the table could name for it is the POOL's, decided when
    /// the slab was allocated and quantified over by no method here. A
    /// dispatch's `Elem^axes` match never indexes this slot.
    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prim {
    F32,

    I32,

    U32,

    Bool,
    /// A TENSOR element only, never a host scalar's run: the byte mask a
    /// selection writes and a selected attention reads.
    U8,
}

/// What a plane is, to a declaration: the payloads its marks carry.
pub trait Plane {
    type Tensor<T: Scalar>: Elem + ConstRun;

    /// What a `Const` slot carries for a QUANTISED BANK: the plane's own
    /// view of the byte planes the repr stores one bank as.
    ///
    /// NOT A SECOND `Elem`, and that is the whole reason this type exists
    /// beside [`Plane::Tensor`]. A `Tensor<T>` is a rectangle of `T`, so one
    /// address describes it and `Elem` names the arithmetic. An mxfp4 bank
    /// is two planes of bytes — 4-bit codes in one, a per-32-element E8M0
    /// exponent in the other — with no element type at all: the numbers only
    /// exist once a kernel has multiplied one plane by the other. A payload
    /// that carried it as `Tensor<u8>` would be describing the first plane
    /// and silently dropping the second, which is exactly what the `Const`
    /// slot on `matmul_select_bias` used to do.
    ///
    /// ONE ASSOCIATED TYPE PER REPR AXIS, quantified the way `Tensor` is
    /// quantified over `Scalar`. It is `ConstRun` and not `Elem` because a
    /// bank is only ever a `Const`: no fire mints one, no arena sizes one,
    /// and nothing reads a row out of one except the kernel that decodes it.
    type Bank<R: Repr>: ConstRun;

    /// What a `Cache` slot carries for a RECURRENT row: the plane's own
    /// view of the per-request slab pair — the conv window and the
    /// recurrent state — as one object, which is how every plane's ssm
    /// routine already takes it (`In<Struct<RecurrentState>>` on cuda).
    ///
    /// One associated type per POOL and not one per family.
    type Recurrent: Elem;

    /// What a `Cache` slot carries for a PAGED KV row: the plane's own view
    /// of the request's page table and the pool it indexes, as one object —
    /// how every plane's attention routine already takes it
    /// (`In<Struct<KvCache>>` on cuda).
    ///
    /// ONE ASSOCIATED TYPE PER POOL, and four families read this one:
    /// [`Mla`]'s latent pages, [`Index`]'s indexer keys, [`Pool`]'s
    /// compressed entries, and attention's own. A pool row's element type
    /// was chosen when the slab was allocated, which is why no method
    /// quantifies over it and every `Cache` slot's dtype column is
    /// [`Dtype::Opaque`].
    type Pages: Elem;
}

/// A scalar element: pointer-shaped on every plane. The bound the family
/// methods quantify over; `Tensor<T>` carries the plane's payload for it.
pub trait Scalar: Elem<Read = *const Self, Write = *mut Self> + Sized {}

impl<T: Elem<Read = *const T, Write = *mut T>> Scalar for T {}

/// A quantised bank's STORAGE FORM: what a `Const<Self::Bank<R>>` slot's
/// bytes mean.
///
/// A CLOSED MARKER SET AND NOT AN ENUM, on the [`Scalar`] precedent: a repr
/// has to be a TYPE so a method can quantify over it (`<R: Repr>`), so a
/// plane can carry one payload per repr (`Bank<R>`), and so a body can
/// branch on it at compile time. The enum this would otherwise be is
/// [`Form`], which exists for the one place a repr has to be a VALUE — the
/// data→type crossing a generated dispatch performs.
///
/// [`Scalar`] is open (every pointer-shaped element is one); this is closed,
/// because a repr is a decoding rule some kernel has to have been written
/// for. Today the set has one member. That is the point of the axis rather
/// than an argument against it: `matmul_select_bias`'s bank slot USED to be
/// `Const<Self::Tensor<T>>`, which pinned the point to one storage form and
/// to the activation's element at the same time; `<R: Repr>` unpins both, so
/// a caller whose bank ships at some other form is a member added here and
/// an arm added to the body, not a second declaration.
pub trait Repr: 'static {
    /// This repr, as the value a dispatch matches on.
    const FORM: Form;

    /// How many BYTE PLANES one bank of this repr is stored as, which is how
    /// many weight columns its `Const` slot occupies. The planes are bound in
    /// the order the repr states them, under the bank's own parameter name
    /// and that name plus the repr's suffix for each plane after the first.
    const PLANES: usize;
}

/// A [`Repr`] as a value: what a bound statement answers when a generated
/// dispatch asks which form the bank at a weight column is stored in.
///
/// The [`crate::bound::Axis`] of the repr axis, and the same crossing: an
/// element is decided by the arena that minted a rectangle, a repr by the
/// model text that declared the bank, and both have to be read off the fire
/// before a turbofish can be spelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Mxfp4,
}

/// OCP MX FP4: 4-bit E2M1 codes packed two to a byte, with one E8M0
/// exponent byte per 32 consecutive codes.
///
/// TWO PLANES, which is the whole reason `Bank` is not `Tensor`. A logical
/// `[.., N, K]` bank lands as `[.., N, K/32, 16]` bytes of codes and
/// `[.., N, K/32]` bytes of scale, and gpt-oss ships exactly those two
/// tensors (`*_blocks` and `*_scales`) rather than one.
pub enum Mxfp4 {}

impl Repr for Mxfp4 {
    const FORM: Form = Form::Mxfp4;
    const PLANES: usize = 2;
}

#[points]
pub trait Norm: Plane {
    fn rmsnorm<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, weight, eps, y);
        Err(Refusal::Absent {
            what: "norm.rmsnorm",
        })
    }

    /// Normalise each `head_dim`-wide slice of a row independently; the
    /// weight is one head wide. Stated, not derived: a `Const` weight
    /// carries no rectangle at the fire.
    fn rmsnorm_per_head<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, weight, head_dim, eps, y);
        Err(Refusal::Absent {
            what: "norm.rmsnorm_per_head",
        })
    }

    /// [`Norm::rmsnorm`] against a bank stored as an OFFSET: the scale is
    /// `1 + weight`, not `weight`.
    ///
    /// A SEPARATE POINT AND NOT A FLAG, because it is a fact about the
    /// CHECKPOINT and a text states one or the other for its whole life. The
    /// two conventions have been in this tree since Gemma — every plane's
    /// kernel carries the `WEIGHT_PLUS_ONE` template parameter that picks
    /// between them, and the legacy eDSL carries it as `NormVariant::Gemma`
    /// against `NormVariant::Plain` — and the declaration floor was the one
    /// place that had only the plain half. Qwen3.5 stores every norm this
    /// way except its gated out-norm; a text that says `rmsnorm` where the
    /// checkpoint says `1 + w` computes a different model and nothing
    /// refuses, which is exactly the failure a declaration is for.
    fn rmsnorm_plus_one<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, weight, eps, y);
        Err(Refusal::Absent {
            what: "norm.rmsnorm_plus_one",
        })
    }

    /// [`Norm::rmsnorm_per_head`] against an offset bank; see
    /// [`Norm::rmsnorm_plus_one`] for why the convention is a point and not
    /// a flag. Qwen3.5's `q_norm`/`k_norm` are both this.
    fn rmsnorm_per_head_plus_one<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, weight, head_dim, eps, y);
        Err(Refusal::Absent {
            what: "norm.rmsnorm_per_head_plus_one",
        })
    }

    fn rmsnorm_no_scale<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, head_dim, eps, y);
        Err(Refusal::Absent {
            what: "norm.rmsnorm_no_scale",
        })
    }

    /// `y = rmsnorm(x) * silu(gate)`, PER HEAD; the core arrives f32 from a
    /// recurrent mixer, the gate and the result ride the activation dtype.
    ///
    /// `head_dim` is STATED for [`Norm::rmsnorm_per_head`]'s reason, and the
    /// absence was a live bug rather than a tidiness point. A gated
    /// out-norm's weight is ONE head wide -- qwen's is `[value_head_dim]`
    /// against a `value_heads * value_head_dim` row -- and a `Const` weight
    /// carries an address with no rectangle at the fire, so a plane reading
    /// the width off its operands can only read the WHOLE row. That is what
    /// every plane did: it reduced sixteen heads into one mean of squares
    /// and walked `weight[i]` off the end of a 128-float buffer. The legacy
    /// eDSL carries the same number under the name `per_head`
    /// (`model-legacy/src/qwen_3_5/forward/mod.rs:383-391` records the same
    /// bug being fixed there), and both shader planes already take it as
    /// `vd: Const<i32>` on their `gated_rms` rows -- so this parameter is
    /// not new to the tree, only to the declaration that stands for them.
    fn rmsnorm_gated<T: Scalar>(
        &self,
        x: In<Self::Tensor<f32>>,
        gate: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<f32>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, gate, weight, head_dim, eps, y);
        Err(Refusal::Absent {
            what: "norm.rmsnorm_gated",
        })
    }

    /// KDA's per-head form: normalise each of `heads` heads of `x`, gate
    /// by `gate`. The head width is the row over `heads`.
    fn rmsnorm_gated_by<T: Scalar>(
        &self,
        x: In<Self::Tensor<f32>>,
        gate: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<f32>>,
        heads: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, gate, weight, heads, eps, y);
        Err(Refusal::Absent {
            what: "norm.rmsnorm_gated_by",
        })
    }

    fn residual_add<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        y: InOut<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, y);
        Err(Refusal::Absent {
            what: "norm.residual_add",
        })
    }

    fn add_bias<T: Scalar>(
        &self,
        bias: Const<Self::Tensor<T>>,
        out: InOut<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (bias, out);
        Err(Refusal::Absent {
            what: "norm.add_bias",
        })
    }

    /// Multiply by a host constant.
    fn mul_scalar<T: Scalar>(&self, s: f32, x: InOut<Self::Tensor<T>>) -> Result<(), Refusal> {
        let _ = (s, x);
        Err(Refusal::Absent {
            what: "norm.mul_scalar",
        })
    }

    /// Multiply by a learned `[1]` scalar living on the device (gemma's
    /// laurel scale). Unclaimed everywhere today — the measured gap.
    fn scale<T: Scalar>(
        &self,
        s: Const<Self::Tensor<T>>,
        x: InOut<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (s, x);
        Err(Refusal::Absent { what: "norm.scale" })
    }

    /// Kimi's residual blend: normalise the stack of earlier residual
    /// snapshots, project it, and add the result into the running stream.
    ///
    /// A NORM POINT AND NOT AN ATTENTION ONE. The cuda routine is filed
    /// under `attn/` and spelled `attn_res_blend`, which is where kimi's
    /// author put it and not a claim about the family: nothing here reads a
    /// query, a page or a log-sum-exp. What it reads is the residual stream
    /// and what it writes is the residual stream, through an rmsnorm and a
    /// projection — `norm.residual_add` with a normalise and a matmul fused
    /// into it, sitting beside the other residual-stream arithmetic.
    ///
    /// THE VARIADIC LEDGER ITEM, and it is why this point is claim-only on
    /// every plane. The text states ONE value per earlier block and the
    /// count grows with the layer — `blocks: &[Value]` in the builder — so
    /// the statement's arity is a function of where in the model it stands.
    /// A point's arity IS its slot list, so the slot list states the ONE
    /// concatenated rectangle the routine takes, which is what the operands
    /// are once the arena has laid them out end to end. Until a `Vararg`
    /// mark exists, that gap is stated here and bridged by nothing: the
    /// routine keeps its own `canon` and the point resolves through it.
    fn res_blend<T: Scalar>(
        &self,
        prefix: In<Self::Tensor<T>>,
        blocks: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        proj: Const<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (prefix, blocks, weight, eps, proj, y);
        Err(Refusal::Absent {
            what: "norm.res_blend",
        })
    }
}

/// The gated activation family. Every point but `geglu_tanh` reads ONE
/// packed `[gate | up]` row and writes a row half as wide — the first
/// declared family whose `Out` is not shaped like its first `In`. The
/// intermediate width is therefore STATED rather than derived: the
/// `intermediate` scalar IS the output row, and a `#[shape]` rule is the
/// later consumer of it. The table below records the slots; nothing here
/// computes a rectangle.
#[points]
pub trait Mlp: Plane {
    /// `y = silu(gate) * up` over a packed `[gate | up]` row.
    fn swiglu<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (packed, intermediate, y);
        Err(Refusal::Absent { what: "mlp.swiglu" })
    }

    /// Swiglu with both halves clamped to `limit` before the gate.
    fn swiglu_clamp<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        limit: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (packed, intermediate, limit, y);
        Err(Refusal::Absent {
            what: "mlp.swiglu_clamp",
        })
    }

    /// Clamped swiglu whose sigmoid carries a stated `alpha` (gpt-oss).
    /// Unclaimed on every plane today: cuda's `gpt_oss_glu` takes gate and
    /// up as two rows, and no kernel reads the packed form the text states.
    /// The gap is a measured row under this name.
    fn swiglu_clamp_alpha<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (packed, intermediate, limit, alpha, y);
        Err(Refusal::Absent {
            what: "mlp.swiglu_clamp_alpha",
        })
    }

    /// The unpacked geglu: gate and up arrive as two rows, so the result is
    /// shaped like the first `In` after all — the one point of this family
    /// that states no intermediate.
    fn geglu_tanh<T: Scalar>(
        &self,
        gate: In<Self::Tensor<T>>,
        up: In<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (gate, up, y);
        Err(Refusal::Absent {
            what: "mlp.geglu_tanh",
        })
    }

    /// The same activation over one packed `[gate | up]` row.
    fn geglu_tanh_packed<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (packed, intermediate, y);
        Err(Refusal::Absent {
            what: "mlp.geglu_tanh_packed",
        })
    }

    /// Kimi's gated form: a `beta`-scaled silu gate and a linear term
    /// capped at `up_cap`. The cap is optional in the text and rides a
    /// `0.0` sentinel here, which is the encoding the statement has always
    /// carried — an open ledger item, not a decision this family makes.
    fn situ<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        beta: f32,
        up_cap: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (packed, intermediate, beta, up_cap, y);
        Err(Refusal::Absent { what: "mlp.situ" })
    }
}

/// The `Gemm` family: an activation against a model weight, which is most of
/// what a forward pass is. Three points and one arithmetic — `lm_head` and
/// `attention_landing` are the same matmul wearing a purpose, and a plane
/// claims them by an explicit one-line override that calls its own `matmul`.
///
/// That override is the whole of what the retired `canon::DEFAULTS` table
/// did, said where it can be read: a delegation IS a claim, it lands in the
/// plane's claim table, and resolution sees it without a second walk.
#[points]
pub trait Gemm: Plane {
    /// `y = act @ wᵀ`. The result is the activation's rows by the weight's
    /// output width — stated by neither operand, since a `Const` weight
    /// carries an address and no rectangle, so the plane reads it off `y`.
    fn matmul<T: Scalar>(
        &self,
        act: In<Self::Tensor<T>>,
        w: Const<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (act, w, y);
        Err(Refusal::Absent {
            what: "gemm.matmul",
        })
    }

    /// The vocabulary projection that closes a text. The same arithmetic; a
    /// plane may answer it with a wider accumulator or a sharded reduction,
    /// which is why it is a point of its own rather than a call site of
    /// `matmul`.
    fn lm_head<T: Scalar>(
        &self,
        act: In<Self::Tensor<T>>,
        w: Const<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (act, w, y);
        Err(Refusal::Absent {
            what: "gemm.lm_head",
        })
    }

    /// Attention's output projection. `layer` says WHICH attention landed
    /// here — the driver finds a layer's attention output by it — and the
    /// arithmetic never reads it: a plane answering with a plain matmul
    /// ignores the scalar, and the statement's own layer tag is what the
    /// driver actually reads.
    fn attention_landing<T: Scalar>(
        &self,
        act: In<Self::Tensor<T>>,
        w: Const<Self::Tensor<T>>,
        layer: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (act, w, layer, y);
        Err(Refusal::Absent {
            what: "gemm.attention_landing",
        })
    }
}

/// The `Dist` family: what one tensor-parallel rank says to its peers. One
/// point today — the shard sum that closes a split projection.
#[points]
pub trait Dist: Plane {
    /// Sum `buf` across the tensor-parallel group, in place: every rank
    /// leaves holding the whole row it entered holding a shard of.
    fn all_reduce<T: Scalar>(&self, buf: InOut<Self::Tensor<T>>) -> Result<(), Refusal> {
        let _ = buf;
        Err(Refusal::Absent {
            what: "dist.all_reduce",
        })
    }
}

/// Rotary position embedding. Every point here rotates its operands IN
/// PLACE, which is what `InOut` on `q` and `k` says and why a statement's
/// results are the operands it rotated. `positions` is the token's absolute
/// index: a fixed-dtype slot, because a position is an `i32` on every plane
/// and no axis of these methods quantifies over it.
///
/// Stated, not derived: `head_dim`, `rotary_dim` and `interleaved`. An
/// operand's rectangle is the ROW — `heads * head_dim` wide — so a head
/// COUNT follows from the row and the stated head width, but the head width
/// itself is nowhere in the operands to read. `interleaved` is not geometry
/// and is not a convention this family may fix either: false pairs `d` with
/// `d + rotary_dim / 2` (NeoX), true pairs `2d` with `2d + 1` (GPT-J), and
/// the texts disagree — gpt-oss's YaRN rotation is NeoX, deepseek-v4's
/// trailing rotation is not. A point that fixed it would be right for one
/// checkpoint and silently wrong for the other.
#[points]
pub trait Rope: Plane {
    /// The whole head rotates: `rotary_dim` IS `head_dim`, so it is not
    /// stated twice.
    ///
    /// Named `full`, not `rope`: a method's path is its point's name, and
    /// `Rope::rope` would state `"rope.rope"`. The bare `rope` point this
    /// replaces always meant the whole-head rotation, and `"rope.full"` is
    /// that meaning spelled.
    fn full<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        k: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        let _ = (q, k, positions, head_dim, theta, interleaved);
        Err(Refusal::Absent { what: "rope.full" })
    }

    /// The head's LEADING `rotary_dim` channels rotate; the tail passes
    /// through unrotated. NeoX pairing throughout — the kernel that rotates
    /// a leading slice branches on nothing, so nothing is stated.
    fn partial<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        k: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        let _ = (q, k, positions, rotary_dim, head_dim, theta);
        Err(Refusal::Absent {
            what: "rope.partial",
        })
    }

    /// [`Rope::partial`] with no `k`: the text rotates a `q` it has already
    /// split away from its key.
    fn partial_q<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        let _ = (q, positions, rotary_dim, head_dim, theta);
        Err(Refusal::Absent {
            what: "rope.partial_q",
        })
    }

    /// The head's TRAILING `rotary_dim` channels rotate — the leading
    /// `head_dim - rotary_dim` are the nope half of an MLA head and pass
    /// through. Different arithmetic from [`Rope::partial_q`], not a
    /// re-parameterisation of it.
    fn partial_last<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        let _ = (q, positions, rotary_dim, head_dim, theta, interleaved);
        Err(Refusal::Absent {
            what: "rope.partial_last",
        })
    }

    /// The whole head rotates on YaRN-interpolated frequencies. The
    /// checkpoint's YaRN block arrives flattened: a builder mirrors the
    /// declaration one parameter at a time, and a struct is not a slot.
    fn yarn<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        k: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
    ) -> Result<(), Refusal> {
        let _ = (
            q,
            k,
            positions,
            head_dim,
            theta,
            factor,
            beta_fast,
            beta_slow,
            attention_factor,
            original_max_position,
            interleaved,
        );
        Err(Refusal::Absent { what: "rope.yarn" })
    }
}

/// The mixture-of-experts family: the router that chooses experts, the
/// expert-bank matmuls a route selects, and the two combines that fold the
/// experts back into one row.
///
/// A router states TWO results, never one. `routes` names the chosen
/// experts and `weights` says how much each one counts — `matmul_select`
/// reads the first, `weighted_sum` the second, and a single value standing
/// for both would be a fiction no plane's kernel writes.
///
/// STATED GEOMETRY. `experts` and `top_k` are the router's own two numbers.
/// `top_k` sizes both results, and an `Out` is allocated FROM the statement
/// rather than read for it; `experts` is the fan the checkpoint declares,
/// and every shader plane's entrypoint asks for it by name.
#[points]
pub trait Moe: Plane {
    /// Softmax over the router logits, then the top `top_k` of them.
    fn topk_softmax<T: Scalar>(
        &self,
        logits: In<Self::Tensor<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<Self::Tensor<i32>>,
        weights: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (logits, experts, top_k, routes, weights);
        Err(Refusal::Absent {
            what: "moe.topk_softmax",
        })
    }

    /// Sigmoid over the logits, top `top_k`, then `scaling` — and, when
    /// `renormalize`, the kept weights rescaled to sum to one.
    fn topk_sigmoid<T: Scalar>(
        &self,
        logits: In<Self::Tensor<T>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Self::Tensor<i32>>,
        weights: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            logits,
            experts,
            top_k,
            renormalize,
            scaling,
            routes,
            weights,
        );
        Err(Refusal::Absent {
            what: "moe.topk_sigmoid",
        })
    }

    /// `sqrt(softplus(x))` scoring with a learned per-expert correction
    /// bias added before the selection — deepseek's router.
    fn topk_sqrt_softplus<T: Scalar>(
        &self,
        logits: In<Self::Tensor<T>>,
        bias: Const<Self::Tensor<f32>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Self::Tensor<i32>>,
        weights: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            logits,
            bias,
            experts,
            top_k,
            renormalize,
            scaling,
            routes,
            weights,
        );
        Err(Refusal::Absent {
            what: "moe.topk_sqrt_softplus",
        })
    }

    /// `y[r] = x[r] @ bank[routes[r]]`: one matmul per route, against the
    /// expert that route names.
    ///
    /// `bank` is the `[E, N, K]` stack and wears `Const` because that is
    /// the mark a weight wears.
    ///
    /// A DENSE BANK ON PURPOSE, and not the [`Plane::Bank`] payload its
    /// biased sibling below now carries. A dense expert stack IS a rectangle
    /// of elements, and `Self::Tensor<T>` says the one thing about it that
    /// matters and that a `Bank<R>` could not: the bank rides the SAME
    /// element as the activation. qwen3.5-a3b and dsv4 both state this point
    /// against bf16 stacks and both mean exactly that tie. A caller whose
    /// unbiased bank is quantised wants a `Bank<R>` slot — which is a second
    /// point beside this one, not a widening of it, because it would be
    /// dropping the tie.
    ///
    /// (kimi-k3 declares its stacks at the `mxfp4` repr on the model side
    /// while its import table copies dense `w1`/`w3` rows into them. That
    /// disagreement is the checkpoint-unverified debt baker-todo records for
    /// kimi/glm/dsv4, and it is a MODEL-side lie about a dense declaration,
    /// not a gap in this one.)
    fn matmul_select<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        bank: Const<Self::Tensor<T>>,
        routes: In<Self::Tensor<i32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, bank, routes, y);
        Err(Refusal::Absent {
            what: "moe.matmul_select",
        })
    }

    /// `matmul_select` with the expert's own bias row added to the result —
    /// gptoss's banks, the only ones that carry one.
    ///
    /// THE ONE POINT WITH TWO KINDS OF AXIS, and the two are different
    /// questions about the same statement. `T` is what the ACTIVATION and
    /// the BIAS ride — one element, checked equal at every instantiation,
    /// which is what makes the bias add a bias add. `R` is what the BANK is
    /// STORED as, which the activation's element says nothing about: gpt-oss
    /// hands this point a bf16 row, a bf16 bias and an mxfp4 stack, and no
    /// single `T` can spell that trio.
    ///
    /// The slot used to be `Const<Self::Tensor<T>>` with the debt written in
    /// prose; `Const<Self::Bank<R>>` is the debt paid. What changed is not
    /// the arity a text writes — a text still names ONE bank — but what the
    /// declaration ADMITS about it: a bank is however many byte planes its
    /// repr stores, and the weight columns follow [`Repr::PLANES`].
    fn matmul_select_bias<T: Scalar, R: Repr>(
        &self,
        x: In<Self::Tensor<T>>,
        bank: Const<Self::Bank<R>>,
        bias: Const<Self::Tensor<T>>,
        routes: In<Self::Tensor<i32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, bank, bias, routes, y);
        Err(Refusal::Absent {
            what: "moe.matmul_select_bias",
        })
    }

    /// Fold a token's `top_k` expert rows back into one, weighted by the
    /// router's own weights.
    fn weighted_sum<T: Scalar>(
        &self,
        routed: In<Self::Tensor<T>>,
        weights: In<Self::Tensor<f32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (routed, weights, y);
        Err(Refusal::Absent {
            what: "moe.weighted_sum",
        })
    }

    /// `y = routed + shared * sigmoid(gate)`: the shared expert joining the
    /// routed sum through its own learned gate, which the statement hands
    /// over as the `[tokens, 1]` column it already computed.
    fn sigmoid_gate_add<T: Scalar>(
        &self,
        routed: In<Self::Tensor<T>>,
        shared: In<Self::Tensor<T>>,
        gate: In<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (routed, shared, gate, y);
        Err(Refusal::Absent {
            what: "moe.sigmoid_gate_add",
        })
    }
}

/// Output gating: a mixer's result scaled by the sigmoid of its own gate
/// column. Not an MoE combine — no expert route comes near it — and its own
/// family for that reason. The trait IS the name, and every plane files
/// this kernel beside its attention, never beside its experts.
#[points]
pub trait Gate: Plane {
    /// `x *= sigmoid(gate)`, elementwise over the whole rectangle.
    fn sigmoid_mul<T: Scalar>(
        &self,
        x: InOut<Self::Tensor<T>>,
        gate: In<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, gate);
        Err(Refusal::Absent {
            what: "gate.sigmoid_mul",
        })
    }
}

/// The `Layout` family: what a text does to a rectangle's SHAPE, never to
/// the numbers inside it. One gather that turns token ids into rows, three
/// cuts that take a packed projection apart, and the layer slice of a
/// relayed table. No arithmetic anywhere in the family.
///
/// STATED GEOMETRY throughout. A cut's halves are allocated FROM the
/// statement, so every width here is the statement's own number:
/// `q_width`/`kv_width` size `split_qkv`'s three results, `head_dim` is the
/// pitch `split_q_gate` grids by, `width` is where `split_rows` divides. A
/// plane's kernel reads those same numbers back off the rectangles it was
/// handed, which is why a claim may state a width and never look at it —
/// the same shape [`Moe`]'s `experts` and `top_k` take.
#[points]
pub trait Layout: Plane {
    /// Gather one row of `table` per token id.
    ///
    /// `ids` is fixed at `i32`: a token id is an `i32` on every plane, and
    /// no axis of this method quantifies over it. The table and the rows it
    /// yields ride the activation dtype.
    ///
    /// THE SHAPE NO CONVENTION HERE COVERS. The result is `ids`' ROWS by the
    /// TABLE's width, which is neither "like the first `In`" nor anything
    /// else these declarations assume — and a `Const` table carries an
    /// address and no rectangle, so the width is not in the operands to read
    /// at all. The statement allocates the result and the plane reads the
    /// width back off it. This is the family's `#[shape]` row when that
    /// annotation lands, the way [`Mlp`]'s stated intermediate is.
    ///
    /// STATED GEOMETRY: `vocab`, the table's ROW count, for the reason the
    /// width above is READ. The width is on the result, which the statement
    /// allocated; the row count is on nothing at all — a `Const` table is an
    /// address, `ids` carries token values and not their range, and no
    /// result is sized from it. A gather that clamps its index needs that
    /// bound, and a plane that invented one (`i32::MAX`) would be retiring
    /// the clamp rather than answering it, so the number comes from the text
    /// that already knows it. This is `rmsnorm_per_head`'s law reaching the
    /// one geometry that guards a READ rather than sizing a write.
    fn embed<T: Scalar>(
        &self,
        ids: In<Self::Tensor<i32>>,
        table: Const<Self::Tensor<T>>,
        vocab: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (ids, table, vocab, y);
        Err(Refusal::Absent {
            what: "layout.embed",
        })
    }

    /// Cut one packed `[q | k | v]` projection into its three parts. THREE
    /// results, never one: a statement that named the packed row and let its
    /// consumers index into it would hand every plane a stride rule none of
    /// their kernels write.
    fn split_qkv<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        q_width: u32,
        kv_width: u32,
        q: Out<Self::Tensor<T>>,
        k: Out<Self::Tensor<T>>,
        v: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (packed, q_width, kv_width, q, k, v);
        Err(Refusal::Absent {
            what: "layout.split_qkv",
        })
    }

    /// Cut an INTERLEAVED per-head `[query | gate]` row into its two halves.
    /// The interleaving is per head, so `head_dim` is the pitch the cut
    /// walks rather than a width either half carries — the halves come out
    /// the same width as each other, each half the packed row.
    fn split_q_gate<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        head_dim: u32,
        q: Out<Self::Tensor<T>>,
        gate: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (packed, head_dim, q, gate);
        Err(Refusal::Absent {
            what: "layout.split_q_gate",
        })
    }

    /// Cut a row in two at `width`: the leading `width` channels, then the
    /// rest. The plain two-way divide the interleaved cuts above are not.
    fn split_rows<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        width: u32,
        left: Out<Self::Tensor<T>>,
        right: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, width, left, right);
        Err(Refusal::Absent {
            what: "layout.split_rows",
        })
    }

    /// One layer's slice of a relayed table — gemma's per-layer PLE plane,
    /// picked out of the stack the prologue gathered once. The relay is
    /// `[rows, layers * width]` and the slice is `[rows, width]` at column
    /// `layer * width`: a strided per-row copy.
    ///
    /// STATED GEOMETRY: `width`, for [`Norm::rmsnorm_per_head`]'s reason
    /// turned inside out. There the width was missing from a `Const`
    /// weight; here it is missing from the OPERAND — `layer` says WHICH
    /// slice, never how many there are, so the packed row divides by a
    /// number no slot carries and the walk cannot size the result. Gemma's
    /// text holds that number already (`Ple::dim`), and the whole content of
    /// the stated-geometry law is that the statement says what its results
    /// are sized from.
    ///
    /// The arithmetic is a base and an offset, so a plane may one day answer
    /// this at BINDING with a view over the relayed rows and never launch at
    /// all. Cuda answers it with a copy, which is the honest form until a
    /// binder can state the aliasing.
    fn select<T: Scalar>(
        &self,
        table: In<Self::Tensor<T>>,
        layer: u32,
        width: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (table, layer, width, y);
        Err(Refusal::Absent {
            what: "layout.select",
        })
    }
}

/// The recurrent mixers: a depthwise causal convolution over a rolling
/// window, the gated-delta rule that carries qwen's linear attention, and
/// kimi's KDA rule. The first family whose statements name a CACHE ROW —
/// every point here reads a per-request recurrent slab and leaves the next
/// state in it, which is what `Cache<Self::Recurrent>` says and why this
/// family is where that mark had to be invented.
///
/// TWO READINGS OF EVERY RULE, and they are different arithmetic rather
/// than one kernel under two launches. A `_chunked` point runs a prefill
/// WINDOW: it takes the fire's `qo_indptr` beside its rows — the ragged CSR
/// that says where one request's tokens end and the next begin — and the
/// plain form runs the one-token step, whose boundaries are the rows
/// themselves. The CSR is an ordinary `In<Self::Tensor<i32>>`: a device
/// buffer the runtime stages per fire, not a pool row and not a weight.
///
/// STATED GEOMETRY: `conv_width`. A conv's kernel width lives in the
/// `[channels, width]` weight and nowhere else, and a `Const` weight
/// carries an address and no rectangle at the fire — the same reason
/// `norm.rmsnorm_per_head` states its head width. The channel count is NOT
/// stated, because it is the operand's own row.
#[points]
pub trait Ssm: Plane {
    /// Depthwise causal conv with a fused SiLU, one token per request: the
    /// slot's rolling window shifts by one and the new column lands in it.
    fn causal_conv1d<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        state: Cache<Self::Recurrent>,
        conv_width: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, weight, state, conv_width, y);
        Err(Refusal::Absent {
            what: "ssm.causal_conv1d",
        })
    }

    /// [`Ssm::causal_conv1d`] over a prefill window. `indptr` is the fire's
    /// query CSR — `[requests + 1]` boundaries into the token rows — and
    /// the window's tail is what lands in the slot.
    fn causal_conv1d_chunked<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        weight: Const<Self::Tensor<T>>,
        state: Cache<Self::Recurrent>,
        conv_width: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, indptr, weight, state, conv_width, y);
        Err(Refusal::Absent {
            what: "ssm.causal_conv1d_chunked",
        })
    }

    /// Qwen's gated-delta prologue: the `[a | b]` projection becomes the
    /// rule's decay and beta columns, through the checkpoint's `dt_bias`
    /// and `a_log`. The result rides f32 — a decay is accumulated, not
    /// activated — and the mixer downstream reads it as one row.
    ///
    /// `a_log` IS f32 AND `dt_bias` IS NOT, and the split is the kernel's
    /// rather than a tidiness point. `ssm/gated_delta_net_prep.cuh` spells
    /// the two apart in one argument list — `const float* __restrict__
    /// A_log` beside `const T* __restrict__ dt_bias` — and its own header
    /// says why: *"HF Qwen3.5 stores `A_log` and the RMSNormGated weight in
    /// fp32 (matches the FLA fast-path expectation), even when the rest of
    /// the model is bf16. dt_bias stays bf16."* The launch that answered
    /// this point, `qwen_gdn_post_conv_prep_bf16`, has taken `a_log:
    /// Const<Tensor<f32>>` beside `dt_bias: Const<Tensor<bf16>>` the whole
    /// time. THIS DECLARATION SAID `T` FOR BOTH, which is the one slot
    /// where the floor disagreed with the plane claiming it, and the
    /// shipped 35B-A3B checkpoint settles it from the third side: `A_log`
    /// is `F32 [value_heads]` and `dt_bias` is `BF16 [value_heads]` in the
    /// same BF16 file. Handing that `float*` a bf16 bank is not a cast — it is
    /// half the decays per head, read as nonsense.
    fn gdn_prep<T: Scalar>(
        &self,
        ba: In<Self::Tensor<T>>,
        dt_bias: Const<Self::Tensor<T>>,
        a_log: Const<Self::Tensor<f32>>,
        gates: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (ba, dt_bias, a_log, gates);
        Err(Refusal::Absent {
            what: "ssm.gdn_prep",
        })
    }

    /// The gated-delta rule for one token per request, against the slot's
    /// `[k_heads, k_dim, v_dim]` state.
    ///
    /// The four head numbers are STATED because a GQA rule cannot read them
    /// off its rows: `qkv` arrives as one packed row and the key heads that
    /// divide it are the checkpoint's, not the rectangle's.
    fn gated_delta<T: Scalar>(
        &self,
        qkv: In<Self::Tensor<T>>,
        z: In<Self::Tensor<T>>,
        gates: In<Self::Tensor<f32>>,
        state: Cache<Self::Recurrent>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (qkv, z, gates, state, k_heads, v_heads, k_dim, v_dim, y);
        Err(Refusal::Absent {
            what: "ssm.gated_delta",
        })
    }

    /// [`Ssm::gated_delta`] over a prefill window, `indptr` the fire's query
    /// CSR. The chunked rule is not the step in a loop — it blocks the
    /// window and folds the blocks — which is why it is a point of its own.
    fn gated_delta_chunked<T: Scalar>(
        &self,
        qkv: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        z: In<Self::Tensor<T>>,
        gates: In<Self::Tensor<f32>>,
        state: Cache<Self::Recurrent>,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            qkv, indptr, z, gates, state, k_heads, v_heads, k_dim, v_dim, y,
        );
        Err(Refusal::Absent {
            what: "ssm.gated_delta_chunked",
        })
    }

    /// Kimi's KDA rule for one token per request. `f` and `b` are the
    /// forget and beta projections the text hands over; `norm_eps` is the
    /// rule's own internal normalisation, fused rather than stated as a
    /// separate `norm` point — which is what every KDA kernel does.
    ///
    /// `mixed` IS THE PACKED `[q | k | v]` PLANE, three `heads * head_dim`
    /// slices of one post-convolution row, and `heads`/`head_dim` are what
    /// divide it. The text convolves the packed row once
    /// (`model/src/kimi_k3/forward.rs`) where the legacy ran three separate
    /// projections and three separate convs, so the cut is arithmetic this
    /// point owns and no upstream statement carries.
    ///
    /// BOTH DECAY WEIGHTS ARE F32 AND NEITHER RIDES `T`. `ssm/kda.cuh`
    /// spells `kda_gate_beta`'s pair `const float* __restrict__ A_log`
    /// beside `const float* __restrict__ dt_bias` — the same argument list,
    /// both `float` — and the routine claiming that launch has taken
    /// `Const<Tensor<f32>>` for both since it was written. That is a
    /// DIFFERENT answer from [`Ssm::gdn_prep`]'s one line up, where the
    /// kernel takes `A_log` float beside `dt_bias` at `T`, and the
    /// asymmetry is each kernel's own rather than a house style.
    ///
    /// `dt_bias` is `[heads, head_dim]` and `a_log` is `[heads]`, which the
    /// kernel also settles: it reads `dt_bias[h * D + d]` per channel and
    /// `A_log[h]` per head. KDA's forget gate is CHANNEL-wise where the
    /// gated-delta rule's is head-wise, and the two shapes are that
    /// difference.
    fn kda_step<T: Scalar>(
        &self,
        mixed: In<Self::Tensor<T>>,
        f: In<Self::Tensor<T>>,
        b: In<Self::Tensor<T>>,
        dt_bias: Const<Self::Tensor<f32>>,
        a_log: Const<Self::Tensor<f32>>,
        state: Cache<Self::Recurrent>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            mixed, f, b, dt_bias, a_log, state, heads, head_dim, norm_eps, y,
        );
        Err(Refusal::Absent {
            what: "ssm.kda_step",
        })
    }

    /// [`Ssm::kda_step`] over a prefill window, `indptr` the fire's query
    /// CSR.
    fn kda_chunked<T: Scalar>(
        &self,
        mixed: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        f: In<Self::Tensor<T>>,
        b: In<Self::Tensor<T>>,
        dt_bias: Const<Self::Tensor<f32>>,
        a_log: Const<Self::Tensor<f32>>,
        state: Cache<Self::Recurrent>,
        heads: u32,
        head_dim: u32,
        norm_eps: f32,
        y: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            mixed, indptr, f, b, dt_bias, a_log, state, heads, head_dim, norm_eps, y,
        );
        Err(Refusal::Absent {
            what: "ssm.kda_chunked",
        })
    }
}

/// Paged attention: the family the migration ends on, and the one whose
/// statements name the KV POOL. Every reading here reads a request's page
/// table and the slab it indexes — one `Cache<Self::Pages>` slot, bound by
/// the cache pool exactly as `Cache<Self::Recurrent>` is, and named by the
/// statement's `.cache(&pages.name)` reference.
///
/// TWO READINGS OF THE SAME ARITHMETIC, and they are different kernels
/// rather than one under two launches — the same split the recurrent family
/// draws. A `decode` form runs ONE query row per request and its boundaries
/// are the rows themselves; a `prefill` form runs a WINDOW and takes the
/// fire's query CSR beside its rows, an ordinary `In<Self::Tensor<i32>>`:
/// a device buffer the runtime stages per fire, not a pool row and not a
/// weight (the `ssm.*_chunked` precedent). `masked` is the prefill window
/// under a custom `(q, kv)` mask the plane stages.
///
/// THE `_lse` FORMS ARE SEPARATE POINTS and not a flag: a statement that
/// wants the log-sum-exp states two results, and two `Out` slots is what
/// that is. Folding them into one point behind an `Option` would put a
/// result's EXISTENCE on the params run, where no shape walk can read it.
///
/// STATED GEOMETRY. `head_dim` is stated because a packed
/// `[heads * head_dim]` query row cannot be divided by reading it;
/// `kv_heads` is stated on the prefill forms for the GQA reason
/// `ssm.gated_delta` states its four head numbers — the key heads are the
/// checkpoint's, not the rectangle's. `window` is the sliding extent, and
/// ZERO IS NO WINDOW: that is precisely what the DSL's `.window()` verb has
/// always recorded (`self.int(w.unwrap_or(0))`), so the declaration states
/// the `u32` the plan carries rather than an `Option` no params run holds.
///
/// WHAT IS NOT HERE is the plane's staging, and on this family that is most
/// of the fa2 core: the decode and prefill plan caches, the host mirrors of
/// the two CSRs, the mask view, the fire's write origin. A body pulls those
/// from `self`; a declaration that stated them would be describing one
/// plane's plan vocabulary as though every plane had it.
///
/// "PULLS THOSE FROM `self`" IS A MECHANISM AND NOT A HOPE, which is what
/// kept six of these points on the default body for as long as it was only a
/// sentence. Every such object is declared with a KEY
/// (`kernels::raises::Raise`), an executor answers keys
/// (`kernels::raises::Answered`), and a plane's context is where the two
/// meet. The declarations below did not move when cuda claimed all eleven —
/// that is the test of whether the split was drawn in the right place.
#[points]
pub trait Attention: Plane {
    /// One query row per request, against the pool row's pages.
    fn decode<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q, pages, window, head_dim, sm_scale, o);
        Err(Refusal::Absent {
            what: "attention.decode",
        })
    }

    /// [`Attention::decode`] over a prefill window. `indptr` is the fire's
    /// query CSR — `[requests + 1]` boundaries into the token rows.
    fn prefill<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q, indptr, pages, window, head_dim, kv_heads, sm_scale, o);
        Err(Refusal::Absent {
            what: "attention.prefill",
        })
    }

    /// The prefill window under a CUSTOM mask — one byte per `(q, kv)`
    /// pair, with its own CSR. The mask is the plane's own staging and
    /// appears in no slot: what makes this a point of its own is that the
    /// text states a different arithmetic, not that it hands over a buffer.
    fn masked<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q, indptr, pages, window, head_dim, sm_scale, o);
        Err(Refusal::Absent {
            what: "attention.masked",
        })
    }

    /// [`Attention::decode`], also leaving the per-row log-sum-exp — the
    /// normaliser a second attention's output is merged against, and what
    /// a sink correction rescales by.
    ///
    /// THE LSE RIDES BASE TWO, and this is the sentence every other lse
    /// slot on the floor points at. A softmax kernel does not compute
    /// `exp`: it folds `log2(e)` into the scale once and runs the whole
    /// row on `exp2`, because that is the instruction the hardware has. So
    /// `m + log2(d)` is what an attention reading HAS at the end, and
    /// every base a floor could name costs somebody a launch to produce
    /// except that one. A plane whose kernel accumulates in another base
    /// converts before it writes this slot; a consumer reads log2 and says
    /// so where it meets a quantity that is not — which is
    /// [`Attention::sink`], and only there.
    fn decode_lse<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
        lse: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (q, pages, window, head_dim, sm_scale, o, lse);
        Err(Refusal::Absent {
            what: "attention.decode_lse",
        })
    }

    /// [`Attention::prefill`], also leaving the per-row log-sum-exp, in
    /// [`Attention::decode_lse`]'s base.
    fn prefill_lse<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        kv_heads: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
        lse: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            q, indptr, pages, window, head_dim, kv_heads, sm_scale, o, lse,
        );
        Err(Refusal::Absent {
            what: "attention.prefill_lse",
        })
    }

    /// The attention-sink correction: fold a learned per-head logit into
    /// the softmax after the fact, by rescaling the output against the
    /// `lse` an `_lse` reading left — `o *= sigmoid(lse·ln2 − sink)`, which
    /// is the softmax a virtual zero-valued key at logit `sink[h]` would
    /// have produced.
    ///
    /// THE ONE POINT WHERE TWO BASES MEET, and the whole reason it states
    /// which. The lse arrives in base two ([`Attention::decode_lse`]); the
    /// sink is a LOGIT OUT OF A CHECKPOINT and a checkpoint's logits are
    /// natural — gpt-oss ships `self_attn.sinks` as the `exp(sink)` term
    /// of an `exp` denominator. Neither fact is anybody's to move, so the
    /// subtraction rebases, and this declaration is where that is written
    /// down instead of being a factor of 0.693 somebody's greedy decode
    /// drifts by (`kernels-cuda/kernels/attn/attn_sink.cuh` records that
    /// bug being found and fixed once already).
    ///
    /// THE SINK RIDES `T` BECAUSE THE CHECKPOINT DOES. It is a per-head
    /// weight, not accumulator state: gpt-oss ships `[64]` at BF16 and
    /// dsv4's import copies `attn.sinks` through at the model's element,
    /// so a slot pinned to f32 would be a declaration no text could state
    /// without lying about its own bytes. The lse beside it stays f32 —
    /// that one really is accumulator state, minted by the fire.
    fn sink<T: Scalar>(
        &self,
        o: InOut<Self::Tensor<T>>,
        lse: In<Self::Tensor<f32>>,
        sink: Const<Self::Tensor<T>>,
        head_dim: u32,
    ) -> Result<(), Refusal> {
        let _ = (o, lse, sink, head_dim);
        Err(Refusal::Absent {
            what: "attention.sink",
        })
    }

    /// Merge two attentions over disjoint key sets into one, by their
    /// log-sum-exps: `(o1, lse1)` and `(o2, lse2)` become the pair a single
    /// softmax over the union would have produced. The merged lse is a
    /// result and not scratch — a third merge reads it.
    ///
    /// BASE-AGNOSTIC ARITHMETIC AT A STATED BASE. The weights this folds by
    /// are RATIOS — `b^(l1 - m)` against `b^(l2 - m)` — so the merge is the
    /// same number in any base, and the base is stated all the same because
    /// the lse it leaves is one an [`Attention::sink`] downstream reads.
    /// Both operands and the result ride [`Attention::decode_lse`]'s base.
    fn merge_lse<T: Scalar>(
        &self,
        o1: In<Self::Tensor<T>>,
        lse1: In<Self::Tensor<f32>>,
        o2: In<Self::Tensor<T>>,
        lse2: In<Self::Tensor<f32>>,
        heads: u32,
        head_dim: u32,
        o: Out<Self::Tensor<T>>,
        lse: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (o1, lse1, o2, lse2, heads, head_dim, o, lse);
        Err(Refusal::Absent {
            what: "attention.merge_lse",
        })
    }

    /// `x = cap * tanh(x / cap)`, in place.
    ///
    /// FILED BY ITS CLAIM AND NOT BY ITS OPERAND. Gemma states this on its
    /// FINAL logits, where the row is a vocabulary and not a score matrix,
    /// so a reader could fairly ask for `gemm.logit_softcap` beside
    /// `gemm.lm_head`. It is here because the kernel is
    /// `::pie::attn::logit_softcap` and because the same cap rides the
    /// attention dispatches above as `logits_soft_cap` — one soft cap, one
    /// family, one place to look when a plane does not answer it.
    fn logit_softcap<T: Scalar>(&self, x: InOut<Self::Tensor<T>>, cap: f32) -> Result<(), Refusal> {
        let _ = (x, cap);
        Err(Refusal::Absent {
            what: "attention.logit_softcap",
        })
    }

    /// Write this fire's keys and values into the pool row's pages.
    ///
    /// AN EFFECT AND NOT A RESULT: the statement names a cache row and
    /// leaves the fire's rows in it, and there is no rectangle for it to
    /// return — which is what a point with no `Out` slot is. WHERE in the
    /// pages the rows land is the pool's arithmetic and the fire's CSR,
    /// never an operand the text places.
    fn kv_append<T: Scalar>(
        &self,
        k: In<Self::Tensor<T>>,
        v: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
    ) -> Result<(), Refusal> {
        let _ = (k, v, pages);
        Err(Refusal::Absent {
            what: "attention.kv_append",
        })
    }

    /// [`Attention::kv_append`] for a checkpoint whose keys and values are
    /// THE SAME PLANE: dsv4 appends one row per token and both halves of
    /// the read address it. One operand and not two, which is why it is a
    /// point rather than the same point called twice.
    fn kv_append_shared<T: Scalar>(
        &self,
        plane: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
    ) -> Result<(), Refusal> {
        let _ = (plane, pages);
        Err(Refusal::Absent {
            what: "attention.kv_append_shared",
        })
    }
}

/// Multi-head latent attention: the family that keeps a COMPRESSED kv row
/// and pays for it with two matmuls against the checkpoint's `kv_b` bank.
/// A text here projects one packed row into the latent `kv_c` and its rope
/// half `k_pe`, absorbs the query into the latent basis, attends there, and
/// absorbs the result back out to the value basis.
///
/// STATED GEOMETRY, and the absorbs are where it bites. `kv_b` is a
/// `[heads, nope_dim + v_head_dim, kv_lora_rank]` bank and each absorb takes
/// the WHOLE bank and slices it itself — a `Const` carries an address and no
/// rectangle, so the head pitch `(nope_dim + v_head_dim) * kv_lora_rank` is
/// nowhere in the operands to read. That is why `absorb_q` states the
/// `v_head_dim` it never multiplies by and `absorb_out` states the
/// `nope_dim` it only skips past: each names the HALF OF THE BANK IT DOES
/// NOT USE, because the pitch is both halves and the statement is the only
/// place either number lives. Both texts already hold them. The token count
/// is NOT stated — a batched absorb runs one gemm per head over the rows it
/// was handed, so it is the operand's own.
///
/// THE CACHE IS THIS FAMILY'S. `kv_append` writes the latent pair into the
/// paged pool the attention points read, so it is declared here rather than
/// under a prefix of its own: a cache write belongs to the family that owns
/// the cache.
#[points]
pub trait Mla: Plane {
    /// Cut the `kv_a` projection into the normalised latent and its rope
    /// half. `kv_lora_rank` is where the cut falls.
    fn latents<T: Scalar>(
        &self,
        kv_a: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Out<Self::Tensor<T>>,
        k_pe: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (kv_a, weight, eps, kv_lora_rank, kv_c, k_pe);
        Err(Refusal::Absent {
            what: "mla.latents",
        })
    }

    /// [`Mla::latents`] with the rope half rotated on the way out — glm's
    /// reading, which never sees the unrotated `k_pe`.
    ///
    /// CLAIMED ON CUDA, AS TWO LAUNCHES. The absence recorded here was a
    /// FUSION rather than a missing kernel — `mla_prepare_bf16` does this
    /// and three more things in one launch, and is `untraced` for that
    /// reason — but the declared statement is only the cut plus a rotation
    /// of its rope half, and that is [`Mla::latents`]'s own routine followed
    /// by an ordinary partial rope over a one-head row. The fused kernel
    /// stays where it is; the point does not need it.
    fn latents_rope<T: Scalar>(
        &self,
        kv_a: In<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        kv_lora_rank: u32,
        rope_dim: u32,
        theta: f32,
        kv_c: Out<Self::Tensor<T>>,
        k_pe: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (
            kv_a,
            positions,
            weight,
            eps,
            kv_lora_rank,
            rope_dim,
            theta,
            kv_c,
            k_pe,
        );
        Err(Refusal::Absent {
            what: "mla.latents_rope",
        })
    }

    /// Cut the `q_b` projection into its nope and rope halves, per head.
    fn split_q_b<T: Scalar>(
        &self,
        q_b: In<Self::Tensor<T>>,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        q_nope: Out<Self::Tensor<T>>,
        q_pe: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q_b, heads, nope_dim, rope_dim, q_nope, q_pe);
        Err(Refusal::Absent {
            what: "mla.split_q_b",
        })
    }

    /// `q_latent[h] = kv_b_k[h] @ q_nope[h]`: the query, absorbed into the
    /// latent basis one head at a time.
    fn absorb_q<T: Scalar>(
        &self,
        q_nope: In<Self::Tensor<T>>,
        kv_b: Const<Self::Tensor<T>>,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (
            q_nope,
            kv_b,
            heads,
            kv_lora_rank,
            nope_dim,
            v_head_dim,
            q_latent,
        );
        Err(Refusal::Absent {
            what: "mla.absorb_q",
        })
    }

    /// `o[h] = kv_b_v[h]ᵀ @ latent[h]`: the attended latent, absorbed back
    /// out to the value basis.
    fn absorb_out<T: Scalar>(
        &self,
        latent: In<Self::Tensor<T>>,
        kv_b: Const<Self::Tensor<T>>,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (latent, kv_b, heads, kv_lora_rank, v_head_dim, nope_dim, o);
        Err(Refusal::Absent {
            what: "mla.absorb_out",
        })
    }

    /// Append the latent pair to the paged pool this family owns.
    ///
    /// CLAIMED ON CUDA, and what closed it was the POOL VIEW rather than a
    /// new slot here. `write_mla_to_pages` wants the layer's two page
    /// planes, the fire's query CSR, its row-validity plane and the request
    /// count beside the page view; the page planes and the page CSR were
    /// always the pool row's, and cuda's `Pages` view now also carries the
    /// query CSR, the row validity and the request count it is BUILT PER
    /// FIRE out of. That is a plane's own answer to "which rows of this
    /// fire, and where" — not a slot, because no text can place it. The
    /// declaration is unchanged: two rectangles and one cache row.
    fn kv_append<T: Scalar>(
        &self,
        kv_c: In<Self::Tensor<T>>,
        k_pe: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
    ) -> Result<(), Refusal> {
        let _ = (kv_c, k_pe, pages);
        Err(Refusal::Absent {
            what: "mla.kv_append",
        })
    }

    /// Attend in the latent basis, one token per request. `q_pe` is the
    /// rotated half the absorb did not fold in, matched against the pool's
    /// own `k_pe` plane.
    ///
    /// THE SCHEDULE IS THE PLANE'S, AND THE BODY ASKS FOR IT BY NAME. This
    /// point and [`Mla::attention_prefill`] were claim-only while the only
    /// thing a claim could delegate to was an operand column: the schedule is
    /// measured on the HOST out of three CSR slices and uploaded into an int
    /// arena the launch reads, which is [`Attention::decode`]'s seam exactly,
    /// and a body that built one would have to read the device CSR back
    /// mid-fire — a sync a capture cannot record. Both are BODIES on cuda
    /// now, and nothing here moved to allow it: an implementation pulls plane
    /// staging off `self`, and `"mla.plan"` is the key it pulls this one by.
    /// An executor that stages no such object refuses with that key, which is
    /// a sentence about the fire rather than about a missing kernel.
    fn attention_decode<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        q_pe: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q, q_pe, pages, heads, kv_lora_rank, sm_scale, o);
        Err(Refusal::Absent {
            what: "mla.attention_decode",
        })
    }

    /// [`Mla::attention_decode`] over a prefill window, `indptr` the fire's
    /// query CSR.
    fn attention_prefill<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        q_pe: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q, indptr, q_pe, pages, heads, kv_lora_rank, sm_scale, o);
        Err(Refusal::Absent {
            what: "mla.attention_prefill",
        })
    }

    /// [`Mla::attention_decode`] over the keys a `selection` KEEPS —
    /// deepseek's sparse indexer, read one token per request.
    ///
    /// THE QUERY IS THE SEPARATE PAIR, and that is a correction rather than
    /// an addition. This point used to state one fused `q` and no `q_pe`,
    /// on the reading that glm's absorb carried the rotated half in the
    /// result's tail; `mla.absorb_q_pe` was the other end of that reading
    /// and is now DELETED, because nothing writes such a tail (the absorb
    /// is a strided batched gemm with one activation operand, and no
    /// per-head scatter exists) and nothing reads one — every latent
    /// attention kernel in this tree addresses `q_nope` and `q_pe` as two
    /// planes with two pitches and no stride between them. The legacy glm
    /// text folded nothing either; it called the same separate-pair absorb
    /// kimi does. So these two read exactly as [`Mla::attention_decode`]
    /// and [`Mla::attention_prefill`] do, plus the selection.
    ///
    /// `selection` IS AN INDEX LIST AND NOT A MASK: `[tokens, top_k]` of
    /// `i32`, ascending, padded with `-1`. A mask would be `[tokens, kv]`
    /// and the kv extent is a per-request runtime number that appears in no
    /// slot of [`Index::topk`] — the one rectangle `model_compiler`'s width
    /// table could not size. `top_k` IS a stated scalar, so the list is
    /// dense and sized, and an attention that consumes it WALKS it instead
    /// of testing every cached key against a byte, which is what makes the
    /// sparse reading actually sparse.
    fn attention_decode_selected<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        q_pe: In<Self::Tensor<T>>,
        selection: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q, q_pe, selection, pages, heads, kv_lora_rank, sm_scale, o);
        Err(Refusal::Absent {
            what: "mla.attention_decode_selected",
        })
    }

    /// [`Mla::attention_decode_selected`] over a prefill window, `indptr`
    /// the fire's query CSR — [`Mla::attention_prefill`]'s slot order with
    /// `selection` after `q_pe`.
    fn attention_prefill_selected<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        q_pe: In<Self::Tensor<T>>,
        selection: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (
            q,
            indptr,
            q_pe,
            selection,
            pages,
            heads,
            kv_lora_rank,
            sm_scale,
            o,
        );
        Err(Refusal::Absent {
            what: "mla.attention_prefill_selected",
        })
    }
}

/// The sparse indexer: a small side attention whose only result is WHICH
/// KEYS the real attention should look at. Its keys ride a paged pool of
/// their own — `Cache<Self::Pages>`, the second pool the floor declares —
/// and the family owns that pool, so the append that fills it is declared
/// here.
///
/// Both rotations are IN PLACE, which is what `InOut` says: a statement's
/// result is the row it rotated, the way every [`Rope`] point reads.
#[points]
pub trait Index: Plane {
    /// Layer-norm the indexer's key row — a norm with a learned BIAS, which
    /// is what makes it a layernorm rather than an rmsnorm — then rotate its
    /// trailing `rope_dim` channels.
    fn layernorm_rope<T: Scalar>(
        &self,
        k: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        weight: Const<Self::Tensor<T>>,
        bias: Const<Self::Tensor<T>>,
        eps: f32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        let _ = (k, positions, weight, bias, eps, rope_dim, theta);
        Err(Refusal::Absent {
            what: "index.layernorm_rope",
        })
    }

    /// Rotate the indexer's query row. `head_dim` is the pitch and
    /// `rope_dim` the rotated slice of it; `heads` is stated because the
    /// launch grids by it and a packed row does not spell it.
    fn rope<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal> {
        let _ = (q, positions, heads, head_dim, rope_dim, theta);
        Err(Refusal::Absent {
            what: "index.rope",
        })
    }

    /// Score every CACHED key against the query and answer WHICH `top_k` of
    /// them the attention should read:
    ///
    /// ```text
    /// logit[t, j] = Σ_h relu(q[t, h] · k[j]) * weights[t, h]
    /// ```
    ///
    /// causal in the request's own absolute positions, `j` running over the
    /// whole cached prefix and not merely over this fire's rows.
    ///
    /// THE RESULT IS AN INDEX LIST AND NOT A MASK, and that is the shape
    /// decision this point exists to record. A byte mask is `[tokens, kv]`,
    /// and the kv extent is a PER-REQUEST RUNTIME NUMBER: it appears in no
    /// operand, no param and no bank of this statement, so no width rule can
    /// size it and `model_compiler::program::out_sizes` carried it as its
    /// one honest `None`. `top_k` is a stated scalar. `[tokens, top_k]` of
    /// `i32` — ascending, `-1` past the end — is therefore dense, sized
    /// from the statement itself, and the form `moe.topk_sigmoid`'s `routes`
    /// already established for "which of many, at a fixed budget". It is
    /// also the CHEAPER value at every kv longer than `4 * top_k` bytes, and
    /// the only one an attention can consume by WALKING rather than by
    /// testing every cached key against a byte.
    ///
    /// The legacy `dsa_index_topk_mask` is not this point and never was: it
    /// scores a TOKEN-PLANE `idx_k`, its causality is `j <= i` inside one
    /// fire, and the mask it wrote was assigned to `let _index_mask` and
    /// thrown away. The paged reading is the new one, and `index_topk_paged`
    /// is the kernel for it.
    fn topk<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        weights: In<Self::Tensor<T>>,
        keys: Cache<Self::Pages>,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        selection: Out<Self::Tensor<i32>>,
    ) -> Result<(), Refusal> {
        let _ = (q, weights, keys, heads, head_dim, top_k, selection);
        Err(Refusal::Absent {
            what: "index.topk",
        })
    }

    /// Append this fire's key rows to the indexer's own pool.
    ///
    /// NO ROUTINE ANYWHERE ANSWERED IT and none had to: the legacy indexer
    /// never paged its keys at all — it scored the token plane it had just
    /// written and kept nothing across fires. The pool the statement names
    /// is the new reading. A SINGLE-PLANE append is what it is: one row per
    /// token, `k` only, no value half, into the page slot the fire's CSR
    /// resolves to — which is [`Mla::kv_append`]'s destination arithmetic
    /// with the second plane empty.
    fn kv_append<T: Scalar>(
        &self,
        k: In<Self::Tensor<T>>,
        keys: Cache<Self::Pages>,
    ) -> Result<(), Refusal> {
        let _ = (k, keys);
        Err(Refusal::Absent {
            what: "index.kv_append",
        })
    }
}

/// DeepSeek-V4's compressed KV plane: one pooled entry per `ratio` tokens,
/// attended beside the full-resolution attention and merged back through the
/// two log-sum-exps. The family owns the entries pool, so the append that
/// fills it is declared here.
///
/// STATED GEOMETRY: `ratio` everywhere, `head_dim` on the gather. The
/// pooling ratio is the checkpoint's and appears in no operand; the gather's
/// head width sizes a result the statement allocates.
///
/// WHAT THIS FAMILY IS MISSING IS AN EXECUTOR, NOT A DELEGATION. The
/// compressed plane is THREE resident objects beside the page table — the
/// state halves, the running scores, the absolute-position table — plus the
/// compressed pool itself and the fire's own row-validity and
/// request-of-token planes, and a statement names ONE cache row and its
/// operands. That kept every point here claim-only until an implementation
/// had somewhere to ask from: each of those objects is declared with a KEY,
/// and a body pulls it off `self` by that key. All five are bodies on cuda.
///
/// The absences each point below records are therefore about the EXECUTOR
/// now: an implementation that stages the object answers, and one that does
/// not refuses with the key in it. `driver-cuda` stages none of them today
/// and says so on its own `UNSTAGED` list.
#[points]
pub trait Pool: Plane {
    /// Which tokens close a pooling window, one token per request: the
    /// boundary's own position and the request it belongs to, with `-1` at
    /// every token that closes nothing.
    ///
    /// TWO RESULTS, AND THE KERNEL WRITES THREE. `dsv4_boundary_meta_decode`
    /// also writes an `out_rope` plane — the boundary's rope position — that
    /// no statement in this tree reads. The declaration records the statement
    /// as it stands rather than inventing a consumer for the third: a result
    /// nothing reads is not a result, and a slot for it would be a rectangle
    /// no text could name. An IMPLEMENTATION sinks it into plane scratch,
    /// which is the right home for a write nobody wants — the same place
    /// every other body puts the buffers a kernel needs and a statement does
    /// not state.
    ///
    /// THE ROW-VALIDITY PLANE IS THIS POINT'S ONE RAISE, and it is the one
    /// place in the tree where that plane cannot be read off a pool row:
    /// this point names no cache. A body asks for it by key, and its ABSENCE
    /// is an answer — null means every row of the fire is valid, which the
    /// kernels test for.
    fn boundary_decode(
        &self,
        positions: In<Self::Tensor<i32>>,
        ratio: u32,
        boundary_pos: Out<Self::Tensor<i32>>,
        boundary_req: Out<Self::Tensor<i32>>,
    ) -> Result<(), Refusal> {
        let _ = (positions, ratio, boundary_pos, boundary_req);
        Err(Refusal::Absent {
            what: "pool.boundary_decode",
        })
    }

    /// [`Pool::boundary_decode`] over a prefill window, `indptr` the fire's
    /// query CSR — which is the one thing the prefill form needs and the
    /// decode form can shortcut, since one row per request makes the request
    /// column the row index. The same two-of-three arity and the same
    /// row-validity raise behind it.
    fn boundary_prefill(
        &self,
        positions: In<Self::Tensor<i32>>,
        indptr: In<Self::Tensor<i32>>,
        ratio: u32,
        boundary_pos: Out<Self::Tensor<i32>>,
        boundary_req: Out<Self::Tensor<i32>>,
    ) -> Result<(), Refusal> {
        let _ = (positions, indptr, ratio, boundary_pos, boundary_req);
        Err(Refusal::Absent {
            what: "pool.boundary_prefill",
        })
    }

    /// Build one pooled entry per boundary, out of the `ratio` tokens ending
    /// there.
    ///
    /// `dsv4_compress_gather_paged` reads the page table — which the
    /// statement names — AND the three dsv4 residents, and wants a `coff`
    /// beside the ratio: the compressor's window multiplier, a pure function
    /// of the stated ratio. THE SCALAR IS DERIVED AND THE RESIDENTS ARE
    /// RAISED, which is the split this family kept getting wrong: a number a
    /// statement already implies has no business being a second slot, and an
    /// object no statement can name has no business being one either.
    fn gather<T: Scalar>(
        &self,
        boundary_pos: In<Self::Tensor<i32>>,
        boundary_req: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        head_dim: u32,
        ratio: u32,
        entries: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (boundary_pos, boundary_req, pages, head_dim, ratio, entries);
        Err(Refusal::Absent {
            what: "pool.gather",
        })
    }

    /// Append the pooled entries to the entries pool.
    ///
    /// `dsv4_store_comp_entries` reads TWO pools — the page table it walks
    /// for the boundary's page, and the compressed plane it writes — where a
    /// statement names ONE cache row. A second `Cache` slot would be a second
    /// pool the text does not state, so the plane is a RAISE and the page
    /// table is the statement's row. They collapse into one slot the day an
    /// executor builds a pool view per named ROW rather than per model layer;
    /// until then the page CSR is fire-wide and identical on every row of a
    /// fire, so reading it off this one is right for the fields the kernel
    /// takes from it.
    fn kv_append<T: Scalar>(
        &self,
        entries: In<Self::Tensor<T>>,
        boundary_pos: In<Self::Tensor<i32>>,
        boundary_req: In<Self::Tensor<i32>>,
        pool: Cache<Self::Pages>,
    ) -> Result<(), Refusal> {
        let _ = (entries, boundary_pos, boundary_req, pool);
        Err(Refusal::Absent {
            what: "pool.kv_append",
        })
    }

    /// Attend over the pooled entries, stating the log-sum-exp beside the
    /// output so the merge with the full-resolution attention is exact.
    ///
    /// The lse rides [`Attention::decode_lse`]'s base, because the only
    /// thing that ever reads it is the merge against one.
    ///
    /// [`Pool::kv_append`]'s two pools, and one more raise on top:
    /// `attention_compressed_paged` reads the fire's request-of-token plane,
    /// which is derivable from the query CSR and derived by the executor —
    /// but a kernel handed one row at a time cannot do the search, so the
    /// plane is staged and a body asks for it by key.
    fn attention_lse<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        entries: Cache<Self::Pages>,
        ratio: u32,
        heads: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
        lse: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            q, positions, entries, ratio, heads, head_dim, sm_scale, o, lse,
        );
        Err(Refusal::Absent {
            what: "pool.attention_lse",
        })
    }
}

/// DeepSeek-V4's hyper-connections: the residual is not one stream but
/// `streams` of them, and every block reads a learned mixture of the stack
/// and writes its result back through a second one. A text expands once at
/// the prologue, gates before each block, folds after it, and collapses once
/// at the end.
///
/// THE MIXER RIDES f32 AND THE STREAMS RIDE `T`. A mix matrix is a small
/// dense thing whose rows and columns are driven to sum to one, and rounding
/// it to the activation dtype would leave it measurably un-stochastic —
/// which the kernels say in as many words. So the mixes, the gate weights
/// and the two mix results are all `f32` slots, and only the residual stack
/// and the layer's own row quantify over the family's axis.
///
/// STATED GEOMETRY: `stream_count`. The stack is `[tokens, streams, hidden]`
/// flattened to a row, so the count is the row over the hidden width — which
/// every claim below reads back off the rectangles it was handed, and which
/// the statement states anyway because the collapsed row is an `Out` the
/// statement allocates and a divisor has to exist before the row does. The
/// same shape [`Moe`]'s `experts` takes.
#[points]
pub trait Hc: Plane {
    /// Broadcast one row into `streams` of them: the stack a text starts
    /// from before any hyper-connection has mixed anything.
    fn expand<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        streams: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, streams, y);
        Err(Refusal::Absent { what: "hc.expand" })
    }

    /// Normalise the stack into f32 — the mixer's own input, which never
    /// rides the activation dtype.
    fn rmsnorm_f32<T: Scalar>(
        &self,
        streams: In<Self::Tensor<T>>,
        eps: f32,
        y: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (streams, eps, y);
        Err(Refusal::Absent {
            what: "hc.rmsnorm_f32",
        })
    }

    /// Split the mix matrix, Sinkhorn-normalise the combiner, and collapse
    /// the stack into the block's input.
    ///
    /// THREE RESULTS, and the first is the one the block consumes: `x` is
    /// the row the block runs on, `post_mix` and `comb_mix` are what
    /// [`Hc::fold`] folds its answer back through. A statement that named
    /// only the row would leave the fold reading two planes nothing states.
    fn gates<T: Scalar>(
        &self,
        normed: In<Self::Tensor<f32>>,
        streams: In<Self::Tensor<T>>,
        scale: Const<Self::Tensor<f32>>,
        base: Const<Self::Tensor<f32>>,
        stream_count: u32,
        gate_eps: f32,
        alpha: f32,
        sinkhorn: u32,
        x: Out<Self::Tensor<T>>,
        post_mix: Out<Self::Tensor<f32>>,
        comb_mix: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (
            normed,
            streams,
            scale,
            base,
            stream_count,
            gate_eps,
            alpha,
            sinkhorn,
            x,
            post_mix,
            comb_mix,
        );
        Err(Refusal::Absent { what: "hc.gates" })
    }

    /// Fold a block's answer back into the stack through the two mixes
    /// [`Hc::gates`] stated.
    fn fold<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        streams: In<Self::Tensor<T>>,
        post_mix: In<Self::Tensor<f32>>,
        comb_mix: In<Self::Tensor<f32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (x, streams, post_mix, comb_mix, y);
        Err(Refusal::Absent { what: "hc.fold" })
    }

    /// Collapse the stack back into one row — the gated sum that closes a
    /// text, with no Sinkhorn behind it.
    ///
    /// CLAIM-ONLY ON EVERY PLANE, and the absence is an OPERAND WITH NO
    /// PRODUCER — which is a different gap from the rest of this family's and
    /// the reason R4b left this one where it stands while claiming the four
    /// pool points beside it.
    ///
    /// `hc_head_postprocess` reads TWO planes: an `[N, streams]` f32 `mixes`
    /// — the head gate logits, which the kernel's own comment says arrive
    /// "after GEMM" — and the `[N, streams, hidden]` residual stack. The
    /// statement names ONE value. The legacy call site passed the bf16 stack
    /// for the f32 `mixes` slot, which reads a stack's leading bytes as
    /// gates; that is a bug in the caller, not a delegation to copy.
    ///
    /// AND THERE IS NOTHING ELSE TO BIND. This is not a raise waiting for an
    /// executor: no text in this tree computes a head-mix plane, and the
    /// import ships no bank one could come from — deepseek-v4's checkpoint
    /// carries `hc_head_scale` `[1]` and `hc_head_base` `[streams]` and no
    /// head mix weight at all, which is exactly the two `Const` slots this
    /// declaration already states. So the honest readings are a new operand
    /// with a producer to write, or a per-stream gate with no per-token term
    /// — and dsv4 has no cached checkpoint to decide between them, which is
    /// the third party this tree's own law says has to. Until one is cached,
    /// the routine keeps its `canon` and this note is the measurement.
    fn collapse<T: Scalar>(
        &self,
        streams: In<Self::Tensor<T>>,
        head_scale: Const<Self::Tensor<f32>>,
        head_base: Const<Self::Tensor<f32>>,
        stream_count: u32,
        gate_eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (streams, head_scale, head_base, stream_count, gate_eps, y);
        Err(Refusal::Absent {
            what: "hc.collapse",
        })
    }
}
