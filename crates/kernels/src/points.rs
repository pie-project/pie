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
    /// the mark a weight wears. The bank a text hands it is quantized as
    /// often as not, and the slot wants the `Bank<R: Repr>` payload the
    /// floor does not carry yet.
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
    /// gptoss's banks, the only ones that carry one. Unclaimed on every
    /// plane today: the measured gap.
    fn matmul_select_bias<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        bank: Const<Self::Tensor<T>>,
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
    fn embed<T: Scalar>(
        &self,
        ids: In<Self::Tensor<i32>>,
        table: Const<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (ids, table, y);
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
    /// picked out of the stack the prologue gathered once.
    ///
    /// Unclaimed on every plane, and the absence is why it is declared. The
    /// arithmetic is nothing: a layer's slice of a laid-out stack is a base
    /// and an offset, so a plane may well answer this at BINDING with a view
    /// over the relayed rows and never launch anything at all. No kernel
    /// claims it because there is nothing for a kernel to do — but until the
    /// binding stage can say that in its own voice, an unnamed gap is an
    /// unmeasured one, and this declaration is what puts gemma's slice on
    /// the backlog under a name.
    fn select<T: Scalar>(
        &self,
        table: In<Self::Tensor<T>>,
        layer: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (table, layer, y);
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
    fn gdn_prep<T: Scalar>(
        &self,
        ba: In<Self::Tensor<T>>,
        dt_bias: Const<Self::Tensor<T>>,
        a_log: Const<Self::Tensor<T>>,
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
    fn kda_step<T: Scalar>(
        &self,
        mixed: In<Self::Tensor<T>>,
        f: In<Self::Tensor<T>>,
        b: In<Self::Tensor<T>>,
        dt_bias: Const<Self::Tensor<T>>,
        a_log: Const<Self::Tensor<T>>,
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
        dt_bias: Const<Self::Tensor<T>>,
        a_log: Const<Self::Tensor<T>>,
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
/// the two CSRs, the mask view. A body pulls those from `self`; a
/// declaration that stated them would be describing one plane's plan
/// vocabulary as though every plane had it.
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

    /// [`Attention::prefill`], also leaving the per-row log-sum-exp.
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
    /// `lse` an `_lse` reading left. The sink rides f32 with the lse —
    /// both are normaliser arithmetic — while the output rides `T`.
    fn sink<T: Scalar>(
        &self,
        o: InOut<Self::Tensor<T>>,
        lse: In<Self::Tensor<f32>>,
        sink: Const<Self::Tensor<f32>>,
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

    /// Rebase a log-sum-exp from log2 to ln, in place.
    ///
    /// NO AXIS: an lse is f32 wherever it came from, so there is nothing
    /// for this point to quantify over and its dispatch is a single arm.
    /// It lives in this family and not in `Norm` because the only thing
    /// that produces one is an `_lse` reading above, and the only things
    /// that consume one are [`Attention::sink`] and
    /// [`Attention::merge_lse`] beside it — the base disagreement is
    /// between two attention kernels and belongs where both are named.
    fn lse_ln(&self, lse: InOut<Self::Tensor<f32>>) -> Result<(), Refusal> {
        let _ = lse;
        Err(Refusal::Absent {
            what: "attention.lse_ln",
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
    /// Unclaimed on every plane, and the absence is a fusion rather than a
    /// missing kernel: cuda's `mla_prepare_bf16` does this and three more
    /// things in one launch (it cuts `q_b` as well, and appends the pair to
    /// the pages), takes the layer's whole `MlaLayer` staging beside a page
    /// view and the fire's row-validity plane, and is `untraced` for exactly
    /// that reason. The measured glm gap.
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

    /// [`Mla::absorb_q`] with the rotated half folded into the same result —
    /// glm's absorb, which carries `q_pe` through rather than handing it to
    /// the attention as a second operand.
    ///
    /// Unclaimed on every plane: cuda's absorb is a strided batched gemm
    /// with ONE activation operand and no place to put a second, and the
    /// fold is not a gemm. The measured glm gap.
    fn absorb_q_pe<T: Scalar>(
        &self,
        q_nope: In<Self::Tensor<T>>,
        q_pe: In<Self::Tensor<T>>,
        kv_b: Const<Self::Tensor<T>>,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (
            q_nope,
            q_pe,
            kv_b,
            heads,
            kv_lora_rank,
            nope_dim,
            v_head_dim,
            q_latent,
        );
        Err(Refusal::Absent {
            what: "mla.absorb_q_pe",
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
    /// Unclaimed on every plane: cuda's `write_mla_to_pages` is `untraced`
    /// and takes the layer's `MlaLayer` staging, the fire's query CSR, its
    /// row-validity plane and the request count beside the page view — none
    /// of which a statement names. The measured kimi/glm gap.
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
    /// UNCLAIMED, and so are the three below it. One `unsafe`, `untraced`
    /// `dispatch_attention_mla_bf16` served all four in the legacy tree: it
    /// takes a host-side `&MlaPlan` measured against the page table, a
    /// `MlaLayer`, an `AttnMask` raise and the request count, branches on
    /// the device's compute capability, and answers with a `MlaDispatch`
    /// rather than a fired point. A routine that returns which kernel it
    /// picked is not a claim, and none of its staging is a statement's. Four
    /// measured rows.
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

    /// Attend in the latent basis over the keys a `selection` mask keeps —
    /// deepseek's sparse indexer, read one token per request. The query
    /// carries its own rotated half, which is why no `q_pe` is stated.
    fn attention_decode_selected<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        selection: In<Self::Tensor<u8>>,
        pages: Cache<Self::Pages>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (q, selection, pages, heads, kv_lora_rank, sm_scale, o);
        Err(Refusal::Absent {
            what: "mla.attention_decode_selected",
        })
    }

    /// [`Mla::attention_decode_selected`] over a prefill window.
    fn attention_prefill_selected<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        selection: In<Self::Tensor<u8>>,
        pages: Cache<Self::Pages>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal> {
        let _ = (
            q,
            indptr,
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

    /// Score every cached key against the query, keep the top `top_k`, and
    /// write the selection out as a byte mask.
    ///
    /// Unclaimed on every plane. `dsa_index_topk_mask` scores a TOKEN-PLANE
    /// `idx_k` — an ordinary rectangle of the keys this fire just wrote —
    /// and the statement names the POOL those keys live in. A cache row and
    /// a staged rectangle are different binders, which is the whole of what
    /// a mark says, so nothing here is a rename away from claiming it. The
    /// measured glm gap.
    fn topk<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        weights: In<Self::Tensor<T>>,
        keys: Cache<Self::Pages>,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        selection: Out<Self::Tensor<u8>>,
    ) -> Result<(), Refusal> {
        let _ = (q, weights, keys, heads, head_dim, top_k, selection);
        Err(Refusal::Absent {
            what: "index.topk",
        })
    }

    /// Append this fire's key rows to the indexer's own pool.
    ///
    /// Unclaimed on every plane, and NO ROUTINE ANYWHERE ANSWERS IT: the
    /// legacy indexer never paged its keys at all — it scored the token
    /// plane it had just written and kept nothing across fires. The pool the
    /// statement names is the new reading, and this is the row that measures
    /// it.
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
/// NOT ONE POINT OF THIS FAMILY DELEGATES, and the reason is the same at
/// every one of them: the compressed plane is THREE resident objects beside
/// the page table — the state halves, the running scores, the absolute-
/// position table — plus the fire's own row-validity and request-of-token
/// planes, and a statement names ONE cache row and its operands. Each point
/// below says which of those it is missing.
#[points]
pub trait Pool: Plane {
    /// Which tokens close a pooling window, one token per request: the
    /// boundary's own position and the request it belongs to, with `-1` at
    /// every token that closes nothing.
    ///
    /// TWO RESULTS, AND THE ROUTINE WRITES THREE. `dsv4_boundary_meta_decode`
    /// also states an `out_rope` plane — the boundary's rope position — that
    /// no statement in this tree reads. The declaration records the
    /// statement as it stands rather than inventing a consumer for the
    /// third: a result nothing reads is not a result, and passing a scratch
    /// rectangle to swallow it would put a slot on the floor that no text
    /// can name. The arity is one reason this point stays claim-only; the
    /// other is the fire's row-validity plane, which the routine reads and
    /// no statement names.
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
    /// query CSR. The same two-of-three arity, and the same row-validity
    /// plane behind it.
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
    /// Claim-only. `dsv4_compress_gather_paged` reads the page table AND the
    /// three dsv4 residents, and asks for a `coff` beside the ratio — the
    /// compressor's window multiplier, which is a pure function of the ratio
    /// the DRIVER owns (`compressed_plane_geometry::compressor_coff`). The
    /// scalar could be derived here; the three residents could not, and a
    /// claim that bound them out of thin air would be a fiction.
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
    /// Claim-only. `dsv4_store_comp_entries` takes TWO cache views — the
    /// page table it walks for the boundary's page, and the compressed pool
    /// it writes — and a statement names one cache row. A second `Cache`
    /// slot would be a second pool the text does not state.
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
    /// Claim-only, for [`Pool::kv_append`]'s two-cache reason and one more:
    /// `attention_compressed_paged` also reads the fire's request-of-token
    /// plane, which is runtime staging and not an operand.
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
    /// Claim-only on cuda, and the absence is an OPERAND rather than a
    /// kernel. `hc_head_postprocess` reads TWO planes: an `[N, streams]`
    /// f32 `mixes` — the head gate logits, which the kernel's own comment
    /// says arrive "after GEMM" — and the `[N, streams, hidden]` residual
    /// stack. The statement names ONE value, and the legacy call site passed
    /// the bf16 stack for the f32 `mixes` slot, which reads a stack's first
    /// bytes as gates. That is a bug in the caller, not a delegation to
    /// copy, and a claim reproducing it would put the lie on the floor.
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
