#![allow(clippy::too_many_arguments)]

use kernels_macros::points;

use crate::plane::{Cache, Const, ConstRun, Elem, In, InOut, Out, Refusal};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point {
    pub name: &'static str,

    pub axes: usize,

    pub reprs: usize,

    pub slots: &'static [Slot],

    pub outs: &'static [Shape],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    pub rows: Fan,

    pub width: Width,

    pub elem: Element,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fan {
    Fire,

    Ride(usize),

    Per(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Width {
    Of(usize),

    Stated(usize),

    Axis(usize, usize),

    Count(u64),

    Times(&'static Width, &'static Width),

    Over(&'static Width, &'static Width),

    Less(&'static Width, &'static Width),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Element {
    Ride(usize),

    Weight(usize),

    Fixed(Prim),

    Activation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    pub name: &'static str,

    pub mark: Mark,

    pub dtype: Dtype,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mark {
    In,

    InOut,

    Out,

    Const,

    Cache,

    Scalar,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Generic(usize),

    Fixed(Prim),

    Bank(usize),

    Opaque,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Prim {
    F32,

    I32,

    U32,

    Bool,

    U8,
}

pub trait Plane {
    type Tensor<T: Scalar>: Elem + ConstRun;

    type Bank<R: Repr>: ConstRun;

    type Recurrent: Elem;

    type Pages: Elem;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarKind {
    Bf16,
    F16,
    F32,
    I32,
    U32,
    U8,
}

pub trait Scalar: Elem<Read = *const Self, Write = *mut Self> + Sized {
    const KIND: ScalarKind;
}

impl Scalar for f32 {
    const KIND: ScalarKind = ScalarKind::F32;
}

impl Scalar for i32 {
    const KIND: ScalarKind = ScalarKind::I32;
}

impl Scalar for u32 {
    const KIND: ScalarKind = ScalarKind::U32;
}

impl Scalar for u8 {
    const KIND: ScalarKind = ScalarKind::U8;
}

pub trait Repr: 'static {
    const FORM: Form;

    const PLANES: usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Form {
    Mxfp4,
}

impl Form {
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "mxfp4" => Some(Self::Mxfp4),
            _ => None,
        }
    }
}

pub enum Mxfp4 {}

impl Repr for Mxfp4 {
    const FORM: Form = Form::Mxfp4;
    const PLANES: usize = 2;
}

#[points]
pub trait Norm: Plane {
    #[shape(y = x)]
    fn rmsnorm<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = x)]
    fn rmsnorm_per_head<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = x)]
    fn rmsnorm_plus_one<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = x)]
    fn rmsnorm_per_head_plus_one<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = x)]
    fn rmsnorm_no_scale<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = gate)]
    fn rmsnorm_gated<T: Scalar>(
        &self,
        x: In<Self::Tensor<f32>>,
        gate: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<f32>>,
        head_dim: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = gate)]
    fn rmsnorm_gated_by<T: Scalar>(
        &self,
        x: In<Self::Tensor<f32>>,
        gate: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<f32>>,
        heads: u32,
        eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    fn residual_add<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        y: InOut<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    fn add_bias<T: Scalar>(
        &self,
        bias: Const<Self::Tensor<T>>,
        out: InOut<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    fn mul_scalar<T: Scalar>(&self, s: f32, x: InOut<Self::Tensor<T>>) -> Result<(), Refusal>;

    fn scale<T: Scalar>(
        &self,
        s: Const<Self::Tensor<T>>,
        x: InOut<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = prefix)]
    fn res_blend<T: Scalar>(
        &self,
        prefix: In<Self::Tensor<T>>,
        blocks: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        proj: Const<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Mlp: Plane {
    #[shape(y = [packed.rows, intermediate])]
    fn swiglu<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [packed.rows, intermediate])]
    fn swiglu_clamp<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        limit: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [packed.rows, intermediate])]
    fn swiglu_clamp_alpha<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        limit: f32,
        alpha: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = gate)]
    fn geglu_tanh<T: Scalar>(
        &self,
        gate: In<Self::Tensor<T>>,
        up: In<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [packed.rows, intermediate])]
    fn geglu_tanh_packed<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [packed.rows, intermediate])]
    fn situ<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        intermediate: u32,
        beta: f32,
        up_cap: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Gemm: Plane {
    #[shape(y = [act.rows, w.axis(0)])]
    fn matmul<T: Scalar>(
        &self,
        act: In<Self::Tensor<T>>,
        w: Const<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [act.rows, w.axis(0)])]
    fn lm_head<T: Scalar>(
        &self,
        act: In<Self::Tensor<T>>,
        w: Const<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [act.rows, w.axis(0)])]
    fn attention_landing<T: Scalar>(
        &self,
        act: In<Self::Tensor<T>>,
        w: Const<Self::Tensor<T>>,
        layer: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Dist: Plane {
    fn all_reduce<T: Scalar>(&self, buf: InOut<Self::Tensor<T>>) -> Result<(), Refusal>;
}

#[points]
pub trait Rope: Plane {
    fn full<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        k: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal>;

    fn partial<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        k: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal>;

    fn partial_q<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal>;

    fn partial_last<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
    ) -> Result<(), Refusal>;

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
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Moe: Plane {
    #[shape(routes = [fire, top_k], weights = [fire, top_k])]
    fn topk_softmax<T: Scalar>(
        &self,
        logits: In<Self::Tensor<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<Self::Tensor<i32>>,
        weights: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal>;

    #[shape(routes = [fire, top_k], weights = [fire, top_k])]
    fn topk_sigmoid<T: Scalar>(
        &self,
        logits: In<Self::Tensor<T>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Self::Tensor<i32>>,
        weights: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal>;

    #[shape(routes = [fire, top_k], weights = [fire, top_k])]
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
    ) -> Result<(), Refusal>;

    #[shape(y = [per(routes), bank.axis(1)])]
    fn matmul_select<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        bank: Const<Self::Tensor<T>>,
        routes: In<Self::Tensor<i32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [per(routes), bank.axis(1)])]
    fn matmul_select_bias<T: Scalar, R: Repr>(
        &self,
        x: In<Self::Tensor<T>>,
        bank: Const<Self::Bank<R>>,
        bias: Const<Self::Tensor<T>>,
        routes: In<Self::Tensor<i32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [fire, routed.width])]
    fn weighted_sum<T: Scalar>(
        &self,
        routed: In<Self::Tensor<T>>,
        weights: In<Self::Tensor<f32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = routed)]
    fn sigmoid_gate_add<T: Scalar>(
        &self,
        routed: In<Self::Tensor<T>>,
        shared: In<Self::Tensor<T>>,
        gate: In<Self::Tensor<T>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Gate: Plane {
    fn sigmoid_mul<T: Scalar>(
        &self,
        x: InOut<Self::Tensor<T>>,
        gate: In<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Layout: Plane {
    #[shape(y = [fire, table.axis(1)])]
    fn embed<T: Scalar>(
        &self,
        ids: In<Self::Tensor<i32>>,
        table: Const<Self::Tensor<T>>,
        vocab: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(q = [packed.rows, q_width],
    k = [packed.rows, kv_width],
    v = [packed.rows, kv_width])]
    fn split_qkv<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        q_width: u32,
        kv_width: u32,
        q: Out<Self::Tensor<T>>,
        k: Out<Self::Tensor<T>>,
        v: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(q = [packed.rows, packed.width / (2 * head_dim) * head_dim],
    gate = [packed.rows, packed.width / (2 * head_dim) * head_dim])]
    fn split_q_gate<T: Scalar>(
        &self,
        packed: In<Self::Tensor<T>>,
        head_dim: u32,
        q: Out<Self::Tensor<T>>,
        gate: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(left = [x.rows, width], right = [x.rows, x.width - width])]
    fn split_rows<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        width: u32,
        left: Out<Self::Tensor<T>>,
        right: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [table.rows, width])]
    fn select<T: Scalar>(
        &self,
        table: In<Self::Tensor<T>>,
        layer: u32,
        width: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Ssm: Plane {
    #[shape(y = x)]
    fn causal_conv1d<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        state: Cache<Self::Recurrent>,
        conv_width: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = x)]
    fn causal_conv1d_chunked<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        weight: Const<Self::Tensor<T>>,
        state: Cache<Self::Recurrent>,
        conv_width: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(gates = ba)]
    fn gdn_prep<T: Scalar>(
        &self,
        ba: In<Self::Tensor<T>>,
        dt_bias: Const<Self::Tensor<T>>,
        a_log: Const<Self::Tensor<f32>>,
        gates: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [qkv.rows, v_heads * v_dim])]
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
    ) -> Result<(), Refusal>;

    #[shape(y = [qkv.rows, v_heads * v_dim])]
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
    ) -> Result<(), Refusal>;

    #[shape(y = [mixed.rows, heads * head_dim])]
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
    ) -> Result<(), Refusal>;

    #[shape(y = [mixed.rows, heads * head_dim])]
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
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Attention: Plane {
    #[shape(o = q)]
    fn decode<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(o = q)]
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
    ) -> Result<(), Refusal>;

    #[shape(o = q)]
    fn masked<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        indptr: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(o = q, lse = [q.rows, q.width / head_dim])]
    fn decode_lse<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
        window: u32,
        head_dim: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
        lse: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal>;

    #[shape(o = q, lse = [q.rows, q.width / head_dim])]
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
    ) -> Result<(), Refusal>;

    fn sink<T: Scalar>(
        &self,
        o: InOut<Self::Tensor<T>>,
        lse: In<Self::Tensor<f32>>,
        sink: Const<Self::Tensor<T>>,
        head_dim: u32,
    ) -> Result<(), Refusal>;

    #[shape(o = o1, lse = lse1)]
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
    ) -> Result<(), Refusal>;

    fn logit_softcap<T: Scalar>(&self, x: InOut<Self::Tensor<T>>, cap: f32) -> Result<(), Refusal>;

    fn kv_append<T: Scalar>(
        &self,
        k: In<Self::Tensor<T>>,
        v: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
    ) -> Result<(), Refusal>;

    fn kv_append_shared<T: Scalar>(
        &self,
        plane: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Mla: Plane {
    #[shape(kv_c = [kv_a.rows, kv_lora_rank],
    k_pe = [kv_a.rows, kv_a.width - kv_lora_rank])]
    fn latents<T: Scalar>(
        &self,
        kv_a: In<Self::Tensor<T>>,
        weight: Const<Self::Tensor<T>>,
        eps: f32,
        kv_lora_rank: u32,
        kv_c: Out<Self::Tensor<T>>,
        k_pe: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(kv_c = [kv_a.rows, kv_lora_rank],
    k_pe = [kv_a.rows, kv_a.width - kv_lora_rank])]
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
    ) -> Result<(), Refusal>;

    #[shape(q_nope = [q_b.rows, heads * nope_dim],
    q_pe = [q_b.rows, heads * rope_dim])]
    fn split_q_b<T: Scalar>(
        &self,
        q_b: In<Self::Tensor<T>>,
        heads: u32,
        nope_dim: u32,
        rope_dim: u32,
        q_nope: Out<Self::Tensor<T>>,
        q_pe: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(q_latent = [q_nope.rows, heads * kv_lora_rank])]
    fn absorb_q<T: Scalar>(
        &self,
        q_nope: In<Self::Tensor<T>>,
        kv_b: Const<Self::Tensor<T>>,
        heads: u32,
        kv_lora_rank: u32,
        nope_dim: u32,
        v_head_dim: u32,
        q_latent: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(o = [latent.rows, heads * v_head_dim])]
    fn absorb_out<T: Scalar>(
        &self,
        latent: In<Self::Tensor<T>>,
        kv_b: Const<Self::Tensor<T>>,
        heads: u32,
        kv_lora_rank: u32,
        v_head_dim: u32,
        nope_dim: u32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    fn kv_append<T: Scalar>(
        &self,
        kv_c: In<Self::Tensor<T>>,
        k_pe: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
    ) -> Result<(), Refusal>;

    #[shape(o = [q.rows, heads * kv_lora_rank])]
    fn attention_decode<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        q_pe: In<Self::Tensor<T>>,
        pages: Cache<Self::Pages>,
        heads: u32,
        kv_lora_rank: u32,
        sm_scale: f32,
        o: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(o = [q.rows, heads * kv_lora_rank])]
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
    ) -> Result<(), Refusal>;

    #[shape(o = [q.rows, heads * kv_lora_rank])]
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
    ) -> Result<(), Refusal>;

    #[shape(o = [q.rows, heads * kv_lora_rank])]
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
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Index: Plane {
    fn layernorm_rope<T: Scalar>(
        &self,
        k: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        weight: Const<Self::Tensor<T>>,
        bias: Const<Self::Tensor<T>>,
        eps: f32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal>;

    fn rope<T: Scalar>(
        &self,
        q: InOut<Self::Tensor<T>>,
        positions: In<Self::Tensor<i32>>,
        heads: u32,
        head_dim: u32,
        rope_dim: u32,
        theta: f32,
    ) -> Result<(), Refusal>;

    #[shape(selection = [q.rows, top_k])]
    fn topk<T: Scalar>(
        &self,
        q: In<Self::Tensor<T>>,
        weights: In<Self::Tensor<T>>,
        keys: Cache<Self::Pages>,
        heads: u32,
        head_dim: u32,
        top_k: u32,
        selection: Out<Self::Tensor<i32>>,
    ) -> Result<(), Refusal>;

    fn kv_append<T: Scalar>(
        &self,
        k: In<Self::Tensor<T>>,
        keys: Cache<Self::Pages>,
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Pool: Plane {
    #[shape(boundary_pos = [fire, 1], boundary_req = [fire, 1])]
    fn boundary_decode(
        &self,
        positions: In<Self::Tensor<i32>>,
        ratio: u32,
        boundary_pos: Out<Self::Tensor<i32>>,
        boundary_req: Out<Self::Tensor<i32>>,
    ) -> Result<(), Refusal>;

    #[shape(boundary_pos = [fire, 1], boundary_req = [fire, 1])]
    fn boundary_prefill(
        &self,
        positions: In<Self::Tensor<i32>>,
        indptr: In<Self::Tensor<i32>>,
        ratio: u32,
        boundary_pos: Out<Self::Tensor<i32>>,
        boundary_req: Out<Self::Tensor<i32>>,
    ) -> Result<(), Refusal>;

    #[shape(entries = [fire, head_dim])]
    fn gather<T: Scalar>(
        &self,
        boundary_pos: In<Self::Tensor<i32>>,
        boundary_req: In<Self::Tensor<i32>>,
        pages: Cache<Self::Pages>,
        head_dim: u32,
        ratio: u32,
        entries: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    fn kv_append<T: Scalar>(
        &self,
        entries: In<Self::Tensor<T>>,
        boundary_pos: In<Self::Tensor<i32>>,
        boundary_req: In<Self::Tensor<i32>>,
        pool: Cache<Self::Pages>,
    ) -> Result<(), Refusal>;

    #[shape(o = q, lse = [q.rows, heads])]
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
    ) -> Result<(), Refusal>;
}

#[points]
pub trait Hc: Plane {
    #[shape(y = [x.rows, x.width * streams])]
    fn expand<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        streams: u32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = streams)]
    fn rmsnorm_f32<T: Scalar>(
        &self,
        streams: In<Self::Tensor<T>>,
        eps: f32,
        y: Out<Self::Tensor<f32>>,
    ) -> Result<(), Refusal>;

    #[shape(x = [streams.rows, streams.width / stream_count],
    post_mix = [streams.rows, stream_count],
    comb_mix = [streams.rows, stream_count * stream_count])]
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
    ) -> Result<(), Refusal>;

    #[shape(y = streams)]
    fn fold<T: Scalar>(
        &self,
        x: In<Self::Tensor<T>>,
        streams: In<Self::Tensor<T>>,
        post_mix: In<Self::Tensor<f32>>,
        comb_mix: In<Self::Tensor<f32>>,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;

    #[shape(y = [streams.rows, streams.width / stream_count])]
    fn collapse<T: Scalar>(
        &self,
        streams: In<Self::Tensor<T>>,
        head_scale: Const<Self::Tensor<f32>>,
        head_base: Const<Self::Tensor<f32>>,
        stream_count: u32,
        gate_eps: f32,
        y: Out<Self::Tensor<T>>,
    ) -> Result<(), Refusal>;
}

pub const FAMILIES: &[&[Point]] = &[
    NORM_POINTS,
    MLP_POINTS,
    GEMM_POINTS,
    DIST_POINTS,
    ROPE_POINTS,
    MOE_POINTS,
    GATE_POINTS,
    LAYOUT_POINTS,
    SSM_POINTS,
    ATTENTION_POINTS,
    MLA_POINTS,
    INDEX_POINTS,
    POOL_POINTS,
    HC_POINTS,
];

pub fn declared() -> impl Iterator<Item = &'static Point> {
    FAMILIES.iter().copied().flatten()
}

#[must_use]
pub fn point_of(name: &str) -> Option<&'static Point> {
    declared().find(|p| p.name == name)
}
