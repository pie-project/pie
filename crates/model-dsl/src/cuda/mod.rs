//! CUDA launch statements for model DSL traces.
//!
//! Functions state semantic operands; driver-owned stream, dims, workspace,
//! and prepare work are bound by the driver contract.

use crate::{ConvW, Kv, MatW, NormW, RaggedVal, Rs, Trace, Val};
use model_ir::trace::{DType, Dim, Shape, StateRef, StateStore};

/// A launch producing multiple values.
fn record_many_with_params(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    params: Vec<u32>,
    inputs: Vec<model_ir::trace::ValueId>,
    outs: Vec<(Shape, DType)>,
) -> Vec<Val> {
    let n = outs.len();
    let ids = t.with(layer, |b| {
        b.launch_with_params(kernel, weights, None, params, inputs, outs)
    });
    assert_eq!(
        ids.len(),
        n,
        "the tape recorded a different arity than stated"
    );
    ids.into_iter()
        .map(|id| Val {
            t: t.clone(),
            id,
            layer,
        })
        .collect()
}

fn record_many(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    inputs: Vec<model_ir::trace::ValueId>,
    outs: Vec<(Shape, DType)>,
) -> Vec<Val> {
    let n = outs.len();
    let ids = t.with(layer, |b| b.launch(kernel, weights, None, inputs, outs));
    assert_eq!(
        ids.len(),
        n,
        "the tape recorded a different arity than stated"
    );
    ids.into_iter()
        .map(|id| Val {
            t: t.clone(),
            id,
            layer,
        })
        .collect()
}

fn record(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch(kernel, weights, state, inputs, out.into_iter().collect())
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record`], plus symbol-defined params.
/// Signed values use two's complement.
fn record_with_params(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_with_params(
            kernel,
            weights,
            state,
            params,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record_with_params`], plus the scalars whose value is an extent the
/// FIRE decides — see [`model_ir::trace::OpKind::Launch`]'s `param_extents`.
/// Constants at those indices are placeholders and written as zero.
#[allow(clippy::too_many_arguments)]
fn record_with_extents(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_with_extents(
            kernel,
            weights,
            state,
            params,
            param_extents,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record_with_extents`], plus the peel-window slots the walk fills.
fn record_devwin(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    peel_slots: Option<(u8, u8)>,
    inputs: Vec<model_ir::trace::ValueId>,
    out: Option<(Shape, DType)>,
) -> Option<Val> {
    let ids = t.with(layer, |b| {
        b.launch_devwin(
            kernel,
            weights,
            state,
            params,
            param_extents,
            peel_slots,
            inputs,
            out.into_iter().collect(),
        )
    });
    ids.first().map(|&id| Val {
        t: t.clone(),
        id,
        layer,
    })
}

/// [`record_many_with_params`], plus fire-decided scalar extents.
#[allow(clippy::too_many_arguments)]
fn record_many_with_extents(
    t: &Trace,
    layer: Option<u32>,
    kernel: &str,
    weights: Vec<String>,
    state: Option<StateRef>,
    params: Vec<u32>,
    param_extents: Vec<(u8, Shape)>,
    inputs: Vec<model_ir::trace::ValueId>,
    outs: Vec<(Shape, DType)>,
) -> Vec<Val> {
    let n = outs.len();
    let ids = t.with(layer, |b| {
        b.launch_with_extents(kernel, weights, state, params, param_extents, inputs, outs)
    });
    assert_eq!(
        ids.len(),
        n,
        "the tape recorded a different arity than stated"
    );
    ids.into_iter()
        .map(|id| Val {
            t: t.clone(),
            id,
            layer,
        })
        .collect()
}

/// Mint one runtime OBJECT operand — a driver-owned view out of the
/// `kernels` runtime vocabulary (`"kv_cache"`, `"fa2.prefill"`, …) — and
/// return its value id for an `inputs` run.
fn rt_object(t: &Trace, name: &str, layer: Option<u32>) -> model_ir::trace::ValueId {
    t.with(layer, |b| b.runtime_object(name, layer))
}

/// Mint one per-fire `[Tokens]` i32 stream by its vocabulary name
/// (`"positions"`, `"row_valid"`, …).
fn rt_tokens(t: &Trace, name: &str) -> model_ir::trace::ValueId {
    t.with(None, |b| {
        b.runtime_tensor(name, None, Shape(vec![Dim::Tokens]), DType::I32)
    })
}

/// Mint one per-fire `[Requests]` i32 stream by its vocabulary name
/// (`"qo_indptr"`, `"attn.score_indptr"`, …). CSR streams state
/// `[Requests]`; the driver stages the `+1` row the convention implies.
fn rt_requests(t: &Trace, name: &str) -> model_ir::trace::ValueId {
    t.with(None, |b| {
        b.runtime_tensor(name, None, Shape(vec![Dim::Requests]), DType::I32)
    })
}

/// The extent pair for a `num_requests`/`r` scalar the fire decides:
/// zero placeholder at `at`, spliced with the fire's request count.
fn requests_extent(at: u8) -> (u8, Shape) {
    (at, Shape(vec![Dim::Requests]))
}

/// The extent pair for a `rows`/`n_max` scalar: the fire's token rows.
fn tokens_extent(at: u8) -> (u8, Shape) {
    (at, Shape(vec![Dim::Tokens]))
}

/// One result's geometry, resolved from the ROUTINE's stated `out(..)` rule
/// against the statement's own operands — the trace-time half of B4-gen
/// (design-no-ask §10). `inputs` is the statement's operand run in slot
/// order; the rule's ordinals index into it.
///
/// Panics — at trace time, which is load time — when the rule does not
/// resolve, because a result the rule cannot shape must not become a
/// statement. `Unstated` never reaches here: an unruled result keeps its
/// `(Shape, DType)` parameter on the generated wrapper.
fn ruled_out(
    t: &Trace,
    routine: &str,
    rule: kernels::OutRule,
    inputs: &[model_ir::trace::ValueId],
    params: &[u32],
) -> (Shape, DType) {
    let b = t.inner.borrow();
    let shapes: Vec<Shape> = inputs.iter().map(|&id| b.value_shape(id)).collect();
    let dtypes: Vec<DType> = inputs.iter().map(|&id| b.value_dtype(id)).collect();
    let refs: Vec<&Shape> = shapes.iter().collect();
    model_ir::kernels::out_shape(rule, &refs, &dtypes, params).unwrap_or_else(|| {
        panic!("`{routine}`'s out rule does not resolve against this statement's operands")
    })
}

fn kv_state(kv: &Kv) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::KvCache,
        layer: kv.l,
    })
}

/// State mark for GDN ops.
fn rs_state(rs: &Rs) -> Option<StateRef> {
    Some(StateRef {
        store: StateStore::RecurrentState,
        layer: rs.l,
    })
}

/// `kernels::rope::rope_standard_table`: the positions stream is minted by
/// name; `[head_dim, theta]` is the run the routine's two `Const` marks claim.
pub fn rope_standard_table(t: &Trace, head_dim: u32, theta: f32) -> Val {
    let positions = rt_tokens(t, "positions");
    record_with_params(
        t,
        None,
        "rope::rope_standard_table",
        vec![],
        None,
        vec![head_dim, theta.to_bits()],
        vec![positions],
        Some((Shape(vec![Dim::Tokens, Dim::Const(head_dim)]), DType::F32)),
    )
    .expect("table launch produces a value")
}

/// `kernels::attn::qkv_decode_qk_norm_rope_write_kv_bf16`.
///
/// The swept signature reads, in order: `packed`, the rope table, the KV
/// view, positions and row validity — plus `[num_kv_heads, head_dim, theta,
/// eps]` on the params run. The table mark is NOT nullable, so a `table` of
/// `None` leaves the statement one operand short and `check_plan` refuses
/// the plan; callers that dispatch without a table must state one.
pub fn qkv_decode_qk_norm_rope_write_kv(
    packed: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    kv: &Kv,
    table: Option<&Val>,
    q_width: u32,
    num_kv_heads: u32,
    theta: f32,
) -> Val {
    let mut inputs = vec![packed.id];
    inputs.extend(table.map(|t| t.id));
    inputs.push(rt_object(&packed.t, "kv_cache", Some(kv.l)));
    inputs.push(rt_tokens(&packed.t, "positions"));
    inputs.push(rt_tokens(&packed.t, "row_valid"));
    let head_dim = q_norm
        .per_head
        .expect("a per-head q norm carries its head dim");
    record_with_params(
        &packed.t,
        Some(kv.l),
        "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        vec![q_norm.name.clone(), k_norm.name.clone()],
        kv_state(kv),
        vec![
            num_kv_heads,
            head_dim,
            theta.to_bits(),
            q_norm.eps.to_bits(),
        ],
        inputs,
        Some((Shape(vec![Dim::Tokens, Dim::Const(q_width)]), DType::BF16)),
    )
    .expect("fused post produces q")
}




/// Defines launch wrappers; `params` and `outs` are positional contracts.
macro_rules! builder {
    () => {};

    (
        $(#[$meta:meta])*
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> Val {
            symbol: $symbol:literal,
            on: $on:ident,
            $(weights: [$($weight:expr),* $(,)?],)?
            $(layer: $layer:expr,)?
            $(state: $state:expr,)?
            // `params` order is the kernel's positional contract.
            params: [$($param:expr),* $(,)?],
            inputs: [$($input:ident),* $(,)?],
            out: [$($dim:expr),* $(,)?] as $dtype:ident,
            made: $made:literal $(,)?
        }
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        #[must_use]
        pub fn $name($($arg: $ty),*) -> Val {
            record_with_params(
                &$on.t,
                builder!(@layer $on $(, $layer)?),
                $symbol,
                vec![$($($weight.to_string()),*)?],
                builder!(@state $($state)?),
                vec![$($param),*],
                vec![$($input.id),*],
                Some((Shape(vec![$($dim),*]), DType::$dtype)),
            )
            .expect($made)
        }
        builder! { $($rest)* }
    };

    (
        $(#[$meta:meta])*
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> Val {
            symbol: $symbol:literal,
            on: $on:ident,
            $(weights: [$($weight:expr),* $(,)?],)?
            $(layer: $layer:expr,)?
            $(state: $state:expr,)?
            inputs: [$($input:ident),* $(,)?],
            out: [$($dim:expr),* $(,)?] as $dtype:ident,
            made: $made:literal $(,)?
        }
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        #[must_use]
        pub fn $name($($arg: $ty),*) -> Val {
            record(
                &$on.t,
                builder!(@layer $on $(, $layer)?),
                $symbol,
                vec![$($($weight.to_string()),*)?],
                builder!(@state $($state)?),
                vec![$($input.id),*],
                Some((Shape(vec![$($dim),*]), DType::$dtype)),
            )
            .expect($made)
        }
        builder! { $($rest)* }
    };

    (
        $(#[$meta:meta])*
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> ($($ret:ty),+ $(,)?) {
            symbol: $symbol:literal,
            on: $on:ident,
            $(weights: [$($weight:expr),* $(,)?],)?
            $(layer: $layer:expr,)?
            params: [$($param:expr),* $(,)?],
            inputs: [$($input:ident),* $(,)?],
            // `outs` order is the returned tuple order.
            outs: [$([$($dim:expr),* $(,)?] as $dtype:ident),+ $(,)?],
            made: $made:literal $(,)?
        }
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        #[must_use]
        pub fn $name($($arg: $ty),*) -> ($($ret),+) {
            let outs = record_many_with_params(
                &$on.t,
                builder!(@layer $on $(, $layer)?),
                $symbol,
                vec![$($($weight.to_string()),*)?],
                vec![$($param),*],
                vec![$($input.id),*],
                vec![$((Shape(vec![$($dim),*]), DType::$dtype)),+],
            );
            let mut it = outs.into_iter();
            ($(builder!(@peel it, $made, $dtype)),+)
        }
        builder! { $($rest)* }
    };

    (
        $(#[$meta:meta])*
        pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) -> ($($ret:ty),+ $(,)?) {
            symbol: $symbol:literal,
            on: $on:ident,
            $(weights: [$($weight:expr),* $(,)?],)?
            $(layer: $layer:expr,)?
            inputs: [$($input:ident),* $(,)?],
            outs: [$([$($dim:expr),* $(,)?] as $dtype:ident),+ $(,)?],
            made: $made:literal $(,)?
        }
        $($rest:tt)*
    ) => {
        $(#[$meta])*
        #[must_use]
        pub fn $name($($arg: $ty),*) -> ($($ret),+) {
            let outs = record_many(
                &$on.t,
                builder!(@layer $on $(, $layer)?),
                $symbol,
                vec![$($($weight.to_string()),*)?],
                vec![$($input.id),*],
                vec![$((Shape(vec![$($dim),*]), DType::$dtype)),+],
            );
            let mut it = outs.into_iter();
            ($(builder!(@peel it, $made, $dtype)),+)
        }
        builder! { $($rest)* }
    };

    (@peel $it:ident, $made:literal, $dtype:ident) => { $it.next().expect($made) };

    (@layer $on:ident) => { $on.layer };
    (@layer $on:ident, $layer:expr) => { $layer };
    (@state) => { None };
    (@state $state:expr) => { $state };
}


mod attn;
mod base;
mod deepseek_v4;
mod gemma;
/// GENERATED named wrappers, one per traced `#[routine]` in
/// `crates/kernels-cuda/src` — see design-no-ask §10 (B4-gen). Deliberately
/// NOT glob-re-exported: callers opt in with `dsl::cuda::generated::`, so no
/// generated name can shadow a hand-written one while both exist.
pub mod generated;
mod mla;
mod moe;
mod qwen_3_5;
mod rope;
mod ssm;
mod tp;

pub use attn::*;
pub use base::*;
pub use deepseek_v4::*;
pub use gemma::*;
pub use mla::*;
pub use moe::*;
pub use qwen_3_5::*;
pub use rope::*;
pub use ssm::*;
pub use tp::*;
