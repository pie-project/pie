//! CUDA launch statements for model DSL traces.
//!
//! Functions state semantic operands; driver-owned stream, dims, workspace,
//! and prepare work are bound by the driver contract.

use crate::{ConvW, Kv, MatW, NormW, Rs, Trace, Val};
use model_ir::trace::{DType, Dim, NormVariant, Shape, StateRef, StateStore};

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

/// `kernels::rope::rope_standard_table`.
pub fn rope_standard_table(t: &Trace, head_dim: u32) -> Val {
    record(
        t,
        None,
        "rope::rope_standard_table",
        vec![],
        None,
        vec![],
        Some((Shape(vec![Dim::Tokens, Dim::Const(head_dim)]), DType::F32)),
    )
    .expect("table launch produces a value")
}

/// `kernels::attn::qkv_decode_qk_norm_rope_write_kv_bf16`.
pub fn qkv_decode_qk_norm_rope_write_kv(
    packed: &Val,
    q_norm: &NormW,
    k_norm: &NormW,
    kv: &Kv,
    table: Option<&Val>,
    q_width: u32,
) -> Val {
    let mut inputs = vec![packed.id];
    inputs.extend(table.map(|t| t.id));
    record(
        &packed.t,
        Some(kv.l),
        "attn::qkv_decode_qk_norm_rope_write_kv_bf16",
        vec![q_norm.name.clone(), k_norm.name.clone()],
        kv_state(kv),
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
