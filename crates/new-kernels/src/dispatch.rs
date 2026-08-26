//! One trait per op family, every method named `dispatch`; only the
//! aggregate's `exec` is called externally, so the shared name never needs
//! disambiguating outside this file (decision #14).

use new_model_ir::{
    Attention, Cuda, Dist, Gate, Gemm, Hc, Index, Layout, Mla, Mlp, Moe, Node, Norm, Operation,
    Pool, Rope, Ssm,
};

use crate::error::KernelError;

/// One list, three artifacts: the 15 family traits, the aggregate requiring
/// them all, and the blanket impl granting `exec` to any type carrying the
/// full set. Pairs are spelled out because `macro_rules!` cannot concatenate
/// `Dispatch` onto a family name — and the table doubles as the contract's
/// index.
macro_rules! dispatch {
    ($($family:ident => $trait:ident),* $(,)?) => {
        $(
            #[doc = concat!(
                "Enqueue one [`", stringify!($family), "`] op. ",
                "[`Dispatch`] states the standing rules."
            )]
            pub trait $trait {
                fn dispatch(&mut self, op: &$family) -> Result<(), KernelError>;
            }
        )*

        /// The whole contract, one bound. Two standing rules:
        ///
        /// - `dispatch` means **enqueue/encode only, never sync** (#15) —
        ///   CUDA graph capture and Metal command buffers depend on it.
        /// - Impls live in `driver-*` on that driver's `Run` type, and arms
        ///   stay dumb: destructure → resolve → call (#13). Kernel selection
        ///   (arch, gemv-vs-dense, dtype) belongs inside the `kernels-*`
        ///   entry fn; an `if arch >= 90` in an arm is logic leaking upward.
        ///
        /// A backend-specific family on a foreign `Run` answers
        /// [`KernelError::Unsupported`] from its impl — the match below is
        /// total by construction, never partial.
        pub trait Dispatch: $($trait +)* {
            /// Enqueue one node's op. UFCS throughout — a method call would
            /// be ambiguous across the same-named supertraits. `cond` and
            /// `layer` are the driver walk's business; this reads only `op`.
            fn exec(&mut self, node: &Node) -> Result<(), KernelError> {
                match &node.op {
                    $(Operation::$family(op) => $trait::dispatch(self, op),)*
                }
            }
        }

        impl<T: $($trait +)*> Dispatch for T {}
    };
}

dispatch! {
    Norm      => DispatchNorm,
    Mlp       => DispatchMlp,
    Gemm      => DispatchGemm,
    Dist      => DispatchDist,
    Rope      => DispatchRope,
    Moe       => DispatchMoe,
    Gate      => DispatchGate,
    Layout    => DispatchLayout,
    Ssm       => DispatchSsm,
    Attention => DispatchAttention,
    Mla       => DispatchMla,
    Index     => DispatchIndex,
    Pool      => DispatchPool,
    Hc        => DispatchHc,
    Cuda      => DispatchCuda,
}
