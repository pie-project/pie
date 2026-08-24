//! The forward-pass authoring surface: model texts state backend-neutral
//! role points onto a [`Plan`]; planes answer them with claims. See
//! `.wiki/baker.md` for the design this crate is converging to — the
//! hand-written [`kernels`] surface is scheduled to be generated from the
//! kernel crates' own declarations, family by family.

#![allow(clippy::too_many_arguments)]

pub mod declare;
pub mod facts;
pub mod forward;
pub mod kernels;
pub mod load;
mod record;

pub use declare::*;
pub use facts::*;
pub use forward::*;
pub use model_dsl_macros::Facts;
pub use model_ir::kernels::Backend as Plane;
pub use model_ir::plan::Plan;
pub use record::{Value, Windows};

pub mod axes {
    /// One STORED PLANE of a bank: what a `Const` bank slot binds, and what
    /// the plan registers a parameter for.
    ///
    /// A dense weight is one plane and always was — the tensor's own name,
    /// the tensor's own shape. A quantised one is as many planes as its repr
    /// stores, which for mxfp4 is two: the packed codes and their block
    /// exponents ship as two checkpoint tensors, load as two allocations and
    /// reach a kernel as two addresses, so they are two parameters and not
    /// one row with a split point somebody has to recompute.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct BankPlane {
        /// Appended to the bank's declared name. Empty for the first plane,
        /// so a dense weight's parameter name is unchanged and a quantised
        /// bank's leading plane keeps the name the text spells.
        pub suffix: &'static str,

        /// What this plane's bytes are shaped as, which for a quantised
        /// plane is NOT the logical rectangle: 32 codes land as 16 bytes and
        /// one exponent.
        pub shape: Vec<u64>,

        /// The plan's repr column for this plane, which is not always the
        /// axis's own `NAME`: the mxfp4 scale plane is `e8m0` and calling it
        /// `mxfp4` would be the join reporting agreement it never checked.
        pub repr: &'static str,
    }

    /// One weight-bearing axis: what the checkpoint holds for a bank of
    /// projections. `NAME` doubles as the plan's repr column and joins
    /// into the catalogued SKU string.
    pub trait Dtype: 'static {
        const NAME: &'static str;

        /// The name suffixes this repr's stored planes are bound under, in
        /// the repr's own order. One empty suffix for a dense weight, which
        /// is why every existing parameter name is unchanged.
        ///
        /// SEPARATE FROM [`Dtype::planes`] BECAUSE THE SHAPE IS. What a
        /// plane is CALLED is a fact about the repr alone; what SHAPE it has
        /// needs the logical rectangle. An import table knows the first and
        /// not the second — it names checkpoint tensors, it does not size
        /// them — so `Import::bank` reads this list and `planes` below reads
        /// it too.
        const PLANE_SUFFIXES: &'static [&'static str] = &[""];

        /// How a bank at this repr is STORED, as the planes a `Bank` slot
        /// binds it through.
        ///
        /// `shape` is the LOGICAL rectangle a model text declares —
        /// `[.., N, K]`, the shape the arithmetic has — and the answer is
        /// what the bytes actually look like. The default is the dense
        /// reading: one plane, that rectangle, under that name, which is
        /// every `Const<Tensor<..>>` weight in every text.
        ///
        /// THIS IS THE REPR'S DEFINITION AND IT LIVES HERE ONCE. The import
        /// table states where the planes come FROM; the kernel states what it
        /// does WITH them; the shape each one has is a property of the repr
        /// itself and neither of those two gets to spell it.
        #[must_use]
        fn planes(shape: &[u64]) -> Vec<BankPlane> {
            vec![BankPlane {
                suffix: Self::PLANE_SUFFIXES[0],
                shape: shape.to_vec(),
                repr: Self::NAME,
            }]
        }
    }

    /// The KV cache axis: which scheme the pages hold — a load-time fact,
    /// stated where the SKU is named.
    pub trait KvDtype: 'static {
        const NATIVE_BF16: bool;
        const NAME: &'static str;
    }

    /// Plain bf16: the repr every dense row ships today.
    pub enum Bf16 {}
    impl Dtype for Bf16 {
        const NAME: &'static str = "bf16";
    }

    /// Plain f32, and NOT a model-wide activation axis: this is the axis a
    /// handful of individual banks wear while everything around them is
    /// bf16, because a KERNEL reads them through a `float*`.
    ///
    /// Qwen3.5's gated-delta mixer has both of them. `norm.rmsnorm_gated`
    /// declares `weight: Const<Self::Tensor<f32>>` at the floor and
    /// `kernels-cuda`'s `rmsnorm_gated_fp32_in` claims it at that dtype;
    /// `ssm.gdn_prep`'s `a_log` is the same story one point over
    /// (`ssm/gated_delta_net_prep.cuh`: `const float* __restrict__
    /// A_log`). The shipped checkpoint agrees -- both ship F32 in an
    /// otherwise BF16 file -- so a model that declared them `Bf16` would
    /// be stating a cast that no kernel performs and no checkpoint wants.
    ///
    /// A `W1` type parameter must NOT be instantiated at this: it is worn
    /// by named fields whose dtype is fixed, never by the family's weight
    /// axis.
    pub enum F32 {}
    impl Dtype for F32 {
        const NAME: &'static str = "f32";
    }

    /// OCP MX FP4 experts: 4-bit E2M1 codes two to a byte, one E8M0
    /// exponent byte per 32 codes. gpt-oss's shipped form, VERBATIM.
    ///
    /// NOT "MARLIN LAYOUT", which is what this doc used to say and what the
    /// legacy repr enum is still named after. The Marlin repack is reached
    /// only through `Mxfp4MoePolicy::NativeGemm`, which is gated on a
    /// `native_mxfp4_moe` capability that every driver in this tree states
    /// FALSE — cuda, metal, vulkan and wgpu alike — so no shipped path has
    /// ever repacked a byte. What ships is `push_direct` of the checkpoint's
    /// own `_blocks` and `_scales`, which is exactly what the planes below
    /// describe and what `quant/dequant_fp4.cuh` reads.
    pub enum Mxfp4 {}
    impl Dtype for Mxfp4 {
        const NAME: &'static str = "mxfp4";

        const PLANE_SUFFIXES: &'static [&'static str] = &["", ".scales"];

        /// `[.., N, K]` → codes `[.., N, K/32, 16]` + scales `[.., N, K/32]`.
        ///
        /// RANK 4 AND NOT A FLATTENED `[.., N, K/2]`, because the checkpoint
        /// ships rank 4 and the canonical form is the checkpoint's bytes
        /// wherever nothing has to move them. Keeping the group axis makes
        /// the import a pure byte move and makes the join CHECK the block
        /// size rather than assume it — a `[.., N, K/32, 16]` demand against
        /// a differently blocked release is a shape mismatch, where a
        /// flattened byte count would have matched anything with the same
        /// total.
        fn planes(shape: &[u64]) -> Vec<BankPlane> {
            let (&k, lead) = shape
                .split_last()
                .expect("an mxfp4 bank's logical shape ends in its contracted axis");
            assert!(
                k % 32 == 0,
                "an mxfp4 bank contracts over {k}, which is not a whole number of \
                 32-code blocks"
            );
            let groups = k / 32;
            let mut codes = lead.to_vec();
            codes.push(groups);
            codes.push(16);
            let mut scales = lead.to_vec();
            scales.push(groups);
            vec![
                BankPlane {
                    suffix: Self::PLANE_SUFFIXES[0],
                    shape: codes,
                    repr: Self::NAME,
                },
                BankPlane {
                    suffix: Self::PLANE_SUFFIXES[1],
                    shape: scales,
                    // NOT `mxfp4`, and the join is why: this plane is a run
                    // of E8M0 exponent BYTES, so calling it by the codes'
                    // repr would have `baker_load` reporting agreement it
                    // never checked.
                    repr: "e8m0",
                },
            ]
        }
    }

    /// WNA16 int4 (glm's routed experts).
    pub enum Wna16 {}
    impl Dtype for Wna16 {
        const NAME: &'static str = "wna16";
    }

    /// Native bf16 KV pages.
    pub enum NativeKv {}
    impl KvDtype for NativeKv {
        const NATIVE_BF16: bool = true;
        const NAME: &'static str = "kv-bf16";
    }
}

/// Enumerate one shipping SKU: a name, the trace fn, and the monomorphized
/// model it instantiates. The row is a `(name, fn(Plane) -> Plan)` pair in
/// a plain table the family's cover file exposes.
#[macro_export]
macro_rules! catalog {
    ($( ($name:literal, $trace:path, $m:expr $(,)?) ),+ $(,)?) => {
        &[ $( ($name, (|plane| $trace($name, &$m, plane)) as _) ),+ ]
    };
}

pub mod seam {
    pub use model_ir::seam::*;

    use crate::record::Value;

    pub trait Sees {
        fn values(&self) -> Vec<&Value>;
    }

    impl Sees for (&Value,) {
        fn values(&self) -> Vec<&Value> {
            vec![self.0]
        }
    }

    impl Sees for (&Value, &Value) {
        fn values(&self) -> Vec<&Value> {
            vec![self.0, self.1]
        }
    }

    pub fn at<S: Sees>(def: Def, sees: S, layer: u32) {
        let values = sees.values();
        values[0].rec.seam(def.name, &values, Some(layer));
    }
}
