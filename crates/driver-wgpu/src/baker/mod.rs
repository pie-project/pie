//! The lane, at this plane: a `model_compiler::program::Program` per fire
//! class, built at load and walked per fire.
//!
//! # What this is
//!
//! `driver-cuda/src/baker/` is the reference implementation, `driver-metal`'s
//! is the first shader plane to follow it, and this is that capability spelled
//! in WebGPU. The chain is the same one end to end — text → `Plan` → lane →
//! `Program` → walk → the plane's claim bodies — and the shape of every module
//! beside this one matches its two siblings by name, so that a decision made
//! once is findable three times.
//!
//! # HALF OF IT IS NOT HERE ANY MORE, AND THAT IS THE POINT
//!
//! [`crate::walk`] holds the walk, the bound statement, the frame's numbering,
//! the resolve pass and the lane. That crate exists because this driver's copy
//! of those five and `driver-metal`'s came out 96% identical — `frame.rs` at
//! 100%, to the character — and the 4% was a crate name, a payload type and two
//! refusal strings. This file's own `frame.rs` said what to do about it: *"If a
//! third plane wants it, that is the moment to weigh a move."* `driver-vulkan`
//! is the third plane.
//!
//! WHAT STAYED IS WHAT DIFFERS, and each of the five is a measurement rather
//! than a preference:
//!
//! * [`marks`] — a wgpu region NAMES ITS BUFFER, because a
//!   `wgpu::BufferBinding` is an object and two offsets and there is no address
//!   space to do arithmetic in. Metal's is an address and an extent.
//! * [`stage`] — this driver stages a fourteenth plane for its split decode
//!   ([`stage::FireTable::AttnPartials`]) and states a KV head count metal's
//!   pool does not carry.
//! * [`views`] — `Plane::Pages` here is [`views::attn_fire`]'s `AttnFireView`:
//!   the pool row PLUS the five per-fire planes every sdpa arm reads, because a
//!   `Ctx` that is `dyn Encode` has no env for a body to pull them off. Metal's
//!   is the pool row alone.
//! * [`encode`] and [`dispatch`] — a `Dispatch` here carries LANES and no
//!   hazard set: WGSL DECLARES its workgroup size where MSL leaves it to the
//!   driver, and wgpu-core emits a barrier before every dispatch and will not
//!   be told not to. Metal carries total threads and a `Touches` set.
//!
//! And [`plane`] is the join: two impls carrying the region, the encoder, the
//! census, the staging door and the dispatch, so the list of what makes this
//! plane different is a thing a reader can hold in one screen.
//!
//! # THE MEASUREMENT THIS EXECUTOR EXISTS TO MAKE, AND WHAT IT SAYS TODAY
//!
//! THE HEADLINE GAP WAS `gemm.matmul` AND IT IS SHUT. What stood here said
//! **there is no dense matmul on this plane at all** — every matmul in
//! `kernels/quant/` reads a bank as three weights while the three `Gemm` points
//! declare `Const<Self::Tensor<T>>`, which `kernels_wgpu::quant` called "one
//! TYPE the floor does not have" rather than three shaders nobody wrote. It was
//! right, and `kernels/gemm/dense.wgsl` is that type answered rather than
//! worked around: the weight arrives as the dense tensor the declaration
//! states, and two entry points read it — a staged 32x32x32 tile for
//! `M >= 32` and a K-split vector arm below it.
//!
//! No catalog row's lane binds YET, and the reason has moved: `gemm.matmul`,
//! `layout.embed`, `layout.select` and `layout.split_rows` no longer appear in
//! any lane's refusal list, and what does is each SKU's own family —
//! `mlp.geglu_tanh_packed` and `norm.mul_scalar` for gemma, `moe.*` and
//! `rope.yarn` for gpt-oss, `ssm.*` for qwen3.5. The catalog walk at the foot
//! of this file is still the measurement and still reports none binding;
//! [`crate::walk::resolve::check`] is what turns a gap into a load-time
//! sentence naming the point and the first statement that asked.
//!
//! What is CHECKED rather than asserted: the walk is mutation-checked with no
//! adapter (`tests/the_walk_is_the_program.rs`), and `norm.rmsnorm` is driven
//! through this whole path onto a real adapter and compared against a host
//! reference (`tests/device_fire.rs`).
//!
//! # ONE CATALOG (R3), and this driver joins it at P5b
//!
//! The pools used to be the OTHER catalog's, and a `WgpuBinding` carried what
//! the LOAD had observed back into the text a driver ran. Every one of those
//! pieces is gone; see [`crate::walk::lane`] for what replaced them.
//!
//! # The plane is named, not observed
//!
//! `model::trace_of(sku)` takes a `model_ir::kernels::Backend`, and this driver
//! hands it `Wgpu` — a variant P5b added for exactly this call, through
//! [`Wgpu`], the type [`Baked::of`] is instantiated at. That one argument is
//! the whole of what used to be the binding: which plane's claim tables
//! `sweep::resolve` joins a lane's points against, and therefore which lanes
//! bind at all. What a bank is stored as rides on the plan's own `repr` column
//! and is read at the slot (`BoundOp::form`), so a driver no longer measures a
//! quantisation and hands its answer back to the catalog.

pub mod dispatch;
pub mod encode;
pub mod marks;
pub mod plane;
pub mod stage;
pub mod views;

pub use crate::walk::frame;
pub use crate::walk::{BANK_ALIGN, Baked, READABLE_BASE, arena_of, join, readable_base, word_of};
pub use marks::{Bindings, Bound as BoundRegion, BufferId, NOTHING, Rect, Slice};
pub use plane::Wgpu;
pub use stage::{FireTable, KvGeometry, Pools, Slab, Splits};

/// A weight on the device, in this plane's regions.
pub type Bank = crate::walk::Bank<Slice>;

/// The walk, bound to this plane.
///
/// A MODULE OF ALIASES AND NOT A RE-EXPORT, so that `baker::walk::Fire` still
/// names one thing rather than a generic a reader has to instantiate in their
/// head. What is behind each name is [`crate::walk::fire`]'s, at [`Wgpu`].
pub mod walk {
    use super::{Slice, Wgpu};

    pub use crate::walk::fire::{Cursor, Refused};

    /// Everything one fire of the baker addresses.
    pub type Fire<'a> = crate::walk::fire::Fire<'a, Wgpu>;

    /// How big one fire is — the half of a fire no plan holds.
    pub type Extent = crate::walk::fire::Extent<Slice>;

    /// A copy an `InOut` point forced, in walk order.
    pub type Blit = crate::walk::fire::Blit<Slice>;
}

/// The eager resolve pass, at this plane — and the census assertions that are
/// about THIS plane's claim tables rather than about the pass.
pub mod resolve {
    use model_compiler::program::Program;
    use model_ir::plan::Plan;

    pub use crate::walk::resolve::Unresolved;

    use super::Wgpu;

    /// Check every step of `program` against the plane this driver fires.
    ///
    /// See [`crate::walk::resolve::check`], which this names `Wgpu` for.
    #[must_use]
    pub fn check(plan: &Plan, program: &Program) -> Vec<Unresolved> {
        crate::walk::resolve::check::<Wgpu>(plan, program)
    }

    #[cfg(test)]
    mod tests {
        use std::collections::BTreeSet;

        use kernels::bound::Site;
        use kernels_wgpu::points_dispatch::{CLAIMED, TIER2};
        use model_compiler::program::Call;

        /// Every point the generated census claims is one the compiler can
        /// actually route to a point call ON THIS PLANE.
        ///
        /// The failure this catches is a family whose claim table names
        /// something `model_compiler` does not route as a point — a spelling
        /// that resolves nothing, whose only symptom is a load-time refusal for
        /// a point that IS implemented. It also checks the thing P5b added:
        /// that `model_ir::kernels::Backend` grew a `Wgpu` row and that
        /// `point_claims(Wgpu)` reaches THIS crate's tables.
        #[test]
        fn every_claimed_point_is_a_point_the_compiler_routes() {
            for (point, _, _) in CLAIMED {
                let call =
                    model_compiler::program::call_of(model_ir::kernels::Backend::Wgpu, point);
                assert_eq!(
                    call,
                    Some(Call::Point((*point).to_string())),
                    "`{point}` is claimed by this plane but `call_of` answers {call:?}",
                );
            }
        }

        /// THIS PLANE DECLARES NO TIER-2 SURFACE, and it cannot.
        ///
        /// Stronger than metal's version of this test. There, a tier-2 point is
        /// declared by an inherent `impl Ctx<'_>` block that `kernels-metal`
        /// simply has not written. Here `Ctx<'a>` is `dyn Encode + 'a`, and
        /// Rust has no inherent impl for a trait object — so the row that would
        /// make this fail cannot be written at all, and the assertion is a
        /// statement about the generator rather than about restraint.
        #[test]
        fn the_tier2_census_is_empty_because_this_plane_can_state_none() {
            assert!(
                TIER2.is_empty(),
                "this plane grew a tier-2 surface: {:?}",
                TIER2.iter().map(|(p, _, _)| *p).collect::<Vec<_>>(),
            );
        }

        /// The census is a set: a duplicated point would make the second row
        /// unreachable and its elements silently unclaimed.
        #[test]
        fn the_claim_table_names_each_point_once() {
            let mut seen = BTreeSet::new();
            for (point, _, _) in CLAIMED {
                assert!(seen.insert(*point), "`{point}` is claimed twice");
            }
        }

        /// THE WITNESS IS THE GENERATOR'S, and this is the row that proves the
        /// census carries it rather than assuming the result.
        ///
        /// `attention.kv_append` states no result at all, so a pass that read
        /// `outputs.first()` would have had nothing to read and answered `None`
        /// — which reads as "claimed by name" and would have been the RIGHT
        /// answer for the wrong reason.
        #[test]
        fn the_census_reads_the_witness_the_arm_reads() {
            let of = |point: &str| {
                CLAIMED
                    .iter()
                    .find(|(p, _, _)| *p == point)
                    .map(|(_, w, _)| *w)
            };
            assert_eq!(of("attention.kv_append"), Some(Some(Site::In(0))));
            assert_eq!(of("norm.rmsnorm"), Some(Some(Site::Out(0))));
        }

        /// THE GAP CLOSED, AND THIS IS THE ROW THAT SAYS WHICH POINTS SHUT IT.
        ///
        /// This row used to be named for the gap and assert the EMPTY set of
        /// `gemm.` claims, because every SKU in the catalog states
        /// `gemm.matmul` and this tree had no dense matmul at all — every
        /// matmul it stamped read a bank as three weights, which was called
        /// "one TYPE the floor does not have" rather than three shaders nobody
        /// wrote. `kernels/gemm/dense.wgsl` is that type answered: the weight
        /// arrives as the `Const<Self::Tensor<T>>` the declaration states, and
        /// two entry points read it.
        ///
        /// The replacement is the same measurement pointed the other way. All
        /// THREE `Gemm` points are claimed and no other is, so a fourth
        /// appearing — or one of these three going away — is a change to this
        /// plane's headline that fails here rather than passing quietly.
        #[test]
        fn the_gemm_family_is_claimed_whole_and_that_is_what_shut_the_gap() {
            let gemm: Vec<&str> = CLAIMED
                .iter()
                .map(|(p, _, _)| *p)
                .filter(|p| p.starts_with("gemm."))
                .collect();
            assert_eq!(
                gemm,
                ["gemm.matmul", "gemm.lm_head", "gemm.attention_landing"],
                "the `Gemm` family this plane answers changed",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY CATALOG ROW TRACES FOR THIS PLANE, and none of them binds a lane.
    ///
    /// The catalog rows with at least one lane that binds on this plane.
    ///
    /// A list and not a count, because WHICH rows bind says which families
    /// landed: the two gemma rows and the three qwen rows state the dense
    /// tower — embed, gemm, the norms, rope, the packed activations — and the
    /// routed qwen row puts the moe family on top of it.
    ///
    /// THE TWO GPTOSS ROWS ARE THE NEWEST TWO, and they are here for exactly
    /// three points. Their text states `attention.{decode_lse, prefill_lse,
    /// sink}` — publish the softmax denominator, then rescale the output by a
    /// learned per-head sink against it — and this plane claimed none of the
    /// three, so both rows failed to bind on an attention statement while every
    /// other family they name was already answered. `attn/attn_sink.wgsl` and
    /// the two `_lse` stamps in `attn/sdpa_paged.wgsl` are what moved them, and
    /// the trace diff reads 16/16 for both where it read 13/16.
    const BOUND: &[&str] = &[
        "gemma4-e4b-bf16-kv-bf16",
        "gemma4-31b-bf16-kv-bf16",
        "gptoss-20b-bf16-mxfp4-kv-bf16",
        "gptoss-120b-bf16-mxfp4-kv-bf16",
        "qwen35-a3b-bf16-kv-bf16",
        "qwen35-d3b-bf16-kv-bf16",
        "qwen35-d0.8b-bf16-kv-bf16",
    ];

    /// Two assertions in one walk, and the second one changed.
    ///
    /// Tracing is the plane-naming path — `trace(Backend::Wgpu)` — and it must
    /// work for every row whose pools `model::deployment` can describe.
    /// Binding is the claim join, and it used to fail for EVERY row because no
    /// `Gemm` point was claimed here. This test said in its own words that a
    /// dense matmul landing would be its notification.
    ///
    /// IT LANDED, with `ssm` whole and the packed activations and the routed
    /// points beside it — 21 claimed points to 50 in one wave — and five rows
    /// bind. The assertion is INVERTED rather than deleted, into [`BOUND`], so
    /// that a regression which un-binds a row is as loud as the landing was.
    ///
    /// BINDING IS NOT SERVING, and the message this test used to carry asked
    /// for both: "one that says which lanes bind and fires them". This is the
    /// first half. A lane that binds has an answer for every point it states;
    /// whether the whole tower computes the right tokens is what
    /// `scripts/banked-argmaxes.sh` asks of cuda, and no shader plane can be
    /// asked it yet.
    #[test]
    fn every_catalog_row_traces_for_this_plane_and_the_bound_ones_are_named() {
        let mut bound_rows = Vec::new();
        let mut traced = 0usize;
        for row in model::serve::ROWS {
            let Ok(baked) = Baked::of::<Wgpu>(row.id) else {
                // A row `Deployment::of` refuses is a fact about the POOL and
                // not about this plane; `driver-metal`'s twin of this test
                // skips the same set for the same reason.
                continue;
            };
            traced += 1;
            assert_eq!(baked.plan.plane, model_ir::kernels::Backend::Wgpu);
            assert!(
                !baked.plan.ops.is_empty(),
                "`{}` traced an empty plan",
                row.id
            );
            if baked.lanes.iter().any(Result::is_ok) {
                bound_rows.push(row.id);
            }
        }
        assert!(traced > 0, "no catalog row traced for this plane at all");
        bound_rows.sort_unstable();
        let mut want = BOUND.to_vec();
        want.sort_unstable();
        assert_eq!(
            bound_rows, want,
            "the rows whose lanes bind on this plane have moved. A row GAINED \
             belongs in `BOUND` — that edit is the record of the point that \
             landed. A row LOST is a claim this plane used to answer and does \
             not, which is a regression rather than a list to edit.",
        );
    }
}
