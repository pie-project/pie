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
//! `kernels-wgpu` claims 21 of the floor's 81 points, and no catalog row's lane
//! binds on this plane. Every SKU refuses at `gemm.matmul` before it refuses
//! anywhere else, because **there is no dense matmul on this plane at all**:
//! every matmul in `kernels/quant/` reads a bank as three weights, and the
//! three `Gemm` points declare `Const<Self::Tensor<T>>`. `kernels_wgpu::quant`
//! calls that "one TYPE the floor does not have" rather than three shaders
//! nobody wrote, and it is right.
//!
//! So this driver serves nothing yet, and that is a MEASUREMENT rather than a
//! defect — the same one `driver-metal` landed with at P5a ("0/35 metal lanes
//! BUILD; gemm/Bank gates every SKU — asserted so the day one binds is loud").
//! [`crate::walk::resolve::check`] is what turns it into a load-time sentence
//! naming the point and the first statement that asked, and [`resolve`]'s own
//! tests assert the gap so its closing is loud.
//!
//! What DOES work is everything either side of that gap, and it is checked
//! rather than asserted: the walk is mutation-checked with no adapter
//! (`tests/the_walk_is_the_program.rs`), and one claimed point —
//! `norm.rmsnorm` — is driven through this whole path onto a real adapter and
//! compared against a host reference (`tests/device_fire.rs`).
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

        /// THE GAP, ASSERTED SO THE DAY IT CLOSES IS LOUD.
        ///
        /// `kernels-wgpu` claims no `Gemm` point, which is why no catalog row's
        /// lane binds on this plane: every SKU in the catalog states
        /// `gemm.matmul`, and this tree has no dense matmul at all — every
        /// matmul it stamps reads a bank as three weights. That is not three
        /// shaders nobody wrote; it is one TYPE the floor does not have at
        /// these points' declarations.
        ///
        /// This is the same shape as `driver-metal`'s "0/35 lanes BUILD"
        /// assertion and it is here for the same reason: a measurement worth
        /// failing on when it changes.
        #[test]
        fn no_gemm_point_is_claimed_and_that_is_this_planes_headline_gap() {
            let gemm: Vec<&str> = CLAIMED
                .iter()
                .map(|(p, _, _)| *p)
                .filter(|p| p.starts_with("gemm."))
                .collect();
            assert!(
                gemm.is_empty(),
                "this plane grew a `Gemm` claim: {gemm:?} — if a dense matmul now \
                 exists here, the catalog's lanes may bind and this test should be \
                 replaced by one that says which",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY CATALOG ROW TRACES FOR THIS PLANE, and none of them binds a lane.
    ///
    /// Two assertions in one walk, and the second is the one that will change.
    /// Tracing is the plane-naming path — `trace(Backend::Wgpu)` — and it must
    /// work for every row whose pools `model::deployment` can describe. Binding
    /// is the claim join, and it fails for every row today because no `Gemm`
    /// point is claimed here.
    ///
    /// When a dense matmul lands on this plane, THIS TEST IS THE NOTIFICATION.
    #[test]
    fn every_catalog_row_traces_for_this_plane_and_none_binds_yet() {
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
        assert!(
            bound_rows.is_empty(),
            "these rows now BIND a lane on the wgpu plane: {bound_rows:?} — that is \
             the day this driver can serve something, and this test is how you find \
             out. Replace it with one that says which lanes bind and fires them.",
        );
    }
}
