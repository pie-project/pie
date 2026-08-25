//! The lane, at this plane: a `model_compiler::program::Program` per fire
//! class, built at load and walked per fire.
//!
//! # What this is
//!
//! `driver-cuda/src/baker/` is the reference implementation and this is that
//! capability spelled in Metal. The chain is the same one end to end — text →
//! `Plan` → lane → `Program` → walk → the plane's claim bodies — and the shape
//! of every module beside this one matches its cuda sibling by name, so that a
//! decision made once is findable twice.
//!
//! # HALF OF IT IS NOT HERE ANY MORE, AND THAT IS THE POINT
//!
//! [`crate::walk`] holds the walk, the bound statement, the frame's numbering,
//! the resolve pass and the lane. That crate exists because
//! `driver-wgpu/src/baker/` was written against the same cuda reference five
//! months later and came out 96% identical to this one — `frame.rs` at 100%, to
//! the character — and the 4% was a crate name, a payload type and two refusal
//! strings.
//!
//! WHAT STAYED IS WHAT DIFFERS, and each of the five is a measurement rather
//! than a preference:
//!
//! * [`marks`] — a Metal region is an ADDRESS and an extent, because a buffer is
//!   bound by a GPU virtual address. A `wgpu::BufferBinding` is an object and
//!   two offsets and there is no address space at all.
//! * [`stage`] — this driver stages thirteen planes; the wgpu one stages a
//!   fourteenth for its split decode, and states a KV head count this pool does
//!   not carry.
//! * [`views`] — `Plane::Pages` here is the pool row PLUS three per-fire
//!   planes (`AttnFireView`), because this plane's sdpa arms read the
//!   positions, the owning request and the mask triple off the same object.
//!   The wgpu sibling folds five, and the two it has that this does not are
//!   its split-decode partials and a KV head count; there are no split entry
//!   points here, and the count divides out of the strides the pool row
//!   already carries.
//! * [`encode`] and [`dispatch`] — a `Dispatch` here carries TOTAL THREADS and a
//!   [`dispatch::Touches`] hazard set, because a Metal compute encoder runs
//!   dispatches concurrently by default and this driver decides where the
//!   barriers go. wgpu-core emits the barrier itself and will not be told not
//!   to, so its sibling carries LANES and no hazard set at all.
//!
//! And [`plane`] is the join: one impl of [`crate::walk::Plane`] carrying the
//! region, the encoder, the census, the staging door and the dispatch, so the
//! list of what makes this plane different is a thing a reader can hold in one
//! screen.
//!
//! # ONE CATALOG (R3), and this driver joins it at P5
//!
//! The pools used to be the OTHER catalog's: `model_legacy::deployment` sized
//! and strided the KV pages and the recurrent slabs off a config-projected
//! registry, while a `MetalBinding` carried what the LOAD had observed back
//! into `row.trace(class, Deployed::metal(&binding))` so that the text a driver
//! ran depended on what its own loader had found in the checkpoint. Every one
//! of those pieces is gone; see [`crate::walk::lane`] for what replaced them.
//!
//! # The plane is named, not observed
//!
//! `model::trace_of(sku)` takes a `model_ir::kernels::Backend`, and this driver
//! hands it `Metal` — through [`Metal`], the type [`Baked::of`] is instantiated
//! at. That one argument is the whole of what used to be the binding: which
//! plane's claim tables `sweep::resolve` joins a lane's points against, and
//! therefore which lanes bind at all. What a bank is stored as rides on the
//! plan's own `repr` column and is read at the slot (`BoundOp::form`), so a
//! driver no longer measures a quantisation and hands its answer back to the
//! catalog.

pub mod dispatch;
pub mod encode;
pub mod marks;
pub mod plane;
pub mod stage;
pub mod views;

pub use crate::walk::frame;
pub use crate::walk::{
    Arena, BANK_ALIGN, Baked, READABLE_BASE, arenas_of, join, readable_base, word_of,
};
pub use marks::{Bindings, Bound as BoundRegion, NOTHING, Rect, Slice};
pub use plane::Metal;
pub use stage::{FireTable, KvGeometry, Pools, Slab};

/// A weight on the device, in this plane's regions.
pub type Bank = crate::walk::Bank<Slice>;

/// The walk, bound to this plane.
///
/// A MODULE OF ALIASES AND NOT A RE-EXPORT, so that `baker::walk::Fire` still
/// names one thing rather than a generic a reader has to instantiate in their
/// head. What is behind each name is [`crate::walk::fire`]'s, at [`Metal`].
pub mod walk {
    use super::{Metal, Slice};

    pub use crate::walk::fire::{Cursor, Refused};

    /// Everything one fire of the baker lane addresses.
    pub type Fire<'a> = crate::walk::fire::Fire<'a, Metal>;

    /// How big one fire is — the half of a fire no plan holds.
    pub type Extent = crate::walk::fire::Extent<Slice>;

    /// A blit an `InOut` point forced, in walk order.
    pub type Blit = crate::walk::fire::Blit<Slice>;
}

/// The eager resolve pass, at this plane — and the census assertions that are
/// about THIS plane's claim tables rather than about the pass.
pub mod resolve {
    use model_compiler::program::Program;
    use model_ir::plan::Plan;

    pub use crate::walk::resolve::Unresolved;

    use super::Metal;

    /// Check every step of `program` against the plane this driver fires.
    ///
    /// See [`crate::walk::resolve::check`], which this names `Metal` for.
    #[must_use]
    pub fn check(plan: &Plan, program: &Program) -> Vec<Unresolved> {
        crate::walk::resolve::check::<Metal>(plan, program)
    }

    #[cfg(test)]
    mod tests {
        use std::collections::BTreeSet;

        use kernels::bound::Site;
        use kernels_metal::points_dispatch::{CLAIMED, TIER2};
        use model_compiler::program::Call;

        /// Every point the generated census claims is one the compiler can
        /// actually route to a point call ON THIS PLANE.
        ///
        /// The failure this catches is a family whose claim table names
        /// something `model_compiler` does not route as a point — a spelling
        /// that resolves nothing, whose only symptom is a load-time refusal for
        /// a point that IS implemented.
        #[test]
        fn every_claimed_point_is_a_point_the_compiler_routes() {
            for (point, _, _) in CLAIMED {
                let call =
                    model_compiler::program::call_of(model_ir::kernels::Backend::Metal, point);
                assert_eq!(
                    call,
                    Some(Call::Point((*point).to_string())),
                    "`{point}` is claimed but `model_compiler` routes it as {call:?}",
                );
            }
        }

        /// THIS PLANE DECLARES NO TIER-2 SURFACE, and the census says so.
        ///
        /// Not a placeholder: a tier-2 point is declared and claimed by one
        /// inherent method on `Ctx`, an impl block only the plane crate can
        /// write, and `kernels-metal` states none. The row that would make this
        /// fail is a `#[claims] impl Ctx<'_>` block appearing there without the
        /// generator's `Surface::Tier2` being added beside it — which would be
        /// a surface whose points nothing dispatches.
        #[test]
        fn the_tier2_census_is_empty_because_this_plane_states_none() {
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
        /// `ssm.gdn_prep` states an f32 result over a bf16 operand and its arm
        /// selects on `Site::In(0)`. A pass that read `outputs.first()` would
        /// have asked whether the plane instantiates the point at F32 and got
        /// the right answer for the wrong reason — and the WRONG answer for
        /// `attention.kv_append`, which states no result at all.
        #[test]
        fn the_census_reads_the_witness_the_arm_reads() {
            let of = |point: &str| {
                CLAIMED
                    .iter()
                    .find(|(p, _, _)| *p == point)
                    .map(|(_, w, _)| *w)
            };
            assert_eq!(of("ssm.gdn_prep"), Some(Some(Site::In(0))));
            assert_eq!(of("attention.kv_append"), Some(Some(Site::In(0))));
            assert_eq!(of("norm.rmsnorm"), Some(Some(Site::Out(0))));
        }
    }
}
