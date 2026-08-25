//! The baker executor, in Vulkan: a `Program` walked per fire, against the
//! claim bodies `kernels-vulkan` states.
//!
//! # What this is
//!
//! `driver-cuda/src/baker/` is the reference implementation, `driver-metal`'s
//! is the first shader plane to follow it, `driver-wgpu`'s is the second, and
//! this is that capability spelled in Vulkan. The chain is the same one end to
//! end — text → `Plan` → lane → `Program` → walk → the plane's claim bodies —
//! and the shape of every module beside this one matches its three siblings by
//! name, so that a decision made once is findable four times.
//!
//! # HALF OF IT IS NOT HERE, AND THAT IS THE POINT
//!
//! [`crate::walk`] holds the walk, the bound statement, the frame's numbering,
//! the resolve pass and the lane — a thousand lines a plane does not get to
//! have an opinion about. It is this crate's own copy of a program both
//! siblings also hold their own copy of; that module's header carries the
//! argument for why there are three copies and not one crate.
//!
//! WHAT IS HERE IS WHAT DIFFERS, and each of the five is a measurement rather
//! than a preference:
//!
//! * [`marks`] — a Vulkan region names its ALLOCATION and an offset, because a
//!   descriptor is written from a `VkDescriptorBufferInfo` and that is
//!   `{buffer, offset, range}`. Metal's is an address and an extent. wgpu's is
//!   the same three fields as this one, and that agreement is a measurement
//!   about the two APIs rather than a shared file.
//! * [`stage`] — this driver stages a partials plane for its split decode
//!   ([`stage::FireTable::AttnPartials`]), which metal does not, and states no
//!   KV head count, which wgpu does. It is a third list.
//! * [`views`] — `Plane::PagesView` here is the pool row alone, metal's shape,
//!   reached by a route neither sibling has: `kernels_vulkan::attn` pulls the
//!   per-fire planes off the `Staged` trait rather than off the view. That
//!   module states what the route costs today.
//! * [`encode`] and [`dispatch`] — a `Dispatch` here carries LANES (the
//!   workgroup width is declared by `[numthreads]` and recovered from the
//!   SPIR-V) *and* a [`dispatch::Touches`] hazard set (`vkCmdDispatch` runs
//!   concurrently until a `vkCmdPipelineBarrier` says otherwise). One answer
//!   from each sibling, which is the clearest evidence in the tree that these
//!   files could not have been one.
//!
//! And [`plane`] is the join: two impls carrying the region, the encoder, the
//! census, the staging door and the dispatch, so the list of what makes this
//! plane different is a thing a reader can hold in one screen.
//!
//! # WHAT THIS EXECUTOR MEASURES, AND WHAT IT SAYS TODAY
//!
//! `kernels-vulkan` claims 32 of the floor's points across seven families, and
//! two facts about that set decide what this driver can serve:
//!
//! * **`layout.embed` IS claimed here**, which neither shader sibling manages —
//!   wgpu claims no point that can seed a tower at all, and metal reaches the
//!   embed through a `CANON` symbol rather than a claim. A point whose result
//!   width comes off a weight rather than off an operand is what lets
//!   `model_compiler::program::bound` size a first rectangle, so a synthetic
//!   tower on this plane can be BOUND rather than stated by hand.
//! * **no `Gemm` point is claimed**, and that is the headline gap. Every matmul
//!   `kernels-vulkan` stamps is quantised, so all three `Gemm` points wait on
//!   the floor's `Bank<R: Repr>` payload reaching their declarations — the same
//!   sentence `kernels-wgpu`'s `quant` makes, and the same reason a dense bf16
//!   SKU refuses at `gemm.matmul` before it can refuse anywhere else.
//!
//! [`resolve::check`] is what turns the second into a load-time sentence naming
//! the point and the first statement that asked, rather than a mid-fire
//! surprise. The tests below assert both, so the day either changes is loud.
//!
//! # AND ONE REFUSAL THIS EXECUTOR CANNOT MOVE
//!
//! Four of the claimed points — `attention.decode`, `attention.prefill`,
//! `attention.masked` and `attention.kv_append` — pass [`resolve::check`] and
//! then refuse AT THE FIRE, because their bodies reach the fire's staging
//! through `kernels_vulkan::points::Staged`, whose blanket impl on
//! `dyn Encode` refuses all five of its methods by name. That is exactly the
//! shape the eager pass exists to prevent, and it cannot be prevented from
//! here: the door is on the floor. [`views`] names the file it opens in.

/// One encodable dispatch, and the hazard set a barrier is placed from.
pub mod dispatch;
/// The door a claim body fires through.
pub mod encode;
/// Rectangles, the regions they name, and the marks a point takes them as.
pub mod marks;
/// What this plane is to the executor: the two impls the walk takes.
pub mod plane;
/// What a fire stages beside its arena.
pub mod stage;
/// The raised views this driver builds.
pub mod views;

pub use crate::walk::frame;
pub use crate::walk::{BANK_ALIGN, Baked, READABLE_BASE, arena_of, join, readable_base, word_of};
pub use marks::{Bindings, Bound as BoundRegion, BufferId, NOTHING, Rect, Slice};
pub use plane::Vulkan;
pub use stage::{FireTable, KvGeometry, Pools, Slab, Splits};

/// A weight on the device, in this plane's regions.
pub type Bank = crate::walk::Bank<Slice>;

/// The walk, bound to this plane.
///
/// A MODULE OF ALIASES AND NOT A RE-EXPORT, so that `baker::walk::Fire` still
/// names one thing rather than a generic a reader has to instantiate in their
/// head. What is behind each name is [`crate::walk::fire`]'s, at [`Vulkan`].
pub mod walk {
    use super::{Slice, Vulkan};

    pub use crate::walk::fire::{Cursor, Refused};

    /// Everything one fire of the baker addresses.
    pub type Fire<'a> = crate::walk::fire::Fire<'a, Vulkan>;

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

    use super::Vulkan;

    /// Check every step of `program` against the plane this driver fires.
    ///
    /// See [`crate::walk::resolve::check`], which this names [`Vulkan`] for.
    #[must_use]
    pub fn check(plan: &Plan, program: &Program) -> Vec<Unresolved> {
        crate::walk::resolve::check::<Vulkan>(plan, program)
    }

    #[cfg(test)]
    mod tests {
        use std::collections::BTreeSet;

        use kernels::bound::Site;
        use kernels_vulkan::points_dispatch::{CLAIMED, TIER2};
        use model_compiler::program::Call;

        /// Every point the generated census claims is one the compiler can
        /// actually route to a point call ON THIS PLANE.
        ///
        /// The failure this catches is a family whose claim table names
        /// something `model_compiler` does not route as a point — a spelling
        /// that resolves nothing, whose only symptom is a load-time refusal for
        /// a point that IS implemented. It also checks the thing this wave
        /// added: that `model_ir::kernels::Backend` grew a `Vulkan` row and
        /// that `point_claims(Vulkan)` reaches THIS crate's tables. Before it
        /// did, `kernels-vulkan`'s seven `#[claims]` blocks emitted their
        /// tables and nothing on earth read them.
        #[test]
        fn every_claimed_point_is_a_point_the_compiler_routes() {
            for (point, _, _) in CLAIMED {
                let call =
                    model_compiler::program::call_of(model_ir::kernels::Backend::Vulkan, point);
                assert_eq!(
                    call,
                    Some(Call::Point((*point).to_string())),
                    "`{point}` is claimed by this plane but `call_of` answers {call:?}",
                );
            }
        }

        /// THIS PLANE DECLARES NO TIER-2 SURFACE, and it cannot.
        ///
        /// A tier-2 point is an inherent method on `Ctx`, and `Ctx<'a>` here is
        /// `dyn Encode + 'a` — Rust has no inherent impl for a trait object. So
        /// the row that would make this fail cannot be written at all, and the
        /// assertion is a statement about the generator rather than about
        /// restraint.
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
        /// `outputs.first()` would have had nothing to read and answered
        /// `None` — which reads as "claimed by name" and would have been the
        /// RIGHT answer for the wrong reason.
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

        /// THE GAP CLOSED, AND THIS IS WHAT REPLACED THE ASSERTION THAT SAID
        /// IT WAS OPEN.
        ///
        /// This test used to be `no_gemm_point_is_claimed_and_that_is_this_\
        /// planes_headline_gap`, and it read: *"`kernels-vulkan` claims no
        /// `Gemm` point, which is why no catalog row's lane binds on this
        /// plane: every SKU in the catalog states `gemm.matmul`, and this tree
        /// has no dense matmul at all — every matmul it stamps reads a bank as
        /// three weights."* It fired, exactly as it was written to.
        ///
        /// The reading that made it a gap was half right. The bank half still
        /// holds — `points::Staged::bank` refuses here, and a QUANTISED matmul
        /// on this plane is still waiting on the floor for a
        /// `Const<Bank<R>>` mark — but `gemm.matmul`'s weight is
        /// `Const<Tensor<T>>`, ONE address and one dense bf16 rectangle, so
        /// the bank refusal was never what stood in the way of these three.
        /// `kernels/gemm/dense.slang` is the shader that says so.
        ///
        /// All THREE are claimed and not one, because the three are one
        /// arithmetic under three names: `lm_head` and `attention_landing`
        /// forward to `matmul` here exactly as they do on `kernels-cuda`,
        /// which is where that shape was settled.
        #[test]
        fn the_three_gemm_points_are_claimed_and_that_is_what_lets_a_lane_bind() {
            let gemm: Vec<&str> = CLAIMED
                .iter()
                .map(|(p, _, _)| *p)
                .filter(|p| p.starts_with("gemm."))
                .collect();
            assert_eq!(
                gemm,
                ["gemm.matmul", "gemm.lm_head", "gemm.attention_landing"],
                "the dense matmul is what every catalog SKU states first; a \
                 plane that lost it would stop binding every lane at once",
            );
        }

        /// THE ONE POINT THAT CAN SEED A TOWER, and neither shader sibling has
        /// it.
        ///
        /// A result can only be SIZED at load if its width rule does not read
        /// an operand's rectangle, and across the whole floor exactly four
        /// points qualify: `layout.embed` (an embedding table's axis) and the
        /// three `gemm.*` (a weight's). `kernels-wgpu` claims none of the four,
        /// which is why `driver-wgpu/tests/the_walk_is_the_program.rs` has to
        /// state its `Program` by hand rather than bind one.
        ///
        /// This plane claims `layout.embed`, so its walk test binds a real
        /// `Program` through `model_compiler::program::bound`. THIS ROW IS THAT
        /// FILE'S PREMISE: if the claim goes, the fixture stops binding and the
        /// failure would otherwise read as a compiler bug.
        #[test]
        fn layout_embed_is_claimed_and_it_is_what_lets_a_tower_be_seeded() {
            assert!(
                CLAIMED.iter().any(|(p, _, _)| *p == "layout.embed"),
                "`layout.embed` is no longer claimed here; \
                 `tests/the_walk_is_the_program.rs` binds a real Program because it is",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// EVERY CATALOG ROW TRACES FOR THIS PLANE, AND THE GEMMA ROWS NOW BIND.
    ///
    /// This test used to end `..._and_none_binds_yet`, and its last assertion
    /// read: *"these rows now BIND a lane on the vulkan plane — that is the
    /// day this driver can serve something, and this test is how you find out.
    /// Replace it with one that says which lanes bind and fires them."* It
    /// fired on `["gemma4-e4b-bf16-kv-bf16", "gemma4-31b-bf16-kv-bf16"]` the
    /// day `kernels/gemm/dense.slang` landed, which is the first time any
    /// catalog row has bound a lane on this plane.
    ///
    /// WHY GEMMA AND NOT THE OTHER TWELVE, which is the measurement and not a
    /// coincidence. A lane binds when EVERY point its text states is claimed,
    /// so the rows that bind are the rows whose whole point set this plane
    /// answers. Gemma-4's is now closed: the dense matmul was the last of its
    /// twenty-two. Qwen-3.5 still states the five `ssm.*` and gpt-oss the
    /// three `attention.*_lse`/`sink`, and neither family has a claim here
    /// yet — so those rows trace, refuse, and are named by
    /// [`a_refused_lane_names_a_few_points_and_not_every_point_it_states`].
    ///
    /// The list is asserted as a SET and not a count. A row joining it is news
    /// worth reading in a diff, and a row LEAVING it is a regression that
    /// would otherwise be silent — the previous spelling could only notice the
    /// first of those two.
    #[test]
    fn every_catalog_row_traces_and_the_rows_whose_points_are_all_claimed_bind() {
        let mut bound_rows = Vec::new();
        let mut traced = 0usize;
        for row in model::serve::ROWS {
            let Ok(baked) = Baked::of::<Vulkan>(row.id) else {
                // A row `Deployment::of` refuses is a fact about the POOL and
                // not about this plane; both siblings' twins of this test skip
                // the same set for the same reason.
                continue;
            };
            traced += 1;
            assert_eq!(baked.plan.plane, model_ir::kernels::Backend::Vulkan);
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
        assert_eq!(
            bound_rows,
            [
                "gemma4-e4b-bf16-kv-bf16",
                "gemma4-31b-bf16-kv-bf16",
                "qwen35-a3b-bf16-kv-bf16",
                "qwen35-d3b-bf16-kv-bf16",
                "qwen35-d0.8b-bf16-kv-bf16",
            ],
            "the rows that bind on this plane are the rows whose every stated \
             point is claimed; a row joining this list is a family landing and \
             a row leaving it is a claim regressing",
        );
    }

    /// WHAT A LOAD ON THIS PLANE ACTUALLY PRINTS: A SHORT LIST OF NAMED POINTS,
    /// NOT "EVERYTHING".
    ///
    /// This is the row that tells a BACKLOG from a WIRING FAULT, and the
    /// distinction is the whole reason `Backend::Vulkan` had to reach
    /// [`point_claims`]. A plane whose claim tables nothing joined would refuse
    /// every point of every lane as [`Why::Unclaimed`] — which reads exactly
    /// like a plane that implements nothing.
    ///
    /// WHAT THIS ROW USED TO SAY, and why the point it names had to move.
    ///
    /// It read: *"What a load prints today, for `gemma4-e4b-bf16-kv-bf16`, is
    /// five rows — `norm.mul_scalar`, `gemm.matmul`, `gemm.attention_landing`,
    /// `layout.select`, `gemm.lm_head` — against a lane that states many more
    /// points than that"*, and it asserted `gaps.contains("gemm.matmul")`.
    /// Those five ARE the five this wave landed, so gemma-4 has no gaps left
    /// at all and the row it was written against now binds. Both halves of the
    /// test had to move: the SKU it reads and the point it names.
    ///
    /// It reads `qwen35-d0.8b-bf16-kv-bf16` now and names `ssm.gdn_prep`.
    /// Qwen-3.5 is short of exactly the five `ssm.*` points — a family with no
    /// `#[claims]` block on this plane at all — which is the same shape the
    /// gemm five had: a NAMED, SHORT backlog against a lane stating far more.
    /// The SKU is named rather than "the first row that refuses" because which
    /// row that is changes every time a family lands, and a test that silently
    /// re-aims is one that stops measuring what it was written for.
    ///
    /// The assertion stays *contains* rather than *is first*, for the reason
    /// it always gave: the order is the PLAN's, and which unclaimed point a
    /// text happens to state first is not a fact about this plane.
    ///
    /// [`point_claims`]: model_ir::kernels::point_claims
    /// [`Why::Unclaimed`]: model_compiler::program::Why::Unclaimed
    #[test]
    fn a_refused_lane_names_a_few_points_and_not_every_point_it_states() {
        // GPT-OSS AND NOT QWEN-3.5, AND THE SWAP IS THE RECORD OF A LANDING.
        // This read `qwen35-d0.8b` and expected it short of `ssm.gdn_prep`,
        // because the ssm family was unclaimed here. It is claimed now — five
        // of its seven points — and that row BINDS. A test whose subject is a
        // refused lane has to be re-pointed the day its example stops
        // refusing, or it goes on passing for a reason that is no longer the
        // one written here.
        //
        // gpt-oss is short of `attention.{decode_lse, prefill_lse, sink}`,
        // which metal and wgpu both closed this week and this plane has not.
        let baked = Baked::of::<Vulkan>("gptoss-20b-bf16-mxfp4-kv-bf16")
            .expect("`gptoss-20b-bf16-mxfp4-kv-bf16` traces for this plane");
        let refusal = baked
            .lanes
            .iter()
            .find_map(|l| l.as_ref().err())
            .expect("gpt-oss states the attention sink trio, unclaimed here");
        let gaps: Vec<&str> = refusal.gaps.iter().map(|g| g.point.as_str()).collect();
        assert!(
            gaps.contains(&"attention.sink"),
            "`{}` should be short of the sink rescale: {gaps:?}",
            baked.sku,
        );
        // A CLAIMED POINT MUST NOT BE IN THE LIST. Every gap is a point this
        // plane does not answer, so a row naming one of the thirty-two would
        // mean the join reached the wrong table rather than that a shader is
        // missing. THIS IS THE HALF THAT WOULD HAVE FAILED before
        // `Backend::Vulkan` existed: with no arm in `point_claims`, every
        // statement of the lane would be here.
        for gap in &refusal.gaps {
            assert!(
                !kernels_vulkan::points_dispatch::CLAIMED
                    .iter()
                    .any(|(p, _, _)| *p == gap.point),
                "`{}` is claimed here and still reported as a gap: {gap}",
                gap.point,
            );
        }
        // And the backlog is SHORTER than the lane, which is the difference a
        // reader wants: a plane that answered nothing would report one row per
        // distinct point the plan states.
        let stated: std::collections::BTreeSet<&str> =
            baked.plan.ops.iter().map(|o| o.kernel.as_str()).collect();
        assert!(
            gaps.len() < stated.len(),
            "`{}` states {} distinct points and is short of {} — that is not a \
             backlog, it is a plane whose claim tables nothing joined",
            baked.sku,
            stated.len(),
            gaps.len(),
        );
    }
}
