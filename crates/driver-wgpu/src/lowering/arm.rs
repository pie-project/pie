//! The routine plane's driver side: an arm per crossed kernel.
//!
//! # What an arm is for
//!
//! A `kernels-wgpu` routine BODY states the entrypoint, the module and the
//! grid, and takes its operands as typed arguments. What it cannot do is find
//! them: a body is handed `Buf`s and `i32`s, and turning a traced statement
//! into that argument list is the driver's half. An [`Arm`] is that half, one
//! per kernel, and it is the only thing the routine path needs that the table
//! path expressed as a `KernelSig`'s `operands` column.
//!
//! ```text
//!   table path:    row.operands ──► reorder ──► slots
//!   routine path:  arm          ──► ArgValue list ──► body ──► Fire + args
//! ```
//!
//! # Why this is a skeleton with one arm in it
//!
//! `refactor-bigplan.md` §7 Stage 2: *"the first commit in a backend is the
//! driver-side arm registry skeleton, with `ROUTINES` empty and behaviour
//! unchanged. It compiles, both suites stay green, and every family port
//! afterwards has somewhere to land."* This backend crossed its bodies first
//! — all ninety-nine of them — so the skeleton arrives late and lands into a
//! tree that already has somewhere to come FROM.
//!
//! The one arm is `sample::argmax_logits`, and it is first for the reason
//! `kernels-metal` gives for the same choice: its row states no `launch`
//! rule, so [`crate::geometry::lanes`] has always refused it `Unstated` and
//! **the table path has never dispatched this kernel at all**. There is no
//! prior behaviour for the crossing to preserve, which makes it the cheapest
//! place a seam can be proven.
//!
//! # What it does not do yet
//!
//! Dispatch. Nothing calls [`arm_for`] from the launch path; the fork in
//! [`crate::dispatch::plan_one`] is the next commit, and it needs one thing
//! this file cannot supply on its own — see [`Handles::params_block`].

use kernels::routine::Refusal;
use kernels_wgpu::routine::ArgValue;
use model_compiler::lower::Arg;

use crate::dispatch::Geometry;

/// The numbers a body takes as `Env` arguments.
///
/// Every one is a fact about the FIRE or the model, not about the statement,
/// which is what `Provenance::Env` means: the kernel never reads them and
/// they size the grid. A body asks for the ones it needs by name and this is
/// where they come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facts {
    /// How many rows this launch covers — its rectangle's height.
    pub rows: u32,
    /// The row width the launch writes.
    pub width: u32,
    /// The row width it reads, where the two differ.
    pub in_width: u32,
    /// How many requests the fire holds, which is not its row count.
    pub requests: u32,
    /// Query heads.
    pub q_heads: u32,
    /// Key/value heads.
    pub kv_heads: u32,
    /// The head width.
    pub head_dim: u32,
    /// How many channels of each head rotate.
    pub rotary_dims: u32,
    /// Experts in the mixture, or zero.
    pub n_experts: u32,
    /// How many of them each token picks.
    pub experts_per_token: u32,
    /// The norm axis, where a row holds more than one.
    pub axis: u32,
}

// NO `group` OR `bits`, and that is a real difference from
// `driver-metal::lowering::arm::Facts`.
//
// Metal's fire carries the affine point because its kernels take it as a
// launch fact. On this backend the point is in the ENTRYPOINT NAME --
// `affine_qmv_fast_bfloat16_gs_64_b_4` -- and a body picks the whole spelling
// out of a literal table with it, which is `kernels-wgpu::layout`'s
// `affine_point` and the four tables beside it. `dispatch::Geometry` has no
// such field, so inventing one here would be a second statement of something
// the name already says.
//
// The arms that need it (`layout`'s gathers, `moe`'s routed GEMMs, all of
// `quant`) will read it where the table path does: off `Declared`, which is
// the module the shell resolved. That is the fork's to wire, not this
// struct's to carry.

/// Everything a launch knows about itself, as a body's arguments want it.
///
/// # Errors
///
/// Infallible: every field is read off the launch or the fire, and a launch
/// with no rows is refused before this by [`crate::geometry`].
#[must_use]
pub fn facts(rows: u32, fire: Geometry, requests: u32, width: u32, in_width: u32) -> Facts {
    Facts {
        rows,
        width,
        in_width,
        requests,
        q_heads: fire.q_heads,
        kv_heads: fire.kv_heads,
        head_dim: fire.head_dim,
        rotary_dims: fire.rotary_dims,
        n_experts: fire.n_experts,
        experts_per_token: fire.experts_per_token,
        axis: fire.head_dim,
    }
}

/// The launch's operands, handed out as the handles a body binds.
///
/// A body's `Buf` is an opaque handle; this is what mints them. Each call
/// appends to [`Self::taken`] and returns the handle that indexes it, so the
/// ORDER a body asks in is the order the driver binds — which is the whole
/// point, and is what `every_routine_binds_a_buffer_for_every_binding_its_
/// module_declares` measures against the shader.
pub struct Handles<'a> {
    /// The statement's arguments, as the lowering states them.
    args: &'a [Arg],
    /// Which of them are inputs, outputs and weights, in that order.
    ins: Vec<usize>,
    outs: Vec<usize>,
    weights: Vec<usize>,
    /// What the body has asked for so far, in the order it asked.
    taken: Vec<Arg>,
    /// Whether the body has asked for its parameter block.
    block: bool,
}

impl<'a> Handles<'a> {
    /// The operands of one launch, ready to be asked for.
    ///
    /// Takes the operand SLICE and the op's result count, because that is all
    /// it reads, and the narrower argument is what lets this be tested
    /// without building a lowering.
    ///
    /// `results` is how many of the non-weight operands are OUTPUTS, and the
    /// rule is `driver-metal::lowering::arm::split`'s: weights are their own
    /// list, and of what is left the LAST `results` are the outputs. It is
    /// shared semantics rather than a wgpu choice — the lowering states a
    /// statement's reads before its writes — so the two backends read one
    /// convention the same way.
    ///
    /// A first draft here guessed instead: *"the first `Arena` is the
    /// output"*. `argmax_logits` has two of each and it refused its own
    /// statement.
    ///
    /// # The ORDER an arm asks in does not matter
    ///
    /// Worth stating because it is not obvious and it makes one class of
    /// sabotage look like a defect when it is not. A handle is an index into
    /// [`Self::taken`], assigned when the operand is ASKED for, and the body
    /// hands the handles back in its own order — so an arm that reads its
    /// second input before its first produces different indices AND a
    /// differently ordered `taken`, and the two shifts cancel exactly.
    ///
    /// What matters is which operand each NAME binds to. `gate = input(0)`
    /// against `gate = input(1)` is a real defect and
    /// `the_routine_path_plans_what_the_table_path_planned` fails on it;
    /// reordering the two `let`s is not, and it does not.
    #[must_use]
    pub fn over(args: &'a [Arg], results: usize) -> Self {
        let widthed: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| !matches!(a, Arg::Weight(_)))
            .map(|(i, _)| i)
            .collect();
        let weights: Vec<usize> = args
            .iter()
            .enumerate()
            .filter(|(_, a)| matches!(a, Arg::Weight(_)))
            .map(|(i, _)| i)
            .collect();
        let results = results.min(widthed.len());
        let (ins, outs) = widthed.split_at(widthed.len() - results);
        Self {
            args,
            ins: ins.to_vec(),
            outs: outs.to_vec(),
            weights,
            taken: Vec::new(),
            block: false,
        }
    }

    /// The `n`th INPUT, as a handle.
    ///
    /// # Errors
    ///
    /// [`Refusal::Empty`] when the statement has no such input, which is a
    /// disagreement between the arm and the trace rather than a caller's
    /// mistake.
    pub fn input(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self.ins.get(n).ok_or(Refusal::Empty { what: "an input" })?;
        Ok(self.take(at))
    }

    /// The `n`th OUTPUT, as a handle.
    ///
    /// # Errors
    ///
    /// As [`Self::input`].
    pub fn output(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self
            .outs
            .get(n)
            .ok_or(Refusal::Empty { what: "an output" })?;
        Ok(self.take(at))
    }

    /// The `n`th WEIGHT, as a handle.
    ///
    /// # Errors
    ///
    /// As [`Self::input`].
    pub fn weight(&mut self, n: usize) -> Result<ArgValue, Refusal> {
        let at = *self
            .weights
            .get(n)
            .ok_or(Refusal::Empty { what: "a weight" })?;
        Ok(self.take(at))
    }

    /// The parameter block, as a handle that STANDS FOR the staged run.
    ///
    /// **The one thing this file cannot answer on its own.** A statement's
    /// scalars are packed into a block, and on this backend that block is
    /// usually a `@group(1)` uniform the encoder builds from the dispatch's
    /// own scalar list — so it has no address at plan time and no place in
    /// the buffer list at all. Where a shader declares it as a `@group(0)`
    /// storage entry instead, it does have a slot, and the two cases are told
    /// apart by the module rather than by the row.
    ///
    /// `kernels-metal` met this crossing its own `mlp` — *"the parameter
    /// block was a buffer with no address"* — and answered it with a handle
    /// that stands for the staged run, resolved to a packed slot at lay-out
    /// rather than to an address. This returns the same kind of handle and
    /// the resolution is the fork's to write, which is why nothing calls it
    /// yet.
    pub fn params_block(&mut self) -> ArgValue {
        self.block = true;
        ArgValue::Buffer(u32::try_from(self.taken.len()).expect("a small operand count"))
    }

    /// What the body asked for, in the order it asked.
    #[must_use]
    pub fn taken(&self) -> &[Arg] {
        &self.taken
    }

    /// Whether the body asked for its parameter block.
    #[must_use]
    pub const fn wants_block(&self) -> bool {
        self.block
    }

    fn take(&mut self, at: usize) -> ArgValue {
        let handle = u32::try_from(self.taken.len()).expect("a small operand count");
        self.taken.push(self.args[at].clone());
        ArgValue::Buffer(handle)
    }
}

/// One kernel's operand resolution.
///
/// Returns the argument list its BODY takes, which the body then turns into a
/// `Fire` and a bound list. The two halves are deliberately separate: this
/// one knows the trace and nothing about the shader, and the body knows the
/// shader and nothing about the trace.
pub type Arm = fn(&mut Handles<'_>, Facts) -> Result<Vec<ArgValue>, Refusal>;

/// `sample::argmax_logits`.
///
/// The first arm, and the only one, for the reason this module's docs give:
/// its row states no launch rule, so nothing has ever dispatched it through
/// the table and there is no prior behaviour to preserve.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement does not carry the four operands the
/// body takes.
pub fn argmax_logits(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let logits = o.input(0)?;
    let next_token = o.output(0)?;
    let params = o.input(1)?;
    let eos_flag = o.output(1)?;
    Ok(vec![
        logits,
        next_token,
        params,
        eos_flag,
        ArgValue::U32(f.rows),
    ])
}

/// `ptir::copy_logits_bf16`.
///
/// The second arm, and dark for the same reason the first is: this backend's
/// channel-plane interpreter never dispatches it, so no text names the symbol
/// and arming it cannot change what any model computes.
///
/// # `vocab` is the launch's WIDTH
///
/// The body takes `vocab` and halves it, because `logits_copy.wgsl` packs two
/// bf16 into a `u32` and one lane owns one word. What arrives here is the row
/// width the launch writes, which is the vocabulary — the halving is the
/// SHADER's fact and stays in the body, and `kernels-metal`'s row states the
/// unhalved `[vocab, rows, 1]` for the same kernel and is equally right about
/// its own shader. That split is `refactor-bigplan.md` §2.
///
/// # Errors
///
/// [`Refusal::Empty`] when the statement does not carry the three operands
/// the body takes.
pub fn copy_logits_bf16(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let source = o.input(0)?;
    let destination = o.output(0)?;
    let params = o.input(1)?;
    Ok(vec![
        source,
        destination,
        params,
        ArgValue::U32(f.width),
        ArgValue::U32(f.rows),
    ])
}

/// `mlp::silu_mul`.
///
/// **The first LIVE arm on this backend.** Every gated MLP names this symbol,
/// so unlike `sample` and `ptir` a mistake here changes what a real model
/// computes — which is why `the_routine_path_plans_what_the_table_path_planned`
/// exists and derives every field of its dispatch twice.
///
/// The one kernel in `mlp/gated.wgsl` that reads no parameter block: it needs
/// no scalar the grid does not give it. That is what makes it armable before
/// the block's resolution is written, and it is the same reason
/// `driver-vulkan` gives at its own crossing.
///
/// # Errors
///
/// [`Refusal::Empty`] for an operand the statement does not carry.
pub fn silu_mul(o: &mut Handles<'_>, f: Facts) -> Result<Vec<ArgValue>, Refusal> {
    let gate = o.input(0)?;
    let up = o.input(1)?;
    let out = o.output(0)?;
    Ok(vec![
        gate,
        up,
        out,
        ArgValue::I32(f.width.cast_signed()),
        ArgValue::I32(f.rows.cast_signed()),
    ])
}

/// One crossed kernel: the entrypoint STEM a plan spells it with, and the arm
/// that feeds it.
///
/// # Why a stem and not a name
///
/// A plan names `silu_mul_bfloat16`, not `silu_mul`. A row could be found from
/// that because `kernels::sig_in` falls back to an axis match, but the fork
/// sits ABOVE the row lookup — that is what lets rows be deleted — so it has
/// to answer from the symbol alone. This is the same answer
/// `driver-vulkan::arm::Crossed` and `driver-metal`'s `crossed` reached.
///
/// Matching by name alone worked for exactly the two DARK families and hid
/// this: `argmax_logits` and `copy_logits_bf16` have no axis, so their symbol
/// IS their stem. `silu_mul` was the first live arm and it was never reached —
/// `the_routine_path_plans_what_the_table_path_planned` compared zero
/// rectangles and said so, which is why that test asserts a floor.
pub struct Crossed {
    /// The longest prefix of a symbol that names this kernel.
    pub stem: &'static str,
    /// The arm, or `None` for a stem that is CLAIMED but not yet armed.
    pub arm: Option<Arm>,
}

/// The crossed kernels, by stem.
///
/// # Longest match, and why an unarmed stem still belongs here
///
/// Stems nest. `silu_mul` is a prefix of `silu_mul_strided_bfloat16`, and a
/// first match would hand a STRIDED rectangle to the contiguous body: it binds
/// real buffers, dispatches, and reads every row from the wrong offset.
/// Nothing downstream would notice — the shapes are identical and both are
/// storage buffers. So `silu_mul_strided` is listed with no arm, which makes
/// it the longer match and sends its symbols back to the table path where
/// they still belong.
///
/// `every_entrypoint_is_claimed_by_the_stem_that_owns_it` checks that over the
/// whole census rather than over the pairs anyone thought of.
static LIVE: &[Crossed] = &[
    Crossed {
        stem: "argmax_logits",
        arm: Some(argmax_logits as Arm),
    },
    Crossed {
        stem: "copy_logits_bf16",
        arm: Some(copy_logits_bf16 as Arm),
    },
    Crossed {
        stem: "silu_mul",
        arm: Some(silu_mul as Arm),
    },
    // CLAIMED, NOT ARMED. `mlp/gated.wgsl`'s strided variant walks rows by a
    // pitch the contiguous body does not take, and it keeps `silu_mul` from
    // claiming its symbols by prefix.
    Crossed {
        stem: "silu_mul_strided",
        arm: None,
    },
];

/// The stem and arm for a symbol, if this backend has crossed AND armed it.
///
/// `None` is the ordinary answer today and means *"take the table path"*.
#[must_use]
pub fn crossed(symbol: &str) -> Option<(&'static str, Arm)> {
    let found = LIVE
        .iter()
        .filter(|c| {
            symbol
                .strip_prefix(c.stem)
                .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
        })
        .max_by_key(|c| c.stem.len())?;
    Some((found.stem, found.arm?))
}

/// The arm for a symbol, if this backend has crossed and armed it.
#[must_use]
pub fn arm_for(symbol: &str) -> Option<Arm> {
    crossed(symbol).map(|(_, arm)| arm)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every entrypoint is claimed by the stem that OWNS it, or by none.
    ///
    /// The nesting trap over the whole census rather than over the pairs
    /// anyone thought of. `silu_mul` is a prefix of `silu_mul_strided_bfloat16`
    /// and the two are different kernels; a claim there would hand a strided
    /// rectangle to the contiguous body, which binds real buffers, dispatches,
    /// and reads every row from the wrong offset. Both are storage buffers of
    /// the same length, so nothing downstream would see it.
    ///
    /// What makes this checkable without a second list: an entrypoint belongs
    /// to the ROW or ROUTINE whose name is its longest prefix, and
    /// `kernels-wgpu` already states both. So the claim is only correct when
    /// the claiming stem IS that owner.
    #[test]
    fn every_entrypoint_is_claimed_by_the_stem_that_owns_it() {
        let owners: Vec<&str> = kernels_wgpu::KERNELS
            .iter()
            .map(|k| k.name)
            .chain(kernels_wgpu::routines().into_iter().map(|r| r.name))
            .collect();

        let mut claimed = 0u32;
        for point in kernels_wgpu::entrypoints() {
            let Some((stem, _)) = crossed(&point) else {
                continue;
            };
            claimed += 1;
            let owner = owners
                .iter()
                .filter(|n| {
                    point
                        .strip_prefix(**n)
                        .is_some_and(|rest| rest.is_empty() || rest.starts_with('_'))
                })
                .max_by_key(|n| n.len())
                .unwrap_or_else(|| panic!("`{point}` is named by no row and no routine"));
            assert_eq!(
                stem, *owner,
                "`{point}` is claimed by `{stem}` and owned by `{owner}`. A \
                 stem that claims a sibling's symbols sends them to the wrong \
                 body, which binds, dispatches and answers wrongly",
            );
        }
        assert!(
            claimed > 0,
            "no entrypoint is claimed, so this test compared nothing"
        );
    }

    /// The two dark families and the first live one are armed.
    ///
    /// `sample` and `ptir` are the two symbols no text on this backend names,
    /// so arming them could not change what any model computes — which is why
    /// they went first, and the same pair `kernels-metal` and `kernels-vulkan`
    /// crossed first for the same reason.
    ///
    /// The number is asserted rather than bounded so that the next family to
    /// land here is a deliberate edit. `arm_for` returning `None` is what
    /// keeps every other kernel on the table path, so a name added here
    /// CHANGES how a real fire is planned and should never arrive quietly.
    #[test]
    fn the_armed_stems_are_the_ones_registered_and_nothing_else() {
        assert!(arm_for("argmax_logits").is_some());
        assert!(arm_for("copy_logits_bf16").is_some());
        // A crossed routine whose arm has NOT landed: the body exists in
        // `kernels-wgpu` and the driver still plans it from its row.
        assert!(arm_for("rms_single_row").is_none());
        // Armed, and reached through the SYMBOL a plan actually spells.
        assert!(arm_for("silu_mul").is_some());
        assert!(arm_for("silu_mul_bfloat16").is_some());
        assert!(arm_for("argmax_logits_bfloat16").is_some());
        // The nesting trap: a longer stem with no arm sends its symbols back
        // to the table path rather than to the contiguous body.
        assert!(arm_for("silu_mul_strided_bfloat16").is_none());
        assert!(arm_for("silu_mul_strided").is_none());
        // And a stem may not end mid-word.
        assert!(arm_for("silu_multiply").is_none());
        // And a name no backend has.
        assert!(arm_for("not_a_kernel").is_none());

        let armed = kernels_wgpu::routines()
            .into_iter()
            .filter(|r| arm_for(r.name).is_some())
            .count();
        assert_eq!(
            armed, 3,
            "{armed} crossed routines have an arm. Adding one changes how a \
             real fire is planned, so the count moves only on purpose."
        );
    }

    /// A body asks in its own order, and that order is what gets bound.
    ///
    /// `argmax_logits` takes `logits, next_token, params, eos_flag` — an
    /// input, an output, an input, an output — so the handles it is given
    /// must be 0, 1, 2, 3 in the order ASKED and not in the order the trace
    /// states. That is the whole reason `Handles` mints handles on demand
    /// instead of returning the statement's own indices.
    #[test]
    fn handles_are_minted_in_the_order_the_body_asks() {
        let args: Vec<Arg> = (0..4)
            .map(|n| Arg::Arena {
                at: n * 64,
                width: 1,
                bytes: 2,
            })
            .collect();
        let mut o = Handles::over(&args, 2);
        let f = facts(7, Geometry::default(), 1, 1024, 1024);
        let args = argmax_logits(&mut o, f).expect("four operands");
        // The handle SEQUENCE is 0, 1, 2, 3 by construction -- they are
        // minted in order -- so asserting it proves nothing on its own. What
        // it does pin is that the body asked four times and got four
        // distinct slots.
        assert_eq!(
            args,
            vec![
                ArgValue::Buffer(0),
                ArgValue::Buffer(1),
                ArgValue::Buffer(2),
                ArgValue::Buffer(3),
                ArgValue::U32(7),
            ]
        );

        // THE CLAIM. `argmax_logits` binds `logits, next_token, params,
        // eos_flag`, which is in, OUT, in, out; the lowering states its reads
        // before its writes, so the statement is `logits, params, next_token,
        // eos_flag`. Handle 1 must therefore be the statement's THIRD operand
        // and handle 2 its second -- the reorder is the whole reason this
        // type exists, and it is invisible in the handle numbers.
        let at = |arg: &Arg| match arg {
            Arg::Arena { at, .. } => *at,
            _ => unreachable!("this fixture states only arena operands"),
        };
        let taken: Vec<u32> = o
            .taken()
            .iter()
            .map(at)
            .map(u32::try_from)
            .map(|v| v.expect("a small offset"))
            .collect();
        assert_eq!(
            taken,
            vec![0, 128, 64, 192],
            "the body's order was not applied: handle 1 should be the \
             statement's third operand and handle 2 its second"
        );
        assert!(!o.wants_block(), "argmax reads no packed parameter block");
    }

    /// A statement short of an operand is refused, not indexed past.
    #[test]
    fn a_statement_the_arm_cannot_fill_is_refused() {
        let args = [Arg::Arena {
            at: 0,
            width: 1,
            bytes: 2,
        }];
        let mut o = Handles::over(&args, 2);
        let f = facts(1, Geometry::default(), 1, 1024, 1024);
        assert!(matches!(
            argmax_logits(&mut o, f),
            Err(Refusal::Empty { .. })
        ));
    }
}
