//! What the FIRE answers, on this backend. The rest of binding is shared.
//!
//! `hold.rs` was a hundred functions that each said, in Rust, where every
//! argument of one routine comes from. Every one of them was a transcription
//! of something the routine's own signature already states -- the `sources`
//! column `kernels_wgpu::declared()` derives -- and that column is not merely
//! similar to metal's, it is HELD IDENTICAL:
//! `shader_backends_agree::two_backends_that_crossed_the_same_kernel_agree_on_its_signature`
//! compares two hundred of them and fails on a difference.
//!
//! So reading it is [`kernels::bind`], shared by all three planes, and what
//! is left here is the half that is honestly per-driver: which handle a FACT
//! names. That is [`Held::fact`], and the slot accessors around it are this
//! backend's [`Handles`].
//!
//! # What the deletion corrected
//!
//! Before the arms went, both readings existed, so they were run side by
//! side over the same statement and compared by WHICH BUFFER each named.
//! All hundred routines agreed except three, and in all three the arm was
//! the stale one:
//!
//! - `copy_logits_bf16` read the statement's second input where the
//!   signature says `Block<Buf>`. The packed parameter block is not an
//!   operand: it resolves to `Placed::Params`, is staged rather than
//!   addressed, and every other norm and MLP row already asked for it that
//!   way.
//! - `router_topk_scaled` read the statement's second input where the
//!   signature says `Weight<0, Buf>`. `per_expert_scale` is a checkpoint
//!   tensor, and finding one among the arena's operands would be an
//!   accident.
//! - `mxfp4_qmv_routed_bias` passed the SCALES a second time where all three
//!   planes say `Null<Env<Buf>>`. The MXFP4 codec has no bias plane, and
//!   `moe/qmv_routed.wgsl`'s `//#if` arm declares six bindings where the row
//!   states seven -- so the arm handed a seventh buffer to a shader that
//!   never declared it.
//!
//! Two of the three were in the families this backend crossed FIRST, before
//! `params_block` existed to ask. That is the shape of the whole argument
//! for deleting them: an arm is held against nothing, a signature is held
//! against two other planes and run on a device, and the gap between them
//! only ever grows in one direction.
//!
//! `.wiki/kilimanjaro4.md` did this for metal and deleted 3,483 lines of
//! arms. This deletion is 3,848, and it was cheaper because the shared
//! reader already existed.

use kernels::bind::Holds;
use kernels::routine::Refusal;
use kernels::shader::ShaderValue;
use kernels::{Source, Ty};
use kernels_wgpu::routine::ArgValue;

use crate::lowering::hold::{Facts, Handles};

/// Bind one launch's arguments from the row the signature derived.
///
/// [`kernels::bind::one`] per argument, paired with this backend's
/// [`Handles`] and the launch's [`Facts`] — plus the one carrier the shared
/// reader cannot answer: a `Ty::Raised` operand becomes a driver-built view
/// through `views`.
///
/// # Errors
///
/// [`Refusal::Unstated`] when an argument has no source, or has one this
/// backend cannot answer. Otherwise whatever the statement's own absences
/// produce: [`Refusal::Absent`] for a slot or scalar the trace does not
/// carry.
pub fn bind(
    args: &[Ty],
    sources: &[Option<Source>],
    o: &mut Handles<'_>,
    f: Facts,
    views: &mut super::views::Views,
) -> Result<Vec<ArgValue>, Refusal> {
    // Argument by argument rather than the shared list reader, because ONE
    // carrier is this plane's to answer before the reader sees it: a
    // `Ty::Raised` operand is a HOST view the driver builds
    // (`lowering::views`), and the shared binder has no door for a value
    // that is neither a handle nor a scalar. Everything else goes through
    // `kernels::bind::one` exactly as the list reader would have sent it —
    // same order, same slot numbering, same refusals.
    let mut out = Vec::with_capacity(args.len());
    for (at, ty) in args.iter().enumerate() {
        let source = sources
            .get(at)
            .copied()
            .flatten()
            .ok_or(Refusal::Unstated {
                what: "an argument whose signature does not say where it comes from",
            })?;
        if matches!(ty, Ty::Raised) {
            out.push(views.raise(source, o)?);
            continue;
        }
        out.push(kernels::bind::one::<ArgValue, _>(
            *ty,
            source,
            &mut Held { o, f },
        )?);
    }
    Ok(out)
}

/// ONE value, for a body that ASKS rather than a column that declares.
///
/// The same resolver, entered at one argument instead of a list. Nothing new
/// answers — what changed is only where the question is asked from.
///
/// # Errors
///
/// [`Refusal::Unstated`] for a fact this backend does not answer, and whatever
/// the fact's own absence means otherwise.
pub fn one(ty: Ty, source: Source, o: &mut Handles<'_>, f: Facts) -> Result<ArgValue, Refusal> {
    kernels::bind::one::<ArgValue, _>(ty, source, &mut Held { o, f })
}

/// This backend's answers, for the shared reader.
///
/// It borrows rather than owns the [`Handles`] because binding MUTATES them
/// -- every `input` numbers a handle -- and the caller keeps them afterwards
/// to build the bind group.
struct Held<'a, 'h> {
    o: &'h mut Handles<'a>,
    f: Facts,
}

/// The handle inside a bound value.
///
/// This backend's accessors return an `ArgValue` where the shared reader
/// wants the number inside it, because the CARRIER is the signature's
/// business rather than the driver's: `InSlot<0, I32s>` and `InSlot<0, Buf>`
/// are the same handle at two spellings, and a driver choosing between them
/// would be reading the column a second time to do it.
///
/// Unwrapping cannot fail for a value these accessors produced -- every one
/// of them mints `ArgValue::Buffer` -- and a `Refusal` rather than a panic
/// is what keeps that a claim this file makes rather than one it assumes.
fn at(v: ArgValue) -> Result<u32, Refusal> {
    v.as_buffer().ok_or(Refusal::Unstated {
        what: "a slot accessor that answered with something other than a buffer",
    })
}

impl Holds for Held<'_, '_> {
    fn input(&mut self, n: usize) -> Result<u32, Refusal> {
        at(self.o.input(n)?)
    }

    fn output(&mut self, n: usize) -> Result<u32, Refusal> {
        at(self.o.output(n)?)
    }

    // THE RECTANGLE, WHICH THE MARK NOW CARRIES. Without these two the
    // shared binder's `shaped` reads a width of zero for every operand and
    // every body that takes an `In<Tensor<_>>` refuses `Empty`.
    fn in_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.in_width(n)
    }

    fn out_width(&self, n: usize) -> Result<i32, Refusal> {
        self.o.out_width(n)
    }

    fn weight(&mut self, n: usize) -> Result<u32, Refusal> {
        at(self.o.weight(n)?)
    }

    fn param(&self, n: usize) -> Result<i32, Refusal> {
        self.o.param(n)
    }

    fn param_f32(&self, n: usize) -> Result<f32, Refusal> {
        self.o.param_f32(n)
    }

    fn null(&mut self) -> u32 {
        at(self.o.unbound()).unwrap_or_default()
    }

    fn rows(&mut self) -> i32 {
        // The launch rectangle's — what `keys::Rows` used to answer.
        self.f.rows.cast_signed()
    }
}

#[cfg(test)]
// The crate denies `print_stdout` and means it -- a driver that prints is a
// driver whose output is somebody's log. A TEST that prints is how the counts
// below are read, and the two rules do not conflict once said apart.
#[allow(clippy::print_stdout, reason = "these tests report counts to be read")]
mod tests {
    use super::bind;
    use crate::lowering::hold::{Facts, Handles};
    use kernels::routine::Refusal;
    use model_compiler::lower::Arg;

    /// The statement every routine below is bound against.
    ///
    /// Wide enough that no arm runs out: eight inputs, eight results, eight
    /// weights and sixteen scalars, all distinct so that a value landing in
    /// the wrong place is visible rather than accidentally equal.
    fn statement() -> (Vec<Arg>, Vec<u32>) {
        let args = (0..24)
            .map(|n| {
                if n >= 16 {
                    Arg::Weight(format!("w{n}"))
                } else {
                    Arg::Arena {
                        at: n * 64,
                        width: 1,
                        bytes: 2,
                    }
                }
            })
            .collect();
        // Non-zero, because a group size of zero is `Refusal::Empty` and a
        // zero axis makes a head count a division by it.
        ((args), (1..=16).collect())
    }

    fn facts() -> Facts {
        Facts {
            rows: 4,
            width: 64,
            in_width: 64,
            requests: 2,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 64,
            rotary_dims: 64,
            n_experts: 8,
            experts_per_token: 2,
            v_heads: 4,
            v_dim: 128,
            axis: 64,
            group: 64,
            bits: 4,
            tile_m: 32,
            tile_n: 32,
        }
    }

    /// **Every source in the column is one the binder answers.**
    ///
    /// This is the gate that took metal from 802 unstated arguments to
    /// zero, ported to this plane and run for the first time. It asks the
    /// REAL binder rather than a list describing it -- a second list is what
    /// `hold.rs` is, and building one to test the first would be the same
    /// mistake at a smaller scale.
    ///
    /// Two things make it device-free, and both matter:
    ///
    /// Each argument is bound ALONE, on `routine.args[at..=at]`. Binding a
    /// routine whole stops at its first required `Param`, and every
    /// argument after that one goes unasked -- which on metal hid twenty-one
    /// routines behind the first scalar of each.
    ///
    /// And the statement carries NO scalars, so every `ParamOr` chain falls
    /// through to its second half. That half is the one no shipped fire
    /// reaches: every deployment in the suite states its own rotary width,
    /// so metal's `rotary_width` fallback was unanswered for the length of
    /// the migration and nothing went red.
    ///
    /// `Absent` is therefore expected and ignored -- the statement really
    /// does not carry those. `Unstated` is the failure: it means the column
    /// says something this backend has no answer for, or says nothing at
    /// all.
    // RETIRED: `every_fact_the_mint_declares_is_one_this_binder_has_heard_of`.
    //
    // It walked `kernels::keys` -- the named-fact mint -- and asked `super::named`
    // whether this backend answered each. Both are gone. Commit 9c1ed0e6e
    // ("no-ask S4: the vocabulary is deleted") retired `keys.rs`, the whole
    // `Source::Named` variant and every backend's `named()` answerer along
    // with them, because a fact a body used to reach through `ctx.ask::<_, keys::X>()`
    // now arrives as an OPERAND -- a mark on the statement, or a runtime
    // value the driver stages under a `kernels::runtime` name -- and the
    // shared binder walks the operand column instead of a free-form key
    // vocabulary. There is nothing for this gate to walk any more; the
    // ratchet it moved was over a mint that no longer exists.
    //
    // `every_source_in_the_column_is_one_the_binder_answers` below is what
    // survives, and it is now the whole check: the operand column IS the
    // vocabulary, so binding every argument of every routine is the same
    // question stated once instead of split in two.

    #[test]
    fn every_source_in_the_column_is_one_the_binder_answers() {
        let (args, scalars) = statement();
        let _ = scalars;
        let f = facts();
        let mut unanswered: Vec<String> = Vec::new();
        let mut asked = 0usize;
        let mut whole = 0usize;
        let mut routines = 0usize;

        for routine in kernels_wgpu::routines() {
            routines += 1;
            let mut bad = 0usize;
            for at in 0..routine.args.len() {
                let mut o = Handles::over(&args, 8);
                asked += 1;
                let mut views = super::super::views::Views::default();
                let one = bind(
                    &routine.args[at..=at],
                    &routine.sources[at..=at],
                    &mut o,
                    f,
                    &mut views,
                );
                if let Err(Refusal::Unstated { what }) = one {
                    // The RAISED-OPERAND FILTER. `Ty::Raised` operands cross
                    // through `views::raise`, which reads an `Arg::Raised`
                    // off the statement to know which view to build --
                    // `kv_cache` vs `attention_mask` vs `attn.split_policy`.
                    // This fixture is a plain arena, and mixing raise keys
                    // into it would break every non-raised routine that
                    // uses the same slot, so the fixture cannot exercise
                    // the raise path and the refusal here is the FIXTURE's
                    // absence rather than a missing case in the binder --
                    // `views.rs`'s `raise` is the case, and it is reached
                    // by `every_gdn_arm_declines_the_slab_this_driver_does_not_hold`
                    // and by the serving suite. This test is about the
                    // sources the shared binder walks, so the raise path
                    // is excluded here by NAME and not by silence.
                    if what == "a raised view where the statement placed an ordinary operand" {
                        continue;
                    }
                    bad += 1;
                    unanswered.push(format!(
                        "  {}[{at}]: {}",
                        routine.name,
                        Refusal::Unstated { what }
                    ));
                }
            }
            if bad == 0 {
                whole += 1;
            }
        }

        println!("routines stated WHOLE: {whole} of {routines}");
        println!(
            "arguments with no source this backend answers: {}",
            unanswered.len()
        );
        for line in &unanswered {
            println!("{line}");
        }
        // THE FLOOR MOVED BECAUSE THE COLUMN DID. It read `> 900` against a
        // column of a thousand-odd, and the marks migration took it to 563:
        // every fact a body now reaches for with `ctx.ask` has left the
        // parameter run, and this walk enumerates the run. It is still a floor
        // -- it catches the walk going quiet, which is what it is for.
        assert!(
            asked > 500,
            "only {asked} arguments were asked about, which means the walk \
             stopped finding them rather than that they were answered"
        );

        // ZERO, which is what makes this a gate and not a ladder any more.
        // The column was 43 short when the binder first read it; every one
        // of those was a signature still written in the vocabulary that
        // predates slots, and stating them was transcription rather than
        // design.
        //
        // It may not rise. A signature added without a source column fails
        // here, at the point where the only cost is writing the column --
        // rather than at a dispatch, where the cost is a shader reading a
        // number nobody supplied and returning fluently.
        assert!(
            unanswered.is_empty(),
            "{} arguments have no source this backend answers. Every routine \
             on this backend states its whole column, and a new one must \
             too:\n{}",
            unanswered.len(),
            unanswered.join("\n")
        );
    }
}
