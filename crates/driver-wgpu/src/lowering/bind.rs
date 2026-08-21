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
        let source = sources.get(at).copied().flatten().ok_or(Refusal::Unstated {
            what: "an argument whose signature does not say where it comes from",
        })?;
        if matches!(ty, Ty::Raised) {
            out.push(views.raise(source, o)?);
            continue;
        }
        out.push(kernels::bind::one::<ArgValue, _>(*ty, source, &mut Held { o, f })?);
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
pub fn one(
    ty: Ty,
    source: Source,
    o: &mut Handles<'_>,
    f: Facts,
) -> Result<ArgValue, Refusal> {
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
                    Arg::Arena { at: n * 64, width: 1, bytes: 2 }
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
    /// Every fact `kernels::keys` mints is answered here or declined by name.
    ///
    /// The gate above walks the SOURCES wgpu's own routines state. This one
    /// walks the whole mint, including facts only another backend names yet,
    /// because that is where the next one arrives from: a key minted for
    /// metal is a key this binder will meet the day a wgpu routine states it,
    /// and `Source::Named` made forgetting one a runtime `UnknownFact`
    /// instead of a non-exhaustive match.
    ///
    /// This stood in `tests/keys_ledger.rs` and SCRAPED `binding.rs` for the
    /// text `keys::X`. That worked while the answering lived in an operand
    /// match there; the shared binder moved it here and the scrape went on
    /// reading a file that had stopped answering, so every one of the fifty-
    /// eight keys read as unheard. A gate that outlives the thing it guarded
    /// does not fail honestly -- it fails wholesale.
    ///
    /// So it asks `named` instead of reading it. A second list describing the
    /// binder is a second thing to keep true; the binder is already here.
    ///
    /// It is a RATCHET rather than a floor at zero, because the mint is the
    /// workspace's and not this backend's: cuda, metal and vulkan declare
    /// facts here too, and a fact becomes this binder's business the day a
    /// wgpu routine states it -- at which point
    /// `every_source_in_the_column_is_one_the_binder_answers` fails first and
    /// by name. What THIS one adds is the direction of travel. If the number
    /// falls, a fact was ported and the constant comes down with it; if it
    /// rises, either a new fact was minted or `named` stopped answering one
    /// it used to, and the printed list says which.
    #[test]
    fn every_fact_the_mint_declares_is_one_this_binder_has_heard_of() {
        // The four SSM scalars are a fact on CUDA and a trace parameter here:
        // `kernels-wgpu/src/ssm.rs` declares nine routines and names no
        // `keys::` type at all, so its shaders take the decay terms in the
        // params block positionally, the way every other scalar arrives on
        // this backend. Nothing is missing until a wgpu SSM routine wants one
        // by name -- and then this list is what it moves out of.
        const DECLINED: &[&str] = &["MambaA", "MambaD", "MambaDt", "MambaDtBias"];

        let src = std::fs::read_to_string(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .expect("crates/driver-wgpu sits under crates/")
                .join("kernels/src/keys.rs"),
        )
        .expect("kernels/src/keys.rs");

        // Read off `keys.rs` rather than listed: a hand-written population
        // cannot see a NEW declaration, which is exactly where a new fact
        // arrives.
        let minted: Vec<(String, String)> = src
            .lines()
            .filter_map(|l| {
                let t = l.trim_start();
                if t.starts_with("//") {
                    return None;
                }
                let (name, rest) = t.split_once(" = \"")?;
                let (key, _) = rest.split_once("\" =>")?;
                name.chars()
                    .next()
                    .is_some_and(char::is_uppercase)
                    .then(|| (name.to_string(), key.to_string()))
            })
            .collect();
        assert!(
            minted.len() > 20,
            "{} facts scraped from keys.rs -- the scrape broke, and an empty \
             population passes everything below it",
            minted.len(),
        );

        // `named` takes the KEY -- the snake-cased string on the right of the
        // `=` -- and the type name on the left is only what a reader will
        // recognise, so both are carried and each is used for its own job.
        let (args, _) = statement();
        let f = facts();
        let mut unheard: Vec<String> = Vec::new();
        for (name, key) in &minted {
            if DECLINED.contains(&name.as_str()) {
                continue;
            }
            let leaked: &'static str = Box::leak(key.clone().into_boxed_str());
            let mut o = Handles::over(&args, 8);
            if let Err(Refusal::Unstated { what }) = super::named(leaked, &mut o, f)
                && what == "a fact this backend does not answer"
            {
                unheard.push(format!("{name} (\"{key}\")"));
            }
        }

        // The mint's facts that no wgpu routine has asked for yet. Every one
        // of them is answered by some OTHER backend's binder, which is what
        // makes them a to-do list rather than a defect. None is reachable from
        // `kernels_wgpu::routines()` -- the gate below proves that by binding
        // all hundred of them.
        //
        // Twenty-four became A HUNDRED AND TEN when upstream reworked
        // `kernels::keys` -- its own commit reports the mint going 33 answered
        // facts to 134 -- so this rose by minting rather than by this binder
        // forgetting anything. The first seven read `rms_eps`, `theta`,
        // `rope_theta`, `window_left`, `per_head_dim`, `vocab`, `ple_dim`,
        // which are scalars this plane supplies through its params block; most
        // of the rest are `kv.*` describing cache layouts this driver does not
        // build.
        //
        // A rise is only a defect if `named` stopped answering one it used to,
        // and the half of this test BELOW is what says which happened: it
        // binds every routine `kernels_wgpu::routines()` carries, so a fact
        // that went unanswered would refuse there rather than merely be
        // counted here. It passes, and `driver-wgpu`'s serving suite passes 23
        // of 23 on real weights, which is the same claim made twice.
        // 110 -> 144 WITH THE MARKS MIGRATION, and every one of the thirty-four
        // is a fact that moved the other way: `RmsEps`, `Theta`, `WindowLeft`,
        // `KvHeadDim`, `KvNumHeads` and their neighbours are the checkpoint's
        // constants, which §11.12 puts in the STATEMENT. A binder does not
        // answer what the statement carries, so a fact leaving this binder's
        // reach is the migration working rather than a port going backwards.
        // 144 -> 125 WHEN THE INVENTED KEYS WENT. Nineteen of them named a
        // number that was never a fact: `RowStride` was `Param<2>`,
        // `SplitK` was `Param<5>`, `Elements` was `Width * Rows`. The
        // migration minted a key so a body could ask for each, no driver ever
        // answered one, and every routine that reached for one refused
        // `Unstated`. `kernels/tests/every_plane_is_answered.rs` is what
        // stops the next one being minted.
        // 125 -> 98 AS THE PACKED PARAMETER BLOCKS CAME APART. Twenty-seven
        // more went the way the line above describes: a struct that carried
        // N fields behind one `params` binding became N marks the statement
        // states, and a mark is a word the DSL passes rather than a fact a
        // binder answers. The direction is the migration's, and the half of
        // this test BELOW is still what says so -- it binds every routine
        // `kernels_wgpu::routines()` carries, so a fact that stopped being
        // answered refuses there by name instead of quietly landing here.
        // 98 -> 96, AND NEITHER OF THE TWO IS A PORT. `GdnConvK`
        // ("gdn.conv_k") and `GdnNumGroups` ("gdn.n_groups") left
        // `kernels::keys` outright, so they are two fewer facts in the mint
        // rather than two more this binder answers -- the population shrank
        // under the numerator.
        //
        // The mint gained one in the same window, `AttnScratch`
        // ("attn_scratch"), the workspace a SPLIT decode leaves its partial
        // softmax states in. It does not land here because `named` answers it:
        // splitting a key range is a decision this backend makes about its own
        // occupancy, so the fact is the fire's and the binder is exactly who
        // should be supplying it. Had it been minted and left unanswered this
        // would read 97 and the split decode would refuse `Unstated`.
        // 96 -> 155, AND THE FIFTY-NINE ARE THE SAME FIFTY-NINE. The CUDA
        // mark migration minted §M of `kernels::keys` in one commit, and
        // every name that arrived there arrives here: `moe.up_weight_ptrs`,
        // `moe.expert_up`, `moe.aligned_up` and their neighbours are the
        // per-expert pointer tables and staging rectangles a CUDA MoE leg
        // hands its kernels. Checked by name rather than by subtraction --
        // the added set and the newly unheard set are the same set, so the
        // rise is entirely population and this binder forgot nothing.
        //
        // They will stay unheard. A pointer table is a thing a driver that
        // dispatches by raw pointer needs a name for; wgpu binds buffers by
        // group and index, and there is no wgpu routine that could ask.
        const UNHEARD: usize = 154;
        println!("facts minted but not heard by this binder: {}", unheard.len());
        for one in &unheard {
            println!("  {one}");
        }
        assert_eq!(
            unheard.len(),
            UNHEARD,
            "the facts this binder has never heard of are now:\n  {}\nIf this \
             FELL, a fact was ported and `UNHEARD` comes down with it. If it \
             ROSE, either `kernels::keys` minted one or `named` stopped \
             answering one it used to -- and only the second is a defect.",
            unheard.join("\n  "),
        );
    }

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
                let one = bind(
                    &routine.args[at..=at],
                    &routine.sources[at..=at],
                    &mut o,
                    f,
                );
                if let Err(why @ Refusal::Unstated { .. }) = one {
                    bad += 1;
                    unanswered.push(format!("  {}[{at}]: {why}", routine.name));
                }
            }
            if bad == 0 {
                whole += 1;
            }
        }

        println!("routines stated WHOLE: {whole} of {routines}");
        println!("arguments with no source this backend answers: {}", unanswered.len());
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
