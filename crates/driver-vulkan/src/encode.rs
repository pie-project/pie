//! [`Encode`] over this driver's own resources: what a routine body
//! dispatches through.
//!
//! `kernels-vulkan` declares routines and cannot run them. Its
//! `[dependencies]` is `kernels` and nothing else — `ash` is a dev-dependency,
//! deliberately — so it names no device, no `vk::Buffer` and no pipeline. It
//! states an ENTRYPOINT and a number of LANES; everything from there is this
//! crate's.
//!
//! This is `.wiki/kernel-x/refactor-bigplan.md` §7 Stage 2, and the order it
//! insists on is the reason this file exists before any family beyond
//! `sample` has crossed: the seam lands first, with behaviour unchanged, so
//! that every family port afterwards has somewhere to land. Starting with
//! machinery alone starts with code no caller reaches.
//!
//! # What this does that a body must not
//!
//! **The division into workgroups.** `[numthreads]` is stated in the Slang
//! text and recovered from the SPIR-V's `OpExecutionMode LocalSize` — which
//! [`crate::spirv`]'s own comment calls *"the divisor a grid is built with"* —
//! so the divisor is a property of the shader rather than of the launch. A
//! body that divided by it would carry a second copy of a number it cannot
//! see. Here it is one `div_ceil` per axis, in one place, exactly as
//! [`crate::geometry::groups`] has always done it.
//!
//! **Choosing the capability tier.** One entrypoint compiles to up to three
//! `.spv` modules, and the BODY picks which one it fires: it asks
//! [`Encode::best`] for the ceiling this adapter advertises and hands the
//! answer to `kernels_vulkan::module::path`, which steps down to the tier the
//! build actually compiled. What arrives here is the artifact's own name, and
//! this loads exactly that.
//!
//! It used to be the other way round — a body named a bare entrypoint and this
//! driver walked `Capability::PREFERENCE` behind it — and the crate header
//! records what that cost: the walk could not reach a tiered artifact at all,
//! so 146 cooperative-matrix modules and 20 fp16 ones were dead on every
//! device from the first commit, and nothing failed. A body that names its
//! module cannot have that bug, because a variant nothing names is a file
//! nothing reads.
//!
//! The safety the walk was protecting is kept by where the ceiling comes from:
//! a body can only compose a tier [`Encode::best`] gave it, so it still cannot
//! name one the device lacks — which would fault inside
//! `vkCreateComputePipelines` with the validation layer entirely silent.
//!
//! **The split of arguments into descriptors and scalars.** A routine hands
//! over its argument list in signature order and this reads the split off the
//! VARIANTS: [`ArgValue::Buffer`] takes a descriptor and every other variant
//! is a word of the scalar block. Whether that block is push constants or a
//! struct in a storage buffer is the MODULE's decision, read by
//! [`crate::binding::params_from`] off the reflected declaration — the
//! reachable symbols split almost evenly on it, so neither answer could be
//! assumed.
//!
//! # Why it accumulates rather than records
//!
//! A fire is a few hundred rectangles and this driver plans all of them, then
//! records them into ONE command buffer with barriers only between the pairs
//! that touch the same bytes. So a body's dispatch appends a [`Dispatch`] —
//! which stays plain data — and the recording is untouched. That is also what
//! keeps the write bit load-bearing: see [`ArgValue::Buffer`]'s `writes`.

use core::cell::RefCell;
use std::rc::Rc;

use kernels::routine::Refusal;
use kernels_vulkan::Capability;
use kernels_vulkan::routine::{ArgValue, Encode, Fire};

use crate::binding::params_from;
use crate::device::Bound;
use crate::dispatch::Dispatch;
use crate::geometry::Module;
use crate::hold::Facts;
use crate::spirv::Declared;

/// What a module declares, for one entrypoint the driver has already built.
///
/// A trait rather than a map so that the caller keeps the cache it already
/// has. `serve::fire` reads each module ONCE PER SYMBOL and not once per
/// launch — measured at 22 milliseconds of a 24-millisecond pass when it was
/// the other way round — and this must not undo that.
pub trait Reflect {
    /// The workgroup size and tile, and what the module binds.
    ///
    /// `file` is the ARTIFACT the body named and `entrypoint` the point inside
    /// it. Two arguments and not one, because they stopped being the same
    /// question: the artifact is what gets loaded, and the entrypoint is what
    /// `geometry::Module::named` reads a tile out of.
    ///
    /// `None` when this build has no module under that name, which is a
    /// refusal and not a panic: a routine may name an instantiation this build
    /// did not produce. There is no walk down to a lesser tier here — the body
    /// already resolved that when it composed `file`, and a driver that walked
    /// again could only disagree with it.
    ///
    /// # Why the declaration is shared rather than borrowed
    ///
    /// A cache that PARSES on a miss must mutate on a miss, and a `RefCell`
    /// cannot hand a borrow of its contents back out past the guard. The
    /// entrypoint is not knowable before the body runs -- that is the whole
    /// point of the crossing, and a routine that instantiates an axis spells
    /// a name no plan carries -- so eager population is not open to this
    /// either. An [`Rc`] is the smallest thing that lets the cache stay lazy:
    /// one refcount bump per dispatch, against the few-thousand-word SPIR-V
    /// walk it exists to avoid.
    fn of(&self, file: &str, entrypoint: &str) -> Option<(Module, Rc<Declared>)>;

    /// The highest tier this adapter advertises, for a body to compose with.
    ///
    /// The device's ceiling and not a choice: [`kernels_vulkan::module::path`]
    /// steps down from it to what the build compiled. It is here because the
    /// body needs it and the encoder is what the body holds.
    fn best(&self) -> Capability;
}

/// One fire's worth of driver, handed to a routine body.
///
/// Borrows rather than owns, for the reason the whole `dyn Encode` shape
/// exists: the arena, the resolved operands and the module cache all live in
/// the caller's frame for the length of one fire.
pub struct Encoder<'a, 'h, R: Reflect> {
    reflect: &'h R,
    /// The statement's scalar run, for a body that forwards
    /// [`crate::hold::Handles::params_block`].
    ///
    /// Held rather than passed per dispatch because it is the STATEMENT's,
    /// and every rectangle a body fires belongs to one statement.
    block: &'h [u32],
    /// Handle to bound range. A routine's [`ArgValue::Buffer`] handle is an
    /// index into this, and the caller decides what a buffer is — which is
    /// the same contract [`crate::lowering`] has always had.
    ///
    /// Two lifetimes, and the split is load-bearing. `'a` is the DEVICE
    /// memory a [`Bound`] points into — the arena and the weight store, which
    /// outlive the whole fire — while `'h` is the arm's own handle vector,
    /// which lives for one launch. Fused, every [`Dispatch`] this produced
    /// would borrow a `Vec` that is dropped before the command buffer is
    /// recorded, and the planning pass could not accumulate them.
    buffers: &'h [Bound<'a>],
    /// The traced op these dispatches are attributed to, so a refusal points
    /// at a statement rather than at a routine.
    op: u32,
    /// `Encode::fire` takes `&self`, because the machinery hands a body
    /// `&B::Ctx`. Accumulation is mutation; interior mutability is the whole
    /// of the difference and it is confined here.
    out: RefCell<Vec<Dispatch<'a>>>,
    /// WHAT THIS FIRE ANSWERS, for a body that asks.
    ///
    /// `Env` left the parameter list, so a fact only the fire can answer —
    /// a page table, the KV pool, this batch's row count — is no longer bound
    /// into `values` before the body runs. The body asks for it instead, and
    /// this is what answers: the same [`crate::hold::Handles`] the binder used
    /// and the same [`Facts`], through the same [`kernels::bind::one`].
    ///
    /// `RefCell` because answering MINTS: a staged fact takes a handle, which
    /// is a mutation of the handle vector, and the body holds only a `&self`.
    /// The binder's own borrow has ended by the time a body runs, so the two
    /// never overlap.
    ///
    /// `None` on an encoder built for a probe, which has no fire behind it.
    answers: Option<(&'h RefCell<crate::hold::Handles<'a, 'h>>, Facts)>,
}

impl<'a, 'h, R: Reflect> Encoder<'a, 'h, R> {
    /// A view for one op's worth of dispatches.
    pub fn new(reflect: &'h R, buffers: &'h [Bound<'a>], block: &'h [u32], op: u32) -> Self {
        Self {
            reflect,
            buffers,
            block,
            op,
            out: RefCell::new(Vec::new()),
            answers: None,
        }
    }

    /// The same view, able to ANSWER a body that asks.
    ///
    /// Separate from [`Self::new`] because a probe encoder has no fire behind
    /// it and must not pretend to: a body that asks on one gets
    /// [`Refusal::Unstated`], which is the honest answer.
    #[must_use]
    pub fn answering(
        mut self,
        handles: &'h RefCell<crate::hold::Handles<'a, 'h>>,
        facts: Facts,
    ) -> Self {
        self.answers = Some((handles, facts));
        self
    }

    /// What the bodies asked for, in the order they asked.
    #[must_use]
    pub fn finish(self) -> Vec<Dispatch<'a>> {
        self.out.into_inner()
    }
}

/// The scalar arguments of a call, as the words a parameter block is built
/// from.
///
/// One word per scalar in signature order, and TWO for an
/// [`ArgValue::Usize`], low first: nothing in this shader tree declares a
/// 64-bit integer, so a shader that receives an extent receives it as two
/// `uint`s. That is the same convention `kernels-wgpu` states as
/// `vec2<u32>`, and it is why [`kernels_vulkan::routine::Vulkan`]'s `USIZE`
/// spelling is empty rather than a guess.
///
/// An extent is also ALIGNED to its own width, which a packer that only
/// concatenated would get wrong. `PIE_STRIDE` is `uint2`, whose push-constant
/// alignment is eight bytes, so `attn/kv_write.slang`'s
/// `struct Push { int head_dim; PIE_STRIDE k_head_stride; ... }` puts the
/// first stride at offset 8 and leaves a four-byte hole after `head_dim`.
/// This is the same rule `kernels_vulkan::push_layout` applies to a row --
/// `at.next_multiple_of(size)` -- written here for the routine path, which
/// packs from a signature rather than from a table.
///
/// The tail is NOT rounded up. `push_size` rounds a row's block to its widest
/// member because a `VkPushConstantRange` covering the whole block must be,
/// but the range this path is checked against is the one reflected out of the
/// SPIR-V, which ends where the last member ends: `sdpa_vector.slang`'s block
/// is 44 bytes and not 48.
///
/// Getting this wrong is loud rather than silent, which is the only reason
/// this was ever safe to leave: `Device::dispatch` refuses a push run whose
/// length is not exactly the pipeline's range. Twenty bytes where the module
/// declared twenty-four is a `Fault::PushRange` and not a shader reading a
/// stride's high half as its low one.
fn words(args: &[ArgValue]) -> Vec<u32> {
    // Sized rather than grown: an `ArgValue::Usize` is two words and the rest
    // are one, so `args.len() + 1` is the answer for every call this tree
    // makes and an over-estimate for none of them.
    let mut out = Vec::with_capacity(args.len() + 1);
    for a in args {
        match *a {
            // A raised view is HOST data the body already read; it reaches
            // no shader and packs no word.
            ArgValue::Buffer { .. } | ArgValue::Raised(_) => {}
            ArgValue::I32(v) => out.push(v.cast_unsigned()),
            ArgValue::U32(v) => out.push(v),
            ArgValue::F32(v) => out.push(v.to_bits()),
            ArgValue::Usize(v) => {
                if out.len() % 2 == 1 {
                    out.push(0);
                }
                out.push(u32::try_from(v & 0xffff_ffff).unwrap_or(u32::MAX));
                out.push(u32::try_from(v >> 32).unwrap_or(u32::MAX));
            }
        }
    }
    out
}

impl<R: Reflect> Encode for Encoder<'_, '_, R> {
    fn best(&self) -> Capability {
        self.reflect.best()
    }

    fn resolve(&self, ty: kernels::Ty, source: kernels::Source) -> Result<ArgValue, Refusal> {
        let (handles, facts) = self.answers.ok_or(Refusal::Unstated {
            what: "a fact, on an encoder with no fire behind it",
        })?;
        crate::bind::one(ty, source, &mut handles.borrow_mut(), facts)
    }

    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Refusal> {
        // A body with nothing to do should have refused already; a zero here
        // would become `vkCmdDispatch(0, 1, 1)`, which is legal Vulkan that
        // runs nothing and reports success over a buffer that kept its zeros.
        // This backend has paid for that once: a shared expert's gate came
        // back untouched and every routed token was combined under
        // `sigmoid(0)`.
        if fire.lanes.contains(&0) {
            return Err(Refusal::Empty {
                what: "the lanes a routine asked for",
            });
        }

        let (module, declared) = self
            .reflect
            .of(fire.file, fire.entrypoint)
            .ok_or(Refusal::Undeclared)?;
        let declared = &*declared;

        // The divisor, off the shader rather than off a body.
        let local = module.local;
        let groups = [
            fire.lanes[0].div_ceil(local.at(0)),
            fire.lanes[1].div_ceil(local.at(1)),
            fire.lanes[2].div_ceil(local.at(2)),
        ];

        let mut buffers = Vec::with_capacity(args.len());
        let mut writes = Vec::with_capacity(args.len());
        // THE LIVE HANDLE LIST, NOT THE SNAPSHOT `Encoder::new` WAS GIVEN.
        // Binding the column mints a handle per operand and the planner
        // copies that list out BEFORE it runs the body -- but a body that
        // asks mints more, into the `Handles` the cell holds, and their
        // indices point past the copy. The lookup below then answered
        // `Refusal::Absent { what: "a buffer" }` for every fact a body reached
        // through `ctx.ask`: the token ids, the positions table, the KV
        // pages. `layout::embed_gather_mb` asks for the ids in its first line,
        // so this was the first statement of every fire this driver planned.
        //
        // AND THE STAGED BLOCK IS LIVE FOR THE SAME REASON: `ctx.params()`
        // goes through the same resolver, so a body that forwards its own run
        // mints the block while it runs -- after the snapshot was taken.
        //
        // A probe encoder has no cell and keeps the snapshot, which is right:
        // nothing can have minted anything. `driver-metal`'s planner carries
        // this same pair of lines, for the same defect.
        let live = self.answers.map(|(cell, _)| cell.borrow());
        let bounds = live
            .as_deref()
            .map_or(self.buffers, crate::hold::Handles::bound);
        let staged = live.as_deref().map(crate::hold::Handles::staged);
        let block: &[u32] = staged.as_deref().unwrap_or(self.block);
        // Where the caller's scalar block goes in the DENSE list a descriptor
        // set is written from. A module that reads its parameters from a
        // storage buffer binds that buffer at an index its own ABI chooses --
        // `combine_sorted` at 3 of 5, `route_sort` at 4 of 6, each with an
        // operand after it -- so it is taken from WHERE THE BODY PUT IT in
        // its argument list and never guessed at "one past the operands".
        let mut minted: Option<usize> = None;
        for a in args {
            // EVERY FIELD NAMED, and that is the point. This split used to read
            // `if let ArgValue::Buffer { handle, writes: w }` beside a separate
            // `Shaped` variant that carried the rectangle — and an `if let`
            // does not have to be exhaustive, so every shaped operand was
            // skipped here silently: never pushed, never bound, never in a
            // descriptor set. One variant makes the same mistake a compile
            // error, which is how this line was found.
            if let ArgValue::Buffer {
                handle, writes: w, ..
            } = *a
            {
                if handle == crate::hold::BLOCK {
                    if minted.is_some() {
                        return Err(Refusal::Device {
                            why: "a routine forwarded its parameter block twice",
                        });
                    }
                    minted = Some(buffers.len());
                    continue;
                }
                let bound = bounds
                    .get(handle as usize)
                    .ok_or(Refusal::Absent { what: "a buffer" })?;
                buffers.push(*bound);
                writes.push(w);
            }
        }

        // A body that forwarded the sentinel states its own block; one that
        // did not leaves the scalars to be read off the module, which is
        // either a push range or nothing.
        let (params, block_at) = if let Some(at) = minted {
            // The statement's run, then whatever the body packed in after it.
            // `row_gather`'s block is `{ width, count }` where `width` is the
            // trace's scalar and `count` is the routine's own -- eight bytes,
            // and four of them would be a struct whose tail reads as zero,
            // which is a gather of no rows reporting success.
            let mut held = Vec::with_capacity(block.len() + args.len() + 1);
            held.extend_from_slice(block);
            held.extend(words(args));
            // A statement with nothing to say still gets one word: the shader
            // dereferences the pointer whether or not it reads a field, and a
            // descriptor over an empty range is a device fault.
            if held.is_empty() {
                held.push(0);
            }
            let mut bytes = Vec::with_capacity(held.len() * 4);
            for word in &held {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            (crate::binding::Params::Block { at, bytes }, Some(at))
        } else {
            // By VALUE first, because a routine may hand over a 64-bit
            // stride and only the value knows it is two words wide. The word
            // run is still what the parameter-BUFFER search wants: a block is
            // a std430 struct, so its image is the padded run and not the
            // members placed at reflected offsets.
            let params = match crate::binding::push_from(args, declared) {
                Some(placed) => placed,
                None => params_from(&words(args), declared).map_err(|_| Refusal::Device {
                    why: "the module has no room for the scalars this routine states",
                })?,
            };
            let at = match params {
                crate::binding::Params::Block { at, .. } => Some(at),
                crate::binding::Params::Push(_) | crate::binding::Params::None => None,
            };
            (params, at)
        };

        let real = declared.bindings as usize - declared.holes();
        if buffers.len() + usize::from(block_at.is_some()) != real {
            return Err(Refusal::Arity {
                want: real,
                got: buffers.len() + usize::from(block_at.is_some()),
            });
        }

        self.out.borrow_mut().push(Dispatch {
            file: fire.file.to_owned().into(),
            symbol: fire.entrypoint.to_owned().into(),
            buffers,
            writes,
            params,
            block_at,
            groups,
            op: self.op,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::Buffer;

    /// A `Declared` this test can state without a SPIR-V module.
    fn declared(bindings: u32, push: &[u32]) -> Declared {
        Declared {
            local: [1, 1, 1],
            bindings,
            used: vec![true; bindings as usize],
            writable: vec![true; bindings as usize],
            reads_workgroup_count: false,
            grid_axes: [true, true, false],
            push_offsets: push.to_vec(),
            block_bytes: vec![None; bindings as usize],
        }
    }

    /// One module, at whatever local size the test wants to divide by.
    struct One {
        entrypoint: &'static str,
        local: [u32; 3],
        declared: Rc<Declared>,
    }

    impl Reflect for One {
        // The ARTIFACT is ignored and the ENTRYPOINT is what has to match.
        // A body composes both -- a file per tier, a point inside it -- and
        // this fixture holds one module, so the point is the whole question.
        fn of(&self, _file: &str, entrypoint: &str) -> Option<(Module, Rc<Declared>)> {
            (entrypoint == self.entrypoint).then(|| {
                (
                    Module::named(entrypoint, self.local),
                    Rc::clone(&self.declared),
                )
            })
        }

        fn best(&self) -> Capability {
            Capability::Baseline
        }
    }

    /// A resolver that answers everything with one placeholder.
    ///
    /// `argmax_logits` asks for its ROW COUNT and nothing else, and a row
    /// count is a `Facts` field rather than a resource, so nothing below ever
    /// reaches these. They are here because `Handles` takes a resolver and a
    /// body that grew an ask would otherwise fail in a way that read as a
    /// fixture bug.
    struct Anything(Buffer);
    impl crate::binding::Resolve for Anything {
        fn weight(&self, _: &str) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn named(&self, _: model_ir::trace::ValueId) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn kv(&self, _: u16, _: bool) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn table(&self, _: crate::binding::FireTable) -> Option<&Buffer> {
            Some(&self.0)
        }
        fn number(&self, _: crate::binding::FireNumber) -> Option<u32> {
            Some(0)
        }
    }

    /// What one run of `argmax_logits` came to, in owned pieces.
    ///
    /// Owned because a `Dispatch` borrows the buffers it was bound from, and
    /// the helper below owns those: the four fields here are everything these
    /// tests ask about and none of them borrows.
    struct Ran {
        result: Result<(), Refusal>,
        fired: Vec<(String, [u32; 3], u32, Vec<bool>)>,
    }

    /// Bind `argmax_logits`'s operands, run its BODY, and report what it fired.
    ///
    /// The whole seam, because a routine is no longer a function a test can
    /// call with four marked buffers: it takes a `Ctx` and reads its row count
    /// through it, so there has to be a binder and a `Facts` behind the
    /// encoder for the body to reach anything at all. This is
    /// `serve::plan_routine` with the device taken out.
    fn run(reflect: &impl Reflect, rows: u32, op: u32) -> Ran {
        let routine = kernels_vulkan::routines()
            .into_iter()
            .find(|r| r.name == "argmax_logits")
            .expect("the crate serves `argmax_logits`");
        let bufs: Vec<Buffer> = (0..4).map(|_| Buffer::placeholder(64)).collect();
        let args: Vec<Bound<'_>> = bufs.iter().map(Bound::whole).collect();
        let widths = [1024i32; 4];
        let ins = [0usize, 2];
        let outs = [1usize, 3];
        let weights: [usize; 0] = [];
        let params: [Option<u32>; 0] = [];
        let anything = Anything(Buffer::placeholder(64));
        let handles = RefCell::new(crate::hold::Handles::new(
            &args, &widths, &ins, &outs, &weights, &params, &anything,
        ));
        let facts = crate::hold::Facts {
            rows,
            width: 1024,
            in_width: 1024,
            ..Default::default()
        };
        let values = crate::bind::bind(
            routine.args,
            routine.sources,
            &mut handles.borrow_mut(),
            facts,
        )
        .expect("the four operands bind");
        let taken = handles.borrow().bound().to_vec();
        let staged = handles.borrow().staged();
        let enc = Encoder::new(reflect, &taken, &staged, op).answering(&handles, facts);
        let result = (routine.body)(&enc, &values);
        let fired = enc
            .finish()
            .into_iter()
            .map(|d| (d.symbol.into_owned(), d.groups, d.op, d.writes))
            .collect();
        Ran { result, fired }
    }

    /// The grid is the body's LANES divided by the module's own workgroup
    /// size, rounded up, and the body never sees the divisor.
    ///
    /// This is the whole of what the routine shape moves. Under `kernel!` the
    /// number came from `geometry::lanes` matching a `LaunchRule`, and a rule
    /// the table named but the driver did not implement was a compile error
    /// by design. Here the body states elements of work and the divisor comes
    /// off `OpExecutionMode LocalSize`, which is the only place it is written
    /// down.
    #[test]
    fn the_grid_is_the_lanes_divided_by_the_modules_own_workgroup() {
        let one = One {
            entrypoint: "argmax_logits_bfloat16",
            local: [256, 1, 1],
            declared: Rc::new(declared(4, &[])),
        };
        let ran = run(&one, 5, 7);
        ran.result.expect("five rows is a launch");

        assert_eq!(ran.fired.len(), 1);
        assert_eq!(ran.fired[0].0, "argmax_logits_bfloat16");
        assert_eq!(
            ran.fired[0].1,
            [4, 5, 1],
            "1024 lanes over a 256-wide workgroup is 4, and one group per row on y"
        );
        assert_eq!(
            ran.fired[0].2, 7,
            "the refusal points at the statement, not the fn"
        );
    }

    /// The write bit reaches the driver, and it comes from the argument TYPE.
    ///
    /// This is what decides whether two neighbouring dispatches get a barrier
    /// between them. Getting it wrong in the permissive direction is not a
    /// crash: it is a race, and a race here is a plausible number. Under
    /// `kernel!` it was read off the row's operand types; the row is gone from
    /// the launch path once a family crosses, so `BufMut` against `Buf` in the
    /// signature is now the only statement of it anywhere.
    #[test]
    fn the_buffers_a_kernel_may_write_are_the_ones_its_signature_spells_bufmut() {
        let one = One {
            entrypoint: "argmax_logits_bfloat16",
            local: [1024, 1, 1],
            declared: Rc::new(declared(4, &[])),
        };
        let ran = run(&one, 1, 0);
        ran.result.expect("one row is a launch");

        assert_eq!(
            ran.fired[0].3,
            vec![false, true, false, true],
            "logits and params are read, next_token and eos_flag are written"
        );
    }

    /// An entrypoint this adapter has no module for is refused, not dispatched.
    #[test]
    fn a_routine_naming_a_module_this_build_did_not_produce_is_refused() {
        let one = One {
            entrypoint: "something_else",
            local: [1, 1, 1],
            declared: Rc::new(declared(4, &[])),
        };
        let ran = run(&one, 1, 0);
        assert_eq!(ran.result, Err(Refusal::Undeclared));
        assert!(ran.fired.is_empty());
    }

    /// A body that states more buffers than the module binds is refused here,
    /// rather than becoming a descriptor set no device accepts.
    #[test]
    fn a_launch_that_does_not_fill_the_modules_bindings_is_refused() {
        let one = One {
            entrypoint: "argmax_logits_bfloat16",
            local: [1, 1, 1],
            declared: Rc::new(declared(6, &[])),
        };
        let ran = run(&one, 1, 0);
        assert_eq!(ran.result, Err(Refusal::Arity { want: 6, got: 4 }));
    }

    /// A `usize` becomes two words, low first.
    ///
    /// Nothing in the Slang tree declares a 64-bit integer, so an extent
    /// reaches a shader as two `uint`s -- which is why `Vulkan`'s `USIZE`
    /// spelling is empty rather than a guess at one.
    #[test]
    fn an_extent_is_two_words_low_first() {
        assert_eq!(words(&[ArgValue::Usize(0x0000_0002_0000_0001)]), vec![1, 2]);
        assert_eq!(
            words(&[
                ArgValue::Buffer {
                    handle: 3,
                    writes: true,
                    rows: 1,
                    width: 1,
                },
                ArgValue::I32(-1),
                ArgValue::F32(1.0),
            ]),
            vec![0xffff_ffff, 1.0f32.to_bits()],
            "a buffer takes a descriptor and no word"
        );
    }

    /// An extent starts on its own eight-byte boundary, and the hole is a word
    /// the packer emits rather than a word the shader is left to find.
    ///
    /// `attn/kv_write.slang`'s contiguous branch is the shape this is for:
    /// `struct Push { int head_dim; PIE_STRIDE k_head_stride; PIE_STRIDE
    /// k_seq_stride; }`, where `PIE_STRIDE` is `uint2` and therefore
    /// eight-byte aligned, so the first stride sits at offset 8 and not at 4.
    /// Concatenating would produce five words -- and five words is twenty
    /// bytes against a twenty-four byte range, which `Device::dispatch`
    /// refuses. The nearer miss is the one worth naming: a packer that padded
    /// the TAIL instead would produce twenty-four bytes with `k_head_stride`
    /// read out of `head_dim`'s hole, dispatch happily, and address the cache
    /// at a stride of zero.
    #[test]
    fn an_extent_is_aligned_to_its_own_width() {
        assert_eq!(
            words(&[
                ArgValue::I32(64),
                ArgValue::Usize(0x0000_0002_0000_0001),
                ArgValue::Usize(0x0000_0004_0000_0003),
            ]),
            vec![64, 0, 1, 2, 3, 4],
            "the pad belongs BEFORE the first stride"
        );
        assert_eq!(
            words(&[
                ArgValue::I32(1),
                ArgValue::I32(2),
                ArgValue::Usize(3),
                ArgValue::F32(1.0),
            ]),
            vec![1, 2, 3, 0, 1.0f32.to_bits()],
            "two words in front of it is already aligned, and a pad there \
             would push `sdpa_vector.slang` past its own block"
        );
    }
}
