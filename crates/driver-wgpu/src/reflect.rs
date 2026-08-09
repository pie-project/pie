//! What a WGSL module says about itself.
//!
//! A driver holds the source anyway — `kernels_wgpu::entrypoint_source` hands
//! it the expanded text — and several of the numbers it needs to dispatch one
//! are inside that text rather than in any table: the workgroup size, the
//! bindings a bind-group layout must cover, and, for deciding whether a grid
//! may be rounded, whether the shader reads its own workgroup count.
//!
//! Reading them here rather than restating them in `kernels-wgpu`'s table is
//! not a preference. A table entry can drift from the shader it describes; a
//! value read out of the module cannot.
//!
//! # Nothing is hand-parsed, and that is the whole difference from `spirv.rs`
//!
//! `driver-vulkan/src/spirv.rs` is 993 lines of word-stream walking: opcode
//! constants, a type graph, a size recursion, a bound on how deep it will
//! follow a pointer so that a corrupt file cannot hang the driver. It has to
//! be, because SPIR-V arrives as bytes from a build directory and nothing in
//! the Rust ecosystem was going to read them for it.
//!
//! WGSL arrives as TEXT and `naga` is a WGSL front end written in Rust, so
//! this module is a walk over `naga::Module` and there is no parser here at
//! all. `naga` is depended on DIRECTLY rather than through `wgpu`'s re-export,
//! for the reason `Cargo.toml` gives: every question below is about the
//! source, so it belongs in the portable half, and that half must not need an
//! adapter to answer.
//!
//! # Which of `spirv.rs`'s answers survive, and which have no analog
//!
//! This is the part worth reading, because three of the seven do not transfer
//! and each one fails to transfer for a different reason.
//!
//! **`local` survives, and is easier.** `@workgroup_size(16, 16)` is
//! `EntryPoint::workgroup_size`, already normalised to three axes. WGSL also
//! allows an `override` to size a workgroup, which `naga` keeps separately as
//! `workgroup_size_overrides`; no shader in this tree uses one, and
//! [`Unreadable::OverriddenWorkgroup`] refuses rather than reporting the
//! default a pipeline may not end up with.
//!
//! **`bindings` survives, and means something narrower.** It counts
//! `@group(0)` only, because `kernels-wgpu`'s ABI puts every buffer operand in
//! group 0 and the one uniform block in group 1. Vulkan has a single binding
//! numbering shared by both kinds; WebGPU has a bind group per number, so
//! "highest binding plus one" is a question you have to ask of a GROUP.
//!
//! **`used` survives, and is still worth having.** `naga` keeps a
//! `GlobalVariable` the entry point never reads, exactly as `glslc` keeps a
//! `binding` decoration for a buffer a variant never touches. The consequence
//! differs: Vulkan needs a descriptor at every number up to the highest and a
//! driver has to find something to put in a hole, while `wgpu` requires the
//! bind group to MATCH its layout — and the layout is derived from this same
//! reading, so a declared-and-unread binding is a slot the shell must still
//! fill with something. Same question, same answer, different consequence for
//! getting it wrong.
//!
//! **`push_offsets` becomes [`Declared::uniform_offsets`], and the check it
//! feeds points the OTHER WAY.** `kernels-vulkan`'s `layout-10069` finding is
//! that a push block DECLARED wider than the pipeline's range is a validation
//! error: Vulkan checks the shader against the layout and complains when the
//! shader over-declares. WebGPU has no such rule, because a uniform binding is
//! a buffer: WGSL requires the BOUND range to be at least the struct's size,
//! and `wgpu` refuses a binding that is too small. So the direction of the
//! check inverts — there the module must not ask for more than the layout
//! promised, here the shell must not offer less than the module's struct
//! needs. [`Declared::uniform_bytes`] is what a shell compares its block
//! against, and being OVER is fine.
//!
//! **`reads_workgroup_count` survives unchanged.** `@builtin(num_workgroups)`
//! is a WGSL builtin and means what `gl_NumWorkGroups` means, so the rounding
//! argument in [`crate::geometry::groups`] transfers word for word.
//!
//! **`grid_axes` survives but is WEAKER, and the weakness is structural.** In
//! SPIR-V, `gl_GlobalInvocationID` is a global variable and every read of a
//! component is an `OpAccessChain` or an `OpCompositeExtract` in the entry
//! point's own instruction stream. In WGSL the builtin is a function
//! ARGUMENT, and this tree's bodies pass the whole `vec3<u32>` to a helper —
//! `norm/rms.wgsl` calls `row_base(wg)` — so the component that is really read
//! is indexed against the CALLEE's parameter. This walk follows calls to
//! recover it (see [`Declared::grid_axes`]), with a depth bound, and where it
//! cannot follow it answers "every axis" rather than "no axis", because the
//! unknown answer must not let a wrong grid pass a check.
//!
//! **`block_bytes` survives for the same reason it existed.** A parameter
//! struct in a storage buffer is what `rms_single_row`'s `params: Buf` operand
//! binds, and a buffer shorter than the struct the shader reads is a defect
//! with no symptom on this backend too: WGSL's bounds checking means an
//! out-of-range read returns zero rather than faulting, so a missing field is
//! a plausible number and not a crash.
//!
//! **`Malformed` mostly does not survive.** Truncation, a wrong magic word, a
//! zero-length instruction: all three are byte-level conditions of a binary
//! format, and this module is handed a `&str`. What is left is
//! [`Unreadable`], which is `naga`'s own parse error plus the two structural
//! conditions a table can disagree with a shader about.
//!
//! # Nothing here is cached
//!
//! Parsing a module is a few hundred microseconds and a caller does it once
//! per distinct kernel at model load, because what a caller actually caches is
//! the PIPELINE. A cache in this module would be a second lifetime to reason
//! about for a saving nobody measured.

use std::collections::{BTreeMap, BTreeSet};

use naga::{AddressSpace, Binding, BuiltIn, Expression, Handle, Module, ShaderStage};

/// The bind group `kernels-wgpu`'s ABI puts every buffer operand in.
///
/// Named rather than written as `0` at each use, because the OTHER group
/// number is equally meaningful and a bare literal cannot say which is which.
pub const STORAGE_GROUP: u32 = 0;

/// The bind group holding the one uniform block a row's scalars are fields of.
///
/// Group 1 and not another slot of group 0, for the reason `kernels-wgpu`'s
/// module docs give: a uniform in group 0 would take an index that MOVES with
/// the row's buffer count, so every shader in a family would declare its
/// params block at a different number than its neighbour, and a family's
/// shaders are one file.
pub const UNIFORM_GROUP: u32 = 1;

/// The binding within [`UNIFORM_GROUP`] the block sits at. Always zero: the
/// group holds one entry, which is the point of giving it a group.
pub const UNIFORM_BINDING: u32 = 0;

/// How deep the walk will follow a call chain looking for a grid component.
///
/// A grid position reaches at most one or two frames in this tree
/// (`main` -> `row_base`), and a bound is here because the graph is data: a
/// module `naga` accepted can still recurse, and a reflection pass that hangs
/// is worse than one that answers conservatively.
const MAX_CALL_DEPTH: u32 = 8;

/// What a module declares about how it must be launched and bound.
///
/// The same SHAPE `driver-vulkan`'s `spirv::Declared` has wherever the concept
/// survives; see this module's own docs for the three that do not survive
/// intact and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declared {
    /// `@workgroup_size(...)`, the divisor a grid is built with.
    ///
    /// Three axes always, because `naga` fills the omitted ones with 1 — WGSL
    /// writes `@workgroup_size(16, 16)` and means `(16, 16, 1)`.
    pub local: [u32; 3],
    /// One past the highest `@group(0) @binding(N)` the module declares.
    ///
    /// The COUNT of them is the wrong number, for the reason the Vulkan port
    /// records: a variant that never reads a buffer may not declare it, so a
    /// module's binding set can have HOLES, and a layout still needs an entry
    /// at every number up to the highest. `wgpu` checks a bind group against
    /// its layout entry for entry, so a hole is a slot the shell has to fill
    /// with something rather than one it can leave out.
    pub bindings: u32,
    /// Does the module read `@builtin(num_workgroups)`?
    ///
    /// When it does, the workgroup count is a QUANTITY the shader computes
    /// with and not merely a bound it is clipped to, so its grid may not be
    /// rounded up. See [`crate::geometry::groups`].
    pub reads_workgroup_count: bool,
    /// Which of the three grid axes the shader is actually indexed by.
    ///
    /// The component of `workgroup_id` or `global_invocation_id` it reads.
    /// This is what makes a grid on the WRONG AXIS detectable, and that is not
    /// hypothetical: the Vulkan port's first `Rule::Rms` put the row count on
    /// y while `norm/rms` reads its row from the x workgroup id. Every row but
    /// the first was left holding the zeros its buffer was born with, four
    /// dispatches returned success, and the lane-coverage sweeps all passed --
    /// they counted lanes and never asked which axis carried them.
    ///
    /// # Why this is a weaker answer than SPIR-V's, and which way it errs
    ///
    /// A WGSL builtin is a function argument, and this tree hands the whole
    /// `vec3<u32>` to helpers. The walk follows a call whose argument is a
    /// grid position and asks the same question of the callee's parameter, to
    /// a bounded depth. Where a grid position is used in a way the walk
    /// does not model — stored in a local, arithmetic on the whole vector, a
    /// call past the depth bound — every axis is reported.
    ///
    /// That direction is deliberate. "Every axis" makes a check that compares
    /// a rule's axes against the module's PERMISSIVE, which is a check that
    /// finds nothing; "no axis" would make the same check REJECT a correct
    /// grid, and the reader would delete the check.
    pub grid_axes: [bool; 3],
    /// The byte offset of every member of the `@group(1) @binding(0)` uniform
    /// block, in declaration order.
    ///
    /// What the SHADER thinks the block looks like, so that it can be compared
    /// to what `kernels_wgpu::uniform_layout` says the driver will write. A
    /// disagreement here is not a crash and not an error: it is the shader
    /// reading a different scalar than the one that was written -- a stride
    /// where a head count belongs -- and the result is a plausible number.
    ///
    /// Empty means the module declares no uniform block, which is legal and is
    /// what a kernel taking only buffers wants.
    pub uniform_offsets: Vec<u32>,
    /// The bytes that uniform block requires.
    ///
    /// The number a shell's own block must be at LEAST, which is the opposite
    /// direction from Vulkan's push-range check; see the module docs. Zero
    /// when there is no block.
    pub uniform_bytes: u32,
    /// Which `@group(0)` binding numbers the module actually declares.
    ///
    /// [`Self::bindings`] is one past the HIGHEST, so the two disagree
    /// wherever the set has a hole. Indexed by binding number and
    /// [`Self::bindings`] long, so a hole reads as `false` rather than as an
    /// absence a caller has to notice.
    ///
    /// `false` also covers the case a table cannot see: a global that IS
    /// declared and that the entry point never reads. `naga` keeps it, so the
    /// question survives from the Vulkan port, and it is answered here by
    /// walking the entry point's expressions and every function it calls.
    pub used: Vec<bool>,
    /// The bytes each `@group(0)` binding's block requires, when that is a
    /// fixed number.
    ///
    /// Indexed by binding number, so it is [`Self::bindings`] long and a hole
    /// is `None` like an unsized block is. `None` is the ORDINARY answer: a
    /// tensor binding is `array<u32>`, whose length is the binding's own, so
    /// its size is genuinely a property of the CALL and reporting one would be
    /// inventing it.
    ///
    /// The ones that are knowable are the PARAMETER structs -- `RmsParams` and
    /// its siblings -- and those are exactly the ones where a short binding is
    /// a defect rather than a shape. It is a quiet defect here as it is on
    /// Vulkan: WGSL requires an implementation to bounds-check, so a read past
    /// the bound range yields zero, and a missing field is a plausible number
    /// no layer objects to.
    pub block_bytes: Vec<Option<u32>>,
}

impl Declared {
    /// How many `@group(0)` binding numbers below [`Self::bindings`] the module
    /// declares and the entry point never reads.
    ///
    /// # This does not mean what `spirv::Declared::holes` means
    ///
    /// There a hole is a binding number with NO declaration at all, because
    /// `glslc` deletes the declaration of a buffer a variant never reads. The
    /// driver has nothing to put at that number and the row says `Unbound`.
    ///
    /// `naga` deletes nothing, so a hole here is a binding that EXISTS and that
    /// this entry point happens not to read -- the row still names a real
    /// tensor for it and the driver still binds one. Measured over the whole
    /// tree, 19 of the 480 entrypoints have at least one: `kv_append_paged`
    /// keeps six ring-ABI placeholder slots and every `sdpa_paged_*` declares
    /// an attention-sink buffer its non-sink variants do not read.
    ///
    /// So this is a number a caller may use to size a MINIMUM -- every read
    /// binding must be filled -- and never a number to subtract from an arity.
    /// [`crate::dispatch::plan_one`] says at its own line what subtracting it
    /// cost.
    #[must_use]
    pub fn holes(&self) -> usize {
        self.used.iter().filter(|u| !**u).count()
    }
}

/// Why a module could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Unreadable {
    /// `naga` refused the source.
    ///
    /// Carried as a string rather than as `naga::front::wgsl::ParseError`
    /// because that type is not `PartialEq` and the useful content of it is
    /// the rendered message anyway -- which for `naga` names the line and the
    /// span, so nothing is lost by rendering it early.
    Unparseable(
        /// What `naga` said, with its source location.
        String,
    ),
    /// The module declares no compute entry point.
    ///
    /// Every shader in this tree is a compute shader, so this is drift rather
    /// than a runtime condition: the source that reached here is not a kernel.
    NoComputeEntry,
    /// The module declares more than one compute entry point.
    ///
    /// Refused rather than silently taking the first, for the reason the
    /// Vulkan walk gives about two `LocalSize` modes: two entry points need
    /// two grids, and a driver that took either one would launch the other at
    /// the wrong shape. `kernels-wgpu` expands one variant at a time, so this
    /// means an `//#if` arm left two `@compute` functions in one expansion.
    TwoComputeEntries {
        /// The names it found.
        names: Vec<String>,
    },
    /// The entry point's workgroup size is an `override` expression.
    ///
    /// Legal WGSL and a shape this driver cannot dispatch: an override is
    /// resolved at pipeline creation from a value the shell supplies, so the
    /// number in the IR is not the number that will run, and
    /// [`crate::geometry`] divides a whole fire's extent by it. Refused rather
    /// than defaulted -- an undershot grid writes nothing and reports success.
    OverriddenWorkgroup,
    /// The tree has no source for this entrypoint at this tier.
    ///
    /// Forwarded from `kernels_wgpu::entrypoint_source`. For a tier above
    /// baseline this is ORDINARY and is how a driver learns to fall back; for
    /// baseline it is a defect the build should already have caught.
    NoSource(
        /// What `kernels-wgpu` said.
        String,
    ),
}

impl core::fmt::Display for Unreadable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Unparseable(why) => write!(f, "naga refused the source: {why}"),
            Self::NoComputeEntry => write!(f, "no @compute entry point"),
            Self::TwoComputeEntries { names } => {
                write!(f, "two @compute entry points: {}", names.join(", "))
            }
            Self::OverriddenWorkgroup => {
                write!(f, "the workgroup size is an override, not a literal")
            }
            Self::NoSource(why) => write!(f, "{why}"),
        }
    }
}

impl core::error::Error for Unreadable {}

/// Read what one WGSL source declares.
///
/// The source must hold exactly one `@compute` entry point, which every
/// expansion `kernels-wgpu` produces does.
///
/// # Errors
///
/// [`Unreadable`] when `naga` will not take the text, or when the module is
/// not the one-compute-entry-point shape a dispatch needs.
pub fn declared(source: &str) -> Result<Declared, Unreadable> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|e| Unreadable::Unparseable(e.emit_to_string(source)))?;
    of_module(&module)
}

/// The same, from an already-parsed module.
///
/// Split out so that a caller who has a `naga::Module` for another reason --
/// a shell that is about to hand the same source to
/// `wgpu::Device::create_shader_module` -- does not parse it twice.
///
/// # Errors
///
/// [`Unreadable`], less [`Unreadable::Unparseable`], which cannot arise from a
/// module that already exists.
pub fn of_module(module: &Module) -> Result<Declared, Unreadable> {
    let mut computes = module
        .entry_points
        .iter()
        .filter(|e| e.stage == ShaderStage::Compute);
    let entry = computes.next().ok_or(Unreadable::NoComputeEntry)?;
    if computes.next().is_some() {
        return Err(Unreadable::TwoComputeEntries {
            names: module
                .entry_points
                .iter()
                .filter(|e| e.stage == ShaderStage::Compute)
                .map(|e| e.name.clone())
                .collect(),
        });
    }
    // `workgroup_size_overrides` is `Some` only when at least one axis is an
    // override expression, so this is not the same question as "is the size a
    // literal" asked three times.
    if entry
        .workgroup_size_overrides
        .is_some_and(|axes| axes.iter().any(Option::is_some))
    {
        return Err(Unreadable::OverriddenWorkgroup);
    }

    // Every global the entry point can reach, which is the entry point's own
    // expressions plus those of every function it calls. A helper in
    // `common/*.inc.wgsl` that touches a binding makes that binding read, and
    // an analysis that stopped at the entry point would report it unread --
    // the direction that turns a real operand into a hole.
    let reachable = reachable_globals(module, entry);

    let mut highest: Option<u32> = None;
    let mut declared_at: BTreeMap<u32, (bool, Option<u32>)> = BTreeMap::new();
    let mut uniform_offsets = Vec::new();
    let mut uniform_bytes = 0;
    for (handle, global) in module.global_variables.iter() {
        let Some(binding) = &global.binding else {
            continue;
        };
        match global.space {
            AddressSpace::Storage { .. } if binding.group == STORAGE_GROUP => {
                highest = Some(highest.map_or(binding.binding, |h: u32| h.max(binding.binding)));
                declared_at.insert(
                    binding.binding,
                    (reachable.contains(&handle), fixed_size(module, global.ty)),
                );
            }
            // A uniform in group 0 would be a row that ignored the ABI, and a
            // storage buffer in group 1 likewise. Both are counted where they
            // sit rather than where they were meant to sit, so a shader that
            // put them elsewhere reads as the layout it really has.
            AddressSpace::Uniform
                if binding.group == UNIFORM_GROUP && binding.binding == UNIFORM_BINDING =>
            {
                (uniform_offsets, uniform_bytes) = uniform_layout(module, global.ty);
            }
            _ => {}
        }
    }

    let bindings = highest.map_or(0, |h| h + 1);
    let mut used = vec![false; bindings as usize];
    let mut block_bytes = vec![None; bindings as usize];
    for (at, (read, size)) in declared_at {
        if let Some(slot) = used.get_mut(at as usize) {
            *slot = read;
        }
        if let Some(slot) = block_bytes.get_mut(at as usize) {
            *slot = size;
        }
    }

    let builtins = entry_builtins(entry);
    Ok(Declared {
        local: entry.workgroup_size,
        bindings,
        reads_workgroup_count: builtins.workgroup_count,
        grid_axes: axes_read(module, &entry.function, &builtins.grid_args, 0),
        uniform_offsets,
        uniform_bytes,
        used,
        block_bytes,
    })
}

/// What one of `kernels-wgpu`'s 480 entrypoints declares, at one tier.
///
/// The whole path from a name to a `Declared`: `kernels-wgpu` expands the
/// variant's `//#include`s and `//#if` arms and defines, `naga` parses the
/// result, and this reads it. Nothing on disk, nothing from a build directory,
/// no device — which is what makes a check of every entrypoint in the table a
/// thing that runs in a plain `cargo test`.
///
/// Nothing is cached. A caller that dispatches the same kernel a thousand
/// times caches the PIPELINE, and a cache here would be a second lifetime for
/// a saving nobody measured.
///
/// # Errors
///
/// [`Unreadable::NoSource`] when the tree declares no such variant at this
/// tier -- which above [`kernels_wgpu::Capability::Baseline`] is the ordinary
/// answer and a driver's cue to fall back -- and the rest of [`Unreadable`]
/// when it does and the source is not a dispatchable module.
pub fn entrypoint(name: &str, tier: kernels_wgpu::Capability) -> Result<Declared, Unreadable> {
    let source = kernels_wgpu::entrypoint_source(name, tier)
        .map_err(|why| Unreadable::NoSource(why.to_string()))?;
    declared(&source)
}

/// The builtins an entry point takes, as the two questions this module asks.
struct Builtins {
    /// Does it take `@builtin(num_workgroups)`?
    workgroup_count: bool,
    /// Which of its arguments carry a grid POSITION -- a workgroup id or a
    /// global invocation id.
    ///
    /// `local_invocation_id` is deliberately not one: it indexes within a
    /// workgroup and says nothing about which grid axis carries the work,
    /// which is the question [`Declared::grid_axes`] is for.
    grid_args: BTreeSet<u32>,
}

/// Which builtins the entry point's own signature names.
fn entry_builtins(entry: &naga::EntryPoint) -> Builtins {
    let mut out = Builtins {
        workgroup_count: false,
        grid_args: BTreeSet::new(),
    };
    for (at, arg) in entry.function.arguments.iter().enumerate() {
        // A struct argument carries its builtins on its MEMBERS rather than on
        // itself, so its own binding is `None`. No shader in this tree writes
        // one; if one does, its grid positions are simply not found, and
        // `axes_read` then answers no axis for an argument nobody indexed --
        // which `grid_axes`'s own doc names as the case where the walk gives a
        // lower bound.
        let Some(Binding::BuiltIn(builtin)) = arg.binding else {
            continue;
        };
        match builtin {
            BuiltIn::NumWorkGroups => out.workgroup_count = true,
            // `GlobalInvocationId` is `WorkGroupId * WorkGroupSize +
            // LocalInvocationId`, so it says the same thing about WHICH axis
            // carries the work.
            BuiltIn::WorkGroupId | BuiltIn::GlobalInvocationId => {
                out.grid_args.insert(at as u32);
            }
            _ => {}
        }
    }
    out
}

/// Which components of `grid_args` this function reads, following calls.
///
/// `grid_args` names argument INDICES of `function` that hold a grid position.
/// For an entry point they come from its `@builtin` attributes; for a callee
/// they come from which of the caller's arguments were passed into it, which
/// is what makes `row_base(wg)` legible.
fn axes_read(
    module: &Module,
    function: &naga::Function,
    grid_args: &BTreeSet<u32>,
    depth: u32,
) -> [bool; 3] {
    let mut axes = [false; 3];
    if grid_args.is_empty() {
        return axes;
    }
    // The expressions that ARE a grid position. One per argument at most, but
    // `naga` may emit the same `FunctionArgument` expression once and reuse
    // the handle, so this is a set rather than a map.
    let positions: BTreeSet<Handle<Expression>> = function
        .expressions
        .iter()
        .filter_map(|(handle, expr)| match expr {
            Expression::FunctionArgument(i) if grid_args.contains(i) => Some(handle),
            _ => None,
        })
        .collect();

    // Every one of them has to be accounted for, or the answer is "every
    // axis". See `Declared::grid_axes` for why that is the safe direction.
    let mut explained: BTreeSet<Handle<Expression>> = BTreeSet::new();
    for (_, expr) in function.expressions.iter() {
        match expr {
            // `wg.x` -- the ordinary case, and the only one that names an axis.
            Expression::AccessIndex { base, index } if positions.contains(base) => {
                explained.insert(*base);
                if let Some(slot) = axes.get_mut(*index as usize) {
                    *slot = true;
                }
            }
            // `wg[i]` for a computed `i`. Legal, and the component is not
            // knowable, so every axis is in play.
            Expression::Access { base, .. } if positions.contains(base) => {
                explained.insert(*base);
                axes = [true; 3];
            }
            _ => {}
        }
    }

    // And the calls, which is where this tree's grid positions actually go.
    if depth < MAX_CALL_DEPTH {
        for (callee, arguments) in calls(&function.body) {
            let passed: BTreeSet<u32> = arguments
                .iter()
                .enumerate()
                .filter(|(_, a)| positions.contains(a))
                .map(|(at, _)| at as u32)
                .collect();
            if passed.is_empty() {
                continue;
            }
            for a in arguments.iter().filter(|a| positions.contains(a)) {
                explained.insert(*a);
            }
            let inner = axes_read(module, &module.functions[callee], &passed, depth + 1);
            for (slot, got) in axes.iter_mut().zip(inner) {
                *slot |= got;
            }
        }
    }

    if positions.iter().any(|p| !explained.contains(p)) {
        return [true; 3];
    }
    axes
}

/// Every `Statement::Call` in a block, nested blocks included.
///
/// Written out because `naga::Block` is a tree and a call to a helper that
/// reads a grid axis is as likely to sit inside an `if` as at the top level --
/// `norm/rms.wgsl`'s `row_base` is called unconditionally, but nothing makes
/// that a rule.
fn calls(block: &naga::Block) -> Vec<(Handle<naga::Function>, Vec<Handle<Expression>>)> {
    let mut out = Vec::new();
    for statement in block.iter() {
        match statement {
            naga::Statement::Call {
                function,
                arguments,
                ..
            } => out.push((*function, arguments.clone())),
            naga::Statement::Block(inner) => out.extend(calls(inner)),
            naga::Statement::If { accept, reject, .. } => {
                out.extend(calls(accept));
                out.extend(calls(reject));
            }
            naga::Statement::Loop {
                body, continuing, ..
            } => {
                out.extend(calls(body));
                out.extend(calls(continuing));
            }
            naga::Statement::Switch { cases, .. } => {
                for case in cases {
                    out.extend(calls(&case.body));
                }
            }
            _ => {}
        }
    }
    out
}

/// Every global variable the entry point can reach, itself and through calls.
fn reachable_globals(
    module: &Module,
    entry: &naga::EntryPoint,
) -> BTreeSet<Handle<naga::GlobalVariable>> {
    let mut out = BTreeSet::new();
    let mut seen: BTreeSet<Handle<naga::Function>> = BTreeSet::new();
    let mut queue: Vec<&naga::Function> = vec![&entry.function];
    while let Some(function) = queue.pop() {
        for (_, expr) in function.expressions.iter() {
            if let Expression::GlobalVariable(handle) = expr {
                out.insert(*handle);
            }
        }
        for (callee, _) in calls(&function.body) {
            if seen.insert(callee) {
                queue.push(&module.functions[callee]);
            }
        }
    }
    out
}

/// The byte offsets of a struct's members, and the bytes the whole thing needs.
///
/// `naga` has already done the layout: WGSL's alignment rules are in the
/// specification, the front end applies them, and `StructMember::offset` is
/// the answer. That is the entire reason `spirv.rs`'s type-graph recursion has
/// no counterpart here -- SPIR-V carries `Offset` decorations that a walk has
/// to collect and a size that has to be derived from strides, and WGSL carries
/// a layout the parser computed.
///
/// A uniform block that is not a struct -- a bare `vec4<f32>`, say -- has one
/// member at offset zero, which is what a caller comparing against
/// `kernels_wgpu::uniform_layout` should see. No row in the table has one.
fn uniform_layout(module: &Module, ty: Handle<naga::Type>) -> (Vec<u32>, u32) {
    match &module.types[ty].inner {
        naga::TypeInner::Struct { members, span } => {
            (members.iter().map(|m| m.offset).collect(), *span)
        }
        other => (vec![0], other.size(module.to_ctx())),
    }
}

/// The bytes a binding's block requires, when that is a fixed number.
///
/// `None` for anything ending in a runtime-sized array, which is every tensor
/// binding in this tree: `array<u32>` with no length is exactly the shape
/// whose size is the CALL's and not the module's.
fn fixed_size(module: &Module, ty: Handle<naga::Type>) -> Option<u32> {
    if has_dynamic_tail(module, ty) {
        return None;
    }
    Some(module.types[ty].inner.size(module.to_ctx()))
}

/// Does this type end in an array whose length the binding decides?
fn has_dynamic_tail(module: &Module, ty: Handle<naga::Type>) -> bool {
    match &module.types[ty].inner {
        naga::TypeInner::Array {
            size: naga::ArraySize::Dynamic,
            ..
        } => true,
        // A struct's last member may be the runtime array; an earlier one may
        // not, so checking only the last is not a shortcut -- it is the rule.
        naga::TypeInner::Struct { members, .. } => members
            .last()
            .is_some_and(|m| has_dynamic_tail(module, m.ty)),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use kernels_wgpu::Capability;

    /// A module in the shape this tree writes: two storage bindings, a uniform
    /// block, and a grid position handed to a helper.
    const SAMPLE: &str = r"
struct Params { head_dim: i32, stride: vec2<u32> }

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
@group(1) @binding(0) var<uniform> params: Params;

fn row_base(wg: vec3<u32>) -> u32 { return wg.x * 4u; }

@compute @workgroup_size(64, 2)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let at = row_base(wg);
    if (at < arrayLength(&out_)) { out_[at] = x[at] + u32(params.head_dim); }
}
";

    #[test]
    fn a_module_states_its_workgroup_on_every_axis() {
        let d = declared(SAMPLE).expect("the sample parses");
        // WGSL wrote two numbers and means three, which is the whole reason
        // `local` is an array rather than an option per axis.
        assert_eq!(d.local, [64, 2, 1]);
    }

    #[test]
    fn the_bindings_are_the_group_zero_ones_and_the_uniform_is_not_one() {
        let d = declared(SAMPLE).expect("parses");
        assert_eq!(d.bindings, 2, "the uniform is group 1 and takes no slot");
        assert_eq!(d.used, [true, true]);
        assert_eq!(d.holes(), 0);
    }

    /// A tensor binding has no knowable size and a params struct does.
    ///
    /// The distinction `block_bytes` exists for: `None` is the ordinary
    /// answer, and the ones that answer are the blocks where a short binding
    /// is a defect.
    #[test]
    fn a_runtime_array_has_no_size_and_a_struct_does() {
        let d = declared(SAMPLE).expect("parses");
        assert_eq!(d.block_bytes, vec![None, None]);

        let with_params = r"
struct RmsParams { eps: f32, axis_size: u32 }
@group(0) @binding(0) var<storage, read> params: RmsParams;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out_[gid.x] = params.axis_size;
}
";
        let d = declared(with_params).expect("parses");
        assert_eq!(d.block_bytes, vec![Some(8), None]);
    }

    /// The uniform block's offsets are the SHADER's, padding included.
    ///
    /// The check this field exists for: `head_dim` is four bytes and the
    /// `vec2<u32>` after it aligns to eight, so the second member starts at 8
    /// and not at 4. A shell that packed by concatenation would write the
    /// stride four bytes low, and the shader would read half of one number and
    /// half of the next -- a number, in the right place, that is not a stride.
    #[test]
    fn the_uniform_blocks_offsets_are_read_off_the_shader() {
        let d = declared(SAMPLE).expect("parses");
        assert_eq!(d.uniform_offsets, vec![0, 8]);
        assert_eq!(
            d.uniform_bytes, 16,
            "12 bytes of members, rounded to the 16 the uniform address space \
             requires"
        );
    }

    /// A module with no uniform block says so with nothing, not with a zero.
    #[test]
    fn a_module_with_no_uniform_block_declares_none() {
        let plain = r"
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) { out_[gid.x] = 1u; }
";
        let d = declared(plain).expect("parses");
        assert!(d.uniform_offsets.is_empty());
        assert_eq!(d.uniform_bytes, 0);
    }

    /// A grid component read through a helper is still found.
    ///
    /// The case that separates this walk from a one-function scan. `main`
    /// never indexes `wg`; `row_base` does, against its own parameter, and a
    /// reflection that stopped at the entry point would report no axis at all
    /// for every shader in this tree that factors its addressing out.
    #[test]
    fn a_grid_axis_indexed_inside_a_callee_is_followed() {
        let d = declared(SAMPLE).expect("parses");
        assert_eq!(
            d.grid_axes,
            [true, false, false],
            "`row_base` reads `wg.x` and nothing reads y or z"
        );
    }

    /// Two axes, read directly, are two axes.
    #[test]
    fn every_axis_a_body_indexes_is_reported() {
        let two = r"
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    out_[gid.y * 16u + gid.x] = 1u;
}
";
        assert_eq!(declared(two).unwrap().grid_axes, [true, true, false]);
    }

    /// A grid position the walk cannot follow reports every axis.
    ///
    /// The conservative direction, stated as a test so that it is a decision
    /// rather than an accident. Here the whole vector is summed, so no
    /// component is named and the honest answer is "unknown" -- which must
    /// read as "any", or a check comparing a rule's axis against this would
    /// reject a correct grid and be deleted for it.
    #[test]
    fn a_grid_position_used_whole_is_reported_on_every_axis() {
        let whole = r"
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>) {
    let v = dot(wg, vec3<u32>(1u, 1u, 1u));
    out_[v] = 1u;
}
";
        assert_eq!(declared(whole).unwrap().grid_axes, [true, true, true]);
    }

    /// `num_workgroups` is found, and its absence is found too.
    #[test]
    fn the_modules_that_read_their_own_workgroup_count_are_named() {
        assert!(!declared(SAMPLE).unwrap().reads_workgroup_count);
        let counts = r"
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@compute @workgroup_size(1)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) n: vec3<u32>,
) {
    out_[wg.x] = n.x;
}
";
        assert!(declared(counts).unwrap().reads_workgroup_count);
    }

    /// A declared binding the entry point never reads is a hole.
    ///
    /// The question that survives from SPIR-V unchanged: `naga` keeps the
    /// global, so the module still says "there is a binding at 1", and only a
    /// walk of the expressions can say whether anything looks at it.
    #[test]
    fn a_binding_nothing_reads_is_declared_and_unused() {
        let unread = r"
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@group(0) @binding(1) var<storage, read> spare: array<u32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>) { out_[wg.x] = 1u; }
";
        let d = declared(unread).expect("parses");
        assert_eq!(d.bindings, 2, "the layout still needs an entry at 1");
        assert_eq!(d.used, [true, false]);
        assert_eq!(d.holes(), 1);
    }

    /// A binding read only inside a helper is read.
    ///
    /// The other half of the same claim, and the one that would silently turn
    /// a real operand into a hole: `common/bf16.inc.wgsl` takes a
    /// `ptr<storage, array<u32>, read>`, so the call site names the global and
    /// the read happens a frame down.
    #[test]
    fn a_binding_touched_only_through_a_helper_is_not_a_hole() {
        let helped = r"
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<u32>;
fn load(words: ptr<storage, array<u32>, read>, i: u32) -> u32 { return (*words)[i]; }
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>) { out_[wg.x] = load(&x, wg.x); }
";
        assert_eq!(declared(helped).unwrap().used, [true, true]);
    }

    /// A source `naga` refuses is a refusal that names the line.
    #[test]
    fn a_source_naga_will_not_take_is_refused_with_its_message() {
        let broken = "@compute @workgroup_size(64) fn main( { }";
        let Err(Unreadable::Unparseable(why)) = declared(broken) else {
            panic!("naga accepted a module that is not WGSL");
        };
        assert!(
            why.contains("error"),
            "the refusal carries naga's rendered diagnostic, not a bare kind: {why}"
        );
    }

    /// A module with no compute entry point is refused, not read.
    #[test]
    fn a_module_with_no_compute_entry_point_is_refused() {
        let none = "@group(0) @binding(0) var<storage, read> x: array<u32>;";
        assert_eq!(declared(none), Err(Unreadable::NoComputeEntry));
    }

    /// Two entry points need two grids, so one answer would be a guess.
    #[test]
    fn two_compute_entry_points_are_refused_rather_than_the_first_taken() {
        let two = r"
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@compute @workgroup_size(64) fn a(@builtin(workgroup_id) w: vec3<u32>) { out_[w.x] = 1u; }
@compute @workgroup_size(256) fn b(@builtin(workgroup_id) w: vec3<u32>) { out_[w.x] = 2u; }
";
        assert_eq!(
            declared(two),
            Err(Unreadable::TwoComputeEntries {
                names: vec!["a".into(), "b".into()]
            })
        );
    }

    /// An overridable workgroup is refused rather than read at its default.
    ///
    /// The number in the IR is not the number that will run, and
    /// `crate::geometry` divides a whole fire's extent by it -- so taking the
    /// default is an undershoot waiting for a shell that supplies a bigger
    /// one, and an undershoot writes nothing and reports success.
    #[test]
    fn an_overridden_workgroup_size_is_refused() {
        let over = r"
override wg_size: u32 = 64u;
@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@compute @workgroup_size(wg_size)
fn main(@builtin(workgroup_id) w: vec3<u32>) { out_[w.x] = 1u; }
";
        assert_eq!(declared(over), Err(Unreadable::OverriddenWorkgroup));
    }

    /// Every entrypoint of the real table whose source parses, read end to end.
    ///
    /// The claim that the path from a NAME to a `Declared` closes: no file, no
    /// build directory, no adapter. Written as a sweep rather than a single
    /// case because the failure it catches is a shader SHAPE nobody thought
    /// about -- a second `@compute` left behind by an `//#if` arm, a workgroup
    /// sized by an `override`.
    ///
    /// # It fails on a source `naga` refuses, and that is a recent change
    ///
    /// While the WGSL tree was being ported this asserted a FLOOR instead --
    /// "at least a hundred read" -- because a half-written body with a missing
    /// `//#include` is `kernels-wgpu`'s failure and turning this suite red for
    /// it would have taught the next reader to ignore both.
    ///
    /// The tree is complete: all 480 entrypoints expand and every one parses.
    /// So the floor is gone and the assertion is the whole set, which is a
    /// strictly stronger claim and a cheaper one to read. If a shader stops
    /// parsing, two suites go red and both are right -- `kernels-wgpu` because
    /// its module is broken, this one because a driver that cannot read a
    /// module cannot dispatch it.
    #[test]
    fn every_entrypoint_of_the_table_is_read_from_its_own_source() {
        let mut read = 0;
        let mut refused = Vec::new();
        for name in kernels_wgpu::entrypoints() {
            match entrypoint(&name, Capability::Baseline) {
                Ok(d) => {
                    assert!(
                        d.local.iter().all(|n| *n >= 1),
                        "`{name}` declares a workgroup of {:?}",
                        d.local
                    );
                    assert_eq!(
                        d.used.len(),
                        d.bindings as usize,
                        "`{name}`'s used-set is indexed by binding number"
                    );
                    assert_eq!(d.block_bytes.len(), d.bindings as usize);
                    assert_eq!(
                        d.uniform_offsets.is_empty(),
                        d.uniform_bytes == 0,
                        "`{name}` declares a uniform block of {} bytes with {} \
                         members, and those two have to agree",
                        d.uniform_bytes,
                        d.uniform_offsets.len()
                    );
                    read += 1;
                }
                Err(why) => refused.push(format!("{name}: {why}")),
            }
        }
        assert!(
            refused.is_empty(),
            "{} of the table's entrypoints do not read:\n  {}",
            refused.len(),
            refused.join("\n  ")
        );
        assert_eq!(
            read,
            kernels_wgpu::entrypoints().len(),
            "every entrypoint the table names has a source and reads"
        );
        assert_eq!(read, 481, "the table is 100 rows over 481 entrypoints");
    }

    /// And the module's own uniform block agrees with the row's.
    ///
    /// The disagreement this is for is silent: the row says where the driver
    /// will write each scalar and the shader says where it will read them, the
    /// two are computed from different places, and a mismatch is a plausible
    /// number rather than an error. Checked over whatever the tree has today,
    /// for the stated rows only -- an unstated row has no layout to compare.
    #[test]
    fn a_stated_rows_uniform_layout_is_the_one_its_shader_declares() {
        let mut compared = 0;
        for sig in kernels_wgpu::KERNELS {
            if sig.operands.is_empty() {
                continue;
            }
            let want: Vec<u32> = kernels_wgpu::uniform_layout(sig)
                .iter()
                .map(|f| f.offset)
                .collect();
            for name in sig.entrypoints() {
                let d = entrypoint(&name, Capability::Baseline)
                    .unwrap_or_else(|e| panic!("`{name}` has no readable module: {e}"));
                assert_eq!(
                    d.uniform_offsets, want,
                    "`{name}`: the row writes its scalars at {want:?} and the \
                     shader reads them at {:?}",
                    d.uniform_offsets
                );
                // And the block the shell will send is big enough, which is
                // the WebGPU direction of the check -- over is fine, under is
                // a binding `wgpu` refuses.
                assert!(
                    kernels_wgpu::uniform_size(sig) >= d.uniform_bytes,
                    "`{name}`: the row's block is {} bytes and the shader's \
                     struct needs {}",
                    kernels_wgpu::uniform_size(sig),
                    d.uniform_bytes
                );
                compared += 1;
            }
        }
        assert_eq!(
            compared, 189,
            "44 rows state operands, over 189 entrypoints, and every one of \
             them is compared -- a floor here would let the sweep shrink"
        );
    }
}
