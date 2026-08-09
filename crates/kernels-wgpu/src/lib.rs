//! WEBGPU's kernel signature table — one row per KERNEL in `kernels/`, and one
//! row is many entrypoints.
//!
//! ## Why this crate is the one that needs no toolchain
//!
//! `kernels-cuda` costs nvcc. `kernels-vulkan` costs `glslc`, because Vulkan
//! has no runtime shader compiler and a pipeline is built from a SPIR-V module
//! that something else had to produce. `kernels-metal` costs a Mac to RUN,
//! though not to read.
//!
//! WGSL costs nothing. `wgpu` carries `naga`, a WGSL front end written in Rust,
//! so the process that dispatches a kernel is the process that compiled it —
//! the Metal model, with the compiler in the dependency graph instead of in the
//! operating system. That is why this crate has no `native` feature: there is
//! no build product to gate, so the table, the shaders and every structural
//! test are reachable on any machine that can build Rust.
//!
//! ## The coverage is `kernels-metal`'s, deliberately
//!
//! Row for row, axis for axis, point for point: **100 kernels over 481
//! entrypoints**, the same names in the same ten families, and
//! `tests/entrypoints.rs` pins all three of those numbers against
//! `kernels-metal`'s own source.
//!
//! That is not imitation for its own sake. `crates/kernels` is the
//! backend-neutral vocabulary — [`kernels::KernelSig`], [`kernels::Axis`],
//! [`kernels::Ty`], [`kernels::Operand`], [`kernels::Source`],
//! [`kernels::LaunchRule`] — and `model-compiler` resolves a traced program
//! against it through `kernels::sig_in` without ever learning which backend it
//! is compiling for. A backend table is therefore not a design surface. It is
//! an *answer* to a question the compiler already asked, and the answer must be
//! the same shape on every backend or the compiler's plan stops being portable.
//!
//! A divergence between two tables is then a STATEMENT that one backend covers
//! something the other does not, rather than an accident nobody wrote down. If
//! you add a wgpu-only kernel you are no longer porting — you are forking the
//! vocabulary, and that belongs in `crates/kernels` first.
//!
//! ## The WebGPU launch ABI, which is this backend's one real divergence
//!
//! A Metal row is positional over BUFFERS: `constant int& n [[buffer(4)]]` is a
//! buffer like any other, so a scalar and a tensor occupy the same kind of slot
//! and the row's index is the binding. WebGPU does not work that way, and
//! neither does Vulkan — but the two disagree about the answer, so the rule
//! here is a third one and it is worth stating precisely.
//!
//! Vulkan sends the scalar run to a `layout(push_constant)` block. **WebGPU has
//! no push constants.** `wgpu` exposes them as `Features::PUSH_CONSTANTS`,
//! which is a native-only extension no WebGPU implementation is obliged to
//! offer and which the browser backend cannot offer at all. A table that
//! depended on it would be a table that runs on `wgpu` and not on WebGPU, which
//! gives up the only thing this backend has that its siblings do not.
//!
//! So the scalar run becomes **one uniform buffer**, and the two runs are put
//! in two different bind GROUPS rather than sharing one numbering:
//!
//! * every operand whose [`kernels::Ty`] is a BUFFER kind takes the next
//!   `@group(0) @binding(N)` storage slot, in row order, from 0;
//! * every operand whose kind is a SCALAR (`I32`, `U32`, `F32`, `Usize`,
//!   `Bool`) becomes the next field of the single
//!   `@group(1) @binding(0) var<uniform> params` struct, in row order;
//! * [`kernels::Ty::InPacked`] takes neither, for the reason
//!   [`Binding::Packed`] gives.
//!
//! **Two groups, not one, and the reason is not tidiness.** Vulkan's binding
//! numbers are shared between its buffers and nothing else, because its scalars
//! left the numbering entirely; a WebGPU uniform is a binding like any other, so
//! putting it in group 0 would give it an index that MOVES with the row's
//! buffer count. Every shader in a family would then declare its params block at
//! a different number than its neighbour, and a family's shaders are one file.
//! Group 1 binding 0 is the same in all 100 rows, so a shader states it once.
//!
//! [`bindings`] is the rule as code, so that a shell binding a launch and a
//! test checking a shader compute it from one place rather than agreeing by
//! habit. The rule is stated here rather than per-row on purpose: it is a
//! property of the API, not of any kernel, and a table that spelled it 99 times
//! would let row 100 spell it differently.
//!
//! ### The trap this rule exists to make unrepresentable
//!
//! `kernels-vulkan`'s own notes record sixty entrypoints that read a descriptor
//! the shell never wrote, every one of them a shader author transcribing
//! *Metal's* buffer indices into a Vulkan `binding`. The two numbers differ by
//! however many scalars precede the operand. They must never be copied across,
//! and the same is true here — with the extra wrinkle that a WGSL binding that
//! is declared and never written is not a validation error either. `wgpu` will
//! refuse a bind group that does not MATCH its layout, which is more than
//! Vulkan does, but the layout is derived from the same wrong reading.
//!
//! [`uniform_layout`] is the other half. `Binding::Uniform(n)` is a field
//! INDEX, and a shell needs an OFFSET, and turning one into the other is not
//! multiplication: WGSL's uniform address space aligns a member to its own
//! size, so an eight-byte value after a lone four-byte one starts at 8 and not
//! at 4, and the struct itself rounds up to 16.
//!
//! ## What a shell that RUNS these has to do
//!
//! Three things a shader cannot check for itself. They are written down here
//! rather than discovered, because two of the three already cost the Vulkan
//! port a debugging session.
//!
//! * **Grids are workgroup-granular.** `dispatch_workgroups` counts
//!   WORKGROUPS where Metal's `dispatchThreads` counts threads, so a shell that
//!   ports a Metal grid arithmetically launches a 256th of it. Every pointwise
//!   body here guards its own tail against the length of the buffer it writes,
//!   so an overshoot is harmless. An UNDERSHOOT is not, and it is the direction
//!   that fails silently: a lane that never launches writes nothing, the gap
//!   reads back as whatever the buffer held, and the dispatch completes.
//!   [`kernels::KernelSig::grid_param`], `head_param`, `heads_param` and
//!   `rows_param` say which of the STATEMENT's params give the shape.
//! * **Ask the adapter for its limits.** WebGPU's guaranteed floor is **8**
//!   storage buffers per shader stage, and [`over_downlevel_storage_limit`]
//!   names the rows that need more — `sdpa_paged_decode` binds eleven. A shell
//!   that requests `Limits::downlevel_defaults()` will fail to create those
//!   pipelines on hardware that would have run them.
//! * **A workgroup size is fixed when the module compiles.** WGSL's
//!   `@workgroup_size` is a compile-time attribute (an `override` may size it,
//!   but not a uniform), where a Metal threadgroup is sized at dispatch. A body
//!   ported from Metal that assumed "one lane per channel" is correct only up
//!   to its own declared width — the defect `kernels-vulkan` records in
//!   `gated_rms`, which this tree inherits the fix for and not the bug.

pub use kernels::{Axis, Cap, KernelSig, Prepare};

mod capability;
pub use crate::capability::Capability;

pub mod preproc;
pub use crate::preproc::{Directive, Malformed, Variant, expand, instantiations};

pub mod source;
pub use crate::source::{Missing, SOURCES, entrypoint_source, source};

pub mod axes;

pub mod attn;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod ptir;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;

/// The family tables, concatenated.
///
/// A `const fn` fold rather than a `Vec`, so the whole table stays a `&'static`
/// the compiler can read at load with no allocation — the same shape both
/// siblings use for the same reason.
pub static KERNELS: &[KernelSig] = &CONCAT;

const FAMILIES: &[&[KernelSig]] = &[
    attn::KERNELS,
    layout::KERNELS,
    mlp::KERNELS,
    moe::KERNELS,
    norm::KERNELS,
    ptir::KERNELS,
    quant::KERNELS,
    rope::KERNELS,
    sample::KERNELS,
    ssm::KERNELS,
];

const fn total() -> usize {
    let mut n = 0;
    let mut i = 0;
    while i < FAMILIES.len() {
        n += FAMILIES[i].len();
        i += 1;
    }
    n
}

const N: usize = total();

const EMPTY: KernelSig = KernelSig {
    name: "",
    symbol: "",
    file: None,
    launch: kernels::LaunchRule::Unstated,
    whole: false,
    needs: Prepare::None,
    lacks: &[],
    sink: None,
    in_place: &[],
    depth_prefix_plan: false,
    args: &[],
    operands: &[],
    axes: &[],
    grid_param: None,
    head_param: None,
    heads_param: None,
    rows_param: None,
};

const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name,
        symbol: k.symbol,
        file: k.file,
        launch: k.launch,
        whole: k.whole,
        needs: k.needs,
        lacks: k.lacks,
        sink: k.sink,
        in_place: k.in_place,
        depth_prefix_plan: k.depth_prefix_plan,
        args: k.args,
        operands: k.operands,
        axes: k.axes,
        grid_param: k.grid_param,
        head_param: k.head_param,
        heads_param: k.heads_param,
        rows_param: k.rows_param,
    }
}

const CONCAT: [KernelSig; N] = {
    let mut out = [EMPTY; N];
    let mut at = 0;
    let mut f = 0;
    while f < FAMILIES.len() {
        let family = FAMILIES[f];
        let mut i = 0;
        while i < family.len() {
            out[at] = copy_sig(&family[i]);
            at += 1;
            i += 1;
        }
        f += 1;
    }
    out
};

/// Every entrypoint the table names, sorted.
///
/// The set `entrypoints.generated.txt` records, and — one for one — the set of
/// variants the `// pie:instantiate` directives in `kernels/` declare.
#[must_use]
pub fn entrypoints() -> Vec<String> {
    let mut out: Vec<String> = KERNELS.iter().flat_map(KernelSig::entrypoints).collect();
    out.sort();
    out
}

/// The row that covers `symbol`, or `None`.
///
/// A thin forward to [`kernels::sig_in`] over this crate's table, so a caller
/// that has the symbol and not the table does not have to name both.
#[must_use]
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    kernels::sig_in(KERNELS, symbol)
}

/// Where one operand rides. See [`bindings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    /// `@group(0) @binding(N) var<storage, ...>`.
    Storage(u32),
    /// The `N`-th field of the `@group(1) @binding(0) var<uniform>` block.
    Uniform(u32),
    /// Nowhere of its own: a FIELD of the packed buffer ahead of it.
    ///
    /// [`kernels::Ty::InPacked`] is how a row says "the driver has to supply
    /// this value, but it does not get a slot" — the value belongs to a struct
    /// some earlier `Buf` operand already binds, so the driver writes it while
    /// filling that buffer and the shader reads it as a struct member.
    ///
    /// Metal could fold this into the scalar run, because there a packed slot
    /// IS the buffer and a trailing scalar lands in the same argument. Both
    /// this backend and Vulkan split the two runs, so folding it into the
    /// uniform block would push a word no shader reads and leave the struct
    /// field unwritten — the defect this variant exists to make
    /// unrepresentable. `layout/row_gather` is the row that has it: it declares
    /// no uniform block at all while the row states `count`, because `count` is
    /// the second field of the `RowGatherParams` struct that a storage buffer
    /// already binds.
    Packed,
}

/// Which storage binding each operand of `sig` takes, and which uniform field,
/// under the two-pass rule this module's own docs state.
///
/// Both runs are indexed from zero and both follow the row's order, so the
/// answer for operand `k` is a function of the row alone — which is the point:
/// a shell binding a launch and a test checking a shader compute the same thing
/// from the same place.
///
/// A row with no operands is UNSTATED (see [`kernels::KernelSig::operands`]),
/// and this answers with an empty vector rather than inventing a nullary
/// layout. An unstated row is still LAUNCHABLE — a shell falls back to the
/// lowered plan's own argument order — but it is not launchable from here.
#[must_use]
pub fn bindings(sig: &KernelSig) -> Vec<Binding> {
    let mut storages = 0;
    let mut uniforms = 0;
    sig.operands
        .iter()
        .map(|op| {
            if matches!(op.ty, kernels::Ty::InPacked) {
                // Consumes neither run: see `Binding::Packed`.
                Binding::Packed
            } else if is_buffer(op.ty) {
                let at = storages;
                storages += 1;
                Binding::Storage(at)
            } else {
                let at = uniforms;
                uniforms += 1;
                Binding::Uniform(at)
            }
        })
        .collect()
}

/// How many entries a row's `@group(0)` bind group layout declares.
#[must_use]
pub fn storage_count(sig: &KernelSig) -> u32 {
    sig.operands.iter().filter(|op| is_buffer(op.ty)).count() as u32
}

/// WebGPU's guaranteed floor for storage buffers in one shader stage.
///
/// `wgpu::Limits::downlevel_defaults().max_storage_buffers_per_shader_stage`,
/// restated as a number so this crate can name it without depending on `wgpu`.
/// A row above it is not wrong; it is a row whose pipeline needs a device that
/// reports more than the floor, and [`over_downlevel_storage_limit`] is how a
/// shell finds out which rows those are before it picks its limits.
pub const DOWNLEVEL_STORAGE_BUFFERS: u32 = 8;

/// The rows whose `@group(0)` is wider than [`DOWNLEVEL_STORAGE_BUFFERS`].
///
/// Returned rather than asserted, because "too many" is a property of the
/// DEVICE and not of the row: `sdpa_paged_decode`'s eleven buffers are eleven
/// real tensors, and every desktop adapter offers far more than eight. A shell
/// that requests `Limits::downlevel_defaults()` out of caution would fail to
/// create exactly these pipelines, and the failure would arrive at model load
/// with a wgpu message about a limit rather than about attention.
#[must_use]
pub fn over_downlevel_storage_limit() -> Vec<&'static KernelSig> {
    KERNELS
        .iter()
        .filter(|sig| storage_count(sig) > DOWNLEVEL_STORAGE_BUFFERS)
        .collect()
}

/// One scalar's place in the uniform block, in BYTES.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UniformField {
    /// The operand's name, as the row spells it.
    pub name: &'static str,
    /// Byte offset from the start of the block.
    pub offset: u32,
    /// Width in bytes: four, or eight for a 64-bit value.
    pub size: u32,
    /// Whether the field is a 64-bit value with no WGSL scalar type.
    ///
    /// WGSL has `u32`, `i32` and `f32` and no 64-bit integer at all. A
    /// [`kernels::Ty::Usize`] or [`kernels::Ty::I64`] operand is therefore
    /// declared as `vec2<u32>` — low word first — which is what gives it an
    /// eight-byte alignment as well as an eight-byte width, and is the same
    /// answer `kernels-vulkan` arrived at from the other direction when a
    /// validation layer objected to its `uint64_t` requiring `shaderInt64`.
    pub split: bool,
}

/// The uniform block's byte layout, which [`Binding::Uniform`] does NOT give.
///
/// `Binding::Uniform(n)` is a field INDEX and a shell needs an offset, and
/// turning one into the other is not multiplication. WGSL's uniform address
/// space aligns a member to its own alignment, and the `vec2<u32>` a 64-bit
/// operand becomes has an alignment of 8 — so `attn/kv_write`'s
///
/// ```text
/// head_dim: i32, k_head_stride: vec2<u32>, k_seq_stride: vec2<u32>
/// ```
///
/// is 24 bytes with four bytes of padding after the first field, where the
/// naive sum of widths says 20. A shell that packs by concatenation writes both
/// strides four bytes low and the shader reads two halves of two different
/// numbers. Nothing reports it: a uniform buffer is bytes, and wgpu does not
/// know what they were supposed to mean.
///
/// Derived here rather than hand-computed in each caller, for the same reason
/// [`bindings`] exists rather than each caller counting buffers.
#[must_use]
pub fn uniform_layout(sig: &KernelSig) -> Vec<UniformField> {
    let mut at = 0u32;
    sig.operands
        .iter()
        .filter(|op| !is_buffer(op.ty) && !matches!(op.ty, kernels::Ty::InPacked))
        .map(|op| {
            let size = uniform_width(op.ty);
            at = at.next_multiple_of(size);
            let field = UniformField {
                name: op.name,
                offset: at,
                size,
                split: size == 8,
            };
            at += size;
            field
        })
        .collect()
}

/// The uniform block's total size in bytes, padding included.
///
/// Rounded up to **16**, not to the widest member. That is not a Vulkan push
/// block: WGSL's uniform address space gives every host-shareable struct an
/// alignment of at least 16, and `wgpu` rejects a binding whose size is not a
/// multiple of it. A row with no scalars answers zero, which is a row that
/// declares no `@group(1)` at all.
#[must_use]
pub fn uniform_size(sig: &KernelSig) -> u32 {
    let fields = uniform_layout(sig);
    let Some(last) = fields.last() else { return 0 };
    (last.offset + last.size).next_multiple_of(16)
}

/// WebGPU's guaranteed floor for one uniform binding's size, in bytes.
///
/// `wgpu::Limits::downlevel_defaults().max_uniform_buffer_binding_size` is
/// 16 KiB, which no row comes close to. It is named so `tests/entrypoints.rs`
/// can pin the ceiling the way `kernels-vulkan` pins its 128-byte push limit —
/// the check is cheap and the failure it prevents is a pipeline that refuses to
/// build on the one device that mattered.
pub const DOWNLEVEL_UNIFORM_BYTES: u32 = 16 * 1024;

/// Every scalar kind a row can name is four or eight bytes wide.
fn uniform_width(ty: kernels::Ty) -> u32 {
    match ty {
        kernels::Ty::Usize | kernels::Ty::I64 => 8,
        _ => 4,
    }
}

/// Whether a kind crosses as a device allocation rather than as a value.
///
/// Read off the KIND and not off a list of operand names, so a row that grows a
/// buffer cannot land in the uniform block by omission. The struct-shaped and
/// handle kinds of the CUDA vocabulary are not reachable from a row here —
/// there is no stream and no cuBLAS handle in WebGPU — so they answer `false`,
/// and a row that used one would put a plan cache in a uniform: a failure at
/// the row, where it can be read, rather than a silent binding.
const fn is_buffer(ty: kernels::Ty) -> bool {
    use kernels::Ty;
    matches!(
        ty,
        Ty::BufMut
            | Ty::Buf
            | Ty::I32s
            | Ty::I64s
            | Ty::U32s
            | Ty::U8s
            | Ty::F32sMut
            | Ty::F32s
            | Ty::I32sMut
            | Ty::U32sMut
            | Ty::U8sMut
            | Ty::U16s
            | Ty::U16sMut
            | Ty::I8s
            | Ty::BufArray
            | Ty::BufArrayMut
            | Ty::BufArrayOut
            | Ty::BufArrayOutMut
            | Ty::U8Array
            | Ty::I32Array
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two runs number independently, and neither one skips.
    ///
    /// `sdpa_paged_decode` is the row this is written for: its operands
    /// ALTERNATE, so the runs interleave, and a backend that numbered scalars
    /// alongside buffers would put every binding after the first scalar one
    /// too high. Nothing static catches that — only a number does.
    #[test]
    fn the_two_runs_are_numbered_apart() {
        let sig = sig("sdpa_paged_decode_bfloat16_d_64")
            .expect("the table covers the decode step's attention");
        let bindings = bindings(sig);

        let mut storages = 0;
        let mut uniforms = 0;
        for (at, binding) in bindings.iter().enumerate() {
            match binding {
                Binding::Storage(n) => {
                    assert_eq!(*n, storages, "operand {at} is out of the storage run");
                    storages += 1;
                }
                Binding::Uniform(n) => {
                    assert_eq!(*n, uniforms, "operand {at} is out of the uniform run");
                    uniforms += 1;
                }
                Binding::Packed => {}
            }
        }
        assert_eq!(storages, storage_count(sig));
        assert_eq!(uniforms as usize, uniform_layout(sig).len());
        assert!(
            storages > 1 && uniforms > 1,
            "this row was chosen because it has both runs; it now has \
             {storages} buffers and {uniforms} scalars, so it no longer \
             tests the interleaving and another row should be picked",
        );
    }

    /// A packed operand consumes neither run.
    ///
    /// `row_gather`'s `count` is the case: the shader declares no uniform block
    /// and the row states the operand, because the value is a FIELD of the
    /// params struct an earlier buffer binds.
    #[test]
    fn a_packed_operand_takes_no_slot_of_its_own() {
        let sig = sig("row_gather_bfloat16").expect("the table covers the row gather");
        let bindings = bindings(sig);

        let packed = bindings
            .iter()
            .filter(|b| matches!(b, Binding::Packed))
            .count();
        assert_eq!(packed, 1, "row_gather states exactly one packed operand");

        assert!(
            uniform_layout(sig).is_empty(),
            "row_gather's only scalar is packed, so it declares no uniform block",
        );
        assert_eq!(uniform_size(sig), 0);
    }

    /// An unstated row answers with nothing rather than with a nullary layout.
    ///
    /// `sdpa_paged_tiled` is one of the 56 rows that carry axes and a name and
    /// no operands. Such a row is not unlaunchable — a shell falls back to the
    /// lowered plan's own argument order, which is what `driver-metal` does —
    /// but it is not launchable from HERE, and the difference is the whole
    /// content of this check.
    #[test]
    fn an_unstated_row_has_no_bindings() {
        let sig = sig("sdpa_paged_tiled_bfloat16_d_64")
            .expect("the table covers the tiled paged attention");
        assert!(
            sig.operands.is_empty(),
            "this row is chosen for being unstated; state it and pick another",
        );
        assert!(bindings(sig).is_empty());
        assert_eq!(storage_count(sig), 0);
        assert_eq!(uniform_size(sig), 0);
    }

    /// The table is `kernels-metal`'s coverage, and these are the numbers.
    ///
    /// Pinned here as well as in `tests/entrypoints.rs` because a `cargo test
    /// -p kernels-wgpu --lib` should be able to say whether the port is whole.
    #[test]
    fn the_table_is_one_hundred_rows_over_four_hundred_and_eighty_one_entrypoints() {
        assert_eq!(KERNELS.len(), 100, "one row per kernel in `kernels/`");
        assert_eq!(entrypoints().len(), 481, "the product of every row's axes");
        assert_eq!(
            KERNELS.iter().filter(|k| k.operands.is_empty()).count(),
            56,
            "the rows that state no operands. `kernels-vulkan` carries the same \
             56, and the decision about whether they should exist belongs with \
             `kernels-metal`, which carries them too",
        );
    }

    /// A uniform member is aligned to its own alignment, and the block to 16.
    ///
    /// `kv_append` is the shape that proves it: a four-byte `head_dim` followed
    /// by two eight-byte strides. The naive sum is 20 bytes; the real block is
    /// 24 bytes of fields rounded to 32 by the uniform address space, with four
    /// bytes of padding after the first field.
    #[test]
    fn a_wide_field_after_a_narrow_one_is_padded() {
        let sig = sig("kv_append_bfloat16").expect("the table covers the KV append");
        let fields = uniform_layout(sig);

        let named: Vec<_> = fields.iter().map(|f| (f.name, f.offset, f.size)).collect();
        assert_eq!(
            named,
            vec![
                ("head_dim", 0, 4),
                ("k_head_stride", 8, 8),
                ("k_seq_stride", 16, 8),
            ],
            "the eight-byte strides align to eight, not to four",
        );
        assert!(
            fields[1].split && fields[2].split,
            "a 64-bit operand crosses as vec2<u32>: WGSL has no u64",
        );
        assert_eq!(uniform_size(sig), 32, "24 bytes of fields, rounded to 16");
    }

    /// No row's uniform block exceeds what WebGPU guarantees.
    ///
    /// Cheap, and the failure it prevents is a pipeline that refuses to build
    /// on the one device that mattered. `kernels-vulkan` pins the same ceiling
    /// at 128 bytes, which is a real constraint there; 16 KiB is not one here,
    /// and the test says so by also asserting the widest row is far under it.
    #[test]
    fn no_row_asks_for_a_uniform_block_webgpu_will_not_bind() {
        let widest = KERNELS
            .iter()
            .map(|sig| (uniform_size(sig), sig.symbol))
            .max()
            .expect("the table is not empty");

        assert!(
            widest.0 <= DOWNLEVEL_UNIFORM_BYTES,
            "`{}` asks for {} bytes of uniform, over WebGPU's {DOWNLEVEL_UNIFORM_BYTES}",
            widest.1,
            widest.0,
        );
        assert!(
            widest.0 < 256,
            "the widest block is now {} bytes (`{}`), which is close enough to \
             a real limit that the ceiling above stopped being decorative",
            widest.0,
            widest.1,
        );
    }

    /// The rows that need more than WebGPU's floor are NAMED, not tolerated.
    ///
    /// A shell reads this to decide what limits to request. The assertion is
    /// that the answer is a small, known set — if it grows, a person should
    /// look, because the alternative to raising a limit is splitting a row.
    #[test]
    fn the_rows_over_the_storage_floor_are_the_ones_we_know_about() {
        let over = over_downlevel_storage_limit();
        let mut names: Vec<_> = over.iter().map(|sig| sig.symbol).collect();
        names.sort_unstable();
        names.dedup();

        assert!(
            !names.is_empty(),
            "attention alone binds eleven buffers, so an empty answer means \
             `storage_count` stopped counting",
        );
        for sig in &over {
            assert!(
                storage_count(sig) <= 16,
                "`{}` binds {} storage buffers, past what a mainstream adapter \
                 reports; this row needs splitting rather than a bigger limit",
                sig.symbol,
                storage_count(sig),
            );
        }
    }
}
