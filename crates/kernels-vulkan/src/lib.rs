//! VULKAN's kernel signature table — one row per KERNEL in `kernels/`, and one
//! row is many entrypoints.
//!
//! ## Why this is shaped like Metal's and not like CUDA's
//!
//! `kernels-cuda` has one row per launcher symbol, because a CUDA launcher is
//! an authored C++ function and there is nothing else it could be. A GLSL
//! compute shader is the other extreme: it has exactly ONE entry point and it
//! is always called `main`, so a name like
//! `affine_qmm_t_bfloat16_gs_64_b_4_bm_32_bn_32` cannot be a symbol at all. It
//! is the name of a SPIR-V MODULE — one `.spv`, compiled from one `.comp` with
//! one `-D` set, exactly as llama.cpp's `vulkan-shaders-gen` produces its
//! `matmul_q4_k_f16_f32.spv` and its 900 siblings.
//!
//! That lands this table on Metal's shape rather than CUDA's, for nearly
//! Metal's reason: an entrypoint here is GENERATED from a template evaluated at
//! a point, so enumerating the product by hand would state the generator's job
//! a second time. A row carries its [`Axis`]es and the product is the
//! entrypoint set.
//!
//! **The coverage is `kernels-metal`'s, deliberately.** Row for row, axis for
//! axis, point for point: 99 kernels over 480 entrypoints, the same names in
//! the same ten families. That is not imitation for its own sake — it is what
//! makes the two backends comparable. `model-compiler` checks a lowered plan
//! against whichever table the deployment selected, so a text that runs on
//! Metal either runs here or names exactly which row it wanted; and a
//! divergence between the two tables is then a STATEMENT that one backend
//! covers something the other does not, rather than an accident nobody wrote
//! down.
//!
//! ## The Vulkan launch ABI, which is this backend's one real divergence
//!
//! A Metal row is positional over BUFFERS: `constant int& n [[buffer(4)]]` is a
//! buffer like any other, so a scalar and a tensor occupy the same kind of slot
//! and the row's index is the binding. Vulkan does not work that way. Binding a
//! four-byte scalar as a storage buffer costs a descriptor write, a device
//! allocation and a barrier for a number that fits in a push constant, and no
//! Vulkan backend written by anyone does it — llama.cpp puts every scalar in
//! one `layout(push_constant)` block and every tensor in a `binding`.
//!
//! So the row is read in TWO passes, and the rule is mechanical:
//!
//! * every operand whose [`kernels::Ty`] is a BUFFER kind (`Buf`, `BufMut`,
//!   `I32s`, `U32s`, `U8s`, `F32s`, and the rest of the pointer family) takes
//!   the next `layout(std430, binding = N)` slot, in row order, from 0;
//! * every operand whose kind is a SCALAR (`I32`, `U32`, `F32`, `Usize`,
//!   `Bool`, `InPacked`) takes the next field of the single
//!   `layout(push_constant)` block, in row order.
//!
//! Both passes read the SAME row in the SAME order, so nothing about which
//! operand is which moves; only where the value rides does. [`bindings`] is the
//! rule as code, so a shell binding a launch and a test checking a shader
//! compute it from one place rather than agreeing by habit.
//!
//! The rule is stated here rather than per-row on purpose: it is a property of
//! the API, not of any kernel, and a table that spelled it 99 times would let
//! row 100 spell it differently.
//!
//! ## What keeps it honest
//!
//! Three checks, at three distances — the same three the Metal tree has, with
//! the middle one doing more work because the generator is ours:
//!
//! * `kernels`' own unit tests pin the matcher — that a row covers every point
//!   of its axes and refuses a partial or permuted spelling.
//! * `tests/entrypoints.rs` pins the table's product against
//!   `entrypoints.generated.txt`.
//! * `scripts/vulkan-kernel-audit.py` pins that file against the shaders, by
//!   reading the `// pie:instantiate` directives the BUILD compiles from — so
//!   the audit and the build cannot disagree about what exists — and
//!   `--compile` runs `glslc` over every one of them, which proves a declared
//!   variant is a variant that builds.
//!
//! And from the other end, `model-compiler`'s `kernels::check_plan` refuses any
//! launched symbol no row declares, so a lowered text cannot state a kernel
//! this table has not heard of.
//!
//! ## The validation layer, which is not optional and is not installed
//!
//! Everything above is a comparison between two descriptions. Whether a DRIVER
//! agrees is a separate question, and the driver these were developed against
//! answers a malformed request by building the pipeline anyway. Three real
//! defects survived a green suite for weeks because of it: the coopmat tier
//! not naming `vulkanMemoryModel`, the baseline tier requiring an optional
//! `shaderInt64`, and 120 entrypoints declaring a push block wider than the
//! range their row builds.
//!
//! `tests/gpu.rs` enables `VK_LAYER_KHRONOS_validation` when the loader can see
//! it, with synchronization and GPU-assisted validation on, and an ERROR ends
//! the process. It is a soft dependency, because a build machine will not have
//! the layer and "no validation here" must not be a test failure — which does
//! mean a clean CI run is weaker evidence than a clean local one.
//!
//! It does not have to be installed system-wide. `apt-get download
//! vulkan-validationlayers`, `dpkg-deb -x` it somewhere, rewrite the
//! manifest's `library_path` to the absolute path of the extracted `.so`, and
//! point `VK_LAYER_PATH` at the directory holding the manifest.
//!
//! ## What a shell that RUNS these has to do
//!
//! There is no `driver-vulkan` yet, so these are written down here rather than
//! discovered by whoever writes one. All three are things a shader cannot
//! check for itself.
//!
//! * **Grids are workgroup-granular.** `vkCmdDispatch` counts WORKGROUPS where
//!   Metal's `dispatchThreads` counts threads, so a shell that ports a Metal
//!   grid arithmetically will round UP and launch invocations with no work.
//!   Every pointwise body here guards its own tail against the bound length of
//!   the buffer it writes, so an overshoot is harmless. An UNDERSHOOT is not,
//!   and it is the direction that fails silently: a lane that never launches
//!   writes nothing, the gap reads back as whatever the buffer held, and the
//!   dispatch completes. `KernelSig::grid_param`, `head_param` and
//!   `heads_param` say which of the STATEMENT's params give the shape, and a
//!   shell that builds a grid from the fire's numbers instead gets a wrong
//!   answer on any deployment that states two head shapes.
//! * **Enable `robustBufferAccess` unless there is a reason not to.** It makes
//!   an out-of-range access defined and discarded rather than undefined, which
//!   turns the worst residual class of shader bug from memory corruption into
//!   a wrong number. The GPU tests enable it and say plainly that doing so
//!   makes the tail guards unobservable -- the guards are there for a shell
//!   that turns it off.
//! * **One tensor, one descriptor.** Index arithmetic in these shaders is
//!   32-bit, which is safe because
//!   `VkPhysicalDeviceLimits::maxStorageBufferRange` is a `uint32_t` and so a
//!   bound range is at most 4 GiB - 1. A shell that means to address more than
//!   that has a binding problem to solve, not a shader to change.
//! * **An UNSTATED row does not describe a layout, and mistaking that for an
//!   empty one is fatal.** 56 of the 99 rows name no operands.
//!   [`buffer_count`] answers 0 for them honestly -- the row describes nothing
//!   -- but the shader behind the name still declares its bindings, so a
//!   layout built from such a row is missing every descriptor the module
//!   reads. That is not a request a driver rejects: on the machine this was
//!   written against it is a segmentation fault inside
//!   `vkCreateComputePipelines`.
//!
//!   It does NOT mean those 292 entrypoints are unlaunchable, which is what
//!   this said first. `driver-metal/src/lowering/dispatch.rs` shows the other
//!   source: where `sig.operands.is_empty()`, it falls back to the lowered
//!   plan's own argument order. The row's operand list is a reordering and
//!   verification layer over the plan, not the only description of it, and a
//!   Vulkan shell has the same fallback -- it needs a descriptor COUNT at
//!   layout time, and the plan has one. What a shell must not do is build a
//!   layout from the row and dispatch anyway.
//!
//! ## Reading this without a GPU, and without a Vulkan SDK
//!
//! All of the above runs anywhere. `default-features = false` gives the table
//! and nothing else — which is what `model-compiler` wants and all it wants —
//! and `native` adds the `glslc` pass that turns the GLSL tree into SPIR-V.
//! Unlike Metal, that pass is not optional for a shell that means to RUN:
//! `vkCreateShaderModule` takes words, not source.

mod capability;
pub use crate::capability::Capability;

use kernels::{KernelSig, Prepare};
// Named only by the doc links above, which rustdoc still has to resolve.
#[allow(unused_imports)]
use kernels::Axis;

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
/// sibling tables use for the same reason.
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
    // Empty, and it is a fact about this table rather than a placeholder: an
    // aux slot is a mamba block's, and no kernel here is one. `Prepare::Ssm`
    // does not appear in this file either.
    publishes_aux: &[],
    operands: &[],
    returns: "",
    axes: &[],
    grid_param: None,
    head_param: None,
    heads_param: None,
    rows_param: None,
    lowered_as: None,
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
        publishes_aux: k.publishes_aux,
        operands: k.operands,
        returns: k.returns,
        axes: k.axes,
        grid_param: k.grid_param,
        head_param: k.head_param,
        heads_param: k.heads_param,
        rows_param: k.rows_param,
        lowered_as: k.lowered_as,
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
/// The set `scripts/vulkan-kernel-audit.py` compares against the shader tree,
/// and — one for one — the set of `.spv` module names a `native` build writes.
pub fn entrypoints() -> Vec<String> {
    let mut out: Vec<String> = KERNELS.iter().flat_map(KernelSig::entrypoints).collect();
    out.sort();
    out
}

/// Where one operand rides. See [`bindings`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Binding {
    /// `layout(std430, binding = N)`.
    Buffer(u32),
    /// The `N`-th field of the push-constant block.
    Push(u32),
    /// Nowhere of its own: a FIELD of the packed buffer ahead of it.
    ///
    /// [`kernels::Ty::InPacked`] is how a row says "the driver has to supply
    /// this value, but it does not get a slot" — the value belongs to a struct
    /// some earlier `Buf` operand already binds, so the driver writes it while
    /// filling that buffer and the shader reads it as a struct member.
    ///
    /// Metal could fold this into the scalar run, because there a packed slot
    /// IS the buffer and a trailing scalar lands in the same argument. Vulkan
    /// splits the two runs, so folding it into the push block would push a word
    /// no shader reads and leave the struct field unwritten — the defect this
    /// variant exists to make unrepresentable.
    Packed,
}

/// Which descriptor binding each operand of `sig` takes, and which
/// push-constant field, under the two-pass rule this module's own docs state.
///
/// Both runs are indexed from zero and both follow the row's order, so the
/// answer for operand `k` is a function of the row alone — which is the point:
/// a shell binding a launch and a test checking a shader compute the same thing
/// from the same place.
///
/// A row with no operands is UNSTATED (see [`KernelSig::operands`]), and this
/// answers with an empty vector rather than inventing a nullary layout.
#[must_use]
pub fn bindings(sig: &KernelSig) -> Vec<Binding> {
    let mut buffers = 0;
    let mut pushes = 0;
    sig.operands
        .iter()
        .map(|op| {
            if matches!(op.ty, kernels::Ty::InPacked) {
                // Consumes neither run: see `Binding::Packed`.
                Binding::Packed
            } else if is_buffer(op.ty) {
                let at = buffers;
                buffers += 1;
                Binding::Buffer(at)
            } else {
                let at = pushes;
                pushes += 1;
                Binding::Push(at)
            }
        })
        .collect()
}

/// How many descriptor bindings a row's pipeline layout declares.
#[must_use]
pub fn buffer_count(sig: &KernelSig) -> u32 {
    sig.operands.iter().filter(|op| is_buffer(op.ty)).count() as u32
}

/// One scalar's place in the push-constant block, in BYTES.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PushField {
    /// The operand's name, as the row spells it.
    pub name: &'static str,
    /// Byte offset from the start of the block.
    pub offset: u32,
    /// Width in bytes: four or eight.
    pub size: u32,
}

/// The push-constant block's byte layout, which [`Binding::Push`] does NOT give.
///
/// `Binding::Push(n)` is a field INDEX, and a driver needs an offset. Turning
/// one into the other is not multiplication, because a push block follows
/// std430: a member is aligned to its own width, so an eight-byte scalar after
/// a lone four-byte one starts at 8 and not at 4. `attn/kv_write.comp` is
/// exactly that shape --
///
/// ```text
/// int head_dim; uint64_t k_head_stride; uint64_t k_seq_stride;
/// ```
///
/// -- so the naive sum of widths says 20 bytes and the real block is 24, with
/// four bytes of padding after the first field. A driver that packs by
/// concatenation writes both strides four bytes low and the shader reads two
/// halves of two different numbers. Nothing reports it: Vulkan does not know
/// what the bytes were supposed to mean.
///
/// That padding used to exist only as a hand-computed constant in a GPU test.
/// It is derived here instead, so the test, `dump_layout`, the audit and any
/// future driver all get it from one place — which is the same reason
/// [`bindings`] exists rather than each of them counting buffers by hand.
#[must_use]
pub fn push_layout(sig: &KernelSig) -> Vec<PushField> {
    let mut at = 0u32;
    sig.operands
        .iter()
        .filter(|op| !is_buffer(op.ty) && !matches!(op.ty, kernels::Ty::InPacked))
        .map(|op| {
            let size = push_width(op.ty);
            at = at.next_multiple_of(size);
            let field = PushField {
                name: op.name,
                offset: at,
                size,
            };
            at += size;
            field
        })
        .collect()
}

/// The push block's total size in bytes, padding included.
///
/// Rounded up to the block's own alignment — the widest member — because that
/// is what a `VkPushConstantRange` covering the whole block has to be, and
/// because `vkCmdPushConstants` takes a size that must be a multiple of four.
#[must_use]
pub fn push_size(sig: &KernelSig) -> u32 {
    let fields = push_layout(sig);
    let Some(last) = fields.last() else { return 0 };
    let align = fields.iter().map(|f| f.size).max().unwrap_or(4);
    (last.offset + last.size).next_multiple_of(align)
}

/// Every scalar kind a row can name is four or eight bytes wide.
fn push_width(ty: kernels::Ty) -> u32 {
    match ty {
        kernels::Ty::Usize | kernels::Ty::I64 => 8,
        _ => 4,
    }
}

/// Whether a kind crosses as a device allocation rather than as a value.
///
/// Read off the KIND and not off a list of operand names, so a row that grows a
/// buffer cannot land in the push block by omission. The struct-shaped and
/// handle kinds of the CUDA vocabulary are not reachable from a Vulkan row —
/// there is no stream and no cuBLAS handle here — so they answer `false`, and a
/// row that used one would put a plan cache in a push constant: a failure at
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
            | Ty::StructuredMasks
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two runs are independent and both start at zero — the property a
    /// shell's binder and a shader's `layout` qualifiers have to agree on.
    #[test]
    fn buffers_and_push_constants_are_numbered_independently() {
        let row = kernels::sig_in(KERNELS, "rms_single_row_bfloat16").expect("a stated row");
        assert_eq!(
            bindings(row),
            vec![
                Binding::Buffer(0), // x
                Binding::Buffer(1), // w
                Binding::Buffer(2), // out
                Binding::Buffer(3), // params
            ]
        );

        let row = kernels::sig_in(KERNELS, "affine_qmv_fast_bfloat16_gs_64_b_4").expect("a row");
        assert_eq!(
            bindings(row),
            vec![
                Binding::Buffer(0), // w
                Binding::Buffer(1), // scales
                Binding::Buffer(2), // biases
                Binding::Buffer(3), // x
                Binding::Buffer(4), // y
                Binding::Push(0),   // in_vec_size
                Binding::Push(1),   // out_vec_size
            ]
        );
        assert_eq!(buffer_count(row), 5);
    }

    /// A packed field consumes NEITHER run.
    ///
    /// Found by comparing every shader's push block against the table field by
    /// field: `row_gather.comp` declared no push constants at all, because the
    /// count it needs is a member of the params struct it already binds. The
    /// table was right and `bindings()` was wrong — it had inherited Metal's
    /// "append to the scalars" reading, which would have had the driver push a
    /// word the shader never reads while `p.count` stayed whatever the params
    /// buffer happened to hold.
    #[test]
    fn a_packed_field_takes_no_slot_of_its_own() {
        let row = kernels::sig_in(KERNELS, "row_gather_bfloat16").expect("a stated row");
        assert_eq!(
            bindings(row),
            vec![
                Binding::Buffer(0), // input
                Binding::Buffer(1), // out
                Binding::Buffer(2), // rows
                Binding::Buffer(3), // params — the struct the count lives in
                Binding::Packed,    // count
            ]
        );
        assert_eq!(buffer_count(row), 4);
        assert!(!bindings(row).iter().any(|b| matches!(b, Binding::Push(_))));
    }

    /// An unstated row gets no layout rather than a nullary one.
    #[test]
    fn an_unstated_row_has_no_bindings() {
        let row = kernels::sig_in(KERNELS, "argmax_logits_bfloat16").expect("a row");
        assert!(row.operands.is_empty());
        assert!(bindings(row).is_empty());
    }

    /// `maxPushConstantsSize` is 128 bytes on the floor of the desktop Vulkan
    /// implementations (and llama.cpp treats 128 as the number to respect), so
    /// a row whose scalars overflow it is a row whose launch cannot be issued.
    ///
    /// This used to sum the widths, which is the wrong number: a push block is
    /// std430, so an eight-byte scalar after a lone four-byte one is preceded
    /// by four bytes of padding, and the sum UNDER-counts. Under-counting is
    /// the dangerous direction for a ceiling — it lets a row that really does
    /// overflow pass — so it asks [`push_size`] now.
    #[test]
    fn no_row_overflows_the_push_constant_floor() {
        for row in KERNELS {
            let bytes = push_size(row);
            assert!(
                bytes <= 128,
                "`{}` wants {bytes} bytes of push constants; the floor is 128",
                row.symbol
            );
        }
    }

    /// The padding is real, and this is the row that has it.
    ///
    /// `attn/kv_write.comp` declares `int head_dim; uint64_t k_head_stride;
    /// uint64_t k_seq_stride;`, so the block is 4 + 4 pad + 8 + 8 = 24 and not
    /// the 20 that adding the widths gives. A driver packing by concatenation
    /// writes both strides four bytes low, and the shader reads two halves of
    /// two different numbers with nothing to report it.
    #[test]
    fn an_eight_byte_scalar_after_a_four_byte_one_is_padded() {
        let row = kernels::sig_in(KERNELS, "kv_append_bfloat16").expect("a row");
        let fields = push_layout(row);
        let places: Vec<(&str, u32)> = fields.iter().map(|f| (f.name, f.offset)).collect();
        assert_eq!(
            places,
            vec![("head_dim", 0), ("k_head_stride", 8), ("k_seq_stride", 16)]
        );
        assert_eq!(push_size(row), 24);

        // A block of one width needs no padding, and must not acquire any.
        let plain =
            kernels::sig_in(KERNELS, "affine_qmv_routed_bias_bfloat16_gs_64_b_4").expect("a row");
        assert_eq!(
            push_layout(plain)
                .iter()
                .map(|f| f.offset)
                .collect::<Vec<_>>(),
            vec![0, 4, 8, 12, 16]
        );
        assert_eq!(push_size(plain), 20);
    }
}
