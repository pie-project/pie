//! Layer 2: the rows that name a device kernel, as opposed to a launcher.
//!
//! # The difference between this table and [`crate::table`]
//!
//! A row in [`crate::table`] describes a `pie_k_*` entry point: a C++ host
//! function that holds a `<<<>>>`, takes a stream as an argument, and was
//! compiled by nvcc months ago. A row HERE describes a `__global__` template
//! and the type to instantiate it at — three strings, no entry point, no
//! launcher, no `.cu`:
//!
//! * WHICH kernel — [`DeviceKernel::template_path`] and [`DeviceKernel::elem`],
//!   which together spell a C++ instantiation. `nvrtcAddNameExpression` takes
//!   that string and `nvrtcGetLoweredName` answers with the mangled symbol, so
//!   the instantiation SET lives in a Rust table rather than in an explicit-
//!   instantiation list, a `.def` file and a CMake regex.
//! * WHAT to pass it — `KernelSig::operands`, as every row already states.
//! * HOW to launch it — `KernelSig::launch`, which is what the C++ launcher
//!   used to hold inside its `<<<>>>`.
//!
//! That is the whole of what a kernel needs to exist under a JIT, and the
//! measured consequence is in the source these rows name: adding an fp16
//! `residual_add` the ahead-of-time build never had cost one row and no C++
//! at all, because the template and its `Elem` specialisation were already
//! there. Under nvcc the same addition costs a translation unit's worth of
//! `cicc` for a kernel nobody has asked for yet, which is exactly why it was
//! never added.
//!
//! # These rows were `kernels-cuda/src/norm_device.rs` until the seam closed
//!
//! They were authored there, beside the ahead-of-time twins they replace, and
//! this module was a `pub use` of them for [`crate::table`]'s reason: while
//! both paths must run, one file is one contract, and a copy here would be a
//! second contract that agrees until the day it does not. That file said the
//! same thing in the same words — *"a symbol `model-compiler` can state must
//! have exactly one contract, and until the shim path retires, the twin is
//! it."*
//!
//! The rows moved rather than being copied, which is what the argument
//! always demanded; what changed is which crate reads which. `kernels-cuda`
//! now takes [`DeviceKernel`] FROM here and re-exports it under
//! `kernels_cuda::norm_device`, so its `abi::emit_device_typecheck` and
//! `driver-cuda`'s dispatch generator kept compiling untouched.
//!
//! One name changed with the move and it changed in the export rather than
//! here: `ENTRIES` is [`ALTUP_AUX`], because a unit's name is the file it
//! compiles and `ENTRIES` says nothing about which file that is.
//! `kernels_cuda::norm_device::ENTRIES` is an alias of it, spelled once, in
//! the re-export.
//!
//! # And the one thing a row could not say: which of two kernels
//!
//! Seventeen kernels across the family modules are refused for one shared
//! reason, phrased as *"one arm of a host run-time choice"*. `rmsnorm_vec8`
//! is the clearest: `rmsnorm.cu` reaches it only through `rmsnorm_vec8_ok`,
//! which tests three pointer alignments and three strides and falls back to
//! the scalar twin. A row cannot state that, and the refusal was correct —
//! freezing one arm is choosing a kernel rather than describing one, and the
//! frozen row is wrong on the other arm.
//!
//! **Under a JIT it is not a blocker, it is the point.** The ahead-of-time
//! build pushed the decision into a host `if` because it had to choose its
//! instantiations months earlier; a fire knows the actual pointers and the
//! actual strides. [`Specialisation`] is that, and the four types under it —
//! [`Fact`], [`Term`], [`Take`], [`Arm`] — exist to keep it from becoming a
//! licence for a row to say *"and then decide something"*:
//!
//! * [`Fact`] fixes what a predicate MAY read — an address as a number, an
//!   integer already bound, or a host FLAG. There is no variant for a
//!   device-side value, so a predicate needing a synchronisation cannot be
//!   written.
//! * [`Term`] is DATA, so the predicate can be printed beside the C++ it
//!   reproduces and swept case by case. A function pointer would be
//!   unauditable and would not stop the next edit adding a `cudaMemcpy`.
//! * [`Take`] states the correspondence between two kernels' parameter lists,
//!   which `rmsnorm.cu` states twice as two `<<<>>>` argument lists eight
//!   lines apart.
//! * [`Specialisation::agrees`] refuses an arm that changes the launch, the
//!   unit, the argument types or the arity — so an arm can only ever choose a
//!   different INSTANTIATION of the same contract.
//!
//! # The second shape of that decision: a flag, and a `template <bool>`
//!
//! `rmsnorm_vec8_ok` is a predicate the host COMPUTES. The other shape is a
//! flag the host was HANDED, and it is the larger group: nine rows in `attn`
//! are refused because a `bool` is said to select between two arms of a
//! `template <bool>`. `kv_paged.cu:51` is the whole of it —
//!
//! ```text
//! if (hnd_layout) { write_kv<true> <<<n, 256, 0, s>>>(k_curr, ..., first_token); }
//! else            { write_kv<false><<<n, 256, 0, s>>>(k_curr, ..., first_token); }
//! ```
//!
//! — and every word of the refusal was right: the two arms write a correctly
//! shaped cache in different memory orders, so a row that froze one is *"no
//! fault, no error, a wrong read on the next decode"*.
//!
//! [`Fact::Bool`] and [`Term::Is`] are what a row states instead, and the
//! shape is not quite `rmsnorm_vec8`'s. **The flag is an operand of the
//! CONTRACT that no INSTANTIATION takes** — `write_kv`'s parameter list is
//! the same fifteen either way — so the base row carries it, each arm drops
//! it through [`Take`], and the arms must cover BOTH values or a fire falls
//! through to a base row whose argument list is one cell longer than the
//! kernel it names. That last is checked, without a device, by
//! [`Specialisation::agrees`].
//!
//! What a family module writes, for the first of the nine, in full:
//!
//! ```text
//! // Three rows: the contract, and the two instantiations.
//! static WRITE_KV_ROWS: &[DeviceKernel] = &[
//!     DeviceKernel { sig: &SIGS[0], template_path: "attn::device::write_kv",
//!                    elem: "device::false_type::value" },   // the contract
//!     DeviceKernel { sig: &SIGS[1], template_path: "attn::device::write_kv",
//!                    elem: "device::true_type::value"  },   // #hnd
//!     DeviceKernel { sig: &SIGS[2], template_path: "attn::device::write_kv",
//!                    elem: "device::false_type::value" },   // #nhd
//! ];
//!
//! // SIGS[0] is the launcher's contract: the kernel's fifteen operands and
//! // `hnd_layout: Bool` last. SIGS[1] and SIGS[2] are the kernel's fifteen.
//!
//! pub static WRITE_KV: Specialisation = Specialisation {
//!     base: "attn::write_kv_bf16",
//!     arms: &[
//!         Arm { name: "hnd", when: &[Term::Is { operand: 15, value: true }],
//!               row: &WRITE_KV_ROWS[1], take: FIFTEEN,
//!               because: "kv_paged.cu:51 `if (hnd_layout)` -> write_kv<true> at :52" },
//!         Arm { name: "nhd", when: &[Term::Is { operand: 15, value: false }],
//!               row: &WRITE_KV_ROWS[2], take: FIFTEEN,
//!               because: "kv_paged.cu:51 `else` -> write_kv<false> at :62" },
//!     ],
//! };
//! // where FIFTEEN is `&[Take::From(0), ..., Take::From(14)]` — the base's
//! // operands in the kernel's order, with the flag at 15 not forwarded.
//! ```
//!
//! plus one line in [`SPECIALISED`]. `elem` is spelled
//! `device::true_type::value` and not `true` because
//! [`DeviceKernel::instantiation`] prefixes the FIRST template argument with
//! `::pie_cuda_driver::kernels::` and a bare literal cannot survive that;
//! [`args`] records the eleven forms this was measured over, the last three
//! of which are exactly this shape.
//!
//! **Every line of that is executed.** `tests/specialise.rs`'s `flag_arms`
//! states exactly those rows, compiles them against the shipped
//! `attn/kv_paged` unit, and fires both arms of the real
//! `attn::device::write_kv` on an L40S: 220 800 bf16 cells against a
//! hand-computed scatter, 0 differing, over five shapes at both layouts —
//! and a negative control that fires the arm the flag did not name and moves
//! 34 273 of 55 200 cells while writing the same NUMBER of values, which is
//! why a count or a norm would not have caught it. It also confirms the one
//! detail the shape above depends on and nothing else in the tree does: the
//! base row and the `#nhd` row name the same instantiation, so the unit asks
//! NVRTC for one name expression twice, and it resolves rather than
//! conflicting.
//!
//! ## What this does NOT reach, named individually
//!
//! Nine rows were reported blocked on a flag. This vocabulary reaches five of
//! them and the record is worth more than the count:
//!
//! * **Reached.** `write_kv`, `write_kv_at_positions`, `write_kv_explicit`,
//!   `write_kv_explicit_devwin` and `copy_kv_cells` are all
//!   `template <bool HND_LAYOUT>` over a host `if (hnd_layout)` at
//!   `kv_paged.cu:51`, `:203`, `:251`, `:339` and `:386`, all
//!   `<<<n, 256, 0, s>>>`. Two arms, one flag, [`LaunchRule::PerRow`].
//! * **`write_kv_per_token_head<UseFp8>` is not.** Its selector is not a
//!   flag: `kv_paged.cu:122` is a `switch (layer.scheme)` over FOUR cases,
//!   two of which reach this template and two of which reach
//!   `write_kv_fp8_per_tensor` and `write_kv_fp4_block` — different kernels,
//!   not different instantiations. Spelling that as a `Bool` operand would
//!   describe a four-valued choice as a two-valued one. Its geometry is also
//!   `dim3(total_tokens, num_kv_heads)` with `2 * (BLOCK/32) * sizeof(float)`
//!   of dynamic shared memory, which no rule states.
//! * **`qkv_decode_qk_norm_rope_write_kv_bf16` and its `_devwin` twin are
//!   not**, and the flag was never the blocker. `hnd_layout` is a RUN-TIME
//!   parameter of that kernel (`qkv_fused.cu:466`), so no arm bakes it; what
//!   the launcher templates on is `rope_table != nullptr` and a `head_dim` of
//!   64, 128 or 256, and the two families it dispatches between launch at
//!   DIFFERENT geometries — `warp_grid` at 256 against
//!   `dim3(num_requests, num_q_heads + num_kv_heads)` at 128. An arm may not
//!   change the geometry, so that is a refusal [`Specialisation::agrees`]
//!   already makes for the right reason. The file is also still
//!   `crates/kernels-cuda/csrc/src/attn/qkv_fused.cu` and has no `.cuh` here.
//! * **`attention_mtp_paged_history_bf16` is not**, and it is not a
//!   `template <bool>` at all: `attention_naive.cuh:410` is
//!   `template <class T>` and takes `bool hnd_layout` and `bool prefix_global`
//!   as kernel PARAMETERS. Its blockers are the ones its own header states —
//!   a per-head `dim3(num_q_heads, num_tokens)`, `extern __shared__` sized by
//!   `max_global_tokens + history_steps`, and a launcher that switches to a
//!   different kernel on a comparison no [`Term`] can spell. The ROW is gone
//!   since §38 — nothing reached it — so this is now a finding about
//!   `attention_naive.cu:80`, the launcher, which stays: it is the only
//!   caller of `attention_mtp_history_bf16` at `:52`, and deleting it would
//!   orphan two launchers and two `<<<>>>` at once.
//!
//! So: five launchers, four of them rows, and four findings that a flag was
//! the wrong diagnosis.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Lit;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

/// AltUp's epsilon, which is the ALGORITHM's and not the model's -- the same
/// constant, and the same argument, as [`crate::table::norm`]'s.
const ALTUP_EPS: f32 = 1e-5;

/// One kernel a row can state, as a template and the type to instantiate it
/// at.
///
/// Three strings rather than one symbol, because a C++ template instantiation
/// is not a name -- it is a name applied to a type, and the operand list a
/// row declares is only meaningful once the type is chosen. Every one of the
/// three is checked: `kernels_cuda::abi::emit_device_typecheck` spells all of them
/// into a translation unit that does not compile if any is wrong.
pub struct DeviceKernel {
    /// The contract: operands, launch rule, in-place claims.
    pub sig: &'static KernelSig,
    /// The `__global__` template, under `::pie_cuda_driver::kernels`.
    pub template_path: &'static str,
    /// The element type to instantiate it at, under the same root -- or
    /// [`DeviceKernel::PLAIN`], if the `__global__` takes no template
    /// arguments at all.
    ///
    /// Two strings and not three. An earlier version of this row carried a
    /// TAG type and its storage type separately, because `bf16` and `f16`
    /// were both `unsigned short` and a template keyed on storage would have
    /// held one instantiation where a row means two. Testing that version
    /// found it was not merely redundant but UNCHECKABLE: a row that swapped
    /// the two formats compiled, because there were no two types to swap.
    ///
    /// Making the formats distinct structs in the header fixed both -- the
    /// tag became unnecessary and `const bf16*` where `const f16*` is meant
    /// became a conversion C++ refuses. The measurement is in the pilot
    /// write-up; the shorter row is the visible half of it.
    pub elem: &'static str,
}

impl DeviceKernel {
    /// What a row states in [`DeviceKernel::elem`] when its `__global__` has
    /// **no template parameter list at all**, so its name is its qualified
    /// path and nothing else.
    ///
    /// # Why this exists, and why it is a named constant rather than `""`
    ///
    /// Five kernels across three independent migrations were refused with one
    /// sentence -- *"[`DeviceKernel::instantiation`] always emits `path<...>`,
    /// so a plain `__global__` cannot be named"*. The first half is a fact
    /// about a Rust `format!`. The second half is a claim about NVRTC, and it
    /// is FALSE: `examples/argform_probe.rs`'s twelfth case hands
    /// `nvrtcAddNameExpression` the bare path
    /// `::pie_cuda_driver::kernels::probe::plain`, NVRTC accepts it,
    /// `nvrtcGetLoweredName` answers `_ZN15pie_cuda_driver7kernels5probe5plainEPii`
    /// and `cuModuleGetFunction` RESOLVES that name on this L40S. A bare path
    /// is not a weaker kind of name than a template-id; the same probe
    /// resolves `oneflag<true>` beside it and both come back a `CUfunction`.
    ///
    /// So the limit was in this file, and it is gone. What is left is the
    /// question that decides whether the fix is safe: **what does a row with
    /// no `elem` MEAN**, and can it be confused with a row whose `elem` was
    /// left unfilled?
    ///
    /// The empty string would leave that ambiguous. `elem: ""` is what a
    /// half-written row looks like, and reading it as *"this kernel takes no
    /// template arguments"* would make the most common editing mistake into a
    /// silent change of meaning -- `path<>` for a template becomes `path`,
    /// which for an OVERLOADED name would resolve to something. This constant
    /// is not writable by accident, and it is not spellable as C++ either, so
    /// a `format!` that forgot to branch on it produces
    /// `path<::pie_cuda_driver::kernels::(no template arguments)>` -- an NVRTC
    /// diagnostic that quotes this sentence back.
    ///
    /// # The two ways to get it wrong are both REFUSED, measured
    ///
    /// That is what makes the meaning unambiguous rather than merely
    /// permitted, and neither refusal is this crate's -- both are NVRTC's,
    /// which means `tests/units.rs` catches them for every row on the box:
    ///
    /// * A plain kernel stated WITH an `elem` spells `plain<device::bf16>`,
    ///   and NVRTC answers `type name is not allowed`.
    /// * A template kernel stated as [`DeviceKernel::PLAIN`] spells a bare
    ///   `oneflag`, and NVRTC answers `cannot determine which instance of
    ///   function template "..." is intended`.
    /// * An empty `elem` on a template kernel spells `path<>`, which is
    ///   `expected an expression` -- so even the reading this constant exists
    ///   to forbid would not have compiled silently.
    ///
    /// # What it does NOT change
    ///
    /// The linkage of the header the kernel lives in. §21.6's measurement
    /// still holds: a non-template `__global__` in a `.cuh` takes external
    /// linkage, so that header may be included by exactly one translation
    /// unit and a second includer is a `multiple definition` at link. A row
    /// does not include anything -- NVRTC compiles the root on its own -- so
    /// naming a plain kernel and lifting its single-includer constraint are
    /// now two independent decisions, and the second is a linkage decision to
    /// be made per header for its own reasons.
    pub const PLAIN: &'static str = "(no template arguments)";

    /// Whether this row names a `__global__` with no template parameter list.
    ///
    /// One predicate rather than an `== PLAIN` at each reader, because the
    /// readers are in three crates and a comparison spelled four times is a
    /// comparison three of which can be forgotten.
    #[must_use]
    pub const fn is_plain(&self) -> bool {
        // `const` string equality, byte for byte: `str::eq` is not const.
        let (a, b) = (self.elem.as_bytes(), Self::PLAIN.as_bytes());
        if a.len() != b.len() {
            return false;
        }
        let mut i = 0;
        while i < a.len() {
            if a[i] != b[i] {
                return false;
            }
            i += 1;
        }
        true
    }

    /// The instantiation, as C++ spells it and as `nvrtcAddNameExpression`
    /// takes it.
    ///
    /// Fully qualified from the global namespace, because a name expression
    /// is resolved in no particular scope and an unqualified one would depend
    /// on where NVRTC happened to look.
    ///
    /// Two shapes, and which one a row gets is decided by
    /// [`DeviceKernel::is_plain`] alone:
    ///
    /// ```text
    /// a template   ::pie_cuda_driver::kernels::PATH<::pie_cuda_driver::kernels::ELEM>
    /// a plain one  ::pie_cuda_driver::kernels::PATH
    /// ```
    ///
    /// The prefix is glued to the FRONT of the whole `elem` string and
    /// therefore reaches its first TOKEN only -- see [`args`] for the eleven
    /// argument forms that were measured against that, and
    /// [`DeviceKernel::PLAIN`] for the twelfth, which is this branch.
    #[must_use]
    pub fn instantiation(&self) -> String {
        if self.is_plain() {
            return format!("::pie_cuda_driver::kernels::{}", self.template_path);
        }
        format!(
            "::pie_cuda_driver::kernels::{}<::pie_cuda_driver::kernels::{}>",
            self.template_path, self.elem
        )
    }
}

// ---------------------------------------------------------------------------
// The rows themselves, and why there are two tables rather than one.
//
// A unit is ONE NVRTC compile over one source, so a table is per-`.cuh` and
// not per-family: `altup_aux.cuh` and `elementwise.cuh` are two files and
// therefore two compiles (§6.4 of `new-horizon.md` fixes that granularity).
//
// Neither is the end state. When the C++ launcher for a row is deleted, the
// row's twin in `crate::table::norm` loses its `stream` operand, gains a
// `launch`, and the two tables become one. Both exist while both paths must
// run, because the measurement is an A/B and an A/B needs both arms.
// ---------------------------------------------------------------------------

/// The `__global__` templates `csrc/src/norm/altup_aux.cuh` holds, and the
/// instantiations of them these rows state.
pub static ALTUP_AUX: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &SIGS[0],
        template_path: "norm::device::compute_rms",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[1],
        template_path: "norm::device::magnitude_rescale",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[2],
        template_path: "norm::device::mean_streams",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[3],
        template_path: "norm::device::unpack_predict_coefs",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[4],
        template_path: "norm::device::unpack_correct_coefs",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &SIGS[5],
        template_path: "norm::device::tanh_inplace",
        elem: "device::bf16",
    },
    // The fp16 kernel this tree never had, and the whole of what it cost:
    // one row naming a different element type. There is no new C++ anywhere
    // -- the template and its `Elem` specialisation were already there.
    DeviceKernel {
        sig: &SIGS[6],
        template_path: "norm::device::tanh_inplace",
        elem: "device::f16",
    },
];

/// The contracts, in [`ALTUP_AUX`]' order.
///
/// Split from the instantiations so a row stays a [`KernelSig`] -- the type
/// `model-compiler`, `check_plan` and every other reader already takes. What
/// Tier A adds is beside it, not inside it.
#[rustfmt::skip]
static SIGS: [KernelSig; 7] = [
    // One block per row, and the row width is read by a stride loop, so `h`
    // is the only extent the kernel sees. `Rms` means on CUDA what it means
    // on Metal -- a row-wise reduction, one group per row -- and the
    // arithmetic that picks the block width is `driver-cuda`'s, beside the
    // sentence that explains it.
    kernel!(compute_rms "norm::compute_rms_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Rms,
        operands = operands![
            reference: Buf <- Source::In(0),
            target_rms_out: F32sMut <- Source::Out(0),
            h: I32 <- Source::InWidth(0),
            eps: F32 <- Source::Lit(Lit::F32(ALTUP_EPS)),
        ]),
    kernel!(magnitude_rescale "norm::magnitude_rescale_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Rms,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            target_rms: F32s <- Source::In(1),
            h: I32 <- Source::OutWidth(0),
            eps: F32 <- Source::Lit(Lit::F32(ALTUP_EPS)),
        ]),
    // `t_stride` SURVIVES here and nowhere else: `streams` is `[K, T, H]`,
    // so the k-th plane begins at `k * T * H` and the kernel cannot address
    // its input without it. The grid still covers the rows -- what is passed
    // is a stride that happens to equal an extent, which is exactly the pair
    // the old signatures could not tell apart.
    kernel!(mean_streams "norm::mean_streams_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::ElementwiseRows,
        operands = operands![
            streams: Buf <- Source::In(0),
            out: BufMut <- Source::Out(0),
            k: I32 <- Source::CtxNonZero("altup_streams"),
            t_stride: I32 <- Source::Rows,
            h: I32 <- Source::OutWidth(0),
        ]),
    // `RouteRows` -- one block per row, as wide as the row. The row is `K*K`
    // wide and `K` is its integer square root, which is the same `Source`
    // the twin states; nothing about how an operand is SOURCED changes in
    // Tier A, only which operands there are.
    kernel!(altup_unpack_predict_coefs "norm::altup_unpack_predict_coefs",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            in_bf16: Buf <- Source::In(0),
            out_fp32: F32sMut <- Source::Out(0),
            k: I32 <- Source::Isqrt(&Source::Width(&Source::In(0))),
        ]),
    kernel!(altup_unpack_correct_coefs "norm::altup_unpack_correct_coefs",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            in_bf16: Buf <- Source::In(0),
            out_fp32: F32sMut <- Source::Out(0),
            k: I32 <- Source::InWidth(0),
        ]),
    // The flat one: rows stack, so the extent is elements and the guard is
    // the kernel's own. `numel` is not geometry the rule can recover --
    // `Elementwise` reads `rows * width` and this operand says the same
    // number -- so it stays an argument.
    kernel!(tanh "norm::tanh_bf16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            numel: I32 <- Source::OutElements(0),
        ]),
    // A DIFFERENT symbol for a different numeric format, because a symbol is
    // what a text states and a text that says `tanh` must not get to choose
    // its own precision. The row above and this one share every other word.
    kernel!(tanh_f16 "norm::tanh_f16",
        file = Some("norm/altup_aux.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            numel: I32 <- Source::OutElements(0),
        ]),
];

// ---------------------------------------------------------------------------
// The pointwise pair: `csrc/src/norm/elementwise.cuh`.
//
// A second unit rather than more rows in `ALTUP_AUX`, because a unit is ONE
// NVRTC compile over one source and these live in a different file. §6.4 of
// `new-horizon.md` fixes the granularity: one compile per module, many name
// expressions per compile.
// ---------------------------------------------------------------------------

/// The `__global__` templates `csrc/src/norm/elementwise.cuh` holds.
///
/// Three rows over two templates. The third is the fp16 `residual_add` the
/// AOT build never had — not because it was hard, but because instantiating
/// it cost a translation unit's worth of cicc for a kernel nothing had asked
/// for yet. Under a JIT it costs this line.
pub static ELEMENTWISE: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &ELEMENTWISE_SIGS[0],
        template_path: "norm::device::residual_add",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ELEMENTWISE_SIGS[1],
        template_path: "norm::device::scalar_mul",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &ELEMENTWISE_SIGS[2],
        template_path: "norm::device::residual_add",
        elem: "device::f16",
    },
];

/// The contracts, in [`ELEMENTWISE`]'s order.
///
/// Each is its AOT twin minus the stream: a stream is `cuLaunchKernel`'s
/// sixth PARAMETER, outside the `void**`, so it is not an operand and stating
/// it as one was the shim's requirement rather than the kernel's
/// (`new-horizon.md` §4.2).
#[rustfmt::skip]
static ELEMENTWISE_SIGS: [KernelSig; 3] = [
    // `Elementwise` IS the launcher these replace: `(n + 255) / 256` blocks
    // of 256, and an empty `n` refused rather than launched. Both halves were
    // four lines of C++ that the rule now states once.
    kernel!(residual_add_cuda "norm::residual_add_bf16",
        file = Some("norm/elementwise.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut <- Source::Out(0),
            x: Buf <- Source::In(1),
            n: Usize <- Source::OutElements(0),
        ]),
    kernel!(scalar_mul "norm::scalar_mul_bf16",
        file = Some("norm/elementwise.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            x: BufMut <- Source::Out(0),
            // A NUMBER when the statement carries one, a NAME otherwise. The
            // first form is the one a reader can check against the text; the
            // second is what it replaces.
            s: F32 <- Source::Or(&Source::ParamF32(0), &Source::NamedScale),
            n: Usize <- Source::OutElements(0),
        ]),
    // The fp16 twin, which has no AOT counterpart and therefore no symbol to
    // share. Named for what it is.
    kernel!(residual_add_f16_cuda "norm::residual_add_f16",
        file = Some("norm/elementwise.cuh"),
        launch = LaunchRule::Elementwise,
        in_place = &[(0, 0)],
        operands = operands![
            y: BufMut <- Source::Out(0),
            x: Buf <- Source::In(1),
            n: Usize <- Source::OutElements(0),
        ]),
];

/// The rows the DISPATCHER routes to the JIT path, by symbol.
///
/// A subset of [`ELEMENTWISE`], and the distinction is the one that decides
/// whether a `.cu` can be deleted. Every row in `ELEMENTWISE` is COMPILED at
/// run time; a row named here is also LAUNCHED that way, which means its
/// generated arm calls `bind::jit::fire` and its shim entry is not emitted —
/// so the host launcher behind it has no consumer left.
///
/// `norm::residual_add_bf16` is deliberately absent. Its launcher is called
/// from `gemm.cpp` as a building block — C++ composing with C++, which no
/// Rust dispatch can intercept — so deleting it would break a kernel that
/// has not migrated. The rule is the one §10.2 found for shared headers, one
/// level up: **a launcher goes when its whole consumer set has gone**, and
/// the shim is only one consumer.
///
/// This named `gemma4_vision.cu` as a second caller and that half is now
/// **stale**: the tower converted to direct launches and calls
/// `norm::rmsnorm_bf16` six times, not `residual_add`. Left recorded rather
/// than silently corrected, because a consumer set that decays without
/// anything noticing is the same failure as a wall in front of a door nobody
/// opens — the sentence stayed true-sounding for exactly as long as nobody
/// re-measured it. The current set is 11 symbols wide and lives in
/// `examples/dispatch_countdown.rs`, where a stale entry exits 101.
///
/// **And the paragraph above decayed in its turn, within the day.** The six
/// `rmsnorm_bf16` calls it records are now zero: `towers-move` inlined
/// `rmsnorm_strided_bf16`'s body into the tower (byte-identical on real
/// gemma-4-E4B-it weights, 133,156 bytes, with a negative control forcing the
/// wrong arm that moves 44.79% of them) and relocated the three towers to
/// `driver-cuda/csrc/vision/` entirely. `norm::rmsnorm_bf16`'s C++ consumer
/// set is `norm/rmsnorm.cu` alone. Appended rather than rewritten, in this
/// comment's own idiom — but the lesson has changed shape on the second pass:
/// recording a decay does not stop the next one, it only dates it. The set is
/// 9 symbols now, not 11, and what actually catches this is that
/// `dispatch_countdown` re-derives every citation from the sources on each
/// run and exits 101; prose that describes a measurement is not one.
pub static JIT_DISPATCHED: &[&str] = &[
    // Tier A's six, and the payoff it never collected. `537294a7a` moved the
    // `__global__`s into `altup_aux.cuh` and proved them on a device, but the
    // dispatcher went on calling the shim -- so `altup_aux.cu` kept its own
    // copy of all six kernels AND their launchers, and the two could drift
    // with every test passing on whichever half it exercised.
    "norm::compute_rms_bf16",
    "norm::magnitude_rescale_bf16",
    "norm::mean_streams_bf16",
    "norm::altup_unpack_predict_coefs",
    "norm::altup_unpack_correct_coefs",
    "norm::tanh_bf16",
    // The pointwise pair's scalar half (§10.8).
    "norm::scalar_mul_bf16",
    // BATCH 1 -- `layout`, the rows that state their sources. §37.
    //
    // The first rows dispatched since the list was written, and the reason
    // the gap was three months wide is not that these were hard: it is that
    // `jit_dispatched()` joined against two tables and these live in eight
    // others, so naming one here failed with "is dispatched to the JIT and
    // has no row" -- a message about the row, when the row was there and the
    // LOOKUP was narrow. `device::row` now scans `unit::rows()`.
    //
    // `layout` is first because it is the family with the least to argue
    // about: every row is a copy or a permutation with a stated rectangle,
    // none takes a plan or a cache view, and none has a C++ caller -- the
    // whole family is absent from the held set, which is 8 symbols across
    // `gemm.cpp`, `rmsnorm.cu` and `quant_bf16_to_fp8.cu`.
    //
    // FOUR OF THE EIGHT ARE NOT HERE, and the four are the lesson. Adding
    // all eight failed `routed_rows_have_an_arm` in `driver-cuda/build.rs`
    // on `concat_bf16_rows`, `copy_if_valid_slot`, `deinterleave_rows_bf16`
    // and `deinterleave_vec_bf16` -- rows whose operands carry
    // `Source::Unbound`, which `emit_rust_dispatch` skips WHOLE, before it
    // ever reaches the JIT branch. They have no generated arm of either
    // kind; a hand-written arm calls `ffi::pie_k_*` for them. Naming one
    // here deletes the shim entry that hand arm links against, which is
    // §22.1's link error rather than §22.1's fire-time lie -- the same
    // defect from the other side.
    //
    // So "migrated" and "routable" differ by more than a C++ caller: a row
    // must also SAY WHERE ITS ARGUMENTS COME FROM. Four of layout's eight do
    // not, and their doc comments already said so ("this row generates no
    // dispatch and claims none") -- one page away from a list that would
    // have broken on it.
    "layout::gather_bf16_rows",
    "layout::split_bf16_rows",
    "layout::split_qwen_gdn_ba_bf16",
    "layout::transpose_bf16_nld_to_lnd",
    // BATCH 2 -- `mlp`, the whole ready set of the unit, and the first batch
    // landed on MEASURED parity rather than on argument. §40.
    //
    // The unit is the batch, not the family: a unit is one `.cuh`, one NVRTC
    // translation unit and one set of instantiations, so a batch that is a
    // unit either compiles as a whole or fails as a whole, and the failure
    // names the file. `mlp` has 12 ready-or-held rows across
    // `mlp/swiglu.cuh` and `mlp/gaussian_topk.cuh`; the eleven here are the
    // ready ones and `chunked_swiglu_bf16` is held by `mlp/swiglu.cu`.
    //
    // EVERY ONE OF THESE FIRED BOTH WAYS AND MATCHED BYTE FOR BYTE.
    // `driver-cuda/tests/jit_parity.rs` fires the same `BoundLaunch` through
    // `bind::dispatch` (the archive, while the row is still unrouted) and
    // through `bind::dispatch_jit_probe` (the arm routing emits, generated
    // by the same emitter), and compares whole allocations including a
    // 256-byte guard tail: 33 shapes, 0 differing bytes. Three shapes per
    // row, one of them narrow on the axis the row's `LaunchRule` adds,
    // because §21's 16x RMSNorm grid wrote 491,520 values past the rectangle
    // with 0 differing INSIDE it and `AltUpStreams` hid at `kv_heads = 8`.
    //
    // The inputs are spread over 57 exponents, 2^-27 to 2^29, on purpose:
    // the archive is compiled by nvcc with `--fmad` at its default TRUE and
    // NVRTC compiles with `--fmad=false`, so these two arms are NOT obliged
    // to agree by construction. They do -- the bf16 round absorbs the ulp on
    // every shape measured -- but that is a MEASUREMENT, and it is why the
    // fixtures do not use benign data: `env-audit` measured benign data
    // showing 0 difference everywhere while wide exponents exposed five real
    // bytes at three shapes.
    "mlp::chunked_geglu_tanh_bf16",
    "mlp::chunked_situ_bf16",
    "mlp::chunked_swiglu_clamp_bf16",
    "mlp::gaussian_topk_bf16",
    "mlp::geglu_tanh_bf16",
    "mlp::gpt_oss_glu_bf16",
    "mlp::relu2_bf16",
    "mlp::sigmoid_dot_scalar_gate_add_bf16",
    "mlp::situ_bf16",
    "mlp::swiglu_bf16",
    "mlp::swiglu_clamp_bf16",
    // BATCH 3 -- `norm/rmsnorm`, the unit whose GRID has been wrong before.
    //
    // Five of the unit's eight rows. `rmsnorm_bf16` and `rmsnorm_strided_bf16`
    // stay because `norm/rmsnorm.cu` still calls them from C++ —
    // `rmsnorm_bf16_with_fp16` forwarding into both — and `add_bias_bf16` is
    // held by `gemm.cpp`.
    //
    // This sentence said "and `vision/gemma4_vision.cu`, six calls in the
    // vision tower alone" when the batch landed, which was true then and was
    // false eight hours later: `towers-move` converted the tower to direct
    // launches, inlined `rmsnorm_strided_bf16`'s body verbatim, and moved all
    // three towers out of the archive. The two rows are still held, so the
    // batch is unaffected — but the REASON changed under a comment that had
    // no way to notice, which is why the count these rows belong to is
    // re-derived from the sources by `dispatch_countdown` on every run rather
    // than read from here.
    //
    // This unit is where the failures this migration actually has were
    // measured: §21's 16x grid wrote 491,520 values PAST the rectangle with
    // ZERO differing inside it, and five rows here once moved 35,266-61,757
    // bytes carrying the same 32,768 values. Both are shape errors that no
    // per-row text comparison can see, so the fixtures give the two
    // `RowsPerHead` rows a fourth shape with `per_head_dim = Some(64)` and a
    // width of 320 -- five heads, not a power of two -- which is the axis the
    // rule splits on and the one every `Rms` shape collapses. 18 shapes, 0
    // differing bytes.
    "norm::residual_add_rmsnorm_bf16",
    "norm::rmsnorm_gated_bf16",
    "norm::rmsnorm_gemma_bf16",
    "norm::rmsnorm_no_scale_bf16",
    "norm::rmsnorm_residual_add_bf16",
    // BATCH 4 -- five small units, and the first rows this migration HELD
    // BACK on measured evidence rather than on a C++ caller. §40.
    //
    // `ssm/gated_delta_net_prep` (3 of 4), `attn/softcap` (1 of 1),
    // `attn/attn_sink` (2 of 2), `attn/attn_res` (1 of 1), `norm/altup`
    // (2 of 2). 33 shapes fired both ways, 0 differing bytes.
    //
    // TWO ROWS OF `attn/head_dim_pad` ARE ABSENT AND THEY ARE THE POINT.
    // `pad_head_dim_bf16` and `strip_head_dim_bf16` pass their head count to
    // the kernel as `Width(In(0)) / CtxNonZero("head_dim")`, and the
    // archive's launcher builds its grid from that same argument --
    // `dim3(num_heads, num_tokens)` in `attn/head_dim_pad.cu`. The JIT's grid
    // comes from `LaunchRule::PerHead`, which reads `Dims::kv_heads`, i.e.
    // `ctx.num_kv_heads` -- a field no part of the row mentions. At 12 heads
    // of 64 with `num_kv_heads = 6` the two arms differ in 6,100 of 12,544
    // bytes and 4,588 of 9,472: the JIT writes half the rectangle. They agree
    // exactly when the context happens to equal the quotient. That is a
    // `LaunchRule` fix, not a routing decision, and until it lands these two
    // rows keep their launcher.
    //
    // `ssm::repeat_interleave_heads_fp32` is the fourth prep row and is also
    // absent: its arm guards on `gdn.is_some() && join_out(spec, 0, frame,
    // resolver)`, so proving it needs a `GdnCtx` and an output resolved as a
    // REGION, neither of which the parity fixtures can state yet.
    "attn::attention_sink_rescale_bf16",
    "attn::attn_res_blend_bf16",
    "attn::logit_softcap_bf16",
    // THREE ROWS BELOW WERE PROVED AGAINST POISON, AND THE WINDOW HAS CLOSED.
    //
    // `attn::lse_log2_to_ln`, `mlp::gaussian_topk_bf16` and
    // `attn::logit_softcap_bf16` are in-place transforms that name no
    // `Source::In` at all: the operand each one reads IS its output. Their
    // parity fixtures left that output poison-filled, so both arms agreed
    // exactly about what the kernel does to `0xa5, 0xc4, 0xe3, ...` read as
    // floats, and all three were routed on that agreement.
    //
    // The harness's permutation control found them -- rotating every input
    // moved nothing, because there were no inputs -- and by then all three
    // archive launchers had been deleted, the deletions these very routings
    // authorised. Un-routing any of them now fails to COMPILE:
    // `lse_log2_to_ln` is no longer a member of
    // `pie_cuda_driver::kernels::attn`. Their fixtures are corrected and
    // their digests re-measured from the JIT arm alone, and the harness
    // counts them under `recalled`, never `compared`.
    //
    // All three are almost certainly right -- each is a few lines of
    // arithmetic whose text §8 proves byte-identical -- but this file should
    // not claim a comparison that no longer exists. It is the sharpest thing
    // this session measured about why the harness had to come first, and it
    // is now prevented rather than described:
    // `jit_parity.rs::every_case_names_a_row_with_both_implementations`
    // refuses a fixture that leaves an in-place row's operand poison.
    //
    // `attn::lse_log2_to_ln`, and the window for re-proving it HAS CLOSED.
    //
    // Routed in batch 4 on a fixture that stated an input slot the row does
    // not have: its only operand is `lse: F32sMut <- Source::Out(0)`, an
    // in-place transform with no `In` at all, and the fixture left that
    // output poison-filled. Both arms therefore agreed byte-for-byte about
    // what log2->ln does to `0xa5, 0xc4, 0xe3, ...` reinterpreted as f32.
    // The parity harness's permutation control found it -- rotating every
    // input moved nothing, because there were no inputs -- and by then the
    // archive's launcher had already been deleted, the deletion this row's
    // own routing authorised. Un-routing it now fails to COMPILE:
    // `lse_log2_to_ln` is no longer a member of
    // `pie_cuda_driver::kernels::attn`.
    //
    // So its corrected fixture is recorded as a JIT-only digest and the
    // harness counts it under `recalled`, never `compared`. The row is
    // almost certainly right -- it is four lines of arithmetic and its text
    // is proved byte-identical by §8 -- but this file should not claim a
    // comparison that no longer exists. It is the clearest demonstration
    // this session produced of why the harness had to come first.
    "attn::lse_log2_to_ln",
    "norm::altup_correct_bf16",
    "norm::altup_predict_bf16",
    "ssm::bf16_to_fp32",
    "ssm::fp32_to_bf16",
    "ssm::l2norm_scale_bf16_to_fp32",
    // BATCH 5 -- two rows, and the third one is why the batch is two.
    //
    // `attn/kimi_mla` states three rows; `moe/topk_sigmoid` states one. All
    // four are armed and hosted. The two below agreed with the archive in
    // every byte of every allocation at three shapes each, including
    // `topk_sigmoid` at `experts = 4, k = 2` where the routing table is
    // narrower than a warp.
    //
    // `attn::kimi_split_q_b_bf16` is HELD, and the harness is the only reason
    // anyone knows to hold it. The row's extent is `total <- InElements(0)`
    // -- the input's element count -- but `LaunchRule::Elementwise` sizes the
    // grid from `width_of(b, n_in + 0)`, the FIRST OUTPUT's width. This row
    // splits a q projection into `q_nope` and `q_pe`, so its input is
    // strictly wider than out 0 by construction, and the JIT under-covers by
    // exactly that ratio: at 6 rows of 8 heads (nope 128, rope 64) it wrote 4
    // of 6 rows, leaving 4,082 of 12,544 bytes of `q_nope` and 2,041 of 6,400
    // of `q_pe` still holding the poison fill. The archive's launcher reads
    // the same `total` the row states, which is why the two disagree.
    //
    // It is worth saying how nearly this was missed. The row's third shape --
    // 1 row, 1 head -- agrees in every byte, because 200 elements round up
    // into a 256-thread block that covers all 255 the kernel wanted. One
    // shape would have certified it. The divergence is now an INVERTED
    // assertion in `driver-cuda/tests/jit_parity.rs`: if those shapes ever
    // agree, the test fails and says to route the row.
    "attn::kimi_split_kv_a_norm_bf16",
    "moe::topk_sigmoid_bf16",
    // BATCH 6 -- one row of two, and the one held is held for a reason no
    // gate in this crate could have raised.
    //
    // `moe/topk_softmax` states two armed rows. This one agreed with the
    // archive in every byte at 8x64, 3x4 (k = 1, narrower than a warp) and
    // 1x129. It is also the row that answers whether `Bool` and `I32` can
    // spell the same operand: the AOT row declares `normalize: Bool` and the
    // device row `I32`, because the C++ host function narrows a `bool` with
    // `? 1 : 0` and the `__global__` takes an `int`. Both put the same four
    // bytes in the cell, and only comparing the RESULT could say so.
    //
    // `moe::topk_softmax_bf16` is HELD, and not because anything is wrong
    // with it. The ahead-of-time symbol is a LADDER: its launcher picks
    // between five warp rungs and the block form on `num_experts` and `K` at
    // run time, and the device row names the block form -- the one thing a
    // row cannot state, as this family's own header says. So the two arms
    // fire DIFFERENT `__global__`s at 64 and 129 experts and fold the same
    // logits in different orders: 53 of 512 bytes of `topk_idx` differ at
    // 8x64 and 4 of 276 at 1x129, while `topk_w` agrees in every byte at
    // every shape.
    //
    // The binding is right, and here is the measurement that says so: with
    // `PIE_TOPK_WARP=0` -- the documented switch that forces the launcher to
    // the block form -- all three shapes agree in EVERY byte. Routing this
    // row would therefore not be a migration but a behaviour change, and the
    // countdown is not worth a silently different expert ranking.
    "moe::topk_sigmoid_bias_fp32",
    // BATCH 7 -- EVERY REMAINING ARMED ROW, IN ONE COMMIT.
    //
    // 34 rows carried a generated arm; 30 are here. The four absent are the
    // four this session already measured divergent, and no re-measurement was
    // done to leave them out -- the numbers were in hand:
    // `pad_head_dim_bf16` 6,100/12,544 and `strip_head_dim_bf16` 4,588/9,472
    // (`LaunchRule::PerHead` reads `ctx.num_kv_heads`, which no part of the
    // row mentions); `kimi_split_q_b_bf16` 4,082/12,544 + 2,041/6,400
    // unwritten (`Elementwise` grids on out 0, the extent is `InElements(0)`);
    // `topk_softmax_bf16` 53/512 index bytes, a run-time rung ladder the row
    // can only name one rung of. All four are `LaunchRule` work, not routing
    // decisions.
    //
    // The other 30 are proven by the build, which is what the four compulsory
    // compiles check: `routed_rows_have_an_arm` for the arm, the `bridge`
    // link for the shim edge, and `cargo check` toolkit-free.
    "attn::qkv_packed_qk_norm_rope_vnorm_write_kv_bf16",
    "attn::attention_naive_paged",
    "mlp::chunked_swiglu_bf16",
    "moe::topk_sqrtsoftplus_bf16",
    "moe::token_batched_weighted_sum_bf16",
    "moe::token_batched_weighted_sum_add_bf16",
    "moe::gather_moe_aligned_inputs_bf16",
    "moe::moe_align_decode",
    "norm::attn_sink_correction_bf16",
    "norm::per_head_rmsnorm_bf16",
    "norm::hc_expand_bf16",
    "quant::mxfp4_moe_gate_up_decode_bf16",
    "quant::mxfp4_moe_down_decode_bf16",
    "quant::wna16_gate_up_decode_bf16",
    "quant::wna16_down_decode_bf16",
    "rope::rope_standard_table",
    "rope::rope_partial_bf16",
    "rope::qk_rmsnorm_rope_bf16",
    "ssm::repeat_interleave_heads_fp32",
    "ssm::recurrent_gated_delta_step_batched",
    "ssm::recurrent_gated_delta_step_batched_state_bf16",
    "ssm::recurrent_gated_delta_step_batched_gqa",
    "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa",
    "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16",
    "ssm::nemotron_prepare_mamba_params",
    "ssm::nemotron_prepare_mamba_dt_da",
    "ssm::zamba_rmsnorm_gated_bf16",
    "ssm::causal_conv1d_update_batched_bf16",
    "ssm::kda_gate_beta_bf16",
    "ssm::kda_o_norm_gated_bf16",
];

/// [`ELEMENTWISE`]'s rows that [`JIT_DISPATCHED`] names, as the emitters take
/// them.
///
/// A function rather than a second static, so the two lists cannot drift: the
/// symbols are stated once and this is the join.
///
/// # The join is over every unit, and the narrow version was the whole gate
///
/// This filtered `ALTUP_AUX.iter().chain(ELEMENTWISE)`, which is two units of
/// fifteen families. Both build scripts call it — `driver-cuda/build.rs:324`
/// and `kernels-cuda/build.rs:215` — and it decides two things:
/// [`crate::abi::emit_c_shim`] SKIPS a row it names, and
/// [`crate::abi::emit_rust_dispatch`] reads that row's device operands
/// instead of its AOT ones. So a migrated family symbol added to
/// [`JIT_DISPATCHED`] kept its shim entry and kept being routed to it; the
/// name would have been inert, and the only thing standing between that and
/// a silent no-op was `assert_eq!(jit_dispatched().len(),
/// JIT_DISPATCHED.len())` in [`every_dispatched_symbol_has_a_row`].
///
/// That assertion is why this was a loud failure rather than a quiet one,
/// and it is worth keeping for exactly that reason: **the list and the join
/// must have the same length, or one of them is not what it claims.**
#[must_use]
pub fn jit_dispatched() -> Vec<&'static DeviceKernel> {
    crate::unit::rows()
        .filter(|d| JIT_DISPATCHED.contains(&d.sig.symbol))
        .collect()
}

/// A multi-argument instantiation, spelled in a row's `elem`.
///
/// # The blocker that was not one
///
/// Two migrating agents independently reported the same ceiling: *"`DeviceKernel`
/// always emits `path<elem>`, so a kernel with more than one template argument
/// cannot be rowed at all."* Thirty-seven kernels were parked behind it —
/// `rmsnorm<T, int BLOCK>` (8), `gated_delta_net`'s recurrence forms (13),
/// `causal_conv1d<T, bool SILU>`, `kimi_mla<T, int BLOCK_DIM>` and the rest.
///
/// It is not a ceiling. [`DeviceKernel::elem`] is a `&'static str` that
/// [`DeviceKernel::instantiation`] pastes between angle brackets, so it can
/// carry an argument LIST as easily as a type — and `nvrtcAddNameExpression`
/// accepts the result, because it parses C++ rather than a name. Measured on
/// an L40S, NVRTC 13.0:
///
/// ```text
/// ::pie_cuda_driver::kernels::probe::scaled<float, 128>
///   -> _ZN15pie_cuda_driver7kernels5probe6scaledIfLi128EEEvPT_i
/// ```
///
/// So a row for `rmsnorm<T, BLOCK>` states `elem: "device::bf16, 256"` and
/// nothing about the type, the table or the compile path changes. This
/// function exists to make that legible at the call site rather than leaving
/// a reader to wonder whether the comma is a typo.
///
/// # Where the boundary actually is — two slots, two different rules
///
/// [`DeviceKernel::instantiation`] qualifies the `elem` string **once, at the
/// front**:
///
/// ```text
/// ::pie_cuda_driver::kernels::{template_path}<::pie_cuda_driver::kernels::{elem}>
/// ```
///
/// so the first argument and every later one obey different rules, and the
/// difference is not intuitive in either direction. Measured on an L40S with
/// NVRTC 13.0 — `examples/argform_probe.rs` reproduces all eleven:
///
/// ```text
/// "device::bf16"                         OK       scaled<bf16, 256>   (default applied)
/// "device::bf16, 256"                    OK       identical mangling
/// "device::bf16, true"                   OK       flagged<bf16, true>
/// "256"                                  REFUSED  expected an identifier
/// "device::kBlock256"                    OK       sized<256>
/// "device::false_type::value, false"     OK       flags<false, false>
/// "device::bf16, device::kBlock256"      REFUSED  name followed by "::" must be a class or namespace
/// "device::bf16, ::pie_...::kBlock256"   OK       sizedT<bf16, 256>
/// "true"                                 REFUSED  expected an identifier
/// "device::true_type::value"             OK       oneflag<true>     ...ILb1EEEvPi
/// "device::false_type::value"            OK       oneflag<false>    ...ILb0EEEvPi
/// ```
///
/// **Slot 1 is prefixed, so it must RESOLVE inside
/// `::pie_cuda_driver::kernels::` — but it need not be a type.** A `constexpr`
/// variable, a `static constexpr` member or a functional cast all work, which
/// is why `rope`'s `rotate<bool kWriteKv, bool kHnd>` IS nameable:
/// `"device::false_type::value, false"`. The prelude already ships
/// `device::true_type`/`false_type` and `device::i32` for exactly this.
///
/// **Slots 2 and after are NOT prefixed, so they resolve at global scope.**
/// Bare literals work; a NAME must be spelled in full from `::`. The natural
/// spelling — repeating slot 1's style — is the one that fails.
///
/// The last three rows are the ones the nine `template <bool>` rows turn on,
/// and they were added because the first eight did not settle the question.
/// `write_kv<HND_LAYOUT>` and its five siblings take ONE template parameter
/// and it is the flag, so the flag lands in slot 1 — the prefixed one — and
/// the spelling a reader would reach for first, `elem: "true"`, is the one
/// form of the eight-case table that was already known to fail. It fails the
/// same way and for the same reason. `device::true_type::value` and
/// `device::false_type::value` are the spellings that work, they mangle to
/// `Lb1` and `Lb0` as they must, and they are two DISTINCT symbols, which is
/// what makes the two arms nameable at all.
///
/// This entry is a correction. An earlier version of it, written from the
/// first two rows of that table alone, claimed that slot 1 must be a type and
/// that "everything after it may be anything C++ accepts". Both halves were
/// wrong, and the agent that found it had gone and asked the compiler about
/// the cases the first probe had not covered. The lesson is the one §17.6
/// already records, applied to its own answer: **a boundary is worth probing
/// past the first case that confirms it.**
///
/// # What it still does not make safe
///
/// A non-type argument is a value the kernel is compiled AGAINST, so it is
/// part of the contract in a way an element type is not: `rmsnorm<T, 256>` and
/// `rmsnorm<T, 512>` reassociate a reduction differently, and a row that picks
/// the wrong one produces a plausible number. The value must be the one the
/// ahead-of-time launcher passed — read the `<<<>>>` and the template's
/// default, and cite both.
///
#[must_use]
pub fn args(elem: &str, rest: &[&str]) -> String {
    let mut out = elem.to_string();
    for arg in rest {
        out.push_str(", ");
        out.push_str(arg);
    }
    out
}

/// A fact about one bound value that a [`Term`] is allowed to read.
///
/// **This enum is where the line is drawn, and it is drawn in the type system
/// rather than in a convention.** A fire-time choice is only defensible if the
/// thing it reads is already in the caller's hand: an ahead-of-time build
/// pushed `rmsnorm_vec8_ok` into a host `if` because it had to fix its
/// instantiations months earlier, and the fire has the pointers and the
/// strides right there. What the fire does NOT have is anything the device
/// wrote. A predicate over a token count in device memory would have to
/// `cudaMemcpy` it back, and a `cudaMemcpy` on the launch path is a
/// synchronisation between two kernels that were meant to be enqueued. The
/// arithmetic is not close: `tests/specialise.rs` measured this crate's one
/// specialisation buying between 0.29 and 1.86 us of kernel, and a
/// round-trip through a device-side count costs a pipeline drain of tens of
/// microseconds — once per layer per token.
///
/// So there is no variant for it. [`Address`](Fact::Address) is a `u64` and
/// not a pointer, so nothing downstream can dereference what a term reads;
/// [`Int`](Fact::Int) is the operand's own value, already marshalled;
/// [`Bool`](Fact::Bool) is a host FLAG, and it is a variant of its own for the
/// reason below; [`Opaque`](Fact::Opaque) is every other kind, and a term that
/// names one faults rather than guessing. A predicate that wanted a
/// device-side count could not be spelled here at all, which is the refusal
/// stated as a type.
///
/// # Why a flag is not an [`Int`](Fact::Int), which is the shorter change
///
/// Nine rows are blocked on a `bool` operand choosing between two arms of a
/// `template <bool>` — `attn::device::write_kv<HND_LAYOUT>` and its five
/// siblings in `attn/kv_paged.cuh` — and the obvious unblocking is one line:
/// widen `Fact::Int` to carry `0` and `1` and reuse the [`Term`]s that are
/// already here. It was rejected, and the argument is the one §18.4 of
/// `new-horizon.md` makes about the whole design.
///
/// **`Fact::Int` does not actually reach the case.** Neither existing term is
/// an equality: [`Term::Aligned`] wants an address and [`Term::Multiple`]
/// wants a divisor, and *"the flag is true"* is neither. So a flag needs a
/// new term whichever fact carries it, and the saving the `Int` spelling
/// promises is zero.
///
/// **What `Fact::Int` does buy is arithmetic on a flag, and that is a
/// liability.** With a bool arriving as `Int(0 | 1)`,
/// `Multiple { operand: flag, of: 2 }` becomes a well-formed clause that is
/// true exactly when the flag is FALSE, and `Multiple { of: 1 }` a
/// well-formed clause that is always true. Neither is a typo a reader would
/// catch beside the C++ it claims to reproduce, and §18.4's finding is
/// precisely that the fatal shape is a predicate that is **well formed and
/// wrong** — the one that dropped a clause and moved 96 of 6 144 cases onto
/// the wrong arm, invisible to every check inside the crate.
///
/// With `Fact::Bool`, dividing a flag by 8 is a [`Fault::Kind`] instead of an
/// answer, and testing a stride for truth is the same fault the other way.
/// That is the SAME move this enum already makes for a device-side value,
/// applied one level in: **a number is not a flag, so a predicate that treats
/// one as the other is unspellable rather than merely discouraged.**
///
/// The third reason is that `Int` would change an existing answer. A
/// [`Term::Multiple`] naming a `Ty::Bool` operand faults today, because the
/// bool arrives as [`Opaque`](Fact::Opaque); widening `Int` turns that
/// refusal into an answer, silently, for every term already written. Turning
/// a refusal into an answer is the direction this crate does not move in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fact {
    /// A device address, as a number. Never dereferenced, and it cannot be:
    /// the pointer was cast away at the boundary this enum is.
    Address(u64),
    /// An integer operand's value — a width, a stride, a count the HOST
    /// supplied.
    Int(i64),
    /// A host flag's value — a `Ty::Bool` operand, which in this tree is
    /// always a decision the host already made: a KV cache's `hnd_layout`, a
    /// router's `renormalize`, a rotation's `interleaved`.
    ///
    /// Readable by [`Term::Is`] and by nothing else. See this enum's header
    /// for why it is not an [`Int`](Fact::Int) of 0 and 1.
    Bool(bool),
    /// Any other kind of bound value. Present so the mapping from an argument
    /// list to facts is TOTAL: a term naming one of these faults, and a fault
    /// is a refusal.
    Opaque,
}

/// One clause of a selection predicate, over one operand.
///
/// Data, not a function pointer, and the difference is the whole reason a
/// reader can check this against the C++. A `fn(&[ArgValue]) -> bool` would
/// be correct-looking and unauditable: nothing could enumerate what it reads,
/// nothing could sweep it, and nothing could stop a later edit from putting a
/// `cudaMemcpy` in it. Three variants over [`Fact`] can be printed beside
/// `rmsnorm_vec8_ok`'s six clauses — or beside `kv_paged.cu:51`'s one — and
/// compared clause for clause, which is what `tests/specialise.rs` does.
///
/// Each variant reads exactly one [`Fact`] kind and faults on the others, so
/// the vocabulary is closed in both directions: there is no term that reads
/// an address as a number, and none that reads a flag as one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Term {
    /// The operand is a pointer whose address is a multiple of `bytes`.
    ///
    /// `(reinterpret_cast<uintptr_t>(p) & 15u) == 0` is how `rmsnorm.cu`
    /// spells the 16-byte case; a mask and a modulus agree for every power of
    /// two, and `bytes` is checked to be one by [`Specialisation::agrees`] so
    /// the two spellings cannot part company on a value nobody swept.
    Aligned {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// The alignment in bytes — a power of two.
        bytes: u64,
    },
    /// The operand is an integer that divides evenly by `of`.
    Multiple {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// The divisor — `hidden % 8 == 0` and its two stride twins.
        of: i64,
    },
    /// The operand is a host flag with this value.
    ///
    /// The clause the `template <bool>` kernels need, and the only one that
    /// reads a [`Fact::Bool`]. `kv_paged.cu:51` is the C++ it reproduces:
    ///
    /// ```text
    /// if (hnd_layout) { write_kv<true> <<<...>>>(...); }
    /// else            { write_kv<false><<<...>>>(...); }
    /// ```
    ///
    /// An equality and not a truth test — `Is { value: false }` rather than
    /// a `Not` around `Is { value: true }` — because both arms of that `if`
    /// are instantiations a fire must be able to NAME, and an arm spelled as
    /// the negation of another reads as the fallback it is not. The two arms
    /// are peers; the data says so.
    ///
    /// [`Specialisation::agrees`] refuses one that names an operand which is
    /// not `Ty::Bool`, so a clause testing a stride for truth is a build-time
    /// refusal rather than a fire-time [`Fault`].
    Is {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// The value that selects this arm.
        value: bool,
    },
    /// The operand is a pointer the fire published, or one it did not.
    ///
    /// The clause a `template <bool USE_TABLE>` kernel needs when the host
    /// picks its arm by testing a POINTER rather than by reading a flag.
    /// `attn/qkv_fused.cu:56` and `:100` are the C++ it reproduces:
    ///
    /// ```text
    /// if (rope_table != nullptr) { ..._warp<HEAD_DIM, true ><<<...>>>(...); }
    /// else                       { ..._warp<HEAD_DIM, false><<<...>>>(...); }
    /// ```
    ///
    /// `value: true` is *published* — a non-null address. `value: false` is
    /// *absent*, which for an operand the row declares nullable is a state the
    /// binder can actually produce.
    ///
    /// # Why [`Term::Aligned`] is not this clause, measured rather than
    /// argued
    ///
    /// **`Aligned` HOLDS OF ADDRESS ZERO.** `0 % 16 == 0` for every
    /// alignment, so an arm that said `Aligned { operand: rope_table, bytes:
    /// 16 }` where the launcher says `!= nullptr` would take the TABLE arm
    /// for a fire that published no table — and `qkv_fused.cuh` records what
    /// that costs: `false` computes the angle with `powf`/`__sincosf` and
    /// `true` reads a precomputed `[max_pos, head_dim]` table, *"different
    /// numbers — close, not equal"*. §18's measurement is of exactly this
    /// species: a wrong specialisation arm was 99.83% of the right answer, 7
    /// of 4,095 values moved and 0 of the 4,088 actually written. Here the
    /// wrong arm is worse than that — it dereferences null — but nothing
    /// about the CLAUSE tells you so, which is why the clause has to be its
    /// own thing.
    ///
    /// # Why it does not need a [`Fact`] of its own
    ///
    /// It reads [`Fact::Address`], which already carries the number and is
    /// already produced for exactly the operand kinds a null test is
    /// meaningful over. A `Fact::Present(bool)` would be the
    /// [`Fact::Int`]-for-a-flag mistake in the other direction: it would make
    /// *"the width is present"* and *"the flag is present"* well-formed
    /// clauses, and §21.14's whole argument is that a vocabulary is wrong
    /// when it lets someone spell something meaningless. `Int` was refused
    /// because it made `Multiple { operand: flag, of: 2 }` — a clause that is
    /// true exactly when a flag is false — spellable. `Present` as a FACT
    /// would be that mistake generalised; `Present` as a TERM over
    /// `Fact::Address` is not, because every non-pointer operand faults.
    ///
    /// So this term faults on an `Int`, a `Bool` and an `Opaque` the same way
    /// [`Term::Aligned`] does, and [`Specialisation::agrees`] adds two
    /// build-time refusals: one naming a SCALAR operand (a null test over a
    /// width could never be taken), and one naming an operand the row does
    /// not declare NULLABLE (the binder cannot produce a null there, so the
    /// `true` clause holds for every fire that reaches it and the `false` arm
    /// is an instantiation that compiles and never runs). An arm that can
    /// never be taken is worse than a missing one — it reads as a covered
    /// case.
    ///
    /// # Why it carries a `value` rather than being a bare `NonNull`
    ///
    /// The same reason [`Term::Is`] is an equality and not a truth test:
    /// **both arms of that `if` are instantiations a fire must be able to
    /// NAME.** A bare `NonNull` would leave the `else` branch as whatever the
    /// base row happens to be, which is a correspondence a reader has to
    /// reconstruct from an `elem` string — and it would leave the launcher's
    /// `else` line with nowhere to be cited. With a `value` the two arms are
    /// peers, each carries its own [`Arm::because`], and
    /// `tests/specialise.rs` can check that the `#norope` arm names
    /// `<..., false>` the same way it checks `#nhd` names `<false>`.
    Present {
        /// Index into the BASE row's operand list.
        operand: usize,
        /// `true` selects the arm for a published pointer, `false` the arm
        /// for an absent one.
        value: bool,
    },
}

impl Term {
    /// Which operand of the base row this clause reads.
    #[must_use]
    pub const fn operand(&self) -> usize {
        match self {
            Term::Aligned { operand, .. }
            | Term::Multiple { operand, .. }
            | Term::Is { operand, .. }
            | Term::Present { operand, .. } => *operand,
        }
    }

    /// Whether this clause holds over the facts a fire supplies.
    ///
    /// # Errors
    ///
    /// [`Fault`] when the clause names an operand the list does not have, or
    /// names one whose [`Fact`] is not the kind it tests. Both are drift
    /// between a term and the row it is written against — checked without a
    /// GPU by [`Specialisation::agrees`], and refused rather than defaulted
    /// here, because a term that cannot read what it names is not a
    /// predicate that answered `false`.
    pub fn holds(&self, facts: &[Fact]) -> Result<bool, Fault> {
        let at = self.operand();
        let Some(fact) = facts.get(at) else {
            return Err(Fault::Range { operand: at, arity: facts.len() });
        };
        match (self, fact) {
            (Term::Aligned { bytes, .. }, Fact::Address(address)) => Ok(address % bytes == 0),
            (Term::Multiple { of, .. }, Fact::Int(value)) => Ok(value % of == 0),
            (Term::Is { value, .. }, Fact::Bool(flag)) => Ok(value == flag),
            // The one clause `Aligned` cannot express: `0 % bytes == 0` for
            // every alignment, so an alignment term answers TRUE for the
            // absent pointer this term answers FALSE for.
            (Term::Present { value, .. }, Fact::Address(address)) => Ok(*value == (*address != 0)),
            (Term::Aligned { .. }, _) => Err(Fault::Kind { operand: at, wanted: "an address" }),
            (Term::Multiple { .. }, _) => Err(Fault::Kind { operand: at, wanted: "an integer" }),
            (Term::Is { .. }, _) => Err(Fault::Kind { operand: at, wanted: "a flag" }),
            (Term::Present { .. }, _) => Err(Fault::Kind { operand: at, wanted: "an address" }),
        }
    }
}

/// Why a [`Term`] could not be evaluated at all.
///
/// Distinct from "the predicate was false", and it has to be: a false
/// predicate fires the base row, which is correct and slower, while a term
/// that cannot read its operand means the specialisation is not the decision
/// it claims to be. Firing the base on a fault would be the right number for
/// the wrong reason, and the next edit that made the fault permanent would
/// look exactly like a specialisation that never applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// The term names operand `operand` and the row has `arity` of them.
    Range {
        /// The index the term named.
        operand: usize,
        /// How many operands were bound.
        arity: usize,
    },
    /// The operand is bound to a [`Fact`] the term cannot read.
    Kind {
        /// The index the term named.
        operand: usize,
        /// What the term needed there.
        wanted: &'static str,
    },
}

impl std::fmt::Display for Fault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fault::Range { operand, arity } => {
                write!(f, "a term reads operand {operand} of a row with {arity}")
            }
            Fault::Kind { operand, wanted } => {
                write!(f, "a term reads operand {operand} and wanted {wanted} there")
            }
        }
    }
}

/// Where one of a variant's arguments comes from.
///
/// A specialisation is two instantiations of two DIFFERENT templates, and
/// their parameter lists need not agree: `rmsnorm<T, BLOCK>` takes seven
/// arguments and `rmsnorm_vec8<BLOCK, WPO, EMIT_FP16>` takes eight, with the
/// optional fp16 output third. `rmsnorm.cu` spells that correspondence twice,
/// eight lines apart, as two `<<<>>>` argument lists a reader has to diff by
/// eye. Here it is one line of data, and [`Specialisation::agrees`] checks
/// that every [`From`](Take::From) lands on an operand of the same [`Ty`] —
/// so the reshape cannot silently swap a stride for a width, which is the one
/// mistake that produces a launch that runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Take {
    /// The base row's operand at this index, verbatim.
    From(usize),
    /// A null pointer, for a parameter the variant declares and this arm does
    /// not use — `y_fp16` under `EMIT_FP16 = false`, which the kernel reads
    /// only inside an `if constexpr` that is not compiled.
    Null,
}

/// One specialised instantiation, and the predicate that chooses it.
pub struct Arm {
    /// What this arm is called in a diagnosis and in the audit. Short, and
    /// not a symbol: the symbol is `row.sig.symbol`.
    pub name: &'static str,
    /// The clauses, ANDed. Empty would mean "always", which
    /// [`Specialisation::agrees`] refuses — an arm that always applies is not
    /// a specialisation, it is a different base row written in the wrong
    /// place.
    pub when: &'static [Term],
    /// The instantiation to fire instead. A row of the same unit, so the
    /// existing per-unit compile produces it and no second cache slot exists.
    pub row: &'static DeviceKernel,
    /// The variant's argument list, in the variant's order, over the base's
    /// values.
    pub take: &'static [Take],
    /// The host code this arm reproduces, cited so the two can be compared.
    ///
    /// Prose, and load-bearing prose: the failure this whole design has to
    /// answer for is a predicate that DISAGREES with the C++ and therefore
    /// produces a fast wrong answer. A reader checks the agreement by reading
    /// this line, opening the file it names, and comparing the clauses; a
    /// test checks it by pinning the C++ text and sweeping both sides.
    pub because: &'static str,
}

/// A base row and the arms a fire may choose instead of it.
///
/// # What a specialisation may and may not be
///
/// A row states a contract. Specialisation must not turn that into "a
/// contract, and then a decision" — so an arm is constrained to the one thing
/// that leaves the contract alone: it fires a DIFFERENT INSTANTIATION at the
/// SAME launch geometry with the SAME values, chosen by a predicate over
/// facts the caller already handed in. [`Specialisation::agrees`] enforces
/// every clause of that sentence without a GPU:
///
/// * the arm's row states the base's [`kernels::LaunchRule`], so the grid,
///   the block and the shared memory are the base's and the specialisation
///   cannot smuggle a second decision in as a geometry;
/// * the arm's row lives in the base's UNIT, so one compile produces both and
///   the module cache stays one `OnceLock` per unit;
/// * every argument the variant takes is the base's own value at a matching
///   [`Ty`], or an explicit null on a nullable pointer;
/// * every term reads an operand that exists and is of the kind it tests;
/// * a flag that no arm forwards to a kernel is covered by an arm for each of
///   its values, so a fire cannot fall through to a base row that would bind
///   an argument its instantiation does not declare.
///
/// What is left for a reader to check by hand is the only thing a machine in
/// this crate cannot: that `when` says what the C++ predicate says. That is
/// [`Arm::because`]'s job, and `tests/specialise.rs` pins the C++ text so the
/// day it changes is a failing test rather than a silent divergence.
pub struct Specialisation {
    /// The symbol a fire names — the row whose contract this is.
    pub base: &'static str,
    /// The arms, in order. The FIRST whose `when` holds is chosen, so a
    /// narrower arm must precede a wider one; with one arm the order is not a
    /// decision and is documented as such rather than left implicit.
    pub arms: &'static [Arm],
}

impl Specialisation {
    /// The first arm whose predicate holds over `facts`, or `None` for the
    /// base row.
    ///
    /// # Errors
    ///
    /// [`Fault`], from the first term that could not be read at all.
    pub fn choose(&self, facts: &[Fact]) -> Result<Option<&'static Arm>, Fault> {
        for arm in self.arms {
            let mut all = true;
            for term in arm.when {
                if !term.holds(facts)? {
                    all = false;
                    // No `break`: every term is evaluated on every arm, so a
                    // term that names a bad operand faults whether or not an
                    // earlier clause already answered `false`. A predicate
                    // that is only checked when the values happen to reach it
                    // is a predicate that is checked in production.
                }
            }
            if all {
                return Ok(Some(arm));
            }
        }
        Ok(None)
    }

    /// Everything about this specialisation a machine can check, checked.
    ///
    /// Called by `tests/specialise.rs` on every entry of [`SPECIALISED`], and
    /// callable with no device, no cubin and no driver — which is the point.
    /// The failure it exists to prevent is the one the whole idea is on
    /// trial for: an arm that fires a kernel taking different arguments, at a
    /// different width, out of a different compile, and reports success.
    ///
    /// # Errors
    ///
    /// A sentence naming the arm and what does not line up.
    pub fn agrees(&self) -> Result<(), String> {
        let Some((_, unit)) = crate::unit::unit_of(self.base) else {
            return Err(format!("`{}` is specialised and no unit hosts it", self.base));
        };
        let base = unit.row(self.base).ok_or_else(|| format!("`{}` has no row", self.base))?.sig;
        if self.arms.is_empty() {
            return Err(format!("`{}` states a specialisation with no arms", self.base));
        }
        for arm in self.arms {
            let variant = arm.row.sig;
            let at = format!("`{}` arm `{}`", self.base, arm.name);
            if !unit.hosts(variant.symbol) {
                return Err(format!(
                    "{at} fires `{}`, which unit `{}` does not compile — a second unit \
                     would be a second cubin and a second first-fire stall",
                    variant.symbol, unit.name
                ));
            }
            if variant.launch != base.launch {
                return Err(format!(
                    "{at} states {:?} where the base states {:?}; a specialisation chooses an \
                     instantiation, not a geometry",
                    variant.launch, base.launch
                ));
            }
            if arm.when.is_empty() {
                return Err(format!("{at} applies always, which is not a specialisation"));
            }
            if arm.take.len() != variant.operands.len() {
                return Err(format!(
                    "{at} takes {} arguments and `{}` declares {}",
                    arm.take.len(),
                    variant.symbol,
                    variant.operands.len()
                ));
            }
            for (slot, take) in arm.take.iter().enumerate() {
                let wants = variant.operands[slot];
                match take {
                    Take::From(index) => {
                        let Some(source) = base.operands.get(*index) else {
                            return Err(format!(
                                "{at} fills `{}` from operand {index} of a row with {}",
                                wants.name,
                                base.operands.len()
                            ));
                        };
                        if source.ty != wants.ty {
                            return Err(format!(
                                "{at} fills `{}` ({:?}) from `{}` ({:?})",
                                wants.name, wants.ty, source.name, source.ty
                            ));
                        }
                    }
                    Take::Null => {
                        if !wants.nullable {
                            return Err(format!(
                                "{at} nulls `{}`, which the row does not declare nullable",
                                wants.name
                            ));
                        }
                        if scalar(wants.ty) {
                            return Err(format!(
                                "{at} nulls `{}`, which is {:?} and not a pointer",
                                wants.name, wants.ty
                            ));
                        }
                    }
                }
            }
            for term in arm.when {
                let index = term.operand();
                let Some(read) = base.operands.get(index) else {
                    return Err(format!(
                        "{at} reads operand {index} of a row with {}",
                        base.operands.len()
                    ));
                };
                match term {
                    Term::Aligned { bytes, .. } => {
                        if scalar(read.ty) {
                            return Err(format!(
                                "{at} tests the alignment of `{}`, which is {:?}",
                                read.name, read.ty
                            ));
                        }
                        if !bytes.is_power_of_two() {
                            return Err(format!(
                                "{at} tests alignment to {bytes}, and `rmsnorm.cu` spells the \
                                 same test as a MASK — the two agree only on powers of two"
                            ));
                        }
                    }
                    Term::Multiple { of, .. } => {
                        if read.ty != kernels::Ty::I32 {
                            return Err(format!(
                                "{at} divides `{}`, which is {:?} and not an i32",
                                read.name, read.ty
                            ));
                        }
                        if *of <= 0 {
                            return Err(format!("{at} divides `{}` by {of}", read.name));
                        }
                    }
                    Term::Is { .. } => {
                        if read.ty != kernels::Ty::Bool {
                            return Err(format!(
                                "{at} selects on `{}`, which is {:?} and not a Bool — a flag \
                                 clause reads a `Fact::Bool` and every other kind faults, so \
                                 this arm could never be taken",
                                read.name, read.ty
                            ));
                        }
                    }
                    Term::Present { .. } => {
                        if scalar(read.ty) {
                            return Err(format!(
                                "{at} tests `{}` for null, which is {:?} — a null clause reads \
                                 a `Fact::Address` and a scalar never supplies one, so this \
                                 arm could never be taken",
                                read.name, read.ty
                            ));
                        }
                        if !read.nullable {
                            return Err(format!(
                                "{at} tests `{}` for null and the row does not declare it \
                                 nullable — the binder refuses a null there, so the clause is \
                                 decided for every fire that reaches it and one of the two \
                                 arms is an instantiation that compiles and never runs",
                                read.name
                            ));
                        }
                    }
                }
            }
        }
        self.flags_are_covered(base)
    }

    /// Every flag a fire cannot fall through on, checked.
    ///
    /// # The hazard this exists for, in the shape the nine rows have
    ///
    /// `attn::device::write_kv<HND_LAYOUT>` has no `bool` PARAMETER: the flag
    /// is a template argument in both instantiations, so the fifteen
    /// arguments the kernel takes are the same fifteen either way. The
    /// decision still has to reach the fire from somewhere, and the place it
    /// reaches it is the base row's operand list — `hnd_layout` is an operand
    /// of the CONTRACT (`kv_paged.cu:660` takes it as a host argument) that no
    /// INSTANTIATION takes.
    ///
    /// So an arm forwards fifteen of the base's sixteen operands and drops
    /// the flag, which [`Take`] already expresses. What is new is the
    /// consequence for the base row: if a fire's flag matches no arm, it
    /// falls through to the base and binds SIXTEEN cells for a kernel that
    /// declares fifteen. `cuLaunchKernel` reads the parameter count out of
    /// the cubin, so the sixteenth is never read and the launch succeeds —
    /// which is the failure mode this whole design is built against, a wrong
    /// launch that reports success.
    ///
    /// The rule that removes it: **a flag no arm forwards to a kernel exists
    /// only to choose, so the arms must cover both of its values.** Then the
    /// base is unreachable by construction rather than by inspection, and it
    /// is checked here, with no device.
    ///
    /// Coverage is decided by enumeration over the SELECTORS the arms read,
    /// and an arm counts towards it only if every clause it states is a
    /// [`Term::Is`] or a [`Term::Present`] — the two terms whose operand has
    /// exactly two interesting states. An arm that also tests an alignment or
    /// a divisor cannot be discharged without the pointer or the width, so it
    /// is conservatively assumed not to fire — which makes this check refuse a
    /// table it cannot prove rather than accept one it merely likes.
    ///
    /// # [`Term::Present`] is enumerated here even though today's pair is
    /// exempt
    ///
    /// `attn::qkv_decode_qk_norm_rope_write_kv`'s two arms select on a
    /// POINTER, and both instantiations declare it — `qkv_fused.cu:64` and
    /// `:77` pass `rope_table` to `<…, true>` and `<…, false>` alike. So both
    /// arms FORWARD it, the retain below drops it, and there is nothing left
    /// to enumerate. That is the right answer for the right reason: the
    /// hazard is a base row binding one cell more than the instantiation
    /// reads, and it cannot arise when nothing is dropped.
    ///
    /// The enumeration covers `Present` anyway, because the exemption is a
    /// property of TODAY'S argument lists and not of the term. A kernel that
    /// baked its pointer's presence into a template and stopped taking the
    /// pointer would have `write_kv`'s exact shape with an address in place of
    /// a flag, and would need this check for the identical reason.
    ///
    /// # The case this would refuse and should not, stated because it is real
    ///
    /// A base kernel that takes the flag as a RUN-TIME parameter, with arms
    /// that bake it into a template, needs no coverage at all: the base
    /// handles every value it is handed. No kernel in this tree has that
    /// shape — `attn_mtp_paged_history<T>` takes `bool hnd_layout` at run
    /// time and has no `template <bool>` twin; `rope::device::rotate<kWriteKv,
    /// kHnd>` bakes both and takes `bool interleaved` at run time — so the
    /// rule is written for the shape that exists. The day one appears, the
    /// trigger below is what moves: a flag that some arm FORWARDS is already
    /// exempt, and a flag the base itself consumes would be too.
    fn flags_are_covered(&self, base: &'static KernelSig) -> Result<(), String> {
        let mut flags: Vec<usize> = Vec::new();
        for arm in self.arms {
            for term in arm.when {
                if let Term::Is { operand, .. } | Term::Present { operand, .. } = term
                    && !flags.contains(operand)
                {
                    flags.push(*operand);
                }
            }
        }
        // A flag some arm hands to a kernel is a parameter, not only a
        // decision, and the base plausibly takes it too. Only the ones that
        // reach no `__global__` at all are the subject here.
        flags.retain(|flag| {
            !self
                .arms
                .iter()
                .any(|arm| arm.take.contains(&Take::From(*flag)))
        });
        if flags.is_empty() {
            return Ok(());
        }
        // Bounded because the enumeration is 2^n and because a predicate over
        // nine independent flags is not a specialisation anyone can check by
        // reading it. Eight is past every launcher in the tree, which spells
        // one.
        if flags.len() > 8 {
            return Err(format!(
                "`{}` selects on {} flags that reach no kernel; the coverage this check \
                 proves is 2^n cases and a predicate over that many is not one a reader \
                 can compare to the C++",
                self.base,
                flags.len()
            ));
        }
        for assignment in 0..(1u32 << flags.len()) {
            let value = |operand: usize| {
                flags.iter().position(|f| *f == operand).map(|bit| assignment >> bit & 1 == 1)
            };
            let covered = self.arms.iter().any(|arm| {
                !arm.when.is_empty()
                    && arm.when.iter().all(|term| match term {
                        Term::Is { operand, value: wanted }
                        | Term::Present { operand, value: wanted } => {
                            value(*operand) == Some(*wanted)
                        }
                        _ => false,
                    })
            });
            if covered {
                continue;
            }
            let uncovered = flags
                .iter()
                .enumerate()
                .map(|(bit, operand)| {
                    format!("`{}` = {}", base.operands[*operand].name, assignment >> bit & 1 == 1)
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!(
                "`{}` selects on a flag no arm forwards, and states no arm for {uncovered}. \
                 A fire with that flag falls through to the base row, which binds {} cells \
                 for `{}` — and a flag that reaches no kernel is one cell more than the \
                 instantiation declares, which `cuLaunchKernel` accepts and never reads. \
                 State the other arm.",
                self.base,
                base.operands.len(),
                self.base,
            ));
        }
        Ok(())
    }
}

/// Whether a [`kernels::Ty`] is bound by a value rather than by an address.
///
/// The complement of `runtime::args::is_pointer`, and deliberately written as
/// the SCALAR list rather than as a second copy of the pointer one: the
/// pointer list is long and grows, this one is closed, and a type on neither
/// is refused by `Args::bind` before a launch can happen. So the two lists
/// drifting apart costs a refusal, never a launch.
const fn scalar(ty: kernels::Ty) -> bool {
    use kernels::Ty;
    matches!(
        ty,
        Ty::I32
            | Ty::U32
            | Ty::F32
            | Ty::Usize
            | Ty::I64
            | Ty::Bool
            | Ty::Stream
            // The two by-value enums. Scalars in the sense this list means —
            // a `cuLaunchKernel` cell holding a value and not an address — so
            // `Term::Aligned` and `Term::Present` over one are a build-time
            // refusal rather than a fire-time `Fault::Kind`, and `Take::Null`
            // over one is refused for the same reason a null `head_dim` is.
            | Ty::KvScheme
            | Ty::KvDType
    )
}

/// Every specialised row in the tree, one entry per FAMILY.
///
/// Listed by name and not discovered, for [`crate::families::ALL`]'s reason:
/// a specialisation that registered itself would be a second kernel reachable
/// through a symbol with nothing in this file admitting it, and "what does a
/// fire decide for itself?" has to stay a list a reader can finish.
///
/// # Why this is a slice of slices, which it was not
///
/// It named `families::norm::RMSNORM_STRIDED_VEC8` directly while norm was
/// the only family that specialised. That shape put **one line in this file
/// per specialised ROW**, and this file belongs to the runtime rather than to
/// any family — so the nine `kv_paged` arms, whose predicate and arms are
/// `families::attn`'s to write and whose author does not own `device.rs`,
/// could be written and not registered, or registered and not written. A
/// const nobody registered is dead code; a registration nobody wrote is a
/// build break. Neither is a thing to hand between two authors.
///
/// A family owns one `SPECIALISATIONS` slice and appears here once, so the
/// cost of specialising a tenth row in an already-listed family is zero lines
/// in this file. That is the same trade [`crate::families::ALL`] makes for
/// units, made for the same reason.
///
/// # Adding a family, in full
///
/// Write `pub static SPECIALISATIONS: &[&Specialisation] = &[..];` in the
/// family module beside the consts it names, then add one line here. The
/// order does not matter — [`specialisation`] scans by base symbol and
/// [`Specialisation::agrees`] is run over every entry by
/// `tests/specialise.rs`, so a family that lands here wrong fails on a
/// machine with no GPU rather than on a decode.
///
/// An empty slice is not the way to reserve a place. `tests/specialise.rs`
/// refuses one, because a family listed and empty reads as "specialises,
/// pending" and there is no such state — the row either states its arms or
/// it does not exist.
pub static SPECIALISED: &[&[&Specialisation]] = &[
    crate::families::norm::SPECIALISATIONS,
    // The five `attn/kv_paged.cuh` appenders — `write_kv`,
    // `write_kv_at_positions`, `write_kv_explicit`, `write_kv_explicit_devwin`
    // and `copy_kv_cells`, each `template <bool HND_LAYOUT>` and each chosen
    // by a host flag the launcher reads per layer.
    crate::families::attn::SPECIALISATIONS,
];

/// Every specialised row, flattened — what a reader and a test want.
///
/// [`SPECIALISED`]'s grouping is a registration convenience and says nothing
/// about the decisions; anything checking that the table is well formed wants
/// the rows, not the families.
pub fn specialisations() -> impl Iterator<Item = &'static Specialisation> {
    SPECIALISED.iter().copied().flatten().copied()
}

/// The specialisation a symbol carries, if it carries one.
///
/// A linear scan over a list with one entry on the launch path, which is
/// cheaper than the hash it would take to avoid it and stays cheaper while
/// the list is short. When it is not short, it becomes what
/// [`crate::unit::unit_of`] is: an index handed out once and reused.
#[must_use]
pub fn specialisation(symbol: &str) -> Option<&'static Specialisation> {
    specialisations().find(|spec| spec.base == symbol)
}

/// The row a symbol names, over every device table this crate knows.
///
/// A convenience over [`crate::unit::unit_of`] for a caller that wants the
/// ROW and not the unit — a typecheck emitter, say, or a test. The launch
/// path uses the unit form, because it needs the index to reach the module
/// cache anyway.
///
/// # This scanned two tables and said it scanned all of them
///
/// The body was `ALTUP_AUX.iter().chain(ELEMENTWISE)` while the sentence
/// above said "every device table this crate knows" — true when it was
/// written, when those two WERE the tables, and false from the first family
/// unit onwards. [`crate::unit::UNITS`] is `families::ALL` concatenated, and
/// `device::ALTUP_AUX`/`device::ELEMENTWISE` are merely the `rows` of two
/// units in [`crate::families::norm::UNITS`] — so the old body was a strict
/// subset of this one, by 15 families to 2.
///
/// It mattered in exactly one place and it mattered a lot:
/// [`every_dispatched_symbol_has_a_row`] uses this, so a migrated family
/// symbol added to [`JIT_DISPATCHED`] failed with **`is dispatched to the
/// JIT and has no row`** — a message that blames the row, when the row is
/// there and the LOOKUP could not see it. `git log -S "JIT_DISPATCHED"`
/// returns one commit: the list has never been extended past altup, and this
/// is why.
///
/// [`crate::runtime::fire`] has its own `row`, through `unit_of`, which is
/// the wide one — so the LAUNCH path was always correct and only the
/// gatekeeping was narrow. Two functions of one name with different domains
/// is how a defect hides in plain sight for a whole migration.
#[must_use]
pub fn row(symbol: &str) -> Option<&'static DeviceKernel> {
    crate::unit::rows().find(|entry| entry.sig.symbol == symbol)
}

#[cfg(test)]
mod tests {
    use super::{ALTUP_AUX, ELEMENTWISE, JIT_DISPATCHED, jit_dispatched, row};
    use kernels::LaunchRule;

    /// A device row's symbol is unique across the tables, not merely within
    /// one. Two units claiming one symbol is unresolvable by construction —
    /// `unit_of` would answer with whichever it scanned first — so it is
    /// checked here rather than left to the scan order.
    #[test]
    fn a_symbol_belongs_to_one_device_row() {
        let mut seen: Vec<&str> = Vec::new();
        for entry in ALTUP_AUX.iter().chain(ELEMENTWISE) {
            assert!(!seen.contains(&entry.sig.symbol), "{} is stated twice", entry.sig.symbol);
            seen.push(entry.sig.symbol);
        }
    }

    /// Every symbol the dispatcher routes to the JIT is a symbol some unit
    /// can actually compile.
    ///
    /// This is the check that would have caught the defect the switch was
    /// built to fix: a row named in `JIT_DISPATCHED` whose kernel is in no
    /// unit has no shim entry left and nowhere to go, so its fire reaches a
    /// hand-written arm that does not exist and is diagnosed as an unknown
    /// kernel — a lie about what went wrong.
    #[test]
    fn every_dispatched_symbol_has_a_row() {
        for symbol in JIT_DISPATCHED {
            assert!(row(symbol).is_some(), "{symbol} is dispatched to the JIT and has no row");
        }
        assert_eq!(jit_dispatched().len(), JIT_DISPATCHED.len());
    }

    /// The lookup is the tables.
    #[test]
    fn the_lookup_is_the_tables() {
        for entry in ALTUP_AUX.iter().chain(ELEMENTWISE) {
            assert_eq!(row(entry.sig.symbol).map(|r| r.sig.symbol), Some(entry.sig.symbol));
        }
        assert!(row("norm::a_kernel_nobody_wrote").is_none());
    }

    /// ...and "the tables" means EVERY unit's, not the two this module holds.
    ///
    /// `the_lookup_is_the_tables` above passes on a `row` that scans only
    /// `ALTUP_AUX` and `ELEMENTWISE`, because those are the rows it feeds it
    /// — a gate asserting its own denominator, which is the failure
    /// `new-horizon.md` §21 names five times. This one feeds it
    /// [`crate::unit::rows`], so it fails on the narrow body for every
    /// migrated family row at once.
    ///
    /// It is not a hypothetical regression. The narrow body is why
    /// [`JIT_DISPATCHED`] has never grown past altup: adding a family symbol
    /// to it failed with *"is dispatched to the JIT and has no row"*, which
    /// names the row as the problem and sent three attempts looking at the
    /// row.
    #[test]
    fn the_lookup_is_every_unit_s_tables() {
        let mut checked = 0usize;
        for entry in crate::unit::rows() {
            assert_eq!(
                row(entry.sig.symbol).map(|r| r.sig.symbol),
                Some(entry.sig.symbol),
                "`{}` is a row of unit `{}` and `device::row` cannot find it",
                entry.sig.symbol,
                crate::unit::unit_of(entry.sig.symbol).map_or("<none>", |(_, u)| u.name),
            );
            checked += 1;
        }
        // A DENOMINATOR, because the loop above is vacuous over an empty
        // iterator and would pass on one.
        assert!(
            checked > ALTUP_AUX.len() + ELEMENTWISE.len(),
            "only {checked} rows scanned; this module alone holds {}, so the \
             iterator is not the whole table and this test proves nothing",
            ALTUP_AUX.len() + ELEMENTWISE.len(),
        );
    }

    /// Every Tier A row states a rule. `Unstated` is what a row says when it
    /// has not been ported, and a row in THIS table that said it would be
    /// launched by a driver that has nothing to launch it with.
    #[test]
    fn every_entry_states_its_launch() {
        for k in ALTUP_AUX {
            assert_ne!(k.sig.launch, LaunchRule::Unstated, "{} states no rule", k.sig.symbol);
        }
    }

    /// No row here needed a rule the Metal tables had not already stated.
    ///
    /// This is the pilot's headline and it is cheap to keep honest: if a
    /// later port adds a variant to serve one CUDA kernel, this fails and
    /// the claim gets re-measured rather than repeated.
    #[test]
    fn the_pilot_added_no_launch_rules() {
        const REUSED: &[LaunchRule] = &[
            LaunchRule::Rms,
            LaunchRule::ElementwiseRows,
            LaunchRule::RouteRows,
            LaunchRule::Elementwise,
        ];
        for k in ALTUP_AUX {
            assert!(
                REUSED.contains(&k.sig.launch),
                "{} states {:?}, which is not one of the rules Metal already had",
                k.sig.symbol,
                k.sig.launch
            );
        }
    }

    /// A stream is not an operand, and a Tier A row may not say it is.
    #[test]
    fn no_entry_takes_a_stream() {
        for k in ALTUP_AUX {
            assert!(
                k.sig.operands.iter().all(|o| o.ty != kernels::Ty::Stream),
                "{} takes a stream as an operand",
                k.sig.symbol
            );
        }
    }

    /// Two symbols may not name one instantiation, and one symbol may not be
    /// stated twice.
    ///
    /// The first is what would happen if a row were copied and its tag not
    /// changed -- two rows firing one kernel, which no test that runs either
    /// of them alone can see. The second is the same defect from the other
    /// end.
    #[test]
    fn the_map_from_symbol_to_instantiation_is_a_bijection() {
        let mut seen: Vec<(String, &str)> = Vec::new();
        for k in ALTUP_AUX {
            let inst = k.instantiation();
            assert!(
                !seen.iter().any(|(i, _)| *i == inst),
                "{} names an instantiation another row already claims: {inst}",
                k.sig.symbol
            );
            assert!(
                !seen.iter().any(|(_, s)| *s == k.sig.symbol),
                "{} is stated twice",
                k.sig.symbol
            );
            seen.push((inst, k.sig.symbol));
        }
        assert_eq!(seen.len(), ALTUP_AUX.len());
    }

    /// The two `tanh` rows are the same template at different element types,
    /// which is the claim "a second numeric format costs a row".
    #[test]
    fn a_second_numeric_format_is_a_row_and_not_a_kernel() {
        let bf16 = ALTUP_AUX.iter().find(|k| k.sig.symbol == "norm::tanh_bf16").expect("bf16 row");
        let f16 = ALTUP_AUX.iter().find(|k| k.sig.symbol == "norm::tanh_f16").expect("f16 row");
        assert_eq!(bf16.template_path, f16.template_path);
        assert_ne!(bf16.elem, f16.elem);
        assert_eq!(bf16.sig.operands.len(), f16.sig.operands.len());
    }

    /// Each row is shorter than its twin, which is the deletion the
    /// experiment claims. Stated as a total so the number is in the test
    /// output rather than in a commit message.
    ///
    /// Thirty-one operands become twenty-one. The ten are the six streams
    /// and the four extents the rules recover -- which is to say every one
    /// of them was a fact the table already held, spelled a second time
    /// because a C++ function had no other way to receive it.
    #[test]
    fn tier_a_rows_are_shorter_than_their_twins() {
        let mine: usize = ALTUP_AUX
            .iter()
            .filter(|k| k.sig.symbol != "norm::tanh_f16")
            .map(|k| k.sig.operands.len())
            .sum();
        let twins: usize = crate::table::norm::KERNELS
            .iter()
            .filter(|t| ALTUP_AUX.iter().any(|k| k.sig.symbol == t.symbol))
            .map(|k| k.operands.len())
            .sum();
        assert_eq!(ALTUP_AUX.len(), 7, "six ported kernels and the fp16 extra");
        assert_eq!((twins, mine), (31, 21));
    }
}
