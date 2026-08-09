//! `layout`'s JIT units — three headers, and nine rows over them.
//!
//! # Eleven rows became nine, and two units went with them
//!
//! `new-horizon.md` §28.4 measured `layout::split_gate_up_bf16` and
//! `layout::embed_bf16_vocab_shard` as second names for jobs a reached row
//! already does: `lower.rs:1477` sends `SplitQGate` to
//! `layout::split_q_gate_bf16` and `lower.rs:1462` sends every `Embed`,
//! sharded or not, to `layout::embed_bf16`. Neither duplicate had a caller —
//! `cuda::split_gate_up` and `cuda::embed_vocab_shard` had zero call sites,
//! no golden named either, and no `vocab_offset`/`local_vocab` appears in
//! `crates/model`, `model-loader` or `driver-cuda/src` at all. Each was the
//! ONLY row of its unit, and `tests/units.rs` fails a unit with an empty row
//! set, so `layout/split_gate_up` and `layout/embed` are no longer units.
//! Their device text is still carried — [`crate::source::DEVICE_HEADERS`]
//! walks `csrc/src` for `*.cuh` — and the ahead-of-time `.cu`s still include
//! and compile it, which is where `layout::embed_bf16` actually fires:
//! `driver-cuda/tests/bridge_smoke.rs` names it at five call sites and
//! `real_hybrid.rs:577` at a sixth. **What was lost is the NVRTC compile
//! check on `layout/embed.cuh`**,
//! and it was lost because the flat gather cannot be a row: `embed.cu:7`
//! records that its `VEC` choice is a host alignment test and no `Source`
//! produces "is this pointer 16-byte aligned".
//!
//! Ten `.cu` files held twenty-nine `__global__`s interleaved with
//! thirty-one host launchers. Every one of the ten is split now: the device
//! text lives in a `.cuh` the `.cu` includes, so exactly ONE definition of
//! each kernel exists in the tree. That is a SPLIT and not a copy on
//! purpose — `norm/altup_aux` shipped a release with two definitions of six
//! kernels, they agreed the day they were written, each stayed right for
//! whichever half of the tests exercised it, and the drift was invisible
//! until one half was edited.
//!
//! Three of the ten headers get a unit. Six of the other seven —
//! `layout/envelope`, `layout/gather_tokens`, `layout/geometry`,
//! `layout/graph_pad`, `layout/embed` and `layout/split_gate_up` — get none,
//! because
//! `tests/units.rs` asserts `!unit.rows.is_empty()` and a unit with no rows
//! would be a claim about a caller that does not exist. Their
//! device text is still carried: [`crate::source::DEVICE_HEADERS`] walks
//! `csrc/src` for `*.cuh`, so those four are in the JIT's header set and a
//! later row costs a line here and no C++ anywhere. `layout/slot_ops` was
//! the fifth of them until [`kernels::LaunchRule::Single`] landed, which is
//! exactly the shape that paragraph predicted.
//!
//! # What the launch-rule port changed here, and what the arity fix changed
//!
//! Twelve of sixteen rules evaluate now — `PerHead`, `PerHeadElementwise`,
//! `GatedRms`, `SdpaVector`, `SplitPacked`, `Rope`, `RouterLane` and
//! `RouterSort` landed, and `runtime::launch`'s `Dims` gained
//! nine fields including `q_heads`, `kv_heads` and `head_dim`. Every
//! `gridDim.y` refusal was re-checked against the rule that now exists, by
//! reading the `<<<>>>` and working out what `eval` returns for the same
//! extents. Six kernels came back geometrically satisfied and none of them
//! could be rowed, for ONE remaining reason: they were not templates at all,
//! and [`DeviceKernel::instantiation`](crate::device::DeviceKernel) spells
//! `path<elem>` — a plain `__global__` has nowhere to put an `elem`, so
//! `nvrtcAddNameExpression` takes the string, `nvrtcGetLoweredName` has
//! nothing to hand back, and `tests/units.rs` reports the row as defective.
//! Three of the six are templates now and two of those three carry rows:
//!
//! * `split_gate_up` is `dim3(ceil(inter / 256), n_tokens)` at 256, which is
//!   `Rule::SplitPacked` — `grid [ceil(in_width / 256), rows]` at 256 — and
//!   NOT the transposition of `ElementwiseRows` this file called it before
//!   the port. The rule sizes on the packed input's `2 * inter` and so hands
//!   over twice the blocks; the kernel's two loops stride
//!   `j += blockDim.x * gridDim.x` and bound on `inter`, which is precisely
//!   the licence `attn/split_packed.cuh` wrote down for that rule. **Was
//!   rowed**, in its own unit; both went as §28.4 duplicates above, and the
//!   geometric finding is what stays — it is what the `.cu` fires under.
//! * `deinterleave::split_q_gate` is `dim3(N, num_heads)` at
//!   `(head_dim < 128) ? 64 : 128`, and `Rule::PerHeadElementwise` is
//!   `grid [rows, q_heads]` at `clamp(head_dim, 32, 128)` — the same axes in
//!   the same order, and the same block at every head width the launcher
//!   picks, because the kernel strides `i += blockDim.x` and bounds itself on
//!   `head_dim`. **Rowed.**
//! * `deinterleave::repeat_interleave_heads` is the identical launcher over
//!   `q_heads` and is a template for the same one-line reason, but carries
//!   **no row**: `repeat_interleave_heads_bf16` is declared in
//!   `deinterleave.hpp`, defined in `deinterleave.cu`, and called from
//!   nowhere in the tree — no model text, no driver-internal row, no sibling
//!   `.cu`. A row would name a caller that does not exist and would sit in
//!   `migration_status`' denominator forever.
//!
//! The other three of the six are `envelope`'s, and they stay plain
//! `__global__`s deliberately — see the refusal list below. A template with
//! one instantiation and no row is a rename.
//!
//! The arity ceiling that was also reported against this family does not
//! exist and never did: `elem` is pasted whole between the angle brackets, so
//! it carries an argument LIST, and the `::pie_cuda_driver::kernels::` prefix
//! lands on its first token only. Measured here under NVRTC 13.0 —
//! `"device::i32(128)"` and `"device::true_type::value"` both resolve a
//! lowered name, `"128"` and `"true"` fail the name-map pragma with
//! `expected an identifier`, because a bare literal cannot take a namespace.
//! It moved nothing in `layout`: the two kernels it was cited against,
//! `envelope::dot<BLOCK>` and `embed<VEC>`, are refused by a page-window
//! rectangle and by a host alignment choice respectively, and both refusals
//! are recorded below without it. It carries one caveat for the families it
//! DID move — `abi::device_cpp_ty` builds an operand's C++ type as
//! `const ::pie_cuda_driver::kernels::{elem}*`, which a multi-argument `elem`
//! turns into text that is not a type, so a row with several template
//! arguments and a buffer operand compiles and resolves at run time and
//! silently loses the OFFLINE typecheck. No row in this file has one.
//!
//! # The kernels that could not be rowed, and what each one is blocked on
//!
//! **Re-audited at `LaunchRule` 21 → 28.** Every entry below was re-decided
//! against the eight rules §21.13 added. Three of them touch this family and
//! none of them moves a row: `RowsFlat` is `geometry`'s two launchers to the
//! digit and is named in its own doc as not serving them, `WarpTiledScan`
//! produces the third grid axis `gather_tokens` was refused for and is not
//! that axis, and `Tile16` is a 16×16 block where this family's 2-D blocks
//! are `dim3(32, 8)`. `Slab`, `PerRowNarrow`, `RowsPerHead`, `AxialRope` and
//! `RoutedQmv` have no launcher of their shape here at all.
//!
//! * **The arithmetic would have to change.** `envelope::merge_written_fused`
//!   and `envelope::merge_written` are `dim3(num_tokens, num_kv_heads)`,
//!   which `Rule::GatedRms` spells exactly — and they narrow with
//!   `f32_to_bf16_rd` and `f32_to_bf16_ru`, DIRECTED rounding that is what
//!   makes a stored envelope a true bound. `Elem<T>::from_f32` is
//!   round-to-nearest-even and the trait has no directed member, so retyping
//!   them would round a bound back inside the range it must contain: a
//!   wrong answer, never a failure. `merge_written` also drives
//!   `atomic_min`/`atomic_max` CAS loops written on bf16's sign-magnitude
//!   order. That is a rewrite, not a retype, and §8 wants parity evidence
//!   for a rewrite.
//! * **One symbol, two launches.** `envelope::reset_started_pages` is clean —
//!   no directed rounding, no atomics, `GatedRms` geometry — and still
//!   cannot be rowed: `launch_envelope_merge_written_bf16` fires
//!   `merge_written_fused` alone when `num_tokens <= 128` and
//!   `reset_started_pages` then `merge_written` when it is more. A row is one
//!   symbol firing one launch; a row for either half states half a contract.
//! * **Not a rectangle.** `envelope::recompute` is
//!   `dim3(num_pages, num_kv_heads)` and `envelope::update_appended` is
//!   `dim3(max_touched, num_kv_heads)`: `GatedRms` puts `dims.rows` on
//!   `grid.x`, and a page count and a host-computed touched-page bound are
//!   neither of them a fire's row count. `envelope::dot<BLOCK>` is
//!   `dim3(p_max, num_kv_heads)` at 128 over a page window, and its `<int>`
//!   is spellable now — `elem: "device::i32(128)"` resolves — so the
//!   rectangle is the whole of what refuses it. Resolving is NVRTC finding
//!   the instantiation, which is weaker than it sounds: a head that is a
//!   VALUE rather than a type resolves exactly as well, and
//!   `abi::emit_device_typecheck` refuses this very spelling because it
//!   cannot build a pointer from it. Existence is not agreement on an
//!   operand list, and no row here rests on the difference.
//!   `slot_ops::zero_slots_if_fresh` is
//!   `dim3(request_count, layer_count)` at 256 — the second axis is LAYERS,
//!   and every headed rule would tell it that number is a head count.
//! * **A 3-D grid.** `gather_tokens`' two kernels launch
//!   `dim3(num_ops, 1, num_layers)`. A third axis is no longer the
//!   vocabulary's gap — [`kernels::LaunchRule::WarpTiledScan`] produces one
//!   — and this is still not it: that rule's axes are
//!   `[rows, kv_heads, ceil(value_width / 4)]` at 128 threads, and this grid
//!   has a LITERAL 1 on `y` and a LAYER count on `z`. No `Dims` field holds
//!   a layer count, and a rule that filled `y` from `kv_heads` would make the
//!   middle axis a head count for a copy that has no heads.
//! * **"One block, whatever the rectangle."** `graph_pad::graph_pad_rows` is
//!   `<<<1, padding>>>` and `slot_ops::copy_if_valid_slot` is `<<<1, 256>>>`,
//!   and both are idempotent only once: `RouteRows` would launch `dims.rows`
//!   blocks racing on one CSR, or repeat the same copy `dims.rows` times.
//!   `RouterSort` is the one rule that launches a single block and it fixes
//!   1024 threads with `n_experts`-sized shared memory, which is a mixture's
//!   sort and not a pad. [`kernels::LaunchRule::RowsFlat`] was checked
//!   against both and is not either: it answers `ceil(rows / 256)`, a
//!   QUOTIENT, and the `1` in these two launchers is a literal the host
//!   wrote — equal only at `rows <= 256` and a growing over-launch above it.
//!
//!   **HALF of this is retired and half stands.**
//!   [`kernels::LaunchRule::Single`] was written from
//!   `layout/slot_ops.cu:60` and `attn/kv_paged.cu:516` together, and
//!   `copy_if_valid_slot` is a row ([`SLOT_OPS`]). Every sentence above
//!   survives as the ARGUMENT for that rule rather than against it — the
//!   quotient reading is what the rule's doc refuses, in those words.
//!   `graph_pad_rows` is refused still, on the clause that was already
//!   separate: **its block is `padding`, which no rule computes.** `Single`
//!   fixes 256, and a rule whose block came off a `Dims` field is the design
//!   §21.14 refuses — a block width is the launcher's property, and a fire
//!   can make no statement about it true or false.
//! * **A host choice the device cannot make.** `embed`'s `VEC`,
//!   `gather_rows`' `transpose_nld_to_lnd_vec4` and `gather_tokens`' `int4`
//!   form are selected on the HOST from pointer alignment or `dim & 7`, and
//!   the element count launched over DEPENDS on the answer. No `Source` in
//!   `kernels/src/lib.rs` produces "is this pointer 16-byte aligned", and a
//!   launcher that picks a vector width from an alignment is choosing a
//!   KERNEL, not a rectangle. The rows fire the scalar twins, which is what
//!   the vectorised forms were measured against. `embed<bool VEC>`'s
//!   non-type parameter is NOT a second reason: it is spellable as
//!   `elem: "device::true_type::value"`, measured under NVRTC 13.0 alongside
//!   `rope::rotate`'s two bools. Naming one arm of a host decision is
//!   precisely what a row must not do, so being able to name it changes
//!   nothing here.
//! * **No fire, no `Source`.** `geometry`'s two kernels are
//!   `<<<ceil(n / 256), 256>>>`, which is
//!   [`kernels::LaunchRule::RowsFlat`] to the digit —
//!   `runtime::launch::rows_flat` cites `layout/geometry.cu:29-31` and
//!   `:45-47` by name among the launchers it reproduces, and says of these
//!   two that they *"stay rowless for a reason that is not geometry"* —
//!   and they still get no row. They are composed by the DRIVER while it
//!   builds a plan, not stated by any model text, so a row for either would
//!   be a contract naming a caller that does not exist, and the symbol would
//!   sit in `migration_status`' denominator forever. Both are also plain
//!   `__global__`s, so the decision costs nothing to act on. If a statement
//!   for them ever exists, the rule is already right — which is now
//!   literally true rather than a hope, and is the whole of what the new
//!   rule changed here.
//!   `gather_rows::embed_scaled_concat` has no ahead-of-time `KernelSig` at
//!   all, and `vocab`, `scale` and `hidden_first` are three operands with no
//!   `Source` between them.
//!
//! # Every launcher stays
//!
//! Including the ones this family does not call: `envelope`'s seven are fired
//! from `attn/kv_paged.cu` and `gather_tokens`' three from
//! `quant/dequant_fp4.cu` and `rope/rope.cu`, both outside this family. This
//! migration extracts device text and adds rows; it deletes nothing, and
//! `new-horizon.md` §10.10 fixes that order so the two paths can be measured
//! against each other before either is retired.

use kernels::KernelSig;
use kernels::LaunchRule;
use kernels::Source;
use kernels::kernel;
use kernels::operands;

use crate::device::DeviceKernel;
use crate::unit::Unit;

/// The packed-bank splits and the row concat — gpt-oss's parity
/// deinterleave, Qwen's GDN halves, and the concat/split pair.
pub const DEINTERLEAVE: Unit = Unit {
    name: "layout/deinterleave",
    root: include_str!("../../csrc/src/layout/deinterleave.cuh"),
    rows: DEINTERLEAVE_ROWS,
    options: &[],
};

/// The epilogue's row gather and the PLE relay's transpose.
pub const GATHER_ROWS: Unit = Unit {
    name: "layout/gather_rows",
    root: include_str!("../../csrc/src/layout/gather_rows.cuh"),
    rows: GATHER_ROWS_ROWS,
    options: &[],
};

/// The slot-conditional byte copy — one kernel of `slot_ops.cuh`'s two.
///
/// The other, `zero_slots_if_fresh`, is `dim3(request_count, layer_count)` at
/// 256, and its second axis is a LAYER count no [`kernels::LaunchRule`] and no
/// `Dims` field carries. It compiles here anyway — a unit compiles its root,
/// not its rows — which is what makes the refusal cheap to retire.
pub const SLOT_OPS: Unit = Unit {
    name: "layout/slot_ops",
    root: include_str!("../../csrc/src/layout/slot_ops.cuh"),
    rows: SLOT_OPS_ROWS,
    options: &[],
};

/// The units `layout` compiles.
pub static UNITS: &[Unit] = &[DEINTERLEAVE, GATHER_ROWS, SLOT_OPS];

/// [`SLOT_OPS`]'s one instantiation, which is not one: `copy_if_valid_slot`
/// is a plain `__global__` over `u8`, so it is [`DeviceKernel::PLAIN`].
static SLOT_OPS_ROWS: &[DeviceKernel] = &[DeviceKernel {
    sig: &SLOT_OPS_SIGS[0],
    template_path: "layout::device::copy_if_valid_slot",
    elem: DeviceKernel::PLAIN,
}];

/// The contract, which is the launcher's five parameters minus the stream.
///
/// `layout/slot_ops.cu:59-62`:
///
/// ```text
/// :58   if (bytes == 0) return;
/// :59   constexpr int kThreads = 256;
/// :60   device::copy_if_valid_slot<<<1, kThreads, 0, stream>>>(
/// :61       src, dst, bytes, slot_ids, request);
/// ```
///
/// which is [`kernels::LaunchRule::Single`] on every field, and the rule was
/// written from this launcher and `attn/kv_paged.cu:516` together.
///
/// **This module header's "one block, whatever the rectangle" refusal is
/// retired for this kernel and NOT for its neighbour.** The refusal named
/// three launchers — this one, `attn`'s two, and `graph_pad::graph_pad_rows`
/// — and the fourth stays refused for the reason it was always given
/// separately: `graph_pad_rows` is `<<<1, padding>>>`, whose BLOCK is a
/// host-computed extent, and `Single` fixes 256. A rule whose block came off
/// a `Dims` field is the design §21.14's test refuses, because a block width
/// is the launcher's property and a fire can make no statement about it true
/// or false.
///
/// **Unsourced, and the twin is too.** `table/layout.rs:35` states
/// `layout::copy_if_valid_slot` with no `Source` on any operand: `src` and
/// `dst` are driver-owned slot arenas, `request` is an index into a batch the
/// driver holds, and `bytes` is a slot stride the model text never names.
/// `crate::abi` skips a row with any [`kernels::Source::Unbound`] operand
/// whole, so this row generates no dispatch and claims none — it states what
/// the kernel is and how it launches, which is what the migration counts.
#[rustfmt::skip]
static SLOT_OPS_SIGS: [KernelSig; 1] = [
    kernel!(copy_if_valid_slot "layout::copy_if_valid_slot", whole = true,
        file = Some("layout/slot_ops.cuh"),
        launch = LaunchRule::Single,
        operands = operands![
            src: U8s,
            dst: U8sMut,
            bytes: Usize,
            slot_ids: I32s,
            request: Usize,
        ]),
];

/// [`DEINTERLEAVE`]'s instantiations — six of the header's seven kernels.
///
/// All six are `bf16` today and all six are written over `T`. A second
/// format costs one row and no C++, which is the measurement
/// `norm/elementwise` made with `residual_add_f16`; none is declared here
/// because no fire asks for one, and a row for a kernel nothing states is a
/// claim about a caller that does not exist. The seventh,
/// `repeat_interleave_heads`, is a template for exactly that reason and no
/// other — nothing in the tree calls its launcher.
static DEINTERLEAVE_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &DEINTERLEAVE_SIGS[0],
        template_path: "layout::device::deinterleave_rows",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEINTERLEAVE_SIGS[1],
        template_path: "layout::device::deinterleave_vec",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEINTERLEAVE_SIGS[2],
        template_path: "layout::device::concat_rows",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEINTERLEAVE_SIGS[3],
        template_path: "layout::device::split_rows",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEINTERLEAVE_SIGS[4],
        template_path: "layout::device::split_qwen_gdn_ba",
        elem: "device::bf16",
    },
    DeviceKernel {
        sig: &DEINTERLEAVE_SIGS[5],
        template_path: "layout::device::split_q_gate",
        elem: "device::bf16",
    },
];

/// The contracts, in [`DEINTERLEAVE_ROWS`]' order.
///
/// Each is its ahead-of-time twin in `kernels-cuda-new/src/table/layout.rs`
/// minus the
/// stream — `cuLaunchKernel`'s SIXTH PARAMETER, outside the `void**` — and
/// minus whatever extent its rule recovers. The pilot's rows lost ten of
/// thirty-one operands that way, and an operand restating an extent the grid
/// already fixes is an operand that can disagree with the grid.
#[rustfmt::skip]
static DEINTERLEAVE_SIGS: [KernelSig; 6] = [
    // gpt-oss packs gate and up ROW BY ROW, so splitting them is a parity
    // deinterleave and not a slice. Weight-shaped: `i` is a WEIGHT's row
    // count, which is why the twin sourced neither extent, and `RouteRows`
    // recovers it from the fire's rectangle regardless.
    kernel!(deinterleave_rows "layout::deinterleave_rows_bf16",
        file = Some("layout/deinterleave.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            fused: Buf,
            gate_out: BufMut,
            up_out: BufMut,
            h: I32,
        ]),
    // The flat form of the same split, one thread per output element.
    //
    // `i` SURVIVES where the row form lost it. `LaunchRule::Elementwise`
    // rounds the element count up to a whole block, so the tail threads of
    // the last block have to be told to stop -- an extent a rule RECOVERS is
    // not an operand, and an extent a rule ROUNDS is. `norm::tanh` states the
    // same distinction.
    kernel!(deinterleave_vec "layout::deinterleave_vec_bf16",
        file = Some("layout/deinterleave.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            fused: Buf,
            gate_out: BufMut,
            up_out: BufMut,
            i: I32,
        ]),
    // `[N, left] ++ [N, right] -> [N, left+right]`, one block per row.
    // Sourced by nothing, exactly as its twin sources nothing.
    kernel!(concat_rows "layout::concat_bf16_rows",
        file = Some("layout/deinterleave.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            left: Buf,
            right: Buf,
            out: BufMut,
            left_dim: I32,
            right_dim: I32,
        ]),
    // The inverse: one packed row out to two. Two results, so both widths
    // come off the results and the source needs no extent of its own.
    kernel!(split_rows "layout::split_bf16_rows",
        file = Some("layout/deinterleave.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            src: Buf <- Source::In(0),
            left: BufMut <- Source::Out(0),
            right: BufMut <- Source::Out(1),
            left_dim: I32 <- Source::OutWidth(0),
            right_dim: I32 <- Source::OutWidth(1),
        ]),
    // Qwen's GDN bank, split by HALVES where `deinterleave_rows` splits by
    // parity. Same shape, different layout, checkpoint decides -- which is
    // why they are two kernels and not one with a flag.
    kernel!(split_qwen_gdn_ba "layout::split_qwen_gdn_ba_bf16",
        file = Some("layout/deinterleave.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            ba: Buf <- Source::In(0),
            b_out: BufMut <- Source::Out(0),
            a_out: BufMut <- Source::Out(1),
            v_h: I32 <- Source::OutWidth(0),
        ]),
    // Full attention's q_proj packs the query and the per-token output gate
    // PER HEAD -- `[N, heads, 2*head_dim]`, query first -- so this is strided
    // by head, not a halves cut like `split_gate_up`.
    //
    // `PerHeadElementwise`, checked against `deinterleave.cu`'s
    // `split_q_gate_bf16`: `dim3 grid(N, num_heads)` with
    // `(head_dim < 128) ? 64 : 128` threads and no shared memory, against the
    // rule's `grid [rows, q_heads, 1]`, `block clamp(head_dim, 32, 128)`,
    // `smem 0`. Same axes in the same order. The block agrees at every width
    // of 128 and above and at every width in `[32, 128)` the rule gives
    // `head_dim` where the launcher gives 64 -- both cover the head, because
    // the two copy loops stride `i += blockDim.x` and stop at `i < head_dim`.
    // Under 32 the clamp is WIDER than the head; the surplus lanes fail that
    // test on their first iteration and this kernel declares no shared array
    // for them to touch.
    //
    // The symbol is `driver_internal`'s, not the model DSL's, which is the
    // precedent `attn::split_qkv_bf16` set: a driver-internal launcher is
    // still stated by a trace and still looked up by symbol.
    //
    // `N` and `num_heads` SURVIVE as operands where `RouteRows`' rows do not.
    // The kernel guards `if (n >= N || h >= num_heads) return;` and, more to
    // the point, multiplies both back into every address it forms --
    // `(n * num_heads + h) * 2 * head_dim` -- so they are addressing
    // arithmetic the grid happens to agree with rather than an extent the
    // grid recovers. `kda_gate_beta` keeps its `t` for the same reason.
    kernel!(split_q_gate "layout::split_q_gate_bf16",
        file = Some("layout/deinterleave.cuh"),
        launch = LaunchRule::PerHeadElementwise,
        operands = operands![
            packed: Buf <- Source::In(0),
            q_out: BufMut <- Source::Out(0),
            gate_out: BufMut <- Source::Out(1),
            n: I32 <- Source::Rows,
            // Off the QUERY half, not the packed operand: `packed` is
            // `[N, heads, 2*head_dim]` and only the query's half of it lands
            // here, so the head count comes from what is written.
            num_heads: I32 <- Source::Div(&Source::Width(&Source::Out(0)), &Source::CtxNonZero("head_dim")),
            head_dim: I32 <- Source::Ctx("head_dim"),
        ]),
];

/// [`GATHER_ROWS`]'s instantiations — two of the header's four kernels.
///
/// `device::u16` and not `device::bf16` on both, because both are pure
/// copies: neither ever converts to float, and the ahead-of-time launchers
/// take `u16*` for exactly that reason. A tag type that promises arithmetic
/// nobody performs is a tag type that invites it.
static GATHER_ROWS_ROWS: &[DeviceKernel] = &[
    DeviceKernel {
        sig: &GATHER_ROWS_SIGS[0],
        template_path: "layout::device::gather_rows",
        elem: "device::u16",
    },
    DeviceKernel {
        sig: &GATHER_ROWS_SIGS[1],
        template_path: "layout::device::transpose_nld_to_lnd",
        elem: "device::u16",
    },
];

/// The contracts, in [`GATHER_ROWS_ROWS`]' order.
#[rustfmt::skip]
static GATHER_ROWS_SIGS: [KernelSig; 2] = [
    // THE EPILOGUE'S GATHER. A prefill streams one row per token and reads
    // one distribution per request, so the rows actually sampled have to be
    // collected before the final norm and the head -- and they are not a
    // contiguous run, which is why this is a gather rather than a slice.
    //
    // The last operand is the row WIDTH, not a vocabulary: the header names
    // it `vocab` and the caller passes `H`, because this gathers hidden rows
    // on their way INTO the head. The kernel's parameter is named `width`
    // now, since a template over `T` reading `vocab` would be a comment that
    // is wrong at every call site.
    //
    // `num_dst_rows` leaves -- `RouteRows` is one block per destination row.
    // The kernel strides by `blockDim.x` for the same reason: the rule sizes
    // the block `min(1024, ceil(width/32)*32)`, and the file-scope
    // `constexpr int BLOCK = 256` it used to stride by would have dropped
    // every element past the 256th the moment the rule disagreed.
    kernel!(gather_rows "layout::gather_bf16_rows",
        file = Some("layout/gather_rows.cuh"),
        launch = LaunchRule::RouteRows,
        operands = operands![
            src: U16s <- Source::In(0),
            row_indices: I32s <- Source::SamplingIndices,
            dst: U16sMut <- Source::Out(0),
            width: I32 <- Source::OutWidth(0),
        ]),
    // The PLE relay: `[N, L, D] -> [L, N, D]`, so a layer reads a contiguous
    // slice. Addressing, not arithmetic.
    //
    // `Elementwise` recovers NOTHING the kernel needs: it flattens the
    // rectangle to an element count and rounds it up to a block, so `n`,
    // `layers` and `dim` all stay -- the kernel divides the flat index by
    // them -- and `total` joins them because the tail threads of the last
    // block have to be told to stop. Six operands where the twin had six and
    // a stream; the stream is the only thing that left.
    //
    // Neither extent is the plan's, which is what put this row on the
    // generator's wall. The PLE dim is a fire fact the driver holds, and the
    // layer count is what is left of the operand's row once that is divided
    // out -- exactly the arithmetic the hand-written arm did, refusal on an
    // unset `ple_dim` included.
    kernel!(transpose_nld_to_lnd "layout::transpose_bf16_nld_to_lnd",
        file = Some("layout/gather_rows.cuh"),
        launch = LaunchRule::Elementwise,
        operands = operands![
            src: U16s <- Source::In(0),
            dst: U16sMut <- Source::Out(0),
            n: I32 <- Source::Rows,
            layers: I32 <- Source::Div(
                &Source::Width(&Source::In(0)),
                &Source::CtxNonZero("ple_dim"),
            ),
            dim: I32 <- Source::Ctx("ple_dim"),
            total: Usize <- Source::OutElements(0),
        ]),
];
