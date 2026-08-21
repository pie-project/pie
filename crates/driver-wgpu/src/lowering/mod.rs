//! Turning a statement into the two things a dispatch needs.
//!
//! This is where WebGPU diverges from Metal, and the divergence is not
//! cosmetic. Metal binds every operand to a numbered buffer slot, scalars
//! included — a `setBytes` at index `k` for the `k`-th argument — so a driver
//! there has one run to fill and its order IS the answer. WebGPU splits
//! the same list in two: buffers become entries of a `@group(0)` bind group,
//! scalars become fields of the ONE `@group(1) @binding(0)` uniform block, and
//! NEITHER run is indexed by the operand's position in the list.
//!
//! Vulkan splits it in two as well and puts the scalars somewhere else again —
//! a `layout(push_constant)` block, which does not exist here. `wgpu` offers
//! push constants as `Features::PUSH_CONSTANTS`, a native-only extension a
//! browser cannot provide, so a driver that used them would run on `wgpu` and
//! not on WebGPU. The uniform buffer is the whole of this backend's answer.
//!
//! So the mapping has to be computed, and computing it wrongly has no symptom.
//! A buffer bound one entry over reads the wrong tensor and produces numbers —
//! and `wgpu` will not object, because a bind group is typed by its LAYOUT and
//! every operand here is a storage buffer, so a shuffled set matches. A scalar
//! packed at the wrong offset reads a stride where a head count belongs and
//! produces numbers. Nothing returns an error and no layer complains, which is
//! why the arithmetic lives here, in one place, checked — rather than at each
//! of the call sites that would otherwise each get it slightly right.

// Lowerings, kept by the fire shape that produced them. Its own docs say why;
// they are `//!` and not `///` deliberately, because a doc comment HERE is
// merged with the module's and then resolves its intra-doc links in THIS
// scope, where `Shape` and `CAP` do not exist. Three broken links, caught by
// `cargo doc -D warnings`.
// The routine plane's driver side: one arm per crossed kernel, turning a
// traced statement into the argument list a body takes. Stage 2 of
// `refactor-bigplan.md` §7.
pub mod hold;

/// Binding a launch from the signature, through the shared reader.
pub mod bind;

/// The raised views the binder builds for `In<Struct<..>>` operands.
pub mod views;

pub mod cached;

// Running a crossed body to get a dispatch: the other half of the routine
// plane, where `arm` is the first.
pub mod routine;

// RETIRED: `pack`, and the `Value`/`Call`/`Mismatch` vocabulary it spoke in.
//
// It had no production caller left once `plan_one` stopped forking, and that
// is the occasion rather than the whole fact: `pack` never had an in-crate
// caller at any point in its life. `dispatch.rs` did not call it before the
// fork went and does not now. It was live only as `pub use lowering::{Call,
// Mismatch, Value, pack}` at the crate root -- a door held open for a caller
// that still held a `KernelSig` -- and every call in the tree was in this
// file's own `mod tests`. `driver-metal` deleted its table interpreter first
// and `driver-vulkan` deleted its whole `lowering.rs` second, 539 lines and
// the same `Call`/`Value`/`Mismatch` vocabulary. This is the third and last.
//
// # What it did
//
// It took a row's operand description and a positional run of `Value`s and
// split them into the two runs a WebGPU dispatch needs.
// `kernels_wgpu::bindings(sig)` said which run each operand belonged to, and
// buffers went to the `@group(0)` list in BINDING order, which is not the
// row's order.
//
// The uniform block is the part with arithmetic in it. `pack` sized it from
// `kernels_wgpu::uniform_size(sig)`, zero-filled it, and wrote each scalar at
// the offset `uniform_layout(sig)` gave that field -- never by appending,
// because the block HAS holes and a run built end to end closes them and
// moves every field behind them. WGSL has no 64-bit integer of any kind, so
// `kernels-wgpu` declares a `Usize` or `I64` operand as `vec2<u32>`, and a
// `vec2<u32>` is eight-ALIGNED as well as eight wide: `kv_append`'s
// `head_dim: i32` sits at 0, the four bytes behind it are padding nothing
// writes, and its two strides start at 8 and 16. Adding the widths gives 20.
// The layout said 24, and the uniform address space rounds the binding to 32,
// because it gives every host-shareable struct an alignment of at least 16
// and `wgpu` refuses a binding whose size is not a multiple of it. A 64-bit
// value crossed as two little-endian `u32` words, low word first, and was
// CONSTRUCTED as two words rather than stored as one `u64`: byte for byte the
// same, and the spelling was the argument, because the shader does not read a
// 64-bit integer -- it reads two words and rebuilds `x + y * 2^32`.
//
// Nothing was placed until everything was checked, so a call bound whole or
// not at all -- a dispatch with some operands placed is a dispatch that reads
// whatever was in the others. `Mismatch::Unstated` refused a row with an
// empty operand list, which cannot say what a call looks like; answering with
// an empty `Call` would have been a dispatch that binds nothing and reports
// success. `Mismatch::Arity` refused a value count that was not the row's
// operand count, in either direction. `Mismatch::Kind` refused a value of the
// wrong kind for the operand it was handed to, and it is the one that
// mattered: a scalar where a buffer belongs binds every later tensor one
// entry early and `wgpu` accepts it, because a bind group is typed by its
// LAYOUT and every operand here is a storage buffer, so a shuffled set
// matches; and an eight-byte value in a four-byte field does not truncate, it
// overwrites its neighbour. A `Binding::Packed` operand took neither run and
// came back to the caller as `(index, value)` -- it is a FIELD of a struct an
// earlier buffer binds, so folding it into the uniform run would have pushed
// a word no shader reads AND left the struct member unwritten, one mistake
// producing two wrong things.
//
// # Where the same work happens now
//
// `routine::bind`, in this directory. It makes the same uniform run from a
// body's `ArgValue`s, in the order the body passed them, aligning each to its
// own width -- four for a word, eight for a `Usize` -- and states the same
// WGSL rule for the same reason; `encode::Encoder::block` carries it on the
// device path. It pads to `reflect::Declared::uniform_bytes`, which is what
// `naga` read out of the module, where `pack` padded to what a row derived.
// Buffers come from the body's handles and scalars from its `ArgValue`s, so
// neither run can renumber the other, and it refuses by name through
// `Unplanned::{Handle, Scalars, Blocks, Operand, Silent, NoCache, Absent}`.
// Its unit tests are `a_usize_scalar_is_eight_aligned_in_the_block`,
// `scalars_wider_than_the_modules_block_are_refused_by_name` and
// `a_handle_past_what_the_arm_minted_is_refused_by_name`; over the fleet,
// `tests/arena.rs`'s
// `every_launchs_scalars_land_where_its_module_reads_them` holds the offsets
// a body's scalars pack to against the offsets `naga` says its module reads
// at, for every rectangle of every text.
//
// # What is lost, and not merely moved
//
// `Mismatch::Kind` has no counterpart anywhere. `bind` never compares a value
// against a declared operand type, because there is no declaration to compare
// against -- the body IS the statement of the ABI. A buffer where a scalar
// belongs is a Rust type error and needs no run-time refusal, but a `U32`
// where the module declares `vec2<u32>` is not one: `bind` aligns it to four,
// writes four, and every field behind it moves. `pack` asserted the row's
// width against the field's width at every field it wrote; nothing asserts
// that at the seam now. What catches it is the fleet walk,
// `every_launchs_scalars_land_where_its_module_reads_them`, over the plans
// six texts happen to build -- and it compares a PREFIX, so a body passing
// too FEW scalars is collected into `short` and PRINTED, where
// `Mismatch::Arity` refused it. Too many is `Unplanned::Scalars`, which
// counts BYTES against the module's span rather than values against a
// declared count, so a run short one field and long another is invisible to
// it.
//
// The eight-byte alignment is covered more narrowly than it reads. That same
// walk asserts `eight_byte == 0`: no body any of those texts launches passes
// an `ArgValue::Usize` at all, because the shapes that do -- `kv_append` and
// the contiguous vector decodes -- want a cache none of them configures. So
// the rule this module was built around is witnessed by one hand-written
// `Stated` in `routine.rs` and by no real plan, where `pack`'s offsets came
// from `uniform_layout` over a transcribed row. The word ORDER goes with it:
// `bind` writes `v.to_le_bytes()`, so there are no halves in the code to swap
// and no test names them. It is right, and it is no longer argued.
//
// `Binding::Packed`'s hand-back is gone too, and what replaced it is checked
// by length only. A body wanting a field of somebody else's struct passes a
// scalar, and `bind` concatenates the statement's run and the body's scalars
// into `Params::Block { at: ParamSlot::Storage(_) }`; the fleet walk holds
// that block's total SIZE against `Declared::block_bytes[at]` and never asks
// where inside it each value landed. `row_gather`'s `count` is checked to be
// present, not to be in the right place.
//
// And the ceiling check that needed no GPU is gone.
// `every_block_this_packs_is_one_webgpu_will_bind` asserted, with no adapter
// and no fixture, that every block this crate packs is a multiple of 16 and
// inside WebGPU's guaranteed 16 KiB binding size. `bind` deliberately does
// not round -- it pads to `naga`'s span for the struct, 20 bytes for five
// words -- so both the rounding and the limit live in `Device::uniform` now,
// behind the `native` feature, measured against a real adapter rather than
// against the guaranteed floor. `Ceiling::UniformBinding` is on
// `tests/citations.rs`'s UNNAMED list: no test names it.
//
// The cheapness of the proof goes with all of it. `pack` settled a property
// of the ROW, so a `KernelSig` written down in a test file and a list of
// values were the whole apparatus -- no device, no plan, no `naga`, and a row
// the table already had needed no code here to receive it. Everything that
// replaced it costs more to ask: `bind`'s unit tests build a `Stated` and a
// `Declared` by hand, the fleet walk wants six lowered texts, a placeholder
// arena and a resolver before it can plan the 6680 rectangles it questions,
// and a kernel arrives with an arm and a body rather than with a line. That
// is a better test and a worse instrument -- the row was a thing a reader
// could state in eight lines, and a body is not.
//
// `Mismatch::Unstated` is the one thing here that is not a loss. It refused a
// row with no operands, and there are no rows; a symbol no stem claims is
// `Undispatchable::Unknown` at `plan_one`, by name, at the same seam.

#[cfg(test)]
mod tests {
    // RETIRED: THE TABLE IS EMPTY, so the walk has nothing to walk.
    //
    // It asserted the claim this module exists for, over the whole table at
    // once: every row that stated its operands packed from its OWN
    // description, with no arm written for any of it -- 43 stated rows, ten
    // operand kinds, and a `pack` that matches on the operand's KIND and
    // never on the kernel's name, so a row the table already had needed no
    // code here to receive it. Each row's call had to come out with
    // `kernels_wgpu::storage_count(sig)` buffers and a
    // `kernels_wgpu::uniform_size(sig)` block, and the floor of ten was there
    // so the sweep could not quietly shrink.
    //
    // It did NOT go blind, and the floor is why: `for sig in KERNELS` over an
    // empty table runs its body zero times, so `packed` stayed at zero and
    // `assert!(packed >= 10)` failed rather than an empty loop passing in
    // silence. Retiring it converts a loud failure into a recorded absence,
    // which is the honest trade and not a repair -- the sweep is gone because
    // its subject is gone, not because it started agreeing.
    //
    // The same claim is made per kernel on the routine plane, and by the same
    // means: `crate::lowering::routine::bind` splits a body's arguments into
    // buffers (from its handles) and a scalar run (from its `ArgValue`s) with
    // no match on any kernel's name either, refusing by name through
    // `Unplanned::{Handle,Scalars,Operand,Silent,Blocks,Absent,NoCache}` where
    // this refused through `Mismatch::{Unstated,Arity,Kind}`. What is walked
    // instead of a table is `crate::lowering::hold`'s `CROSSED` registry -- one
    // stem per crossed kernel -- and `driver-wgpu/tests/arena.rs`'s
    // `every_launchs_scalars_land_where_its_module_reads_them` is what derived
    // every field of a rectangle twice, by the row and by the arm, for as long
    // as there were rows to derive it from.
}
