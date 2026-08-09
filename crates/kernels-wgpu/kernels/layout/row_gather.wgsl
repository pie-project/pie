// Compact the rows a fire will actually sample.
//
// A prefill's stream is one row per TOKEN and its readout is one distribution
// per REQUEST, so the sampled rows are picked out before the lm head runs.
//
// ## The row this file exists to get right
//
// **There is no `@group(1)` here, and that is the contract, not an omission.**
// The row states five operands and the fifth, `count`, is `Ty::InPacked`: a
// value the driver must supply that gets no slot of its own, because it is the
// second FIELD of the `RowGatherParams` struct binding 3 already carries. The
// driver writes it while filling that buffer.
//
// Folding it into a uniform block instead would push a word no shader reads and
// leave `params.count` holding whatever the params buffer happened to contain --
// so the guard below would bound the gather by garbage. `kernels-vulkan`'s table
// reader made exactly that mistake, inheriting Metal's reading, where a packed
// slot IS the buffer and a trailing scalar lands in the same argument.
// `kernels_wgpu::bindings` answers `Binding::Packed` for it; `cargo run -p
// kernels-wgpu --example dump_layout -- row_gather` prints "0 bytes of uniform
// block", and this file agrees with it.
//
// ## A gather moves bits, so nothing here widens a bf16
//
// `common/bf16.inc.wgsl` is not included. A copy reproduces the bit pattern,
// and rounding it through f32 and back would be a lossless operation at best
// and a rounding of a NaN at worst. Both tensors are `array<u32>` holding two
// bf16 apiece, so one invocation moves one WORD -- columns `2c` and `2c+1` of
// its row -- which also makes the write race-free without a sub-word atomic
// WGSL does not have.
//
// That word is whole only if `width` is EVEN, which is the same thing every
// bf16 body in this tree needs: an odd row pitch means row `i+1` starts in the
// middle of row `i`'s last word, and no invocation can own it. Every hidden
// size the tree loads is a multiple of 64.

@group(0) @binding(0) var<storage, read> input_: array<u32>;
@group(0) @binding(1) var<storage, read_write> out_: array<u32>;
@group(0) @binding(2) var<storage, read> rows: array<u32>;
// Width then count, exactly as `layout/row_gather_params.glsl` and Metal's
// `row_gather_params.h` spell it. The statement states `[width]` and the driver
// appends the count, giving `[width, count]`.
struct RowGatherParams { width: u32, count: u32 }
@group(0) @binding(3) var<storage, read> params: RowGatherParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    let i = gid.y;
    let width = params.width;
    // `count` is the packed field, and this is the only thing that reads it.
    if (2u * c + 1u >= width || i >= params.count) { return; }

    let pitch = width >> 1u;
    let at = i * pitch + c;
    // The row count comes from a buffer the driver filled, so it can be right
    // where the grid is wrong. Guard the destination anyway: a WGSL store past
    // the end is clamped rather than dropped, which corrupts the last word
    // instead of doing nothing.
    if (at >= arrayLength(&out_)) { return; }

    out_[at] = input_[rows[i] * pitch + c];
}

// pie:instantiate row_gather_bfloat16
