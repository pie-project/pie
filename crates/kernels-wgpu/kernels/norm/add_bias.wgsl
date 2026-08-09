// Add a per-column bias to every row, in place: `out[r][c] += bias[c]`.
//
// The Qwen-2 family's attention biases, and the reason this file exists at all:
// the shared text omitted them because no Metal kernel added a bias, and a
// Qwen2 computed without its q/k/v biases is not a crash and not a NaN -- it is
// fluent, wrong text.
//
// # Why the grid is two-dimensional and `residual_add`'s is not
//
// A residual is elementwise against a buffer of the same extent, so a flat
// index over `rows * width` reads both operands correctly. A bias is
// BROADCAST: one vector of `width`, re-read by every row. The column has to be
// recoverable from the invocation, and `LaunchRule::RouteRows` gives it
// directly -- it dispatches `[width, rows, 1]`, so `x` is the column and `y` is
// the row.
//
// A flat grid with a `% width` would work and is worse: the modulus is a
// division per element, and it puts the width in the shader where the launch
// rule already carries it.
//
// # Why two buffers and not three
//
// The op is IN PLACE, and not by accident: the row declares `in_place =
// &[(0, 0)]` and sources its only mutable buffer from `Out(0)`, so the trace
// hands every backend the SAME allocation for the value and the result. A
// three-buffer shader here would bind that one allocation twice and read a row
// it had already written -- correct only by luck of the launch order.
//
// # The bf16 pairing, which is this backend's divergence — and the bug it had
//
// `kernels-vulkan`'s copy of this has `uint16_t[]` and one invocation per
// COLUMN. WGSL has no 16-bit storage type, so `out_` is `array<u32>` holding
// two columns per word — and a read-modify-write of one half would race the
// lane that owns the other. WGSL gives no sub-word atomic, so there is no way
// to make that safe from inside.
//
// The first draft answered that by giving one invocation a PAIR OF COLUMNS,
// `2*x` and `2*x + 1`. That is wrong at an odd `width`, and a GPU test at
// `width = 461` found it: a row's elements start at `row * width`, which is odd
// on every odd row, so a column pair is not a WORD. Twelve of thirteen rows
// disagreed, every even row at its last column and every odd row at its first,
// each holding the other's value — and the word they shared was written by two
// invocations at once, so which value survived was a race and the failure was
// not even stable.
//
// The fix is to own the WORD instead. `x` counts words, not columns; a word is
// owned by the row that owns its LOW half, so every word has exactly one
// writer whatever the width; and each half's column is derived from its own
// element index, so the word straddling two rows gets each half the bias its
// own row asked for.
//
// That costs a modulus per half. `kernels-vulkan`'s copy says a flat grid with
// a `% width` "would work and is worse", and it is right about ITS shader,
// where one invocation is one element and words do not exist. Here the modulus
// is what makes the odd case correct, and correct is not negotiable against a
// division.
//
// # The bounds test is not decoration
//
// The grid is rounded up to whole workgroups AND to whole words, so lanes run
// past the row in both directions. The buffer is WRITTEN, so an unguarded lane
// does not merely read a zero back — it biases a column of another row a second
// time. Neither is reported.
//
// The high half of the buffer's last word may be padding, when `rows * width`
// is odd. Biasing it is harmless because nothing reads it, and the caller
// allocates whole words by construction: a `array<u32>` cannot hold half a
// word.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> out_: array<u32>;
@group(0) @binding(1) var<storage, read> bias: array<u32>;

struct Params { width: i32 }
@group(1) @binding(0) var<uniform> params: Params;

// One element, biased by the column it actually sits in.
//
// The column comes from the ELEMENT index rather than from the invocation,
// which is the whole content of the fix: the word this invocation owns may
// straddle two rows, and then its two halves are column `width - 1` of one row
// and column `0` of the next.
fn biased(word: u32, e: u32, width: u32) -> f32 {
    let col = e % width;
    return pie_bf16_at(word, e) + pie_bf16_at(bias[col >> 1u], col);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = u32(params.width);
    let row = gid.y;
    let base = row * width;

    // `x` counts WORDS of this row's span, from the word holding its first
    // element. The launch rule dispatches `width` lanes on x, and a row spans
    // at most `width / 2 + 1` words, so this always over-dispatches — which is
    // the safe direction, and what the guards below are for.
    let w = (base >> 1u) + gid.x;
    if (w >= arrayLength(&out_)) { return; }

    let lo_e = w * 2u;

    // One word, one writer. The row owning the word's LOW half owns the word.
    //
    // At an odd width the word at `base >> 1` holds the PREVIOUS row's last
    // column in its low half, so without this line that word has two writers.
    //
    // Removing it does NOT fail the odd-width test, and the reason is worth
    // stating rather than hiding: `biased` derives each half's column from its
    // own element index, so it is row-agnostic — both writers compute the same
    // two values and the bytes agree. What the line buys is that the write is
    // not a race at all. WGSL gives no sub-word atomic and no ordering between
    // two invocations writing one location; "they happen to write the same
    // bytes" is a property of this body that the memory model does not know
    // about, and the next edit to `biased` could quietly stop it being true.
    //
    // So this is a guard against a future change, not against a measured
    // failure, and it is cheap. It is also what makes the paragraph above
    // honest: the first draft of this file claimed a guard prevented a
    // cross-row write when it did not, and a GPU test found the difference.
    if (lo_e / width != row) { return; }

    // Past this row's span: the round-up on x.
    if (lo_e >= base + width) { return; }

    let word = out_[w];
    out_[w] = pie_pack_bf16(
        biased(word, lo_e, width),
        biased(word, lo_e + 1u, width),
    );
}

// pie:instantiate add_bias_bfloat16
