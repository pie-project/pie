//! The dense bf16 projection, on the GPU, for the first time.
//!
//! `gemm/dense.metal` is `gemm.matmul`, `gemm.lm_head` and
//! `gemm.attention_landing` -- three of this plane's fifty-one points, one
//! arithmetic, `y[M, N] = act[M, K] @ w[N, K]^T`. On CUDA the same three are
//! cuBLAS. Here they are two hand-written kernels, and the file's own header
//! closed with the word UNVERIFIED: "No Metal toolchain and no Apple device
//! existed where this was written. It has not been compiled by `metal`, and
//! no number it computes has been compared against anything."
//!
//! # The reference is a CPU model, and the bound is the dot product's
//!
//! There is no CUDA on this machine, so the model is here. It cannot be
//! bit-exact and the reason is stated in the shader: `dense_gemm_t`
//! accumulates through `simdgroup_matrix` in `BK`-sized steps while
//! `dense_gemv_t` walks `K` in strides of 32, and a CPU model walks it in
//! ones. Three associations of one sum.
//!
//! So the bound is the one numerical analysis gives for a dot product rather
//! than one reasoned from the output's element: the error of an f32
//! accumulation is bounded by a small multiple of `u * Σ|a_k · b_k|`, and the
//! bf16 store on top of it by `2^-9 · |y|`, which is at most `2^-9` of that
//! same sum. `SCALE_BOUND` is `2^-8` of `Σ|a_k · b_k|` computed per output
//! element by the model, and the tail of each test REPORTS what fraction of
//! it the device actually used.
//!
//! That is not a relative bound on the answer, and it must not be turned into
//! one. A dot product whose terms cancel has an answer near zero and an error
//! that does not care: dividing by `|want|` there produces an enormous ratio
//! for an answer that is right to every bit that matters, and the hand that
//! then widens the bound to quiet it has widened it for every element.
//!
//! # Every shape here is RAGGED, because M never is not
//!
//! The tile kernel computes a whole `32 x 32` tile whatever the shape, and
//! bounds-checks the edges with `load_safe` and `store_result_safe`. A GEMM's
//! M is the token count -- 1 at decode, anything at prefill -- so a tile that
//! divides is the case that never happens. `M = 70`, `N = 50`, `K = 44`:
//! two whole row tiles and six rows over, one whole column tile and eighteen
//! columns over, one whole `BK` and a twelve-column K tail that is its own
//! trailing block. Nothing here is a multiple of anything.
//!
//! # The two bodies are fired at the SAME shape
//!
//! `kernels_metal::gemm::act_x_wt` picks between them on `M < TILE_M`, so
//! nothing in production ever runs both over one rectangle. They are two
//! spellings of one point and this tree does not take that on trust -- see
//! `sdpa_paged_mma`, where three bodies that answered the same softmax did
//! not share a contract -- so the third test below drives the tile kernel at
//! `M = 3`, which it will never be asked for and which its ragged path is
//! supposed to handle, and requires the two answers to agree.
//!
//! # The guard that is not decoration
//!
//! `vector_grid` rounds the column extent up to whole threadgroups -- 128
//! threads, four simdgroups, four columns -- so at `N = 50` the launch has
//! 52 columns' worth of simdgroups and two of them own no column. What stops
//! those two writing is `if (n >= N ...) return;`, and every result here is
//! allocated with a slack row that must still hold its poison when the fire
//! retires. Removing that guard is one of the mutations, and what it trips is
//! the slack rather than the comparison.

#![cfg(target_vendor = "apple")]

mod plane;

use driver_metal::skip::skipped;
use plane::{Arg, Rig};

const FILE: &str = "gemm/dense.metal";

const TILE: &str = "dense_gemm_t_bfloat16_bm_32_bn_32";
const VECTOR: &str = "dense_gemv_t_bfloat16";

/// Two whole row tiles and six rows over.
const M: usize = 70;
/// One whole column tile and eighteen columns over.
const N: usize = 50;
/// One whole `BK` and a twelve-column tail.
const K: usize = 44;

/// The decode shape: fewer rows than the tile, which is what sends
/// `act_x_wt` down the vector road.
const M_DECODE: usize = 3;

/// One bf16 step, against the sum of the dot product's term magnitudes.
const SCALE_BOUND: f32 = 1.0 / 256.0;

const POISON: f32 = -99.0;

/// A whole tile past the end of the rectangle.
///
/// Sized for the WORST mutation rather than for the kernel: swapping
/// `store_result_safe` for `store_result` makes the last `32 x 32` tile
/// write its whole self at a stride of `N`, which reaches 1263 elements past
/// a `70 x 50` result. A mutation that faults measures nothing, so the slack
/// is what makes that one a comparison instead of a crash.
const SLACK: usize = 32 * N + 64;

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_tile_kernel_lands_a_ragged_rectangle() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `gemm.matmul`'s tile road was not fired");
        return;
    };
    let model = Model::of(M);
    let got = model.fire(&rig, plane::kernels_dir().as_path(), TILE);
    model.agrees(&got, "gemm.matmul, the tile road at M=70 N=50 K=44");

    // The K tail, dropped. `K = 44` is one whole `BK` and twelve columns, and
    // the trailing block is what reads them; without it the projection is a
    // perfectly well-formed contraction over the first 32.
    model.bites(&rig, TILE, "if (k < K) {", "if (false) {");

    // The output tile's offset into the weight, dropped: every column tile
    // contracts against the bank's first `BN` rows.
    model.bites(
        &rig,
        TILE,
        "loader_b(w + size_t(y_col) * size_t(K), K, Bs, simd_gid, simd_lid)",
        "loader_b(w, K, Bs, simd_gid, simd_lid)",
    );

    // The ragged store, taken as a whole one. `store_result_safe` is what
    // keeps a `32 x 32` tile inside a `70 x 50` rectangle, and the six rows
    // and eighteen columns past the edge are where it matters.
    model.bites(
        &rig,
        TILE,
        "mma_op.store_result_safe(dst, N, short2(cols_left, rows_left));",
        "mma_op.store_result(dst, N);",
    );
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_vector_kernel_lands_the_decode_shape() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: `gemm.matmul`'s vector road was not fired");
        return;
    };
    let model = Model::of(M_DECODE);
    let got = model.fire(&rig, plane::kernels_dir().as_path(), VECTOR);
    model.agrees(&got, "gemm.matmul, the vector road at M=3 N=50 K=44");

    // The weight's row stride, dropped. Every output column contracts
    // against the same 44 elements of the bank, which is a rank-one answer
    // that is finite everywhere.
    model.bites(
        &rig,
        VECTOR,
        "const device T* w_row = w + size_t(n) * size_t(K);",
        "const device T* w_row = w + size_t(n);",
    );

    // The surplus simdgroups' guard. `vector_grid` launches 52 columns'
    // worth at `N = 50`, so what this trips is the slack past the rectangle
    // rather than any element inside it -- which is why the slack is checked
    // at all.
    model.bites(&rig, VECTOR, "if (n >= N || m >= M) {", "if (m >= M) {");
}

#[test]
#[ignore = "needs a Metal 4 device"]
fn the_two_roads_answer_the_same_projection() {
    let Some(rig) = Rig::open() else {
        skipped("no Metal 4 device: the two dense roads were not compared");
        return;
    };
    let model = Model::of(M_DECODE);
    let root = plane::kernels_dir();

    // The tile kernel at a shape the host would never send it: three rows
    // against a `BM` of 32, entirely inside the ragged path.
    let tiled = model.fire(&rig, root.as_path(), TILE);
    let vector = model.fire(&rig, root.as_path(), VECTOR);
    model.agrees(&tiled, "gemm.matmul, the tile road at the decode shape");

    let worst = model.against(&tiled, &vector);
    assert!(
        worst <= SCALE_BOUND,
        "the tile road and the vector road disagree by {worst} of the dot \
         product's own scale, past the {SCALE_BOUND} one bf16 step allows"
    );
    plane::measured(
        "gemm.matmul, tile against vector",
        &format!("worst {worst} against the scale bound {SCALE_BOUND}"),
    );
}

/// The reference, the fixture it was taken over, and the scale its bound is
/// taken against.
///
/// The operands live here rather than beside the tests because a mutation
/// takes five things -- a rig, an entry point, and the two halves of an edit
/// -- and everything else it needs is a property of the rectangle. A helper
/// that took the rectangle too would take eight.
struct Model {
    act: Vec<f32>,
    w: Vec<f32>,
    y: Vec<f32>,
    /// `sum |a_k * b_k|` per output element: the dot product's own magnitude,
    /// which is what bounds the error of accumulating it in any order.
    scale: Vec<f32>,
    rows: usize,
}

impl Model {
    /// `[rows, K]` and `[N, K]` at values a bf16 buffer holds exactly, and
    /// the projection of one against the other.
    ///
    /// 29 and 31 are prime and share no factor with 70, 50 or 44, so no two
    /// rows of either operand repeat within the rectangle and a dropped
    /// stride lands on a different number rather than on the same one.
    fn of(rows: usize) -> Self {
        let act: Vec<f32> = (0..rows * K)
            .map(|i| plane::narrowed(((i * 7) % 29) as f32 * 0.07 - 1.0))
            .collect();
        let w: Vec<f32> = (0..N * K)
            .map(|i| plane::narrowed(((i * 5) % 31) as f32 * 0.06 - 0.9))
            .collect();
        let mut y = vec![0.0; rows * N];
        let mut scale = vec![0.0; rows * N];
        for m in 0..rows {
            for n in 0..N {
                let mut acc = 0.0f32;
                let mut mag = 0.0f32;
                for k in 0..K {
                    let term = act[m * K + k] * w[n * K + k];
                    acc += term;
                    mag += term.abs();
                }
                y[m * N + n] = acc;
                scale[m * N + n] = mag;
            }
        }
        Self {
            act,
            w,
            y,
            scale,
            rows,
        }
    }

    /// The widest disagreement between two rectangles, as a fraction of each
    /// element's own scale.
    fn against(&self, got: &[f32], other: &[f32]) -> f32 {
        got.iter()
            .take(self.rows * N)
            .zip(other)
            .zip(&self.scale)
            .map(|((g, o), s)| (g - o).abs() / s.max(f32::MIN_POSITIVE))
            .fold(0.0, f32::max)
    }

    fn agrees(&self, got: &[f32], what: &str) {
        assert!(
            got[self.rows * N..].iter().all(|v| *v == POISON),
            "{what} wrote past the rectangle its point states"
        );
        let worst = self.against(got, &self.y);
        assert!(
            worst <= SCALE_BOUND,
            "{what}: the widest element is {worst} of the dot product\'s own \
             scale, past the {SCALE_BOUND} one bf16 step allows"
        );
        plane::tolerance_holds(worst, SCALE_BOUND, what);
        plane::measured(
            what,
            &format!("worst {worst} against the scale bound {SCALE_BOUND}"),
        );
    }

    /// Fire a SABOTAGED shader and require the same comparison to fail --
    /// either inside the rectangle, or by writing past it.
    fn bites(&self, rig: &Rig, symbol: &'static str, from: &str, to: &str) {
        let root = plane::mutant(FILE, from, to);
        let got = self.fire(rig, root.path(), symbol);
        let inside = self.against(&got, &self.y);
        let slack = got[self.rows * N..].iter().any(|v| *v != POISON);
        assert!(
            inside > SCALE_BOUND || slack,
            "replacing `{from}` with `{to}` left every element inside {inside} \
             of the scale bound and wrote nothing past the rectangle, so the \
             comparison above would not have caught it"
        );
        plane::measured(
            symbol,
            &format!(
                "`{from}` -> `{to}`: worst {inside} against the scale bound \
                 {SCALE_BOUND} inside the rectangle, slack {}",
                if slack { "overwritten" } else { "intact" }
            ),
        );
    }

    /// One dispatch, at the grid `kernels_metal::gemm` states for the entry
    /// point being fired.
    fn fire(&self, rig: &Rig, root: &std::path::Path, symbol: &'static str) -> Vec<f32> {
        let (grid, group) = if symbol == TILE {
            (
                [
                    N.div_ceil(32) as u32 * 32,
                    self.rows.div_ceil(32) as u32 * 2,
                    2,
                ],
                [32, 2, 2],
            )
        } else {
            (
                [N.div_ceil(4) as u32 * 128, self.rows as u32, 1],
                [128, 1, 1],
            )
        };
        let act = plane::alloc_bf16(&rig.context, &self.act, "act");
        let w = plane::alloc_bf16(&rig.context, &self.w, "w");
        let y = plane::alloc_bf16(&rig.context, &vec![POISON; self.rows * N + SLACK], "y");
        plane::fire(
            rig,
            root,
            FILE,
            symbol,
            grid,
            group,
            &[
                Arg::Buf(&act),
                Arg::Buf(&w),
                Arg::Buf(&y),
                Arg::I32(self.rows as i32),
                Arg::I32(N as i32),
                Arg::I32(K as i32),
            ],
        );
        plane::read_bf16(&y, self.rows * N + SLACK)
    }
}
