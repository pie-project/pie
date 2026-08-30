//! **THE TOWER'S EMBED MERGE IS A ROW COPY, AND THIS IS WHAT IT PROMISES.**
//!
//! ```text
//! cargo test -p kernels-cuda --features cuda-13 --test tower_scatter_rows -- --nocapture
//! ```
//!
//! The third of the multimodal design's three ops
//! (`.wiki/alto/multimodal.md` §2: `layout::scatter_rows(y, src, routes)` —
//! "tower output into the image-placeholder token rows, a one-line
//! gather/copy kernel") is a kernel this plane **already carries**:
//! `layout::scatter_rows`, the second half of `Fallback::Copy`. The M2 wave
//! wrote no new kernel for it, so what this file does is pin the properties
//! the tower merge is about to depend on — the ones the copy pair's own
//! callers never needed to state, because a fallback copy scatters back
//! exactly the rows it gathered and the merge will not.
//!
//! ```text
//! (a) row i of the tight rectangle lands at wide row routes[i], bit for bit,
//!     for a route vector that is a PERMUTATION and for one that is not
//!     (scattered, ascending, descending, and a single row)
//! (b) the rows no route names are NOT written — the whole reason the merge
//!     can land patch rows into a rectangle whose other rows are live text
//! (c) the round trip: gather by the same routes and get the tight rectangle
//!     back, so the two halves cannot drift into different maps
//! (d) the copy unit is an optimisation and never a rounding: the same
//!     permutation moves bf16 rows of 16, 2 and 3 elements (a 16-byte unit, a
//!     4-byte one, and the byte the odd width forces) and an f32 rectangle,
//!     and every one of them is bit-identical
//! (e) the refusals: a route vector that is not i32, and one whose length
//!     disagrees with the rows it is supposed to name
//! ```

#![cfg(feature = "_cuda")]

mod common;

use common::Gpu;

use dtype::Dtype;
use kernels_cuda::layout;
use kernels_cuda::tensor::Tensor;

/// The sentinel a wide row carries before anything scatters into it — a
/// pattern no payload word takes, so "not written" is checkable.
const UNTOUCHED: u16 = 0xa5a5;

const WIDE_ROWS: u32 = 9;

/// One copy shape, and the unit the entry will pick for it.
struct Shape {
    what: &'static str,
    width: u32,
    dtype: Dtype,
    /// Bytes per element of `dtype` — the harness moves raw words, so it
    /// needs the size the entry computes from the handle.
    element: usize,
}

const SHAPES: &[Shape] = &[
    Shape {
        what: "a 16-wide bf16 row (a 16-byte copy unit)",
        width: 16,
        dtype: Dtype::Bf16,
        element: 2,
    },
    Shape {
        what: "a 2-wide bf16 row (a 4-byte copy unit)",
        width: 2,
        dtype: Dtype::Bf16,
        element: 2,
    },
    Shape {
        what: "a 3-wide bf16 row (six bytes: the byte unit, the one every row admits)",
        width: 3,
        dtype: Dtype::Bf16,
        element: 2,
    },
    Shape {
        what: "an 8-wide f32 rectangle (the log-sum-exp shape the pair also serves)",
        width: 8,
        dtype: Dtype::F32,
        element: 4,
    },
];

/// Route vectors worth distinguishing: a scattered permutation, ascending
/// runs, a descending one, and a lone row.
const ROUTES: &[&[i32]] = &[&[5, 0, 8, 2], &[0, 1, 2, 3], &[8, 6, 4, 1], &[7]];

/// The words of tight row `i`, in the harness's raw currency (`u16` pairs,
/// which is what a bf16 element is and half of what an f32 element is).
fn payload(row: usize, words: usize) -> Vec<u16> {
    (0..words)
        .map(|word| u16::try_from((row * 977 + word * 31 + 1) & 0x7fff).expect("in range"))
        .collect()
}

/// Scatter `routes` and hand back the wide rectangle, in raw `u16` words.
fn scatter(shape: &Shape, routes: &[i32]) -> (Vec<u16>, Vec<u16>) {
    let mut gpu = Gpu::open();
    let words = shape.width as usize * shape.element / 2;
    let tight_rows = routes.len();

    let mut tight_words = Vec::with_capacity(tight_rows * words);
    for row in 0..tight_rows {
        tight_words.extend(payload(row, words));
    }
    let wide_words = vec![UNTOUCHED; WIDE_ROWS as usize * words];

    let tight_at = gpu.up(&tight_words);
    let wide_at = gpu.up(&wide_words);
    let routes_at = gpu.up(routes);

    let tight = Tensor::new(
        tight_at,
        u32::try_from(tight_rows).expect("a test's rows fit"),
        shape.width,
        shape.dtype,
    );
    let mut wide = Tensor::new(wide_at, WIDE_ROWS, shape.width, shape.dtype);
    let index = Tensor::new(
        routes_at,
        u32::try_from(routes.len()).expect("a test's routes fit"),
        1,
        Dtype::I32,
    );
    layout::scatter_rows(&gpu.ctx(), tight, index, &mut wide).expect("the scatter enqueues");
    gpu.sync();

    // And back the other way, into a rectangle of its own, so the round trip
    // reads the map rather than the bytes it just wrote.
    let back_at = gpu.zeros(tight_words.len() * core::mem::size_of::<u16>());
    let mut back = Tensor::new(
        back_at,
        u32::try_from(tight_rows).expect("a test's rows fit"),
        shape.width,
        shape.dtype,
    );
    layout::gather_rows(
        &gpu.ctx(),
        Tensor::new(wide_at, WIDE_ROWS, shape.width, shape.dtype),
        index,
        &mut back,
    )
    .expect("the gather enqueues");
    gpu.sync();

    (
        gpu.down(wide_at, wide_words.len()),
        gpu.down(back_at, tight_words.len()),
    )
}

/// (a), (b), (c) and (d), over every shape and every route vector.
#[test]
fn a_scattered_row_lands_where_its_route_names_and_nowhere_else() {
    for shape in SHAPES {
        let words = shape.width as usize * shape.element / 2;
        for routes in ROUTES {
            let (wide, back) = scatter(shape, routes);

            for (row, route) in routes.iter().enumerate() {
                let at = *route as usize * words;
                assert_eq!(
                    &wide[at..at + words],
                    &payload(row, words)[..],
                    "{}: tight row {row} did not land at wide row {route}",
                    shape.what
                );
            }

            for row in 0..WIDE_ROWS as usize {
                if routes.contains(&i32::try_from(row).expect("in range")) {
                    continue;
                }
                for word in 0..words {
                    assert_eq!(
                        wide[row * words + word],
                        UNTOUCHED,
                        "{}: wide row {row} is nobody's route and was written",
                        shape.what
                    );
                }
            }

            for row in 0..routes.len() {
                assert_eq!(
                    &back[row * words..(row + 1) * words],
                    &payload(row, words)[..],
                    "{}: the round trip did not return tight row {row}",
                    shape.what
                );
            }
        }
    }
}

/// (e): the refusals a merge with a mis-assembled route vector must hit.
#[test]
fn the_refusals_fire_by_name() {
    let mut gpu = Gpu::open();
    let ctx = gpu.ctx();
    let width = 8u32;
    let tight_at = gpu.zeros(4 * width as usize * 2);
    let wide_at = gpu.zeros(WIDE_ROWS as usize * width as usize * 2);
    let routes_at = gpu.up(&[0i32, 1, 2, 3]);

    let tight = Tensor::new(tight_at, 4, width, Dtype::Bf16);
    let mut wide = Tensor::new(wide_at, WIDE_ROWS, width, Dtype::Bf16);

    let wrong_dtype = layout::scatter_rows(
        &ctx,
        tight,
        Tensor::new(routes_at, 4, 1, Dtype::F32),
        &mut wide,
    );
    assert!(
        wrong_dtype.is_err(),
        "a route vector that is not i32 is refused"
    );

    let wrong_length = layout::scatter_rows(
        &ctx,
        tight,
        Tensor::new(routes_at, 3, 1, Dtype::I32),
        &mut wide,
    );
    assert!(
        wrong_length.is_err(),
        "a route vector that names fewer rows than there are to move is refused"
    );
}
