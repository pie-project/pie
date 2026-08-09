//! Does `pie_mma.cuh` compute what `nvcuda::wmma` computes?
//!
//! # The question this answers, and why compiling is not it
//!
//! `csrc/src/pie_mma.cuh` restates one WMMA shape --
//! `16x16x16`, bf16 in, fp32 accumulate -- over the `mma.sync.aligned.m16n8k16`
//! instruction, because `examples/header_probe.rs` measured NVRTC 13.0
//! refusing `mma.h` and vendoring it is a redistribution decision this crate
//! exists to avoid. The header's whole content is a per-lane REGISTER MAP:
//! which of the 256 elements of a tile each of the 32 lanes holds, and in
//! which of its registers.
//!
//! A map that is wrong compiles. `ptxas` accepts any assignment of registers
//! to lanes; the instruction runs; the kernel returns numbers. A transposed
//! store, an A fragment whose two k-halves are swapped, a B whose column is
//! `groupID` where it should be `threadID_in_group` -- every one of those is a
//! **silent wrong answer**, and in a MoE decode path it is a wrong answer that
//! looks like a slightly worse model rather than like a bug. So a probe that
//! merely compiles the shim proves nothing at all, and the only gate worth
//! having is a numeric comparison against the thing being replaced.
//!
//! # Two paths, one body
//!
//! Both kernels below are the SAME nine lines of `wmma` calls. The only
//! difference between the two translation units is the four lines above them:
//!
//! * **reference** -- `#include <cuda_bf16.h>` and `#include <mma.h>`,
//!   compiled by the machine's own `nvcc` to a cubin. This is the path that
//!   catches a layout error, and it is why the toolkit is shelled out to
//!   rather than a host matmul being trusted: a host reference checks the
//!   MATH, and every mistake this file is looking for produces the right
//!   arithmetic on the wrong elements.
//! * **under test** -- `#include "pie_mma.cuh"`, compiled by NVRTC against a
//!   header set built here out of [`kernels_cuda_new::source`], with no
//!   include path on disk. Two headers, deliberately: the prelude the shim
//!   takes its `bf16` from, and the shim. If a third were needed the compile
//!   would fail, which is itself part of the claim.
//!
//! Both then run on the same device on the same bytes, and the results are
//! compared **element by element with no tolerance**. Same instruction, same
//! hardware, same accumulate order, so the difference must be exactly zero;
//! a tolerance here would be a way of not noticing that it was not.
//!
//! A host fp32 matmul runs beside them as a third opinion. Three of the four
//! cases are built from multiples of `1/8`, for which every partial product
//! and every partial sum is exact in fp32 -- so on those the host must agree
//! to the bit as well, and does. The full-entropy case spreads its exponents
//! over sixteen octaves so that the accumulation is genuinely lossy; its host
//! difference is reported and not gated on, because a tensor core's internal
//! accumulation is not obliged to match a sequential fp32 loop, and the point
//! of the case is that the two KERNELS still agree exactly while the host
//! does not.
//!
//! # The four cases, and what each one catches
//!
//! | case | the mistake it catches |
//! |---|---|
//! | exact grid | the map is wrong somewhere, anywhere |
//! | full bf16 entropy | a map that is right only for values with short mantissas |
//! | `A = I` | a transposed store, or an A/B confusion -- the result IS `B` |
//! | one non-zero `B` column | a B column map off by a lane group; every other output column must be exactly zero |
//!
//! and then one more, which is about this file rather than about the header:
//! the shim is compiled a second time with the store's layout flipped, and
//! the comparison is required to CATCH it. Four passes prove nothing if the
//! harness cannot produce a failure.
//!
//! ```text
//! cargo run -p kernels-cuda-new --features cuda-13 --example mma_probe
//! ```
//!
//! # It exits non-zero, which is the only reason it can be gated on
//!
//! This probe printed its verdict and exited 0 either way for as long as it
//! existed, which made it a report to a human who had to be reading rather
//! than something CI could hold. It now returns a verdict:
//!
//! * **1** — a comparison differed, or the transposed-store control was NOT
//!   caught. Four passes are worth nothing if the harness cannot fail, so an
//!   unmeasured control is a failure and not a shrug;
//! * **0** — all four bit-identical and the control caught. Also 0 when there
//!   is no device or no `nvcc`, because those are skips: this crate's normal
//!   state is a machine with neither, and `tests/fire.rs` draws the same line.
//!   The skip says so on stdout.
//!
//! Verified by mutation rather than by reading: changing the store's row map
//! from `g + 8 * (i >> 1)` to `g + 4 * (i >> 1)` — a lane-map error of exactly
//! the class this file exists to catch — reports `PARITY FAILED on 4 of 4`
//! and exits 1.
//!
//! Needs `nvcc` on `PATH` or at `$CUDA_HOME/bin/nvcc`, and a device of
//! `sm_80` or newer -- `mma.sync` with bf16 operands does not exist before
//! Ampere.

#[cfg(not(feature = "_cuda"))]
fn main() {
    // Declared with no `required-features` in `Cargo.toml`, which this file
    // does not own, so a default-feature `cargo test` compiles it. The gate
    // is here instead: layers 1 and 2 build with no CUDA at all, and a probe
    // that exists to show the toolkit is unnecessary must not be the thing
    // that drags it in.
    println!(
        "mma_probe needs layer 3: cargo run -p kernels-cuda-new --features cuda-13 \
         --example mma_probe"
    );
}

#[cfg(feature = "_cuda")]
fn main() {
    // Exit code, so this can be gated on. Everything above prints its own
    // reasoning for a reader; the code is for whatever runs it unattended.
    if !probe::run() {
        std::process::exit(1);
    }
}

#[cfg(feature = "_cuda")]
mod probe {
    use std::ffi::{CStr, CString, c_void};
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::Instant;

    use cudarc::driver::sys as dr;
    use cudarc::nvrtc::sys as nv;
    use cudarc::runtime::sys as rt;

    use kernels_cuda_new::source::{DEVICE_HEADERS, Header, as_nvrtc_arrays};

    /// The shim, carried the way every other device source in this crate is.
    ///
    /// `include_str!` and not a read: the bytes compared here are the bytes
    /// that ship, so a probe that passed against a file on disk while the
    /// binary carried something else would be measuring the wrong header.
    const PIE_MMA: &str = include_str!("../csrc/src/pie_mma.cuh");

    /// The nine lines both paths run, verbatim.
    ///
    /// Shared as one string rather than written twice, because the claim is
    /// that these calls mean the same thing in both worlds -- and two copies
    /// of them would let a typo make that true by accident.
    const BODY: &str = concat!(
        "extern \"C\" __global__ void mma16(\n",
        "    const __nv_bfloat16* a, const __nv_bfloat16* b, float* c)\n",
        "{\n",
        "    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> fa;\n",
        "    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> fb;\n",
        "    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;\n",
        "    wmma::fill_fragment(acc, 0.0f);\n",
        "    wmma::load_matrix_sync(fa, a, 16);\n",
        "    wmma::load_matrix_sync(fb, b, 16);\n",
        "    wmma::mma_sync(acc, fa, fb, acc);\n",
        "    wmma::store_matrix_sync(c, acc, 16, wmma::mem_row_major);\n",
        "}\n",
    );

    /// The tile edge. One number, because the shim implements one shape --
    /// see its header for why that is a bound and not a simplification.
    const N: usize = 16;

    /// The one column left non-zero by the fourth case. Odd, and not a
    /// multiple of eight, so that it is neither the first column of the tile
    /// nor the first of either 16x8 half -- a B map that dropped `groupID`
    /// or confused the two halves has to move it.
    const LIVE_COLUMN: usize = 5;

    /// One comparison, and the mistake it exists to catch.
    struct Case {
        /// What appears in the report.
        what: &'static str,
        /// `A`, row-major, as bf16 bits.
        a: Vec<u16>,
        /// `B`, column-major -- element `(k, n)` at index `n * 16 + k`.
        b: Vec<u16>,
        /// Whether the host fp32 matmul must agree to the bit, which is true
        /// exactly when the inputs are multiples of `1/8`: every partial
        /// product is then a multiple of `1/64` under 1, and sixteen of them
        /// sum well inside fp32's exact-integer range.
        host_is_exact: bool,
    }

    /// Runs every check. `true` only when all four agree exactly AND the
    /// transposed-store control was caught.
    pub fn run() -> bool {
        let arch = match kernels_cuda_new::jit::cache::arch() {
            Some(arch) => arch,
            None => {
                // A skip, not a failure -- `tests/fire.rs` draws the same line
                // and for the same reason: "could not check here" must not be
                // reported as "checked and wrong". The caller gets `true` and
                // the reader gets the sentence.
                println!("SKIP: no CUDA device is current; this probe needs one to compare on");
                return true;
            }
        };

        println!("pie_mma.cuh parity probe -- the shim against the real nvcuda::wmma\n");
        println!("  device        {} ({arch})", device_name());
        println!("  NVRTC         {}", nvrtc_version());
        let Some(nvcc) = find_nvcc() else {
            println!(
                "\n  no `nvcc` on PATH or at $CUDA_HOME/bin/nvcc.\n\
                 The reference path IS the check -- a host matmul would agree with a\n\
                 transposed store -- so there is nothing useful to report without it."
            );
            // Also a skip: the toolkit's absence is this crate's normal state,
            // and the reference path is what needs it, not the shim.
            return true;
        };
        println!("  nvcc          {}", nvcc.display());
        println!("  pie_mma.cuh   {} bytes", PIE_MMA.len());

        // The set the shim resolves against: the prelude it takes `bf16`
        // from, and itself. Nothing else, which is the point -- a third entry
        // would mean the shim had grown a dependency the two call sites do
        // not have.
        let Some(prelude) = DEVICE_HEADERS.iter().find(|h| h.name == "pie_device.cuh") else {
            println!("the header set has no `pie_device.cuh`, which the shim includes");
            return false;
        };
        let headers = [*prelude, Header { name: "pie_mma.cuh", text: PIE_MMA }];
        println!(
            "  header set    {}",
            headers
                .iter()
                .map(|h| format!("{} ({} B)", h.name, h.text.len()))
                .collect::<Vec<_>>()
                .join(", ")
        );

        let reference_source =
            format!("#include <cuda_bf16.h>\n#include <mma.h>\nusing namespace nvcuda;\n\n{BODY}");
        let under_test_source =
            format!("#include \"pie_mma.cuh\"\nusing namespace nvcuda;\n\n{BODY}\n");

        println!("\ncompiling:\n");
        let reference = match compile_with_nvcc(&nvcc, &reference_source, arch) {
            Ok(built) => {
                println!("  reference   nvcc -cubin, <mma.h>          {:8.1} ms", built.millis);
                built
            }
            Err(why) => {
                println!("  reference   nvcc REFUSED: {why}");
                return false;
            }
        };
        let under_test = match compile_with_nvrtc(&under_test_source, arch, &headers) {
            Ok(built) => {
                println!("  under test  NVRTC, \"pie_mma.cuh\"         {:8.1} ms", built.millis);
                built
            }
            Err(why) => {
                println!("  under test  NVRTC REFUSED:\n{why}");
                return false;
            }
        };

        let reference = match Module::load(&reference.image) {
            Ok(module) => module,
            Err(why) => {
                println!("\nloading the reference cubin: {why}");
                return false;
            }
        };
        let under_test = match Module::load(&under_test.image) {
            Ok(module) => module,
            Err(why) => {
                println!("\nloading the shim's cubin: {why}");
                return false;
            }
        };

        println!(
            "\n{:<44} {:>6}  {:>13}  {:>15}  {:>13}",
            "check", "result", "max |Δ| wmma", "first differing", "host rel |Δ|"
        );
        println!("{}", "-".repeat(100));

        let mut failures = 0usize;
        let cases = cases();
        let total = cases.len();
        for case in &cases {
            let (Ok(want), Ok(got)) =
                (reference.run(&case.a, &case.b), under_test.run(&case.a, &case.b))
            else {
                println!("{:<44} {:>6}  a launch failed", case.what, "ERROR");
                failures += 1;
                continue;
            };

            let (max_delta, first_bad) = compare(&want, &got);
            let host = host_matmul(&case.a, &case.b);
            let (host_delta, _) = compare(&host, &got);
            // Relative, and absolute would mislead: the entropy case spans
            // sixteen octaves, so its tile reaches ~1e10 and a one-ulp
            // disagreement prints as a four-figure absolute number that looks
            // like a defect. Scaled by the tile's own magnitude it reads as
            // what it is -- fp32 epsilon.
            let scale = host.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let host_rel = if scale == 0.0 { 0.0 } else { host_delta / scale };

            let mut passed = max_delta == 0.0;
            if case.host_is_exact && host_delta != 0.0 {
                passed = false;
            }
            if !contract_holds(case, &got) {
                passed = false;
            }
            if !passed {
                failures += 1;
            }

            println!(
                "{:<44} {:>6}  {:>13}  {:>15}  {:>13}",
                case.what,
                if passed { "PASS" } else { "FAIL" },
                format_delta(max_delta),
                first_bad.map_or_else(|| "--".to_string(), |at| format!("[{},{}]", at / N, at % N)),
                format_delta(host_rel),
            );
        }
        println!("{}", "-".repeat(100));

        if failures == 0 {
            println!(
                "PARITY: bit-identical on {total} of {total} checks. The shim's lane maps\n\
                 agree with `nvcuda::wmma` exactly, on this device, for this shape.\n\
                 `host rel |Δ|` is the same tile against a sequential fp32 matmul, scaled by\n\
                 the tile's magnitude: zero where the inputs make the arithmetic exact,\n\
                 and one fp32 epsilon where sixteen octaves of exponent make it lossy."
            );
        } else {
            println!(
                "PARITY FAILED on {failures} of {total} checks. A non-zero `max |Δ| wmma`\n\
                 is a lane map that disagrees with the hardware's own; `first differing`\n\
                 is the (row, col) to read the map at. Do not loosen the comparison --\n\
                 both paths ran the same instruction on the same bytes, so the only\n\
                 correct difference is zero."
            );
        }

        identity_contract_note(&reference, &under_test);
        let sensitive = sensitivity(&reference, arch, &headers);

        // A probe whose process exits 0 on failure is a probe nothing can be
        // gated on -- it reports to a human who has to be reading. Both halves
        // of the claim are returned so `main` can set an exit code: the four
        // comparisons AND the control that proves they could have failed.
        failures == 0 && sensitive
    }

    /// Can this probe fail?
    ///
    /// Four PASSes are worth nothing unless the harness would have reported a
    /// FAIL, and the ways it might not are not exotic: a comparison over an
    /// empty slice, a result read back from the wrong buffer, a launch whose
    /// failure was swallowed all print a clean zero. So the shim is compiled
    /// a second time with ONE character of its call site changed --
    /// `mem_row_major` becomes `mem_col_major`, the transposed store that is
    /// the single most likely mistake in a hand-written lane map -- and the
    /// comparison is required to catch it.
    ///
    /// The mutation is in the probe's own kernel text and not in
    /// `pie_mma.cuh`, so this measures the harness rather than the header.
    /// `true` when the deliberately-wrong shim was caught. An unmeasured
    /// gate answers `false`: it is not a pass, and the caller must not treat
    /// "could not check" as "checked".
    fn sensitivity(reference: &Module, arch: &str, headers: &[Header]) -> bool {
        let mutant = format!(
            "#include \"pie_mma.cuh\"\nusing namespace nvcuda;\n\n{}\n",
            BODY.replace("wmma::mem_row_major", "wmma::mem_col_major")
        );
        let Ok(built) = compile_with_nvrtc(&mutant, arch, headers) else {
            println!("\nsensitivity: the mutant would not compile, so the gate is unmeasured");
            return false;
        };
        let Ok(module) = Module::load(&built.image) else {
            println!("\nsensitivity: the mutant would not load, so the gate is unmeasured");
            return false;
        };

        let mut rng = Rng::new(0x5eed_1e55_c0ff_ee01);
        let a: Vec<u16> = (0..N * N).map(|_| f32_to_bf16(rng.eighth())).collect();
        let b: Vec<u16> = (0..N * N).map(|_| f32_to_bf16(rng.eighth())).collect();
        let (Ok(want), Ok(got)) = (reference.run(&a, &b), module.run(&a, &b)) else {
            println!("\nsensitivity: a launch failed, so the gate is unmeasured");
            return false;
        };
        let (delta, first) = compare(&want, &got);
        if delta == 0.0 {
            println!(
                "\nsensitivity: FAIL. A transposed store measured identical to wmma, which\n\
                 means the four PASSes above are not evidence of anything -- the comparison\n\
                 is not reading what the kernels wrote."
            );
            false
        } else {
            println!(
                "\nsensitivity: a transposed store (mem_col_major) is caught -- max |Δ| {:.6e}\n\
                 at [{},{}]. The four PASSes are a measurement and not an empty comparison.",
                delta,
                first.unwrap_or(0) / N,
                first.unwrap_or(0) % N,
            );
            true
        }
    }

    /// The four cases, built from a deterministic generator so that a failure
    /// is reproducible and a report names inputs someone else can regenerate.
    fn cases() -> Vec<Case> {
        let mut rng = Rng::new(0x5eed_1e55_c0ff_ee01);

        let exact_a: Vec<u16> = (0..N * N).map(|_| f32_to_bf16(rng.eighth())).collect();
        let exact_b: Vec<u16> = (0..N * N).map(|_| f32_to_bf16(rng.eighth())).collect();

        let entropy_a: Vec<u16> = (0..N * N).map(|_| rng.bf16_bits()).collect();
        let entropy_b: Vec<u16> = (0..N * N).map(|_| rng.bf16_bits()).collect();

        // `A = I`. The result must be `B` itself, which is what makes this the
        // case a transposed store cannot survive.
        let mut identity = vec![0u16; N * N];
        for i in 0..N {
            identity[at(i, i)] = f32_to_bf16(1.0);
        }
        let identity_b: Vec<u16> = (0..N * N).map(|_| rng.eighth()).map(f32_to_bf16).collect();

        // One non-zero column of `B`. `LIVE_COLUMN` is odd and not a
        // multiple of 8, so it is in the second lane-group half of its own
        // 16x8 tile: a B map that lost the `groupID` term would put it in the
        // wrong column and every other column would stop being zero.
        let column_a: Vec<u16> = (0..N * N).map(|_| f32_to_bf16(rng.eighth())).collect();
        let mut column_b = vec![0u16; N * N];
        for k in 0..N {
            column_b[at(LIVE_COLUMN, k)] = f32_to_bf16(rng.eighth());
        }

        vec![
            Case {
                what: "pseudo-random, multiples of 1/8 (exact)",
                a: exact_a,
                b: exact_b,
                host_is_exact: true,
            },
            Case {
                what: "pseudo-random, full bf16 mantissa entropy",
                a: entropy_a,
                b: entropy_b,
                host_is_exact: false,
            },
            Case {
                what: "A = I  (result must be B, untransposed)",
                a: identity,
                b: identity_b,
                host_is_exact: true,
            },
            Case {
                what: "B with exactly one non-zero column (n=5)",
                a: column_a,
                b: column_b,
                host_is_exact: true,
            },
        ]
    }

    /// The case-specific claim, beyond agreeing with the reference.
    ///
    /// Agreement alone would be satisfied by two implementations that are
    /// wrong in the same way -- which is not a hypothetical here, since the
    /// shim was written by reading the same specification the reference
    /// implements. These are the two structural facts that can be asserted
    /// without either implementation's help.
    fn contract_holds(case: &Case, got: &[f32]) -> bool {
        if case.what.starts_with("A = I") {
            // `C[m][n] = sum_k I[m][k] * B[k][n] = B[m][n]`, and `B[k][n]`
            // lives at `b[n * 16 + k]` because B was loaded column-major. So
            // the row-major output is B's memory TRANSPOSED -- and a store
            // that wrote `(n, m)` would produce B's memory unchanged, which
            // is exactly the mistake this catches.
            for m in 0..N {
                for n in 0..N {
                    if got[at(m, n)] != bf16_to_f32(case.b[at(n, m)]) {
                        return false;
                    }
                }
            }
        }
        if case.what.starts_with("B with exactly one") {
            for m in 0..N {
                for n in 0..N {
                    if n != LIVE_COLUMN && got[at(m, n)] != 0.0 {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// The identity case, said out loud, because "untransposed" is a claim
    /// about a contract and the point of the case is to state which one.
    fn identity_contract_note(reference: &Module, under_test: &Module) {
        let mut identity = vec![0u16; N * N];
        for i in 0..N {
            identity[at(i, i)] = f32_to_bf16(1.0);
        }
        // A B whose every element is distinct, so no two positions can be
        // confused for one another.
        let b: Vec<u16> = (0..N * N).map(|i| f32_to_bf16(i as f32 / 8.0)).collect();
        let (Ok(want), Ok(got)) = (reference.run(&identity, &b), under_test.run(&identity, &b))
        else {
            return;
        };
        // The single element the whole contract turns on: row 0, column 1 of
        // the result. Named through `at` rather than spelled `0 * N + 1`, so
        // that the (row, column) reading survives and the arithmetic is not
        // something a reader has to reconstruct.
        let (row, col) = (0, 1);
        println!(
            "\nthe store's contract, with A = I and B[k][n] = (n*16+k)/8:\n\
             \x20 C[{row}][{col}] = {:.3}, and B's memory at [{col}*16+{row}] is {:.3} -- the \
             accumulator is\n\
             \x20 indexed (m, n) with n being B's COLUMN, so a row-major store of C is B\n\
             \x20 TRANSPOSED relative to B's own column-major memory. wmma agrees: {:.3}.",
            got[at(row, col)],
            bf16_to_f32(b[at(col, row)]),
            want[at(row, col)],
        );
    }

    /// `(row, column)` of a 16x16 tile, as an index into a row-major buffer.
    ///
    /// Also the index into a COLUMN-major one with the arguments swapped,
    /// which is the whole of the identity case's claim and the reason this is
    /// a function rather than open-coded arithmetic at each site.
    fn at(row: usize, col: usize) -> usize {
        row * N + col
    }

    /// Element-by-element, with no tolerance: the largest absolute difference
    /// and where the first one is.
    fn compare(want: &[f32], got: &[f32]) -> (f32, Option<usize>) {
        let mut max = 0.0f32;
        let mut first = None;
        for (at, (w, g)) in want.iter().zip(got).enumerate() {
            let delta = (w - g).abs();
            if delta > max {
                max = delta;
            }
            if delta != 0.0 && first.is_none() {
                first = Some(at);
            }
        }
        (max, first)
    }

    /// `C = A @ B` in fp32, the third opinion. `A` is row-major and `B` is
    /// column-major, which is what the two fragments say they are.
    fn host_matmul(a: &[u16], b: &[u16]) -> Vec<f32> {
        let mut c = vec![0.0f32; N * N];
        for m in 0..N {
            for n in 0..N {
                let mut sum = 0.0f32;
                for k in 0..N {
                    sum += bf16_to_f32(a[at(m, k)]) * bf16_to_f32(b[at(n, k)]);
                }
                c[at(m, n)] = sum;
            }
        }
        c
    }

    /// A difference, printed so that zero is unmistakably zero.
    fn format_delta(delta: f32) -> String {
        if delta == 0.0 { "0 (exact)".to_string() } else { format!("{delta:.6e}") }
    }

    // ---------------------------------------------------------------------
    // bf16, on the host
    // ---------------------------------------------------------------------

    /// `f32 -> bf16`, round-to-nearest-even -- the same arithmetic
    /// `pie_device.cuh`'s `f32_to_bf16` does, so the bytes this probe uploads
    /// are the bytes that kernel would have produced.
    fn f32_to_bf16(value: f32) -> u16 {
        let bits = value.to_bits();
        if bits & 0x7fff_ffff > 0x7f80_0000 {
            return ((bits >> 16) as u16) | 0x0040;
        }
        let rounding = 0x7fff + ((bits >> 16) & 1);
        (bits.wrapping_add(rounding) >> 16) as u16
    }

    /// `bf16 -> f32`, a shift, and exact -- bfloat16 is fp32 with the low
    /// sixteen bits dropped.
    fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits(u32::from(bits) << 16)
    }

    /// A deterministic generator, so a failure names inputs that can be
    /// regenerated rather than "some random matrix".
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        /// xorshift64*, which is enough entropy for a 256-element tile and
        /// short enough to read.
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        /// A multiple of `1/8` in `[-7/8, 7/8]`. Exact in bf16 -- three
        /// significant bits against seven stored -- and its products and
        /// sixteen-term sums are exact in fp32, which is what lets the host
        /// reference be an equality rather than a tolerance.
        fn eighth(&mut self) -> f32 {
            ((self.next() % 15) as f32 - 7.0) / 8.0
        }

        /// A finite bf16 with a full random mantissa and an exponent spread
        /// over `[2^-15, 2^16)`, well short of an infinity or a NaN.
        ///
        /// The spread is deliberate and was MEASURED into existence. A first
        /// version pinned the exponent to `[2^-1, 2^1)`, and the host fp32
        /// matmul then agreed with both kernels to the bit -- correctly, and
        /// uselessly: a bf16 product has at most sixteen significant bits and
        /// sixteen of them in one octave sum inside fp32's exact range, so
        /// that case could not tell a rounding difference from an equality.
        /// Sixteen octaves of exponent make the sum genuinely lossy, which is
        /// what turns "the two kernels agree exactly while the host does not"
        /// into a statement with content.
        fn bf16_bits(&mut self) -> u16 {
            let bits = self.next();
            let sign = ((bits >> 20) & 1) as u16;
            let exponent = 0x70 + ((bits >> 8) & 0x1f) as u16;
            let mantissa = (bits & 0x7f) as u16;
            (sign << 15) | (exponent << 7) | mantissa
        }
    }

    // ---------------------------------------------------------------------
    // compiling, both ways
    // ---------------------------------------------------------------------

    /// One compiled kernel image, and what it cost.
    struct Built {
        image: Vec<u8>,
        millis: f64,
    }

    /// Compile with the machine's `nvcc`, against its own `mma.h`.
    ///
    /// Shelling out, and reading the toolkit, is exactly what the shipped
    /// crate refuses to do -- and it is right for a probe: this path exists
    /// to produce the answer the shim is being held to, so it must be the
    /// vendor's implementation and not another statement of the shim's.
    ///
    /// The files land in `OUT_DIR`, which is this build's own scratch inside
    /// `target/`.
    fn compile_with_nvcc(nvcc: &PathBuf, source: &str, arch: &str) -> Result<Built, String> {
        let scratch = PathBuf::from(env!("OUT_DIR")).join("mma_probe");
        std::fs::create_dir_all(&scratch).map_err(|e| e.to_string())?;
        let cu = scratch.join("reference.cu");
        let cubin = scratch.join("reference.cubin");
        std::fs::write(&cu, source).map_err(|e| e.to_string())?;

        let started = Instant::now();
        let out = Command::new(nvcc)
            .arg(format!("-arch={arch}"))
            .args(["-std=c++17", "--cubin", "-o"])
            .arg(&cubin)
            .arg(&cu)
            .output()
            .map_err(|e| format!("could not run nvcc: {e}"))?;
        let millis = started.elapsed().as_secs_f64() * 1e3;
        if !out.status.success() {
            return Err(String::from_utf8_lossy(&out.stderr).trim().to_string());
        }
        let image = std::fs::read(&cubin).map_err(|e| e.to_string())?;
        Ok(Built { image, millis })
    }

    /// Compile with NVRTC, against the header set built in `run`.
    ///
    /// Its own helper rather than `runtime::nvrtc::compile_with`, because
    /// that one takes a `Unit` and a row list and mangles template names --
    /// none of which this kernel has, being `extern "C"` on purpose so that
    /// both paths are found by the same string.
    ///
    /// `sm_XY` and not `compute_XY`: the reference is a cubin for this
    /// device, so the thing under test has to be one too, or the comparison
    /// would include a difference in who ran the back end.
    fn compile_with_nvrtc(source: &str, arch: &str, headers: &[Header]) -> Result<Built, String> {
        let src = CString::new(source).map_err(|_| "a NUL in the probe source")?;
        let name = c"mma_probe.cu";
        let (texts, names) = as_nvrtc_arrays(headers)?;
        let text_ptrs: Vec<_> = texts.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string outlives the call, and the two arrays are the
        // same length -- the whole of `nvrtcCreateProgram`'s contract. The
        // header set is an in-memory filesystem: nothing is read from disk,
        // which is the property this probe is here to keep honest.
        let code = unsafe {
            nv::nvrtcCreateProgram(
                &raw mut program,
                src.as_ptr(),
                name.as_ptr(),
                i32::try_from(text_ptrs.len()).unwrap(),
                text_ptrs.as_ptr(),
                name_ptrs.as_ptr(),
            )
        };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcCreateProgram: {code:?}"));
        }

        let gpu = CString::new(format!("--gpu-architecture={arch}")).unwrap();
        let options = [gpu.as_ptr(), c"-std=c++17".as_ptr()];

        let started = Instant::now();
        // SAFETY: the program is live and the options outlive the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(
                program,
                i32::try_from(options.len()).unwrap(),
                options.as_ptr(),
            )
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;
        let log = program_log(program);

        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: destroyed exactly once, and not used after.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        let mut size = 0;
        // SAFETY: the program compiled, so a cubin exists; `size` is live.
        let code = unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: as above.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(format!("nvrtcGetCUBINSize: {code:?}"));
        }
        let mut image = vec![0u8; size];
        // SAFETY: the buffer is `size` bytes, which is what NVRTC just asked
        // for.
        let code = unsafe { nv::nvrtcGetCUBIN(program, image.as_mut_ptr().cast()) };
        // SAFETY: destroyed exactly once, after the last read out of it.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            return Err(format!("nvrtcGetCUBIN: {code:?}"));
        }
        Ok(Built { image, millis })
    }

    /// Whatever NVRTC had to say, whether or not it compiled.
    fn program_log(program: nv::nvrtcProgram) -> String {
        let mut size = 0;
        // SAFETY: `program` is live and `size` is a live out-parameter.
        unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
        let mut log = vec![0u8; size.max(1)];
        // SAFETY: the buffer is the size NVRTC asked for.
        unsafe { nv::nvrtcGetProgramLog(program, log.as_mut_ptr().cast()) };
        CStr::from_bytes_until_nul(&log)
            .map_or_else(|_| String::new(), |s| s.to_string_lossy().trim().to_string())
    }

    /// `nvcc`, wherever this machine keeps it.
    fn find_nvcc() -> Option<PathBuf> {
        let mut candidates: Vec<PathBuf> = Vec::new();
        if let Ok(path) = std::env::var("PATH") {
            candidates.extend(std::env::split_paths(&path).map(|dir| dir.join("nvcc")));
        }
        for root in ["CUDA_HOME", "CUDA_PATH"] {
            if let Ok(dir) = std::env::var(root) {
                candidates.push(PathBuf::from(dir).join("bin").join("nvcc"));
            }
        }
        candidates.push(PathBuf::from("/usr/local/cuda/bin/nvcc"));
        candidates.into_iter().find(|c| c.is_file())
    }

    // ---------------------------------------------------------------------
    // running, once per path per case
    // ---------------------------------------------------------------------

    /// A loaded cubin and the one entry point in it.
    ///
    /// Its own type rather than `runtime::KernelModule`, for the reason the
    /// compile helper is its own: that one is keyed on rows and units this
    /// kernel does not have. The unload in `Drop` is the part worth keeping
    /// either way -- two modules are live at once here, and a leaked one
    /// would keep a stale cubin resident for the process.
    struct Module {
        module: dr::CUmodule,
        function: dr::CUfunction,
    }

    impl Module {
        fn load(image: &[u8]) -> Result<Self, String> {
            ensure_context()?;
            let mut module: dr::CUmodule = std::ptr::null_mut();
            // SAFETY: the image is a cubin this process just produced and
            // outlives the call; `module` is a live out-parameter.
            let code = unsafe { dr::cuModuleLoadData(&raw mut module, image.as_ptr().cast()) };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleLoadData: {code:?}"));
            }
            let mut function: dr::CUfunction = std::ptr::null_mut();
            // SAFETY: `module` came from a successful load and `mma16` is a
            // NUL-terminated literal.
            let code =
                unsafe { dr::cuModuleGetFunction(&raw mut function, module, c"mma16".as_ptr()) };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuModuleGetFunction(mma16): {code:?}"));
            }
            Ok(Self { module, function })
        }

        /// One warp, one tile, one 16x16 fp32 result.
        ///
        /// A block of exactly 32 threads because a fragment op is
        /// warp-collective and the shim reads `threadIdx.x % 32` to place a
        /// lane -- a wider block would run the same kernel several times over
        /// the same output, which is a race rather than a comparison.
        fn run(&self, a: &[u16], b: &[u16]) -> Result<Vec<f32>, String> {
            let da = Device::upload(a)?;
            let db = Device::upload(b)?;
            let dc = Device::alloc(N * N * std::mem::size_of::<f32>())?;

            let mut pa = da.ptr;
            let mut pb = db.ptr;
            let mut pc = dc.ptr;
            let mut params = [
                (&raw mut pa).cast::<c_void>(),
                (&raw mut pb).cast::<c_void>(),
                (&raw mut pc).cast::<c_void>(),
            ];

            // SAFETY: the function came from a live module; the three
            // allocations outlive the launch because `params` borrows locals
            // that outlive the synchronise below; the geometry is one warp.
            let code = unsafe {
                dr::cuLaunchKernel(
                    self.function,
                    1,
                    1,
                    1,
                    32,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            };
            if code != dr::CUresult::CUDA_SUCCESS {
                return Err(format!("cuLaunchKernel: {code:?}"));
            }
            // SAFETY: no arguments, and a fault inside the kernel surfaces
            // here rather than at the copy below.
            let code = unsafe { rt::cudaDeviceSynchronize() };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaDeviceSynchronize: {code:?}"));
            }

            let mut out = vec![0.0f32; N * N];
            // SAFETY: both sides are `N * N` floats, and the device side was
            // allocated at that size above.
            let code = unsafe {
                rt::cudaMemcpy(
                    out.as_mut_ptr().cast(),
                    dc.ptr,
                    std::mem::size_of_val(out.as_slice()),
                    rt::cudaMemcpyKind::cudaMemcpyDeviceToHost,
                )
            };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMemcpy D2H: {code:?}"));
            }
            Ok(out)
        }
    }

    impl Drop for Module {
        fn drop(&mut self) {
            // SAFETY: the handle came from `cuModuleLoadData`, every launch
            // that named it has been synchronised, and nothing else holds it.
            unsafe { dr::cuModuleUnload(self.module) };
        }
    }

    /// One device allocation, freed when it goes out of scope.
    struct Device {
        ptr: *mut c_void,
    }

    impl Device {
        fn alloc(bytes: usize) -> Result<Self, String> {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            // SAFETY: `ptr` is a live out-parameter and `bytes` is non-zero.
            let code = unsafe { rt::cudaMalloc(&raw mut ptr, bytes) };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMalloc({bytes}): {code:?}"));
            }
            Ok(Self { ptr })
        }

        fn upload<T>(values: &[T]) -> Result<Self, String> {
            let bytes = std::mem::size_of_val(values);
            let owned = Self::alloc(bytes)?;
            // SAFETY: the destination is `bytes` long by construction and the
            // source is the slice's own storage.
            let code = unsafe {
                rt::cudaMemcpy(
                    owned.ptr,
                    values.as_ptr().cast(),
                    bytes,
                    rt::cudaMemcpyKind::cudaMemcpyHostToDevice,
                )
            };
            if code != rt::cudaError::cudaSuccess {
                return Err(format!("cudaMemcpy H2D: {code:?}"));
            }
            Ok(owned)
        }
    }

    impl Drop for Device {
        fn drop(&mut self) {
            // SAFETY: the pointer came from `cudaMalloc` and nothing else
            // holds it; every launch that read it has been synchronised.
            unsafe { rt::cudaFree(self.ptr) };
        }
    }

    /// A context the driver API can load a module into.
    ///
    /// The runtime API creates the primary context lazily and pushes it onto
    /// the calling thread, which is why a `cudaFree(null)` is enough. The
    /// explicit retain is the fallback for the case where it is not -- and it
    /// is a real case rather than defensiveness: `cuModuleLoadData` with no
    /// current context fails with `CUDA_ERROR_INVALID_CONTEXT`, which reads
    /// like a broken cubin.
    fn ensure_context() -> Result<(), String> {
        // SAFETY: a null pointer is the documented no-op that forces runtime
        // initialisation.
        unsafe { rt::cudaFree(std::ptr::null_mut()) };
        let mut current: dr::CUcontext = std::ptr::null_mut();
        // SAFETY: `current` is a live out-parameter.
        unsafe { dr::cuCtxGetCurrent(&raw mut current) };
        if !current.is_null() {
            return Ok(());
        }
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a live out-parameter; the driver is initialised
        // by the runtime call above.
        unsafe { dr::cuDeviceGet(&raw mut device, 0) };
        let mut context: dr::CUcontext = std::ptr::null_mut();
        // SAFETY: `context` is live and `device` came from `cuDeviceGet`.
        let code = unsafe { dr::cuDevicePrimaryCtxRetain(&raw mut context, device) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(format!("cuDevicePrimaryCtxRetain: {code:?}"));
        }
        // SAFETY: `context` came from a successful retain.
        let code = unsafe { dr::cuCtxSetCurrent(context) };
        if code != dr::CUresult::CUDA_SUCCESS {
            return Err(format!("cuCtxSetCurrent: {code:?}"));
        }
        Ok(())
    }

    /// What the driver calls this GPU, so the report names the machine it was
    /// measured on.
    fn device_name() -> String {
        let mut device: dr::CUdevice = 0;
        // SAFETY: `device` is a live out-parameter; `arch()` has already
        // initialised the driver by the time this is called.
        if unsafe { dr::cuDeviceGet(&raw mut device, 0) } != dr::CUresult::CUDA_SUCCESS {
            return "unknown".to_string();
        }
        let mut name = [0u8; 128];
        // SAFETY: the buffer is 128 bytes and that is what is claimed.
        let code = unsafe {
            dr::cuDeviceGetName(
                name.as_mut_ptr().cast(),
                i32::try_from(name.len()).unwrap(),
                device,
            )
        };
        if code != dr::CUresult::CUDA_SUCCESS {
            return "unknown".to_string();
        }
        CStr::from_bytes_until_nul(&name)
            .map_or_else(|_| "unknown".to_string(), |s| s.to_string_lossy().into_owned())
    }

    /// `libnvrtc`'s own version, because the answer to "does `mma.h` resolve"
    /// is a property of it and not of the toolkit beside it.
    fn nvrtc_version() -> String {
        let (mut major, mut minor) = (0, 0);
        // SAFETY: both are live out-parameters for the call's duration.
        let code = unsafe { nv::nvrtcVersion(&raw mut major, &raw mut minor) };
        if code == nv::nvrtcResult::NVRTC_SUCCESS {
            format!("{major}.{minor}")
        } else {
            format!("unavailable ({code:?})")
        }
    }
}
