//! Is vendored Marlin *structurally* unmigrable, or is it a host choice?
//!
//! # The claim this was written to settle
//!
//! `examples/migration_status.rs` classifies two rows [`Wall::Library`] —
//! *"there is no device text in this tree"*:
//!
//! ```text
//! gemm::act_x_wt_mxfp4_marlin                    vendored Marlin, csrc/third_party/marlin
//! marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16   vendored Marlin MoE, csrc/third_party/marlin_moe
//! ```
//!
//! The reason is false and this probe measures why. `csrc/third_party/marlin`
//! holds 727 `__global__` and `marlin_moe` holds 22, and `marlin_template.h`
//! is 2,082 lines of kernel body sitting right there in the repository. Marlin
//! is not cuBLAS. `new-horizon.md` §5 said so in the first place — *"**Marlin
//! JITs.** Its device headers need only `cuda_{fp16,bf16,fp8}.h`; the host STL
//! in its wrappers is launcher code that Rust replaces anyway"* — and three
//! files in the vendored tree (`pie_marlin_rtc.h`, `pie_marlin_intrinsics.h`,
//! and `marlin.cuh`'s `#ifndef __CUDACC_RTC__` around `<iostream>`) exist for
//! no other purpose than to make that true.
//!
//! So `Library` is the wrong wall. The probe asks the three questions that
//! decide the right one, in the order in which each can invalidate the next.
//!
//! # §1 — How many kernels stand behind one symbol, and what picks one?
//!
//! Counted from the files, not estimated. Every `template __global__` in the
//! fourteen generated `sm80_*.cu` instantiation lists, every branch of the
//! generated `kernel_selector.h`, and the subset each of the two rows can
//! actually reach — which is narrower than either, because `marlin.cu` builds
//! three of the fourteen lists and the mxfp4 entry point pins all four scalar
//! types.
//!
//! The host code is read for the same reason `gemm.rs`'s header read
//! `gemv.cu`: a row names ONE instantiation, so what the launcher reads to
//! choose is the whole question.
//!
//! # §2 — Would NVRTC compile it?
//!
//! §13.1 measured NVRTC 13.0 answering **0 of 31** external includes with an
//! empty header set — not `<cstdint>`, not `<cuda_runtime.h>`. So every
//! directive Marlin's device text reaches is enumerated here and marked
//! against the set the library actually hands NVRTC ([`source::LIBRARY`]),
//! and then the compile is RUN. An enumeration that says "all answered" and a
//! compile that fails would both be findings; only running it distinguishes
//! them.
//!
//! The two things to watch for were `<cub/…>`/`<cuda/pipeline>`/`<cuda/barrier>`
//! and `mma.h`. Marlin has neither, and the mechanism is worth naming because
//! it is why this differs from CUTLASS: **Marlin writes its own PTX.**
//! `cp.async.cg.shared.global`, `ldmatrix.sync.aligned.m8n8.x4`, `mma.sync`,
//! `lop3.b32`, `prmt.b32` and `red.relaxed.gpu.global.add.s32` are `asm
//! volatile` in `marlin.cuh`, `marlin_mma.h`, `dequant.h` and
//! `marlin_template.h` — 40 of them — so the pipeline and the tensor cores
//! arrive with no library at all. `pie_mma.cuh` is not needed and is not
//! reached.
//!
//! # §3 — Could a row NAME the instantiation it picked?
//!
//! [`DeviceKernel::instantiation`] emits
//! `::pie_cuda_driver::kernels::{path}<::pie_cuda_driver::kernels::{elem}>`
//! — the qualification glued ONCE to the front, which `examples/argform_probe.rs`
//! measured over sixteen cases: slot 1 resolves under the prefix, slots 2+ at
//! global scope. Marlin's list is twelve arguments whose first four are
//! `vllm::kBFloat16.id()` — a `constexpr` **method call on a namespace-scope
//! object**, not a type name, from a `namespace vllm` at global scope. That is
//! two questions at once and this asks both separately, because they have
//! different answers and the difference is the finding.
//!
//! # What this probe is NOT
//!
//! It migrates nothing. No row, no unit, no shim, no edit to `csrc/**`. It
//! reads the vendored tree from disk the way `flashinfer_probe` reads
//! `$CUDA_HOME`, hands the text to NVRTC, and prints what NVRTC said.
//!
//! [`Wall::Library`]: https://example.invalid
//! [`DeviceKernel::instantiation`]: kernels_cuda_new::device::DeviceKernel::instantiation
//! [`source::LIBRARY`]: kernels_cuda_new::source::LIBRARY
fn main() {
    probe::run();
}

mod probe {
    use cudarc::nvrtc::sys as nv;
    use kernels_cuda_new::source::{self, Header};
    use std::collections::{BTreeMap, BTreeSet};
    use std::ffi::{CStr, CString};
    use std::path::{Path, PathBuf};

    /// sm_89 is the L40S in this box. Marlin's files are named `sm80_*` and
    /// §1 reports why that is a *generator* label rather than a target: the
    /// device text guards on `__CUDA_ARCH__ >= 750` and nothing above it, and
    /// `cp.async` + `mma.m16n8k16` are sm_80 instructions an sm_89 runs.
    const ARCH: &str = "sm_89";

    /// The vendored tree, relative to this crate's manifest.
    fn marlin_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kernels-cuda/csrc/third_party/marlin")
    }

    fn marlin_moe_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kernels-cuda/csrc/third_party/marlin_moe")
    }

    pub fn run() {
        println!("Marlin under NVRTC -- is `Library` the right wall?\n");
        census();
        let closure = includes();
        let verdict = compiles(&closure);
        naming(&closure, verdict);
        classification(verdict);
    }

    // -----------------------------------------------------------------------
    // §4  The wall these three measurements actually name
    // -----------------------------------------------------------------------

    /// What the two rows should be classified, and the one sentence for it.
    ///
    /// This is the part that matters more than anything landing. A wrong
    /// REASON on a correct refusal is what sends the next reader down a dead
    /// end — `Library` says *"closes only by the library being replaced"*, so
    /// a reader who believes it will look for a replacement GEMM. The right
    /// wall says what would actually have to change, and it is host code in
    /// Rust, which is work this migration does every day.
    fn classification(compiled: bool) {
        println!("\n== 4. The correct classification ==\n");
        if !compiled {
            println!("   Withheld: §2 did not run to a verdict.");
            return;
        }
        for line in VERDICT {
            println!("   {line}");
        }
    }

    const VERDICT: &[&str] = &[
        "NOT `Library`. That wall reads \"there is no device text in this tree\" and",
        "\"closes only by the library being replaced\", and both halves are false:",
        "`marlin_template.h` is 2,082 lines of `__global__` in this repository, and",
        "NVRTC compiled one of its instantiations to a 55,024-byte cubin above.",
        "",
        "`HostChoice` -- the same wall as `gemm::gemv3_bf16`, and for the same reason",
        "in the same words. `Wall::HostChoice` is defined as \"a host `if` whose arms",
        "reach different kernels, on a fact that is not the fire's shape: an",
        "environment variable, A SHARED-MEMORY BUDGET, a `constexpr` in a file the row",
        "cannot see.\" Marlin's `determine_exec_config` walks a tuning table and takes",
        "the first entry whose `get_kernel_cache_size(...)` fits",
        "`cudaDevAttrMaxSharedMemoryPerBlockOptin` AND whose twelve-argument",
        "instantiation exists; `stages` comes from `cudaDevAttrComputeCapabilityMajor`;",
        "the grid and `max_par` come from `cudaDevAttrMultiProcessorCount`. Fifteen",
        "instantiations stand behind each row and a host heuristic picks one per call,",
        "on three device queries a name expression is fixed before it can see.",
        "",
        "Two things compound it, and both are worth stating because they change what",
        "closing the row would cost:",
        "",
        "  * `gemm::act_x_wt_mxfp4_marlin` is ALSO `TwoLaunches`. `marlin.cu:443`'s",
        "    `while (rest_m)` issues one `<<<>>>` per prob_m chunk -- at M=100 with",
        "    max_par=128 that is two launches of two DIFFERENT `thread_m_blocks`,",
        "    hence two different instantiations, from one C++ call.",
        "    `marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16` has no such loop and is one",
        "    launch, so that half of the compounding is the GEMM row's alone.",
        "",
        "  * NEITHER row is `Declines`, and the difference from `gemv.cu` is real.",
        "    `gemv.cu` returns `false` meaning \"I did not launch, use cuBLAS\";",
        "    `marlin_mm` returns void and THROWS. A row cannot decline, but it also",
        "    cannot throw -- and the failure modes differ: a JIT row that reproduced",
        "    `gemv.cu`'s arm would read past a buffer, whereas one that reproduced",
        "    Marlin's would launch a kernel whose shared-memory opt-in was never",
        "    granted and return the caller its own uninitialised output buffer.",
        "    `marlin_moe/ops.cu:543` says so in its own comment.",
        "",
        "What the row would need, stated so it is a plan and not a mood:",
        "  1. `Specialisation` over the fifteen (`agrees()` refuses arms whose",
        "     `LaunchRule` differs -- these share a geometry, so they are legal).",
        "  2. `determine_exec_config` in Rust, including the two device queries",
        "     `crate::device` would have to carry. §14.6 did exactly this for",
        "     FlashInfer's planner, byte for byte.",
        "  3. Two shim intrinsics, to §15.2's bar: `__hadd2` and `atomicAdd` over",
        "     `__nv_bfloat162`.",
        "  4. Two `namespace` aliases in the unit source, per §3 above.",
        "  5. A `LaunchRule` for a grid that is `sms * blocks_per_sm` -- and, the one",
        "     genuinely new thing here, somewhere to put the per-instantiation",
        "     `cudaFuncSetAttribute(cudaFuncAttributeMaxDynamicSharedMemorySize)`,",
        "     which is module state rather than a launch parameter.",
        "",
        "None of those is a wall. All of them are host work, which is what",
        "`HostChoice` means and `Library` does not.",
    ];

    // -----------------------------------------------------------------------
    // §1  The count, and the selector
    // -----------------------------------------------------------------------

    /// Every `template __global__` in the generated lists, and every branch of
    /// the generated selector.
    fn census() {
        println!("== 1. How many instantiations, and what chooses among them ==\n");

        let dir = marlin_dir();
        let mut lists: Vec<(String, usize)> = Vec::new();
        let mut total = 0usize;
        let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
            .expect("the vendored marlin tree is in this repository")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("sm80_kernel_") && n.ends_with(".cu"))
            })
            .collect();
        entries.sort();
        for path in &entries {
            let text = std::fs::read_to_string(path).expect("a generated list is readable");
            let n = text.matches("template __global__").count();
            total += n;
            lists.push((path.file_name().unwrap().to_string_lossy().into_owned(), n));
        }

        // The three `marlin.cu` actually `#include`s. A macro cannot expand to
        // an `#include`, so the built subset is spelled out in the `.cu` and
        // CMake reconciles it against `kernels.def` -- which is the thirty-
        // eight lines of build system `src/families/marlin.rs` says a JIT
        // deletes rather than ports.
        let driver = std::fs::read_to_string(dir.join("marlin.cu")).expect("marlin.cu is readable");
        let built: BTreeSet<&str> = lists
            .iter()
            .map(|(n, _)| n.as_str())
            .filter(|n| driver.contains(&format!("#include \"{n}\"")))
            .collect();
        let built_count: usize =
            lists.iter().filter(|(n, _)| built.contains(n.as_str())).map(|(_, c)| c).sum();

        println!("   {:<44}  {:>5}   {}", "generated instantiation list", "insts", "built?");
        for (name, n) in &lists {
            println!(
                "   {:<44}  {:>5}   {}",
                name,
                n,
                if built.contains(name.as_str()) { "yes" } else { "-" }
            );
        }
        println!("   {:<44}  {:>5}", "TOTAL on disk", total);
        println!("   {:<44}  {:>5}   ({} of 14 lists)", "TOTAL compiled by marlin.cu", built_count, built.len());

        let selector =
            std::fs::read_to_string(dir.join("kernel_selector.h")).expect("the selector is readable");
        let branches = selector.matches("kernel = Marlin<").count();
        println!("   {:<44}  {:>5}", "branches in kernel_selector.h", branches);

        // The two sets are the same set, which is what makes the selector a
        // complete account of the built kernels rather than a superset.
        let named: BTreeSet<String> = normalised(&selector, "kernel = ");
        let mut defined: BTreeSet<String> = BTreeSet::new();
        for (name, _) in lists.iter().filter(|(n, _)| built.contains(n.as_str())) {
            let text = std::fs::read_to_string(dir.join(name)).expect("readable");
            defined.extend(normalised(&text, "template __global__ void "));
        }
        println!(
            "   selector branches == compiled instantiations?  {}",
            if named == defined { "YES, exactly" } else { "NO" }
        );

        // What each ROW can reach. `marlin_wrapper.cpp` pins all four scalar
        // types, so the reachable set is one generated list, not the built 150.
        let mxfp4 = lists
            .iter()
            .find(|(n, _)| n == "sm80_kernel_bfloat16_fe2m1f_bfloat16.cu")
            .map(|(_, c)| *c)
            .unwrap_or(0);

        let moe_dir = marlin_moe_dir();
        let moe_list = moe_dir.join("sm80_kernel_bfloat16_fe2m1f_bfloat16.cu");
        let moe = std::fs::read_to_string(&moe_list)
            .map(|t| t.matches("template __global__").count())
            .unwrap_or(0);
        let moe_sel = std::fs::read_to_string(moe_dir.join("kernel_selector.h"))
            .map(|t| t.matches("kernel = Marlin<").count())
            .unwrap_or(0);

        println!("\n   -- what stands behind each ROW --");
        println!(
            "   gemm::act_x_wt_mxfp4_marlin                   {mxfp4:>3} instantiations \
             (bf16 x FE2M1f x bf16 x FE8M0fnu)"
        );
        println!(
            "   marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16  {moe:>3} instantiations \
             ({moe_sel} selector branches)"
        );
        println!(
            "   ...reached through `marlin_mm`, which can also serve {} others \
             (u4, u4b8) behind the SAME C++ symbol.",
            built_count - mxfp4
        );

        selector_reads();
    }

    /// `Marlin<…>` argument lists, whitespace-collapsed, from a generated file.
    fn normalised(text: &str, after: &str) -> BTreeSet<String> {
        let mut out = BTreeSet::new();
        for chunk in text.split(after).skip(1) {
            if let Some(end) = chunk.find('>') {
                let args: String =
                    chunk[..end].split_whitespace().collect::<Vec<_>>().join(" ");
                out.insert(args);
            }
        }
        out
    }

    /// Exactly what the host reads to pick one of them. Cited, not summarised.
    fn selector_reads() {
        println!("\n   -- what the selector READS, `marlin.cu` line by line --");
        for (what, cite) in SELECTOR {
            println!("   {what:<40}  {cite}");
        }
        println!("\n   -- and how it fails --");
        for line in DECLINES {
            println!("   {line}");
        }
    }

    /// The facts `get_marlin_kernel`'s twelve template arguments are computed
    /// from, each with the line that computes it.
    const SELECTOR: &[(&str, &str)] = &[
        ("a/b/c/s scalar types", "pinned by the entry point; marlin_wrapper.cpp:142-145"),
        ("prob_m, prob_n, prob_k", "the fire's shape; marlin.cu:355 splits prob_m into chunks"),
        ("group_size -> group_blocks", "`group_size / 16`, or -1 per-channel; marlin.cu:391-401"),
        ("K % group_blocks", "TORCH_CHECK(prob_k % group_blocks == 0); marlin.cu:398"),
        ("K % 32 (mxfp4)", "throws before marlin_mm is reached; marlin_wrapper.cpp:146-148"),
        ("prob_k % thread_k, prob_n % thread_n", "is_valid_config; marlin.cu:257-259"),
        ("a SHARED-MEMORY budget", "get_kernel_cache_size(...) <= max_shared_mem; marlin.cu:275"),
        ("cudaDevAttrMaxSharedMemoryPerBlockOptin", "a DEVICE QUERY; marlin.cu:411-413"),
        ("cudaDevAttrComputeCapabilityMajor/Minor", "a DEVICE QUERY -> `stages`; marlin.cu:415-424"),
        ("cudaDevAttrMultiProcessorCount", "a DEVICE QUERY -> grid AND `max_par`; wrapper:60-64"),
        ("a TUNING TABLE, walked in order", "small_/large_batch_thread_configs; marlin.cu:156-171"),
        ("`kernel == MarlinDefault` retry", "the table is walked until one RESOLVES; marlin.cu:334"),
        ("m_block_size_8", "`prob_m_split <= 8 && a_type.size_bits() == 16`; marlin.cu:454"),
        ("a second is_valid_config pass", "a {128,64,128} override when the grid underfills; :477-487"),
        ("max_thread_m_blocks, decremented", "4 -> 3 -> 2 -> 1 and `continue`; marlin.cu:490-493"),
    ];

    /// How the launcher behaves when it cannot serve the call — the question
    /// `gemm.rs`'s third reason turns on.
    const DECLINES: &[&str] = &[
        "`marlin_mm` returns void. It does not decline the way `gemv.cu` does --",
        "it THROWS: TORCH_CHECK(false, \"Unsupported shapes: MNK = [...]\") at marlin.cu:519.",
        "So the arm a row would have to reproduce is an exception, not a `false`.",
        "It also launches MORE THAN ONCE: `while (rest_m)` at marlin.cu:443 issues one",
        "`kernel<<<>>>` per prob_m chunk (up to max_par=128 blocks' worth at a time),",
        "and `has_act_order` adds a `permute_cols_kernel<<<>>>` before the loop.",
        "And each launch is preceded by `cudaFuncSetAttribute(kernel,",
        "cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` -- per INSTANTIATION state,",
        "not a launch parameter (marlin.cu:531; marlin_moe/ops.cu:543 checks its result",
        "and says in its own comment why an unchecked one is a silent wrong answer).",
    ];

    // -----------------------------------------------------------------------
    // §2  The includes, and the compile
    // -----------------------------------------------------------------------

    /// The in-tree files a compile of `marlin_template.h` must be handed, in
    /// the spelling each includer writes.
    ///
    /// NVRTC matches `includeNames[]` against the LITERAL string in the
    /// directive (`src/source.rs`), so `marlin_moe/marlin_template.h` reaching
    /// `"../marlin/marlin.cuh"` needs that spelling and not a tidied one.
    struct Closure {
        /// name -> text, for `nvrtcCreateProgram`.
        files: Vec<(String, String)>,
        /// Every angle-bracket directive the closure reaches, with where.
        external: Vec<(String, String)>,
        /// Directives deleted by `#ifndef __CUDACC_RTC__`. Reported by
        /// [`report`] on the way through; kept on the struct because "which
        /// directives the guards DELETE" is half of §2's answer and a reader
        /// extending this probe will want them.
        #[allow(dead_code)]
        guarded: Vec<(String, String)>,
    }

    /// Walk `marlin_template.h`'s quoted includes, honouring the RTC guards.
    ///
    /// A real preprocessor is not needed and would hide the finding: the
    /// question is which directives SURVIVE `__CUDACC_RTC__`, and that is a
    /// nesting count over `#ifdef`/`#ifndef __CUDACC_RTC__`. Every other
    /// conditional is treated as taken, which over-counts rather than under-
    /// counts — and the compile below is the check on that.
    fn includes() -> Closure {
        let dir = marlin_dir();
        let mut files: Vec<(String, String)> = Vec::new();
        let mut external: Vec<(String, String)> = Vec::new();
        let mut guarded: Vec<(String, String)> = Vec::new();
        let mut seen: BTreeSet<String> = BTreeSet::new();
        // Both seeds, because a unit's source is both directives: `kernel.h`
        // holds `MARLIN_KERNEL_PARAMS` and the `__global__` DECLARATION,
        // `marlin_template.h` holds its body. Every generated `sm80_*.cu`
        // opens with exactly this pair, and a walk seeded on one of them
        // misses the other's includes.
        let mut queue: Vec<String> = vec!["marlin_template.h".into(), "kernel.h".into()];

        while let Some(name) = queue.pop() {
            if !seen.insert(name.clone()) {
                continue;
            }
            let leaf = name.rsplit('/').next().unwrap_or(&name);
            let Ok(text) = std::fs::read_to_string(dir.join(leaf)) else {
                continue;
            };
            for (spelling, live) in directives(&text) {
                let here = format!("{leaf}");
                if spelling.starts_with('<') {
                    let bare = spelling.trim_matches(['<', '>']).to_string();
                    if live {
                        external.push((bare, here));
                    } else {
                        guarded.push((bare, here));
                    }
                } else if live {
                    queue.push(spelling.trim_matches('"').to_string());
                }
            }
            files.push((name, text));
        }
        external.sort();
        external.dedup();
        guarded.sort();
        guarded.dedup();

        // Both `marlin/` and `marlin_moe/` reach the same five headers; the MoE
        // tree spells them `../marlin/x`, so register that spelling too.
        let extra: Vec<(String, String)> = files
            .iter()
            .filter(|(n, _)| n != "marlin_template.h")
            .map(|(n, t)| (format!("../marlin/{n}"), t.clone()))
            .collect();
        files.extend(extra);

        report(&external, &guarded);
        Closure { files, external, guarded }
    }

    /// `(spelling, survives __CUDACC_RTC__)` for every `#include` in a file.
    fn directives(text: &str) -> Vec<(String, bool)> {
        let mut out = Vec::new();
        // >0 inside `#ifdef __CUDACC_RTC__`, <0 inside `#ifndef`, tracked as a
        // stack so an unrelated `#if` in between does not shift the meaning.
        let mut stack: Vec<i32> = Vec::new();
        for raw in text.lines() {
            let line = raw.trim();
            if let Some(rest) = line.strip_prefix('#') {
                let rest = rest.trim_start();
                if let Some(cond) = rest.strip_prefix("ifdef ") {
                    stack.push(if cond.trim().starts_with("__CUDACC_RTC__") { 1 } else { 0 });
                    continue;
                }
                if let Some(cond) = rest.strip_prefix("ifndef ") {
                    stack.push(if cond.trim().starts_with("__CUDACC_RTC__") { -1 } else { 0 });
                    continue;
                }
                if rest.starts_with("if") {
                    stack.push(0);
                    continue;
                }
                if rest.starts_with("else") {
                    if let Some(top) = stack.last_mut() {
                        *top = -*top;
                    }
                    continue;
                }
                if rest.starts_with("endif") {
                    stack.pop();
                    continue;
                }
                if let Some(inc) = rest.strip_prefix("include") {
                    // A directive is live under NVRTC unless some enclosing
                    // guard says `__CUDACC_RTC__` is FALSE.
                    let live = !stack.contains(&-1);
                    let inc = inc.trim();
                    let end = inc.find(|c| c == '>' || c == '"').map(|i| i + 1).unwrap_or(0);
                    let rest_after_open = &inc[1..];
                    let close = rest_after_open.find(|c| c == '>' || c == '"');
                    if let Some(close) = close {
                        let _ = end;
                        out.push((inc[..close + 2].to_string(), live));
                    }
                }
            }
        }
        out
    }

    /// Mark every external directive against the set the library carries.
    fn report(external: &[(String, String)], guarded: &[(String, String)]) {
        println!("\n== 2. Would NVRTC compile Marlin at all? ==\n");
        println!("   Every include the device text reaches, marked against the SAME objects");
        println!("   `runtime::nvrtc::compile` hands NVRTC -- `source::LIBRARY` (`csrc/src`:");
        println!("   the shims) and `source::VENDOR` (`csrc/vendor`: the four crutches whose");
        println!("   guards §13.6 measured and REFUSED, because the names in them reach");
        println!("   device code).\n");

        let shims: BTreeSet<&str> =
            ["cuda_fp16.h", "cuda_bf16.h", "cuda_fp8.h", "cuda_fp4.h", "cooperative_groups.h"]
                .into_iter()
                .collect();
        let library: BTreeMap<&str, &Header> =
            source::LIBRARY.iter().map(|h| (h.name, h)).collect();
        let vendor: BTreeMap<&str, &Header> = source::VENDOR.iter().map(|h| (h.name, h)).collect();

        // One row per directive, with every includer that writes it.
        let mut merged: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (name, from) in external {
            merged.entry(name.as_str()).or_default().push(from.as_str());
        }

        println!("   {:<20} {:<12} {}", "directive", "answered by", "which file, and its size");
        let mut unanswered = 0;
        for (name, from) in &merged {
            let (kind, where_) = if shims.contains(name) {
                let bytes = library.get(name).map_or(0, |h| h.text.len());
                ("SHIM", format!("csrc/src/{name}  ({bytes} B)"))
            } else if let Some(h) = library.get(name) {
                ("LIBRARY", format!("csrc/src/{}  ({} B)", h.name, h.text.len()))
            } else if let Some(h) = vendor.get(name) {
                ("VENDOR", format!("csrc/vendor/{}  ({} B)", h.name, h.text.len()))
            } else {
                unanswered += 1;
                ("NOT ANSWERED", String::new())
            };
            println!("   {:<20} {:<12} {}", format!("<{name}>"), kind, where_);
            println!("   {:<20} {:<12} reached from {}", "", "", from.join(", "));
        }
        println!(
            "\n   {} distinct external directives, {} not answered.",
            merged.len(),
            unanswered
        );

        if !guarded.is_empty() {
            println!("\n   Deleted by `#ifndef __CUDACC_RTC__`, and therefore never asked:");
            let mut host: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
            for (name, from) in guarded {
                host.entry(name.as_str()).or_default().push(from.as_str());
            }
            for (name, from) in &host {
                println!("   {:<20} {:<12} {}", format!("<{name}>"), "(host-only)", from.join(", "));
            }
            println!(
                "\n   `pie_marlin_rtc.h` answers <tuple> and <utility> in their place -- 275 lines"
            );
            println!("   carrying `pair`, `tuple`, `tuple_cat` and `apply` -- because");
            println!("   `ScalarType::from_id` is a constexpr FOLD evaluated the moment NVRTC");
            println!("   resolves `Marlin<vllm::kBFloat16.id(), …>`. Not host code.");
        }

        println!("\n   Loud absences, checked rather than assumed:");
        for (what, verdict) in ABSENT {
            println!("   {what:<34} {verdict}");
        }
    }

    /// The things a reader would reasonably expect a 2,000-line GEMM to need.
    const ABSENT: &[(&str, &str)] = &[
        ("<cub/…>", "NOT REACHED -- 0 directives in the closure"),
        ("<cuda/pipeline>, <cuda/barrier>", "NOT REACHED -- `cp.async` is raw PTX, marlin.cuh:115-186"),
        ("<cuda/std/…>", "NOT REACHED"),
        ("mma.h / nvcuda::wmma", "NOT REACHED -- `mma.sync`/`ldmatrix` are raw PTX, marlin_mma.h"),
        ("cutlass / cute", "NOT REACHED -- this is what makes it not §5's island"),
        ("cooperative_groups.h", "NOT REACHED -- no grid sync"),
        ("torch / ATen / c10", "already detoxified; marlin.cuh:5-7 records it"),
        ("vllm::ScalarType", "header-only: scalar_type.hpp, an int64 fold, IN THIS TREE"),
    ];

    /// The vendored closure, then everything `csrc/` carries.
    ///
    /// [`source::ALL_HEADERS`] and not [`source::LIBRARY`], because four of
    /// the eleven directives are answered by `csrc/vendor` — `cstdint`,
    /// `type_traits`, `cuda.h` and `cuda_runtime.h`, each of which exists
    /// because §13.6 measured guarding it and REFUSED. A unit for Marlin would
    /// state [`Headers::LibraryAndVendor`] for exactly those four and would be
    /// paying 1.7 MB of FlashInfer it never reads, which is a real cost and is
    /// reported rather than hidden.
    ///
    /// The vendored files go FIRST, so a Marlin header always wins a spelling
    /// contest against a carried one — the ordering rule `flashinfer_probe`'s
    /// `dedup` records, for the same reason.
    ///
    /// [`Headers::LibraryAndVendor`]: kernels_cuda_new::unit::Headers
    fn header_set(closure: &Closure) -> Vec<(String, String)> {
        let mut headers: Vec<(String, String)> = closure.files.clone();
        for h in source::ALL_HEADERS {
            headers.push((h.name.to_string(), h.text.to_string()));
        }
        let mut seen = BTreeSet::new();
        headers.retain(|(n, _)| seen.insert(n.clone()));
        headers
    }

    /// Hand the thing to NVRTC and print what it says.
    ///
    /// Three compiles, because they answer three different questions and only
    /// the third is the one §2 asked:
    ///
    /// 1. The template alone. Type-checks the non-dependent text — 2,082 lines
    ///    of it — and proves the include closure resolves.
    /// 2. One instantiation, by NAME EXPRESSION rather than by a
    ///    `template __global__` line, because that is what a row does.
    /// 3. The same, with the names step 2 reported missing declared in the
    ///    PROBE's own TU. Not a shim and not an edit to `csrc/` — it measures
    ///    whether the gap is a bounded list or the first of many.
    fn compiles(closure: &Closure) -> bool {
        println!("\n   -- the compile, run rather than argued --\n");

        let headers = header_set(closure);
        let bytes: usize = headers.iter().map(|(_, t)| t.len()).sum();
        println!(
            "   header set: {} entries, {} B ({} from the vendored tree, {} from `csrc/`)",
            headers.len(),
            bytes,
            closure.files.len(),
            source::ALL_HEADERS.len()
        );
        println!("   arch:       --gpu-architecture={ARCH}");
        println!("   options:    the crate's, plus `--device-as-default-execution-space`.");
        println!("               That flag is NOT decoration and it is the same one");
        println!("               `flashinfer_probe` needed: `scalar_type.hpp`'s members are");
        println!("               unannotated `constexpr`, and NVRTC calls an unannotated");
        println!("               function a HOST function. nvcc has the same problem under a");
        println!("               different name -- `csrc/CMakeLists.txt:865` passes");
        println!("               `--expt-relaxed-constexpr` for it, and without that the AOT");
        println!("               build fails too (measured: 24 errors out of `dequant.h`).");
        println!("               `Unit::options` is where a Marlin unit would put it.\n");

        // 1. the template, uninstantiated
        let bare = format!("{PRELUDE}");
        match compile(&bare, &headers, None) {
            Ok(b) => println!(
                "   [1] marlin_template.h, no instantiation      COMPILED  {:.0} ms, {} B cubin",
                b.millis,
                b.cubin.len()
            ),
            Err(log) => {
                println!("   [1] marlin_template.h, no instantiation      REFUSED");
                println!("{}", indent(&log));
                return false;
            }
        }

        // 2. one instantiation, named the way a row names one
        let expr = format!("::marlin::Marlin<{MXFP4_ARGS}>");
        let missing = match compile(&bare, &headers, Some(&expr)) {
            Ok(b) => {
                println!(
                    "   [2] + one instantiation, by name expr       COMPILED  {:.0} ms, {} B cubin",
                    b.millis,
                    b.cubin.len()
                );
                Vec::new()
            }
            Err(log) => {
                let names = undefined_names(&log);
                println!("   [2] + one instantiation, by name expr       REFUSED");
                for line in log.lines().filter(|l| l.contains("error")) {
                    println!("       {}", line.trim());
                }
                println!(
                    "\n       Read this precisely. The name expression was ACCEPTED and the\n       \
                     template INSTANTIATED -- NVRTC's own note says\n       \
                     `a_type_id=1125899906909960LL`, which means it evaluated\n       \
                     `vllm::kBFloat16.id()`, a constexpr fold through `pie_marlin_rtc.h`'s\n       \
                     `tuple`, on the way in. What failed is {} device intrinsic(s) the\n       \
                     shim set does not carry. That is a SHIM GAP, not a structure.",
                    names.len()
                );
                names
            }
        };

        // 3. the same, with the gap closed inside the probe's own TU
        let patched = format!("{PATCH}{PRELUDE}");
        let ok = match compile(&patched, &headers, Some(&expr)) {
            Ok(b) => {
                println!(
                    "\n   [3] + those names, declared in the PROBE    COMPILED  {:.0} ms, {} B cubin",
                    b.millis,
                    b.cubin.len()
                );
                println!(
                    "       lowered: {}",
                    b.lowered.as_deref().unwrap_or("(none)")
                );
                println!(
                    "\n       So the gap is exactly {}: `__hadd2` and `atomicAdd` over\n       \
                     `__nv_bfloat162`. Both ARE in NVIDIA's `cuda_bf16.h`; both are the\n       \
                     packed-half2 SIMD family `new-horizon.md` §10.5 already reverted once\n       \
                     (`__hfma2`, `__hsub2`), for the stated reason that emulating one in\n       \
                     fp32 changes the rounding -- so closing it is a parity commit with\n       \
                     its own evidence, which is a KNOWN cost and not an unknown.",
                    if missing.is_empty() { "nothing".into() } else { format!("{} names", missing.len()) }
                );
                true
            }
            Err(log) => {
                println!("\n   [3] + those names, declared in the PROBE    STILL REFUSED");
                println!("{}", indent(&log));
                false
            }
        };

        control();

        if ok {
            println!(
                "\n   VERDICT on §2: Marlin compiles under NVRTC 13.0 for {ARCH}. `Library` --\n   \
                 \"there is no device text in this tree\" -- is refuted twice: the text is\n   \
                 here, and the compiler accepts it."
            );
        }
        ok
    }

    /// The same instantiation through nvcc and NVIDIA's real headers.
    ///
    /// Without this, `[2]`'s two errors read as *"Marlin does not compile"*.
    /// With it they read as *"two names are missing from a shim"*, which is a
    /// different sentence and the one that decides the classification.
    ///
    /// `-Xcompiler=-iquote,<dir>` and never `-I`: §21.10 measured that the
    /// five shims shadow real toolkit headers on the angle-bracket path, that
    /// both spellings compile without a diagnostic, and that the objects
    /// export DIFFERENT mangled symbols. `-I` here would put our `cuda_bf16.h`
    /// in front of NVIDIA's and make the control a control over nothing.
    fn control() {
        let nvcc = std::env::var("CUDA_HOME")
            .map(|h| PathBuf::from(h).join("bin/nvcc"))
            .ok()
            .filter(|p| p.exists())
            .or_else(|| {
                let p = PathBuf::from("/usr/local/cuda-13.0/bin/nvcc");
                p.exists().then_some(p)
            });
        let Some(nvcc) = nvcc else {
            println!("\n   [control] nvcc not found -- skipped.");
            return;
        };

        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target/marlin_probe");
        if std::fs::create_dir_all(&dir).is_err() {
            println!("\n   [control] could not make a scratch directory -- skipped.");
            return;
        }
        let cu = dir.join("control.cu");
        let cubin = dir.join("control.cubin");
        let text = format!(
            "{PRELUDE}namespace marlin {{\ntemplate __global__ void \
             Marlin<{MXFP4_ARGS}>(MARLIN_KERNEL_PARAMS);\n}}\n"
        );
        if std::fs::write(&cu, text).is_err() {
            println!("\n   [control] could not write the scratch TU -- skipped.");
            return;
        }

        let out = std::process::Command::new(&nvcc)
            .args(["-std=c++17", &format!("-arch={ARCH}"), "-cubin", "--expt-relaxed-constexpr"])
            .arg("-o")
            .arg(&cubin)
            .arg(&cu)
            .arg(format!("-Xcompiler=-iquote,{}", marlin_dir().display()))
            .output();

        println!("\n   [control] the SAME instantiation, nvcc + NVIDIA's real headers");
        println!("             (-Xcompiler=-iquote, never -I -- §21.10)");
        match out {
            Ok(o) if o.status.success() => {
                let size = std::fs::metadata(&cubin).map(|m| m.len()).unwrap_or(0);
                println!("             COMPILED  {size} B cubin");
                println!(
                    "             So the SOURCE instantiates. The two names `[2]` reported\n             \
                     are the shim's, not Marlin's."
                );
            }
            Ok(o) => {
                let log = String::from_utf8_lossy(&o.stderr);
                println!("             REFUSED:\n{}", indent(log.lines().take(6).collect::<Vec<_>>().join("\n").as_str()));
            }
            Err(e) => println!("             could not run nvcc: {e}"),
        }
        let _ = std::fs::remove_file(&cu);
        let _ = std::fs::remove_file(&cubin);
    }

    /// The TU every compile here starts from — what a Marlin unit's source
    /// would be, and it is two directives.
    const PRELUDE: &str = "#define MARLIN_NAMESPACE_NAME marlin\n\
                           #include \"kernel.h\"\n\
                           #include \"marlin_template.h\"\n";

    /// The twelve arguments, copied from `sm80_kernel_bfloat16_fe2m1f_bfloat16.cu:5`.
    const MXFP4_ARGS: &str = "vllm::kBFloat16.id(), vllm::kFE2M1f.id(), \
                              vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), \
                              256, 1, 8, 8, true, 4, 2, false";

    /// The two names the shims do not carry, declared in the probe's TU.
    ///
    /// `atomicAdd`'s body is deliberately WRONG — it returns `*p` and adds
    /// nothing. This compiles a name, it does not implement one, and a probe
    /// that shipped a plausible-looking emulation is how a wrong answer gets
    /// adopted. The question here is *"is the gap bounded"*, and a stub
    /// answers it; the real one is a parity commit (§15.2's bar: bit-identity
    /// against the vendor header on the device).
    const PATCH: &str = "#include <cuda_bf16.h>\n\
        __device__ inline __nv_bfloat162 __hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {\n\
        \x20 __nv_bfloat162 r;\n\
        \x20 r.x = __float2bfloat16(__bfloat162float(a.x) + __bfloat162float(b.x));\n\
        \x20 r.y = __float2bfloat16(__bfloat162float(a.y) + __bfloat162float(b.y));\n\
        \x20 return r;\n}\n\
        __device__ inline __nv_bfloat162 atomicAdd(__nv_bfloat162* p, __nv_bfloat162 v) {\n\
        \x20 return *p;  /* NOT an implementation -- see PATCH's doc comment */\n}\n";

    /// The identifiers an NVRTC log calls undefined or unmatched.
    fn undefined_names(log: &str) -> Vec<String> {
        let mut out = BTreeSet::new();
        for line in log.lines() {
            if let Some(rest) = line.split_once("identifier \"") {
                if let Some((name, _)) = rest.1.split_once('"') {
                    out.insert(name.to_string());
                }
            }
            if let Some(rest) = line.split_once("no instance of overloaded function \"") {
                if let Some((name, _)) = rest.1.split_once('"') {
                    out.insert(name.to_string());
                }
            }
        }
        out.into_iter().collect()
    }

    // -----------------------------------------------------------------------
    // §3  Could a row name one?
    // -----------------------------------------------------------------------

    /// The forms `instantiation()` can and cannot produce for this kernel.
    fn naming(closure: &Closure, compiled: bool) {
        println!("\n== 3. Could a ROW name one of them? ==\n");
        if !compiled {
            println!("   Moot: §2 refused. Nothing to name.");
            return;
        }
        println!("   `DeviceKernel::instantiation()` emits");
        println!("     `{PREFIX}{{path}}<{PREFIX}{{elem}}>`");
        println!("   -- the qualification glued ONCE to the front, so slot 1 resolves UNDER");
        println!("   the prefix and slots 2+ at global scope (`argform_probe`, 16 cases).");
        println!("   Marlin's `elem` would be the twelve-argument list, whose slot 1 is");
        println!("   `vllm::kBFloat16.id()` -- a constexpr METHOD CALL on a namespace-scope");
        println!("   object in a `namespace vllm` at GLOBAL scope.\n");

        let headers = header_set(closure);
        let base = format!("{PATCH}{PRELUDE}");
        let aliased = format!("{base}{ALIASES}");

        let cases: &[(&str, &String, String)] = &[
            (
                "as instantiation() emits it TODAY",
                &base,
                format!("{PREFIX}marlin::Marlin<{PREFIX}{MXFP4_ARGS}>"),
            ),
            (
                "...+ two namespace aliases in the unit",
                &aliased,
                format!("{PREFIX}marlin::Marlin<{PREFIX}{MXFP4_ARGS}>"),
            ),
            ("unprefixed, at global scope", &base, format!("::marlin::Marlin<{MXFP4_ARGS}>")),
            (
                "a DIFFERENT shape of the same 15",
                &base,
                "::marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), \
                 vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), 128, 4, 4, 8, false, 4, 2, false>"
                    .to_string(),
            ),
            (
                "a shape in NO generated list",
                &base,
                "::marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(), \
                 vllm::kBFloat16.id(), vllm::kFE8M0fnu.id(), 128, 4, 4, 8, false, 4, 3, false>"
                    .to_string(),
            ),
        ];

        for (what, src, expr) in cases {
            match compile(src, &headers, Some(expr)) {
                Ok(built) => {
                    let lowered = built.lowered.unwrap_or_else(|| "(none)".into());
                    println!("   {what:<40}  OK       {} B cubin", built.cubin.len());
                    // The tail, not the head: the four `ScalarTypeId` folds are
                    // identical across every case and the SHAPE is what differs,
                    // so an elision from the front would hide the finding.
                    println!("   {:<40}           …{}", "", shape_tail(&lowered));
                }
                Err(log) => {
                    println!("   {what:<40}  REFUSED  {}", first_diagnostic(&log));
                    for line in log.lines().filter(|l| l.contains("static_assert")) {
                        println!("   {:<40}           {}", "", line.trim());
                    }
                }
            }
        }

        println!("\n   The two aliases, and they are the WHOLE of what the prefix costs:");
        for line in ALIASES.lines().filter(|l| l.contains("namespace")) {
            println!("       {}", line.trim());
        }
        println!(
            "\n   So §3's answer is YES -- a row CAN name a Marlin instantiation, `elem` and\n   \
             all, once its unit source carries two `namespace` aliases. Neither §22.3's\n   \
             `PLAIN` nor a twelfth argument slot is needed: `elem` is one string and\n   \
             eleven commas are legal inside it (`argform_probe` case 2)."
        );
        let _ = &closure.external;
    }

    /// What a unit's source adds so the prefixed spelling resolves. Reported
    /// as a cost, not adopted: this probe writes nothing into `csrc/`.
    const ALIASES: &str = "namespace pie_cuda_driver { namespace kernels {\n\
                           namespace marlin = ::marlin;\n\
                           namespace vllm = ::vllm;\n\
                           } }\n";

    const PREFIX: &str = "::pie_cuda_driver::kernels::";

    /// The part of a mangled Marlin name that encodes the SHAPE.
    ///
    /// `_ZN6marlin6MarlinILx…ELx…ELx…ELx…E` then `Li256ELi1ELi8ELi8ELb1ELi4ELi2ELb0E`.
    /// The four `Lx` folds are the scalar types and are the same in every case
    /// here; the eight after them are `threads, thread_m_blocks,
    /// thread_n_blocks, thread_k_blocks, m_block_size_8, stages, group_blocks,
    /// is_zp_float`, and they are the whole reason two rows would differ.
    fn shape_tail(s: &str) -> String {
        match s.find("ELi") {
            Some(i) => s[i + 1..].split("EEEv").next().unwrap_or(&s[i + 1..]).to_string(),
            None => s.to_string(),
        }
    }

    /// The first line of an NVRTC log that is a diagnostic, trimmed.
    fn first_diagnostic(log: &str) -> String {
        let line = log
            .lines()
            .find(|l| l.contains("error") || l.contains("warning"))
            .unwrap_or_else(|| log.lines().next().unwrap_or(""))
            .trim();
        let line = line.split_once("error: ").map_or(line, |(_, r)| r);
        if line.len() > 70 { format!("{}...", &line[..67]) } else { line.to_string() }
    }

    fn indent(text: &str) -> String {
        text.lines().map(|l| format!("      {l}")).collect::<Vec<_>>().join("\n")
    }

    // -----------------------------------------------------------------------
    // The NVRTC call, which is `flashinfer_probe`'s minus the crutch
    // -----------------------------------------------------------------------

    struct Built {
        millis: f64,
        lowered: Option<String>,
        cubin: Vec<u8>,
        /// NVRTC's warnings on the SUCCESS path. Empty on every compile here,
        /// which is itself worth keeping: a Marlin compile that starts warning
        /// is a change in the vendored text.
        #[allow(dead_code)]
        log: String,
    }

    /// Compile `source` against `headers`, optionally instantiating `expr`.
    ///
    /// The options are `runtime::nvrtc::options`' — `-std=c++17`,
    /// `--fmad=false`, `--prec-div=true`, `--prec-sqrt=true` — plus
    /// `--device-as-default-execution-space`, which is a MEASURED requirement
    /// and not a convenience: without it NVRTC refuses `scalar_type.hpp` at
    /// line 72 with *"a function without execution space annotations … is
    /// considered a host function"*, seven times over, because
    /// `ScalarType`'s constructor and its five factories are unannotated
    /// `constexpr`. nvcc has the identical problem and `csrc/CMakeLists.txt:865`
    /// answers it with `--expt-relaxed-constexpr`. `Unit::options` is the
    /// mechanism a real row would use, and the flag is spanned by the cache
    /// key, so this is a per-unit fact rather than a global loosening.
    fn compile(
        source: &str,
        headers: &[(String, String)],
        expr: Option<&str>,
    ) -> Result<Built, String> {
        let src = CString::new(source).map_err(|_| "a NUL in the probe source")?;
        let name = c"marlin_probe.cu";

        let texts: Vec<CString> = headers
            .iter()
            .map(|(n, t)| CString::new(t.as_str()).map_err(|_| format!("NUL in {n}")))
            .collect::<Result<_, _>>()?;
        let names: Vec<CString> =
            headers.iter().map(|(n, _)| CString::new(n.as_str()).expect("no NUL")).collect();
        let text_ptrs: Vec<_> = texts.iter().map(|t| t.as_ptr()).collect();
        let name_ptrs: Vec<_> = names.iter().map(|n| n.as_ptr()).collect();

        let mut program: nv::nvrtcProgram = std::ptr::null_mut();
        // SAFETY: every string outlives the call and the two arrays are the
        // same length, which is the whole of `nvrtcCreateProgram`'s contract.
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

        let expression = expr.map(|e| CString::new(e).expect("no NUL in a name expression"));
        if let Some(expression) = &expression {
            // SAFETY: the program is live and the string outlives the call.
            let code = unsafe { nv::nvrtcAddNameExpression(program, expression.as_ptr()) };
            if code != nv::nvrtcResult::NVRTC_SUCCESS {
                return Err(format!("nvrtcAddNameExpression: {code:?}"));
            }
        }

        let gpu = CString::new(format!("--gpu-architecture={ARCH}")).unwrap();
        let options = [
            gpu.as_ptr(),
            c"-std=c++17".as_ptr(),
            c"--fmad=false".as_ptr(),
            c"--prec-div=true".as_ptr(),
            c"--prec-sqrt=true".as_ptr(),
            // See `compiles`: `scalar_type.hpp`'s members are unannotated
            // `constexpr`, and this is the flag that says so. nvcc spells the
            // same need `--expt-relaxed-constexpr`, which `csrc/CMakeLists.txt:865`
            // already passes for exactly this tree.
            c"--device-as-default-execution-space".as_ptr(),
        ];

        let started = std::time::Instant::now();
        // SAFETY: the program is live and the options outlive the call.
        let code = unsafe {
            nv::nvrtcCompileProgram(program, i32::try_from(options.len()).unwrap(), options.as_ptr())
        };
        let millis = started.elapsed().as_secs_f64() * 1e3;
        let log = program_log(program);

        if code != nv::nvrtcResult::NVRTC_SUCCESS {
            // SAFETY: destroyed exactly once, and not used after.
            unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
            return Err(log);
        }

        let mut lowered = None;
        if let Some(expression) = &expression {
            let mut out: *const std::ffi::c_char = std::ptr::null();
            // SAFETY: the program compiled with this expression registered, so
            // NVRTC owns a string for it until the program is destroyed.
            let code =
                unsafe { nv::nvrtcGetLoweredName(program, expression.as_ptr(), &raw mut out) };
            if code == nv::nvrtcResult::NVRTC_SUCCESS && !out.is_null() {
                // SAFETY: NVRTC returned a NUL-terminated string it still owns.
                lowered = Some(unsafe { CStr::from_ptr(out) }.to_string_lossy().into_owned());
            } else {
                return Err(format!("nvrtcGetLoweredName: {code:?} -- the expression compiled but named nothing"));
            }
        }

        let mut size = 0;
        // SAFETY: the program compiled, so a cubin exists; `size` is live.
        unsafe { nv::nvrtcGetCUBINSize(program, &raw mut size) };
        let mut cubin = vec![0u8; size.max(1)];
        // SAFETY: the buffer is exactly the size NVRTC just asked for.
        unsafe { nv::nvrtcGetCUBIN(program, cubin.as_mut_ptr().cast()) };
        cubin.truncate(size);

        // SAFETY: destroyed exactly once, and not used after.
        unsafe { nv::nvrtcDestroyProgram(&raw mut program) };
        Ok(Built { millis, lowered, cubin, log })
    }

    fn program_log(program: nv::nvrtcProgram) -> String {
        let mut size = 0;
        // SAFETY: the program is live.
        unsafe { nv::nvrtcGetProgramLogSize(program, &raw mut size) };
        if size <= 1 {
            return String::new();
        }
        let mut buf = vec![0u8; size];
        // SAFETY: the buffer is exactly the size NVRTC asked for.
        unsafe { nv::nvrtcGetProgramLog(program, buf.as_mut_ptr().cast()) };
        buf.pop();
        String::from_utf8_lossy(&buf).into_owned()
    }
}
