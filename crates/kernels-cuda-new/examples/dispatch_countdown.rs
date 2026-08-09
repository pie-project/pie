//! What stands between here and deleting `kernels-cuda`, counted.
//!
//! The archive dies when no `pie_k_*` entry point is called. Those calls are
//! generated — `abi::emit_rust_dispatch` writes one per stated row that is
//! NOT in [`device::jit_dispatched`] — so the count is a function of one
//! list, and this prints that function's arguments.
//!
//! # Why this is not `migration_status`
//!
//! `migration_status` answers *"is this symbol's kernel text migrated"*.
//! That question was answered 116 times and the archive did not move an
//! inch, because a migrated row still routes through the shim until its
//! symbol is named in [`device::JIT_DISPATCHED`]. **Text migration and
//! dispatch migration are different verbs**, and conflating them is how a
//! migration reports 80% while the thing it is migrating away from has lost
//! nothing.
//!
//! Three facts decide whether a row can be dispatched, and this prints all
//! three per row rather than a verdict, because the third is the one that
//! needs a human:
//!
//! 1. **Migrated** — some unit holds a device row for the symbol.
//! 2. **Unheld** — no C++ translation unit CALLS its host launcher. A
//!    launcher goes when its whole consumer set has gone (§10.10), and the
//!    shim is only one consumer; a `.cu` composing with a `.cu` is a caller
//!    no Rust dispatch can intercept.
//! 3. **Proven** — something has fired it through the JIT and compared
//!    bytes. This file cannot know that, so it prints the first two and
//!    leaves the third to the operator.
//!
//! The held set is stated below rather than computed, because computing it
//! means parsing C++ and the failure mode of a bad parse here is deleting a
//! launcher that is still called. It was measured with a sweep over
//! `.cu`, `.cpp`, `.cuh` AND `.hpp` classifying each occurrence as
//! definition / declaration / call — and the previous measurement, which
//! read `.cu` alone and matched on the `kernel!` macro's first identifier
//! rather than the symbol's last `::` segment, reported **2** where the
//! truth is **11**.

use kernels_cuda_new::{KernelSig, abi, device, table, unit};

/// The tables `driver-cuda/build.rs` hands the emitters, in its order.
///
/// Duplicated from there on purpose and checked by nothing, because the
/// alternative is `driver-cuda` depending on this example or this example on
/// a build script's private function. What it costs if it drifts is one row
/// reported ARMLESS that is armed — visible the moment it is routed, since
/// `routed_rows_have_an_arm` reads the real one.
fn tables() -> Vec<&'static [KernelSig]> {
    vec![
        table::attn::KERNELS,
        table::rope::KERNELS,
        table::norm::KERNELS,
        table::mlp::KERNELS,
        table::gemm::KERNELS,
        table::moe::KERNELS,
        table::ssm::KERNELS,
        table::quant::KERNELS,
        table::layout::KERNELS,
        table::sample::KERNELS,
        table::adapter::KERNELS,
        table::driver_internal::DRIVER_KERNELS,
    ]
}

/// The symbols a C++ translation unit still CALLS, with the caller.
///
/// Re-measured 2026-08-13 (second sweep, same day) over
/// `crates/kernels-cuda/csrc/src/**/*.{cu,cpp}`. `abi::cpp_path` formats a
/// symbol as `::pie_cuda_driver::kernels::{symbol}`, so the C++ function name
/// is the symbol's last `::` segment — NOT the `kernel!` macro's first
/// identifier, which is the row's ABI name and matched only 33 of 218 names
/// when used by mistake.
///
/// **The first sweep's eleven are nine, and the two that left are the point.**
/// This table is a dated measurement of somebody else's tree, and between the
/// two sweeps of one afternoon:
///
/// * `mlp::chunked_swiglu_bf16` lost its only caller when
///   `chunked_swiglu_strided_bf16` was deleted — a hold that was itself
///   unreachable, which `mlp/swiglu.cu:41` now calls "orphaned at one remove".
/// * `norm::add_bias_bf16` lost its when the GEMM epilogue absorbed the
///   addition; `gemm/gemm.cpp:2236` records that the second call *"is why it
///   moved"*.
/// * `norm::rmsnorm_bf16` was cited as seven calls across
///   `norm/rmsnorm.cu` and `vision/gemma4_vision.cu`. The vision tower's
///   `.cu` no longer exists at all, and the count is two.
/// * Six of the nine survivors moved by 22 to 100 lines.
///
/// Nine of eleven entries were wrong within hours, and nothing said so,
/// because [`the_citations_still_resolve`] did not exist and the check that
/// did only asked whether the SYMBOL was still migrated — the half this table
/// does not get wrong. A held row is a row this migration is not allowed to
/// touch, so an entry that has quietly stopped being true costs the countdown
/// a row it could have had.
const HELD: &[(&str, &str)] = &[
    ("norm::residual_add_bf16", "gemm/gemm.cpp:2030,2124"),
    ("norm::rmsnorm_bf16", "norm/rmsnorm.cu:59,63"),
    ("norm::rmsnorm_strided_bf16", "norm/rmsnorm.cu:42"),
    ("quant::bf16_to_fp16", "norm/rmsnorm.cu:64"),
    ("quant::dequant_fp8_e4m3_to_bf16", "gemm/gemm.cpp:1732"),
    ("quant::dequant_fp8_e4m3_to_bf16_per_channel", "gemm/gemm.cpp:1721"),
    ("quant::dequant_fp8_e4m3_to_bf16_per_group", "gemm/gemm.cpp:1714"),
    ("quant::dequant_mxfp4_to_bf16", "gemm/gemm.cpp:2163"),
    ("quant::quantize_bf16_to_int8_per_channel", "quant/quant_bf16_to_fp8.cu:84"),
];

/// Every [`HELD`] citation names a file that exists and a line that calls the
/// symbol — checked here rather than trusted.
///
/// This is the check that was missing, and it is missing in a specific way
/// worth naming: the staleness check at the bottom of `main` asks whether a
/// held SYMBOL is still a migrated row, which is a question about this
/// crate's own tables. The citation is a claim about somebody else's tree,
/// and it is the half that rots. A gate whose denominator is a set the
/// claimant supplies will pass for as long as the claimant is consistent with
/// itself; §39 lost three months to exactly that shape, and §21 named it.
///
/// It reads the archive's sources directly, so it fails loudly when the
/// archive moves under it — which is the intended behaviour, not a
/// fragility. The alternative is a table that is silently wrong.
fn the_citations_still_resolve() -> (Vec<String>, Vec<String>) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../kernels-cuda/csrc/src");
    let mut wrong = Vec::new();
    let mut freed = Vec::new();
    for (symbol, citation) in HELD {
        // `norm::rmsnorm_bf16` -> `rmsnorm_bf16`, which is what C++ spells.
        let name = symbol.rsplit("::").next().unwrap_or(symbol);
        for part in citation.split(" + ") {
            let (path, lines) = match part.split_once(':') {
                Some((p, l)) => (p, l),
                None => (part, ""),
            };
            let file = root.join(path.trim());
            let text = std::fs::read_to_string(&file).ok();
            let calls: Vec<usize> = text
                .as_deref()
                .map(|t| {
                    t.lines()
                        .enumerate()
                        .filter(|(_, l)| {
                            let s = l.trim_start();
                            !s.starts_with("//") && !s.starts_with('*') && s.contains(name)
                        })
                        .filter(|(_, l)| !l.contains(&format!("void {name}")))
                        .map(|(i, _)| i + 1)
                        .collect()
                })
                .unwrap_or_default();
            if calls.is_empty() {
                // THE TWO WAYS A CITATION STOPS RESOLVING ARE OPPOSITES, AND
                // THIS USED TO REPORT BOTH AS exit 101.
                //
                // A cited line that moved is rot: the table is wrong and the
                // countdown is reading a stale claim. A symbol with no caller
                // ANYWHERE is the migration succeeding -- somebody deleted the
                // last C++ consumer and the row is now routable. Four agents
                // are cutting exactly these call sites as this runs
                // (`gemm.cpp`'s six, `rmsnorm.cu`, `quant_bf16_to_fp8.cu`),
                // so treating a freed row as a hard failure would fire on the
                // integrator's single pass and cost an edit to say "good".
                //
                // The distinction is measurable and costs one sweep: ask
                // whether ANY tree still calls it, using the same predicate
                // `no_c_caller_hides_outside_the_archive` uses, so the two
                // gates cannot disagree about what a call is.
                let elsewhere = where_it_is_called(name);
                if elsewhere.is_empty() {
                    freed.push(format!(
                        "{symbol}: no C++ caller left in any tree (cited {path}) — \
                         DELETE this HELD row, the countdown gains one"
                    ));
                } else {
                    wrong.push(format!(
                        "{symbol}: cites {path}, which no longer calls it, but it is \
                         still called from {} — re-cite, do not free",
                        elsewhere.join(", ")
                    ));
                }
                continue;
            }
            for want in lines.split(',').filter(|s| !s.is_empty()) {
                let Ok(want) = want.trim().parse::<usize>() else { continue };
                if !calls.contains(&want) {
                    wrong.push(format!(
                        "{symbol}: cites {path}:{want}, but the calls are at {calls:?}"
                    ));
                }
            }
        }
    }
    (wrong, freed)
}

/// Every `file:line` in either C++ tree that calls `name`.
///
/// Shared with [`no_c_caller_hides_outside_the_archive`] so that "is this
/// still called" has exactly one answer in this file. Two gates with two
/// predicates is how a set comes to be held by one and free by the other.
fn where_it_is_called(name: &str) -> Vec<String> {
    let mut out = Vec::new();
    for tree in c_trees() {
        let mut files = Vec::new();
        collect_sources(&tree, &mut files);
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else { continue };
            let hits: Vec<usize> = text
                .lines()
                .enumerate()
                .filter(|(_, l)| {
                    let s = l.trim_start();
                    !s.starts_with("//") && !s.starts_with('*')
                })
                .filter(|(_, l)| calls(l, name))
                .map(|(i, _)| i + 1)
                .collect();
            if !hits.is_empty() {
                let shown = file.strip_prefix(&tree).unwrap_or(&file).display();
                out.push(format!("{shown}:{hits:?}"));
            }
        }
    }
    out
}

/// The C++ trees that can call a launcher. Was one path hard-coded in two
/// places; `towers-move` made it two on 2026-08-14 and there is no reason to
/// think two is final.
fn c_trees() -> Vec<std::path::PathBuf> {
    let here = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    vec![
        here.join("../driver-cuda/csrc"),
        here.join("../kernels-cuda/csrc/src"),
    ]
}

/// The archive is no longer the only C++ tree that calls a launcher — and
/// [`HELD`] was never checked for the entries it LACKS.
///
/// [`the_citations_still_resolve`] reads `kernels-cuda/csrc/src`, because for
/// the whole of this migration that was where C++ callers lived, and "held"
/// has quietly meant "called from THAT directory" ever since. On 2026-08-14
/// `towers-move` relocated the three multimodal towers to
/// `driver-cuda/csrc/vision/`, on the argument that a host walk over JIT-tree
/// device text is this crate's kind of object — an argument I agree with and
/// which does not change the fact that the towers still call `kernels::` by
/// name. So there is now a second tree, and a held symbol could move into it
/// and be reported FREE, which is worse than being reported held: the
/// countdown would hand it to a router as provable and its launcher deletion
/// would break a tower.
///
/// That is the denominator failing again in its cheapest form — the set was
/// never "callers", it was "callers in one hard-coded path", and the path
/// stopped being the whole story without the constant changing. But writing
/// it exposed a second and larger hole. `the_citations_still_resolve` can
/// only ever validate the nine claims [`HELD`] makes; **nothing has ever
/// asked whether a tenth is missing.** So this sweeps BOTH trees against the
/// FULL migrated set, because a table cannot be used to check itself — §21,
/// and §39's five-line lookup, and the count is now nine times this session.
///
/// **It is checked in both directions rather than trusted in either.**
///
/// * *Negative control.* Pointed at the archive with [`calls`] in its first
///   form it returned 20 findings, every one of them `device::name<<<...>>>`
///   — a launcher's own body launching the kernel it shares a name with. The
///   predicate is documented with what that cost.
/// * *Positive control.* With the [`HELD`] filter emptied, the sweep over the
///   archive returns **exactly the nine held symbols at exactly their nine
///   cited line numbers** — `gemm.cpp:2030,2124`, `rmsnorm.cu:42`, `:59,63`,
///   `:64`, `quant_bf16_to_fp8.cu:84` and the four `gemm.cpp` dequants. A
///   table measured by hand and a sweep derived from the sources agree
///   symbol-for-symbol and line-for-line, which is the first time in this
///   session two independent methods have agreed on the held set without one
///   of them having supplied the other's denominator.
///
/// So a green gate here means the sweep looked and found nothing, not that it
/// could not look. Today the towers call `gemm::act_x_wt_bf16`,
/// `attn::make_prefill_plan` and the two FlashInfer prefill entry points,
/// none of which is a migrated row.
fn no_c_caller_hides_outside_the_archive(migrated: &[&'static str]) -> Vec<String> {
    let trees = c_trees();
    let held: Vec<&str> = HELD.iter().map(|(s, _)| *s).collect();
    let mut found = Vec::new();
    for tree in &trees {
        let mut files = Vec::new();
        collect_sources(tree, &mut files);
        for file in files {
            let Ok(text) = std::fs::read_to_string(&file) else { continue };
            let shown = file.strip_prefix(tree).unwrap_or(&file).display().to_string();
            for symbol in migrated {
                let name = symbol.rsplit("::").next().unwrap_or(symbol);
                let hits: Vec<usize> = text
                    .lines()
                    .enumerate()
                    .filter(|(_, l)| {
                        let t = l.trim_start();
                        !t.starts_with("//") && !t.starts_with('*')
                    })
                    .filter(|(_, l)| calls(l, name))
                    .map(|(i, _)| i + 1)
                    .collect();
                if hits.is_empty() || held.contains(symbol) {
                    continue;
                }
                found.push(format!(
                    "{symbol}: called from {shown}:{hits:?}, and it is not in HELD — \
                     the countdown is offering this row as FREE while a C++ caller \
                     still needs its launcher"
                ));
            }
        }
    }
    found
}

/// A call to `name`, not a mention of it, and not its own implementation.
///
/// Three exclusions, each one measured rather than guessed. The first sweep
/// of the archive with this predicate returned 20 findings and every one I
/// read was `device::name<<<...>>>` — a launcher's own body launching the
/// `__global__` kernel it shares a name with. That is not a caller; it is the
/// definition. §39 records the 2-vs-11 disagreement coming from a sweep that
/// matched the row's ABI name instead of the C++ symbol, and this is the same
/// error reflected: matching the KERNEL where the LAUNCHER was meant. So:
///
/// * an identifier boundary before, and `(` or `<` after — a mention in prose
///   or a longer identifier is not a call;
/// * not qualified by `device::`, which in this archive is exactly the
///   namespace the `__global__` kernels live in and the launchers do not;
/// * not followed by `<<<`, with or without a template argument list, because
///   a launch configuration is the one syntax C++ has that a host call cannot
///   wear.
///
/// Deliberately narrow: it will under-report rather than invent a caller,
/// because the cost of a false hold is one row left in the countdown and the
/// cost of a false free is a deleted launcher and a broken tower.
fn calls(line: &str, name: &str) -> bool {
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(at) = line[from..].find(name) {
        let start = from + at;
        let end = start + name.len();
        let before_ok = start == 0 || {
            let c = bytes[start - 1] as char;
            !c.is_alphanumeric() && c != '_'
        };
        let after_ok = matches!(bytes.get(end).map(|b| *b as char), Some('(') | Some('<'));
        let is_kernel = line[..start].trim_end().ends_with("device::")
            || line[end..].trim_start().starts_with("<<<")
            || line[end..]
                .split_once(">>>")
                .map(|(head, _)| head.starts_with('<') && !head.contains('('))
                .unwrap_or(false);
        if before_ok && after_ok && !is_kernel && !line.contains(&format!("void {name}")) {
            return true;
        }
        from = end;
    }
    false
}

fn collect_sources(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(listing) = std::fs::read_dir(dir) else { return };
    for entry in listing.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_sources(&path, out);
        } else if matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("cu") | Some("cpp") | Some("cuh") | Some("hpp")
        ) {
            out.push(path);
        }
    }
}

fn main() {
    let aot: Vec<&'static str> = table::KERNELS.iter().map(|k| k.symbol).collect();
    let migrated: Vec<&'static str> = unit::rows().map(|r| r.sig.symbol).collect();
    let dispatched = device::JIT_DISPATCHED;

    // The three sets, and the one that is the countdown.
    let mut ready: Vec<&'static str> = Vec::new();
    let mut held: Vec<(&'static str, &'static str)> = Vec::new();
    let mut already: Vec<&'static str> = Vec::new();
    for symbol in &migrated {
        if !aot.contains(symbol) {
            // A device row with no AOT twin has no shim entry to remove: it
            // was never in the archive, so it is not part of this countdown.
            continue;
        }
        if dispatched.contains(symbol) {
            already.push(symbol);
        } else if let Some((_, by)) = HELD.iter().find(|(s, _)| s == symbol) {
            held.push((symbol, by));
        } else {
            ready.push(symbol);
        }
    }
    ready.sort_unstable();
    held.sort_unstable();

    // WHICH READY ROWS ACTUALLY HAVE AN ARM.
    //
    // "READY" above means only *hosted, twinned and unrouted*. It is a
    // statement about tables. Routing needs one more thing that no table
    // says: `emit_rust_dispatch` must WRITE an arm for the row, and it
    // silently declines two ways -- an operand carrying `Source::Unbound`
    // skips the row whole (link error at route time, because a hand-written
    // arm has been calling the shim entry routing deletes), and an operand
    // whose `Ty` has no `ArgValue` variant skips only the JIT branch
    // (`UnknownKernel` at fire time). Four of `layout`'s eight rows were the
    // first kind and `driver-cuda/build.rs::routed_rows_have_an_arm` caught
    // them AFTER they were named here.
    //
    // The probe emitter answers the question before the name is written, and
    // it is the same emitter, so this cannot drift from what routing does.
    let probe = abi::emit_rust_dispatch_probe(&tables(), &unit::rows().collect::<Vec<_>>());
    let armed = |symbol: &str| probe.lines().any(|l| l.starts_with(&format!("\"{symbol}\"")));
    let (arm_ready, arm_less): (Vec<&str>, Vec<&str>) =
        ready.iter().partition(|s| armed(s));

    println!("stated rows (AOT tables)     {}", aot.len());
    println!("migrated rows (units)        {}", migrated.len());
    println!("  of those, with an AOT twin {}", ready.len() + held.len() + already.len());
    println!();
    println!("ALREADY DISPATCHED           {:3}", already.len());
    println!("HELD by a C++ caller         {:3}", held.len());
    println!("READY, unproven              {:3}   <- the countdown", ready.len());
    println!("  of those, ARMED            {:3}   <- routable today", arm_ready.len());
    println!("  of those, ARMLESS          {:3}   <- needs a Source or an ArgValue first", arm_less.len());
    println!();

    println!("== held ==");
    for (s, by) in &held {
        println!("  {s:52} {by}");
    }
    println!();

    // BY UNIT, because a unit is one NVRTC compile and therefore the
    // granularity at which a batch either works or fails with the file
    // named. A batch spread over units fails with a symbol.
    println!("== ready, by unit ==");
    for u in unit::UNITS {
        let mine: Vec<&str> = u
            .rows
            .iter()
            .map(|r| r.sig.symbol)
            .filter(|s| ready.contains(s))
            .collect();
        if mine.is_empty() {
            continue;
        }
        let (a, l): (Vec<&str>, Vec<&str>) = mine.iter().partition(|s| armed(s));
        println!("  {:44} {:2} armed, {:2} armless", u.name, a.len(), l.len());
        for s in &a {
            println!("      {s}");
        }
        for s in &l {
            println!("      {s}   [ARMLESS]");
        }
    }

    // A HELD ENTRY THAT NAMES NOTHING IS A STALE MEASUREMENT.
    //
    // The sweep behind `HELD` is dated. A symbol that has since been deleted
    // or renamed would sit here forever, holding a launcher that no longer
    // exists — the same shape of error as a wall in front of a door nobody
    // opens, one layer down.
    let stale: Vec<&str> =
        HELD.iter().map(|(s, _)| *s).filter(|s| !migrated.contains(s)).collect();
    if !stale.is_empty() {
        println!();
        println!("!! {} HELD entries name no migrated row: {stale:?}", stale.len());
        println!("   Either the sweep is stale or the symbol moved. Re-measure.");
        std::process::exit(101);
    }

    // AND THE OTHER HALF, which is the one that actually rots.
    let (wrong, freed) = the_citations_still_resolve();
    if !freed.is_empty() {
        println!();
        println!("** {} HELD row(s) have been FREED:", freed.len());
        for f in &freed {
            println!("   {f}");
        }
        println!("   Not a failure. Delete the HELD lines and route them.");
    }
    if !wrong.is_empty() {
        println!();
        println!("!! {} HELD citation(s) no longer hold:", wrong.len());
        for w in &wrong {
            println!("   {w}");
        }
        println!("   A held row is a row nobody is allowed to route. Re-measure.");
        std::process::exit(101);
    }

    // AND THE TREE THE CITATIONS DO NOT COVER.
    let hidden = no_c_caller_hides_outside_the_archive(&migrated);
    if !hidden.is_empty() {
        println!();
        println!("!! {} C++ caller(s) live outside the archive:", hidden.len());
        for h in &hidden {
            println!("   {h}");
        }
        println!(
            "   Add them to HELD with a re-measured citation, or the countdown is offering \
             a row whose launcher another tree still needs."
        );
        std::process::exit(101);
    }
}
