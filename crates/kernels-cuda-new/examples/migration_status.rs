//! How far the migration has got, by symbol.
//!
//! # Why a report rather than a gate
//!
//! `tests/units.rs` is the gate: every unit that exists must compile and every
//! row must resolve. It says nothing about what does NOT exist yet, and that
//! is the number a migration is actually steered by — which families are done,
//! which are half-done, and which symbol is next.
//!
//! So this walks the two tables and joins them on the one thing they share.
//! An ahead-of-time row in [`kernels_cuda_new::table`] describes a `pie_k_*`
//! entry point: a host launcher holding a `<<<>>>`, taking a stream. A JIT row
//! in a [`kernels_cuda_new::families`] module describes a template
//! instantiation and states its geometry as a `LaunchRule`. **The symbol is
//! the same string in both**, deliberately — `model-compiler` writes it into a
//! trace, and a kernel that changed its name on migration would be a kernel no
//! existing model text can state.
//!
//! # What "migrated" means here, and what it does not
//!
//! A symbol counts as migrated when some unit hosts it. That means its device
//! text is extracted, its geometry is a rule, and `tests/units.rs` has
//! compiled it and resolved its mangled name.
//!
//! It does **not** mean the `.cu` launcher is gone. `new-horizon.md` §10.10
//! fixes the order and the last step is the one this report cannot see: a
//! launcher goes when its whole consumer set has gone, and the shim is only
//! one consumer. `norm::residual_add_bf16` is the standing example — migrated,
//! fireable, and still called from `gemm.cpp` and `gemma4_vision.cu`.
//!
//! # The denominator is wrong a second time, and this is the fix
//!
//! `Prepare` rows left the denominator because they dispatch into FlashInfer
//! and are not kernels of ours to migrate. That reasoning does not stop at
//! `Prepare`. A row for `gemm::act_x_wt_bf16_out_fp32` names a launcher in
//! `gemm/gemm.cpp`, which is host C++ compiled by `g++` — one `cublasGemmEx`,
//! no `__global__` and no `<<<>>>` of its own. There is nothing to extract.
//! A row for `gemm::gemv3_bf16` names a launcher that reads
//! `cudaDevAttrComputeCapabilityMajor` and `getenv` to pick one of two
//! instantiations and RETURNS `bool` to mean *"I did not launch — use
//! cuBLAS"*; a row names one instantiation and a row cannot decline.
//! `qwen35_verify_stash_store` names no C++ function at all.
//!
//! None of those becomes a row by adding vocabulary. Counting them as
//! unmigrated makes the migration look like it has 81 units of work left
//! when it has 45, and — worse — implies the number can reach 100%, which it
//! cannot. **The migration can be COMPLETE at a percentage below 100.**
//!
//! So this reports two denominators: `rows`, which is every non-`Prepare`
//! row, and `reachable`, which is `rows` minus [`Class::Structural`]. The
//! four classes below partition the refused set, [`assert_total`] holds that
//! they do, and the classification is a TABLE here rather than a scan of the
//! family modules' prose — `families/*.rs` is being edited by another agent
//! and a report that read its sentences would measure the edit, not the tree.
//!
//! # And the denominator was wrong a THIRD time: three kinds, not one
//!
//! `reachable` still says every row it counts is a kernel somebody could
//! migrate. That is false of two groups, and they fail differently:
//!
//! * a **service** — `dist::all_reduce_bf16` is NCCL, which the DRIVER links
//!   and this crate has never included; `moe::flashinfer_cutlass_moe_bf16` is
//!   CUTLASS, whose source CPM fetches at configure time. Never a kernel of
//!   ours, so never work, so never a percentage.
//! * an **op** — `gemm::act_x_wt_channel_scaled` is a quantize, a library
//!   GEMM, a dequant and sometimes a residual-add, three of the four kernels
//!   OURS AND TWO OF THEM ALREADY MIGRATED. The kernels underneath are not
//!   the wall; the COMPOSITION is, and a row has no way to say it.
//!
//! [`kernels_cuda_new::execution::Execution`] is that distinction as data,
//! beside the table where `DeviceKernel` already lives. This report is where
//! it is counted: [`kind_of_wall`] maps each [`Wall`] onto a [`Kind`], the
//! three kinds partition all 198 stated symbols, and [`assert_total`] derives
//! the service set twice — once from the walls here, once from
//! `execution::SERVED` — and asserts the two agree as SETS.
//!
//! The classification is only as good as the row-by-row evidence, and the bar
//! is deliberately high: **no `__global__` in the symbol's closure, or a
//! library whose source is not in this repository.** Applying it moved six
//! rows OUT of `Wall::Library` — every dense and quantized `gemm::` entry
//! point on record as "cuBLASLt" turned out to reach `gemm/gemv.cu`, or
//! `quant::`'s dequant kernels, or `norm::add_bias_bf16`. `Wall::Library`'s
//! own doc keeps the tally, because the count of corrections is the argument
//! for making the claim checkable in the first place.
//!
//! ```text
//! cargo run -p kernels-cuda-new --example migration_status
//! ```

use std::collections::{BTreeMap, BTreeSet};

use kernels_cuda_new::execution::{self, Kind};
use kernels_cuda_new::{table, unit};

/// The stated symbols this report is about: every row that is not a
/// `Prepare`.
///
/// One definition, used by the count, by [`kind_of`]'s fold and by
/// [`assert_total`] — because a denominator derived three times three ways is
/// three chances to disagree with itself.
fn candidates() -> impl Iterator<Item = &'static str> {
    table::KERNELS
        .iter()
        .filter(|row| row.needs == kernels_cuda_new::Prepare::None)
        .map(|row| row.symbol)
}

fn main() {
    // Every symbol some unit can compile, and which unit compiles it.
    let hosted: BTreeMap<&str, &str> = unit::UNITS
        .iter()
        .flat_map(|u| u.rows.iter().map(move |row| (row.sig.symbol, u.name)))
        .collect();

    // The ahead-of-time table, grouped by the family its symbol names. A
    // symbol is `family::kernel`, which is what makes this a join rather than
    // a guess.
    //
    // Rows that state a `Prepare` are counted SEPARATELY and excluded from the
    // denominator, because they are not kernels of ours to migrate. A
    // `Prepare::DecodePlan` row dispatches into FlashInfer — the launcher
    // calls a library, the library launches its own kernels, and there is no
    // `__global__` in this tree to extract or `LaunchRule` to state. Counting
    // them as unmigrated made `attn` read 24% when 10 of its 49 rows were
    // never candidates; the number was wrong in a way that pointed work at the
    // wrong family. FlashInfer is migrating on its own track — vendored,
    // guarded and JIT-compiled (`new-horizon.md` §14) — and that is a
    // different project with a different gate.
    let mut families: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    let mut library: BTreeMap<&str, usize> = BTreeMap::new();
    let mut refused: BTreeSet<&str> = BTreeSet::new();
    for row in table::KERNELS {
        let family = row.symbol.split("::").next().unwrap_or("?");
        if row.needs != kernels_cuda_new::Prepare::None {
            *library.entry(family).or_default() += 1;
            continue;
        }
        let entry = families.entry(family).or_default();
        entry.0 += 1;
        if hosted.contains_key(row.symbol) {
            entry.1 += 1;
        } else {
            refused.insert(row.symbol);
        }
    }

    println!("\nMigration status -- ahead-of-time rows with a JIT twin\n");
    println!(
        "{:<12} {:>7} {:>9} {:>7}   {:<22} {}",
        "family", "rows", "migrated", "", "", "library"
    );
    println!("{}", "-".repeat(74));

    let (mut total, mut done) = (0, 0);
    for (family, (rows, migrated)) in &families {
        total += rows;
        done += migrated;
        let bar = bar(*migrated, *rows);
        let lib = library.get(*family).copied().unwrap_or(0);
        let lib = if lib == 0 { String::new() } else { format!("+{lib} library") };
        println!(
            "{family:<12} {rows:>7} {migrated:>9} {:>6.0}%   {bar}  {lib}",
            percent(*migrated, *rows)
        );
    }
    println!("{}", "-".repeat(74));
    let lib_total: usize = library.values().sum();
    println!(
        "{:<12} {total:>7} {done:>9} {:>6.0}%   {}  +{lib_total} library\n",
        "TOTAL",
        percent(done, total),
        bar(done, total)
    );
    println!(
        "`library` rows state a `Prepare` -- they dispatch into FlashInfer rather\n\
than launching a kernel this tree holds, so they are not candidates and are\n\
out of the denominator below. Read in this report's vocabulary they are\n\
SERVICES too, and they are counted apart only because they were out of the\n\
denominator before the word existed. See `new-horizon.md` §14 for their track.\n"
    );

    report_partition(&refused, total, done, lib_total);

    // The units, and what each carries. A unit with one row is usually a file
    // that has only begun to be split; a unit with none cannot exist, because
    // `tests/units.rs` refuses it.
    println!("{} unit(s):", unit::UNITS.len());
    for u in unit::UNITS {
        println!("  {:<34} {:>3} row(s)", u.name, u.rows.len());
    }

    // Rows a unit hosts that the ahead-of-time table does not state. These are
    // not errors — they are the kernels a JIT costs a line and an AOT build
    // costs a translation unit, which is the whole argument for the design.
    // `norm::residual_add_f16` is the first of them.
    let extra: Vec<&str> = hosted
        .keys()
        .filter(|symbol| !table::KERNELS.iter().any(|row| row.symbol == **symbol))
        .copied()
        .collect();
    if !extra.is_empty() {
        println!(
            "\n{} row(s) with no ahead-of-time twin -- kernels the AOT build never had,\n\
             because instantiating one cost a translation unit of `cicc` for something\n\
             nobody had asked for yet:",
            extra.len()
        );
        for symbol in extra {
            println!("  {symbol}");
        }
    }

    // The next thing to do, named. A family that is started and unfinished is
    // where the cheapest remaining work is, because its `.cuh` already exists.
    let started: Vec<&str> = families
        .iter()
        .filter(|(_, (rows, migrated))| *migrated > 0 && migrated < rows)
        .map(|(family, _)| *family)
        .collect();
    if started.is_empty() {
        println!("\nNo family is half-migrated.");
    } else {
        println!("\nHalf-migrated, and therefore cheapest to finish: {}", started.join(", "));
    }
}

/// The partition of the refused set, printed, after [`assert_total`] has held
/// that it is one.
fn report_partition(refused: &BTreeSet<&str>, rows: usize, migrated: usize, library: usize) {
    assert_total(refused, rows, migrated, library);

    let of = |symbol: &str| CLASSIFIED.iter().find(|(s, _, _)| *s == symbol).map(|(_, c, r)| (*c, *r));
    let mut structural: Vec<(&str, Wall, &str)> = Vec::new();
    let mut text: Vec<(&str, &str)> = Vec::new();
    let mut stale: Vec<(&str, &str)> = Vec::new();
    let mut vocabulary: Vec<&str> = Vec::new();
    for symbol in refused {
        match of(symbol) {
            Some((Class::Structural(wall), why)) => structural.push((symbol, wall, why)),
            Some((Class::Text, why)) => text.push((symbol, why)),
            Some((Class::Stale, why)) => stale.push((symbol, why)),
            None => vocabulary.push(symbol),
        }
    }
    let (a, b, c, d) = (structural.len(), vocabulary.len(), text.len(), stale.len());
    let reachable = rows - a;

    // The kind partition, over every candidate and not merely the refused
    // ones. `kinds[Kind]` is how many of the 198 stated symbols are that
    // kind; `unmigrable_kernels` is the part of the floor that is still a
    // kernel, which is the only part of it that is WORK.
    let mut kernels = 0;
    let mut ops = 0;
    let mut services = 0;
    for symbol in candidates() {
        match kind_of(symbol) {
            Kind::Kernel => kernels += 1,
            Kind::Op => ops += 1,
            Kind::Service => services += 1,
        }
    }
    let unmigrable_kernels = structural.iter().filter(|(_, wall, _)| kind_of_wall(*wall) == Kind::Kernel).count();

    // The ops, and how far step two got with each. `composed` is how many of
    // the ten state a `Composition`; `fireable` is how many of THOSE this
    // crate can actually run end to end, which is smaller and is the number
    // that has fire evidence behind it. Keeping them apart is the same
    // honesty as keeping `rows` apart from `reachable`: a composition whose
    // first step is an unmigrated kernel is a true statement of the op's
    // shape and not a thing that runs.
    let composed = execution::COMPOSED.iter().filter(|c| candidates().any(|s| s == c.symbol)).count();
    let fireable = execution::COMPOSED.iter().filter(|c| c.fireable()).count();

    println!("{}", "=".repeat(74));
    println!("{migrated:>3} of {rows:>3} stated symbols      {:>3.0}%", percent(migrated, rows));
    println!(
        "{migrated:>3} of {kernels:>3} KERNELS             {:>3.0}%   {services} services and {ops} ops are not kernels",
        percent(migrated, kernels)
    );
    println!(
        "{migrated:>3} of {reachable:>3} reachable KERNELS   {:>3.0}%   {unmigrable_kernels} kernels behind a structural wall",
        percent(migrated, reachable)
    );
    println!(
        "{composed:>3} of {ops:>3} OPS composed        {:>3.0}%   {fireable} of them fires here; the rest are stated, not run",
        percent(composed, ops)
    );
    println!(
        "{:>3} of {:>3} symbols OURS TO RUN {:>3.0}%   kernels + ops, which is every row no library serves",
        migrated + composed,
        kernels + ops,
        percent(migrated + composed, kernels + ops)
    );
    println!("{}\n", "=".repeat(74));

    println!(
        "Every stated symbol is a KERNEL ({kernels}), an OP ({ops}) or a SERVICE ({services}).\n\
         `Execution` in `src/execution.rs` is that distinction as data; `kind_of_wall`\n\
         is the mapping, and `assert_total` derives the service set twice -- from the\n\
         walls here and from `execution::SERVED` -- and asserts the two agree.\n\
         \n  \
         A SERVICE is never work: cuBLAS, CUTLASS, NCCL, the P2P all-reduce plane, or\n  \
         the driver itself. It cannot migrate because it was never a kernel of ours.\n  \
         An OP is a host program over kernels of ours, most of which migrate already;\n  \
         what a row cannot say is the composition -- and `Step` now says some of it.\n  \
         So the third line is the migration's real denominator, and the second is the\n  \
         one that shows how much of the shortfall is not work at all.\n"
    );

    println!("The refused {} rows, partitioned. Every row lands in exactly one class.\n", refused.len());
    println!("  A  structurally unmigrable  {a:>3}   no device text here, or a host decision a row cannot make");
    println!("  B  waiting on vocabulary    {b:>3}   a rule, a `Ty`, a `Term`, a `Source`, a `Dims` field");
    println!("  C  waiting on text          {c:>3}   the `.cu` is not split into a `.cuh` yet");
    println!("  D  refused for a stale reason {d:>1}   the vocabulary has grown past the refusal on record");
    println!("     {}", "-".repeat(60));
    println!("     total                   {:>3}   == {} refused\n", a + b + c + d, refused.len());

    // A, broken down twice: by kind, because that is what decides whether a
    // row is work, and then by wall, because a floor nobody can decompose is
    // a floor nobody can argue with.
    println!(
        "A. Structurally unmigrable ({a}) -- and only {unmigrable_kernels} of them are KERNELS:\n\
         \n  \
         {:>3} service(s)   never a kernel; `execution::SERVED` names who runs each\n  \
         {:>3} op(s)        a host program over kernels of ours -- step two\n  \
         {unmigrable_kernels:>3} kernel(s)    one launch, an instantiation a row cannot choose\n",
        structural.iter().filter(|(_, w, _)| kind_of_wall(*w) == Kind::Service).count(),
        structural.iter().filter(|(_, w, _)| kind_of_wall(*w) == Kind::Op).count(),
    );
    let mut by_wall: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (symbol, wall, _) in &structural {
        by_wall.entry(format!("{wall:?}")).or_default().push(symbol);
    }
    for (wall, symbols) in &by_wall {
        let kind = structural
            .iter()
            .find(|(_, w, _)| format!("{w:?}") == *wall)
            .map(|(_, w, _)| kind_of_wall(*w))
            .expect("a wall with members has a kind");
        println!("  {:<14} {:>2}   -> {kind:?}", wall, symbols.len());
        for symbol in symbols {
            let why = of(symbol).map(|(_, why)| why).unwrap_or("");
            println!("      {symbol:<48} {why}");
        }
    }

    // The service rows again, by WHO runs them rather than by which wall
    // stopped them — the join `execution::SERVED` exists to make possible,
    // and the answer to "what would it take to close this?" for each.
    println!("\nThe services ({services}), by who executes them:\n");
    let mut by_service: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for (symbol, service, _) in execution::SERVED {
        by_service.entry(format!("{} ({service:?})", service.label())).or_default().push(symbol);
    }
    for (service, symbols) in &by_service {
        println!("  {:<40} {:>2}   {}", service, symbols.len(), symbols.join(", "));
    }

    // The ops, one line each, saying what step two could and could not say
    // about it. A `Step` names a SYMBOL, never an execution — so the steps
    // print as symbols and the reader cannot tell from them which are JIT and
    // which are cuBLAS, which is the design rather than an omission.
    println!("\nThe ops ({ops}), and what `Step` can say about each:\n");
    for symbol in candidates().filter(|s| kind_of(s) == Kind::Op) {
        match execution::composition(symbol) {
            Some(composition) => {
                let mark = if composition.fireable() { "FIRES" } else { "stated" };
                let steps: Vec<&str> = composition.steps.iter().map(|s| s.symbol()).collect();
                println!("  {mark:<6} {symbol:<46} {}", steps.join(" -> "));
            }
            None => {
                let why = of(symbol).map(|(_, why)| why).unwrap_or("");
                println!("  {:<6} {symbol:<46} {why}", "--");
            }
        }
    }

    if !text.is_empty() {
        println!("\nC. Waiting on text ({c}):\n");
        for (symbol, why) in &text {
            println!("  {symbol:<48} {why}");
        }
    }

    // D last and loudest: it is the class most likely to be under-counted,
    // and every entry is a piece of work somebody has already stopped doing.
    if !stale.is_empty() {
        println!(
            "\nD. Refused for a reason that is now stale ({d}) -- NOT fixed here;\n   \
             `families/*.rs` belongs to another agent. Re-derive each from its\n   \
             launcher before believing either the refusal or this line:\n"
        );
        for (symbol, why) in &stale {
            println!("  {symbol}\n      {why}");
        }
    }

    // The classification going stale is the good failure, so it is REPORTED
    // rather than asserted: a class-D row that migrated is the point. It has
    // fired once already — see `Class::Stale` for the two `quant` rows it
    // caught — which is why it is here and not an `assert!`.
    let overtaken: Vec<&str> = CLASSIFIED
        .iter()
        .filter(|(symbol, _, _)| !refused.contains(symbol))
        .map(|(symbol, _, _)| *symbol)
        .collect();
    if overtaken.is_empty() {
        println!("\nNo classified row has migrated since the table was written.");
    } else {
        println!(
            "\n{} classified row(s) have MIGRATED since this table was written --\n\
             the table is stale in the direction that is good, and each entry below\n\
             should be deleted from `CLASSIFIED` by whoever moved it:",
            overtaken.len()
        );
        for symbol in overtaken {
            let class = of(symbol).map(|(c, _)| format!("{c:?}")).unwrap_or_default();
            println!("  {symbol:<48} was {class}");
        }
    }
    println!();
}

/// Percent, with an empty table reading as zero rather than dividing by it.
fn percent(part: usize, whole: usize) -> f64 {
    if whole == 0 { 0.0 } else { 100.0 * part as f64 / whole as f64 }
}

/// What kind of wall a [`Class::Structural`] row is behind.
///
/// Named rather than lumped, because the six close differently — or, in three
/// of the six cases, provably never close — and a count of 36 that cannot be
/// broken down is a count nobody can argue with.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Wall {
    /// **There is no device text in this tree.** The launcher calls cuBLAS,
    /// cuBLASLt, CUTLASS or NCCL, and the kernel that runs is the library's.
    /// Extracting a `.cuh` is not blocked; there is nothing to extract.
    /// Closes only by the library being replaced.
    ///
    /// **This is the wall that means [`Kind::Service`]**, so an entry here is
    /// also an entry in [`kernels_cuda_new::execution::SERVED`], with the
    /// service that runs it — and [`assert_total`] derives the two sets by
    /// different routes and asserts they are equal.
    ///
    /// # This wall has now been over-claimed TEN times, and the corrections
    /// are the whole reason the classification exists
    ///
    /// It used to end *"…or Marlin"*, and both Marlin rows sat here. That was
    /// read off the call site rather than measured, and it was **false**:
    /// `csrc/third_party/marlin` holds 2,082 lines of device text that NVRTC
    /// 13.0 compiles to a **55,024-byte sm_89 cubin**, reaching seven external
    /// includes of which **zero** are unanswered — three by the shims, four by
    /// `csrc/vendor` — and needing exactly two intrinsics (`__hadd2`,
    /// `atomicAdd(__nv_bfloat162*)`) and two namespace aliases.
    /// `examples/marlin_probe.rs` is the instrument.
    ///
    /// Both rows are still refused, and both for real reasons — but for
    /// [`Wall::TwoLaunches`] and [`Wall::HostChoice`], not for this one. The
    /// distinction is not pedantry: *"there is nothing to extract"* tells the
    /// next reader to stop, and there was something to extract.
    ///
    /// **Six more left this wall the same way**, when the `Service`
    /// classification made "nothing to extract" a claim somebody had to
    /// check row by row rather than family by family. `gemm/gemm.cpp` holds
    /// 0 `__global__` in 2,470 lines — and CALLS kernels of ours from ten
    /// places. `act_x_wt_bf16` tries our `gemv.cu` before cuBLAS;
    /// `act_x_wt_bias_bf16` runs `norm::add_bias_bf16`, which THIS CRATE
    /// ALREADY FIRES, straight after the GEMM; `act_x_wt_channel_scaled` is a
    /// quantize, a GEMM, a dequant and sometimes a residual-add, three of the
    /// four ours. Every one of those was on record as "cuBLASLt", and the
    /// three that said "cuBLASLt" about a `cublasGemm*Ex` call were wrong
    /// about the library too.
    ///
    /// The lesson generalises to every row below: **a wall named from the
    /// call site is a guess.** `flashinfer::*` in a `.cu` says which library
    /// is called; it does not say whether that library's source is in this
    /// tree — and for FlashInfer's cascade merge it demonstrably is
    /// (`csrc/vendor/flashinfer/attention/cascade.cuh` holds seven
    /// `__global__`s including `MergeStatesKernel`).
    Library,
    /// **The launcher returns `bool` and declines.** `K % 8 != 0`, or a
    /// pointer not 16-byte aligned, and it returns `false` meaning *"I did
    /// not launch"*. A row cannot decline: dispatching one through the JIT
    /// launches the kernel the C++ refused, over the buffer it refused for.
    /// Compounded here by template arguments taken from a device query and
    /// `getenv`, which a name expression is fixed before it can see.
    Declines,
    /// **The symbol names no C++ function.** A pseudo-symbol: an operation of
    /// the declared executor — a `cudaMemcpyAsync` pair, a staged LoRA apply
    /// built out of GEMM calls. `driver-cuda`'s
    /// `the_unstated_rows_are_exactly_the_ones_with_a_written_reason` calls
    /// this `Unstated::NotACppFunction` and says of it: *"never closes"*.
    ///
    /// The second wall that means [`Kind::Service`] — [`Service::DriverOp`],
    /// where the library is the driver.
    NotAKernel,
    /// **A `switch` over a scheme, whose cases reach different kernels.**
    /// `write_kv_to_pages` reads `layer.scheme` and lands in one of four
    /// `__global__`s with four different page formats — and throws on one
    /// combination before it gets there. A row names one kernel.
    ///
    /// One of the two walls that mean [`Kind::Op`]: the kernels underneath
    /// are ours and several are migrated, so what a row cannot say here is
    /// the CHOICE, not the kernel.
    SchemeSwitch,
    /// **A host `if` whose arms reach different kernels**, on a fact that is
    /// not the fire's shape: an environment variable, a shared-memory budget,
    /// a `constexpr` in a file the row cannot see.
    ///
    /// This is the wall [`Class::Stale`]'s hardest member is measured
    /// against. `crate::device::Specialisation` DOES let one row choose an
    /// arm at fire time — but `agrees()` refuses an arm whose `LaunchRule`
    /// differs from the base's, *"a specialisation chooses an instantiation,
    /// not a geometry"*. So arms that share a geometry are vocabulary and
    /// arms that do not are a wall, and each row below was decided on that
    /// line rather than on how the C++ spells the branch.
    HostChoice,
    /// **One symbol, more than one launch.** A row is one `<<<>>>`. The
    /// fallback arm of `rmsnorm_bf16_with_fp16` is a norm and then a cast;
    /// `compact_page_csr` is a count and then a scan-and-scatter;
    /// `chunk_gated_delta_prefill` is a host `for` issuing T of them;
    /// `act_x_wt_channel_scaled` is a quantize, a library GEMM, a dequant and
    /// sometimes a residual-add.
    ///
    /// The other wall that means [`Kind::Op`], and the one that makes the
    /// distinction worth having: a `TwoLaunches` row is not blocked on device
    /// text — `norm::add_bias_bf16` and `norm::residual_add_bf16` are rows
    /// this crate hosts and fires TODAY. It is blocked on a vocabulary for
    /// SEQUENCE, which is step two.
    TwoLaunches,
}

/// What a symbol IS, per its wall — a kernel, an op, or a service.
///
/// The mapping is a `match` on [`Wall`] rather than a second column of
/// [`CLASSIFIED`], and that is load-bearing twice over: a wall added tomorrow
/// is a **compile error here**, so nobody can add one without saying which
/// kind it makes a symbol; and the six rows that left [`Wall::Library`] this
/// session changed kind for free, because the kind is derived from the
/// measured wall rather than typed in beside it.
///
/// * [`Wall::Library`], [`Wall::NotAKernel`] → **service**. A library the
///   driver links, or the driver itself. Never a kernel, so never work.
/// * [`Wall::SchemeSwitch`], [`Wall::TwoLaunches`] → **op**. A host program
///   over kernels of ours, most of which migrate already.
/// * [`Wall::HostChoice`], [`Wall::Declines`] → **kernel**. One launch of one
///   kernel; what is unstatable is WHICH INSTANTIATION, from a device query,
///   a `getenv`, a shared-memory budget or an alignment the row cannot see.
///   These are unmigrated kernels behind a real wall, and they stay in the
///   kernel denominator on purpose: pretending a device query is a
///   composition is how a wrong reason gets written down.
///
/// The last arm is the one to argue with, and the argument is bounded: some
/// of those ten (`merge_attention_states_bf16`'s two-geometry host `if`,
/// `nemotron_mamba_split_bf16`'s `gate == nullptr`) may well be CHOICES that
/// step two's vocabulary reaches. Deciding that requires re-deriving each
/// launcher against a `Choose` that does not exist yet, so it is named here
/// and left, which is the honest half of a classification.
const fn kind_of_wall(wall: Wall) -> Kind {
    match wall {
        Wall::Library | Wall::NotAKernel => Kind::Service,
        Wall::SchemeSwitch | Wall::TwoLaunches => Kind::Op,
        Wall::HostChoice | Wall::Declines => Kind::Kernel,
    }
}

/// What a symbol is, over EVERY candidate row and not merely the refused ones.
///
/// A migrated row is a kernel because a unit hosts it, which is
/// [`kernels_cuda_new::execution::Execution::Jit`] by definition. A row
/// refused into class B, C or D is a kernel too — those are rows waiting on
/// vocabulary or on text, which is work, and work on a kernel.
fn kind_of(symbol: &str) -> Kind {
    match CLASSIFIED.iter().find(|(s, _, _)| *s == symbol) {
        Some((_, Class::Structural(wall), _)) => kind_of_wall(*wall),
        _ => Kind::Kernel,
    }
}

/// Which of the four classes a refused row is in — or rather three of them,
/// because **B is the residue** and is deliberately not spellable here.
///
/// A table of B rows would be a table that goes stale silently: `vocab-last`
/// is adding rules and types right now, and every row it moves would leave a
/// dead entry behind. A, C and D are claims about the tree that only a HUMAN
/// re-reading a launcher can retire, so those are the ones written down, and
/// "everything else" is what B means.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Class {
    /// **A.** Structurally unmigrable. Not work — a floor.
    Structural(Wall),
    /// **C.** Waiting on text: the `__global__` is still inside a `.cu` that
    /// has not been split into a `.cuh`, so there is nothing for a unit to
    /// compile. Nine `__global__`s remain in `kernels-cuda/csrc/src` outside
    /// `third_party` — four in `attn/attention_flashinfer.cu`, one in
    /// `attn/attention_xqa.cu`, one in `comm/custom_all_reduce.cu` and three
    /// in `gemm/gemv.cu` — and only the ones a TABLE ROW names are here. The
    /// other eight are internal helpers of a library dispatch, or `gemv.cu`'s
    /// three, which are [`Wall::Declines`] and whose file is left whole on
    /// purpose.
    Text,
    /// **D.** Refused for a reason that is now stale.
    ///
    /// The vocabulary grew a great deal this session — `LaunchRule` 21 → 36,
    /// `Fact::Bool`, `Term::Is`, `Term::Present`, `DeviceKernel::PLAIN`,
    /// `Ty::{Bf16s,F16s,I8sMut}`, `Dims::{stated_head_dim,requests,
    /// altup_streams}` — and a refusal written before a rule landed is a
    /// refusal about a vocabulary that no longer exists. These are NOT fixed
    /// here: `families/*.rs` belongs to another agent. They are named, with
    /// the rule that overturns them, so that the next reader re-derives the
    /// refusal from the launcher instead of believing the report.
    ///
    /// # The class is real, and two entries proved it inside an hour
    ///
    /// This table was written with NINE entries. Two of them —
    /// `quant::mxfp4_moe_gate_up_decode_bf16` and
    /// `quant::mxfp4_moe_down_decode_bf16`, both called stale on the grounds
    /// that `LaunchRule::RoutedQmvQuad` cites `quant/dequant_fp4.cu:67-70`
    /// and `:152-156` by line — were rowed by the agent that owns
    /// `families/quant.rs` while this report was being written, and the
    /// `overtaken` block below is what noticed. They are gone from the table
    /// because the table describes REFUSED rows; the count they left behind
    /// is the evidence that a refusal on record is not a refusal that holds.
    Stale,
}

/// The refused rows that are **not** simply waiting on vocabulary, one entry
/// each, with the evidence that decides it.
///
/// Citations are into `crates/kernels-cuda/csrc/**` — the archive — and not
/// into `families/*.rs`, on purpose: the family modules are being edited, the
/// C++ is not, and a reason that cites the launcher can be checked by anyone
/// at any commit.
#[rustfmt::skip]
static CLASSIFIED: &[(&str, Class, &str)] = &[
    // ── A: no device text in this tree ───────────────────────────────────
    //
    // `gemm/gemm.cpp` is host C++ compiled by `g++`: 0 `__global__` in 2,470
    // lines. That fact was read as "so every `gemm::` row is a library call",
    // and it is NOT what it means -- the file CALLS kernels of ours from ten
    // places (`:544`, `:962`, `:2356` reach `gemm/gemv.cu`'s warp-per-row
    // GEMV; `:1814`, `:1855`, `:1912`, `:2085`, `:2122`, `:2263` reach
    // `quant::`; `:2130`, `:2224`, `:2393` reach `norm::`). SIX rows below
    // therefore left `Library` this session, measured one body at a time, and
    // are now `HostChoice` or `TwoLaunches`. What survives here is the set
    // whose bodies reach NONE of those: see `execution::SERVED`, which is the
    // same set with the service that runs it.
    ("gemm::act_x_wt_bf16",            Class::Structural(Wall::HostChoice), "MEASURED, not read: NOT a library call. `gemm.cpp:958-963` is `if (M == 1 && beta == 0 && gemv_bf16(...)) return;` -- OUR warp-per-row GEMV out of `gemm/gemv.cu`, which holds three `__global__`s -- and the tuner one branch up can pick `GemmKind::Gemv` at any M (`:528-545`). Two arms, one ours and one cuBLASLt's, chosen on a tuning table and an alignment"),
    ("gemm::act_x_wt_bias_bf16",       Class::Structural(Wall::TwoLaunches), "MEASURED: `gemm.cpp:2391-2394` is a GEMM and then `kernels::norm::add_bias_bf16` -- a kernel THIS CRATE ALREADY HOSTS AND FIRES. The M=1 arm folds the bias into the gemv epilogue instead, so the row is one launch or two depending on a tuner"),
    ("gemm::act_x_wt_bf16_out_fp32",   Class::Structural(Wall::Library), "one `cublasGemmEx`, bf16 in / fp32 out; `gemm.cpp:1030-1058` is the whole body"),
    ("gemm::batched_act_x_wt_bf16",    Class::Structural(Wall::Library), "`cublasGemmGroupedBatchedEx` with a `cublasGemmBatchedEx` fallback; both arms the library's"),
    ("gemm::grouped_act_x_wt_bf16",    Class::Structural(Wall::Library), "one `cublasGemmGroupedBatchedEx`; `gemm.cpp:1242-1294`. CLASSIC cuBLAS -- the \"cuBLASLt grouped\" on record was wrong"),
    ("gemm::act_x_wt_tensor_scaled",   Class::Structural(Wall::HostChoice), "MEASURED: `gemm.cpp:1982-1989` picks on `ctx.fp8_native_supported`, a LATCHED device probe, between one `cublasLtMatmul` and `gemm_fp8_dequant_then_bf16_fallback`, which launches our `quant::dequant_fp8_e4m3_to_bf16` (`:1832`) and then a bf16 GEMM"),
    ("gemm::act_x_wt_channel_scaled",  Class::Structural(Wall::TwoLaunches), "MEASURED: `gemm.cpp:2085-2133` is `quant::quantize_bf16_to_int8_per_token`, then an INT8 `cublasGemmEx`, then `quant::dequant_int32_w8a8_to_bf16`, then `norm::residual_add_bf16` when `beta != 0`. THREE OR FOUR LAUNCHES, three of them ours"),
    ("gemm::act_x_wt_grouped_scaled",  Class::Structural(Wall::TwoLaunches), "MEASURED: `gemm.cpp:1912-1953` is `quant::quantize_bf16_to_fp8_e4m3_per_token_group` and then a block-scaled `cublasLtMatmul`; when the Lt heuristic returns nothing it latches off and the dequant fallback (`:1814`) is two more"),
    ("gemm::act_x_wt_mxfp4_marlin",    Class::Structural(Wall::TwoLaunches), "MEASURED, not read: Marlin's device text IS here and NVRTC 13.0 compiles it to a 55,024 B sm_89 cubin (`examples/marlin_probe.rs`). Two walls, neither `Library`: `marlin.cu:443` loops `while (rest_m)` issuing MORE THAN ONE launch, and the selector picks 1 of 15 instantiations from a shared-memory budget and three device queries"),
    ("gemm::mla_absorb_q_to_latent_bf16", Class::Structural(Wall::Library), "one `cublasGemmStridedBatchedEx` over the head axis; `gemm.cpp:2419-2442` is the whole body"),
    ("gemm::mla_absorb_latent_to_v_bf16", Class::Structural(Wall::Library), "the second absorb, same single call; `gemm.cpp:2444-2468`"),
    ("marlin_moe::launch_mxfp4_moe_gemm_w4a16_bf16", Class::Structural(Wall::HostChoice), "MEASURED: the text is here and compiles under NVRTC. One launch, so not `TwoLaunches` — but the selector walks a tuning table against `MaxSharedMemoryPerBlockOptin`, `ComputeCapabilityMajor/Minor` and `MultiProcessorCount` to pick 1 of 15, and a row names one instantiation. Not built unless `PIE_CUDA_BUILD_MARLIN_MOE`"),
    ("moe::flashinfer_cutlass_moe_bf16", Class::Structural(Wall::Library), "CUTLASS grouped-GEMM MoE pipeline, returns `bool`. THE ONE THAT SURVIVES: `csrc/third_party/flashinfer_moe/*.cu` holds 0 `__global__`, `src/moe/flashinfer_moe.cu` holds 0 and calls no kernel of ours, and `cutlass/` is in no SOURCE directory of this repo -- CPM fetches it into `target/**/_deps/flashinfer-src/3rdparty/cutlass` at configure time. The kernels are templates in headers we do not have"),
    ("ssm::flashinfer_mamba_ssu_bf16",   Class::Structural(Wall::Declines), "MEASURED (`examples/vendor_probe.rs`): 5 files / 1,665 lines vendor with 7 guards + 2 shims, NVRTC COMPILES them to a 396,128 B cubin and 9 of 9 symbols resolve. Not `Library`. The wall is a device query: `CTAS_PER_HEAD` is a TEMPLATE argument from `clamp(GetCudaMultiProcessorCount()*10/(batch*nheads), 1, DIM/64)`, and `flashinfer_mamba_ssu_enabled()` needs sm90 -- on this L40S the row never fires"),
    ("attn::merge_attention_states_bf16", Class::Structural(Wall::HostChoice), "MEASURED: ZERO files to move -- `csrc/vendor/flashinfer/attention/cascade.cuh` is already here, byte-for-byte upstream, and NVRTC compiles it to 96,176 B with 8 of 8 symbols resolving. The wall is `cascade.cuh:638-666`, a host `if` over TWO kernels with two geometries: `MergeStatesLargeNumIndexSetsKernel` is grid `(seq_len, num_heads)` with dynamic smem, `MergeStatesKernel` is grid `(seq_len)` with none"),
    // NCCL, and not even in this crate: `csrc/src/dist/` does not exist.
    // These are methods on the DRIVER's `NcclComm`, which is what
    // `launch_abi::the_unstated_rows...` calls `SecondNamespaceRoot`.
    // `kernels-cuda` neither includes `nccl.h` nor links NCCL.
    ("dist::all_reduce_bf16",     Class::Structural(Wall::Library), "NCCL; no `csrc/src/dist/`, a method on the driver's `NcclComm`"),
    ("dist::all_reduce_bf16_out", Class::Structural(Wall::Library), "NCCL, out-of-place"),
    ("dist::all_gather_bf16",     Class::Structural(Wall::Library), "NCCL all-gather"),
    // `comm/` DOES exist and holds one `__global__` — but it backs
    // `all_reduce_residual_rmsnorm_bf16_exact`, which has NO TABLE ROW. Both
    // rowed entry points take a `CustomAllReduce*` the driver owns and
    // forward into headers this repo does not carry: `csrc/vendor/flashinfer`
    // holds `attention/` only, and there is no in-repo copy of
    // `flashinfer/comm/vllm_custom_all_reduce.cuh`.
    ("comm::all_reduce_bf16", Class::Structural(Wall::Library), "`car->all_reduce_bf16` -> `impl_->allreduce<__nv_bfloat16>`, vLLM's NVLink kernel; `custom_all_reduce.cu:641-658`. A null `car` is a REFUSAL rather than a fallback"),
    ("comm::all_reduce_residual_rmsnorm_bf16", Class::Structural(Wall::Library), "`flashinfer::trtllm_allreduce_fusion`'s `kARResidualRMSNorm`; `custom_all_reduce.cu:661-700`. The one `__global__` in that file is the `_exact` twin, which has no row"),

    // ── A: the launcher declines ─────────────────────────────────────────
    ("gemm::gemv3_bf16", Class::Structural(Wall::Declines), "`gemv.cu` returns `false` on `K % 8` or misalignment; `gemv_unroll_depth()` reads `cudaDevAttrComputeCapabilityMajor` and `PIE_GEMV_B200_TUNING` to pick between exactly TWO arms at different BLOCK shapes"),

    // ── A: not a C++ function ────────────────────────────────────────────
    ("qwen35_verify_stash_store", Class::Structural(Wall::NotAKernel), "a `cudaMemcpyAsync` trio the executor performs; `Unstated::NotACppFunction`, and `KernelSig::operands` is empty"),
    ("qwen35_verify_stash_load",  Class::Structural(Wall::NotAKernel), "the load half of the same trio"),
    ("pie_lora_qkv_correction",   Class::Structural(Wall::NotAKernel), "the driver's own arm: `bind/mod.rs:1895` calls `(*state).apply(ctx.cublas, ...)`, built out of grouped GEMM calls it already had"),

    // ── A: a switch over the cache scheme ────────────────────────────────
    ("attn::write_kv_to_pages", Class::Structural(Wall::SchemeSwitch), "`switch (layer.scheme)` over four page formats, and throws on `first_token != 0` off native bf16; `attn/kv_paged.cu:107-160`"),
    ("attn::dequant_kv_cache_layer_to_bf16_active", Class::Structural(Wall::SchemeSwitch), "one launcher over all four schemes, four dequant kernels"),

    // ── A: a host `if` whose arms are different kernels ──────────────────
    ("attn::attention_mtp_paged_history_bf16", Class::Structural(Wall::HostChoice), "falls back to `attn_mtp_history`, a DIFFERENT symbol, when `max_global_tokens + history_steps > 8192` -- a hard-coded literal and NOT a shared-memory query; `attn/attention_naive.cu:80-135`"),
    ("ssm::nemotron_mamba_split_bf16", Class::Structural(Wall::HostChoice), "`gate == nullptr` reaches `mamba_split_conv_dt`, a different kernel; `runtime::launch`'s `ElementwiseIn` doc says so itself"),
    ("ssm::nemotron_mamba_ssm_batched_bf16", Class::Structural(Wall::HostChoice), "TWO live kernel forms chosen on `sequence_prefill`, a `bool` OPERAND -- the middle arm is `if constexpr (false)`, dead. `Term::Is` already expresses the predicate; what is missing is a row for either arm"),
    ("ssm::recurrent_gated_delta_step_batched_gqa_state_bf16", Class::Structural(Wall::HostChoice), "`getenv(\"PIE_QWEN35_GDN_SMEM_STEP\")` picks a DIFFERENT KERNEL, not a different grid"),

    // ── A: one symbol, more than one launch ──────────────────────────────
    ("attn::compact_page_csr", Class::Structural(Wall::TwoLaunches), "`count_kept` then `scan_and_scatter`; a row for either states half a contract"),
    ("norm::rmsnorm_bf16_with_fp16", Class::Structural(Wall::TwoLaunches), "unaligned rows fall back to `rmsnorm_bf16` PLUS `quant::bf16_to_fp16`; `norm/rmsnorm.cu:50-53` says so"),

    // ── C: the text is not split yet ─────────────────────────────────────
    // `attn::attn_score_fold_heads` WAS HERE, and its departure is the
    // milestone: **class C reached zero, and kernel text migration is
    // complete.** Every `__global__` this project migrates now lives in
    // `kernels-cuda-new/csrc`.
    //
    // It is still refused, and it is class B now — the residue — because the
    // blocker moved rather than closed. `attention_flashinfer.cu:828-829`
    // launches `dim3(requests, 64u)` at 256 threads with nothing shared, and
    // `LaunchRule::PerRequest` is ONE NUMBER AWAY: same `grid.x`, same block,
    // `grid.y` of 1 against the launcher's 64.
    //
    // That near miss is the reason it is a refusal rather than a row, and the
    // reason deserves stating: the body strides `i += blockDim.x * gridDim.y`,
    // so the wrong rule computes **the same floats in 64x fewer blocks**. It
    // is wrong only in LATENCY — no parity test here could fail on it, and
    // §22.7's near miss (0 of 20,480 bytes differing behind a bounds guard)
    // is the same shape. The rule it needs is `PerRequest` with a fixed
    // y-fanout: `dim3(requests, 64)` at 256, nothing shared.
    //
    // Kept as a comment rather than deleted because the count moving from 1
    // to 0 is exactly the kind of change a reader will want a reason for.

    // ── D: the stated refusal is stale ───────────────────────────────────
    //
    // These are the highest-value entries in the file and the ones most
    // likely to be wrong in the other direction — a rule whose grid matches
    // to the digit and whose OPERANDS do not. Each names the rule and the
    // launcher, so overturning an entry is a re-read rather than an argument.
    //
    // FOUR OF THE SEVEN WERE RE-DERIVED AND LANDED, and they are gone from
    // this list rather than marked: `rope::qk_rmsnorm_rope_bf16` and
    // `rope::qk_rmsnorm_mrope_bf16` on `RowsPackedHeadsNarrow`,
    // `attn::attention_compressed_paged_bf16` on `PagedScoresDecode`, and
    // `attn::write_kv_explicit_bf16_devwin` on `PerRow`. Each is pinned in
    // `tests/launch_rules.rs::mod transcribed` and fired byte-identical
    // against a raw `cuLaunchKernel` at three shapes.
    //
    // The three below are the ones that survived re-derivation. Two of them
    // are still class D — the CLASS is right, the row IS refused for a reason
    // that is not the one that was on record — and the entries have been
    // rewritten to the true reason. That is the outcome this class exists to
    // produce as much as a landing is: a stale refusal and a stale
    // un-refusal are the same kind of error.
    ("rope::qk_rmsnorm_rope_bf16_rounded", Class::Stale, "NOT `rope/rope.cu:214-215`, which is `dim3(num_tokens, num_q_heads)` — Q ONLY. The real call site (`model/src/gemma_4/forward/mod.rs:217`) reaches it with `k = nullptr` and `num_kv_heads = 0`, while `Dims::kv_heads` is the FIRE's head count (`bind/mod.rs:1379`) and is non-zero there. `RowsPackedHeadsNarrow` would open `q + kv` blocks and the excess would rotate a null `k`; `packed_heads`' zero-refusal cannot fire, because the zero is in the OPERAND and not in the `Dims`. NEEDS a separate symbol for the q-only form (as `_devwin` has) or a rule whose head axis reads the operand — reproducing `rope/rope.cu:214`"),
    ("attn::qkv_decode_qk_norm_rope_write_kv_bf16", Class::Stale, "the reason on record was the vocabulary and the vocabulary caught up: `WarpPackedHeads` and `RowsPackedHeadsNarrow` were BOTH ported from this launcher (`attn/qkv_fused.cu:51-53`, `:98-102`) and `Term::Present` reads `rope_table != nullptr`. What refuses it NOW is that a `Specialisation` may not change a `LaunchRule` — the launcher picks between those two GEOMETRIES on `head_dim` — and this audit KEPT that invariant: `fire.rs:176-186` evaluates the grid from the base row before consulting the specialisation, four readers take `KernelSig::launch` as a contract, and lifting it would not land the row anyway because `head_dim == 64 | 128 | 256` is still unspellable (`Term::Multiple { of: 64 }` holds of 192). `tests/specialise.rs::agrees_refuses_an_arm_that_changes_the_launch_rule`"),
    ("ssm::build_nemotron_moe_ptrs_decode_batched_bf16", Class::Stale, "refused as an extent no shape produces; `ElementwiseIn` is `rows * in_width`, which is `rows * top_k` to the digit, and `Ty::BufArray*` now names its `void**` operands. NOT RE-DERIVED: `families/ssm.rs` is outside this session's grant"),
];

/// Every claim this report makes about its own arithmetic, checked.
///
/// # Why an example asserts
///
/// Five gates this session passed while measuring nothing (`new-horizon.md`
/// §21.9, §22.6). The shape is always the same: something filters, the filter
/// is right, and nothing anywhere states how many rows it was supposed to
/// remove — so the day it removes the wrong ones, or none, it still passes.
/// **A gate that filters must assert its own denominator.** This one filters
/// twice, so it asserts twice: `Prepare` rows out of `rows`, and
/// [`Class::Structural`] rows out of `reachable`.
///
/// The failure mode it exists against is a classification that silently drops
/// a row — an entry whose symbol was renamed, a class that overlaps another,
/// a residue computed by subtraction that hides a miscount. Every one of
/// those is a panic here.
///
/// # The residue trap
///
/// B is a residue, and a residue makes its own total-ness assertion
/// vacuous: define `b = refused - a - c - d` and `a + b + c + d == refused`
/// holds for every possible classification, including one that puts all 78
/// rows in the wrong class. So the partition below is derived twice by
/// different routes — folded over the refused set, and filtered over the
/// classification table — and the two are asserted equal. That comparison is
/// the assertion; the sum is the receipt.
///
/// # The kind partition, and how to make it fail
///
/// The same trap applies to kernel/op/service, and harder: `kernels` is the
/// residue there. So the SERVICE set is derived twice by genuinely
/// independent routes — from [`kind_of_wall`] over the walls in this file,
/// and from `execution::SERVED` in the crate — and asserted equal as SETS
/// rather than as counts. Deleting a row from `SERVED` fails it; moving a row
/// to `Wall::Library` without serving it fails it; and a service row that
/// some unit HOSTS fails a third assertion, because a symbol cannot be both
/// JIT-compiled and library-served.
///
/// It has already failed once for real: `moe::flashinfer_cutlass_moe_bf16`
/// and the five surviving `gemm::` rows were the residue of a set that
/// started at seventeen, and the six that left it did so because this
/// assertion made "no `__global__` in the closure" a claim somebody had to
/// check per row.
///
/// # Panics
///
/// Naming the row and the arithmetic that does not close.
fn assert_total(refused: &BTreeSet<&str>, rows: usize, migrated: usize, library: usize) {
    // 1. The classification names rows that EXIST and are candidates. A
    //    symbol that no table row states — a typo, or a row deleted under us
    //    — would otherwise sit here forever contributing nothing.
    let stated: BTreeSet<&str> = candidates().collect();
    let mut seen: BTreeSet<&str> = BTreeSet::new();
    for (symbol, class, _) in CLASSIFIED {
        assert!(
            stated.contains(symbol),
            "`{symbol}` is classified {class:?} and is not a non-`Prepare` row of \
             `table::KERNELS` — the classification names a row that does not exist"
        );
        assert!(seen.insert(symbol), "`{symbol}` is classified twice");
    }

    // 2. Both denominators, stated rather than assumed. `rows + library` is
    //    every row there is; `reachable + structural` is every candidate.
    assert_eq!(
        rows + library,
        table::KERNELS.len(),
        "the `Prepare` filter dropped rows into neither denominator"
    );

    // 3. The partition is TOTAL, and this is checked by deriving it TWICE by
    //    different routes and making them agree. Deriving it once is worth
    //    nothing: B is the residue, so `b = refused - a - c - d` makes
    //    `a + b + c + d == refused` true by arithmetic no matter how wrong
    //    the classification is. That assert is one of the five this session
    //    found (§21.9, §22.6) wearing a different hat.
    //
    //    Route one: fold every refused symbol to the class it lands in,
    //    B being "refused and unclassified", and count the folds. This is
    //    total by construction over the refused set — a symbol contributes
    //    exactly one increment because the `match` has one arm per symbol.
    let mut by_iteration = [0usize; 4];
    for symbol in refused {
        let idx = match CLASSIFIED.iter().find(|(s, _, _)| s == symbol) {
            Some((_, Class::Structural(_), _)) => 0,
            Some((_, Class::Text, _)) => 2,
            Some((_, Class::Stale, _)) => 3,
            None => 1,
        };
        by_iteration[idx] += 1;
    }
    let [a, b, c, d] = by_iteration;

    //    Route two: filter the CLASSIFIED table per class. This one CAN
    //    disagree — an entry classified twice under different classes, a
    //    `Class` variant added that two predicates both match, a symbol in
    //    the table that is not in `refused` — and each disagreement is a row
    //    that route one silently placed somewhere route two did not.
    let counted = |want: fn(&Class) -> bool| {
        CLASSIFIED
            .iter()
            .filter(|(symbol, class, _)| refused.contains(symbol) && want(class))
            .count()
    };
    assert_eq!(a, counted(|c| matches!(c, Class::Structural(_))), "class A disagrees between the two derivations");
    assert_eq!(c, counted(|c| matches!(c, Class::Text)), "class C disagrees between the two derivations");
    assert_eq!(d, counted(|c| matches!(c, Class::Stale)), "class D disagrees between the two derivations");
    assert_eq!(
        b,
        refused.len() - (a + c + d),
        "class B is not the residue it is documented to be — a refused row is \
         in no class and in the residue at once, or in two classes"
    );
    assert_eq!(a + b + c + d, refused.len(), "the four classes do not sum to the refused set");

    // 4. And the headline identity, which is the one a reader will do in
    //    their head and get wrong: `reachable` is not `refused` minus
    //    anything, it is `rows` minus the floor, and what is left to do in it
    //    is exactly B + C + D.
    let reachable = rows - a;
    assert_eq!(
        migrated + b + c + d,
        reachable,
        "migrated + (B + C + D) must be the reachable denominator, or a row is \
         in neither the numerator nor the work"
    );

    // 5. The kind partition is TOTAL over the candidates. Every stated symbol
    //    is a kernel, an op or a service, and the three sum to `rows`. This
    //    one is total by construction — `kind_of` has one arm per symbol —
    //    so the sum is a receipt and the assertions that follow are the
    //    substance.
    let of_kind = |want: Kind| candidates().filter(|s| kind_of(s) == want).count();
    let (kernels, ops, services) = (of_kind(Kind::Kernel), of_kind(Kind::Op), of_kind(Kind::Service));
    assert_eq!(
        kernels + ops + services,
        rows,
        "kernel + op + service must be every stated symbol — a row is in no kind or in two"
    );

    // 6. The SERVICE set, derived twice by independent routes and compared as
    //    a SET. Route one is this file's walls; route two is the crate's
    //    `execution::SERVED`, which is where the driver reads it. Counts
    //    would agree by accident; sets name the row that disagrees.
    let by_wall: BTreeSet<&str> = candidates().filter(|s| kind_of(s) == Kind::Service).collect();
    let by_table: BTreeSet<&str> =
        execution::SERVED.iter().map(|(symbol, _, _)| *symbol).filter(|s| stated.contains(s)).collect();
    assert_eq!(
        by_wall, by_table,
        "the service set disagrees between the walls here and `execution::SERVED` — \
         a `Wall::Library`/`Wall::NotAKernel` row that no service runs, or a served \
         row this file still calls work"
    );
    assert_eq!(
        by_table.len(),
        execution::SERVED.len(),
        "`execution::SERVED` names a symbol that is not a candidate row — a renamed \
         row, or a `Prepare` row that was never in this denominator"
    );

    // 7. A service is never a JIT row, and never migrated. This is the
    //    assertion that would fire if somebody "migrated" cuBLAS: a symbol
    //    cannot be both library-served and hosted by a unit, and if one ever
    //    is, the numerator and the service count are both wrong.
    for symbol in &by_table {
        assert!(
            unit::unit_of(symbol).is_none(),
            "`{symbol}` is served by a library AND hosted by a unit — one of the two \
             is a lie, and `Execution` cannot be both arms at once"
        );
        assert!(
            refused.contains(symbol),
            "`{symbol}` is a service and is not in the refused set — a service cannot \
             be migrated, so this is a counting error, not a landing"
        );
    }

    // 8. The kind derived from the WALL and the kind derived from the crate's
    //    `Execution` agree wherever both have an opinion. This is the
    //    assertion that keeps the next one from being algebra: `kind_of`
    //    reads `CLASSIFIED`, so "every op and every service is structural" is
    //    true BY CONSTRUCTION here, and an identity resting on it would hold
    //    for any classification whatsoever — the residue trap again, one
    //    level up. `execution::execution` has never heard of `CLASSIFIED`; it
    //    reads the unit tables and `SERVED`. A migrated row this file called
    //    an op, or a served row it called work, disagrees here.
    for symbol in candidates() {
        let Some(how) = execution::execution(symbol) else { continue };
        assert_eq!(
            how.kind(),
            kind_of(symbol),
            "`{symbol}`: `execution::execution` says {:?} and the wall here says {:?} \
             — the two routes to a symbol's kind disagree",
            how.kind(),
            kind_of(symbol)
        );
    }

    // 9. Every op and every service is behind a structural wall — stated,
    //    because it is the premise the identity below rests on and it is
    //    exactly the kind of thing that stays true until it quietly does not.
    for symbol in candidates().filter(|s| kind_of(s) != Kind::Kernel) {
        assert!(
            CLASSIFIED.iter().any(|(s, class, _)| *s == symbol && matches!(class, Class::Structural(_))),
            "`{symbol}` is not a kernel and is not in class A — an op or a service \
             outside the structural floor breaks both denominators at once"
        );
    }

    // 10. And the identity that makes the two denominators one claim: since
    //     every op and every service is structural (9), the reachable ROWS
    //     and the reachable KERNELS are the same set. Derived by subtraction
    //     here and by the fold above.
    let unmigrable_kernels = a - (ops + services);
    assert_eq!(
        kernels - unmigrable_kernels,
        reachable,
        "kernels minus the kernels behind a wall must be the reachable denominator — \
         an op or a service is outside class A, so one of the two partitions has a \
         row the other does not"
    );
    assert!(
        migrated <= kernels,
        "more symbols are migrated than there are kernels to migrate"
    );

    // 11. The OP set, derived twice, exactly as 6 does for services — and
    //     with one deliberate asymmetry. `execution::SERVED` must EQUAL the
    //     wall-derived service set, because a service that no wall names is a
    //     row this file calls work and the crate calls a library call.
    //     `execution::COMPOSED` is only a SUBSET of the wall-derived op set,
    //     because an op that states no composition yet is honest and an op
    //     that states one this file does not call an op is not. Asserting
    //     equality here would be asserting that step two finished, which it
    //     did not; asserting containment is asserting that nothing was
    //     invented, which is the property that has to hold every day.
    let ops_by_wall: BTreeSet<&str> = candidates().filter(|s| kind_of(s) == Kind::Op).collect();
    let ops_by_table: BTreeSet<&str> = execution::COMPOSED.iter().map(|c| c.symbol).collect();
    for symbol in &ops_by_table {
        assert!(
            ops_by_wall.contains(symbol),
            "`{symbol}` states a `Composition` and no wall here calls it an op — either the \
             crate composed something this file thinks is one launch, or a wall moved"
        );
    }
    assert_eq!(
        ops_by_table.len(),
        execution::COMPOSED.len(),
        "`execution::COMPOSED` states a symbol twice"
    );

    // 12. Every composition agrees with the rows it composes, and the graph
    //     is acyclic. `tests/layers.rs` asserts both already; they are
    //     repeated here because this example is what a reader runs, and a
    //     number printed from a table that does not typecheck is worse than
    //     no number. The acyclicity check in particular is new capability:
    //     a `Step` names a symbol and a symbol may be composed, so the table
    //     is now a graph and a graph can loop.
    for composition in execution::COMPOSED {
        composition
            .agrees()
            .unwrap_or_else(|why| panic!("`{}` does not agree with its rows: {why}", composition.symbol));
    }
    execution::acyclic(execution::COMPOSED)
        .unwrap_or_else(|why| panic!("the composition graph is not acyclic: {why}"));
}

/// Twenty characters of progress, because a column of numbers hides the shape.
fn bar(part: usize, whole: usize) -> String {
    const WIDTH: usize = 20;
    let filled = if whole == 0 { 0 } else { WIDTH * part / whole };
    format!("{}{}", "#".repeat(filled), ".".repeat(WIDTH - filled))
}
