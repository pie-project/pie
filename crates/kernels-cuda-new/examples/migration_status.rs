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
//! A row for `gemm::gemv3_bf16` named a launcher that reads
//! `cudaDevAttrComputeCapabilityMajor` and `getenv` to pick one of two
//! instantiations and RETURNS `bool` to mean *"I did not launch — use
//! cuBLAS"*; a row names one instantiation and a row cannot decline. (That
//! row is gone — nothing called its wrapper — but the shape of the argument
//! is what this paragraph is for, and `gemm::act_x_wt_bf16` still has it.)
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

/// The refused set: every candidate with no JIT twin.
///
/// Exposed because `tests/consumer.rs` mounts this file as a module and must
/// build the same denominator [`assert_consumers`] does, rather than a
/// convenient one of its own.
pub fn refused_set() -> BTreeSet<&'static str> {
    candidates().filter(|s| unit::unit_of(s).is_none()).collect()
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

    let of = |symbol: &str| CLASSIFIED.iter().find(|r| r.symbol == symbol);
    let mut structural: Vec<&Refusal> = Vec::new();
    let mut text: Vec<&Refusal> = Vec::new();
    let mut stale: Vec<&Refusal> = Vec::new();
    let mut vocabulary: Vec<&str> = Vec::new();
    for symbol in refused {
        match of(symbol) {
            Some(row) if matches!(row.class, Class::Structural(_)) => structural.push(row),
            Some(row) if matches!(row.class, Class::Text) => text.push(row),
            Some(row) if matches!(row.class, Class::Stale) => stale.push(row),
            Some(_) => unreachable!("`Class` has three variants and all three are matched"),
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
    let wall_of = |row: &Refusal| match row.class {
        Class::Structural(wall) => wall,
        _ => unreachable!("`structural` holds only `Class::Structural` rows"),
    };
    let unmigrable_kernels = structural.iter().filter(|r| kind_of_wall(wall_of(r)) == Kind::Kernel).count();

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
        structural.iter().filter(|r| kind_of_wall(wall_of(r)) == Kind::Service).count(),
        structural.iter().filter(|r| kind_of_wall(wall_of(r)) == Kind::Op).count(),
    );
    let mut by_wall: BTreeMap<String, Vec<&&Refusal>> = BTreeMap::new();
    for row in &structural {
        by_wall.entry(format!("{:?}", wall_of(row))).or_default().push(row);
    }
    for (wall, rows) in &by_wall {
        let kind = kind_of_wall(wall_of(rows[0]));
        println!("  {:<14} {:>2}   -> {kind:?}", wall, rows.len());
        for row in rows {
            println!("      {:<48} {}", row.symbol, row.why);
            println!("      {:<48} consumer: {:<11} {}", "", row.consumer.channel(), row.consumer.evidence());
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
                let why = of(symbol).map(|row| row.why).unwrap_or("");
                println!("  {:<6} {symbol:<46} {why}", "--");
            }
        }
    }

    report_consumers();

    if !text.is_empty() {
        println!("\nC. Waiting on text ({c}):\n");
        for row in &text {
            println!("  {:<48} {}", row.symbol, row.why);
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
        for row in &stale {
            println!("  {}\n      {}", row.symbol, row.why);
        }
    }

    // The classification going stale is the good failure, so it is REPORTED
    // rather than asserted: a class-D row that migrated is the point. It has
    // fired once already — see `Class::Stale` for the two `quant` rows it
    // caught — which is why it is here and not an `assert!`.
    let overtaken: Vec<&str> = CLASSIFIED
        .iter()
        .filter(|r| !refused.contains(r.symbol))
        .map(|r| r.symbol)
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
            let class = of(symbol).map(|row| format!("{:?}", row.class)).unwrap_or_default();
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
pub enum Wall {
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
    /// `examples/marlin_probe.rs` was the instrument; both it and the vendored
    /// tree it measured were deleted in §47, once the OTHER question -- not
    /// "can NVRTC compile this?" but "does anything CALL it?" -- was asked and
    /// answered no. The correction recorded here stands: the wall was never
    /// `Library`, and establishing that is what made the deletion legible.
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
    ///
    /// **No row is classified here today, and that is a measurement.** Both
    /// rows that were — `gemm::gemv3_bf16` and `ssm::flashinfer_mamba_ssu_bf16`
    /// — turned out to be reached by nothing: no model text called their
    /// `dsl::cuda` wrappers, so the wall stood in front of a door nobody
    /// opened. The variant stays because the argument is sound and the next
    /// declining launcher a text actually names belongs in it; what it is no
    /// longer is a floor under this migration's percentage.
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
    match CLASSIFIED.iter().find(|r| r.symbol == symbol) {
        Some(Refusal { class: Class::Structural(wall), .. }) => kind_of_wall(*wall),
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
pub enum Class {
    /// **A.** Structurally unmigrable. Not work — a floor.
    Structural(Wall),
    /// **C.** Waiting on text: the `__global__` is still inside a `.cu` that
    /// has not been split into a `.cuh`, so there is nothing for a unit to
    /// compile. Nine `__global__`s remain in `kernels-cuda/csrc/src` outside
    /// `third_party` — four in `attn/attention_flashinfer.cu`, one in
    /// `attn/attention_xqa.cu`, one in `comm/custom_all_reduce.cu` (**both
    /// that `__global__` and that file are now DELETED** — the `_exact` twin
    /// it backed had an empty caller set, and what was left was a host
    /// program, now `driver-cuda/src/fire/all_reduce.rs`) and three
    /// in `gemm/gemv.cu` — and only the ones a TABLE ROW names are here. The
    /// other nine are internal helpers of a library dispatch, or `gemv.cu`'s
    /// three, whose file is left whole on purpose: `gemv3_bf16`'s row was
    /// deleted (nothing called its wrapper) and `gemv_bf16` is called from
    /// `gemm.cpp`'s live `act_x_wt_bf16` path, so no row names any of them.
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
    /// that `LaunchRule::RoutedQmvQuad` cited `quant/dequant_fp4.cu:67-70`
    /// and `:152-156` by line (both since deleted with their launchers in
    /// §43) — were rowed by the agent that owns
    /// `families/quant.rs` while this report was being written, and the
    /// `overtaken` block below is what noticed. They are gone from the table
    /// because the table describes REFUSED rows; the count they left behind
    /// is the evidence that a refusal on record is not a refusal that holds.
    Stale,
}

/// A citation: a repo-relative `path:line`, and the token that must be there.
///
/// This is [`kernels_cuda_new::device::LaunchRule`]'s discipline — *"a rule
/// with no cited launcher is a guess"* — applied to the other side of the
/// question. [`resolve`] opens the file and looks; a citation that does not
/// resolve is a **test failure**, not a comment somebody will get around to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cite {
    /// `crates/model/src/glm_5/forward/mod.rs:127`. The line may be omitted;
    /// the file may not.
    pub at: &'static str,
    /// The token that must appear at `at` — a DSL method, a symbol, a fact,
    /// an entry point. Checking THIS is what makes the citation a fact
    /// rather than a decoration.
    pub names: &'static str,
}

/// **What reaches the symbol.** The field this whole file was missing.
///
/// # The defect this closes
///
/// Sixteen symbols were classified as hard migration problems — *"the
/// launcher returns `bool` and declines"*, *"a tuning table picks 1 of 15"*,
/// *"needs sm90"* — and then found to be reached by nothing at all. Every one
/// of those sentences is a true, checked statement about a **launcher**. None
/// of them is a statement about whether anything **calls** it. The DSL
/// surface was generated from the launcher headers (`6d02452de`, `c0e57c7f1`
/// — *"read the HEADERS to learn what a launcher IS"*), so a `dsl::cuda`
/// wrapper exists whether or not a model ever asked for one, and the wrapper
/// then reads as demand to any tool that stops at it. `new-horizon.md` §28
/// measured the population: **62 of 226 rows (27.4%) are reached by
/// nothing**, spread across eleven of twelve family files.
///
/// [`Wall`] is a good instrument pointed one hop short. This is the hop.
///
/// # Why an enum, and not free text with a citation
///
/// §21.14's test: *does the shape make a wrong claim well-formed?* Free text
/// admits `"reachable"`, which is worth nothing and cannot be checked; an
/// enum makes that string unspellable. The counter-argument is real — an
/// enum can be confidently WRONG in a way prose is not, because picking
/// `ModelText` for a test-only row is a well-formed lie — and it is answered
/// not by the enum but by the [`Cite`] every variant carries: the variant
/// says which CHANNEL, the citation says WHERE, and the citation is opened
/// and read by [`assert_consumers`]. Wrong file, wrong token, drifted line:
/// all three are failures that name the symbol.
///
/// Two channels are deliberately NOT variants, because admitting them would
/// make the gate an escape hatch:
///
/// * **A report is not a consumer.** `examples/vendor_probe.rs` names
///   `attn::merge_attention_states_bf16` twice and compiles its text to a
///   96,176-byte cubin. That is a finding about a launcher, not a demand for
///   a row, and every one of §31's four deleted rows had at least one report
///   naming it.
/// * **A C++-internal caller is not a consumer of the ROW.** It is
///   [`Consumer::Cpp`], which is a fact about the `.cu` — §10.10 keeps the
///   launcher — and it does NOT satisfy the structural gate. Twelve of §28's
///   62 unreached rows are in exactly this position: the kernel runs on every
///   fire of its outer launcher and the row is still dead vocabulary.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Consumer {
    /// **A model text calls a `dsl::cuda` wrapper that records this symbol.**
    /// The citation is the CALL SITE, not the wrapper — `dsl.rs` naming a
    /// symbol is supply, and this field is about demand. `goldens` is how
    /// many of the 73 traces under `crates/model/tests/golden/` name the
    /// symbol, counted at test time and compared against the number here.
    ModelText { cite: Cite, goldens: u32 },
    /// **`lower.rs::semantic()` maps a semantic `OpKind` straight to the
    /// symbol, with no `dsl::cuda` wrapper anywhere in the chain.** §28.5's
    /// mechanism 2, and the reason a one-hop DSL sweep is wrong in the
    /// generous direction: **21 rows reach production only this way**.
    /// No row below is one of them, and that is a measurement rather than an
    /// omission — the variant is here so that the next row reached this way
    /// is not written down as a [`Consumer::ModelText`] citing `lower.rs`.
    Lowering { cite: Cite, goldens: u32 },
    /// **A shipped model text reaches it, but only when a deployment fact
    /// selects it.** `MatW::gemm_symbol` and thirteen other DSL functions
    /// quote more than one symbol and choose on a checkpoint's `WeightRepr`,
    /// a `ScaleLayout`, a `state_bf16` flag. `arm` cites the selector arm;
    /// `publishes` cites the ONE place in the tree that constructs the fact.
    /// Both are checked. Goldens must be zero — a fixture that published the
    /// fact would produce one, and then the row is plainly [`Self::ModelText`].
    FactGated { arm: Cite, publishes: Cite },
    /// **Hand-written driver or loader code fires it**, through
    /// `ffi::pie_k_*` or a load-time transform. Nine symbols have such a call
    /// site (§28.7); being in the GENERATED dispatch is not evidence, because
    /// `emit_rust_dispatch` matches on every stated row whether or not
    /// anything fires it. Cite the hand-written line, outside `#[cfg(test)]`.
    Driver { cite: Cite },
    /// **Another launcher in the C++ archive calls it.** The kernel runs; the
    /// ROW is reached by nothing. §10.10 keeps the `.cu` and this table is
    /// about rows, so **this does not satisfy the structural gate** — a
    /// `Structural` verdict resting on it is refused exactly as `Nothing` is,
    /// with its own message.
    Cpp { cite: Cite },
    /// **Only a test reaches it.** Worth its own variant precisely because a
    /// test is a consumer that will not complain when the row it pins is the
    /// wrong one — twelve rows table-wide are in this class (§28.8). And a
    /// test that NAMES a row is not a test that FIRES it: `launch_rules.rs`
    /// asserting a `Rule` and a `file` is the table restated.
    TestOnly { cite: Cite },
    /// **Nothing reaches it, and somebody wrote down why and what would.**
    /// `executor_bind.rs`'s `AWAITING_THE_VERIFY_STASH_POOL`, `serve/load.rs`'s
    /// *"ported and has no forward path to serve"*. Distinct from
    /// [`Self::Nothing`] because the pin is a real line that a deletion would
    /// have to argue with — and amber rather than green, so the report prints
    /// it beside the reds rather than letting it read as demand.
    Awaiting { cite: Cite },
    /// **Nothing. Swept through every channel §28.5 names and found none.**
    ///
    /// `wrapper` is the `dsl::cuda` function that records the symbol, or `""`
    /// where there is none. It is not decoration: [`assert_consumers`] checks
    /// that it exists in `dsl.rs` and that **no file under `crates/model/src`
    /// mentions it**, and that the symbol appears in **zero** goldens. So
    /// this variant is measured, not asserted — which is what stops it being
    /// the lazy answer and, more importantly, stops the OPPOSITE mistake:
    /// calling a live row dead.
    Nothing { wrapper: &'static str, swept: &'static str },
}

impl Consumer {
    /// Every citation this consumer makes, for [`resolve`] to open.
    pub fn cites(&self) -> Vec<Cite> {
        match self {
            Consumer::ModelText { cite, .. }
            | Consumer::Lowering { cite, .. }
            | Consumer::Driver { cite }
            | Consumer::Cpp { cite }
            | Consumer::TestOnly { cite }
            | Consumer::Awaiting { cite } => vec![*cite],
            Consumer::FactGated { arm, publishes } => vec![*arm, *publishes],
            Consumer::Nothing { .. } => vec![],
        }
    }

    /// How many goldens this consumer CLAIMS name the symbol.
    ///
    /// Every variant but the two live ones claims zero, and the claim is
    /// checked: a `TestOnly` row with a golden is a row a model text traced,
    /// which is the classification going stale in the good direction.
    pub fn goldens(&self) -> u32 {
        match self {
            Consumer::ModelText { goldens, .. } | Consumer::Lowering { goldens, .. } => *goldens,
            _ => 0,
        }
    }

    /// Whether this may stand under a [`Class::Structural`] verdict.
    ///
    /// A wall in front of a door nobody opens is not a wall; it is a deletion
    /// candidate wearing one. [`Consumer::Cpp`] is refused for the same
    /// reason with a different sentence: the kernel runs, the row does not.
    pub fn holds_up_a_wall(&self) -> bool {
        !matches!(self, Consumer::Nothing { .. } | Consumer::Cpp { .. })
    }

    /// The one-word channel, for the report's column.
    pub fn channel(&self) -> &'static str {
        match self {
            Consumer::ModelText { .. } => "model text",
            Consumer::Lowering { .. } => "lowering",
            Consumer::FactGated { .. } => "fact-gated",
            Consumer::Driver { .. } => "driver",
            Consumer::Cpp { .. } => "C++ only",
            Consumer::TestOnly { .. } => "test only",
            Consumer::Awaiting { .. } => "awaiting",
            Consumer::Nothing { .. } => "NOTHING",
        }
    }

    /// The citation as a reader checks it: `path:line via token`.
    pub fn evidence(&self) -> String {
        match self {
            Consumer::Nothing { wrapper, swept } => {
                if wrapper.is_empty() {
                    format!("no wrapper; {swept}")
                } else {
                    format!("`cuda::{wrapper}` has no caller; {swept}")
                }
            }
            Consumer::FactGated { arm, publishes } => {
                format!("{} via {}; the fact is built only at {} ({})", arm.at, arm.names, publishes.at, publishes.names)
            }
            _ => {
                let cite = self.cites()[0];
                let goldens = self.goldens();
                if goldens == 0 {
                    format!("{} via {}", cite.at, cite.names)
                } else {
                    format!("{} via {} ({goldens} golden(s))", cite.at, cite.names)
                }
            }
        }
    }
}

/// A refused row: what it is, **what reaches it**, and why it is refused.
///
/// `consumer` sits beside `why` on purpose. `why` is a statement about the
/// launcher and `consumer` is a statement about demand, and the whole defect
/// this file is closing is that the first was being read as if it implied the
/// second.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Refusal {
    pub symbol: &'static str,
    pub class: Class,
    pub consumer: Consumer,
    pub why: &'static str,
}

/// The refused rows that are **not** simply waiting on vocabulary, one entry
/// each, with the evidence that decides it.
///
/// `why` cites into `crates/kernels-cuda/csrc/**` — the archive — and not
/// into `families/*.rs`, on purpose: the family modules are being edited, the
/// C++ is not, and a reason that cites the launcher can be checked by anyone
/// at any commit. `consumer` cites the opposite direction — into
/// `crates/model/src`, the driver, the goldens and the tests — and is checked
/// by opening the file.
#[rustfmt::skip]
pub static CLASSIFIED: &[Refusal] = &[
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
    Refusal { symbol: "gemm::act_x_wt_bf16", class: Class::Structural(Wall::HostChoice),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/glm_5/forward/mod.rs:127", names: "gemm_xwt" }, goldens: 2 },
        why: "MEASURED, not read: NOT a library call. `gemm.cpp:958-963` is `if (M == 1 && beta == 0 && gemv_bf16(...)) return;` -- OUR warp-per-row GEMV out of `gemm/gemv.cu`, which holds three `__global__`s -- and the tuner one branch up can pick `GemmKind::Gemv` at any M (`:528-545`). Two arms, one ours and one cuBLASLt's, chosen on a tuning table and an alignment" },
    Refusal { symbol: "gemm::act_x_wt_bias_bf16", class: Class::Structural(Wall::TwoLaunches),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/gpt_oss/forward/mod.rs:145", names: "gemm_bias" }, goldens: 2 },
        why: "MEASURED: `gemm.cpp:2391-2394` was a GEMM and then `kernels::norm::add_bias_bf16` -- a kernel THIS CRATE ALREADY HOSTS AND FIRES. The M=1 arm folded the bias into the gemv epilogue instead, so the row was one launch or two depending on a tuner. §45 executes the COMPOSITION `execution::COMPOSED` already stated, in `bind::service`, and the fused arm is gone: always two launches now, and `driver-cuda/tests/gemm_service_parity.rs` measures the fold bit-identical over 14,497 values on hostile input, which `gemv.hpp:25-28` had only asserted" },
    Refusal { symbol: "gemm::act_x_wt_bf16_out_fp32", class: Class::Structural(Wall::Library),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/nemotron_h/forward/mod.rs:188", names: "gemm_out_fp32" }, goldens: 2 },
        why: "one `cublasGemmEx`, bf16 in / fp32 out; `gemm.cpp:1030-1058` WAS the whole body. Still refused as a JIT extraction -- there is no kernel to extract -- but no longer C++: §45 moved the tuple to `driver-cuda/src/bind/service.rs` and `emit_dispatch` gives it a third arm. `Wall::Library` was always a statement about the BODY, never about the language" },
    Refusal { symbol: "gemm::batched_act_x_wt_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::Nothing { wrapper: "gemm_batched_xwt", swept: "0 callers of the wrapper, 0 goldens, 0 `pie_k_*`, 0 `lower.rs`; `gemm.hpp:401` DECLARES `batched_act_x_wt_bf16` and no `csrc` line CALLS it. §28.9 lists it, nearest consumer `kernels/src/lib.rs:710` -- a DOC using it as an example of `BufArray` naming" },
        why: "`cublasGemmGroupedBatchedEx` with a `cublasGemmBatchedEx` fallback; both arms the library's" },
    Refusal { symbol: "gemm::grouped_act_x_wt_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::Driver { cite: Cite { at: "crates/driver-cuda/src/fire/lora.rs:800", names: "gemm_grouped_act_x_wt_bf16" } },
        why: "one `cublasGemmGroupedBatchedEx`; `gemm.cpp:1242-1294` before §45 moved it to `driver-cuda/src/bind/service.rs`. CLASSIC cuBLAS -- the \"cuBLASLt grouped\" on record was wrong" },
    Refusal { symbol: "gemm::act_x_wt_channel_scaled", class: Class::Structural(Wall::TwoLaunches),
        consumer: Consumer::FactGated { arm: Cite { at: "crates/model-compiler/src/dsl.rs:196", names: "ScaleLayout::PerChannel" }, publishes: Cite { at: "crates/model/tests/kernels_table.rs:515", names: "ScaleLayout::PerChannel" } },
        why: "MEASURED: `gemm.cpp:2085-2133` is `quant::quantize_bf16_to_int8_per_token`, then an INT8 `cublasGemmEx`, then `quant::dequant_int32_w8a8_to_bf16`, then `norm::residual_add_bf16` when `beta != 0`. THREE OR FOUR LAUNCHES, three of them ours" },
    Refusal { symbol: "gemm::act_x_wt_grouped_scaled", class: Class::Structural(Wall::TwoLaunches),
        consumer: Consumer::FactGated { arm: Cite { at: "crates/model-compiler/src/dsl.rs:192", names: "ScaleLayout::PerGroup" }, publishes: Cite { at: "crates/model/tests/tp_quantized_spec.rs:150", names: "act_x_wt_grouped_scaled" } },
        why: "MEASURED: `gemm.cpp:1912-1953` is `quant::quantize_bf16_to_fp8_e4m3_per_token_group` and then a block-scaled `cublasLtMatmul`; when the Lt heuristic returns nothing it latches off and the dequant fallback (`:1814`) is two more" },
    Refusal { symbol: "gemm::act_x_wt_mxfp4_marlin", class: Class::Structural(Wall::TwoLaunches),
        consumer: Consumer::FactGated { arm: Cite { at: "crates/model-compiler/src/dsl.rs:190", names: "WeightRepr::Mxfp4Marlin" }, publishes: Cite { at: "crates/model/src/gemma_4/project.rs:519", names: "WeightRepr::Mxfp4Marlin" } },
        why: "MEASURED, and the NAME is the trap: this row never reached the vendored Marlin at all. `gemm.hpp:206` sets `DType::MXFP4_PACKED` and calls `act_x_w`, whose MXFP4 arm has no marlin `#ifdef` -- it runs `quant::dequant_mxfp4_to_bf16` into an `LtCtx` scratch and then `gemm_bf16_impl`. So the wall is TwoLaunches for a reason that outlived §47's deletion of `csrc/third_party/marlin`: a dequant kernel, then a GEMM" },
    Refusal { symbol: "gemm::mla_absorb_q_to_latent_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/kimi_k2/forward/mod.rs:127", names: "mla_absorbed_attention" }, goldens: 6 },
        why: "one `cublasGemmStridedBatchedEx` over the head axis; `gemm.cpp:2419-2442` was the whole body, and is `bind::service::gemm_mla_absorb_q_to_latent_bf16` since §45" },
    Refusal { symbol: "gemm::mla_absorb_latent_to_v_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/glm_5/forward/mod.rs:158", names: "mla_absorbed_attention" }, goldens: 6 },
        why: "the second absorb, same single call; `gemm.cpp:2444-2468`, in `bind::service` since §45. Its weight pointer is the SECOND half of each head's bank and the archive stepped it in `__nv_bfloat16`, not bytes -- `driver-cuda/tests/gemm_service_parity.rs` fails on that one substitution" },
    Refusal { symbol: "moe::flashinfer_cutlass_moe_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/qwen_3_5/forward/mod.rs:362", names: "moe_fused_cutlass" }, goldens: 1 },
        why: "CUTLASS grouped-GEMM MoE pipeline, returns `bool`. THE ONE THAT SURVIVES: `csrc/third_party/flashinfer_moe/*.cu` holds 0 `__global__`, `src/moe/flashinfer_moe.cu` holds 0 and calls no kernel of ours, and `cutlass/` is in no SOURCE directory of this repo -- CPM fetches it into `target/**/_deps/flashinfer-src/3rdparty/cutlass` at configure time. The kernels are templates in headers we do not have" },
    Refusal { symbol: "attn::merge_attention_states_bf16", class: Class::Structural(Wall::HostChoice),
        consumer: Consumer::Nothing { wrapper: "merge_attention_states", swept: "0 callers of the wrapper in any tracked file, 0 goldens, 0 `pie_k_*`, 0 `lower.rs`, no peel stem, no fact gate; `launch_rules.rs` STATES its two arms and fires neither, `vendor_probe.rs` compiles its text, and `attention_flashinfer_hopper_stub.cpp:53` names it in a COMMENT. §28.9" },
        why: "MEASURED: ZERO files to move -- `csrc/vendor/flashinfer/attention/cascade.cuh` is already here, byte-for-byte upstream, and NVRTC compiles it to 96,176 B with 8 of 8 symbols resolving. The wall is `cascade.cuh:644-664`, a host `if` over TWO kernels with two geometries: `MergeStatesLargeNumIndexSetsKernel` is grid `(seq_len, num_heads)` with dynamic smem, `MergeStatesKernel` is grid `(seq_len)` with none. TWO WALLS, and the predicate is only the first: `:644` is `num_index_sets >= seq_len`, a comparison of TWO OPERANDS, and every `Term` is unary (`Aligned`, `Multiple` and `Is` test against a literal; `Present` tests for null) while `Source`'s combinators stop at `Ne`, which is equality. The second is that BOTH arms launch a 2-D thread block `(bdx, bdy)` with `bdx = HEAD_DIM / vec_size`, and `Tile16`'s constant (16, 16) is the vocabulary's only 2-D block. Measured smem 8,704 B at head_dim 64/128/256 and 16,896 B at 512, all under 48 KB, so the `cudaFuncSetAttribute` at `:656` is a no-op" },
    // NCCL, and not even in this crate: `csrc/src/dist/` does not exist.
    // These are methods on the DRIVER's `NcclComm`, which is what
    // `launch_abi::the_unstated_rows...` calls `SecondNamespaceRoot`.
    // `kernels-cuda` neither includes `nccl.h` nor links NCCL.
    Refusal { symbol: "dist::all_reduce_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::TestOnly { cite: Cite { at: "crates/model/tests/tp_quantized_spec.rs:109", names: "cuda::all_reduce" } },
        why: "NCCL; no `csrc/src/dist/`, a method on the driver's `NcclComm`" },
    Refusal { symbol: "dist::all_reduce_bf16_out", class: Class::Structural(Wall::Library),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/shared/llama_like/forward/mod.rs:139", names: "all_reduce_out" }, goldens: 1 },
        why: "NCCL, out-of-place" },
    Refusal { symbol: "dist::all_gather_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::Awaiting { cite: Cite { at: "crates/driver-cuda/tests/launch_abi.rs:1555", names: "dist::all_gather_bf16" } },
        why: "NCCL all-gather" },
    // `comm/` IS GONE. `custom_all_reduce.cu` was measured at zero
    // `__global__` and zero `<<<>>>` — a 664-line HOST PROGRAM — and it, its
    // header and its sm100/sm120 stub are DELETED; the lifecycle is
    // `driver-cuda/src/fire/all_reduce.rs`. Both rowed entry points still
    // take a `CustomAllReduce*` the driver owns (a Rust struct behind an
    // opaque handle now) and still forward into headers this repo does not
    // carry: `csrc/vendor/flashinfer` holds `attention/` only, and there is
    // no in-repo copy of `flashinfer/comm/vllm_custom_all_reduce.cuh` or
    // `flashinfer/comm/trtllm_allreduce_fusion.cuh`.
    //
    // So both stay `Structural(Wall::Library)` and the class is unchanged —
    // but the WALL moved. It used to be "the launcher is C++"; it is now
    // exactly and only "the vendored tree has no `comm/`", which is
    // `vendor-role`'s to close and is what `examples/vendor_probe.rs`'s
    // `TRTLLM` candidate probes. Everything on this side of it is Rust, and
    // the refusal names the resolved template point.
    Refusal { symbol: "comm::all_reduce_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/shared/llama_like/forward/mod.rs:144", names: "all_reduce_p2p" }, goldens: 1 },
        why: "`impl_->allreduce<__nv_bfloat16>`, vLLM's NVLink kernel; `fire/all_reduce.rs::CustomAllReduce::all_reduce_bf16`, header fetched not vendored. A null `car` is a REFUSAL rather than a fallback (`Decline::NoInstance`)" },
    Refusal { symbol: "comm::all_reduce_residual_rmsnorm_bf16", class: Class::Structural(Wall::Library),
        consumer: Consumer::TestOnly { cite: Cite { at: "crates/model/tests/tp_quantized_spec.rs:104", names: "all_reduce_residual_rmsnorm" } },
        why: "`flashinfer::trtllm_allreduce_fusion`'s `kARResidualRMSNorm` — 1 of 240 template points; `fire/all_reduce.rs::CustomAllReduce::all_reduce_residual_rmsnorm_bf16`. The `_exact` twin that held this file's one `__global__` was deleted with its empty caller set" },

    // ── A: the launcher declines ─────────────────────────────────────────

    // ── A: not a C++ function ────────────────────────────────────────────
    Refusal { symbol: "qwen35_verify_stash_store", class: Class::Structural(Wall::NotAKernel),
        consumer: Consumer::Awaiting { cite: Cite { at: "crates/driver-cuda/tests/executor_bind.rs:463", names: "qwen35_verify_stash_store" } },
        why: "a `cudaMemcpyAsync` trio the executor performs; `Unstated::NotACppFunction`, and `KernelSig::operands` is empty" },
    Refusal { symbol: "qwen35_verify_stash_load", class: Class::Structural(Wall::NotAKernel),
        consumer: Consumer::Awaiting { cite: Cite { at: "crates/driver-cuda/tests/executor_bind.rs:463", names: "qwen35_verify_stash_load" } },
        why: "the load half of the same trio" },
    Refusal { symbol: "pie_lora_qkv_correction", class: Class::Structural(Wall::NotAKernel),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/shared/llama_like/forward/mod.rs:1117", names: "seam::ATTN_QV" }, goldens: 13 },
        why: "the driver's own arm: `bind/mod.rs:1895` calls `(*state).apply(ctx.cublas, ...)`, built out of grouped GEMM calls it already had" },

    // ── A: a switch over the cache scheme ────────────────────────────────
    Refusal { symbol: "attn::write_kv_to_pages", class: Class::Structural(Wall::SchemeSwitch),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/shared/llama_like/forward/mod.rs:1198", names: "write_kv_to_pages" }, goldens: 27 },
        why: "`switch (layer.scheme)` over four page formats, and throws on `first_token != 0` off native bf16; `attn/kv_paged.cu:107-160`" },
    Refusal { symbol: "attn::dequant_kv_cache_layer_to_bf16_active", class: Class::Structural(Wall::SchemeSwitch),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/shared/llama_like/forward/mod.rs:1330", names: "dequant_only" }, goldens: 9 },
        why: "one launcher over all four schemes, four dequant kernels" },

    // ── A: a host `if` whose arms are different kernels ──────────────────
    Refusal { symbol: "attn::attention_mtp_paged_history_bf16", class: Class::Structural(Wall::HostChoice),
        consumer: Consumer::Nothing { wrapper: "attention_mtp_paged_history", swept: "0 callers of the wrapper in any tracked file, 0 goldens, 0 `pie_k_*`, 0 `lower.rs`, no peel stem, no fact gate; the only other occurrences are the row, the wrapper, this report and `launch_rules.rs`, which STATES its fallback and fires neither arm. §28.9" },
        why: "falls back to `attn_mtp_history`, a DIFFERENT symbol, when `max_global_tokens + history_steps > 8192` -- a hard-coded literal and NOT a shared-memory query; `attn/attention_naive.cu:114`. THREE-way and not two: `:105`'s `max_global_tokens <= 0` falls through to the same place, and that place is another HOST LAUNCHER (`:52`) rather than a kernel. RE-MEASURED: the GRID is no longer part of the wall -- `SdpaVector` reproduces `dim3 grid(num_q_heads, num_tokens)` at a 256 block at the rectangle a fire supplies, at both shapes. The shared allocation is the whole of what is missing: `4 * (max_global_tokens + history_steps + 256)` where `max_global_tokens` is an operand no `Dims` axis carries. The fallback arm's `4 * (history_steps + 256)` IS `SdpaVector`'s `4 * (rows + 256)` -- on exactly the fires where `history_steps == num_tokens`, and an under-allocation on the rest is an out-of-bounds shared read and not a wrong number. No `Term` states a SUM of two operands against a literal either" },
    Refusal { symbol: "ssm::nemotron_mamba_split_bf16", class: Class::Structural(Wall::HostChoice),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/nemotron_h/forward/mod.rs:105", names: "nemotron_mamba_split" }, goldens: 2 },
        why: "`gate == nullptr` (`nemotron_h.cu:37`) reaches `mamba_split_conv_dt`, a different kernel. RE-MEASURED: the arm this row keeps its wall for is the OTHER one -- `#split` has had a row since `ElementwiseIn` landed and fires byte-identical at two shapes. Sweeping all 40 rules over every assignment of the launcher's own numbers finds `Elementwise` reproducing `ceil(N * (conv_dim + num_heads) / 256)` EXACTLY, at `width = conv_dim + num_heads` -- a number no fire supplies, because `abi.rs:1801` fills `Dims::width` from the FIRST result's width. One `Dims` axis, not a row and not a rule. `Term::Present` reads `Fact::Address` and cannot be used either: `table/ssm.rs:39` binds `gate <- Source::Out(0)` and does not declare it `| null`, which `Specialisation::agrees` refuses outright" },
    Refusal { symbol: "ssm::nemotron_mamba_ssm_batched_bf16", class: Class::Structural(Wall::HostChoice),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/nemotron_h/forward/mod.rs:135", names: "nemotron_mamba_ssm" }, goldens: 2 },
        why: "TWO live kernel forms chosen on `sequence_prefill`, a `bool` OPERAND -- the middle arm at `nemotron_h.cu:144` is `if constexpr (false)`, dead, and a fourth form at `:185` is behind an unconditional `return`. CONFIRMED. `Term::Is` does express the predicate. What is missing is NOT a row: the prefill arm is `(R, num_heads, ceil(head_dim/16))` at a 512 block and no rule of the 40 pairs those; the warp arm is reached by exactly one shape function, `Rope`'s grid `(rows, q_heads)` at 256 with `4 * head_dim`, which would agree only where `rows == R` AND `q_heads == v_h` AND `head_dim == 2 * k_d`. Two `Dims` axes (the mamba head count `Gdn(\"v_h\")` and the state width `Gdn(\"k_d\")`) and at least one rule" },
    Refusal { symbol: "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16", class: Class::Structural(Wall::HostChoice),
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/qwen_3_5/forward/mod.rs:658", names: "gdn_step_batched" }, goldens: 1 },
        why: "picks a DIFFERENT KERNEL on `V_d == 128 && K_d == 128` -- a fire's shape now, not `getenv(\"PIE_QWEN35_GDN_SMEM_STEP\")`, which is deleted; the two arms are BYTE-IDENTICAL (§30), so what is missing is a row for `..._gqa_smem<128>`, not a semantics" },

    // ── A: one symbol, more than one launch ──────────────────────────────
    Refusal { symbol: "attn::compact_page_csr", class: Class::Structural(Wall::TwoLaunches),
        consumer: Consumer::TestOnly { cite: Cite { at: "crates/driver-cuda/tests/page_mask_parity.rs:128", names: "compact_page_csr" } },
        why: "`count_kept` then `scan_and_scatter`; a row for either states half a contract" },
    Refusal { symbol: "norm::rmsnorm_bf16_with_fp16", class: Class::Structural(Wall::TwoLaunches),
        consumer: Consumer::TestOnly { cite: Cite { at: "crates/kernels-cuda-new/tests/layers.rs:489", names: "norm::rmsnorm_bf16_with_fp16" } },
        why: "unaligned rows fall back to `rmsnorm_bf16` PLUS `quant::bf16_to_fp16`; `norm/rmsnorm.cu:50-53` says so" },

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
    Refusal { symbol: "rope::qk_rmsnorm_rope_bf16_rounded", class: Class::Stale,
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/gemma_4/forward/mod.rs:217", names: "qk_rmsnorm_rope_rounded_q_only" }, goldens: 4 },
        why: "NOT `rope/rope.cu:214-215`, which is `dim3(num_tokens, num_q_heads)` — Q ONLY. The real call site (`model/src/gemma_4/forward/mod.rs:217`) reaches it with `k = nullptr` and `num_kv_heads = 0`, while `Dims::kv_heads` is the FIRE's head count (`bind/mod.rs:1379`) and is non-zero there. `RowsPackedHeadsNarrow` would open `q + kv` blocks and the excess would rotate a null `k`; `packed_heads`' zero-refusal cannot fire, because the zero is in the OPERAND and not in the `Dims`. NEEDS a separate symbol for the q-only form (as `_devwin` has) or a rule whose head axis reads the operand — reproducing `rope/rope.cu:214`" },
    Refusal { symbol: "attn::qkv_decode_qk_norm_rope_write_kv_bf16", class: Class::Stale,
        consumer: Consumer::ModelText { cite: Cite { at: "crates/model/src/shared/llama_like/forward/mod.rs:1514", names: "qkv_decode_qk_norm_rope_write_kv_region" }, goldens: 1 },
        why: "the reason on record was the vocabulary and the vocabulary caught up: `WarpPackedHeads` and `RowsPackedHeadsNarrow` were BOTH ported from this launcher (`attn/qkv_fused.cu:51-53`, `:98-102`) and `Term::Present` reads `rope_table != nullptr`. What refuses it NOW is that a `Specialisation` may not change a `LaunchRule` — the launcher picks between those two GEOMETRIES on `head_dim` — and this audit KEPT that invariant: `fire.rs:176-186` evaluates the grid from the base row before consulting the specialisation, four readers take `KernelSig::launch` as a contract, and lifting it would not land the row anyway because `head_dim == 64 | 128 | 256` is still unspellable (`Term::Multiple { of: 64 }` holds of 192). `tests/specialise.rs::agrees_refuses_an_arm_that_changes_the_launch_rule`" },
    Refusal { symbol: "ssm::build_nemotron_moe_ptrs_decode_batched_bf16", class: Class::Stale,
        consumer: Consumer::Nothing { wrapper: "build_nemotron_moe_ptrs_decode", swept: "0 callers of the wrapper, 0 goldens, 0 `pie_k_*`, 0 `lower.rs`, no C++ CALL; its whole occurrence set is the table row, the wrapper and this report. §28.9" },
        why: "refused as an extent no shape produces; `ElementwiseIn` is `rows * in_width`, which is `rows * top_k` to the digit, and `Ty::BufArray*` now names its `void**` operands. NOT RE-DERIVED: `families/ssm.rs` is outside this session's grant" },
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
    for Refusal { symbol, class, .. } in CLASSIFIED {
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
        let idx = match CLASSIFIED.iter().find(|r| r.symbol == *symbol) {
            Some(Refusal { class: Class::Structural(_), .. }) => 0,
            Some(Refusal { class: Class::Text, .. }) => 2,
            Some(Refusal { class: Class::Stale, .. }) => 3,
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
            .filter(|r| refused.contains(r.symbol) && want(&r.class))
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
    //
    //    ── KNOWN DRIFT: A WALK IS AN `Op` AND `Wall` CANNOT SAY SO ─────────
    //
    //    `Execution::Walk` answers `Kind::Op` (`execution.rs`'s `kind`), and
    //    `kind_of` answers `Kind::Kernel` for anything not in `CLASSIFIED`.
    //    So EVERY walked row that no unit hosts fails this assertion. That
    //    was already true of two before the `norm/`+`rope/` port —
    //    `moe::moe_grouped_gemm_bf16` and `sample::lm_head_gemv_argmax_int8`,
    //    both walks, neither classified, and both fired under a device row
    //    with a DIFFERENT name so `unit_of` is `None` — and the port made it
    //    seventeen, because the fifteen launchers of `norm/rmsnorm.cu`,
    //    `norm/dsv4_hc.cu` and `rope/rope.cu` became
    //    `driver-cuda/src/fire/{rmsnorm,dsv4_hc,rope}.rs` and twelve of their
    //    device rows were renamed for the same reason (`families/rope.rs`'s
    //    `ROPE_SIGS` doc has the table).
    //
    //    What closes it is a `Wall` whose `kind_of_wall` is `Kind::Op` and
    //    that is HONEST about these: not `SchemeSwitch` (four of the fifteen
    //    do pick a scheme, eleven do not) and not `TwoLaunches` (all fifteen
    //    are one launch). The shape they share is `Control::Supplies`: one
    //    kernel, and a grid or a template argument the host computes from a
    //    comparison the `Source` grammar cannot spell. That is a seventh
    //    wall, and adding it is a `migration_status` edit, not a port edit —
    //    left for whoever owns this report, with the count above as the
    //    receipt for having measured it.
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
            CLASSIFIED.iter().any(|r| r.symbol == symbol && matches!(r.class, Class::Structural(_))),
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

    // 13. **DEMAND, beside supply.** Checks 1-12 are all statements about
    //     supply: what the launcher is, which table holds it, which service
    //     runs it, what it composes into. Not one of them asks whether
    //     anything WANTS it, and that gap is the most expensive one this file
    //     has had: sixteen symbols classified hard — "the launcher returns
    //     `bool` and declines", "a tuning table picks 1 of 15", "needs sm90"
    //     — and then found to be reached by nothing at all.
    //
    //     Check 1 above is the near miss: it holds that every classification
    //     names a live ROW. `kernels_table.rs` is the other: table ⊇ dsl.
    //     Both stop at the wrapper, and the wrapper exists whether or not a
    //     model asked, because the DSL surface was GENERATED from the
    //     launcher headers (`6d02452de`, `c0e57c7f1`). One hop further is
    //     `assert_consumers`, and it is deliberately the last check: the
    //     twelve above must hold before "what reaches this?" is even a
    //     well-posed question.
    assert_consumers();
}

/// Demand, beside supply — the column this file was missing.
///
/// `why` is a statement about a LAUNCHER; `consumer` is a statement about
/// whether anything calls it. Reading the first as if it implied the second
/// cost sixteen wrong classifications and twenty-one duplicate rows, because
/// the DSL surface was generated FROM the launcher headers (`6d02452de`,
/// `c0e57c7f1`) and so a wrapper exists whether or not a model asked.
///
/// Printed by [`assert_consumers`] too, on its way to failing: a gate that
/// turns a row red should show the reader the whole table it judged it in.
pub fn report_consumers() {
    // Demand, beside supply. The column this file was missing: `why` is a
// statement about a launcher and `consumer` is a statement about whether
// anything calls it, and reading the first as if it implied the second
// cost sixteen wrong classifications and twenty-one duplicate rows.
println!("\nWhat reaches each classified symbol ({}) -- checked citations, not adjectives:\n", CLASSIFIED.len());
let mut by_channel: BTreeMap<&str, Vec<&Refusal>> = BTreeMap::new();
for row in CLASSIFIED {
    by_channel.entry(row.consumer.channel()).or_default().push(row);
}
for (channel, rows) in &by_channel {
    println!("  {channel} ({})", rows.len());
    for row in rows {
        println!("      {:<54} {}", row.symbol, row.consumer.evidence());
    }
}
let red: Vec<&Refusal> = CLASSIFIED.iter().filter(|r| !r.consumer.holds_up_a_wall()).collect();
println!(
    "\n{} classified symbol(s) are reached by NOTHING. `assert_consumers` refuses a\n\
     `Structural` verdict for any of them -- a wall in front of a door nobody opens is\n\
     a deletion candidate wearing one -- so any that survive here are class B/C/D:\n",
    red.len()
);
for row in &red {
    println!("  {:<54} {:?}", row.symbol, row.class);
    println!("      {}", row.consumer.evidence());
}
}

/// How far a citation's line may drift before it is a failure.
///
/// The token must be in the file — that is not negotiable and is what makes a
/// fabricated consumer catchable. The LINE is checked to a screen, because
/// its whole job is to put a reader's eye on the evidence in seconds, and
/// because five crates in this tree are being edited by other agents while
/// this file is read. Beyond a screen the citation is not helping, and
/// [`resolve`] fails NAMING THE TRUE LINE, so the fix is one character.
const DRIFT: usize = 40;

/// The repository root, from this file's crate.
fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

/// Open the cited file and look for the token.
///
/// This is the whole argument for an enum over free text: the variant says
/// which CHANNEL and the citation says WHERE, and a `consumer` that says
/// `"reachable"` is not merely discouraged — it cannot be written down. What
/// makes the citation more than a nicer string is that this function reads it.
///
/// # Errors
///
/// The file does not exist; the token is nowhere in it; the token is in it
/// but more than [`DRIFT`] lines from the cited line. The third names the
/// true line.
pub fn resolve(cite: Cite) -> Result<(), String> {
    let (path, line) = match cite.at.rsplit_once(':') {
        Some((path, n)) if !n.is_empty() && n.chars().all(|c| c.is_ascii_digit()) => {
            (path, Some(n.parse::<usize>().expect("digits parse")))
        }
        _ => (cite.at, None),
    };
    let full = repo_root().join(path);
    let text = std::fs::read_to_string(&full)
        .map_err(|err| format!("cites `{}`, which does not read: {err}", cite.at))?;
    let hits: Vec<usize> = text
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(cite.names))
        .map(|(i, _)| i + 1)
        .collect();
    let Some(&first) = hits.first() else {
        return Err(format!(
            "cites `{}` for `{}` — and `{}` does not contain that token ANYWHERE. \
             A consumer nobody can find is not a consumer",
            cite.at, cite.names, path
        ));
    };
    if let Some(line) = line {
        let nearest = hits.iter().copied().min_by_key(|h| h.abs_diff(line)).unwrap_or(first);
        if nearest.abs_diff(line) > DRIFT {
            return Err(format!(
                "cites `{}` for `{}`, and the nearest occurrence in that file is line {nearest} \
                 — {} lines away. The citation has drifted; write `{}:{nearest}`",
                cite.at,
                cite.names,
                nearest.abs_diff(line),
                path
            ));
        }
    }
    Ok(())
}

/// The golden corpus, read once: for every trace, the symbols it names.
///
/// Route two of §28.9's extraction, done the cheap way — the raw text scanned
/// for the QUOTED symbol, which is how a `Launch` names its kernel in these
/// files and which is what keeps `dist::all_reduce_bf16` from matching
/// `dist::all_reduce_bf16_out`. That substring collision is not hypothetical:
/// it made an unreached NCCL row read as 64 golden ops on the first pass of
/// this instrument.
pub fn golden_corpus() -> Vec<(String, String)> {
    let dir = repo_root().join("crates/model/tests/golden");
    let mut traces = Vec::new();
    let entries = std::fs::read_dir(&dir).unwrap_or_else(|err| {
        panic!("the golden corpus at `{}` does not read: {err}", dir.display())
    });
    for entry in entries {
        let path = entry.expect("a directory entry").path();
        if path.extension().is_some_and(|e| e == "json") {
            let name = path.file_name().expect("a file name").to_string_lossy().into_owned();
            traces.push((name, std::fs::read_to_string(&path).expect("a golden reads")));
        }
    }
    traces.sort();
    traces
}

/// How many traces name this symbol.
pub fn goldens_naming(corpus: &[(String, String)], symbol: &str) -> u32 {
    let quoted = format!("\"{symbol}\"");
    corpus.iter().filter(|(_, text)| text.contains(&quoted)).count() as u32
}

/// **A `Structural` verdict for a symbol nothing names is refused.**
///
/// # The rule, and why it is a gate rather than a convention
///
/// `Declines`, `Tuned`, `Arch` are all true, checked statements about a
/// LAUNCHER; none of them is a statement about whether anything CALLS it.
/// Sixteen symbols were classified hard on that basis and then found to be
/// reached by nothing; twenty-one more were deleted as duplicates. Both
/// existing tests stop exactly one hop short — `kernels_table.rs` checks
/// table ⊇ dsl, and [`assert_total`]'s check 1 holds that every
/// classification names a live ROW — and **neither traverses wrapper →
/// caller**. What would have caught all four of §31's deletions costs
/// seconds: `git grep -w <dsl_fn>`, zero for all four, and zero on the day
/// each was classified.
///
/// So: made structural, it is this function.
///
/// # What is checked, and what each check can catch
///
/// 1. **`Structural` + [`Consumer::Nothing`] fails.** The headline rule.
/// 2. **`Structural` + [`Consumer::Cpp`] fails too**, with its own sentence.
///    A C++-internal caller keeps the `.cu` (§10.10) and says nothing about
///    the row; twelve of §28's 62 unreached rows are exactly there.
/// 3. **Every citation resolves** — file, token, and line to a screen. This
///    is what makes the opposite perturbation catchable: marking a dead
///    symbol as consumed requires naming a file and a token that are really
///    there, and `merge_attention_states` is in no file under
///    `crates/model/src`.
/// 4. **Every claimed golden count is the measured one**, over all 73 traces.
///    A live claim is corroborated by a second, independent instrument; a
///    `TestOnly`/`FactGated`/`Awaiting`/`Nothing` row claims zero and a
///    golden appearing means a text has started tracing it — the
///    classification going stale in the good direction.
/// 5. **[`Consumer::Nothing`] is MEASURED, not asserted.** Its wrapper must
///    exist in `dsl.rs` and must appear in no file under `crates/model/src`.
///    Without this the variant would be the lazy answer; with it, calling a
///    live row dead is a failure that names the file that calls it.
/// 6. **The denominator.** The count of rows checked, of citations opened and
///    of traces read are all returned and asserted by the caller. Five gates
///    this session passed while measuring nothing (§21.9, §22.6): a gate that
///    filters must assert its own denominator.
///
/// # Why this returns `Result` rather than panicking
///
/// So that it can be pointed at a doctored table. A gate nobody has seen fail
/// is a gate nobody has tested — `tests/consumer.rs` perturbs this one in
/// both directions, marking a live symbol consumer-less and a dead one
/// consumed, and asserts each is caught by name.
pub fn check_consumers(rows: &[Refusal], refused: &BTreeSet<&str>) -> Result<Checked, String> {
    let corpus = golden_corpus();
    if corpus.len() < 2 {
        return Err(format!(
            "the golden corpus holds {} trace(s) — every golden claim below would pass \
             vacuously, which is the exact shape of the five gates §21.9 found",
            corpus.len()
        ));
    }
    let mut checked =
        Checked { rows: 0, refused: 0, citations: 0, traces: corpus.len(), nothing: 0 };
    let mut failures: Vec<String> = Vec::new();
    for row in rows {
        checked.rows += 1;
        let symbol = row.symbol;

        // 1 and 2: the verdict must rest on something.
        if matches!(row.class, Class::Structural(_)) && !row.consumer.holds_up_a_wall() {
            failures.push(match row.consumer {
                Consumer::Cpp { cite } => format!(
                    "`{symbol}` is classified {:?} and the only thing that reaches it is C++ \
                     (`{}`). That keeps the LAUNCHER — §10.10 — and says nothing about the ROW. \
                     A wall in front of a door nobody opens is a deletion candidate wearing one: \
                     state a consumer of the row, or move it out of `Structural`",
                    row.class, cite.at
                ),
                _ => format!(
                    "`{symbol}` is classified {:?} and NOTHING reaches it. A wall in front of a \
                     door nobody opens is not a wall; it is a deletion candidate wearing one. \
                     Either name what calls it — a model text and its DSL method, a driver path, \
                     a golden, a test — or take the `Structural` verdict off it. Deleting it is a \
                     separate task with its own evidence (§10.10: a launcher goes only when its \
                     WHOLE consumer set has gone)",
                    row.class
                ),
            });
        }

        // 3: every citation opened and read.
        for cite in row.consumer.cites() {
            checked.citations += 1;
            if let Err(why) = resolve(cite) {
                failures.push(format!("`{symbol}`'s consumer {why}"));
            }
        }

        // 4: the golden count, derived here rather than believed.
        let measured = goldens_naming(&corpus, symbol);
        let claimed = row.consumer.goldens();
        if measured != claimed {
            failures.push(format!(
                "`{symbol}`'s consumer claims {claimed} golden(s) and {measured} of the {} traces \
                 name it. {}",
                corpus.len(),
                if claimed == 0 {
                    "A row nothing is supposed to trace is being traced — the classification has \
                     gone stale in the GOOD direction, and the consumer should say which text"
                } else {
                    "Either the citation is wrong or the goldens moved; re-run the sweep"
                }
            ));
        }

        // 5: `Nothing` is measured. A wrapper that a model text mentions is
        //    not a wrapper nothing calls.
        if let Consumer::Nothing { wrapper, .. } = row.consumer {
            checked.nothing += 1;
            if !wrapper.is_empty() {
                let dsl = repo_root().join("crates/model-compiler/src/dsl.rs");
                let text = std::fs::read_to_string(&dsl)
                    .map_err(|err| format!("`{symbol}`: `dsl.rs` does not read: {err}"))?;
                if !text.contains(&format!("fn {wrapper}(")) {
                    failures.push(format!(
                        "`{symbol}` states `Nothing` behind the wrapper `cuda::{wrapper}`, and \
                         `dsl.rs` declares no `fn {wrapper}(`. Name the wrapper that records the \
                         symbol, or `\"\"` if there is none"
                    ));
                }
                if let Some(at) = names_in(&repo_root().join("crates/model/src"), wrapper) {
                    failures.push(format!(
                        "`{symbol}` states that NOTHING reaches it — and `{at}` mentions its \
                         wrapper `{wrapper}`. A model text naming the wrapper is exactly the \
                         consumer this field is for; state it"
                    ));
                }
            }
        }

        // The scope, counted rather than assumed. A classification for a
        // symbol that has since been MIGRATED is stale in the good direction
        // and `report_partition` prints it as `overtaken`; the rules above
        // still ran on it, because a wall is refused on its own terms and not
        // on whether the refusal list still holds the symbol.
        if refused.contains(symbol) {
            checked.refused += 1;
        }
    }

    // Every failure, not the first. A gate that stops at one turns a report
    // into a queue, and the whole point of this field is that the reader sees
    // demand beside supply for the WHOLE table at once.
    if !failures.is_empty() {
        return Err(format!(
            "{} consumer claim(s) do not hold:\n\n{}",
            failures.len(),
            failures.iter().map(|f| format!("  * {f}\n")).collect::<String>()
        ));
    }
    Ok(checked)
}

/// The first file under `dir` whose text names `token`, as a repo-relative
/// path — the measurement behind [`Consumer::Nothing`].
fn names_in(dir: &std::path::Path, token: &str) -> Option<String> {
    let mut stack = vec![dir.to_path_buf()];
    let word = |line: &str| {
        line.match_indices(token).any(|(i, _)| {
            let before = line[..i].chars().next_back();
            let after = line[i + token.len()..].chars().next();
            let plain = |c: Option<char>| !c.is_some_and(|c| c.is_alphanumeric() || c == '_');
            plain(before) && plain(after)
        })
    };
    while let Some(path) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&path) else { continue };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                let Ok(text) = std::fs::read_to_string(&path) else { continue };
                if let Some((n, _)) = text.lines().enumerate().find(|(_, l)| word(l)) {
                    let shown = path.strip_prefix(repo_root()).unwrap_or(&path).display();
                    return Some(format!("{shown}:{}", n + 1));
                }
            }
        }
    }
    None
}

/// What [`check_consumers`] measured, so the caller can assert its size.
///
/// A gate that filters must assert its own denominator (§21.9). This is that
/// denominator, and it is four numbers rather than one because the gate reads
/// four things: the classification, the citations, the golden corpus, and the
/// `Nothing` rows it had to measure rather than believe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Checked {
    pub rows: usize,
    pub refused: usize,
    pub citations: usize,
    pub traces: usize,
    pub nothing: usize,
}

/// [`check_consumers`] over the real table, panicking, with the counts held.
///
/// # Panics
///
/// Naming the symbol, and what would close it.
pub fn assert_consumers() -> Checked {
    let refused = refused_set();
    let checked = check_consumers(CLASSIFIED, &refused).unwrap_or_else(|why| {
        report_consumers();
        panic!(
            "\n{why}\nThis is the report, not a bug to route around. §10.10: a launcher goes \n\
             only when its WHOLE consumer set has gone, and that is a separate task with its \n\
             own evidence. Do not silence this by editing the `consumer:` field -- every claim \n\
             in it is opened and read.\n"
        )
    });

    // The denominator, stated. Without these four the gate above could
    // silently check nothing at all: a table that lost its rows, a corpus
    // that moved, a `Consumer` variant that stopped citing anything.
    assert_eq!(checked.rows, CLASSIFIED.len(), "the gate did not visit every classified row");
    let live = CLASSIFIED.iter().filter(|r| refused.contains(r.symbol)).count();
    assert_eq!(checked.refused, live, "the gate did not visit every refused classified row");
    assert!(live > 0, "no classified symbol is still refused — the gate is measuring nothing");
    assert!(
        checked.traces >= 60,
        "the golden corpus fell to {} traces — every golden claim in the table is now \
         checked against almost nothing",
        checked.traces
    );
    let citations: usize = CLASSIFIED.iter().map(|r| r.consumer.cites().len()).sum();
    assert_eq!(checked.citations, citations, "a citation went unopened");
    assert!(
        citations >= CLASSIFIED.len() - checked.nothing,
        "a consumer other than `Nothing` cited no file — the enum's whole claim is that \
         it cannot"
    );
    checked
}

/// Twenty characters of progress, because a column of numbers hides the shape.
fn bar(part: usize, whole: usize) -> String {
    const WIDTH: usize = 20;
    let filled = if whole == 0 { 0 } else { WIDTH * part / whole };
    format!("{}{}", "#".repeat(filled), ".".repeat(WIDTH - filled))
}
