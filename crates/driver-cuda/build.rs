//! The launch bridge (retirement plan phase A), behind the `bridge` feature.
//!
//! Without `bridge` this script does nothing at all — the crate's
//! toolkit-free build is load-bearing for CI and must not learn a CUDA
//! dependency here. With it, the Rust half of the flat launch ABI is
//! generated from the kernel table at build time and never committed:
//!
//! * `launch_bindings.rs` — `emit_rust_bindings` over every family table,
//!   included by `bind::abi::ffi`. These are DECLARATIONS, which is why
//!   they live with their caller: they are spelled in this crate's own
//!   `#[repr(C)]` mirrors (`WeightView`, `DType`, the workspace views), and
//!   any number of crates may declare one symbol.
//! * `rust_dispatch.rs` — `emit_rust_dispatch` over the same tables, the
//!   statement-keyed `match` the binder includes.
//!
//! Both emitters are `kernels_cuda_new::abi`'s, and that is a recent change
//! with a measurable point. They were `kernels_cuda::abi`'s while the archive
//! owned the rows, so THIS SCRIPT depended on a crate that runs CMake and
//! nvcc to obtain a function from rows to text. It does not any more: the
//! emitters read `kernels_cuda_new::table` and `kernels_cuda_new::device`,
//! neither needs a toolkit, and `kernels-cuda` is off this crate's
//! `[build-dependencies]` entirely. What is left of the edge to that crate is
//! the archive twice — `bridge = ["kernels-cuda/native"]` and the
//! dev-dependency — which is what it should be.
//!
//! The C shim that DEFINES those symbols is still built by `kernels-cuda`,
//! from the same `emit_c_shim`, under its `native` feature — which `bridge`
//! turns on. A definition may exist once, so the crate that owns the
//! launchers owns the entry points forwarding into them; this crate was only
//! ever the first caller.
//!
//! The link directives for BOTH archives live HERE, and the order is
//! load-bearing: a static archive is scanned once, in place, so the shim that
//! references the launchers must precede `pie_kernels_cuda` on the link line.
//!
//! That sentence used to end *"`kernels-cuda` emits the shim's directive, and
//! cargo puts a dependency's ahead of its dependent's; ours follow."* It was
//! false, it was the whole defect, and the correction is the rule this script
//! now follows for both: **the crate that references a symbol names the
//! archive that defines it.** Cargo hands a build script's `-l` to its own
//! package's lib and nowhere else; a `-l` crosses a crate boundary only in
//! rustc's metadata, and only for a crate rustc LOADED. `kernels_cuda` is not
//! one — nothing has named `kernels_cuda::` since §21.8 — so its `-l` and the
//! `+bundle`d archive inside its rlib both went nowhere, while the identical
//! `static=pie_kernels_cuda` below has always worked because `driver_cuda`
//! is loaded by everything that links it.
//!
//! # The third thing this script decides: which rows do not go that way
//!
//! A row named by `kernels_cuda_new::device::JIT_DISPATCHED` has no shim
//! entry — `emit_c_shim` skips it — and its generated arm calls
//! `bind::jit::fire`, which forwards to `kernels-cuda-new`. So this script is
//! where the two halves of that decision are read together, and
//! [`bridge::routed_rows_are_hosted`] is where the agreement between them is
//! checked while it is still a build error.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var_os("CARGO_FEATURE_BRIDGE").is_none() {
        return;
    }
    bridge::build();
}

mod bridge {
    use std::path::{Path, PathBuf};

    /// Every family table, in the crate's own concatenation order.
    ///
    /// Named one by one rather than taken from `table::TABLES`, and that is
    /// the same choice `kernels_cuda::KERNELS`'s re-export made for the same
    /// reason: a family that appeared silently would be in the concatenation
    /// and absent from every reader that walks the modules. This is one of
    /// those readers.
    fn tables() -> Vec<&'static [kernels_cuda_new::KernelSig]> {
        vec![
            kernels_cuda_new::table::attn::KERNELS,
            kernels_cuda_new::table::rope::KERNELS,
            kernels_cuda_new::table::norm::KERNELS,
            kernels_cuda_new::table::mlp::KERNELS,
            kernels_cuda_new::table::gemm::KERNELS,
            kernels_cuda_new::table::moe::KERNELS,
            kernels_cuda_new::table::ssm::KERNELS,
            kernels_cuda_new::table::quant::KERNELS,
            kernels_cuda_new::table::layout::KERNELS,
            kernels_cuda_new::table::sample::KERNELS,
            kernels_cuda_new::table::adapter::KERNELS,
            // The second table: launchers the driver fires with no DSL
            // statement (envelope tier, QKV split, mask packers, cell
            // moves). Same rows, same proof, outside the DSL-surface
            // equality — see `kernels_cuda_new::table::driver_internal`.
            kernels_cuda_new::table::driver_internal::DRIVER_KERNELS,
        ]
    }

    fn cuda_home() -> PathBuf {
        std::env::var_os("CUDA_HOME")
            .or_else(|| std::env::var_os("CUDA_PATH"))
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"))
    }

    /// Every routed symbol is one `kernels-cuda-new` will compile — checked
    /// here, where it is a build error.
    ///
    /// # Why this gate exists, and what it is guarding against
    ///
    /// The routed set is `kernels_cuda_new::device::JIT_DISPATCHED` and it is
    /// read by TWO emitters: `emit_c_shim` skips those rows, so the
    /// `pie_k_*` entry is not emitted, and `emit_rust_dispatch` sends them to
    /// `bind::jit::fire`, which forwards to `kernels_cuda_new::fire`. One list
    /// feeding both is what makes a row reversible — shorten it and the shim
    /// entry comes back with the arm that calls it — and it is also what makes
    /// a routed row's absence from the JIT crate unrecoverable: the shim entry
    /// is gone, so there is nothing to fall back to and the fire would be
    /// refused at run time as `Error::Unknown`, on a device, in a model text,
    /// once per launch.
    ///
    /// Layer 2 of `kernels-cuda-new` answers "does some unit host this" from
    /// the table alone — no cudarc, no device, no toolkit — which is exactly
    /// what lets the question be asked here instead. `unit::unit_of` is the
    /// ungated spelling; `kernels_cuda_new::hosts` is the same predicate
    /// behind that crate's `_cuda` gate, and its own doc calls it *"what a
    /// dispatcher asks before it emits an arm"*. This is that dispatcher, and
    /// this is where it emits the arm.
    ///
    /// A second list, local to this driver, was considered and rejected. Two
    /// lists that can disagree have two failure modes and both are silent
    /// until late: a symbol routed here but absent from `JIT_DISPATCHED` still
    /// has a shim entry, so one row gets two live implementations and each
    /// test passes on whichever half it exercises (`new-horizon.md` §10.10's
    /// worst case); a symbol in `JIT_DISPATCHED` but absent here loses its
    /// shim entry and links against a `pie_k_*` nothing defines.
    fn routed_rows_are_hosted(jit: &[&'static kernels_cuda_new::device::DeviceKernel]) {
        let orphans: Vec<&str> = jit
            .iter()
            .map(|row| row.sig.symbol)
            .filter(|symbol| kernels_cuda_new::unit::unit_of(symbol).is_none())
            .collect();
        assert!(
            orphans.is_empty(),
            "these symbols are routed to the JIT (kernels_cuda_new::device::JIT_DISPATCHED) \
             and no kernels-cuda-new unit hosts them, so their shim entry is not emitted and \
             nothing would fire them: {}. Either add the row to a unit in that crate, or take \
             the symbol off JIT_DISPATCHED to put it back on the ahead-of-time path.",
            orphans.join(", ")
        );
    }

    /// Every entry point the generated text can reach is one the archive this
    /// script names actually defines — checked here, where it is a build
    /// error.
    ///
    /// # The sibling of [`routed_rows_are_hosted`], for the rows that go the
    /// other way
    ///
    /// That gate reads the ROUTED direction: a symbol on `JIT_DISPATCHED` must
    /// be hosted by some `kernels-cuda-new` unit, because its shim entry is
    /// deliberately not emitted and nothing else would fire it. This one reads
    /// the UNROUTED direction, which had no gate at all: a symbol the dispatch
    /// calls, or the bindings declare, must be one the shim defines. The two
    /// together say that every row is reachable exactly once, by exactly one
    /// path, and neither weakens the other — they are read from the same
    /// `jit` list, which is the property `routed_rows_are_hosted`'s doc calls
    /// *"one list feeding both is what makes a row reversible"*.
    ///
    /// It is two assertions because the two halves fail differently.
    ///
    /// **Calls must be defined.** This is the one that would have caught
    /// §21.11: in the arrangement that failed, this script named no shim
    /// archive at all, so there was nothing to resolve against and the
    /// `expect` above is already a build error. It also catches a stale
    /// archive and a row whose shim entry vanished under an edit.
    ///
    /// **Declarations that are not defined must be JIT rows.**
    /// `emit_rust_bindings` writes an `unsafe extern "C"` block for every
    /// stated row — 219 today — while `emit_c_shim` skips the routed ones, so
    /// the archive holds 212 and **seven declarations have no definition**.
    /// That is correct and must stay narrow: those seven are exactly the rows
    /// `bind::jit::fire` serves. Nothing calls them today, and a hand-written
    /// arm that did would compile — the declaration is right there — and fail
    /// at LINK, in one feature combination, with the symbol list this whole
    /// section is about. The assertion pins the difference to the `jit` list,
    /// so a row leaving the shim for any OTHER reason is a build error the
    /// hour it happens.
    ///
    /// Reading the archive's own symbol index rather than shelling out to
    /// `nm`: the index lists defined externals and nothing else, it is eight
    /// lines of format, and a build script that needs binutils on PATH to
    /// check an invariant is one that silently stops checking on a machine
    /// without it.
    fn every_call_resolves_in_the_shim(
        archive: &Path,
        dispatch: &str,
        bindings: &str,
        jit: &[&'static kernels_cuda_new::device::DeviceKernel],
    ) {
        let defined = archive_defines(archive);
        let named = |text: &str| -> Vec<String> {
            let mut out: Vec<String> = text
                .match_indices("pie_k_")
                .map(|(i, _)| {
                    text[i..]
                        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                        .next()
                        .unwrap_or_default()
                        .to_string()
                })
                .collect();
            out.sort();
            out.dedup();
            out
        };

        let missing: Vec<String> = named(dispatch)
            .into_iter()
            .filter(|s| !defined.contains(s))
            .collect();
        assert!(
            missing.is_empty(),
            "`rust_dispatch.rs` calls {} entry point(s) that {} does not define: {}. \
             The dispatch and the shim are generated from ONE table by one pair of \
             emitters, so this is not a disagreement between them -- it is the archive \
             being older than the rows, or a row that lost its shim entry without \
             losing its arm.",
            missing.len(),
            archive.display(),
            missing.join(", ")
        );

        // A declaration with no definition is only legitimate for a routed
        // row. `entry_name` is the same function both emitters spell the
        // symbol with, so this compares like with like rather than
        // re-deriving the mangling here.
        let routed: Vec<String> = jit
            .iter()
            .map(|d| kernels_cuda_new::abi::entry_name(d.sig.symbol))
            .collect();
        let stray: Vec<String> = named(bindings)
            .into_iter()
            .filter(|s| !defined.contains(s) && !routed.contains(s))
            .collect();
        assert!(
            stray.is_empty(),
            "`launch_bindings.rs` declares {} entry point(s) that neither the shim defines \
             nor JIT_DISPATCHED explains: {}. A declaration like that compiles and fails at \
             link, which is how new-horizon.md §21.11 spent an afternoon.",
            stray.len(),
            stray.join(", ")
        );

        // A ROUTED ROW MUST HAVE SOMEWHERE TO GO.
        //
        // The two checks above look for a NAME with no definition. This one
        // looks for the opposite and worse shape: a symbol in
        // `JIT_DISPATCHED` that produced **no arm at all**.
        //
        // `emit_rust_dispatch` builds the JIT arm from the DEVICE row's
        // operands, and `continue`s the whole row if any one of them has no
        // `ArgValue` variant. That skip is silent and it is total: the row
        // gets no JIT arm, and `emit_c_shim` has already skipped its shim
        // entry because the same list named it. So the fire reaches a
        // hand-written arm that does not exist and is diagnosed as
        // `UnknownKernel` -- a lie about what went wrong, and the exact
        // failure §22.1 names as a "fire-time lie".
        //
        // Neither check above sees it, because the failure is an ABSENCE of a
        // name in both files at once. Only the list knows the row was meant
        // to be there.
        let armless: Vec<&str> = jit
            .iter()
            .map(|d| d.sig.symbol)
            .filter(|symbol| !dispatch.contains(&format!("\"{symbol}\",")))
            .collect();
        assert!(
            armless.is_empty(),
            "{} symbol(s) are in `JIT_DISPATCHED` and got no arm in `rust_dispatch.rs`: {}. \
             There are two ways to land here and both are fatal to a routed row. Either an \
             operand carries `Source::Unbound`, in which case `emit_rust_dispatch` skipped \
             the row WHOLE and a hand-written arm has been calling `ffi::pie_k_*` for it -- \
             so dropping the shim entry breaks that arm at LINK time; or a device operand \
             has no `ArgValue` variant, in which case only the JIT branch was skipped. \
             `emit_c_shim` has already dropped the shim entry either way, so the first \
             fails to link and the second reports `UnknownKernel`, blaming the statement \
             for a gap in the row. State the missing source or operand, or take the symbol \
             back out of the list.",
            armless.len(),
            armless.join(", ")
        );

        // A HAND-WRITTEN ARM IS A CONSUMER TOO.
        //
        // Everything above reads GENERATED text, and that is the blind spot:
        // `driver-cuda` also calls `ffi::pie_k_*` by hand, from arms that no
        // emitter knows about. A routed row loses its shim entry, so a hand
        // arm naming it stops linking -- and the check that would have caught
        // it reads `rust_dispatch.rs`, where the hand arm is not.
        //
        // §22.1 measured what that costs: 114 undefined symbols at once, and
        // `rust-lld`'s `--error-limit=20` reporting a fifth of them, so the
        // first four fixes each revealed twenty more.
        let hand = format!("{}/src", env!("CARGO_MANIFEST_DIR"));
        let mut by_hand: Vec<String> = Vec::new();
        for d in jit {
            let entry = kernels_cuda_new::abi::entry_name(d.sig.symbol);
            if grep_tree(Path::new(&hand), &entry) {
                by_hand.push(format!("{} (as `{entry}`)", d.sig.symbol));
            }
        }
        assert!(
            by_hand.is_empty(),
            "{} routed symbol(s) are still named by a HAND-WRITTEN arm under \
             `driver-cuda/src`: {}. Routing drops the shim entry these link against, and \
             no generated file mentions them -- so every check that reads generated text \
             passes and the build fails at link. Move the hand arm to the JIT path first, \
             then route.",
            by_hand.len(),
            by_hand.join(", ")
        );
    }

    /// The probe file holds, arm for arm, what the dispatcher holds for every
    /// ROUTED symbol.
    ///
    /// # What this is guarding, and why a comment would not do it
    ///
    /// The parity harness fires an unrouted row through the probe and calls
    /// the result "what routing will do". That claim is only worth something
    /// if the probe's arm IS the arm routing emits — same guard, same
    /// staging, same `jit_dims` call, same operand expressions in the same
    /// order. Both come out of one function in one mode-parameterised loop,
    /// which is the structural half of the argument; this is the measured
    /// half, and it is cheap because the two strings are right here.
    ///
    /// It can only be checked on the rows that are ALREADY routed — an
    /// unrouted row has no arm in the dispatcher to compare against, which is
    /// the whole reason the probe exists. So the check grows a row at a time
    /// with the routed set, and every row it can see is one the harness
    /// certified before the flip: if the two ever came apart, the harness
    /// would have been proving something about a string nothing runs.
    fn the_probe_is_the_arm_routing_emits(
        dispatch: &str,
        probe: &str,
        jit: &[&'static kernels_cuda_new::device::DeviceKernel],
    ) {
        let mut checked = 0usize;
        for row in jit {
            let symbol = row.sig.symbol;
            let (Some(shipped), Some(probed)) = (arm_of(dispatch, symbol), arm_of(probe, symbol))
            else {
                // A routed row with no arm in one of the two files is
                // `routed_rows_have_an_arm`'s failure and is reported there,
                // with the two causes named. Saying it twice here would put
                // the worse message first.
                continue;
            };
            assert!(
                shipped == probed,
                "`{symbol}`'s arm differs between the dispatcher and the parity probe. \
                 They are emitted by one function and must be one string, or the harness \
                 that fires a row through the probe before routing it is proving something \
                 about text the dispatcher does not contain.\n--- dispatcher\n{shipped}\n\
                 --- probe\n{probed}"
            );
            checked += 1;
        }
        // AND THE DENOMINATOR, because a comparison that found no pairs
        // passes for the same reason it would pass on two empty files --
        // §21's rule, and §39.2's instance of it, one layer down.
        assert!(
            checked > 0 || jit.is_empty(),
            "{} row(s) are routed and NONE of them was found in both generated files. \
             `arm_of` looks for a line starting `\"symbol\"`; if the emitter's arm \
             preamble changed shape, this check is now reading nothing and passing.",
            jit.len()
        );
    }

    /// One arm's text, from the line whose pattern names `symbol` to the
    /// closing brace at column zero.
    ///
    /// Textual, because the input is generated text and the alternative is
    /// re-deriving the arm here — which would compare the emitter against a
    /// second copy of itself. A pattern line is `"sym" => {` or
    /// `"sym" | "alias" if guard => {`, always at column zero, and the arm
    /// always closes with a `}` at column zero, because that is what
    /// `emit_rust_dispatch` writes.
    fn arm_of(text: &str, symbol: &str) -> Option<String> {
        let head = format!("\"{symbol}\"");
        let mut lines = text.lines().skip_while(|l| !l.starts_with(&head));
        let first = lines.next()?;
        let mut arm = String::from(first);
        for line in lines {
            arm.push('\n');
            arm.push_str(line);
            if line == "}" {
                return Some(arm);
            }
        }
        None
    }

    /// Whether any `.rs` under `dir` contains `needle`, comments included.
    ///
    /// Deliberately textual and deliberately over-eager: a mention in a
    /// comment is a false positive that costs one edit, and a missed call is
    /// a link error that costs an afternoon. The asymmetry is the whole
    /// design — §21's rule that *a textual gate names what it looks at* is
    /// satisfied by the panic message above, which says "named by", not
    /// "called by".
    fn grep_tree(dir: &Path, needle: &str) -> bool {
        let Ok(entries) = std::fs::read_dir(dir) else { return false };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                if grep_tree(&path, needle) {
                    return true;
                }
            } else if path.extension().is_some_and(|e| e == "rs")
                && std::fs::read_to_string(&path).is_ok_and(|t| t.contains(needle))
            {
                return true;
            }
        }
        false
    }

    /// The defined external symbols in a `!<arch>` archive, from its own
    /// symbol index.
    ///
    /// GNU `ar` writes the index as the first member, named `/` (or `/SYM64/`
    /// with 64-bit offsets): a big-endian count, that many offsets, then that
    /// many NUL-terminated names. Both `ar` and `llvm-ar` write one, and a
    /// missing one is a hard error rather than an empty set — a check that
    /// quietly passes because it read nothing is worse than no check.
    fn archive_defines(path: &Path) -> Vec<String> {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        assert!(
            bytes.starts_with(b"!<arch>\n"),
            "{} is not an ar archive",
            path.display()
        );
        let header = &bytes[8..];
        assert!(header.len() > 60, "{} is truncated", path.display());
        let name = String::from_utf8_lossy(&header[..16])
            .trim_end()
            .to_string();
        let size: usize = String::from_utf8_lossy(&header[48..58])
            .trim()
            .parse()
            .unwrap_or_else(|e| panic!("{}: bad member size: {e}", path.display()));
        let (word, is64) = match name.as_str() {
            "/" => (4usize, false),
            "/SYM64/" => (8usize, true),
            other => panic!(
                "{}'s first member is {other:?}, not a symbol index. The shim archive is \
                 built by `cc` through `ar`, which always writes one; without it this check \
                 would pass by reading nothing.",
                path.display()
            ),
        };
        let data = &header[60..60 + size];
        let count = if is64 {
            u64::from_be_bytes(data[..8].try_into().unwrap()) as usize
        } else {
            u32::from_be_bytes(data[..4].try_into().unwrap()) as usize
        };
        let mut out = Vec::with_capacity(count);
        let mut at = word * (count + 1);
        for _ in 0..count {
            let end = data[at..]
                .iter()
                .position(|&b| b == 0)
                .unwrap_or_else(|| panic!("{}: unterminated name in symbol index", path.display()));
            out.push(String::from_utf8_lossy(&data[at..at + end]).into_owned());
            at += end + 1;
        }
        out
    }

    pub fn build() {
        let out_dir = PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        let tables = tables();

        let bindings = kernels_cuda_new::abi::emit_rust_bindings(&tables);
        std::fs::write(out_dir.join("launch_bindings.rs"), &bindings).expect("write bindings");

        // The DISPATCH, from the same rows. `emit_rust_dispatch` is the
        // sibling of the C++ `emit_dispatch` the declared executor
        // includes — same table, same guards, different strings — and it
        // is generated here for the reason the bindings are: a second
        // hand-written switch over a table that already knows the answer
        // is the duplication this whole arc removes.
        // THE ROWS THAT GO THE OTHER WAY. A symbol with a `DeviceKernel`
        // is compiled by NVRTC at run time and fired from Rust, so its arm
        // calls `bind::jit::fire` instead of a `pie_k_*` entry -- which is
        // what leaves its `.cu` launcher with no consumer.
        let jit: Vec<&'static kernels_cuda_new::device::DeviceKernel> =
            kernels_cuda_new::device::jit_dispatched();
        routed_rows_are_hosted(&jit);
        let dispatch = kernels_cuda_new::abi::emit_rust_dispatch(&tables, &jit);
        std::fs::write(out_dir.join("rust_dispatch.rs"), &dispatch).expect("write dispatch");

        // AND THE SAME ARMS ONE STEP EARLY, for the harness that has to fire
        // a row both ways BEFORE it is routed.
        //
        // Written unconditionally and included only under `jit-parity`: a
        // string this build script already holds costs a `write` to produce
        // and nothing to leave on disk, while making the file's existence
        // depend on a feature would mean the check below only ran in the
        // builds that least need it.
        let hosted: Vec<&'static kernels_cuda_new::device::DeviceKernel> =
            kernels_cuda_new::unit::rows().collect();
        let probe = kernels_cuda_new::abi::emit_rust_dispatch_probe(&tables, &hosted);
        std::fs::write(out_dir.join("rust_dispatch_probe.rs"), &probe).expect("write probe");
        the_probe_is_the_arm_routing_emits(&dispatch, &probe, &jit);

        println!("cargo:rerun-if-env-changed=CUDA_HOME");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        let cuda_include = cuda_home().join("include");
        if !cuda_include.join("cuda_runtime.h").is_file() {
            panic!(
                "the `bridge` feature needs the CUDA toolkit headers, and \
                 {cuda_include:?} has no cuda_runtime.h. Set $CUDA_HOME/$CUDA_PATH \
                 or install the toolkit — or drop `bridge` for a toolkit-free build."
            );
        }

        // The supergraph's set-cond kernel, which is the ONE thing in this
        // crate that has to be device code: `cudaGraphSetConditional` is a
        // `__device__` function, so arming a conditional handle from inside a
        // graph — the whole point, since it is what removes the host
        // round-trip — cannot be spelled in Rust or in `.cpp`.
        //
        // Its own archive rather than a `.file` on the one above, because
        // that build is `cpp(true)` and this needs nvcc. It goes in
        // `driver-cuda` rather than `kernels-cuda` because its argument
        // is a conditional handle — a shell object — not a tensor; see
        // `src/device/graph.rs`'s header for the same argument.
        let supergraph = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/supergraph.cu");
        println!("cargo:rerun-if-changed=csrc/supergraph.cu");
        cc::Build::new()
            .cuda(true)
            .include(&cuda_include)
            .file(&supergraph)
            .compile("pie_supergraph");

        // THE THREE MULTIMODAL TOWERS, moved here out of `kernels-cuda`.
        //
        // Same argument as the supergraph above, and it is measured rather
        // than asserted. A tower's `.cu` includes exactly two kinds of thing:
        // `.cuh` device headers, and `.hpp` host declarations. EVERY `.cuh`
        // the three towers include -- `vision/gemma4_vision.cuh`,
        // `gemma4_audio.cuh`, `gemma4_naive_kernels.cuh`,
        // `qwen3_vl_tower.cuh`, `tower_naive_kernels.cuh`, plus
        // `norm/rmsnorm.cuh`, `norm/elementwise.cuh`, `mlp/swiglu.cuh`,
        // `ssm/causal_conv1d.cuh` -- already lived in the JIT tree, reached
        // through `-iquote`. None of them was ever in the archive. So what
        // `kernels-cuda` compiled for a tower was never device code: it was a
        // HOST WALK over device code that belongs to `kernels-cuda-new`, and
        // a host walk with a `cudaStream_t` in its signature is this crate's
        // kind of object, not that one's.
        //
        // The walk is also not expressible as a kernel row. `Execution` has
        // `Jit | Composed | Service`, and `Composed` is the near miss: it
        // carries a `&'static [Step]`, a list fixed at compile time, and a
        // tower is a data-dependent loop -- `for im in 0..num_images`, a
        // per-layer body, and host-side position-embedding interpolation
        // computed BETWEEN launches from a grid size only known at call
        // time. A static step list cannot say that. The rows therefore stay
        // `driver_internal` with `whole = true`, which is also what keeps
        // the owner's constraint true: `model-compiler` cannot name these
        // symbols at all, so it cannot tell what is behind them.
        //
        // The five `vision/*.hpp` stay in `kernels-cuda/csrc/src/vision/`.
        // They are the launch shim's compile-time contract -- the shim is
        // generated over there and forwards `pie_k_vision_*` into
        // `kernels::vision::*` -- and a contract is a header. The
        // DEFINITIONS are here, so the final link resolves shim -> towers,
        // which is why `pie_vision_towers` is emitted between the shim and
        // the kernel archive in the `-l` order below.
        //
        // Toolkit-free builds are unaffected: this is inside `bridge`, and
        // `cargo check -p driver-cuda` with no features never reaches it.
        let towers_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("csrc/vision");
        let archive_src = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kernels-cuda/csrc/src");
        let jit_headers = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kernels-cuda-new/csrc/src");
        println!("cargo:rerun-if-changed=csrc/vision");
        let mut towers = cc::Build::new();
        towers
            .cuda(true)
            .std("c++17")
            .include(&cuda_include)
            // `vision/*.hpp`, and the two host headers a tower still calls
            // into: `gemm/gemm.hpp` (cuBLAS) and `attn/*.hpp` (FlashInfer).
            .include(&archive_src)
            // `-iquote`, NOT `-I`, and for the reason `kernels-cuda`'s
            // CMakeLists spells out at length: this tree carries shims
            // wearing NVIDIA's filenames (`cuda_bf16.h`) so NVRTC has
            // something to answer with, and an `-I` would let one shadow the
            // real toolkit header. `-iquote` answers only `#include "..."`.
            .flag(&format!("-Xcompiler=-iquote,{}", jit_headers.display()))
            .flag("-gencode")
            .flag("arch=compute_89,code=sm_89")
            .flag("--expt-relaxed-constexpr")
            .flag("--extended-lambda")
            // The `-l` is emitted BY HAND below, after the shim's, because
            // `cc` would print it here and a static archive is scanned in
            // place: the shim's `pie_k_vision_*` bodies CALL into these
            // objects, so the caller has to precede the callee. Printing it
            // at this point in the file would put the towers ahead of the
            // shim and every tower symbol would be undefined.
            .cargo_metadata(false)
            .warnings(false);
        let listing = std::fs::read_dir(&towers_dir).unwrap_or_else(|e| {
            panic!(
                "{towers_dir:?} does not read ({e}), and the `-l static=pie_vision_towers` \
                 below names its archive unconditionally. The towers moved here out of \
                 `kernels-cuda`; if that move is being reverted, the `-l` goes with the \
                 directory."
            )
        });
        let mut units = 0usize;
        for entry in listing {
            let path = entry.expect("csrc/vision entry").path();
            match path.extension().and_then(|e| e.to_str()) {
                Some("cu") | Some("cpp") => {
                    towers.file(&path);
                    units += 1;
                }
                _ => {}
            }
        }
        // The floor 1832b170f argues for, applied to the scan two hundred
        // lines from the one it fixed. This is a `read_dir` over a directory
        // whose contents are landing in another agent's commits, and its
        // result is an archive NAMED on the link line: zero translation units
        // still produces a `libpie_vision_towers.a`, still satisfies the
        // `-l`, and defines nothing. The failure then surfaces at the final
        // link as undefined `kernels::vision::*` — which `rust-lld` reports
        // twenty at a time under `--error-limit=20`, so a rename that drops
        // one file reads as a fifth of a symbol list attached to whatever
        // test target happened to link first. A count is the only thing
        // between those two readings.
        //
        // The floor is 1 and not the 5 or 6 files the move will settle on,
        // deliberately: a literal count would be a second place to edit
        // every time a tower is split or `vis_helpers.cpp` is deleted, and a
        // stale floor fails the build for a file that was correctly removed.
        // What is not allowed to be true is that the scan found NOTHING and
        // said so by succeeding.
        assert!(
            units > 0,
            "{towers_dir:?} holds no .cu or .cpp, so `pie_vision_towers` would be an empty \
             archive that the `-l` below still names and every `kernels::vision::*` the \
             launch shim forwards into would be undefined at the final link, reported \
             twenty at a time. Either the tower sources have not landed yet, or their \
             extensions changed and this scan no longer recognises them."
        );
        towers.compile("pie_vision_towers");
        let towers_out = PathBuf::from(std::env::var_os("OUT_DIR").expect("OUT_DIR"));

        // THE SHIM, which is what `rust_dispatch.rs` above actually calls.
        //
        // `kernels-cuda` COMPILES `libpie_launch_shim.a` (one definition, in
        // the crate that owns the launchers it forwards into) and does not
        // link it, exactly as it compiles `libpie_kernels_cuda.a` and does not
        // link that. Both `-l`s are ours because we are what references them,
        // and this one is FIRST because the shim's bodies call the launchers:
        // a static archive is scanned in place, so the caller precedes the
        // callee.
        //
        // This is a fix, not a tidy-up. Until this round the shim's directive
        // was `cc`'s default `cargo:rustc-link-lib=static=pie_launch_shim`
        // over in `kernels-cuda`, and nothing linked it: cargo hands a build
        // script's `-l` only to its own package's lib, `static=` defaults to
        // `+bundle` so rustc puts the objects inside `libkernels_cuda.rlib`
        // rather than re-emitting a `-l`, and since §21.8 no crate names
        // `kernels_cuda::` — so rustc never loads that `--extern` and its rlib
        // is not among the 118 on our link line. Every `pie_k_*` this
        // dispatch calls was undefined; `rust-lld`'s `--error-limit=20` is the
        // only reason it read as seven.
        //
        // The directory comes through `kernels-cuda`'s `links` key rather than
        // from cargo's implicit `-L` propagation. That propagation is real and
        // did work, but an archive this crate NAMES is one it should be able
        // to point at.
        let shim_dir = PathBuf::from(std::env::var_os("DEP_PIE_KERNELS_CUDA_LAUNCH_SHIM").expect(
            "`bridge` is on but kernels-cuda published no `launch_shim` path. That key \
                 is printed by its `shim()` under the `native` feature, which `bridge` \
                 turns on — so this means the two features have come apart.",
        ));
        let shim_archive = shim_dir.join("libpie_launch_shim.a");
        assert!(
            shim_archive.is_file(),
            "kernels-cuda published {shim_dir:?} as the launch shim's directory and there is \
             no libpie_launch_shim.a in it. `rust_dispatch.rs` calls `pie_k_*` entry points \
             that only that archive defines, so linking would fail one test target at a time \
             with a symbol list instead of failing here with a reason."
        );
        println!("cargo:rerun-if-changed={}", shim_archive.display());
        every_call_resolves_in_the_shim(&shim_archive, &dispatch, &bindings, &jit);
        println!("cargo:rustc-link-search=native={}", shim_dir.display());
        println!("cargo:rustc-link-lib=static=pie_launch_shim");

        // ...and the towers immediately after it, before the kernel archive.
        // The shim forwards `pie_k_vision_*` into `kernels::vision::*`, whose
        // definitions are in `libpie_vision_towers.a`; those in turn still
        // call `gemm::` and the FlashInfer wrappers, which are in
        // `libpie_kernels_cuda.a`. Caller before callee, twice.
        println!("cargo:rustc-link-search=native={}", towers_out.display());
        println!("cargo:rustc-link-lib=static=pie_vision_towers");

        // The kernels archive the shim forwards into. Search paths come from
        // `kernels-cuda`'s own build script (the `native` feature this
        // crate's `bridge` turns on); the `-l` is ours so it lands AFTER the
        // shim's.
        println!("cargo:rustc-link-lib=static=pie_kernels_cuda");

        // The archive's own closure, `driver-cuda/build.rs`'s list minus
        // nvrtc (the NVRTC JIT is pipeline code, which stayed C++): dynamic
        // cudart + cublas + cublasLt, the driver-API stub for `cuMem*`, NCCL
        // for the custom all-reduce, and the C++ runtime.
        let cuda_lib = cuda_home().join("lib64");
        println!("cargo:rustc-link-search=native={}", cuda_lib.display());
        for lib in ["cudart", "cublas", "cublasLt"] {
            println!("cargo:rustc-link-lib={lib}");
        }
        let stubs = cuda_lib.join("stubs");
        if stubs.is_dir() {
            println!("cargo:rustc-link-search=native={}", stubs.display());
        }
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=nccl");
        for lib in ["stdc++", "pthread", "m", "dl", "rt"] {
            println!("cargo:rustc-link-lib={lib}");
        }
    }
}
