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
            // `rope`'s twelve, from `x::rope::SIGS` — the same rows, derived
            // from the `contract!` block in `x::rope` rather than
            // written out by hand. They state no `operands`, so the shim
            // emitter drops them (`abi::stated`) and the dispatch emitter
            // writes no arm for them; they are here so that this reader
            // still walks every symbol the crate declares, which is what
            // `armless` and `by_hand` below check.
            kernels_cuda_new::x::rope::SIGS,
            // `norm`'s twenty-eight, from `x::norm::SIGS`, for
            // `x::rope::SIGS`' reason above. `table::norm::KERNELS` stood
            // here; §5 step 5 took `norm` into fn-world and deleted it. The
            // count went 26 -> 28 in the move: `norm::add_bias_bf16` and
            // `norm::rmsnorm_gated_fp32_in_bf16` are lowered
            // (`model-compiler/src/lower.rs` `OpKind::AddBias` and
            // `OpKind::RmsnormGated`) and had no ahead-of-time row, so they
            // reached this reader through `families::norm`'s JIT rows only.
            kernels_cuda_new::x::norm::SIGS,
            // `mlp`'s twelve, from `x::mlp::SIGS`, for `x::rope::SIGS`'
            // reason above.
            kernels_cuda_new::x::mlp::SIGS,
            // `gemm`'s twelve, from `x::gemm::SIGS`, for `x::rope::SIGS`'
            // reason above: derived from the `contract!` block, stating no
            // `operands`, so the shim emitter drops them and the dispatch
            // emitter writes no arm — and they are still walked here.
            kernels_cuda_new::x::gemm::SIGS,
            kernels_cuda_new::table::moe::KERNELS,
            // `ssm`'s twenty-seven, from `x::ssm::SIGS`, for `x::rope::SIGS`'
            // reason above. `table::ssm::KERNELS` stood here; §5 step 5 took
            // `ssm`'s five roots into fn-world and deleted it. The count is
            // unchanged at twenty-seven — four of them are `bind!` `none:`
            // arms, which are contracts all the same and must still be walked
            // or `check_plan` would stop refusing symbols they declare.
            kernels_cuda_new::x::ssm::SIGS,
            // `quant`'s eleven, from `x::quant::SIGS`, for `x::rope::SIGS`'
            // reason above. The four routed MoE decode GEMVs that carry a
            // `quant::` symbol are NOT here — their contracts are
            // `table::moe`'s, which this list already walks, and they still
            // state operands and still get an arm.
            kernels_cuda_new::x::quant::SIGS,
            // `layout`'s seven and `sample`'s one, from their `contract!`
            // blocks, for `x::rope::SIGS`' reason above. `adapter`'s one is
            // here for that reason and one more: it never had a `pie_k_`
            // entry point at all — the LoRA seam is cuBLAS batched GEMMs —
            // so it was always a row the shim emitter had to skip, and now
            // it is a row that states no operands and is skipped by the
            // general mechanism instead of by a special case.
            kernels_cuda_new::x::layout::SIGS,
            kernels_cuda_new::x::sample::SIGS,
            kernels_cuda_new::x::adapter::SIGS,
            // THE SECOND TABLE IS GONE. It was
            // `kernels_cuda_new::table::driver_internal::DRIVER_KERNELS` —
            // launchers the driver fires with no DSL statement, outside the
            // DSL-surface equality — and §5 step 5 took its six rows to
            // `x::driver_internal` as plain `fn`s with **no `contract!`**.
            // There is nothing to append: no rows, and no `SIGS` either,
            // because a family with no contract declares no signatures. The
            // launchers are called directly and need no entry point.
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
        let Ok(entries) = std::fs::read_dir(dir) else {
            return false;
        };
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

        // THE THREE MULTIMODAL TOWERS ARE GONE, and nvcc with them.
        //
        // This block used to compile `csrc/vision/` into
        // `libpie_vision_towers.a`. There is no `csrc/vision/` — all three
        // host walks are Rust, at `src/tower/gemma4_vision.rs`,
        // `.../gemma4_audio.rs` and `.../qwen3_vl.rs`, firing JIT rows one at
        // a time. What is kept here is the measurement that made the move
        // possible, because the FA2 block below still cites it ("for the
        // towers' reason"):
        //
        // A tower's `.cu` included exactly two kinds of thing: `.cuh` device
        // headers, and `.hpp` host declarations. EVERY `.cuh` the three
        // included -- `vision/gemma4_vision.cuh`, `gemma4_audio.cuh`,
        // `gemma4_naive_kernels.cuh`, `qwen3_vl_tower.cuh`,
        // `tower_naive_kernels.cuh`, plus `norm/rmsnorm.cuh`,
        // `norm/elementwise.cuh`, `mlp/swiglu.cuh`, `ssm/causal_conv1d.cuh`
        // -- already lived in the JIT tree, reached through `-iquote`. None
        // was ever in the archive. So what was being compiled for a tower was
        // never device code: it was a HOST WALK over device code that belongs
        // to `kernels-cuda-new`. `qwen3_vl_tower.cu` was the last of the
        // three and made the point without argument -- **zero `__global__`
        // and sixteen `<<<>>>`**, a host program wearing a `.cu` extension
        // because `<<<>>>` needs nvcc to parse.
        //
        // The walk was also never expressible as a kernel row, which is what
        // kept it out of `Execution::Composed` for as long as it was C++:
        // `Composed` carries a `&'static [Step]`, a list fixed at compile
        // time, and a tower is a data-dependent loop -- `for im in
        // 0..num_images`, a per-layer body whose depth comes from the
        // checkpoint, and host-side position-embedding interpolation computed
        // BETWEEN launches from a grid size known only at call time. A static
        // step list cannot say that. A Rust function can, and that is what
        // each of them now is.
        //
        // The five `vision/*.hpp` stay in `kernels-cuda/csrc/src/vision/`:
        // `kernels-cuda/build.rs::includes()` scans that directory wholesale,
        // and a header declaring a function nothing calls costs a parse. The
        // shim no longer emits `pie_k_vision_gemma4_*` or
        // `pie_k_vision_qwen3vl_scatter` at all -- those rows left
        // `driver_internal.rs`, and `abi::emit_c_shim` emits one entry per
        // stated row.
        //
        // THE THREE SURVIVING VARIABLES ARE GONE TOO. `archive_src`,
        // `jit_headers` and `jit_shims` were kept past the towers' deletion
        // with the note *"Three variables survive the deletion because the
        // FA2 block below uses all three"*, and the FA2 block below is now
        // deleted as well. There is no `cc::Build` left in this script, so
        // there is no include path to name and nothing for the
        // `jit_shims/cuda_fp16.h` assertion to protect: that assertion
        // guarded a C++ TU from silently resolving `"cuda_fp16.h"` to the
        // toolkit's header, and this crate compiles no C++ TU.
        //
        // The measurement it carried is not lost, it has moved to where the
        // risk moved: NVRTC is what reads `csrc/shim` now, and
        // `kernels-cuda-new`'s own build is what asserts the impersonating
        // headers are present. `csrc/shim/cuda/cmath:245-280` in particular
        // is the file recording that this shim's `__fast_div_modulo` is
        // `{u32 @0, u64 @8}` align 8 against CCCL's `{u32,u32,u32,i32}`
        // align 4, which put `paged_kv_t::num_heads` at +24 under the shim
        // and +20 under CCCL with `sizeof` reconverging at 96.
        //
        // THAT HAZARD IS WHAT THIS DELETION ENDS, and it is the reason the
        // `.cu` and this block had to go in the same pass.
        // `kernels-cuda-new/src/fa2/params.rs` pins its mirror to the SHIM's
        // layout, which is right for every JIT fire;
        // `attention_flashinfer.cu` compiled against real CCCL and filled the
        // +20 layout. Both were correct and neither could ever read the
        // other's block — but only because nothing happened to hand one
        // across. With the C++ gone there is exactly one layout in the
        // process and the question cannot be asked again.
        //
        // The two `rerun-if-changed` lines for the JIT trees stay: this
        // script does not compile them, but `kernels-cuda-new`'s generated
        // dispatch and bindings are re-emitted from tables that describe
        // them, and a source edit there still has to invalidate this crate.
        println!("cargo:rerun-if-changed=../kernels-cuda-new/csrc/src");
        println!("cargo:rerun-if-changed=../kernels-cuda-new/csrc/shim");

        // THE LAST `.cuda(true)` IN THIS BUILD SCRIPT WAS HERE, AND IT IS
        // GONE. North star §5 step 7; `families/fa2.rs`' header holds the
        // deletion list this hunk was cut against.
        //
        // WHAT WENT, item by item, because a deletion is a claim about a
        // whole consumer set:
        //
        //  * The `cc::Build` (`.cuda(true).std("c++17")`), its four include
        //    groups, the `-gencode arch=compute_89,code=sm_89`, the
        //    `--extended-lambda` / `--expt-relaxed-constexpr` pair, the two
        //    `-Xcompiler=-iquote` flags, `.cargo_metadata(false)`, the
        //    `read_dir` scan of `csrc/attn`, the `attn_units > 0` floor and
        //    `fa2.compile("pie_attn_flashinfer")`.
        //  * The two `DEP_PIE_KERNELS_CUDA_{FLASHINFER,CCCL}` `expect`s that
        //    fed it.
        //  * The hand-printed `rustc-link-search` / `rustc-link-lib=
        //    static=pie_attn_flashinfer` pair below. Those had to be deleted
        //    BY HAND and were not implied by dropping the `cc::Build`,
        //    because `.cargo_metadata(false)` meant this target printed its
        //    own link lines instead of letting `cc` do it.
        //  * The four-point "why nvcc and not NVRTC" argument. Points 1, 3
        //    and 4 were about `attention_flashinfer.cu` specifically and die
        //    with it; the two that are still true about OTHER text are
        //    repointed by name below rather than deleted, because they were
        //    never claims about this file alone.
        //
        // WHAT MADE IT DELETABLE: `csrc/attn/attention_flashinfer.cu` (1,258
        // lines) and `csrc/attn/plan_lifecycle.cpp` (105) held `__global__`
        // 0, `__device__` 0 and exactly one `<<<>>>` — `attn_score_fold_heads`
        // launching `device::attn_score_fold_heads`, which is ours, already
        // rowed (`attn::attn_score_fold_heads`) and already fired from Rust
        // by `fire/attn_score.rs`. The four dispatches and the two planners
        // are `bind::service::attn_*` and `fire::flashinfer_fa2*`; the six
        // rows are on `kernels_cuda_new::execution::RUST_SERVED`, so
        // `emit_c_shim` emits no forwarder for them and no `-l` has to
        // resolve one. `plan_lifecycle.cpp`'s own header said it existed for
        // *"a `unique_ptr` with a custom deleter"*; the replacement caches are
        // plain Rust structs whose deleter is `Drop`. `csrc/` is now empty
        // and gone, and the driver's `<<<>>>` census is ZERO.
        //
        // WHAT IS STILL TRUE AND WHERE IT NOW POINTS. The old point 2 quoted
        // `kernels-cuda-new/src/plan/mod.rs` — *"the only remaining reason
        // `driver-cuda` compiles C++ for attention is the kernels themselves
        // ... §13.6 prices that separately"* — and that price is exactly what
        // `families/fa2.rs` has now paid for FA2: 56 units, 460 rows, one
        // root, all NVRTC. It has NOT been paid for **FA3 (the SM90 Hopper
        // prefill)** or for **MLA**, whose headers are CPM-only —
        // `csrc/vendor` carries no `attention/hopper/` at all — so NVRTC
        // cannot see them and `Headers::LibraryAndVendor` cannot be widened
        // to reach them. Old point 3's *"`cascade.cuh` does not generalise to
        // `prefill.cuh`"* survives as the same warning about those two: the
        // FA2 port is not evidence that the next one is cheap.
        //
        // The sm_89-only `gencode` and its named regression against the
        // archive build are NOT lost with this hunk. They are recorded in
        // `fire/flashinfer_fa2.rs`' header, together with the reason the
        // deletion RECOVERS the coverage rather than merely dropping the
        // claim: NVRTC compiles for the device it is loaded on, so a machine
        // that is not sm_89 stops being excluded. §44.7's rule — every sm_90
        // claim in this migration is argued from the call graph and none from
        // a run — is recorded in the same place and still binds.

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

        // `libpie_attn_flashinfer.a` USED TO BE NAMED HERE, immediately after
        // the shim and before the kernel archive, by a hand-printed
        // `rustc-link-search` / `rustc-link-lib=static=pie_attn_flashinfer`
        // pair. Both lines are deleted with the `cc::Build` that produced the
        // archive — and they had to be deleted explicitly, because that
        // target set `.cargo_metadata(false)` and printed its own directives
        // rather than letting `cc` emit them.
        //
        // The edge they ordered is gone rather than reordered. It was:
        //
        //   shim  -> `kernels::attn::dispatch_attention_flashinfer_*`
        //   there -> `AttnHd<HD>::*`, instantiated by the four
        //            `attention_flashinfer_hd<N>.cu` units       (archive)
        //
        // The six FlashInfer FA2 rows are on `execution::RUST_SERVED`, so
        // `emit_c_shim` no longer emits the forwarders that were the first
        // edge's tail, and there is nothing left to name the second edge's
        // head. `every_call_resolves_in_the_shim` above checks only the calls
        // the GENERATED dispatch makes, so it is satisfied by the same fact.

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
