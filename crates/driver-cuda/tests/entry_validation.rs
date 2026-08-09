//! No entry point reads a descriptor without validating it.
//!
//! North star rule 4: *a shared capability must not be optional — if it
//! can be skipped, it will be.* `driver-api` ships seventeen `validate_*`
//! functions. `driver-dummy`, the reference implementation of this
//! contract, calls them; this shell called NONE and re-derived similar
//! checks by hand at 51 sites. The capability was built, shipped, and
//! routed around.
//!
//! `serve::checked` makes the dereference and the validation one
//! operation — there is no way to obtain a `&PieKvCopyDesc` without
//! having run a validator over it, because the only thing that turns the
//! pointer into a reference takes the validator as an argument. This test
//! holds the other half: that nothing goes around it.
//!
//! ## Why a source scan, and what it is worth
//!
//! The rule is "every raw descriptor pointer in this file is dereferenced
//! through `checked`", and the compiler cannot state it — `as_ref` is
//! inherent on every raw pointer and always will be. So the check reads
//! the source, which is the same trade `executor_bind`'s arm scan makes
//! and for the same reason: a narrower question than the one the code
//! answers, aimed at the failure that actually keeps happening.
//!
//! The failure it catches is a NEW entry point written the old way. Every
//! one of the eleven took its descriptor with a bare
//! `unsafe { p.as_ref() }`, because that is what the one above it did.

#![cfg(feature = "abi")]

use std::collections::BTreeSet;

/// The shell's source, read once.
///
/// EVERY module of it, concatenated. The shell was one file when this
/// test was written, became six, and is now TWO DIRECTORIES —
/// `serve/` holds the doors and `fire/` holds the pass they open
/// onto. Reading only the one that happens to hold a door today would
/// let a new entry point written the old way land next door and pass.
/// The guarantee is about the whole shell, so the source is.
///
/// This test has been broken by a move twice. Both times it kept
/// passing, because a shrinking corpus finds fewer violations rather
/// than more — which is why the emptiness assertions below are not
/// decoration.
fn shell_source() -> String {
    let dirs = [
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/serve"),
        concat!(env!("CARGO_MANIFEST_DIR"), "/src/fire"),
    ];
    let mut files: Vec<_> = dirs
        .iter()
        .flat_map(|dir| {
            std::fs::read_dir(dir)
                .expect("the shell's source is beside this test")
                .filter_map(|e| e.ok().map(|e| e.path()))
        })
        .filter(|p| p.extension().is_some_and(|x| x == "rs"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "the shell has modules");
    files
        .iter()
        .map(|p| std::fs::read_to_string(p).expect("a shell module reads"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Every `unsafe { <ident>.as_ref() }` in the shell, by the name it
/// dereferences.
fn bare_derefs(src: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for line in src.lines() {
        let Some(at) = line.find("unsafe { ") else {
            continue;
        };
        let rest = &line[at + "unsafe { ".len()..];
        let Some(end) = rest.find(".as_ref() }") else {
            continue;
        };
        let name = rest[..end].trim();
        if !name.is_empty() && name.chars().all(|c| c.is_alphanumeric() || c == '_') {
            out.insert(name.to_string());
        }
    }
    out
}

/// A descriptor reaches an entry point through `checked`, or not at all.
///
/// The allowed set is ONE name and it is `checked`'s own — the function
/// whose whole job is to pair the dereference with the validator. A
/// second name here is an entry point that took its descriptor raw.
#[test]
fn every_descriptor_is_dereferenced_through_its_validator() {
    let src = shell_source();

    // SELF-VACUITY, and it is the same shape as the arm scan's: a
    // scanner that stopped recognising the pattern would find nothing
    // and report a clean file. `checked` is the one site that must
    // always match, because it is the site the pattern exists to allow.
    let found = bare_derefs(&src);
    assert!(
        !found.is_empty(),
        "the scan found no `unsafe {{ x.as_ref() }}` at all, so its shape \
         assumption broke rather than the shell being clean"
    );

    // `p` is `checked`'s parameter. Nothing else may name one.
    let allowed: BTreeSet<String> = ["p"].iter().map(|s| (*s).to_string()).collect();
    let bare: BTreeSet<&String> = found.difference(&allowed).collect();
    assert!(
        bare.is_empty(),
        "these dereference a descriptor pointer WITHOUT its validator: {bare:?}\n\
         Use `checked(ptr, driver_api::local::validate_*, \"<entry>\")`, which \
         pairs the null test and the validation so neither can be forgotten. \
         If the descriptor has no validator, say so at the call with a \
         closure and a reason, as `launch` and `encode` do."
    );
}

/// The entry points that take a descriptor all call `checked`.
///
/// The scan above proves nothing goes AROUND it; this proves the calls
/// are there at all, so that deleting one is a failure rather than a
/// silent return to the old shape.
#[test]
fn every_descriptor_entry_point_calls_checked() {
    let src = shell_source();
    let calls = src.matches("checked(").count();
    // Eleven entry points take a descriptor; `checked(` also appears in
    // its own definition, so the floor is the count of entry points.
    assert!(
        calls >= 10,
        "only {calls} `checked(` sites — an entry point stopped validating \
         its descriptor, or the helper was renamed and this floor was not"
    );
}

/// EVERY entry point that takes a descriptor validates it.
///
/// This test used to hold the opposite: that `validate_frame_desc` and
/// `validate_encode_desc` stayed DEFERRED, because their rules were right
/// and the callers were short. They are not short any more, so the claim
/// inverts — and the inversion is the point of having written it as a
/// test rather than a note.
///
/// What the two of them caught, all of it in fixtures that had passed for
/// as long as they existed:
///
/// * `roster_rows` stated one entry per TOKEN of a prefill, every one
///   zero — which the validator reads as N requests all claiming roster
///   index 0. The engine builds it with
///   `Vec::with_capacity(instance_ids.len())`, one per REQUEST.
/// * `sub_batch_indptr` partitioned tokens where it partitions roster
///   rows.
/// * steps with no terminal cell at all, and then two steps SHARING one —
///   which would have had the frame report whichever finished last as the
///   answer for both.
/// * `rs_slot_ids` with no matching `rs_slot_flags`: a recurrent slot the
///   driver cannot know whether to reset or continue.
/// * encode descriptors with no `image_grids`, which the engine sends
///   (`engine/src/driver/abi.rs:323`) and this shell never reads.
///
/// Not one of those was reachable through the engine, because the engine
/// builds these correctly. They were reachable through the tests, which
/// is the point of `validators-unskippable`: a test fixture is a caller,
/// and an unchecked contract lets it drift exactly as far as any other.
#[test]
fn no_validator_is_deferred() {
    let src = shell_source();
    for validator in [
        "validate_driver_create_desc",
        "validate_model_load_desc",
        "validate_program_desc",
        "validate_channel_desc",
        "validate_instance_desc",
        "validate_frame_desc",
        "validate_encode_desc",
        "validate_kv_copy_desc",
        "validate_state_copy_desc",
        "validate_pool_resize_desc",
    ] {
        // The NAME, not a call: a validator reaches `checked` as a value
        // (`checked(p, local::validate_x, "…")`) or inside a closure when
        // it is `unsafe`, so matching an open paren would find only half
        // of them and report the other half as missing.
        let named = format!("local::{validator}");
        assert!(
            src.contains(&named),
            "`{validator}` is no longer called — an entry point stopped \
             validating its descriptor, and the shared checks it names are \
             the ones this shell used to hand-roll at 51 sites"
        );
    }
}

/// Every ABI entry point catches its own panics.
///
/// The same rule as the validators, and the same failure: `guard` was
/// built to turn a panic into a failed REQUEST rather than a dead
/// process, and eleven of the thirteen entries used it. `create` and
/// `destroy` did not — so a panic while parsing the boot TOML, or while
/// draining a fire on the way out, took the whole engine down along with
/// every other request it was serving.
///
/// Two out of thirteen is what an unenforced convention decays to. This
/// makes the fourteenth entry point's author find out at CI time.
///
/// # Why the entry list is derived rather than written down
///
/// A hand-kept list is the same shape of debt: someone adds
/// `pie_cuda_something` and does not add it here, and the test passes
/// while the hole exists. So the entries come from the source, and the
/// only way to be exempt is to not be an entry point.
#[test]
fn no_entry_point_lets_a_panic_out() {
    let src = shell_source();
    let lines: Vec<&str> = src.lines().collect();

    let mut entries = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let Some(rest) = line.strip_prefix("pub fn pie_cuda_") else { continue };
        let name = rest.split('(').next().unwrap_or("").to_string();
        // The signature runs to the brace; the body starts after it.
        let body_at = (i..lines.len().min(i + 30))
            .find(|&j| lines[j].contains('{'))
            .map_or(i, |j| j + 1);
        // `guard` is the FIRST thing the body does or it is not a guard —
        // work above it is work outside the catch.
        let guarded = lines[body_at..lines.len().min(body_at + 6)]
            .iter()
            .any(|l| l.contains("guard(\""));
        entries.push((name, guarded));
    }

    assert!(
        entries.len() >= 13,
        "the entry scan found {} entries, so it stopped recognising them \
         and would pass while every one of them was unguarded: {entries:?}",
        entries.len()
    );

    let naked: Vec<&str> =
        entries.iter().filter(|(_, g)| !*g).map(|(n, _)| n.as_str()).collect();
    assert!(
        naked.is_empty(),
        "these entry points let a panic reach the caller: {naked:?}. \
         `serve::guard` turns it into a failed request instead; a \
         panic that escapes an `extern \"C\"` boundary is undefined \
         behaviour, and even where it is not, it kills every other \
         request the engine was serving"
    );
}
