//! No entry point reads a plan without validating it, and none lets a
//! panic out.
//!
//! North star rule 4: *a shared capability must not be optional -- if it
//! can be skipped, it will be.* `driver-api` shipped seventeen `validate_*`
//! functions and nothing called them: this shell called NONE and re-derived
//! similar checks by hand at 51 sites, and the one caller that did --
//! `driver-dummy`, the interpreter backend -- is deleted. The capability was
//! built, shipped, and routed around by every implementation there was.
//!
//! ## What changed, and what that costs this file
//!
//! Half of that is now settled, and not by anyone remembering to call
//! anything. The entry points stopped being `extern "C"` (see
//! `serve::guard`'s header) and take typed references and owned plans, so
//! the descriptors that needed a `validate_*_desc` -- the null test, the
//! ptr/len agreement -- do not arrive as pointers to be checked. Three
//! tests here scanned for that mechanism, and when it went they found ZERO
//! `checked(` sites and ZERO validators and FAILED, rather than passing over
//! an empty corpus. That is the only reason this file could be repaired
//! instead of quietly believed; see `no_entry_point_takes_a_raw_descriptor`,
//! which is what replaced them, and which holds the door shut behind them.
//!
//! What did NOT go away is the half the type system cannot state: the
//! frame's roster bounds, the copy's memory domains, the encode plan's CSR
//! partitioning. Those are `plan.validate()` calls that a new entry point
//! can simply not make, and the three `*_validates_its_*` tests below run
//! each one against a plan that is wrong in exactly that way.
//!
//! ## Why source scans, and what they are worth
//!
//! Two of these tests read the source, because the compiler cannot say
//! "every method that answers a status catches its own panics" -- the same
//! trade `executor_bind`'s arm scan makes, aimed at the failure that keeps
//! happening: a NEW entry point written like the one above it.
//!
//! A source scan's whole risk is finding nothing and reporting success. This
//! test has been broken by a move twice, and both times it kept passing,
//! because a shrinking corpus finds fewer violations rather than more. So
//! every scan below derives its own subjects from the source and asserts a
//! floor on how many it found. The lists are never hand-kept: a hand-kept
//! list is the same debt in a different place.

#![cfg(feature = "abi")]

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
fn shell_modules() -> Vec<String> {
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
        .collect()
}

/// The same modules, concatenated, for the checks that ask "anywhere".
fn shell_source() -> String {
    shell_modules().join("\n")
}

/// Every method of `impl Shell`, with its signature and its body.
///
/// The entry points are derived, never listed: someone adds a verb and does
/// not add it here is the same debt the validators decayed into.
fn shell_entries() -> Vec<(String, String, String)> {
    // Scoped to the `impl Shell` blocks, not to the whole shell source.
    // `fire/` is full of `*const c_void` -- device buffers, which are raw
    // because a device address IS raw -- and counting those as entry points
    // taking a descriptor would make the test below fire on the wrong thing
    // and then be relaxed until it fired on nothing.
    let mut out = Vec::new();
    for src in shell_modules() {
        let Some(at) = src.find("\nimpl Shell {") else {
            continue;
        };
        // To the `}` in column zero that closes the impl.
        let block = &src[at..];
        let block = &block[..block[1..].find("\n}").map_or(block.len(), |e| e + 2)];
        let mut rest = block;
        while let Some(at) = rest.find("\n    pub fn ") {
            let after = &rest[at + "\n    pub fn ".len()..];
            let name: String = after.chars().take_while(|c| *c != '(').collect();
            let Some(open) = after.find(" {") else { break };
            let sig = after[..open].to_string();
            let body_end = after[open..]
                .find("\n    }")
                .map_or(after.len(), |e| open + e);
            out.push((name, sig, after[open..body_end].to_string()));
            rest = &after[open..];
        }
    }
    out
}

/// No entry point takes a raw descriptor, so none can dereference one
/// unvalidated.
///
/// # What replaced `checked`, and why this test is not the one that was here
///
/// This file used to hold three tests over a mechanism that is gone.
/// `serve::checked` paired the dereference with a `driver_api::local::
/// validate_*`, and the tests asserted that every entry point called it, that
/// nothing went around it with a bare `unsafe { p.as_ref() }`, and that four
/// named validators were still reached. All three scanned for a shape that no
/// longer occurs: they found ZERO `checked(` sites, zero bare dereferences and
/// zero validators, and said so rather than passing -- which is the emptiness
/// assertion in each of them doing exactly its job, and the reason this could
/// be repaired instead of discovered later.
///
/// The thirteen entry points are not `extern "C"` any more (see
/// `serve::guard`'s header). They take `&driver_api::ModelLoadDesc`,
/// `&ChannelRegistrationPlan`, `FrameSubmission` -- typed references and owned
/// plans. There is no pointer to be null, no length to be mismatched against
/// its buffer, and nothing for a `validate_*_desc` to check that the type does
/// not already state. The capability was not routed around this time; it was
/// subsumed.
///
/// So the claim that survives is structural, and this holds it: a raw
/// descriptor cannot come back without failing here first. The rules that were
/// NOT about the C shape -- the frame's roster bounds, the copy's domains, the
/// encode plan's CSR partitioning -- are still real and still skippable, and
/// the three `*_validates_its_*` tests below are what keep them on the path.
#[test]
fn no_entry_point_takes_a_raw_descriptor() {
    let entries = shell_entries();
    assert!(
        entries.len() >= 12,
        "the scan found {} `impl Shell` methods, so it stopped recognising \
         them and would pass while every one of them took a raw pointer: {:?}",
        entries.len(),
        entries.iter().map(|(n, ..)| n).collect::<Vec<_>>()
    );

    let raw: Vec<&str> = entries
        .iter()
        .filter(|(_, sig, _)| sig.contains("*const ") || sig.contains("*mut "))
        .map(|(n, ..)| n.as_str())
        .collect();
    assert!(
        raw.is_empty(),
        "these entry points take a RAW pointer: {raw:?}\n  \
         A descriptor reaching this shell as a pointer brings back everything \
         `driver_api::local::validate_*` existed for -- the null test, the \
         ptr/len agreement, the reserved words -- and this crate no longer has \
         a `checked` to pair the dereference with the validation. Take a typed \
         reference or an owned plan, as the other entry points do; if it must \
         be a pointer, restore `checked` and the scan that proved nothing went \
         around it (this file, before the shell stopped being `extern \"C\"`)."
    );
}

/// The launch entry point validates its frame.
///
/// It is separate from `no_validator_is_deferred` because the frame is the
/// one verb that no longer takes a C descriptor: `validate_frame_desc` and
/// `checked` do not apply to a `FrameSubmission`, which owns its `Vec`s and
/// has no pointer to be null. The RULES did not go anywhere — they are
/// `FrameSubmission::validate`, and `driver-api`'s `validation_tests` pin
/// each one — but the call site here is what keeps them on the path a
/// launch actually takes, which is the whole point of
/// `validators-unskippable`.
#[test]
fn launch_validates_its_frame() {
    let src = shell_source();
    assert!(
        src.contains("frame.validate()"),
        "`pie_cuda_launch` no longer calls `frame.validate()` — the frame \
         verb stopped validating, and its rules (roster bounds, one terminal \
         cell per member and distinct across steps, the CSRs, the \
         recurrent-state parallelism) are checked nowhere else on this path"
    );
}

/// The KV-copy verb validates its plan.
///
/// The other two transfer verbs need no floor: `validate_state_copy_desc`
/// and `validate_pool_resize_desc` stated nothing but `ptr/len mismatch` and
/// the header words, so an owned plan carries their whole content in its
/// type. `copy_kv` had two rules that a `Vec` does not state — both domains
/// name a real one, and the page lists are parallel — and this is what keeps
/// them on the path a copy takes.
#[test]
fn copy_kv_validates_its_plan() {
    let src = shell_source();
    assert!(
        src.contains("copy.validate()"),
        "`pie_cuda_copy_kv` no longer calls `copy.validate()` — a copy with \
         an unknown memory domain, or with more source pages than \
         destinations, is checked nowhere else on this path"
    );
}

/// The encode verb validates its plan.
///
/// The heaviest of the three: `validate_encode_desc` was mostly NOT about
/// the C shape. It checks that the image and audio planes describe the same
/// counts, that each byte payload is `f32`-aligned and EXACTLY partitioned
/// by its CSR, and that a plane with no anchors carries no payload — none of
/// which a `Vec` states. All of it is `MediaEncodePlan::validate`.
#[test]
fn encode_validates_its_plan() {
    let src = shell_source();
    assert!(
        src.contains("encode.validate()"),
        "`pie_cuda_encode` no longer calls `encode.validate()` — a media \
         payload with no anchor to attach it to, a misaligned pixel buffer, \
         or a CSR that does not partition its bytes is checked nowhere else"
    );
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
    // An entry point is one that ANSWERS A STATUS. That is the derivation,
    // and it is why `device_facts` and `live_device_bytes` are not on the
    // list without being written down as exceptions: they return a borrowed
    // field and an integer, they cannot fail, and there is no status for
    // `guard` to answer with. Anything that can report `Err(i32)` to the
    // engine is a door, and every door catches.
    let entries: Vec<(String, bool)> = shell_entries()
        .into_iter()
        .filter(|(_, sig, _)| sig.contains(", i32>"))
        .map(|(name, _, body)| {
            // `guard` is the FIRST thing the body does or it is not a guard --
            // work above it is work outside the catch. Six lines of slack for
            // the multi-line call form, which is what three of them use.
            let guarded = body
                .lines()
                .take(7)
                .any(|l| l.contains("guard(\"") || l.trim() == "guard(");
            (name, guarded)
        })
        .collect();

    assert!(
        entries.len() >= 12,
        "the entry scan found {} entries, so it stopped recognising them \
         and would pass while every one of them was unguarded: {entries:?}",
        entries.len()
    );

    let naked: Vec<&str> = entries
        .iter()
        .filter(|(_, g)| !*g)
        .map(|(n, _)| n.as_str())
        .collect();
    assert!(
        naked.is_empty(),
        "these entry points let a panic reach the caller: {naked:?}. \
         `serve::guard` turns it into a failed request instead. These are \
         plain Rust now, so the panic unwinds rather than being undefined \
         behaviour -- which is precisely why it can be caught, and why not \
         catching it is a choice. It unwinds through the engine's worker and \
         kills every other request that worker was serving, over one bad \
         plan. Answer the status these signatures already return."
    );
}
