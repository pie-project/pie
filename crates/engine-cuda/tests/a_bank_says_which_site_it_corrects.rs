//! **A BANK NAME CAN SAY WHICH PROJECTION IT CORRECTS** (alto next B3,
//! closing A-2's finding as A-6 renewed it).
//!
//! # The gap this file judges
//!
//! Until this wave a bank was named `layer.{l}.{role}` and `role` was matched
//! against `lora_a`/`lora_b` LITERALLY, so a text had no way to state WHICH
//! site its banks corrected: `layer.3.mixer.lora_a` parsed as a role of its
//! own at layer zero and every shared bind refused it. The consequence was the
//! silent-wrongness class — a guest's `Site::Q` took the one site the text
//! happened to correct (the mixer output) and nobody was told.
//!
//! The widening is a name grammar with an OPTIONAL middle segment:
//!
//! ```text
//! layer.{n}.{role}              the pre-B3 spelling — states no site
//! layer.{n}.{site}.{role}       states one, from the guest surface's own
//!                               vocabulary (`inferlet::eta::adapter::Site`)
//! ```
//!
//! and the discipline that absent is a VALUE — the text's own default site —
//! and not a wildcard. The claims:
//!
//! ```text
//! (a) both spellings parse: layer, site and role, off one name
//! (b) a middle segment outside the vocabulary is NOT a site, so the whole
//!     name stays the role and `layer.3.mixer.lora_a` goes on refusing
//! (c) a manifest's optional `site` key selects the sited banks
//! (d) a `site` outside the vocabulary is refused BY NAME, and so is one
//!     no bank of this load declares
//! (e) an absent site is today's meaning, byte for byte: the untagged banks
//!     the six A-6 family texts declare land exactly what they landed
//! ```
//!
//! No device: every claim is a string, a manifest, or a host-side slice.
//!
//! ```text
//! cargo test -p engine-cuda --test a_bank_says_which_site_it_corrects
//! ```

use std::path::{Path, PathBuf};

use engine_cuda::blob::{Adapters, Site};
use engine_cuda::{BankSeat, layer_of, role_of, site_of};

// ── the fixture ──────────────────────────────────────────────────────────

const LAYERS: u64 = 2;
const BANK_RANK: u64 = 4;
const HIDDEN: u64 = 8;
const ELEM: u64 = 2;

fn scratch(what: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or(0);
    let at = std::env::temp_dir().join(format!("pie-site-{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&at).expect("a scratch directory");
    at
}

/// One layer's pair of banks under `prefix` — `A` rank-major, `B` out-major,
/// which is §6.3's statute stated as shapes.
fn pair(prefix: &str) -> [BankSeat; 2] {
    let slot = BANK_RANK * HIDDEN * ELEM;
    [
        BankSeat {
            name: format!("{prefix}.lora_a"),
            adapters: 2,
            slot,
            rows: BANK_RANK,
            cols: HIDDEN,
            elem: ELEM,
        },
        BankSeat {
            name: format!("{prefix}.lora_b"),
            adapters: 2,
            slot,
            rows: HIDDEN,
            cols: BANK_RANK,
            elem: ELEM,
        },
    ]
}

/// The banks a PRE-B3 text declares: `layer.{l}.lora_a`, no site stated.
/// This is what all six A-6 family texts name, verbatim.
fn untagged() -> Vec<BankSeat> {
    (0..LAYERS)
        .flat_map(|layer| pair(&format!("layer.{layer}")))
        .collect()
}

/// The banks a text that STATES its site declares, at the same site every
/// family text actually corrects — the mixer output.
fn sited() -> Vec<BankSeat> {
    (0..LAYERS)
        .flat_map(|layer| pair(&format!("layer.{layer}.o")))
        .collect()
}

/// Write an adapter directory whose manifest states `site`, or does not.
fn write_adapter(mount: &Path, name: &str, rank: u64, site: Option<&str>) {
    let dir = mount.join(name);
    std::fs::create_dir_all(&dir).expect("an adapter directory");
    let at = match site {
        Some(site) => format!("site = \"{site}\"\n"),
        None => String::new(),
    };
    std::fs::write(
        dir.join("adapter.toml"),
        format!(
            "rank = {rank}\n\n\
             [[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\nlayout = \"rank_major\"\n{at}\n\
             [[plane]]\nrole = \"lora_b\"\nfile = \"b.bin\"\nlayout = \"out_major\"\n{at}"
        ),
    )
    .expect("a manifest");
    let ramp: Vec<u8> = (0..(LAYERS * rank * HIDDEN) as usize)
        .flat_map(|at| ((at as u16) | 0x0100).to_le_bytes())
        .collect();
    std::fs::write(dir.join("a.bin"), &ramp).expect("an A plane");
    std::fs::write(dir.join("b.bin"), &ramp).expect("a B plane");
}

fn mounted(what: &str) -> (PathBuf, Adapters) {
    let mount = scratch(what);
    let mut adapters = Adapters::new(4);
    adapters.mount(Some(mount.clone()));
    (mount, adapters)
}

// ── (a) and (b): the name grammar ────────────────────────────────────────

/// **(a)** Both spellings parse, and the optional segment is the only
/// difference between them.
#[test]
fn both_spellings_of_a_bank_name_parse() {
    // The pre-B3 spelling: no site, and every other answer unmoved.
    assert_eq!(role_of("layer.7.lora_a"), "lora_a");
    assert_eq!(layer_of("layer.7.lora_a"), 7);
    assert_eq!(site_of("layer.7.lora_a"), None, "states no site");

    // The sited spelling, once per word of the vocabulary — the spellings ARE
    // the contract, so every one of them is checked rather than a sample.
    for site in Site::ALL {
        let name = format!("layer.7.{}.lora_b", site.spelled());
        assert_eq!(role_of(&name), "lora_b", "{name}");
        assert_eq!(layer_of(&name), 7, "{name}");
        assert_eq!(site_of(&name), Some(site), "{name}");
    }

    // A two-word site is one segment and reads whole — `gate_up` is not a
    // `gate` at some layer called `up`.
    assert_eq!(site_of("layer.0.gate_up.lora_a"), Some(Site::GateUp));

    // And the vocabulary's bits are the guest surface's own, which is what
    // makes a guest's ask and a bank's declaration one number to compare.
    assert_eq!(Site::Q.bit(), 1 << 0);
    assert_eq!(Site::O.bit(), 1 << 3);
    assert_eq!(Site::Down.bit(), 1 << 5);
}

/// **(b)** A middle segment outside the vocabulary is NOT a site: the name
/// has no prefix to cut, so the WHOLE of it is the role — which is exactly
/// how `layer.3.mixer.lora_a` refused before this wave and goes on refusing.
///
/// This is the half of the widening that is a promise about what did NOT
/// change. A grammar that accepted any middle word would have turned an
/// operator's typo into a bank at "whatever the text corrects", which is the
/// silent wrongness this wave exists to close.
#[test]
fn a_middle_segment_that_is_not_a_site_is_not_a_site() {
    assert_eq!(
        role_of("layer.3.mixer.lora_a"),
        "layer.3.mixer.lora_a",
        "an unknown middle word leaves the whole name as its own role"
    );
    assert_eq!(layer_of("layer.3.mixer.lora_a"), 0, "and no layer");
    assert_eq!(site_of("layer.3.mixer.lora_a"), None, "and no site");
    assert_eq!(Site::parse("mixer"), None, "`mixer` is not of the vocabulary");

    // A name with no numbered component at all is still its own role at layer
    // zero, unmoved from before.
    assert_eq!(role_of("lora_a"), "lora_a");
    assert_eq!(layer_of("lora_a"), 0);
    // A site where the layer should be is not a layer.
    assert_eq!(role_of("o.lora_a"), "o.lora_a");
}

// ── (c) and (d): the manifest key ────────────────────────────────────────

/// **(c)** A manifest that states `site` lands into the banks that declare
/// it, per layer and by name.
#[test]
fn a_manifest_that_states_a_site_lands_into_that_sites_banks() {
    let (mount, adapters) = mounted("stated");
    write_adapter(&mount, "alice-v2", 2, Some("o"));
    let (built, fingerprint) = adapters
        .planes("alice-v2", &sited())
        .expect("a manifest at the site the text names");

    assert_ne!(fingerprint, 0, "the identity's content half is recorded");
    let landed: Vec<&str> = built.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        landed,
        vec![
            "layer.0.o.lora_a",
            "layer.1.o.lora_a",
            "layer.0.o.lora_b",
            "layer.1.o.lora_b",
        ],
        "one plane per sited bank, per layer, in role order"
    );
    for (name, plane) in &built {
        assert_eq!(
            plane.len() as u64,
            BANK_RANK * HIDDEN * ELEM,
            "`{name}` is one whole slot"
        );
    }
}

/// **(d)** Both site refusals, by name: a spelling nobody can name, and a
/// site this load's banks do not declare.
#[test]
fn a_site_the_shell_cannot_serve_is_refused_by_name() {
    let (mount, adapters) = mounted("refused");

    // A spelling outside the vocabulary, refused at the manifest with the six
    // words a bank can be named at.
    write_adapter(&mount, "typo", 2, Some("mixer"));
    let why = adapters
        .planes("typo", &sited())
        .expect_err("`mixer` is not a site");
    let said = why.to_string();
    assert!(said.contains("mixer"), "names the word it was given: {said}");
    assert!(said.contains("`gate_up`"), "and the vocabulary: {said}");

    // A site of the vocabulary that this load's banks do not declare, refused
    // at the resolver with the banks there are.
    write_adapter(&mount, "elsewhere", 2, Some("q"));
    let why = adapters
        .planes("elsewhere", &sited())
        .expect_err("`q` is not a site these banks correct");
    let said = why.to_string();
    assert!(said.contains("site `q`"), "names the site asked for: {said}");
    assert!(
        said.contains("layer.0.o.lora_a"),
        "and the banks this load declares: {said}"
    );

    // And the mirror: a manifest that states nothing, against banks that all
    // state a site. Absent is a VALUE — the unstated default — so it does not
    // quietly match `o`.
    write_adapter(&mount, "unstated", 2, None);
    let why = adapters
        .planes("unstated", &sited())
        .expect_err("no site is not `o`");
    assert!(
        why.to_string().contains("at no stated site"),
        "says what was asked: {why}"
    );
}

// ── (e) byte-compatibility ───────────────────────────────────────────────

/// **(e) THE SIX FAMILY TEXTS PAY NOTHING FOR THIS WAVE.**
///
/// A-6's texts every one declare `layer.{l}.lora_a` / `layer.{l}.lora_b` and
/// every operator manifest written before this wave states no `site`. Both
/// halves of that pair are the absent value, and the pair lands the SAME
/// BYTES in the SAME BANKS as it did before the parser was widened — which is
/// the whole licence for widening it without touching a text.
#[test]
fn an_untagged_text_and_a_siteless_manifest_land_what_they_always_landed() {
    let (mount, adapters) = mounted("compat");
    write_adapter(&mount, "alice-v2", 2, None);
    let (built, _) = adapters
        .planes("alice-v2", &untagged())
        .expect("the pre-B3 pair, unchanged");

    let landed: Vec<&str> = built.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(
        landed,
        vec![
            "layer.0.lora_a",
            "layer.1.lora_a",
            "layer.0.lora_b",
            "layer.1.lora_b",
        ],
        "the names the six family texts declare, untouched"
    );

    // And the contents: the source is a `(index | 0x0100)` u16 ramp over
    // `[LAYERS, rank, HIDDEN]`, so layer 1's `A` starts at element
    // `1 * rank * HIDDEN` and the trailing ranks are the zero rows the
    // orientation asks for. A widened parser that had picked up a different
    // bank would fail here as a wrong NUMBER, not only a wrong name.
    let rank = 2usize;
    let hidden = HIDDEN as usize;
    let a = &built
        .iter()
        .find(|(name, _)| name == "layer.1.lora_a")
        .expect("layer 1's A")
        .1;
    for row in 0..BANK_RANK as usize {
        for col in 0..hidden {
            let at = (row * hidden + col) * 2;
            let want = match row < rank {
                true => (((hidden * rank + row * hidden + col) as u16) | 0x0100).to_le_bytes(),
                false => [0, 0],
            };
            assert_eq!(&a[at..at + 2], &want, "A row {row} col {col} of layer 1");
        }
    }

    // The sited spelling is a DIFFERENT load, not a second reading of this
    // one: an untagged manifest against untagged banks and a sited manifest
    // against sited banks land the same bytes under different names.
    write_adapter(&mount, "bob-v1", 2, Some("o"));
    let (there, _) = adapters
        .planes("bob-v1", &sited())
        .expect("the sited pair lands too");
    for ((_, here), (_, there)) in built.iter().zip(there.iter()) {
        assert_eq!(here, there, "the site is a name, not a different arithmetic");
    }
}
