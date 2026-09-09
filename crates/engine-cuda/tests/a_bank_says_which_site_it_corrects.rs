use std::path::{Path, PathBuf};

use engine_cuda::blob::{Adapters, Site};
use engine_cuda::{BankSeat, layer_of, role_of, site_of};

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

fn sited() -> Vec<BankSeat> {
    (0..LAYERS)
        .flat_map(|layer| pair(&format!("layer.{layer}.o")))
        .collect()
}

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

#[test]
fn a_bank_says_which_site_it_corrects_every_case() {
    both_spellings_of_a_bank_name_parse();
    a_manifest_that_states_a_site_lands_into_that_sites_banks();
    a_site_the_shell_cannot_serve_is_refused_by_name();
}

fn both_spellings_of_a_bank_name_parse() {
    assert_eq!(role_of("layer.7.lora_a"), "lora_a");
    assert_eq!(layer_of("layer.7.lora_a"), 7);
    assert_eq!(site_of("layer.7.lora_a"), None, "states no site");

    for site in Site::ALL {
        let name = format!("layer.7.{}.lora_b", site.spelled());
        assert_eq!(role_of(&name), "lora_b", "{name}");
        assert_eq!(layer_of(&name), 7, "{name}");
        assert_eq!(site_of(&name), Some(site), "{name}");
    }

    assert_eq!(site_of("layer.0.gate_up.lora_a"), Some(Site::GateUp));

    assert_eq!(Site::Q.bit(), 1 << 0);
    assert_eq!(Site::O.bit(), 1 << 3);
    assert_eq!(Site::Down.bit(), 1 << 5);
}

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

fn a_site_the_shell_cannot_serve_is_refused_by_name() {
    let (mount, adapters) = mounted("refused");

    write_adapter(&mount, "typo", 2, Some("mixer"));
    let why = adapters
        .planes("typo", &sited())
        .expect_err("`mixer` is not a site");
    let said = why.to_string();
    assert!(said.contains("mixer"), "names the word it was given: {said}");
    assert!(said.contains("`gate_up`"), "and the vocabulary: {said}");

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

    write_adapter(&mount, "unstated", 2, None);
    let why = adapters
        .planes("unstated", &sited())
        .expect_err("no site is not `o`");
    assert!(
        why.to_string().contains("at no stated site"),
        "says what was asked: {why}"
    );
}
