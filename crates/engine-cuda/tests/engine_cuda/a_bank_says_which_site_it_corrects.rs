use engine_cuda::blob::Site;
use engine_cuda::{layer_of, role_of, site_of};

#[test]
fn a_bank_says_which_site_it_corrects_every_case() {
    both_spellings_of_a_bank_name_parse();
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
