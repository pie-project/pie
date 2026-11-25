//! Operator-declared adapters are refused on every backend: an adapter
//! arrives through a program's `lora` sink at bind and lands in `Adapter`
//! slots; `register_adapter` names that replacement, loaded or not.

use engine::adapter::{AdapterPlane, AdapterRegistration};
use engine_metal::{DeviceBoot, Metal};

#[test]
fn an_operator_declared_adapter_is_refused() {
    let mut metal = Metal::new(
        DeviceBoot {
            profiles: None,
            measure: engine::rates::Measuring::Time,
            ..DeviceBoot::default()
        },
        |_, _| Err("no contract: nothing loads here".to_string()),
    );
    let why = engine::Programs::adapter(&mut metal, &AdapterRegistration {
            id: 0,
            planes: vec![AdapterPlane {
                bank: "layer.0.lora_a".to_string(),
                bytes: vec![0u8; 64],
            }],
        })
        .expect_err("an operator-declared adapter is refused")
        .to_string();
    assert!(
        why.contains("operator-declared") && why.contains("`lora` sink"),
        "the refusal names the replacement: {why}"
    );
}
