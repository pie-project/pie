//! A program that reads attention scores states what it observes in an
//! optional section after the container's tables: `Rows(n)` or `FireRows`
//! round-trips, a container built before the section (the SDKs' golden
//! containers) decodes as it always did, observing nothing stated, and a
//! section of another tag or kind is refused by name.

use eta_ir::container::{ContainerDecodeError, Observe, OBSERVE_SECTION, TraceContainer, decode};

const GOLDENS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../packages/python/inferlet/tests/goldens/eta_containers.txt"
);

#[test]
fn a_container_states_what_it_observes() {
    for observe in [Observe::Rows(32), Observe::Rows(1), Observe::FireRows] {
        let stated = TraceContainer {
            observe: Some(observe),
            ..TraceContainer::default()
        };
        let bytes = stated.encode();
        assert_eq!(decode(&bytes), Ok(stated), "{observe:?} round-trips");
        let plain = TraceContainer::default().encode();
        assert!(bytes.starts_with(&plain), "the section follows every table");
    }

    // Containers built before the section: each decodes, states nothing,
    // and encodes to its own bytes.
    let goldens = std::fs::read_to_string(GOLDENS).expect("the SDKs' golden containers");
    let mut decoded = 0;
    for line in goldens.lines().filter(|line| !line.trim().is_empty()) {
        let hex = line.split_whitespace().last().expect("a golden's bytes");
        let bytes: Vec<u8> = (0..hex.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(&hex[at..at + 2], 16).expect("hex"))
            .collect();
        let container = decode(&bytes).unwrap_or_else(|why| panic!("{line:.40}: {why}"));
        assert_eq!(container.observe, None, "an old container states no observation");
        assert_eq!(container.encode(), bytes, "and keeps its bytes");
        decoded += 1;
    }
    assert!(decoded >= 5, "the goldens hold containers: {decoded}");

    let mut other = TraceContainer::default().encode();
    other.push(OBSERVE_SECTION + 1);
    assert!(matches!(
        decode(&other),
        Err(ContainerDecodeError::UnknownTag { what: "section", .. })
    ));
    let mut kind = TraceContainer::default().encode();
    kind.extend([OBSERVE_SECTION, 7]);
    assert!(matches!(
        decode(&kind),
        Err(ContainerDecodeError::UnknownTag { what: "observe", .. })
    ));
}
