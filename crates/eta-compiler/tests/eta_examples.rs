use eta_ir::container::{decode, encode};
use eta_ir::container_hash;
use eta_ir::registry::ModelProfile;
use eta_ir::validate::bind;

#[path = "common/traces.rs"]
mod traces;
use traces::*;

#[test]
fn eta_examples_every_case() {
    section3_serializes_validates_hashes_stably();
    beam_epilogue_serializes_validates_hashes_stably();
}

fn section3_serializes_validates_hashes_stably() {
    let c = section3_trace();
    let bytes = encode(&c);
    let h = container_hash(&bytes);
    let c2 = decode(&bytes).expect("decode");
    assert_eq!(c2, c);
    assert_eq!(container_hash(&encode(&c2)), h);
    let bound = bind(c, ModelProfile::dummy()).expect("bind");
    assert_eq!(bound.hash, h);
}

fn beam_epilogue_serializes_validates_hashes_stably() {
    let c = beam_trace();
    let bytes = encode(&c);
    let h = container_hash(&bytes);
    assert_eq!(decode(&bytes).expect("decode"), c);
    let bound = bind(c.clone(), beam_profile()).expect("bind");
    assert_eq!(bound.hash, h);
    assert_eq!(container_hash(&encode(&decode(&bytes).unwrap())), h);
}
