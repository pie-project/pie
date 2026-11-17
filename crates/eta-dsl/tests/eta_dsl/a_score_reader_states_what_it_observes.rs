//! A program that reads attention scores states what it observes on its
//! builder: the traced container carries the observe section, which
//! round-trips through the wire; a program that states nothing carries none.

use eta_dsl::builder::Builder;
use eta_dsl::eta::container::{Observe, decode};
use eta_dsl::prelude::*;

const V: u32 = 8;

fn traced(observe: Option<Observe>) -> Vec<u8> {
    let out = Channel::new([1], dtype::i32);
    let mut b = Builder::new(V, 2);
    b.outputs(crate::outputs::logits(V));
    if let Some(observe) = observe {
        b.observe(observe);
    }
    b.stage(Stage::Epilogue, move || {
        out.put(cast(
            reduce_argmax(intrinsics::output(0, "logits")),
            dtype::i32,
        ));
    });
    b.build().expect("an output and a sink trace").encode()
}

#[test]
fn a_score_reader_states_what_it_observes() {
    for observe in [Observe::Rows(32), Observe::FireRows] {
        let bytes = traced(Some(observe));
        let container = decode(&bytes).expect("the stated container decodes");
        assert_eq!(container.observe, Some(observe));
        assert_eq!(container.encode(), bytes, "and keeps its bytes");
    }
    let bare = decode(&traced(None)).expect("a plain container decodes");
    assert_eq!(bare.observe, None, "a program that states nothing carries no section");
}
