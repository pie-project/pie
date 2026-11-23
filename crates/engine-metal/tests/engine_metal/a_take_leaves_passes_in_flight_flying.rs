#![cfg(target_vendor = "apple")]

//! The host takes a run-ahead instance's committed cell without waiting out
//! its passes in flight, where its program never takes from that channel: a
//! pass in flight puts past the committed cell and its commit moves the
//! tail. The take returns the committed value and nothing of the pass still
//! in flight. From a channel the program takes from, a take still waits out
//! every pass of the instance.

use std::sync::mpsc;

use engine_metal::serve::Seated;
use engine_metal::Lane;

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_ahead, word};
use super::an_envelope_keeps_two_passes_in_flight::{bind_envelope, heard, submit};

/// The envelope's host-read channel (the program only puts there) and its
/// token channel (the program takes from it each pass).
const OUT: u32 = 0;
const TOK: u32 = 1;

#[test]
fn a_take_leaves_passes_in_flight_flying() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_ahead(&fixture, 1024, None, engine::runahead::Runahead::of(2));
    let tokens: Vec<u32> = vec![7];
    shell.open(3).expect("slot 3 opens");
    let vocab = shell
        .fire_seated(&[Seated {
            replay_stated: None,
            held: Some(0),
            ..Seated::of(Lane {
                slot: 3,
                word: word(&fixture.trace, 1),
                tokens: &tokens,
            })
        }])
        .expect("a plain fire lands")[0]
        .len() as u32;
    let id = bind_envelope(&mut shell, vocab, 0);
    shell.open(0).expect("the slot opens");
    let (tx, rx) = mpsc::channel();

    // Pass 1 lands and commits; pass 2 is submitted and left in flight.
    submit(&mut shell, id, &tokens, 1, false, &tx);
    heard(&rx, 1);
    shell.reap().expect("pass 1 publishes");
    submit(&mut shell, id, &tokens, 2, false, &tx);

    let session = shell
        .program_instance_taking(id, OUT)
        .expect("the take resolves")
        .expect("the instance is bound");
    let expected = session.peek(OUT, 0).expect("pass 1's cell");
    let took = session.take(OUT).expect("the take reads").expect("pass 1 committed a cell");
    assert_eq!(took, expected, "the take returns pass 1's committed cell");
    assert_eq!(session.depth(OUT), 0, "nothing of pass 2 is committed or read");
    let flying = shell.frames_carrying(id);

    // From the channel the program takes from, the take waits pass 2 out.
    shell
        .program_instance_taking(id, TOK)
        .expect("the take resolves")
        .expect("the instance is bound");
    assert_eq!(shell.frames_carrying(id), 0, "a take from a channel the program takes waits out its passes");
    heard(&rx, 2);
    assert_eq!(
        flying, 1,
        "the take from the host-read channel left pass 2 in flight (it had not landed and been waited out)"
    );
    shell.reap().expect("pass 2 publishes");
    shell.close_program_instance(id).expect("the instance closes");
}
