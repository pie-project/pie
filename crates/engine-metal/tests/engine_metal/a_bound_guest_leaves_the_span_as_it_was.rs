#![cfg(target_vendor = "apple")]
//! A guest program's state is its instances' and its pair's, in `Instance`
//! slots (`engine::Engine::guest_state`): nothing of it lies in the
//! activation span. Bound 64 times over, a program leaves the span the
//! rung's fires take as it was before the first bind. (Laid in the span,
//! its batch tables and lanes' scratch cut the widest rung that fit beside
//! them to 32 rows on a load sized for 1024.)

use eta_ir::registry::{GeometryClass, PortMask};

use super::a_scattered_pool_serves_the_same_logits::{fixture_of, load_with};
use super::a_score_reader_holds_its_rows_through_a_park::reader;

const CONTEXT: u32 = 256;
const BOUND: u32 = 64;

#[test]
fn a_bound_guest_leaves_the_span_as_it_was() {
    if !engine_metal::device::present() {
        eprintln!("skipping: no Metal device");
        return;
    }
    let fixture = fixture_of("qwen3-micro-text", None);
    let mut shell = load_with(&fixture, CONTEXT);
    let before = shell.span_covered();
    assert!(before > 0, "the rung's carve is backed");

    let program = shell.register_program(&reader(CONTEXT)).expect("the reader registers");
    let facts = shell.program_facts(program).expect("the reader's facts");
    let extents = eta_exec::Extents {
        row_count: 1,
        token_count: 1,
        sampled_rows: 1,
        query_len: 1,
        ..eta_exec::Extents::default()
    };
    let state = shell.guest_state(program, &extents).expect("its guest state");
    assert!(
        state.per_pair > 0,
        "the reader's pair holds its batch sets in its state: {state:?}"
    );

    let layout = shell.pool_layout();
    let instance = layout.instance.expect("an instance shape");
    let channel = layout.channel.expect("a channel shape");
    let free = layout.free_chunk();
    let ring_slots: u32 = facts.rings.iter().map(|ring| ring.footprint.slots).sum();
    let mut ids = Vec::new();
    for nth in 0..BOUND {
        // Each instance's tables a run of the first free chunk's slots, its
        // rings a run of the next one's.
        let first = free * instance.per_chunk + nth * facts.instance.slots;
        let slots: Vec<u32> = (first..first + facts.instance.slots).collect();
        let mut next = (free + 1) * channel.per_chunk + nth * ring_slots;
        let rings: Vec<engine::RingSlots> = facts
            .rings
            .iter()
            .map(|ring| {
                let slots = (next..next + ring.footprint.slots).collect();
                next += ring.footprint.slots;
                engine::RingSlots {
                    channel: ring.channel,
                    slots,
                }
            })
            .collect();
        let (scratch, pair) = crate::guest_places(&mut shell, program, &extents, nth);
        ids.push(
            shell
                .bind_program(
                    program,
                    &[],
                    extents,
                    GeometryClass::Host,
                    PortMask::NONE,
                    &[engine::ChannelId(0x6a00_0000 + u64::from(nth))],
                    &slots,
                    &rings,
                    &scratch,
                    &pair,
                    0,
                )
                .expect("the reader binds"),
        );
    }
    assert_eq!(
        shell.span_covered(),
        before,
        "{BOUND} bound instances leave the span the rung's fires take as it was"
    );
    for id in ids {
        shell.close_program_instance(id).expect("the reader closes");
    }
    assert_eq!(shell.span_covered(), before, "and their closes leave it so");
}
