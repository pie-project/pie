#![cfg(feature = "device")]
//! A device-geometry program's ports state the lanes it is attached over,
//! and those alone: an `embed_indptr` describing one lane more would set
//! the next request's positions and kv slots, through that request's own
//! table. The attachment's lanes are refused by name before any position
//! or slot is written, and the neighbour reads as it did.

use crate::a_row_placed_below_its_floor_is_refused::{Fixture, PAGE, bind, fixture, read};
use engine_cuda::serve::Attached;
use engine_cuda::{Fault, Lane, Seated, Shell};

/// The placing lane (lane 0, slot 1, reading through `translation`, its
/// one lane the attachment's) beside a neighbour decoding one token on
/// slot 0 over `reader`, as lane 1 of the same fire.
fn beside(
    shell: &mut Shell,
    word: u64,
    instance: engine::InstanceId,
    translation: &[u32],
    reader: &[u32],
) -> Result<Vec<Vec<f32>>, Fault> {
    let (placed, read) = ([0u32], [21u32]);
    shell.fire_media(
        &[
            Seated {
                translation,
                folded: PAGE,
                write_end: 2 * PAGE,
                ..Seated::of(Lane {
                    slot: 1,
                    word,
                    tokens: &placed,
                })
            },
            Seated {
                pages: reader,
                held: Some(PAGE),
                ..Seated::of(Lane {
                    slot: 0,
                    word,
                    tokens: &read,
                })
            },
        ],
        &[Attached {
            lane: 0,
            lanes: 1,
            instance,
            adapter_only: false,
        }],
        &[],
    )
}

#[test]
fn a_port_describing_a_neighbours_lane_is_refused() {
    if !engine_cuda::device::present() {
        eprintln!("no CUDA device: skipping");
        return;
    }
    let Fixture {
        mut shell,
        mut guest,
        shared,
        own,
        placer,
        word,
        want,
        _dir,
    } = fixture();
    let reader = [shared, own];
    let table = [shared, placer];

    // Attached over lane 0, its ports describe lane 1 too: a row at slot
    // 3 of the neighbour's table, its shared page.
    let over = bind(&mut shell, &mut guest, &[PAGE, 3], 31);
    let refused = beside(&mut shell, word, over, &table, &reader)
        .expect_err("ports describing a lane past the attachment");
    assert!(
        matches!(
            &refused,
            Fault::Overreach {
                instance,
                lanes,
                carried: 2,
            } if *instance == over && *lanes == (0..1)
        ),
        "the attachment's lanes are refused by name: {refused}"
    );
    let why = refused.to_string();
    assert!(
        why.contains("embed_indptr") && why.contains("lanes 0..1"),
        "{why}"
    );
    assert_eq!(
        read(&mut shell, word, &reader),
        want,
        "the neighbour's page is unchanged"
    );

    // Its ports stating its own lane alone, the same fire lands, and the
    // neighbour beside it reads what it read alone.
    let own_lane = bind(&mut shell, &mut guest, &[PAGE], 41);
    let rows = beside(&mut shell, word, own_lane, &table, &reader)
        .expect("ports describing the attachment's own lane");
    assert!(
        rows.iter().any(|row| *row == want),
        "the neighbour answers as it did alone"
    );
    assert_eq!(
        read(&mut shell, word, &reader),
        want,
        "the neighbour's page is unchanged"
    );
}
