// Generated from the measured links; regenerate rather than edit by hand.
use super::memory::{Anchor, Code, Moving};
use crate::compile::kernels::KernelSet;
use crate::compile::plan::Plan;

pub const ANCHORS: &[Anchor] = &[
    Anchor {
        kernels: KernelSet::Own,
        weight_used: 47696.0,
        head_used: 38496.0,
        other_room: 21184.0,
        moving: Moving {
            weight: 30304.0,
            head: 27648.0,
        },
        head_moves: true,
        code: &[],
        attention: 0.0,
    },
    Anchor {
        kernels: KernelSet::Shared,
        weight_used: 47280.0,
        head_used: 40640.0,
        other_room: 5088.0,
        moving: Moving {
            weight: 28960.0,
            head: 29696.0,
        },
        head_moves: true,
        code: SHARED,
        attention: 0.0,
    },
    Anchor {
        kernels: KernelSet::Windows,
        weight_used: 47696.0,
        head_used: 46160.0,
        other_room: 22224.0,
        moving: Moving {
            weight: 31230.0,
            head: 66816.0,
        },
        head_moves: false,
        code: WINDOWS,
        attention: 26796.0,
    },
    Anchor {
        kernels: KernelSet::Experts,
        weight_used: 47840.0,
        head_used: 30144.0,
        other_room: 28416.0,
        moving: Moving {
            weight: 34560.0,
            head: 4352.0,
        },
        head_moves: false,
        code: &[],
        attention: 0.0,
    },
];

/// The shared kernels' weight image (`kv.csl`): positions over KV rows not a power of two take
/// software multiplies and divides; a sliding ring that is not the window exactly masks its
/// entries; sliding entries not a power of two index by multiplies (the e2b links: 692, 268 and
/// 224 B over the e4b anchor's 128 rows, exact ring and 4 entries); a ring of seats' code past
/// what its caches and references give, and its sliding rings a seat's where those take fewer
/// than the pool's entries (the e4b links at 2 to 6 seats: 734 and 120 B).
const SHARED: &[Code] = &[
    Code {
        what: "KV rows not a power of two",
        on: |p: &Plan| !p.partition.kv_rows.is_power_of_two(),
        anchor: false,
        bytes: 692.0,
    },
    Code {
        what: "a masked sliding window",
        on: |p: &Plan| crate::compile::kernels::stacked::masked_window(p) != 0,
        anchor: false,
        bytes: 268.0,
    },
    Code {
        what: "sliding entries not a power of two",
        on: |p: &Plan| {
            !p.partition
                .kv_entries
                .sliding
                .unwrap_or(1)
                .is_power_of_two()
        },
        anchor: false,
        bytes: 224.0,
    },
    Code {
        what: "a ring of seats",
        on: |p: &Plan| p.seats > 1,
        anchor: false,
        bytes: 734.0,
    },
    Code {
        what: "sliding rings a seat",
        on: fixed_rings,
        anchor: false,
        bytes: 120.0,
    },
];

/// A ring of seats whose sliding rings, a seat's each, take fewer entries than the pool's.
fn fixed_rings(p: &Plan) -> bool {
    let e = &p.partition.kv_entries;
    p.seats > 1
        && e.sliding
            .is_some_and(|n| n * p.seats < e.global.unwrap_or(e.per_row))
}

/// The windowed rows' weight image (`kv.csl`): a ring of seats' code past what its attention
/// rows' moving bytes give (the 12B links at 2 seats and 13 entries a row: 168 B), and its
/// sliding rings a seat's (the shared kernels' measure).
const WINDOWS: &[Code] = &[
    Code {
        what: "a ring of seats",
        on: |p: &Plan| p.seats > 1,
        anchor: false,
        bytes: 168.0,
    },
    Code {
        what: "sliding rings a seat",
        on: fixed_rings,
        anchor: false,
        bytes: 120.0,
    },
];
