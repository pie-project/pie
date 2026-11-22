// Generated from the measured links; regenerate rather than edit by hand.
use super::memory::{Anchor, Moving};
use crate::compile::kernels::KernelSet;

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
    },
];
