#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeEntry {
    pub name: &'static str,

    pub resident: bool,
}

pub const TIER1: &[RuntimeEntry] = &[
    RuntimeEntry {
        name: "kv_cache",
        resident: true,
    },
    RuntimeEntry {
        name: "recurrent_state",
        resident: true,
    },
    RuntimeEntry {
        name: "positions",
        resident: false,
    },
    RuntimeEntry {
        name: "token_ids",
        resident: false,
    },
    RuntimeEntry {
        name: "request_of_token",
        resident: false,
    },
    RuntimeEntry {
        name: "qo_indptr",
        resident: false,
    },
    RuntimeEntry {
        name: "row_valid",
        resident: false,
    },
    RuntimeEntry {
        name: "attention_mask",
        resident: false,
    },
    RuntimeEntry {
        name: "sampling_indices",
        resident: false,
    },
    RuntimeEntry {
        name: "first_token",
        resident: false,
    },
];

#[must_use]
pub fn tier1(name: &str) -> Option<&'static RuntimeEntry> {
    TIER1.iter().find(|e| e.name == name)
}
