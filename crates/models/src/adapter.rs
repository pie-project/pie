use poem::{Dtype, Weight};

#[derive(Clone, Copy, Debug)]
pub struct Adapters {
    pub slots: u32,
    pub rank: u32,
}

#[must_use]
pub fn banks(prefix: &str, a: Adapters, hidden: u64, dense: Dtype) -> (Weight, Weight) {
    banks_at(prefix, None, a, hidden, dense)
}

#[must_use]
pub fn banks_at(
    prefix: &str,
    site: Option<Site>,
    a: Adapters,
    hidden: u64,
    dense: Dtype,
) -> (Weight, Weight) {
    let slots = u64::from(a.slots);
    let rank = u64::from(a.rank);
    let at = match site {
        Some(site) => format!(".{}", site.spelled()),
        None => String::new(),
    };
    (
        Weight::sym(format!("{prefix}{at}.lora_a"), [slots, rank, hidden], dense).registered(),
        Weight::sym(format!("{prefix}{at}.lora_b"), [slots, hidden, rank], dense).registered(),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Site {
    Q,
    K,
    V,
    O,
    GateUp,
    Down,
}

impl Site {
    #[must_use]
    pub const fn spelled(self) -> &'static str {
        match self {
            Site::Q => "q",
            Site::K => "k",
            Site::V => "v",
            Site::O => "o",
            Site::GateUp => "gate_up",
            Site::Down => "down",
        }
    }
}
