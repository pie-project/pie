use crate::weights::BankSeat;

#[must_use]
pub fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// This load's banks as `engine::bundle` fills them.
#[must_use]
pub fn banks(seats: &[BankSeat]) -> Vec<engine::bundle::Bank<'_>> {
    seats
        .iter()
        .map(|seat| engine::bundle::Bank {
            name: &seat.name,
            rows: seat.rows,
            cols: seat.cols,
            slot: seat.slot,
            dtype: seat.dtype,
            shard: &seat.shard,
        })
        .collect()
}
