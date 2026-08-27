use super::extent::ValueDesc;

pub const ALIGN: u64 = 256;

pub const DUMMY_BYTES: u64 = ALIGN;

pub const MAX_BYTES: u64 = 512 << 20;

const TEMPORARIES_PER_ELEMENT: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TooLarge {
    Bound { bytes: u64, limit: u64 },

    Overflow,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub values: Vec<u64>,

    pub temporary: u64,

    pub temporary_bytes: u64,

    pub total: u64,
}

fn align_up(value: u64) -> Option<u64> {
    value.checked_next_multiple_of(ALIGN)
}

pub fn layout(descriptors: &[ValueDesc]) -> Result<Layout, TooLarge> {
    let mut values = Vec::with_capacity(descriptors.len());
    let mut at = DUMMY_BYTES;

    let mut widest: u64 = 1;

    for descriptor in descriptors {
        at = align_up(at).ok_or(TooLarge::Overflow)?;
        values.push(at);
        let span = align_up(descriptor.device_bytes()).ok_or(TooLarge::Overflow)?;
        at = at.checked_add(span).ok_or(TooLarge::Overflow)?;
        widest = widest.max(u64::from(descriptor.len));
    }

    let temporary = align_up(at).ok_or(TooLarge::Overflow)?;
    let temporary_bytes = widest
        .checked_mul(size_of::<u32>() as u64)
        .and_then(|bytes| bytes.checked_mul(TEMPORARIES_PER_ELEMENT))
        .and_then(align_up)
        .ok_or(TooLarge::Overflow)?;
    let total = temporary
        .checked_add(temporary_bytes)
        .ok_or(TooLarge::Overflow)?;

    if total > MAX_BYTES {
        return Err(TooLarge::Bound {
            bytes: total,
            limit: MAX_BYTES,
        });
    }
    Ok(Layout {
        values,
        temporary,
        temporary_bytes,
        total,
    })
}
