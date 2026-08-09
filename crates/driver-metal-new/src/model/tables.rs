//! The fire's own tables, staged once.
//!
//! A row names which table a slot wants; something has to put them on the
//! device. That was written twice — once at the engine seam and once in
//! `tests/device_real_weights.rs` — and the two drifted the moment the text
//! named a table only one of them knew: the gate held both fire classes to
//! MLX while the seam was still staging seven of nine, so a rotation with no
//! frequency ladder and a readout that answered the first token were invisible
//! from either side.
//!
//! One builder, and the resolver's index map beside it. What a caller supplies
//! is the FRAME's data; what this owns is the order.

use crate::error::Result;
use crate::metal::{Context, Handle, allocate};
use crate::model::executor::{FireTable, Slice};
use crate::region::Region as _;

/// The tables a fire states, in the order [`Staged::at`] indexes them.
///
/// Borrowed rather than owned: every one of these is the frame's, already
/// laid out, and copying them to describe them would be the third place they
/// live.
#[derive(Debug, Default, Clone, Copy)]
pub struct Frame<'a> {
    /// One token id per row.
    pub token_ids: &'a [u32],
    /// One position per row.
    pub position_ids: &'a [u32],
    /// Which request owns each token.
    pub req_of_token: &'a [u32],
    /// The page list, and the CSR into it.
    pub kv_page_indices: &'a [u32],
    /// See [`Self::kv_page_indices`].
    pub kv_page_indptr: &'a [u32],
    /// Per token: the physical page its KV row is written into, and the row
    /// within it.
    pub kv_write_page: &'a [u32],
    /// See [`Self::kv_write_page`].
    pub kv_write_offset: &'a [u32],
    /// The rotary inverse frequencies, as f32 bits.
    pub rope_frequencies: &'a [u32],
    /// Which rows the readout samples, one per request.
    pub sampling_indices: &'a [u32],
}

/// One region holding every table, and where each one starts.
#[derive(Debug)]
pub struct Staged {
    /// The region. **Held, not dropped** — every span points into it.
    pub region: Handle,
    /// Each table's `(start, len)`, in u32s, indexed as [`Frame`] declares.
    spans: Vec<(usize, usize)>,
}

impl Staged {
    /// Where `which` is, or `None` for a table this fire has none of.
    ///
    /// A zero-length table answers `None` rather than an empty region: a slot
    /// nobody fills is better than one pointed at nothing, because the second
    /// is a valid address the kernel will read.
    #[must_use]
    pub fn at(&self, which: FireTable) -> Option<Slice> {
        let i = match which {
            FireTable::TokenIds => 0,
            FireTable::Positions => 1,
            FireTable::RequestOfToken => 2,
            FireTable::KvPageIndices => 3,
            FireTable::KvPageIndptr => 4,
            FireTable::KvWritePage => 5,
            FireTable::KvWriteOffset => 6,
            FireTable::RopeFrequencies => 7,
            FireTable::SamplingIndices => 8,
            // No custom mask on this path yet, and the pool's numbers are
            // answered by `Resolver::pool` rather than by an address.
            FireTable::AttentionMask
            | FireTable::AttentionMaskEnabled
            | FireTable::KvHeadStride
            | FireTable::KvSeqStride
            | FireTable::KvPageSize => return None,
        };
        let (at, len) = *self.spans.get(i)?;
        (len > 0).then(|| Slice {
            address: self.region.gpu_address() + (at * 4) as u64,
            bytes: (len * 4) as u64,
        })
    }
}

/// Stage every table of `frame` into one region.
///
/// # Errors
///
/// The allocation or the write.
pub fn stage(context: &Context, frame: Frame<'_>) -> Result<Staged> {
    let mut blob: Vec<u32> = Vec::new();
    let mut spans: Vec<(usize, usize)> = Vec::new();
    // The ORDER is the contract, and it is this list — `Staged::at` indexes
    // into it and nothing else may.
    for table in [
        frame.token_ids,
        frame.position_ids,
        frame.req_of_token,
        frame.kv_page_indices,
        frame.kv_page_indptr,
        frame.kv_write_page,
        frame.kv_write_offset,
        frame.rope_frequencies,
        frame.sampling_indices,
    ] {
        spans.push((blob.len(), table.len()));
        blob.extend_from_slice(table);
    }
    let region = allocate(context, ((blob.len() * 4).max(4)) as u64, "fire tables")?;
    // SAFETY: freshly allocated and not yet encoded against.
    unsafe {
        let raw = core::slice::from_raw_parts(blob.as_ptr().cast::<u8>(), blob.len() * 4);
        region.write(0, raw)?;
    }
    Ok(Staged { region, spans })
}

#[cfg(test)]
// A device test that finds no device SAYS so and passes. Silence would make
// "the machine has no Metal 4" and "the table staged correctly" the same
// observation, which is the failure mode a skip is meant to avoid.
#[allow(clippy::print_stderr)]
mod tests {
    use super::*;

    #[test]
    fn a_table_this_fire_has_none_of_answers_nothing() {
        let Ok(context) = Context::new() else {
            eprintln!("SKIP: no Metal 4 device");
            return;
        };
        let ids = [7u32, 8, 9];
        let staged = stage(
            &context,
            Frame {
                token_ids: &ids,
                ..Frame::default()
            },
        )
        .expect("it stages");
        let tokens = staged.at(FireTable::TokenIds).expect("the ids are there");
        assert_eq!(tokens.bytes, 12);
        assert_eq!(tokens.address, staged.region.gpu_address());
        assert!(
            staged.at(FireTable::SamplingIndices).is_none(),
            "an empty table is a slot nobody fills, not an empty region"
        );
    }

    #[test]
    fn every_table_starts_where_the_one_before_it_ended() {
        let Ok(context) = Context::new() else {
            eprintln!("SKIP: no Metal 4 device");
            return;
        };
        let (a, b) = ([1u32, 2], [3u32, 4, 5]);
        let staged = stage(
            &context,
            Frame {
                token_ids: &a,
                position_ids: &b,
                ..Frame::default()
            },
        )
        .expect("it stages");
        let t = staged.at(FireTable::TokenIds).expect("tokens");
        let p = staged.at(FireTable::Positions).expect("positions");
        assert_eq!(p.address, t.address + t.bytes, "packed, in order");
    }
}
