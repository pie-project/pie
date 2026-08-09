//! The memory a plan never mentions.
//!
//! Everything else in this crate reads a statement and works out what it
//! means. This module holds what no statement can: the paged KV cache, and
//! the tables a fire assembles for the rows it is about to run.
//!
//! The distinction is not a matter of taste. A plan is compiled once and run
//! against many deployments, so any number that belongs to a DEPLOYMENT --
//! how many pages the cache has, how many rows a page holds, which pages a
//! request happens to own -- cannot be in it. A text that stated its page
//! size would be right for one server and silently wrong for the next, which
//! is why the kernel rows name these as [`Source`](kernels::Source)s and ask
//! the driver rather than reading them off the statement.
//!
//! # The cache layout is the shaders', not this module's
//!
//! `attn/kv_write.comp` writes
//!
//! ```text
//! slot = page[i] * page_size + off[i]
//! at   = slot * (kv_heads * head_dim) + h * head_dim + d
//! ```
//!
//! and `attn/sdpa_paged.comp` reads
//! `(slot * n_kv_heads + kv_head) * head_dim + d_out`, the same expression. Two modules compiled separately
//! from separate sources agree on it, so this file transcribes a fact rather
//! than choosing a convention, and [`Shape::slot`] is where a driver can ask
//! for the arithmetic instead of repeating it.

use crate::binding::{FireNumber, FireTable, Resolve};
use crate::device::{Buffer, Device, Failed};
use model_compiler::trace::ValueId;
use std::collections::BTreeMap;

/// What a deployment decided about its cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    /// How many layers the model has. One key and one value buffer each.
    pub layers: u16,
    /// Key/value heads, which is what the cache is wide in.
    pub kv_heads: u32,
    /// Elements per head.
    pub head_dim: u32,
    /// Rows per page.
    ///
    /// The number `Source::KvPageSize` asks for, and the one a statement
    /// cannot carry.
    pub page_size: u32,
    /// How many pages the pool holds, across all requests.
    pub pages: u32,
    /// Bytes per element. Two for the `bfloat16` cache every current
    /// entrypoint is built for.
    pub bytes: u32,
}

impl Shape {
    /// Elements in one row of the cache, across every head.
    #[must_use]
    pub const fn row(&self) -> u64 {
        self.kv_heads as u64 * self.head_dim as u64
    }

    /// Bytes in one layer's key cache, which is also one layer's value cache.
    #[must_use]
    pub const fn layer_bytes(&self) -> u64 {
        self.elements() * self.bytes as u64
    }

    /// Where the element `(page, offset, head, at)` lives, in ELEMENTS.
    ///
    /// Transcribed from the two shaders rather than chosen. A driver that
    /// needs to read a row out of the cache -- to check it, to evict it, to
    /// copy it -- should ask here instead of writing the expression again,
    /// because writing it again is how the two copies come to disagree.
    #[must_use]
    pub const fn slot(&self, page: u32, offset: u32, head: u32, at: u32) -> u64 {
        let slot = page as u64 * self.page_size as u64 + offset as u64;
        slot * self.row() + head as u64 * self.head_dim as u64 + at as u64
    }

    /// Elements the pool holds in one layer, which is one past the largest
    /// index [`Shape::slot`] can return.
    #[must_use]
    pub const fn elements(&self) -> u64 {
        self.pages as u64 * self.page_size as u64 * self.row()
    }

    /// The number a row asks the driver for, or `None` if it does not fit the
    /// channel the push block carries it in.
    ///
    /// Here rather than on [`Pool`] because none of it needs a device, and a
    /// claim about the cache's arithmetic that can only be made against a card
    /// gets checked on the handful of positions a card test has time for.
    ///
    /// The two strides are in ELEMENTS: `attn/kv_write.comp` adds them to an
    /// index, not to a byte offset. Which of them is which is fixed by
    /// [`Shape::slot`] and not free -- see
    /// `the_two_stride_numbers_are_the_only_pair_that_agrees_with_slot`.
    #[must_use]
    pub fn number(&self, which: FireNumber) -> Option<u32> {
        match which {
            FireNumber::KvPageSize => Some(self.page_size),
            FireNumber::KvHeadStride => Some(self.head_dim),
            FireNumber::KvSeqStride => u32::try_from(self.row()).ok(),
        }
    }
}

/// The driver's own memory for one fire.
///
/// Holds the cache, which outlives a fire, and the tables, which do not. Kept
/// together because a [`Resolve`] has to answer for both and a caller holding
/// two objects would eventually hand a kernel one fire's tables and another
/// fire's cache.
pub struct Pool {
    shape: Shape,
    keys: Vec<Buffer>,
    values: Vec<Buffer>,
    tables: BTreeMap<FireTable, Buffer>,
    /// The stand-in for a weight or a seam value, if the caller gave one.
    named: Option<Buffer>,
}

impl Pool {
    /// Allocate one key and one value buffer per layer.
    ///
    /// Zeroed. A cache that came up holding the previous fire's rows would
    /// produce attention over sequences nobody asked about, and the attention
    /// would look plausible.
    ///
    /// # Errors
    ///
    /// [`Failed`] from the first allocation that does not fit.
    pub fn open(device: &Device, shape: Shape) -> Result<Self, Failed> {
        let zeros = vec![0u8; usize::try_from(shape.layer_bytes()).unwrap_or(usize::MAX)];
        let mut keys = Vec::with_capacity(shape.layers as usize);
        let mut values = Vec::with_capacity(shape.layers as usize);
        // Freed on the way out of a partial failure: an allocator that leaks
        // the layers it did get is an allocator whose second call fails for a
        // reason that has nothing to do with the second call.
        for _ in 0..shape.layers {
            match device.buffer(&zeros) {
                Ok(b) => keys.push(b),
                Err(e) => {
                    for b in keys.into_iter().chain(values) {
                        device.free(b);
                    }
                    return Err(e);
                }
            }
            match device.buffer(&zeros) {
                Ok(b) => values.push(b),
                Err(e) => {
                    for b in keys.into_iter().chain(values) {
                        device.free(b);
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self {
            shape,
            keys,
            values,
            tables: BTreeMap::new(),
            named: None,
        })
    }

    /// What the cache was built to.
    #[must_use]
    pub const fn shape(&self) -> Shape {
        self.shape
    }

    /// Give the pool one of the fire's tables, replacing any it held.
    ///
    /// Takes the words rather than a buffer so that the pool owns every
    /// allocation it hands out. A caller that kept the buffer could free it
    /// while a command buffer still named it, which is a use-after-free the
    /// layer reports and the caller does not.
    ///
    /// # Errors
    ///
    /// [`Failed`] if the table does not allocate.
    pub fn state(
        &mut self,
        device: &Device,
        which: FireTable,
        words: &[u32],
    ) -> Result<(), Failed> {
        let mut bytes = Vec::with_capacity(words.len() * 4);
        for w in words {
            bytes.extend_from_slice(&w.to_le_bytes());
        }
        let buffer = device.buffer(&bytes)?;
        if let Some(old) = self.tables.insert(which, buffer) {
            device.free(old);
        }
        Ok(())
    }

    /// A single buffer standing in for every weight and seam value.
    ///
    /// A driver that has loaded a model answers those from its own tables;
    /// this exists so that a caller exercising the cache does not have to.
    ///
    /// # Errors
    ///
    /// [`Failed`] if it does not allocate.
    pub fn stand_in(&mut self, device: &Device, bytes: u64) -> Result<(), Failed> {
        let buffer = device.buffer(&vec![0u8; usize::try_from(bytes).unwrap_or(0)])?;
        if let Some(old) = self.named.replace(buffer) {
            device.free(old);
        }
        Ok(())
    }

    /// One layer's cache, for a caller that wants to read it back.
    #[must_use]
    pub fn cache(&self, layer: u16, values: bool) -> Option<&Buffer> {
        let side = if values { &self.values } else { &self.keys };
        side.get(layer as usize)
    }

    /// Give every allocation back.
    ///
    /// Not [`Drop`]: freeing a Vulkan buffer needs the device that made it,
    /// and a `Drop` that cannot reach one either stores a handle it must not
    /// outlive or leaks. Stated as a call so the leak is the caller's to
    /// avoid rather than this module's to hide.
    pub fn close(self, device: &Device) {
        for b in self
            .keys
            .into_iter()
            .chain(self.values)
            .chain(self.tables.into_values())
            .chain(self.named)
        {
            device.free(b);
        }
    }
}

impl Resolve for Pool {
    fn weight(&self, _name: &str) -> Option<&Buffer> {
        self.named.as_ref()
    }

    fn named(&self, _value: ValueId) -> Option<&Buffer> {
        self.named.as_ref()
    }

    fn kv(&self, layer: u16, values: bool) -> Option<&Buffer> {
        self.cache(layer, values)
    }

    fn table(&self, which: FireTable) -> Option<&Buffer> {
        self.tables.get(&which)
    }

    fn number(&self, which: FireNumber) -> Option<u32> {
        self.shape.number(which)
    }
}

#[cfg(test)]
mod tests {
    use super::{FireNumber, Shape};

    /// Small enough to walk exhaustively, and no two dimensions equal, so a
    /// transposition of any two of them shows.
    const SMALL: Shape = Shape {
        layers: 2,
        kv_heads: 3,
        head_dim: 5,
        page_size: 4,
        pages: 7,
        bytes: 2,
    };

    /// Every element of the pool is one element of the pool.
    ///
    /// A scatter is only safe if this holds: `kv_write` computes a
    /// destination per invocation and never checks for a collision, so two
    /// distinct positions sharing an index would have one of them silently
    /// overwrite the other. Walked exhaustively over the whole small pool,
    /// which is what a card test cannot afford.
    #[test]
    fn every_element_of_the_cache_has_exactly_one_address() {
        let n = usize::try_from(SMALL.elements()).expect("small");
        let mut seen = vec![false; n];
        for page in 0..SMALL.pages {
            for off in 0..SMALL.page_size {
                for head in 0..SMALL.kv_heads {
                    for at in 0..SMALL.head_dim {
                        let ix = SMALL.slot(page, off, head, at);
                        let ix = usize::try_from(ix).expect("in range");
                        assert!(
                            ix < n,
                            "({page}, {off}, {head}, {at}) is at {ix}, past the {n} the pool holds"
                        );
                        assert!(!seen[ix], "({page}, {off}, {head}, {at}) collides at {ix}");
                        seen[ix] = true;
                    }
                }
            }
        }
        // Onto, not merely into: a layout that left gaps would allocate memory
        // the cache can never use, and one that packed too tightly would have
        // collided above.
        assert!(seen.iter().all(|&s| s), "the layout leaves holes");
    }

    /// The two strides are the only pair that describes the same memory as
    /// [`Shape::slot`].
    ///
    /// `attn/kv_write.comp`'s contiguous half writes
    /// `h * k_head_stride + pos * k_seq_stride + d`; its paged half writes
    /// what `slot` says. Both are
    /// checked on a card over six positions and two heads. Here the same
    /// identity is walked over every position and head the small pool has,
    /// and every other assignment of the two numbers is checked to break it --
    /// the card test can only afford to try the swap.
    #[test]
    fn the_two_stride_numbers_are_the_only_pair_that_agrees_with_slot() {
        let head = SMALL
            .number(FireNumber::KvHeadStride)
            .expect("a head stride") as u64;
        let seq = SMALL.number(FireNumber::KvSeqStride).expect("a seq stride") as u64;
        let contiguous = |h: u64, pos: u64, d: u64, head: u64, seq: u64| h * head + pos * seq + d;
        let slots = SMALL.pages as u64 * SMALL.page_size as u64;
        for pos in 0..slots {
            for h in 0..u64::from(SMALL.kv_heads) {
                for d in 0..u64::from(SMALL.head_dim) {
                    let page = u32::try_from(pos).expect("small") / SMALL.page_size;
                    let off = u32::try_from(pos).expect("small") % SMALL.page_size;
                    let want = SMALL.slot(
                        page,
                        off,
                        u32::try_from(h).expect("small"),
                        u32::try_from(d).expect("small"),
                    );
                    assert_eq!(
                        contiguous(h, pos, d, head, seq),
                        want,
                        "position {pos}, head {h}, channel {d}"
                    );
                }
            }
        }
        // And no other pair does. Anything drawn from the shape's own numbers
        // is a plausible mistake -- the row's comment names one of them -- so
        // each is tried and each has to fail somewhere.
        let candidates = [
            u64::from(SMALL.head_dim),
            SMALL.row(),
            u64::from(SMALL.page_size),
            slots * u64::from(SMALL.head_dim),
            1,
        ];
        for &a in &candidates {
            for &b in &candidates {
                if (a, b) == (head, seq) {
                    continue;
                }
                let agrees = (0..slots).all(|pos| {
                    (0..u64::from(SMALL.kv_heads)).all(|h| {
                        (0..u64::from(SMALL.head_dim)).all(|d| {
                            let page = u32::try_from(pos).expect("small") / SMALL.page_size;
                            let off = u32::try_from(pos).expect("small") % SMALL.page_size;
                            contiguous(h, pos, d, a, b)
                                == SMALL.slot(
                                    page,
                                    off,
                                    u32::try_from(h).expect("small"),
                                    u32::try_from(d).expect("small"),
                                )
                        })
                    })
                });
                assert!(
                    !agrees,
                    "a head stride of {a} and a sequence stride of {b} also describe the cache, \
                     so the pair this driver states is not forced"
                );
            }
        }
    }

    /// A page size the cache is not a multiple of would put a row across the
    /// end of the buffer.
    #[test]
    fn the_buffer_is_exactly_as_big_as_the_addresses_it_has_to_hold() {
        let last = SMALL.slot(
            SMALL.pages - 1,
            SMALL.page_size - 1,
            SMALL.kv_heads - 1,
            SMALL.head_dim - 1,
        );
        assert_eq!(
            last + 1,
            SMALL.elements(),
            "the highest address and the allocation disagree"
        );
        assert_eq!(
            SMALL.layer_bytes(),
            SMALL.elements() * u64::from(SMALL.bytes),
            "bytes and elements disagree about the same buffer"
        );
    }

    /// A pool wider than four billion elements per layer cannot state its own
    /// sequence stride, and says so rather than wrapping.
    ///
    /// `Source::KvSeqStride` reaches the shader through a 32-bit channel --
    /// `PIE_STRIDE` is a `uvec2` whose low half is all the shaders read -- so
    /// a `row()` past `u32::MAX` has nowhere to go. Truncating it would put
    /// every position after the first at a wrong address, on a card, with no
    /// error anywhere.
    #[test]
    fn a_cache_too_wide_to_state_refuses_rather_than_wraps() {
        let wide = Shape {
            kv_heads: 1 << 20,
            head_dim: 1 << 13,
            ..SMALL
        };
        assert!(wide.row() > u64::from(u32::MAX), "the premise");
        assert_eq!(
            wide.number(FireNumber::KvSeqStride),
            None,
            "a stride that does not fit was handed over anyway"
        );
        // The narrow one still answers, so the refusal is about the width and
        // not about the method.
        assert_eq!(
            SMALL.number(FireNumber::KvSeqStride),
            Some(SMALL.kv_heads * SMALL.head_dim)
        );
    }
}
