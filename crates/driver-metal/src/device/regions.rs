//! Which buffer an address belongs to.
//!
//! # Why this has to exist
//!
//! This driver binds by **address**. `Slice { address, bytes }` is what a
//! `Resolver` answers, `MTL4ArgumentTable::setAddress` is what consumes it,
//! and the executor never learns which `MTLBuffer` a weight lives in — an
//! alias is two names for one address, so the address is the honest unit.
//!
//! An indirect command buffer does not work that way.
//! `MTLIndirectComputeCommand::setKernelBuffer:offset:atIndex:` takes a
//! **buffer object** and an offset, because a recorded command has to keep
//! its operand resident and an integer cannot do that.
//!
//! So recording a fire (`.wiki/driver/graph-metal.md` §5②) needs the inverse
//! of what the fire computes: given the address an operand resolved to, which
//! allocation is that, and how far into it. Every address in a fire comes
//! from a [`Handle`], and a `Handle` holds its buffer — so the inverse exists,
//! it has simply never been written down.
//!
//! # Why a registry and not a field on `Slice`
//!
//! Because it would make every `Resolver` carry a buffer it has no use for,
//! and because the answer is a property of the DEPLOYMENT rather than of the
//! fire: the weight arena, the KV pool's layers, the fire tables and the
//! activation arena are the same allocations for every fire that follows.
//! Registering them once and asking per record is the cheaper shape.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use crate::device::handle::Handle;
use crate::layout::region::Region as _;

/// One registered allocation.
struct Span {
    base: u64,
    len: u64,
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
}

/// Every allocation a fire's addresses may fall in.
#[derive(Default)]
pub struct Regions {
    spans: Vec<Span>,
    /// The stand-in for an operand that addresses NOTHING.
    ///
    /// `dispatch::bind` answers an unfilled slot with address zero — *"the
    /// bound operand at `at`, or one that addresses nothing… nothing rather
    /// than a skip: a skipped slot shifts every operand after it"* — and
    /// `encode` binds that zero into the argument table quite happily.
    ///
    /// A recorded command cannot: it binds a BUFFER. So a recording needs one
    /// real allocation to point those slots at, and the caller supplies it
    /// rather than the registry inventing one, because the caller is who
    /// knows it is never written.
    null: Option<Span>,
}

impl Regions {
    /// An empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `handle`'s span.
    ///
    /// Re-registering the same base replaces it, which is what a pooled
    /// region needs: the address is the same and the buffer is the same, and
    /// a caller that registers per fire should not grow the list.
    pub fn add(&mut self, handle: &Handle) {
        let base = handle.gpu_address();
        let span = Span {
            base,
            len: handle.len(),
            buffer: Retained::from(handle.buffer()),
        };
        match self.spans.iter_mut().find(|s| s.base == base) {
            Some(existing) => *existing = span,
            None => self.spans.push(span),
        }
    }

    /// Register the region that stands in for "addresses nothing".
    ///
    /// See `Regions::null`. Any allocation will do as long as nothing reads
    /// what a kernel writes there; the point is that it is a real buffer.
    pub fn set_null(&mut self, handle: &Handle) {
        self.null = Some(Span {
            base: handle.gpu_address(),
            len: handle.len(),
            buffer: Retained::from(handle.buffer()),
        });
    }

    /// The buffer holding `address`, and how far into it that is.
    ///
    /// Address zero answers the null region when one is registered — that is
    /// an unfilled slot, not a defect. Anything else in no registered
    /// allocation is `None`, which for a recording is a refusal rather than a
    /// default, because the alternative is a command bound to some other
    /// operand's buffer.
    #[must_use]
    pub fn resolve(&self, address: u64) -> Option<(&ProtocolObject<dyn MTLBuffer>, u64)> {
        if address == 0 {
            return self.null.as_ref().map(|s| (&*s.buffer, 0));
        }
        self.spans
            .iter()
            .find(|s| address >= s.base && address < s.base + s.len.max(1))
            .map(|s| (&*s.buffer, address - s.base))
    }

    /// How many allocations are registered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether nothing is registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

impl std::fmt::Debug for Regions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Regions")
            .field("spans", &self.spans.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::Regions;
    use crate::device::{Allocation, Context};

    /// An address resolves to the allocation it is in, and to an offset.
    #[test]
    fn an_address_names_its_buffer_and_how_far_in() {
        let Ok(context) = Context::new() else {
            return;
        };
        let a = Allocation::new(&context, 4096, "regions a").expect("a region");
        let b = Allocation::new(&context, 4096, "regions b").expect("a region");
        let mut regions = Regions::new();
        regions.add(&a);
        regions.add(&b);

        let (buffer, offset) = regions
            .resolve(a.gpu_address() + 128)
            .expect("an address inside a registered span resolves");
        assert_eq!(
            offset, 128,
            "the offset is into the allocation, not absolute"
        );
        assert!(
            std::ptr::eq(buffer, a.buffer()),
            "and it names the right one -- binding an operand to the wrong \
             buffer is a command that reads someone else's bytes"
        );

        // The other allocation, so the search is not "the first span".
        let (buffer, offset) = regions.resolve(b.gpu_address()).expect("b resolves too");
        assert_eq!(offset, 0);
        assert!(std::ptr::eq(buffer, b.buffer()));

        // An address in nothing registered is None rather than the nearest
        // span: a recording that guessed would bind a command to another
        // operand's buffer.
        assert!(regions.resolve(a.gpu_address() + 1_000_000).is_none());

        // Address ZERO is the unfilled slot `dispatch::bind` answers with, and
        // it needs a real buffer because a recorded command binds one. Until
        // a null region is registered it is refused like any other unknown.
        assert!(regions.resolve(0).is_none());
        regions.set_null(&b);
        let (buffer, offset) = regions.resolve(0).expect("zero is the null region now");
        assert_eq!(offset, 0);
        assert!(std::ptr::eq(buffer, b.buffer()));

        // Re-registering one region does not grow the list, which is what a
        // POOLED region needs -- the same address every fire.
        assert_eq!(regions.len(), 2);
        regions.add(&a);
        assert_eq!(regions.len(), 2, "the same allocation is one span");
    }

    /// MANY spans, because a real deployment has many.
    ///
    /// A serving driver registers the weight arena, the fire tables, the
    /// activation arena, the scalars — and **two spans per layer** for the KV
    /// pool's K and V, which is 48 on a 24-layer model and 120 on a 60-layer
    /// one. Resolution is a linear search over all of them, and every one of
    /// its answers becomes a `setKernelBuffer` on a recorded command: an
    /// address resolved to the wrong span is a kernel reading another layer's
    /// cache, silently.
    ///
    /// The two-span test above cannot see an off-by-one at a span boundary.
    /// This walks every registered region at its first byte, its middle and
    /// its LAST byte.
    #[test]
    fn every_span_resolves_at_its_edges_however_many_there_are() {
        use crate::layout::region::Region as _;
        let Ok(context) = Context::new() else {
            return;
        };
        // Deliberately mixed sizes: a pool's layers are one size and the
        // tables are another, and equal sizes would hide an arithmetic error
        // that a stride would not.
        let handles: Vec<_> = (0..48u64)
            .map(|i| Allocation::new(&context, 4096 + i * 256, "many").expect("a region"))
            .collect();
        let mut regions = Regions::new();
        for h in &handles {
            regions.add(h);
        }
        assert_eq!(regions.len(), handles.len());

        for (i, h) in handles.iter().enumerate() {
            let base = h.gpu_address();
            for probe in [0, h.len() / 2, h.len() - 1] {
                let (buffer, offset) = regions
                    .resolve(base + probe)
                    .unwrap_or_else(|| panic!("span {i} does not resolve at +{probe}"));
                assert_eq!(offset, probe, "span {i} answered the wrong offset");
                assert!(
                    std::ptr::eq(buffer, h.buffer()),
                    "span {i} at +{probe} resolved to ANOTHER allocation -- a \
                     recorded command would read someone else's bytes"
                );
            }
            // One past the end is NOT this span. It may be another
            // allocation's first byte, which is why the assertion is on the
            // buffer rather than on `is_none`.
            if let Some((buffer, _)) = regions.resolve(base + h.len()) {
                assert!(
                    !std::ptr::eq(buffer, h.buffer()),
                    "span {i} claims a byte past its own end"
                );
            }
        }
    }
}
