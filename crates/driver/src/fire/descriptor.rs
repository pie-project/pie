//! The [`FireDescriptor`]: the window table as a struct, and the same table as
//! bytes.
//!
//! **THE ONE MUTABLE CHANNEL INTO A RECORDED GRAPH** (design §5). Everything
//! else a fire varies by — composition, window bounds, row counts, lane order,
//! the plan structs prepare hoists out — is absorbed here, which is what makes
//! "one immutable graph per bucket, recaptured never" a true sentence. A
//! kernel inside the graph does not know which lanes it got; it reads its
//! region's count and offset out of this table and returns immediately if the
//! count is zero.
//!
//! # Why a flat form at all
//!
//! Because the reader is a device. The eager walk reads the struct in place
//! and never packs anything; the CUDA shell writes [`pack`](FireDescriptor::pack)
//! into pinned memory and uploads it, once, in front of the launch. The bytes
//! are the interface (design §6's standing doctrine), so the layout lives HERE
//! — one implementation, in the shared crate, with a roundtrip test — rather
//! than once per shell with a drift test between them. That is the lesson
//! `program/lane.rs` learned the expensive way: five copies of a layout, each
//! safe only because something watched it.
//!
//! # The layout
//!
//! ```text
//! offset  bytes  field
//!      0      4  magic "FIRE"
//!      4      4  abi version
//!      8      4  rows          total token rows this fire carries
//!     12      4  lanes         how many lane records follow the class table
//!     16      4  bucket        the shape bucket, i.e. which graph
//!     20      4  classes       how many class records follow the header
//!     24      8  reserved      zero; keeps the class table 8-byte aligned
//!     32     16  class[0]      row_offset, rows, lane_offset, lanes
//!    ...     16  class[n-1]
//!    ...     24  lane[0]       word (8), source, class, row_offset, rows
//!    ...     24  lane[m-1]
//! ```
//!
//! Little-endian throughout, `u32` fields except the fact word — every device
//! this ships on is little-endian, and stating it beats deriving it from the
//! host's byte order at pack time. **Fixed-width records, no padding beyond
//! the reserved word**: a device indexes `class[i]` with a multiply, which it
//! cannot do against a table it has to parse.
//!
//! `ABI_VERSION` is checked on unpack and never negotiated. A shell and a
//! driver that disagree about the layout are a build mismatch, not a
//! compatibility case (`driver-api`'s note on the ABI-version-on-an-in-process
//! -call is the counterexample this deliberately is not: that number rode on
//! a call inside one process, this one rides on bytes that cross to a device
//! and back into a `--force-warn`-clean unpack).

use model_ir::ClassSet;

use crate::fire::Fault;
use crate::fire::compose::{ClassWindow, Composition, LaneRow, MaskSpan, WindowTable};
use crate::{Error, Result};

/// `"FIRE"`, big-endian in the spelling and little-endian on the wire — the
/// first four bytes of any descriptor.
pub const MAGIC: u32 = 0x4649_5245;

/// The layout's version. Bumped by any change to the table below; checked, not
/// negotiated.
pub const ABI_VERSION: u32 = 1;

/// Bytes before the class table.
pub const HEADER_BYTES: u64 = 32;

/// Bytes per class record: `row_offset, rows, lane_offset, lanes`.
pub const CLASS_BYTES: u64 = 16;

/// Bytes per lane record: `word, source, class, row_offset, rows`.
pub const LANE_BYTES: u64 = 24;

/// One fire's window table, as the walk and the shells read it.
///
/// A COMPOSITION MINUS THE PROVENANCE. [`Composition`] knows how it was built
/// — the submitted lanes, the class order it chose — and this is what survives
/// the trip to a device: counts, offsets, words. Building one is a clone and
/// nothing else, which is deliberate; the descriptor is what a shell keeps
/// resident and overwrites in place every fire, so it must be a plain table
/// with no borrow into the batch that produced it.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FireDescriptor {
    /// Total token rows.
    pub rows: u32,
    /// The shape bucket these rows round up to — which recorded graph runs.
    pub bucket: u32,
    /// One window per class of the artifact, indexed by class.
    pub classes: WindowTable,
    /// The lanes in fire order, carrying the permutation back to submission
    /// order.
    pub lanes: Vec<LaneRow>,
}

impl FireDescriptor {
    /// The descriptor for a composition.
    #[must_use]
    pub fn of(composition: &Composition) -> FireDescriptor {
        FireDescriptor {
            rows: composition.rows(),
            bucket: composition.bucket(),
            classes: composition.classes().clone(),
            lanes: composition.lanes().to_vec(),
        }
    }

    /// How many lanes this fire carries.
    #[must_use]
    pub fn lane_count(&self) -> u32 {
        self.lanes.len() as u32
    }

    /// How many token rows a node with this class mask runs over — the
    /// zero-row question, asked against the table that crossed to the device.
    #[must_use]
    pub fn rows_of(&self, mask: &ClassSet) -> u32 {
        self.classes.rows_of(mask)
    }

    /// The one row-and-lane interval a node with this class mask runs over —
    /// the window question, asked against the table that crossed to the
    /// device. [`WindowTable::span`] states what the two answers mean.
    ///
    /// # Errors
    ///
    /// The number of runs the mask covers, when that is more than one.
    pub fn span(&self, mask: &ClassSet) -> core::result::Result<Option<MaskSpan>, usize> {
        self.classes.span(mask)
    }

    /// How many bytes [`pack`](FireDescriptor::pack) will write.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        HEADER_BYTES
            + CLASS_BYTES * self.classes.len() as u64
            + LANE_BYTES * self.lanes.len() as u64
    }

    /// The descriptor, flat.
    #[must_use]
    pub fn pack(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes() as usize);
        put32(&mut out, MAGIC);
        put32(&mut out, ABI_VERSION);
        put32(&mut out, self.rows);
        put32(&mut out, self.lane_count());
        put32(&mut out, self.bucket);
        put32(&mut out, self.classes.len() as u32);
        out.extend_from_slice(&0u64.to_le_bytes());

        for window in self.classes.as_slice() {
            put32(&mut out, window.row_offset);
            put32(&mut out, window.rows);
            put32(&mut out, window.lane_offset);
            put32(&mut out, window.lanes);
        }
        for lane in &self.lanes {
            out.extend_from_slice(&lane.word.to_le_bytes());
            put32(&mut out, lane.source);
            put32(&mut out, lane.class);
            put32(&mut out, lane.row_offset);
            put32(&mut out, lane.rows);
        }
        out
    }

    /// The descriptor these bytes carry.
    ///
    /// WHAT IT CHECKS IS WHAT A DEVICE CANNOT. A malformed descriptor does not
    /// fault on the far side — it computes, over whatever rows the wrong
    /// numbers name — so the length has to be exact rather than sufficient,
    /// and the class rows have to add up to the total the header claims.
    ///
    /// # Errors
    ///
    /// [`Fault::Descriptor`], naming which of those it was.
    pub fn unpack(bytes: &[u8]) -> Result<FireDescriptor> {
        if (bytes.len() as u64) < HEADER_BYTES {
            return Err(refuse("shorter than a header"));
        }
        if take32(bytes, 0) != MAGIC {
            return Err(refuse("no FIRE magic in the first four bytes"));
        }
        if take32(bytes, 4) != ABI_VERSION {
            return Err(refuse("a descriptor ABI version this build does not speak"));
        }
        let rows = take32(bytes, 8);
        let lanes = u64::from(take32(bytes, 12));
        let bucket = take32(bytes, 16);
        let classes = u64::from(take32(bytes, 20));

        let want = HEADER_BYTES + CLASS_BYTES * classes + LANE_BYTES * lanes;
        if bytes.len() as u64 != want {
            return Err(refuse(
                "a length its own header disagrees with, so a record would be \
                 read half out of the next one",
            ));
        }

        let mut table = Vec::with_capacity(classes as usize);
        let mut counted: u64 = 0;
        for c in 0..classes {
            let at = (HEADER_BYTES + CLASS_BYTES * c) as usize;
            let window = ClassWindow {
                row_offset: take32(bytes, at),
                rows: take32(bytes, at + 4),
                lane_offset: take32(bytes, at + 8),
                lanes: take32(bytes, at + 12),
            };
            counted += u64::from(window.rows);
            table.push(window);
        }
        if counted != u64::from(rows) {
            return Err(refuse(
                "class windows that do not add up to the row count in its \
                 header",
            ));
        }

        let base = HEADER_BYTES + CLASS_BYTES * classes;
        let mut placed = Vec::with_capacity(lanes as usize);
        for l in 0..lanes {
            let at = (base + LANE_BYTES * l) as usize;
            placed.push(LaneRow {
                word: take64(bytes, at),
                source: take32(bytes, at + 8),
                class: take32(bytes, at + 12),
                row_offset: take32(bytes, at + 16),
                rows: take32(bytes, at + 20),
            });
        }

        Ok(FireDescriptor {
            rows,
            bucket,
            classes: WindowTable::new(table),
            lanes: placed,
        })
    }
}

fn refuse(what: &'static str) -> Error {
    Error::Fire(Fault::Descriptor { what })
}

fn put32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

/// The `u32` at `at`. Every caller has already checked the length against the
/// header, so a short read here is impossible; the `unwrap_or` is what that
/// impossibility costs, and it reads zero rather than panicking in a fire
/// path.
fn take32(bytes: &[u8], at: usize) -> u32 {
    bytes
        .get(at..at + 4)
        .and_then(|b| b.try_into().ok())
        .map_or(0, u32::from_le_bytes)
}

fn take64(bytes: &[u8], at: usize) -> u64 {
    bytes
        .get(at..at + 8)
        .and_then(|b| b.try_into().ok())
        .map_or(0, u64::from_le_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fire::compose::{Lane, compose};
    use crate::fire::fixture::{Build, fact};
    use crate::{Error, fire::Fault};
    use model_compiler::{Budgets, DeviceProfile, compile};
    use model_ir::Cond;

    fn budgets() -> Budgets {
        Budgets::new(8, 64)
    }

    /// The design §0 split, in four nodes.
    fn plan() -> Build {
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Cond::Always);
        let d = b.op(q, 4, fact(0));
        let p = b.op(q, 4, Cond::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Cond::not(fact(0)))], 4);
        let y = b.op(o, 4, Cond::Always);
        b.out(y);
        b
    }

    /// The §0 diagram's fire, as a descriptor.
    fn descriptor() -> FireDescriptor {
        let b = plan();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let lanes = [
            Lane::new(0, 7),
            Lane::new(0, 3),
            Lane::new(1, 1),
            Lane::new(1, 1),
            Lane::new(1, 1),
        ];
        FireDescriptor::of(&compose(&baked, &budgets(), &lanes).expect("composes"))
    }

    #[test]
    fn a_descriptor_survives_the_round_trip_whole() {
        let before = descriptor();
        let bytes = before.pack();

        assert_eq!(bytes.len() as u64, before.bytes());
        assert_eq!(
            bytes.len() as u64,
            HEADER_BYTES + CLASS_BYTES * 2 + LANE_BYTES * 5,
        );
        assert_eq!(FireDescriptor::unpack(&bytes), Ok(before));
    }

    #[test]
    fn the_header_says_fire_and_which_layout_it_is() {
        let bytes = descriptor().pack();
        assert_eq!(&bytes[0..4], &MAGIC.to_le_bytes());
        assert_eq!(&bytes[4..8], &ABI_VERSION.to_le_bytes());
        // rows, lanes, bucket, classes.
        assert_eq!(&bytes[8..12], &13u32.to_le_bytes());
        assert_eq!(&bytes[12..16], &5u32.to_le_bytes());
        assert_eq!(&bytes[16..20], &13u32.to_le_bytes());
        assert_eq!(&bytes[20..24], &2u32.to_le_bytes());
        // The reserved word is zero, and it is what keeps the class table
        // 8-byte aligned.
        assert_eq!(&bytes[24..32], &0u64.to_le_bytes());
    }

    #[test]
    fn the_table_a_device_reads_is_the_table_the_walk_reads() {
        let b = plan();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let fire = compose(
            &baked,
            &budgets(),
            &[Lane::new(0, 7), Lane::new(1, 1), Lane::new(1, 1)],
        )
        .expect("composes");
        let packed = FireDescriptor::unpack(&FireDescriptor::of(&fire).pack()).expect("unpacks");

        assert_eq!(packed.rows, fire.rows());
        assert_eq!(packed.lane_count(), fire.lane_count());
        for region in baked.template() {
            assert_eq!(
                packed.rows_of(&region.mask),
                fire.classes().rows_of(&region.mask),
                "the trip through the bytes changed a window",
            );
        }
    }

    #[test]
    fn bytes_that_are_not_a_descriptor_are_refused_and_named() {
        let good = descriptor().pack();

        assert!(matches!(
            FireDescriptor::unpack(&good[..16]),
            Err(Error::Fire(Fault::Descriptor { .. })),
        ));
        assert!(matches!(
            FireDescriptor::unpack(&[0u8; 8]),
            Err(Error::Fire(Fault::Descriptor { .. })),
        ));

        let mut foreign = good.clone();
        foreign[0] ^= 0xff;
        assert!(matches!(
            FireDescriptor::unpack(&foreign),
            Err(Error::Fire(Fault::Descriptor { .. })),
        ));

        let mut older = good.clone();
        older[4..8].copy_from_slice(&(ABI_VERSION + 1).to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&older),
            Err(Error::Fire(Fault::Descriptor { .. })),
        ));

        // A header that claims one more class than the bytes carry: the record
        // after it would be read half out of the lane table.
        let mut miscounted = good.clone();
        miscounted[20..24].copy_from_slice(&3u32.to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&miscounted),
            Err(Error::Fire(Fault::Descriptor { .. })),
        ));

        // Windows that do not add up to the total the header claims — the
        // corruption that does not fault on the far side, it computes.
        let mut wrong = good;
        wrong[8..12].copy_from_slice(&12u32.to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&wrong),
            Err(Error::Fire(Fault::Descriptor { .. })),
        ));
    }

    #[test]
    fn an_empty_fire_still_packs_a_full_class_table() {
        // Zero lanes, and every class still has a window — a zero one. That is
        // what makes "an empty window is a count, not an absence" true on the
        // wire as well as in the struct.
        let b = plan();
        let baked = compile(&b.plan, &budgets(), &DeviceProfile::default()).expect("bakes");
        let fire = compose(&baked, &budgets(), &[]).expect("composes");
        let descriptor = FireDescriptor::of(&fire);

        assert_eq!(descriptor.classes.len(), 2);
        assert_eq!(descriptor.rows, 0);
        assert!(descriptor.lanes.is_empty());
        assert_eq!(
            FireDescriptor::unpack(&descriptor.pack()),
            Ok(descriptor.clone()),
        );
        assert_eq!(descriptor.bytes(), HEADER_BYTES + CLASS_BYTES * 2);
    }
}
