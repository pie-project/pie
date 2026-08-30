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
//!     24      4  patch_rows    total PATCH rows this fire carries
//!     28      4  patch_bucket  the patch rung, i.e. which tower graph
//!     32     16  class[0]      row_offset, rows, lane_offset, lanes
//!    ...     16  class[n-1]
//!    ...     24  lane[0]       word (8), source, class, row_offset, rows
//!    ...     24  lane[m-1]
//!  --- the patch trailer, present iff patch_rows > 0 ---
//!    ...     16  patch_class[0]  patch_offset, patches, image_offset, images
//!    ...     16  patch_class[n-1]
//!    ...     16  patch_lane[0]   patch_offset, patches, image_offset, images
//!    ...     16  patch_lane[m-1]
//! ```
//!
//! Little-endian throughout, `u32` fields except the fact word — every device
//! this ships on is little-endian, and stating it beats deriving it from the
//! host's byte order at pack time. **Fixed-width records, no padding**: a
//! device indexes `class[i]` with a multiply, which it cannot do against a
//! table it has to parse.
//!
//! # ABI 2, and what it costs a fire that carries no image
//!
//! **FOUR BYTES, AND THEY ARE THE VERSION NUMBER.** The reserved `u64` at 24
//! was always written as zero; ABI 2 splits it into two `u32`s that a
//! text-only fire also writes as zero, and the patch trailer is keyed on
//! `patch_rows > 0` rather than on a count of its own — so a fire with no
//! patch rows packs the same length, the same class table and the same lane
//! table it packed under ABI 1, byte for byte, and the only difference on the
//! wire is `4..8`. That is what makes multimodal gate (a) — "a fire with no
//! image lane is the fire this engine always fired" — an arithmetic property
//! of the layout rather than a measurement.
//!
//! The trailer is keyed on `patch_rows` and NOT on a second class count
//! because there is no second count to carry: the patch table has one window
//! per class of the same artifact and one record per lane of the same fire,
//! so the header's `classes` and `lanes` size both halves. A separate count
//! would be a number free to disagree with the one beside it.
//!
//! `ABI_VERSION` is checked on unpack and never negotiated. A shell and an
//! engine that disagree about the layout are a build mismatch, not a
//! compatibility case (`engine`'s note on the ABI-version-on-an-in-process
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
///
/// **1 → 2 AT M3**, for the patch header words and the patch trailer. A
/// descriptor carrying version 1 is REFUSED by name
/// ([`Fault::DescriptorAbi`]) and never regenerated: nothing persists a
/// descriptor — it is built and dropped once per fire — so the only way v1
/// bytes reach this unpack is a shell and an engine from two builds, and
/// filling in the fields the older side never wrote would be this one
/// pretending to know what it meant.
pub const ABI_VERSION: u32 = 2;

/// Bytes before the class table.
pub const HEADER_BYTES: u64 = 32;

/// Bytes per class record: `row_offset, rows, lane_offset, lanes`.
pub const CLASS_BYTES: u64 = 16;

/// Bytes per lane record: `word, source, class, row_offset, rows`.
pub const LANE_BYTES: u64 = 24;

/// Bytes per PATCH lane record: `patch_offset, patches, image_offset, images`.
///
/// Four words and not a second copy of the token record: the word, the source
/// and the class are properties of the LANE and are already on the wire once.
/// A second copy of them would be three numbers free to disagree with
/// themselves.
pub const PATCH_LANE_BYTES: u64 = 16;

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
    /// order — and, since ABI 2, each lane's place in BOTH seriations.
    pub lanes: Vec<LaneRow>,
    /// Total PATCH rows. Zero for every fire of a text-only artifact and for
    /// every text-only fire of a tower one, which is the same number and the
    /// true one in both cases.
    pub patch_rows: u32,
    /// How many IMAGES this fire carries — the patch axis's lane count.
    pub images: u32,
    /// The patch rung these patch rows round up to — which tower graph runs.
    pub patch_bucket: u32,
    /// One PATCH window per class of the artifact, indexed by class.
    ///
    /// **A SECOND TABLE OVER THE SAME CLASSES IN A DIFFERENT ORDER**, which
    /// is the whole of multimodal §5.1: patch rows and token rows do not
    /// break at the same places, so a region on the patch axis asks its
    /// window of this table and a region on the token axis asks it of the one
    /// above. All-zero, and packed as nothing at all, for a fire with no
    /// patch rows.
    pub patch_classes: WindowTable,
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
            patch_rows: composition.patch_rows(),
            images: composition.images(),
            patch_bucket: composition.patch_bucket(),
            patch_classes: composition.patch_classes().clone(),
        }
    }

    /// Does this fire carry the second row axis at all?
    ///
    /// THE ONE PREDICATE THE TRAILER IS KEYED ON, and the one a shell asks
    /// before it launches a tower exec: an axis-empty fire does not launch
    /// that unit (multimodal §1), and this is what "axis-empty" is.
    #[must_use]
    pub fn has_patches(&self) -> bool {
        self.patch_rows > 0
    }

    /// How many PATCH rows a node with this class mask runs over — the
    /// zero-row question on the second axis.
    #[must_use]
    pub fn patch_rows_of(&self, mask: &ClassSet) -> u32 {
        self.patch_classes.rows_of(mask)
    }

    /// The one patch-row-and-image interval a node with this class mask runs
    /// over.
    ///
    /// # Errors
    ///
    /// The number of runs the mask covers, when that is more than one.
    pub fn patch_span(&self, mask: &ClassSet) -> core::result::Result<Option<MaskSpan>, usize> {
        self.patch_classes.span(mask)
    }

    /// Every patch interval a node with this class mask runs over, into a
    /// buffer the caller keeps.
    pub fn patch_spans_into(&self, mask: &ClassSet, out: &mut Vec<MaskSpan>) {
        self.patch_classes.spans_into(mask, out);
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

    /// Every interval a node with this class mask runs over, ascending — the
    /// slow path's launches, asked against the table that crossed to the
    /// device. [`WindowTable::spans`] states what the list means.
    #[must_use]
    pub fn spans(&self, mask: &ClassSet) -> Vec<MaskSpan> {
        self.classes.spans(mask)
    }

    /// The same, into a buffer the caller keeps — what the walk asks once per
    /// region ([`WindowTable::spans_into`] says why it is not a `Vec`).
    pub fn spans_into(&self, mask: &ClassSet, out: &mut Vec<MaskSpan>) {
        self.classes.spans_into(mask, out);
    }

    /// How many bytes [`pack`](FireDescriptor::pack) will write.
    #[must_use]
    pub fn bytes(&self) -> u64 {
        HEADER_BYTES
            + CLASS_BYTES * self.classes.len() as u64
            + LANE_BYTES * self.lanes.len() as u64
            + if self.has_patches() {
                CLASS_BYTES * self.patch_classes.len() as u64
                    + PATCH_LANE_BYTES * self.lanes.len() as u64
            } else {
                0
            }
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
        put32(&mut out, self.patch_rows);
        put32(&mut out, self.patch_bucket);

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
        // THE TRAILER, AND NOTHING AT ALL WHEN THERE ARE NO PATCH ROWS. That
        // is what makes a text-only fire's bytes the bytes ABI 1 wrote.
        if self.has_patches() {
            for window in self.patch_classes.as_slice() {
                put32(&mut out, window.row_offset);
                put32(&mut out, window.rows);
                put32(&mut out, window.lane_offset);
                put32(&mut out, window.lanes);
            }
            for lane in &self.lanes {
                put32(&mut out, lane.patch_offset);
                put32(&mut out, lane.patches);
                put32(&mut out, lane.image_offset);
                put32(&mut out, lane.images);
            }
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
        // **REFUSED BY NAME AND NEVER REGENERATED.** The two numbers are in
        // the refusal because "an ABI mismatch" without them sends an
        // operator reading a hexdump.
        let saw = take32(bytes, 4);
        if saw != ABI_VERSION {
            return Err(Error::Fire(Fault::DescriptorAbi {
                saw,
                speaks: ABI_VERSION,
            }));
        }
        let rows = take32(bytes, 8);
        let lanes = u64::from(take32(bytes, 12));
        let bucket = take32(bytes, 16);
        let classes = u64::from(take32(bytes, 20));
        let patch_rows = take32(bytes, 24);
        let patch_bucket = take32(bytes, 28);

        let trailer = if patch_rows > 0 {
            CLASS_BYTES * classes + PATCH_LANE_BYTES * lanes
        } else {
            0
        };
        let want = HEADER_BYTES + CLASS_BYTES * classes + LANE_BYTES * lanes + trailer;
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
                ..LaneRow::default()
            });
        }

        // The trailer. Its two halves are checked the way the token halves
        // are: the patch windows have to add up to the patch row count the
        // header claims, because a patch table that does not is the same
        // corruption the token one is — it does not fault, it computes.
        let mut patch_table = vec![ClassWindow::default(); classes as usize];
        if patch_rows > 0 {
            let at_classes = base + LANE_BYTES * lanes;
            let mut counted: u64 = 0;
            for c in 0..classes {
                let at = (at_classes + CLASS_BYTES * c) as usize;
                let window = ClassWindow {
                    row_offset: take32(bytes, at),
                    rows: take32(bytes, at + 4),
                    lane_offset: take32(bytes, at + 8),
                    lanes: take32(bytes, at + 12),
                };
                counted += u64::from(window.rows);
                patch_table[c as usize] = window;
            }
            if counted != u64::from(patch_rows) {
                return Err(refuse(
                    "patch windows that do not add up to the patch row count in \
                     its header",
                ));
            }
            let at_lanes = at_classes + CLASS_BYTES * classes;
            for (l, lane) in placed.iter_mut().enumerate() {
                let at = (at_lanes + PATCH_LANE_BYTES * l as u64) as usize;
                lane.patch_offset = take32(bytes, at);
                lane.patches = take32(bytes, at + 4);
                lane.image_offset = take32(bytes, at + 8);
                lane.images = take32(bytes, at + 12);
            }
        }
        let images = placed.iter().map(|lane| lane.images).sum();

        Ok(FireDescriptor {
            rows,
            bucket,
            classes: WindowTable::new(table),
            lanes: placed,
            patch_rows,
            images,
            patch_bucket,
            patch_classes: WindowTable::new(patch_table),
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
    use model_compiler::{Budget, DeviceProfile, compile};
    use model_ir::Guard;

    fn budget() -> Budget {
        Budget::new(8, 64)
    }

    /// The design §0 split, in four nodes.
    fn plan() -> Build {
        let mut b = Build::new();
        let x = b.input(4);
        let q = b.op(x, 4, Guard::Always);
        let d = b.op(q, 4, fact(0));
        let p = b.op(q, 4, Guard::not(fact(0)));
        let o = b.merge(&[(d, fact(0)), (p, Guard::not(fact(0)))], 4);
        let y = b.op(o, 4, Guard::Always);
        b.out(y);
        b
    }

    /// The §0 diagram's fire, as a descriptor.
    fn descriptor() -> FireDescriptor {
        let b = plan();
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let lanes = [
            Lane::new(0, 7),
            Lane::new(0, 3),
            Lane::new(1, 1),
            Lane::new(1, 1),
            Lane::new(1, 1),
        ];
        FireDescriptor::of(&compose(&compiled, &budget(), &lanes).expect("composes"))
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
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let fire = compose(
            &compiled,
            &budget(),
            &[Lane::new(0, 7), Lane::new(1, 1), Lane::new(1, 1)],
        )
        .expect("composes");
        let packed = FireDescriptor::unpack(&FireDescriptor::of(&fire).pack()).expect("unpacks");

        assert_eq!(packed.rows, fire.rows());
        assert_eq!(packed.lane_count(), fire.lane_count());
        for region in compiled.template() {
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

        let mut newer = good.clone();
        newer[4..8].copy_from_slice(&(ABI_VERSION + 1).to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&newer),
            Err(Error::Fire(Fault::DescriptorAbi { .. })),
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
        let compiled = compile(&b.trace, &budget(), &DeviceProfile::default()).expect("bakes");
        let fire = compose(&compiled, &budget(), &[]).expect("composes");
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

    /// **THE ABI DECISION, STATED AS A TEST**: version 1 bytes are REFUSED by
    /// name, and never regenerated into version 2.
    ///
    /// A descriptor is built and dropped once per fire and nothing persists
    /// one, so v1 bytes can only reach a v2 unpack across a build mismatch —
    /// and this side filling in the two header words the other side never
    /// wrote would be it guessing a patch row count. The refusal carries both
    /// numbers so the mismatch is legible without a hexdump.
    #[test]
    fn a_version_one_descriptor_is_refused_by_name_and_not_regenerated() {
        // What ABI 1 wrote, exactly: the same header with a zero `u64` where
        // the two patch words now stand, and the same two tables.
        let before = descriptor();
        let mut v1 = before.pack();
        v1[4..8].copy_from_slice(&1u32.to_le_bytes());

        let refusal = FireDescriptor::unpack(&v1).expect_err("v1 is not v2");
        assert_eq!(
            refusal,
            Error::Fire(Fault::DescriptorAbi {
                saw: 1,
                speaks: ABI_VERSION,
            }),
        );
        let said = refusal.to_string();
        assert!(said.contains('1') && said.contains('2'), "{said}");
        assert!(said.contains("never negotiated"), "{said}");
    }

    /// **AND WHAT THE BUMP COSTS A TEXT-ONLY FIRE: FOUR BYTES, AND THEY ARE
    /// THE VERSION.** The reserved `u64` at 24 was zero and the two words
    /// that replaced it are zero too, and a fire with no patch rows packs no
    /// trailer — so the bytes ABI 2 writes for a text-only fire are the bytes
    /// ABI 1 wrote, at `4..8` and nowhere else.
    #[test]
    fn a_text_only_fire_packs_the_bytes_abi_one_packed() {
        let bytes = descriptor().pack();
        let mut as_v1 = bytes.clone();
        as_v1[4..8].copy_from_slice(&1u32.to_le_bytes());

        assert_eq!(bytes[..4], as_v1[..4]);
        assert_eq!(bytes[8..], as_v1[8..]);
        assert_eq!(
            bytes.len() as u64,
            HEADER_BYTES + CLASS_BYTES * 2 + LANE_BYTES * 5,
            "a fire with no patch rows carries no patch trailer",
        );
        assert_eq!(&bytes[24..32], &0u64.to_le_bytes(), "both patch words zero");
    }

    /// A fire that DOES carry patch rows round-trips both seriations, and the
    /// trailer is exactly the two tables the header's own counts size.
    #[test]
    fn a_patch_carrying_descriptor_survives_the_round_trip_whole() {
        use crate::fire::compose::compose_axes;
        use model_compiler::{Budgets, PatchLadder, compile_axes};

        let b = plan();
        let budgets = Budgets::of(budget()).with_patches(PatchLadder {
            max_patches: 64,
            buckets: vec![16, 64],
            max_images: 4,
        });
        // The plan states no patch row, so the artifact carries no patch
        // axis and no lane may submit an image — which is refusal (ii), and
        // it is checked by its own test. Here the descriptor is built by
        // hand, because what is under test is the LAYOUT and not the compose.
        let compiled = compile_axes(&b.trace, &budgets, &DeviceProfile::default()).expect("bakes");
        let fire = compose_axes(&compiled, &budgets, &[Lane::new(0, 7), Lane::new(1, 1)])
            .expect("composes");
        let mut before = FireDescriptor::of(&fire);
        before.patch_rows = 12;
        before.patch_bucket = 16;
        before.images = 3;
        before.patch_classes = WindowTable::new(vec![
            ClassWindow { row_offset: 0, rows: 8, lane_offset: 0, lanes: 2 },
            ClassWindow { row_offset: 8, rows: 4, lane_offset: 2, lanes: 1 },
        ]);
        before.lanes[0].patch_offset = 0;
        before.lanes[0].patches = 8;
        before.lanes[0].image_offset = 0;
        before.lanes[0].images = 2;
        before.lanes[1].patch_offset = 8;
        before.lanes[1].patches = 4;
        before.lanes[1].image_offset = 2;
        before.lanes[1].images = 1;

        let bytes = before.pack();
        assert_eq!(bytes.len() as u64, before.bytes());
        assert_eq!(
            bytes.len() as u64,
            HEADER_BYTES
                + CLASS_BYTES * 2
                + LANE_BYTES * 2
                + CLASS_BYTES * 2
                + PATCH_LANE_BYTES * 2,
        );
        assert_eq!(&bytes[24..28], &12u32.to_le_bytes());
        assert_eq!(&bytes[28..32], &16u32.to_le_bytes());
        assert_eq!(FireDescriptor::unpack(&bytes), Ok(before));
    }

    /// A patch table that does not add up to the header's patch row count is
    /// the corruption that computes rather than faults — refused for the
    /// token table's reason, on the second axis.
    #[test]
    fn patch_windows_that_do_not_add_up_are_refused() {
        let mut before = descriptor();
        before.patch_rows = 4;
        before.patch_bucket = 4;
        before.images = 1;
        before.patch_classes = WindowTable::new(vec![
            ClassWindow { row_offset: 0, rows: 4, lane_offset: 0, lanes: 1 },
            ClassWindow::default(),
        ]);
        let good = before.pack();
        assert!(FireDescriptor::unpack(&good).is_ok());

        let mut wrong = good;
        wrong[24..28].copy_from_slice(&5u32.to_le_bytes());
        assert!(matches!(
            FireDescriptor::unpack(&wrong),
            Err(Error::Fire(Fault::Descriptor { .. })),
        ));
    }
}
