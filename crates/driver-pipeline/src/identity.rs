//! What the compile cache is keyed on.
//!
//! A compiled pipeline state is expensive enough that this shell caches it in
//! memory, on disk as an `mtl4archive`, and — for the failures worth not
//! retrying — negatively. All three are keyed on one string, and that string
//! has exactly one job: **two runs that would compile different code must never
//! produce the same key.** Every field below exists because it is something a
//! compilation depends on and that can change without the program changing.
//!
//! The key is a 22-byte record in hexadecimal, followed by a `-v` suffix
//! carrying the four version numbers in the clear. The split is not decorative.
//! The record is what identifies the compilation; the suffix is what a human
//! reads off a stale archive filename to see *which* version moved. Only the
//! compiler version appears in both.
//!
//! # The suffix is a concatenation with no separators
//!
//! `-v` is followed by four fixed-width hex fields run together. Nothing marks
//! their boundaries, so the widths must match the field types exactly. The C++
//! wrote them with `std::setw`, which **pads but never truncates**: a value too
//! wide for its field is printed at full width and shifts every field after it,
//! and the suffix becomes ambiguous — two different version tuples encoding to
//! one string, which is precisely the collision the key exists to prevent. The
//! C++ was safe from this only because its field widths happened to match its
//! `uint16_t`/`uint32_t` field types, not because anything checked. Here the
//! widths are derived from the types and a test pins that the encoding's length
//! is the same for all-zero versions as for all-max ones.
//!
//! # Why the emitter version is `u32` here and was `u16` there
//!
//! The C++ driver hardcoded `kMetalM1EmitterVersion = 23` — a copy, kept by
//! hand, of a number owned by the host emitter. The copy has already drifted:
//! the host is at 36. A driver-side copy of another component's version cannot
//! do the job the comment claims for it ("so a stale mtl4archive never survives
//! a host-side emitter bump"), because the host can bump without the driver
//! noticing. The launch ABI already carries the real number —
//! [`ProgramRegistration::emitter_version`], documented as "part of the
//! driver's compile-cache key, so a bump must miss rather than reuse" — and it
//! is a `u32`. Narrowing it to `u16` to fit the old field would be a silent
//! collision at 65536, so the field is a `u32` and its suffix width is eight.
//!
//! [`ProgramRegistration::emitter_version`]: driver::plan::ProgramRegistration::emitter_version

use std::fmt::Write as _;

use driver::plan::LaunchStagePlan;
use tensor_ir::fnv1a64;

/// Which device shell is compiling.
///
/// This was a hardcoded `BACKEND_METAL: u8 = 1` while the layer was a module
/// of the Metal shell, and it stopped being defensible the moment the CUDA
/// shell read the same file: a byte that only ever holds one value keys
/// nothing, and two shells writing one value would have their caches alias.
/// They would not collide in memory — the caches are per-process — but they
/// share `$PIE_HOME/cache`, and a cubin answering a request for a `.metallib`
/// is a class of failure worth making unrepresentable.
///
/// The discriminants are `model_loader::types::BackendKind`'s, written as
/// numbers rather than taken as a dependency: this crate would import the
/// loader for one enum, and the numbers are ABI either way — they are already
/// in every archive filename on disk.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Backend {
    /// `BackendKind::Cuda`. Compiles NVRTC cubins.
    Cuda = 0,
    /// `BackendKind::Metal`. Compiles pipeline state objects.
    Metal = 1,
}

/// The row bucket this shell compiles for.
///
/// The C++ wrote a literal zero with the comment "generic row bucket". It is
/// zero because this shell does not specialize a pipeline by row count; the
/// byte is reserved so that a shell which does can vary it without moving every
/// field after it.
const ROW_BUCKET_GENERIC: u8 = 0;

/// The lane-count bucket this shell compiles for. Generic, for the same reason
/// as [`ROW_BUCKET_GENERIC`].
const LANE_BUCKET_GENERIC: u8 = 0;

/// `SemanticMode::Exact` — this shell never compiles an approximating variant,
/// so the byte is constant. It is still in the key because a shell that gained
/// one would have to miss against every archive built by this one.
const SEMANTIC_EXACT: u8 = 0;

/// The version numbers a compilation depends on.
///
/// Each is owned by a component upstream of this driver and none can be derived
/// from the program, which is why they are a parameter rather than a constant:
/// this crate deliberately does not depend on the compiler, so it cannot read
/// the compiler's own version and must be told.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Versions {
    /// `tensor_compiler::plan::COMPILER_VERSION` — bumped when the normalized
    /// form itself changes shape.
    pub compiler: u16,
    /// `tensor_compiler::plan::REGION_PLAN_VERSION` — bumped when region
    /// partitioning changes.
    pub region_plan: u16,
    /// `tensor_compiler::plan::LANE_TABLE_ABI_VERSION` — the layout the lane
    /// table is written in, which the emitted kernels index directly.
    pub lane_table: u32,
    /// [`ProgramRegistration::emitter_version`](driver::plan::ProgramRegistration::emitter_version)
    /// — the host emitter that produced the kernels being compiled. Taken from
    /// the registration, never hardcoded; see the module docs.
    pub emitter: u32,
}

/// `tensor_compiler::plan::COMPILER_VERSION`, mirrored.
///
/// Mirrored rather than imported for the reason [`Versions`] gives: this crate
/// does not depend on the compiler, because the compiler produces what it
/// consumes and the dependency would run the wrong way. Mirrored rather than
/// left to each shell because there is no such thing as a per-backend answer —
/// it is one number, owned upstream — and two shells each writing their own
/// copy is two chances to be a version behind.
///
/// A hand-copied constant that nothing checks drifts, so a test compares it
/// against the compiler's through the dev-dependency, exactly as
/// [`lane::ABI_VERSION`](crate::LANE_ABI_VERSION) and `status::FAULT_CLASSES`
/// are checked.
pub const COMPILER_VERSION: u16 = 3;

/// `tensor_compiler::plan::REGION_PLAN_VERSION`, mirrored on the same terms as
/// [`COMPILER_VERSION`].
pub const REGION_PLAN_VERSION: u16 = 4;

impl Versions {
    /// The three compiler-side numbers from their mirrors, and the emitter
    /// from the wire.
    ///
    /// This is the constructor a driver shell should use, and the split it
    /// encodes is the whole point. Three of the four are facts about the
    /// toolchain that produced the program and are mirrored here, checked
    /// against the compiler by test. The fourth cannot be: the C++ hardcoded
    /// `kMetalM1EmitterVersion = 23` — a driver-side copy of a number the HOST
    /// owns — and it had already drifted to 36 by the time anyone looked. A
    /// copy of another process's version cannot do the job of noticing when
    /// that process changes, so `emitter` is a parameter and comes from
    /// [`PieProgramDesc::emitter_version`](driver::local::PieProgramDesc).
    #[must_use]
    pub const fn mirrored(emitter: u32) -> Self {
        Self {
            compiler: COMPILER_VERSION,
            region_plan: REGION_PLAN_VERSION,
            lane_table: crate::lane::ABI_VERSION,
            emitter,
        }
    }
}

/// How many bytes the identity record holds.
///
/// Summed from the fields rather than written as `22`, so widening a field is a
/// compile-time change to this number instead of a hand-recomputed literal and
/// an out-of-bounds write. The C++ wrote `std::array<std::uint8_t, 22>` and
/// indexed it at hand-computed offsets with `operator[]`, which is unchecked:
/// getting the arithmetic wrong there is undefined behaviour, not a panic.
const RECORD_BYTES: usize = size_of::<u8>()      // backend
    + size_of::<u64>()                            // device
    + size_of::<u16>()                            // compiler version
    + size_of::<u64>()                            // signature
    + size_of::<u8>()                             // row bucket
    + size_of::<u8>()                             // lane bucket
    + size_of::<u8>(); // semantic mode

/// The C++'s hand-summed literal, kept as an assertion rather than as the
/// definition. If a field is ever widened this fires and says so; if it never
/// is, it documents that the two agree.
const _: () = assert!(RECORD_BYTES == 22);

/// Fills a fixed record left to right so no offset is ever written by hand.
struct Record {
    bytes: [u8; RECORD_BYTES],
    at: usize,
}

impl Record {
    fn new() -> Self {
        Record {
            bytes: [0; RECORD_BYTES],
            at: 0,
        }
    }

    /// Append a field's little-endian bytes.
    ///
    /// Little-endian because the record is a key, not a sort order: nothing
    /// compares two of these for magnitude, and every producer and consumer of
    /// it runs on a little-endian machine, so the host's own byte order is the
    /// one with no conversion in it.
    fn put(&mut self, bytes: &[u8]) {
        self.bytes[self.at..self.at + bytes.len()].copy_from_slice(bytes);
        self.at += bytes.len();
    }

    /// The finished record, checked to have been filled exactly.
    ///
    /// # Panics
    ///
    /// If a field was left out. A short record would silently key two different
    /// compilations alike on whatever bytes remained zero.
    fn finish(self) -> [u8; RECORD_BYTES] {
        assert_eq!(
            self.at, RECORD_BYTES,
            "the identity record must be filled exactly; a gap is a cache collision"
        );
        self.bytes
    }
}

/// Encode the compile-cache identity for one program on one device.
///
/// `backend` is which shell is asking, because the two compile the same
/// program to different machine code and share a cache directory. `device` is
/// the registry id of the GPU — two GPUs of different families compile the
/// same source differently, so an archive is not portable between them.
/// `signature` is [`combined_signature`] over the program's stage plans.
#[must_use]
pub fn cache_identity(backend: Backend, device: u64, signature: u64, versions: Versions) -> String {
    let mut record = Record::new();
    record.put(&[backend as u8]);
    record.put(&device.to_le_bytes());
    record.put(&versions.compiler.to_le_bytes());
    record.put(&signature.to_le_bytes());
    record.put(&[ROW_BUCKET_GENERIC, LANE_BUCKET_GENERIC, SEMANTIC_EXACT]);
    let record = record.finish();

    // Two hex digits per record byte, then `-v` and the four version fields.
    let mut out = String::with_capacity(RECORD_BYTES * 2 + 2 + 4 + 4 + 8 + 8);
    for byte in record {
        // Writing to a `String` cannot fail; the `Result` is `fmt::Write`'s
        // signature, not a real outcome.
        let _ = write!(out, "{byte:02x}");
    }
    let _ = write!(
        out,
        "-v{:04x}{:04x}{:08x}{:08x}",
        versions.compiler, versions.region_plan, versions.lane_table, versions.emitter
    );
    out
}

/// Hash a program's stage signatures into the one number the identity carries.
///
/// The stages are folded in order, and order is part of the identity: the same
/// stages composed differently are a different program and must compile to
/// different code. FNV-1a is [`tensor_ir::fnv1a64`], which the CUDA driver and
/// the host program cache also use, so the three agree on what a program is.
///
/// A program with no stages hashes to the FNV offset basis rather than to zero.
/// That is the correct answer — it is the hash of the empty byte string — and
/// it is distinct from any real program's, which is all the key needs.
#[must_use]
pub fn combined_signature(plans: &[LaunchStagePlan]) -> u64 {
    let bytes: Vec<u8> = plans
        .iter()
        .flat_map(|plan| plan.signature_hash.to_le_bytes())
        .collect();
    fnv1a64(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan(signature_hash: u64) -> LaunchStagePlan {
        LaunchStagePlan {
            signature_hash,
            ..LaunchStagePlan::default()
        }
    }

    /// The exact string the C++ produced, for one fixed input, so the port can
    /// be checked against the format rather than only against itself. Byte for
    /// byte: backend `01`, the device little-endian, the compiler version, the
    /// signature little-endian, then the three constant bytes.
    #[test]
    fn the_record_is_the_fields_in_order_little_endian() {
        let identity = cache_identity(
            Backend::Metal,
            0x0011_2233_4455_6677,
            0x8899_aabb_ccdd_eeff,
            Versions {
                compiler: 3,
                region_plan: 4,
                lane_table: 3,
                emitter: 36,
            },
        );
        let (record, suffix) = identity.split_at(RECORD_BYTES * 2);
        assert_eq!(
            record,
            concat!(
                "01",               // BackendKind::Metal
                "7766554433221100", // device, little-endian
                "0300",             // compiler version, little-endian u16
                "ffeeddccbbaa9988", // signature, little-endian
                "00",               // generic row bucket
                "00",               // generic lane-count bucket
                "00",               // SemanticMode::Exact
            )
        );
        assert_eq!(suffix, "-v000300040000000300000024");
    }

    /// The suffix has no separators, so a field that outgrew its width would
    /// shift the ones after it and let two version tuples collide. Pinning the
    /// length at both extremes is what says the widths match the types.
    #[test]
    fn the_encoding_is_fixed_width_for_every_version_a_field_can_hold() {
        let zero = cache_identity(Backend::Metal, 0, 0, Versions::default());
        let max = cache_identity(
            Backend::Metal,
            u64::MAX,
            u64::MAX,
            Versions {
                compiler: u16::MAX,
                region_plan: u16::MAX,
                lane_table: u32::MAX,
                emitter: u32::MAX,
            },
        );
        assert_eq!(
            zero.len(),
            max.len(),
            "a version too wide for its field is not truncated, it shifts every \
             field after it and makes the suffix ambiguous"
        );
        assert_eq!(zero.len(), RECORD_BYTES * 2 + 2 + 4 + 4 + 8 + 8);
    }

    /// The whole point of the key: anything a compilation depends on must move
    /// it. One test per input, because a field left out of the record is
    /// invisible until an archive built by one version is loaded by another.
    #[test]
    fn every_input_changes_the_identity() {
        let versions = Versions {
            compiler: 3,
            region_plan: 4,
            lane_table: 3,
            emitter: 36,
        };
        let base = cache_identity(Backend::Metal, 7, 9, versions);
        let variants = [
            cache_identity(Backend::Cuda, 7, 9, versions),
            cache_identity(Backend::Metal, 8, 9, versions),
            cache_identity(Backend::Metal, 7, 10, versions),
            cache_identity(
                Backend::Metal,
                7,
                9,
                Versions {
                    compiler: 4,
                    ..versions
                },
            ),
            cache_identity(
                Backend::Metal,
                7,
                9,
                Versions {
                    region_plan: 5,
                    ..versions
                },
            ),
            cache_identity(
                Backend::Metal,
                7,
                9,
                Versions {
                    lane_table: 4,
                    ..versions
                },
            ),
            cache_identity(
                Backend::Metal,
                7,
                9,
                Versions {
                    emitter: 37,
                    ..versions
                },
            ),
        ];
        for variant in variants {
            assert_ne!(
                base, variant,
                "an input that does not move the key is not in it"
            );
        }
    }

    /// An emitter version past `u16` is exactly where the C++'s field width
    /// would have started colliding. It does not here.
    #[test]
    fn an_emitter_version_past_u16_does_not_alias_its_low_half() {
        let low = Versions {
            emitter: 36,
            ..Versions::default()
        };
        let high = Versions {
            emitter: 36 + (1 << 16),
            ..Versions::default()
        };
        assert_ne!(
            cache_identity(Backend::Metal, 0, 0, low),
            cache_identity(Backend::Metal, 0, 0, high)
        );
    }

    /// The one test that makes the mirrors safe. A hand-copied version number
    /// that nothing checks drifts — the C++ proved it, with a driver-side
    /// `kMetalM1EmitterVersion = 23` that was still 23 when the host reached
    /// 36 — and a stale version in the key does not fail loudly: it makes a
    /// cache HIT that should have been a miss, so the driver runs code the
    /// current compiler would not have produced.
    #[test]
    fn the_mirrored_versions_still_match_the_compilers() {
        assert_eq!(
            COMPILER_VERSION,
            tensor_compiler::plan::COMPILER_VERSION,
            "the normalized form changed shape and this mirror did not"
        );
        assert_eq!(
            REGION_PLAN_VERSION,
            tensor_compiler::plan::REGION_PLAN_VERSION,
            "region partitioning changed and this mirror did not"
        );
        assert_eq!(
            crate::lane::ABI_VERSION,
            tensor_compiler::plan::lane_table::LANE_TABLE_ABI_VERSION,
            "the lane table's layout changed and this mirror did not"
        );
    }

    /// The emitter is the one field that must NOT be mirrored: it belongs to
    /// the host, which can bump it without this driver being rebuilt. The
    /// constructor takes it and everything else is a mirror, and that split is
    /// the correction to the C++'s hardcoded copy.
    #[test]
    fn only_the_emitter_version_comes_from_the_caller() {
        let versions = Versions::mirrored(41);
        assert_eq!(versions.emitter, 41);
        assert_eq!(versions.compiler, COMPILER_VERSION);
        assert_eq!(versions.region_plan, REGION_PLAN_VERSION);
        assert_eq!(versions.lane_table, crate::lane::ABI_VERSION);
        assert_ne!(
            cache_identity(Backend::Cuda, 0, 0, Versions::mirrored(41)),
            cache_identity(Backend::Cuda, 0, 0, Versions::mirrored(42)),
            "a host-side emitter bump must miss the cache, which is the whole              reason the number crosses the ABI instead of being written here"
        );
    }

    /// Stage order is program identity: the same stages in another order are
    /// another program and must not share a compilation.
    #[test]
    fn stage_order_is_part_of_the_combined_signature() {
        let forward = combined_signature(&[plan(1), plan(2)]);
        let backward = combined_signature(&[plan(2), plan(1)]);
        assert_ne!(forward, backward);
    }

    /// The empty program hashes to the FNV offset basis, not to zero -- the
    /// hash of no bytes, which is a distinct value and therefore a usable key.
    #[test]
    fn a_program_with_no_stages_hashes_to_the_offset_basis() {
        assert_eq!(combined_signature(&[]), 0xcbf2_9ce4_8422_2325);
        assert_ne!(combined_signature(&[]), combined_signature(&[plan(0)]));
    }

    /// The fold is over the signatures' little-endian bytes and nothing else,
    /// which is what lets the CUDA driver and the host reach the same number
    /// from the same plans.
    #[test]
    fn the_combined_signature_is_fnv1a_over_the_little_endian_signatures() {
        let plans = [plan(0x0102_0304_0506_0708), plan(0xdead_beef_dead_beef)];
        let mut bytes = Vec::new();
        for stage in &plans {
            bytes.extend_from_slice(&stage.signature_hash.to_le_bytes());
        }
        assert_eq!(combined_signature(&plans), fnv1a64(&bytes));
    }
}
