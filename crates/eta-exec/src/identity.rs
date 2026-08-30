use std::fmt::Write as _;

use eta_compiler::codegen::launch::LaunchStagePlan;
use eta_ir::fnv1a64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Backend {
    Cuda = 0,

    Metal = 1,

    Vulkan = 2,
}

const ROW_BUCKET_GENERIC: u8 = 0;

const LANE_BUCKET_GENERIC: u8 = 0;

const SEMANTIC_EXACT: u8 = 0;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Versions {
    pub compiler: u16,

    pub region_plan: u16,

    pub lane_table: u32,

    pub emitter: u32,
}

pub use eta_compiler::plan::{COMPILER_VERSION, REGION_PLAN_VERSION};

impl Versions {
    #[must_use]
    pub const fn from_compiler(emitter: u32) -> Self {
        Self {
            compiler: COMPILER_VERSION,
            region_plan: REGION_PLAN_VERSION,
            lane_table: super::lane::ABI_VERSION,
            emitter,
        }
    }
}

const RECORD_BYTES: usize = size_of::<u8>()
    + size_of::<u64>()
    + size_of::<u16>()
    + size_of::<u64>()
    + size_of::<u8>()
    + size_of::<u8>()
    + size_of::<u8>();

const _: () = assert!(RECORD_BYTES == 22);

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

    fn put(&mut self, bytes: &[u8]) {
        self.bytes[self.at..self.at + bytes.len()].copy_from_slice(bytes);
        self.at += bytes.len();
    }

    fn finish(self) -> [u8; RECORD_BYTES] {
        assert_eq!(
            self.at, RECORD_BYTES,
            "the identity record must be filled exactly; a gap is a cache collision"
        );
        self.bytes
    }
}

#[must_use]
pub fn cache_identity(backend: Backend, device: u64, signature: u64, versions: Versions) -> String {
    let mut record = Record::new();
    record.put(&[backend as u8]);
    record.put(&device.to_le_bytes());
    record.put(&versions.compiler.to_le_bytes());
    record.put(&signature.to_le_bytes());
    record.put(&[ROW_BUCKET_GENERIC, LANE_BUCKET_GENERIC, SEMANTIC_EXACT]);
    let record = record.finish();

    let mut out = String::with_capacity(RECORD_BYTES * 2 + 2 + 4 + 4 + 8 + 8);
    for byte in record {
        let _ = write!(out, "{byte:02x}");
    }
    let _ = write!(
        out,
        "-v{:04x}{:04x}{:08x}{:08x}",
        versions.compiler, versions.region_plan, versions.lane_table, versions.emitter
    );
    out
}

#[must_use]
pub fn combined_signature(plans: &[LaunchStagePlan]) -> u64 {
    let bytes: Vec<u8> = plans
        .iter()
        .flat_map(|plan| plan.signature_hash.to_le_bytes())
        .collect();
    fnv1a64(&bytes)
}
