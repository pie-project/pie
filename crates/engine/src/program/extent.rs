use engine_api::program::{Axis, LaunchPlanValue};
use tensor_ir::DType;
use tensor_ir::types::MAX_RANK;

/// Which runtime quantity a symbolic axis resolves against.
///
/// The contract's [`ExtentRole`](engine_api::program::ExtentRole), re-exported
/// under the name this plane has always called it. It used to be declared
/// here, a third copy of a seven-variant enum that `tensor-compiler` also
/// declares (`SymbolicExtent`) and `engine-api` also carried as a `u8` tag
/// space; the wire tags were kept in step by nothing.
pub use engine_api::program::ExtentRole as Role;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extents {
    pub kv_len: u32,

    pub page_count: u32,

    pub row_count: u32,

    pub token_count: u32,

    pub sampled_rows: u32,

    pub query_len: u32,

    pub key_len: u32,
}

impl Default for Extents {
    fn default() -> Self {
        Extents {
            kv_len: 1,
            page_count: 1,
            row_count: 1,
            token_count: 1,
            sampled_rows: 1,
            query_len: 1,
            key_len: 1,
        }
    }
}

impl Extents {
    #[must_use]
    pub fn get(&self, role: Role) -> u32 {
        match role {
            Role::KvLen => self.kv_len,
            Role::PageCount => self.page_count,
            Role::RowCount => self.row_count,
            Role::TokenCount => self.token_count,
            Role::SampledRows => self.sampled_rows,
            Role::QueryLen => self.query_len,
            Role::KeyLen => self.key_len,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct ValueDesc {
    pub len: u32,

    pub rows: u32,

    pub last: u32,

    pub rank: u32,

    pub dtype: u32,

    pub dims: [u32; MAX_RANK],
}

const _: () = assert!(size_of::<ValueDesc>() == 36);

impl Default for ValueDesc {
    fn default() -> Self {
        ValueDesc {
            len: 1,
            rows: 1,
            last: 1,
            rank: 0,
            dtype: 0,
            dims: [1; MAX_RANK],
        }
    }
}

impl ValueDesc {
    #[must_use]
    pub fn device_bytes(&self) -> u64 {
        let len = u64::from(self.len);
        let bytes = if self.dtype() == DType::Bool {
            len
        } else {
            len * 4
        };
        bytes.max(4)
    }

    #[must_use]
    pub fn wire_bytes(&self) -> u64 {
        super::value::wire_cell_bytes(self.dtype(), self.len as usize) as u64
    }

    /// The element type this descriptor's `dtype` word names.
    ///
    /// `dtype` is a `u32` because the struct is `#[repr(C)]` and read by a
    /// device kernel; `F32` for a word no dtype claims, which is what the
    /// `concrete_dtype` round trip this replaced also answered.
    #[must_use]
    fn dtype(&self) -> DType {
        u8::try_from(self.dtype)
            .ok()
            .and_then(DType::from_wire)
            .unwrap_or(DType::F32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unresolvable {
    Rank { rank: usize, limit: usize },

    ZeroExtent { axis: usize },

    Overflow { axis: usize },
}

pub fn describe(value: &LaunchPlanValue, extents: &Extents) -> Result<ValueDesc, Unresolvable> {
    // `Unresolvable::Mismatch` used to stand here, guarding two parallel
    // vectors — `extents: Vec<u8>` and `dims: Vec<u32>` — against being
    // different lengths. `LaunchPlanValue::axes` is one vector of `Axis`, so
    // there is nothing left to disagree with itself. `UnknownRole` went the
    // same way: a role is an enum now, not a byte with a sentinel carved out
    // of it.
    let rank = value.axes.len();
    if rank > MAX_RANK {
        return Err(Unresolvable::Rank {
            rank,
            limit: MAX_RANK,
        });
    }

    let mut descriptor = ValueDesc {
        rank: rank as u32,
        dtype: value.dtype as u32,
        ..ValueDesc::default()
    };

    let mut len: u32 = 1;
    for (axis, &entry) in value.axes.iter().enumerate() {
        let dim = match entry {
            Axis::Static(literal) => literal,
            Axis::Symbolic(role) => extents.get(role),
        };
        if dim == 0 {
            return Err(Unresolvable::ZeroExtent { axis });
        }
        len = len
            .checked_mul(dim)
            .ok_or(Unresolvable::Overflow { axis })?;
        descriptor.dims[axis] = dim;
    }

    descriptor.len = len;

    descriptor.rows = descriptor.dims[..rank.saturating_sub(1)].iter().product();
    descriptor.last = if rank == 0 {
        1
    } else {
        descriptor.dims[rank - 1]
    };
    Ok(descriptor)
}
