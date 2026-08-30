use eta_compiler::codegen::launch::LaunchPlanValue;
use eta_compiler::plan::Dimension;
use eta_ir::Dtype;
use eta_ir::types::{MAX_RANK, from_wire, wire_dtype};

/// Which runtime quantity a symbolic axis resolves against.
///
/// [`eta_compiler::plan::SymbolicExtent`], re-exported under the name this
/// plane has always called it — a rename, not a copy. There were three of this
/// seven-variant enum: one declared here, one the launch package declared as
/// `ExtentRole`, and a `u8` tag space the contract carried, with nothing
/// keeping the wire tags in step. The declaration here went first; `ExtentRole`
/// went when the package stopped being declared in a crate that could not name
/// the planner's. One is left, and this alias is a spelling of it rather than a
/// fourth thing to keep right.
pub use eta_compiler::plan::SymbolicExtent as Role;

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
        let bytes = if self.dtype() == Dtype::Bool {
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
    fn dtype(&self) -> Dtype {
        u8::try_from(self.dtype)
            .ok()
            .and_then(from_wire)
            .unwrap_or(Dtype::F32)
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
    // different lengths. `LaunchPlanValue::axes` is one vector of
    // `Dimension`, so there is nothing left to disagree with itself.
    // `UnknownRole` went the same way: a role is an enum now, not a byte with
    // a sentinel carved out of it.
    let rank = value.axes.len();
    if rank > MAX_RANK {
        return Err(Unresolvable::Rank {
            rank,
            limit: MAX_RANK,
        });
    }

    let mut descriptor = ValueDesc {
        rank: rank as u32,
        dtype: u32::from(wire_dtype(value.dtype)),
        ..ValueDesc::default()
    };

    let mut len: u32 = 1;
    for (axis, &entry) in value.axes.iter().enumerate() {
        let dim = match entry {
            Dimension::Static(literal) => literal,
            Dimension::Symbolic(role) => extents.get(role),
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
