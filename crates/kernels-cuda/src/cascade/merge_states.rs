
use std::ffi::c_void;

use crate::jit::Ctx;
use kernels::Refusal;
use crate::jit::abi::bf16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[must_use]
pub enum Merged {

    Launched,
    Declined(Refusal),
}

impl Merged {

    pub fn expect_launched(self, what: &str) {
        if let Self::Declined(why) = self {
            panic!("{what}: the split-KV merge declined: {why}");
        }
    }

    #[must_use]
    pub fn launched(self) -> bool {
        matches!(self, Self::Launched)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VarLen {

    pub v: u64,
    pub s: u64,
    pub indptr: u64,
    pub v_merged: u64,
    pub s_merged: u64,
    pub max_seq_len: u32,
    pub seq_len: u64,
    pub num_heads: u32,
    pub head_dim: u32,
}

pub unsafe fn variable_length(job: VarLen, stream: *mut c_void) -> Merged {

    fn ptr<T>(addr: u64) -> *mut T {
    addr as usize as *mut T
    }

    let ctx = unsafe { Ctx::on(stream) };
    let fired = super::merge_states_varlen(
        &ctx,
        ptr::<bf16>(job.v),
        ptr::<f32>(job.s),
        ptr::<i32>(job.indptr),
        ptr::<bf16>(job.v_merged),
        ptr::<f32>(job.s_merged),
        job.max_seq_len,
        ptr::<u32>(job.seq_len),
        job.num_heads,
        job.head_dim,
    );
    match fired {
        Ok(()) => Merged::Launched,
        Err(why) => Merged::Declined(why),
    }
}
