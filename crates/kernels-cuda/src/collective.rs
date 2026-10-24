mod all_reduce;
mod combine;
mod dispatch;

#[cfg(feature = "cuda")]
pub(crate) use all_reduce::answered;
pub use all_reduce::{Peers, SIGNAL_BYTES, all_gather, all_reduce, grouped, reduce_scatter};
pub use combine::combine;
pub use dispatch::{
    EP_MAX_RANKS, EP_SIGNAL_BYTES, Ep, Landing, Placement, Tokens, dispatch, ep_scratch_bytes,
};

/// Loads every dispatch and combine kernel on the current device. A module
/// load may wait on the device, where a peer may already spin in a barrier
/// for this rank: an expert-parallel group loads them on each of its devices
/// before any rank fires one.
pub fn load_ep() -> Result<(), crate::error::Error> {
    const OP: &str = "collective.load_ep";
    #[cfg(feature = "cuda")]
    {
        for (file, name) in dispatch::KERNELS.iter().chain(&combine::KERNELS) {
            crate::jit::kernel(file, name)
                .resolve()
                .map_err(|fault| fault.at(OP))?;
        }
        Ok(())
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(crate::jit::runtimeless(OP))
    }
}
