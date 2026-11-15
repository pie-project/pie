//! What a suite load states of its device, from the load's own fit
//! (`Shell::need`, `Gpu::need`): a device that reports no memory size is
//! stated what the load's fit says it takes, never a byte count picked for
//! a model.

use engine_gpu::serve::Need;

/// The loads the suite holds on one device at once: a load takes one share
/// in this many of what the device has.
pub const LOADS: u64 = 4;

/// The bytes a load fitted as `need` takes to serve `seats` sequences,
/// `guests` chunks a segment a test places by hand: its weights and, in
/// each segment the pool is bound in, a working set a seat, a fire's span
/// and the guests' chunks.
///
/// The fit's own floor (the weights, one working set and a fire) does not
/// load: the seats a load lays outside the pool (readouts, inputs, staging,
/// the weight caches) are on the books before the pool is sized and are in
/// no figure of the fit, and a test moves slots between segments. They come
/// out of this room; a load they leave short of its floor is refused by
/// name (`engine::chunk::fit`).
pub fn of(need: Need, seats: u32, guests: u32) -> u64 {
    let segment =
        u64::from(seats) * u64::from(need.working_set) + u64::from(need.fire) + u64::from(guests);
    need.weights + kernels_slang::SEGMENTS as u64 * segment * need.chunk
}
