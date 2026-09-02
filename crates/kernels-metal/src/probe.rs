//! A recording [`Encode`] sink — the host-side half of a kernel entry's
//! contract, made assertable with no GPU. Records the [`Fire`] and its
//! marshalled arguments; resolves no handle, allocates nothing, and never
//! claims a launch ran.

use core::cell::RefCell;

use crate::encode::{ArgValue, Encode, Fire};
use crate::error::Error;

/// One recorded launch: the shader named, and the arguments marshalled for
/// it in the order the entry stated them.
pub(crate) type Recorded = (Fire, Vec<ArgValue>);

#[derive(Default)]
pub(crate) struct Probe {
    fires: RefCell<Vec<Recorded>>,
}

impl Probe {
    /// Every launch this probe was handed, in order.
    pub(crate) fn fires(&self) -> Vec<Recorded> {
        self.fires.borrow().clone()
    }

}

impl Encode for Probe {
    fn fire(&self, fire: Fire, args: &[ArgValue]) -> Result<(), Error> {
        self.fires.borrow_mut().push((fire, args.to_vec()));
        Ok(())
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(u32::MAX))
    }
}
