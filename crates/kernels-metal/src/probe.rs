//! A recording [`Encode`] sink — the host-side half of a kernel entry's
//! contract, made assertable on a box with no GPU.
//!
//! Every entry in this crate answers two questions before any device sees
//! it: WHICH point it selected, and WHAT geometry it hands the driver. Both
//! are pure functions of the handles, and both are exactly what a stamp
//! ladder or a grid formula gets wrong. A probe stands where the driver
//! stands, records the [`Fire`] and its marshalled arguments, and lets a unit
//! test read them back — so `dense_bidirectional_bfloat16_d_128` being the
//! point a 72-wide head selects is a test rather than a comment.
//!
//! It is deliberately not a fake driver: it resolves no handle, allocates
//! nothing, and never claims a launch RAN. What a passing test here proves
//! is that the entry would have asked for the right thing.

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

    /// The one launch an entry that fires once was handed. Panics if the
    /// entry fired a different number of times, which is itself the claim
    /// most of these tests are making.
    pub(crate) fn only(&self) -> Recorded {
        let fires = self.fires();
        assert_eq!(fires.len(), 1, "expected exactly one launch");
        fires.into_iter().next().expect("one launch")
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
