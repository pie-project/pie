use core::cell::RefCell;

use crate::encode::{ArgValue, Encode, Fire};
use crate::error::Error;

pub(crate) type Recorded = (Fire, Vec<ArgValue>);

#[derive(Default)]
pub(crate) struct Probe {
    fires: RefCell<Vec<Recorded>>,
}

impl Probe {
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
        Ok(ArgValue::Buffer(crate::encode::ABSENT))
    }
}
