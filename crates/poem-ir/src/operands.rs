use crate::check::Expect;
use crate::value::ValueId;

pub trait Operands {
    fn inputs(&self, sink: &mut Vec<ValueId>);
    fn outputs(&self, sink: &mut Vec<ValueId>);
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>);
    fn name(&self) -> &'static str;
    /// The operands whose type the op fixes, by field name: what the
    /// validator checks beyond def-use. `#[derive(Op)]` writes it from
    /// `#[input(I32)]`-style field attributes.
    fn ports(&self, _sink: &mut Vec<(&'static str, ValueId, Expect)>) {}
}

/// What an `#[input]` or `#[output]` field holds: none, one or several
/// values.
pub trait Slot {
    fn each(&self, f: &mut dyn FnMut(ValueId));

    fn push(&self, sink: &mut Vec<ValueId>) {
        self.each(&mut |id| sink.push(id));
    }

    /// The single value an alias pairs; `None` when the slot is empty.
    fn one(&self) -> Option<ValueId> {
        let mut one = None;
        self.each(&mut |id| one = Some(id));
        one
    }
}

impl Slot for ValueId {
    fn each(&self, f: &mut dyn FnMut(ValueId)) {
        f(*self);
    }
}

impl<T: Slot> Slot for Option<T> {
    fn each(&self, f: &mut dyn FnMut(ValueId)) {
        if let Some(slot) = self {
            slot.each(f);
        }
    }
}

impl<T: Slot> Slot for [T] {
    fn each(&self, f: &mut dyn FnMut(ValueId)) {
        for slot in self {
            slot.each(f);
        }
    }
}

impl<T: Slot> Slot for Vec<T> {
    fn each(&self, f: &mut dyn FnMut(ValueId)) {
        self.as_slice().each(f);
    }
}

impl<T: Slot, const N: usize> Slot for [T; N] {
    fn each(&self, f: &mut dyn FnMut(ValueId)) {
        self.as_slice().each(f);
    }
}
