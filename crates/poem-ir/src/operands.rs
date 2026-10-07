use crate::value::ValueId;

pub trait Operands {
    fn inputs(&self, sink: &mut Vec<ValueId>);
    fn outputs(&self, sink: &mut Vec<ValueId>);
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>);
    fn name(&self) -> &'static str;
}
