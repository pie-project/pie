use kernels_slang::Tensor;

#[derive(Clone, Debug, Default, Hash)]
pub struct SlotTable(pub Vec<Option<Tensor>>);
