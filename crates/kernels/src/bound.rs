use crate::plane::{Cache, Const, In, InOut, Out, Refusal};
use crate::points::{Form, Plane, Repr, Scalar, ScalarKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    In(usize),

    Out(usize),

    Const(usize),
}

pub trait BoundOp {
    type Plane: Plane + ?Sized;

    fn point(&self) -> &str;

    fn dtype(&self, at: Site) -> Result<ScalarKind, Refusal>;

    fn tin<T: Scalar>(&self, at: usize) -> Result<In<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    fn tout<T: Scalar>(&self, at: usize)
    -> Result<Out<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    fn tinout<T: Scalar>(
        &self,
        from: usize,
        to: usize,
    ) -> Result<InOut<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    fn tconst<T: Scalar>(
        &self,
        at: usize,
    ) -> Result<Const<<Self::Plane as Plane>::Tensor<T>>, Refusal>;

    fn form(&self, at: usize) -> Result<Form, Refusal>;

    fn bank<R: Repr>(&self, at: usize) -> Result<Const<<Self::Plane as Plane>::Bank<R>>, Refusal>;

    fn recurrent(&self) -> Result<Cache<<Self::Plane as Plane>::Recurrent>, Refusal>;

    fn pages(&self) -> Result<Cache<<Self::Plane as Plane>::Pages>, Refusal>;

    fn u32(&self, at: usize) -> Result<u32, Refusal>;

    fn f32(&self, at: usize) -> Result<f32, Refusal>;

    fn bool(&self, at: usize) -> Result<bool, Refusal>;

    fn layer(&self) -> Result<u32, Refusal>;
}
