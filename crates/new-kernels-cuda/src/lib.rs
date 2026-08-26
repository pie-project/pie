//! Pure CUDA kernel definitions — jit unit names, argument marshalling, and
//! launch geometry over a stream; no IR types, no execution state. A driver
//! `Run` resolves plan ids to handles and calls these entry functions (design §8).

#[cfg(all(feature = "_cuda", not(any(feature = "cuda-12", feature = "cuda-13"))))]
compile_error!(
    "new-kernels-cuda's runtime needs exactly one CUDA runtime version: \
     enable `cuda-12` or `cuda-13`, matching the libcudart this binary will load"
);

#[cfg(all(feature = "cuda-12", feature = "cuda-13"))]
compile_error!(
    "new-kernels-cuda: `cuda-12` and `cuda-13` are mutually exclusive -- a binary \
     loads one libcudart, and the two disagree on `cudaGraphAddNode`'s arity"
);

pub mod attn;
pub mod dist;
pub mod gate;
pub mod gemm;
pub mod hc;
pub mod jit;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod source;
pub mod ssm;
pub mod tensor;

pub use jit::{Arg, ArgValue, Ctx, Fire, Launch};
pub use new_kernels::KernelError;
pub use tensor::{KvPool, RaggedTensor, RecurrentPool, Tensor};
