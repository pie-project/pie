//! Sampling. One kernel: the device argmax with its EOS compare.

use kernels::{KernelSig, kernel};

use crate::axes::*;

pub static KERNELS: &[KernelSig] = &[
    // 1 in argmax.metal
    kernel!(argmax_logits "argmax_logits", axes = &[BF16]),
];
