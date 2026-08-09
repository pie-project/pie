use crate::unit::Unit;

pub mod attn;
/// FlashInfer's cascade merge — the split-KV path's other half.
pub mod cascade;
/// The FlashInfer FA2 lattice — 56 units over four axes, the last thing in
pub mod fa2;
/// The supergraph's two arming kernels — the one family named after a SHELL
pub mod graph;
pub mod marlin;
pub mod vision;

/// Every family's units, in a stable order.
pub static ALL: &[&[Unit]] = &[
    crate::x::adapter::UNITS,
    crate::x::attn::UNITS,
    attn::UNITS,
    crate::x::xqa::UNITS,
    cascade::UNITS,
    fa2::UNITS,
    crate::x::gemm::UNITS,
    graph::UNITS,
    crate::x::layout::UNITS,
    marlin::UNITS,
    crate::x::mlp::UNITS,
    crate::x::moe::UNITS,
    crate::x::norm::UNITS,
    crate::x::quant::UNITS,
    crate::x::rope::UNITS,
    crate::x::sample::UNITS,
    crate::x::ssm::UNITS,
    vision::UNITS,
];
