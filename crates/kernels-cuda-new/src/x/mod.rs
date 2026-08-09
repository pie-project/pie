pub mod abi;
pub mod contract;
pub mod cx;
pub mod launch;
#[macro_use]
pub mod macros;
pub mod adapter;
pub mod attn;
pub mod driver_internal;
pub mod gemm;
pub mod layout;
pub mod mlp;
pub mod moe;
pub mod norm;
pub mod quant;
pub mod rope;
pub mod sample;
pub mod ssm;
pub mod xqa;

#[cfg(feature = "_cuda")]
pub mod fire;

pub use abi::{Abi, ByValue, Layout, fp8_kind};
pub use contract::{Contract, Entry, Fired, Refusal};
#[cfg(feature = "_cuda")]
pub use contract::Route;
pub use cx::{
    AttnWorkspace, Cx, Facts, Gdn, KvDType, KvLayer, KvScheme, MlaLayer, MlaPlan, Plan, Rows, Slab,
    Yarn,
};

/// Every family that has crossed into fn-world.
#[cfg(feature = "_cuda")]
pub static FAMILIES: &[&[Entry]] = &[
    rope::ENTRIES,
    layout::ENTRIES,
    sample::ENTRIES,
    quant::ENTRIES,
    mlp::ENTRIES,
    norm::ENTRIES,
    ssm::ENTRIES,
    moe::ENTRIES,
    attn::ENTRIES,
    xqa::ENTRIES,
];

/// Every [`Contract`] fn-world declares — the list the third registration
pub static CONTRACTS: &[&[Contract]] = &[
    rope::CONTRACTS,
    layout::CONTRACTS,
    sample::CONTRACTS,
    adapter::CONTRACTS,
    quant::CONTRACTS,
    mlp::CONTRACTS,
    norm::CONTRACTS,
    ssm::CONTRACTS,
    moe::CONTRACTS,
    gemm::CONTRACTS,
    attn::CONTRACTS,
    xqa::CONTRACTS,
];

/// The [`Contract`] for one symbol, or `None` if fn-world declares no such
#[must_use]
pub fn contract(symbol: &str) -> Option<&'static Contract> {
    CONTRACTS
        .iter()
        .flat_map(|family| family.iter())
        .find(|contract| contract.symbol == symbol)
}

/// The [`Entry`] for one symbol, or `None` if no family declares it.
#[cfg(feature = "_cuda")]
#[must_use]
pub fn entry(symbol: &str) -> Option<&'static Entry> {
    FAMILIES
        .iter()
        .flat_map(|family| family.iter())
        .find(|entry| entry.contract.symbol == symbol)
}

/// What will fire one symbol — §5 step 4's resolution, in the one crate that
#[cfg(feature = "_cuda")]
#[must_use]
pub fn route(symbol: &str) -> Route {
    if let Some(entry) = entry(symbol) {
        return match (entry.bind, entry.unbound) {
            (Some(_), _) => Route::Bound(entry),
            (None, Some(why)) => Route::Unbound(entry, why),
            (None, None) => Route::Unbound(entry, "this symbol is not trace-fired"),
        };
    }
    if matches!(
        crate::execution::service(symbol),
        Some(crate::execution::Service::DriverOp)
    ) {
        return Route::Driver;
    }
    if crate::table::sig(symbol).is_some() {
        return Route::Rows;
    }
    Route::Unknown
}

/// Every contract in fn-world, as the `KernelSig` rows `model-compiler`
pub static SIGS: &[&[kernels::KernelSig]] = &[
    rope::SIGS,
    layout::SIGS,
    sample::SIGS,
    adapter::SIGS,
    quant::SIGS,
    mlp::SIGS,
    norm::SIGS,
    ssm::SIGS,
    moe::SIGS,
    gemm::SIGS,
    attn::SIGS,
    xqa::SIGS,
];

#[cfg(test)]
mod tests {
    use super::{CONTRACTS, SIGS, contract};

    /// [`CONTRACTS`] and [`SIGS`] cover the same symbols.
    #[test]
    fn the_two_registries_cover_the_same_symbols() {
        let contracts: Vec<&str> = CONTRACTS
            .iter()
            .flat_map(|family| family.iter())
            .map(|c| c.symbol)
            .collect();
        let rows: Vec<&str> = SIGS
            .iter()
            .flat_map(|family| family.iter())
            .map(|k| k.symbol)
            .collect();

        assert!(
            contracts.len() > 50,
            "the walk of `CONTRACTS` found {} symbols, which means it stopped \
             walking rather than that fn-world shrank",
            contracts.len(),
        );

        let unreachable: Vec<&&str> = rows.iter().filter(|s| !contracts.contains(s)).collect();
        assert!(
            unreachable.is_empty(),
            "{unreachable:?} have a row in `SIGS` and no declaration in \
             `CONTRACTS`: `contract()` answers `None` for a symbol \
             `table::sig` answers for, and every read that resolves a \
             `Contract` silently skips them",
        );

        let rowless: Vec<&&str> = contracts.iter().filter(|s| !rows.contains(s)).collect();
        assert!(
            rowless.is_empty(),
            "{rowless:?} are declared in `CONTRACTS` and have no row in \
             `SIGS`: `check_plan` refuses them at model load",
        );
    }

    /// [`contract`] answers for a DRIVER OP, which is the whole point of it.
    #[test]
    fn a_driver_op_has_a_reachable_contract() {
        let c = contract("gemm::act_x_wt_bf16")
            .expect("`gemm`'s twelve are driver ops and still declare contracts");
        assert_eq!(c.lowered_as, Some("gemm::act_x_w"));
    }
}
