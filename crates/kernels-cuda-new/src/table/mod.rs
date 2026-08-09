use kernels::{KernelSig, Prepare};

pub mod attn;

/// Every kernel a lowered declaration may state.
pub static KERNELS: &[KernelSig] = &concat_tables();

/// The per-family tables, in the concatenation's order.
pub static TABLES: &[&[KernelSig]] = &concat_lists();

/// The families still written as rows. **EMPTY.**
static ROW_TABLES: &[&[KernelSig]] = &[];

const N_LISTS: usize = ROW_TABLES.len() + crate::x::SIGS.len();

/// The row lists and the fn-world lists as one list of lists.
const fn concat_lists() -> [&'static [KernelSig]; N_LISTS] {
    let mut out = [EMPTY_LIST; N_LISTS];
    let mut w = 0;
    let mut i = 0;
    while i < ROW_TABLES.len() {
        out[w] = ROW_TABLES[i];
        w += 1;
        i += 1;
    }
    let mut j = 0;
    while j < crate::x::SIGS.len() {
        out[w] = crate::x::SIGS[j];
        w += 1;
        j += 1;
    }
    out
}

const EMPTY_LIST: &[KernelSig] = &[];

/// The row a symbol names, or nothing.
#[must_use]
pub fn sig(symbol: &str) -> Option<&'static KernelSig> {
    KERNELS.iter().find(|row| row.symbol == symbol)
}

/// `[&[T]] -> [T]` at compile time, because [`KERNELS`] must stay a
const fn concat_tables() -> [KernelSig; TOTAL] {
    let mut out = [EMPTY; TOTAL];
    let mut w = 0;
    let mut t = 0;
    while t < TABLES.len() {
        let table = TABLES[t];
        let mut i = 0;
        while i < table.len() {
            out[w] = copy_sig(&table[i]);
            w += 1;
            i += 1;
        }
        t += 1;
    }
    out
}

const TOTAL: usize = total();

const fn total() -> usize {
    let mut n = 0;
    let mut t = 0;
    while t < TABLES.len() {
        n += TABLES[t].len();
        t += 1;
    }
    n
}

const EMPTY: KernelSig = KernelSig {
    name: "", symbol: "", file: None, launch: kernels::LaunchRule::Unstated,
    whole: false, needs: Prepare::None,
    lacks: &[], sink: None, in_place: &[], depth_prefix_plan: false,
    operands: &[],
    axes: &[], grid_param: None,
    head_param: None, heads_param: None, rows_param: None,
};

const fn copy_sig(k: &KernelSig) -> KernelSig {
    KernelSig {
        name: k.name, symbol: k.symbol, file: k.file, launch: k.launch,
        whole: k.whole, needs: k.needs,
        lacks: k.lacks, sink: k.sink, in_place: k.in_place,
        depth_prefix_plan: k.depth_prefix_plan,
        operands: k.operands, axes: k.axes,
        grid_param: k.grid_param,
        head_param: k.head_param, heads_param: k.heads_param,
        rows_param: k.rows_param,
    }
}

#[cfg(test)]
mod tests {
    use super::{KERNELS, TABLES, sig};

    /// A symbol names at most one row. Two rows sharing one symbol is the
    #[test]
    fn a_symbol_names_one_row() {
        let mut seen: Vec<&str> = Vec::with_capacity(KERNELS.len());
        for row in KERNELS {
            assert!(!seen.contains(&row.symbol), "{} is stated twice", row.symbol);
            seen.push(row.symbol);
        }
    }

    /// The lookup finds what the table holds, and refuses what it does not.
    #[test]
    fn the_lookup_is_the_table() {
        for row in KERNELS {
            assert_eq!(sig(row.symbol).map(|r| r.symbol), Some(row.symbol));
        }
        assert!(sig("norm::a_kernel_nobody_wrote").is_none());
    }

    /// The concatenation spans every family, so a table listed in [`TABLES`]
    #[test]
    fn the_concatenation_is_the_tables() {
        let counted: usize = TABLES.iter().map(|t| t.len()).sum();
        assert_eq!(KERNELS.len(), counted);
        assert!(KERNELS.len() > 100, "{} rows is not the CUDA table", KERNELS.len());
        for table in TABLES {
            assert!(!table.is_empty(), "a family listed in TABLES declares no rows");
        }
    }

    /// `driver_internal`'s rows were reachable and NOT in [`KERNELS`].
    const _THE_DRIVER_INTERNAL_ROWS_ARE_NOT_STATABLE: () = ();
}
