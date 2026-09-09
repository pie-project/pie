use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use checkpoint::contract::{Expr, ModelContract, TensorContract, Visibility};
use model_dsl::{Dtype, Param, ParamSource, Platform, Shard};

const GROUP: u64 = 32;

struct Stated {
    name: &'static str,
    tp: u32,
    params: Vec<Param>,
    contract: ModelContract,
}

fn scratch() -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);

    let dir = std::env::temp_dir().join(format!(
        "model_load_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

fn write_checkpoint(path: &Path, params: &[Param]) {
    let banks: BTreeSet<&str> = params
        .iter()
        .filter(|p| p.dtype == Dtype::Mxfp4)
        .map(|p| p.name.as_str())
        .collect();
    let mut planes: Vec<&Param> = params
        .iter()
        .filter(|p| {
            !p.name
                .strip_suffix(".scales")
                .is_some_and(|stem| banks.contains(stem))
        })
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));

    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    for param in planes {
        state(&mut writer, param);
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}

fn state(writer: &mut ztensor::Writer, param: &Param) {
    match param.dtype {
        Dtype::Bf16 => raw(writer, param, ztensor::Leaf::BF16, 2),
        Dtype::F16 => raw(writer, param, ztensor::Leaf::F16, 2),
        Dtype::F32 => raw(writer, param, ztensor::Leaf::F32, 4),
        Dtype::I32 => raw(writer, param, ztensor::Leaf::I32, 4),
        Dtype::U32 => raw(writer, param, ztensor::Leaf::U32, 4),
        Dtype::U8 => raw(writer, param, ztensor::Leaf::U8, 1),
        Dtype::I8 => raw(writer, param, ztensor::Leaf::I8, 1),
        Dtype::E4m3 => raw(writer, param, ztensor::Leaf::E4M3, 1),
        Dtype::E8m0 => raw(writer, param, ztensor::Leaf::E8M0, 1),
        Dtype::Mxfp4 => codes(writer, param),
        Dtype::E2m1 => panic!(
            "`{}` is declared fp4, which names a kv-page scheme and no stored plane",
            param.name
        ),
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => panic!(
            "`{}` is declared {:?}, which this fixture does not state; the \
             affine rows are exercised through `Model::import`, not `load`",
            param.name, param.dtype
        ),
        Dtype::I64 => raw(writer, param, ztensor::Leaf::I64, 8),
        Dtype::E5m2 | Dtype::I16 | Dtype::U64 | Dtype::U16 | Dtype::Bool => {
            panic!(
                "`{}` is declared {:?}, which no SKU in the catalog stores",
                param.name, param.dtype
            )
        }
        Dtype::Nvfp4
        | Dtype::E4m3row
        | Dtype::E4m3tile128
        | Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k => panic!(
            "`{}` is declared `{}`, which this fixture does not state; a \
             stored block wants its own bytes and no SKU declares one yet",
            param.name, param.dtype,
        ),
    }
}

fn raw(writer: &mut ztensor::Writer, param: &Param, leaf: ztensor::Leaf, width: usize) {
    let data = vec![0u8; width];
    let shape = vec![1u64; param.shape.len()];
    writer
        .add(param.name.as_str(), shape, leaf, &data)
        .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
}

fn codes(writer: &mut ztensor::Writer, param: &Param) {
    let codes = [0u8; 16];
    let scale = [0u8; 1];
    let axis = block_axis(param);
    let mut shape = vec![1u64; axis];
    shape.push(GROUP);
    let term = ztensor::Term::parse(&format!("g{GROUP}_e2m1_e8m0_n")).expect("the mxfp4 term");
    writer
        .object(param.name.as_str(), |o| {
            o.shape(shape)
                .term(term)
                .planes([codes.as_slice(), scale.as_slice()])
        })
        .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
}

fn block_axis(param: &Param) -> usize {
    param.shape.len().checked_sub(2).unwrap_or_else(|| {
        panic!(
            "`{}` is an mxfp4 plane stated {:?}, and a bank's codes are its \
             logical axes, then its blocks of {GROUP}, then the sixteen bytes \
             one block packs into",
            param.name, param.shape,
        )
    })
}

fn stated() -> &'static [Stated] {
    static EVERY: OnceLock<Vec<Stated>> = OnceLock::new();

    EVERY.get_or_init(state_every_sku)
}

fn state_every_sku() -> Vec<Stated> {
    let dir = scratch();
    let mut out = Vec::new();

    for row in models::skus() {
        if !by_load(row) {
            continue;
        }
        let (name, tp) = (row.name.as_str(), row.recipe.tp);
        let trace = (row.trace)(Platform::Cuda);
        let path = dir.join(format!("{name}.zt"));
        write_checkpoint(&path, &trace.params);

        let src = ztensor::Source::open(&path).unwrap_or_else(|why| {
            panic!("`{name}`: the checkpoint just written does not open again: {why}")
        });
        let contract =
            checkpoint_dsl::own_contract(&src, &trace.params, tp, model_dsl::Platform::Cuda)
                .unwrap_or_else(|why| {
                    panic!(
                        "`{name}` refuses a checkpoint that states its own plan, plane for \
                 plane, in the dtypes it asked for: {why}"
                    )
                });
        drop(src);
        out.push(Stated {
            name,
            tp,
            params: trace.params,
            contract,
        });
    }

    let _ = std::fs::remove_dir_all(&dir);
    out
}

fn published(contract: &ModelContract) -> BTreeMap<&str, &TensorContract> {
    contract
        .tensors
        .iter()
        .filter(|entry| entry.visibility == Visibility::Public)
        .map(|entry| (entry.name.as_str(), entry))
        .collect()
}

fn nodes(expr: &Expr, wanted: &dyn Fn(&Expr) -> bool) -> usize {
    let mut found = 0;
    expr.visit(&mut |node| {
        if wanted(node) {
            found += 1;
        }
    });
    found
}

fn by_load(row: &models::Sku) -> bool {
    row.recipe
        .weights
        .iter()
        .all(|w| matches!(w, Dtype::Bf16 | Dtype::Mxfp4))
}

#[test]
fn the_zt_contract_states_the_cut_every_case() {
    one_entry_per_plan_param_under_the_plans_own_names();
    a_cut_param_carries_a_shard_per_leg();
    a_replicated_param_reads_its_own_name_and_nothing_else();
    a_declared_shape_is_the_whole_tensors();
    an_identity_load_states_no_cast();
    a_bank_the_checkpoint_ships_unquantized_is_cast_on_the_way_in();
}

fn one_entry_per_plan_param_under_the_plans_own_names() {
    let mut faults = Vec::new();

    for one in stated() {
        let supply = published(&one.contract);
        let demand: BTreeSet<&str> = one
            .params
            .iter()
            .filter(|p| p.source == ParamSource::Checkpoint)
            .map(|p| p.name.as_str())
            .collect();
        let named: BTreeSet<&str> = supply.keys().copied().collect();
        for name in demand.symmetric_difference(&named) {
            faults.push(format!(
                "`{}`: `{name}` is in one of the plan and the load contract and \
                 not the other",
                one.name,
            ));
        }
        assert_eq!(
            one.contract.alignment, 256,
            "`{}` lands its bytes on {}",
            one.name, one.contract.alignment,
        );
        assert!(
            one.contract.groups.is_empty(),
            "`{}`: a derived group",
            one.name,
        );
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn a_cut_param_carries_a_shard_per_leg() {
    let mut faults = Vec::new();

    for one in stated() {
        let supply = published(&one.contract);
        for param in &one.params {
            let Some(entry) = supply.get(param.name.as_str()) else {
                continue;
            };
            let found = nodes(&entry.expr, &|expr| matches!(expr, Expr::Shard { .. }));
            let want = match &param.shard {
                Shard::Replicated => 0,
                Shard::Cut { segments, .. } => segments.len(),
            };
            if found != want {
                faults.push(format!(
                    "`{}` at tp {}: `{}` is declared {:?} and its expression \
                     carries {found} `Expr::Shard` node(s), not {want}",
                    one.name, one.tp, param.name, param.shard,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn a_replicated_param_reads_its_own_name_and_nothing_else() {
    let mut faults = Vec::new();

    for one in stated() {
        let supply = published(&one.contract);
        for param in &one.params {
            match param.shard {
                Shard::Cut { .. } => continue,
                Shard::Replicated => {}
            }
            let Some(entry) = supply.get(param.name.as_str()) else {
                continue;
            };
            match &entry.expr {
                Expr::Src(from) if *from == param.name => {}
                other => faults.push(format!(
                    "`{}` at tp {}: `{}` is replicated, so every rank holds the \
                     stored tensor whole, and its expression is {other:?}",
                    one.name, one.tp, param.name,
                )),
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn a_declared_shape_is_the_whole_tensors() {
    let mut faults = Vec::new();

    for one in stated() {
        let supply = published(&one.contract);
        for param in &one.params {
            let Some(entry) = supply.get(param.name.as_str()) else {
                continue;
            };
            let Some(declared) = entry.shape.as_deref() else {
                continue;
            };
            let want: Vec<i64> = param
                .shape
                .iter()
                .enumerate()
                .map(|(at, extent)| {
                    let extent = i64::try_from(*extent).expect("an extent no i64 holds");
                    match &param.shard {
                        Shard::Cut { axis, .. } if *axis as usize == at => {
                            extent * i64::from(one.tp)
                        }
                        Shard::Cut { .. } | Shard::Replicated => extent,
                    }
                })
                .collect();
            if declared != want.as_slice() {
                faults.push(format!(
                    "`{}` at tp {}: `{}` is traced {:?} per rank and declared \
                     {declared:?}, where the whole tensor is {want:?}",
                    one.name, one.tp, param.name, param.shape,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn an_identity_load_states_no_cast() {
    let mut faults = Vec::new();

    for one in stated() {
        if one.tp != 1 {
            continue;
        }
        for entry in &one.contract.tensors {
            let found = nodes(&entry.expr, &|expr| matches!(expr, Expr::Cast { .. }));
            if found > 0 {
                faults.push(format!(
                    "`{}`: `{}` carries {found} `Expr::Cast` node(s) against a \
                     checkpoint stored in the very dtype it asked for; an \
                     identity load converts nothing",
                    one.name, entry.name,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn a_bank_the_checkpoint_ships_unquantized_is_cast_on_the_way_in() {
    let dir = scratch();
    let mut faults = Vec::new();

    for row in models::skus() {
        let (name, tp, trace) = (row.name.as_str(), row.recipe.tp, row.trace);
        if !name.starts_with("kimik3") {
            continue;
        }
        let trace = trace(Platform::Cuda);
        let path = dir.join(format!("{name}-unquantized.zt"));
        write_unquantized_checkpoint(&path, &trace.params);

        let src = ztensor::Source::open(&path)
            .unwrap_or_else(|why| panic!("`{name}`: {} does not open: {why}", path.display()));
        let contract =
            checkpoint_dsl::own_contract(&src, &trace.params, tp, model_dsl::Platform::Cuda)
                .unwrap_or_else(|why| {
                    panic!(
                        "`{name}` refuses a checkpoint that ships its banks unquantized, \
                 which is the file a runtime-quantizing SKU exists to read: {why}"
                    )
                });
        drop(src);

        let supply = published(&contract);
        for param in &trace.params {
            if param.source != ParamSource::Checkpoint {
                continue;
            }
            let Some(stem) = param.name.strip_suffix(".scales") else {
                if !supply.contains_key(param.name.as_str()) {
                    faults.push(format!(
                        "`{}`: the plan binds `{}` and the contract publishes \
                         nothing under that name",
                        name, param.name,
                    ));
                }
                continue;
            };
            if supply.contains_key(param.name.as_str()) {
                faults.push(format!(
                    "`{}`: the contract declares `{}`, and the encode of `{stem}` \
                     already publishes it -- two producers for one plane",
                    name, param.name,
                ));
            }
            let Some(payload) = supply.get(stem) else {
                faults.push(format!(
                    "`{}`: the plan binds `{}` and the contract publishes no \
                     `{stem}` whose encode would produce it",
                    name, param.name,
                ));
                continue;
            };
            let encodes = nodes(&payload.expr, &|expr| {
                matches!(
                    expr,
                    Expr::Cast {
                        to: checkpoint::types::Encoding::Quant(_),
                        ..
                    }
                )
            });
            if encodes != 1 {
                faults.push(format!(
                    "`{}`: `{stem}` carries {encodes} cast(s) into a quantized \
                     encoding, so nothing here produces `{}`",
                    name, param.name,
                ));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

fn write_unquantized_checkpoint(path: &Path, params: &[Param]) {
    let mut planes: Vec<&Param> = params
        .iter()
        .filter(|param| !param.name.ends_with(".scales"))
        .collect();
    planes.sort_by(|a, b| a.name.cmp(&b.name));

    let mut writer =
        ztensor::Writer::create(path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    for param in planes {
        match param.dtype {
            Dtype::Mxfp4 => {
                let logical = param.shape.len().saturating_sub(1);
                writer
                    .add(
                        param.name.as_str(),
                        vec![1u64; logical],
                        ztensor::Leaf::BF16,
                        &[0u8, 0u8],
                    )
                    .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
            }
            _ => state(&mut writer, param),
        }
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}
