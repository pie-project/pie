use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use model::contract::ModelError;
use model_dsl::{Dtype, Param, Plane, Shard};
use model_loader::contract::{Expr, ModelContract, TensorContract, Visibility};

type Load = fn(&ztensor::Source) -> Result<ModelContract, ModelError>;

struct Sku {
    name: &'static str,
    tp: u32,
    load: Load,
}

struct Stated {
    sku: Sku,
    params: Vec<Param>,
    contract: ModelContract,
}

fn sku(name: &'static str, tp: u32, load: Load) -> Sku {
    Sku { name, tp, load }
}

fn skus() -> Vec<Sku> {
    vec![
        sku("dsv4-base-bf16-kv-bf16", 1, |src| {
            model::deepseek_v4::model::Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
                .load(src)
        }),
        sku("dsv4-base-bf16-kv-bf16-tp2", 2, |src| {
            model::deepseek_v4::model::Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
                .load(src)
        }),
        sku("gemma4-e4b-bf16-kv-bf16", 1, |src| {
            model::gemma_4::model::Model::e4b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).load(src)
        }),
        sku("gemma4-31b-bf16-kv-bf16", 1, |src| {
            model::gemma_4::model::Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).load(src)
        }),
        sku("gemma4-31b-bf16-kv-bf16-tp2", 2, |src| {
            model::gemma_4::model::Model::b31(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2).load(src)
        }),
        sku("glm5-a12b-bf16-bf16-kv-bf16", 1, |src| {
            model::glm_5::model::Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1)
                .load(src)
        }),
        sku("glm5-a12b-bf16-bf16-kv-bf16-tp2", 2, |src| {
            model::glm_5::model::Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2)
                .load(src)
        }),
        sku("gptoss-20b-bf16-mxfp4-kv-bf16", 1, |src| {
            model::gpt_oss::model::Model::b20(
                Dtype::Bf16,
                Dtype::Mxfp4,
                Dtype::Bf16,
                Dtype::Bf16,
                1,
            )
            .load(src)
        }),
        sku("gptoss-120b-bf16-mxfp4-kv-bf16", 1, |src| {
            model::gpt_oss::model::Model::b120(
                Dtype::Bf16,
                Dtype::Mxfp4,
                Dtype::Bf16,
                Dtype::Bf16,
                1,
            )
            .load(src)
        }),
        sku("gptoss-120b-bf16-mxfp4-kv-bf16-tp2", 2, |src| {
            model::gpt_oss::model::Model::b120(
                Dtype::Bf16,
                Dtype::Mxfp4,
                Dtype::Bf16,
                Dtype::Bf16,
                2,
            )
            .load(src)
        }),
        sku("kimik3-bf16-mxfp4-kv-bf16", 1, |src| {
            model::kimi_k3::model::Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, Dtype::Bf16, 1)
                .load(src)
        }),
        sku("kimik3-bf16-mxfp4-kv-bf16-tp2", 2, |src| {
            model::kimi_k3::model::Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, Dtype::Bf16, 2)
                .load(src)
        }),
        sku("qwen35-a3b-bf16-kv-bf16", 1, |src| {
            model::qwen_3::model::Model::a3b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).load(src)
        }),
        sku("qwen35-d3b-bf16-kv-bf16", 1, |src| {
            model::qwen_3::model::Model::d3b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).load(src)
        }),
        sku("qwen35-d0.8b-bf16-kv-bf16", 1, |src| {
            model::qwen_3::model::Model::d0_8b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 1).load(src)
        }),
        sku("qwen35-a3b-bf16-kv-bf16-tp2", 2, |src| {
            model::qwen_3::model::Model::a3b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, 2).load(src)
        }),
    ]
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
    let mut planes: Vec<&Param> = params.iter().collect();
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
        Dtype::Bf16 => raw(writer, param, ztensor::DType::BF16, None, 2),
        Dtype::F16 => raw(writer, param, ztensor::DType::F16, None, 2),
        Dtype::F32 => raw(writer, param, ztensor::DType::F32, None, 4),
        Dtype::I32 => raw(writer, param, ztensor::DType::I32, None, 4),
        Dtype::U32 => raw(writer, param, ztensor::DType::U32, None, 4),
        Dtype::U8 => raw(writer, param, ztensor::DType::U8, None, 1),
        Dtype::I8 => raw(writer, param, ztensor::DType::I8, None, 1),
        Dtype::Fp8E4m3 => raw(writer, param, ztensor::DType::U8, Some("f8_e4m3fn"), 1),
        Dtype::E8m0 => raw(writer, param, ztensor::DType::U8, Some("f8_e8m0"), 1),
        Dtype::Mxfp4 => codes(writer, param),
        Dtype::Fp4 => panic!(
            "`{}` is declared fp4, which names a kv-page scheme and no stored plane",
            param.name
        ),
    }
}

fn raw(
    writer: &mut ztensor::Writer,
    param: &Param,
    dtype: ztensor::DType,
    logical: Option<&str>,
    width: usize,
) {
    let data = vec![0u8; width];
    let shape = vec![1u64; param.shape.len()];
    writer
        .object(param.name.as_str(), |o| {
            o.shape(shape).part("data", |p| {
                let p = p.dtype(dtype);
                match logical {
                    Some(id) => p.logical(id),
                    None => p,
                }
                .bytes(&data)
            })
        })
        .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
}

fn codes(writer: &mut ztensor::Writer, param: &Param) {
    let data = vec![0u8; 16];
    let axis = block_axis(param);
    let mut shape = vec![1u64; axis];
    shape.push(32);
    let stated = u64::try_from(axis).expect("an axis no u64 holds");
    writer
        .object(param.name.as_str(), |o| {
            o.shape(shape)
                .layout("zt.mx/1")
                .attr("axis", stated)
                .attr("block_size", 32u64)
                .part("data", |p| {
                    p.dtype(ztensor::DType::U8).logical("f4_e2m1").bytes(&data)
                })
        })
        .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
}

fn block_axis(param: &Param) -> usize {
    param.shape.len().checked_sub(2).unwrap_or_else(|| {
        panic!(
            "`{}` is an mxfp4 plane stored {:?}, and a bank's codes are its \
             logical axes, then its blocks, then the sixteen bytes one block \
             packs into",
            param.name, param.shape,
        )
    })
}

fn stated() -> Vec<Stated> {
    let dir = scratch();
    let mut out = Vec::new();

    for sku in skus() {
        let trace = model::trace_of(sku.name)
            .unwrap_or_else(|| panic!("`{}` names no catalog row", sku.name));
        let plan = trace(Plane::Cuda);
        let path = dir.join(format!("{}.zt", sku.name));
        write_checkpoint(&path, &plan.params);

        let src = ztensor::Source::open(&path).unwrap_or_else(|why| {
            panic!(
                "`{}`: the checkpoint just written does not open again: {why}",
                sku.name
            )
        });
        let contract = (sku.load)(&src).unwrap_or_else(|why| {
            panic!(
                "`{}` refuses a checkpoint that states its own plan, plane for \
                 plane, in the dtypes it asked for: {why}",
                sku.name
            )
        });
        drop(src);
        out.push(Stated {
            sku,
            params: plan.params,
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

fn nodes(expr: Expr, wanted: &dyn Fn(&Expr) -> bool) -> usize {
    fn walk(expr: Expr, wanted: &dyn Fn(&Expr) -> bool, found: &mut usize) -> Expr {
        if wanted(&expr) {
            *found += 1;
        }
        expr.map_children(|child| Ok(walk(child, wanted, found)))
            .expect("a count rebuilds every node it walks")
    }

    let mut found = 0;
    walk(expr, wanted, &mut found);
    found
}

#[test]
fn every_catalog_row_states_how_it_lands() {
    let asked: BTreeSet<&str> = skus().iter().map(|sku| sku.name).collect();
    let shipped: BTreeSet<&str> = model::catalog().into_iter().map(|(name, _)| name).collect();

    let mut faults = Vec::new();
    for name in asked.symmetric_difference(&shipped) {
        faults.push(format!(
            "`{name}` is in one of the catalog and this test's table and not \
             the other; a SKU whose landing nobody asks about ships untested"
        ));
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn one_entry_per_plan_param_under_the_plans_own_names() {
    let mut faults = Vec::new();

    for one in stated() {
        let supply = published(&one.contract);
        let demand: BTreeSet<&str> = one.params.iter().map(|p| p.name.as_str()).collect();
        let named: BTreeSet<&str> = supply.keys().copied().collect();
        for name in demand.symmetric_difference(&named) {
            faults.push(format!(
                "`{}`: `{name}` is in one of the plan and the load contract and \
                 not the other",
                one.sku.name,
            ));
        }
        assert_eq!(
            one.contract.alignment, 256,
            "`{}` lands its bytes on {}",
            one.sku.name, one.contract.alignment,
        );
        assert!(
            one.contract.groups.is_empty(),
            "`{}`: a derived group",
            one.sku.name,
        );
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn a_cut_param_carries_a_shard_per_leg() {
    let mut faults = Vec::new();

    for one in stated() {
        let supply = published(&one.contract);
        for param in &one.params {
            let Some(entry) = supply.get(param.name.as_str()) else {
                continue;
            };
            let found = nodes(entry.expr.clone(), &|expr| {
                matches!(expr, Expr::Shard { .. })
            });
            let want = match &param.shard {
                Shard::Replicated => 0,
                Shard::Cut { segments, .. } => segments.len(),
            };
            if found != want {
                faults.push(format!(
                    "`{}` at tp {}: `{}` is declared {:?} and its expression \
                     carries {found} `Expr::Shard` node(s), not {want}",
                    one.sku.name, one.sku.tp, param.name, param.shard,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
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
                    one.sku.name, one.sku.tp, param.name,
                )),
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
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
                            extent * i64::from(one.sku.tp)
                        }
                        Shard::Cut { .. } | Shard::Replicated => extent,
                    }
                })
                .collect();
            if declared != want.as_slice() {
                faults.push(format!(
                    "`{}` at tp {}: `{}` is traced {:?} per rank and declared \
                     {declared:?}, where the whole tensor is {want:?}",
                    one.sku.name, one.sku.tp, param.name, param.shape,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

#[test]
fn an_identity_load_states_no_cast() {
    let mut faults = Vec::new();

    for one in stated() {
        if one.sku.tp != 1 {
            continue;
        }
        for entry in &one.contract.tensors {
            let found = nodes(entry.expr.clone(), &|expr| {
                matches!(expr, Expr::Cast { .. })
            });
            if found > 0 {
                faults.push(format!(
                    "`{}`: `{}` carries {found} `Expr::Cast` node(s) against a \
                     checkpoint stored in the very dtype it asked for; an \
                     identity load converts nothing",
                    one.sku.name, entry.name,
                ));
            }
        }
    }

    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
