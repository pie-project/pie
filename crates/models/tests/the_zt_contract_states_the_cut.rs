use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use checkpoint::contract::{Expr, ModelContract, TensorContract, Visibility};
use checkpoint_dsl::Error;
use model_dsl::{Dtype, Param, ParamSource, Platform, Shard};

const GROUP: u64 = 32;

/// **A ROW'S LANDING, ASKED AT THE DEGREE THE CATALOG STATES.**
///
/// The `u32` is not this table's to choose. Every closure below used to write
/// its own — `Model::k3(.., 2)` — which made this table a second statement of
/// a fact `models::catalog()` already holds, and a test that restates the fact
/// it is checking cannot check it: `models::kimi_k3::IMPORTS` said 1 for that
/// same row for as long as this file said 2, and nothing here could see it.
/// So `state_every_sku` reads the degree off the catalog row and hands it in,
/// and `models::ImportFn` takes it for the same reason one crate over.
type Load = fn(&ztensor::Source, u32) -> Result<ModelContract, Error>;

struct Sku {
    name: &'static str,
    load: Load,
}

struct Stated {
    name: &'static str,
    tp: u32,
    params: Vec<Param>,
    contract: ModelContract,
}

fn sku(name: &'static str, load: Load) -> Sku {
    Sku { name, load }
}

fn skus() -> Vec<Sku> {
    vec![
        sku("dsv4-base-bf16-kv-bf16", |src, tp| {
            models::deepseek_v4::model::Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp)
                .load(src)
        }),
        sku("dsv4-base-bf16-kv-bf16-tp2", |src, tp| {
            models::deepseek_v4::model::Model::base(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp)
                .load(src)
        }),
        sku("dsv4-flash-bf16-kv-bf16", |src, tp| {
            models::deepseek_v4::model::Model::flash(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp)
                .load(src)
        }),
        sku("gemma4-e4b-bf16-kv-bf16", |src, tp| {
            models::gemma_4::model::Model::e4b(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("gemma4-e4b-eagle-bf16-kv-bf16", |src, tp| {
            models::gemma_4::model::Model::e4b_eagle(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("gemma4-e4b-vision-bf16-kv-bf16", |src, tp| {
            models::gemma_4::model::Model::e4b_vision(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("gemma4-31b-bf16-kv-bf16", |src, tp| {
            models::gemma_4::model::Model::b31(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("gemma4-31b-bf16-kv-bf16-tp2", |src, tp| {
            models::gemma_4::model::Model::b31(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("glm5-a12b-bf16-bf16-kv-bf16", |src, tp| {
            models::glm_5::model::Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("glm5-a12b-bf16-bf16-kv-bf16-tp2", |src, tp| {
            models::glm_5::model::Model::a12b(Dtype::Bf16, Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("gptoss-20b-bf16-mxfp4-kv-bf16", |src, tp| {
            models::gpt_oss::model::Model::b20(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp).load(src)
        }),
        sku("gptoss-120b-bf16-mxfp4-kv-bf16", |src, tp| {
            models::gpt_oss::model::Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp)
                .load(src)
        }),
        sku("gptoss-120b-bf16-mxfp4-kv-bf16-tp2", |src, tp| {
            models::gpt_oss::model::Model::b120(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp)
                .load(src)
        }),
        sku("kimik3-bf16-mxfp4-kv-bf16", |src, tp| {
            models::kimi_k3::model::Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp).load(src)
        }),
        sku("kimik3-bf16-mxfp4-kv-bf16-tp2", |src, tp| {
            models::kimi_k3::model::Model::k3(Dtype::Bf16, Dtype::Mxfp4, Dtype::Bf16, tp).load(src)
        }),
        // The C3 row: the one SKU of this catalog whose checkpoint publishes
        // a draft head, so the one whose bijection covers `mtp.*`. Fifteen
        // stored tensors land as fifteen declared planes and the count is a
        // coincidence of two opposite regroupings — `mtp.fc` is one bank cut
        // into two, and `gate_proj`/`up_proj` are two fused into one.
        sku("qwen36-27b-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d27b(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        // The shadowed twin (see `qwen_3::IMPORTS`): the same text as the
        // qwen36 row, so the same landing — what tells the two apart is the
        // tokenizer contract, which no `.zt` bijection can see.
        sku("qwen38-27b-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d27b(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen35-a3b-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::a3b(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen38-flash-bf16-kv-bf16", |src, tp| {
            models::qwen_4::model::Model::flash(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen35-d3b-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d3b(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen35-d0.8b-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d0_8b(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        // The M-1 and M-2 rows: the same trunks reading the vision towers
        // their own checkpoints ship. The bijection here covers `visual.*` —
        // a hundred and forty-eight planes for the twelve-block one and three
        // hundred and twenty-eight for the twenty-seven-block one — and it is
        // one-to-one at every entry but the patch embed, whose `Conv3d` kernel
        // the plan reads as the matmul bank it already is.
        sku("qwen35-d0.8b-vision-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d0_8b_vision(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen35-d0.8b-vision-eagle-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d0_8b_vision_eagle(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen36-27b-vision-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d27b_vision(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen38-27b-vision-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d27b_vision(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        // The M-4 row: the same trunk with an EAGLE head overlaid, so the one
        // SKU whose bijection covers `aux.*`. Eleven declared planes against
        // twelve stored tensors, and the difference is the same regrouping the
        // 27B's row makes in the other direction -- `aux.fc` is one bank cut
        // into two and `gate_proj`/`up_proj` are two fused into one.
        sku("qwen35-d0.8b-eagle-bf16-kv-bf16", |src, tp| {
            models::qwen_3::model::Model::d0_8b_eagle(Dtype::Bf16, Dtype::Bf16, tp).load(src)
        }),
        sku("qwen35-a3b-bf16-kv-bf16-tp2", |src, tp| {
            models::qwen_3::model::Model::a3b(Dtype::Bf16, Dtype::Bf16, tp).load(src)
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
        Dtype::E4m3 => raw(writer, param, ztensor::DType::U8, Some("f8_e4m3fn"), 1),
        Dtype::E8m0 => raw(writer, param, ztensor::DType::U8, Some("f8_e8m0"), 1),
        Dtype::Mxfp4 => codes(writer, param),
        Dtype::E2m1 => panic!(
            "`{}` is declared fp4, which names a kv-page scheme and no stored plane",
            param.name
        ),
        // The catalog's affine rows (`*-mlxu4-*`, `*-mlxu2-*`) are read
        // through `Model::import`, off an MLX checkpoint that ships the triplet, and
        // not through the `load` this fixture exercises. None of the SKUs
        // above reaches this arm, and stating an affine bank here would mean
        // inventing a canonical layout for the codes plus a `.scales` and a
        // `.biases` — a claim about bytes no load in this file reads.
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => panic!(
            "`{}` is declared {:?}, which this fixture does not state; the \
             affine rows are exercised through `Model::import`, not `load`",
            param.name,
            param.dtype
        ),
        // The checkpoint vocabulary the merged dtype enum brought with it. No
        // SKU in the catalog declares a weight in one, so this fixture has
        // never had to state one, and inventing a layout here would be a
        // claim about a plane no model ships.
        // **THE ONE LOOKUP TABLE A CATALOG ROW DECLARES.**
        // `ffn.gate.tid2eid` is dsv4-flash's `[vocab, top_k]` token-id →
        // expert-id table, which `linear.moe_hash_route` gathers. It is real
        // bytes in the most ordinary layout there is — eight per element,
        // dense, no companion plane — so it is stated here like every other
        // raw row, and not refused like the block-quantized ones below whose
        // layout would have to be invented.
        Dtype::I64 => raw(writer, param, ztensor::DType::I64, None, 8),
        Dtype::E5m2 | Dtype::I16 | Dtype::U64 | Dtype::U16 | Dtype::Bool => {
            panic!(
                "`{}` is declared {:?}, which no SKU in the catalog stores",
                param.name, param.dtype
            )
        }
        // The same argument as the affine rows above, one wave later: a
        // stored K-quant plane is real bytes in a real block layout, and no
        // catalog SKU declares one yet (alto `next.md` §J5 — the import leg
        // that would feed one is item 1 and has not landed). Writing a
        // plausible super-block here would be a claim about bytes no load in
        // this file reads.
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
    let stated = u64::try_from(axis).expect("an axis no u64 holds");
    let mut shape = vec![1u64; axis];
    shape.push(GROUP);
    writer
        .object(param.name.as_str(), |o| {
            o.shape(shape)
                .layout("zt.mx/1")
                .attr("axis", stated)
                .attr("block_size", GROUP)
                .part("data", |p| {
                    p.dtype(ztensor::DType::U8).logical("f4_e2m1").bytes(&data)
                })
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
    let rows = models::catalog();
    let mut out = Vec::new();

    for sku in skus() {
        let (_, tp, trace, _) = *rows
            .iter()
            .find(|(name, ..)| *name == sku.name)
            .unwrap_or_else(|| panic!("`{}` names no catalog row", sku.name));
        let trace = trace(Platform::Cuda);
        let path = dir.join(format!("{}.zt", sku.name));
        write_checkpoint(&path, &trace.params);

        let src = ztensor::Source::open(&path).unwrap_or_else(|why| {
            panic!(
                "`{}`: the checkpoint just written does not open again: {why}",
                sku.name
            )
        });
        let contract = (sku.load)(&src, tp).unwrap_or_else(|why| {
            panic!(
                "`{}` refuses a checkpoint that states its own plan, plane for \
                 plane, in the dtypes it asked for: {why}",
                sku.name
            )
        });
        drop(src);
        out.push(Stated {
            name: sku.name,
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

/// **AND THE AFFINE ROWS ARE EXEMPT, BY THE FIXTURE'S OWN ARGUMENT.**
///
/// `state`'s affine arm refuses to write a plane: an affine bank would mean
/// inventing a canonical layout for the codes plus a `.scales` and a
/// `.biases`, which is a claim about bytes no load in this file reads. The
/// `*-mlxu4-*` and `*-mlxu2-*` rows are landed through `Model::import` off a
/// real MLX checkpoint, and `the_checkpoints_state_what_the_texts_read` is
/// where that is asked. So they are named here as exempt rather than left to
/// read as an omission — a row that ships untested and a row tested somewhere
/// else are different sentences, and this test only makes the first one.
///
/// **A PREFIX AND NOT A LIST OF WIDTHS.** It was `-mlxu4-` while four bits was
/// the only affine width the catalog spent; the exemption is about the SCHEME
/// — a triplet this fixture cannot invent — and the width was never the reason,
/// so a second width should not have been a second line. Every affine row this
/// tree can spell is `-mlxu<bits>-` (`model_dsl::Dtype::U2g32`).
const NOT_BY_LOAD: &str = "-mlxu";

#[test]
fn every_catalog_row_states_how_it_lands() {
    let asked: BTreeSet<&str> = skus().iter().map(|sku| sku.name).collect();
    let shipped: BTreeSet<&str> = models::catalog()
        .into_iter()
        .map(|(name, ..)| name)
        .filter(|name| !name.contains(NOT_BY_LOAD))
        .collect();

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
        // **A REGISTERED PLANE IS NOT THE CHECKPOINT'S** (palo design §8). An
        // adapter bank is a `Param` because it is a `Def::Weight` the shell
        // reserves and the routed op indexes, and it is `ParamSource::
        // Registered` because its bytes arrive through `register_adapter`
        // instead of out of a `.zt`. Demanding it here would be demanding that
        // a pretrained checkpoint ship somebody's LoRA.
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

#[test]
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
                    one.name, one.tp, param.name,
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

#[test]
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

/// The other door: a checkpoint that has NOT been quantized yet.
///
/// Every fixture above states the plan's own dtypes, so no SKU is ever asked
/// to convert anything and `an_identity_load_states_no_cast` holds. This one
/// states the opposite file — kimi's mxfp4 expert banks written as the BF16 a
/// published checkpoint actually ships, with no scales plane at all, because
/// an unquantized bank has none to ship — and asks what the contract says
/// about it.
///
/// # The accord this pins
///
/// A plan param has ONE producer. For every param here but one that is the
/// contract entry of its own name; for `<w>.scales` it is the ENCODE the
/// contract's `Cast { to: Quant(mxfp4) }` compiles into, which publishes both
/// planes and names the second `<w>.scales` — the same spelling
/// `model_dsl::scales_name` writes and `Weight::planes` interns. So the
/// contract must NOT declare that entry: a declaration would be a second
/// producer for a plane that already has exactly one, which is the thing this
/// file's other tests exist to forbid.
///
/// The loader's half is proved where it lives, on the bytes:
/// `checkpoint`'s `executor::walk::tests`
/// `an_expert_bank_encodes_to_the_same_bytes_as_the_rows_it_stacks` compiles a
/// rank-3 bank's `Cast`, runs it, and reads the published `experts.scales`
/// back. This half is the claim on the model side of that seam, and the two
/// are what "one producer" means when the producer is a kernel.
///
/// Recorded open by menlo M18 as "kimi mxfp4 runtime-quant needs the loader to
/// grow rank-3 encode and a `.scales`-vs-`_scale` naming accord".
///
/// **AND "ON THE WAY IN" IS `pie model import` NOW** (§M-3). The sentence the
/// contract says is unchanged and the party that runs it is not: no device
/// target carries an encode any more, so this cast compiles for CONVERSION
/// and is refused for a serving load — which is what `storage_compiler`'s
/// `a_serving_target_refuses_the_encode_a_conversion_target_runs` pins, on
/// the plan, where the mask is. Read together, the two say the whole ruling:
/// this SKU states the conversion, and only the converter may run it.
#[test]
fn a_bank_the_checkpoint_ships_unquantized_is_cast_on_the_way_in() {
    let dir = scratch();
    let mut faults = Vec::new();

    for sku in skus() {
        if !sku.name.starts_with("kimik3") {
            continue;
        }
        let rows = models::catalog();
        let (_, tp, trace, _) = *rows
            .iter()
            .find(|(name, ..)| *name == sku.name)
            .unwrap_or_else(|| panic!("`{}` names no catalog row", sku.name));
        let trace = trace(Platform::Cuda);
        let path = dir.join(format!("{}-unquantized.zt", sku.name));
        write_unquantized_checkpoint(&path, &trace.params);

        let src = ztensor::Source::open(&path).unwrap_or_else(|why| {
            panic!("`{}`: {} does not open: {why}", sku.name, path.display())
        });
        let contract = (sku.load)(&src, tp).unwrap_or_else(|why| {
            panic!(
                "`{}` refuses a checkpoint that ships its banks unquantized, \
                 which is the file a runtime-quantizing SKU exists to read: {why}",
                sku.name
            )
        });
        drop(src);

        let supply = published(&contract);
        for param in &trace.params {
            // **A REGISTERED PLANE IS NOT THE CHECKPOINT'S**, and the argument
            // is the one `one_entry_per_plan_param_under_the_plans_own_names`
            // makes next door: an adapter bank is a `Param` because the shell
            // reserves it and the routed op indexes it, and its bytes arrive
            // through `register_adapter` rather than out of a `.zt`. Demanding
            // it here would be demanding that an unquantized checkpoint ship
            // somebody's LoRA.
            if param.source != ParamSource::Checkpoint {
                continue;
            }
            let Some(stem) = param.name.strip_suffix(".scales") else {
                // Every other plane is declared under its own name.
                if !supply.contains_key(param.name.as_str()) {
                    faults.push(format!(
                        "`{}`: the plan binds `{}` and the contract publishes \
                         nothing under that name",
                        sku.name, param.name,
                    ));
                }
                continue;
            };
            // The scales plane. Its producer is the payload's encode, so the
            // contract declares the payload with a cast into a quantized
            // encoding and declares nothing at all here.
            if supply.contains_key(param.name.as_str()) {
                faults.push(format!(
                    "`{}`: the contract declares `{}`, and the encode of `{stem}` \
                     already publishes it -- two producers for one plane",
                    sku.name, param.name,
                ));
            }
            let Some(payload) = supply.get(stem) else {
                faults.push(format!(
                    "`{}`: the plan binds `{}` and the contract publishes no \
                     `{stem}` whose encode would produce it",
                    sku.name, param.name,
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
                    sku.name, param.name,
                ));
            }
        }
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

/// The same fixture writer, for a checkpoint the loader will have to quantize.
///
/// Two differences from [`write_checkpoint`], and both are what makes the file
/// an unquantized one rather than the same file with a flag: an mxfp4 param is
/// written as the BF16 values it was quantized FROM, and its `.scales`
/// companion is not written at all. The mxfp4 plane's declared shape is its
/// codes' — leading axes, then blocks of `GROUP`, then the sixteen bytes one
/// block packs into — so the values it holds are one axis shorter.
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
                let data = [0u8, 0u8];
                writer
                    .object(param.name.as_str(), |o| {
                        o.shape(vec![1u64; logical])
                            .part("data", |p| p.dtype(ztensor::DType::BF16).bytes(&data))
                    })
                    .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
            }
            _ => state(&mut writer, param),
        }
    }
    writer
        .finish()
        .unwrap_or_else(|why| panic!("{}: {why}", path.display()));
}



/// **A FILE OF A TEXT'S OWN PLANES IS IMPORTED AS A LOAD** (§M-4a).
///
/// Every `IMPORTS` row opens by sniffing a WITNESS NAME — the embedding, in
/// transformers' or mlx_lm's or GGUF's spelling — and no such name is in a
/// file whose tensors are this text's plane names. So before §M-4a the
/// artifact `pie model import` writes was refused by the very table that
/// would have to read it, and the 194 `read_own` statements every family
/// wrote for it had no production caller at all.
///
/// This is that door, asserted through the public table rather than through
/// `Model::load` directly: `models::import_of(sku)` is what
/// `runtime::engine::load::{identify, contract_for, conversion_contract}`
/// reach, so a green run here is the statement that a serving load can open
/// an artifact it wrote.
///
/// **AND THE TWO DOORS MUST AGREE, WHICH IS WHY THE CONTRACTS ARE COMPARED
/// AND NOT MERELY BOTH ACCEPTED.** `import` answering `Ok` over a file of
/// landed planes would be worth little if it answered with a DIFFERENT
/// contract — one that had gone down a foreign arm and found the names by
/// luck. Equality with the `load` contract is what says it took the native
/// door.
///
/// **AND EVERY ROW NOW, NOT THE tp1 ONES ONLY** (§M-4g). This read "tp = 1
/// only: every `IMPORTS` row builds its model at one rank (`qwen_3.rs`'s
/// tp2-named row included), so a tp2 catalog row has no import to compare
/// against and the two contracts would differ by every band" — a true sentence
/// about a table that was wrong, and a prediction that turns out to be false
/// in its second half. The rows are built at their catalog degree now, so the
/// guard goes; and the six tp2 rows pass without a band moving, because a
/// `read_own` landing is tp-invariant (`checkpoint_dsl::claim` declares
/// `per-rank × tp` and an `Expr::Shard` carries an axis and no world). That
/// invariance was what made the disagreement survivable, and it is worth
/// having a test SAY it rather than a comment predict the opposite.
#[test]
fn an_artifact_of_a_texts_own_planes_imports_as_a_load() {
    let dir = scratch();
    let mut faults = Vec::new();

    for row in stated() {
        let Some(import) = models::import_of(row.name) else {
            faults.push(format!("`{}` names no import row", row.name));
            continue;
        };
        let path = dir.join(format!("{}.zt", row.name));
        write_checkpoint(&path, &row.params);
        let src = ztensor::Source::open(&path).unwrap_or_else(|why| {
            panic!(
                "`{}`: the artifact just written does not open again: {why}",
                row.name
            )
        });
        match import(&src) {
            Ok(native) if native == row.contract => {}
            Ok(_) => faults.push(format!(
                "`{}` imports a file of its own planes and states a DIFFERENT \
                 contract than `Model::load` does over the same bytes; the \
                 native door was not the one it took",
                row.name
            )),
            Err(why) => faults.push(format!(
                "`{}` refuses a file holding every plane it declares, under \
                 the names it declares them by — which is the artifact \
                 `pie model import` writes out of this very text: {why}",
                row.name
            )),
        }
        drop(src);
    }

    let _ = std::fs::remove_dir_all(&dir);
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}

/// **THE TWO TABLES AGREE ON HOW MANY RANKS A ROW IS FOR** (§M-4g).
///
/// `models::catalog()` states the width a SKU is TRACED for — what its bands
/// are cut for, and what §M's artifact stamp will say the file is — and
/// `models::imports()` states the width the contract that LANDS those bands is
/// built at. Two hand-written numbers about one deployment, and until this
/// test they were never compared: `models::kimi_k3::IMPORTS` said 1 where its
/// catalog row said 2, and so did the other five tp2 rows, and the only test
/// that could have seen it hard-coded its OWN correct 2 in `skus()` instead of
/// reading the shipped table.
///
/// **AND IT READS BOTH FROM THEIR SOURCES**, through
/// `models::tp_disagreements`, which walks the family `const`s directly.
/// `models::imports()` asserts on the same list — so the refusal stands on
/// every production path and not only under `cargo test` — and a walk that
/// went through `imports()` could report at most the first fault. This reports
/// them all.
#[test]
fn the_two_tables_agree_on_how_many_ranks_a_row_is_for() {
    let faults = models::tp_disagreements();
    assert!(faults.is_empty(), "\n{}\n", faults.join("\n"));
}
