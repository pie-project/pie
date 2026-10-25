//! Ranks outnumbering a model's kv heads each hold a whole kv head, a group
//! of them in a row the same one, while they split the query heads.

use checkpoint::contract::Expr;
use poem::{Attn, CacheRow, Operation, Platform};

/// qwen 3.5 0.8b attends 8 query heads to 2 kv heads 256 wide.
const ROW: &str = "qwen35-d0.8b-bf16-kv-bf16-tp4";

#[test]
fn a_kv_head_is_held_by_a_group_of_ranks() {
    let row = poem_compiler::catalog::deployment(ROW).expect("the 0.8b row splits four ways");
    let trace = row.trace(Platform::Cuda);

    let mut plans = 0;
    for node in &trace.nodes {
        if let Operation::Attn(
            Attn::PlanPrefill {
                q_heads, kv_heads, ..
            }
            | Attn::PlanDecode {
                q_heads, kv_heads, ..
            },
        ) = &node.op
        {
            plans += 1;
            assert_eq!((*q_heads, *kv_heads), (2, 1), "a rank's heads");
        }
    }
    assert!(plans > 0, "the row plans its attention");

    for row in &trace.caches {
        if let CacheRow::Kv { name, planes, .. } = row {
            assert_eq!(planes, &[256, 256], "`{name}` holds one head a rank");
        }
    }

    // q and its gate, k and v are one product: two query heads and their
    // gates, and a kv head each, a rank.
    let qkv = trace
        .params
        .iter()
        .find(|p| p.name == JOINED)
        .unwrap_or_else(|| panic!("the row reads `{JOINED}`"));
    assert_eq!(
        qkv.shape,
        [2 * 2 * 256 + 256 + 256, 1024],
        "two query heads and their gates, one k and one v head"
    );
}

/// The 0.8b's q and gate, k and v, joined.
const JOINED: &str = "layers.3.attn.qg_proj_k_proj_v_proj";

/// A rank's k projection is the share of the whole one its group reads: the
/// four ranks read it in two groups of two.
#[test]
fn the_contract_reads_a_kv_head_once_per_group() {
    let row = poem_compiler::catalog::deployment(ROW).expect("the 0.8b row splits four ways");
    let trace = row.trace(Platform::Cuda);
    let dir = std::env::temp_dir().join(format!("kv_group_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a scratch directory");
    let path = dir.join("row.zt");
    let mut writer = ztensor::Writer::create(&path).expect("a checkpoint to write");
    let mut params: Vec<_> = trace.params.iter().collect();
    params.sort_by(|a, b| a.name.cmp(&b.name));
    for param in params {
        writer
            .add(
                param.name.as_str(),
                vec![1u64; param.shape.len()],
                ztensor::Leaf::BF16,
                &[0u8; 2],
            )
            .unwrap_or_else(|why| panic!("`{}`: {why}", param.name));
    }
    writer.finish().expect("the checkpoint closes");
    let src = ztensor::Source::open(&path).expect("the checkpoint opens");
    let contract = poem::import::own_contract(&src, &trace.params, 4, Platform::Cuda)
        .expect("the row reads a checkpoint of its own planes");
    let read = |name: &str| {
        contract
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("the contract reads `{name}`"))
            .expr
            .clone()
    };
    // The query heads cut once per rank, the one k and one v head each
    // read by a group of two.
    let qkv = || Expr::src(JOINED);
    assert_eq!(
        read(JOINED),
        Expr::concat(
            0,
            vec![
                qkv().slice(0, 0, 8 * 2 * 256).shard(0),
                qkv().slice(0, 8 * 2 * 256, 2 * 256).shard_among(0, 2),
                qkv()
                    .slice(0, 8 * 2 * 256 + 2 * 256, 2 * 256)
                    .shard_among(0, 2),
            ]
        ),
    );
    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
}
