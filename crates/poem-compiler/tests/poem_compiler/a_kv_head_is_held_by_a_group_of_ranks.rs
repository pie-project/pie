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

    // k and v are one product, a kv head each a rank.
    let kv = trace
        .params
        .iter()
        .find(|p| p.name == JOINED)
        .unwrap_or_else(|| panic!("the row reads `{JOINED}`"));
    assert_eq!(kv.shape, [256 + 256, 1024], "one k and one v head");
    let q = trace
        .params
        .iter()
        .find(|p| p.name == "layers.3.attn.qg_proj")
        .expect("the row reads its query projection");
    assert_eq!(
        q.shape,
        [2 * 2 * 256, 1024],
        "two query heads and their gates"
    );
}

/// The 0.8b's k and v, joined (its q states no heads, so stays apart).
const JOINED: &str = "layers.3.attn.k_proj_v_proj";

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
    // The one k and one v head each read by a group of two.
    let src = || Expr::src(JOINED);
    assert_eq!(
        read(JOINED),
        Expr::concat(
            0,
            vec![
                src().slice(0, 0, 2 * 256).shard_among(0, 2),
                src().slice(0, 2 * 256, 2 * 256).shard_among(0, 2),
            ]
        ),
    );
    assert_eq!(
        read("layers.3.attn.qg_proj"),
        Expr::src("layers.3.attn.qg_proj").shard(0),
        "the query heads are cut once per rank",
    );
    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
}
