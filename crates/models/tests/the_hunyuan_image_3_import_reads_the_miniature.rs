use std::collections::BTreeMap;
use std::path::PathBuf;

use checkpoint::contract::Expr;
use model_dsl::{Dtype, Platform};
use models::hunyuan_image_3::model::{Dims, Model};

const MINI: &str = "hunyuanimage3-mini-bf16-kv-bf16";

fn golden() -> PathBuf {
    let root = std::env::var("PIE_IMAGEGEN_GOLDEN")
        .unwrap_or_else(|_| "/root/.cache/pie-imagegen/golden".to_string());
    PathBuf::from(root).join("hy3").join("hy3_mini.safetensors")
}

#[test]
fn the_hunyuan_image_3_import_reads_the_miniature_every_case() {
    the_miniature_reads_the_golden_and_rearranges_where_the_study_says();
    the_rotary_channel_permutation_is_one();
}

fn the_miniature_reads_the_golden_and_rearranges_where_the_study_says() {
    let path = golden();
    if !path.exists() {
        eprintln!(
            "skipped: no `{}` (run `scripts/imagegen/hy3_golden.py --mini`)",
            path.display()
        );
        return;
    }
    let src = ztensor_compat::open(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let row = models::sku(MINI).expect("this build ships the miniature");
    let contract = row
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the miniature does not read its own golden: {why}"));

    let mut reads: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &contract.tensors {
        for name in tensor.expr.sources() {
            *reads.entry(name.to_string()).or_default() += 1;
        }
    }
    let held: Vec<String> = src.names().map(|n| n.to_string()).collect();
    let missed: Vec<&String> = held
        .iter()
        .filter(|name| !reads.contains_key(*name))
        .collect();
    assert!(
        missed.is_empty(),
        "the golden holds {} tensors this import never reads: {:?}",
        missed.len(),
        &missed[..missed.len().min(12)]
    );
    for name in reads.keys() {
        assert!(
            held.contains(name),
            "the import reads `{name}`, which the golden does not hold"
        );
    }

    let d = Dims::mini();
    let mut gathers: Vec<usize> = Vec::new();
    let mut concats = 0usize;
    for tensor in &contract.tensors {
        tensor.expr.visit(&mut |e| match e {
            Expr::Gather { indices, .. } => gathers.push(indices.len()),
            Expr::Concat { .. } => concats += 1,
            _ => {}
        });
    }
    let qkv_rows = (d.q_heads + 2 * d.kv_heads) as usize * d.head_dim as usize;
    let per_layer = [qkv_rows, d.head_dim as usize, d.head_dim as usize];
    let layers = d.layers as usize;
    let mut counts: BTreeMap<usize, usize> = BTreeMap::new();
    for n in &gathers {
        *counts.entry(*n).or_default() += 1;
    }
    assert_eq!(
        counts.get(&qkv_rows),
        Some(&layers),
        "one `qkv_proj` gather a layer, {qkv_rows} rows wide: {counts:?}"
    );
    assert_eq!(
        counts.get(&(d.head_dim as usize)),
        Some(&(2 * layers)),
        "the two QK-norm gains are permuted with the channels they scale"
    );
    assert_eq!(
        gathers.len(),
        layers * per_layer.len(),
        "and nothing else gathers"
    );
    assert_eq!(
        concats,
        layers * (3 + d.experts as usize) + 3,
        "the expert banks stack, every `gate_and_up_proj` swaps its halves, \
         and `timestep_emb.mlp.2` is stacked twice"
    );

    let ones = contract
        .tensors
        .iter()
        .find(|t| t.name == "special.ones")
        .expect("the flag bank is declared");
    assert!(
        ones.expr.sources().is_empty(),
        "a constant reads no checkpoint tensor"
    );

    let model = Model::mini(Dtype::Bf16, Dtype::Bf16, 1);
    assert_eq!(model.dims, d);
    assert_eq!(model.layers.len(), layers);
}

fn the_rotary_channel_permutation_is_one() {
    for head_dim in [64u32, 128] {
        let scale = models::hunyuan_image_3::model::rope_x_scale(head_dim);
        let want = 10_000f32.powf(-2.0 / head_dim as f32);
        assert!((scale - want).abs() < 1e-9, "head {head_dim}");
        assert!(head_dim.is_multiple_of(4));
    }
}
