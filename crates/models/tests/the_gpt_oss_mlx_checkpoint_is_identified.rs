use std::path::{Path, PathBuf};

use poem::Platform;
use ztensor::Leaf;
use ztensor::provide::{Catalog, Entry, Location, Store, StoreId};

type Named = (String, Vec<u64>, Leaf);

/// The tensors `mlx-community/gpt-oss-20b-MXFP4-Q4` stores: affine u4g64
/// trunk, the experts split into `gate_proj`/`up_proj`/`down_proj` MXFP4 words.
fn mlx_checkpoint() -> Vec<Named> {
    let mut out: Vec<Named> = Vec::new();
    let mut affine = |stem: &str, rows: u64, cols: u64| {
        out.push((format!("{stem}.weight"), vec![rows, cols / 8], Leaf::U32));
        out.push((format!("{stem}.scales"), vec![rows, cols / 64], Leaf::BF16));
        out.push((format!("{stem}.biases"), vec![rows, cols / 64], Leaf::BF16));
    };
    affine("model.embed_tokens", 201_088, 2880);
    affine("lm_head", 201_088, 2880);
    for l in 0..24 {
        let ck = |what: &str| format!("model.layers.{l}.{what}");
        affine(&ck("self_attn.q_proj"), 4096, 2880);
        affine(&ck("self_attn.k_proj"), 512, 2880);
        affine(&ck("self_attn.v_proj"), 512, 2880);
        affine(&ck("self_attn.o_proj"), 2880, 4096);
    }
    out.push(("model.norm.weight".into(), vec![2880], Leaf::BF16));
    for l in 0..24 {
        let ck = |what: &str| format!("model.layers.{l}.{what}");
        out.push((ck("input_layernorm.weight"), vec![2880], Leaf::BF16));
        out.push((
            ck("post_attention_layernorm.weight"),
            vec![2880],
            Leaf::BF16,
        ));
        for (proj, rows) in [
            ("q_proj", 4096),
            ("k_proj", 512),
            ("v_proj", 512),
            ("o_proj", 2880),
        ] {
            out.push((
                ck(&format!("self_attn.{proj}.bias")),
                vec![rows],
                Leaf::BF16,
            ));
        }
        out.push((ck("self_attn.sinks"), vec![64], Leaf::BF16));
        out.push((ck("mlp.router.weight"), vec![32, 720], Leaf::U32));
        out.push((ck("mlp.router.scales"), vec![32, 45], Leaf::BF16));
        out.push((ck("mlp.router.biases"), vec![32, 45], Leaf::BF16));
        out.push((ck("mlp.router.bias"), vec![32], Leaf::BF16));
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            out.push((
                ck(&format!("mlp.experts.{proj}.weight")),
                vec![32, 2880, 360],
                Leaf::U32,
            ));
            out.push((
                ck(&format!("mlp.experts.{proj}.scales")),
                vec![32, 2880, 90],
                Leaf::U8,
            ));
            out.push((
                ck(&format!("mlp.experts.{proj}.bias")),
                vec![32, 2880],
                Leaf::BF16,
            ));
        }
    }
    out
}

#[test]
fn the_split_experts_of_mlx_lm_are_recognized() {
    let dir = scratch();
    let src = synthetic(&dir, &mlx_checkpoint());
    let identified = models::identify(&src, Platform::Cuda).unwrap_or_else(|unmatched| {
        panic!("no deployment claims the mlx_lm checkpoint: {unmatched:?}")
    });
    assert_eq!(identified, "gptoss-20b-u4g64-mxfp4-kv-bf16");
    let _ = std::fs::remove_dir_all(&dir);
}

fn bytes_of(leaf: Leaf) -> u64 {
    match leaf {
        Leaf::U8 => 1,
        Leaf::BF16 => 2,
        Leaf::U32 => 4,
        other => panic!("no synthetic tensor here is {other:?}"),
    }
}

fn synthetic(dir: &Path, tensors: &[Named]) -> ztensor::Source {
    let path = dir.join("synthetic.bin");
    let mut catalog = Catalog::new();
    let mut offset = 0u64;
    for (name, shape, leaf) in tensors {
        let len = shape.iter().product::<u64>() * bytes_of(*leaf);
        catalog.insert(
            name.clone(),
            Entry::leaf(
                shape.clone(),
                *leaf,
                Location {
                    store: StoreId(0),
                    offset,
                    len,
                },
            ),
        );
        offset += len;
    }
    let file =
        std::fs::File::create(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    file.set_len(offset.max(1))
        .unwrap_or_else(|why| panic!("{}: a sparse file of {offset} bytes: {why}", path.display()));
    drop(file);
    let store = Store::index(&path, "safetensors").unwrap_or_else(|why| panic!("{why}"));
    ztensor::Source::from_parts(vec![store], catalog).unwrap_or_else(|why| panic!("{why}"))
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gpt_oss_mlx_identify_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}
