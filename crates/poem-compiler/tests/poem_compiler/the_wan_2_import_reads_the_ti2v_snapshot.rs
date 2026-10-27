use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::wan_2_dims::Dims;
use checkpoint::contract::infer::{CheckpointTypes, Resolver};
use checkpoint::contract::{ModelContract, Partition, TensorType};
use checkpoint::plan::StorageTarget;
use poem::Platform;
use ztensor::Leaf;
use ztensor::provide::{Catalog, Entry, Location, Store, StoreId};

const TI2V: &str = "wan22-ti2v-5b-bf16-kv-bf16";
const D128: &str = "wan22-mini-d128-bf16-kv-bf16";
const NANO: &str = "wan22-mini-nano-bf16-kv-bf16";

type Named = (String, Vec<u64>, Leaf);

fn transformer(d: &Dims, leaf: Leaf) -> Vec<Named> {
    let dim = u64::from(d.dim);
    let ffn = u64::from(d.ffn);
    let mut out: Vec<Named> = Vec::new();
    let mut push = |name: String, shape: Vec<u64>| out.push((name, shape, leaf));
    let linear = |push: &mut dyn FnMut(String, Vec<u64>), stem: &str, out_: u64, in_: u64| {
        push(format!("{stem}.weight"), vec![out_, in_]);
        push(format!("{stem}.bias"), vec![out_]);
    };
    for i in 0..d.layers {
        let n = |s: &str| format!("blocks.{i}.{s}");
        for attn in ["attn1", "attn2"] {
            for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
                linear(&mut push, &n(&format!("{attn}.{proj}")), dim, dim);
            }
            push(n(&format!("{attn}.norm_q.weight")), vec![dim]);
            push(n(&format!("{attn}.norm_k.weight")), vec![dim]);
        }
        linear(&mut push, &n("ffn.net.0.proj"), ffn, dim);
        linear(&mut push, &n("ffn.net.2"), dim, ffn);
        push(n("norm2.weight"), vec![dim]);
        push(n("norm2.bias"), vec![dim]);
        push(n("scale_shift_table"), vec![1, 6, dim]);
    }
    linear(
        &mut push,
        "condition_embedder.text_embedder.linear_1",
        dim,
        u64::from(d.text_dim),
    );
    linear(
        &mut push,
        "condition_embedder.text_embedder.linear_2",
        dim,
        dim,
    );
    linear(
        &mut push,
        "condition_embedder.time_embedder.linear_1",
        dim,
        u64::from(d.freq_dim),
    );
    linear(
        &mut push,
        "condition_embedder.time_embedder.linear_2",
        dim,
        dim,
    );
    linear(&mut push, "condition_embedder.time_proj", 6 * dim, dim);
    push(
        "patch_embedding.weight".into(),
        vec![dim, u64::from(d.in_channels), 1, 2, 2],
    );
    push("patch_embedding.bias".into(), vec![dim]);
    linear(&mut push, "proj_out", u64::from(d.patch_out()), dim);
    push("scale_shift_table".into(), vec![1, 2, dim]);
    out
}

fn prefixed(prefix: &str, tensors: Vec<Named>) -> Vec<Named> {
    tensors
        .into_iter()
        .map(|(name, shape, leaf)| (format!("{prefix}{name}"), shape, leaf))
        .collect()
}

fn bytes_of(leaf: Leaf) -> u64 {
    match leaf {
        Leaf::F32 => 4,
        Leaf::BF16 => 2,
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
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "wan_2_import_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    dir
}

struct Types<'a>(&'a ztensor::Source);

impl CheckpointTypes for Types<'_> {
    fn tensor_type(&self, name: &str) -> Option<TensorType> {
        let tensor = self.0.get(name)?;
        let encoding = checkpoint::file::encoding_of(&tensor).ok()?;
        Some(TensorType {
            shape: tensor.shape().iter().map(|&n| n as i64).collect(),
            encoding,
        })
    }
}

fn reads(contract: &ModelContract) -> BTreeMap<String, usize> {
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for tensor in &contract.tensors {
        for source in tensor.expr.sources() {
            *counts.entry(source.to_string()).or_default() += 1;
        }
    }
    counts
}

fn type_checks(contract: &ModelContract, src: &ztensor::Source) {
    let types = Types(src);
    let mut resolver = Resolver::new(&types, Partition::WHOLE);
    for tensor in &contract.tensors {
        let ty = resolver
            .infer(&tensor.expr, &tensor.name)
            .unwrap_or_else(|why| panic!("`{}` does not type: {why}", tensor.name));
        if let Some(shape) = &tensor.shape {
            assert_eq!(
                &ty.shape, shape,
                "`{}` declares {shape:?} and its expression yields {:?}",
                tensor.name, ty.shape
            );
        }
        assert_eq!(
            ty.encoding, tensor.encoding,
            "`{}` declares {:?} and its expression yields {:?}",
            tensor.name, tensor.encoding, ty.encoding
        );
        resolver.publish(&tensor.name, ty);
    }
    let metadata = checkpoint::file::zt::describe(src)
        .unwrap_or_else(|why| panic!("the source does not describe: {why}"));
    let plan = checkpoint::plan::compile(&metadata, contract, StorageTarget::default())
        .unwrap_or_else(|why| panic!("the contract does not lower to a load plan: {why}"));
    assert!(!plan.instrs.is_empty());
}

fn expected_dit_reads(prefix: &str, d: &Dims) -> BTreeMap<String, usize> {
    transformer(d, Leaf::F32)
        .into_iter()
        .map(|(name, ..)| {
            let count = if name == "scale_shift_table" {
                2
            } else if name.ends_with(".scale_shift_table")
                || name.starts_with("condition_embedder.time_proj.")
            {
                6
            } else if name.starts_with("condition_embedder.time_embedder.linear_2.") {
                3
            } else {
                1
            };
            (format!("{prefix}{name}"), count)
        })
        .collect()
}

fn check_mini(deployment: &str, d: &Dims, src: &ztensor::Source, prefix: &str) {
    let row =
        poem_compiler::catalog::deployment(deployment).expect("the catalog ships the miniature");
    let contract = row
        .contract(src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("`{deployment}` does not read its checkpoint: {why}"));
    assert_eq!(
        reads(&contract),
        expected_dit_reads(prefix, d),
        "`{deployment}` reads its whole state_dict at the counts the cuts imply"
    );
    type_checks(&contract, src);
}

#[test]
fn the_wan_2_import_reads_the_ti2v_snapshot_every_case() {
    each_miniature_reads_a_synthetic_state_dict_bare_and_prefixed();
    the_flagship_refuses_a_bare_miniature();
}

fn each_miniature_reads_a_synthetic_state_dict_bare_and_prefixed() {
    for (deployment, d) in [(D128, Dims::mini_d128()), (NANO, Dims::mini_nano())] {
        for prefix in ["", "dit."] {
            let dir = scratch();
            let tensors = prefixed(prefix, transformer(&d, Leaf::F32));
            assert_eq!(tensors.len(), 69, "the golden's config lists 69 tensors");
            let src = synthetic(&dir, &tensors);
            check_mini(deployment, &d, &src, prefix);
            drop(src);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

fn the_flagship_refuses_a_bare_miniature() {
    let dir = scratch();
    let bare = transformer(&Dims::mini_d128(), Leaf::F32);
    let src = synthetic(&dir, &bare);
    let flagship = poem_compiler::catalog::deployment(TI2V).unwrap();
    assert!(
        flagship.contract(&src, Platform::Cuda).is_err(),
        "the flagship read a bare 256-wide transformer with no encoder"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
