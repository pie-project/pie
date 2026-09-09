use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use checkpoint::contract::infer::{CheckpointTypes, Resolver};
use checkpoint::contract::{ModelContract, Partition, TensorType};
use checkpoint::plan::StorageTarget;
use model_dsl::Platform;
use models::wan_2::model::{self, Dims};
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

fn text_encoder() -> Vec<Named> {
    let hidden = u64::from(model::TE_HIDDEN);
    let inner = u64::from(model::TE_HEADS * model::TE_HEAD_DIM);
    let inter = u64::from(model::TE_INTER);
    let mut out: Vec<Named> = Vec::new();
    let mut push = |name: String, shape: Vec<u64>| out.push((name, shape, Leaf::BF16));
    push(
        "shared.weight".into(),
        vec![u64::from(model::TE_VOCAB), hidden],
    );
    for l in 0..model::TE_LAYERS {
        let n = |s: &str| format!("encoder.block.{l}.{s}");
        for proj in ["q", "k", "v", "o"] {
            push(
                n(&format!("layer.0.SelfAttention.{proj}.weight")),
                vec![inner, hidden],
            );
        }
        push(
            n("layer.0.SelfAttention.relative_attention_bias.weight"),
            vec![u64::from(model::TE_BUCKETS), u64::from(model::TE_HEADS)],
        );
        push(n("layer.0.layer_norm.weight"), vec![hidden]);
        push(n("layer.1.DenseReluDense.wi_0.weight"), vec![inter, hidden]);
        push(n("layer.1.DenseReluDense.wi_1.weight"), vec![inter, hidden]);
        push(n("layer.1.DenseReluDense.wo.weight"), vec![hidden, inter]);
        push(n("layer.1.layer_norm.weight"), vec![hidden]);
    }
    push("encoder.final_layer_norm.weight".into(), vec![hidden]);
    out
}

fn vae() -> Vec<Named> {
    let dims = model::VAE_DECODER_DIMS;
    let mut out: Vec<Named> = Vec::new();
    let mut push = |name: String, shape: Vec<u64>| out.push((name, shape, Leaf::F32));
    let conv =
        |push: &mut dyn FnMut(String, Vec<u64>), stem: &str, c_out: u32, c_in: u32, k: [u64; 3]| {
            push(
                format!("{stem}.weight"),
                vec![u64::from(c_out), u64::from(c_in), k[0], k[1], k[2]],
            );
            push(format!("{stem}.bias"), vec![u64::from(c_out)]);
        };
    let resnet = |push: &mut dyn FnMut(String, Vec<u64>), stem: &str, c_in: u32, c_out: u32| {
        push(
            format!("{stem}.norm1.gamma"),
            vec![u64::from(c_in), 1, 1, 1],
        );
        conv(push, &format!("{stem}.conv1"), c_out, c_in, [3, 3, 3]);
        push(
            format!("{stem}.norm2.gamma"),
            vec![u64::from(c_out), 1, 1, 1],
        );
        conv(push, &format!("{stem}.conv2"), c_out, c_out, [3, 3, 3]);
        if c_in != c_out {
            conv(
                push,
                &format!("{stem}.conv_shortcut"),
                c_out,
                c_in,
                [1, 1, 1],
            );
        }
    };
    let z = model::VAE_Z;
    conv(&mut push, "post_quant_conv", z, z, [1, 1, 1]);
    conv(&mut push, "decoder.conv_in", dims[0], z, [3, 3, 3]);
    resnet(&mut push, "decoder.mid_block.resnets.0", dims[0], dims[0]);
    resnet(&mut push, "decoder.mid_block.resnets.1", dims[0], dims[0]);
    push(
        "decoder.mid_block.attentions.0.norm.gamma".into(),
        vec![u64::from(dims[0]), 1, 1],
    );
    push(
        "decoder.mid_block.attentions.0.to_qkv.weight".into(),
        vec![u64::from(3 * dims[0]), u64::from(dims[0]), 1, 1],
    );
    push(
        "decoder.mid_block.attentions.0.to_qkv.bias".into(),
        vec![u64::from(3 * dims[0])],
    );
    push(
        "decoder.mid_block.attentions.0.proj.weight".into(),
        vec![u64::from(dims[0]), u64::from(dims[0]), 1, 1],
    );
    push(
        "decoder.mid_block.attentions.0.proj.bias".into(),
        vec![u64::from(dims[0])],
    );
    for i in 0..4 {
        let (c_in, c_out) = (dims[i], dims[i + 1]);
        for r in 0..model::VAE_RESNETS {
            resnet(
                &mut push,
                &format!("decoder.up_blocks.{i}.resnets.{r}"),
                if r == 0 { c_in } else { c_out },
                c_out,
            );
        }
        if i != 3 {
            let stem = format!("decoder.up_blocks.{i}.upsampler");
            if model::VAE_TEMPORAL_UP[i] {
                conv(
                    &mut push,
                    &format!("{stem}.time_conv"),
                    2 * c_out,
                    c_out,
                    [3, 1, 1],
                );
            }
            push(
                format!("{stem}.resample.1.weight"),
                vec![u64::from(c_out), u64::from(c_out), 3, 3],
            );
            push(format!("{stem}.resample.1.bias"), vec![u64::from(c_out)]);
        }
    }
    push(
        "decoder.norm_out.gamma".into(),
        vec![u64::from(dims[4]), 1, 1, 1],
    );
    conv(
        &mut push,
        "decoder.conv_out",
        model::VAE_PIX_CHANNELS,
        dims[4],
        [3, 3, 3],
    );
    conv(&mut push, "encoder.conv_in", 160, 12, [3, 3, 3]);
    conv(&mut push, "quant_conv", 96, 96, [1, 1, 1]);
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

fn check_mini(sku: &str, d: &Dims, src: &ztensor::Source, prefix: &str) {
    let row = models::sku(sku).expect("the catalog ships the miniature");
    let contract = row
        .contract(src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("`{sku}` does not read its checkpoint: {why}"));
    assert_eq!(
        reads(&contract),
        expected_dit_reads(prefix, d),
        "`{sku}` reads its whole state_dict at the counts the cuts imply"
    );
    type_checks(&contract, src);
}

#[test]
fn the_wan_2_import_reads_the_ti2v_snapshot_every_case() {
    each_miniature_reads_a_synthetic_state_dict_bare_and_prefixed();
    the_flagship_refuses_a_bare_miniature();
    the_flagship_reads_the_real_snapshot();
    each_miniature_reads_its_golden_fixture();
}

fn each_miniature_reads_a_synthetic_state_dict_bare_and_prefixed() {
    for (sku, d) in [(D128, Dims::mini_d128()), (NANO, Dims::mini_nano())] {
        for prefix in ["", "dit."] {
            let dir = scratch();
            let tensors = prefixed(prefix, transformer(&d, Leaf::F32));
            assert_eq!(tensors.len(), 69, "the golden's config lists 69 tensors");
            let src = synthetic(&dir, &tensors);
            check_mini(sku, &d, &src, prefix);
            drop(src);
            let _ = std::fs::remove_dir_all(&dir);
        }
    }
}

fn the_flagship_refuses_a_bare_miniature() {
    let dir = scratch();
    let bare = transformer(&Dims::mini_d128(), Leaf::F32);
    let src = synthetic(&dir, &bare);
    let flagship = models::sku(TI2V).unwrap();
    assert!(
        flagship.contract(&src, Platform::Cuda).is_err(),
        "the flagship read a bare 256-wide transformer with no encoder"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn hub() -> PathBuf {
    if let Some(dir) = std::env::var_os("HF_HUB_CACHE").filter(|v| !v.is_empty()) {
        return PathBuf::from(dir);
    }
    if let Some(home) = std::env::var_os("HF_HOME").filter(|v| !v.is_empty()) {
        return PathBuf::from(home).join("hub");
    }
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default()).join(".cache/huggingface/hub")
}

fn snapshot() -> Option<PathBuf> {
    let snapshots = hub().join("models--Wan-AI--Wan2.2-TI2V-5B-Diffusers/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("model_index.json").is_file())
}

fn the_flagship_reads_the_real_snapshot() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no Wan-AI/Wan2.2-TI2V-5B-Diffusers snapshot in the HuggingFace cache");
        return;
    };
    let src = checkpoint::file::diffusers::open(&root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let index: BTreeSet<String> = src.names().map(str::to_string).collect();

    for (component, synthesized) in [
        (
            "dit.",
            prefixed("dit.", transformer(&Dims::ti2v_5b(), Leaf::F32)),
        ),
        ("te.", prefixed("te.", text_encoder())),
    ] {
        for (name, shape, _) in &synthesized {
            let real = src
                .get(name)
                .unwrap_or_else(|| panic!("the snapshot holds no `{name}`"));
            assert_eq!(real.shape(), shape.as_slice(), "`{name}`");
        }
        let synthesized: BTreeSet<&String> = synthesized.iter().map(|(name, ..)| name).collect();
        let real: BTreeSet<&String> = index.iter().filter(|n| n.starts_with(component)).collect();
        assert_eq!(
            synthesized, real,
            "the synthetic `{component}` component and the snapshot's are one list"
        );
    }
    assert_eq!(index.iter().filter(|n| n.starts_with("dit.")).count(), 825);
    assert_eq!(index.iter().filter(|n| n.starts_with("te.")).count(), 242);
    assert_eq!(index.iter().filter(|n| n.starts_with("vae.")).count(), 196);
    for (name, shape, _) in prefixed("vae.", vae()) {
        let real = src
            .get(&name)
            .unwrap_or_else(|| panic!("the snapshot holds no `{name}`"));
        assert_eq!(real.shape(), shape.as_slice(), "`{name}`");
    }

    let row = models::sku(TI2V).expect("the catalog ships the flagship");
    let contract = row
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the flagship does not read this checkpoint: {why}"));
    let counts = reads(&contract);

    let dit: BTreeMap<String, usize> = counts
        .iter()
        .filter(|(n, _)| n.starts_with("dit."))
        .map(|(n, c)| (n.clone(), *c))
        .collect();
    assert_eq!(dit, expected_dit_reads("dit.", &Dims::ti2v_5b()));

    let te_read: BTreeSet<&String> = counts.keys().filter(|n| n.starts_with("te.")).collect();
    let te_want: BTreeSet<&String> = index.iter().filter(|n| n.starts_with("te.")).collect();
    assert_eq!(te_read, te_want, "every encoder plane is read");

    let vae_read: BTreeSet<&String> = counts.keys().filter(|n| n.starts_with("vae.")).collect();
    let vae_want: BTreeSet<&String> = index.iter().filter(|n| n.starts_with("vae.")).collect();
    assert_eq!(vae_read, vae_want, "every VAE plane is read");

    let odd: BTreeSet<&String> = counts
        .iter()
        .filter(|(n, c)| !n.starts_with("dit.") && **c != 1)
        .map(|(n, _)| n)
        .collect();
    assert!(odd.is_empty(), "read at an unexpected count: {odd:?}");

    type_checks(&contract, &src);

    let identified = models::identify(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the snapshot identifies as nothing: {why}"));
    assert_eq!(identified, TI2V);
}

fn golden(file: &str) -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let file = root.join("wan22").join(file);
    file.is_file().then_some(file)
}

fn each_miniature_reads_its_golden_fixture() {
    for (sku, d, file) in [
        (D128, Dims::mini_d128(), "wan22_mini_d128.safetensors"),
        (NANO, Dims::mini_nano(), "wan22_mini_nano.safetensors"),
    ] {
        let Some(file) = golden(file) else {
            eprintln!("skipping: no {file} under $PIE_IMAGEGEN_GOLDEN/wan22");
            continue;
        };
        let src =
            ztensor_compat::open(&file).unwrap_or_else(|why| panic!("{}: {why}", file.display()));
        let index: BTreeSet<String> = src.names().map(str::to_string).collect();
        let want: BTreeSet<String> = transformer(&d, Leaf::F32)
            .into_iter()
            .map(|(name, ..)| name)
            .collect();
        assert_eq!(index, want, "`{sku}`'s fixture is its state_dict");
        for (name, shape, _) in transformer(&d, Leaf::F32) {
            assert_eq!(
                src.get(&name).unwrap().shape(),
                shape.as_slice(),
                "`{name}`"
            );
        }
        check_mini(sku, &d, &src, "");
    }
}
