use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use checkpoint::contract::Partition;
use checkpoint::contract::infer::{CheckpointTypes, Resolver};
use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint::plan::StorageTarget;
use model_dsl::Platform;
use models::z_image::model::{self, Dims};
use ztensor::Leaf;
use ztensor::provide::{Catalog, Entry, Location, Store, StoreId};

const TURBO: &str = "z-image-turbo-bf16-kv-bf16";
const MINI: &str = "z-image-mini-bf16-kv-bf16";

type Named = (String, Vec<u64>, Leaf);

fn transformer(d: &Dims, leaf: Leaf) -> Vec<Named> {
    let dim = u64::from(d.dim);
    let hd = u64::from(d.head_dim);
    let inter = u64::from(d.inter);
    let mut out: Vec<Named> = Vec::new();
    let mut push = |name: String, shape: Vec<u64>| out.push((name, shape, leaf));
    push("all_x_embedder.2-1.weight".into(), vec![dim, 64]);
    push("all_x_embedder.2-1.bias".into(), vec![dim]);
    push(
        "all_final_layer.2-1.adaLN_modulation.1.weight".into(),
        vec![dim, 256],
    );
    push(
        "all_final_layer.2-1.adaLN_modulation.1.bias".into(),
        vec![dim],
    );
    push("all_final_layer.2-1.linear.weight".into(), vec![64, dim]);
    push("all_final_layer.2-1.linear.bias".into(), vec![64]);
    push("cap_embedder.0.weight".into(), vec![u64::from(d.cap_width)]);
    push(
        "cap_embedder.1.weight".into(),
        vec![dim, u64::from(d.cap_width)],
    );
    push("cap_embedder.1.bias".into(), vec![dim]);
    push("cap_pad_token".into(), vec![1, dim]);
    push("x_pad_token".into(), vec![1, dim]);
    push("t_embedder.mlp.0.weight".into(), vec![1024, 256]);
    push("t_embedder.mlp.0.bias".into(), vec![1024]);
    push("t_embedder.mlp.2.weight".into(), vec![256, 1024]);
    push("t_embedder.mlp.2.bias".into(), vec![256]);
    for (stem, count, modulated) in [
        ("noise_refiner", d.refiner_layers, true),
        ("context_refiner", d.refiner_layers, false),
        ("layers", d.joint_layers, true),
    ] {
        for i in 0..count {
            let n = |s: &str| format!("{stem}.{i}.{s}");
            if modulated {
                push(n("adaLN_modulation.0.weight"), vec![4 * dim, 256]);
                push(n("adaLN_modulation.0.bias"), vec![4 * dim]);
            }
            for norm in [
                "attention_norm1",
                "attention_norm2",
                "ffn_norm1",
                "ffn_norm2",
            ] {
                push(n(&format!("{norm}.weight")), vec![dim]);
            }
            for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
                push(n(&format!("attention.{proj}.weight")), vec![dim, dim]);
            }
            push(n("attention.norm_q.weight"), vec![hd]);
            push(n("attention.norm_k.weight"), vec![hd]);
            push(n("feed_forward.w1.weight"), vec![inter, dim]);
            push(n("feed_forward.w3.weight"), vec![inter, dim]);
            push(n("feed_forward.w2.weight"), vec![dim, inter]);
        }
    }
    out
}

fn text_encoder() -> Vec<Named> {
    let hidden = u64::from(model::TE_HIDDEN);
    let q = u64::from(model::TE_Q_HEADS * model::TE_HEAD_DIM);
    let kv = u64::from(model::TE_KV_HEADS * model::TE_HEAD_DIM);
    let hd = u64::from(model::TE_HEAD_DIM);
    let inter = u64::from(model::TE_INTER);
    let mut out: Vec<Named> = Vec::new();
    let mut push = |name: String, shape: Vec<u64>| out.push((name, shape, Leaf::BF16));
    push(
        "model.embed_tokens.weight".into(),
        vec![u64::from(model::TE_VOCAB), hidden],
    );
    push("model.norm.weight".into(), vec![hidden]);
    for l in 0..model::TE_DEPTH {
        let n = |s: &str| format!("model.layers.{l}.{s}");
        push(n("input_layernorm.weight"), vec![hidden]);
        push(n("post_attention_layernorm.weight"), vec![hidden]);
        push(n("self_attn.q_proj.weight"), vec![q, hidden]);
        push(n("self_attn.k_proj.weight"), vec![kv, hidden]);
        push(n("self_attn.v_proj.weight"), vec![kv, hidden]);
        push(n("self_attn.o_proj.weight"), vec![hidden, q]);
        push(n("self_attn.q_norm.weight"), vec![hd]);
        push(n("self_attn.k_norm.weight"), vec![hd]);
        push(n("mlp.gate_proj.weight"), vec![inter, hidden]);
        push(n("mlp.up_proj.weight"), vec![inter, hidden]);
        push(n("mlp.down_proj.weight"), vec![hidden, inter]);
    }
    out
}

fn vae() -> Vec<Named> {
    let channels: [u64; 4] = [128, 256, 512, 512];
    let mut out: Vec<Named> = Vec::new();
    let mut push = |name: String, shape: Vec<u64>| out.push((name, shape, Leaf::BF16));
    fn norm(push: &mut impl FnMut(String, Vec<u64>), stem: &str, c: u64) {
        push(format!("{stem}.weight"), vec![c]);
        push(format!("{stem}.bias"), vec![c]);
    }
    fn conv(push: &mut impl FnMut(String, Vec<u64>), stem: &str, c_out: u64, c_in: u64, k: u64) {
        push(format!("{stem}.weight"), vec![c_out, c_in, k, k]);
        push(format!("{stem}.bias"), vec![c_out]);
    }
    fn resnet(push: &mut impl FnMut(String, Vec<u64>), stem: &str, c_in: u64, c_out: u64) {
        norm(push, &format!("{stem}.norm1"), c_in);
        conv(push, &format!("{stem}.conv1"), c_out, c_in, 3);
        norm(push, &format!("{stem}.norm2"), c_out);
        conv(push, &format!("{stem}.conv2"), c_out, c_out, 3);
        if c_in != c_out {
            conv(push, &format!("{stem}.conv_shortcut"), c_out, c_in, 1);
        }
    }
    fn mid(push: &mut impl FnMut(String, Vec<u64>), stem: &str, c: u64) {
        resnet(push, &format!("{stem}.resnets.0"), c, c);
        norm(push, &format!("{stem}.attentions.0.group_norm"), c);
        for proj in ["to_q", "to_k", "to_v", "to_out.0"] {
            push(format!("{stem}.attentions.0.{proj}.weight"), vec![c, c]);
            push(format!("{stem}.attentions.0.{proj}.bias"), vec![c]);
        }
        resnet(push, &format!("{stem}.resnets.1"), c, c);
    }
    conv(&mut push, "decoder.conv_in", 512, 16, 3);
    mid(&mut push, "decoder.mid_block", 512);
    let mut c_prev = 512;
    for (i, &c) in channels.iter().rev().enumerate() {
        for r in 0..3 {
            resnet(
                &mut push,
                &format!("decoder.up_blocks.{i}.resnets.{r}"),
                c_prev,
                c,
            );
            c_prev = c;
        }
        if i < 3 {
            conv(
                &mut push,
                &format!("decoder.up_blocks.{i}.upsamplers.0.conv"),
                c,
                c,
                3,
            );
        }
    }
    norm(&mut push, "decoder.conv_norm_out", 128);
    conv(&mut push, "decoder.conv_out", 3, 128, 3);
    conv(&mut push, "encoder.conv_in", 128, 3, 3);
    let mut c_prev = 128;
    for (i, &c) in channels.iter().enumerate() {
        for r in 0..2 {
            resnet(
                &mut push,
                &format!("encoder.down_blocks.{i}.resnets.{r}"),
                c_prev,
                c,
            );
            c_prev = c;
        }
        if i < 3 {
            conv(
                &mut push,
                &format!("encoder.down_blocks.{i}.downsamplers.0.conv"),
                c,
                c,
                3,
            );
        }
    }
    mid(&mut push, "encoder.mid_block", 512);
    norm(&mut push, "encoder.conv_norm_out", 512);
    conv(&mut push, "encoder.conv_out", 32, 512, 3);
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
        "z_image_import_{}_{}",
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

fn check_turbo(src: &ztensor::Source, index: &BTreeSet<String>) {
    let row = models::sku(TURBO).expect("the catalog ships the flagship");
    let contract = row
        .contract(src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the flagship does not read this checkpoint: {why}"));
    let counts = reads(&contract);

    let dit_index: BTreeSet<&String> = index.iter().filter(|n| n.starts_with("dit.")).collect();
    let dit_read: BTreeSet<&String> = counts.keys().filter(|n| n.starts_with("dit.")).collect();
    let unread: Vec<&&String> = dit_index.difference(&dit_read).collect();
    assert!(
        unread.is_empty(),
        "`dit.` tensors the import never reads: {unread:?}"
    );
    let phantom: Vec<&&String> = dit_read.difference(&dit_index).collect();
    assert!(
        phantom.is_empty(),
        "`dit.` names read that the index lacks: {phantom:?}"
    );

    let te_read: BTreeSet<&String> = counts.keys().filter(|n| n.starts_with("te.")).collect();
    let skipped_layer = format!("te.model.layers.{}.", model::TE_DEPTH - 1);
    let te_want: BTreeSet<&String> = index
        .iter()
        .filter(|n| n.starts_with("te."))
        .filter(|n| !n.starts_with(&skipped_layer) && n.as_str() != "te.model.norm.weight")
        .collect();
    assert_eq!(
        te_read, te_want,
        "the encoder planes read are the layers the plan runs"
    );
    assert_eq!(te_want.len(), 1 + 11 * model::TE_LAYERS as usize);

    let vae_index: BTreeSet<&String> = index.iter().filter(|n| n.starts_with("vae.")).collect();
    let vae_read: BTreeSet<&String> = counts.keys().filter(|n| n.starts_with("vae.")).collect();
    let unread: Vec<&&String> = vae_index.difference(&vae_read).collect();
    assert!(
        unread.is_empty(),
        "`vae.` tensors the import never reads: {unread:?}"
    );
    let phantom: Vec<&&String> = vae_read.difference(&vae_index).collect();
    assert!(
        phantom.is_empty(),
        "`vae.` names read that the index lacks: {phantom:?}"
    );
    assert_eq!(vae_index.len(), 244, "the FLUX VAE is 244 tensors");

    let twice: BTreeSet<String> = counts
        .iter()
        .filter(|(_, count)| **count != 1)
        .map(|(name, _)| name.clone())
        .collect();
    assert!(
        twice.is_empty(),
        "every tensor is read exactly once; these are not: {twice:?}"
    );

    type_checks(&contract, src);
}

fn check_mini(src: &ztensor::Source, index: &BTreeSet<String>) {
    let row = models::sku(MINI).expect("the catalog ships the miniature");
    let contract = row
        .contract(src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the miniature does not read its checkpoint: {why}"));
    let counts = reads(&contract);
    let read: BTreeSet<&String> = counts.keys().collect();
    let want: BTreeSet<&String> = index.iter().collect();
    assert_eq!(
        read, want,
        "the miniature reads its whole state_dict and nothing else"
    );
    let twice: BTreeSet<String> = counts
        .iter()
        .filter(|(_, count)| **count != 1)
        .map(|(name, _)| name.clone())
        .collect();
    assert!(
        twice.is_empty(),
        "every tensor is read exactly once; these are not: {twice:?}"
    );
    type_checks(&contract, src);
}

#[test]
fn the_z_image_import_reads_the_diffusers_pipeline_every_case() {
    the_flagship_reads_a_synthetic_pipeline_shaped_like_the_snapshot();
    the_miniature_reads_its_bare_state_dict_and_the_same_names_prefixed();
    neither_row_serves_the_other_rows_checkpoint();
    the_flagship_reads_the_real_snapshot();
    the_derived_planes_are_stated_through_internal_steps();
}

fn the_flagship_reads_a_synthetic_pipeline_shaped_like_the_snapshot() {
    let dir = scratch();
    let mut tensors = prefixed("dit.", transformer(&Dims::turbo(), Leaf::F32));
    tensors.extend(prefixed("te.", text_encoder()));
    tensors.extend(prefixed("vae.", vae()));
    let index: BTreeSet<String> = tensors.iter().map(|(name, ..)| name.clone()).collect();
    assert_eq!(index.iter().filter(|n| n.starts_with("dit.")).count(), 521);
    assert_eq!(index.iter().filter(|n| n.starts_with("te.")).count(), 398);
    assert_eq!(index.iter().filter(|n| n.starts_with("vae.")).count(), 244);
    let src = synthetic(&dir, &tensors);
    check_turbo(&src, &index);
    drop(src);
    let _ = std::fs::remove_dir_all(&dir);
}

fn the_miniature_reads_its_bare_state_dict_and_the_same_names_prefixed() {
    for prefix in ["", "dit."] {
        let dir = scratch();
        let tensors = prefixed(prefix, transformer(&Dims::mini(), Leaf::F32));
        let index: BTreeSet<String> = tensors.iter().map(|(name, ..)| name.clone()).collect();
        let src = synthetic(&dir, &tensors);
        check_mini(&src, &index);
        drop(src);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

fn neither_row_serves_the_other_rows_checkpoint() {
    let dir = scratch();
    let turbo = prefixed("dit.", transformer(&Dims::turbo(), Leaf::F32));
    let src = synthetic(&dir, &turbo);
    let mini = models::sku(MINI).unwrap();
    let contract = mini
        .contract(&src, Platform::Cuda)
        .expect("the names are there; a raw read is not shape-checked at build");
    let types = Types(&src);
    let mut resolver = Resolver::new(&types, Partition::WHOLE);
    let mistyped = contract.tensors.iter().find(|tensor| {
        let ty = resolver.infer(&tensor.expr, &tensor.name);
        match ty {
            Ok(ty) => {
                let wrong = tensor
                    .shape
                    .as_ref()
                    .is_some_and(|shape| *shape != ty.shape);
                resolver.publish(&tensor.name, ty);
                wrong
            }
            Err(_) => true,
        }
    });
    assert!(
        mistyped.is_some(),
        "the miniature typed every plane of a 3840-wide transformer"
    );
    let bare = transformer(&Dims::mini(), Leaf::F32);
    let src = synthetic(&dir, &bare);
    let flagship = models::sku(TURBO).unwrap();
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
    let snapshots = hub().join("models--Tongyi-MAI--Z-Image-Turbo/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path())
        .find(|path| path.join("model_index.json").is_file())
}

fn the_flagship_reads_the_real_snapshot() {
    let Some(root) = snapshot() else {
        eprintln!("skipping: no Tongyi-MAI/Z-Image-Turbo snapshot in the HuggingFace cache");
        return;
    };
    let src = checkpoint::file::diffusers::open(&root)
        .unwrap_or_else(|why| panic!("{}: {why}", root.display()));
    let index: BTreeSet<String> = src.names().map(str::to_string).collect();

    let mut synthesized = prefixed("dit.", transformer(&Dims::turbo(), Leaf::F32));
    synthesized.extend(prefixed("te.", text_encoder()));
    synthesized.extend(prefixed("vae.", vae()));
    for (name, shape, _) in &synthesized {
        let real = src
            .get(name)
            .unwrap_or_else(|| panic!("the snapshot holds no `{name}`"));
        assert_eq!(real.shape(), shape.as_slice(), "`{name}`");
    }
    let synthesized: BTreeSet<&String> = synthesized.iter().map(|(name, ..)| name).collect();
    let real: BTreeSet<&String> = index.iter().collect();
    assert_eq!(
        synthesized, real,
        "the synthetic index and the snapshot's are one list"
    );

    check_turbo(&src, &index);

    let identified = models::identify(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the snapshot identifies as nothing: {why}"));
    assert_eq!(identified, TURBO);
}

fn the_derived_planes_are_stated_through_internal_steps() {
    let dir = scratch();
    let tensors = prefixed("dit.", transformer(&Dims::mini(), Leaf::F32));
    let src = synthetic(&dir, &tensors);
    let row = models::sku(MINI).unwrap();
    let contract = row.contract(&src, Platform::Cuda).unwrap();
    let named = |name: &str| {
        contract
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no `{name}` in the contract"))
    };
    for bank in ["dit.x_pad_mod", "dit.cap_pad_mod"] {
        let neg = named(&format!("{bank}.neg"));
        assert!(
            matches!(&neg.expr, Expr::Bias { .. }),
            "`{bank}.neg` is a biased fill"
        );
        let bank = named(bank);
        assert_eq!(bank.expr.outputs(), vec![format!("{}.neg", bank.name)]);
        assert_eq!(bank.expr.sources().len(), 1, "the token itself, once");
    }
    let flip = named("dit.t_flip");
    assert_eq!(flip.expr.outputs(), vec!["dit.t_flip.raw".to_string()]);
    assert!(
        flip.expr.sources().is_empty(),
        "a constant reads no checkpoint tensor"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
