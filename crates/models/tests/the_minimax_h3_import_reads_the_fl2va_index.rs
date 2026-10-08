use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use checkpoint::contract::infer::{CheckpointTypes, Resolver};
use checkpoint::contract::{ModelContract, Partition, TensorType};
use checkpoint::plan::StorageTarget;
use poem::Platform;
use ztensor::Leaf;
use ztensor::provide::{Catalog, Entry, Location, Store, StoreId};

/// The widths and constants MiniMax H3's package declares.
#[allow(dead_code)]
mod model {
    use poem::Stream;

    pub const LATENT_CHANNELS: u32 = 24;
    pub const AUDIO_CHANNELS: u32 = 32;
    pub const VIDEO_FEATURES: u32 = LATENT_CHANNELS * 2 * 2;
    pub const HEAD_DIM: u32 = 128;
    pub const ROPE_THETA: f32 = 10_000.0;
    pub const ADALN_SLICES: u32 = 6;
    pub const MODALITIES: u32 = 3;
    pub const FINAL_SLICES: u32 = 2;
    pub const TIMESTEP_SLOTS: u32 = 4;
    pub const VIDEO_SHIFT: f32 = 12.0;
    pub const AUDIO_SHIFT: f32 = 3.0;
    pub const STEPS: u32 = 50;
    pub const TE_HIDDEN: u32 = 5120;
    pub const TE_VOCAB: u32 = 151_936;
    pub const TE_Q_HEADS: u32 = 64;
    pub const TE_KV_HEADS: u32 = 8;
    pub const TE_HEAD_DIM: u32 = 128;
    pub const TE_INTER: u32 = 25_600;
    pub const TE_DEPTH: u32 = 64;
    pub const TE_LAYERS: u32 = 50;

    pub mod port {
        pub const LATENTS: u8 = 0;
        pub const REFERENCE: u8 = 1;
        pub const AUDIO: u8 = 2;
        pub const CONTEXT: u8 = 3;
        pub const CAPTION: u8 = 0;
        pub const TIMESTEP: u8 = 0;
        pub const POSITIONS: u8 = 0;
    }

    #[must_use]
    pub const fn modality(stream: Stream) -> usize {
        match stream {
            Stream::Video | Stream::Reference | Stream::Image => 0,
            Stream::Audio => 2,
            Stream::Text | Stream::Context => 1,
        }
    }

    #[must_use]
    pub const fn timestep_slot(stream: Stream) -> u32 {
        match stream {
            Stream::Video | Stream::Text | Stream::Context => 0,
            Stream::Reference | Stream::Image => 1,
            Stream::Audio => 2,
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Dims {
        pub dim: u32,
        pub heads: u32,
        pub head_dim: u32,
        pub inter: u32,
        pub blocks: u32,
        pub refiners: u32,
        pub text_dim: u32,
        pub t_freq: u32,
        pub t_hidden: u32,
        pub t_dim: u32,
        pub rope_freqs: u32,
    }

    impl Dims {
        #[must_use]
        pub const fn h3() -> Dims {
            Dims {
                dim: 5376,
                heads: 56,
                head_dim: HEAD_DIM,
                inter: 14336,
                blocks: 50,
                refiners: 2,
                text_dim: TE_HIDDEN,
                t_freq: 256,
                t_hidden: 5376,
                t_dim: 2688,
                rope_freqs: 16,
            }
        }

        #[must_use]
        pub const fn mini() -> Dims {
            Dims {
                dim: 128,
                heads: 2,
                head_dim: 64,
                inter: 256,
                blocks: 2,
                refiners: 1,
                text_dim: 64,
                t_freq: 32,
                t_hidden: 128,
                t_dim: 64,
                rope_freqs: 8,
            }
        }

        #[must_use]
        pub const fn inner(&self) -> u32 {
            self.heads * self.head_dim
        }

        #[must_use]
        pub fn sm_scale(&self) -> f32 {
            (self.head_dim as f32).sqrt().recip()
        }

        #[must_use]
        pub const fn rope_dims(&self) -> [u32; 4] {
            let per = 2 * self.rope_freqs;
            [per, per, per, 0]
        }

        #[must_use]
        pub const fn rotary_dim(&self) -> u32 {
            6 * self.rope_freqs
        }

        #[must_use]
        pub const fn adaln_width(&self) -> u32 {
            ADALN_SLICES * self.dim
        }
    }
}

use model::Dims;

const FLAGSHIP: &str = "minimax-h3-fl2va-bf16-kv-bf16";
const MINI: &str = "minimax-h3-mini-bf16-kv-bf16";

type Named = (String, Vec<u64>, Leaf);

fn transformer(d: &Dims) -> Vec<Named> {
    let dim = u64::from(d.dim);
    let inner = u64::from(d.inner());
    let inter = u64::from(d.inter);
    let t_dim = u64::from(d.t_dim);
    let mut out: Vec<Named> = Vec::new();
    macro_rules! push {
        ($name:expr, $shape:expr, $leaf:expr $(,)?) => {
            out.push(($name, $shape, $leaf))
        };
    }
    macro_rules! linear {
        ($stem:expr, $out:expr, $in:expr, $leaf:expr $(,)?) => {{
            let stem: &str = $stem;
            push!(format!("{stem}.weight"), vec![$out, $in], $leaf);
            push!(format!("{stem}.bias"), vec![$out], $leaf);
        }};
    }
    macro_rules! block {
        ($stem:expr, $adaln:expr $(,)?) => {{
            let stem: String = $stem;
            push!(format!("{stem}.norm1.weight"), vec![dim], Leaf::BF16);
            push!(format!("{stem}.norm2.weight"), vec![dim], Leaf::BF16);
            push!(
                format!("{stem}.attn.qkv_proj.weight"),
                vec![3 * inner, dim],
                Leaf::BF16
            );
            push!(
                format!("{stem}.attn.q_norm.weight"),
                vec![u64::from(d.head_dim)],
                Leaf::BF16
            );
            push!(
                format!("{stem}.attn.k_norm.weight"),
                vec![u64::from(d.head_dim)],
                Leaf::BF16
            );
            push!(
                format!("{stem}.attn.out_proj.weight"),
                vec![dim, inner],
                Leaf::BF16
            );
            push!(
                format!("{stem}.mlp.fc1.weight"),
                vec![2 * inter, dim],
                Leaf::BF16
            );
            push!(
                format!("{stem}.mlp.fc2.weight"),
                vec![dim, inter],
                Leaf::BF16
            );
            if $adaln {
                let rows = u64::from(model::ADALN_SLICES * model::MODALITIES) * dim;
                push!(
                    format!("{stem}.adaln_proj.linear.weight"),
                    vec![rows, t_dim],
                    Leaf::BF16
                );
                push!(
                    format!("{stem}.adaln_proj.linear.bias"),
                    vec![rows],
                    Leaf::BF16
                );
            }
        }};
    }
    linear!(
        "video_patch_proj",
        dim,
        u64::from(model::VIDEO_FEATURES),
        Leaf::F32,
    );
    linear!(
        "audio_patch_proj",
        dim,
        u64::from(model::AUDIO_CHANNELS),
        Leaf::F32,
    );
    linear!("condition_proj", dim, u64::from(d.text_dim), Leaf::BF16);
    linear!(
        "time_embedder.proj_in",
        u64::from(d.t_hidden),
        u64::from(d.t_freq),
        Leaf::F32,
    );
    linear!(
        "time_embedder.proj_out",
        t_dim,
        u64::from(d.t_hidden),
        Leaf::F32,
    );
    push!(
        "rope.inv_freq".to_string(),
        vec![u64::from(d.rope_freqs)],
        Leaf::F32,
    );

    for i in 0..d.refiners {
        block!(format!("token_refiner.blocks.{i}"), false);
    }
    push!(
        "token_refiner.final_norm.weight".to_string(),
        vec![dim],
        Leaf::BF16,
    );
    for i in 0..d.blocks {
        block!(format!("blocks.{i}"), true);
    }
    push!("final_layer.norm.weight".to_string(), vec![dim], Leaf::BF16);
    linear!(
        "final_layer.adaln_proj.linear",
        u64::from(model::FINAL_SLICES) * dim,
        t_dim,
        Leaf::BF16,
    );
    linear!(
        "final_layer.video_out",
        u64::from(model::VIDEO_FEATURES),
        dim,
        Leaf::F32,
    );
    linear!(
        "final_layer.audio_out",
        u64::from(model::AUDIO_CHANNELS),
        dim,
        Leaf::F32,
    );
    out
}

fn text_encoder(depth: u32) -> Vec<Named> {
    let hidden = u64::from(model::TE_HIDDEN);
    let hd = u64::from(model::TE_HEAD_DIM);
    let inter = u64::from(model::TE_INTER);
    let q = u64::from(model::TE_Q_HEADS) * hd;
    let kv = u64::from(model::TE_KV_HEADS) * hd;
    let mut out: Vec<Named> = vec![(
        "model.language_model.embed_tokens.weight".to_string(),
        vec![u64::from(model::TE_VOCAB), hidden],
        Leaf::BF16,
    )];
    for l in 0..depth {
        let n = |s: &str| format!("model.language_model.layers.{l}.{s}");
        for (tail, shape) in [
            ("input_layernorm.weight", vec![hidden]),
            ("self_attn.q_proj.weight", vec![q, hidden]),
            ("self_attn.k_proj.weight", vec![kv, hidden]),
            ("self_attn.v_proj.weight", vec![kv, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, q]),
            ("self_attn.q_norm.weight", vec![hd]),
            ("self_attn.k_norm.weight", vec![hd]),
            ("post_attention_layernorm.weight", vec![hidden]),
            ("mlp.gate_proj.weight", vec![inter, hidden]),
            ("mlp.up_proj.weight", vec![inter, hidden]),
            ("mlp.down_proj.weight", vec![hidden, inter]),
        ] {
            out.push((n(tail), shape, Leaf::BF16));
        }
    }
    out.push((
        "model.language_model.norm.weight".to_string(),
        vec![hidden],
        Leaf::BF16,
    ));
    out.push((
        "lm_head.weight".to_string(),
        vec![u64::from(model::TE_VOCAB), hidden],
        Leaf::BF16,
    ));
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
        "minimax_h3_import_{}_{}",
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

fn expected_reads(prefix: &str, d: &Dims) -> BTreeMap<String, usize> {
    let mut want: BTreeMap<String, usize> = BTreeMap::new();
    for (name, ..) in transformer(d) {
        if name == "rope.inv_freq" {
            continue;
        }
        let count = if name.ends_with("attn.qkv_proj.weight") {
            3
        } else if name.starts_with("blocks.") && name.contains(".adaln_proj.linear.") {
            (model::MODALITIES * model::ADALN_SLICES) as usize
        } else if name.starts_with("final_layer.adaln_proj.linear.") {
            model::FINAL_SLICES as usize
        } else {
            1
        };
        want.insert(format!("{prefix}{name}"), count);
    }
    want
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

#[test]
fn the_minimax_h3_import_reads_the_fl2va_index_every_case() {
    the_flagship_reads_a_synthetic_partition_at_the_counts_its_cuts_imply();
    the_flagship_refuses_a_bare_transformer();
    the_flagships_names_are_the_partitions_index();
    the_miniature_reads_its_golden_fixture();
}

fn the_flagship_reads_a_synthetic_partition_at_the_counts_its_cuts_imply() {
    let dir = scratch();
    let d = Dims::h3();
    let mut tensors = prefixed("dit.", transformer(&d));
    tensors.extend(prefixed("te.", text_encoder(model::TE_DEPTH)));
    let src = synthetic(&dir, &tensors);

    let row = models::deployment(FLAGSHIP).expect("the catalog ships the flagship");
    let contract = row
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the flagship does not read a synthetic partition: {why}"));
    let counts = reads(&contract);

    let dit: BTreeMap<String, usize> = counts
        .iter()
        .filter(|(n, _)| n.starts_with("dit."))
        .map(|(n, c)| (n.clone(), *c))
        .collect();
    assert_eq!(dit, expected_reads("dit.", &d));

    let te_read: BTreeSet<String> = counts
        .keys()
        .filter(|n| n.starts_with("te."))
        .cloned()
        .collect();
    let te_want: BTreeSet<String> = prefixed("te.", text_encoder(model::TE_LAYERS))
        .into_iter()
        .map(|(name, ..)| name)
        .filter(|name| {
            !name.ends_with("language_model.norm.weight") && !name.ends_with("lm_head.weight")
        })
        .collect();
    assert_eq!(te_read, te_want, "the encoder planes read are the cut's");
    for (name, count) in counts.iter().filter(|(n, _)| n.starts_with("te.")) {
        assert_eq!(*count, 1, "`{name}` is read {count} times");
    }

    type_checks(&contract, &src);
    let _ = std::fs::remove_dir_all(&dir);
}

fn the_flagship_refuses_a_bare_transformer() {
    let dir = scratch();
    let src = synthetic(&dir, &transformer(&Dims::mini()));
    let row = models::deployment(FLAGSHIP).expect("the catalog ships the flagship");
    assert!(
        row.contract(&src, Platform::Cuda).is_err(),
        "the flagship read a bare 128-wide transformer with no encoder"
    );
    let mini = models::deployment(MINI).expect("the catalog ships the miniature");
    let contract = mini
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the miniature does not read its own state_dict: {why}"));
    assert_eq!(
        reads(&contract),
        expected_reads("", &Dims::mini()),
        "the miniature's counts"
    );
    type_checks(&contract, &src);
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

fn partition() -> Option<PathBuf> {
    let snapshots = hub().join("models--MiniMaxAI--MiniMax-H3/snapshots");
    std::fs::read_dir(snapshots)
        .ok()?
        .flatten()
        .map(|entry| entry.path().join("FL2VA"))
        .find(|path| path.join("model_index.json").is_file())
}

fn index_names(component: &Path) -> Option<BTreeSet<String>> {
    let mut found: Option<BTreeSet<String>> = None;
    for stem in ["diffusion_pytorch_model", "model"] {
        let path = component.join(format!("{stem}.safetensors.index.json"));
        if !path.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&path).ok()?;
        let value: serde_json::Value = serde_json::from_str(&text).ok()?;
        let map = value.get("weight_map")?.as_object()?;
        found = Some(map.keys().cloned().collect());
    }
    found
}

fn the_flagships_names_are_the_partitions_index() {
    let Some(root) = partition() else {
        eprintln!("skipping: no MiniMaxAI/MiniMax-H3 FL2VA partition in the HuggingFace cache");
        return;
    };
    let Some(dit_index) = index_names(&root.join("transformer")) else {
        eprintln!("skipping: the partition's transformer has no shard index yet");
        return;
    };
    let d = Dims::h3();
    let synthesized: BTreeSet<String> =
        transformer(&d).into_iter().map(|(name, ..)| name).collect();
    assert_eq!(
        synthesized, dit_index,
        "the transformer this text declares and the partition's index are one list"
    );

    let Some(te_index) = index_names(&root.join("text_encoder")) else {
        eprintln!("skipping the encoder half: no shard index yet");
        return;
    };
    let read: BTreeSet<String> = text_encoder(model::TE_LAYERS)
        .into_iter()
        .map(|(name, ..)| name)
        .filter(|name| {
            !name.ends_with("language_model.norm.weight") && !name.ends_with("lm_head.weight")
        })
        .collect();
    let missing: Vec<&String> = read.difference(&te_index).collect();
    assert!(
        missing.is_empty(),
        "the encoder index holds none of {missing:?}"
    );
    let left: BTreeSet<&String> = te_index.difference(&read).collect();
    for name in &left {
        let past_the_cut = (model::TE_LAYERS..model::TE_DEPTH)
            .any(|l| name.starts_with(&format!("model.language_model.layers.{l}.")));
        assert!(
            past_the_cut
                || name.starts_with("model.visual.")
                || *name == "model.language_model.norm.weight"
                || *name == "lm_head.weight",
            "`{name}` is in the encoder's index and this row neither reads it nor drops it \
             for a reason the reference states"
        );
    }
    assert!(
        left.iter().any(|name| name.starts_with("model.visual.")),
        "the shipped encoder is a VLM and its tower is in the index"
    );
}

fn golden(file: &str) -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let file = root.join("minimax_h3").join(file);
    file.is_file().then_some(file)
}

fn the_miniature_reads_its_golden_fixture() {
    let Some(path) = golden("h3_mini.safetensors") else {
        eprintln!("skipping: no h3_mini.safetensors; run `h3_golden.py --mini`");
        return;
    };
    let src = ztensor_compat::open(&path).unwrap_or_else(|why| panic!("{}: {why}", path.display()));
    let index: BTreeSet<String> = src.names().map(str::to_string).collect();
    let want: BTreeSet<String> = transformer(&Dims::mini())
        .into_iter()
        .map(|(name, ..)| name)
        .collect();
    assert_eq!(index, want, "the fixture is the miniature's state_dict");
    for (name, shape, _) in transformer(&Dims::mini()) {
        assert_eq!(
            src.get(&name)
                .unwrap_or_else(|| panic!("no `{name}`"))
                .shape(),
            shape.as_slice(),
            "`{name}`"
        );
    }
    let row = models::deployment(MINI).expect("the catalog ships the miniature");
    let contract = row
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the miniature does not read its own fixture: {why}"));
    assert_eq!(
        reads(&contract),
        expected_reads("", &Dims::mini()),
        "the miniature's counts over its fixture"
    );
    type_checks(&contract, &src);
}
