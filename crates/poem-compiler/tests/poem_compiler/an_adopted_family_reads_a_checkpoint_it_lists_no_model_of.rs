use poem::Platform;
use poem_compiler::catalog::repository;
use ztensor::format::cbor::Value;

fn checkpoint(
    name: &str,
    tensors: &[(String, Vec<u64>)],
    config: &serde_json::Value,
) -> ztensor::Source {
    let dir = std::env::temp_dir().join(format!("adopted_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    let path = dir.join("model.zt");
    let mut writer = ztensor::Writer::create(&path).expect("the container opens");
    let mut tensors = tensors.to_vec();
    tensors.sort();
    for (tensor, shape) in &tensors {
        let bytes = vec![0u8; 2 * shape.iter().product::<u64>() as usize];
        writer
            .add(tensor, shape.clone(), ztensor::Leaf::BF16, &bytes)
            .expect("the plane lands");
    }
    writer.finish().expect("the container closes");
    ztensor::Source::open(&path)
        .expect("it reads back")
        .with_attribute(checkpoint::file::snapshot::CONFIG, cbor(config))
}

fn cbor(value: &serde_json::Value) -> Value {
    match value {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => match n.as_u64() {
            Some(u) => Value::Uint(u),
            None => Value::Float(n.as_f64().expect("a number")),
        },
        serde_json::Value::String(s) => Value::Text(s.clone()),
        serde_json::Value::Array(items) => Value::Array(items.iter().map(cbor).collect()),
        serde_json::Value::Object(map) => Value::Map(
            map.iter()
                .map(|(k, v)| (Value::Text(k.clone()), cbor(v)))
                .collect(),
        ),
    }
}

struct Trunk {
    hidden: u64,
    layers: u64,
    q: u64,
    kv: u64,
    head_dim: u64,
    inter: u64,
    vocab: u64,
    qk_norm: bool,
    tied: bool,
}

fn reading<'a>(
    src: &'a ztensor::Source,
    config: &'a serde_json::Value,
) -> poem_compiler::kin::Checkpoint<'a> {
    static SPECIAL: std::sync::LazyLock<serde_json::Value> =
        std::sync::LazyLock::new(|| serde_json::json!({}));
    poem_compiler::kin::Checkpoint {
        src,
        config,
        special: &SPECIAL,
        origin: "a test",
    }
}

fn trunk(shape: Trunk) -> Vec<(String, Vec<u64>)> {
    let Trunk {
        hidden,
        layers,
        q,
        kv,
        head_dim,
        inter,
        vocab,
        qk_norm,
        tied,
    } = shape;
    let mut t = vec![
        ("model.embed_tokens.weight".to_string(), vec![vocab, hidden]),
        ("model.norm.weight".to_string(), vec![hidden]),
    ];
    if !tied {
        t.push(("lm_head.weight".to_string(), vec![vocab, hidden]));
    }
    for l in 0..layers {
        let at = |s: &str| format!("model.layers.{l}.{s}");
        t.push((at("input_layernorm.weight"), vec![hidden]));
        t.push((at("self_attn.q_proj.weight"), vec![q * head_dim, hidden]));
        t.push((at("self_attn.k_proj.weight"), vec![kv * head_dim, hidden]));
        t.push((at("self_attn.v_proj.weight"), vec![kv * head_dim, hidden]));
        t.push((at("self_attn.o_proj.weight"), vec![hidden, q * head_dim]));
        if qk_norm {
            t.push((at("self_attn.q_norm.weight"), vec![head_dim]));
            t.push((at("self_attn.k_norm.weight"), vec![head_dim]));
        }
        t.push((at("post_attention_layernorm.weight"), vec![hidden]));
        t.push((at("mlp.gate_proj.weight"), vec![inter, hidden]));
        t.push((at("mlp.up_proj.weight"), vec![inter, hidden]));
        t.push((at("mlp.down_proj.weight"), vec![hidden, inter]));
    }
    t
}

#[test]
fn a_qwen3_at_unlisted_widths_is_adopted_and_read() {
    let config = serde_json::json!({
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 64,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "intermediate_size": 192,
        "vocab_size": 151936,
        "tie_word_embeddings": true,
        "rope_theta": 1000000,
        "rms_norm_eps": 1e-6,
        "rope_scaling": null,
        "hidden_act": "silu",
        "attention_bias": false,
        "use_sliding_window": false,
    });
    let src = checkpoint(
        "qwen3",
        &trunk(Trunk {
            hidden: 64,
            layers: 3,
            q: 4,
            kv: 2,
            head_dim: 32,
            inter: 192,
            vocab: 151936,
            qk_norm: true,
            tied: true,
        }),
        &config,
    );
    let catalog = repository();
    assert!(
        catalog.identify(&src, Platform::Cuda).is_err(),
        "no listed model is built at these numbers"
    );

    let kin = catalog.kin(&src, Platform::Cuda);
    let qwen3 = kin
        .iter()
        .find(|k| k.package.name() == "qwen3_dense")
        .expect("Qwen 3's formats recognize it by its per-head q norm");
    assert!(qwen3.adopts);
    let departing: Vec<&str> = qwen3
        .nearest
        .departing
        .iter()
        .map(|a| a.key.as_str())
        .collect();
    assert!(departing.contains(&"hidden_size"), "{departing:?}");
    assert!(
        !kin.iter().any(|k| k.package.name() == "llama_3"),
        "Llama 3's formats leave a q-normed trunk alone"
    );

    let adopted = catalog
        .adopt(
            qwen3,
            "adopted_qwen3_tiny",
            "qwen3-tiny",
            &reading(&src, &config),
            Platform::Cuda,
        )
        .expect("the family adopts it");
    assert_eq!(adopted.adoption.rebinds, ["DEPLOYMENTS", "DIMS", "MODELS"]);
    let (bf16, read) = adopted
        .deployments
        .iter()
        .find(|(d, _)| d.name == "qwen3-tiny-bf16-kv-bf16")
        .expect("a bf16 deployment");
    read.as_ref()
        .expect("the bf16 deployment reads the checkpoint");
    let trace = bf16.try_trace(Platform::Cuda).expect("it traces");
    let o = trace
        .params
        .iter()
        .find(|p| p.name == "layers.2.o")
        .expect("a third layer");
    assert_eq!(o.shape, vec![64, 128], "four heads 32 wide, out to 64");
    bf16.try_trace(Platform::Metal)
        .expect("it traces for Metal too");
}

#[test]
fn a_llama_3_2_with_a_tied_head_is_adopted_and_read() {
    let config = serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "intermediate_size": 128,
        "vocab_size": 128256,
        "tie_word_embeddings": true,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-5,
        "rope_scaling": {"rope_type": "llama3", "factor": 32.0, "low_freq_factor": 1.0, "high_freq_factor": 4.0, "original_max_position_embeddings": 8192},
        "hidden_act": "silu",
    });
    let src = checkpoint(
        "llama",
        &trunk(Trunk {
            hidden: 64,
            layers: 2,
            q: 4,
            kv: 2,
            head_dim: 16,
            inter: 128,
            vocab: 128256,
            qk_norm: false,
            tied: true,
        }),
        &config,
    );
    let catalog = repository();
    let kin = catalog.kin(&src, Platform::Cuda);
    let llama = kin
        .iter()
        .find(|k| k.package.name() == "llama_3")
        .expect("Llama 3's formats recognize it");
    let adopted = catalog
        .adopt(
            llama,
            "adopted_llama_tiny",
            "llama-tiny",
            &reading(&src, &config),
            Platform::Cuda,
        )
        .expect("the family adopts it");
    let (bf16, read) = adopted
        .deployments
        .iter()
        .find(|(d, _)| d.name == "llama-tiny-bf16-kv-bf16")
        .expect("a bf16 deployment");
    read.as_ref()
        .expect("the tied checkpoint reads without an lm_head");
    bf16.try_trace(Platform::Cuda).expect("it traces");
}

#[test]
fn a_family_refuses_what_its_forward_does_not_run() {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 128256,
        "rope_scaling": {"rope_type": "mrope", "mrope_section": [4, 2, 2]},
    });
    let src = checkpoint(
        "mrope",
        &trunk(Trunk {
            hidden: 64,
            layers: 2,
            q: 4,
            kv: 2,
            head_dim: 16,
            inter: 128,
            vocab: 128256,
            qk_norm: false,
            tied: false,
        }),
        &config,
    );
    let catalog = repository();
    let kin = catalog.kin(&src, Platform::Cuda);
    let llama = kin
        .iter()
        .find(|k| k.package.name() == "llama_3")
        .expect("Llama 3's formats recognize it");
    let why = catalog
        .adopt(
            llama,
            "adopted_llama_mrope",
            "llama-mrope",
            &reading(&src, &config),
            Platform::Cuda,
        )
        .err()
        .expect("the family refuses a rope it does not turn")
        .to_string();
    assert!(why.contains("rope scaling of type `mrope`"), "{why}");
}

#[test]
fn a_llama_family_adopts_a_sliding_window_and_a_scaled_rope() {
    let config = serde_json::json!({
        "model_type": "mistral",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 128256,
        "max_position_embeddings": 64,
        "sliding_window": 16,
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 4.0,
            "original_max_position_embeddings": 16,
        },
    });
    let src = checkpoint(
        "windowed",
        &trunk(Trunk {
            hidden: 64,
            layers: 2,
            q: 4,
            kv: 2,
            head_dim: 16,
            inter: 128,
            vocab: 128256,
            qk_norm: false,
            tied: false,
        }),
        &config,
    );
    let catalog = repository();
    let kin = catalog.kin(&src, Platform::Cuda);
    let llama = kin
        .iter()
        .find(|k| k.package.name() == "llama_3")
        .expect("Llama 3's formats recognize it");
    let adopted = catalog
        .adopt(
            llama,
            "adopted_mistral_windowed",
            "mistral-windowed",
            &reading(&src, &config),
            Platform::Cuda,
        )
        .expect("the family adopts a window and a yarn rope it turns");
    assert!(
        adopted.adoption.block.contains("\"sliding_window\": 16"),
        "the adoption records the window it read"
    );
    assert!(
        adopted.adoption.block.contains("\"rope_theta\": 1000000.0"),
        "the theta transformers 5 states under rope_parameters is read"
    );
    let (deployment, read) = adopted
        .deployments
        .iter()
        .find(|(_, read)| read.is_ok())
        .expect("a deployment reads the checkpoint");
    assert!(read.is_ok());
    deployment
        .try_trace(Platform::Cuda)
        .expect("the windowed, yarn-scaled model traces");
}

#[test]
fn a_llama_3_0_is_not_read_as_a_llama_3_1() {
    let config = serde_json::json!({
        "model_type": "llama",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
    });
    let src = checkpoint(
        "llama30",
        &trunk(Trunk {
            hidden: 64,
            layers: 2,
            q: 4,
            kv: 2,
            head_dim: 16,
            inter: 128,
            vocab: 128256,
            qk_norm: false,
            tied: false,
        }),
        &config,
    );
    let catalog = repository();
    let kin = catalog.kin(&src, Platform::Cuda);
    let llama = kin
        .iter()
        .find(|k| k.package.name() == "llama_3")
        .expect("Llama 3's formats recognize it");
    assert!(
        llama
            .nearest
            .departing
            .iter()
            .any(|a| a.key == "rope_scaling.factor"),
        "a checkpoint stating no rope_scaling departs from a llama3-scaled row"
    );
    let adopted = catalog
        .adopt(
            llama,
            "adopted_llama_30",
            "llama-30",
            &reading(&src, &config),
            Platform::Cuda,
        )
        .expect("the family adopts it with its rope unscaled");
    assert!(
        adopted.adoption.block.contains("\"rope_theta\": 500000"),
        "{}",
        adopted.adoption.block
    );
}
