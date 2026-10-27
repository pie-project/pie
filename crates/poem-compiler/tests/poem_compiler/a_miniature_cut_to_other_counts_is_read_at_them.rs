use poem::Platform;
use ztensor::format::cbor::Value;

/// A checkpoint of one stranger whose configuration states `config`.
fn stating(name: &str, config: Vec<(&str, Value)>) -> ztensor::Source {
    let dir = std::env::temp_dir().join(format!("miniature_cut_{}_{name}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    let path = dir.join("model.zt");
    let mut writer = ztensor::Writer::create(&path).expect("the container opens");
    writer
        .add("a.stranger", vec![1u64], ztensor::Leaf::U8, &[0u8])
        .expect("the plane lands");
    writer.finish().expect("the container closes");
    let config = Value::Map(
        config
            .into_iter()
            .map(|(key, value)| (Value::Text(key.into()), value))
            .collect(),
    );
    ztensor::Source::open(&path)
        .expect("it reads back")
        .with_attribute(checkpoint::file::snapshot::CONFIG, config)
}

#[test]
fn a_miniature_cut_to_other_counts_is_read_at_them() {
    let listed = poem_compiler::catalog::deployment("glm53-mini-bf16-kv-bf16").expect("listed");

    let src = stating(
        "deeper",
        vec![
            ("num_hidden_layers", Value::Uint(8)),
            ("n_routed_experts", Value::Uint(16)),
        ],
    );
    let cut = listed
        .cut(&src, Platform::Cuda)
        .expect("a cut of other counts");
    assert_eq!(cut.name, "glm53-mini-bf16-kv-bf16-l8-e16");
    let parsed = poem_compiler::catalog::repository()
        .parse(&cut.name)
        .expect("the cut's name parses");
    assert_eq!(parsed.deploy, cut.deploy);
    let trace = cut.try_trace(Platform::Cuda).expect("the cut traces");
    let router = trace
        .params
        .iter()
        .find(|p| p.name == "layers.7.mlp.router")
        .expect("an eighth layer, routed");
    assert_eq!(router.shape, vec![16, 6144]);

    let src = stating(
        "listed",
        vec![
            ("num_hidden_layers", Value::Uint(7)),
            ("n_routed_experts", Value::Uint(8)),
        ],
    );
    assert!(
        listed.cut(&src, Platform::Cuda).is_none(),
        "a checkpoint at the listed counts is read as listed"
    );
}
