use std::path::{Path, PathBuf};

use checkpoint::executor::Execution;
use checkpoint::executor::sink::MemorySink;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{StorageTarget, compile_streaming};
use model_dsl::Platform;
use models::z_image::model::Dims;

const MINI: &str = "z-image-mini-bf16-kv-bf16";

fn fixture() -> Option<PathBuf> {
    let root = std::env::var_os("PIE_IMAGEGEN_GOLDEN")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/root/.cache/pie-imagegen/golden"));
    let file = root.join("z-image/zimage_mini.safetensors");
    file.is_file().then_some(file)
}

fn stage(fixture: &Path) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("z_image_mini_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|why| panic!("{}: {why}", dir.display()));
    std::fs::copy(fixture, dir.join("model.safetensors"))
        .unwrap_or_else(|why| panic!("{}: {why}", fixture.display()));
    dir
}

fn bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|quad| f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]))
        .collect()
}

fn round_bf16(x: f32) -> f32 {
    let bits = x.to_bits();
    let lsb = (bits >> 16) & 1;
    let rounded = bits.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000;
    f32::from_bits(rounded)
}

#[test]
fn the_miniature_lands_its_derived_planes() {
    let Some(fixture) = fixture() else {
        eprintln!("skipping: no zimage_mini.safetensors under $PIE_IMAGEGEN_GOLDEN/z-image");
        return;
    };
    let dir = stage(&fixture);
    let src = ztensor_compat::open(dir.join("model.safetensors")).unwrap();
    let metadata = parse_metadata(&dir).unwrap();
    let row = models::sku(MINI).expect("the catalog ships the miniature");
    let contract = row
        .contract(&src, Platform::Cuda)
        .unwrap_or_else(|why| panic!("the miniature does not read its fixture: {why}"));
    let plan = compile_streaming(&metadata, &contract, StorageTarget::default())
        .unwrap_or_else(|why| panic!("the miniature's contract does not compile: {why}"));
    let mut sink = MemorySink::default();
    Execution::new(&plan, &dir)
        .streaming()
        .sink(&mut sink)
        .run()
        .unwrap_or_else(|why| panic!("the load plan does not execute: {why}"));

    let d = Dims::mini();
    let dim = d.dim as usize;
    let stored = |name: &str| src.tensor(name).unwrap().bytes().unwrap().into_owned();
    let landed = |name: &str| -> &Vec<u8> {
        sink.tensors
            .get(name)
            .unwrap_or_else(|| panic!("`{name}` was not published"))
    };

    for (bank, token) in [
        ("dit.x_pad_mod", "x_pad_token"),
        ("dit.cap_pad_mod", "cap_pad_token"),
    ] {
        let got = bf16(landed(bank));
        assert_eq!(got.len(), 2 * dim, "{bank} is [2·dim, 1]");
        let (scale, shift) = got.split_at(dim);
        assert!(
            scale.iter().all(|&v| v == -1.0),
            "{bank}: a pad row scales by −1"
        );
        let want: Vec<f32> = f32s(&stored(token)).into_iter().map(round_bf16).collect();
        assert_eq!(
            shift,
            want.as_slice(),
            "{bank}: a pad row shifts by the learned token"
        );
    }

    assert_eq!(f32s(landed("dit.t_flip")), vec![1000.0]);

    let qkv = bf16(landed("dit.layer.0.qkv"));
    let mut want = Vec::new();
    for proj in ["to_q", "to_k", "to_v"] {
        want.extend(
            f32s(&stored(&format!("layers.0.attention.{proj}.weight")))
                .into_iter()
                .map(round_bf16),
        );
    }
    assert_eq!(qkv.len(), 3 * dim * dim);
    assert_eq!(
        qkv, want,
        "the packed bank is the three projections in order"
    );

    for name in ["dit.layer.1.out", "dit.noise.0.gate_up", "dit.x_embed"] {
        let elements = match name {
            "dit.layer.1.out" => dim * dim,
            "dit.noise.0.gate_up" => 2 * d.inter as usize * dim,
            _ => dim * 64,
        };
        assert_eq!(landed(name).len(), 2 * elements, "{name} landed bf16");
    }

    let _ = std::fs::remove_dir_all(&dir);
}
