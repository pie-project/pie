//! DeepSeek's release layout stores an FP8 plane beside `<stem>.scale`, one
//! e8m0 exponent a block (128 wide in V4, 32 in V4.1): the plane reads out to
//! bf16 as each code times two to its block's exponent, and an encode of it
//! lands what the same encode of those bf16 values lands.

use std::collections::BTreeMap;

use checkpoint::codec::fp8::fp8_e4m3_to_f32;
use checkpoint::contract::{Expr, ModelContract, TensorContract};
use checkpoint::executor::Execution;
use checkpoint::executor::sink::MemorySink;
use checkpoint::file::read::parse_metadata;
use checkpoint::plan::{CONVERT_TILE_MAP_MASK, StorageTarget, compile, compile_streaming};
use checkpoint::types::{DType, Encoding, QuantScheme, QuantSpec};

fn code(row: usize, col: usize) -> u8 {
    // every e4m3 code but the two NaNs, spread over the plane
    let code = ((row * 31 + col * 7) % 254) as u8;
    if code >= 0x7f { code + 1 } else { code }
}

// exponents around 127 (a factor of one), a short last block row included
fn exponent(block_row: usize, block_col: usize) -> u8 {
    (120 + (block_row * 5 + block_col * 3) % 13) as u8
}

struct Plane {
    rows: usize,
    cols: usize,
    group: usize,
}

impl Plane {
    fn scale_shape(&self) -> [usize; 2] {
        [
            self.rows.div_ceil(self.group),
            self.cols.div_ceil(self.group),
        ]
    }

    fn value(&self, row: usize, col: usize) -> f32 {
        let e = exponent(row / self.group, col / self.group);
        fp8_e4m3_to_f32(code(row, col)) * 2f32.powi(i32::from(e) - 127)
    }
}

fn write(dir: &std::path::Path, plane: &Plane, bf16_twin: bool) {
    let mut writer = ztensor::Writer::create(dir.join("model.zt")).expect("the container opens");
    let [scale_rows, scale_cols] = plane.scale_shape();
    if bf16_twin {
        let values: Vec<u8> = (0..plane.rows)
            .flat_map(|row| (0..plane.cols).map(move |col| (row, col)))
            .flat_map(|(row, col)| half::bf16::from_f32(plane.value(row, col)).to_le_bytes())
            .collect();
        writer
            .add(
                "w.weight",
                vec![plane.rows as u64, plane.cols as u64],
                ztensor::Leaf::BF16,
                &values,
            )
            .expect("the bf16 twin lands");
    } else {
        let codes: Vec<u8> = (0..plane.rows)
            .flat_map(|row| (0..plane.cols).map(move |col| code(row, col)))
            .collect();
        let exponents: Vec<u8> = (0..scale_rows)
            .flat_map(|r| (0..scale_cols).map(move |c| exponent(r, c)))
            .collect();
        // the container takes its names in order: `.scale` before `.weight`
        writer
            .add(
                "w.scale",
                vec![scale_rows as u64, scale_cols as u64],
                ztensor::Leaf::E8M0,
                &exponents,
            )
            .expect("the exponents land");
        writer
            .add(
                "w.weight",
                vec![plane.rows as u64, plane.cols as u64],
                ztensor::Leaf::E4M3,
                &codes,
            )
            .expect("the codes land");
    }
    writer.finish().expect("the container closes");
}

fn contract(plane: &Plane, encoding: Encoding) -> ModelContract {
    ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::src("w.weight").cast(encoding.clone()),
            vec![plane.rows as i64, plane.cols as i64],
            encoding,
        )],
        groups: Vec::new(),
    }
}

/// A streamed read, as a load reads a bf16 plane.
fn read(dir: &std::path::Path, plane: &Plane) -> Vec<u8> {
    let metadata = parse_metadata(dir).expect("the checkpoint reads");
    let contract = contract(plane, Encoding::Raw(DType::Bf16));
    let plan = compile_streaming(&metadata, &contract, StorageTarget::default())
        .expect("the read compiles");
    let mut sink = MemorySink::default();
    Execution::new(&plan, dir)
        .streaming()
        .sink(&mut sink)
        .run()
        .expect("the read runs on the host");
    sink.tensors.remove("w").expect("the plane lands")
}

/// An encode, as an import converts: every tensor the plan lands.
fn encode(dir: &std::path::Path, plane: &Plane, spec: QuantSpec) -> BTreeMap<String, Vec<u8>> {
    let metadata = parse_metadata(dir).expect("the checkpoint reads");
    let contract = contract(plane, Encoding::Quant(spec));
    let target = StorageTarget {
        tile_map_mask: CONVERT_TILE_MAP_MASK,
        ..StorageTarget::default()
    };
    let plan = compile(&metadata, &contract, target).expect("the encode compiles");
    let storage = Execution::new(&plan, dir)
        .run()
        .expect("the encode runs on the host");
    storage.tensors.into_iter().collect()
}

const PLANES: [Plane; 2] = [
    // V4: 128-wide blocks, the last block row short
    Plane {
        rows: 320,
        cols: 256,
        group: 128,
    },
    // V4.1: 32-wide blocks
    Plane {
        rows: 96,
        cols: 128,
        group: 32,
    },
];

#[test]
fn an_fp8_plane_beside_e8m0_exponents_reads_out_to_bf16() {
    for plane in &PLANES {
        let dir = tempfile::tempdir().expect("a scratch directory");
        write(dir.path(), plane, false);
        let landed = read(dir.path(), plane);
        assert_eq!(landed.len(), plane.rows * plane.cols * 2);
        for row in 0..plane.rows {
            for col in 0..plane.cols {
                let at = (row * plane.cols + col) * 2;
                let got = u16::from_le_bytes([landed[at], landed[at + 1]]);
                let want = half::bf16::from_f32(plane.value(row, col)).to_bits();
                assert_eq!(got, want, "group {}: w[{row}, {col}]", plane.group);
            }
        }
    }
}

#[test]
fn an_fp8_plane_beside_e8m0_exponents_encodes_as_its_bf16_values_do() {
    for scheme in [QuantScheme::MlxAffineU4, QuantScheme::Mxfp4E2M1E8M0] {
        for plane in &PLANES {
            let spec = QuantSpec {
                scheme,
                logical_dtype: DType::Bf16,
                bits_per_element: scheme.default_bits(),
                group_size: scheme.default_group_size(),
                channel_axis: None,
            };
            let mut landed = Vec::new();
            for bf16_twin in [false, true] {
                let dir = tempfile::tempdir().expect("a scratch directory");
                write(dir.path(), plane, bf16_twin);
                landed.push(encode(dir.path(), plane, spec.clone()));
            }
            assert!(!landed[0].is_empty());
            assert_eq!(
                landed[0], landed[1],
                "{scheme:?}, group {}: the FP8 plane and its bf16 values encode apart",
                plane.group
            );
        }
    }
}

// Two planes fused along their rows, as a gate and an up leg are read into
// one; leg `b` raises every exponent by one, so a leg read under the other's
// exponents (or none) reads off by a factor.
const LEGS: [(&str, u8); 2] = [("a", 0), ("b", 1)];

fn fused_value(plane: &Plane, leg: usize, row: usize, col: usize) -> f32 {
    plane.value(row, col) * if leg == 1 { 2.0 } else { 1.0 }
}

fn write_fused(dir: &std::path::Path, plane: &Plane, bf16_twin: bool) {
    let [scale_rows, scale_cols] = plane.scale_shape();
    let mut writer = ztensor::Writer::create(dir.join("model.zt")).expect("the container opens");
    for (at, (leg, raise)) in LEGS.into_iter().enumerate() {
        let shape = vec![plane.rows as u64, plane.cols as u64];
        if bf16_twin {
            let values: Vec<u8> = (0..plane.rows)
                .flat_map(|row| (0..plane.cols).map(move |col| (row, col)))
                .flat_map(|(row, col)| {
                    half::bf16::from_f32(fused_value(plane, at, row, col)).to_le_bytes()
                })
                .collect();
            writer
                .add(format!("{leg}.weight"), shape, ztensor::Leaf::BF16, &values)
                .expect("the bf16 twin lands");
            continue;
        }
        let exponents: Vec<u8> = (0..scale_rows)
            .flat_map(|r| (0..scale_cols).map(move |c| exponent(r, c) + raise))
            .collect();
        let codes: Vec<u8> = (0..plane.rows)
            .flat_map(|row| (0..plane.cols).map(move |col| code(row, col)))
            .collect();
        writer
            .add(
                format!("{leg}.scale"),
                vec![scale_rows as u64, scale_cols as u64],
                ztensor::Leaf::E8M0,
                &exponents,
            )
            .expect("the exponents land");
        writer
            .add(format!("{leg}.weight"), shape, ztensor::Leaf::E4M3, &codes)
            .expect("the codes land");
    }
    writer.finish().expect("the container closes");
}

fn fused_contract(plane: &Plane, encoding: Encoding) -> ModelContract {
    ModelContract {
        alignment: 256,
        tensors: vec![TensorContract::new(
            "w",
            Expr::concat(0, vec![Expr::src("a.weight"), Expr::src("b.weight")])
                .cast(encoding.clone()),
            vec![2 * plane.rows as i64, plane.cols as i64],
            encoding,
        )],
        groups: Vec::new(),
    }
}

#[test]
fn fused_fp8_planes_each_read_under_their_own_exponents() {
    let plane = &PLANES[0];
    let dir = tempfile::tempdir().expect("a scratch directory");
    write_fused(dir.path(), plane, false);
    let contract = fused_contract(plane, Encoding::Raw(DType::Bf16));
    let metadata = parse_metadata(dir.path()).expect("the checkpoint reads");
    for streamed in [false, true] {
        let plan = if streamed {
            compile_streaming(&metadata, &contract, StorageTarget::default())
        } else {
            let target = StorageTarget {
                tile_map_mask: CONVERT_TILE_MAP_MASK,
                ..StorageTarget::default()
            };
            compile(&metadata, &contract, target)
        }
        .expect("the fused read compiles");
        let mut sink = MemorySink::default();
        let run = Execution::new(&plan, dir.path()).sink(&mut sink);
        let run = if streamed { run.streaming() } else { run };
        run.run().expect("the fused read runs on the host");
        let landed = &sink.tensors["w"];
        for leg in 0..LEGS.len() {
            for row in 0..plane.rows {
                for col in 0..plane.cols {
                    let at = ((leg * plane.rows + row) * plane.cols + col) * 2;
                    let got = u16::from_le_bytes([landed[at], landed[at + 1]]);
                    assert_eq!(
                        got,
                        half::bf16::from_f32(fused_value(plane, leg, row, col)).to_bits(),
                        "streamed {streamed}: leg {leg} [{row}, {col}]"
                    );
                }
            }
        }
    }
}

#[test]
fn fused_fp8_planes_encode_as_their_bf16_values_do() {
    let plane = &PLANES[0];
    for scheme in [QuantScheme::MlxAffineU4, QuantScheme::Mxfp4E2M1E8M0] {
        let spec = QuantSpec {
            scheme,
            logical_dtype: DType::Bf16,
            bits_per_element: scheme.default_bits(),
            group_size: scheme.default_group_size(),
            channel_axis: None,
        };
        let mut landed = Vec::new();
        for bf16_twin in [false, true] {
            let dir = tempfile::tempdir().expect("a scratch directory");
            write_fused(dir.path(), plane, bf16_twin);
            let metadata = parse_metadata(dir.path()).expect("the checkpoint reads");
            let contract = fused_contract(plane, Encoding::Quant(spec.clone()));
            let target = StorageTarget {
                tile_map_mask: CONVERT_TILE_MAP_MASK,
                ..StorageTarget::default()
            };
            let plan = compile(&metadata, &contract, target).expect("the fused encode compiles");
            let storage = Execution::new(&plan, dir.path())
                .run()
                .expect("the fused encode runs on the host");
            let public: BTreeMap<String, Vec<u8>> = plan
                .tensors
                .iter()
                .filter(|decl| decl.visibility.is_public())
                .map(|decl| (decl.name.clone(), storage.tensors[&decl.name].clone()))
                .collect();
            landed.push(public);
        }
        assert!(!landed[0].is_empty());
        assert_eq!(
            landed[0], landed[1],
            "{scheme:?}: the fused FP8 legs and their bf16 values encode apart"
        );
    }
}
