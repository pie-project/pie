//! End-to-end test of the algebra: declaration -> compiled gather -> value.
//!
//! `contract/compile.rs` checks the lowering against a per-coordinate oracle in
//! index space. This file checks the layer below it: that the byte offsets and
//! strides the gather carries, replayed by the reference evaluator, reproduce
//! the tensor the expression names — and that the type checker rejects a gather
//! that does not.

use std::collections::HashMap;

use pie_loader::contract::compile::compile;
use pie_loader::contract::{CheckpointTypes, Expr, TensorType, infer_type};
use pie_loader::ir::{GatherDim, GatherPiece, LayoutExpr, LayoutPlan};
use pie_loader::optimizer::optimize;
use pie_loader::reference::{TensorValue, evaluate};
use pie_loader::typecheck::typecheck;
use pie_loader::types::{DType, Encoding, ExprId, TensorDecl, TensorId, encoding_nbytes};

const MAX_RUNS: usize = 1 << 16;
const DTYPE: DType = DType::I32;

/// A checkpoint of small `i32` tensors whose values are their flat indices,
/// offset per tensor so that every element in the fixture is distinct.
struct Fixture {
    tensors: Vec<(String, Vec<i64>)>,
}

impl Fixture {
    fn new(tensors: &[(&str, &[i64])]) -> Self {
        Self {
            tensors: tensors
                .iter()
                .map(|(name, shape)| ((*name).to_string(), shape.to_vec()))
                .collect(),
        }
    }

    fn index(&self, name: &str) -> usize {
        self.tensors
            .iter()
            .position(|(candidate, _)| candidate == name)
            .unwrap_or_else(|| panic!("no tensor '{name}'"))
    }

    fn shape(&self, name: &str) -> &[i64] {
        &self.tensors[self.index(name)].1
    }

    /// Element `i` of the `n`th tensor is `1000 * n + i`.
    fn data(&self, name: &str) -> Vec<i64> {
        let at = self.index(name);
        let count: i64 = self.tensors[at].1.iter().product();
        (0..count).map(|i| 1000 * at as i64 + i).collect()
    }

    fn values(&self) -> HashMap<TensorId, TensorValue> {
        self.tensors
            .iter()
            .enumerate()
            .map(|(at, (name, shape))| {
                let decl = decl(at as u32, name, shape);
                (
                    TensorId(at as u32),
                    TensorValue::new(decl, self.data(name)).unwrap(),
                )
            })
            .collect()
    }
}

impl CheckpointTypes for Fixture {
    fn tensor_type(&self, name: &str) -> Option<TensorType> {
        self.tensors
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, shape)| TensorType::raw(shape.clone(), DTYPE))
    }
}

fn decl(id: u32, name: &str, shape: &[i64]) -> TensorDecl {
    TensorDecl {
        id: TensorId(id),
        name: name.to_string(),
        shape: shape.to_vec(),
        encoding: Encoding::Raw(DTYPE),
        alignment: 256,
    }
}

/// Build the one-gather plan the frontend builds for a single expression, and
/// run the reference evaluator over it.
fn realize(expr: &Expr, fixture: &Fixture) -> TensorValue {
    let (ty, checked) = infer_type(expr, fixture).unwrap();
    let lowering = compile(expr, &checked, MAX_RUNS).unwrap();
    assert!(
        !lowering.needs_zero_fill(),
        "this helper does not model padding"
    );

    let mut plan = LayoutPlan::new();
    let inputs = lowering
        .leaves
        .iter()
        .map(|leaf| {
            let name = match leaf {
                pie_loader::contract::compile::Leaf::Checkpoint(name) => name.clone(),
                pie_loader::contract::compile::Leaf::Contract(name) => {
                    panic!("this helper takes no contract references, got '{name}'")
                }
            };
            let at = fixture.index(&name);
            plan.push(LayoutExpr::Source {
                tensor: TensorId(at as u32),
                decl: decl(at as u32, &name, fixture.shape(&name)),
            })
        })
        .collect::<Vec<_>>();

    let out_decl = decl(900, "out", &ty.shape);
    let gathered = plan.push(LayoutExpr::Gather {
        inputs,
        pieces: lowering
            .byte_pieces(&ty.encoding)
            .unwrap()
            .into_iter()
            .map(|piece| GatherPiece {
                input: piece.leaf as u32,
                src_offset: piece.src_offset,
                dst_offset: piece.dst_offset,
                dims: piece
                    .dims
                    .into_iter()
                    .map(|dim| GatherDim {
                        count: dim.count,
                        src_stride: dim.src_stride,
                        dst_stride: dim.dst_stride,
                    })
                    .collect(),
            })
            .collect(),
        zero_fill: false,
        decl: out_decl.clone(),
    });
    let final_node = plan.push(LayoutExpr::Realize {
        input: gathered,
        runtime_name: "out".to_string(),
        decl: out_decl,
    });
    plan.outputs.push(final_node);

    typecheck(&plan).unwrap();
    let plan = optimize(plan).unwrap();
    evaluate(&plan, &fixture.values())
        .unwrap()
        .remove("out")
        .unwrap()
}

/// What the expression means, worked out by hand with no reference to the
/// lowering: resolve one output coordinate at a time.
fn oracle(expr: &Expr, fixture: &Fixture) -> Vec<i64> {
    let (ty, _) = infer_type(expr, fixture).unwrap();
    let count: i64 = ty.shape.iter().product();
    (0..count)
        .map(|flat| {
            let mut index = Vec::with_capacity(ty.shape.len());
            let mut rest = flat;
            for dim in ty.shape.iter().rev() {
                index.push(rest % dim);
                rest /= dim;
            }
            index.reverse();
            resolve(expr, &index, fixture).expect("the helper expressions have no holes")
        })
        .collect()
}

fn resolve(expr: &Expr, index: &[i64], fixture: &Fixture) -> Option<i64> {
    match expr {
        Expr::Src(name) => {
            let shape = fixture.shape(name);
            let mut flat = 0i64;
            for (at, coordinate) in index.iter().enumerate() {
                flat = flat * shape[at] + coordinate;
            }
            Some(fixture.data(name)[flat as usize])
        }
        Expr::Slice {
            src: input,
            axis,
            start,
            len,
            step,
        } => {
            let mut inner = index.to_vec();
            let at = usize::from(axis.0);
            assert!(inner[at] < *len);
            inner[at] = start + inner[at] * step;
            resolve(input, &inner, fixture)
        }
        Expr::Cat { axis, parts } => {
            let at = usize::from(axis.0);
            let mut offset = 0;
            for part in parts {
                let (ty, _) = infer_type(part, fixture).unwrap();
                let extent = ty.shape[at];
                if index[at] < offset + extent {
                    let mut inner = index.to_vec();
                    inner[at] -= offset;
                    return resolve(part, &inner, fixture);
                }
                offset += extent;
            }
            panic!("cat index out of range")
        }
        Expr::Reshape { src: input, shape } => {
            let mut flat = 0i64;
            for (at, coordinate) in index.iter().enumerate() {
                flat = flat * shape[at] + coordinate;
            }
            let (inner_ty, _) = infer_type(input, fixture).unwrap();
            let mut inner = vec![0; inner_ty.shape.len()];
            let mut rest = flat;
            for (at, dim) in inner_ty.shape.iter().enumerate().rev() {
                inner[at] = rest % dim;
                rest /= dim;
            }
            resolve(input, &inner, fixture)
        }
        other => panic!("the oracle does not model {other:?}"),
    }
}

fn check(expr: Expr, fixture: &Fixture) {
    let realized = realize(&expr, fixture);
    assert_eq!(realized.data, oracle(&expr, fixture), "for {expr:?}");
}

#[test]
fn a_whole_tensor_is_itself() {
    let fixture = Fixture::new(&[("w", &[4, 6])]);
    check(Expr::src("w"), &fixture);
}

#[test]
fn a_row_shard_reads_a_window() {
    let fixture = Fixture::new(&[("w", &[8, 6])]);
    check(Expr::src("w").slice(0, 2, 4), &fixture);
}

#[test]
fn a_column_shard_reads_a_stride() {
    let fixture = Fixture::new(&[("w", &[8, 6])]);
    check(Expr::src("w").slice(1, 2, 3), &fixture);
}

#[test]
fn a_strided_slice_picks_every_other_expert() {
    let fixture = Fixture::new(&[("e", &[8, 3])]);
    check(Expr::src("e").slice_step(0, 1, 4, 2), &fixture);
}

#[test]
fn fusing_three_projections_concatenates_them() {
    let fixture = Fixture::new(&[("q", &[4, 6]), ("k", &[2, 6]), ("v", &[2, 6])]);
    check(
        Expr::cat(0, vec![Expr::src("q"), Expr::src("k"), Expr::src("v")]),
        &fixture,
    );
}

#[test]
fn fusing_shards_interleaves_them() {
    // Each part is a column window, so every part contributes a strided read
    // and the fusion cannot collapse into one copy.
    let fixture = Fixture::new(&[("q", &[4, 8]), ("k", &[4, 8])]);
    check(
        Expr::cat(
            1,
            vec![Expr::src("q").slice(1, 0, 4), Expr::src("k").slice(1, 4, 4)],
        ),
        &fixture,
    );
}

#[test]
fn reshaping_does_not_move_anything() {
    let fixture = Fixture::new(&[("w", &[4, 6])]);
    check(Expr::src("w").reshape(vec![2, 2, 6]), &fixture);
    check(Expr::src("w").reshape(vec![24]), &fixture);
}

#[test]
fn slicing_a_fusion_reaches_through_it() {
    let fixture = Fixture::new(&[("a", &[8, 4]), ("b", &[8, 4])]);
    check(
        Expr::cat(0, vec![Expr::src("a").slice(0, 2, 4), Expr::src("b")]).slice(0, 2, 6),
        &fixture,
    );
}

#[test]
fn stacking_experts_is_a_reshaped_concatenation() {
    let fixture = Fixture::new(&[("e0", &[2, 3]), ("e1", &[2, 3]), ("e2", &[2, 3])]);
    check(
        Expr::cat(
            0,
            vec![
                Expr::src("e0").reshape(vec![1, 2, 3]),
                Expr::src("e1").reshape(vec![1, 2, 3]),
                Expr::src("e2").reshape(vec![1, 2, 3]),
            ],
        ),
        &fixture,
    );
}

#[test]
fn typecheck_rejects_a_gather_that_leaves_a_hole() {
    let mut plan = LayoutPlan::new();
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: decl(0, "w", &[4, 4]),
    });
    let out = decl(1, "out", &[4, 4]);
    let final_node = plan.push(LayoutExpr::Gather {
        inputs: vec![source],
        pieces: vec![GatherPiece::span(0, 0, 0, 32)],
        zero_fill: false,
        decl: out,
    });
    plan.outputs.push(final_node);

    let err = typecheck(&plan).unwrap_err().to_string();
    assert!(err.contains("covers 32 of 64 bytes"), "{err}");
}

#[test]
fn typecheck_rejects_a_gather_that_reads_past_its_input() {
    let mut plan = LayoutPlan::new();
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: decl(0, "w", &[4, 4]),
    });
    let out = decl(1, "out", &[4, 4]);
    let final_node = plan.push(LayoutExpr::Gather {
        inputs: vec![source],
        pieces: vec![GatherPiece::span(0, 8, 0, 64)],
        zero_fill: false,
        decl: out,
    });
    plan.outputs.push(final_node);

    let err = typecheck(&plan).unwrap_err().to_string();
    assert!(err.contains("reads past the end"), "{err}");
}

#[test]
fn typecheck_rejects_a_gather_that_names_a_missing_input() {
    let mut plan = LayoutPlan::new();
    let source = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: decl(0, "w", &[4, 4]),
    });
    let out = decl(1, "out", &[4, 4]);
    let final_node = plan.push(LayoutExpr::Gather {
        inputs: vec![source],
        pieces: vec![GatherPiece::span(1, 0, 0, 64)],
        zero_fill: false,
        decl: out,
    });
    plan.outputs.push(final_node);

    let err = typecheck(&plan).unwrap_err().to_string();
    assert!(err.contains("names input 1"), "{err}");
}

#[test]
fn the_optimizer_drops_what_no_output_reaches() {
    let mut plan = LayoutPlan::new();
    let kept = plan.push(LayoutExpr::Source {
        tensor: TensorId(0),
        decl: decl(0, "kept", &[2, 2]),
    });
    plan.push(LayoutExpr::Source {
        tensor: TensorId(1),
        decl: decl(1, "dropped", &[2, 2]),
    });
    let out = decl(2, "out", &[2, 2]);
    let bytes = encoding_nbytes(&out.shape, &out.encoding).unwrap();
    let gathered = plan.push(LayoutExpr::Gather {
        inputs: vec![kept],
        pieces: vec![GatherPiece::span(0, 0, 0, bytes)],
        zero_fill: false,
        decl: out.clone(),
    });
    let final_node = plan.push(LayoutExpr::Realize {
        input: gathered,
        runtime_name: "out".to_string(),
        decl: out,
    });
    plan.outputs.push(final_node);

    let optimized = optimize(plan).unwrap();
    assert_eq!(optimized.exprs.len(), 3);
    assert_eq!(optimized.outputs, vec![ExprId(2)]);
}
