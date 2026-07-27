//! End-to-end test of the algebra: declaration -> compiled gather -> value.
//!
//! `contract/compile.rs` checks the lowering against a per-coordinate oracle in
//! index space. This file checks the layer below it: that the byte offsets and
//! strides the lowering carries, replayed literally, reproduce the tensor the
//! expression names — and that a lowering which does not is refused.
//!
//! The replay used to go through the middle IR, which meant the frontend that
//! built that IR was inside the thing under test. It is gone, and so is the
//! detour: `reference::replay` reads the lowering directly.

use std::collections::HashMap;

use pie_loader::contract::compile::{Leaf, Lowering, Run, RunSource, compile};
use pie_loader::contract::infer::{CheckpointTypes, infer_type};
use pie_loader::contract::{Expr, TensorType};
use pie_loader::reference::{TensorValue, replay};
use pie_loader::types::{DType, Encoding, TensorDecl, TensorId};

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

    fn values(&self) -> HashMap<String, TensorValue> {
        self.tensors
            .iter()
            .enumerate()
            .map(|(at, (name, shape))| {
                let decl = decl(at as u32, name, shape);
                (
                    name.clone(),
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

/// Compile one expression and replay the bytes its lowering claims.
fn realize(expr: &Expr, fixture: &Fixture) -> Vec<i64> {
    let (ty, checked) = infer_type(expr, fixture).unwrap();
    let lowering = compile(expr, &checked, MAX_RUNS).unwrap();
    replay(&lowering, &ty, &fixture.values()).unwrap()
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
            resolve(expr, &index, fixture).unwrap_or(0)
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
        Expr::Pad {
            src: input,
            axis,
            before,
            after: _,
        } => {
            let at = usize::from(axis.0);
            let (inner_ty, _) = infer_type(input, fixture).unwrap();
            let mut inner = index.to_vec();
            inner[at] -= before;
            if inner[at] < 0 || inner[at] >= inner_ty.shape[at] {
                return None;
            }
            resolve(input, &inner, fixture)
        }
        other => panic!("the oracle does not model {other:?}"),
    }
}

fn check(expr: Expr, fixture: &Fixture) {
    let realized = realize(&expr, fixture);
    assert_eq!(realized, oracle(&expr, fixture), "for {expr:?}");
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

/// A lowering built by hand, so the replay can be shown what a *wrong* one
/// looks like. `compile` cannot produce these; the point is that if it ever
/// did, the replay would say so rather than quietly returning a tensor with
/// garbage in it.
fn hand_rolled(runs: Vec<Run>, elements: i64) -> Lowering {
    Lowering {
        leaves: vec![Leaf::Checkpoint("w".to_string())],
        runs,
        elements,
    }
}

fn out_type(shape: &[i64]) -> TensorType {
    TensorType::raw(shape.to_vec(), DTYPE)
}

#[test]
fn the_replay_rejects_a_lowering_that_leaves_a_hole() {
    let fixture = Fixture::new(&[("w", &[4, 4])]);
    let lowering = hand_rolled(
        vec![Run {
            source: RunSource::Leaf {
                leaf: 0,
                src_elem: 0,
            },
            dst_elem: 0,
            len: 8,
        }],
        16,
    );
    let err = replay(&lowering, &out_type(&[4, 4]), &fixture.values())
        .unwrap_err()
        .to_string();
    assert!(err.contains("uninitialized"), "{err}");
}

#[test]
fn the_replay_rejects_a_lowering_that_reads_past_its_input() {
    let fixture = Fixture::new(&[("w", &[4, 4])]);
    let lowering = hand_rolled(
        vec![Run {
            source: RunSource::Leaf {
                leaf: 0,
                src_elem: 2,
            },
            dst_elem: 0,
            len: 16,
        }],
        16,
    );
    let err = replay(&lowering, &out_type(&[4, 4]), &fixture.values())
        .unwrap_err()
        .to_string();
    assert!(err.contains("reads past the end"), "{err}");
}

#[test]
fn the_replay_rejects_a_lowering_that_names_a_leaf_nobody_supplied() {
    let fixture = Fixture::new(&[("w", &[4, 4])]);
    let mut lowering = hand_rolled(
        vec![Run {
            source: RunSource::Leaf {
                leaf: 0,
                src_elem: 0,
            },
            dst_elem: 0,
            len: 16,
        }],
        16,
    );
    lowering.leaves = vec![Leaf::Checkpoint("absent".to_string())];
    let err = replay(&lowering, &out_type(&[4, 4]), &fixture.values())
        .unwrap_err()
        .to_string();
    assert!(err.contains("no reference value for 'absent'"), "{err}");
}

#[test]
fn the_replay_rejects_a_lowering_that_writes_one_element_twice() {
    // Dead-code elimination used to be a pass, and the test here used to be
    // that it dropped what no output reached. There is nothing to drop now:
    // the builder walks outward from the declared tensors, so an unreachable
    // node is one it never constructs. What is still worth pinning is the
    // other half of "covers the destination exactly once".
    let fixture = Fixture::new(&[("w", &[4, 4])]);
    let run = |dst_elem| Run {
        source: RunSource::Leaf {
            leaf: 0,
            src_elem: 0,
        },
        dst_elem,
        len: 16,
    };
    let lowering = hand_rolled(vec![run(0), run(0)], 16);
    let err = replay(&lowering, &out_type(&[4, 4]), &fixture.values())
        .unwrap_err()
        .to_string();
    assert!(err.contains("twice"), "{err}");
}

#[test]
fn padding_a_head_dim_leaves_zeros_and_folds_the_copies() {
    // spec.md §2 case 9: a model whose head_dim is not one the kernel takes is
    // padded up. This is the one worked example in spec.md §3.3's cost table —
    // a `[4, 4]` padded by one column, priced at 5 — so it is the case that
    // says whether the code and the paper agree.
    let fixture = Fixture::new(&[("q", &[4, 4])]);
    let expr = Expr::src("q").pad(1, 0, 1);
    check(expr.clone(), &fixture);

    let (ty, checked) = infer_type(&expr, &fixture).unwrap();
    assert_eq!(ty.shape, vec![4, 5]);
    let lowering = compile(&expr, &checked, MAX_RUNS).unwrap();
    assert!(lowering.needs_zero_fill());
    // Eight alternating data and zero runs become four data runs and one fill.
    // They do not fold further: the destination stride is 5 and the row is 4,
    // and `fold` will not skip in the destination.
    assert_eq!(lowering.copy_pieces().len(), 4);
    assert_eq!(lowering.cost(), 5, "spec.md §3.3");
}

#[test]
fn padding_the_front_shifts_the_data_instead_of_the_zeros() {
    let fixture = Fixture::new(&[("q", &[2, 3])]);
    check(Expr::src("q").pad(1, 2, 0), &fixture);
    check(Expr::src("q").pad(0, 1, 1), &fixture);
}
