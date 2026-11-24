//! Every dense, quantized and routed matmul form launches the workgroups its
//! output takes, once: a sink records each fire's `groups`, and each axis's
//! groups times what a group of that kernel lands covers the axis with no
//! more than one group's slack (`Encode`'s contract: `groups` counts
//! workgroups, never tiles times threads).

use std::cell::RefCell;

use dtype::Dtype;

use crate::encode::{ABSENT, ArgValue, Encode, Fire, Target, Tier};
use crate::error::Error;
use crate::linear::moe::select::{self, Batch, RoutedScratch};
use crate::linear::qmm::{self, Qmm};
use crate::linear::{gemm, kquant};
use crate::tensor::{Bank, Tensor};

/// A sink that fires nothing and keeps every fire.
struct Recorder {
    target: Target,
    fires: RefCell<Vec<Fire>>,
}

impl Recorder {
    fn new(target: Target) -> Self {
        Self {
            target,
            fires: RefCell::new(Vec::new()),
        }
    }

    fn take(&self) -> Vec<Fire> {
        std::mem::take(&mut *self.fires.borrow_mut())
    }
}

impl Encode for Recorder {
    fn fire(&self, fire: Fire, _args: &[ArgValue]) -> Result<(), Error> {
        self.fires.borrow_mut().push(fire);
        Ok(())
    }

    fn absent(&self) -> Result<ArgValue, Error> {
        Ok(ArgValue::Buffer(ABSENT))
    }

    fn target(&self) -> Target {
        self.target
    }
}

/// The extents a fire lands, by the names its shaders take.
#[derive(Clone, Copy, Debug)]
struct Shape {
    /// Result rows (routes, for a routed fire), columns, contraction.
    m: u32,
    n: u32,
    k: u32,
    /// A routed fire's tokens, routes a token, and sorted rows.
    tokens: u32,
    top_k: u32,
    padded: u32,
}

impl Shape {
    const fn dense(m: u32, n: u32, k: u32) -> Self {
        Self {
            m,
            n,
            k,
            tokens: 0,
            top_k: 0,
            padded: 0,
        }
    }
}

/// `_{key}_<digits>` in `name`.
fn stamp(name: &str, key: &str) -> Option<u32> {
    let at = name.find(&format!("_{key}_"))? + key.len() + 2;
    let digits: String = name[at..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().ok()
}

/// Per axis, the extent `entry` covers along it and what one workgroup
/// lands of it, read off its shader (`kernels/linear/`).
fn axes(entry: &str, target: Target, s: Shape) -> [(u32, u32); 3] {
    let words = |width: u32, rows: u32| (width * rows).div_ceil(2);
    let one = (1, 1);
    match entry {
        // A workgroup a tile, from `SV_GroupID`: columns on x, rows on y.
        "linear_gemm_gemv_v_bf16" => [(s.n, 8), (s.m, 4), one],
        "linear_gemm_gemv_v1_bf16" => [(s.n, 8), (s.m, 1), one],
        "linear_gemm_gemv_v8_bf16" => [(s.n, 8), (s.m, 8), one],
        "linear_gemm_gemv_v16_bf16" => [(s.n, 8), (s.m, 16), one],
        "linear_gemm_tiled_bm_64_bf16" => [(s.n, 64), (s.m, 64), one],
        // `gemm_register.slang`: `PIE_BN` columns, 32 lanes of eight rows.
        "linear_gemm_register_bf16" => [(s.n, 64), (s.m, 256), one],
        "linear_gemm_register_bn_32_bf16" => [(s.n, 32), (s.m, 256), one],
        "linear_gemm_tiled_bf16" => [(s.n, 32), (s.m, 32), one],
        "linear_gemm_pairs_tiled_bm_16_bf16" => [(s.n, 64), (s.m, 16), one],
        "linear_gemm_pairs_tiled_bf16" => [(s.n, 64), (s.m, 64), one],
        // An invocation a column (y) and row (z), 32 lanes folding k.
        "linear_gemm_gemv_bf16" => [one, (s.n, 8), (s.m, 1)],
        "linear_gemm_pairs_gemv_bf16" => [one, (s.n, 8), (s.m, 4)],
        "linear_gemm_pairs_gemv_v1_bf16" => [one, (s.n, 8), (s.m, 1)],
        "linear_gemm_pairs_gemv_v8_bf16" => [one, (s.n, 8), (s.m, 8)],
        "linear_gemm_pairs_gemv_v16_bf16" => [one, (s.n, 8), (s.m, 16)],
        "linear_moe_route_sort" | "linear_moe_route_sort_256" => [one, one, one],
        "linear_moe_route_sort_gather" => match target {
            Target::Spirv => [(s.k, 256), (s.padded, 1), one],
            Target::Wgsl => [(words(s.k, s.padded), 256), one, one],
        },
        "linear_moe_route_sort_scatter" => match target {
            Target::Spirv => [(s.n, 256), (s.m, 1), one],
            Target::Wgsl => [(words(s.n, s.m), 256), one, one],
        },
        // An invocation an output.
        e if e.starts_with("linear_gemm_lane_") => [(s.n, 64), (s.m, 4), one],
        e if e.starts_with("linear_qmm_register") => {
            [(s.n, 64), (s.m, stamp(e, "bm").unwrap()), one]
        }
        e if e.starts_with("linear_qmm_tiled_") || e.starts_with("linear_qmm_quad_") => [
            (s.n, stamp(e, "bn").unwrap()),
            (s.m, stamp(e, "bm").unwrap()),
            one,
        ],
        e if e.starts_with("linear_qmv_row_") => {
            // `_l<lanes>_r<rows>`, or 32 lanes of one column unstamped.
            let (lanes, rows) = e
                .rsplit_once("_l")
                .and_then(|(_, tail)| tail.split_once("_r"))
                .map_or((32, 1), |(l, r)| {
                    (l.parse::<u32>().unwrap(), r.parse::<u32>().unwrap())
                });
            [(s.n, 256 / lanes * rows), one, one]
        }
        e if e.starts_with("linear_qmv_bf16") => {
            let vecs = e.rsplit_once("_v").map_or(4, |(_, v)| v.parse().unwrap());
            [(s.n, 8), (s.m, vecs), one]
        }
        e if e.starts_with("linear_kquant") => {
            let slices = if e.starts_with("linear_kquant_s4") {
                4
            } else {
                2
            };
            [(s.n, 2 * slices), (s.m, 1), one]
        }
        e if e.starts_with("linear_moe_select_tiled") => [
            (s.n, stamp(e, "bn").unwrap()),
            (s.padded, stamp(e, "bm").unwrap()),
            one,
        ],
        // A token (x), eight output rows (y), a route (z).
        e if e.starts_with("linear_moe_select_") => [(s.tokens, 1), (s.n, 8), (s.top_k, 1)],
        e => panic!("no coverage is stated for `{e}`"),
    }
}

/// Every fire `rec` kept launches the groups `shape` takes, once.
fn assert_covers(rec: &Recorder, what: &str, shape: Shape) -> Vec<&'static str> {
    let fires = rec.take();
    assert!(!fires.is_empty(), "{what}: nothing fired");
    fires
        .into_iter()
        .map(|fire| {
            for (axis, (extent, per)) in axes(fire.entrypoint, rec.target, shape)
                .into_iter()
                .enumerate()
            {
                let groups = u64::from(fire.groups[axis]);
                let (extent, per) = (u64::from(extent.max(1)), u64::from(per));
                assert!(
                    groups * per >= extent && (groups - 1) * per < extent,
                    "{what}: `{}` launches {:?} workgroups of {:?}; axis {axis} takes {} of {per}",
                    fire.entrypoint,
                    fire.groups,
                    fire.group,
                    extent.div_ceil(per),
                );
            }
            fire.entrypoint
        })
        .collect()
}

const fn t(buf: u32, rows: u32, width: u32, dtype: Dtype) -> Tensor {
    Tensor::new(buf, rows, width, dtype)
}

#[test]
fn every_dense_matmul_launches_a_workgroup_a_tile() {
    let mut seen = std::collections::BTreeSet::new();
    for target in [Target::Spirv, Target::Wgsl] {
        let tiers: &[Tier] = match target {
            Target::Spirv => &[Tier::Baseline, Tier::Coopmat],
            Target::Wgsl => &[Tier::Baseline, Tier::Subgroup],
        };
        for &tier in tiers {
            for (m, n, k) in [
                (1, 4096, 64),
                (3, 70, 60),
                (2, 24, 64),
                (20, 64, 64),
                (40, 50, 64),
                (100, 96, 32),
                (130, 70, 48),
                (300, 4096, 64),
                (2048, 1024, 1024),
                (4096, 4096, 4096),
            ] {
                if target == Target::Wgsl && n % 2 == 1 {
                    continue;
                }
                let rec = Recorder::new(target);
                let shape = Shape::dense(m, n, k);
                for (vector, wide) in [
                    (0, false),
                    (0, true),
                    (1, false),
                    (4, false),
                    (8, false),
                    (16, false),
                ] {
                    gemm::matmul(
                        &rec,
                        gemm::Tactic { tier, vector, wide },
                        t(0, m, k, Dtype::Bf16),
                        t(1, n, k, Dtype::Bf16),
                        t(2, m, n, Dtype::Bf16),
                        crate::ssm::RowSpan::whole(t(0, m, k, Dtype::Bf16)),
                    )
                    .unwrap();
                    let what = format!(
                        "gemm {m} x {n} x {k} on {target:?} at {} (vector {vector}, wide {wide})",
                        tier.name()
                    );
                    seen.extend(assert_covers(&rec, &what, shape));
                }
                gemm::lane(
                    &rec,
                    "linear.matmul",
                    t(0, m, k, Dtype::F32),
                    t(1, n, k, Dtype::Bf16),
                    t(2, m, n, Dtype::F32),
                )
                .unwrap();
                let what = format!("lane gemm {m} x {n} x {k} on {target:?} at {}", tier.name());
                seen.extend(assert_covers(&rec, &what, shape));
            }
        }
    }
    for form in [
        "linear_gemm_gemv_v_bf16",
        "linear_gemm_gemv_v1_bf16",
        "linear_gemm_gemv_v8_bf16",
        "linear_gemm_gemv_v16_bf16",
        "linear_gemm_gemv_bf16",
        "linear_gemm_tiled_bm_64_bf16",
        "linear_gemm_register_bf16",
        "linear_gemm_register_bn_32_bf16",
        "linear_gemm_tiled_bf16",
        "linear_gemm_pairs_gemv_bf16",
        "linear_gemm_pairs_gemv_v1_bf16",
        "linear_gemm_pairs_gemv_v8_bf16",
        "linear_gemm_pairs_gemv_v16_bf16",
        "linear_gemm_pairs_tiled_bm_16_bf16",
        "linear_gemm_pairs_tiled_bf16",
    ] {
        assert!(seen.contains(form), "no case reached `{form}`: {seen:?}");
    }
}

#[test]
fn every_quantized_matmul_launches_a_workgroup_a_tile() {
    let (group, bits) = (64, 4);
    let mut seen = std::collections::BTreeSet::new();
    for target in [Target::Spirv, Target::Wgsl] {
        let ladders: &[Qmm] = match target {
            Target::Spirv => &[
                Qmm::Coopmat,
                Qmm::RegisterTiled { bm: 64, k16: false },
                Qmm::RegisterTiled { bm: 32, k16: true },
                Qmm::Tiled { crossover_tg: 1 },
                Qmm::Tiled {
                    crossover_tg: 1 << 20,
                },
                Qmm::OneRow { lanes: 32, rows: 1 },
                Qmm::OneRow { lanes: 16, rows: 2 },
                Qmm::Qmv { vecs: 4 },
                Qmm::Qmv { vecs: 2 },
            ],
            Target::Wgsl => &[
                Qmm::Tiled { crossover_tg: 1 },
                Qmm::Tiled {
                    crossover_tg: 1 << 20,
                },
                Qmm::Qmv { vecs: 4 },
                Qmm::Qmv { vecs: 8 },
            ],
        };
        for &rung in ladders {
            for (m, n, k) in [
                (1, 1024, 1024),
                (7, 96, 512),
                (100, 1024, 1024),
                (2048, 4096, 1024),
            ] {
                if matches!(rung, Qmm::OneRow { .. }) && m != 1 {
                    continue;
                }
                let rec = Recorder::new(target);
                let bank = Bank {
                    codes: t(1, n, k * bits / 32, Dtype::U32),
                    scales: t(2, n, k / group, Dtype::Bf16),
                    biases: Some(t(3, n, k / group, Dtype::Bf16)),
                    group,
                    bits,
                };
                qmm::matmul(
                    &rec,
                    &[rung],
                    t(0, m, k, Dtype::Bf16),
                    bank,
                    t(4, m, n, Dtype::Bf16),
                    m.div_ceil(64) * 64,
                )
                .unwrap();
                let what = format!("qmm {m} x {n} x {k} on {target:?} at {rung:?}");
                seen.extend(assert_covers(&rec, &what, Shape::dense(m, n, k)));
            }
        }
        for slices in [2, 4] {
            for m in [1, 100] {
                let (n, k) = (1024, 1024);
                let rec = Recorder::new(target);
                kquant::matmul(
                    &rec,
                    kquant::Tactic {
                        tier: Tier::Baseline,
                        slices,
                    },
                    t(0, m, k, Dtype::Bf16),
                    t(1, n, k / 256 * 144, Dtype::U8),
                    t(2, m, n, Dtype::Bf16),
                )
                .unwrap();
                let what = format!("q4_k {m} x {n} x {k} on {target:?}, {slices} strips");
                seen.extend(assert_covers(&rec, &what, Shape::dense(m, n, k)));
            }
        }
    }
    assert!(
        seen.iter().any(|e| e.starts_with("linear_qmm_register")),
        "no case reached the register-tiled qmm: {seen:?}"
    );
}

#[test]
fn every_routed_matmul_launches_a_workgroup_a_tile() {
    let (tokens, top_k, k, n, experts) = (7, 4, 512, 768, 8);
    let m = tokens * top_k;
    let x = t(0, tokens, k, Dtype::Bf16);
    let routes = t(1, tokens, top_k, Dtype::I32);
    let y = t(2, m, n, Dtype::Bf16);
    let bank = Bank {
        codes: t(3, experts * n, k * 4 / 32, Dtype::U32),
        scales: t(4, experts * n, k / 64, Dtype::Bf16),
        biases: Some(t(5, experts * n, k / 64, Dtype::Bf16)),
        group: 64,
        bits: 4,
    };
    let mut seen = std::collections::BTreeSet::new();
    for target in [Target::Spirv, Target::Wgsl] {
        let rec = Recorder::new(target);
        let routed = Shape {
            tokens,
            top_k,
            ..Shape::dense(m, n, k)
        };
        select::matmul_select(
            &rec,
            Tier::Baseline,
            x,
            t(6, experts * n, k, Dtype::Bf16),
            routes,
            y,
        )
        .unwrap();
        seen.extend(assert_covers(
            &rec,
            &format!("dense select on {target:?}"),
            routed,
        ));
        select::matmul_select_quant(&rec, Tier::Baseline, x, bank, routes, y).unwrap();
        seen.extend(assert_covers(
            &rec,
            &format!("affine select on {target:?}"),
            routed,
        ));

        let tiers: &[Tier] = match target {
            Target::Spirv => &[Tier::Baseline, Tier::Coopmat],
            Target::Wgsl => &[Tier::Baseline],
        };
        for &tier in tiers {
            for tile in [16, 32, 64] {
                let padded = experts * tile;
                let scratch = RoutedScratch {
                    perm: t(7, padded, 1, Dtype::I32),
                    row_expert: t(8, padded, 1, Dtype::I32),
                    tile_expert: t(9, padded / tile, 1, Dtype::I32),
                    inv: t(10, m, 1, Dtype::I32),
                    x: t(11, padded, k, Dtype::Bf16),
                    y: t(12, padded, n, Dtype::Bf16),
                };
                let batch = Batch {
                    tile,
                    padded,
                    tier,
                    sort_lanes: 1024,
                };
                let fired = select::matmul_select_batched(
                    &rec,
                    "linear.moe_select",
                    x,
                    bank,
                    None,
                    routes,
                    experts,
                    scratch,
                    y,
                    batch,
                )
                .unwrap();
                assert!(
                    fired,
                    "a batched route {tile} rows a tile at {tier:?} on {target:?} fires"
                );
                let what = format!(
                    "batched route, {tile}-row tiles at {} on {target:?}",
                    tier.name()
                );
                seen.extend(assert_covers(&rec, &what, Shape { padded, ..routed }));
            }
        }
    }
    assert!(
        seen.iter()
            .any(|e| e.starts_with("linear_moe_select_tiled")),
        "no case reached the tiled route: {seen:?}"
    );
}
