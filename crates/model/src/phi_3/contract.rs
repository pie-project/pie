//! Phi-3's load contract.
//!
//! Phi-3 stores what every other dense decoder stores, fused differently:
//! `qkv_proj` and `gate_up_proj` arrive already joined. So the contract is the
//! ordinary dense one with two source-side *splits* in front of it — undo the
//! checkpoint's fusion, then let the dense join re-fuse on the device's terms.
//!
//! It sits in its own generation directory rather than inside Llama 3's
//! because the splits are Phi-3's alone; what it shares with Llama 3 is the
//! three-pass dense tail, which is spelled out below like every other
//! generation spells it out.

use model_loader::checkpoint::RawTensor;
use model_loader::contract::Expr;
use model_loader::error::Error;

use crate::shared::builder::Builder;

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

/// Phi-3: undo the two source-side fusions first, so the generic tail never
/// sees the fused tensors and the dense join can re-fuse on CUDA's terms.
pub fn author_phi3(b: &mut Builder<'_>) -> Result<(), Error> {
    phi3_fused_splits(b)?;
    // The dense tail, stated rather than bundled: a family's contract is
    // its pass sequence, and hiding three of them behind a helper meant
    // six families' contracts could not be read where they live.
    b.fused_moe_gate_up_tp_slices(false)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// Split Phi-3's fused QKV and gate/up back into the six tensors the
/// llama-like bind path reads.
fn phi3_fused_splits(b: &mut Builder<'_>) -> Result<(), Error> {
    for raw in b.tensors().to_vec() {
        if raw.name.ends_with(".self_attn.qkv_proj.weight") {
            phi3_qkv_split(b, raw)?;
        } else if raw.name.ends_with(".mlp.gate_up_proj.weight") {
            phi3_gate_up_split(b, raw)?;
        }
    }
    Ok(())
}

fn phi3_qkv_split(b: &mut Builder<'_>, raw: &RawTensor) -> Result<(), Error> {
    if raw.shape.len() != 2 {
        return fail(format!("Phi-3 fused QKV '{}' must be 2-D", raw.name));
    }
    let q_rows = raw.shape[1];
    let kv_rows = (raw.shape[0] - q_rows) / 2;
    if q_rows <= 0 || kv_rows <= 0 || q_rows + 2 * kv_rows != raw.shape[0] {
        return fail(format!(
            "Phi-3 fused QKV '{}' has an unsupported shape",
            raw.name
        ));
    }
    let cols = raw.shape[1];
    let base = raw
        .name
        .strip_suffix(".self_attn.qkv_proj.weight")
        .expect("matched above");
    let specs = [
        ("q_proj", 0, q_rows),
        ("k_proj", q_rows, kv_rows),
        ("v_proj", q_rows + kv_rows, kv_rows),
    ];
    for (proj, start, rows) in specs {
        let (expr, local_rows) = b.band(Expr::src(&raw.name), 0, start, rows);
        b.push_expr(
            format!("{base}.self_attn.{proj}.weight"),
            raw,
            vec![local_rows, cols],
            expr,
        );
    }
    Ok(())
}

fn phi3_gate_up_split(b: &mut Builder<'_>, raw: &RawTensor) -> Result<(), Error> {
    if raw.shape.len() != 2 || raw.shape[0] % 2 != 0 {
        return fail(format!(
            "Phi-3 fused gate/up '{}' has an unsupported shape",
            raw.name
        ));
    }
    let half_rows = raw.shape[0] / 2;
    let cols = raw.shape[1];
    let base = raw
        .name
        .strip_suffix(".mlp.gate_up_proj.weight")
        .expect("matched above");
    for (proj, start) in [("gate_proj", 0), ("up_proj", half_rows)] {
        let (expr, local_rows) = b.band(Expr::src(&raw.name), 0, start, half_rows);
        b.push_expr(
            format!("{base}.mlp.{proj}.weight"),
            raw,
            vec![local_rows, cols],
            expr,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::LoadShape;
    use crate::encoding::Encoding as StoredEncoding;
    use crate::shared::policy::Policy;
    use model_loader::checkpoint::CheckpointMetadata;
    use model_loader::contract::ModelContract;
    use model_loader::plan::StorageTarget;
    use model_loader::types::{BackendKind, DType, Encoding, FileId, TensorId};

    const H: i64 = 96; // hidden
    const KV: i64 = 32; // one kv group of 32
    const I: i64 = 128; // mlp intermediate

    fn tensor(id: u32, name: &str, shape: &[i64]) -> RawTensor {
        RawTensor {
            id: TensorId(id),
            name: name.to_string(),
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: u64::try_from(shape.iter().product::<i64>() * 2).unwrap_or(0),
            shape: shape.to_vec(),
            encoding: Encoding::Raw(DType::BF16),
        }
    }

    /// One layer, fused exactly the way Phi-3's release fuses it:
    /// `[q + k + v, hidden]` and `[2 * intermediate, hidden]`.
    fn fused(qkv: &[i64], gate_up: &[i64]) -> Vec<RawTensor> {
        vec![
            tensor(0, "model.layers.0.self_attn.qkv_proj.weight", qkv),
            tensor(1, "model.layers.0.mlp.gate_up_proj.weight", gate_up),
            tensor(2, "model.norm.weight", &[H]),
        ]
    }

    fn run(tensors: Vec<RawTensor>, tp_size: u32) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let enc = StoredEncoding::dense();
        let target = StorageTarget {
            backend: BackendKind::Cuda,
            tp_rank: 0,
            tp_size,
            max_tile_bytes: 1 << 20,
            preferred_alignment: 256,
            tile_map_mask: model_loader::plan::CUDA_TILE_MAP_MASK,
            ..StorageTarget::default()
        };
        let policy = Policy::default();
        let mut b = Builder::new(
            &meta,
            "phi-3-test",
            LoadShape::dense(1, 32, false),
            &enc,
            &target,
            &policy,
        );
        author_phi3(&mut b)?;
        b.finish()
    }

    fn refusal(tensors: Vec<RawTensor>) -> String {
        match run(tensors, 1).expect_err("an unsplittable fusion is refused") {
            Error::Contract(why) => why,
            other => panic!("expected a contract refusal, got {other:?}"),
        }
    }

    fn shape_of(c: &ModelContract, name: &str) -> Vec<i64> {
        c.tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| {
                panic!(
                    "no tensor {name}; have {:?}",
                    c.tensors.iter().map(|t| &t.name).collect::<Vec<_>>()
                )
            })
            .shape
            .clone()
            .unwrap_or_default()
    }

    /// Where a published tensor starts in the fused source it was cut
    /// from. `Shard` wraps the band at tp > 1, so the walk goes through
    /// whatever the sharding added.
    fn band_start(c: &ModelContract, name: &str) -> i64 {
        fn walk(e: &Expr) -> Option<i64> {
            match e {
                Expr::Slice { start, .. } => Some(*start),
                Expr::Shard { src, .. } => walk(src),
                _ => None,
            }
        }
        let t = c
            .tensors
            .iter()
            .find(|t| t.name == name)
            .unwrap_or_else(|| panic!("no tensor {name}"));
        walk(&t.expr).unwrap_or_else(|| panic!("{name} is not a band: {:?}", t.expr))
    }

    /// The two fusions come apart into the six tensors the llama-like
    /// bind path reads, at the widths the fused shapes imply.
    ///
    /// `q_rows` is read off the fused tensor's COLUMN count, because
    /// that column count is the hidden width and Phi-3's q projection is
    /// square. Everything left over is the two kv halves.
    #[test]
    fn the_two_fusions_come_apart_into_six_tensors() {
        let c = run(fused(&[H + 2 * KV, H], &[2 * I, H]), 1).expect("phi-3 authors");
        assert_eq!(
            shape_of(&c, "model.layers.0.self_attn.q_proj.weight"),
            [H, H]
        );
        assert_eq!(
            shape_of(&c, "model.layers.0.self_attn.k_proj.weight"),
            [KV, H]
        );
        assert_eq!(
            shape_of(&c, "model.layers.0.self_attn.v_proj.weight"),
            [KV, H]
        );
        assert_eq!(shape_of(&c, "model.layers.0.mlp.gate_proj.weight"), [I, H]);
        assert_eq!(shape_of(&c, "model.layers.0.mlp.up_proj.weight"), [I, H]);
        // Six tensors of the right WIDTH would also be six copies of the
        // same rows, so every band is checked by where it starts.
        for (name, start) in [
            ("self_attn.q_proj", 0),
            ("self_attn.k_proj", H),
            ("self_attn.v_proj", H + KV),
            ("mlp.gate_proj", 0),
            ("mlp.up_proj", I),
        ] {
            assert_eq!(
                band_start(&c, &format!("model.layers.0.{name}.weight")),
                start,
                "{name} was cut from the wrong offset"
            );
        }
        // And the fused sources do not ALSO ship: the dense tail would
        // otherwise publish them beside the halves it just derived.
        let names: Vec<&str> = c.tensors.iter().map(|t| t.name.as_str()).collect();
        assert!(
            !names
                .iter()
                .any(|n| n.contains("qkv_proj") || n.contains("gate_up_proj")),
            "the fused sources shipped beside their halves: {names:?}"
        );
    }

    /// A fused QKV this pass cannot divide is refused, and refused HERE.
    ///
    /// The split is arithmetic on two numbers, and every way that
    /// arithmetic can be wrong produces a q/k/v of some width -- just not
    /// the model's. A negative kv count silently becomes an empty band; a
    /// remainder silently shortens v. Both would load.
    #[test]
    fn a_fused_qkv_this_pass_cannot_divide_is_refused() {
        for (case, shape, wanted) in [
            ("rank 1", vec![H + 2 * KV], "must be 2-D"),
            ("rank 3", vec![1, H + 2 * KV, H], "must be 2-D"),
            // Columns wider than rows: q alone would be the whole tensor
            // and the kv halves would have negative height.
            ("no room for kv", vec![H, H + 2 * KV], "unsupported shape"),
            ("exactly q", vec![H, H], "unsupported shape"),
            // An odd remainder after q: the kv halves are not equal.
            (
                "a ragged kv split",
                vec![H + 2 * KV + 1, H],
                "unsupported shape",
            ),
        ] {
            let msg = refusal(fused(&shape, &[2 * I, H]));
            assert!(
                msg.contains(wanted) && msg.contains("qkv_proj"),
                "{case}: {msg}"
            );
        }
    }

    /// A fused gate/up that is not two equal halves is refused.
    #[test]
    fn a_fused_gate_up_that_is_not_two_halves_is_refused() {
        for (case, shape) in [
            ("rank 1", vec![2 * I]),
            ("rank 3", vec![2, I, H]),
            ("an odd row count", vec![2 * I + 1, H]),
        ] {
            let msg = refusal(fused(&[H + 2 * KV, H], &shape));
            assert!(
                msg.contains("unsupported shape") && msg.contains("gate_up_proj"),
                "{case}: {msg}"
            );
        }
    }

    /// Tensor parallelism splits the halves, not the fusion.
    ///
    /// The bands are taken on the FULL tensor and `band` returns the
    /// local extent, so each rank publishes its own slice of each
    /// projection rather than its own slice of the fused source. A pass
    /// that sharded first would cut q, k and v at one offset shared
    /// between them.
    #[test]
    fn each_rank_publishes_its_own_slice_of_every_half() {
        let c = run(fused(&[H + 2 * KV, H], &[2 * I, H]), 2).expect("phi-3 authors at tp 2");
        assert_eq!(
            shape_of(&c, "model.layers.0.self_attn.q_proj.weight"),
            [H / 2, H]
        );
        assert_eq!(
            shape_of(&c, "model.layers.0.self_attn.k_proj.weight"),
            [KV / 2, H]
        );
        assert_eq!(
            shape_of(&c, "model.layers.0.mlp.gate_proj.weight"),
            [I / 2, H]
        );
    }
}
