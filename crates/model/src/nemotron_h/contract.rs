//! What Nemotron-H binds.
//!
//! Ported from `crates/driver-cuda/csrc/src/model/nemotron_h/nemotron_h_contract.hpp`.
//! The Mamba2/attention/MoE hybrid keeps its decoder under
//! `language_model.backbone.`, and its MoE GEMM addresses all experts of a
//! layer as one contiguous slab. The contract declares the slab and then
//! declares each expert as a slice of it, so no byte is copied twice.

use model_loader::checkpoint::RawTensor;
use model_loader::contract::{Expr, TensorType};
use model_loader::error::Error;
use model_loader::types::{DType, Encoding};

use crate::shared::builder::{Builder, is_raw};

fn fail<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Contract(what.into()))
}

pub fn author_nemotron_h(b: &mut Builder<'_>) -> Result<(), Error> {
    mamba_tp_shards(b)?;
    b.fused_moe_gate_up_tp_slices(false)?;
    packed_expert_views(b)?;
    b.dense_fused_projection_joins()?;
    b.publish_remaining()
}

/// A rename, written only when there is something to rename.
///
/// The algebra refuses a node that denotes its operand, so a fold whose
/// target is the shape the operand already has must be left out rather than
/// written and ignored. `to` may carry one `-1`, resolved here the way
/// `infer` resolves it.
fn retype_if_different(node: Expr, from: &[i64], mut to: Vec<i64>, encoding: &Encoding) -> Expr {
    let total: i64 = from.iter().product();
    let known: i64 = to.iter().filter(|&&dim| dim >= 0).product();
    for dim in &mut to {
        if *dim < 0 && known > 0 {
            *dim = total / known;
        }
    }
    if to == from {
        return node;
    }
    node.transmute(TensorType::new(to, encoding.clone()))
}

/// This rank's share of a leading axis that is `units` equal blocks.
///
/// `split` divides an extent, and an extent cannot say what it is a
/// concatenation *of*: `[heads * head_dim, cols]` divides by a `tp_size`
/// that does not divide `heads`, and the split then cuts a head in half.
/// Reshaping the axis to `[units, block]` first puts the divisibility
/// question on the value that has to answer it, so the loader rejects an
/// indivisible world naming this tensor instead of loading a wrong one.
///
/// Both transmutes are pure renames, so the composition still compiles to
/// one contiguous run per band.
fn split_by_unit(
    b: &Builder<'_>,
    rows: Expr,
    rows_shape: &[i64],
    units: i64,
    block: i64,
    out_shape: Vec<i64>,
    encoding: &Encoding,
) -> Expr {
    let folded = retype_if_different(rows, rows_shape, vec![units, block], encoding);
    let local_shape = [b.local_extent(units), block];
    retype_if_different(b.split(folded, 0), &local_shape, out_shape, encoding)
}

/// Elements per row of `shape`, i.e. everything but the leading extent.
fn row_elems(shape: &[i64]) -> i64 {
    shape[1..].iter().product()
}

/// `shape` with its leading extent replaced by `-1`.
fn rows_inferred(shape: &[i64]) -> Vec<i64> {
    let mut out = shape.to_vec();
    out[0] = -1;
    out
}

/// Split every Mamba mixer across the tensor-parallel world.
///
/// The knob (`PIE_NEMOTRON_DISABLE_TP_MAMBA_SHARD` in the driver) is a kill
/// switch for bisecting a numerical regression; because it changes the
/// contract, and the cache key is the contract's, flipping it re-plans
/// rather than reusing a cached plan for the other layout.
fn mamba_tp_shards(b: &mut Builder<'_>) -> Result<(), Error> {
    if b.target().tp_size <= 1 || !b.knobs().nemotron_tp_mamba_sharding {
        return Ok(());
    }
    for layer in 0..b.shape().layers {
        let mp = b.source_name(&format!("language_model.backbone.layers.{layer}.mixer."));
        layer_mamba_tp(b, &mp, i64::from(b.shape().mamba_groups))?;
    }
    Ok(())
}

/// Declare this rank's slice of one Mamba mixer.
///
/// The mixer's `in_proj` is five bands stacked on one axis — `z` and `x` of
/// `heads * head_dim` rows, `B` and `C` of `groups * state`, and `dt` of
/// `heads` — and TP splits each band by its own unit. So the local tensor is
/// the concatenation of five independently sharded bands, which is a
/// sentence the algebra can say and a row range cannot.
fn layer_mamba_tp(b: &mut Builder<'_>, mp: &str, groups: i64) -> Result<(), Error> {
    let (
        Some(in_proj),
        Some(conv_w),
        Some(conv_b),
        Some(a_log),
        Some(d),
        Some(dt_bias),
        Some(norm_w),
        Some(out_proj),
    ) = (
        b.find(&format!("{mp}in_proj.weight")),
        b.find(&format!("{mp}conv1d.weight")),
        b.find(&format!("{mp}conv1d.bias")),
        b.find(&format!("{mp}A_log")),
        b.find(&format!("{mp}D")),
        b.find(&format!("{mp}dt_bias")),
        b.find(&format!("{mp}norm.weight")),
        b.find(&format!("{mp}out_proj.weight")),
    )
    else {
        return Ok(()); // an attention or MoE layer; it has no mixer to split.
    };

    // Every extent below is read off a tensor. Only `groups` cannot be: see
    // `crate::catalog::LoadShape::mamba_groups`, which the ROW states
    // because the checkpoint fuses B and C into one bank and a loader
    // holding it can only see the product.
    let in_shape = in_proj.shape.clone();
    let conv_shape = conv_w.shape.clone();
    let out_shape = out_proj.shape.clone();
    if in_shape.len() != 2
        || conv_shape.len() < 2
        || out_shape.len() != 2
        || a_log.shape.len() != 1
        || norm_w.shape.len() != 1
    {
        return fail(format!(
            "nemotron_h: mamba mixer '{mp}' has an unexpected tensor rank"
        ));
    }
    let heads = a_log.shape[0];
    let intermediate = norm_w.shape[0];
    let hidden = in_shape[1];
    let conv_dim = conv_shape[0];
    let group_state = (conv_dim - intermediate) / 2;
    if heads <= 0
        || intermediate % heads != 0
        || conv_dim <= intermediate
        || (conv_dim - intermediate) % 2 != 0
        || groups <= 0
        || group_state % groups != 0
        || in_shape[0] != 2 * intermediate + 2 * group_state + heads
        || out_shape[0] != hidden
        || out_shape[1] != intermediate
    {
        return fail(format!(
            "nemotron_h: mamba mixer '{mp}' does not match heads={heads} groups={groups}"
        ));
    }
    let head_dim = intermediate / heads;
    let state = group_state / groups;

    // The five bands of `in_proj`, in the order the mixer reads them.
    let band_shape = vec![-1, hidden];
    let in_band = |start: i64, rows: i64| Expr::src(&in_proj.name).slice(0, start, rows);
    let z_shape = [intermediate, hidden];
    let bc_shape = [group_state, hidden];
    let enc = &in_proj.encoding;
    let bands = vec![
        split_by_unit(
            b,
            in_band(0, intermediate),
            &z_shape,
            heads,
            head_dim * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(intermediate, intermediate),
            &z_shape,
            heads,
            head_dim * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(2 * intermediate, group_state),
            &bc_shape,
            groups,
            state * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(2 * intermediate + group_state, group_state),
            &bc_shape,
            groups,
            state * hidden,
            band_shape.clone(),
            enc,
        ),
        split_by_unit(
            b,
            in_band(2 * intermediate + 2 * group_state, heads),
            &[heads, hidden],
            heads,
            hidden,
            band_shape,
            enc,
        ),
    ];
    let local_heads = b.local_extent(heads);
    let local_groups = b.local_extent(groups);
    let local_intermediate = local_heads * head_dim;
    let local_group_state = local_groups * state;
    let encoding = in_proj.encoding.clone();
    let in_proj_id = in_proj.id;
    b.define(
        b.output_name(&in_proj.name),
        Expr::concat(0, bands),
        encoding,
        Some(vec![
            2 * local_intermediate + 2 * local_group_state + local_heads,
            hidden,
        ]),
    );
    b.consume(in_proj_id);

    // `conv1d` carries the same bands minus `z` and `dt`, over whatever
    // trailing extents the checkpoint gave it (`[conv_dim, k]` or
    // `[conv_dim, 1, k]`).
    for raw in [conv_w, conv_b] {
        let shape = raw.shape.clone();
        let cols = row_elems(&shape);
        let published = rows_inferred(&shape);
        let mut z_band = shape.clone();
        z_band[0] = intermediate;
        let mut bc_band = shape.clone();
        bc_band[0] = group_state;
        let conv_band = |start: i64, rows: i64| Expr::src(&raw.name).slice(0, start, rows);
        let enc = &raw.encoding;
        let conv_bands = vec![
            split_by_unit(
                b,
                conv_band(0, intermediate),
                &z_band,
                heads,
                head_dim * cols,
                published.clone(),
                enc,
            ),
            split_by_unit(
                b,
                conv_band(intermediate, group_state),
                &bc_band,
                groups,
                state * cols,
                published.clone(),
                enc,
            ),
            split_by_unit(
                b,
                conv_band(intermediate + group_state, group_state),
                &bc_band,
                groups,
                state * cols,
                published,
                enc,
            ),
        ];
        let mut local_shape = shape;
        local_shape[0] = local_intermediate + 2 * local_group_state;
        let encoding = raw.encoding.clone();
        let id = raw.id;
        b.define(
            b.output_name(&raw.name),
            Expr::concat(0, conv_bands),
            encoding,
            Some(local_shape),
        );
        b.consume(id);
    }

    // One entry per head, so the head axis is already the leading extent.
    for raw in [a_log, d, dt_bias] {
        let expr = b.split(Expr::src(&raw.name), 0);
        let encoding = raw.encoding.clone();
        let id = raw.id;
        b.define(
            b.output_name(&raw.name),
            expr,
            encoding,
            Some(vec![local_heads]),
        );
        b.consume(id);
    }
    let norm_expr = split_by_unit(
        b,
        Expr::src(&norm_w.name),
        &[intermediate],
        heads,
        head_dim,
        vec![-1],
        &norm_w.encoding,
    );
    let encoding = norm_w.encoding.clone();
    let norm_id = norm_w.id;
    b.define(
        b.output_name(&norm_w.name),
        norm_expr,
        encoding,
        Some(vec![local_intermediate]),
    );
    b.consume(norm_id);

    // `out_proj` is `[hidden, heads * head_dim]`: the reduction axis is the
    // columns, so the unit fold goes on axis 1 and the split with it.
    let enc = out_proj.encoding.clone();
    let folded = Expr::src(&out_proj.name)
        .transmute(TensorType::new(vec![hidden, heads, head_dim], enc.clone()));
    let out_expr = b
        .split(folded, 1)
        .transmute(TensorType::new(vec![hidden, -1], enc.clone()));
    let out_proj_id = out_proj.id;
    b.define(
        b.output_name(&out_proj.name),
        out_expr,
        enc,
        Some(vec![hidden, local_intermediate]),
    );
    b.consume(out_proj_id);
    Ok(())
}

/// Publish each layer's experts as one packed slab plus per-expert views
/// into it, which is what the Nemotron-H MoE GEMM addresses.
fn packed_expert_views(b: &mut Builder<'_>) -> Result<(), Error> {
    if b.shape().n_experts == 0 {
        return Ok(());
    }
    for layer in 0..b.shape().layers {
        let base = format!("language_model.backbone.layers.{layer}.mixer.experts");
        if b.find(&format!("{base}.up_proj.packed.weight")).is_some()
            || b.find(&format!("{base}.down_proj.packed.weight")).is_some()
        {
            continue;
        }
        let mut up = Vec::new();
        let mut down = Vec::new();
        let mut complete = true;
        for expert in 0..b.shape().n_experts {
            let tag = format!("{base}.{expert}.");
            let (Some(u), Some(d)) = (
                b.find(&format!("{tag}up_proj.weight")),
                b.find(&format!("{tag}down_proj.weight")),
            ) else {
                complete = false;
                break;
            };
            up.push(u);
            down.push(d);
        }
        if complete {
            layer_packed_experts(b, &base, &up, &down);
        }
    }
    Ok(())
}

/// Declare one layer's experts as a slab plus slices of it.
///
/// Every guard below declines by RETURNING rather than refusing, because an
/// unpackable bank is still a loadable one: the experts stay as themselves
/// and only the fused GEMM's slab is missing. So each `return` is silent,
/// and the tests pin the absence of the slab rather than an error.
fn layer_packed_experts(b: &mut Builder<'_>, base: &str, up: &[&RawTensor], down: &[&RawTensor]) {
    // Unreachable as called: `packed_expert_views` returns early at
    // `n_experts == 0` and only calls here when the walk reached all of
    // them, so both slices hold at least one. Written as a `let else`
    // because the two `Option`s have to be opened somehow, not as a check.
    let (Some(first_up), Some(first_down)) = (up.first(), down.first()) else {
        return;
    };
    if first_up.shape.len() != 2
        || first_down.shape.len() != 2
        || !is_raw(&first_up.encoding, DType::BF16)
        || !is_raw(&first_down.encoding, DType::BF16)
    {
        return;
    }
    let full_intermediate = first_up.shape[0];
    let hidden = first_up.shape[1];
    for raw in up {
        if raw.shape != [full_intermediate, hidden] || raw.encoding != first_up.encoding {
            return;
        }
    }
    // This covers `first_down` too -- it is `down[0]`. A separate check on
    // the first pair stood here and could not change an outcome, because it
    // compared the same tensor against the same two extents; a control
    // deleting it was silent. The transpose is asserted once, here.
    for raw in down {
        if raw.shape != [hidden, full_intermediate] || raw.encoding != first_down.encoding {
            return;
        }
    }

    let local_intermediate = b.local_extent(full_intermediate);
    let expert_count = up.len() as i64;
    let bf16 = Encoding::Raw(DType::BF16);

    // Each expert contributes its local row band; the pack is their
    // concatenation. The sharding is in the expression, not in a flag.
    let up_name = format!("{base}.up_proj.packed.weight");
    let up_parts: Vec<Expr> = up
        .iter()
        .map(|raw| b.split(Expr::src(&raw.name), 0))
        .collect();
    b.define(
        up_name.clone(),
        Expr::concat(0, up_parts),
        bf16.clone(),
        Some(vec![expert_count * local_intermediate, hidden]),
    );

    let down_parts: Vec<Expr> = down.iter().map(|raw| Expr::src(&raw.name)).collect();
    let (down_expr, down_shape) = b.shard(
        Expr::concat(0, down_parts),
        vec![expert_count * hidden, full_intermediate],
        Some(1),
    );
    let down_name = format!("{base}.down_proj.packed.weight");
    b.define(down_name.clone(), down_expr, bf16.clone(), Some(down_shape));

    for (index, raw) in up.iter().enumerate() {
        let index = index as i64;
        b.define(
            raw.name.clone(),
            Expr::out(&up_name).slice(0, index * local_intermediate, local_intermediate),
            bf16.clone(),
            Some(vec![local_intermediate, hidden]),
        );
        b.consume(raw.id);
    }
    for (index, raw) in down.iter().enumerate() {
        let index = index as i64;
        b.define(
            raw.name.clone(),
            Expr::out(&down_name).slice(0, index * hidden, hidden),
            bf16.clone(),
            Some(vec![hidden, local_intermediate]),
        );
        b.consume(raw.id);
    }
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
    use model_loader::types::{BackendKind, FileId, TensorId};

    // The fixture's Mamba mixer, sized so that every extent the checker
    // recovers is distinct from every other. heads=4 of head_dim=8 gives
    // intermediate 32; groups=2 of state=8 gives group_state 16.
    const HIDDEN: i64 = 64;
    const HEADS: i64 = 4;
    const INTERMEDIATE: i64 = 32;
    const GROUP_STATE: i64 = 16;
    const CONV_DIM: i64 = INTERMEDIATE + 2 * GROUP_STATE;
    const IN_ROWS: i64 = 2 * INTERMEDIATE + 2 * GROUP_STATE + HEADS;
    const GROUPS: u32 = 2;
    const MP: &str = "language_model.backbone.layers.0.mixer.";
    const EP: &str = "language_model.backbone.layers.0.mixer.experts";

    fn bf16() -> Encoding {
        Encoding::Raw(DType::BF16)
    }
    fn f32e() -> Encoding {
        Encoding::Raw(DType::F32)
    }

    fn tensor(tensors: &mut Vec<RawTensor>, name: String, shape: Vec<i64>, encoding: Encoding) {
        let elements: i64 = shape.iter().product();
        tensors.push(RawTensor {
            id: TensorId(u32::try_from(tensors.len()).expect("a small fixture")),
            name,
            file_id: FileId(0),
            file_offset: 0,
            span_bytes: u64::try_from(elements).unwrap_or(0),
            shape,
            encoding,
        });
    }

    /// A mixer whose every extent AGREES, built from the three numbers the
    /// checker recovers. Damaging one number here keeps all the others
    /// consistent, which is the only way to make a single clause of the
    /// arithmetic guard fire alone: a fixture that just widens `norm.weight`
    /// also breaks the fused height, and then the guard refuses for a reason
    /// the test did not choose.
    fn mamba_of(heads: i64, intermediate: i64, group_state: i64) -> Vec<RawTensor> {
        let conv_dim = intermediate + 2 * group_state;
        let in_rows = 2 * intermediate + 2 * group_state + heads;
        let mut t = Vec::new();
        tensor(
            &mut t,
            format!("{MP}in_proj.weight"),
            vec![in_rows, HIDDEN],
            bf16(),
        );
        tensor(
            &mut t,
            format!("{MP}conv1d.weight"),
            vec![conv_dim, 1, 4],
            bf16(),
        );
        tensor(&mut t, format!("{MP}conv1d.bias"), vec![conv_dim], bf16());
        tensor(&mut t, format!("{MP}A_log"), vec![heads], f32e());
        tensor(&mut t, format!("{MP}D"), vec![heads], f32e());
        tensor(&mut t, format!("{MP}dt_bias"), vec![heads], f32e());
        tensor(
            &mut t,
            format!("{MP}norm.weight"),
            vec![intermediate],
            bf16(),
        );
        tensor(
            &mut t,
            format!("{MP}out_proj.weight"),
            vec![HIDDEN, intermediate],
            bf16(),
        );
        t
    }

    /// One well-formed Mamba mixer, the eight tensors `layer_mamba_tp`
    /// looks for and nothing else.
    fn mamba() -> Vec<RawTensor> {
        let mut t = Vec::new();
        tensor(
            &mut t,
            format!("{MP}in_proj.weight"),
            vec![IN_ROWS, HIDDEN],
            bf16(),
        );
        tensor(
            &mut t,
            format!("{MP}conv1d.weight"),
            vec![CONV_DIM, 1, 4],
            bf16(),
        );
        tensor(&mut t, format!("{MP}conv1d.bias"), vec![CONV_DIM], bf16());
        tensor(&mut t, format!("{MP}A_log"), vec![HEADS], f32e());
        tensor(&mut t, format!("{MP}D"), vec![HEADS], f32e());
        tensor(&mut t, format!("{MP}dt_bias"), vec![HEADS], f32e());
        tensor(
            &mut t,
            format!("{MP}norm.weight"),
            vec![INTERMEDIATE],
            bf16(),
        );
        tensor(
            &mut t,
            format!("{MP}out_proj.weight"),
            vec![HIDDEN, INTERMEDIATE],
            bf16(),
        );
        t
    }

    /// Two experts of a MoE mixer, the shape `layer_packed_experts` packs.
    fn experts() -> Vec<RawTensor> {
        let mut t = Vec::new();
        for e in 0..2 {
            tensor(
                &mut t,
                format!("{EP}.{e}.up_proj.weight"),
                vec![INTERMEDIATE, HIDDEN],
                bf16(),
            );
            tensor(
                &mut t,
                format!("{EP}.{e}.down_proj.weight"),
                vec![HIDDEN, INTERMEDIATE],
                bf16(),
            );
        }
        t
    }

    fn run(tensors: Vec<RawTensor>, n_experts: u32) -> Result<ModelContract, Error> {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors,
        };
        let enc = StoredEncoding::dense();
        // tp_size 2: `mamba_tp_shards` returns early at rank size 1, so a
        // single-rank fixture would exercise none of the fold.
        let target = StorageTarget::for_backend(BackendKind::Cuda, 0, 2);
        let policy = Policy::default();
        let shape = LoadShape {
            layers: 1,
            head_dim: 0,
            n_experts,
            mamba_groups: GROUPS,
            kv_shared_layers: 0,
            tied_embeddings: true,
        };
        let mut b = Builder::new(&meta, "nemotron-h-test", shape, &enc, &target, &policy);
        author_nemotron_h(&mut b)?;
        b.finish()
    }

    fn refusal(tensors: Vec<RawTensor>, n_experts: u32) -> String {
        match run(tensors, n_experts) {
            Ok(_) => panic!("expected a refusal"),
            Err(e) => e.to_string(),
        }
    }

    /// Replace one fixture tensor's shape, by name.
    fn reshaped(mut t: Vec<RawTensor>, name: &str, shape: Vec<i64>) -> Vec<RawTensor> {
        let raw = t
            .iter_mut()
            .find(|r| r.name == name)
            .unwrap_or_else(|| panic!("no fixture tensor named '{name}'"));
        raw.shape = shape;
        t
    }

    fn named(c: &ModelContract, name: &str) -> bool {
        c.tensors.iter().any(|t| t.name == name)
    }

    // ── The two refusals of the Mamba unit fold ──────────────────────────

    /// The eight-way `let else` above returns `Ok(())`, not a refusal, when
    /// a part is missing -- because Nemotron-H is a HYBRID and most of its
    /// layers have no mixer at all. So a layer that is simply not Mamba
    /// must pass through untouched rather than fail the load.
    #[test]
    fn a_layer_with_no_mamba_mixer_is_passed_over_rather_than_refused() {
        let mut t = mamba();
        t.retain(|r| !r.name.ends_with("dt_bias"));
        let c = run(t, 0).expect("a non-Mamba layer is not an error");
        // Untouched: the seven remaining parts are published as themselves,
        // not folded into the five bands the mixer reads.
        assert!(named(&c, &format!("{MP}in_proj.weight")));
        assert!(!named(&c, &format!("{MP}in_proj.z.weight")));
    }

    /// A rank is not a shape. This guard runs BEFORE any extent is read,
    /// and it must, because every line below it indexes `shape[0]` and
    /// `shape[1]` directly -- a 1-D `out_proj` would panic, not refuse.
    #[test]
    fn a_mixer_whose_tensor_lost_a_dimension_is_refused_for_its_rank() {
        let t = reshaped(mamba(), &format!("{MP}out_proj.weight"), vec![HIDDEN]);
        let m = refusal(t, 0);
        assert!(m.contains("unexpected tensor rank"), "{m}");
        assert!(m.contains(MP), "the refusal names the mixer: {m}");
    }

    /// `conv1d.weight` is the one part allowed MORE than two dimensions
    /// (it ships as `[conv_dim, 1, width]`), so the rank check spells it
    /// `< 2` while every other part is an equality. Squashing it to 1-D is
    /// what tells the two apart.
    #[test]
    fn a_conv_kernel_flattened_to_one_dimension_is_refused_for_its_rank() {
        let t = reshaped(mamba(), &format!("{MP}conv1d.weight"), vec![CONV_DIM]);
        assert!(refusal(t, 0).contains("unexpected tensor rank"));
    }

    /// The second guard is the arithmetic one: every extent has the right
    /// rank, but they do not describe one mixer. Here `norm.weight` claims
    /// an intermediate that `heads` does not divide, so `head_dim` would
    /// come out truncated and the five bands of `in_proj` would be sliced
    /// at the wrong offsets -- silently, since a slice of the right total
    /// width is still a legal slice.
    #[test]
    fn an_intermediate_the_head_count_does_not_divide_is_refused() {
        // 33 rows over 4 heads. Every OTHER extent is rebuilt around the 33
        // so that this is the only clause left to fire.
        let t = mamba_of(HEADS, INTERMEDIATE + 1, GROUP_STATE);
        let m = refusal(t, 0);
        assert!(m.contains("does not match heads="), "{m}");
        assert!(
            m.contains(&format!("heads={HEADS}")),
            "the refusal states what it read: {m}"
        );
        assert!(m.contains(&format!("groups={GROUPS}")), "{m}");
    }

    /// `groups` is the one extent NOT read off a tensor -- the row states
    /// it, because the checkpoint fuses B and C and only stores their
    /// product. So it is also the one extent that can be wrong while every
    /// tensor in the file is right, and this is the guard that catches it.
    #[test]
    fn a_row_whose_group_count_does_not_divide_the_fused_band_is_refused() {
        let meta = CheckpointMetadata {
            files: Vec::new(),
            tensors: mamba(),
        };
        let enc = StoredEncoding::dense();
        let target = StorageTarget::for_backend(BackendKind::Cuda, 0, 2);
        let policy = Policy::default();
        let shape = LoadShape {
            layers: 1,
            head_dim: 0,
            n_experts: 0,
            // GROUP_STATE is 16; 5 does not divide it.
            mamba_groups: 5,
            kv_shared_layers: 0,
            tied_embeddings: true,
        };
        let mut b = Builder::new(&meta, "nemotron-h-test", shape, &enc, &target, &policy);
        let m = author_nemotron_h(&mut b)
            .expect_err("a group count that does not divide the band")
            .to_string();
        assert!(m.contains("groups=5"), "{m}");
    }

    /// The fused `in_proj` is the only tensor whose height is a SUM of the
    /// other four extents, so it is the only one that can disagree with all
    /// of them at once. A wrong height here means the five bands do not
    /// tile it and the last band runs past the end.
    #[test]
    fn a_fused_input_projection_of_the_wrong_height_is_refused() {
        let t = reshaped(
            mamba(),
            &format!("{MP}in_proj.weight"),
            vec![IN_ROWS + HEADS, HIDDEN],
        );
        assert!(refusal(t, 0).contains("does not match heads="));
    }

    /// A conv bank no WIDER than the intermediate leaves no rows for B and
    /// C at all, which would make `group_state` zero or negative. Checked
    /// separately from the parity guard beside it because a negative
    /// remainder passes `% 2 == 0` just as happily as a positive one.
    #[test]
    fn a_conv_bank_no_wider_than_the_intermediate_is_refused() {
        // group_state 0, and every other extent rebuilt to agree with it --
        // so the fused height matches and only the width clause is left.
        // The parity clause beside it is happy too, since 0 is even, which
        // is exactly why the two are separate tests.
        let t = mamba_of(HEADS, INTERMEDIATE, 0);
        assert!(refusal(t, 0).contains("does not match heads="));
    }

    /// B and C are two bands of equal width, so their combined row count is
    /// even by construction. An odd one is a checkpoint that fused
    /// something else.
    #[test]
    fn a_conv_bank_with_an_odd_band_remainder_is_refused() {
        let t = reshaped(
            mamba(),
            &format!("{MP}conv1d.weight"),
            vec![CONV_DIM + 1, 1, 4],
        );
        assert!(refusal(t, 0).contains("does not match heads="));
    }

    // ── The silent declines of the packed expert view ────────────────────

    /// The happy path, stated so the declines below mean something: two
    /// well-formed experts become one slab plus two slices of it.
    #[test]
    fn two_well_formed_experts_become_one_slab_addressed_by_slices() {
        let c = run(experts(), 2).expect("a well-formed expert bank");
        let up = c
            .tensors
            .iter()
            .find(|t| t.name == format!("{EP}.up_proj.packed.weight"))
            .expect("the packed slab");
        // The slab is this rank's band of each expert concatenated, not
        // each expert whole: `local_extent` halves the intermediate at
        // tp 2, so the slab is 2 * 16 rows and not 2 * 32. The sharding
        // lives in the expression, so the DECLARED shape is the only place
        // it can be read back.
        assert_eq!(up.shape, Some(vec![2 * (INTERMEDIATE / 2), HIDDEN]));
        // Each expert is re-declared as a view of the slab, not as itself.
        let e0 = c
            .tensors
            .iter()
            .find(|t| t.name == format!("{EP}.0.up_proj.weight"))
            .expect("expert 0");
        // Not merely "reads the slab" -- reads the RIGHT band of it. The
        // slices are what make the pack free: no byte is copied twice, so
        // an expert's offset being wrong is a silent mis-bind rather than a
        // missing tensor.
        assert!(
            matches!(&e0.expr, Expr::Slice { src, start, len, .. }
                if matches!(&**src, Expr::Out(n) if *n == format!("{EP}.up_proj.packed.weight"))
                    && *start == 0
                    && *len == INTERMEDIATE / 2),
            "expert 0 is the slab's first local band: {:?}",
            e0.expr
        );
        let e1 = c
            .tensors
            .iter()
            .find(|t| t.name == format!("{EP}.1.up_proj.weight"))
            .expect("expert 1");
        assert!(
            matches!(&e1.expr, Expr::Slice { start, len, .. }
                if *start == INTERMEDIATE / 2 && *len == INTERMEDIATE / 2),
            "expert 1 follows expert 0 by one local band: {:?}",
            e1.expr
        );
    }

    /// Every guard in `layer_packed_experts` declines by RETURNING, not by
    /// refusing -- so a bank it will not pack still loads. What is lost is
    /// only the slab, and the fused MoE GEMM is the thing that asks for it
    /// by name. These tests pin that the load survives AND that the slab is
    /// absent, because "it loaded" is exactly the observation that hides
    /// this.
    #[test]
    fn experts_of_disagreeing_widths_are_left_unpacked_and_still_load() {
        let t = reshaped(
            experts(),
            &format!("{EP}.1.up_proj.weight"),
            vec![INTERMEDIATE * 2, HIDDEN],
        );
        let c = run(t, 2).expect("an unpackable bank is not a refusal");
        assert!(!named(&c, &format!("{EP}.up_proj.packed.weight")));
        // The experts are still there -- as themselves, straight off disk.
        assert!(named(&c, &format!("{EP}.1.up_proj.weight")));
    }

    /// The same disagreement on the down side. Checked separately because
    /// the two banks are walked by two loops, and one of them can be
    /// deleted without the other noticing.
    #[test]
    fn a_disagreeing_down_projection_is_left_unpacked() {
        let t = reshaped(
            experts(),
            &format!("{EP}.1.down_proj.weight"),
            vec![HIDDEN, INTERMEDIATE * 2],
        );
        let c = run(t, 2).expect("an unpackable bank is not a refusal");
        assert!(!named(&c, &format!("{EP}.down_proj.packed.weight")));
    }

    /// The pack declares itself BF16 unconditionally, so a bank of any
    /// other encoding must not enter it -- the slab would carry a dtype its
    /// bytes do not have and every element would be read wrong.
    #[test]
    fn an_expert_bank_that_is_not_bf16_is_left_unpacked() {
        let mut t = experts();
        for raw in &mut t {
            if raw.name.ends_with("up_proj.weight") {
                raw.encoding = f32e();
            }
        }
        let c = run(t, 2).expect("an unpackable bank is not a refusal");
        assert!(!named(&c, &format!("{EP}.up_proj.packed.weight")));
    }

    /// `down_proj` must be exactly `up_proj` transposed. This is the guard
    /// that reads the FIRST pair, before the two loops below check the rest
    /// against it, so a bank that is uniformly wrong gets caught here and
    /// not by them.
    #[test]
    fn a_down_projection_that_is_not_the_transpose_is_left_unpacked() {
        let mut t = experts();
        for raw in &mut t {
            if raw.name.ends_with("down_proj.weight") {
                raw.shape = vec![INTERMEDIATE, HIDDEN];
            }
        }
        let c = run(t, 2).expect("an unpackable bank is not a refusal");
        assert!(!named(&c, &format!("{EP}.down_proj.packed.weight")));
    }

    /// A checkpoint that already ships the slab is left alone -- including
    /// its per-expert tensors, which are NOT re-derived as slices. Without
    /// this skip the pass would concat a bank onto a name the file already
    /// holds, and the builder would see the same name defined twice.
    #[test]
    fn a_checkpoint_that_already_ships_the_slab_is_not_packed_again() {
        let mut t = experts();
        tensor(
            &mut t,
            format!("{EP}.up_proj.packed.weight"),
            vec![2 * INTERMEDIATE, HIDDEN],
            bf16(),
        );
        let c = run(t, 2).expect("a pre-packed bank");
        let e0 = c
            .tensors
            .iter()
            .find(|t| t.name == format!("{EP}.0.up_proj.weight"))
            .expect("expert 0");
        assert!(
            !matches!(&e0.expr, Expr::Out(name) if name.ends_with("packed.weight")),
            "a pre-packed bank does not re-derive its experts from a slab \
             this pass built: {:?}",
            e0.expr
        );
    }

    /// The `down_proj` half of the same skip. The condition is an `||` of
    /// two probes, so a checkpoint that shipped only one of the two slabs
    /// still skips -- deliberately, since packing the other half alone
    /// would give the GEMM one slab addressed by slices and one not.
    #[test]
    fn shipping_only_the_down_slab_still_skips_the_whole_layer() {
        let mut t = experts();
        tensor(
            &mut t,
            format!("{EP}.down_proj.packed.weight"),
            vec![2 * HIDDEN, INTERMEDIATE],
            bf16(),
        );
        let c = run(t, 2).expect("a half-packed bank");
        assert!(!named(&c, &format!("{EP}.up_proj.packed.weight")));
    }

    /// A row that states no experts skips the walk entirely, so a
    /// checkpoint that HAS experts is left unpacked. The row is
    /// authoritative here, the same way it is for `mamba_groups`.
    #[test]
    fn a_row_stating_no_experts_leaves_an_expert_bank_unpacked() {
        let c = run(experts(), 0).expect("a dense row over an expert bank");
        assert!(!named(&c, &format!("{EP}.up_proj.packed.weight")));
    }

    /// A bank SHORTER than the row states is not packed at all -- the
    /// `complete` flag. This is the hole that `kimi_k2` and `deepseek_v4`
    /// both had and this family does not: their loops probed a name to
    /// decide whether expert `n` existed, so a missing expert read as the
    /// end of the bank and they stacked a short slab. Here the loop counts
    /// to the row's `n_experts` and gives up if it cannot reach it.
    #[test]
    fn a_bank_shorter_than_the_row_states_is_not_packed_short() {
        let c = run(experts(), 3).expect("a short bank is not a refusal");
        assert!(!named(&c, &format!("{EP}.up_proj.packed.weight")));
    }
}
