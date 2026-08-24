//! The numbers the claim-only routines want and the statements do not carry.
//!
//! EVERY ONE IS READ OFF THE PLAN, never off a config file and never off
//! `model::deployment`. The second of those is the tempting one and is the
//! reason this module exists rather than borrowing the driver's own shape:
//! `Deployment` is the LEGACY catalog's account of the checkpoint, and a
//! baker fire that took its head count from there would be firing one
//! catalog's program with the other catalog's numbers. When they agree, the
//! agreement is worth nothing; when they drift, the drift is a wrong answer
//! with no error in it.
//!
//! So: `head_dim` and the four gdn head numbers are statement params,
//! `kv_heads` divides the declared cache row by `head_dim`, `q_heads`
//! divides the decode statement's own operand, and `conv_dim` is the conv
//! statement's operand width. A number this cannot find is a refusal with
//! the point named.
//!
//! Lifted from `baker-smoke/src/smoke.rs:500-556`.

use model_compiler::program::{Program, Slot};
use model_ir::plan::{CacheRow, Plan};

/// One SKU's fire geometry, derived.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Geometry {
    pub head_dim: i32,
    pub kv_heads: i32,
    pub q_heads: i32,
    /// gdn key heads.
    pub k_h: i32,
    /// gdn value heads.
    pub v_h: i32,
    /// gdn key head dim.
    pub k_d: i32,
    /// gdn value head dim.
    pub v_d: i32,
    /// The convolution's width, in tokens.
    pub conv_k: i32,
    /// The convolution's channel count.
    pub conv_dim: i32,
}

impl Geometry {
    /// Read the geometry off `plan`, sizing against `program`'s slots.
    ///
    /// # Errors
    ///
    /// A statement the plan does not make, or a width that does not fit an
    /// `i32` — both named.
    pub(crate) fn of(plan: &Plan, program: &Program) -> Result<Geometry, String> {
        let find = |kernel: &str| plan.ops.iter().find(|o| o.kernel == kernel);
        let decode = find("attention.decode").ok_or("the plan states no `attention.decode`")?;
        let head_dim = i32::try_from(decode.params[1]).map_err(|_| "a wide head_dim")?;
        // `attention.decode`'s operand is the roped `q`, whose width is
        // `q_heads * head_dim`. `program.slots` is where that width lives.
        let q_width = match program.slots[decode.inputs[0] as usize] {
            Slot::Arena { width, .. } => width,
            ref other => return Err(format!("`attention.decode`'s q lives at {other:?}")),
        };
        let kv_row = plan
            .caches
            .iter()
            .find_map(|c| match c {
                CacheRow::Kv { row, .. } => Some(row.clone()),
                CacheRow::State { .. } => None,
            })
            .ok_or("the plan declares no kv cache row")?;
        // `[2, kv_heads * head_dim]`: the k/v pair, then the plane's width.
        // A row whose first extent is 1 is an MLA latent and is a different
        // pool this driver refuses at load, so the pair is asserted.
        if kv_row.first() != Some(&2) {
            return Err(format!(
                "the kv row is {kv_row:?}, not the `[2, kv_heads * head_dim]` \
                 pair a paged cache declares"
            ));
        }
        let kv_width = i32::try_from(kv_row[1]).map_err(|_| "a wide kv row")?;

        let gd = find("ssm.gated_delta").ok_or("the plan states no `ssm.gated_delta`")?;
        let conv = find("ssm.causal_conv1d").ok_or("the plan states no `ssm.causal_conv1d`")?;
        let conv_dim = match program.slots[conv.inputs[0] as usize] {
            Slot::Arena { width, .. } => i32::try_from(width).map_err(|_| "a wide conv")?,
            ref other => return Err(format!("`ssm.causal_conv1d`'s x lives at {other:?}")),
        };
        if head_dim <= 0 {
            return Err(format!("`attention.decode` states head_dim {head_dim}"));
        }
        Ok(Geometry {
            head_dim,
            kv_heads: kv_width / head_dim,
            q_heads: i32::try_from(q_width).map_err(|_| "a wide q")? / head_dim,
            k_h: gd.params[0] as i32,
            v_h: gd.params[1] as i32,
            k_d: gd.params[2] as i32,
            v_d: gd.params[3] as i32,
            conv_k: conv.params[0] as i32,
            conv_dim,
        })
    }

    /// Check this geometry against the LEGACY catalog's account of the same
    /// checkpoint.
    ///
    /// # Why this is the most valuable check in the module
    ///
    /// The baker lane fires the new catalog's program on the legacy
    /// catalog's pools. Those pools are sized, strided and slot-addressed
    /// from `model::deployment` — `bind::views::kv_view` takes
    /// `seq_stride = kv_heads * head_dim` from it, and `GdnCtx` takes
    /// `conv_stride`/`state_stride` from `RecurrentShape`. So if the two
    /// catalogues disagree about a head count, the program indexes one
    /// geometry and the pool is laid out for another, and NOTHING SAYS SO:
    /// every launch succeeds, every pointer is in range, and the answers
    /// are wrong.
    ///
    /// Two independent derivations of one truth are usually a smell. Here
    /// they are the point: they come from two different sources (one text
    /// each), they are ABOUT to be used together, and comparing them is the
    /// only moment at which their disagreement is cheap.
    ///
    /// # Errors
    ///
    /// The first field that disagrees, with both answers.
    pub(crate) fn agrees_with(
        &self,
        dep: &model::deployment::Deployment,
    ) -> Result<(), String> {
        let mut differ = Vec::new();
        let mut check = |what: &str, baker: i64, legacy: i64| {
            if baker != legacy {
                differ.push(format!("{what}: baker says {baker}, the catalog says {legacy}"));
            }
        };
        check("head_dim", self.head_dim.into(), dep.shape.head_dim.into());
        check("kv_heads", self.kv_heads.into(), dep.shape.kv_heads.into());
        check("q_heads", self.q_heads.into(), dep.shape.q_heads.into());
        if let Some(r) = dep.recurrent.as_ref() {
            check("gdn k_heads", self.k_h.into(), r.k_h.into());
            check("gdn v_heads", self.v_h.into(), r.v_h.into());
            check("gdn k_dim", self.k_d.into(), r.k_d.into());
            check("gdn v_dim", self.v_d.into(), r.v_d.into());
            check("conv_dim", self.conv_dim.into(), r.conv_dim.into());
            check("conv_kernel", self.conv_k.into(), r.conv_k.into());
        }
        if differ.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "the two catalogues describe different models — the baker \
                 lane would index one geometry on pools laid out for \
                 another: {}",
                differ.join("; ")
            ))
        }
    }
}
