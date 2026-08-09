//! GPT-OSS's quantization trio, solved from the staged tensors.
//!
//! `config.json` cannot be trusted for any of the three: mlx_lm's
//! quantization predicate leaves a 32×2880 router at 8 bits inside a
//! 4-bit checkpoint without recording that anywhere; the MXFP4-Q8 publish
//! declares a global mxfp4/4/g32 and then overrides every attention
//! projection back to affine/8/g64; and "mxfp4" in the config describes
//! the checkpoint the loader may have since CONVERTED. The only honest
//! witness is what is in the heap, so the facts are read off the staged
//! byte extents — arithmetic a wrong guess turns into fluent wrong text,
//! not a crash, which is why every branch here refuses instead of
//! defaulting.
//!
//! The C++ `router_bits_from_extents` (gptoss/bind.cpp), lifted portable:
//! the solver takes a name→bytes lookup, so the test suite probes it with
//! a table and the device path probes it with the staged handles.

use super::geometry_facts::GeometryRefused;
use super::gptoss::GptOssGeometry;

/// The affine width a weight/scales extent pair implies, at group 64 with
/// bf16 scales: `weight = N·K·bits/8`, `scales = N·(K/64)·2`, so the
/// ratio is `4·bits` and everything else cancels — including N and K,
/// which is what lets one function serve tensors of every shape.
///
/// `None` when the pair implies no width this driver ships a kernel for:
/// a zero extent, a ratio that is not a whole number of bits, or a width
/// other than 4 or 8. Truncating 5.3 "bits" to 5 would be a claim about
/// the packing; refusing is a claim about the solver.
#[must_use]
pub fn bits_from_extents(weight_bytes: u64, scale_bytes: u64) -> Option<u32> {
    if weight_bytes == 0 || scale_bytes == 0 {
        return None;
    }
    let denom = 4 * scale_bytes;
    if !weight_bytes.is_multiple_of(denom) {
        return None;
    }
    match weight_bytes / denom {
        4 => Some(4),
        8 => Some(8),
        _ => None,
    }
}

/// What the staged tensors say about the three facts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StagedQuant {
    /// The router matvec's affine width.
    pub router_bits: u32,
    /// The attention/embedding projections' affine width.
    pub proj_bits: u32,
    /// Whether the expert bank is still the checkpoint's MXFP4.
    pub mxfp4_experts: bool,
}

/// Solve the trio from a name→bytes lookup over the staged tensors.
///
/// Layer 0 is the probe for all three, as in the C++: the quantization
/// predicate is uniform across layers, so one layer's answer is the
/// model's. MXFP4 is recognised by what it LACKS — a `.biases` zero-point
/// companion — the same rule the load contract used when it decided not
/// to decode the bank.
///
/// # Errors
///
/// [`GeometryRefused`] naming the tensor whose extents could not carry
/// the answer. A missing probe means the staging and this solver disagree
/// about the model's shape, and every wrong resolution of that RUNS.
pub fn solve_staged_quant(
    bytes_of: impl Fn(&str) -> Option<u64>,
) -> Result<StagedQuant, GeometryRefused> {
    let refuse = |what: &str| GeometryRefused(format!("gpt-oss: {what}"));
    let pair = |base: &str| -> Result<(u64, u64), GeometryRefused> {
        let w = bytes_of(&format!("{base}.weight"))
            .ok_or_else(|| refuse(&format!("{base}.weight was not staged")))?;
        let s = bytes_of(&format!("{base}.scales"))
            .ok_or_else(|| refuse(&format!("{base}.scales was not staged")))?;
        Ok((w, s))
    };

    let (rw, rs) = pair("layers.0.mlp.router")?;
    let router_bits = bits_from_extents(rw, rs).ok_or_else(|| {
        refuse(&format!(
            "the router's extents ({rw} weight bytes over {rs} scale bytes) \
             imply no affine width this driver ships"
        ))
    })?;

    let (pw, ps) = pair("layers.0.self_attn.q_proj")?;
    let proj_bits = bits_from_extents(pw, ps).ok_or_else(|| {
        refuse(&format!(
            "q_proj's extents ({pw} weight bytes over {ps} scale bytes) \
             imply no affine width this driver ships"
        ))
    })?;

    let gate = "layers.0.mlp.experts.gate_proj";
    if bytes_of(&format!("{gate}.weight")).is_none() {
        return Err(refuse(&format!(
            "{gate}.weight was not staged; every gpt-oss layer routes"
        )));
    }
    let mxfp4_experts = bytes_of(&format!("{gate}.biases")).is_none();

    Ok(StagedQuant {
        router_bits,
        proj_bits,
        mxfp4_experts,
    })
}

/// Write the solved trio into `g` — the one place the geometry's three
/// "solved from the staged tensors" fields are assigned.
///
/// # Errors
///
/// As [`solve_staged_quant`].
pub fn solve_quant_into(
    g: &mut GptOssGeometry,
    bytes_of: impl Fn(&str) -> Option<u64>,
) -> Result<(), GeometryRefused> {
    let q = solve_staged_quant(bytes_of)?;
    g.router_bits = q.router_bits;
    g.proj_bits = q.proj_bits;
    g.mxfp4_experts = q.mxfp4_experts;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// The 20b MXFP4-Q4 publish's layer-0 extents: a 4-bit checkpoint
    /// whose router mlx_lm's predicate left at 8 bits, and an expert bank
    /// still in MXFP4 (scales are one block exponent per 32, no biases).
    fn mxfp4_q4_extents() -> HashMap<&'static str, u64> {
        let mut t = HashMap::new();
        t.insert("layers.0.mlp.router.weight", 32 * 2880 * 8 / 8);
        t.insert("layers.0.mlp.router.scales", 32 * (2880 / 64) * 2);
        t.insert("layers.0.self_attn.q_proj.weight", 4096 * 2880 * 4 / 8);
        t.insert("layers.0.self_attn.q_proj.scales", 4096 * (2880 / 64) * 2);
        t.insert(
            "layers.0.mlp.experts.gate_proj.weight",
            32 * 2880 * 2880 / 2,
        );
        t.insert(
            "layers.0.mlp.experts.gate_proj.scales",
            32 * 2880 * 2880 / 32,
        );
        t
    }

    fn lookup<'t>(t: &'t HashMap<&'static str, u64>) -> impl Fn(&str) -> Option<u64> + 't {
        move |name| t.get(name).copied()
    }

    #[test]
    fn the_router_solves_to_eight_bits_inside_a_four_bit_checkpoint() {
        let q = solve_staged_quant(lookup(&mxfp4_q4_extents())).expect("the shipped publish");
        assert_eq!(
            q,
            StagedQuant {
                router_bits: 8,
                proj_bits: 4,
                mxfp4_experts: true,
            }
        );
        let mut g = GptOssGeometry {
            router_bits: 0,
            proj_bits: 0,
            ..GptOssGeometry::default()
        };
        solve_quant_into(&mut g, lookup(&mxfp4_q4_extents())).expect("same table");
        assert_eq!((g.router_bits, g.proj_bits, g.mxfp4_experts), (8, 4, true));
    }

    #[test]
    fn a_converted_bank_is_affine_because_it_grew_a_zero_point() {
        let mut t = mxfp4_q4_extents();
        t.insert(
            "layers.0.mlp.experts.gate_proj.biases",
            32 * 2880 * (2880 / 64) * 2,
        );
        let q = solve_staged_quant(lookup(&t)).expect("an affine bank");
        assert!(!q.mxfp4_experts);
    }

    #[test]
    fn extents_that_imply_no_shipped_width_are_refused_not_truncated() {
        // The N and K cancel: any shape at 4 or 8 bits solves.
        assert_eq!(bits_from_extents(92_160, 2_880), Some(8));
        assert_eq!(bits_from_extents(5_898_240, 368_640), Some(4));
        // 6-bit packing exists in mlx; this driver ships no kernel for it.
        assert_eq!(bits_from_extents(69_120, 2_880), None);
        // A ratio that is not whole is a shape disagreement, not "about 4".
        assert_eq!(bits_from_extents(92_161, 2_880), None);
        assert_eq!(bits_from_extents(0, 2_880), None);

        let mut t = mxfp4_q4_extents();
        t.insert("layers.0.mlp.router.weight", 32 * 2880 * 6 / 8);
        let err = solve_staged_quant(lookup(&t)).expect_err("no 6-bit router kernel");
        assert!(err.0.contains("router"), "{}", err.0);

        let mut t = mxfp4_q4_extents();
        t.remove("layers.0.mlp.experts.gate_proj.weight");
        let err = solve_staged_quant(lookup(&t)).expect_err("a routeless gpt-oss");
        assert!(err.0.contains("gate_proj"), "{}", err.0);
    }
}
