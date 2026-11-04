//! gpt-oss on the wafer: the fit of its MoE unit (`moe_mxfp4`), from linked
//! sizes, and the layouts it rules out.
//!
//! Every expert is resident and sliced over a PX x PY grid like a dense MLP
//! (row py: Hc hidden dims; column px: Ic intermediate dims of gate, up and
//! down), so a token's top-k experts run on every PE of the grid. A PE holds
//! each expert's tiles as MXFP4 codes (three a u16) and a scale byte for
//! each output of each 32-block run of its inputs (at most two runs a tile).
//!
//! Measured (cslc links, `cs_readelf --syms` on an interior PE, B 1):
//! - the MoE unit alone (moe_mxfp4 on qwen2d_paged8b_hr's collectives, the
//!   layer body the MoE only) at 120 x 180, Hc 16, Ic 24, NE 8: 17,332 B of
//!   code and 20,346 B of data, of which 7,168 B the experts' tiles and
//!   5,580 B the host kernel's attention buffers a MoE grid would not carry;
//! - qwen2d_paged8b_hr's attention with the MoE in place of its MLP (W16,
//!   KV16, HROUTE, no copy kernel) at 128 x 234, Hc 13, Ic 23, NE 8: 24,520 B
//!   of code and 18,642 B of data, of which 5,760 B the experts' tiles and
//!   1,798 B the dense MLP's dead weights.

use super::model::{ModelSpec, Refused, refuse};

/// PE memory (`plan::WaferDims::pe_bytes`) less the stack's margin.
const PE_LIMIT: usize = 0xc000 - 768;

/// The MoE unit's bytes a PE besides the experts' tiles and its own
/// buffers: code, runtime, collectives, x (fitted to the 120 x 180 link).
const MOE_FIXED: usize = 17_332 + 20_346 - 7_168 - 5_580 - MOE_BUF_AT_LINK;
/// [`MoeUnit::buffers`] at the link's shape.
const MOE_BUF_AT_LINK: usize = 5_956;
/// The attention layer's bytes a PE besides the MoE (fitted to the 128 x 234
/// link: everything but the experts, the MoE's buffers and the dead MLP).
const ATTN_MOE_FIXED: usize = 24_520 + 18_642 - 5_760 - 1_798 - ATTN_BUF_AT_LINK;
const ATTN_BUF_AT_LINK: usize = 5_084;

/// One MoE grid's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeUnit {
    pub px: usize,
    pub py: usize,
    pub hc: usize,
    pub ic: usize,
    pub experts: usize,
    pub top_k: usize,
    pub lanes: usize,
    pub mlp_chunk: usize,
}

impl MoeUnit {
    /// Splits `hidden` over `py` rows and `inter` over `px` columns.
    pub fn new(spec: &ModelSpec, px: usize, py: usize, lanes: usize) -> Result<MoeUnit, Refused> {
        let Some(m) = &spec.moe else {
            return refuse(format!("{}: not a mixture of experts", spec.name));
        };
        Ok(MoeUnit {
            px,
            py,
            hc: spec.hidden.div_ceil(py),
            ic: spec.inter.div_ceil(px),
            experts: m.experts,
            top_k: m.top_k,
            lanes,
            mlp_chunk: 1,
        })
    }

    /// An expert's u16 words a PE: gate/up and down codes, two runs of scale
    /// bytes each (moe_mxfp4's `XW`).
    pub fn expert_words(&self) -> usize {
        let g2 = 2 * self.ic;
        (self.hc * g2).div_ceil(3)
            + (self.ic * self.hc).div_ceil(3)
            + 2 * g2.div_ceil(2)
            + 2 * self.hc.div_ceil(2)
    }

    /// The unit's buffers a PE (decode scratch, router tile, biases, lanes).
    pub fn buffers(&self) -> usize {
        let (g2, ne, tk) = (2 * self.ic, self.experts, self.top_k);
        let dec = 3
            * (self.hc * g2)
                .div_ceil(3)
                .max((self.ic * self.hc).div_ceil(3));
        let ner = ne.div_ceil(self.py);
        let nec = ne.div_ceil(self.px);
        4 * (dec
            + self.mlp_chunk * tk * g2
            + self.hc * ne
            + ne
            + ner * g2
            + nec * self.hc
            + tk * self.ic
            + 5 * self.ic
            + 2 * g2.max(self.hc)
            + self.lanes * (ne + 1 + 2 * tk))
    }

    /// A MoE-only grid PE's bytes.
    pub fn bytes(&self) -> usize {
        MOE_FIXED + self.buffers() + 2 * self.experts * self.expert_words()
    }

    /// A PE's bytes when the grid also runs the layer's attention
    /// (qwen2d_paged8b_hr's, bf16).
    pub fn bytes_with_attention(&self) -> usize {
        ATTN_MOE_FIXED + self.buffers() + 2 * self.experts * self.expert_words()
    }

    pub fn pes(&self) -> usize {
        self.px * self.py
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poem_ir::Platform;

    fn spec(name: &str) -> ModelSpec {
        let t = poem_compiler::catalog::repository()
            .parse(name)
            .unwrap()
            .trace(Platform::Cerebras);
        ModelSpec::read(&t).unwrap()
    }

    fn unit(px: usize, py: usize, hc: usize, ic: usize, ne: usize) -> MoeUnit {
        MoeUnit {
            px,
            py,
            hc,
            ic,
            experts: ne,
            top_k: 4,
            lanes: 1,
            mlp_chunk: 1,
        }
    }

    /// The model reproduces both links.
    #[test]
    fn the_memory_model_is_fitted_to_the_links() {
        let u = unit(120, 180, 16, 24, 8);
        assert_eq!(u.buffers(), MOE_BUF_AT_LINK);
        assert_eq!(2 * 8 * u.expert_words(), 7_168);
        assert_eq!(u.bytes(), 17_332 + 20_346 - 5_580);
        let a = unit(128, 234, 13, 23, 8);
        assert_eq!(a.buffers(), ATTN_BUF_AT_LINK);
        assert_eq!(2 * 8 * a.expert_words(), 5_760);
        assert_eq!(a.bytes_with_attention(), 24_520 + 18_642 - 1_798);
    }

    /// gpt-oss-20b's 32 experts do not fit a layer grid that also attends
    /// (128 x 234, the most rows five bands of grids leave), nor a MoE-only
    /// grid of 120 x 180: each layer's MoE wants more PEs than a 24-layer
    /// wafer gives a layer at today's code and three codes a u16.
    #[test]
    fn gptoss_20b_wants_more_pes_a_layer_than_one_grid() {
        let s = spec("gptoss-20b-bf16-mxfp4-kv-bf16");
        let one = MoeUnit::new(&s, 128, 234, 1).unwrap();
        assert_eq!((one.hc, one.ic), (13, 23));
        assert!(
            one.bytes_with_attention() > PE_LIMIT,
            "{}",
            one.bytes_with_attention()
        );
        let moe = MoeUnit::new(&s, 120, 180, 1).unwrap();
        assert!(moe.bytes() > PE_LIMIT, "{}", moe.bytes());
        // The smallest MoE-only grid of 128 columns that fits, and its share
        // of the wafer over 24 layers.
        let py = (180..=1172)
            .find(|&py| MoeUnit::new(&s, 128, py, 1).unwrap().bytes() <= PE_LIMIT)
            .unwrap();
        let fit = MoeUnit::new(&s, 128, py, 1).unwrap();
        let wafer = 762 * 1172;
        assert!(24 * fit.pes() < wafer, "py {py}: {} PEs", 24 * fit.pes());
        println!(
            "gpt-oss-20b MoE grid 128 x {py}: {} B a PE, {} PEs over 24 layers ({:.0}% of the wafer)",
            fit.bytes(),
            24 * fit.pes(),
            100.0 * (24 * fit.pes()) as f64 / wafer as f64
        );
    }
}
