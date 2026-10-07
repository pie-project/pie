use checkpoint::contract::{Expr, ModelContract, TensorType};
use checkpoint_dsl::{Builder, Error, extents};
use poem_dsl::Platform;

use super::model::{HIDDEN, MOD_SLICES, Model};

#[derive(Clone, Copy)]
enum Layout {
    Reference,
}

impl Layout {
    fn at(self, tail: &str) -> String {
        match self {
            Layout::Reference => tail.to_string(),
        }
    }
}

impl Model {
    pub fn import(
        &self,
        src: &ztensor::Source,
        platform: Platform,
    ) -> Result<ModelContract, Error> {
        self.import_from(src, platform, Layout::Reference)
    }

    fn import_from(
        &self,
        src: &ztensor::Source,
        platform: Platform,
        layout: Layout,
    ) -> Result<ModelContract, Error> {
        let mut b = Builder::new(src, 1, platform);
        let at = |tail: &str| layout.at(tail);

        biased(&mut b, &self.x_embed, &at("x_embedder"))?;
        biased(&mut b, &self.final_proj, &at("final_proj"))?;
        reordered(&mut b, &self.final_ada, &at("final_adaLN"), 2)?;

        reordered(&mut b, &self.single.ada, &at("blocks.0.adaLN"), MOD_SLICES)?;
        self_attn(&mut b, &self.single.attn, &at("blocks.0.attn"))?;
        swiglu(&mut b, &self.single.mlp, &at("blocks.0.mlp"))?;

        for (side, stem) in [
            (&self.double.img, "blocks.1.img"),
            (&self.double.txt, "blocks.1.txt"),
        ] {
            reordered(&mut b, &side.ada, &at(&format!("{stem}_adaLN")), MOD_SLICES)?;
            self_attn(&mut b, &side.attn, &at(&format!("{stem}_attn")))?;
            swiglu(&mut b, &side.mlp, &at(&format!("{stem}_mlp")))?;
        }

        let cross = &self.cross;
        b.read_expr(
            &cross.mod_table,
            slices_reordered(at("blocks.2.mod_table"), MOD_SLICES, 1, 0).transmute(
                TensorType::new(
                    extents(&cross.mod_table),
                    encoding_of(src, &at("blocks.2.mod_table"))?,
                ),
            ),
        )?;
        reordered(&mut b, &cross.ada, &at("blocks.2.adaLN"), MOD_SLICES)?;
        self_attn(&mut b, &cross.self_attn, &at("blocks.2.self_attn"))?;
        b.read(&cross.norm, at("blocks.2.norm_cross.weight"))?;
        b.read(&cross.norm_bias, at("blocks.2.norm_cross.bias"))?;
        biased(&mut b, &cross.cross.q, &at("blocks.2.cross_attn.q"))?;
        biased(&mut b, &cross.cross.kv, &at("blocks.2.cross_attn.kv"))?;
        b.read(&cross.cross.q_norm, at("blocks.2.cross_attn.norm_q"))?;
        b.read(&cross.cross.k_norm, at("blocks.2.cross_attn.norm_k"))?;
        biased(&mut b, &cross.cross.out, &at("blocks.2.cross_attn.out"))?;
        swiglu(&mut b, &cross.mlp, &at("blocks.2.mlp"))?;

        Ok(b.build())
    }
}

fn biased(b: &mut Builder, w: &super::model::Linear, stem: &str) -> Result<(), Error> {
    b.read(&w.w, format!("{stem}.weight"))?;
    b.read(&w.bias, format!("{stem}.bias"))
}

fn self_attn(b: &mut Builder, a: &super::model::SelfAttn, stem: &str) -> Result<(), Error> {
    biased(b, &a.qkv, &format!("{stem}.qkv"))?;
    b.read(&a.q_norm, format!("{stem}.norm_q"))?;
    b.read(&a.k_norm, format!("{stem}.norm_k"))?;
    biased(b, &a.out, &format!("{stem}.out"))
}

fn swiglu(b: &mut Builder, m: &super::model::Swiglu, stem: &str) -> Result<(), Error> {
    b.read_concat(
        &m.gate_up.w,
        [
            format!("{stem}.gate_proj.weight"),
            format!("{stem}.up_proj.weight"),
        ],
    )?;
    b.read_concat(
        &m.gate_up.bias,
        [
            format!("{stem}.gate_proj.bias"),
            format!("{stem}.up_proj.bias"),
        ],
    )?;
    biased(b, &m.down, &format!("{stem}.down_proj"))
}

fn reordered(
    b: &mut Builder,
    w: &super::model::Linear,
    stem: &str,
    slices: u32,
) -> Result<(), Error> {
    b.read_expr(
        &w.w,
        slices_reordered(format!("{stem}.weight"), slices, i64::from(HIDDEN), 0),
    )?;
    b.read_expr(
        &w.bias,
        slices_reordered(format!("{stem}.bias"), slices, i64::from(HIDDEN), 0),
    )
}

fn slices_reordered(from: String, slices: u32, width: i64, axis: u8) -> Expr {
    let take = |i: i64| Expr::src(from.clone()).slice(axis, i * width, width);
    let order: Vec<i64> = match slices {
        2 => vec![1, 0],
        6 => vec![1, 0, 2, 4, 3, 5],
        other => panic!("mini-dit states adaLN vectors of 2 or 6 slices, not {other}"),
    };
    Expr::concat(axis, order.into_iter().map(take).collect())
}

fn encoding_of(src: &ztensor::Source, name: &str) -> Result<checkpoint::types::Encoding, Error> {
    let Some(tensor) = src.get(name) else {
        return Err(Error::Missing(name.to_string()));
    };
    checkpoint::file::encoding_of(&tensor).map_err(|why| Error::Illegible {
        name: name.to_string(),
        detail: why.to_string(),
    })
}
