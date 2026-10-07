use poem_dsl::{Dtype, Weight};

pub const HIDDEN: u32 = 256;
pub const HEADS: u32 = 4;
pub const HEAD_DIM: u32 = 64;
pub const INTER: u32 = 512;
pub const CHANNELS: u32 = 16;
pub const PATCH: u32 = 2;
pub const PATCH_FEATURES: u32 = CHANNELS * PATCH * PATCH;
pub const TEXT_WIDTH: u32 = 256;
pub const CONTEXT_WIDTH: u32 = 512;
pub const TIMESTEP_DIM: u32 = 256;
pub const TIMESTEP_MAX_PERIOD: f32 = 10_000.0;
pub const TIMESTEP_FLIP_SIN_COS: bool = false;
pub const TIMESTEP_SCALE: f32 = 1.0;

pub const ROPE_DIMS: [u32; 4] = [16, 24, 24, 0];
pub const ROPE_THETA: f32 = 10_000.0;
pub const ROPE_AXES: u8 = 3;

pub const LN_EPS: f32 = 1e-6;
pub const RMS_EPS: f32 = 1e-6;

pub const SM_SCALE: f32 = 0.125;

pub const MOD_SLICES: u32 = 6;

pub mod port {
    pub const LATENTS: u8 = 0;
    pub const TEXT: u8 = 0;
    pub const CONTEXT: u8 = 1;
    pub const TIMESTEP: u8 = 0;
    pub const POSITIONS: u8 = 0;
}

pub struct Linear {
    pub w: Weight,
    pub bias: Weight,
}

impl Linear {
    fn at(name: &str, out: u32, in_: u32, banks: Dtype) -> Linear {
        Linear {
            w: Weight::sym(name, [u64::from(out), u64::from(in_)], banks),
            bias: Weight::sym(
                format!("{name}.bias"),
                [u64::from(out)],
                crate::dense(banks),
            ),
        }
    }

    fn columns(self) -> Linear {
        Linear {
            w: self.w.columns(),
            bias: self.bias.columns(),
        }
    }

    fn rows(self) -> Linear {
        Linear {
            w: self.w.rows(),
            bias: self.bias,
        }
    }

    fn packed(self, seams: impl IntoIterator<Item = u64> + Clone) -> Linear {
        Linear {
            w: self.w.packed(seams.clone()),
            bias: self.bias.packed(seams),
        }
    }
}

pub struct SelfAttn {
    pub qkv: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Linear,
}

impl SelfAttn {
    fn at(prefix: &str, banks: Dtype) -> SelfAttn {
        let dense = crate::dense(banks);
        let mine = HIDDEN;
        let seams = [u64::from(mine); 3];
        SelfAttn {
            qkv: Linear::at(&format!("{prefix}.qkv"), 3 * mine, HIDDEN, banks).packed(seams),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Linear::at(&format!("{prefix}.o"), HIDDEN, mine, banks).rows(),
        }
    }
}

pub struct CrossAttn {
    pub q: Linear,
    pub kv: Linear,
    pub q_norm: Weight,
    pub k_norm: Weight,
    pub out: Linear,
}

impl CrossAttn {
    fn at(prefix: &str, banks: Dtype) -> CrossAttn {
        let dense = crate::dense(banks);
        let mine = HIDDEN;
        CrossAttn {
            q: Linear::at(&format!("{prefix}.q"), mine, HIDDEN, banks).columns(),
            kv: Linear::at(&format!("{prefix}.kv"), 2 * mine, CONTEXT_WIDTH, banks)
                .packed([u64::from(mine); 2]),
            q_norm: Weight::sym(format!("{prefix}.q_norm"), [u64::from(HEAD_DIM)], dense),
            k_norm: Weight::sym(format!("{prefix}.k_norm"), [u64::from(HEAD_DIM)], dense),
            out: Linear::at(&format!("{prefix}.o"), HIDDEN, mine, banks).rows(),
        }
    }
}

pub struct Swiglu {
    pub gate_up: Linear,
    pub down: Linear,
}

impl Swiglu {
    fn at(prefix: &str, banks: Dtype) -> Swiglu {
        let name = format!("{prefix}.gate_up");
        let mine = INTER;
        let seams = [u64::from(mine), u64::from(mine)];
        Swiglu {
            gate_up: Linear {
                w: Weight::sym(&name, [u64::from(2 * mine), u64::from(HIDDEN)], banks)
                    .packed(seams),
                bias: Weight::sym(
                    format!("{name}.bias"),
                    [u64::from(2 * mine)],
                    crate::dense(banks),
                )
                .packed(seams),
            },
            down: Linear::at(&format!("{prefix}.down"), HIDDEN, mine, banks).rows(),
        }
    }
}

pub struct Single {
    pub ada: Linear,
    pub attn: SelfAttn,
    pub mlp: Swiglu,
}

pub struct Side {
    pub ada: Linear,
    pub attn: SelfAttn,
    pub mlp: Swiglu,
}

impl Side {
    fn at(prefix: &str, banks: Dtype) -> Side {
        Side {
            ada: Linear::at(&format!("{prefix}.ada"), MOD_SLICES * HIDDEN, HIDDEN, banks),
            attn: SelfAttn::at(&format!("{prefix}.attn"), banks),
            mlp: Swiglu::at(&format!("{prefix}.mlp"), banks),
        }
    }
}

pub struct Double {
    pub img: Side,
    pub txt: Side,
}

pub struct Cross {
    pub mod_table: Weight,
    pub ada: Linear,
    pub self_attn: SelfAttn,
    pub norm: Weight,
    pub norm_bias: Weight,
    pub cross: CrossAttn,
    pub mlp: Swiglu,
}

pub struct Model {
    pub banks: Dtype,
    pub x_embed: Linear,
    pub single: Single,
    pub double: Double,
    pub cross: Cross,
    pub final_ada: Linear,
    pub final_proj: Linear,
    pub tap: Option<String>,
}

impl Model {
    #[must_use]
    pub fn mini(banks: Dtype) -> Model {
        let dense = crate::dense(banks);
        Model {
            banks,
            x_embed: Linear::at("x_embed", HIDDEN, PATCH_FEATURES, banks),
            single: Single {
                ada: Linear::at("single.ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
                attn: SelfAttn::at("single.attn", banks),
                mlp: Swiglu::at("single.mlp", banks),
            },
            double: Double {
                img: Side::at("double.img", banks),
                txt: Side::at("double.txt", banks),
            },
            cross: Cross {
                mod_table: Weight::sym("cross.mod_table", [u64::from(MOD_SLICES * HIDDEN)], dense),
                ada: Linear::at("cross.ada", MOD_SLICES * HIDDEN, HIDDEN, banks),
                self_attn: SelfAttn::at("cross.self", banks),
                norm: Weight::sym("cross.norm", [u64::from(HIDDEN)], dense),
                norm_bias: Weight::sym("cross.norm.bias", [u64::from(HIDDEN)], dense),
                cross: CrossAttn::at("cross.x", banks),
                mlp: Swiglu::at("cross.mlp", banks),
            },
            final_ada: Linear::at("final.ada", 2 * HIDDEN, HIDDEN, banks),
            final_proj: Linear::at("final.proj", PATCH_FEATURES, HIDDEN, banks),
            tap: None,
        }
    }

    #[must_use]
    pub fn tapped(mut self, tap: Option<String>) -> Model {
        self.tap = tap;
        self
    }
}
