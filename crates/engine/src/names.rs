#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Naming {
    pub layer_prefix: &'static str,

    pub roles: &'static [(&'static str, &'static [&'static str])],

    pub globals: &'static [(&'static str, &'static [&'static str])],

    pub weight_suffix: &'static [&'static str],

    pub zero_point_suffix: &'static [&'static str],

    pub bias_suffix: &'static [&'static str],
}

const ROLES: &[(&str, &[&str])] = &[
    ("q_proj", &["self_attn.q_proj"]),
    ("k_proj", &["self_attn.k_proj"]),
    ("v_proj", &["self_attn.v_proj"]),
    ("o_proj", &["self_attn.o_proj", "linear_attn.out_proj"]),
    ("q_bias", &["self_attn.q_proj.bias"]),
    ("k_bias", &["self_attn.k_proj.bias"]),
    ("v_bias", &["self_attn.v_proj.bias"]),
    ("o_bias", &["self_attn.o_proj.bias"]),
    ("router_bias", &["mlp.gate.bias", "mlp.router.bias"]),
    ("conv_w", &["linear_attn.conv1d.weight"]),
    ("conv_b", &["linear_attn.conv1d.bias"]),
    ("a_log", &["linear_attn.A_log"]),
    ("dt", &["linear_attn.dt_bias"]),
    ("gate_norm", &["linear_attn.norm"]),
    ("in_proj_qkv", &["linear_attn.in_proj_qkv"]),
    ("in_proj_a", &["linear_attn.in_proj_a"]),
    ("in_proj_b", &["linear_attn.in_proj_b"]),
    ("in_proj_z", &["linear_attn.in_proj_z"]),
    ("q_norm", &["self_attn.q_norm"]),
    ("k_norm", &["self_attn.k_norm"]),
    ("attn_sinks", &["self_attn.sinks"]),
    ("gate_proj", &["mlp.gate_proj"]),
    ("up_proj", &["mlp.up_proj"]),
    ("down", &["mlp.down_proj"]),
    ("router", &["mlp.gate", "mlp.router"]),
    (
        "expert_gate",
        &[
            "mlp.switch_mlp.gate_proj",
            "experts.switch_glu.gate_proj",
            "mlp.experts.gate_proj",
        ],
    ),
    (
        "expert_up",
        &[
            "mlp.switch_mlp.up_proj",
            "experts.switch_glu.up_proj",
            "mlp.experts.up_proj",
        ],
    ),
    (
        "expert_down",
        &[
            "mlp.switch_mlp.down_proj",
            "experts.switch_glu.down_proj",
            "mlp.experts.down_proj",
        ],
    ),
    ("attn_norm", &["input_layernorm"]),
    (
        "mlp_norm",
        &["pre_feedforward_layernorm", "post_attention_layernorm"],
    ),
];

const GLOBALS: &[(&str, &[&str])] = &[
    ("embed", &["shared_embedding", "embed_tokens"]),
    ("lm_head", &["shared_embedding", "lm_head"]),
    ("final_norm", &["final_norm"]),
];

impl Naming {
    #[must_use]
    pub const fn mlx() -> Self {
        Self {
            layer_prefix: "layers.",
            roles: ROLES,
            globals: GLOBALS,

            weight_suffix: &[".weight", ""],
            zero_point_suffix: &[".biases"],
            bias_suffix: &[".bias"],
        }
    }

    #[must_use]
    pub fn spellings(&self, traced: &str) -> Vec<String> {
        let Some(t) = decompose(traced) else {
            return Vec::new();
        };
        let table = if t.layer.is_some() {
            self.roles
        } else {
            self.globals
        };
        let Some((_, paths)) = table.iter().find(|(role, _)| *role == t.role) else {
            return Vec::new();
        };
        let bases: Vec<String> = match t.layer {
            Some(l) => paths
                .iter()
                .map(|p| format!("{}{l}.{p}", self.layer_prefix))
                .collect(),
            None => paths.iter().map(|p| (*p).to_string()).collect(),
        };
        let suffixes: &[&str] = match t.sidecar {
            Sidecar::Packed => self.weight_suffix,

            Sidecar::Scales => &[".scales"],
            Sidecar::Zeros => self.zero_point_suffix,
            Sidecar::Bias => self.bias_suffix,
        };
        bases
            .iter()
            .flat_map(|b| suffixes.iter().map(move |s| format!("{b}{s}")))
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sidecar {
    Packed,
    Scales,
    Zeros,

    Bias,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Traced<'a> {
    layer: Option<u32>,
    role: &'a str,
    sidecar: Sidecar,
}

fn decompose(name: &str) -> Option<Traced<'_>> {
    let (rest, sidecar) = match name.rfind('.') {
        Some(at) if &name[at..] == ".scales" => (&name[..at], Sidecar::Scales),
        Some(at) if &name[at..] == ".zeros" => (&name[..at], Sidecar::Zeros),
        Some(at) if &name[at..] == ".bias" => (&name[..at], Sidecar::Bias),
        _ => (name, Sidecar::Packed),
    };
    if let Some(tail) = rest.strip_prefix("layer.") {
        let (index, role) = tail.split_once('.')?;
        Some(Traced {
            layer: Some(index.parse().ok()?),
            role,
            sidecar,
        })
    } else if rest.is_empty() {
        None
    } else {
        Some(Traced {
            layer: None,
            role: rest,
            sidecar,
        })
    }
}
