//! One profile flag per fused op a rule forms: `fuse.<name>` holds where the
//! chip runs the fused kernel rather than its primitives. A backend registers
//! flags only for the rules it holds, so a profile can turn one off but never
//! on a fusion the backend lacks.

use profile::{Profile, Registry};

use super::Rule;

macro_rules! flags {
    ($($ty:ident = $id:literal => $kernel:literal,)*) => {
        $(
            #[derive(Clone, Copy, Debug, PartialEq, Eq, profile::Decision)]
            #[decision(id = $id, flag, tune = engine)]
            #[doc = concat!("Whether `", $kernel, "` replaces its primitives.")]
            pub struct $ty;
        )*

        /// Every flag as `(profile id, fused kernel)`.
        pub const ALL: &[(&str, &str)] = &[$(($id, $kernel)),*];

        /// Register a flag, on by default, for each fused op `rules` form.
        pub fn register(registry: &mut Registry, rules: &[Rule]) {
            for rule in rules {
                assert!(
                    ALL.iter().any(|(_, k)| *k == rule.kernel()),
                    "`{}` has no `fuse.*` flag",
                    rule.kernel()
                );
            }
            $(
                if rules.iter().any(|rule| rule.kernel() == $kernel) {
                    registry.add_defaulting::<$ty>("true");
                }
            )*
        }
    };
}

flags! {
    SplitQkvFused = "fuse.split_qkv_fused" => "fused.split_qkv_fused",
    QkvFusedQknormRopeVnormWrite = "fuse.qkv_fused_qknorm_rope_vnorm_write" => "fused.qkv_fused_qknorm_rope_vnorm_write",
    RmsnormResidualAdd = "fuse.rmsnorm_residual_add" => "fused.rmsnorm_residual_add",
    EmbedScaleAddSelect = "fuse.embed_scale_add_select" => "fused.embed_scale_add_select",
    EmbedScaleAdd = "fuse.embed_scale_add" => "fused.embed_scale_add",
    ResidualAddRmsnorm = "fuse.residual_add_rmsnorm" => "fused.residual_add_rmsnorm",
    MatmulSwigluMatmul = "fuse.matmul_swiglu_matmul" => "fused.matmul_swiglu_matmul",
    MatmulGlu = "fuse.matmul_glu" => "fused.matmul_glu",
    MoeExperts = "fuse.moe_experts" => "fused.moe_experts",
    MoeSelectGlu = "fuse.moe_select_glu" => "fused.moe_select_glu",
    MatmulSoftcap = "fuse.matmul_softcap" => "fused.matmul_softcap",
    MatmulBias = "fuse.matmul_bias" => "fused.matmul_bias",
    MatmulMoeRoute = "fuse.matmul_moe_route" => "fused.matmul_moe_route",
    MatmulSplitRows = "fuse.matmul_split_rows" => "fused.matmul_split_rows",
    GatedResidualNormModulate = "fuse.gated_residual_norm_modulate" => "fused.gated_residual_norm_modulate",
    NormModulate = "fuse.norm_modulate" => "fused.norm_modulate",
    NormRope = "fuse.norm_rope" => "fused.norm_rope",
    HcNormProject = "fuse.hc_norm_project" => "fused.hc_norm_project",
    SsmGdnStep = "fuse.gdn_step" => "fused.gdn_step",
    QGateKNormRopeAppend = "fuse.q_gate_norm_rope_append" => "fused.q_gate_norm_rope_append",
    NormRopeLastAppend = "fuse.norm_rope_last_append" => "fused.norm_rope_last_append",
    PoolCompress = "fuse.pool_compress" => "fused.pool_compress",
    NormRopePair = "fuse.norm_rope_pair" => "fused.norm_rope_pair",
    IndexTopkRope = "fuse.index_topk_rope" => "fused.index_topk_rope",
    SplitRopeYarnAppend = "fuse.split_rope_yarn_append" => "fused.split_rope_yarn_append",
    RopeYarnAppend = "fuse.rope_yarn_append" => "fused.rope_yarn_append",
    LseSink = "fuse.lse_sink" => "fused.lse_sink",
    PoolAttention = "fuse.pool_attention" => "fused.pool_attention",
    MlaPrefillExpanded = "fuse.mla_prefill_expanded" => "fused.mla_prefill_expanded",
    HcPrenormProject = "fuse.hc_prenorm_project" => "fused.hc_prenorm_project",
    HcGatesNorm = "fuse.hc_gates_norm" => "fused.hc_gates_norm",
    ResBlendRmsnorm = "fuse.res_blend_rmsnorm" => "fused.res_blend_rmsnorm",
    HcPre = "fuse.hc_pre" => "fused.hc_pre",
}

/// The rules of `rules` whose flag `profile` turns on: what
/// [`super::fuse`] runs for this load.
#[must_use]
pub fn on<'r>(profile: &Profile, rules: &'r [Rule]) -> Vec<&'r Rule> {
    rules
        .iter()
        .filter(|rule| {
            ALL.iter()
                .any(|(id, kernel)| *kernel == rule.kernel() && profile.flag(id) == Some(true))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rule_has_a_flag_and_every_flag_a_rule() {
        let rules: Vec<&str> = super::super::rules::all()
            .iter()
            .map(Rule::kernel)
            .collect();
        let flagged: Vec<&str> = ALL.iter().map(|(_, k)| *k).collect();
        for k in &rules {
            assert!(flagged.contains(k), "rule `{k}` has no `fuse.*` flag");
        }
        for k in &flagged {
            assert!(rules.contains(k), "flag for `{k}` names no rule");
        }
        for (id, kernel) in ALL {
            assert_eq!(id.strip_prefix("fuse."), kernel.rsplit('.').next());
            assert!(
                poem_ir::Fused::NAMES.contains(kernel),
                "`{kernel}` names no fused op"
            );
        }
    }

    #[test]
    fn a_backend_registers_what_it_ships_and_a_profile_may_turn_it_off() {
        use super::super::rules;
        let held = [rules::residual_add_rmsnorm(), rules::matmul_swiglu_matmul()];
        let mut r = Registry::new("toy", &[], &["rows"]);
        register(&mut r, &held);
        let labels = |p: &Profile| on(p, &held).iter().map(|r| r.kernel()).collect::<Vec<_>>();
        assert_eq!(
            labels(&r.defaults()),
            ["fused.residual_add_rmsnorm", "fused.matmul_swiglu_matmul"]
        );
        let p = r
            .load_user("api = 1\nfuse = [\"residual_add_rmsnorm\"]\n", "chip.toml")
            .expect("reads");
        assert_eq!(labels(&p), ["fused.residual_add_rmsnorm"]);
        let refused = r.load_user("api = 1\nfuse = [\"matmul_bias\"]\n", "chip.toml");
        assert!(
            refused.is_err(),
            "a chip cannot turn on a kernel its backend does not ship"
        );
    }
}
