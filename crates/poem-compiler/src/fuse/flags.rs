//! One profile flag per fused kernel a rule forms: `fuse.<name>` holds where
//! the chip runs the fused kernel rather than its primitives. A backend
//! registers flags only for the kernels it ships, so a profile can turn one
//! off but never on a kernel the backend lacks.

use profile::{Profile, Registry};

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

        /// Register a flag, on by default, for each fused kernel `shipped`
        /// names. Panics on a name no rule forms.
        pub fn register(registry: &mut Registry, shipped: &[&str]) {
            for kernel in shipped {
                assert!(
                    ALL.iter().any(|(_, k)| k == kernel),
                    "`{kernel}` is a fused kernel no rule forms"
                );
            }
            $(
                if shipped.contains(&$kernel) {
                    registry.add_defaulting::<$ty>("true");
                }
            )*
        }
    };
}

flags! {
    QkvFusedQknormRopeVnormWrite = "fuse.qkv_fused_qknorm_rope_vnorm_write" => "custom_cuda.qkv_fused_qknorm_rope_vnorm_write",
    RmsnormResidualAdd = "fuse.rmsnorm_residual_add" => "elementwise.rmsnorm_residual_add",
    EmbedScaleAddSelect = "fuse.embed_scale_add_select" => "elementwise.embed_scale_add_select",
    EmbedScaleAdd = "fuse.embed_scale_add" => "elementwise.embed_scale_add",
    ResidualAddRmsnorm = "fuse.residual_add_rmsnorm" => "elementwise.residual_add_rmsnorm",
    MlpSwiglu = "fuse.mlp_swiglu" => "linear.mlp_swiglu",
    MatmulGeglu = "fuse.matmul_geglu" => "linear.matmul_geglu",
    LmHeadSoftcap = "fuse.lm_head_softcap" => "linear.lm_head_softcap",
    MatmulBias = "fuse.matmul_bias" => "linear.matmul_bias",
    GatedResidualNormModulate = "fuse.gated_residual_norm_modulate" => "elementwise.gated_residual_norm_modulate",
    NormModulate = "fuse.norm_modulate" => "elementwise.norm_modulate",
    RmsnormRopePartialQ = "fuse.rmsnorm_rope_partial_q" => "elementwise.rmsnorm_rope_partial_q",
}

/// The fused kernels `profile` turns on, in rule order: what [`super::fuse`]
/// forms for this load.
#[must_use]
pub fn kernels(profile: &Profile) -> Vec<&'static str> {
    ALL.iter()
        .filter(|(id, _)| profile.flag(id) == Some(true))
        .map(|(_, kernel)| *kernel)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_rule_has_a_flag_and_every_flag_a_rule() {
        let rules: Vec<&str> = super::super::kernels().collect();
        let flagged: Vec<&str> = ALL.iter().map(|(_, k)| *k).collect();
        for k in &rules {
            assert!(flagged.contains(k), "rule `{k}` has no `fuse.*` flag");
        }
        for k in &flagged {
            assert!(rules.contains(k), "flag for `{k}` names no rule");
        }
        for (id, kernel) in ALL {
            assert_eq!(id.strip_prefix("fuse."), kernel.rsplit('.').next());
        }
    }

    #[test]
    fn a_backend_registers_what_it_ships_and_a_profile_may_turn_it_off() {
        let mut r = Registry::new("toy", &[], &["rows"]);
        register(
            &mut r,
            &["elementwise.residual_add_rmsnorm", "linear.mlp_swiglu"],
        );
        assert_eq!(
            kernels(&r.defaults()),
            ["elementwise.residual_add_rmsnorm", "linear.mlp_swiglu"]
        );
        let p = r
            .load_user("api = 1\nfuse = [\"residual_add_rmsnorm\"]\n", "chip.toml")
            .expect("reads");
        assert_eq!(kernels(&p), ["elementwise.residual_add_rmsnorm"]);
        let refused = r.load_user("api = 1\nfuse = [\"matmul_bias\"]\n", "chip.toml");
        assert!(
            refused.is_err(),
            "a chip cannot turn on a kernel its backend does not ship"
        );
    }
}
