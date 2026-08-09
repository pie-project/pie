use crate::unit::Unit;

/// No device text.
pub static UNITS: &[Unit] = &[];

contract! {
    /// Applies a LoRA correction to a fused qkv projection, in place.
    LORA_QKV_CORRECTION = "pie_lora_qkv_correction" as lora_qkv_correction
}
