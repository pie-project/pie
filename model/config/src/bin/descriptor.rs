//! Prints the `pie.model/1` descriptor for a HuggingFace `config.json`.
//!
//! The Rust half of the round-trip check in
//! `driver/cuda/tests/hf_config_dump/check_descriptor.sh`: this normalizes a
//! config the way `pie model import` does, and the C++ reader on the other
//! side has to turn the result back into exactly what `parse_hf_config` would
//! have produced from the same input.
//!
//!     descriptor path/to/config.json

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: descriptor <config.json>"))?;
    let raw = std::fs::read_to_string(&path)?;
    let root: serde_json::Value = serde_json::from_str(&raw)?;
    let descriptor = pie_model_config::descriptor(&root, &path)?;
    println!("{}", serde_json::to_string_pretty(&descriptor)?);
    Ok(())
}
