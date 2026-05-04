//! Initialize an Evolution Strategies adapter.
//!
//! Creates a new ES adapter (or re-uses an existing one) and optionally
//! loads pre-trained weights from a checkpoint.

use inferlet::{
    adapter::Adapter,
    model::Model,
    runtime,
    Result,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// Name of the adapter to create or open.
    name: String,
    /// LoRA rank.
    rank: u32,
    /// LoRA scaling factor.
    alpha: f32,
    /// ES population size.
    population_size: u32,
    /// Fraction of the population selected as parents.
    mu_fraction: f32,
    /// Initial perturbation standard deviation.
    initial_sigma: f32,
    /// Optional checkpoint path to load weights from.
    #[serde(default)]
    upload: Option<String>,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    let model_name = runtime::models().into_iter().next()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    // Reuse existing adapter or create + initialize a new one.
    let adapter = if let Some(existing) = Adapter::open(&model, &input.name) {
        println!("🔧 Existing adapter found: '{}'", input.name);
        existing
    } else {
        println!("🔧 Initializing new adapter '{}'...", input.name);
        let adapter = Adapter::create(&model, &input.name)?;
        inferlet::zo::zo::initialize(
            &adapter,
            input.rank,
            input.alpha,
            input.population_size,
            input.mu_fraction,
            input.initial_sigma,
        )?;
        adapter
    };

    // Optionally load pre-trained weights.
    if let Some(path) = &input.upload {
        if !path.is_empty() {
            println!("📥 Loading weights from '{}'...", path);
            adapter.load(path)?;
        }
    }

    println!("✅ Adapter '{}' ready.", input.name);
    Ok(format!("Adapter '{}' ready", input.name))
}
