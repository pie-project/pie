//! Update an Evolution Strategies adapter.
//!
//! Receives population seeds + scores, performs the ES parameter update
//! (via `zo::update`), and optionally saves a checkpoint.

use inferlet::{
    adapter::Adapter,
    model::Model,
    runtime,
    Result,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Input {
    /// Adapter name.
    name: String,
    /// Population seeds.
    seeds: Vec<i64>,
    /// Fitness scores (one per seed).
    scores: Vec<f32>,
    /// Maximum sigma for adaptive noise scaling.
    max_sigma: f32,
    /// Optional checkpoint name to save the updated adapter.
    #[serde(default)]
    download: Option<String>,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.seeds.is_empty() {
        return Err("At least one seed and score must be provided.".into());
    }
    if input.seeds.len() != input.scores.len() {
        return Err(format!(
            "Seed count ({}) must match score count ({}).",
            input.seeds.len(), input.scores.len(),
        ));
    }

    let model_name = runtime::models().into_iter().next()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let adapter = Adapter::open(&model, &input.name)
        .ok_or_else(|| format!("Adapter '{}' not found", input.name))?;

    println!(
        "🔧 Updating adapter '{}' ({} scores, max_sigma={})...",
        input.name, input.scores.len(), input.max_sigma,
    );
    inferlet::zo::zo::update(&adapter, &input.scores, &input.seeds, input.max_sigma)?;

    // Optionally save a checkpoint.
    if let Some(path) = &input.download {
        if !path.is_empty() {
            println!("💾 Saving checkpoint to '{}'...", path);
            adapter.save(path)?;
        }
    }

    println!("✅ Adapter '{}' updated.", input.name);
    Ok(format!("Adapter '{}' updated", input.name))
}
