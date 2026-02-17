use inferlet::prelude::*;
use inferlet::{adapter::Adapter, parse_args, runtime, Result};
use inferlet::wstd::time::Duration;

#[inferlet::main]
async fn main(args: Vec<String>) -> Result<String> {
    let mut args = parse_args(args);
    let name: String = args.value_from_str("--name").map_err(|e| e.to_string())?;
    let seeds: Vec<i64> = args.value_from_fn("--seeds", |s| {
        s.split(',')
            .map(|v| v.parse::<i64>())
            .collect::<std::result::Result<Vec<_>, _>>()
    }).map_err(|e| e.to_string())?;
    let scores: Vec<f32> = args.value_from_fn("--scores", |s| {
        s.split(',')
            .map(|v| v.parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()
    }).map_err(|e| e.to_string())?;
    let max_sigma: f32 = args.value_from_str("--max-sigma").map_err(|e| e.to_string())?;
    let download: Option<String> = args.opt_value_from_str("--download").map_err(|e| e.to_string())?;

    // Input validation.
    if seeds.is_empty() {
        return Err("At least one seed and score must be provided.".to_string());
    }
    if seeds.len() != scores.len() {
        return Err(format!(
            "The number of seeds ({}) must match the number of scores ({}).",
            seeds.len(),
            scores.len()
        ));
    }

    // Load the model and look up the adapter.
    let model_name = runtime::models().into_iter().next()
        .ok_or_else(|| "No models available".to_string())?;
    let model = Model::load(&model_name)?;

    println!("🔧 Updating adapter '{}'...", &name);
    let adapter = Adapter::lookup(&model, &name)
        .ok_or_else(|| format!("Adapter '{}' not found", name))?;

    // Perform the ES update.
    println!(
        "Updating adapter '{}' with {} scores (max_sigma = {})...",
        &name,
        scores.len(),
        max_sigma
    );
    inferlet::zo::zo::update(&adapter, &scores, &seeds, max_sigma)?;

    // If a download path was provided, save the adapter weights.
    if let Some(path) = &download {
        if !path.is_empty() {
            println!("📥 Saving adapter '{}' to '{}'...", name, path);
            adapter.save(path)?;
        }
    }

    inferlet::wstd::task::sleep(Duration::from_millis(100)).await;
    println!("✅ Adapter '{}' updated successfully.", name);

    Ok(format!("Adapter '{}' updated", name))
}
