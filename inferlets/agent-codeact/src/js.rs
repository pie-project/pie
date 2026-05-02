//! JavaScript execution helper for the CodeACT agent. The agent now feeds
//! plain JS source through a JSON-Schema field, so there's no fence-
//! extraction to do — just evaluate.

use boa_engine::{Context, Source};

/// Evaluate `code` and return the last expression's value as a string.
/// Errors come back as `"Execution Error: …"` so the caller can detect
/// and surface them as observations.
pub fn eval(code: &str) -> String {
    let mut ctx = Context::default();
    match ctx.eval(Source::from_bytes(code)) {
        Ok(value) => value
            .to_string(&mut ctx)
            .ok()
            .and_then(|s| s.to_std_string().ok())
            .unwrap_or_else(|| "undefined".into()),
        Err(e) => format!("Execution Error: {e}"),
    }
}
