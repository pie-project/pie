//! `import.json`: which checkpoint tensors fill each frozen weight of a
//! trace, by the import contract pie itself reads that checkpoint with.
//! A weight that is one checkpoint tensor as stored is a `copy`; any other
//! read (a slice, a reshape, a join, a dequantization) is an `expr`, the
//! contract's own expression over checkpoint tensors alone, so a trainer
//! that cannot apply it refuses by name.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use checkpoint::contract::{Expr, ModelContract};
use checkpoint::error::Error;
use poem::star::{Deploy, Package};
use poem::{ParamSource, Platform, Trace};
use serde_json::{Value, json};

/// `import.json` for `model`'s deployment `deploy`, whose trace is `trace`:
/// for each checkpoint weight, the tensors under `checkpoint` it reads and how.
pub fn of(
    package: &Package,
    model: &str,
    deploy: &Deploy,
    checkpoint: &Path,
    trace: &Trace,
) -> Result<Value> {
    let src = checkpoint::file::snapshot::open(checkpoint)
        .map_err(|why| anyhow!("open the checkpoint {}: {why}", checkpoint.display()))?;
    let contract = package
        .import(model, deploy, &src, Platform::Metal)
        .map_err(|why| anyhow!("`{model}` does not import {}: {why}", checkpoint.display()))?;
    let mut reads = Vec::new();
    for param in trace
        .params
        .iter()
        .filter(|p| p.source == ParamSource::Checkpoint)
    {
        let Some(tensor) = contract.tensors.iter().find(|t| t.name == param.name) else {
            bail!(
                "`{model}`'s import of {} states no read of the weight `{}`",
                checkpoint.display(),
                param.name
            );
        };
        let expr = inlined(tensor.expr.clone(), &contract)
            .map_err(|why| anyhow!("`{model}`'s read of `{}`: {why}", param.name))?;
        let read = match &expr {
            Expr::Src(name) => json!({"name": param.name, "checkpoint": [name], "op": "copy"}),
            expr => json!({
                "name": param.name,
                "checkpoint": expr.sources(),
                "op": "expr",
                "expr": expr,
            }),
        };
        reads.push(read);
    }
    let checkpoint = std::fs::canonicalize(checkpoint)
        .with_context(|| format!("resolve {}", checkpoint.display()))?;
    Ok(json!({"format": "pie-stanza.import/1", "checkpoint": checkpoint, "params": reads}))
}

/// `expr` with each read of another of `contract`'s tensors (`Out`) replaced
/// by that tensor's own expression, so it reads checkpoint tensors alone.
fn inlined(expr: Expr, contract: &ModelContract) -> Result<Expr, Error> {
    match expr {
        Expr::Out(name) => {
            let Some(tensor) = contract.tensors.iter().find(|t| t.name == name) else {
                return Err(Error::Contract(format!(
                    "it reads `{name}`, which it states no read of"
                )));
            };
            inlined(tensor.expr.clone(), contract)
        }
        expr => expr.map_children(|child| inlined(child, contract)),
    }
}
