use std::collections::BTreeMap;

use poem_ir::{Platform, Shard};

use super::infer;

/// The ranks the report shards each row across.
const TWO: u32 = 2;

/// A param's name with its layer and expert indices elided, so one line
/// stands for the param in every layer.
fn pattern(name: &str) -> String {
    name.split('.')
        .map(|part| {
            if part.bytes().all(|b| b.is_ascii_digit()) {
                "{n}"
            } else {
                part
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

fn spelled(shard: &Shard) -> String {
    match shard {
        Shard::Replicated => "whole".into(),
        Shard::Inferred => "inferred".into(),
        Shard::Cut {
            axis,
            segments,
            heads,
        } if heads.is_empty() => format!("cut {axis} {segments:?}"),
        Shard::Cut {
            axis,
            segments,
            heads,
        } => format!("cut {axis} {segments:?} heads {heads:?}"),
    }
}

/// The cut without the head counts a model states as a hint.
fn unhinted(shard: &Shard) -> Shard {
    match shard {
        Shard::Cut { axis, segments, .. } => Shard::Cut {
            axis: *axis,
            segments: segments.clone(),
            heads: Vec::new(),
        },
        other => other.clone(),
    }
}

/// Over every catalog deployment, the cut inferred of each param against
/// the one its package states: how often they agree, by package, and the
/// params they differ on.
#[test]
#[ignore = "a report, read by hand while the rules grow"]
fn inferred_cuts_agree_with_the_stated_ones() {
    #[derive(Default)]
    struct Tally {
        params: usize,
        agree: usize,
        cut_agrees: usize,
        rows: usize,
        /// Rows whose stated cuts shard two ways, whose inferred ones do,
        /// and those whose two-rank traces are one.
        stated_shard: usize,
        inferred_shard: usize,
        same_trace: usize,
    }
    let mut tallies: BTreeMap<String, Tally> = BTreeMap::new();
    let mut traces = Vec::new();
    let mut differ: BTreeMap<(String, String, String, String), usize> = BTreeMap::new();
    for row in crate::catalog::deployments() {
        let Ok(trace) = row
            .package
            .trace(&row.model.id, &row.deploy, &row.name, Platform::Cuda)
        else {
            continue;
        };
        let package = row.package.name().to_string();
        let tally = tallies.entry(package.clone()).or_default();
        tally.rows += 1;
        let cuts = infer(&trace);
        let two = poem::star::Deploy {
            tp: TWO,
            ..row.deploy.clone()
        };
        if let Ok(whole) = row
            .package
            .trace(&row.model.id, &two, &row.name, Platform::Cuda)
        {
            let mut inferred = whole.clone();
            for (param, cut) in inferred.params.iter_mut().zip(infer(&whole)) {
                param.shard = cut;
            }
            let stated = crate::catalog::lowered(whole, &two);
            let split = crate::catalog::lowered(inferred, &two);
            tally.stated_shard += usize::from(stated.is_ok());
            tally.inferred_shard += usize::from(split.is_ok());
            let why = match (&stated, &split) {
                (Ok(a), Ok(b)) => poem_ir::canon::difference(a, b),
                (Err(a), Err(b)) => Some(format!("neither shards: {a} / {b}")),
                (Err(a), Ok(_)) => Some(format!("only inferred shards: {a}")),
                (Ok(_), Err(b)) => Some(format!("only stated shards: {b}")),
            };
            match why {
                None => tally.same_trace += 1,
                Some(why) => traces.push(format!(
                    "{}: {}",
                    row.name,
                    why.chars().take(400).collect::<String>()
                )),
            }
        }
        for (param, inferred) in trace.params.iter().zip(cuts) {
            tally.params += 1;
            if param.shard == inferred {
                tally.agree += 1;
            }
            if unhinted(&param.shard) == unhinted(&inferred) {
                tally.cut_agrees += 1;
            }
            if param.shard != inferred {
                *differ
                    .entry((
                        package.clone(),
                        pattern(&param.name),
                        spelled(&param.shard),
                        spelled(&inferred),
                    ))
                    .or_default() += 1;
            }
        }
    }
    let (mut all, mut agree, mut cut) = (0, 0, 0);
    for (package, t) in &tallies {
        all += t.params;
        agree += t.agree;
        cut += t.cut_agrees;
        eprintln!(
            "{package:<16} {:>4} rows {:>8} params  {:>6.2}% agree  {:>6.2}% on the cut  \
             tp2: stated {:>3} inferred {:>3} same {:>3}",
            t.rows,
            t.params,
            100.0 * t.agree as f64 / t.params.max(1) as f64,
            100.0 * t.cut_agrees as f64 / t.params.max(1) as f64,
            t.stated_shard,
            t.inferred_shard,
            t.same_trace,
        );
    }
    eprintln!(
        "{:<16} {all:>8} params  {:>6.2}% agree  {:>6.2}% on the cut",
        "all",
        100.0 * agree as f64 / all.max(1) as f64,
        100.0 * cut as f64 / all.max(1) as f64,
    );
    let limit = std::env::var("PIE_INFER_DIFFER")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(80);
    if std::env::var("PIE_INFER_TRACES").is_ok() {
        for line in &traces {
            eprintln!("{line}");
        }
    }
    let mut lines: Vec<_> = differ.into_iter().collect();
    lines.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for ((package, name, stated, inferred), n) in lines.into_iter().take(limit) {
        eprintln!("{n:>6} {package:<14} {name:<44} stated {stated:<36} inferred {inferred}");
    }
}

/// Every op the inference's tables name is an op: a row under a name no op
/// carries never applies.
#[test]
fn every_op_the_inference_names_is_an_op() {
    let named = super::UNPACKS
        .iter()
        .chain(super::AFFINE.iter())
        .chain(super::PER_HEAD.iter().map(|(op, _)| op));
    let stale: Vec<&&str> = named
        .filter(|name| !poem_ir::Operation::names_an_op(name))
        .collect();
    assert!(stale.is_empty(), "named by no op: {stale:?}");
}
