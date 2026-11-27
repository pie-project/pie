//! Contract A's near-tie verdict on one parting, for the gate tools: the
//! one rule ([`eta_compiler::contract::near_tie`]) every probe and gate reads
//! a parting of two runs by, printed as one line of JSON.
//!
//! ```text
//! near_tie --gap G --contested A,B --rows ROW_A ROW_B [--temperature T --top-p P]
//! near_tie --gap G --contested A,B --top TOP --steps STEPS
//! ```
//!
//! `G` is the decision's top-1 less its top-2 in logit units, `A,B` the
//! token each path chose. With `--rows`, each file holds one path's full
//! logit row at the parting's step: bf16 bits as hex, four digits a logit
//! with nothing between (as the margin bench dumps them), or decimal f32
//! logits separated by whitespace or commas. Without rows, `TOP` is the
//! step's top logit and `STEPS` the stated per-logit drift in bf16 steps
//! (the depth bound the logits propagate). `T` and `P` state the top-p
//! sampler the decision was drawn under, which the nucleus-edge clause
//! reads. The verdict names the clause that admitted the parting. Exits 0
//! on a near-tie, 1 on a parting that is not one, 2 on a malformed call.

use eta_compiler::contract::{Drift, Nucleus, near_tie};

fn row(path: &str) -> Result<Vec<f32>, String> {
    let text = std::fs::read_to_string(path).map_err(|why| format!("{path}: {why}"))?;
    let text = text.trim();
    let hex = !text.is_empty()
        && text.len().is_multiple_of(4)
        && text.bytes().all(|b| b.is_ascii_hexdigit());
    if hex {
        return (0..text.len())
            .step_by(4)
            .map(|at| {
                u32::from_str_radix(&text[at..at + 4], 16)
                    .map(|bits| f32::from_bits(bits << 16))
                    .map_err(|why| format!("{path}: {why}"))
            })
            .collect();
    }
    text.split(|c: char| c.is_whitespace() || c == ',')
        .filter(|word| !word.is_empty())
        .map(|word| {
            word.parse::<f32>()
                .map_err(|why| format!("{path}: {word}: {why}"))
        })
        .collect()
}

fn run(args: &[String]) -> Result<bool, String> {
    let value = |flag: &str| -> Option<&str> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|at| args.get(at + 1))
            .map(String::as_str)
    };
    let number = |flag: &str| -> Result<f32, String> {
        value(flag)
            .ok_or(format!("{flag} is missing"))?
            .parse::<f32>()
            .map_err(|why| format!("{flag}: {why}"))
    };
    let gap = number("--gap")?;
    let contested: Vec<usize> = value("--contested")
        .ok_or("--contested is missing")?
        .split(',')
        .map(|t| {
            t.trim()
                .parse::<usize>()
                .map_err(|why| format!("--contested: {why}"))
        })
        .collect::<Result<_, _>>()?;
    let nucleus = match (value("--temperature"), value("--top-p")) {
        (None, None) => None,
        _ => Some(Nucleus {
            temperature: number("--temperature")?,
            top_p: number("--top-p")?,
        }),
    };
    let parting = match args.iter().position(|a| a == "--rows") {
        Some(at) => {
            let (a, b) = match (args.get(at + 1), args.get(at + 2)) {
                (Some(a), Some(b)) => (row(a)?, row(b)?),
                _ => return Err("--rows takes two files".into()),
            };
            if a.len() != b.len() {
                return Err(format!("the rows are {} and {} wide", a.len(), b.len()));
            }
            near_tie(gap, &contested, Drift::Rows { a: &a, b: &b }, nucleus)
        }
        None => near_tie(
            gap,
            &contested,
            Drift::Depth {
                top: number("--top")?,
                steps: number("--steps")?,
            },
            None,
        ),
    };
    let ranks: Vec<String> = parting.ranks.iter().map(ToString::to_string).collect();
    let diffuse = parting
        .diffuse
        .map_or("null".to_string(), |d| d.to_string());
    let excl: Vec<String> = parting
        .excl
        .iter()
        .map(|(a, b)| format!("[{a}, {b}]"))
        .collect();
    let clause = parting
        .clause
        .map_or("null".to_string(), |c| format!("\"{}\"", c.name()));
    println!(
        "{{\"gap\": {}, \"drift_max\": {}, \"drift_rms\": {}, \"ranks\": [{}], \"bound\": {}, \"diffuse\": {}, \"excl\": [{}], \"clause\": {}, \"tie\": {}}}",
        parting.gap,
        parting.drift_max,
        parting.drift_rms,
        ranks.join(", "),
        parting.bound,
        diffuse,
        excl.join(", "),
        clause,
        parting.tie
    );
    Ok(parting.tie)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(why) => {
            eprintln!("near_tie: {why}");
            std::process::exit(2);
        }
    }
}
