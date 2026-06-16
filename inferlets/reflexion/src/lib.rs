//! Reflexion-style scheduling with deterministic evaluation and verbal memory.
//!
//! A plain retry rewinds to the question and resamples. Reflexion instead turns
//! each failed attempt's concrete violations into a compact verbal lesson, then
//! injects accumulated lessons into the next attempt as explicit memory.

use inferlet::{Context, JsonSchema, Result, model::Model, runtime, sample::Sampler};
use serde::{Deserialize, Serialize};

const HORIZON_START: i32 = 9 * 60;
const HORIZON_END: i32 = 13 * 60;
const D_DEADLINE: i32 = 11 * 60;

const SYSTEM_PROMPT: &str = "\
You schedule maintenance work. Return only JSON matching the supplied schema. \
Times are integer minutes after midnight and intervals are half-open [start,end). \
Schedule jobs A=60 minutes, B=60 minutes, C=30 minutes, and D=30 minutes between \
09:00 (540) and 13:00 (780). B must start after A finishes. D must finish by \
11:00 (660). C must start after B finishes. Include one explicit 30-minute recovery \
interval after A and before B. No job or recovery interval may overlap another.";

const SCHEDULE_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "jobs": {
      "type": "array",
      "minItems": 4,
      "maxItems": 4,
      "items": {
        "type": "object",
        "properties": {
          "id": { "type": "string", "enum": ["A", "B", "C", "D"] },
          "start": { "type": "integer", "minimum": 540, "maximum": 780 },
          "end": { "type": "integer", "minimum": 540, "maximum": 780 }
        },
        "required": ["id", "start", "end"],
        "additionalProperties": false
      }
    },
    "recovery": {
      "type": "object",
      "properties": {
        "start": { "type": "integer", "minimum": 540, "maximum": 780 },
        "end": { "type": "integer", "minimum": 540, "maximum": 780 }
      },
      "required": ["start", "end"],
      "additionalProperties": false
    }
  },
  "required": ["jobs", "recovery"],
  "additionalProperties": false
}"#;

const REFLECTION_SCHEMA: &str = r#"{
  "type": "object",
  "properties": {
    "lesson": { "type": "string", "minLength": 1, "maxLength": 240 }
  },
  "required": ["lesson"],
  "additionalProperties": false
}"#;

#[derive(Deserialize)]
struct Input {
    #[serde(default = "default_mode")]
    mode: String,
    #[serde(default = "default_max_iterations")]
    max_iterations: usize,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default = "default_reflection_tokens")]
    reflection_tokens: usize,
}

fn default_mode() -> String {
    "reflexion".into()
}

fn default_max_iterations() -> usize {
    3
}

fn default_max_tokens() -> usize {
    256
}

fn default_reflection_tokens() -> usize {
    128
}

#[derive(Clone, Deserialize, Serialize)]
struct Schedule {
    jobs: Vec<Job>,
    #[serde(default)]
    recovery: Option<Interval>,
}

#[derive(Clone, Deserialize, Serialize)]
struct Job {
    id: String,
    start: i32,
    end: i32,
}

#[derive(Clone, Deserialize, Serialize)]
struct Interval {
    start: i32,
    end: i32,
}

#[derive(Clone, Serialize)]
struct Evaluation {
    valid: bool,
    violations: Vec<Violation>,
}

#[derive(Clone, Serialize)]
struct Violation {
    kind: &'static str,
    message: String,
}

#[derive(Deserialize)]
struct Reflection {
    lesson: String,
}

#[derive(Serialize)]
struct RunResult {
    mode: &'static str,
    success: bool,
    iterations: usize,
    schedule: String,
    violations: Vec<Violation>,
}

#[inferlet::main]
async fn main(input: Input) -> Result<String> {
    if input.max_iterations == 0 {
        return Err("max_iterations must be at least 1".into());
    }

    let model_name = runtime::models()
        .first()
        .cloned()
        .ok_or("No models available")?;
    let model = Model::load(&model_name)?;

    let mut attempt_root = Context::new(&model)?;
    attempt_root.system(SYSTEM_PROMPT);
    attempt_root.flush().await?;

    let mut reflection_root = Context::new(&model)?;
    reflection_root.system(
        "You are a precise scheduling critic. Given a failed schedule and deterministic \
     violations, produce one short actionable lesson for the next attempt. \
     Do not invent a full new schedule. Do not suggest moving a job past its deadline. \
     Mention the exact violated constraint and the current bad interval. \
     If giving a candidate repair, it must preserve all fixed constraints: \
     A=60, B=60, C=30, D=30; recovery=30; recovery must be after A and before B; \
     D must end by 660; no overlap. Prefer saying what must not happen over giving \
     a complete new schedule.",
    );
    reflection_root.flush().await?;

    let output = match input.mode.to_lowercase().as_str() {
        "baseline" => {
            let result = run_mode(&input, &attempt_root, &reflection_root, false).await?;
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
        }
        "reflexion" | "" => {
            let result = run_mode(&input, &attempt_root, &reflection_root, true).await?;
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
        }
        "both" => {
            let baseline = run_mode(&input, &attempt_root, &reflection_root, false).await?;
            let reflexion = run_mode(&input, &attempt_root, &reflection_root, true).await?;
            println!("\n--- Side-by-side summary ---");
            println!(
                "baseline:  iterations={} success={}",
                baseline.iterations, baseline.success
            );
            println!(
                "reflexion: iterations={} success={}",
                reflexion.iterations, reflexion.success
            );
            serde_json::to_string_pretty(&serde_json::json!({
                "baseline": baseline,
                "reflexion": reflexion
            }))
            .map_err(|e| e.to_string())?
        }
        other => {
            return Err(format!(
                "unknown mode '{other}': expected 'reflexion', 'baseline', or 'both'"
            ));
        }
    };

    println!("\n--- Final result ---\n{output}");
    Ok(output)
}

async fn run_mode(
    input: &Input,
    attempt_root: &Context,
    reflection_root: &Context,
    use_reflexion: bool,
) -> Result<RunResult> {
    let mode = if use_reflexion {
        "reflexion"
    } else {
        "baseline"
    };
    println!("\n=== {} ===", mode.to_uppercase());

    let mut memory: Vec<String> = Vec::new();
    let mut previous_evaluation: Option<String> = None;
    let mut latest = String::new();
    let mut last_violations = Vec::new();

    for iteration in 1..=input.max_iterations {
        println!("\n--- Attempt {iteration} ---");

        let mut ctx = attempt_root.fork()?;
        let attempt_prompt = if iteration == 1 {
            "Produce a candidate schedule directly. No evaluator feedback is available \
             on the first attempt."
                .to_string()
        } else if use_reflexion {
            format!(
                "Exact violations from the previous attempt:\n{}\n\n\
                 Accumulated reflection lessons:\n- {}\n\n\
                 Produce a corrected schedule. Preserve all already-satisfied constraints, \
                 but you may move any job if needed to fix the exact violations. \
                 Remember D must finish by 660, so if D conflicts with recovery at 600-630, \
                 try placing D in another non-overlapping 30-minute slot before or at 660. \
                 B may start later as long as it remains after A and recovery. \
                 Check all durations before answering: A=60, B=60, C=30, D=30, recovery=30.",
                previous_evaluation.as_deref().unwrap_or("{}"),
                memory.join("\n- ")
            )
        } else {
            // A retry only rewinds and resamples from the original prompt. It receives
            // neither evaluator feedback nor lessons from earlier failures.
            "Retry from the original instructions and produce a new schedule. \
             No feedback from earlier attempts is available."
                .to_string()
        };
        ctx.user(&attempt_prompt);
        ctx.cue();

        latest = ctx
            .generate(Sampler::TopP {
                temperature: 0.6,
                p: 0.95,
            })
            .max_tokens(input.max_tokens)
            .constrain_with(JsonSchema(SCHEDULE_SCHEMA))?
            .collect_text()
            .await?;

        println!("Schedule: {latest}");

        let schedule = match serde_json::from_str::<Schedule>(&latest) {
            Ok(schedule) => schedule,
            Err(error) => {
                let evaluation = Evaluation {
                    valid: false,
                    violations: vec![Violation {
                        kind: "invalid_json",
                        message: format!("Schedule JSON could not be parsed: {error}"),
                    }],
                };
                last_violations = evaluation.violations.clone();
                if use_reflexion && iteration < input.max_iterations {
                    reflect(
                        reflection_root,
                        &latest,
                        &evaluation,
                        input.reflection_tokens,
                        &mut memory,
                    )
                    .await?;
                    previous_evaluation =
                        Some(serde_json::to_string(&evaluation).map_err(|e| e.to_string())?);
                }
                continue;
            }
        };

        let evaluation = evaluate(&schedule);
        println!(
            "Evaluation: {}",
            serde_json::to_string(&evaluation).map_err(|e| e.to_string())?
        );

        if evaluation.valid {
            println!("Valid schedule found after {iteration} attempt(s).");
            return Ok(RunResult {
                mode,
                success: true,
                iterations: iteration,
                schedule: latest,
                violations: Vec::new(),
            });
        }

        last_violations = evaluation.violations.clone();
        if use_reflexion && iteration < input.max_iterations {
            reflect(
                reflection_root,
                &latest,
                &evaluation,
                input.reflection_tokens,
                &mut memory,
            )
            .await?;
            previous_evaluation =
                Some(serde_json::to_string(&evaluation).map_err(|e| e.to_string())?);
        }
    }

    println!(
        "No valid schedule found within {} attempt(s).",
        input.max_iterations
    );
    Ok(RunResult {
        mode,
        success: false,
        iterations: input.max_iterations,
        schedule: latest,
        violations: last_violations,
    })
}

async fn reflect(
    root: &Context,
    schedule: &str,
    evaluation: &Evaluation,
    max_tokens: usize,
    memory: &mut Vec<String>,
) -> Result<()> {
    let violations = serde_json::to_string(evaluation).map_err(|e| e.to_string())?;
    let mut ctx = root.fork()?;
    ctx.user(&format!(
        "Failed schedule:\n{schedule}\n\nDeterministic evaluation:\n{violations}"
    ));
    ctx.cue();

    let raw = ctx
        .generate(Sampler::Argmax)
        .max_tokens(max_tokens)
        .constrain_with(JsonSchema(REFLECTION_SCHEMA))?
        .collect_text()
        .await?;

    let reflection: Reflection = serde_json::from_str(&raw)
        .map_err(|e| format!("reflection JSON could not be parsed: {e}"))?;
    println!("Reflection: {}", reflection.lesson);
    // Reflexion adds a verbal lesson to explicit episodic memory. Unlike a
    // plain retry, the next attempt receives this lesson in its prompt.
    memory.push(reflection.lesson);
    Ok(())
}

fn evaluate(schedule: &Schedule) -> Evaluation {
    let mut violations = Vec::new();

    for id in ["A", "B", "C", "D"] {
        let count = schedule.jobs.iter().filter(|job| job.id == id).count();
        if count == 0 {
            violations.push(violation("missing_job", format!("Job {id} is missing.")));
        } else if count > 1 {
            violations.push(violation(
                "duplicate_job",
                format!("Job {id} appears {count} times."),
            ));
        }
    }

    for job in &schedule.jobs {
        let expected = expected_duration(&job.id);
        match expected {
            Some(duration) if job.end - job.start != duration => {
                violations.push(violation(
                    "incorrect_duration",
                    format!("Job {} must last {duration} minutes.", job.id),
                ));
            }
            None => violations.push(violation(
                "unknown_job",
                format!("Unknown job identifier {}.", job.id),
            )),
            _ => {}
        }
        check_horizon(&job.id, job.start, job.end, &mut violations);
    }

    if let Some(recovery) = &schedule.recovery {
        check_horizon("recovery", recovery.start, recovery.end, &mut violations);
        if recovery.end - recovery.start != 30 {
            violations.push(violation(
                "recovery",
                "Recovery must be an explicit 30-minute interval.".into(),
            ));
        }
    } else {
        violations.push(violation(
            "missing_recovery",
            "An explicit 30-minute recovery interval is required.".into(),
        ));
    }

    let mut intervals: Vec<(&str, i32, i32)> = schedule
        .jobs
        .iter()
        .map(|job| (job.id.as_str(), job.start, job.end))
        .collect();
    if let Some(recovery) = &schedule.recovery {
        intervals.push(("recovery", recovery.start, recovery.end));
    }
    for left in 0..intervals.len() {
        for right in (left + 1)..intervals.len() {
            let (left_id, left_start, left_end) = intervals[left];
            let (right_id, right_start, right_end) = intervals[right];
            if overlaps(left_start, left_end, right_start, right_end) {
                violations.push(violation(
                    "overlap",
                    format!("{left_id} overlaps {right_id}."),
                ));
            }
        }
    }

    if let (Some(a), Some(b)) = (unique_job(schedule, "A"), unique_job(schedule, "B"))
        && b.start < a.end
    {
        violations.push(violation(
            "ordering",
            "B must start after A finishes.".into(),
        ));
    }

    if let (Some(b), Some(c)) = (unique_job(schedule, "B"), unique_job(schedule, "C"))
        && c.start < b.end
    {
        violations.push(violation(
            "ordering",
            "C must start after B finishes.".into(),
        ));
    }

    if let (Some(a), Some(b), Some(recovery)) = (
        unique_job(schedule, "A"),
        unique_job(schedule, "B"),
        schedule.recovery.as_ref(),
    ) && (recovery.start < a.end || recovery.end > b.start)
    {
        violations.push(violation(
            "ordering",
            "Recovery must occur after A finishes and before B starts.".into(),
        ));
    }

    if let Some(d) = unique_job(schedule, "D")
        && d.end > D_DEADLINE
    {
        violations.push(violation(
            "deadline",
            "D must finish by 11:00 (660).".into(),
        ));
    }

    Evaluation {
        valid: violations.is_empty(),
        violations,
    }
}

fn expected_duration(id: &str) -> Option<i32> {
    match id {
        "A" | "B" => Some(60),
        "C" | "D" => Some(30),
        _ => None,
    }
}

fn unique_job<'a>(schedule: &'a Schedule, id: &str) -> Option<&'a Job> {
    let mut matches = schedule.jobs.iter().filter(|job| job.id == id);
    let job = matches.next()?;
    matches.next().is_none().then_some(job)
}

fn check_horizon(id: &str, start: i32, end: i32, violations: &mut Vec<Violation>) {
    if start >= end {
        violations.push(violation(
            "invalid_interval",
            format!("{id} must start before it ends."),
        ));
    }
    if start < HORIZON_START || end > HORIZON_END {
        violations.push(violation(
            "out_of_horizon",
            format!("{id} must remain within 09:00-13:00 (540-780)."),
        ));
    }
}

fn overlaps(a_start: i32, a_end: i32, b_start: i32, b_end: i32) -> bool {
    a_start < b_end && b_start < a_end
}

fn violation(kind: &'static str, message: String) -> Violation {
    Violation { kind, message }
}
