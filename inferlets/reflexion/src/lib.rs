//! Reflexion-style scheduling with deterministic evaluation and verbal memory.
//!
//! A plain retry rewinds to the question and resamples. Reflexion instead turns
//! each failed attempt's concrete violations into a compact verbal lesson, then
//! injects accumulated lessons into the next attempt as explicit memory.
//!
//! **Low-level ① rewrite (grammar, SEQUENTIAL + fork).** Off the
//! `Context`/`Generator`/`Sampler`/`constrain_with` facade onto the keep-core
//! (`ptir-grammar-tranche-conversion-spec`): `KvWorkingSet::fork()` (COW-shared
//! prefix) per attempt/reflection, `JsonSchema(schema).build_constraint()` (host
//! `Matcher`: `advance`/`mask`/`is_terminated`), and the masked grammar samplers.
//! The shipped facade computed-but-DROPPED the grammar mask (Stage-1); this now
//! ENFORCES the schema. Attempts sample (`TopP` → `grammar_program_sampled`, the
//! masked Gumbel sampler that keeps diversity); reflections are greedy (`Argmax` →
//! `grammar_program`).

use inferlet::inference::ForwardPass;
use inferlet::working_set::KvWorkingSet;
use inferlet::{chat, geometry, model, prefill, sampler, Constrain, JsonSchema, Result, Schema};
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

    let vocab = model::output_vocab_size();
    // Attempts sample (TopP → masked-sampled grammar); reflections are greedy
    // (Argmax → masked grammar). Both grammars ride one program each (the mask
    // supplies the schema per step).
    let g_sampled = sampler::grammar_program_sampled(vocab)?;
    let g_greedy = sampler::grammar_program(vocab)?;
    let stop = chat::stop_tokens();

    let mut attempt_root = Ctx::new();
    attempt_root.prefill(&chat::system(SYSTEM_PROMPT))?;

    let mut reflection_root = Ctx::new();
    reflection_root.prefill(&chat::system(
        "You are a precise scheduling critic. Given a failed schedule and deterministic \
     violations, produce one short actionable lesson for the next attempt. \
     Do not invent a full new schedule. Do not suggest moving a job past its deadline. \
     Mention the exact violated constraint and the current bad interval. \
     If giving a candidate repair, it must preserve all fixed constraints: \
     A=60, B=60, C=30, D=30; recovery=30; recovery must be after A and before B; \
     D must end by 660; no overlap. Prefer saying what must not happen over giving \
     a complete new schedule.",
    ))?;

    let output = match input.mode.to_lowercase().as_str() {
        "baseline" => {
            let result = run_mode(&input, &attempt_root, &reflection_root, &g_sampled, &g_greedy, &stop, false).await?;
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
        }
        "reflexion" | "" => {
            let result = run_mode(&input, &attempt_root, &reflection_root, &g_sampled, &g_greedy, &stop, true).await?;
            serde_json::to_string_pretty(&result).map_err(|e| e.to_string())?
        }
        "both" => {
            let baseline = run_mode(&input, &attempt_root, &reflection_root, &g_sampled, &g_greedy, &stop, false).await?;
            let reflexion = run_mode(&input, &attempt_root, &reflection_root, &g_sampled, &g_greedy, &stop, true).await?;
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

/// In-inferlet decode context (raw-WIT keep-core, no `Context` facade): a KV
/// working set + cursor, forkable with COW-shared prefix. The grammar decode is
/// sequential (no run-ahead carrier: the next mask depends on this token).
struct Ctx {
    kv: KvWorkingSet,
    seq_len: u32,
    fresh: bool,
}

impl Ctx {
    fn new() -> Self {
        Self { kv: KvWorkingSet::new(), seq_len: 0, fresh: true }
    }

    fn fork(&self) -> Result<Self> {
        Ok(Self {
            kv: self.kv.fork().map_err(|e| format!("fork: {e}"))?,
            seq_len: self.seq_len,
            fresh: true,
        })
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<()> {
        prefill::tokens(&self.kv, &mut self.seq_len, tokens)
    }

    /// One SEQUENTIAL grammar fire: geometry + input + the masked grammar sampler
    /// (`g` = greedy or sampled) + execute, advancing the cursor. Fires
    /// `fresh_generate` once.
    fn grammar_fire(
        &mut self,
        g: &sampler::LoweredGrammar,
        tokens: &[u32],
        packed_mask: &[u32],
    ) -> Result<ForwardPass> {
        let n = tokens.len() as u32;
        let pass = ForwardPass::new();
        if self.fresh {
            pass.fresh_generate();
            self.fresh = false;
        }
        let geom = geometry::ensure_pages(
            &self.kv,
            geometry::kv_write_geometry(self.seq_len, n, self.kv.page_size()),
        )?;
        geometry::attach_kv_write(&pass, &self.kv, &geom);
        let positions: Vec<u32> = (self.seq_len..self.seq_len + n).collect();
        pass.input_tokens(tokens, &positions);
        let decode_pos = self.seq_len + n - 1;
        pass.sampler(&g.program, g.bindings(decode_pos, packed_mask)?);
        pass.execute();
        self.seq_len += n;
        Ok(pass)
    }

    /// Prefill `tail`, then sequentially grammar-decode under `matcher` until it
    /// TERMINATES, a stop token fires, or `max_tokens` is hit. Returns the decoded
    /// text. This fork is dropped after the parse, so no residual is preserved.
    async fn grammar_decode(
        &mut self,
        g: &sampler::LoweredGrammar,
        mut matcher: inferlet::GrammarConstraint,
        tail: &[u32],
        max_tokens: usize,
        stop: &[u32],
    ) -> Result<String> {
        let mut decoder = chat::Decoder::new();
        let mut text = String::new();
        let mut pending = tail.to_vec();
        if pending.is_empty() {
            pending = vec![0u32];
        }
        let mut generated = 0usize;

        loop {
            let m = matcher.mask();
            let packed: Vec<u32> = if m.is_empty() {
                vec![u32::MAX; g.mask_words]
            } else {
                m
            };

            let pass = self.grammar_fire(g, &pending, &packed)?;
            let token = read_token(pass).await?;

            if stop.contains(&token) {
                return Ok(text);
            }

            generated += 1;
            match decoder.feed(&[token])? {
                chat::Event::Delta(sd) => text.push_str(&sd),
                chat::Event::Done(sd) => return Ok(sd),
                _ => {}
            }
            matcher.advance(&[token]);
            pending = vec![token];

            if matcher.is_terminated() || generated >= max_tokens {
                return Ok(text);
            }
        }
    }
}

/// Read the sampled token off a finalized pass's single-`Token` output tensor.
async fn read_token(pass: ForwardPass) -> Result<u32> {
    let out = pass.output().await.map_err(|e| format!("output: {e}"))?;
    let bytes = out.read().map_err(|e| format!("tensor read: {e:?}"))?;
    Ok(if bytes.len() >= 4 {
        i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u32
    } else {
        0
    })
}

#[allow(clippy::too_many_arguments)]
async fn run_mode(
    input: &Input,
    attempt_root: &Ctx,
    reflection_root: &Ctx,
    g_sampled: &sampler::LoweredGrammar,
    g_greedy: &sampler::LoweredGrammar,
    stop: &[u32],
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
        let mut tail = chat::user(&attempt_prompt);
        tail.extend(chat::cue());

        let matcher = JsonSchema(SCHEDULE_SCHEMA).build_constraint()?;
        latest = ctx
            .grammar_decode(g_sampled, matcher, &tail, input.max_tokens, stop)
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
                        g_greedy,
                        stop,
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
                g_greedy,
                stop,
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
    root: &Ctx,
    g_greedy: &sampler::LoweredGrammar,
    stop: &[u32],
    schedule: &str,
    evaluation: &Evaluation,
    max_tokens: usize,
    memory: &mut Vec<String>,
) -> Result<()> {
    let violations = serde_json::to_string(evaluation).map_err(|e| e.to_string())?;
    let mut ctx = root.fork()?;
    let mut tail = chat::user(&format!(
        "Failed schedule:\n{schedule}\n\nDeterministic evaluation:\n{violations}"
    ));
    tail.extend(chat::cue());

    let matcher = JsonSchema(REFLECTION_SCHEMA).build_constraint()?;
    let raw = ctx
        .grammar_decode(g_greedy, matcher, &tail, max_tokens, stop)
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
