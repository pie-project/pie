import { useState, type ReactNode } from 'react';
import clsx from 'clsx';
import Layout from '@theme/Layout';
import Link from '@docusaurus/Link';
import CodeBlock from '@theme/CodeBlock';
import ThemedImage from '@theme/ThemedImage';
import useBaseUrl from '@docusaurus/useBaseUrl';
import styles from './index.module.css';

const EXAMPLES = {
    chat: {
        label: 'Basic chat',
        code: `use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};

#[inferlet::main]
async fn main(prompt: String) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let mut ctx = Context::new(&model)?;

    ctx.system("You are a helpful assistant.")
       .user(&prompt)
       .cue();

    ctx.generate(Sampler::TopP { temperature: 0.6, p: 0.95 })
        .max_tokens(256)
        .collect_text()
        .await
}`,
    },
    parallelChat: {
        label: 'Parallel chat',
        code: `use futures::future;
use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};

#[inferlet::main]
async fn main() -> Result<(String, String)> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let mut ctx = Context::new(&model)?;
    ctx.system("You are a helpful assistant.");
    ctx.flush().await?;

    // Two forks share the system prompt's committed KV pages.
    let mut a = ctx.fork()?;
    let task_a = async move {
        a.user("Explain pulmonary embolism.").cue();
        a.generate(Sampler::Argmax).max_tokens(128).collect_text().await
    };

    let mut b = ctx.fork()?;
    let task_b = async move {
        b.user("Explain espresso making, ELI5.").cue();
        b.generate(Sampler::Argmax).max_tokens(128).collect_text().await
    };

    // Decode concurrently; the engine batches their forward passes.
    let (out_a, out_b) = future::join(task_a, task_b).await;
    Ok((out_a?, out_b?))
}`,
    },
    slidingWindow: {
        label: 'Sliding window',
        code: `use inferlet::{Context, Result, chat, model::Model, runtime, sample::Sampler};

#[inferlet::main]
async fn main(prompt: String) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let mut ctx = Context::new(&model)?;

    let mut pending: Vec<u32> = chat::system(&model, "You are a helpful assistant.");
    pending.extend(chat::user(&model, &prompt));
    pending.extend(chat::cue(&model));

    let window = 64u32;
    let mut tokens = Vec::new();
    for _ in 0..512 {
        let mut pass = ctx.forward();
        let total = pass.start_position() + pending.len() as u32;
        pass.input(&pending);

        // Sliding window: only attend to the last \`window\` positions.
        if total > window {
            let mask = vec![total - window, window];
            let masks: Vec<_> = (0..pending.len()).map(|_| mask.clone()).collect();
            pass.attention_mask(&masks);
        }

        let last = (pending.len() - 1) as u32;
        let h = pass.sample(&[last], Sampler::Argmax);
        let out = pass.execute().await?;
        let token = out.token(h).ok_or("no token")?;
        tokens.push(token);
        pending = vec![token];
    }

    Ok(model.tokenizer().decode(&tokens)?)
}`,
    },
    reranking: {
        label: 'Re-ranking',
        code: `use inferlet::{Context, Result, model::Model, runtime, sample::Logits};

// Score each candidate's perplexity under the model; return the most
// likely one. Useful for choosing among paraphrases, spelling variants,
// or best-of-N samples produced elsewhere.
#[inferlet::main]
async fn main(candidates: Vec<String>) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;

    let mut best: (f32, String) = (f32::INFINITY, String::new());
    for text in candidates {
        let mut ctx = Context::new(&model)?;
        let tokens = model.tokenizer().encode(&text)?;
        let mut pass = ctx.forward();
        pass.input(&tokens);

        let probes: Vec<_> = (0..tokens.len() as u32 - 1)
            .map(|i| pass.probe(i, Logits))
            .collect();
        let out = pass.execute().await?;

        let mut nll = 0.0f32;
        for (i, h) in probes.iter().enumerate() {
            let logits = decode_f32(&out.logits(*h).ok_or("no logits")?);
            nll -= log_prob(&logits, tokens[i + 1] as usize);
        }
        // elided: decode_f32 and log_prob (stable log-softmax)
        let avg = nll / probes.len() as f32;
        if avg < best.0 { best = (avg, text); }
    }
    Ok(best.1)
}`,
    },
    codeExec: {
        label: 'Code execution',
        code: `use boa_engine::{Context as JsCtx, Source};
use inferlet::{Context, Result, model::Model, runtime, sample::Sampler};
use schemars::JsonSchema;
use serde::Deserialize;
use std::io::Write;

#[derive(Deserialize, JsonSchema)]
struct Action {
    code: String,
}

#[inferlet::main]
async fn main(task: String) -> Result<String> {
    let model = Model::load(runtime::models().first().ok_or("no models")?)?;
    let mut ctx = Context::new(&model)?;
    ctx.system("Return JSON: { code: <one JavaScript expression> }.")
       .user(&task)
       .cue();

    // Constrained decoding forces the output to match Action's schema.
    let action: Action = ctx.generate(Sampler::Argmax)
        .max_tokens(256)
        .collect_json::<Action>()
        .await?;

    // Evaluate the model's code with Boa, in the inferlet's own process.
    let mut js = JsCtx::default();
    let result = js.eval(Source::from_bytes(&action.code))
        .map(|v| v.display().to_string())
        .unwrap_or_else(|e| format!("error: {e}"));

    // Append a JSONL record to the engine's preopened /scratch directory.
    let mut log = std::fs::OpenOptions::new()
        .append(true).create(true)
        .open("/scratch/runs.jsonl")
        .map_err(|e| e.to_string())?;
    writeln!(log, "{}", serde_json::json!({
        "task": task, "code": action.code, "result": result,
    })).map_err(|e| e.to_string())?;

    Ok(result)
}`,
    },
} as const;

type ExampleKey = keyof typeof EXAMPLES;

function Hero() {
    return (
        <header className={clsx('hero', styles.heroBanner)}>
            <div className="container">
                <h1 className={styles.heroTitle}>
                    A <span>programmable</span> LLM serving system.
                </h1>
                <p className={styles.heroSubtitle}>
                    High-performance inference engine where you write the loop. <br />
                    Forward passes are library calls in your inferlet.
                </p>
                <div className={styles.heroLinks}>
                    <Link className="button button--primary button--lg" to="/docs/overview/what-is-pie">
                        What is Pie?
                    </Link>
                    <Link className="button button--secondary button--lg" to="/docs/guide/install">
                        Get started
                    </Link>
                    <Link className="button button--secondary button--lg" to="https://github.com/pie-project/pie">
                        GitHub
                    </Link>
                </div>
            </div>
        </header>
    );
}

function ServingLoop() {
    return (
        <section className={styles.section}>
            <div className="container">
                <h2 className={styles.sectionTitle}>Serve programs, not prompts</h2>
                <p className={styles.sectionSubtitle}>
                    In existing serving systems, the inference workflow is baked into the engine.
                    In Pie, you write it.
                </p>
                <div className={styles.figuresGrid}>
                    <div className={styles.figureCard}>
                        <h3>Conventional serving systems</h3>
                        <ThemedImage
                            sources={{
                                light: useBaseUrl('/img/current-serving.svg'),
                                dark: useBaseUrl('/img/current-serving-dark.svg'),
                            }}
                            alt="A conventional serving system. Prompts from users enter the engine and pass through a fixed pipeline of batch, embed, prefill or decode, and sample stages, with one global autoregressive loop."
                        />
                        <p>
                            Every request runs through the same fixed pipeline.
                            Branching and tool calls live outside the engine.
                        </p>
                    </div>
                    <div className={styles.figureCard}>
                        <h3>Programmable serving system - Pie</h3>
                        <ThemedImage
                            sources={{
                                light: useBaseUrl('/img/programmable-serving.svg'),
                                dark: useBaseUrl('/img/programmable-serving-dark.svg'),
                            }}
                            alt="Pie's serving model. Each application runs as an inferlet inside the engine, calling into the model's KV cache and forward pass through a control layer."
                        />
                        <p>
                            Each inferlet runs its own workflow inside the engine.
                            It controls the KV cache, forward pass, and tool calls directly.
                        </p>
                    </div>
                </div>
            </div>
        </section>
    );
}

function Sample() {
    const [active, setActive] = useState<ExampleKey>('chat');
    return (
        <section className={clsx(styles.section, styles.codeSection)}>
            <div className="container">
                <h2 className={styles.sectionTitle}>More to optimize, more to customize</h2>
                <p className={styles.sectionSubtitle}>
                    Pie unlocks opportunities for optimization and custom model behavior. <br />
                    Each tab is an inferlet that customizes the inference loop in a different way.
                </p>
                <div className={styles.codeContainer}>
                    <div className={styles.codeTabs}>
                        {(Object.keys(EXAMPLES) as ExampleKey[]).map((key) => (
                            <button
                                key={key}
                                type="button"
                                className={clsx(styles.codeTab, active === key && styles.codeTabActive)}
                                onClick={() => setActive(key)}
                            >
                                {EXAMPLES[key].label}
                            </button>
                        ))}
                    </div>
                    <CodeBlock language="rust">{EXAMPLES[active].code}</CodeBlock>
                </div>
            </div>
        </section>
    );
}

export default function Home(): ReactNode {
    return (
        <Layout
            title="Programmable LLM serving"
            description="Pie is a programmable serving system for LLM inference."
        >
            <Hero />
            <main>
                <ServingLoop />
                <Sample />
            </main>
        </Layout>
    );
}
