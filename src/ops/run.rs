use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use client::client::{Client, ProcessEvent};

use crate::ops::inferlet::{Artifact, load_artifact};

#[derive(Args, Debug)]
pub struct RunArgs {
    pub inferlet: Option<String>,

    #[arg(long, short = 'p')]
    pub path: Option<PathBuf>,

    #[arg(long = "out", short = 'o', value_name = "DIR")]
    pub out: Option<PathBuf>,

    #[arg(last = true, allow_hyphen_values = true)]
    pub arguments: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Target {
    /// A name to resolve against what is installed (or curated in this tree).
    Installed(String),
    Local {
        path: PathBuf,
    },
}

/// Whether a `pie run` positional names a file rather than an installed
/// program: it carries a path separator or an artifact extension.
fn looks_like_a_file(word: &str) -> bool {
    word.contains(['/', '\\'])
        || Path::new(word)
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| matches!(e, "wasm" | "py" | "js" | "mjs"))
}

pub fn target(inferlet: Option<&str>, path: Option<&Path>) -> Result<Target> {
    match (inferlet, path) {
        (None, None) => bail!(
            "name an inferlet to run, or a `.wasm`, `.py` or `.js` to run from a file. \
             `pie inferlet list` shows what is installed."
        ),
        (Some(inferlet), Some(path)) => bail!(
            "both an inferlet name ({inferlet:?}) and `--path {}` -- run one or \
             the other. Arguments for the inferlet go after `--`.",
            path.display()
        ),
        (Some(inferlet), None) if looks_like_a_file(inferlet) => {
            let path = Path::new(inferlet);
            if !path.exists() {
                bail!("no file at {}", path.display());
            }
            Ok(Target::Local {
                path: path.to_path_buf(),
            })
        }
        (Some(inferlet), None) => Ok(Target::Installed(inferlet.to_string())),
        (None, Some(path)) => {
            if !path.exists() {
                bail!("no file at {}", path.display());
            }
            Ok(Target::Local {
                path: path.to_path_buf(),
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Curated {
    pub root: PathBuf,
    pub dir: PathBuf,
    /// A component built from the directory's Rust source.
    pub wasm: Option<PathBuf>,
    /// The directory's script, when the inferlet is one (`main.py`, `index.js`).
    pub script: Option<PathBuf>,
    pub name: String,
}

impl Curated {
    pub fn build_hint(&self) -> String {
        format!(
            "cd {} && cargo build -p {} --release --target wasm32-wasip2",
            crate::ui::short_path(&self.root),
            self.name
        )
    }

    /// What to upload: the script if the inferlet is one, else the build.
    pub fn artifact(&self) -> Option<&PathBuf> {
        self.script.as_ref().or(self.wasm.as_ref())
    }
}

pub fn curated_roots() -> Vec<PathBuf> {
    let mut roots: Vec<PathBuf> = std::env::var("PIE_INFERLETS")
        .unwrap_or_default()
        .split(':')
        .filter(|entry| !entry.is_empty())
        .map(PathBuf::from)
        .filter(|root| root.is_dir())
        .collect();
    if let Ok(cwd) = std::env::current_dir() {
        for ancestor in cwd.ancestors() {
            let candidate = ancestor.join("examples");
            if candidate.join("Cargo.toml").is_file() {
                roots.push(candidate);
                break;
            }
        }
    }
    roots
}

pub fn curated(name: &str) -> Option<Curated> {
    curated_in(&curated_roots(), name)
}

pub fn curated_in(roots: &[PathBuf], name: &str) -> Option<Curated> {
    if name.is_empty() || name.contains(['/', '\\']) || name.starts_with('.') {
        return None;
    }
    let artifact = format!("{}.wasm", name.replace('-', "_"));
    roots.iter().find_map(|root| {
        let dir = root.join(name);
        let script = ["main.py", "index.js"]
            .iter()
            .map(|file| dir.join(file))
            .find(|path| path.is_file());
        if script.is_none() && !dir.join("Cargo.toml").is_file() {
            return None;
        }
        let built = std::env::var_os("CARGO_TARGET_DIR")
            .map_or_else(|| root.join("target"), PathBuf::from)
            .join("wasm32-wasip2");
        let wasm = ["release", "debug"]
            .iter()
            .map(|profile| built.join(profile).join(&artifact))
            .find(|path| path.is_file());
        Some(Curated {
            root: root.clone(),
            dir,
            wasm,
            script,
            name: name.to_string(),
        })
    })
}

pub fn arguments_to_input(arguments: &[String]) -> String {
    let mut object = serde_json::Map::new();
    let mut positional: Vec<serde_json::Value> = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        let Some(key) = flag_key(argument) else {
            positional.push(typed(argument));
            index += 1;
            continue;
        };
        match arguments.get(index + 1) {
            Some(next) if !is_flag(next) => {
                object.insert(key, typed(next));
                index += 2;
            }
            _ => {
                object.insert(key, serde_json::Value::Bool(true));
                index += 1;
            }
        }
    }
    if !positional.is_empty() {
        object.insert(
            "_positional".to_string(),
            serde_json::Value::Array(positional),
        );
    }
    serde_json::Value::Object(object).to_string()
}

fn is_flag(token: &str) -> bool {
    token.starts_with('-') && token.len() > 1 && token.parse::<f64>().is_err()
}

fn flag_key(token: &str) -> Option<String> {
    if !is_flag(token) {
        return None;
    }
    match token.strip_prefix("--") {
        Some(key) if !key.is_empty() => Some(key.replace('-', "_")),
        _ => match token.strip_prefix('-') {
            Some(key) if key.len() == 1 => Some(key.to_string()),
            _ => None,
        },
    }
}

fn typed(value: &str) -> serde_json::Value {
    if let Ok(number) = value.parse::<i64>() {
        return serde_json::Value::from(number);
    }
    if let Ok(number) = value.parse::<f64>()
        && number.is_finite()
    {
        return serde_json::Value::from(number);
    }
    match value {
        "true" => serde_json::Value::Bool(true),
        "false" => serde_json::Value::Bool(false),
        other => serde_json::Value::String(other.to_string()),
    }
}

/// What `pie run` will do once the target is settled: the program to
/// launch, and the artifact to upload first when it comes from a file.
pub struct Plan {
    pub program: String,
    pub upload: Option<(PathBuf, Artifact)>,
}

fn resolve(target: Target) -> Result<Plan> {
    let spec = match target {
        Target::Local { path } => return plan_local(&path),
        Target::Installed(spec) => spec,
    };

    let curated = curated(&spec);
    if let Some(found) = &curated
        && let Some(artifact) = found.artifact()
    {
        return plan_local(artifact);
    }

    match crate::ops::inferlet::resolve_installed(&spec) {
        Ok(program) => Ok(Plan {
            program: program.to_string(),
            upload: None,
        }),
        Err(error) => match curated {
            Some(found) => bail!(
                "{spec:?} is in this tree at {} but has not been built; `{}` builds it",
                crate::ui::short_path(&found.dir),
                found.build_hint()
            ),
            None => Err(error),
        },
    }
}

fn plan_local(path: &Path) -> Result<Plan> {
    let artifact = load_artifact(path, None)?;
    Ok(Plan {
        program: artifact.identity.name.to_string(),
        upload: Some((path.to_path_buf(), artifact)),
    })
}

pub async fn run(
    global: &bootstrap::GlobalArgs,
    args: RunArgs,
    diag: Option<&str>,
) -> Result<crate::ui::Answer> {
    let (cfg_path, origin) = bootstrap::cli_config_path(global);
    let content = std::fs::read_to_string(&cfg_path).with_context(|| {
        format!(
            "no config file at {} ({}); `pie config init` writes one",
            crate::ui::short_path(&cfg_path),
            origin.describe()
        )
    })?;

    let target = target(args.inferlet.as_deref(), args.path.as_deref())?;

    let (controller, gateway, mut worker) = crate::derive::derive_standalone(&content)?;
    if let Some(words) = diag {
        worker.state_diagnostics(words)?;
    }
    let model = worker.model.name.clone();

    let plan = resolve(target)?;

    match &plan.upload {
        Some((path, _)) => println!(
            "Running {} on {model}\n  from {}",
            plan.program,
            crate::ui::short_path(path)
        ),
        None => println!("Running {} on {model}", plan.program),
    }
    println!();

    let pie = crate::compose::run_standalone(controller, gateway, worker)
        .await
        .context("boot the engine")?;
    let outcome = drive(
        &pie.listen_addr.to_string(),
        &plan,
        &args.arguments,
        args.out.as_deref(),
    )
    .await;
    fire_probes().await;
    pie.shutdown().await;
    Ok(crate::ui::Answer::quiet().with_code(outcome?))
}

#[cfg(feature = "profile-fire")]
async fn fire_probes() {
    let s = runtime::scheduler::get_stats().await;
    let fires = s.total_batches.max(1);
    let per = |sum: u64| sum as f64 / fires as f64 / 1000.0;
    println!();
    println!("fires {fires}  tokens {}", s.total_tokens_processed);
    println!(
        "  inter-fire        {:7.3} ms   = execute {:.3} + post-dispatch-to-fire {:.3}",
        per(s.fire.inter_fire_us_sum),
        per(s.fire.execute.total_us_sum),
        per(s.fire.post_dispatch_to_fire_us_sum),
    );
    println!(
        "    execute         {:7.3} ms   batch-build {:.3}  engine-fire {:.3}",
        per(s.fire.execute.total_us_sum),
        per(s.fire.execute.batch_build_us_sum),
        per(s.fire.execute.engine_fire_us_sum),
    );
    println!(
        "    accumulate      {:7.3} ms   fire-prepare {:.3}  recv-block {:.3}",
        per(s.fire.accumulate.accum_loop_us_sum),
        per(s.fire.pre_dispatch.fire_prepare_us_sum),
        per(s.fire.recv_block_wait_us_sum),
    );
    let submits = s.host_submit.submits.max(1);
    let sub = |sum: u64| sum as f64 / submits as f64 / 1000.0;
    println!(
        "  guest submit      {:7.3} ms   over {submits} submits",
        sub(s.host_submit.total_us),
    );
    println!(
        "    drain-settled {:.3}  geometry {:.3}  kv-prepare {:.3}  scheduler-submit {:.3}  shadow {:.3}  validate {:.3}",
        sub(s.host_submit.drain_settled_us),
        sub(s.host_submit.geometry_us),
        sub(s.host_submit.kv_prepare_us),
        sub(s.host_submit.scheduler_submit_us),
        sub(s.host_submit.shadow_advance_us),
        sub(s.host_submit.validate_frame_us),
    );
}

#[cfg(not(feature = "profile-fire"))]
async fn fire_probes() {}

fn write_received(
    dir: &Path,
    file: &client::client::ReceivedFile,
    index: usize,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let name = file.file_name(&format!("file-{index:04}.bin"));
    let path = dir.join(&name);
    if path.parent() != Some(dir) {
        bail!(
            "the inferlet asked for {name:?}, which is not a name in {}",
            dir.display()
        );
    }
    std::fs::write(&path, &file.data).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

async fn drive(
    addr: &str,
    plan: &Plan,
    arguments: &[String],
    out_dir: Option<&Path>,
) -> Result<std::process::ExitCode> {
    let program = &plan.program;
    let client = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "pie-run")
        .await
        .context("connect to the engine this command just booted")?;
    client
        .authenticate("pie-run", &None)
        .await
        .context("authenticate")?;

    if let Some((path, artifact)) = &plan.upload {
        let installed = client
            .add_program_bytes(
                &artifact.bytes,
                &artifact.file,
                Some(&artifact.identity.name.version),
                true,
            )
            .await
            .with_context(|| format!("uploading {}", path.display()))?;
        if installed != *program {
            bail!(
                "{} was installed as {installed}, not {program}",
                path.display()
            );
        }
    }

    let mut process = client
        .launch_process(program.to_string(), arguments_to_input(arguments), true)
        .await
        .with_context(|| format!("launching {program}"))?;

    let mut shown = String::new();
    let mut files_written = 0usize;
    let code = loop {
        match process.recv().await.context("reading process output")? {
            ProcessEvent::Stdout(text) => {
                shown.push_str(&text);
                print!("{text}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            ProcessEvent::Stderr(text) => {
                eprint!("{text}");
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
            ProcessEvent::Message(text) => {
                shown.push_str(&text);
                println!("{text}");
            }
            ProcessEvent::File(file) => match out_dir {
                Some(dir) => match write_received(dir, &file, files_written) {
                    Ok(path) => {
                        files_written += 1;
                        eprintln!("[wrote {} ({} bytes)]", path.display(), file.data.len());
                    }
                    Err(error) => eprintln!("[could not write a received file: {error:#}]"),
                },
                None => {
                    eprintln!(
                        "[received {} ({} bytes) and dropped it; `-o .` writes it here]",
                        file.file_name(&format!("an unnamed file-{files_written:04}.bin")),
                        file.data.len(),
                    );
                    files_written += 1;
                }
            },
            ProcessEvent::Return(value) => {
                let trimmed = value.trim();
                if !trimmed.is_empty() && !shown.contains(trimmed) {
                    println!("{value}");
                }
                break std::process::ExitCode::SUCCESS;
            }
            ProcessEvent::Error(message) => {
                eprintln!("{message}");
                break std::process::ExitCode::FAILURE;
            }
        }
    };

    drop(process);
    client
        .close()
        .await
        .context("closing the client connection")?;
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(arguments: &[&str]) -> serde_json::Value {
        let owned: Vec<String> = arguments.iter().map(|s| s.to_string()).collect();
        serde_json::from_str(&arguments_to_input(&owned)).unwrap()
    }

    fn tree(dir: &Path, name: &str, profile: Option<&str>) {
        std::fs::create_dir_all(dir.join(name)).unwrap();
        std::fs::write(
            dir.join(name).join("Cargo.toml"),
            format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\n"),
        )
        .unwrap();
        if let Some(profile) = profile {
            let built = dir.join("target").join("wasm32-wasip2").join(profile);
            std::fs::create_dir_all(&built).unwrap();
            std::fs::write(
                built.join(format!("{}.wasm", name.replace('-', "_"))),
                b"\0asm",
            )
            .unwrap();
        }
    }

    #[test]
    fn run_every_case() {
        a_curated_name_resolves_to_the_build_of_its_crate();
        an_unbuilt_curated_directory_is_found_without_a_wasm();
        a_script_twin_is_its_source_and_needs_no_build();
        release_outranks_debug();
        a_name_that_is_a_path_is_not_a_curated_name();
        a_positional_with_an_artifact_extension_is_a_file();
        the_documented_invocation_produces_the_documented_input();
    }

    fn a_script_twin_is_its_source_and_needs_no_build() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("text-completion-py")).unwrap();
        std::fs::write(dir.path().join("text-completion-py").join("main.py"), "").unwrap();
        let found = curated_in(&[dir.path().to_path_buf()], "text-completion-py").unwrap();
        assert!(found.wasm.is_none());
        assert_eq!(
            found.artifact().unwrap(),
            &dir.path().join("text-completion-py").join("main.py")
        );
    }

    fn a_positional_with_an_artifact_extension_is_a_file() {
        assert!(looks_like_a_file("main.py"));
        assert!(looks_like_a_file("build/x.wasm"));
        assert!(looks_like_a_file("./text-completion"));
        assert!(!looks_like_a_file("text-completion"));
        assert!(!looks_like_a_file("text-completion@0.1.0"));
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("probe.py");
        std::fs::write(&script, "").unwrap();
        let script_str = script.to_str().unwrap();
        assert_eq!(
            target(Some(script_str), None).unwrap(),
            Target::Local {
                path: script.clone(),
            }
        );
        assert_eq!(
            target(Some("probe"), None).unwrap(),
            Target::Installed("probe".to_string())
        );
    }

    fn a_curated_name_resolves_to_the_build_of_its_crate() {
        let dir = tempfile::tempdir().unwrap();
        tree(dir.path(), "text-to-image", Some("release"));
        let found = curated_in(&[dir.path().to_path_buf()], "text-to-image").unwrap();
        assert_eq!(
            found.wasm.unwrap(),
            dir.path()
                .join("target/wasm32-wasip2/release/text_to_image.wasm")
        );
    }

    fn an_unbuilt_curated_directory_is_found_without_a_wasm() {
        let dir = tempfile::tempdir().unwrap();
        tree(dir.path(), "text-to-image", None);
        let found = curated_in(&[dir.path().to_path_buf()], "text-to-image").unwrap();
        assert!(found.wasm.is_none());
        assert!(found.build_hint().contains("-p text-to-image"));
    }

    fn release_outranks_debug() {
        let dir = tempfile::tempdir().unwrap();
        tree(dir.path(), "frames-probe", Some("debug"));
        tree(dir.path(), "frames-probe", Some("release"));
        let found = curated_in(&[dir.path().to_path_buf()], "frames-probe").unwrap();
        assert!(found.wasm.unwrap().to_string_lossy().contains("/release/"));
    }

    fn a_name_that_is_a_path_is_not_a_curated_name() {
        let dir = tempfile::tempdir().unwrap();
        tree(dir.path(), "text-to-image", Some("release"));
        let roots = [dir.path().to_path_buf()];
        assert!(curated_in(&roots, "../text-to-image").is_none());
        assert!(curated_in(&roots, ".hidden").is_none());
        assert!(curated_in(&roots, "").is_none());
    }

    fn the_documented_invocation_produces_the_documented_input() {
        assert_eq!(
            input(&["--prompt", "The capital of France is"]),
            serde_json::json!({"prompt": "The capital of France is"})
        );
    }
}
