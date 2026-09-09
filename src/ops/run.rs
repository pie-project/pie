use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use client::client::{Client, ProcessEvent};

#[derive(Args, Debug)]
pub struct RunArgs {
    pub inferlet: Option<String>,

    #[arg(long, short = 'p')]
    pub path: Option<PathBuf>,

    #[arg(long, short = 'm')]
    pub manifest: Option<PathBuf>,

    #[arg(long = "out", short = 'o', value_name = "DIR")]
    pub out: Option<PathBuf>,

    #[arg(last = true, allow_hyphen_values = true)]
    pub arguments: Vec<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Target {
    Registry(String),
    Local {
        wasm: PathBuf,
        manifest: PathBuf,
        name: String,
    },
}

pub fn target(
    inferlet: Option<&str>,
    path: Option<&Path>,
    manifest: Option<&Path>,
    read_manifest: impl FnOnce(&Path) -> Result<String>,
) -> Result<Target> {
    match (inferlet, path) {
        (None, None) => bail!(
            "name an inferlet to run, or point `--path` at a local `.wasm`. \
             `pie inferlet list` shows what is already here."
        ),
        (Some(inferlet), Some(path)) => bail!(
            "both an inferlet name ({inferlet:?}) and `--path {}` -- run one or \
             the other. Arguments for the inferlet go after `--`.",
            path.display()
        ),
        (Some(inferlet), None) => {
            if manifest.is_some() {
                bail!(
                    "`--manifest` describes a local build, so it only means something with `--path`"
                );
            }
            Ok(Target::Registry(inferlet.to_string()))
        }
        (None, Some(path)) => {
            if !path.exists() {
                bail!("no file at {}", path.display());
            }
            let manifest = manifest.ok_or_else(|| {
                anyhow!(
                    "`--path` needs `--manifest`: the manifest is where the \
                     program's name and version come from, and a `.wasm` on its \
                     own does not carry them"
                )
            })?;
            if !manifest.exists() {
                bail!("no manifest at {}", manifest.display());
            }
            let name = manifest_program_name(&read_manifest(manifest)?)
                .with_context(|| format!("reading {}", manifest.display()))?;
            Ok(Target::Local {
                wasm: path.to_path_buf(),
                manifest: manifest.to_path_buf(),
                name,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Curated {
    pub root: PathBuf,
    pub manifest: PathBuf,
    pub wasm: Option<PathBuf>,
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
            let candidate = ancestor.join("tests").join("inferlets");
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
        let manifest = root.join(name).join("Pie.toml");
        if !manifest.is_file() {
            return None;
        }
        let built = root.join("target").join("wasm32-wasip2");
        let wasm = ["release", "debug"]
            .iter()
            .map(|profile| built.join(profile).join(&artifact))
            .find(|path| path.is_file());
        Some(Curated {
            root: root.clone(),
            manifest,
            wasm,
            name: name.to_string(),
        })
    })
}

pub fn manifest_program_name(content: &str) -> Result<String> {
    let manifest: toml::Value = toml::from_str(content).context("parse the manifest")?;
    let package = manifest
        .get("package")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| anyhow!("the manifest has no [package] table"))?;
    let field = |key: &str| {
        package
            .get(key)
            .and_then(toml::Value::as_str)
            .ok_or_else(|| anyhow!("the manifest's [package] has no {key}"))
    };
    Ok(format!("{}@{}", field("name")?, field("version")?))
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

async fn resolve(target: Target, registry: &str) -> Result<(Target, String)> {
    let spec = match target {
        Target::Local { ref name, .. } => {
            let name = name.clone();
            return Ok((target, name));
        }
        Target::Registry(spec) => spec,
    };

    if spec.contains('@') {
        let program = crate::ops::inferlet::resolve_inferlet_id(&spec, registry)
            .await
            .with_context(|| format!("resolving {spec:?}"))?
            .to_string();
        return Ok((Target::Registry(spec), program));
    }

    let curated = curated(&spec);
    if let Some(found) = &curated
        && let Some(wasm) = &found.wasm
    {
        let manifest = std::fs::read_to_string(&found.manifest)
            .with_context(|| format!("reading {}", found.manifest.display()))?;
        let name = manifest_program_name(&manifest)
            .with_context(|| format!("reading {}", found.manifest.display()))?;
        return Ok((
            Target::Local {
                wasm: wasm.clone(),
                manifest: found.manifest.clone(),
                name: name.clone(),
            },
            name,
        ));
    }

    if let Some(program) = crate::ops::inferlet::cached_version(&spec) {
        let program = program.to_string();
        return Ok((Target::Registry(spec), program));
    }

    match crate::ops::inferlet::resolve_inferlet_id(&spec, registry).await {
        Ok(program) => Ok((Target::Registry(spec), program.to_string())),
        Err(error) => match curated {
            Some(found) => Err(error.context(format!(
                "{spec:?} is in this tree at {} but has not been built; \
                 `{}` builds it",
                crate::ui::short_path(&found.manifest),
                found.build_hint()
            ))),
            None => Err(error.context(format!("resolving {spec:?}"))),
        },
    }
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

    let target = target(
        args.inferlet.as_deref(),
        args.path.as_deref(),
        args.manifest.as_deref(),
        |path| std::fs::read_to_string(path).map_err(Into::into),
    )?;

    let (controller, gateway, mut worker) = crate::derive::derive_standalone(&content)?;
    if let Some(words) = diag {
        worker.state_diagnostics(words)?;
    }
    let registry = worker.server.registry.clone();
    let model = worker.model.name.clone();

    let (target, program) = resolve(target, &registry).await?;

    match &target {
        Target::Local { wasm, .. } => println!(
            "Running {program} on {model}\n  from {}",
            crate::ui::short_path(wasm)
        ),
        Target::Registry(_) => println!("Running {program} on {model}"),
    }
    println!();

    let pie = crate::compose::run_standalone(controller, gateway, worker)
        .await
        .context("boot the engine")?;
    let outcome = drive(
        &pie.listen_addr.to_string(),
        &target,
        &program,
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
    target: &Target,
    program: &str,
    arguments: &[String],
    out_dir: Option<&Path>,
) -> Result<std::process::ExitCode> {
    let client = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "pie-run")
        .await
        .context("connect to the engine this command just booted")?;
    client
        .authenticate("pie-run", &None)
        .await
        .context("authenticate")?;

    if let Target::Local { wasm, manifest, .. } = target {
        client
            .add_program(wasm, manifest, true)
            .await
            .with_context(|| format!("uploading {}", wasm.display()))?;
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
            dir.join(name).join("Pie.toml"),
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
        a_curated_name_resolves_to_the_build_beside_its_manifest();
        an_unbuilt_curated_directory_is_found_without_a_wasm();
        release_outranks_debug();
        a_name_that_is_a_path_is_not_a_curated_name();
        the_documented_invocation_produces_the_documented_input();
    }

    fn a_curated_name_resolves_to_the_build_beside_its_manifest() {
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
