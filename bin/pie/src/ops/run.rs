//! `pie run` — one inferlet, one engine, one exit.
//!
//! Restored rather than invented. It was `pie/src/pie_cli/commands/run.py`
//! until `2fd8dc09d` retired the Python CLI and shipped the Rust binary; that
//! commit describes the restructuring at length and never mentions dropping
//! this, so it went with the package rather than by decision. The Rust `bin/pie`
//! was then built without it from its first commit.
//!
//! Four months of documentation kept describing it, which is what makes the
//! contract here not a fresh design: `website/docs/guide/setup.mdx` teaches it
//! as the first thing a new user runs, `dev-env.mdx` documents `--path` +
//! `--manifest` for a local build, and `bakery` prints `pie run <name>` as the
//! next step after a successful publish. Those pages are the specification, and
//! the behaviour below matches them.
//!
//! **One-shot, always.** The engine is booted here, holds the device for the
//! length of the run, and shuts down after. That was the Python command's
//! behaviour and it is what the docs promise ("boots a one-shot engine, runs
//! the inferlet, prints its output, and exits"). Attaching to an already
//! serving pie would be a different command with a different failure mode --
//! notably that the model it runs against would be whatever that server was
//! started with, rather than what this config says.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use pie_client::client::{Client, ProcessEvent};

#[derive(Args, Debug)]
pub struct RunArgs {
    /// The inferlet to run, e.g. `chat-completion` or `chat-completion@0.1.0`.
    /// A bare name resolves to the registry's latest, the same way `pie
    /// inferlet download` resolves one. Omit when using `--path`.
    pub inferlet: Option<String>,

    /// Run a local `.wasm` build instead of a published inferlet. Requires
    /// `--manifest`, which is where its name and version come from.
    #[arg(long, short = 'p')]
    pub path: Option<PathBuf>,

    /// The `Pie.toml` beside a local build. Only meaningful with `--path`.
    #[arg(long, short = 'm')]
    pub manifest: Option<PathBuf>,

    /// Arguments for the inferlet itself, after `--`. `--prompt hi` arrives as
    /// `{"prompt": "hi"}`; a bare `--stream` arrives as `{"stream": true}`.
    // `last` rather than `trailing_var_arg`: the separator is then required, so
    // an inferlet's flag can never be mistaken for one of pie's own. `--path`
    // is pie's before the `--` and the inferlet's after it, and there is no
    // position where it is ambiguous. Kept out of the doc comment because it is
    // a note to whoever edits this, not to whoever runs it.
    #[arg(last = true, allow_hyphen_values = true)]
    pub arguments: Vec<String>,
}

/// What to launch, and whether it has to be uploaded first.
#[derive(Debug, PartialEq, Eq)]
pub enum Target {
    /// A published inferlet, resolved against the registry.
    Registry(String),
    /// A local build: upload it under the name its manifest declares, then run
    /// that. `force` is always true — a local `.wasm` is the thing being
    /// iterated on, so re-running after a rebuild has to replace what the last
    /// run uploaded rather than refuse.
    Local {
        wasm: PathBuf,
        manifest: PathBuf,
        name: String,
    },
}

/// Decide what a person meant, and refuse the combinations that mean nothing.
///
/// `--path` with a positional argument is not an error the way it looks: the
/// Python command treated the stray word as the first inferlet argument,
/// because `pie run --path ./x.wasm foo` reads as "run this, with foo". That
/// only worked because arguments were a second positional. Here they are behind
/// `--`, so the word is unambiguous -- and unambiguous nonsense is worth saying
/// out loud rather than silently absorbing.
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
                bail!("`--manifest` describes a local build, so it only means something with `--path`");
            }
            Ok(Target::Registry(inferlet.to_string()))
        }
        (None, Some(path)) => {
            if !path.exists() {
                bail!("no file at {}", path.display());
            }
            // Required rather than guessed. The manifest is not decoration:
            // it carries the name and version the program is stored under, and
            // the engine has no other way to learn them from a bare `.wasm`.
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

/// `name@version` from a `Pie.toml`'s `[package]`.
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

/// Turn what followed `--` into the JSON object an inferlet receives.
///
/// `--prompt hi` is `{"prompt": "hi"}`, a bare `--stream` is `{"stream":
/// true}`, and anything left over lands in `_positional`. Values are typed by
/// what they look like -- `4` is a number, `true` is a boolean -- which is the
/// rule the Python command used and therefore the rule every inferlet in the
/// wild was written against.
///
/// Hyphens become underscores, because `--max-tokens` is how a flag is spelled
/// and `max_tokens` is how a field is.
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
        // The next word is this flag's value unless it is itself a flag --
        // which is what makes `--stream --prompt hi` two arguments and not one.
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

/// Whether a token introduces a flag rather than being a value.
///
/// The leading `-` is not enough on its own, because a negative number wears
/// one. `--seed -1` read `-1` as a flag, so the seed became `true` and a key
/// named `1` appeared beside it -- two wrong arguments out of one correct
/// invocation, and no complaint from anything. Faithfully ported from the
/// Python command, which had the same rule and the same bug.
///
/// A lone `-` stays a value: it is the filename convention for stdin, and
/// nothing spells a flag that way.
fn is_flag(token: &str) -> bool {
    token.starts_with('-') && token.len() > 1 && token.parse::<f64>().is_err()
}

/// The field name a flag token carries, or `None` if it is not a flag.
fn flag_key(token: &str) -> Option<String> {
    if !is_flag(token) {
        return None;
    }
    match token.strip_prefix("--") {
        // `--max-tokens` is how a flag is spelled, `max_tokens` is how a field
        // is.
        Some(key) if !key.is_empty() => Some(key.replace('-', "_")),
        // `-k value`. Longer single-dash clusters (`-abc`) are not a spelling
        // this ever supported, so they stay whole and land in `_positional`
        // rather than being invented into three flags.
        _ => match token.strip_prefix('-') {
            Some(key) if key.len() == 1 => Some(key.to_string()),
            _ => None,
        },
    }
}

/// Integer, then float, then boolean, then string.
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

/// Boot, run, print, exit.
pub async fn run(global: &startup::GlobalArgs, args: RunArgs) -> Result<crate::ui::Answer> {
    let (cfg_path, origin) = startup::cli_config_path(global);
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

    let (controller, gateway, worker) = crate::derive::derive_standalone(&content)?;
    let registry = worker.server.registry.clone();
    let model = worker.model.name.clone();

    // Resolved before the boot, so a name nobody can find fails in a second
    // rather than after the weights are on the device.
    let program = match &target {
        // Cache first. A bare name that is already on disk needs no network to
        // resolve, and the registry does not serve everything `pie inferlet
        // list` shows -- so asking it first turned `pie run <a local program>`
        // into a 404 for something sitting right there.
        Target::Registry(inferlet) if !inferlet.contains('@') => {
            match crate::ops::inferlet::cached_version(inferlet) {
                Some(program) => program.to_string(),
                None => crate::ops::inferlet::resolve_inferlet_id(inferlet, &registry)
                    .await
                    .with_context(|| format!("resolving {inferlet:?}"))?
                    .to_string(),
            }
        }
        Target::Registry(inferlet) => {
            crate::ops::inferlet::resolve_inferlet_id(inferlet, &registry)
                .await
                .with_context(|| format!("resolving {inferlet:?}"))?
                .to_string()
        }
        Target::Local { name, .. } => name.clone(),
    };

    println!("Running {program} on {model}");
    println!();

    // No guess in the context. The first version offered "is something already
    // serving on this port?", which is one of several reasons a boot fails and
    // was the wrong one every time it was seen -- a missing model artifact
    // reads as a port conflict if the top line says so, and the top line is
    // where a reader looks. The chain underneath already names the real cause.
    let pie = crate::compose::run_standalone(controller, gateway, worker)
        .await
        .context("boot the engine")?;
    let outcome = drive(&pie.listen_addr.to_string(), &target, &program, &args.arguments).await;
    pie.shutdown().await;
    // Quiet, because the inferlet's own output already went to stdout as it
    // was produced -- there is nothing left to render. The status is the
    // inferlet's, not this command's: `pie run` succeeded at running something
    // that failed.
    Ok(crate::ui::Answer::quiet().with_code(outcome?))
}

/// Connect, upload if local, launch, and mirror everything the process says.
async fn drive(
    addr: &str,
    target: &Target,
    program: &str,
    arguments: &[String],
) -> Result<std::process::ExitCode> {
    let client = Client::connect_with_identity(&format!("ws://{addr}/v1/ws"), "pie-run")
        .await
        .context("connect to the engine this command just booted")?;
    client
        .authenticate("pie-run", &None)
        .await
        .context("authenticate")?;

    if let Target::Local { wasm, manifest, .. } = target {
        // Always overwriting: a local `.wasm` is what is being iterated on, so
        // the second run after a rebuild must replace the first one's upload.
        client
            .add_program(wasm, manifest, true)
            .await
            .with_context(|| format!("uploading {}", wasm.display()))?;
    }

    let mut process = client
        .launch_process(program.to_string(), arguments_to_input(arguments), true)
        .await
        .with_context(|| format!("launching {program}"))?;

    // Streamed, not collected. An inferlet that prints as it decodes should
    // look like it is printing as it decodes -- collecting until `Return` would
    // turn every run into a silence followed by a wall of text.
    // What the inferlet has already shown, kept so the return value can be
    // recognised as a repeat of it rather than assumed to be one.
    let mut shown = String::new();
    let code = loop {
        match process.recv().await.context("reading process output")? {
            ProcessEvent::Stdout(text) => {
                shown.push_str(&text);
                print!("{text}");
                let _ = std::io::Write::flush(&mut std::io::stdout());
            }
            // stderr to stderr, so `pie run ... > out.txt` separates what the
            // inferlet produced from what it said about producing it.
            ProcessEvent::Stderr(text) => {
                eprint!("{text}");
                let _ = std::io::Write::flush(&mut std::io::stderr());
            }
            ProcessEvent::Message(text) => {
                shown.push_str(&text);
                println!("{text}");
            }
            ProcessEvent::File(bytes) => {
                eprintln!("[received a {} byte file]", bytes.len());
            }
            // Printed unless it is a repeat of what the reader already saw.
            //
            // `chat-completion` returns the completion it just streamed token
            // by token, so printing both showed the whole answer twice -- the
            // second copy arriving all at once, which reads as the model having
            // said it again. But "it streamed something, so skip the return"
            // is too blunt: an inferlet that streams progress and returns a
            // result would lose the result, and the first version of this hid
            // that behind a `debug!` nobody runs at. Comparing is what makes
            // the duplicate go away without taking anything else with it.
            ProcessEvent::Return(value) => {
                let trimmed = value.trim();
                if !trimmed.is_empty() && !shown.contains(trimmed) {
                    println!("{value}");
                }
                break std::process::ExitCode::SUCCESS;
            }
            // The inferlet failed, which is not this command failing: the
            // message is the inferlet's and belongs on stderr unadorned. The
            // exit code is what a script reads.
            ProcessEvent::Error(message) => {
                eprintln!("{message}");
                break std::process::ExitCode::FAILURE;
            }
        }
    };

    // Order matters twice here.
    //
    // The process goes first because it holds a clone of the client's shared
    // state, and `close` waits for the writer task, which only ends once every
    // sender is gone. Held, it waits forever -- the first version of this
    // printed the completion and then hung with the model still resident.
    //
    // The client goes before the engine because the reverse -- tearing down the
    // gateway under a still-open socket -- made the writer fail into "the
    // channel is closed and cannot accept new items", printed after a
    // successful run as though the run had failed.
    drop(process);
    client.close().await.context("closing the client connection")?;
    Ok(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(arguments: &[&str]) -> serde_json::Value {
        let owned: Vec<String> = arguments.iter().map(|s| s.to_string()).collect();
        serde_json::from_str(&arguments_to_input(&owned)).unwrap()
    }

    #[test]
    fn the_documented_invocation_produces_the_documented_input() {
        // `pie run chat-completion -- --prompt "The capital of France is"`,
        // straight out of setup.mdx. Every inferlet in the wild was written
        // against this shape, so it is a contract and not a preference.
        assert_eq!(
            input(&["--prompt", "The capital of France is"]),
            serde_json::json!({"prompt": "The capital of France is"})
        );
    }

    #[test]
    fn a_flag_without_a_value_is_true_and_does_not_eat_the_next_flag() {
        // The failure this prevents: `--stream` swallowing `--prompt` as its
        // value, which loses one argument and corrupts the other.
        assert_eq!(
            input(&["--stream", "--prompt", "hi"]),
            serde_json::json!({"stream": true, "prompt": "hi"})
        );
        assert_eq!(input(&["--stream"]), serde_json::json!({"stream": true}));
    }

    #[test]
    fn values_are_typed_by_what_they_look_like() {
        // The Python command's rule: int, then float, then bool, then string.
        // An inferlet reading `max_tokens` wants a number, not "128".
        assert_eq!(
            input(&["--max-tokens", "128", "--temp", "0.7", "--raw", "false"]),
            serde_json::json!({"max_tokens": 128, "temp": 0.7, "raw": false})
        );
        // And a version-looking string stays a string: `1.2.3` parses as
        // neither an integer nor a float.
        assert_eq!(
            input(&["--model", "1.2.3"]),
            serde_json::json!({"model": "1.2.3"})
        );
    }

    #[test]
    fn a_negative_number_is_a_value_and_not_a_flag() {
        // `--seed -1` produced `{"seed": true, "1": true}` -- two wrong
        // arguments out of one correct invocation, silently. The leading `-`
        // is not enough to call something a flag when it is a number.
        assert_eq!(
            input(&["--seed", "-1"]),
            serde_json::json!({"seed": -1})
        );
        assert_eq!(
            input(&["--temp", "-0.5", "--top-k", "40"]),
            serde_json::json!({"temp": -0.5, "top_k": 40})
        );
        // And a negative number standing alone is still a value.
        assert_eq!(
            input(&["-3"]),
            serde_json::json!({"_positional": [-3]})
        );
    }

    #[test]
    fn a_short_flag_still_reads_as_one() {
        // The negative-number rule must not swallow real short flags.
        assert_eq!(input(&["-k", "5"]), serde_json::json!({"k": 5}));
        assert_eq!(input(&["-v"]), serde_json::json!({"v": true}));
        // A lone `-` is the stdin convention, not a flag.
        assert_eq!(input(&["-"]), serde_json::json!({"_positional": ["-"]}));
        // A single-dash cluster is kept whole rather than invented into three
        // flags -- a spelling this never supported, so guessing at it would be
        // making something up on a person's behalf.
        assert_eq!(input(&["-abc"]), serde_json::json!({"_positional": ["-abc"]}));
    }

    #[test]
    fn a_flag_name_is_spelled_as_a_flag_and_arrives_as_a_field() {
        assert_eq!(
            input(&["--max-new-tokens", "4"]),
            serde_json::json!({"max_new_tokens": 4})
        );
    }

    #[test]
    fn leftovers_are_kept_rather_than_dropped() {
        // Silently discarding a word someone typed is worse than putting it
        // somewhere they can find it.
        assert_eq!(
            input(&["hello", "--n", "2", "world"]),
            serde_json::json!({"n": 2, "_positional": ["hello", "world"]})
        );
    }

    #[test]
    fn no_arguments_is_an_empty_object_rather_than_nothing() {
        // `{}` and `""` are not the same thing to an inferlet parsing its input.
        assert_eq!(arguments_to_input(&[]), "{}");
    }

    #[test]
    fn a_name_and_a_path_are_not_both_run() {
        let error = target(Some("chat-completion"), Some(Path::new("/x.wasm")), None, |_| {
            unreachable!()
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("one or the other"), "got: {error}");
    }

    #[test]
    fn nothing_to_run_says_how_to_find_something() {
        let error = target(None, None, None, |_| unreachable!())
            .unwrap_err()
            .to_string();
        assert!(error.contains("inferlet list"), "got: {error}");
    }

    #[test]
    fn a_local_build_takes_its_name_from_the_manifest() {
        // Not from the filename: the engine stores programs by name@version,
        // and `hello.wasm` says neither.
        let dir = std::env::temp_dir().join(format!("pie-run-target-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wasm = dir.join("hello.wasm");
        let manifest = dir.join("Pie.toml");
        std::fs::write(&wasm, b"\0asm").unwrap();
        std::fs::write(&manifest, "[package]\nname = \"hello\"\nversion = \"0.3.1\"\n").unwrap();

        let resolved = target(None, Some(&wasm), Some(&manifest), |p| {
            std::fs::read_to_string(p).map_err(Into::into)
        })
        .unwrap();
        assert_eq!(
            resolved,
            Target::Local {
                wasm: wasm.clone(),
                manifest: manifest.clone(),
                name: "hello@0.3.1".to_string(),
            }
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_path_without_a_manifest_says_why_it_needs_one() {
        let dir = std::env::temp_dir().join(format!("pie-run-nomanifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wasm = dir.join("hello.wasm");
        std::fs::write(&wasm, b"\0asm").unwrap();

        let error = target(None, Some(&wasm), None, |_| unreachable!())
            .unwrap_err()
            .to_string();
        assert!(error.contains("name and version"), "got: {error}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_file_is_reported_before_anything_boots() {
        // Booting the engine to then discover the file is absent would cost
        // the weight load for nothing.
        let error = target(None, Some(Path::new("/nope/absent.wasm")), None, |_| {
            unreachable!()
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("no file at"), "got: {error}");
    }

    #[test]
    fn a_manifest_without_a_package_says_so() {
        assert!(
            manifest_program_name("[other]\nname = \"x\"\n")
                .unwrap_err()
                .to_string()
                .contains("[package]")
        );
        assert!(
            manifest_program_name("[package]\nname = \"x\"\n")
                .unwrap_err()
                .to_string()
                .contains("version")
        );
    }
}
