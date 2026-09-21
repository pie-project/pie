use std::path::Path;
use std::process::Command;

use anyhow::Result;

use runtime::inferlet::program::{Language, Repository};

use crate::ui::{Mark, Palette};

type Checks = Vec<(String, String, Status)>;

#[derive(serde::Serialize)]
pub struct DoctorReport {
    ready: bool,
    passed: usize,
    warnings: usize,
    blocking: usize,
    sections: Vec<Section>,
}

#[derive(serde::Serialize)]
struct Section {
    section: &'static str,
    checks: Vec<Check>,
}

#[derive(serde::Serialize)]
struct Check {
    check: String,
    detail: String,
    status: Status,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
enum Status {
    Pass,
    Warn,
    #[serde(rename = "blocking")]
    Fail,
}

impl Status {
    fn mark(self) -> Mark {
        match self {
            Status::Pass => Mark::Did,
            Status::Warn => Mark::Warn,
            Status::Fail => Mark::Blocked,
        }
    }
}

impl crate::ui::Report for DoctorReport {
    fn render(&self, palette: &Palette) {
        println!("Pie standalone — environment doctor");
        for section in &self.sections {
            println!("\n{}", palette.bold(format!("[{}]", section.section)));
            let mut table =
                crate::ui::Table::new([crate::ui::Align::Left, crate::ui::Align::Left], 1);
            for check in &section.checks {
                table.push(crate::ui::Row::new(
                    check.status.mark(),
                    [check.check.clone(), check.detail.clone()],
                ));
            }
            table.print(palette);
        }

        println!();
        let plural = if self.warnings == 1 { "" } else { "s" };
        let (mark, line) = if !self.ready {
            (
                Mark::Blocked,
                format!(
                    "pie cannot boot here ({} blocking, {} warning{plural}).",
                    self.blocking, self.warnings
                ),
            )
        } else if self.warnings > 0 {
            (
                Mark::Warn,
                format!(
                    "Ready, with warnings ({} passed, {} warning{plural}).",
                    self.passed, self.warnings
                ),
            )
        } else {
            (Mark::Did, format!("Ready ({} checks).", self.passed))
        };
        println!("{} {line}", mark.render(palette));
    }
}

pub fn run(global: &bootstrap::GlobalArgs) -> Result<crate::ui::Answer> {
    let mut warnings = 0usize;
    let mut passes = 0usize;
    let mut failures = 0usize;

    let mut sections: Vec<(&'static str, Checks)> = Vec::new();

    let (path, origin) = bootstrap::cli_config_path(global);

    let mut system = vec![check_platform()];
    system.extend(Language::ALL.into_iter().map(check_language));
    sections.push(("system", system));
    sections.push(("built-in inferlets", check_builtin_inferlets()));
    sections.push(("gpus", check_gpus(configured_engine(&path).as_deref())));
    sections.push((
        "engines",
        worker::backend::flavor::compiled_embedded()
            .iter()
            .map(|(name, on)| {
                if *on {
                    (name.to_string(), "compiled in".to_string(), Status::Pass)
                } else {
                    (name.to_string(), absent_because(name), Status::Warn)
                }
            })
            .collect(),
    ));
    sections.push(("config", check_config(&path, origin)));
    sections.push(("tuning", check_tuning(&path)));

    for (_, checks) in &sections {
        for (_, _, status) in checks {
            match status {
                Status::Pass => passes += 1,
                Status::Warn => warnings += 1,
                Status::Fail => failures += 1,
            }
        }
    }
    let ready = failures == 0;

    let report = DoctorReport {
        ready,
        passed: passes,
        warnings,
        blocking: failures,
        sections: sections
            .into_iter()
            .map(|(section, checks)| Section {
                section,
                checks: checks
                    .into_iter()
                    .map(|(check, detail, status)| Check {
                        check,
                        detail,
                        status,
                    })
                    .collect(),
            })
            .collect(),
    };

    let answer = crate::ui::Answer::report(report);
    Ok(if ready {
        answer
    } else {
        answer.with_code(std::process::ExitCode::FAILURE)
    })
}

fn check_config(path: &Path, origin: bootstrap::Origin) -> Checks {
    if !path.exists() {
        return if origin == bootstrap::Origin::Default {
            vec![(
                "config".into(),
                format!(
                    "none at {} — running on defaults",
                    crate::ui::short_path(path)
                ),
                Status::Warn,
            )]
        } else {
            vec![(
                "config".into(),
                format!(
                    "{} does not exist ({})",
                    crate::ui::short_path(path),
                    origin.describe()
                ),
                Status::Fail,
            )]
        };
    }

    let combined = match crate::derive::read_config_file(path) {
        Ok(c) => c,
        Err(e) => {
            return vec![(
                "config".into(),
                format!("{}: {e}", crate::ui::short_path(path)),
                Status::Fail,
            )];
        }
    };
    let worker = match crate::derive::derive_standalone(&combined) {
        Ok((_controller, _gateway, worker)) => worker,
        Err(e) => {
            return vec![(
                "config".into(),
                format!("{}: {e:#}", crate::ui::short_path(path)),
                Status::Fail,
            )];
        }
    };

    let mut out = vec![(
        "config".into(),
        format!("{} parses", crate::ui::short_path(path)),
        Status::Pass,
    )];
    let flavor = worker::backend::flavor::resolve(worker.model.engine.kind, &worker.model.name);
    let want = worker::weights::Want {
        backend: flavor.as_ref().ok().map(|flavor| flavor.as_str()),
        sku: worker.model.sku.as_deref(),
    };
    match worker::weights::resolve(&worker.model.model, want) {
        Ok(resolved) => out.push((
            "weights".into(),
            match resolved {
                worker::weights::Model::Artifact(path) => {
                    format!("artifact {}", crate::ui::short_path(&path))
                }
                worker::weights::Model::Snapshot(path) => format!(
                    "raw snapshot {} — `pie model import` makes an artifact",
                    crate::ui::short_path(&path)
                ),
            },
            Status::Pass,
        )),
        Err(error) => out.push(("weights".into(), format!("{error}"), Status::Fail)),
    }
    let kind = worker.model.engine.kind.as_str();
    let compiled = worker::backend::flavor::compiled_embedded()
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, on)| *on)
        .unwrap_or(false);
    out.push(if compiled {
        (
            "model".into(),
            format!("{} on {}", worker.model.name, kind),
            Status::Pass,
        )
    } else {
        (
            "model".into(),
            format!(
                "{} asks for the {kind} engine: {}",
                worker.model.name,
                absent_because(kind)
            ),
            Status::Fail,
        )
    });
    out
}

const KNOWN_ENGINES: &str = "cuda, metal, vulkan, wgpu";

fn absent_because(name: &str) -> String {
    match name {
        "cuda_native" => "not compiled — build with `--features cuda`".to_string(),
        "metal" if cfg!(target_vendor = "apple") => {
            "not compiled — build with `--features metal`".to_string()
        }
        "metal" => "metal engines run on Apple hardware only".to_string(),
        "vulkan" => "not compiled — build with `--features vulkan`".to_string(),
        "wgpu" => "not compiled — build with `--features wgpu`".to_string(),
        other => format!("unknown engine type `{other}`; this build knows: {KNOWN_ENGINES}"),
    }
}

/// The newest version of `name` installed under the inferlets directory.
fn installed_version(name: &str) -> Option<String> {
    let mut repo = Repository::new(bootstrap::paths::inferlets_dir());
    repo.refresh();
    repo.newest(name).map(|program| program.version)
}

/// The programs behind the HTTP routes are built into this binary, one per
/// API. A copy installed on disk shadows the built-in one; a pie built with
/// `PIE_BUILTINS=skip` has none.
fn check_builtin_inferlets() -> Checks {
    let mut programs: Vec<&str> = gateway::ingress::compat::ROUTES
        .iter()
        .map(|(_, name)| *name)
        .collect();
    programs.sort_unstable();
    programs.dedup();
    let inferlets_dir = crate::ui::short_path(&bootstrap::paths::inferlets_dir());
    programs
        .into_iter()
        .map(|name| {
            let routes: Vec<&str> = gateway::ingress::compat::ROUTES
                .iter()
                .filter(|(_, program)| *program == name)
                .map(|(route, _)| *route)
                .collect();
            let routes = routes.join(", ");
            match (builtins::find(name), installed_version(name)) {
                (Some(builtin), None) => (
                    name.to_string(),
                    format!("{name}@{} built in, serves {routes}", builtin.version),
                    Status::Pass,
                ),
                (Some(builtin), Some(installed)) if installed != builtin.version => (
                    name.to_string(),
                    format!(
                        "{name}@{} built in; {name}@{installed} in {inferlets_dir} shadows it for {routes}",
                        builtin.version
                    ),
                    Status::Pass,
                ),
                (Some(builtin), Some(_)) => (
                    name.to_string(),
                    format!(
                        "{name}@{} built in; the copy in {inferlets_dir} is what serves {routes}",
                        builtin.version
                    ),
                    Status::Pass,
                ),
                (None, Some(installed)) => (
                    name.to_string(),
                    format!("{name}@{installed} installed, serves {routes} (not built into this pie)"),
                    Status::Pass,
                ),
                (None, None) => (
                    name.to_string(),
                    format!(
                        "not built into this pie and not installed, so {routes} answer 404; \
                         `pie inferlet install` a build of crates/builtins/inferlets/{name}"
                    ),
                    Status::Warn,
                ),
            }
        })
        .collect()
}

fn check_platform() -> (String, String, Status) {
    let info = format!(
        "{} {} ({})",
        std::env::consts::OS,
        std::env::consts::FAMILY,
        std::env::consts::ARCH,
    );
    ("Platform".to_string(), info, Status::Pass)
}

fn check_language(language: Language) -> (String, String, Status) {
    let path = crate::ops::language::path(language);
    if path.is_file() {
        (
            language.name().to_string(),
            format!("language component at {}", crate::ui::short_path(&path)),
            Status::Pass,
        )
    } else {
        (
            language.name().to_string(),
            format!(
                "no language component at {}; {language} inferlets need it. `pie language \
                 install pie-language-{language}.tar.gz` (a release asset; install.sh \
                 fetches it) puts it there, or {language}/inferlet/language/build.sh \
                 builds one",
                crate::ui::short_path(&path)
            ),
            Status::Warn,
        )
    }
}

fn configured_engine(config_path: &Path) -> Option<String> {
    let file: toml::Value = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|content| toml::from_str(&content).ok())?;
    worker::config::schema::lookup(&file, "engine.type")
        .and_then(|v| v.as_str())
        .map(str::to_string)
}

fn nvidia_probe_applies(named_engine: Option<&str>) -> bool {
    match named_engine {
        Some(kind) => kind == "cuda_native" || kind == "cuda",
        None => worker::backend::flavor::compiled_embedded()
            .iter()
            .any(|(name, on)| *name == "cuda_native" && *on),
    }
}

fn check_gpus(named_engine: Option<&str>) -> Checks {
    if !nvidia_probe_applies(named_engine) {
        return vec![(
            "GPU".into(),
            match named_engine {
                Some(kind) => format!("not probed — this config names the {kind} engine"),
                None => "not probed — this binary carries no CUDA engine".to_string(),
            },
            Status::Pass,
        )];
    }
    match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,driver_version",
            "--format=csv,noheader",
        ])
        .output()
    {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let lines: Vec<&str> = stdout.lines().filter(|l| !l.trim().is_empty()).collect();
            if lines.is_empty() {
                vec![("GPU".into(), "no NVIDIA GPUs detected".into(), Status::Warn)]
            } else {
                lines
                    .into_iter()
                    .map(|line| {
                        let parts: Vec<&str> = line.split(',').map(str::trim).collect();
                        let idx = parts.first().copied().unwrap_or("?");
                        let rest = parts[1..].join(", ");
                        (format!("GPU {idx}"), rest, Status::Pass)
                    })
                    .collect()
            }
        }
        Ok(_) | Err(_) => vec![(
            "GPU".into(),
            "nvidia-smi not available (CPU-only? non-NVIDIA? or driver missing)".into(),
            Status::Warn,
        )],
    }
}

fn check_tuning(config_path: &Path) -> Checks {
    let file: toml::Value = std::fs::read_to_string(config_path)
        .ok()
        .and_then(|content| toml::from_str(&content).ok())
        .unwrap_or_else(|| toml::Value::Table(Default::default()));
    let set = |key: &str| worker::config::schema::lookup(&file, key).map(|v| v.to_string());

    let mut checks = Vec::new();

    match (
        set("engine.max_forward_tokens"),
        set("engine.max_forward_requests"),
    ) {
        (Some(tokens), Some(requests)) => checks.push((
            "forward shape".to_string(),
            format!("pinned at {tokens} tokens x {requests} requests"),
            Status::Pass,
        )),
        (Some(tokens), None) => checks.push((
            "forward shape".to_string(),
            format!("max_forward_tokens pinned at {tokens}, decode width still derived"),
            Status::Warn,
        )),
        (None, Some(requests)) => checks.push((
            "forward shape".to_string(),
            format!("max_forward_requests pinned at {requests}, token budget still derived"),
            Status::Warn,
        )),
        (None, None) => checks.push((
            "forward shape".to_string(),
            "derived from the engine's own defaults; state them to pin this machine's shape"
                .to_string(),
            Status::Warn,
        )),
    }

    let frame_knobs = ["runtime.frame_size", "runtime.frame_dispatch_depth"];
    let pinned: Vec<&str> = frame_knobs
        .iter()
        .copied()
        .filter(|k| set(k).is_some())
        .collect();
    checks.push(if pinned.is_empty() {
        (
            "batching".to_string(),
            "defaults, measured on other hardware (`pie config tune --for ...`)".to_string(),
            Status::Warn,
        )
    } else {
        (
            "batching".to_string(),
            format!(
                "{} of {} knobs set in this config",
                pinned.len(),
                frame_knobs.len()
            ),
            Status::Pass,
        )
    });

    checks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning_of(config: &str) -> Checks {
        let path = std::env::temp_dir().join(format!(
            "pie-doctor-tuning-{}-{:?}.toml",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::write(&path, config).unwrap();
        let checks = check_tuning(&path);
        let _ = std::fs::remove_file(&path);
        checks
    }

    #[test]
    fn every_route_is_served_by_a_built_in_program_and_vice_versa() {
        let routes = gateway::ingress::compat::ROUTES;
        for (route, name) in routes {
            assert!(
                builtins::find(name).is_some(),
                "{route}: {name} is not built in"
            );
        }
        for builtin in builtins::all() {
            assert!(
                routes.iter().any(|(_, name)| *name == builtin.name),
                "{} is built in but no route serves it",
                builtin.name
            );
        }
    }

    #[test]
    fn the_unmeasured_machine_still_serves() {
        for config in ["", "[engine]\nkv_page_size = 32\n"] {
            let checks = tuning_of(config);
            assert!(
                !checks.iter().any(|(_, _, status)| *status == Status::Fail),
                "nothing here blocks a boot: {checks:?}"
            );
        }
    }
}
