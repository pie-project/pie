use std::process::ExitCode;

use clap::{Parser, Subcommand};
use pie::{compose, derive, ops, ui};
#[derive(Parser, Debug)]
#[command(
    name = "pie",
    version,
    about = "Pie — Programmable Inference Engine (standalone)",
    disable_help_subcommand = true
)]
struct Cli {
    #[command(flatten)]
    global: bootstrap::GlobalArgs,

    #[arg(long, global = true)]
    json: bool,

    #[arg(long, global = true, value_name = "WORDS")]
    diag: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Serve,

    Run(ops::run::RunArgs),

    Model {
        #[command(subcommand)]
        cmd: ops::model::ModelCmd,
    },

    Cache {
        #[command(subcommand)]
        cmd: ops::cache::CacheCmd,
    },

    Config {
        #[command(subcommand)]
        cmd: ops::config::ConfigCmd,
    },

    Inferlet {
        #[command(subcommand)]
        cmd: ops::inferlet::InferletCmd,
    },

    Language {
        #[command(subcommand)]
        cmd: ops::language::LanguageCmd,
    },

    Doctor,
}

#[cfg(unix)]
fn die_quietly_on_closed_pipe() {
    // SAFETY: setting a signal disposition to SIG_DFL before any threads that
    // could observe a different one. This is the documented workaround for
    // rust-lang/rust#62569.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
}

#[cfg(not(unix))]
fn die_quietly_on_closed_pipe() {}

fn report(error: &anyhow::Error) {
    let palette = ui::Palette::for_stream(ui::Stream::Stderr);
    eprintln!("{} {error}", ui::Mark::Blocked.render(&palette));
    for cause in error.chain().skip(1) {
        for line in cause.to_string().lines() {
            eprintln!("  {}", palette.dim(line));
        }
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    die_quietly_on_closed_pipe();
    match run().await {
        Ok(code) => code,
        Err(error) => {
            report(&error);
            ExitCode::FAILURE
        }
    }
}

async fn run() -> anyhow::Result<ExitCode> {
    let cli = Cli::parse();

    if let Command::Serve = cli.command {
        return serve(cli.global, cli.diag.as_deref()).await;
    }

    bootstrap::init_cli(&cli.global)?;

    let answer = match cli.command {
        Command::Serve => unreachable!("serve returns before the op dispatch"),

        Command::Model { cmd } => {
            let global = cli.global.clone();
            tokio::task::spawn_blocking(move || ops::model::run(cmd, &global)).await??
        }

        Command::Cache { cmd } => {
            tokio::task::spawn_blocking(move || ops::cache::run(cmd)).await??
        }
        Command::Doctor => {
            let global = cli.global.clone();
            tokio::task::spawn_blocking(move || ops::doctor::run(&global)).await??
        }

        Command::Run(args) => ops::run::run(&cli.global, args, cli.diag.as_deref()).await?,
        Command::Config { cmd } => ops::config::run(cmd, &cli.global).await?,
        Command::Inferlet { cmd } => ops::inferlet::run(cmd).await?,
        Command::Language { cmd } => {
            tokio::task::spawn_blocking(move || ops::language::run(cmd)).await??
        }
    };

    let code = answer.code();
    ui::present(answer, cli.json)?;
    Ok(code)
}

async fn serve(global: bootstrap::GlobalArgs, diag: Option<&str>) -> anyhow::Result<ExitCode> {
    let ctx = bootstrap::init(
        bootstrap::BootSpec::pie().version(env!("CARGO_PKG_VERSION")),
        global,
    )?;
    let (controller, gateway, mut worker) = derive::derive_standalone(ctx.config_str())?;
    if let Some(words) = diag {
        worker.state_diagnostics(words)?;
    }
    let handle = compose::run_standalone(controller, gateway, worker).await?;
    tracing::info!(
        listen = %handle.listen_addr,
        worker = %handle.worker_addr,
        "pie standalone serving",
    );
    println!("✓ Server ready at ws://{}", handle.listen_addr);
    Ok(ctx
        .run_until_signal(async move { handle.shutdown().await })
        .await)
}
