//! The edges the refactor removed, held removed.
//!
//! Nothing in this tree enforces its own dependency graph. Cargo enforces
//! *acyclicity* and nothing else: every edge below was legal the day it was
//! added, cost something real, and survived because no one was looking at the
//! closure. `crates/eta-compiler/tests/module_layering.rs` is the in-crate
//! precedent — the rule that was a build error becoming a test — and this is
//! the same move one level up, over the crate graph instead of the module
//! tree, with `cargo metadata` in the place of a textual scan.
//!
//! # A denylist, not a whitelist
//!
//! This does NOT enumerate the graph. A whitelist of allowed edges has to be
//! edited by every legitimate change, and a check that fails on arrival is a
//! check that gets disabled — the same reasoning `crates/checkpoint/tests/citations.rs`
//! gives for its widening `TREES`. What is written here is narrower and
//! stronger: each entry is a decision somebody made, measured and paid for.
//! Adding a crate, or an edge that no rule below names, passes silently and is
//! meant to.
//!
//! # Measured, feature-resolved
//!
//! Every claim reads `resolve.nodes` from `cargo metadata` — the graph cargo
//! actually resolved — and never the manifest `[dependencies]` tables. An
//! optional dependency appears in a manifest whether or not its feature is on,
//! and reading the tables produced a false finding earlier in this refactor:
//! model-loader (now `checkpoint`) looked free of `model-ir` while a
//! `cuda`-gated edge still
//! dragged it in through `kernels-cuda`.
//!
//! It reads two poles — default features and `--all-features` — because those
//! are the two a single `cargo metadata` invocation can state. An edge that is
//! off at one pole and on at the other cannot hide between them, but a pair of
//! mutually exclusive features is beyond this and is not claimed. Neither pole
//! is filtered by platform, so Apple-only and CUDA-only edges are both in
//! view; that is the widest reading and the one a layering rule wants.
//!
//! Kinds are distinguished per rule. Some of these say "not as a dependency,
//! not as a dev-dependency" and use [`Reach::Any`]; the toolchain-disjointness
//! rule is about production edges only and says so where it is written.
//!
//! # Measured state, 2026-08-30
//!
//! Local-crate closures over normal edges, identical at both poles. Every one
//! of these was taken with `cargo metadata` after the rename landed, not read
//! off a design document:
//!
//! ```text
//! controller-api  1   ids
//! controller      2   controller-api, ids
//! worker-api      3   client-api, controller-api, ids
//! gateway         4   client-api, controller-api, ids, worker-api
//! dtype           0
//! ids             0
//! checkpoint      1   dtype
//! eta-compiler    1   eta-ir
//! eta-exec        2   eta-compiler, eta-ir
//! kernels-cuda    1   dtype
//! kernels-metal   1   dtype
//! model-exec      3   dtype, model-compiler, model-ir
//! engine          4   dtype, eta-compiler, eta-ir, model-ir
//! ```
//!
//! `checkpoint`'s closure is the one that differs by kind and by pole and does
//! not: it is `dtype` alone over normal, dev and build edges at BOTH poles,
//! which is the whole claim its rule makes.
//!
//! # What this does not check
//!
//! Anything inside a crate: that is
//! `crates/eta-compiler/tests/module_layering.rs`'s job and it is a
//! different instrument. Anything about third-party dependencies. And any
//! edge no rule below names — see the first section; that silence is the
//! design, not a gap.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// the graph
// ---------------------------------------------------------------------------

/// Which dependency kinds a rule is written over.
///
/// The distinction is load-bearing in both directions. A dev-dependency is a
/// real edge — it is compiled, it is in the closure of `cargo test -p`, and it
/// is how `runtime` held onto the execution substrate after its production
/// edge was gone. But it is also the escape hatch a layering rule is supposed
/// to leave open: `eta-compiler` may test itself against anything, and this
/// very file is a test depending on the whole workspace.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Reach {
    /// `[dependencies]` only — what a `cargo build` of the crate links.
    Normal,
    /// Every edge cargo records: normal, dev and build.
    Any,
}

impl Reach {
    /// `dep_kinds[].kind` is `null` for a normal dependency and the kind's
    /// name otherwise.
    fn admits(self, kind: &str) -> bool {
        match self {
            Reach::Normal => kind == "normal",
            Reach::Any => true,
        }
    }

    fn describe(self) -> &'static str {
        match self {
            Reach::Normal => "as a normal dependency",
            Reach::Any => "as a dependency of some kind (normal, dev or build)",
        }
    }
}

/// One feature pole's resolved graph, restricted to workspace-local crates.
struct Graph {
    /// How this pole was resolved, for diagnostics: a failure that reproduces
    /// only under `--all-features` is a different bug from one that does not.
    pole: &'static str,
    /// Every local crate, by package name.
    crates: BTreeSet<String>,
    /// `from -> to -> kinds`. Local edges only.
    edges: BTreeMap<String, BTreeMap<String, BTreeSet<String>>>,
}

impl Graph {
    /// The shortest route from `from` to `to` over edges `reach` admits,
    /// rendered with each hop's kinds, or `None` if there is no route.
    ///
    /// The rendering is the whole point of returning a route rather than a
    /// bool. `gateway --normal--> worker-api --normal--> controller-api
    /// --normal--> engine --normal--> model-ir` is the path the dead
    /// `WorkerInfo.capability` field cut, and it tells a reader which of four
    /// edges to go and look at. "gateway must not depend on model-ir" does
    /// not.
    fn route(&self, from: &str, to: &str, reach: Reach) -> Option<String> {
        let mut came: BTreeMap<String, String> = BTreeMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(from.to_string());
        let mut reached = from == to;
        while let Some(node) = queue.pop_front() {
            if node == to {
                reached = true;
                break;
            }
            let Some(outgoing) = self.edges.get(&node) else {
                continue;
            };
            for (next, kinds) in outgoing {
                if next == from || came.contains_key(next) {
                    continue;
                }
                if !kinds.iter().any(|kind| reach.admits(kind)) {
                    continue;
                }
                came.insert(next.clone(), node.clone());
                queue.push_back(next.clone());
            }
        }
        if !reached {
            return None;
        }

        // `came` is a BFS tree whose root `from` is never a key, so walking
        // back from `to` terminates.
        let mut hops = vec![to.to_string()];
        loop {
            let last = hops[hops.len() - 1].clone();
            let Some(previous) = came.get(&last) else {
                break;
            };
            hops.push(previous.clone());
        }
        hops.reverse();
        let mut rendered = hops[0].clone();
        for hop in hops.windows(2) {
            let kinds = self
                .edges
                .get(&hop[0])
                .and_then(|outgoing| outgoing.get(&hop[1]))
                .map(|kinds| kinds.iter().cloned().collect::<Vec<_>>().join("+"))
                .unwrap_or_else(|| "?".to_string());
            rendered.push_str(&format!(" --{kinds}--> {}", hop[1]));
        }
        Some(rendered)
    }

    /// Every local crate reachable from `from` over edges `reach` admits.
    fn reaches(&self, from: &str, reach: Reach) -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let mut stack = vec![from.to_string()];
        while let Some(node) = stack.pop() {
            let Some(outgoing) = self.edges.get(&node) else {
                continue;
            };
            for (next, kinds) in outgoing {
                if !kinds.iter().any(|kind| reach.admits(kind)) {
                    continue;
                }
                if seen.insert(next.clone()) {
                    stack.push(next.clone());
                }
            }
        }
        seen
    }

    /// The kinds on the DIRECT edge `from -> to` that `reach` admits, if any.
    fn direct(&self, from: &str, to: &str, reach: Reach) -> Option<String> {
        let kinds: Vec<String> = self
            .edges
            .get(from)?
            .get(to)?
            .iter()
            .filter(|kind| reach.admits(kind.as_str()))
            .cloned()
            .collect();
        (!kinds.is_empty()).then(|| kinds.join("+"))
    }
}

/// The feature poles read, with the flags that produce them.
///
/// Two, not one: default features alone would let an optional edge hide, which
/// is the mistake this file exists partly to prevent from recurring.
const POLES: &[(&str, &[&str])] = &[
    ("default features", &[]),
    ("--all-features", &["--all-features"]),
];

/// Both poles, resolved once for the whole test binary.
///
/// `cargo metadata` neither builds nor takes the target-directory lock, so
/// running it from inside a test is safe; it is also the only way to read a
/// FEATURE-RESOLVED graph, which is the one thing this file may not get wrong.
fn graphs() -> &'static [Graph] {
    static GRAPHS: OnceLock<Vec<Graph>> = OnceLock::new();
    GRAPHS.get_or_init(|| {
        POLES
            .iter()
            .map(|&(pole, flags)| load(pole, flags))
            .collect()
    })
}

fn load(pole: &'static str, flags: &[&str]) -> Graph {
    // `CARGO_MANIFEST_DIR` for the root `pie` package IS the workspace root.
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml");
    let cargo = std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(&cargo)
        .arg("metadata")
        .arg("--format-version")
        .arg("1")
        .arg("--manifest-path")
        .arg(&manifest)
        .args(flags)
        .output()
        .unwrap_or_else(|err| panic!("could not run `cargo metadata` ({pole}): {err}"));
    assert!(
        output.status.success(),
        "`cargo metadata {}` failed ({}). Every rule in this file is measured \
         from that output, so there is nothing to fall back on.\n{}",
        flags.join(" "),
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );

    let json: serde_json::Value =
        serde_json::from_slice(&output.stdout).expect("`cargo metadata` emits JSON on stdout");

    let mut name_of: BTreeMap<String, String> = BTreeMap::new();
    for package in json["packages"]
        .as_array()
        .expect("`cargo metadata` reports `packages`")
    {
        let (Some(id), Some(name)) = (package["id"].as_str(), package["name"].as_str()) else {
            continue;
        };
        name_of.insert(id.to_string(), name.to_string());
    }

    let crates: BTreeSet<String> = json["workspace_members"]
        .as_array()
        .expect("`cargo metadata` reports `workspace_members`")
        .iter()
        .filter_map(|id| name_of.get(id.as_str()?).cloned())
        .collect();
    // Vacuity guard. Every rule below is of the form "there is no path", so a
    // graph that failed to parse would pass all of them in silence.
    assert!(
        crates.len() > 20,
        "only {} local crates resolved ({pole}); this workspace has ~35, so \
         the metadata did not parse and every rule below would pass \
         vacuously",
        crates.len()
    );

    let nodes = json["resolve"]["nodes"]
        .as_array()
        .expect("`cargo metadata` resolves the graph unless `--no-deps` is passed");
    let mut edges: BTreeMap<String, BTreeMap<String, BTreeSet<String>>> = BTreeMap::new();
    for node in nodes {
        let Some(from) = node["id"].as_str().and_then(|id| name_of.get(id)) else {
            continue;
        };
        if !crates.contains(from) {
            continue;
        }
        for dep in node["deps"].as_array().into_iter().flatten() {
            let Some(to) = dep["pkg"].as_str().and_then(|id| name_of.get(id)) else {
                continue;
            };
            if !crates.contains(to) {
                continue;
            }
            let recorded = dep["dep_kinds"].as_array().expect(
                "`dep_kinds` is how normal, dev and build edges are told apart; without it \
                 every rule here would read the wrong graph",
            );
            let kinds = edges
                .entry(from.clone())
                .or_default()
                .entry(to.clone())
                .or_default();
            for dep_kind in recorded {
                kinds.insert(dep_kind["kind"].as_str().unwrap_or("normal").to_string());
            }
            // An edge cargo reported with no kinds at all would silently drop
            // out of every rule below. Read it as normal rather than as absent.
            if kinds.is_empty() {
                kinds.insert("normal".to_string());
            }
        }
    }

    Graph {
        pole,
        crates,
        edges,
    }
}

// ---------------------------------------------------------------------------
// naming a crate, safely
// ---------------------------------------------------------------------------

/// Fail on a name this file uses that the workspace does not have.
///
/// Crate names in this tree move: model-loader became `checkpoint`,
/// `tensor-*` became `eta-*`, and the suffixed contract crate took the name
/// `engine` as the crate that held it split into `model-exec` and `eta-exec`.
/// A guard test
/// that dies on `Option::unwrap` teaches nobody anything, and the repair is
/// almost never "delete the rule".
fn require(graph: &Graph, krate: &str) {
    assert!(
        graph.crates.contains(krate),
        "this test names the crate `{krate}`, which is not in the workspace \
         ({}). Names in this tree have moved before -- model-loader became \
         checkpoint, tensor-* became eta-* -- so the repair is almost \
         certainly to rename it here, NOT to drop the invariant it carries. \
         The local crates are:\n  {}",
        graph.pole,
        graph.crates.iter().cloned().collect::<Vec<_>>().join(", ")
    );
}

/// No route at all from `from` to `to`, transitively.
fn forbid_reaching(graph: &Graph, from: &str, to: &str, reach: Reach, why: &str) {
    require(graph, from);
    require(graph, to);
    if let Some(route) = graph.route(from, to, reach) {
        panic!(
            "`{from}` reaches `{to}` {} ({}):\n\n    {route}\n\n{why}",
            reach.describe(),
            graph.pole
        );
    }
}

/// No DIRECT edge from `from` to `to`.
///
/// Weaker than [`forbid_reaching`] and deliberately so where it is used: a
/// crate may sit in another's closure through a shell that legitimately links
/// both, while a direct edge means this crate's own source names it.
fn forbid_direct(graph: &Graph, from: &str, to: &str, reach: Reach, why: &str) {
    require(graph, from);
    require(graph, to);
    if let Some(kinds) = graph.direct(from, to, reach) {
        panic!(
            "`{from}` depends directly on `{to}` [{kinds}] ({}):\n\n    \
             {from} --{kinds}--> {to}\n\n{why}",
            graph.pole
        );
    }
}

// ---------------------------------------------------------------------------
// the execution layer, whose names are in motion
// ---------------------------------------------------------------------------

/// Which crate holds `trait Engine` and which crates are the substrate under
/// it — answered against the workspace that is here, not the one in anyone's
/// head.
///
/// THE RENAME SWAPPED A NAME RATHER THAN FREEING ONE, and this function was
/// written while it was mid-flight: the token `engine` named the substrate
/// before the swap and the contract after it, so it told the two worlds apart
/// by whether the old suffixed contract crate still existed. It does not.
/// `engine` is the
/// contract, and the crate that used to wear that name is `model-exec` +
/// `eta-exec`.
///
/// What is left is not a branch but a tolerance: the substrate is whichever of
/// the two halves the workspace actually has, so a rule below can be written
/// once and stay true if a half is renamed again or a third appears.
struct Layer {
    /// The crate that holds `trait Engine`.
    contract: String,
    /// The execution machinery under the contract, whichever halves exist.
    substrate: Vec<String>,
}

fn layer(graph: &Graph) -> Layer {
    let contract = "engine".to_string();
    let substrate: Vec<String> = ["model-exec", "eta-exec"]
        .into_iter()
        .filter(|half| graph.crates.contains(*half))
        .map(str::to_string)
        .collect();
    Layer {
        contract,
        substrate,
    }
}

/// Fail rather than pass silently when neither half of the substrate is
/// found under the names [`layer`] knows.
///
/// Every rule over the substrate is of the form "there is no edge", so an
/// empty list satisfies all of them. That is the failure mode a guard test
/// cannot afford: it looks like a green run.
fn require_substrate(graph: &Graph, names: &Layer) {
    assert!(
        !names.substrate.is_empty(),
        "no execution substrate crate was found ({}). This file looks for \
         `model-exec` and `eta-exec`, the two halves the crate formerly called \
         `engine` was split into. Until one of them is named here, every rule \
         written over the substrate passes vacuously -- which is worse than \
         not having one.",
        graph.pole
    );
}

// ---------------------------------------------------------------------------
// the rules
// ---------------------------------------------------------------------------

/// The IR crates, and the compilers over them. A control-plane crate that
/// reaches any of these is carrying a description of a tensor program it will
/// never read.
const COMPUTE_PLANE: &[&str] = &["model-ir", "eta-ir", "model-compiler", "eta-compiler"];

/// The crates that route requests and never execute one.
const CONTROL_PLANE: &[&str] = &["controller-api", "controller", "worker-api", "gateway"];

#[test]
fn the_control_plane_carries_no_ir() {
    let why = "The control plane routes requests; it does not describe computations. All \
               four of these crates carried both IRs, and the entire cause was one field -- \
               `WorkerInfo.capability`, serialized by the worker, shipped over tarpc and \
               dropped on arrival at `controller/src/actor.rs`. Two doc comments already \
               said it was unused; nothing connected a dead field to four crate closures, \
               so its cost stayed invisible. If capability-aware placement is wanted, the \
               controller takes the specific numbers it needs as its own fields -- the rule \
               `WorkerStatus` already follows -- not the engine's record whole.";
    for graph in graphs() {
        for from in CONTROL_PLANE {
            for to in COMPUTE_PLANE {
                // Any kind: an IR crate in the dev closure is still an IR
                // crate in `cargo test -p gateway`, and the field that put it
                // there could come back through a test fixture just as easily.
                forbid_reaching(graph, from, to, Reach::Any, why);
            }
        }
    }
}

#[test]
fn dtype_and_ids_are_leaves() {
    let why = "This crate exists to be depended on and to depend on nothing. `dtype` is one \
               enum and the reason `model-ir`, `kernels-cuda`, `checkpoint` and the transfer \
               contract can all spell an element type without meeting each other; `ids` is \
               the atom the roles and the wire contracts sit on. Both are load-bearing as \
               FLOORS: the control-plane closures above read 1/2/3/4 only because what they \
               bottom out in has nothing under it, so an edge added here raises numbers in \
               crates whose own manifests did not change.";
    for graph in graphs() {
        for leaf in ["dtype", "ids"] {
            require(graph, leaf);
            let reached = graph.reaches(leaf, Reach::Any);
            assert!(
                reached.is_empty(),
                "`{leaf}` is not a leaf any more ({}): it reaches {}.\n\n    {}\n\n{why}",
                graph.pole,
                reached.iter().cloned().collect::<Vec<_>>().join(", "),
                reached
                    .iter()
                    .filter_map(|other| graph.route(leaf, other.as_str(), Reach::Any))
                    .collect::<Vec<_>>()
                    .join("\n    ")
            );
        }
    }
}

#[test]
fn checkpoint_reaches_nothing_but_dtype() {
    let why = "`checkpoint` (was model-loader) parses checkpoints and answers with memory \
               layouts a device can be pointed at. It knows no model family and no kernel. \
               It HAD an optional `cuda`-gated edge to `kernels-cuda` that dragged `kernels` \
               and `model-ir` in behind it -- invisible under default features, which is how \
               it survived a review that read the manifest. The device executor is \
               `engine-cuda`'s now and the `cuda`/`cuda-12`/`cuda-13` features were deleted \
               outright rather than narrowed, which is why this rule can be stated over \
               `--all-features` at all.";
    for graph in graphs() {
        require(graph, "checkpoint");
        require(graph, "dtype");
        let reached = graph.reaches("checkpoint", Reach::Any);
        let extra: Vec<&String> = reached.iter().filter(|name| *name != "dtype").collect();
        assert!(
            extra.is_empty(),
            "`checkpoint` reaches more than `dtype` ({}):\n\n    {}\n\n{why}",
            graph.pole,
            extra
                .iter()
                .filter_map(|other| graph.route("checkpoint", other.as_str(), Reach::Any))
                .collect::<Vec<_>>()
                .join("\n    ")
        );
    }
}

#[test]
fn the_kernel_libraries_reach_nothing_but_dtype() {
    let why = "A kernel library is a pile of device code and the vocabulary to ask for it. \
               It knows no IR, no plan and no engine -- `engine-cuda` and `engine-metal` are \
               what hold a kernel beside a `CompiledModel`, and they depend on both. Each of \
               these two HAD an edge to a 141-line crate called `kernels` for one \
               three-variant error enum, and that crate's other half -- the six `Dispatch*` \
               traits -- carried `model-ir` in behind it, for a contract neither library \
               ever named. The traits went to `model-exec`, which holds their one caller \
               (`fire::walk`), and each library declares its own `Error`. What that bought \
               is this rule: over normal, dev AND build edges, at both poles.";
    for graph in graphs() {
        for lib in ["kernels-cuda", "kernels-metal"] {
            require(graph, lib);
            require(graph, "dtype");
            let reached = graph.reaches(lib, Reach::Any);
            let extra: Vec<&String> = reached.iter().filter(|name| *name != "dtype").collect();
            assert!(
                extra.is_empty(),
                "`{lib}` reaches more than `dtype` ({}):\n\n    {}\n\n{why}",
                graph.pole,
                extra
                    .iter()
                    .filter_map(|other| graph.route(lib, other.as_str(), Reach::Any))
                    .collect::<Vec<_>>()
                    .join("\n    ")
            );
        }
    }
}

#[test]
fn the_eta_compiler_does_not_depend_on_the_execution_layer() {
    let why = "The producer owns its output type. `LaunchPackage` and the twenty types \
               around it lived in the engine contract crate, so the compiler that EMITS them \
               had to depend on the crate that merely receives them -- an inverted edge its \
               own manifest apologised for. The artifact is `eta_compiler::codegen::launch` \
               now and the edge is deleted. A compiler under an executor is the same \
               inversion whichever executor it is, so contract and substrate are both \
               forbidden here.";
    for graph in graphs() {
        let names = layer(graph);
        forbid_reaching(graph, "eta-compiler", &names.contract, Reach::Any, why);
        for substrate in &names.substrate {
            forbid_reaching(graph, "eta-compiler", substrate, Reach::Any, why);
        }
    }
}

#[test]
fn the_execution_substrate_does_not_depend_on_the_engine_contract() {
    let why = "The substrate executes; the contract is how a shell is ASKED to. `eta-exec` \
               is a job over `eta-ir` and `eta-compiler`, and `model-exec` is one over \
               `model-ir` and `model-compiler`; neither needs to know what an \
               `Engine` is, and the shells that implement one -- `engine-cuda`, \
               `engine-metal` -- link both halves and the contract, which is where the \
               knowledge belongs. This was true of neither half while they shared a crate: \
               the ETA side named the contract crate's `program::*` fifteen times and its \
               `channel` once, and moving the program artifact to `eta-compiler` \
               is what left it contract-free. The direction also has to hold for the \
               contract to be able to name `eta_compiler::LaunchPackage` without a cycle.";
    for graph in graphs() {
        let names = layer(graph);
        require_substrate(graph, &names);
        for substrate in &names.substrate {
            forbid_reaching(graph, substrate, &names.contract, Reach::Any, why);
        }
    }
}

#[test]
fn the_runtime_does_not_depend_on_the_execution_substrate() {
    let why = "`runtime` schedules; it does not execute. It held this edge twice -- as a \
               normal dependency, then demoted to a dev-dependency -- and by the end the \
               only code reading it was one test that drove `Engine::submit` directly, \
               bypassing the pipeline it nominally tested. The test was deleted as misplaced \
               (its four siblings that do the same thing live in `tests/gpu/`) and the edge \
               went with it. A dev-dependency is not a loophole here: it is exactly the form \
               this edge took last time.";
    for graph in graphs() {
        let names = layer(graph);
        require_substrate(graph, &names);
        for substrate in &names.substrate {
            // Direct, not transitive. `runtime` reaches the substrate through
            // `engine-cuda`, which links both halves and is supposed to; what
            // is forbidden is `runtime`'s own source naming it.
            forbid_direct(graph, "runtime", substrate, Reach::Any, why);
        }
    }
}

/// The eDSL, the IR and the compiler of each toolchain, in dependency order.
const ETA_TOOLCHAIN: &[&str] = &["eta-ir", "eta-dsl", "eta-compiler"];
const MODEL_TOOLCHAIN: &[&str] = &["model-ir", "model-dsl", "model-compiler"];

#[test]
fn the_two_toolchains_do_not_meet() {
    let why = "The workspace declares two parallel stacks -- model-dsl -> model-ir -> \
               model-compiler and eta-dsl -> eta-ir -> eta-compiler -- and every argument \
               built on top of that reads them as disjoint: the `eta-*` renaming, and the \
               engine split, which rests on the measurement that the two halves' upstreams \
               share no crate and cross-reference each other zero times. That disjointness \
               is not enforced by anything; one `use model_ir::Dtype` inside `eta-ir` would \
               end it and no build would complain. The element type they DO share is \
               `dtype`, which is a leaf on purpose and is the sanctioned way to share one. \
               Production edges only: a parity test may reach across, and this file is \
               itself an argument for keeping that door open.";
    for graph in graphs() {
        for eta in ETA_TOOLCHAIN {
            for model in MODEL_TOOLCHAIN {
                forbid_reaching(graph, eta, model, Reach::Normal, why);
                forbid_reaching(graph, model, eta, Reach::Normal, why);
            }
        }
    }
}
