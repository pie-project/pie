use std::time::{Duration, Instant};

use super::*;
use crate::policy::wit::{By, Exit, Outcome};

#[derive(Default, Clone)]
struct Fake {
    processes: Vec<(String, Vec<(String, Value)>)>,
    attrs: Vec<(String, String, Value)>,
}

fn spelled(of: &Subject) -> String {
    match of {
        Subject::Session(id) => format!("session:{id}"),
        Subject::Process(id) => format!("process:{id}"),
        Subject::WorkingSet(id) => format!("working-set:{id}"),
        Subject::Worker(id) => format!("worker:{id}"),
    }
}

impl Fake {
    fn with(mut self, process: &str, props: &[(&str, Value)]) -> Self {
        self.processes.push((
            process.to_string(),
            props
                .iter()
                .map(|(name, value)| (name.to_string(), value.clone()))
                .collect(),
        ));
        self
    }

    fn attr(mut self, of: Subject, name: &str, value: Value) -> Self {
        self.attrs.push((spelled(&of), name.to_string(), value));
        self
    }
}

impl Plane for Fake {
    fn candidates(&self, kind: SubjectKind) -> Vec<String> {
        match kind {
            SubjectKind::Process => self.processes.iter().map(|(id, _)| id.clone()).collect(),
            _ => Vec::new(),
        }
    }

    fn attr(&self, of: &Subject, name: &str) -> Lookup {
        let of = spelled(of);
        self.attrs
            .iter()
            .find(|(subject, n, _)| *subject == of && n == name)
            .map_or(Lookup::UnknownName, |(_, _, value)| {
                Lookup::Value(value.clone())
            })
    }

    fn prop(&self, of: &Subject, name: &str) -> Lookup {
        let Subject::Process(id) = of else {
            return Lookup::UnknownSubject;
        };
        let Some((_, props)) = self.processes.iter().find(|(p, _)| p == id) else {
            return Lookup::UnknownSubject;
        };
        match props.iter().find(|(n, _)| n == name) {
            Some((_, value)) => Lookup::Value(value.clone()),
            None => Lookup::Absent,
        }
    }

    fn capability(&self, _name: &str) -> Level {
        Level::Native
    }

    fn terminable(&self, _process: &str) -> Result<(), Refusal> {
        Ok(())
    }

    fn discardable(&self, _working_set: &str) -> Result<(), Refusal> {
        Ok(())
    }
}

fn load(name: &str) -> Program {
    let builtin = builtins::policy(name).unwrap_or_else(|| panic!("no built-in policy {name}"));
    Program::load(name, builtin.component, &|_| Level::Native)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn arrived(process: &str) -> Vec<Edge> {
    vec![Edge::Arrived(process.to_string())]
}

#[test]
fn every_built_in_policy_loads_and_imports_no_router_writes() {
    for builtin in builtins::policies() {
        // A router policy is the gateway's to load.
        if let Err(ManifestError::WrongPlane(_)) =
            Manifest::of(builtin.component, &|_| Level::Native)
        {
            continue;
        }
        let program = load(builtin.name);
        let component = Component::new(engine(), builtin.component).unwrap();
        let imports: Vec<String> = component
            .component_type()
            .imports(engine())
            .map(|(name, _)| name.to_string())
            .collect();
        assert!(
            !imports.iter().any(|name| name.contains("router-writes")),
            "{}: {imports:?}",
            program.name()
        );
    }
}

fn tenant_process(plane: Fake, id: &str, tenant: &str, state: &str) -> Fake {
    let process = || Subject::Process(id.to_string());
    plane
        .with(id, &[])
        .attr(process(), "tenant", Value::Text(tenant.into()))
        .attr(process(), "state", Value::Text(state.into()))
}

#[test]
fn fair_runs_the_least_served_tenant_first_and_banks_no_idle_credit() {
    let mut program = load("fair");
    let busy = |plane| tenant_process(plane, "a1", "a", "running");
    let plane = tenant_process(busy(Fake::default()), "b1", "b", "pending");
    let edges = vec![Edge::Arrived("a1".into()), Edge::Arrived("b1".into())];
    program.react(Box::new(plane.clone()), edges, 0).unwrap();
    assert!(
        program.alarm().is_some(),
        "service is counted while a1 runs"
    );

    // A second of a1 running: tenant a has had more.
    let commit = program
        .react(Box::new(plane), vec![Edge::Timer], 1000)
        .unwrap();
    let schedule = commit.schedule.unwrap();
    assert!(schedule.key("b1") > schedule.key("a1"));

    // Tenant c arrives late and starts at the least an active tenant has
    // (b's 0), not below it.
    let plane = tenant_process(
        tenant_process(busy(Fake::default()), "b1", "b", "pending"),
        "c1",
        "c",
        "pending",
    );
    let edges = vec![Edge::Arrived("c1".into())];
    let commit = program.react(Box::new(plane), edges, 2000).unwrap();
    let schedule = commit.schedule.unwrap();
    assert_eq!(schedule.key("c1"), schedule.key("b1"));
    assert_eq!(schedule.key("a1"), -2000);
}

#[test]
fn deadline_runs_the_soonest_due_first_and_the_undated_last() {
    let mut program = load("deadline");
    let process = |id: &str| Subject::Process(id.to_string());
    let plane = Fake::default()
        .with("late", &[("deadline-ms", Value::Int(5000))])
        .attr(process("late"), "arrival-ms", Value::Int(0))
        .with("soon", &[("deadline-ms", Value::Int(500))])
        .attr(process("soon"), "arrival-ms", Value::Int(1000))
        .with("undated", &[])
        .attr(process("undated"), "arrival-ms", Value::Int(0));
    let commit = program.react(Box::new(plane), arrived("soon"), 0).unwrap();
    let schedule = commit.schedule.unwrap();
    assert!(schedule.key("soon") > schedule.key("late"));
    assert!(schedule.key("late") > schedule.key("undated"));
    assert!(!schedule.keys.contains_key("undated"));
}

#[test]
fn keep_session_keeps_a_session_s_latest_turn_until_it_is_due() {
    let mut program = load("keep-session");
    let ws = |id: &str| Subject::WorkingSet(id.to_string());
    let turn = |plane: Fake, id: &str| {
        plane.with(id, &[("keep-ms", Value::Int(1000))]).attr(
            ws(id),
            "session",
            Value::Text("t/chat".into()),
        )
    };
    let exit = |id: &str| {
        vec![
            Edge::Exited(Exit {
                process: id.into(),
                outcome: Outcome::Completed,
                by: By::Host,
            }),
            Edge::Retained(id.into()),
        ]
    };

    let plane = turn(Fake::default(), "t1");
    program
        .react(Box::new(plane.clone()), arrived("t1"), 0)
        .unwrap();
    let commit = program.react(Box::new(plane), exit("t1"), 100).unwrap();
    let kept = commit.retention.expect("a retention table");
    assert!(matches!(kept.steps["t1"][..], [(None, key)] if key > 0));
    assert_eq!(program.alarm(), Some(1100));

    // The session's next turn takes its place.
    let plane = turn(turn(Fake::default(), "t1"), "t2");
    program
        .react(Box::new(plane.clone()), arrived("t2"), 500)
        .unwrap();
    let commit = program.react(Box::new(plane), exit("t2"), 600).unwrap();
    let kept = commit.retention.unwrap();
    assert_eq!(kept.steps.keys().collect::<Vec<_>>(), ["t2"]);

    // Due: discarded.
    let commit = program
        .react(Box::new(Fake::default()), vec![Edge::Timer], 1600)
        .unwrap();
    assert_eq!(commit.discard, ["t2"]);
    assert!(commit.retention.unwrap().steps.is_empty());
}

#[test]
fn priority_keys_what_the_router_ranked_else_what_the_client_did() {
    let mut program = load("priority");
    let plane = Fake::default()
        .with("a", &[("priority", Value::Int(5))])
        .with("b", &[("tier", Value::Text("gold".into()))])
        .with("c", &[("priority", Value::Int(-3))])
        .with(
            "d",
            &[
                ("priority", Value::Int(9)),
                ("router/priority", Value::Int(1)),
            ],
        );
    let commit = program.react(Box::new(plane), arrived("a"), 0).unwrap();
    let schedule = commit.schedule.expect("priority installs a schedule");
    assert_eq!(schedule.key("a"), 5);
    assert_eq!(schedule.key("b"), 0);
    assert_eq!(schedule.key("c"), -3);
    assert_eq!(schedule.key("d"), 1, "the router's rank wins");
    assert_eq!(schedule.keys.len(), 3, "only ranked processes are named");
    assert_eq!(schedule.floor, None);
}

#[test]
fn an_edge_the_program_did_not_subscribe_to_does_not_run_it() {
    let mut program = load("priority");
    let plane = Fake::default().with("a", &[("priority", Value::Int(5))]);
    let commit = program
        .react(Box::new(plane), vec![Edge::Started("a".into())], 0)
        .unwrap();
    assert!(commit.schedule.is_none());
}

#[test]
fn a_trap_discards_the_call_and_the_program_runs_again() {
    let mut program = load("test-trap");
    for _ in 0..2 {
        let fault = program
            .react(Box::new(Fake::default()), arrived("a"), 0)
            .unwrap_err();
        assert!(matches!(fault, Fault::Trap(_)), "{fault}");
        assert!(
            program.store.data().scratch.is_empty(),
            "the trapped call's put did not commit"
        );
    }
}

#[test]
fn an_overrun_is_cut_off_at_the_deadline() {
    let mut program = load("test-spin");
    let deadline = program.manifest.deadline_ms;
    for _ in 0..2 {
        let started = Instant::now();
        let fault = program
            .react(Box::new(Fake::default()), arrived("a"), 0)
            .unwrap_err();
        assert!(
            matches!(fault, Fault::Deadline(ms) if ms == deadline),
            "{fault}"
        );
        assert!(
            started.elapsed() < Duration::from_millis(deadline * 20),
            "cut off in {:?}",
            started.elapsed()
        );
    }
}

#[test]
fn scratch_commits_and_outlives_a_restart() {
    let mut program = load("test-scratch");
    let plane = Fake::default().with("a", &[]).with("b", &[]);
    for call in 1..=3 {
        let commit = program
            .react(Box::new(plane.clone()), arrived("a"), 100)
            .unwrap();
        let schedule = commit.schedule.expect("test-scratch installs a schedule");
        assert_eq!((schedule.key("a"), schedule.key("b")), (call, call));
    }
    assert_eq!(program.alarm(), Some(1100));

    program.restart().unwrap();
    let commit = program
        .react(Box::new(plane.clone()), vec![Edge::Timer], 200)
        .unwrap();
    assert_eq!(
        commit.schedule.unwrap().key("a"),
        4,
        "the count survived the restart"
    );
    assert_eq!(program.alarm(), Some(1200), "the timer re-armed it");
}
