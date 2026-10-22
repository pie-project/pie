use super::*;
use crate::policy::ManifestError;
use crate::policy::wit::Dispatch;

#[derive(Default, Clone)]
struct Fake {
    processes: Vec<(String, Vec<(String, Value)>)>,
    attrs: Vec<(String, String, Value)>,
    workers: Vec<String>,
    groups: Vec<String>,
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

    fn worker(mut self, id: &str) -> Self {
        self.workers.push(id.to_string());
        self
    }

    fn attr(mut self, process: &str, name: &str, value: Value) -> Self {
        self.attrs
            .push((process.to_string(), name.to_string(), value));
        self
    }
}

impl Plane for Fake {
    fn candidates(&self, kind: SubjectKind) -> Vec<String> {
        match kind {
            SubjectKind::Process => self.processes.iter().map(|(id, _)| id.clone()).collect(),
            SubjectKind::Worker => self.workers.clone(),
            SubjectKind::Group => self.groups.clone(),
            _ => Vec::new(),
        }
    }

    fn attr(&self, of: &Subject, name: &str) -> Option<Value> {
        let (Subject::Process(id) | Subject::Worker(id)) = of else {
            return None;
        };
        self.attrs
            .iter()
            .find(|(p, n, _)| p == id && n == name)
            .map(|(_, _, value)| value.clone())
    }

    fn prop(&self, of: &Subject, name: &str) -> Option<Value> {
        let Subject::Process(id) = of else {
            return None;
        };
        let (_, props) = self.processes.iter().find(|(p, _)| p == id)?;
        props
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, value)| value.clone())
    }

    fn capability(&self, _name: &str) -> bool {
        true
    }

    fn terminable(&self, _process: &str) -> bool {
        true
    }
}

fn load(name: &str) -> Result<Program, LoadError> {
    let builtin = builtins::policy(name).unwrap_or_else(|| panic!("no built-in policy {name}"));
    Program::load(name, builtin.component, &|_| true)
}

#[test]
fn cluster_policies_load_and_worker_policies_are_refused() {
    let mut clusters = 0;
    for builtin in builtins::policies() {
        match load(builtin.name) {
            Ok(_) => {
                clusters += 1;
                let component = Component::new(engine(), builtin.component).unwrap();
                let imports: Vec<String> = component
                    .component_type()
                    .imports(engine())
                    .map(|(name, _)| name.to_string())
                    .collect();
                assert!(
                    !imports.iter().any(|name| name.contains("worker-writes")),
                    "{}: {imports:?}",
                    builtin.name
                );
            }
            Err(LoadError::Manifest(ManifestError::WrongPlane(plane))) => {
                assert_eq!(plane, "worker", "{}", builtin.name);
            }
            Err(error) => panic!("{}: {error}", builtin.name),
        }
    }
    assert!(clusters > 0, "test-route at least");
}

#[test]
fn test_route_orders_holds_and_places_by_its_clients_props() {
    let mut program = load("test-route").unwrap();
    let plane = Fake::default()
        .with(
            "a",
            &[("key", Value::Int(0)), ("worker", Value::Text("7".into()))],
        )
        .with("b", &[]);
    let edges = vec![Edge::Arrived("a".into()), Edge::Arrived("b".into())];
    let commit = program.react(Box::new(plane), edges, 0).unwrap();

    let schedule = commit.schedule.expect("a schedule");
    assert_eq!((schedule.key("a"), schedule.key("b")), (0, 1));
    assert_eq!(schedule.floor, Some(1), "so a is held");

    let placement = commit.placement.expect("a placement table");
    assert_eq!(placement["a"].worker.as_deref(), Some("7"));
    assert_eq!(placement["b"].worker, None, "the gateway's default");
    assert!(matches!(
        placement["a"].props.as_slice(),
        [(name, Value::Text(from))] if name == "from" && from == "a"
    ));
}

#[test]
fn retain_all_retains_every_group_above_key_0() {
    let mut program = load("retain-all").unwrap();
    let plane = Fake {
        groups: vec!["g1".into(), "g2".into()],
        ..Fake::default()
    };
    let copied = Edge::Copied(types::Residency {
        group: "g1".into(),
        site: types::Site {
            worker: "1".into(),
            tier: types::Tier::Device,
        },
    });
    let commit = program.react(Box::new(plane), vec![copied], 0).unwrap();
    let retention = commit.state.retention.expect("a retention table");
    assert_eq!((retention.key("g1"), retention.key("g2")), (1, 1));
    assert_eq!(retention.keys.len(), 2);
}

#[test]
fn an_unsubscribed_edge_does_not_run_the_program() {
    let mut program = load("test-route").unwrap();
    let plane = Fake::default().with("a", &[]);
    let edges = vec![Edge::WorkerDraining("1".into())];
    let commit = program.react(Box::new(plane), edges, 0).unwrap();
    assert!(commit.schedule.is_none() && commit.placement.is_none());
}

#[test]
fn session_affinity_sends_a_session_where_its_last_process_went() {
    let mut program = load("session-affinity").unwrap();
    let in_session = |plane: Fake, id: &str| {
        plane
            .with(id, &[])
            .attr(id, "session", Value::Text("t/chat".into()))
    };
    let first = in_session(Fake::default(), "1");
    let commit = program
        .react(Box::new(first.clone()), vec![Edge::Arrived("1".into())], 0)
        .unwrap();
    assert!(
        commit.placement.unwrap().is_empty(),
        "a first turn goes by default"
    );
    let dispatched = Edge::Dispatched(Dispatch {
        process: "1".into(),
        worker: "7".into(),
    });
    program
        .react(Box::new(first), vec![dispatched], 10)
        .unwrap();

    let next = in_session(Fake::default(), "2").with("alone", &[]);
    let edges = vec![Edge::Arrived("2".into()), Edge::Arrived("alone".into())];
    let commit = program.react(Box::new(next.clone()), edges, 20).unwrap();
    let placement = commit.placement.unwrap();
    assert_eq!(placement["2"].worker.as_deref(), Some("7"));
    assert!(!placement.contains_key("alone"), "no session, no placement");

    let gone = vec![Edge::WorkerRemoved("7".into())];
    let commit = program.react(Box::new(next), gone, 30).unwrap();
    assert!(commit.placement.unwrap().is_empty(), "its worker is gone");
}

#[test]
fn cache_aware_follows_the_cache_until_one_worker_piles_up() {
    let mut program = load("cache-aware").unwrap();
    let waiting = |plane: Fake, id: &str, prompt: i64, on_1: i64, on_2: i64| {
        plane
            .with(id, &[])
            .attr(id, "state", Value::Text("pending".into()))
            .attr(id, "prompt-tokens", Value::Int(prompt))
            .attr(id, "cached-tokens/1", Value::Int(on_1))
            .attr(id, "cached-tokens/2", Value::Int(on_2))
    };
    let running = |plane: Fake, id: &str, on: &str| {
        plane
            .with(id, &[])
            .attr(id, "state", Value::Text("running".into()))
            .attr(id, "worker", Value::Text(on.into()))
    };
    let two = Fake::default().worker("1").worker("2");
    // Worker 2 holds most of "a"'s prompt; "b" is held nowhere much, so it
    // goes to the less loaded, which "a" just made worker 1.
    let plane = waiting(waiting(two.clone(), "a", 1000, 100, 800), "b", 1000, 200, 0);
    let edges = vec![Edge::Arrived("a".into()), Edge::Arrived("b".into())];
    let placement = program
        .react(Box::new(plane), edges, 0)
        .unwrap()
        .placement
        .unwrap();
    assert_eq!(placement["a"].worker.as_deref(), Some("2"));
    assert_eq!(placement["b"].worker.as_deref(), Some("1"));

    // Worker 2 runs ten turns and 1 none: load wins over cache.
    let mut busy = two.clone();
    for i in 0..10 {
        busy = running(busy, &format!("r{i}"), "2");
    }
    let plane = waiting(busy, "c", 1000, 0, 900);
    let placement = program
        .react(Box::new(plane), vec![Edge::Arrived("c".into())], 10)
        .unwrap()
        .placement
        .unwrap();
    assert_eq!(placement["c"].worker.as_deref(), Some("1"));
}

#[test]
fn cluster_priority_orders_the_gateway_and_passes_priority_on() {
    let mut program = load("cluster-priority").unwrap();
    let plane = Fake::default()
        .with("1", &[("priority", Value::Int(3))])
        .with("2", &[]);
    let edges = vec![Edge::Arrived("1".into()), Edge::Arrived("2".into())];
    let commit = program.react(Box::new(plane), edges, 0).unwrap();
    let schedule = commit.schedule.unwrap();
    assert_eq!((schedule.key("1"), schedule.key("2")), (3, 0));
    let placement = commit.placement.unwrap();
    assert_eq!(placement["1"].worker, None);
    assert!(matches!(
        placement["1"].props.as_slice(),
        [(name, Value::Int(3))] if name == "priority"
    ));
    assert!(!placement.contains_key("2"));
}

#[test]
fn a_program_publishes_under_the_publish_grant_and_one_without_it_is_refused() {
    let mut share = load("test-share").unwrap();
    let plane =
        Fake::default()
            .with("7", &[])
            .attr("7", "placement/worker", Value::Text("3".into()));
    let commit = share
        .react(Box::new(plane), vec![Edge::PlacementChanged("7".into())], 0)
        .unwrap();
    assert!(matches!(
        commit.publish.as_slice(),
        [(Subject::Process(p), name, Some(Value::Text(w)))] if p == "7" && name == "placed" && w == "3"
    ));

    let mut route = load("test-route").unwrap();
    let plane = Fake::default().with("8", &[("publish", Value::Text("x".into()))]);
    let commit = route
        .react(Box::new(plane), vec![Edge::Arrived("8".into())], 0)
        .unwrap();
    assert!(
        commit.publish.is_empty(),
        "test-route is not granted `publish`"
    );
    assert!(commit.placement.is_some(), "its granted writes stand");
}

#[test]
fn disaggregate_puts_first_the_prefill_workers_on_the_placed_process_s_host() {
    let mut disaggregate = Program::load(
        "disaggregate",
        builtins::policy("disaggregate").unwrap().component,
        &|_| true,
    )
    .unwrap();
    let text = |t: &str| Value::Text(t.into());
    let plane = Fake::default()
        .with("7", &[])
        .attr("7", "placement/worker", text("1"))
        .worker("1")
        .attr("1", "role", text("decode"))
        .attr("1", "host", text("node-b"))
        .worker("2")
        .attr("2", "role", text("prefill"))
        .attr("2", "host", text("node-a"))
        .attr("2", "running", Value::Int(0))
        .worker("3")
        .attr("3", "role", text("prefill"))
        .attr("3", "host", text("node-b"))
        .attr("3", "running", Value::Int(5));
    let commit = disaggregate
        .react(Box::new(plane), vec![Edge::PlacementChanged("7".into())], 0)
        .unwrap();
    assert_eq!(
        commit.executors.unwrap()["7"],
        ["3", "2"],
        "the loaded one on the placed host first, then the idle one elsewhere"
    );
}
