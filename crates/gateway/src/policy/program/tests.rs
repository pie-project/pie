use super::*;
use crate::policy::ManifestError;
use crate::policy::wit::Dispatch;

#[derive(Default, Clone)]
struct Fake {
    processes: Vec<(String, Vec<(String, Value)>)>,
    attrs: Vec<(String, String, Value)>,
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
            _ => Vec::new(),
        }
    }

    fn attr(&self, of: &Subject, name: &str) -> Lookup {
        let Subject::Process(id) = of else {
            return Lookup::UnknownSubject;
        };
        self.attrs
            .iter()
            .find(|(p, n, _)| p == id && n == name)
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
}

fn load(name: &str) -> Result<Program, LoadError> {
    let builtin = builtins::policy(name).unwrap_or_else(|| panic!("no built-in policy {name}"));
    Program::load(name, builtin.component, &|_| Level::Native)
}

#[test]
fn router_policies_load_and_engine_policies_are_refused() {
    let mut routers = 0;
    for builtin in builtins::policies() {
        match load(builtin.name) {
            Ok(_) => {
                routers += 1;
                let component = Component::new(engine(), builtin.component).unwrap();
                let imports: Vec<String> = component
                    .component_type()
                    .imports(engine())
                    .map(|(name, _)| name.to_string())
                    .collect();
                assert!(
                    !imports.iter().any(|name| name.contains("engine-writes")),
                    "{}: {imports:?}",
                    builtin.name
                );
            }
            Err(LoadError::Manifest(ManifestError::WrongPlane(plane))) => {
                assert_eq!(plane, "engine", "{}", builtin.name);
            }
            Err(error) => panic!("{}: {error}", builtin.name),
        }
    }
    assert!(routers > 0, "test-route at least");
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
    assert_eq!(placement["b"].worker, None, "the router's default");
    assert!(matches!(
        placement["a"].props.as_slice(),
        [(name, Value::Text(from))] if name == "from" && from == "a"
    ));
}

#[test]
fn an_unsubscribed_edge_does_not_run_the_program() {
    let mut program = load("test-route").unwrap();
    let plane = Fake::default().with("a", &[]);
    let edges = vec![Edge::WorkerAdded("1".into())];
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
fn router_priority_orders_the_gateway_and_passes_priority_on() {
    let mut program = load("router-priority").unwrap();
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
