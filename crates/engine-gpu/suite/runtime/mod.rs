mod outputs;
mod port;
mod register;
mod roles;
mod slots;

/// Registers a program and waits until its build has told: the id of a
/// program a bind takes.
fn registered(
    engine: &mut crate::backend::Engine,
    registration: &engine::program::ProgramRegistration,
) -> engine::program::ProgramId {
    use engine::Engine;
    let (tell, told) = std::sync::mpsc::channel();
    let program = engine
        .register_program(
            registration,
            Box::new(move |built| {
                let _ = tell.send(built);
            }),
        )
        .expect("the program registers");
    told.recv()
        .expect("a registration that returned an id is told")
        .expect("the program builds");
    program
}
