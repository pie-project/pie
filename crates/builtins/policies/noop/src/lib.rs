//! Installs nothing: it must behave as no policy at all.

#[derive(Default)]
struct Noop;

impl policy::engine::Policy for Noop {}

policy::engine!(Noop);
