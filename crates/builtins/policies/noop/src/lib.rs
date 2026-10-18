//! Installs nothing: it must behave as no policy at all.

#[derive(Default)]
struct Noop;

impl policy::worker::Policy for Noop {}

policy::worker!(Noop);
