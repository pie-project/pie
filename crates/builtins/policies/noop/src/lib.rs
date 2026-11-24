//! Writes nothing: it must behave as no policy at all (a free baseline
//! arm, PLEX 1.1 principle 2).

use plex::prelude::*;

#[derive(Default)]
struct Noop;

impl Policy for Noop {}

plex::export!(Noop);
