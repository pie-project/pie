//! Library component that exports a greeting function.
//!
//! This component exports `greet(name: String) -> String` which formats a simple
//! greeting message ("Hello, <name>!"). It's a pure standalone library that doesn't
//! import any Pie runtime APIs.
//!
//! Used for testing component linking and composition.

// Generate WIT bindings
wit_bindgen::generate!({
    path: "wit",
    world: "greet-lib",
});

use exports::greet::lib::greet::Guest;

struct Component;

impl Guest for Component {
    fn greet(name: String) -> String {
        format!("Hello, {}!", name)
    }
}

export!(Component);
