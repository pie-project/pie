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
