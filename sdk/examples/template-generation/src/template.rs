//! Template rendering wrapper.
//!
//! Provides a `TemplateRenderer` that renders a Jinja2-style template with
//! JSON data via `minijinja`. Depends only on `minijinja` and `serde_json`.

use minijinja::Environment;

pub struct TemplateRenderer {
    env: Environment<'static>,
    name: String,
}

impl TemplateRenderer {
    pub fn new(name: &str, template_str: &str) -> Self {
        let mut env = Environment::new();
        env.add_template_owned(name.to_owned(), template_str.to_owned())
            .expect("invalid template");
        Self {
            env,
            name: name.to_owned(),
        }
    }

    /// Renders the template with the given JSON data.
    /// Returns `Ok(rendered)` on success, or `Err(error_message)`.
    pub fn render(&self, data: &serde_json::Value) -> Result<String, String> {
        let tmpl = self.env.get_template(&self.name).unwrap();
        tmpl.render(data)
            .map_err(|e| format!("Template rendering error: {}", e))
    }
}
