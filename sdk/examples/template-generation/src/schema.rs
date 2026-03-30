//! JSON Schema validation wrapper.
//!
//! Provides a `SchemaValidator` that parses a JSON string and validates it
//! against a JSON Schema. Depends only on `jsonschema` and `serde_json`.

pub struct SchemaValidator {
    schema: serde_json::Value,
    validator: jsonschema::Validator,
}

impl SchemaValidator {
    pub fn new(schema_str: &str) -> Self {
        let schema: serde_json::Value =
            serde_json::from_str(schema_str).expect("invalid schema JSON");
        let validator = jsonschema::validator_for(&schema).expect("invalid JSON Schema");
        Self { schema, validator }
    }

    /// Parses `json_str` as JSON and validates it against the schema.
    /// Returns `Ok(parsed_value)` on success, or `Err(error_report)` with
    /// either the parse error or all schema validation errors.
    pub fn validate(&self, json_str: &str) -> Result<serde_json::Value, String> {
        let value: serde_json::Value =
            serde_json::from_str(json_str).map_err(|e| format!("JSON parse error: {}", e))?;

        let errors: Vec<String> = self
            .validator
            .iter_errors(&value)
            .map(|e| format!("- {}", e))
            .collect();

        if errors.is_empty() {
            Ok(value)
        } else {
            Err(errors.join("\n"))
        }
    }
}
