//! JSON Schema to grammar converter.
//!
//! Converts a JSON Schema into an EBNF grammar string, then parses it
//! into a `Grammar`. The generated grammar constrains LLM output to valid
//! JSON matching the schema.
//!
//! # Supported features
//! - Types: string, integer, number, boolean, null, array, object
//! - Constraints: enum, const, minLength/maxLength, minimum/maximum,
//!   exclusiveMinimum/exclusiveMaximum, pattern, format
//! - Arrays: items, prefixItems, minItems/maxItems
//! - Objects: properties, required, additionalProperties
//! - Composition: $ref, anyOf, oneOf
//! - Formatting: indent, separators, any_whitespace

use std::collections::HashMap;

use anyhow::{Result, bail};
use serde_json::Value;

use crate::structured::grammar::Grammar;
use crate::structured::regex::regex_to_ebnf;

/// Options for JSON schema to grammar conversion.
#[derive(Debug, Clone)]
pub struct JsonSchemaOptions {
    /// Allow arbitrary whitespace between JSON elements.
    pub any_whitespace: bool,
    /// Indent level (number of spaces). None = single line.
    pub indent: Option<usize>,
    /// Custom separators: (item_separator, key_value_separator).
    pub separators: Option<(String, String)>,
    /// If true, disallow additional properties/items not in schema.
    pub strict_mode: bool,
}

impl Default for JsonSchemaOptions {
    fn default() -> Self {
        Self {
            any_whitespace: true,
            indent: None,
            separators: None,
            strict_mode: true,
        }
    }
}

/// Convert a JSON Schema string to a Grammar.
pub fn json_schema_to_grammar(schema: &str, options: &JsonSchemaOptions) -> Result<Grammar> {
    let schema_val: Value = serde_json::from_str(schema)?;
    let ebnf = json_schema_to_ebnf(&schema_val, options)?;
    Grammar::from_ebnf(&ebnf, "root")
}

/// Convert a parsed JSON Schema to an EBNF grammar string.
pub fn json_schema_to_ebnf(schema: &Value, options: &JsonSchemaOptions) -> Result<String> {
    let mut converter = SchemaConverter::new(options.clone());
    let root_expr = converter.visit_schema(schema, "root")?;
    converter.add_rule("root", &root_expr);
    Ok(converter.to_ebnf())
}

/// Create a grammar for standard JSON (any valid JSON value).
pub fn builtin_json_grammar() -> Result<Grammar> {
    let ebnf = BUILTIN_JSON_EBNF;
    Grammar::from_ebnf(ebnf, "root")
}

const BUILTIN_JSON_EBNF: &str = r#"
root ::= value
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws (pair ("," ws pair)*)? ws "}"
pair ::= ws string ws ":" ws value
array ::= "[" ws (value ("," ws value)*)? ws "]"
string ::= "\"" char* "\""
char ::= [^"\\] | "\\" escape
escape ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]
number ::= integer fraction? exponent?
integer ::= "-"? ("0" | [1-9] [0-9]*)
fraction ::= "." [0-9]+
exponent ::= [eE] [+-]? [0-9]+
ws ::= [ \t\n\r]*
"#;

// ---------------------------------------------------------------------------
// Internal converter
// ---------------------------------------------------------------------------

struct SchemaConverter {
    options: JsonSchemaOptions,
    rules: Vec<(String, String)>,
    rule_cache: HashMap<String, String>,
    aux_counter: usize,
    /// Whitespace rule expression
    ws: String,
    /// Separators
    item_sep: String,
    kv_sep: String,
}

impl SchemaConverter {
    fn new(options: JsonSchemaOptions) -> Self {
        let (item_sep, kv_sep) = match &options.separators {
            Some((is, kvs)) => (is.clone(), kvs.clone()),
            None => {
                if options.any_whitespace {
                    (",".to_string(), ":".to_string())
                } else if options.indent.is_some() {
                    (",".to_string(), ": ".to_string())
                } else {
                    (",".to_string(), ":".to_string())
                }
            }
        };

        let ws = if options.any_whitespace {
            "ws".to_string()
        } else {
            "\"\"".to_string() // no whitespace
        };

        Self {
            options,
            rules: Vec::new(),
            rule_cache: HashMap::new(),
            aux_counter: 0,
            ws,
            item_sep,
            kv_sep,
        }
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        self.aux_counter += 1;
        format!("{}_{}", prefix, self.aux_counter)
    }

    fn add_rule(&mut self, name: &str, body: &str) {
        self.rules.push((name.to_string(), body.to_string()));
    }

    fn to_ebnf(&self) -> String {
        let mut out = String::new();
        for (name, body) in &self.rules {
            out.push_str(&format!("{} ::= {}\n", name, body));
        }
        // Add ws rule if needed
        if self.options.any_whitespace {
            out.push_str("ws ::= [ \\t\\n\\r]*\n");
        }
        out
    }

    fn visit_schema(&mut self, schema: &Value, hint: &str) -> Result<String> {
        // Handle boolean schemas
        if let Some(b) = schema.as_bool() {
            if b {
                return self.visit_any(hint);
            } else {
                bail!("false schema: no values are valid");
            }
        }

        let obj = match schema.as_object() {
            Some(o) => o,
            None => bail!("schema must be an object or boolean"),
        };

        // Check cache
        let cache_key = schema.to_string();
        if let Some(cached) = self.rule_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Handle $ref
        if let Some(ref_val) = obj.get("$ref") {
            // Simple $ref handling (only supports #/definitions/name style)
            if let Some(ref_str) = ref_val.as_str() {
                if let Some(name) = ref_str.strip_prefix("#/definitions/") {
                    return Ok(sanitize_rule_name(name));
                }
                if let Some(name) = ref_str.strip_prefix("#/$defs/") {
                    return Ok(sanitize_rule_name(name));
                }
                bail!("unsupported $ref: {}", ref_str);
            }
        }

        // Handle const
        if let Some(const_val) = obj.get("const") {
            return self.visit_const(const_val);
        }

        // Handle enum
        if let Some(enum_val) = obj.get("enum") {
            return self.visit_enum(enum_val);
        }

        // Handle anyOf / oneOf
        if let Some(any_of) = obj.get("anyOf").or_else(|| obj.get("oneOf")) {
            return self.visit_any_of(any_of, hint);
        }

        // Handle allOf (limited)
        if let Some(all_of) = obj.get("allOf") {
            if let Some(arr) = all_of.as_array() {
                if arr.len() == 1 {
                    return self.visit_schema(&arr[0], hint);
                }
            }
            bail!("allOf with multiple schemas is not fully supported");
        }

        // Handle definitions
        if let Some(defs) = obj.get("definitions").or_else(|| obj.get("$defs")) {
            if let Some(defs_obj) = defs.as_object() {
                for (name, def_schema) in defs_obj {
                    let rule_name = sanitize_rule_name(name);
                    let expr = self.visit_schema(def_schema, &rule_name)?;
                    self.add_rule(&rule_name, &expr);
                }
            }
        }

        // Determine type
        let type_val = obj.get("type");

        match type_val {
            Some(Value::String(t)) => self.visit_typed(schema, t, hint),
            Some(Value::Array(types)) => {
                // Union of types
                let mut alts = Vec::new();
                for t in types {
                    if let Some(t_str) = t.as_str() {
                        let alt = self.visit_typed(schema, t_str, hint)?;
                        alts.push(alt);
                    }
                }
                Ok(format!("({})", alts.join(" | ")))
            }
            None => {
                // No type specified; infer from properties
                if obj.contains_key("properties") || obj.contains_key("required")
                    || obj.contains_key("minProperties") || obj.contains_key("maxProperties") {
                    self.visit_typed(schema, "object", hint)
                } else if obj.contains_key("items") || obj.contains_key("prefixItems") {
                    self.visit_typed(schema, "array", hint)
                } else if obj.contains_key("pattern") || obj.contains_key("minLength") || obj.contains_key("maxLength") || obj.contains_key("format") {
                    self.visit_typed(schema, "string", hint)
                } else if obj.contains_key("minimum") || obj.contains_key("maximum") {
                    self.visit_typed(schema, "number", hint)
                } else {
                    self.visit_any(hint)
                }
            }
            _ => bail!("unexpected type value"),
        }
    }

    fn visit_typed(&mut self, schema: &Value, type_name: &str, hint: &str) -> Result<String> {
        match type_name {
            "string" => self.visit_string(schema),
            "integer" => self.visit_integer(schema),
            "number" => self.visit_number(schema),
            "boolean" => Ok("(\"true\" | \"false\")".to_string()),
            "null" => Ok("\"null\"".to_string()),
            "array" => self.visit_array(schema, hint),
            "object" => self.visit_object(schema, hint),
            _ => bail!("unknown type: {}", type_name),
        }
    }

    fn visit_any(&mut self, hint: &str) -> Result<String> {
        let name = self.fresh_name(&format!("{}_value", hint));
        let ws = self.ws.clone();
        let item_sep = self.item_sep.clone();
        let kv_sep = self.kv_sep.clone();

        // JSON value: object | array | string | number | boolean | null
        let string_rule = format!("{}_string", name);
        let char_rule = format!("{}_char", name);
        let escape_rule = format!("{}_escape", name);
        let number_rule = format!("{}_number", name);
        let integer_rule = format!("{}_integer", name);

        self.add_rule(
            &string_rule,
            &format!("\"\\\"\" {}* \"\\\"\"", char_rule),
        );
        self.add_rule(
            &char_rule,
            &format!("[^\"\\\\] | \"\\\\\" {}", escape_rule),
        );
        self.add_rule(
            &escape_rule,
            "\"\\\"\" | \"\\\\\" | \"/\" | \"b\" | \"f\" | \"n\" | \"r\" | \"t\" | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]",
        );
        self.add_rule(
            &number_rule,
            &format!("{} (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?", integer_rule),
        );
        self.add_rule(
            &integer_rule,
            "\"-\"? (\"0\" | [1-9] [0-9]*)",
        );

        let obj_pair = format!("{} {} {} \"{}\" {} {}", ws, string_rule, ws, kv_sep, ws, name);
        let obj_body = format!("\"{{\" {} ({} (\"{}\"{} {})*)? {} \"}}\"", ws, obj_pair, item_sep, ws, obj_pair, ws);
        let arr_body = format!("\"[\" {} ({} (\"{}\"{} {})*)? {} \"]\"", ws, name, item_sep, ws, name, ws);

        let body = format!(
            "{} | {} | {} | {} | \"true\" | \"false\" | \"null\"",
            obj_body, arr_body, string_rule, number_rule
        );

        self.add_rule(&name, &body);
        Ok(name)
    }

    fn visit_const(&mut self, val: &Value) -> Result<String> {
        Ok(json_value_to_ebnf_literal(val))
    }

    fn visit_enum(&mut self, enum_val: &Value) -> Result<String> {
        let arr = enum_val.as_array().ok_or_else(|| anyhow::anyhow!("enum must be an array"))?;
        let alts: Vec<String> = arr.iter().map(json_value_to_ebnf_literal).collect();
        Ok(format!("({})", alts.join(" | ")))
    }

    fn visit_any_of(&mut self, any_of: &Value, hint: &str) -> Result<String> {
        let arr = any_of.as_array().ok_or_else(|| anyhow::anyhow!("anyOf must be an array"))?;
        let mut alts = Vec::new();
        for (i, schema) in arr.iter().enumerate() {
            let name = format!("{}_{}", hint, i);
            let expr = self.visit_schema(schema, &name)?;
            alts.push(expr);
        }
        Ok(format!("({})", alts.join(" | ")))
    }

    fn visit_string(&mut self, schema: &Value) -> Result<String> {
        let obj = schema.as_object().unwrap();

        // Format takes highest precedence
        if let Some(format_val) = obj.get("format") {
            if let Some(fmt) = format_val.as_str() {
                if let Some(regex_pattern) = format_to_regex(fmt) {
                    let ebnf = regex_to_ebnf(&regex_pattern)?;
                    let body = ebnf
                        .strip_prefix("root ::= ")
                        .unwrap_or(&ebnf)
                        .trim();
                    let name = self.fresh_name("string_format");
                    self.add_rule(&name, body);
                    return Ok(format!("\"\\\"\" {} \"\\\"\"", name));
                }
                // Unknown format: fall through to default string handling
            }
        }

        // Pattern takes next precedence
        if let Some(pattern) = obj.get("pattern") {
            if let Some(p) = pattern.as_str() {
                let ebnf = regex_to_ebnf(p)?;
                // Extract the body (after "root ::= ")
                let body = ebnf
                    .strip_prefix("root ::= ")
                    .unwrap_or(&ebnf)
                    .trim();
                let name = self.fresh_name("string_pattern");
                self.add_rule(&name, body);
                return Ok(format!("\"\\\"\" {} \"\\\"\"", name));
            }
        }

        let min_len = obj.get("minLength").and_then(|v| v.as_u64()).unwrap_or(0);
        let max_len = obj.get("maxLength").and_then(|v| v.as_u64());

        let char_rule = "[^\"\\\\] | \"\\\\\" (\"\\\"\" | \"\\\\\" | \"/\" | \"b\" | \"f\" | \"n\" | \"r\" | \"t\" | \"u\" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])";

        let char_name = self.fresh_name("json_char");
        self.add_rule(&char_name, char_rule);

        let body = match (min_len, max_len) {
            (0, None) => format!("\"\\\"\" {}* \"\\\"\"", char_name),
            (0, Some(max)) => format!("\"\\\"\" {}{{0,{}}} \"\\\"\"", char_name, max),
            (min, None) => format!("\"\\\"\" {}{{{},}} \"\\\"\"", char_name, min),
            (min, Some(max)) => format!("\"\\\"\" {}{{{},{}}} \"\\\"\"", char_name, min, max),
        };

        Ok(body)
    }

    fn visit_integer(&mut self, schema: &Value) -> Result<String> {
        let obj = schema.as_object().unwrap();
        let min = obj.get("minimum").and_then(|v| v.as_i64())
            .or_else(|| obj.get("exclusiveMinimum").and_then(|v| v.as_i64()).map(|v| v + 1));
        let max = obj.get("maximum").and_then(|v| v.as_i64())
            .or_else(|| obj.get("exclusiveMaximum").and_then(|v| v.as_i64()).map(|v| v - 1));

        match (min, max) {
            (None, None) => Ok("\"-\"? (\"0\" | [1-9] [0-9]*)".to_string()),
            (Some(lo), Some(hi)) if lo > hi => bail!("minimum > maximum"),
            _ => {
                let regex = generate_integer_range_regex(min, max);
                let ebnf_body = regex_to_ebnf(&regex)?;
                let body = ebnf_body
                    .strip_prefix("root ::= ")
                    .unwrap_or(&ebnf_body)
                    .trim()
                    .to_string();
                Ok(body)
            }
        }
    }

    fn visit_number(&mut self, schema: &Value) -> Result<String> {
        let obj = schema.as_object().unwrap();
        let has_bounds = obj.contains_key("minimum")
            || obj.contains_key("maximum")
            || obj.contains_key("exclusiveMinimum")
            || obj.contains_key("exclusiveMaximum");

        if !has_bounds {
            return Ok("\"-\"? (\"0\" | [1-9] [0-9]*) (\".\" [0-9]+)? ([eE] [+-]? [0-9]+)?".to_string());
        }

        // For bounded numbers, use integer bounds + optional fraction
        self.visit_integer(schema).map(|int_part| {
            format!("{} (\".\" [0-9]+)?", int_part)
        })
    }

    fn visit_array(&mut self, schema: &Value, hint: &str) -> Result<String> {
        let obj = schema.as_object().unwrap();
        let ws = self.ws.clone();
        let item_sep = self.item_sep.clone();

        let prefix_items = obj.get("prefixItems").and_then(|v| v.as_array());
        let items = obj.get("items");
        let min_items = obj.get("minItems").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let max_items = obj.get("maxItems").and_then(|v| v.as_u64());

        // Build element rules
        let mut fixed_elements: Vec<String> = Vec::new();
        if let Some(prefix) = prefix_items {
            for (i, item_schema) in prefix.iter().enumerate() {
                let name = format!("{}_item_{}", hint, i);
                let expr = self.visit_schema(item_schema, &name)?;
                fixed_elements.push(expr);
            }
        }

        let additional_item = if let Some(items_schema) = items {
            if items_schema.as_bool() == Some(false) {
                None
            } else {
                let name = format!("{}_additional", hint);
                Some(self.visit_schema(items_schema, &name)?)
            }
        } else if self.options.strict_mode {
            None
        } else {
            Some(self.visit_any(hint)?)
        };

        // Build the array body
        let num_fixed = fixed_elements.len();

        if num_fixed == 0 && additional_item.is_none() {
            // Empty array only
            return Ok(format!("\"[\" {} \"]\"", ws));
        }

        if num_fixed == 0 {
            // Homogeneous array
            let item_expr = additional_item.unwrap();
            let body = if min_items == 0 && max_items.is_none() {
                format!(
                    "\"[\" {} ({} {} (\"{}\"{} {} {})*)? {} \"]\"",
                    ws, ws, item_expr, item_sep, ws, ws, item_expr, ws
                )
            } else {
                let _arr_name = self.fresh_name(&format!("{}_arr", hint));
                let min = min_items;
                // max_str accounts for the first element being placed separately
                let max_str = match max_items {
                    Some(m) if m > 0 => format!(",{}", m - 1),
                    Some(_) => ",0".to_string(),
                    None => ",".to_string(),
                };
                // Use repetition syntax
                let elem_rule = self.fresh_name(&format!("{}_elem", hint));
                let sep_elem = format!("\"{}\"{} {} {}", item_sep, ws, ws, item_expr);
                self.add_rule(&elem_rule, &sep_elem);

                if min == 0 {
                    format!(
                        "\"[\" {} ({} {} {}{{{}{}}})? {} \"]\"",
                        ws, ws, item_expr, elem_rule, 0, max_str, ws
                    )
                } else {
                    let first_elem = format!("{} {} {}", ws, ws, item_expr);
                    let min_rest = if min > 0 { min - 1 } else { 0 };
                    format!(
                        "\"[\" {} {} {}{{{}{}}}{} \"]\"",
                        ws, first_elem, elem_rule, min_rest, max_str, ws
                    )
                }
            };
            return Ok(body);
        }

        // Mixed: fixed prefix items + optional additional
        let mut parts = Vec::new();
        for (i, elem) in fixed_elements.iter().enumerate() {
            if i > 0 {
                parts.push(format!("\"{}\" {} {}", item_sep, ws, elem));
            } else {
                parts.push(format!("{} {}", ws, elem));
            }
        }

        let fixed_part = parts.join(" ");

        if let Some(ref addl) = additional_item {
            let addl_min = if min_items > num_fixed { min_items - num_fixed } else { 0 };
            let addl_max = max_items.map(|m| {
                let m = m as usize;
                if m > num_fixed { m - num_fixed } else { 0 }
            });

            let addl_elem_rule = self.fresh_name(&format!("{}_addl_elem", hint));
            let sep_addl = format!("\"{}\"{} {} {}", item_sep, ws, ws, addl);
            self.add_rule(&addl_elem_rule, &sep_addl);

            let addl_part = match (addl_min, addl_max) {
                (0, None) => format!("{}*", addl_elem_rule),
                (0, Some(0)) => String::new(),
                (min, None) => format!("{}{{{},}}", addl_elem_rule, min),
                (min, Some(max)) if min == max => format!("{}{{{}}}", addl_elem_rule, min),
                (min, Some(max)) => format!("{}{{{},{}}}", addl_elem_rule, min, max),
            };
            if addl_part.is_empty() {
                Ok(format!("\"[\" {} {} {} \"]\"", ws, fixed_part, ws))
            } else {
                Ok(format!("\"[\" {} {} {} {} \"]\"", ws, fixed_part, addl_part, ws))
            }
        } else {
            Ok(format!("\"[\" {} {} {} \"]\"", ws, fixed_part, ws))
        }
    }

    fn visit_object(&mut self, schema: &Value, hint: &str) -> Result<String> {
        let obj = schema.as_object().unwrap();
        let ws = self.ws.clone();
        let item_sep = self.item_sep.clone();
        let kv_sep = self.kv_sep.clone();

        let properties = obj.get("properties").and_then(|v| v.as_object());
        let required: Vec<String> = obj
            .get("required")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default();

        let min_properties = obj.get("minProperties").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
        let max_properties = obj.get("maxProperties").and_then(|v| v.as_u64());

        if let Some(max) = max_properties {
            let max = max as usize;
            if min_properties > max {
                bail!(
                    "minProperties ({}) is greater than maxProperties ({})",
                    min_properties,
                    max
                );
            }
            if required.len() > max {
                bail!(
                    "number of required properties ({}) exceeds maxProperties ({})",
                    required.len(),
                    max
                );
            }
        }

        let additional = obj.get("additionalProperties");
        let (allow_additional, additional_schema) = match additional {
            Some(Value::Bool(false)) => (false, None),
            Some(Value::Bool(true)) => (true, None),
            Some(schema_val) if schema_val.is_object() => (true, Some(schema_val.clone())),
            None if self.options.strict_mode => (false, None),
            None => (true, None),
            _ => (true, None),
        };

        let mut prop_rules: Vec<(String, String, bool)> = Vec::new(); // (key, value_expr, is_required)

        if let Some(props) = properties {
            for (key, prop_schema) in props {
                let prop_hint = format!("{}_{}", hint, sanitize_rule_name(key));
                let value_expr = self.visit_schema(prop_schema, &prop_hint)?;
                let is_required = required.contains(key);
                prop_rules.push((key.clone(), value_expr, is_required));
            }
        }

        if prop_rules.is_empty() && !allow_additional {
            return Ok(format!("\"{{\" {} \"}}\"", ws));
        }

        if prop_rules.is_empty() && allow_additional {
            return self.build_any_property_object(
                hint, &additional_schema, min_properties, max_properties,
                &ws, &item_sep, &kv_sep,
            );
        }

        // Build object with known properties
        // Required properties must appear; optional properties may appear
        let required_props: Vec<&(String, String, bool)> =
            prop_rules.iter().filter(|(_, _, r)| *r).collect();
        let optional_props: Vec<&(String, String, bool)> =
            prop_rules.iter().filter(|(_, _, r)| !*r).collect();

        if optional_props.is_empty() {
            // All required: fixed order
            let mut parts = Vec::new();
            for (i, (key, val_expr, _)) in required_props.iter().enumerate() {
                let pair = format!(
                    "\"\\\"{}\\\"\" {} \"{}\" {} {}",
                    escape_json_string(key), ws, kv_sep, ws, val_expr
                );
                if i > 0 {
                    parts.push(format!("\"{}\"{} {} {}", item_sep, ws, ws, pair));
                } else {
                    parts.push(format!("{} {}", ws, pair));
                }
            }
            if allow_additional {
                self.append_additional_suffix(
                    &mut parts, hint, &additional_schema,
                    required_props.len(), min_properties, max_properties,
                    &ws, &item_sep, &kv_sep,
                )?;
            }
            return Ok(format!("\"{{\" {} {} {} \"}}\"", ws, parts.join(" "), ws));
        }

        // Mix of required and optional: use simpler approach
        // List all properties with optional markers
        let mut parts = Vec::new();
        let mut first = true;
        for (key, val_expr, is_req) in &prop_rules {
            let pair = format!(
                "\"\\\"{}\\\"\" {} \"{}\" {} {}",
                escape_json_string(key), ws, kv_sep, ws, val_expr
            );
            let separator = if first {
                format!("{} ", ws)
            } else {
                format!("\"{}\"{} {} ", item_sep, ws, ws)
            };
            if *is_req {
                parts.push(format!("{}{}", separator, pair));
            } else {
                parts.push(format!("({}{})?", separator, pair));
            }
            first = false;
        }

        if allow_additional {
            self.append_additional_suffix(
                &mut parts, hint, &additional_schema,
                required_props.len(), min_properties, max_properties,
                &ws, &item_sep, &kv_sep,
            )?;
        }

        Ok(format!("\"{{\" {} {} \"}}\"", parts.join(""), ws))
    }

    /// Build an object with no named properties, only arbitrary key-value pairs.
    fn build_any_property_object(
        &mut self,
        hint: &str,
        additional_schema: &Option<Value>,
        min_properties: usize,
        max_properties: Option<u64>,
        ws: &str,
        item_sep: &str,
        kv_sep: &str,
    ) -> Result<String> {
        let max_p = max_properties.map(|m| m as usize);

        if max_p == Some(0) {
            return Ok(format!("\"{{\" {} \"}}\"", ws));
        }

        let kv_name = self.fresh_name(&format!("{}_kv", hint));
        let str_expr = self.visit_string(&Value::Object(serde_json::Map::new()))?;
        let value_expr = match additional_schema {
            Some(s) => self.visit_schema(s, &format!("{}_addl", hint))?,
            None => self.visit_any(hint)?,
        };
        let kv_body = format!("{} {} \"{}\" {} {}", str_expr, ws, kv_sep, ws, value_expr);
        self.add_rule(&kv_name, &kv_body);

        if min_properties == 0 && max_p.is_none() {
            return Ok(format!(
                "\"{{\" {} ({} (\"{}\"{} {})*)? {} \"}}\"",
                ws, kv_name, item_sep, ws, kv_name, ws
            ));
        }

        let sep_kv_name = self.fresh_name(&format!("{}_sep_kv", hint));
        let sep_kv_body = format!("\"{}\" {} {}", item_sep, ws, kv_name);
        self.add_rule(&sep_kv_name, &sep_kv_body);

        let rest_min = if min_properties > 0 { min_properties - 1 } else { 0 };
        let rest_suffix = match max_p {
            None => format!("{{{},}}", rest_min),
            Some(max) => {
                let rest_max = max - 1;
                repetition_suffix(rest_min, Some(rest_max))
            }
        };

        if min_properties == 0 {
            if rest_suffix.is_empty() {
                return Ok(format!("\"{{\" {} ({})? {} \"}}\"", ws, kv_name, ws));
            }
            Ok(format!(
                "\"{{\" {} ({} {}{})? {} \"}}\"",
                ws, kv_name, sep_kv_name, rest_suffix, ws
            ))
        } else {
            if rest_suffix.is_empty() {
                return Ok(format!("\"{{\" {} {} {} \"}}\"", ws, kv_name, ws));
            }
            Ok(format!(
                "\"{{\" {} {} {}{} {} \"}}\"",
                ws, kv_name, sep_kv_name, rest_suffix, ws
            ))
        }
    }

    /// Append additional-properties repetition suffix to a parts list.
    fn append_additional_suffix(
        &mut self,
        parts: &mut Vec<String>,
        hint: &str,
        additional_schema: &Option<Value>,
        num_required: usize,
        min_properties: usize,
        max_properties: Option<u64>,
        ws: &str,
        item_sep: &str,
        kv_sep: &str,
    ) -> Result<()> {
        let addl_kv = self.build_additional_kv_rule(hint, additional_schema, ws, kv_sep)?;
        let addl_min = min_properties.saturating_sub(num_required);
        let addl_max = max_properties.map(|m| (m as usize).saturating_sub(num_required));
        let suffix = repetition_suffix(addl_min, addl_max);
        if !suffix.is_empty() {
            parts.push(format!("(\"{}\"{} {} {}){}", item_sep, ws, ws, addl_kv, suffix));
        }
        Ok(())
    }

    /// Build a rule for additional key-value pairs, returning the rule name.
    fn build_additional_kv_rule(
        &mut self,
        hint: &str,
        additional_schema: &Option<Value>,
        ws: &str,
        kv_sep: &str,
    ) -> Result<String> {
        let addl_kv_name = self.fresh_name(&format!("{}_addl_kv", hint));
        let str_expr = self.visit_string(&Value::Object(serde_json::Map::new()))?;
        let addl_value_expr = match additional_schema {
            Some(s) => self.visit_schema(s, &format!("{}_addl_val", hint))?,
            None => self.visit_any(hint)?,
        };
        let addl_kv_body = format!("{} {} \"{}\" {} {}", str_expr, ws, kv_sep, ws, addl_value_expr);
        self.add_rule(&addl_kv_name, &addl_kv_body);
        Ok(addl_kv_name)
    }
}

/// Generate a regex pattern that matches integers in [min, max].
fn generate_integer_range_regex(min: Option<i64>, max: Option<i64>) -> String {
    match (min, max) {
        (None, None) => "-?(?:0|[1-9][0-9]*)".to_string(),
        (Some(lo), Some(hi)) if lo == hi => format!("{}", lo),
        (Some(lo), Some(hi)) => {
            if lo >= 0 {
                positive_range_regex(lo as u64, hi as u64)
            } else if hi < 0 {
                let pos = positive_range_regex((-hi) as u64, (-lo) as u64);
                format!("-{}", pos)
            } else {
                // Range crosses zero
                let neg = positive_range_regex(1, (-lo) as u64);
                let pos = positive_range_regex(0, hi as u64);
                format!("(?:-{}|{})", neg, pos)
            }
        }
        (Some(lo), None) => {
            if lo >= 0 {
                if lo == 0 {
                    "(?:0|[1-9][0-9]*)".to_string()
                } else {
                    format!("(?:{})", positive_range_regex_unbounded(lo as u64))
                }
            } else {
                let neg_bound = positive_range_regex(1, (-lo) as u64);
                format!("(?:-(?:{}|[1-9][0-9]*)|(?:0|[1-9][0-9]*))", neg_bound)
            }
        }
        (None, Some(hi)) => {
            if hi < 0 {
                format!("-(?:{})", positive_range_regex_unbounded((-hi) as u64))
            } else {
                let pos = positive_range_regex(0, hi as u64);
                format!("(?:-[1-9][0-9]*|{})", pos)
            }
        }
    }
}

/// Generate regex for positive integers in [min, max] (both non-negative).
fn positive_range_regex(min: u64, max: u64) -> String {
    if min == max {
        return format!("{}", min);
    }

    let min_s = format!("{}", min);
    let max_s = format!("{}", max);

    if min_s.len() == max_s.len() {
        // Same number of digits: build digit-by-digit
        return same_length_range(&min_s, &max_s);
    }

    // Different lengths: split into same-length sub-ranges
    let mut parts = Vec::new();

    // min to 10^(min_digits) - 1
    let first_ceiling = 10u64.pow(min_s.len() as u32) - 1;
    if min <= first_ceiling {
        parts.push(positive_range_regex(min, first_ceiling));
    }

    // Full digit-count ranges
    for d in (min_s.len() + 1)..max_s.len() {
        let lo = 10u64.pow((d - 1) as u32);
        let hi = 10u64.pow(d as u32) - 1;
        parts.push(format!("[1-9][0-9]{{{}}}", d - 1));
        let _ = (lo, hi); // Just for illustration
    }

    // 10^(max_digits-1) to max
    let last_floor = 10u64.pow((max_s.len() - 1) as u32);
    if last_floor <= max && max_s.len() > min_s.len() {
        parts.push(positive_range_regex(last_floor, max));
    }

    if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        format!("(?:{})", parts.join("|"))
    }
}

fn positive_range_regex_unbounded(min: u64) -> String {
    if min == 0 {
        return "(?:0|[1-9][0-9]*)".to_string();
    }
    if min == 1 {
        return "[1-9][0-9]*".to_string();
    }

    let min_s = format!("{}", min);
    let first_ceiling = 10u64.pow(min_s.len() as u32) - 1;

    let mut parts = Vec::new();
    parts.push(positive_range_regex(min, first_ceiling));
    // All larger digit counts
    parts.push(format!("[1-9][0-9]{{{},}}", min_s.len()));

    format!("(?:{})", parts.join("|"))
}

/// Generate regex for numbers with the same number of digits.
fn same_length_range(min: &str, max: &str) -> String {
    let min_digits: Vec<u8> = min.bytes().map(|b| b - b'0').collect();
    let max_digits: Vec<u8> = max.bytes().map(|b| b - b'0').collect();

    build_digit_range(&min_digits, &max_digits, 0)
}

fn build_digit_range(min: &[u8], max: &[u8], pos: usize) -> String {
    if pos >= min.len() {
        return String::new();
    }

    if pos == min.len() - 1 {
        // Last digit
        return digit_range_char(min[pos], max[pos]);
    }

    if min[pos] == max[pos] {
        // Same digit at this position
        let rest = build_digit_range(min, max, pos + 1);
        return format!("{}{}", min[pos], rest);
    }

    let mut parts = Vec::new();

    // Lower boundary: min[pos] with constrained rest
    if min[pos] < max[pos] {
        let lower_max = vec![9u8; min.len() - pos - 1];
        let lower_rest = build_digit_range(&min[pos + 1..], &lower_max, 0);
        if !lower_rest.is_empty() {
            parts.push(format!("{}{}", min[pos], lower_rest));
        }
    }

    // Middle range: (min[pos]+1) to (max[pos]-1) with any digits
    if min[pos] + 1 < max[pos] {
        let mid_range = digit_range_char(min[pos] + 1, max[pos] - 1);
        let remaining = min.len() - pos - 1;
        if remaining > 0 {
            parts.push(format!("{}[0-9]{{{}}}", mid_range, remaining));
        } else {
            parts.push(mid_range);
        }
    }

    // Upper boundary: max[pos] with constrained rest
    if min[pos] < max[pos] {
        let upper_min = vec![0u8; max.len() - pos - 1];
        let upper_rest = build_digit_range(&upper_min, &max[pos + 1..], 0);
        if !upper_rest.is_empty() {
            parts.push(format!("{}{}", max[pos], upper_rest));
        }
    }

    if parts.len() == 1 {
        parts.into_iter().next().unwrap()
    } else {
        format!("(?:{})", parts.join("|"))
    }
}

fn digit_range_char(lo: u8, hi: u8) -> String {
    if lo == hi {
        format!("{}", lo)
    } else if lo + 1 == hi {
        format!("[{}{}]", lo, hi)
    } else {
        format!("[{}-{}]", lo, hi)
    }
}

/// Convert a JSON value to an EBNF string literal.
fn json_value_to_ebnf_literal(val: &Value) -> String {
    let json_str = serde_json::to_string(val).unwrap_or_else(|_| "null".to_string());
    format!("\"{}\"", escape_for_ebnf_string(&json_str))
}

fn escape_for_ebnf_string(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out
}

fn escape_json_string(s: &str) -> String {
    let mut out = String::new();
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            _ => out.push(ch),
        }
    }
    out
}

/// Map a JSON Schema format name to a regex pattern.
/// Returns None for unknown formats (they are silently ignored per JSON Schema spec).
fn format_to_regex(format: &str) -> Option<String> {
    match format {
        "date" => {
            // RFC 3339 full-date: YYYY-MM-DD
            Some(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01])$".to_string())
        }
        "time" => {
            // RFC 3339 full-time: HH:MM:SS[.frac](Z|+HH:MM|-HH:MM)
            Some(r"^([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$".to_string())
        }
        "date-time" => {
            // RFC 3339 date-time: full-date "T" full-time
            Some(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[1-2]\d|3[01])T([01]\d|2[0-3]):[0-5]\d:([0-5]\d|60)(\.\d+)?(Z|[+-]([01]\d|2[0-3]):[0-5]\d)$".to_string())
        }
        "email" => {
            // Simplified RFC 5321: local@domain
            Some(r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$".to_string())
        }
        "uuid" => {
            Some(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$".to_string())
        }
        "ipv4" => {
            Some(r"^((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)$".to_string())
        }
        "hostname" => {
            Some(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?)*$".to_string())
        }
        _ => None,
    }
}

/// Build an EBNF repetition suffix for a given min/max count.
fn repetition_suffix(min: usize, max: Option<usize>) -> String {
    match (min, max) {
        (0, None) => "*".to_string(),
        (0, Some(0)) => String::new(),
        (1, None) => "+".to_string(),
        (0, Some(1)) => "?".to_string(),
        (min, None) => format!("{{{},}}", min),
        (min, Some(max)) if min == max => format!("{{{}}}", min),
        (min, Some(max)) => format!("{{{},{}}}", min, max),
    }
}

fn sanitize_rule_name(name: &str) -> String {
    let mut out = String::new();
    for ch in name.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            out.push(ch);
        } else {
            out.push('_');
        }
    }
    if out.is_empty() {
        out = "rule".to_string();
    }
    out
}

// Tests are in tests/test_json_schema.rs
