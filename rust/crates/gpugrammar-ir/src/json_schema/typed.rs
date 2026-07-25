use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

use super::{
    JsonSchemaOptions, format_to_regex, generate_bounded_number_regex,
    generate_integer_range_regex, parse_i64_keyword, sanitize_rule_name,
};
use crate::frontend::{FrontendExpr as Expr, FrontendGrammar, FrontendRule};
use crate::regex::regex_to_expr;

pub(super) fn convert(schema: &Value, options: &JsonSchemaOptions) -> Result<FrontendGrammar> {
    Converter::new(options).convert(schema)
}

struct Converter<'a> {
    options: &'a JsonSchemaOptions,
    rules: Vec<FrontendRule>,
    names: HashSet<String>,
    counter: usize,
}

impl<'a> Converter<'a> {
    fn new(options: &'a JsonSchemaOptions) -> Self {
        Self {
            options,
            rules: Vec::new(),
            names: HashSet::new(),
            counter: 0,
        }
    }

    fn convert(mut self, schema: &Value) -> Result<FrontendGrammar> {
        self.register_definitions(schema)?;
        let root = self.visit(schema, "root")?;
        self.define_named("root".to_string(), root)?;
        if self.options.any_whitespace {
            self.define_named(
                "__json_ws".to_string(),
                char_class(
                    false,
                    vec![
                        (b'\t' as u32, b'\t' as u32),
                        (b'\n' as u32, b'\n' as u32),
                        (b'\r' as u32, b'\r' as u32),
                        (b' ' as u32, b' ' as u32),
                    ],
                )
                .repeat(0, None),
            )?;
        }
        Ok(FrontendGrammar {
            rules: self.rules,
            root: "root".to_string(),
        })
    }

    fn register_definitions(&mut self, schema: &Value) -> Result<()> {
        let Some(object) = schema.as_object() else {
            return Ok(());
        };
        for keyword in ["definitions", "$defs"] {
            let Some(definitions) = object.get(keyword) else {
                continue;
            };
            let definitions = definitions
                .as_object()
                .ok_or_else(|| anyhow!("{} must be an object", keyword))?;
            for (name, schema) in definitions {
                let name = sanitize_rule_name(name);
                let body = self.visit(schema, &name)?;
                self.define_named(name, body)?;
            }
        }
        Ok(())
    }

    fn visit(&mut self, schema: &Value, hint: &str) -> Result<Expr> {
        if let Some(accepts_all) = schema.as_bool() {
            return if accepts_all {
                self.visit_any(hint)
            } else {
                bail!("false schema: no values are valid")
            };
        }
        let object = schema
            .as_object()
            .ok_or_else(|| anyhow!("schema must be an object or boolean"))?;

        if let Some(reference) = object.get("$ref") {
            let reference = reference
                .as_str()
                .ok_or_else(|| anyhow!("$ref must be a string"))?;
            for prefix in ["#/definitions/", "#/$defs/"] {
                if let Some(name) = reference.strip_prefix(prefix) {
                    return Ok(Expr::RuleRef(sanitize_rule_name(name)));
                }
            }
            bail!("unsupported $ref: {}", reference);
        }
        if let Some(value) = object.get("const") {
            return Ok(json_literal(value));
        }
        if let Some(values) = object.get("enum") {
            let values = values
                .as_array()
                .ok_or_else(|| anyhow!("enum must be an array"))?;
            if values.is_empty() {
                bail!("enum must not be empty");
            }
            return Ok(Expr::choice(values.iter().map(json_literal).collect()));
        }
        if let Some(options) = object.get("anyOf").or_else(|| object.get("oneOf")) {
            let options = options
                .as_array()
                .ok_or_else(|| anyhow!("anyOf/oneOf must be an array"))?;
            return Ok(Expr::choice(
                options
                    .iter()
                    .enumerate()
                    .map(|(index, schema)| self.visit(schema, &format!("{}_{}", hint, index)))
                    .collect::<Result<Vec<_>>>()?,
            ));
        }
        if let Some(all_of) = object.get("allOf") {
            let all_of = all_of
                .as_array()
                .ok_or_else(|| anyhow!("allOf must be an array"))?;
            if all_of.len() != 1 {
                bail!("allOf with multiple schemas is not supported");
            }
            return self.visit(&all_of[0], hint);
        }

        match object.get("type") {
            Some(Value::String(type_name)) => self.visit_typed(schema, type_name, hint),
            Some(Value::Array(types)) => Ok(Expr::choice(
                types
                    .iter()
                    .map(|type_name| {
                        let type_name = type_name
                            .as_str()
                            .ok_or_else(|| anyhow!("type array entries must be strings"))?;
                        self.visit_typed(schema, type_name, hint)
                    })
                    .collect::<Result<Vec<_>>>()?,
            )),
            Some(_) => bail!("unexpected type value"),
            None => {
                if object.contains_key("properties")
                    || object.contains_key("required")
                    || object.contains_key("minProperties")
                    || object.contains_key("maxProperties")
                {
                    self.visit_object(schema, hint)
                } else if object.contains_key("items") || object.contains_key("prefixItems") {
                    self.visit_array(schema, hint)
                } else if object.contains_key("pattern")
                    || object.contains_key("minLength")
                    || object.contains_key("maxLength")
                    || object.contains_key("format")
                {
                    self.visit_string(schema)
                } else if object.contains_key("minimum") || object.contains_key("maximum") {
                    self.visit_number(schema)
                } else {
                    self.visit_any(hint)
                }
            }
        }
    }

    fn visit_typed(&mut self, schema: &Value, type_name: &str, hint: &str) -> Result<Expr> {
        match type_name {
            "string" => self.visit_string(schema),
            "integer" => self.visit_integer(schema),
            "number" => self.visit_number(schema),
            "boolean" => Ok(Expr::choice(vec![lit("true"), lit("false")])),
            "null" => Ok(lit("null")),
            "array" => self.visit_array(schema, hint),
            "object" => self.visit_object(schema, hint),
            _ => bail!("unknown type: {}", type_name),
        }
    }

    fn visit_any(&mut self, hint: &str) -> Result<Expr> {
        let name = self.fresh_name(&format!("{}_value", hint));
        let value = Expr::RuleRef(name.clone());
        let string = self.json_string(0, None);
        let number = unbounded_number();
        let pair = seq(vec![
            string.clone(),
            self.ws(),
            lit(":"),
            self.ws(),
            value.clone(),
        ]);
        let object = seq(vec![
            lit("{"),
            self.ws(),
            optional(seq(vec![
                pair.clone(),
                seq(vec![lit(","), self.ws(), pair]).repeat(0, None),
            ])),
            self.ws(),
            lit("}"),
        ]);
        let array = seq(vec![
            lit("["),
            self.ws(),
            optional(seq(vec![
                value.clone(),
                seq(vec![lit(","), self.ws(), value.clone()]).repeat(0, None),
            ])),
            self.ws(),
            lit("]"),
        ]);
        self.define_named(
            name.clone(),
            Expr::choice(vec![
                object,
                array,
                string,
                number,
                lit("true"),
                lit("false"),
                lit("null"),
            ]),
        )?;
        Ok(Expr::RuleRef(name))
    }

    fn visit_string(&mut self, schema: &Value) -> Result<Expr> {
        let object = schema.as_object().unwrap();
        let min = length_keyword(object, "minLength")?.unwrap_or(0);
        let max = length_keyword(object, "maxLength")?;
        if max.is_some_and(|max| min > max) {
            bail!("minLength is greater than maxLength");
        }
        let has_length = min != 0 || max.is_some();

        if let Some(format) = object.get("format") {
            let format = format
                .as_str()
                .ok_or_else(|| anyhow!("format must be a string"))?;
            if let Some(pattern) = format_to_regex(format) {
                if has_length || object.contains_key("pattern") {
                    bail!("format cannot be combined with pattern or length constraints");
                }
                return Ok(seq(vec![lit("\""), regex_to_expr(&pattern)?, lit("\"")]));
            }
        }
        if let Some(pattern) = object.get("pattern") {
            let pattern = pattern
                .as_str()
                .ok_or_else(|| anyhow!("pattern must be a string"))?;
            if has_length {
                bail!("pattern cannot be combined with length constraints");
            }
            return Ok(seq(vec![lit("\""), regex_to_expr(pattern)?, lit("\"")]));
        }
        Ok(self.json_string(min, max))
    }

    fn json_string(&self, min: u32, max: Option<u32>) -> Expr {
        seq(vec![
            lit("\""),
            json_character().repeat(min, max),
            lit("\""),
        ])
    }

    fn visit_integer(&mut self, schema: &Value) -> Result<Expr> {
        let (min, max) = integer_bounds(schema)?;
        if matches!((min, max), (Some(min), Some(max)) if min > max) {
            bail!("minimum > maximum");
        }
        regex_to_expr(&generate_integer_range_regex(min, max))
    }

    fn visit_number(&mut self, schema: &Value) -> Result<Expr> {
        let object = schema.as_object().unwrap();
        let has_bounds = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
            .iter()
            .any(|keyword| object.contains_key(*keyword));
        if !has_bounds {
            return Ok(unbounded_number());
        }
        if object.contains_key("exclusiveMinimum") || object.contains_key("exclusiveMaximum") {
            bail!("exclusive bounds are not supported for number schemas");
        }
        let min = parse_i64_keyword(object, "minimum")?;
        let max = parse_i64_keyword(object, "maximum")?;
        if matches!((min, max), (Some(min), Some(max)) if min > max) {
            bail!("minimum > maximum");
        }
        regex_to_expr(&generate_bounded_number_regex(min, max))
    }

    fn visit_array(&mut self, schema: &Value, hint: &str) -> Result<Expr> {
        let object = schema.as_object().unwrap();
        let prefix = object
            .get("prefixItems")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .enumerate()
                    .map(|(index, item)| self.visit(item, &format!("{}_item_{}", hint, index)))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?
            .unwrap_or_default();
        let additional = match object.get("items") {
            Some(Value::Bool(false)) => None,
            Some(items) => Some(self.visit(items, &format!("{}_additional", hint))?),
            None if self.options.strict_mode => None,
            None => Some(self.visit_any(hint)?),
        };
        let min = count_keyword(object, "minItems")?.unwrap_or(0);
        let max = count_keyword(object, "maxItems")?;
        if max.is_some_and(|max| min > max) {
            bail!("minItems is greater than maxItems");
        }

        if prefix.is_empty() {
            let Some(item) = additional else {
                return Ok(seq(vec![lit("["), self.ws(), lit("]")]));
            };
            return Ok(seq(vec![
                lit("["),
                self.ws(),
                separated_items(item, min, max, self.ws()),
                self.ws(),
                lit("]"),
            ]));
        }

        let mut content = Vec::new();
        for (index, item) in prefix.iter().cloned().enumerate() {
            if index > 0 {
                content.extend([lit(","), self.ws()]);
            }
            content.push(item);
        }
        if let Some(additional) = additional {
            let prefix_count = prefix.len() as u32;
            let additional_min = min.saturating_sub(prefix_count);
            let additional_max = max.map(|max| max.saturating_sub(prefix_count));
            content.push(
                seq(vec![lit(","), self.ws(), additional]).repeat(additional_min, additional_max),
            );
        } else if min > prefix.len() as u32 || max.is_some_and(|max| max < prefix.len() as u32) {
            bail!("array bounds cannot be satisfied by prefixItems");
        }
        Ok(seq(vec![
            lit("["),
            self.ws(),
            seq(content),
            self.ws(),
            lit("]"),
        ]))
    }

    fn visit_object(&mut self, schema: &Value, hint: &str) -> Result<Expr> {
        let object = schema.as_object().unwrap();
        let properties = object.get("properties").and_then(Value::as_object);
        let required: HashSet<&str> = object
            .get("required")
            .and_then(Value::as_array)
            .map(|required| required.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
        if required
            .iter()
            .any(|name| properties.is_none_or(|properties| !properties.contains_key(*name)))
        {
            bail!("required properties must be declared in properties");
        }
        let min = count_keyword(object, "minProperties")?.unwrap_or(0);
        let max = count_keyword(object, "maxProperties")?;
        if max.is_some_and(|max| min > max) {
            bail!("minProperties is greater than maxProperties");
        }
        if max.is_some_and(|max| required.len() as u32 > max) {
            bail!("number of required properties exceeds maxProperties");
        }

        let additional = match object.get("additionalProperties") {
            Some(Value::Bool(false)) => None,
            Some(Value::Bool(true)) => Some(self.visit_any(hint)?),
            Some(schema) if schema.is_object() => {
                Some(self.visit(schema, &format!("{}_additional", hint))?)
            }
            None if self.options.strict_mode => None,
            None => Some(self.visit_any(hint)?),
            _ => bail!("additionalProperties must be a boolean or schema"),
        };

        let mut known = Vec::new();
        if let Some(properties) = properties {
            for (name, schema) in properties {
                known.push(Property {
                    pair: seq(vec![
                        lit(&serde_json::to_string(name)?),
                        self.ws(),
                        lit(":"),
                        self.ws(),
                        self.visit(schema, &format!("{}_{}", hint, sanitize_rule_name(name)))?,
                    ]),
                    required: required.contains(name.as_str()),
                });
            }
        }
        let additional_pair = additional.map(|value| {
            seq(vec![
                self.json_string(0, None),
                self.ws(),
                lit(":"),
                self.ws(),
                value,
            ])
        });
        if known.is_empty() {
            return Ok(seq(vec![
                lit("{"),
                self.ws(),
                additional_properties(additional_pair, min, max, self.ws())?,
                self.ws(),
                lit("}"),
            ]));
        }

        let content = if known.iter().all(|property| property.required) {
            let mut sequence = intersperse_properties(
                known.iter().map(|property| property.pair.clone()).collect(),
                self.ws(),
            );
            let tail = additional_tail(known.len() as u32, min, max, additional_pair, self.ws())?
                .ok_or_else(|| anyhow!("object property constraints are unsatisfiable"))?;
            if tail != Expr::Empty {
                sequence.push(tail);
            }
            seq(sequence)
        } else if known.iter().filter(|property| !property.required).count() <= 8 {
            enumerate_properties(&known, min, max, additional_pair, self.ws())?
        } else {
            self.build_property_state(
                hint,
                &known,
                0,
                0,
                min,
                max,
                additional_pair,
                self.ws(),
                &mut HashMap::new(),
            )?
            .ok_or_else(|| anyhow!("object property constraints are unsatisfiable"))?
        };

        Ok(seq(vec![lit("{"), self.ws(), content, self.ws(), lit("}")]))
    }

    fn ws(&self) -> Expr {
        if self.options.any_whitespace {
            Expr::RuleRef("__json_ws".to_string())
        } else {
            Expr::Empty
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_property_state(
        &mut self,
        hint: &str,
        properties: &[Property],
        index: usize,
        emitted: u32,
        min: u32,
        max: Option<u32>,
        additional: Option<Expr>,
        ws: Expr,
        memo: &mut HashMap<(usize, u32), Option<Expr>>,
    ) -> Result<Option<Expr>> {
        if max.is_some_and(|max| emitted > max) {
            return Ok(None);
        }
        if let Some(cached) = memo.get(&(index, emitted)) {
            return Ok(cached.clone());
        }
        if index == properties.len() {
            let tail = additional_tail(emitted, min, max, additional, ws)?;
            memo.insert((index, emitted), tail.clone());
            return Ok(tail);
        }

        let mut alternatives = Vec::new();
        if !properties[index].required {
            if let Some(rest) = self.build_property_state(
                hint,
                properties,
                index + 1,
                emitted,
                min,
                max,
                additional.clone(),
                ws.clone(),
                memo,
            )? {
                alternatives.push(rest);
            }
        }
        if max.is_none_or(|max| emitted < max) {
            let next_emitted = if min == 0 && max.is_none() {
                1
            } else {
                emitted + 1
            };
            if let Some(rest) = self.build_property_state(
                hint,
                properties,
                index + 1,
                next_emitted,
                min,
                max,
                additional,
                ws.clone(),
                memo,
            )? {
                let property = if emitted == 0 {
                    properties[index].pair.clone()
                } else {
                    seq(vec![lit(","), ws, properties[index].pair.clone()])
                };
                alternatives.push(seq(vec![property, rest]));
            }
        }

        let result = if alternatives.is_empty() {
            None
        } else {
            let name = self.fresh_name(&format!("{}_properties_{}_{}", hint, index, emitted));
            self.define_named(name.clone(), Expr::choice(alternatives))?;
            Some(Expr::RuleRef(name))
        };
        memo.insert((index, emitted), result.clone());
        Ok(result)
    }

    fn define_named(&mut self, name: String, body: Expr) -> Result<()> {
        if !self.names.insert(name.clone()) {
            bail!("duplicate generated rule '{}'", name);
        }
        self.rules.push(FrontendRule { name, body });
        Ok(())
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        loop {
            self.counter += 1;
            let name = format!("{}_{}", prefix, self.counter);
            if !self.names.contains(&name) {
                return name;
            }
        }
    }
}

#[derive(Clone)]
struct Property {
    pair: Expr,
    required: bool,
}

fn enumerate_properties(
    properties: &[Property],
    min: u32,
    max: Option<u32>,
    additional: Option<Expr>,
    ws: Expr,
) -> Result<Expr> {
    let optional: Vec<usize> = properties
        .iter()
        .enumerate()
        .filter_map(|(index, property)| (!property.required).then_some(index))
        .collect();
    let mut alternatives = Vec::new();
    for selected in 0usize..(1usize << optional.len()) {
        let mut pairs = Vec::new();
        for (index, property) in properties.iter().enumerate() {
            let included = property.required
                || optional
                    .iter()
                    .position(|&optional| optional == index)
                    .is_some_and(|bit| selected & (1 << bit) != 0);
            if included {
                pairs.push(property.pair.clone());
            }
        }
        let emitted = pairs.len() as u32;
        let Some(tail) = additional_tail(emitted, min, max, additional.clone(), ws.clone())? else {
            continue;
        };
        let mut sequence = intersperse_properties(pairs, ws.clone());
        if tail != Expr::Empty {
            sequence.push(tail);
        }
        alternatives.push(seq(sequence));
    }
    if alternatives.is_empty() {
        bail!("object property constraints are unsatisfiable");
    }
    Ok(Expr::choice(alternatives))
}

fn additional_properties(
    additional: Option<Expr>,
    min: u32,
    max: Option<u32>,
    ws: Expr,
) -> Result<Expr> {
    additional_tail(0, min, max, additional, ws)?
        .ok_or_else(|| anyhow!("object property constraints are unsatisfiable"))
}

fn additional_tail(
    emitted: u32,
    min: u32,
    max: Option<u32>,
    additional: Option<Expr>,
    ws: Expr,
) -> Result<Option<Expr>> {
    if max.is_some_and(|max| emitted > max) {
        return Ok(None);
    }
    let needed = min.saturating_sub(emitted);
    let allowed = max.map(|max| max - emitted);
    let Some(additional) = additional else {
        return if needed == 0 {
            Ok(Some(Expr::Empty))
        } else {
            Ok(None)
        };
    };
    if allowed == Some(0) {
        return if needed == 0 {
            Ok(Some(Expr::Empty))
        } else {
            Ok(None)
        };
    }

    if emitted > 0 {
        return Ok(Some(
            seq(vec![lit(","), ws, additional]).repeat(needed, allowed),
        ));
    }
    Ok(Some(separated_items(additional, needed, allowed, ws)))
}

fn separated_items(item: Expr, min: u32, max: Option<u32>, ws: Expr) -> Expr {
    if max == Some(0) {
        return Expr::Empty;
    }
    let rest = seq(vec![lit(","), ws, item.clone()])
        .repeat(min.saturating_sub(1), max.map(|max| max.saturating_sub(1)));
    let sequence = seq(vec![item, rest]);
    if min == 0 {
        optional(sequence)
    } else {
        sequence
    }
}

fn intersperse_properties(properties: Vec<Expr>, ws: Expr) -> Vec<Expr> {
    let mut result = Vec::new();
    for (index, property) in properties.into_iter().enumerate() {
        if index > 0 {
            result.extend([lit(","), ws.clone()]);
        }
        result.push(property);
    }
    result
}

fn integer_bounds(schema: &Value) -> Result<(Option<i64>, Option<i64>)> {
    let object = schema.as_object().unwrap();
    let inclusive_min = parse_i64_keyword(object, "minimum")?;
    let exclusive_min = parse_i64_keyword(object, "exclusiveMinimum")?
        .map(|value| {
            value
                .checked_add(1)
                .ok_or_else(|| anyhow!("exclusiveMinimum leaves no valid i64 integer"))
        })
        .transpose()?;
    let min = match (inclusive_min, exclusive_min) {
        (Some(inclusive), Some(exclusive)) => Some(inclusive.max(exclusive)),
        (inclusive, exclusive) => inclusive.or(exclusive),
    };

    let inclusive_max = parse_i64_keyword(object, "maximum")?;
    let exclusive_max = parse_i64_keyword(object, "exclusiveMaximum")?
        .map(|value| {
            value
                .checked_sub(1)
                .ok_or_else(|| anyhow!("exclusiveMaximum leaves no valid i64 integer"))
        })
        .transpose()?;
    let max = match (inclusive_max, exclusive_max) {
        (Some(inclusive), Some(exclusive)) => Some(inclusive.min(exclusive)),
        (inclusive, exclusive) => inclusive.or(exclusive),
    };
    Ok((min, max))
}

fn length_keyword(object: &serde_json::Map<String, Value>, name: &str) -> Result<Option<u32>> {
    count_keyword(object, name)
}

fn count_keyword(object: &serde_json::Map<String, Value>, name: &str) -> Result<Option<u32>> {
    object
        .get(name)
        .map(|value| {
            let value = value
                .as_u64()
                .ok_or_else(|| anyhow!("{} must be a non-negative integer", name))?;
            u32::try_from(value).map_err(|_| anyhow!("{} exceeds u32::MAX", name))
        })
        .transpose()
}

fn json_character() -> Expr {
    let unescaped = char_class(
        true,
        vec![
            (0, 0x1f),
            (b'"' as u32, b'"' as u32),
            (b'\\' as u32, b'\\' as u32),
        ],
    );
    let simple_escape = Expr::choice(
        ["\"", "\\", "/", "b", "f", "n", "r", "t"]
            .into_iter()
            .map(lit)
            .collect(),
    );
    let hex = char_class(
        false,
        vec![
            (b'0' as u32, b'9' as u32),
            (b'A' as u32, b'F' as u32),
            (b'a' as u32, b'f' as u32),
        ],
    );
    let unicode_escape = seq(vec![lit("u"), hex.clone(), hex.clone(), hex.clone(), hex]);
    Expr::choice(vec![
        unescaped,
        seq(vec![
            lit("\\"),
            Expr::choice(vec![simple_escape, unicode_escape]),
        ]),
    ])
}

fn unbounded_number() -> Expr {
    regex_to_expr(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
        .expect("builtin number regex is valid")
}

fn json_literal(value: &Value) -> Expr {
    lit(serde_json::to_string(value).expect("JSON values serialize"))
}

fn lit(value: impl AsRef<str>) -> Expr {
    Expr::literal(value.as_ref().as_bytes().to_vec())
}

fn seq(elements: Vec<Expr>) -> Expr {
    Expr::sequence(elements)
}

fn optional(expr: Expr) -> Expr {
    expr.repeat(0, Some(1))
}

fn char_class(negated: bool, ranges: Vec<(u32, u32)>) -> Expr {
    Expr::CharacterClass { negated, ranges }
}
