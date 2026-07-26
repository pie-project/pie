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
    /// Lexeme rules already declared, keyed by shape.
    lexemes: HashMap<String, String>,
}

impl<'a> Converter<'a> {
    fn new(options: &'a JsonSchemaOptions) -> Self {
        Self {
            options,
            rules: Vec::new(),
            names: HashSet::new(),
            counter: 0,
            lexemes: HashMap::new(),
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

    /// A value, followed by whatever whitespace comes after it.
    ///
    /// JSON allows whitespace on both sides of every separator, but a grammar
    /// that writes it on both sides puts two optionals next to each other -
    /// `value ws? ws? ","` - and the parser cannot tell which one took it. LALR
    /// reports that as a conflict, and resolving it is not free: it costs the
    /// schemas that then fail to compile. Attaching the whitespace to the value
    /// instead gives every position exactly one place to put it, so the same
    /// language is described without the ambiguity.
    fn visit(&mut self, schema: &Value, hint: &str) -> Result<Expr> {
        let value = self.visit_bare(schema, hint)?;
        Ok(seq(vec![value, self.ws()]))
    }

    fn visit_bare(&mut self, schema: &Value, hint: &str) -> Result<Expr> {
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
            // `#` is the document itself, which is the root rule. A schema
            // that refers to it is recursive, and the root is already a named
            // rule, so the reference resolves like any other.
            if reference == "#" || reference == "#/" {
                return Ok(Expr::RuleRef("root".to_string()));
            }
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
            // `{"properties": {...}, "anyOf": [{"required": ["a"]},
            // {"required": ["b"]}]}` means "this object, and it has a or b".
            // Lowering the branches alone turns each into "any JSON value" and
            // throws the object away, so the siblings have to come along.
            //
            // Distributing them one branch at a time - `(S and B1) or (S and
            // B2)` - is exact but unparseable: both alternatives begin with the
            // same property and differ only in which ones may be left out, so
            // an LALR parser cannot tell them apart until far past one token of
            // lookahead. Requiring only what every branch requires collapses
            // them to a single object with no choice to resolve. That accepts
            // documents satisfying none of the branches, which is the same
            // direction the parser already errs in for `anyOf`, and it is the
            // difference between a usable grammar and none at all.
            if self.options.precision.merges_branches()
                && let Some(merged) = merge_required_choice(schema, options)?
            {
                return self.visit(&merged, hint);
            }
            return Ok(Expr::choice(
                options
                    .iter()
                    .enumerate()
                    .map(|(index, branch)| self.visit(branch, &format!("{}_{}", hint, index)))
                    .collect::<Result<Vec<_>>>()?,
            ));
        }
        if let Some(all_of) = object.get("allOf") {
            let all_of = all_of
                .as_array()
                .ok_or_else(|| anyhow!("allOf must be an array"))?;
            if all_of.len() == 1 {
                return self.visit(&all_of[0], hint);
            }
            let merged = merge_all_of(schema, all_of)?;
            return self.visit(&merged, hint);
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
        let string = self.json_string(0, None)?;
        let number = self.lexeme("number", unbounded_number())?;
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
            lit("}"),
        ]);
        let array = seq(vec![
            lit("["),
            self.ws(),
            optional(seq(vec![
                value.clone(),
                seq(vec![lit(","), self.ws(), value.clone()]).repeat(0, None),
            ])),
            lit("]"),
        ]);
        self.define_named(
            name.clone(),
            // Every alternative ends with the whitespace that follows it, so
            // the separator that comes next needs none of its own.
            Expr::choice(
                [
                    object,
                    array,
                    string,
                    number,
                    lit("true"),
                    lit("false"),
                    lit("null"),
                ]
                .into_iter()
                .map(|alternative| seq(vec![alternative, self.ws()]))
                .collect(),
            ),
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
            // A grammar cannot intersect two languages, so a pattern and a
            // length bound can only be lowered together when one of them makes
            // the other redundant. That is decidable: the pattern's own length
            // range is computable, and when it lies inside the bound the bound
            // says nothing the pattern does not already say.
            if has_length {
                let expr = regex_to_expr(pattern)?;
                let (shortest, longest) = length_range(&expr);
                let within_min = shortest >= u64::from(min);
                let within_max = max.is_none_or(|max| longest.is_some_and(|l| l <= u64::from(max)));
                if !(within_min && within_max) {
                    bail!(
                        "pattern matches strings of length {shortest}..{} and the \
                         schema also bounds the length, which needs an intersection",
                        longest.map_or("unbounded".to_string(), |l| l.to_string())
                    );
                }
                return Ok(seq(vec![lit("\""), expr, lit("\"")]));
            }
            let body = seq(vec![lit("\""), regex_to_expr(pattern)?, lit("\"")]);
            return self.lexeme("pattern", body);
        }
        self.json_string(min, max)
    }

    /// A property name that a regex has to accept.
    ///
    /// JSON Schema patterns are unanchored - `"ab"` matches the key
    /// `"xaby"` - but the regex frontend reads them as anchored, so an
    /// unanchored pattern has to be widened explicitly. The padding is the
    /// JSON string body rather than `.`, because `.` includes the quote that
    /// ends the key and the lexer would run straight through it.
    fn pattern_key(&mut self, pattern: &str) -> Result<Expr> {
        let anchored_start = pattern.starts_with('^');
        let anchored_end = pattern.ends_with('$') && !pattern.ends_with("\\$");
        let body = regex_to_expr(pattern)?;
        let mut parts = vec![lit("\"")];
        if !anchored_start {
            parts.push(json_character().repeat(0, None));
        }
        parts.push(body);
        if !anchored_end {
            parts.push(json_character().repeat(0, None));
        }
        parts.push(lit("\""));
        self.lexeme("key", seq(parts))
    }

    fn json_string(&mut self, min: u32, max: Option<u32>) -> Result<Expr> {
        self.lexeme(
            "string",
            seq(vec![
                lit("\""),
                json_character().repeat(min, max),
                lit("\""),
            ]),
        )
    }

    /// Declare a lexical unit as its own rule.
    ///
    /// The lexer takes a whole regular rule as one terminal, so this is how the
    /// front end says where a lexeme begins and ends. Left inline, a string
    /// would reach the lexer as `'"'`, a body and `'"'`, and the body class
    /// `[^"\\]*` overlaps every punctuation terminal in the grammar: after a
    /// colon the scanner would keep munching as a string body and never commit
    /// the colon the parser wanted. Identical shapes share one rule.
    fn lexeme(&mut self, hint: &str, body: Expr) -> Result<Expr> {
        let key = format!("{body:?}");
        if let Some(name) = self.lexemes.get(&key) {
            return Ok(Expr::RuleRef(name.clone()));
        }
        let name = self.fresh_name(&format!("{hint}_lexeme"));
        self.lexemes.insert(key, name.clone());
        self.define_named(name.clone(), body)?;
        Ok(Expr::RuleRef(name))
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
        Ok(seq(vec![lit("["), self.ws(), seq(content), lit("]")]))
    }

    fn visit_object(&mut self, schema: &Value, hint: &str) -> Result<Expr> {
        let object = schema.as_object().unwrap();
        let properties = object.get("properties").and_then(Value::as_object);
        let required: HashSet<&str> = object
            .get("required")
            .and_then(Value::as_array)
            .map(|required| required.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();
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

        // `required` may name a property `properties` does not describe. That
        // is not a contradiction: it says the property must be present, and
        // what it may hold is whatever `additionalProperties` allows. Only when
        // additional properties are forbidden is the schema unsatisfiable.
        let mut undeclared: Vec<&str> = required
            .iter()
            .copied()
            .filter(|name| properties.is_none_or(|declared| !declared.contains_key(*name)))
            .collect();
        undeclared.sort_unstable();
        if !undeclared.is_empty() {
            let Some(value) = additional.clone() else {
                bail!("a required property is neither declared nor allowed as additional");
            };
            for name in undeclared {
                known.push(Property {
                    pair: seq(vec![
                        lit(&serde_json::to_string(name)?),
                        self.ws(),
                        lit(":"),
                        self.ws(),
                        value.clone(),
                    ]),
                    required: true,
                });
            }
        }
        // `patternProperties` names its keys by a regex instead of literally,
        // which is the one place a JSON Schema object says something a fixed
        // list of properties cannot. It behaves like `additionalProperties`
        // except that the key is constrained, so it lowers the same way: a
        // pair that may repeat. A schema that pairs it with
        // `additionalProperties: false` allows *only* these keys, which is why
        // the pattern pairs have to survive `additional` being absent.
        let mut repeatable = Vec::new();
        for (pattern, sub) in object
            .get("patternProperties")
            .map(|value| {
                value
                    .as_object()
                    .ok_or_else(|| anyhow!("patternProperties must be an object"))
            })
            .transpose()?
            .into_iter()
            .flatten()
        {
            let key = self.pattern_key(pattern)?;
            let value = self.visit(sub, &format!("{}_pattern", hint))?;
            repeatable.push(seq(vec![key, self.ws(), lit(":"), self.ws(), value]));
        }
        if let Some(value) = additional {
            let name = self.json_string(0, None)?;
            repeatable.push(seq(vec![name, self.ws(), lit(":"), self.ws(), value]));
        }
        let additional_pair = match repeatable.len() {
            0 => None,
            _ => Some(Expr::choice(repeatable)),
        };
        if known.is_empty() {
            return Ok(seq(vec![
                lit("{"),
                self.ws(),
                additional_properties(additional_pair, min, max, self.ws())?,
                lit("}"),
            ]));
        }

        // A JSON object is a set, not a sequence, but a grammar can only
        // describe a sequence. The usual answer - XGrammar's too - is to fix
        // the order at the one the schema declares, which rejects every other
        // permutation of a perfectly valid document.
        //
        // It can be done exactly. What the order was standing in for is the
        // question "have the required properties appeared yet", and that is a
        // subset of the required set, not an ordering of everything. Carrying
        // the subset in the parser state makes the properties free to arrive in
        // any order while `required` is still enforced. The cost is one rule
        // per subset, so it is affordable exactly while the required set is
        // small - which, in practice, is nearly always.
        if min <= required.len() as u32
            && max.is_none()
            && self.options.precision.unordered()
            && let Some(object) = self.build_unordered(hint, &known, &additional_pair)?
        {
            return Ok(object);
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
        } else {
            // Never enumerate subsets of the optional properties. That is
            // exponential in their number and, worse, copies every property's
            // value grammar into every subset, so a schema with eight optional
            // properties duplicates each value 256 times. The state
            // construction shares them behind one rule per (index, count).
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

        Ok(seq(vec![lit("{"), self.ws(), content, lit("}")]))
    }

    /// Objects whose properties may arrive in any order.
    ///
    /// Two rules per subset of the required properties: `item` picks the next
    /// property, `tail` decides whether the object ends. A required property
    /// moves to a larger subset, an optional or additional one stays put, and
    /// only the full subset lets `tail` be empty, so the object closes exactly
    /// when everything required has been seen.
    ///
    /// `None` when the required set is too large to enumerate - the caller
    /// then falls back to the declared order.
    fn build_unordered(
        &mut self,
        hint: &str,
        known: &[Property],
        additional: &Option<Expr>,
    ) -> Result<Option<Expr>> {
        let required: Vec<usize> = known
            .iter()
            .enumerate()
            .filter(|(_, property)| property.required)
            .map(|(index, _)| index)
            .collect();
        // A declared name also scans as a generic one, so where both are
        // possible the matcher carries a configuration per reading. The
        // readings differ only in the subset they claim to have completed, so
        // the count is bounded by the number of subsets rather than by the
        // number of properties - and it collapses to one as soon as the names
        // are a closed set, because then there is no generic reading to fork
        // on. The budget is therefore the matcher's configuration budget seen
        // from the other side, and it is tighter when a generic key exists.
        let budget = match additional {
            Some(_) => UNORDERED_REQUIRED_BUDGET_OPEN,
            None => UNORDERED_REQUIRED_BUDGET_CLOSED,
        };
        if required.len() > budget {
            return Ok(None);
        }
        let full: u32 = (1u32 << required.len()) - 1;

        // The rules refer to each other, so every name has to exist before any
        // body can be written - and nothing may be committed until they all
        // do, or a later refusal would leave the earlier rules pointing at
        // names that never get defined.
        let items: Vec<String> = (0..=full)
            .map(|mask| self.fresh_name(&format!("{hint}_item_{mask}")))
            .collect();
        let tails: Vec<String> = (0..=full)
            .map(|mask| self.fresh_name(&format!("{hint}_tail_{mask}")))
            .collect();
        let mut pending: Vec<(String, Expr)> = Vec::new();

        for mask in 0..=full {
            let more = seq(vec![
                lit(","),
                self.ws(),
                Expr::RuleRef(items[mask as usize].clone()),
            ]);
            // Only the full subset may stop: anything less is an object still
            // missing a property the schema requires.
            let tail = if mask == full { optional(more) } else { more };
            pending.push((tails[mask as usize].clone(), tail));

            let mut item = Vec::new();
            for (index, property) in known.iter().enumerate() {
                let next = match required.iter().position(|&r| r == index) {
                    Some(bit) if mask & (1 << bit) != 0 => continue,
                    Some(bit) => mask | (1 << bit),
                    None => mask,
                };
                item.push(seq(vec![
                    property.pair.clone(),
                    Expr::RuleRef(tails[next as usize].clone()),
                ]));
            }
            if let Some(additional) = additional {
                item.push(seq(vec![
                    additional.clone(),
                    Expr::RuleRef(tails[mask as usize].clone()),
                ]));
            }
            if item.is_empty() {
                return Ok(None);
            }
            pending.push((items[mask as usize].clone(), Expr::choice(item)));
        }
        for (name, body) in pending {
            self.define_named(name, body)?;
        }

        let content = Expr::RuleRef(items[0].clone());
        // With nothing required the object may also be empty.
        let content = if full == 0 {
            optional(content)
        } else {
            content
        };
        Ok(Some(seq(vec![lit("{"), self.ws(), content, lit("}")])))
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

/// How many required properties an order-free object will enumerate subsets
/// of when `additionalProperties` leaves the names open. Each subset is a
/// parse the matcher may have to carry at once, so this is bounded by its
/// configuration budget; four covers 96% of the objects in JSONSchemaBench.
const UNORDERED_REQUIRED_BUDGET_OPEN: usize = 4;

/// The same, for objects whose property names are a closed set. Nothing forks
/// there, so the only cost is grammar size and the budget can be looser.
const UNORDERED_REQUIRED_BUDGET_CLOSED: usize = 6;

#[derive(Clone)]
struct Property {
    pair: Expr,
    required: bool,
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

/// The shortest and longest strings an expression matches, in bytes.
///
/// `None` for the longest means unbounded. Used to decide whether a length
/// bound adds anything to a pattern that already constrains the length.
fn length_range(expr: &Expr) -> (u64, Option<u64>) {
    match expr {
        Expr::Empty => (0, Some(0)),
        Expr::Literal(bytes) => (bytes.len() as u64, Some(bytes.len() as u64)),
        // A character class is one codepoint, and UTF-8 spends up to four bytes
        // on one. Both ends are needed, since the bound is over characters.
        Expr::CharacterClass { .. } => (1, Some(4)),
        // A rule reference could be anything without resolving it, and guessing
        // would defeat the point of deciding rather than approximating.
        Expr::RuleRef(_) => (0, None),
        Expr::Group(inner) => length_range(inner),
        Expr::Sequence(parts) => parts.iter().fold((0, Some(0)), |(low, high), part| {
            let (part_low, part_high) = length_range(part);
            (
                low.saturating_add(part_low),
                high.zip(part_high).map(|(a, b)| a.saturating_add(b)),
            )
        }),
        Expr::Choice(alternatives) => {
            alternatives
                .iter()
                .fold((u64::MAX, Some(0)), |(low, high), alternative| {
                    let (alt_low, alt_high) = length_range(alternative);
                    (low.min(alt_low), high.zip(alt_high).map(|(a, b)| a.max(b)))
                })
        }
        Expr::Repeat { expr, min, max } => {
            let (inner_low, inner_high) = length_range(expr);
            (
                inner_low.saturating_mul(u64::from(*min)),
                max.and_then(|max| inner_high.map(|high| high.saturating_mul(u64::from(max)))),
            )
        }
    }
}

/// Fold an `anyOf` whose branches only list required properties.
///
/// Returns `None` when the branches say anything else, since then they are
/// real alternatives and have to stay a choice.
fn merge_required_choice(schema: &Value, options: &[Value]) -> Result<Option<Value>> {
    if options.is_empty() {
        return Ok(None);
    }
    let mut shared: Option<Vec<Value>> = None;
    for branch in options {
        let Some(branch) = branch.as_object() else {
            return Ok(None);
        };
        if branch.keys().any(|key| key != "required") {
            return Ok(None);
        }
        let required = match branch.get("required") {
            Some(value) => value
                .as_array()
                .ok_or_else(|| anyhow!("required must be an array"))?
                .clone(),
            None => Vec::new(),
        };
        shared = Some(match shared {
            None => required,
            Some(previous) => previous
                .into_iter()
                .filter(|name| required.contains(name))
                .collect(),
        });
    }

    let mut merged = strip_keys(schema, &["anyOf", "oneOf"]);
    let object = merged.as_object_mut().expect("built from an object");
    if !describes_shape(&Value::Object(object.clone())) {
        return Ok(None);
    }
    let mut required = object
        .get("required")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    for name in shared.unwrap_or_default() {
        if !required.contains(&name) {
            required.push(name);
        }
    }
    object.insert("required".to_string(), Value::Array(required));
    Ok(Some(merged))
}

/// Does this schema constrain what a value looks like?
///
/// `required`, `minProperties` and friends restrict an object without saying
/// it is one, so a schema built only from those admits every JSON value.
fn describes_shape(schema: &Value) -> bool {
    let Some(object) = schema.as_object() else {
        return true;
    };
    const SHAPE: &[&str] = &[
        "type",
        "properties",
        "patternProperties",
        "items",
        "prefixItems",
        "enum",
        "const",
        "pattern",
        "format",
        "$ref",
        "anyOf",
        "oneOf",
        "allOf",
    ];
    object.keys().any(|key| SHAPE.contains(&key.as_str()))
}

/// A schema with some keywords removed.
fn strip_keys(schema: &Value, drop: &[&str]) -> Value {
    Value::Object(
        schema
            .as_object()
            .expect("checked by the caller")
            .iter()
            .filter(|(key, _)| !drop.contains(&key.as_str()))
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    )
}

/// Combine `allOf` branches into one schema.
///
/// A grammar cannot intersect two languages, so the branches have to be merged
/// before lowering. That is exact for the shape `allOf` is actually used in -
/// several partial object descriptions, each naming properties and requirements
/// the others do not - because a conjunction of those is the union of their
/// properties and requirements. It is refused rather than approximated whenever
/// two branches say something different about the same thing, since a mask that
/// is nearly right is a mask that lets an invalid token through.
fn merge_all_of(parent: &Value, branches: &[Value]) -> Result<Value> {
    let mut merged = serde_json::Map::new();
    for (key, value) in parent.as_object().expect("checked by the caller") {
        if key != "allOf" {
            merged.insert(key.clone(), value.clone());
        }
    }

    for branch in branches {
        let branch = branch
            .as_object()
            .ok_or_else(|| anyhow!("allOf branches must be objects"))?;
        for (key, value) in branch {
            match (key.as_str(), merged.get_mut(key)) {
                (_, None) => {
                    merged.insert(key.clone(), value.clone());
                }
                ("properties", Some(Value::Object(into))) => {
                    for (name, schema) in value
                        .as_object()
                        .ok_or_else(|| anyhow!("properties must be an object"))?
                    {
                        if into.get(name).is_some_and(|existing| existing != schema) {
                            bail!("allOf branches disagree about property '{name}'");
                        }
                        into.insert(name.clone(), schema.clone());
                    }
                }
                ("required", Some(Value::Array(into))) => {
                    for name in value
                        .as_array()
                        .ok_or_else(|| anyhow!("required must be an array"))?
                    {
                        if !into.contains(name) {
                            into.push(name.clone());
                        }
                    }
                }
                (_, Some(existing)) if existing == value => {}
                (key, _) => bail!("allOf branches disagree about '{key}'"),
            }
        }
    }
    Ok(Value::Object(merged))
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
