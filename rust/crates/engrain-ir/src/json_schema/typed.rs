use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{Result, anyhow, bail};
use serde_json::Value;

use super::{
    JsonSchemaOptions, digit_tail, format_to_regex, generate_bounded_number_regex,
    generate_integer_range_regex, parse_i64_keyword, sanitize_rule_name,
};
use crate::frontend::{FrontendExpr as Expr, FrontendGrammar, FrontendRule};
use crate::regex::regex_to_expr;

pub(super) fn convert(schema: &Value, options: &JsonSchemaOptions) -> Result<FrontendGrammar> {
    Converter::new(options, schema).convert(schema)
}

struct Converter<'a> {
    options: &'a JsonSchemaOptions,
    rules: Vec<FrontendRule>,
    names: HashSet<String>,
    /// The whole schema, so a `$ref` can be resolved as the JSON pointer it is.
    document: &'a Value,
    /// Pointers already turned into rules, so two references to the same place
    /// share one and a recursive one terminates.
    pointers: HashSet<String>,
    counter: usize,
    /// Lexeme rules already declared, keyed by shape.
    lexemes: HashMap<String, String>,
}

impl<'a> Converter<'a> {
    fn new(options: &'a JsonSchemaOptions, document: &'a Value) -> Self {
        Self {
            options,
            rules: Vec::new(),
            names: HashSet::new(),
            document,
            pointers: HashSet::new(),
            counter: 0,
            lexemes: HashMap::new(),
        }
    }

    fn convert(mut self, schema: &Value) -> Result<FrontendGrammar> {
        self.register_definitions(schema)?;
        // `visit` attaches whitespace *after* every value so that each position
        // has exactly one place to put it. At the root that trailing run has
        // nothing after it to separate, and it is the difference between a
        // document that ends and one that does not: a model handed a mask that
        // still admits a space will emit one, and then another, until it runs
        // out of budget. Measured through vLLM at batch 256, that was 24,576
        // tokens generated against XGrammar's 6,635 for the same requests, and
        // 1.86 s against 0.85.
        //
        // So the root is the bare value. This narrows the language by the
        // trailing whitespace JSON would allow around a document - which is
        // deliberate, is what XGrammar does too, and cannot reject a document,
        // only its padding.
        let root = self.visit_bare(schema, "root")?;
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
                // Only a direct child of the root's definitions block, which is
                // what `register_definitions` declares. A pointer that goes on
                // through nested blocks - `#/definitions/a/definitions/b`, which
                // this corpus has plenty of - sanitises to a name nobody
                // defines, and that was most of the "undefined frontend rule"
                // refusals. Those fall through to the general pointer path.
                if let Some(name) = reference.strip_prefix(prefix)
                    && !name.contains('/') {
                        return Ok(Expr::RuleRef(sanitize_rule_name(name)));
                    }
            }
            // Any other pointer into the document. `$ref` is a JSON pointer,
            // not a name in a definitions block, and schemas in the wild point
            // at properties, items and definition blocks under other names -
            // `#/properties/author`, `#/defs/scope`. Resolving the pointer and
            // lowering what it finds treats all of them alike, and the result
            // is named after the pointer so that two references to the same
            // place share one rule and a recursive one terminates.
            if let Some(pointer) = reference.strip_prefix('#') {
                let name = format!("ref_{}", sanitize_rule_name(pointer));
                if !self.pointers.insert(name.clone()) {
                    return Ok(Expr::RuleRef(name));
                }
                let target = resolve_pointer(self.document, pointer)
                    .ok_or_else(|| anyhow!("$ref points nowhere: {reference}"))?
                    .clone();
                let body = self.visit(&target, &name)?;
                self.define_named(name.clone(), body)?;
                return Ok(Expr::RuleRef(name));
            }
            bail!("unsupported $ref: {}", reference);
        }
        // `not` is a complement, and a grammar cannot complement a language.
        // The front end has always lowered the rest of the schema and let the
        // `not` go, which is a widening and so the direction a mask may err in -
        // but it did so silently, and silence is what makes a limitation into a
        // surprise. Two shapes are worth treating properly.
        if let Some(negated) = object.get("not") {
            // `not {}` and `not true` accept nothing, since `{}` accepts
            // everything. That is exact rather than a widening, and it is how
            // schemas spell "no additional properties" when they want to be
            // pedantic about it.
            if negated == &Value::Object(Default::default()) || negated == &Value::Bool(true) {
                // Accepts nothing. Some positions can say that exactly - as
                // `additionalProperties` it is `false`, as a branch of a choice
                // it is a branch that goes - and those are handled where the
                // position is known. Reaching here means the position cannot,
                // and a grammar has no empty language to put in its place, so
                // this widens to any value rather than losing the schema. That
                // is the direction a mask may err in, and refusing outright
                // costs seven schemas that otherwise compile.
                return self.visit_any(hint);
            }
            // Anything else: lower what is left and record that the complement
            // was dropped. `soundness.py` attributes its failures to this.
            let rest = strip_keys(schema, &["not"]);
            return self.visit_bare(&rest, hint);
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
            // A branch that accepts nothing contributes nothing to a union, so
            // dropping it is exact - and it is one fewer production for the
            // parser to tell apart.
            let kept: Vec<Value> = options
                .iter()
                .filter(|branch| !is_unsatisfiable(branch))
                .cloned()
                .collect();
            let options = &kept;
            if options.is_empty() {
                bail!("every branch of the choice accepts nothing");
            }
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
            // The same idea, applied to branches that differ in more than what
            // they require. Two object branches are two productions that begin
            // with the same brace and the same property names, and an LALR
            // parser cannot tell which to reduce by - reduce/reduce conflicts
            // are the largest single reason a schema has no parser here. One
            // object naming every property any branch names, required only
            // where every branch requires it, has no choice left to resolve.
            if self.options.precision.merges_objects()
                && let Some(merged) = merge_object_choice(schema, options)
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
            let merged = merge_all_of(self.document, schema, all_of)?;
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
        let number = self.lexeme("number", unbounded_number(self.options.max_digits))?;
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
                // The bound may already be implied by the pattern, in which
                // case there is nothing to do; otherwise push it into the
                // repeat it constrains, which is exact where it applies.
                let (shortest, longest) = character_range(&expr);
                let within_min = shortest >= u64::from(min);
                let within_max = max.is_none_or(|max| longest.is_some_and(|l| l <= u64::from(max)));
                if !(within_min && within_max) {
                    if let Some(narrowed) = constrain_to_length(&expr, min, max) {
                        return Ok(seq(vec![lit("\""), narrowed, lit("\"")]));
                    }
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
        regex_to_expr(&generate_integer_range_regex(min, max, self.options.max_digits))
    }

    fn visit_number(&mut self, schema: &Value) -> Result<Expr> {
        let object = schema.as_object().unwrap();
        let has_bounds = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
            .iter()
            .any(|keyword| object.contains_key(*keyword));
        if !has_bounds {
            return Ok(unbounded_number(self.options.max_digits));
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
            // `{"not": {}}` accepts nothing, since `{}` accepts everything, so
            // as `additionalProperties` it says exactly what `false` says. This
            // is how a schema spells it when it wants to be pedantic, and it is
            // the shape most of the corpus's uses of `not` take.
            Some(schema) if is_unsatisfiable(schema) => None,
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
        // Whether a generic key can spell a name this object declares. False
        // means the two readings are disjoint and nothing forks here, which is
        // what lets `build_unordered` use the larger budget.
        let mut disjoint = true;
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
            // Every name `properties` declares is governed by that declaration
            // and not by `additionalProperties`, so this arm must exclude
            // them. Doing so is also what stops a declared name from having
            // two readings - the literal and the generic key - which is one of
            // the two things that forks a configuration at run time.
            let declared: Vec<String> = properties
                .into_iter()
                .flat_map(|declared| declared.keys())
                .cloned()
                .collect();
            // Per object, over its own names. One complement over every name
            // the *document* declares was tried - one terminal for the schema
            // instead of one per object, with each object handing back the
            // names it does not declare - on the theory that terminals are
            // what compile time is spent on. They are, but not by count: over
            // 40 corpus schemas the shared shape took the median group count
            // from 28,389 to 85,366 and the median compile from 102.7 ms to
            // 478.9, because the complement of ninety-six names is a far
            // larger automaton than the complement of three, and every state
            // of it is a lexer state with an admitted set of its own.
            let excluded = self
                .options
                .precision
                .excludes_declared_names()
                .then(|| string_body_excluding(&declared))
                .flatten();
            disjoint = excluded.is_some() || declared.is_empty();
            let name = match excluded {
                Some(body) => self.lexeme("key", seq(vec![lit("\""), body, lit("\"")]))?,
                None => self.json_string(0, None)?,
            };
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
        if self.options.precision.enforces_counting()
            && min <= required.len() as u32
            && max.is_none()
            && let Some(object) =
                self.build_unordered(hint, &known, &additional_pair, disjoint)?
        {
            return Ok(object);
        }
        // This object alone widens; the rest of the schema stays exact. The
        // fallback is deliberately local, because the alternative - refusing
        // the schema so the search retries it at a coarser level - relaxes
        // every other object too, and those did fit.
        Ok(self.relaxed_object(&known, additional_pair))
    }

    /// The shape of an object without any of its counting constraints.
    ///
    /// Any declared pair, or an additional one where the schema allows it, in
    /// any order and any number of times. `required`, `minProperties` and
    /// `maxProperties` are not enforced - they are exactly the keywords that
    /// need a tally in the parser state, and a tally is what the exact
    /// lowering could not afford here.
    ///
    /// This is a superset of the schema, so a caller that needs those three
    /// keywords must check the finished document. It is never a subset, which
    /// is the property that matters: the model can still produce every
    /// document the schema allows.
    fn relaxed_object(&mut self, known: &[Property], additional: Option<Expr>) -> Expr {
        let mut alternatives: Vec<Expr> =
            known.iter().map(|property| property.pair.clone()).collect();
        if let Some(additional) = additional {
            alternatives.push(additional);
        }
        let body = if alternatives.is_empty() {
            Expr::Empty
        } else {
            let pair = if alternatives.len() == 1 {
                alternatives.pop().expect("one alternative")
            } else {
                Expr::Choice(alternatives)
            };
            optional(seq(vec![
                pair.clone(),
                seq(vec![lit(","), self.ws(), pair]).repeat(0, None),
            ]))
        };
        seq(vec![lit("{"), self.ws(), body, lit("}")])
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
        disjoint: bool,
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
            // An open object whose generic key excludes the declared names has
            // no second reading to fork on, so its subsets cost grammar size
            // rather than live configurations - the same position a closed
            // object is in, and it gets the same budget.
            Some(_) if !disjoint => UNORDERED_REQUIRED_BUDGET_OPEN,
            _ => UNORDERED_REQUIRED_BUDGET_CLOSED,
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
            // Nothing may follow. Once every required property has been seen
            // that is not a failure, it is the object being closed: a schema
            // whose properties are all required and whose `additionalProperties`
            // is false admits exactly one more token, `}`. Treating it as a
            // failure made the caller fall back to the relaxed object, which
            // does not enforce `required` at all - and it did so silently, on
            // the commonest small closed object there is.
            let tail = if item.is_empty() {
                if mask != full {
                    // Required properties are still missing and no property can
                    // supply them, so the object can never be closed.
                    return Ok(None);
                }
                Expr::Empty
            } else {
                let more = seq(vec![
                    lit(","),
                    self.ws(),
                    Expr::RuleRef(items[mask as usize].clone()),
                ]);
                // Only the full subset may stop: anything less is an object
                // still missing a property the schema requires.
                if mask == full { optional(more) } else { more }
            };
            pending.push((tails[mask as usize].clone(), tail));
            if !item.is_empty() {
                pending.push((items[mask as usize].clone(), Expr::choice(item)));
            }
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
/// of when `additionalProperties` leaves the names open.
///
/// Past it `required` is not enforced, and `required` is what rejects 92% of
/// the documents this engine admits and the schema does not - far ahead of
/// `anyOf` at 10 and `dependencies` at 2. So raising it was tried, and it is
/// not the fix.
///
/// Each subset is a parse the matcher carries at once, because an open
/// object's declared name can also be read as a generic key, so the ceiling
/// here is the configuration budget rather than grammar size. Four covers
/// 94.4% of the objects in JSONSchemaBench, though only 89.1% of its
/// *schemas*. Seven - with ten for the closed case - took over-acceptance from
/// 153 random walks to 101 and validity-given-completion from 80.4% to 86.7%
/// for no compile time, and then end to end, over 409 corpus schemas at batch
/// 512 with each document validated against its own schema rather than
/// walked, produced **exactly the same 195 valid documents of 512** and cost
/// 15% of the throughput. Raising only the closed budget, which carries no
/// runtime configurations at all, also produced exactly 195.
///
/// A random walk is uniform over the mask and a model is not. The walk
/// harness generates the deeply nested `required` violations these subsets
/// catch; the model does not generate them in the first place. The subsets are
/// paid for on every step and collected on almost none.
pub const UNORDERED_REQUIRED_BUDGET_OPEN: usize = 4;

/// The same, for objects whose property names are a closed set. Nothing forks
/// there, so the subsets are grammar states rather than live configurations
/// and the budget is bounded by size alone, which is why it is higher.
pub const UNORDERED_REQUIRED_BUDGET_CLOSED: usize = 6;



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

/// The shortest and longest strings an expression matches, in bytes.
///
/// `None` for the longest means unbounded. Used to decide whether a length
/// bound adds anything to a pattern that already constrains the length.
/// A pattern's length in *characters*, exactly when it has one.
///
/// `length_range` answers in bytes, because that is what the lexer counts, and
/// a character class spends one to four of them. A JSON Schema length bound is
/// over characters, so narrowing a repeat against one needs this instead.
fn character_range(expr: &Expr) -> (u64, Option<u64>) {
    match expr {
        Expr::Empty => (0, Some(0)),
        Expr::Literal(bytes) => {
            let count = String::from_utf8_lossy(bytes).chars().count() as u64;
            (count, Some(count))
        }
        Expr::CharacterClass { .. } => (1, Some(1)),
        Expr::RuleRef(_) => (0, None),
        Expr::Group(inner) => character_range(inner),
        Expr::Sequence(parts) => parts.iter().fold((0, Some(0)), |(low, high), part| {
            let (part_low, part_high) = character_range(part);
            (
                low.saturating_add(part_low),
                high.zip(part_high).map(|(a, b)| a.saturating_add(b)),
            )
        }),
        Expr::Choice(alternatives) => alternatives
            .iter()
            .fold((u64::MAX, Some(0)), |(low, high), alternative| {
                let (alt_low, alt_high) = character_range(alternative);
                (low.min(alt_low), high.zip(alt_high).map(|(a, b)| a.max(b)))
            }),
        Expr::Repeat { expr, min, max } => {
            let (inner_low, inner_high) = character_range(expr);
            (
                inner_low.saturating_mul(u64::from(*min)),
                max.and_then(|max| inner_high.map(|high| high.saturating_mul(u64::from(max)))),
            )
        }
    }
}

/// Narrow a pattern so that it only matches strings the length bound allows.
///
/// A grammar cannot intersect two languages in general, but it does not have to
/// here: what a length bound constrains is how many times something repeats, and
/// that is a number the repeat already carries. For `A x{p,q} B` with `A` and
/// `B` of fixed length, the bound on the whole string is a bound on the count,
/// so the two meet exactly by narrowing `p` and `q`. A pattern of fixed length
/// needs no narrowing, only checking, and a choice distributes - the bound
/// applies to each alternative and the ones it empties simply go.
///
/// Returns `None` where the shape is not one of those, which is the honest
/// answer: refusing the schema is better than a mask that admits a string the
/// bound forbids.
fn constrain_to_length(expr: &Expr, min: u32, max: Option<u32>) -> Option<Expr> {
    let min = u64::from(min);
    let max = max.map(u64::from);
    let fits = |low: u64, high: Option<u64>| {
        low >= min && max.is_none_or(|max| high.is_some_and(|high| high <= max))
    };

    match expr {
        Expr::Group(inner) => constrain_to_length(inner, min as u32, max.map(|m| m as u32)),
        Expr::Choice(alternatives) => {
            let kept: Vec<Expr> = alternatives
                .iter()
                .filter_map(|alternative| {
                    constrain_to_length(alternative, min as u32, max.map(|m| m as u32))
                })
                .collect();
            (!kept.is_empty()).then(|| Expr::choice(kept))
        }
        Expr::Repeat { .. } => narrow_repeat(std::slice::from_ref(expr), min, max),
        Expr::Sequence(parts) => {
            let (low, high) = character_range(expr);
            if fits(low, high) {
                return Some(expr.clone());
            }
            narrow_repeat(parts, min, max)
        }
        other => {
            let (low, high) = character_range(other);
            fits(low, high).then(|| other.clone())
        }
    }
}

/// Push a length bound into the one repeat of a sequence.
fn narrow_repeat(parts: &[Expr], min: u64, max: Option<u64>) -> Option<Expr> {
    let mut variable = None;
    let mut fixed = 0u64;
    for (index, part) in parts.iter().enumerate() {
        let flat = match part {
            Expr::Group(inner) => inner.as_ref(),
            other => other,
        };
        if let Expr::Repeat {
            expr,
            min: low,
            max: high,
        } = flat
            && Some(*low) != *high {
                if variable.is_some() {
                    // Two repeats share the budget between them, and which
                    // split to pick is a choice rather than a computation.
                    return None;
                }
                let (each_low, each_high) = character_range(expr);
                if each_low == 0 || Some(each_low) != each_high {
                    return None;
                }
                variable = Some((index, each_low, *low, *high));
                continue;
            }
        let (low, high) = character_range(part);
        if Some(low) != high {
            return None;
        }
        fixed = fixed.saturating_add(low);
    }

    let (index, each, low, high) = variable?;
    if fixed > max.unwrap_or(u64::MAX) {
        return None;
    }
    // `count * each + fixed` has to land in the bound, so the count does too.
    let wanted_low = min.saturating_sub(fixed).div_ceil(each);
    let wanted_high = max.map(|max| (max - fixed) / each);
    let new_low = u64::from(low).max(wanted_low);
    let new_high = match (high, wanted_high) {
        (Some(high), Some(wanted)) => Some(u64::from(high).min(wanted)),
        (Some(high), None) => Some(u64::from(high)),
        (None, wanted) => wanted,
    };
    if new_high.is_some_and(|high| high < new_low) {
        return None;
    }

    let mut rebuilt = parts.to_vec();
    let inner = match &parts[index] {
        Expr::Group(boxed) => boxed.as_ref(),
        other => other,
    };
    let Expr::Repeat { expr, .. } = inner else {
        return None;
    };
    rebuilt[index] = Expr::Repeat {
        expr: expr.clone(),
        min: u32::try_from(new_low).ok()?,
        max: new_high.map(|high| u32::try_from(high).unwrap_or(u32::MAX)),
    };
    Some(Expr::Sequence(rebuilt))
}

/// Fold branches that all describe objects into a single object.
///
/// The union of their properties, required only where every branch requires it,
/// and closed only where every branch closes it. That accepts more than the
/// branches do - a document may satisfy none of them exactly - which is the
/// direction a mask is allowed to err in, and it is the difference between a
/// grammar an LALR parser can build and no grammar at all.
///
/// Returns `None` when a branch is not an object, since then there is a real
/// choice of shape and collapsing it would throw the other shapes away.
fn merge_object_choice(parent: &Value, branches: &[Value]) -> Option<Value> {
    let mut properties = serde_json::Map::new();
    let mut required: Option<Vec<Value>> = None;
    let mut closed = true;
    let mut seen = 0;

    for branch in branches {
        let branch = branch.as_object()?;
        let names = match branch.get("properties") {
            Some(Value::Object(names)) => names,
            // A branch with no properties of its own accepts any object, so
            // the union is any object and there is nothing to gain.
            _ => return None,
        };
        if branch
            .get("type")
            .is_some_and(|kind| kind != &Value::String("object".into()))
        {
            return None;
        }
        for (name, schema) in names {
            match properties.get(name) {
                // Branches that describe the same property differently are
                // left unconstrained rather than intersected: the union of two
                // property schemas is not something this front end can build,
                // and accepting any value there is the safe direction.
                Some(held) if held != schema => {
                    properties.insert(name.clone(), Value::Object(Default::default()));
                }
                _ => {
                    properties.insert(name.clone(), schema.clone());
                }
            }
        }
        let here: Vec<Value> = branch
            .get("required")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        required = Some(match required {
            None => here,
            Some(so_far) => so_far.into_iter().filter(|name| here.contains(name)).collect(),
        });
        closed &= branch.get("additionalProperties") == Some(&Value::Bool(false));
        seen += 1;
    }
    if seen < 2 {
        return None;
    }

    let mut merged = serde_json::Map::new();
    for (key, value) in parent.as_object()? {
        if key != "anyOf" && key != "oneOf" {
            merged.insert(key.clone(), value.clone());
        }
    }
    // The parent may name properties too, and they join the union.
    if let Some(Value::Object(theirs)) = merged.get("properties") {
        for (name, schema) in theirs {
            properties.entry(name.clone()).or_insert(schema.clone());
        }
    }
    merged.insert("type".into(), Value::String("object".into()));
    merged.insert("properties".into(), Value::Object(properties));
    merged.insert("required".into(), Value::Array(required.unwrap_or_default()));
    if closed {
        merged.insert("additionalProperties".into(), Value::Bool(false));
    } else {
        merged.remove("additionalProperties");
    }
    Some(Value::Object(merged))
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

/// Keywords that describe a schema without constraining what it accepts.
///
/// Two `allOf` branches that disagree only about their prose are not in
/// conflict, and refusing them cost real schemas: `allOf` carries a lift of 21
/// over the corpus's lowering failures, and most of those branches differ only
/// here or in a bound that has a perfectly good intersection.
const ANNOTATIONS: &[&str] = &[
    "$comment",
    "$id",
    "$schema",
    "default",
    "definitions",
    "deprecated",
    "description",
    "examples",
    "readOnly",
    "title",
    "writeOnly",
    "$defs",
];

/// Combine `allOf` branches into one schema.
///
/// A grammar cannot intersect two languages, so the branches have to be merged
/// before lowering, and the merge has to be exact: a mask that is nearly right
/// is a mask that lets an invalid token through. Most of JSON Schema's
/// keywords do have an exact conjunction - a bound meets its partner at the
/// tighter of the two, a set of types at its intersection, two property maps at
/// their union with shared names merged in turn - so the merge computes it, and
/// refuses only where it genuinely cannot.
fn merge_all_of(document: &Value, parent: &Value, branches: &[Value]) -> Result<Value> {
    let mut merged = Value::Object(
        parent
            .as_object()
            .expect("checked by the caller")
            .iter()
            .filter(|(key, _)| key.as_str() != "allOf")
            .map(|(key, value)| (key.clone(), value.clone()))
            .collect(),
    );
    for branch in branches {
        merged = meet(document, &merged, branch, 0)?;
    }
    Ok(merged)
}

/// How far `meet` will follow `$ref` before giving up.
///
/// A reference can point at a schema that refers back, and a conjunction of two
/// mutually recursive schemas is not something this front end can build. The
/// limit is what stops it looping rather than a claim about how deep is enough.
const MEET_DEPTH: usize = 8;

/// The conjunction of two schemas, or an error if it cannot be computed.
fn meet(document: &Value, left: &Value, right: &Value, depth: usize) -> Result<Value> {
    if depth > MEET_DEPTH {
        bail!("allOf nests references too deeply to meet");
    }
    // A branch that is a reference is met by meeting what it points at. This is
    // the commonest shape `allOf` takes - `[{"$ref": "#/definitions/base"},
    // {"properties": {...}}]` - and refusing it was the single largest lowering
    // failure left.
    let left = &resolved(document, left);
    let right = &resolved(document, right);
    let (Some(left_object), Some(right_object)) = (left.as_object(), right.as_object()) else {
        // `true` and `false` as schemas: one accepts everything, the other
        // nothing, and neither shape appears in the corpus.
        bail!("allOf branches must be objects");
    };
    let mut merged = left_object.clone();

    // A branch that is itself an `allOf` is flattened rather than refused: the
    // conjunction is associative, so the inner branches simply join the outer.
    if let Some(inner) = right_object.get("allOf") {
        let inner = inner
            .as_array()
            .ok_or_else(|| anyhow!("allOf must be an array"))?;
        let rest = Value::Object(
            right_object
                .iter()
                .filter(|(key, _)| key.as_str() != "allOf")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let mut folded = meet(document, &Value::Object(merged), &rest, depth + 1)?;
        for branch in inner {
            folded = meet(document, &folded, branch, depth + 1)?;
        }
        return Ok(folded);
    }

    // A conjunction distributes over a disjunction, exactly:
    // `A and (B or C)` is `(A and B) or (A and C)`. Both are expressible, so
    // this needs no approximation.
    if let Some(options) = right_object.get("anyOf").or_else(|| right_object.get("oneOf")) {
        let options = options
            .as_array()
            .ok_or_else(|| anyhow!("anyOf/oneOf must be an array"))?;
        let rest = Value::Object(
            right_object
                .iter()
                .filter(|(key, _)| key.as_str() != "anyOf" && key.as_str() != "oneOf")
                .map(|(key, value)| (key.clone(), value.clone()))
                .collect(),
        );
        let base = meet(document, &Value::Object(merged), &rest, depth + 1)?;
        let distributed = options
            .iter()
            .map(|option| meet(document, &base, option, depth + 1))
            .collect::<Result<Vec<_>>>()?;
        return Ok(Value::Object(
            [("anyOf".to_string(), Value::Array(distributed))]
                .into_iter()
                .collect(),
        ));
    }

    for (key, value) in right_object {
        let Some(existing) = merged.get(key) else {
            merged.insert(key.clone(), value.clone());
            continue;
        };
        if existing == value {
            continue;
        }
        let combined = match key.as_str() {
            // Prose. Whichever is kept, the language is the same.
            key if ANNOTATIONS.contains(&key) => continue,

            // The lowering drops a complement it cannot express, so refusing
            // the whole schema here for a keyword the next stage discards would
            // be a refusal that buys nothing.
            "not" => continue,

            // A bound meets its partner at whichever is tighter.
            "minLength" | "minItems" | "minProperties" | "minimum" | "exclusiveMinimum" => {
                tighter(existing, value, true)?
            }
            "maxLength" | "maxItems" | "maxProperties" | "maximum" | "exclusiveMaximum" => {
                tighter(existing, value, false)?
            }

            // Both must hold, and `false` admits nothing beyond what is named.
            "additionalProperties" | "additionalItems" | "unevaluatedProperties" => {
                if existing == &Value::Bool(false) || value == &Value::Bool(false) {
                    Value::Bool(false)
                } else {
                    meet(document, existing, value, depth + 1)?
                }
            }
            "uniqueItems" => Value::Bool(
                existing.as_bool().unwrap_or(false) || value.as_bool().unwrap_or(false),
            ),

            // Sets meet at their intersection, and an empty one is a schema
            // nothing satisfies - which is a refusal rather than a merge.
            "type" => {
                let both = intersect(&as_set(existing), &as_set(value));
                if both.is_empty() {
                    bail!("allOf branches ask for incompatible types");
                }
                if both.len() == 1 {
                    Value::String(both[0].clone())
                } else {
                    Value::Array(both.into_iter().map(Value::String).collect())
                }
            }
            "enum" => {
                let left = existing
                    .as_array()
                    .ok_or_else(|| anyhow!("enum must be an array"))?;
                let right = value
                    .as_array()
                    .ok_or_else(|| anyhow!("enum must be an array"))?;
                let both: Vec<Value> = left
                    .iter()
                    .filter(|entry| right.contains(entry))
                    .cloned()
                    .collect();
                if both.is_empty() {
                    bail!("allOf branches ask for disjoint enums");
                }
                Value::Array(both)
            }

            // A property named by both branches has to satisfy both.
            "properties" | "patternProperties" | "definitions" | "$defs" => {
                let (Some(left), Some(right)) = (existing.as_object(), value.as_object())
                else {
                    bail!("{key} must be an object");
                };
                let mut into = left.clone();
                for (name, schema) in right {
                    match into.get(name) {
                        Some(held) if held != schema => {
                            let combined = meet(document, held, schema, depth + 1)?;
                            into.insert(name.clone(), combined);
                        }
                        _ => {
                            into.insert(name.clone(), schema.clone());
                        }
                    }
                }
                Value::Object(into)
            }
            "required" => {
                let mut into = existing
                    .as_array()
                    .ok_or_else(|| anyhow!("required must be an array"))?
                    .clone();
                for name in value
                    .as_array()
                    .ok_or_else(|| anyhow!("required must be an array"))?
                {
                    if !into.contains(name) {
                        into.push(name.clone());
                    }
                }
                Value::Array(into)
            }
            "items" => meet(document, existing, value, depth + 1)?,

            // `$ref` cannot be met without resolving it, and two different
            // patterns have no intersection expressible as a pattern.
            key => bail!("allOf branches disagree about '{key}'"),
        };
        merged.insert(key.clone(), combined);
    }
    Ok(Value::Object(merged))
}

/// Whichever of two numeric bounds is the tighter.
fn tighter(left: &Value, right: &Value, lower: bool) -> Result<Value> {
    let (Some(a), Some(b)) = (left.as_f64(), right.as_f64()) else {
        bail!("a bound must be a number");
    };
    let keep_left = if lower { a >= b } else { a <= b };
    Ok(if keep_left {
        left.clone()
    } else {
        right.clone()
    })
}

fn as_set(value: &Value) -> Vec<String> {
    match value {
        Value::String(name) => vec![name.clone()],
        Value::Array(names) => names
            .iter()
            .filter_map(|name| name.as_str().map(str::to_string))
            .collect(),
        _ => Vec::new(),
    }
}

fn intersect(left: &[String], right: &[String]) -> Vec<String> {
    left.iter()
        .filter(|name| right.contains(name))
        .cloned()
        .collect()
}

/// Does this schema accept nothing at all?
///
/// `false`, and `{"not": {}}` or `{"not": true}` - the complement of the schema
/// that accepts everything.
fn is_unsatisfiable(schema: &Value) -> bool {
    if schema == &Value::Bool(false) {
        return true;
    }
    let Some(object) = schema.as_object() else {
        return false;
    };
    object.len() == 1
        && object
            .get("not")
            .is_some_and(|negated| {
                negated == &Value::Object(Default::default()) || negated == &Value::Bool(true)
            })
}

/// A schema with its `$ref` replaced by what it points at, if it has one.
///
/// The reference's siblings are kept and win, which is what JSON Schema 2019-09
/// onwards says and what schemas in the wild assume: `{"$ref": "#/x",
/// "description": "..."}` is the referenced schema with that description.
fn resolved(document: &Value, schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };
    let Some(Value::String(reference)) = object.get("$ref") else {
        return schema.clone();
    };
    let Some(pointer) = reference.strip_prefix('#') else {
        return schema.clone();
    };
    let Some(Value::Object(target)) = resolve_pointer(document, pointer) else {
        return schema.clone();
    };
    let mut merged = target.clone();
    for (key, value) in object {
        if key != "$ref" {
            merged.insert(key.clone(), value.clone());
        }
    }
    Value::Object(merged)
}

/// Follow a JSON pointer into the document, unescaping as RFC 6901 says.
fn resolve_pointer<'v>(document: &'v Value, pointer: &str) -> Option<&'v Value> {
    let mut here = document;
    for token in pointer.trim_start_matches('/').split('/') {
        if token.is_empty() {
            continue;
        }
        let token = token.replace("~1", "/").replace("~0", "~");
        here = match here {
            Value::Object(map) => map.get(&token)?,
            Value::Array(items) => items.get(token.parse::<usize>().ok()?)?,
            _ => return None,
        };
    }
    Some(here)
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
    json_character_except(&[])
}

/// One JSON string character, except that it may not be any of `excluded`.
///
/// `excluded` holds plain bytes. An escape sequence is never one of them - it
/// is two bytes or more and starts with a backslash - so the escape arm is
/// unaffected, and only the unescaped class narrows.
fn json_character_except(excluded: &[u8]) -> Expr {
    let mut forbidden = vec![
        (0, 0x1f),
        (b'"' as u32, b'"' as u32),
        (b'\\' as u32, b'\\' as u32),
    ];
    forbidden.extend(excluded.iter().map(|byte| (*byte as u32, *byte as u32)));
    let unescaped = char_class(true, forbidden);
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

/// A JSON string that is *not* any of `names`.
///
/// `additionalProperties` governs the names `properties` does not declare, so
/// the arm that carries it must not also spell a declared name. Left open, an
/// object with one declared string property admits `{"a": 1}`, because the
/// additional arm accepts any name at all - including `a` - with any value.
///
/// "Any name except these" is a regular set, so this is a complement rather
/// than an approximation. Walk a trie of the declared names: a string is not a
/// name exactly when it stops at a node no name ends at, or when it leaves the
/// trie by a character no edge carries - after which the rest is
/// unconstrained. Those two are built separately so that the unconstrained
/// tail is written once rather than once per character of every name.
///
/// `None` when some name does not survive the round trip through JSON as plain
/// bytes - anything needing an escape, or non-ASCII. The trie walks bytes and
/// an escape spells one character in several of them, so rather than get that
/// subtly wrong the caller keeps the unrestricted key, which only ever admits
/// more.
fn string_body_excluding(names: &[String]) -> Option<Expr> {
    // Excluding nothing is the plain string, and it has to be spelled the way
    // `json_string` spells it or the lexeme cache sees a different shape and
    // gives this object a key terminal of its own. An object with no declared
    // properties is common enough - every free-form value has one - that the
    // duplicates were a measurable part of the lexer.
    if names.is_empty() {
        return None;
    }
    let mut bodies: Vec<Vec<u8>> = Vec::with_capacity(names.len());
    for name in names {
        if !name.is_ascii() || name.bytes().any(|byte| byte < 0x20 || byte == b'"' || byte == b'\\')
        {
            return None;
        }
        bodies.push(name.as_bytes().to_vec());
    }
    let suffixes: Vec<&[u8]> = bodies.iter().map(|body| body.as_slice()).collect();
    // A string is not one of the names exactly when it either stops at a node
    // no name ends at, or leaves the trie and then does whatever it likes.
    let mut alternatives = vec![seq(vec![
        leaves_trie(&suffixes),
        json_character().repeat(0, None),
    ])];
    alternatives.extend(stops_inside(&suffixes));
    Some(Expr::choice(alternatives))
}

/// The paths into the trie that stop at a node no name ends at.
///
/// `None` when every node reachable from here ends a name, so there is no such
/// path and the alternative does not exist.
fn stops_inside(suffixes: &[&[u8]]) -> Option<Expr> {
    let mut alternatives = Vec::new();
    if !suffixes.iter().any(|suffix| suffix.is_empty()) {
        alternatives.push(Expr::Empty);
    }
    for (byte, rest) in trie_edges(suffixes) {
        if let Some(tail) = stops_inside(&rest) {
            alternatives.push(seq(vec![byte_literal(byte), tail]));
        }
    }
    (!alternatives.is_empty()).then(|| Expr::choice(alternatives))
}

/// The shortest paths that leave the trie: a walk to some node followed by one
/// character no edge out of it carries.
///
/// Everything after such a prefix is unconstrained, which is why the free tail
/// is concatenated once by the caller rather than at every node. Building it
/// per node instead is correct and was measured to be far more expensive - the
/// lexer determinises a copy of "any string" for every character of every
/// declared name.
fn leaves_trie(suffixes: &[&[u8]]) -> Expr {
    let edges = trie_edges(suffixes);
    let taken: Vec<u8> = edges.iter().map(|(byte, _)| *byte).collect();
    let mut alternatives = vec![json_character_except(&taken)];
    for (byte, rest) in edges {
        // Recurse even when every remaining suffix is empty. That node ends a
        // name and has no edges, so *any* character leaves the trie there -
        // which is what lets a name's own extensions through.
        alternatives.push(seq(vec![byte_literal(byte), leaves_trie(&rest)]));
    }
    Expr::choice(alternatives)
}

fn trie_edges<'a>(suffixes: &[&'a [u8]]) -> Vec<(u8, Vec<&'a [u8]>)> {
    let mut edges: BTreeMap<u8, Vec<&'a [u8]>> = BTreeMap::new();
    for suffix in suffixes {
        if let Some((first, rest)) = suffix.split_first() {
            edges.entry(*first).or_default().push(rest);
        }
    }
    edges.into_iter().collect()
}

fn byte_literal(byte: u8) -> Expr {
    Expr::literal(vec![byte])
}

fn unbounded_number(max_digits: Option<u32>) -> Expr {
    // Every run of digits is bounded by the same budget, because a fraction or
    // an exponent runs away exactly as an integer part does.
    let tail = digit_tail(max_digits);
    let run = match max_digits {
        Some(budget) => format!("[0-9]{{1,{budget}}}"),
        None => "[0-9]+".to_string(),
    };
    regex_to_expr(&format!(
        r"-?(?:0|[1-9]{tail})(?:\.{run})?(?:[eE][+-]?{run})?"
    ))
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Walk the expression over a candidate string. Only the shapes
    /// `string_body_excluding` builds appear, which is why this is a test
    /// helper rather than a general matcher.
    fn matches(expr: &Expr, input: &[u8]) -> bool {
        fn walk(expr: &Expr, input: &[u8], rest: &mut Vec<usize>) {
            match expr {
                Expr::Empty => rest.push(0),
                Expr::Literal(bytes) => {
                    if input.starts_with(bytes) {
                        rest.push(bytes.len());
                    }
                }
                Expr::CharacterClass { negated, ranges } => {
                    if let Some(byte) = input.first() {
                        let inside = ranges
                            .iter()
                            .any(|(low, high)| (*low..=*high).contains(&(*byte as u32)));
                        if inside != *negated {
                            rest.push(1);
                        }
                    }
                }
                Expr::Choice(alternatives) => {
                    for alternative in alternatives {
                        walk(alternative, input, rest);
                    }
                }
                Expr::Sequence(elements) => {
                    let mut heads = vec![0usize];
                    for element in elements {
                        let mut next = Vec::new();
                        for head in &heads {
                            let mut tails = Vec::new();
                            walk(element, &input[*head..], &mut tails);
                            next.extend(tails.into_iter().map(|tail| head + tail));
                        }
                        next.sort_unstable();
                        next.dedup();
                        heads = next;
                    }
                    rest.extend(heads);
                }
                Expr::Repeat { expr, min, max } => {
                    let mut heads = vec![0usize];
                    let mut count = 0u32;
                    if *min == 0 {
                        rest.push(0);
                    }
                    while !heads.is_empty() && max.is_none_or(|max| count < max) {
                        let mut next = Vec::new();
                        for head in &heads {
                            let mut tails = Vec::new();
                            walk(expr, &input[*head..], &mut tails);
                            next.extend(
                                tails
                                    .into_iter()
                                    .filter(|tail| *tail > 0)
                                    .map(|tail| head + tail),
                            );
                        }
                        next.sort_unstable();
                        next.dedup();
                        count += 1;
                        if count >= *min {
                            rest.extend(next.iter().copied());
                        }
                        heads = next;
                    }
                }
                other => panic!("unexpected shape in an excluded key: {other:?}"),
            }
        }
        let mut ends = Vec::new();
        walk(expr, input, &mut ends);
        ends.contains(&input.len())
    }

    #[test]
    fn a_key_may_be_any_string_that_is_not_a_declared_name() {
        let names = ["al".to_string(), "alpha".to_string(), "b".to_string()];
        let body = string_body_excluding(&names).expect("plain ASCII names");
        for declared in ["al", "alpha", "b"] {
            assert!(
                !matches(&body, declared.as_bytes()),
                "{declared} is declared, so it is not an additional property"
            );
        }
        // Prefixes, extensions and neighbours of a declared name are not the
        // name, and excluding them would narrow the language - the one thing a
        // mask may not do.
        for allowed in ["", "a", "alp", "alphas", "ala", "bb", "z", "AL"] {
            assert!(
                matches(&body, allowed.as_bytes()),
                "{allowed} is not a declared name, so it must still be allowed"
            );
        }
    }

    #[test]
    fn a_name_needing_an_escape_gives_up_rather_than_guess() {
        // The trie walks bytes and an escape spells one character in several,
        // so the answer is None and the caller keeps the unrestricted key.
        assert!(string_body_excluding(&["a\"b".to_string()]).is_none());
        assert!(string_body_excluding(&["é".to_string()]).is_none());
    }

    #[test]
    fn excluding_nothing_is_left_to_the_shared_string_lexeme() {
        assert!(string_body_excluding(&[]).is_none());
    }
}
