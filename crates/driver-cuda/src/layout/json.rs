//! `nlohmann::json::dump()`, reproduced for `serde_json::Value`.
//!
//! # Why not `serde_json::to_string_pretty`
//!
//! Three reasons, each independently sufficient.
//!
//! **Key order.** `nlohmann`'s default object is a `std::map`, so `dump()`
//! emits keys in byte-wise sorted order. `serde_json`'s order depends on the
//! `preserve_order` feature -- and Cargo unifies features across a build
//! graph, so whether this crate sorted would depend on a sibling crate's
//! dependency list. `crates/engine` already turns that feature on. Sorting
//! here explicitly cannot be reconfigured from outside.
//!
//! **Indentation.** `to_string_pretty` and `dump(2)` differ in where they put
//! spaces around `:` and how they render empty containers.
//!
//! **Doubles.** See [`super::dtoa`]: `nlohmann` uses Grisu2, Rust uses
//! shortest-correctly-rounded, and they disagree about 0.07% of the time.
//!
//! The planner profile cache is read-merge-rewritten by whichever process
//! calibrates next, and either implementation may be the one doing it. If the
//! two did not produce identical bytes, every write would rewrite entries it
//! never touched.

use serde_json::Value;

use super::dtoa::write_f64;

/// Escape a string the way `nlohmann::json::dump()` does.
///
/// The five short escapes, `\"`, `\\`, and `\u00XX` for every other control
/// character. Notably it does **not** escape `/`, and it leaves non-ASCII
/// UTF-8 as literal bytes rather than emitting surrogate pairs -- both are
/// observable in the cache file, and a GPU name is a vendor string this code
/// does not get to constrain.
pub fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str("\\u00");
                let n = c as u32;
                out.push(char::from_digit(n >> 4, 16).unwrap_or('0'));
                out.push(char::from_digit(n & 0xf, 16).unwrap_or('0'));
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// `value.dump(indent)`, plus the trailing newline the C++ writes after it.
#[must_use]
pub fn dump_pretty(value: &Value, indent: usize) -> String {
    let mut out = String::with_capacity(1024);
    write_value(&mut out, value, indent, 0);
    out.push('\n');
    out
}

fn write_value(out: &mut String, value: &Value, indent: usize, depth: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(n) => write_number(out, n),
        Value::String(s) => write_string(out, s),
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline_indent(out, indent, depth + 1);
                write_value(out, item, indent, depth + 1);
            }
            newline_indent(out, indent, depth);
            out.push(']');
        }
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str("{}");
                return;
            }
            // Sorted, because `nlohmann`'s object is a `std::map`. `str`'s
            // `Ord` is byte-wise, which is what `std::string::operator<` is.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_unstable();
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                newline_indent(out, indent, depth + 1);
                write_string(out, key);
                out.push(':');
                if indent > 0 {
                    out.push(' ');
                }
                let v = map.get(key.as_str()).unwrap_or(&Value::Null);
                write_value(out, v, indent, depth + 1);
            }
            newline_indent(out, indent, depth);
            out.push('}');
        }
    }
}

fn newline_indent(out: &mut String, indent: usize, depth: usize) {
    if indent == 0 {
        return;
    }
    out.push('\n');
    for _ in 0..(indent * depth) {
        out.push(' ');
    }
}

/// A number as `dump()` renders it: integers verbatim, everything else
/// through Grisu2.
fn write_number(out: &mut String, n: &serde_json::Number) {
    if let Some(u) = n.as_u64() {
        out.push_str(itoa(u, false).as_str());
    } else if let Some(i) = n.as_i64() {
        // `unsigned_abs` is used rather than `-i as u64` because i64::MIN has
        // no positive counterpart and would overflow the negation.
        out.push_str(itoa(i.unsigned_abs(), i < 0).as_str());
    } else if let Some(f) = n.as_f64() {
        write_f64(out, f);
    } else {
        // `serde_json` with `arbitrary_precision` keeps the source text. It is
        // not enabled here, so this is unreachable; emitting the debug form is
        // still better than silently dropping the field.
        out.push_str(&n.to_string());
    }
}

fn itoa(v: u64, negative: bool) -> String {
    if negative {
        format!("-{v}")
    } else {
        v.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn objects_are_emitted_in_sorted_key_order() {
        // Not insertion order: `nlohmann`'s object is a `std::map`, so a file
        // it writes is sorted and a file this writes must be too.
        let v = json!({"b": 1, "a": 2, "C": 3});
        assert_eq!(
            dump_pretty(&v, 2),
            "{\n  \"C\": 3,\n  \"a\": 2,\n  \"b\": 1\n}\n"
        );
    }

    #[test]
    fn empty_containers_stay_on_one_line() {
        let v = json!({"a": [], "c": {}, "b": 1});
        assert_eq!(
            dump_pretty(&v, 2),
            "{\n  \"a\": [],\n  \"b\": 1,\n  \"c\": {}\n}\n"
        );
    }

    #[test]
    fn nesting_indents_by_two_per_level() {
        let v = json!({"outer": {"inner": [1, 2]}});
        assert_eq!(
            dump_pretty(&v, 2),
            "{\n  \"outer\": {\n    \"inner\": [\n      1,\n      2\n    ]\n  }\n}\n"
        );
    }

    #[test]
    fn zero_indent_drops_the_space_after_the_colon() {
        // `dump()` with no argument is compact, and the colon separator
        // changes with it -- not just the whitespace between members.
        let v = json!({"a": 1, "b": [2]});
        assert_eq!(dump_pretty(&v, 0), "{\"a\":1,\"b\":[2]}\n");
    }

    #[test]
    fn floats_go_through_grisu2_and_integers_do_not() {
        // A float that happens to be integral still prints a fractional part,
        // and an integer must not acquire one.
        let v = json!({"f": 1.0, "i": 1, "n": -3, "big": 18446744073709551615u64});
        assert_eq!(
            dump_pretty(&v, 0),
            "{\"big\":18446744073709551615,\"f\":1.0,\"i\":1,\"n\":-3}\n"
        );
    }

    #[test]
    fn control_characters_use_lowercase_four_digit_escapes() {
        let v = json!({"k": "a\u{1}b\tc\"d\\e/f"});
        assert_eq!(
            dump_pretty(&v, 0),
            "{\"k\":\"a\\u0001b\\tc\\\"d\\\\e/f\"}\n"
        );
    }
}
