//! Regex-to-grammar converter.
//!
//! Converts a regular expression pattern to an EBNF grammar string,
//! then parses it into a `Grammar`. Follows JavaScript regex semantics.
//!
//! # Supported features
//! - Literals, character classes `[a-z]`, negated `[^...]`
//! - Quantifiers: `*`, `+`, `?`, `{n}`, `{n,}`, `{n,m}`
//! - Groups: `(...)`, `(?:...)`, `(?<name>...)`
//! - Alternation: `|`
//! - Escapes: `\d`, `\w`, `\s`, `\D`, `\W`, `\S`, `\uXXXX`, `\u{XXXXX}`, `\xHH`
//! - Any char: `.`
//! - Anchors: `^`, `$` (ignored with warning)
//!
//! # Unsupported (errors)
//! - Lookahead/lookbehind: `(?=...)`, `(?!...)`, `(?<=...)`, `(?<!...)`
//! - Backreferences: `\1`, `\k<name>`
//! - Unicode properties: `\p{...}`, `\P{...}`
//! - Word boundaries: `\b`, `\B`

use anyhow::{Result, bail};

use crate::structured::grammar::Grammar;

/// Convert a regex pattern to a Grammar.
///
/// The resulting grammar has a single root rule matching the pattern.
///
/// # Example
/// ```
/// use pie_grammar::regex::regex_to_grammar;
///
/// let grammar = regex_to_grammar("[a-z]+").unwrap();
/// ```
pub fn regex_to_grammar(pattern: &str) -> Result<Grammar> {
    let ebnf = regex_to_ebnf(pattern)?;
    Grammar::from_ebnf(&ebnf, "root")
}

/// Convert a regex pattern to an EBNF grammar string.
///
/// Returns a string like `root ::= [a-z]+\n`.
pub fn regex_to_ebnf(pattern: &str) -> Result<String> {
    let mut converter = RegexConverter::new(pattern);
    let body = converter.convert()?;
    Ok(format!("root ::= {}\n", body))
}

struct RegexConverter<'a> {
    input: &'a [u8],
    pos: usize,
}

impl<'a> RegexConverter<'a> {
    fn new(pattern: &'a str) -> Self {
        Self {
            input: pattern.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let ch = self.input.get(self.pos).copied();
        if ch.is_some() {
            self.pos += 1;
        }
        ch
    }

    fn at_end(&self) -> bool {
        self.pos >= self.input.len()
    }

    /// Main entry: parse alternation.
    fn convert(&mut self) -> Result<String> {
        // Handle anchors
        if self.peek() == Some(b'^') {
            self.advance();
        }

        let result = self.parse_alternation()?;

        if self.peek() == Some(b'$') {
            self.advance();
        }

        if !self.at_end() && self.peek() != Some(b')') {
            bail!("unexpected character at position {}: '{}'",
                self.pos, self.input[self.pos] as char);
        }

        if result.is_empty() {
            Ok("\"\"".to_string())
        } else {
            Ok(result)
        }
    }

    /// Parse alternation: `a|b|c`
    fn parse_alternation(&mut self) -> Result<String> {
        let mut alternatives = vec![self.parse_sequence()?];

        while self.peek() == Some(b'|') {
            self.advance();
            alternatives.push(self.parse_sequence()?);
        }

        if alternatives.len() == 1 {
            Ok(alternatives.into_iter().next().unwrap())
        } else {
            // Wrap empty alternatives
            let parts: Vec<String> = alternatives
                .into_iter()
                .map(|a| if a.is_empty() { "\"\"".to_string() } else { a })
                .collect();
            Ok(format!("({})", parts.join(" | ")))
        }
    }

    /// Parse sequence: concatenated atoms with quantifiers.
    fn parse_sequence(&mut self) -> Result<String> {
        let mut segments = Vec::new();

        while !self.at_end() {
            match self.peek() {
                Some(b'|') | Some(b')') | Some(b'$') => break,
                _ => {
                    let atom = self.parse_atom_with_quantifier()?;
                    if !atom.is_empty() {
                        segments.push(atom);
                    }
                }
            }
        }

        Ok(segments.join(" "))
    }

    /// Parse an atom with optional quantifier.
    fn parse_atom_with_quantifier(&mut self) -> Result<String> {
        let atom = self.parse_atom()?;

        if self.at_end() {
            return Ok(atom);
        }

        match self.peek() {
            Some(b'*') => {
                self.advance();
                self.skip_non_greedy();
                self.check_no_consecutive_quantifier()?;
                Ok(format!("{}*", atom))
            }
            Some(b'+') => {
                self.advance();
                self.skip_non_greedy();
                self.check_no_consecutive_quantifier()?;
                Ok(format!("{}+", atom))
            }
            Some(b'?') => {
                self.advance();
                self.skip_non_greedy();
                self.check_no_consecutive_quantifier()?;
                Ok(format!("{}?", atom))
            }
            Some(b'{') => {
                let (min, max) = self.parse_repetition()?;
                self.skip_non_greedy();
                self.check_no_consecutive_quantifier()?;
                match max {
                    Some(m) if m == min => Ok(format!("{}{{{}}}", atom, min)),
                    Some(m) => Ok(format!("{}{{{},{}}}", atom, min, m)),
                    None => Ok(format!("{}{{{},}}", atom, min)),
                }
            }
            _ => Ok(atom),
        }
    }

    /// Skip non-greedy modifier (? after quantifier) — ignored for grammar.
    fn skip_non_greedy(&mut self) {
        if self.peek() == Some(b'?') {
            self.advance();
        }
    }

    /// Check that the next character is not another quantifier (consecutive quantifiers are invalid).
    fn check_no_consecutive_quantifier(&self) -> Result<()> {
        if matches!(self.peek(), Some(b'*') | Some(b'+') | Some(b'{')) {
            bail!("consecutive quantifiers are not supported at position {}", self.pos);
        }
        // Note: ? after a quantifier is handled as non-greedy by skip_non_greedy,
        // so by the time we get here, an extra ? would be a third quantifier.
        Ok(())
    }

    /// Parse a single atom: literal, escape, group, char class, or dot.
    fn parse_atom(&mut self) -> Result<String> {
        match self.peek() {
            None => Ok(String::new()),
            Some(b'(') => self.parse_group(),
            Some(b'[') => self.parse_char_class(),
            Some(b'.') => {
                self.advance();
                // . = any character (full Unicode range)
                Ok("[\\u0000-\\U0010ffff]".to_string())
            }
            Some(b'\\') => self.parse_escape(),
            Some(b'^') => {
                self.advance();
                // Anchor — ignore
                Ok(String::new())
            }
            Some(ch) if is_metachar(ch) => {
                bail!("unexpected metacharacter '{}' at position {}", ch as char, self.pos);
            }
            Some(_) => self.parse_literal(),
        }
    }

    /// Parse a parenthesized group.
    fn parse_group(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'('));

        // Check for special group types
        if self.peek() == Some(b'?') {
            self.advance();
            match self.peek() {
                Some(b':') => {
                    self.advance(); // non-capturing group
                }
                Some(b'<') => {
                    self.advance();
                    match self.peek() {
                        Some(b'=') | Some(b'!') => {
                            bail!("lookbehind assertions are not supported");
                        }
                        _ => {
                            // Named capture group (?<name>...) — skip name
                            while self.peek() != Some(b'>') {
                                if self.at_end() {
                                    bail!("unterminated named group");
                                }
                                self.advance();
                            }
                            self.advance(); // skip >
                        }
                    }
                }
                Some(b'=') | Some(b'!') => {
                    bail!("lookahead assertions are not supported");
                }
                _ => {
                    bail!("unsupported group modifier");
                }
            }
        }

        let inner = self.parse_alternation()?;

        if self.peek() != Some(b')') {
            bail!("unmatched '(' at position {}", self.pos);
        }
        self.advance();

        if inner.is_empty() {
            Ok("\"\"".to_string())
        } else {
            Ok(format!("({})", inner))
        }
    }

    /// Parse a character class `[...]`.
    fn parse_char_class(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'['));
        let mut result = String::from("[");

        if self.peek() == Some(b'^') {
            self.advance();
            result.push('^');
        }

        // Allow ] as first char in class
        if self.peek() == Some(b']') {
            self.advance();
            result.push_str("\\]");
        }

        while self.peek() != Some(b']') {
            if self.at_end() {
                bail!("unclosed character class");
            }

            match self.peek() {
                Some(b'\\') => {
                    let esc = self.parse_char_class_escape()?;
                    result.push_str(&esc);
                }
                Some(ch) => {
                    self.advance();
                    // Check for range
                    if self.peek() == Some(b'-') && self.input.get(self.pos + 1) != Some(&b']') {
                        self.advance(); // consume -
                        result.push(ch as char);
                        result.push('-');
                        if self.peek() == Some(b'\\') {
                            let esc = self.parse_char_class_escape()?;
                            result.push_str(&esc);
                        } else if let Some(end) = self.advance() {
                            result.push(end as char);
                        }
                    } else {
                        // Escape special EBNF char-class chars
                        match ch {
                            b']' => result.push_str("\\]"),
                            b'\\' => result.push_str("\\\\"),
                            b'^' => result.push_str("\\^"),
                            _ => result.push(ch as char),
                        }
                    }
                }
                None => bail!("unclosed character class"),
            }
        }

        self.advance(); // consume ]
        result.push(']');
        Ok(result)
    }

    /// Parse an escape in a character class context.
    fn parse_char_class_escape(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'\\'));
        match self.peek() {
            None => bail!("truncated escape"),
            Some(b'd') => { self.advance(); Ok("0-9".to_string()) }
            Some(b'D') => { self.advance(); Ok("\\u0000-/:-\\U0010ffff".to_string()) }
            Some(b'w') => { self.advance(); Ok("a-zA-Z0-9_".to_string()) }
            Some(b'W') => { self.advance(); Ok("\\u0000-/:-@\\[-\\^`\\{-\\U0010ffff".to_string()) }
            Some(b's') => { self.advance(); Ok("\\t\\n\\r \\u000b\\u000c".to_string()) }
            Some(b'S') => { self.advance(); Ok("\\u0000-\\u0008\\u000e-\\u001f!-\\U0010ffff".to_string()) }
            Some(b'u') => self.parse_unicode_escape_for_class(),
            Some(b'x') => self.parse_hex_escape_for_class(),
            Some(ch) => {
                self.advance();
                match ch {
                    b'n' => Ok("\\n".to_string()),
                    b'r' => Ok("\\r".to_string()),
                    b't' => Ok("\\t".to_string()),
                    b'f' => Ok("\\u000c".to_string()),
                    b'v' => Ok("\\u000b".to_string()),
                    b'-' => Ok("\\-".to_string()),
                    b']' => Ok("\\]".to_string()),
                    b'\\' => Ok("\\\\".to_string()),
                    b'^' => Ok("\\^".to_string()),
                    _ if is_regex_metachar(ch) => {
                        let mut s = String::new();
                        s.push(ch as char);
                        Ok(s)
                    }
                    _ => {
                        let mut s = String::new();
                        s.push(ch as char);
                        Ok(s)
                    }
                }
            }
        }
    }

    /// Parse an escape sequence outside character classes.
    fn parse_escape(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'\\'));
        match self.peek() {
            None => bail!("truncated escape"),
            Some(b'd') => { self.advance(); Ok("[0-9]".to_string()) }
            Some(b'D') => { self.advance(); Ok("[^0-9]".to_string()) }
            Some(b'w') => { self.advance(); Ok("[a-zA-Z0-9_]".to_string()) }
            Some(b'W') => { self.advance(); Ok("[^a-zA-Z0-9_]".to_string()) }
            Some(b's') => { self.advance(); Ok("[\\t\\n\\r \\u000b\\u000c]".to_string()) }
            Some(b'S') => { self.advance(); Ok("[^\\t\\n\\r \\u000b\\u000c]".to_string()) }
            Some(b'b') | Some(b'B') => {
                bail!("word boundaries (\\b, \\B) are not supported");
            }
            Some(b'p') | Some(b'P') => {
                bail!("unicode property escapes (\\p, \\P) are not supported");
            }
            Some(ch) if ch >= b'1' && ch <= b'9' => {
                bail!("backreferences are not supported");
            }
            Some(b'k') => {
                bail!("backreferences are not supported");
            }
            Some(b'u') => self.parse_unicode_escape(),
            Some(b'x') => self.parse_hex_escape(),
            Some(b'n') => { self.advance(); Ok("\"\\n\"".to_string()) }
            Some(b'r') => { self.advance(); Ok("\"\\r\"".to_string()) }
            Some(b't') => { self.advance(); Ok("\"\\t\"".to_string()) }
            Some(b'f') => { self.advance(); Ok("\"\\x0c\"".to_string()) }
            Some(b'v') => { self.advance(); Ok("\"\\x0b\"".to_string()) }
            Some(b'0') => { self.advance(); Ok("\"\\x00\"".to_string()) }
            Some(ch) => {
                self.advance();
                // Escaped metachar or literal
                let mut s = String::new();
                s.push('"');
                match ch {
                    b'\\' => s.push_str("\\\\"),
                    b'"' => s.push_str("\\\""),
                    _ => s.push(ch as char),
                }
                s.push('"');
                Ok(s)
            }
        }
    }

    /// Parse `\uXXXX` or `\u{XXXXX}`.
    fn parse_unicode_escape(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'u'));
        if self.peek() == Some(b'{') {
            self.advance();
            let mut hex = String::new();
            while self.peek() != Some(b'}') {
                if self.at_end() {
                    bail!("unterminated unicode escape");
                }
                hex.push(self.advance().unwrap() as char);
            }
            self.advance(); // consume }
            let cp = u32::from_str_radix(&hex, 16)
                .map_err(|_| anyhow::anyhow!("invalid unicode escape: \\u{{{}}}", hex))?;
            Ok(codepoint_to_ebnf_literal(cp))
        } else {
            let hex = self.read_hex(4)?;
            let cp = u32::from_str_radix(&hex, 16)
                .map_err(|_| anyhow::anyhow!("invalid unicode escape: \\u{}", hex))?;
            Ok(codepoint_to_ebnf_literal(cp))
        }
    }

    /// Parse `\uXXXX` inside a character class (returns char-class-compatible string).
    fn parse_unicode_escape_for_class(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'u'));
        if self.peek() == Some(b'{') {
            self.advance();
            let mut hex = String::new();
            while self.peek() != Some(b'}') {
                if self.at_end() { bail!("unterminated unicode escape"); }
                hex.push(self.advance().unwrap() as char);
            }
            self.advance();
            Ok(format!("\\u{{{}}}", hex))
        } else {
            let hex = self.read_hex(4)?;
            Ok(format!("\\u{}", hex))
        }
    }

    /// Parse `\xHH`.
    fn parse_hex_escape(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'x'));
        let hex = self.read_hex(2)?;
        let byte = u8::from_str_radix(&hex, 16)
            .map_err(|_| anyhow::anyhow!("invalid hex escape: \\x{}", hex))?;
        if byte < 0x80 {
            Ok(codepoint_to_ebnf_literal(byte as u32))
        } else {
            Ok(format!("\"\\x{}\"", hex))
        }
    }

    /// Parse `\xHH` inside a character class.
    fn parse_hex_escape_for_class(&mut self) -> Result<String> {
        assert_eq!(self.advance(), Some(b'x'));
        let hex = self.read_hex(2)?;
        Ok(format!("\\u00{}", hex))
    }

    /// Parse a literal character and wrap in EBNF quotes.
    fn parse_literal(&mut self) -> Result<String> {
        // Collect consecutive literals
        let mut s = String::new();
        while !self.at_end() {
            match self.peek() {
                Some(ch) if !is_metachar(ch) && ch != b'\\' && ch != b'[' && ch != b'(' && ch != b'.' => {
                    self.advance();
                    match ch {
                        b'"' => s.push_str("\\\""),
                        b'\\' => s.push_str("\\\\"),
                        _ => s.push(ch as char),
                    }
                }
                _ => break,
            }
            // Don't collect across a quantifier boundary
            if matches!(self.peek(), Some(b'*') | Some(b'+') | Some(b'?') | Some(b'{')) {
                // Only the last char gets quantified; must break if more than one
                if s.len() > 1 {
                    // Back up one character
                    let last = s.pop().unwrap();
                    self.pos -= last.len_utf8();
                    break;
                }
                break;
            }
        }

        if s.is_empty() {
            Ok(String::new())
        } else {
            Ok(format!("\"{}\"", s))
        }
    }

    /// Parse `{n}`, `{n,}`, or `{n,m}`.
    fn parse_repetition(&mut self) -> Result<(u32, Option<u32>)> {
        assert_eq!(self.advance(), Some(b'{'));
        let min = self.read_int()?;

        match self.peek() {
            Some(b'}') => {
                self.advance();
                Ok((min, Some(min)))
            }
            Some(b',') => {
                self.advance();
                if self.peek() == Some(b'}') {
                    self.advance();
                    Ok((min, None))
                } else {
                    let max = self.read_int()?;
                    if self.peek() != Some(b'}') {
                        bail!("expected '}}' in repetition");
                    }
                    self.advance();
                    Ok((min, Some(max)))
                }
            }
            _ => bail!("expected ',' or '}}' in repetition"),
        }
    }

    fn read_int(&mut self) -> Result<u32> {
        let mut n: u32 = 0;
        let mut any = false;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                n = n * 10 + (ch - b'0') as u32;
                self.advance();
                any = true;
            } else {
                break;
            }
        }
        if !any {
            bail!("expected integer in repetition");
        }
        Ok(n)
    }

    fn read_hex(&mut self, count: usize) -> Result<String> {
        let mut s = String::new();
        for _ in 0..count {
            match self.advance() {
                Some(ch) if ch.is_ascii_hexdigit() => s.push(ch as char),
                _ => bail!("expected hex digit"),
            }
        }
        Ok(s)
    }
}

fn is_metachar(ch: u8) -> bool {
    matches!(ch, b'*' | b'+' | b'?' | b'{' | b'}' | b'|' | b')' | b'^' | b'$')
}

fn is_regex_metachar(ch: u8) -> bool {
    matches!(
        ch,
        b'.' | b'*' | b'+' | b'?' | b'(' | b')' | b'[' | b']'
            | b'{' | b'}' | b'|' | b'^' | b'$' | b'\\'
    )
}

/// Convert a Unicode codepoint to an EBNF-safe string literal.
fn codepoint_to_ebnf_literal(cp: u32) -> String {
    if let Some(c) = char::from_u32(cp) {
        match c {
            '"' => "\"\\\"\"".to_string(),
            '\\' => "\"\\\\\"".to_string(),
            '\n' => "\"\\n\"".to_string(),
            '\r' => "\"\\r\"".to_string(),
            '\t' => "\"\\t\"".to_string(),
            c if c.is_ascii_graphic() || c == ' ' => format!("\"{}\"", c),
            _ => {
                let mut buf = [0u8; 4];
                let s = c.encode_utf8(&mut buf);
                let mut r = String::from("\"");
                for &b in s.as_bytes() {
                    r.push_str(&format!("\\x{:02x}", b));
                }
                r.push('"');
                r
            }
        }
    } else {
        format!("\"\\x{:02x}\"", cp)
    }
}
