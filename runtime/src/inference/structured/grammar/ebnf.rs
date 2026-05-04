//! EBNF parser: parses EBNF grammar strings into `Grammar`.
//!
//! Supports:
//! - String literals with UTF-8 and escape sequences
//! - Character classes with Unicode ranges, negation
//! - Rule references
//! - Sequences, choices (|), grouping with parentheses
//! - Quantifiers: *, +, ?, {n}, {n,m}, {n,}
//! - Lookahead assertions: (= ...)
//! - Comments: # to end of line

use anyhow::{Result, bail};

use super::builder::GrammarBuilder;
use super::{Grammar, Expr, ExprId, RuleId};

// ─── UTF-8 / Escape helpers ──────────────────────────────────────────

fn hex_char_to_u32(c: u8) -> Option<u32> {
    match c {
        b'0'..=b'9' => Some((c - b'0') as u32),
        b'a'..=b'f' => Some((c - b'a' + 10) as u32),
        b'A'..=b'F' => Some((c - b'A' + 10) as u32),
        _ => None,
    }
}

/// Parse an escape sequence starting at `\`. Returns (codepoint, bytes_consumed).
fn parse_escape(data: &[u8], extra_escapes: &[(u8, u32)]) -> Result<(u32, usize)> {
    if data.len() < 2 || data[0] != b'\\' {
        bail!("expected escape sequence");
    }
    // Check extra escapes first
    for &(ch, cp) in extra_escapes {
        if data[1] == ch {
            return Ok((cp, 2));
        }
    }
    match data[1] {
        b'\'' => Ok((b'\'' as u32, 2)),
        b'"' => Ok((b'"' as u32, 2)),
        b'?' => Ok((b'?' as u32, 2)),
        b'\\' => Ok((b'\\' as u32, 2)),
        b'a' => Ok((0x07, 2)),
        b'b' => Ok((0x08, 2)),
        b'f' => Ok((0x0C, 2)),
        b'n' => Ok((0x0A, 2)),
        b'r' => Ok((0x0D, 2)),
        b't' => Ok((0x09, 2)),
        b'v' => Ok((0x0B, 2)),
        b'0' => Ok((0x00, 2)),
        b'e' => Ok((0x1B, 2)),
        b'x' => {
            // \xHH... (variable length hex)
            let mut cp = 0u32;
            let mut len = 0;
            while 2 + len < data.len() {
                if let Some(d) = hex_char_to_u32(data[2 + len]) {
                    cp = cp * 16 + d;
                    len += 1;
                } else {
                    break;
                }
            }
            if len == 0 {
                bail!("invalid \\x escape: no hex digits");
            }
            Ok((cp, 2 + len))
        }
        b'u' => {
            // \uXXXX (exactly 4 hex digits)
            if data.len() < 6 {
                bail!("invalid \\u escape: need 4 hex digits");
            }
            let mut cp = 0u32;
            for i in 0..4 {
                let d = hex_char_to_u32(data[2 + i])
                    .ok_or_else(|| anyhow::anyhow!("invalid hex digit in \\u escape"))?;
                cp = cp * 16 + d;
            }
            Ok((cp, 6))
        }
        b'U' => {
            // \UXXXXXXXX (exactly 8 hex digits)
            if data.len() < 10 {
                bail!("invalid \\U escape: need 8 hex digits");
            }
            let mut cp = 0u32;
            for i in 0..8 {
                let d = hex_char_to_u32(data[2 + i])
                    .ok_or_else(|| anyhow::anyhow!("invalid hex digit in \\U escape"))?;
                cp = cp * 16 + d;
            }
            Ok((cp, 10))
        }
        _ => bail!("invalid escape sequence: \\{}", data[1] as char),
    }
}

/// Parse next UTF-8 char or escape sequence. Returns (codepoint, bytes_consumed).
fn parse_next_utf8_or_escaped(data: &[u8], extra_escapes: &[(u8, u32)]) -> Result<(u32, usize)> {
    if data.is_empty() {
        bail!("unexpected end of input");
    }
    if data[0] == b'\\' {
        return parse_escape(data, extra_escapes);
    }
    // Decode one UTF-8 character
    let s = std::str::from_utf8(data).unwrap_or("");
    if let Some(c) = s.chars().next() {
        Ok((c as u32, c.len_utf8()))
    } else {
        bail!("invalid UTF-8 sequence");
    }
}

// ─── Token types ─────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum TokenType {
    RuleName,
    Identifier,
    StringLiteral,
    LBracket,
    RBracket,
    Caret,
    CharInCharClass(u32), // codepoint
    Dash,
    Assign,
    LParen,
    RParen,
    LBrace,
    RBrace,
    Pipe,
    Star,
    Plus,
    Question,
    LookaheadLParen,
    IntegerLiteral(i64),
    Comma,
    EndOfFile,
}

#[derive(Debug, Clone)]
struct Token {
    ty: TokenType,
    /// The string value (for StringLiteral: decoded UTF-8, for Identifier/RuleName: the name)
    value: String,
    line: usize,
    col: usize,
}

// ─── Lexer ───────────────────────────────────────────────────────────

struct Lexer<'a> {
    input: &'a [u8],
    pos: usize,
    line: usize,
    col: usize,
}

impl<'a> Lexer<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input: input.as_bytes(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn peek_at(&self, delta: usize) -> Option<u8> {
        self.input.get(self.pos + delta).copied()
    }

    fn advance(&mut self) {
        if let Some(b) = self.peek() {
            if b == b'\n' || (b == b'\r' && self.peek_at(1) != Some(b'\n')) {
                self.line += 1;
                self.col = 1;
            } else {
                self.col += 1;
            }
            self.pos += 1;
        }
    }

    fn advance_n(&mut self, n: usize) {
        for _ in 0..n {
            self.advance();
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            match self.peek() {
                Some(b' ') | Some(b'\t') | Some(b'\n') | Some(b'\r') => {
                    self.advance();
                }
                Some(b'#') => {
                    self.advance();
                    while let Some(b) = self.peek() {
                        if b == b'\n' || b == b'\r' {
                            break;
                        }
                        self.advance();
                    }
                    // consume the newline
                    if let Some(b) = self.peek() {
                        self.advance();
                        if b == b'\r' && self.peek() == Some(b'\n') {
                            self.advance();
                        }
                    }
                }
                _ => break,
            }
        }
    }

    fn err(&self, msg: &str) -> anyhow::Error {
        anyhow::anyhow!("EBNF lexer error at line {}, column {}: {}", self.line, self.col, msg)
    }

    fn is_name_char(c: u8, is_first: bool) -> bool {
        c == b'_'
            || c == b'-'
            || c == b'.'
            || c.is_ascii_alphabetic()
            || (!is_first && c.is_ascii_digit())
    }

    fn lex_identifier(&mut self) -> String {
        let start = self.pos;
        let mut first = true;
        while let Some(c) = self.peek() {
            if Self::is_name_char(c, first) {
                self.advance();
                first = false;
            } else {
                break;
            }
        }
        String::from_utf8_lossy(&self.input[start..self.pos]).to_string()
    }

    fn lex_string(&mut self) -> Result<Token> {
        let start_line = self.line;
        let start_col = self.col;
        self.advance(); // skip opening "

        let mut codepoints = Vec::new();
        loop {
            match self.peek() {
                None | Some(b'\n') | Some(b'\r') => {
                    return Err(self.err("unterminated string literal"));
                }
                Some(b'"') => break,
                _ => {
                    let remaining = &self.input[self.pos..];
                    let (cp, len) = parse_next_utf8_or_escaped(remaining, &[])
                        .map_err(|e| self.err(&e.to_string()))?;
                    self.advance_n(len);
                    codepoints.push(cp);
                }
            }
        }
        self.advance(); // skip closing "

        // Convert codepoints to UTF-8 string
        let mut value = String::new();
        for cp in codepoints {
            if let Some(c) = char::from_u32(cp) {
                value.push(c);
            } else {
                return Err(self.err(&format!("invalid codepoint: U+{:04X}", cp)));
            }
        }

        Ok(Token {
            ty: TokenType::StringLiteral,
            value,
            line: start_line,
            col: start_col,
        })
    }

    fn lex_char_class(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        let line = self.line;
        let col = self.col;
        tokens.push(Token { ty: TokenType::LBracket, value: String::new(), line, col });
        self.advance(); // skip [

        if self.peek() == Some(b'^') {
            let line = self.line;
            let col = self.col;
            tokens.push(Token { ty: TokenType::Caret, value: String::new(), line, col });
            self.advance();
        }

        // Extra escape chars inside character classes
        let char_class_escapes: &[(u8, u32)] = &[
            (b'^', b'^' as u32), (b'$', b'$' as u32), (b'\\', b'\\' as u32),
            (b'.', b'.' as u32), (b'*', b'*' as u32), (b'+', b'+' as u32),
            (b'?', b'?' as u32), (b'(', b'(' as u32), (b')', b')' as u32),
            (b'[', b'[' as u32), (b']', b']' as u32), (b'{', b'{' as u32),
            (b'}', b'}' as u32), (b'|', b'|' as u32), (b'/', b'/' as u32),
            (b'-', b'-' as u32),
        ];

        while self.peek().is_some() && self.peek() != Some(b']') {
            let line = self.line;
            let col = self.col;
            match self.peek().unwrap() {
                b'\r' | b'\n' => {
                    return Err(self.err("character class should not contain newline"));
                }
                b'-' => {
                    tokens.push(Token { ty: TokenType::Dash, value: String::new(), line, col });
                    self.advance();
                }
                _ => {
                    let remaining = &self.input[self.pos..];
                    let (cp, len) = parse_next_utf8_or_escaped(remaining, char_class_escapes)
                        .map_err(|e| self.err(&e.to_string()))?;
                    self.advance_n(len);
                    tokens.push(Token {
                        ty: TokenType::CharInCharClass(cp),
                        value: String::new(),
                        line,
                        col,
                    });
                }
            }
        }

        if self.peek().is_none() {
            return Err(self.err("unterminated character class"));
        }

        let line = self.line;
        let col = self.col;
        tokens.push(Token { ty: TokenType::RBracket, value: String::new(), line, col });
        self.advance(); // skip ]

        Ok(tokens)
    }

    fn lex_integer(&mut self) -> Result<Token> {
        let start_line = self.line;
        let start_col = self.col;
        let mut negative = false;

        if self.peek() == Some(b'-') {
            negative = true;
            self.advance();
        } else if self.peek() == Some(b'+') {
            self.advance();
        }

        let mut num: i64 = 0;
        let mut has_digit = false;
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                has_digit = true;
                num = num * 10 + (c - b'0') as i64;
                self.advance();
                if num > 1_000_000_000_000_000 {
                    return Err(self.err("integer too large"));
                }
            } else {
                break;
            }
        }
        if !has_digit {
            return Err(self.err("expected integer"));
        }
        if negative {
            num = -num;
        }
        Ok(Token {
            ty: TokenType::IntegerLiteral(num),
            value: String::new(),
            line: start_line,
            col: start_col,
        })
    }

    fn tokenize(&mut self) -> Result<Vec<Token>> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace_and_comments();

            let line = self.line;
            let col = self.col;

            match self.peek() {
                None => {
                    tokens.push(Token { ty: TokenType::EndOfFile, value: String::new(), line, col });
                    break;
                }
                Some(b'(') => {
                    if self.peek_at(1) == Some(b'=') {
                        self.advance_n(2);
                        tokens.push(Token {
                            ty: TokenType::LookaheadLParen,
                            value: String::new(),
                            line,
                            col,
                        });
                    } else {
                        self.advance();
                        tokens.push(Token { ty: TokenType::LParen, value: String::new(), line, col });
                    }
                }
                Some(b')') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::RParen, value: String::new(), line, col });
                }
                Some(b'{') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::LBrace, value: String::new(), line, col });
                }
                Some(b'}') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::RBrace, value: String::new(), line, col });
                }
                Some(b'|') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::Pipe, value: String::new(), line, col });
                }
                Some(b',') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::Comma, value: String::new(), line, col });
                }
                Some(b'*') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::Star, value: String::new(), line, col });
                }
                Some(b'+') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::Plus, value: String::new(), line, col });
                }
                Some(b'?') => {
                    self.advance();
                    tokens.push(Token { ty: TokenType::Question, value: String::new(), line, col });
                }
                Some(b':') => {
                    if self.peek_at(1) == Some(b':') && self.peek_at(2) == Some(b'=') {
                        self.advance_n(3);
                        tokens.push(Token { ty: TokenType::Assign, value: String::new(), line, col });
                    } else {
                        return Err(self.err("unexpected character ':'"));
                    }
                }
                Some(b'"') => {
                    tokens.push(self.lex_string()?);
                }
                Some(b'[') => {
                    tokens.extend(self.lex_char_class()?);
                }
                Some(c) if Self::is_name_char(c, true) => {
                    let name = self.lex_identifier();
                    tokens.push(Token { ty: TokenType::Identifier, value: name, line, col });
                }
                Some(c) if c.is_ascii_digit() => {
                    tokens.push(self.lex_integer()?);
                }
                Some(c) => {
                    return Err(self.err(&format!("unexpected character: '{}'", c as char)));
                }
            }
        }

        // Convert identifiers before ::= to RuleNames
        convert_identifiers_to_rule_names(&mut tokens)?;

        Ok(tokens)
    }
}

fn convert_identifiers_to_rule_names(tokens: &mut [Token]) -> Result<()> {
    for i in 0..tokens.len() {
        if tokens[i].ty == TokenType::Assign {
            if i == 0 {
                bail!(
                    "EBNF parser error at line {}, column {}: ::= should not be the first token",
                    tokens[i].line, tokens[i].col
                );
            }
            if tokens[i - 1].ty != TokenType::Identifier {
                bail!(
                    "EBNF parser error at line {}, column {}: ::= should be preceded by an identifier",
                    tokens[i - 1].line, tokens[i - 1].col
                );
            }
            // Check rule name is at start of line
            if i >= 2 && tokens[i - 2].line == tokens[i - 1].line {
                bail!(
                    "EBNF parser error at line {}, column {}: rule name should be at the beginning of the line",
                    tokens[i - 1].line, tokens[i - 1].col
                );
            }
            tokens[i - 1].ty = TokenType::RuleName;
        }
    }
    Ok(())
}

// ─── Parser ──────────────────────────────────────────────────────────

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    builder: GrammarBuilder,
    cur_rule_name: String,
    aux_rule_counter: usize,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            builder: GrammarBuilder::new(),
            cur_rule_name: String::new(),
            aux_rule_counter: 0,
        }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }

    fn consume(&mut self) {
        self.pos += 1;
    }

    fn expect(&mut self, expected: &TokenType, msg: &str) -> Result<()> {
        if std::mem::discriminant(&self.peek().ty) != std::mem::discriminant(expected) {
            return Err(self.parse_error(msg));
        }
        self.consume();
        Ok(())
    }

    fn parse_error(&self, msg: &str) -> anyhow::Error {
        let tok = self.peek();
        anyhow::anyhow!(
            "EBNF parser error at line {}, column {}: {}",
            tok.line, tok.col, msg
        )
    }

    fn new_aux_rule_name(&mut self) -> String {
        self.aux_rule_counter += 1;
        format!("{}_{}", self.cur_rule_name, self.aux_rule_counter)
    }

    // ── Init: collect all rule names (two-pass) ──

    fn init_rule_names(&mut self, root_rule_name: &str) -> Result<()> {
        for tok in &self.tokens {
            if tok.ty == TokenType::RuleName {
                if self.builder.find_rule(&tok.value).is_some() {
                    bail!(
                        "EBNF parser error at line {}, column {}: rule \"{}\" defined multiple times",
                        tok.line, tok.col, tok.value
                    );
                }
                self.builder.add_rule(&tok.value);
            }
        }
        if self.builder.find_rule(root_rule_name).is_none() {
            bail!("EBNF parser error: root rule \"{}\" not found", root_rule_name);
        }
        Ok(())
    }

    // ── Parsing ──

    fn parse_char_class(&mut self) -> Result<ExprId> {
        // Expect LBracket
        self.expect(&TokenType::LBracket, "expected [")?;

        let mut negated = false;
        if self.peek().ty == TokenType::Caret {
            negated = true;
            self.consume();
        }

        let mut ranges: Vec<(u32, u32)> = Vec::new();

        while self.peek().ty != TokenType::RBracket && self.peek().ty != TokenType::EndOfFile {
            let cp = match &self.peek().ty {
                TokenType::CharInCharClass(cp) => *cp,
                TokenType::Dash => b'-' as u32,
                _ => return Err(self.parse_error("unexpected token in character class")),
            };
            self.consume();

            // Check for range expression: char-char
            if self.peek().ty == TokenType::Dash {
                let next_is_char = matches!(
                    self.tokens.get(self.pos + 1).map(|t| &t.ty),
                    Some(TokenType::CharInCharClass(_)) | Some(TokenType::Dash)
                );
                if next_is_char {
                    self.consume(); // skip dash
                    let cp2 = match &self.peek().ty {
                        TokenType::CharInCharClass(cp) => *cp,
                        TokenType::Dash => b'-' as u32,
                        _ => unreachable!(),
                    };
                    self.consume();
                    if cp > cp2 {
                        return Err(self.parse_error("invalid character class: lower bound > upper bound"));
                    }
                    ranges.push((cp, cp2));
                } else {
                    ranges.push((cp, cp));
                }
            } else {
                ranges.push((cp, cp));
            }
        }

        self.expect(&TokenType::RBracket, "expected ]")?;
        Ok(self.builder.add_character_class(negated, ranges))
    }

    fn parse_string(&mut self) -> Result<ExprId> {
        if self.peek().ty != TokenType::StringLiteral {
            return Err(self.parse_error("expected string literal"));
        }
        let value = self.peek().value.clone();
        self.consume();

        if value.is_empty() {
            Ok(self.builder.add_empty_string())
        } else {
            Ok(self.builder.add_byte_string(value.as_bytes()))
        }
    }

    fn parse_rule_ref(&mut self) -> Result<ExprId> {
        if self.peek().ty != TokenType::Identifier {
            return Err(self.parse_error("expected identifier"));
        }
        let name = self.peek().value.clone();
        self.consume();

        match self.builder.find_rule(&name) {
            Some(rule_id) => Ok(self.builder.add_rule_ref(rule_id)),
            None => Err(self.parse_error(&format!("rule \"{}\" is not defined", name))),
        }
    }

    fn parse_element(&mut self) -> Result<ExprId> {
        match &self.peek().ty {
            TokenType::LParen => {
                self.consume();
                if self.peek().ty == TokenType::RParen {
                    self.consume();
                    return Ok(self.builder.add_empty_string());
                }
                let expr = self.parse_choices()?;
                self.expect(&TokenType::RParen, "expected )")?;
                Ok(expr)
            }
            TokenType::LBracket => self.parse_char_class(),
            TokenType::StringLiteral => self.parse_string(),
            TokenType::Identifier => self.parse_rule_ref(),
            _ => Err(self.parse_error(&format!(
                "expected element, got {:?}",
                self.peek().ty
            ))),
        }
    }

    fn parse_integer(&mut self) -> Result<i64> {
        match self.peek().ty {
            TokenType::IntegerLiteral(n) => {
                self.consume();
                Ok(n)
            }
            _ => Err(self.parse_error("expected integer")),
        }
    }

    fn parse_repetition_range(&mut self) -> Result<(i64, i64)> {
        self.expect(&TokenType::LBrace, "expected {")?;
        let lower = self.parse_integer()?;
        if lower < 0 {
            return Err(self.parse_error("lower bound cannot be negative"));
        }

        if self.peek().ty == TokenType::Comma {
            self.consume();
            if self.peek().ty == TokenType::RBrace {
                self.consume();
                return Ok((lower, -1)); // unbounded
            }
            let upper = self.parse_integer()?;
            if upper < lower {
                return Err(self.parse_error("lower bound is larger than upper bound"));
            }
            self.expect(&TokenType::RBrace, "expected }")?;
            Ok((lower, upper))
        } else if self.peek().ty == TokenType::RBrace {
            self.consume();
            Ok((lower, lower)) // exact
        } else {
            Err(self.parse_error("expected ',' or '}' in repetition range"))
        }
    }

    /// Extract a RuleId from an expression: if it's already a RuleRef, return that RuleId;
    /// otherwise wrap the expression in a new auxiliary rule.
    fn wrap_in_rule(&mut self, expr_id: ExprId) -> RuleId {
        if let BorrowedExpr::RuleRef(rid) = self.builder_get_expr(expr_id) {
            return rid;
        }
        let aux_name = self.new_aux_rule_name();
        let aux_rule = self.builder.add_rule(&aux_name);
        self.builder.set_rule_body(aux_rule, expr_id);
        aux_rule
    }

    fn handle_star(&mut self, expr_id: ExprId) -> ExprId {
        // Check if it's a character class → CharacterClassStar optimization
        if let BorrowedExpr::CharacterClass { negated, ranges } = self.builder_get_expr(expr_id) {
            return self.builder.add_character_class_star(negated, ranges);
        }

        // a* → Repeat(rule_for(a), 0, None)
        let rule_id = self.wrap_in_rule(expr_id);
        self.builder.add_repeat(rule_id, 0, None)
    }

    fn handle_plus(&mut self, expr_id: ExprId) -> ExprId {
        // a+ → Repeat(rule_for(a), 1, None)
        let rule_id = self.wrap_in_rule(expr_id);
        self.builder.add_repeat(rule_id, 1, None)
    }

    fn handle_question(&mut self, expr_id: ExprId) -> ExprId {
        // a? → Repeat(rule_for(a), 0, Some(1))
        let rule_id = self.wrap_in_rule(expr_id);
        self.builder.add_repeat(rule_id, 0, Some(1))
    }

    fn handle_repetition(&mut self, expr_id: ExprId, lower: i64, upper: i64) -> ExprId {
        let rule_id = self.wrap_in_rule(expr_id);
        let lower = lower as u32;

        if upper == -1 {
            // {n,}: n mandatory, then unbounded
            self.builder.add_repeat(rule_id, lower, None)
        } else {
            // {n} or {n,m}
            self.builder.add_repeat(rule_id, lower, Some(upper as u32))
        }
    }

    fn parse_element_with_quantifier(&mut self) -> Result<ExprId> {
        let expr = self.parse_element()?;

        match self.peek().ty {
            TokenType::Star => {
                self.consume();
                Ok(self.handle_star(expr))
            }
            TokenType::Plus => {
                self.consume();
                Ok(self.handle_plus(expr))
            }
            TokenType::Question => {
                self.consume();
                Ok(self.handle_question(expr))
            }
            TokenType::LBrace => {
                let (lower, upper) = self.parse_repetition_range()?;
                Ok(self.handle_repetition(expr, lower, upper))
            }
            _ => Ok(expr),
        }
    }

    fn parse_sequence(&mut self) -> Result<ExprId> {
        let mut elements = Vec::new();
        loop {
            elements.push(self.parse_element_with_quantifier()?);
            match self.peek().ty {
                TokenType::Pipe
                | TokenType::RParen
                | TokenType::LookaheadLParen
                | TokenType::RuleName
                | TokenType::EndOfFile => break,
                _ => {}
            }
        }
        Ok(self.builder.add_sequence(elements))
    }

    fn parse_choices(&mut self) -> Result<ExprId> {
        let mut choices = Vec::new();
        choices.push(self.parse_sequence()?);

        while self.peek().ty == TokenType::Pipe {
            self.consume();
            choices.push(self.parse_sequence()?);
        }

        Ok(self.builder.add_choices(choices))
    }

    fn parse_rule(&mut self) -> Result<()> {
        if self.peek().ty != TokenType::RuleName {
            return Err(self.parse_error("expected rule name"));
        }
        let name = self.peek().value.clone();
        self.cur_rule_name = name.clone();
        self.aux_rule_counter = 0;
        self.consume();

        self.expect(&TokenType::Assign, "expected ::=")?;

        let body = self.parse_choices()?;

        let rule_id = self.builder.find_rule(&name).unwrap();
        self.builder.set_rule_body(rule_id, body);

        // Optional lookahead assertion
        if self.peek().ty == TokenType::LookaheadLParen {
            self.consume();
            let la_expr = self.parse_choices()?;
            self.expect(&TokenType::RParen, "expected )")?;
            self.builder.set_rule_lookahead(rule_id, la_expr, false);
        }

        Ok(())
    }

    fn parse(mut self, root_rule_name: &str) -> Result<Grammar> {
        self.init_rule_names(root_rule_name)?;

        while self.peek().ty != TokenType::EndOfFile {
            self.parse_rule()?;
        }

        self.builder.build(root_rule_name)
    }

    /// Helper to read an expr from the builder (needed for star optimization check).
    fn builder_get_expr(&self, id: ExprId) -> BorrowedExpr {
        // We need to peek at the expr type without consuming the builder.
        // Since we only use this for the star optimization, we'll check the type.
        // The builder stores exprs in a Vec, so we can index directly.
        // We need to expose this from the builder...
        // For now, we can match on the data stored in exprs.
        self.builder.peek_expr(id)
    }
}

/// A borrowed view of an expression, for optimization checks in the parser.
pub(crate) enum BorrowedExpr {
    CharacterClass { negated: bool, ranges: Vec<(u32, u32)> },
    RuleRef(RuleId),
    Other,
}

impl GrammarBuilder {
    /// Peek at an expression type (used by parser for CharacterClass star optimization).
    pub(crate) fn peek_expr(&self, id: ExprId) -> BorrowedExpr {
        match &self.exprs[id.0 as usize] {
            Expr::CharacterClass { negated, ranges } => BorrowedExpr::CharacterClass {
                negated: *negated,
                ranges: ranges.to_vec(),
            },
            Expr::RuleRef(rid) => BorrowedExpr::RuleRef(*rid),
            _ => BorrowedExpr::Other,
        }
    }
}

// ─── Public API ──────────────────────────────────────────────────────

impl Grammar {
    /// Parse an EBNF grammar string.
    ///
    /// # Example
    /// ```
    /// use pie_grammar::grammar::Grammar;
    ///
    /// let grammar = Grammar::from_ebnf(r#"root ::= "hello" | "world""#, "root").unwrap();
    /// assert_eq!(grammar.num_rules(), 1);
    /// ```
    pub fn from_ebnf(source: &str, root_rule_name: &str) -> Result<Grammar> {
        let mut lexer = Lexer::new(source);
        let tokens = lexer.tokenize()?;
        let parser = Parser::new(tokens);
        parser.parse(root_rule_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_display(input: &str) -> String {
        Grammar::from_ebnf(input, "root").unwrap().to_string()
    }

    #[test]
    fn test_basic_string_literal() {
        let g = parse_and_display("root ::= \"hello\"");
        assert_eq!(g, "root ::= ((\"hello\"))");
    }

    #[test]
    fn test_empty_string() {
        let g = parse_and_display("root ::= \"\"");
        assert_eq!(g, "root ::= ((\"\"))");
    }

    #[test]
    fn test_character_class() {
        let g = parse_and_display("root ::= [a-z]");
        assert_eq!(g, "root ::= (([a-z]))");
    }

    #[test]
    fn test_negated_character_class() {
        let g = parse_and_display("root ::= [^a-z]");
        assert_eq!(g, "root ::= (([^a-z]))");
    }

    #[test]
    fn test_sequence() {
        let g = parse_and_display("root ::= \"a\" \"b\" \"c\"");
        assert_eq!(g, "root ::= ((\"a\" \"b\" \"c\"))");
    }

    #[test]
    fn test_choice() {
        let g = parse_and_display("root ::= \"a\" | \"b\" | \"c\"");
        assert_eq!(g, "root ::= ((\"a\") | (\"b\") | (\"c\"))");
    }

    #[test]
    fn test_grouping() {
        let g = parse_and_display("root ::= (\"a\" \"b\") | (\"c\" \"d\")");
        assert_eq!(g, "root ::= ((((\"a\" \"b\"))) | (((\"c\" \"d\"))))");
    }

    #[test]
    fn test_star_quantifier_string() {
        let g = parse_and_display("root ::= \"a\"*");
        assert_eq!(g, "root ::= ((root_1{0,}))\nroot_1 ::= \"a\"");
    }

    #[test]
    fn test_plus_quantifier() {
        let g = parse_and_display("root ::= \"a\"+");
        assert_eq!(g, "root ::= ((root_1{1,}))\nroot_1 ::= \"a\"");
    }

    #[test]
    fn test_question_quantifier() {
        let g = parse_and_display("root ::= \"a\"?");
        assert_eq!(g, "root ::= ((root_1{0,1}))\nroot_1 ::= \"a\"");
    }

    #[test]
    fn test_character_class_star() {
        let g = parse_and_display("root ::= [a-z]*");
        assert_eq!(g, "root ::= (([a-z]*))");
    }

    #[test]
    fn test_repetition_exact() {
        let g = parse_and_display("root ::= \"a\"{3}");
        assert_eq!(g, "root ::= ((root_1{3,3}))\nroot_1 ::= \"a\"");
    }

    #[test]
    fn test_repetition_range() {
        let g = parse_and_display("root ::= \"a\"{2,4}");
        assert_eq!(g, "root ::= ((root_1{2,4}))\nroot_1 ::= \"a\"");
    }

    #[test]
    fn test_repetition_unbounded() {
        let g = parse_and_display("root ::= \"a\"{2,}");
        assert_eq!(g, "root ::= ((root_1{2,}))\nroot_1 ::= \"a\"");
    }

    #[test]
    fn test_lookahead_assertion() {
        let g = parse_and_display("root ::= \"a\" (=\"b\")");
        assert_eq!(g, "root ::= ((\"a\")) (= ((\"b\")))");
    }

    #[test]
    fn test_complex_lookahead() {
        let g = parse_and_display("root ::= \"a\" (=\"b\" \"c\" [0-9])");
        assert_eq!(g, "root ::= ((\"a\")) (= ((\"b\" \"c\" [0-9])))");
    }

    #[test]
    fn test_rule_reference() {
        let g = parse_and_display("root ::= digit\ndigit ::= [0-9]");
        assert_eq!(g, "root ::= ((digit))\ndigit ::= (([0-9]))");
    }

    #[test]
    fn test_comment() {
        let g = parse_and_display("# comment\nroot ::= \"hello\" # inline\n");
        assert_eq!(g, "root ::= ((\"hello\"))");
    }

    #[test]
    fn test_unicode_string() {
        let g = Grammar::from_ebnf(r#"root ::= "\u0041\u0042""#, "root").unwrap();
        match g.get_expr(g.root().body) {
            Expr::Choices(choices) => {
                match g.get_expr(choices[0]) {
                    Expr::Sequence(seq) => {
                        match g.get_expr(seq[0]) {
                            Expr::ByteString(bytes) => assert_eq!(bytes, b"AB"),
                            other => panic!("expected ByteString, got {:?}", other),
                        }
                    }
                    other => panic!("expected Sequence, got {:?}", other),
                }
            }
            other => panic!("expected Choices, got {:?}", other),
        }
    }

    #[test]
    fn test_multiple_rules() {
        let g = parse_and_display(
            "root ::= item+\nitem ::= [a-z] | [0-9]"
        );
        assert!(g.contains("root ::= ((item{1,}))"));
        assert!(g.contains("item ::= (([a-z]) | ([0-9]))"));
    }

    #[test]
    fn test_error_undefined_rule() {
        let result = Grammar::from_ebnf("root ::= missing", "root");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not defined"));
    }

    #[test]
    fn test_error_duplicate_rule() {
        let result = Grammar::from_ebnf("root ::= \"a\"\nroot ::= \"b\"", "root");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("multiple times"));
    }

    #[test]
    fn test_error_missing_root() {
        let result = Grammar::from_ebnf("foo ::= \"a\"", "root");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_complex_character_class() {
        let g = parse_and_display(r"root ::= [a-zA-Z0-9_\-]");
        assert_eq!(g, "root ::= (([a-zA-Z0-9_\\-]))");
    }

    #[test]
    fn test_escape_sequences_in_string() {
        let g = Grammar::from_ebnf(r#"root ::= "\n\t\\\"""#, "root").unwrap();
        match g.get_expr(g.root().body) {
            Expr::Choices(choices) => match g.get_expr(choices[0]) {
                Expr::Sequence(seq) => match g.get_expr(seq[0]) {
                    Expr::ByteString(bytes) => assert_eq!(bytes, b"\n\t\\\""),
                    other => panic!("expected ByteString, got {:?}", other),
                },
                other => panic!("expected Sequence, got {:?}", other),
            },
            other => panic!("expected Choices, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_parens() {
        let g = parse_and_display("root ::= ()");
        assert_eq!(g, "root ::= ((\"\"))");
    }

    #[test]
    fn test_nested_groups() {
        let g = parse_and_display("root ::= ((\"a\"))");
        assert_eq!(g, "root ::= ((((((\"a\"))))))");
    }
}
