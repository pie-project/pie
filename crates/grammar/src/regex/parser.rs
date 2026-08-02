use anyhow::{Result, bail};

use crate::frontend::FrontendExpr;

const UNICODE_MAX: u32 = 0x10ffff;
const SURROGATE_START: u32 = 0xd800;
const SURROGATE_END: u32 = 0xdfff;

pub(super) fn parse(pattern: &str) -> Result<FrontendExpr> {
    RegexParser::new(pattern).parse()
}

struct RegexParser<'a> {
    input: &'a [u8],
    pos: usize,
}

#[derive(Debug)]
enum ClassItem {
    Codepoint(u32),
    Ranges(Vec<(u32, u32)>),
}

impl ClassItem {
    fn into_ranges(self) -> Vec<(u32, u32)> {
        match self {
            Self::Codepoint(codepoint) => vec![(codepoint, codepoint)],
            Self::Ranges(ranges) => ranges,
        }
    }

    fn codepoint(self) -> Result<u32> {
        match self {
            Self::Codepoint(codepoint) => Ok(codepoint),
            Self::Ranges(_) => bail!("character-class shorthand cannot be a range endpoint"),
        }
    }
}

impl<'a> RegexParser<'a> {
    fn new(pattern: &'a str) -> Self {
        Self {
            input: pattern.as_bytes(),
            pos: 0,
        }
    }

    fn parse(mut self) -> Result<FrontendExpr> {
        if self.peek() == Some(b'^') {
            self.advance();
        }
        let result = self.parse_alternation()?;
        if self.peek() == Some(b'$') {
            self.advance();
        }
        if !self.at_end() {
            bail!(
                "unexpected character at position {}: '{}'",
                self.pos,
                self.input[self.pos] as char
            );
        }
        Ok(result)
    }

    fn parse_alternation(&mut self) -> Result<FrontendExpr> {
        let mut alternatives = vec![self.parse_sequence()?];
        while self.peek() == Some(b'|') {
            self.advance();
            alternatives.push(self.parse_sequence()?);
        }
        Ok(FrontendExpr::choice(alternatives))
    }

    fn parse_sequence(&mut self) -> Result<FrontendExpr> {
        let mut elements = Vec::new();
        while !self.at_end() && !matches!(self.peek(), Some(b'|') | Some(b')') | Some(b'$')) {
            elements.push(self.parse_quantified_atom()?);
        }
        Ok(FrontendExpr::sequence(elements))
    }

    fn parse_quantified_atom(&mut self) -> Result<FrontendExpr> {
        let atom = self.parse_atom()?;
        let repeated = match self.peek() {
            Some(b'*') => {
                self.advance();
                atom.repeat(0, None)
            }
            Some(b'+') => {
                self.advance();
                atom.repeat(1, None)
            }
            Some(b'?') => {
                self.advance();
                atom.repeat(0, Some(1))
            }
            Some(b'{') => {
                let (min, max) = self.parse_repetition()?;
                atom.repeat(min, max)
            }
            _ => return Ok(atom),
        };

        if self.peek() == Some(b'?') {
            self.advance();
        }
        if matches!(
            self.peek(),
            Some(b'*') | Some(b'+') | Some(b'?') | Some(b'{')
        ) {
            bail!(
                "consecutive quantifiers are not supported at position {}",
                self.pos
            );
        }
        Ok(repeated)
    }

    fn parse_atom(&mut self) -> Result<FrontendExpr> {
        match self.peek() {
            None => Ok(FrontendExpr::Empty),
            Some(b'(') => self.parse_group(),
            Some(b'[') => self.parse_character_class(),
            Some(b'.') => {
                self.advance();
                Ok(character_class(vec![(0, UNICODE_MAX)]))
            }
            Some(b'\\') => self.parse_escape(),
            Some(b'^') => {
                self.advance();
                Ok(FrontendExpr::Empty)
            }
            Some(ch) if is_metachar(ch) => {
                bail!(
                    "unexpected metacharacter '{}' at position {}",
                    ch as char,
                    self.pos
                )
            }
            Some(_) => self.parse_literal(),
        }
    }

    fn parse_group(&mut self) -> Result<FrontendExpr> {
        self.expect_byte(b'(')?;
        if self.peek() == Some(b'?') {
            self.advance();
            match self.peek() {
                Some(b':') => {
                    self.advance();
                }
                Some(b'=') | Some(b'!') => bail!("lookahead assertions are not supported"),
                Some(b'<') if matches!(self.input.get(self.pos + 1), Some(b'=') | Some(b'!')) => {
                    bail!("lookbehind assertions are not supported")
                }
                Some(b'<') => {
                    self.advance();
                    while !self.at_end() && self.peek() != Some(b'>') {
                        self.advance_char();
                    }
                    if self.peek() != Some(b'>') {
                        bail!("unterminated named group");
                    }
                    self.advance();
                }
                _ => bail!("unsupported group modifier"),
            }
        }

        let expression = self.parse_alternation()?;
        if self.peek() != Some(b')') {
            bail!("unmatched '(' at position {}", self.pos);
        }
        self.advance();
        Ok(FrontendExpr::Group(Box::new(expression)))
    }

    fn parse_character_class(&mut self) -> Result<FrontendExpr> {
        self.expect_byte(b'[')?;
        let negated = if self.peek() == Some(b'^') {
            self.advance();
            true
        } else {
            false
        };
        let mut ranges = Vec::new();

        if self.peek() == Some(b']') {
            self.advance();
            ranges.push((b']' as u32, b']' as u32));
        }

        while self.peek() != Some(b']') {
            if self.at_end() {
                bail!("unclosed character class");
            }
            let first = self.parse_class_item()?;
            if self.peek() == Some(b'-') && self.input.get(self.pos + 1) != Some(&b']') {
                self.advance();
                let start = first.codepoint()?;
                if self.at_end() {
                    bail!("unclosed character class");
                }
                let end = self.parse_class_item()?.codepoint()?;
                if start > end {
                    bail!("invalid character class range");
                }
                ranges.push((start, end));
            } else {
                ranges.extend(first.into_ranges());
            }
        }
        self.advance();

        Ok(FrontendExpr::CharacterClass {
            negated,
            ranges: normalize_ranges(ranges),
        })
    }

    fn parse_class_item(&mut self) -> Result<ClassItem> {
        if self.peek() == Some(b'\\') {
            self.advance();
            return self.parse_class_escape();
        }
        let codepoint = self.advance_char() as u32;
        Ok(ClassItem::Codepoint(codepoint))
    }

    fn parse_class_escape(&mut self) -> Result<ClassItem> {
        let escaped = self
            .peek()
            .ok_or_else(|| anyhow::anyhow!("truncated escape"))?;
        match escaped {
            b'd' => {
                self.advance();
                Ok(ClassItem::Ranges(digit_ranges()))
            }
            b'D' => {
                self.advance();
                Ok(ClassItem::Ranges(complement_ranges(&digit_ranges())))
            }
            b'w' => {
                self.advance();
                Ok(ClassItem::Ranges(word_ranges()))
            }
            b'W' => {
                self.advance();
                Ok(ClassItem::Ranges(complement_ranges(&word_ranges())))
            }
            b's' => {
                self.advance();
                Ok(ClassItem::Ranges(whitespace_ranges()))
            }
            b'S' => {
                self.advance();
                Ok(ClassItem::Ranges(complement_ranges(&whitespace_ranges())))
            }
            b'u' => Ok(ClassItem::Codepoint(self.parse_unicode_escape()?)),
            b'x' => Ok(ClassItem::Codepoint(self.parse_hex_escape()?)),
            _ => {
                self.advance();
                Ok(ClassItem::Codepoint(simple_escape(escaped)))
            }
        }
    }

    fn parse_escape(&mut self) -> Result<FrontendExpr> {
        self.expect_byte(b'\\')?;
        let escaped = self
            .peek()
            .ok_or_else(|| anyhow::anyhow!("truncated escape"))?;
        match escaped {
            b'd' => {
                self.advance();
                Ok(character_class(digit_ranges()))
            }
            b'D' => {
                self.advance();
                Ok(character_class(complement_ranges(&digit_ranges())))
            }
            b'w' => {
                self.advance();
                Ok(character_class(word_ranges()))
            }
            b'W' => {
                self.advance();
                Ok(character_class(complement_ranges(&word_ranges())))
            }
            b's' => {
                self.advance();
                Ok(character_class(whitespace_ranges()))
            }
            b'S' => {
                self.advance();
                Ok(character_class(complement_ranges(&whitespace_ranges())))
            }
            b'b' | b'B' => bail!("word boundaries (\\b, \\B) are not supported"),
            b'p' | b'P' => bail!("unicode property escapes (\\p, \\P) are not supported"),
            b'1'..=b'9' | b'k' => bail!("backreferences are not supported"),
            b'u' => codepoint_literal(self.parse_unicode_escape()?),
            b'x' => codepoint_literal(self.parse_hex_escape()?),
            _ => {
                self.advance();
                codepoint_literal(simple_escape(escaped))
            }
        }
    }

    fn parse_unicode_escape(&mut self) -> Result<u32> {
        self.expect_byte(b'u')?;
        let codepoint = if self.peek() == Some(b'{') {
            self.advance();
            let start = self.pos;
            while self.peek().is_some_and(|ch| ch.is_ascii_hexdigit()) {
                self.advance();
            }
            let digits = self.pos - start;
            if digits == 0 || digits > 6 || self.peek() != Some(b'}') {
                bail!("invalid unicode escape");
            }
            let value = parse_radix(&self.input[start..self.pos], 16)?;
            self.advance();
            value
        } else {
            let bytes = self.read_hex(4)?;
            parse_radix(bytes, 16)?
        };
        validate_codepoint(codepoint)
    }

    fn parse_hex_escape(&mut self) -> Result<u32> {
        self.expect_byte(b'x')?;
        let bytes = self.read_hex(2)?;
        parse_radix(bytes, 16)
    }

    fn parse_literal(&mut self) -> Result<FrontendExpr> {
        let mut bytes = Vec::new();
        let mut count = 0;
        while !self.at_end() {
            let Some(first) = self.peek() else {
                break;
            };
            if is_metachar(first) || matches!(first, b'\\' | b'[' | b'(' | b'.') {
                break;
            }
            let start = self.pos;
            let ch = self.advance_char();
            let quantified = matches!(
                self.peek(),
                Some(b'*') | Some(b'+') | Some(b'?') | Some(b'{')
            );
            if quantified && count > 0 {
                self.pos = start;
                break;
            }
            let mut encoded = [0; 4];
            bytes.extend_from_slice(ch.encode_utf8(&mut encoded).as_bytes());
            count += 1;
            if quantified {
                break;
            }
        }
        Ok(FrontendExpr::literal(bytes))
    }

    fn parse_repetition(&mut self) -> Result<(u32, Option<u32>)> {
        self.expect_byte(b'{')?;
        let min = self.read_integer()?;
        match self.peek() {
            Some(b'}') => {
                self.advance();
                Ok((min, Some(min)))
            }
            Some(b',') => {
                self.advance();
                if self.peek() == Some(b'}') {
                    self.advance();
                    return Ok((min, None));
                }
                let max = self.read_integer()?;
                if max < min {
                    bail!("repetition lower bound exceeds upper bound");
                }
                self.expect_byte(b'}')?;
                Ok((min, Some(max)))
            }
            _ => bail!("expected ',' or '}}' in repetition"),
        }
    }

    fn read_integer(&mut self) -> Result<u32> {
        let mut value = 0u32;
        let mut has_digit = false;
        while let Some(ch) = self.peek() {
            if !ch.is_ascii_digit() {
                break;
            }
            has_digit = true;
            value = value
                .checked_mul(10)
                .and_then(|value| value.checked_add((ch - b'0') as u32))
                .ok_or_else(|| anyhow::anyhow!("repetition count exceeds u32::MAX"))?;
            self.advance();
        }
        if !has_digit {
            bail!("expected integer in repetition");
        }
        Ok(value)
    }

    fn read_hex(&mut self, count: usize) -> Result<&'a [u8]> {
        let start = self.pos;
        for _ in 0..count {
            if !self.peek().is_some_and(|ch| ch.is_ascii_hexdigit()) {
                bail!("expected hex digit");
            }
            self.advance();
        }
        Ok(&self.input[start..self.pos])
    }

    fn expect_byte(&mut self, expected: u8) -> Result<()> {
        if self.peek() != Some(expected) {
            bail!("expected '{}'", expected as char);
        }
        self.advance();
        Ok(())
    }

    fn peek(&self) -> Option<u8> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let value = self.peek();
        if value.is_some() {
            self.pos += 1;
        }
        value
    }

    fn advance_char(&mut self) -> char {
        let first = self.input[self.pos];
        if first.is_ascii() {
            self.pos += 1;
            return first as char;
        }
        let ch = std::str::from_utf8(&self.input[self.pos..])
            .expect("regex source remains valid UTF-8")
            .chars()
            .next()
            .unwrap();
        self.pos += ch.len_utf8();
        ch
    }

    fn at_end(&self) -> bool {
        self.pos == self.input.len()
    }
}

fn character_class(ranges: Vec<(u32, u32)>) -> FrontendExpr {
    FrontendExpr::CharacterClass {
        negated: false,
        ranges: normalize_ranges(ranges),
    }
}

fn codepoint_literal(codepoint: u32) -> Result<FrontendExpr> {
    let codepoint = validate_codepoint(codepoint)?;
    let ch = char::from_u32(codepoint).unwrap();
    let mut bytes = [0; 4];
    Ok(FrontendExpr::literal(
        ch.encode_utf8(&mut bytes).as_bytes().to_vec(),
    ))
}

fn validate_codepoint(codepoint: u32) -> Result<u32> {
    if codepoint > UNICODE_MAX || (SURROGATE_START..=SURROGATE_END).contains(&codepoint) {
        bail!("invalid Unicode codepoint U+{:X}", codepoint);
    }
    Ok(codepoint)
}

fn simple_escape(escaped: u8) -> u32 {
    match escaped {
        b'n' => b'\n' as u32,
        b'r' => b'\r' as u32,
        b't' => b'\t' as u32,
        b'f' => 0x0c,
        b'v' => 0x0b,
        b'0' => 0,
        _ => escaped as u32,
    }
}

fn digit_ranges() -> Vec<(u32, u32)> {
    vec![(b'0' as u32, b'9' as u32)]
}

fn word_ranges() -> Vec<(u32, u32)> {
    vec![
        (b'0' as u32, b'9' as u32),
        (b'A' as u32, b'Z' as u32),
        (b'_' as u32, b'_' as u32),
        (b'a' as u32, b'z' as u32),
    ]
}

fn whitespace_ranges() -> Vec<(u32, u32)> {
    vec![(b'\t' as u32, b'\r' as u32), (b' ' as u32, b' ' as u32)]
}

fn normalize_ranges(mut ranges: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    ranges.sort_unstable();
    let mut normalized: Vec<(u32, u32)> = Vec::new();
    for (start, end) in ranges {
        if let Some(last) = normalized.last_mut() {
            if start <= last.1.saturating_add(1) {
                last.1 = last.1.max(end);
                continue;
            }
        }
        normalized.push((start, end));
    }
    normalized
}

fn complement_ranges(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let ranges = normalize_ranges(ranges.to_vec());
    let mut result = Vec::new();
    let mut next = 0;
    for (start, end) in ranges {
        if next < start {
            result.push((next, start - 1));
        }
        next = end.saturating_add(1);
    }
    if next <= UNICODE_MAX {
        result.push((next, UNICODE_MAX));
    }
    result
}

fn parse_radix(bytes: &[u8], radix: u32) -> Result<u32> {
    let text = std::str::from_utf8(bytes).expect("hex digits are ASCII");
    u32::from_str_radix(text, radix).map_err(Into::into)
}

fn is_metachar(ch: u8) -> bool {
    matches!(
        ch,
        b'*' | b'+' | b'?' | b'{' | b'}' | b'|' | b')' | b'^' | b'$'
    )
}
