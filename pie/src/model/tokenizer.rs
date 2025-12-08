use base64::{Engine as _, engine::general_purpose};
use fancy_regex::Regex;
use std::collections::{HashMap, HashSet};
use std::fs;
pub type Rank = u32;

// The code below is copied from the tiktoken.
// https://github.com/openai/tiktoken/blob/main/src/lib.rs

fn _byte_pair_merge(ranks: &HashMap<Vec<u8>, Rank>, piece: &[u8]) -> Vec<(usize, Rank)> {
    let mut parts = Vec::with_capacity(piece.len() + 1);

    let mut min_rank: (Rank, usize) = (Rank::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = *ranks.get(&piece[i..i + 2]).unwrap_or(&Rank::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, Rank::MAX));
    parts.push((piece.len(), Rank::MAX));

    let get_rank = {
        |parts: &Vec<(usize, Rank)>, i: usize| {
            if (i + 3) < parts.len() {
                *ranks
                    .get(&piece[parts[i].0..parts[i + 3].0])
                    .unwrap_or(&Rank::MAX)
            } else {
                Rank::MAX
            }
        }
    };

    while min_rank.0 != Rank::MAX {
        let i = min_rank.1;

        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (Rank::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    parts
}

pub fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<Rank> {
    if piece.len() == 1 {
        return vec![ranks[piece]];
    }
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| ranks[&piece[part[0].0..part[1].0]])
        .collect()
}

pub fn byte_pair_split<'a>(piece: &'a [u8], ranks: &HashMap<Vec<u8>, Rank>) -> Vec<&'a [u8]> {
    assert!(piece.len() > 1);
    _byte_pair_merge(ranks, piece)
        .windows(2)
        .map(|part| &piece[part[0].0..part[1].0])
        .collect()
}

#[derive(Debug, Clone)]
pub struct DecodeKeyError {
    pub token: Rank,
}

impl std::fmt::Display for DecodeKeyError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Invalid token for decoding: {}", self.token)
    }
}

impl std::error::Error for DecodeKeyError {}

#[derive(Debug, Clone)]
pub struct DecodeError {
    pub message: String,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Could not decode tokens: {}", self.message)
    }
}

impl std::error::Error for DecodeError {}

#[derive(Clone, Debug)]
pub struct BytePairEncoder {
    num_vocab: usize,
    encoder: HashMap<Vec<u8>, Rank>,
    special_tokens_encoder: HashMap<String, Rank>,
    decoder: HashMap<Rank, Vec<u8>>,
    special_tokens_decoder: HashMap<Rank, Vec<u8>>,
    regex: Regex,
    special_regex: Regex,
    escape_non_printable: bool,
}

impl BytePairEncoder {
    pub fn num_vocab(&self) -> usize {
        self.num_vocab
    }

    pub fn decode(&self, tokens: &[Rank]) -> Result<String, DecodeError> {
        // First, decode raw bytes from the tokens.
        let decoded_bytes = self.decode_bytes(tokens).map_err(|err| DecodeError {
            message: err.to_string(),
        })?;

        // Then, convert the bytes to a UTF-8 string.
        // Using `from_utf8_lossy` would silently replace invalid sequences with
        // the Unicode replacement character
        Ok(String::from_utf8_lossy(&*decoded_bytes).to_string())
    }

    fn decode_bytes(&self, tokens: &[Rank]) -> Result<Vec<u8>, DecodeKeyError> {
        let mut ret = Vec::with_capacity(tokens.len() * 2);
        for &token in tokens {
            let token_bytes = match self.decoder.get(&token) {
                Some(bytes) => bytes,
                None => self
                    .special_tokens_decoder
                    .get(&token)
                    .ok_or(DecodeKeyError { token })?,
            };

            let is_special = self.special_tokens_decoder.contains_key(&token);

            if !is_special && self.escape_non_printable {
                let decoded_string = String::from_utf8_lossy(token_bytes);
                let decoded_bytes = unescape_non_printable(&decoded_string).unwrap();
                ret.extend(&decoded_bytes);
            } else {
                ret.extend(token_bytes);
            }
        }
        Ok(ret)
    }

    pub fn encode(&self, text: &str, allowed_special: &HashSet<&str>) -> Vec<Rank> {
        let mut ret = vec![];

        let mut start = 0;
        loop {
            let mut next_special;
            let mut start_find = start;
            loop {
                next_special = self.special_regex.find_from_pos(text, start_find).unwrap();
                match next_special {
                    Some(m) => {
                        if allowed_special.contains(&text[m.start()..m.end()]) {
                            break;
                        }
                        start_find = m.start() + 1;
                    }
                    None => break,
                }
            }
            let end = next_special.map_or(text.len(), |m| m.start());

            for mat in self.regex.find_iter(&text[start..end]) {
                let mut piece = mat.unwrap().as_str().as_bytes();

                let escaped_piece = escape_non_printable(piece);
                if self.escape_non_printable {
                    piece = escaped_piece.as_bytes();
                }

                if let Some(token) = self.encoder.get(piece) {
                    ret.push(*token);
                    continue;
                }
                let tokens = byte_pair_encode(piece, &self.encoder);
                ret.extend(&tokens);
            }

            match next_special {
                // And here we push the special token
                Some(m) => {
                    let piece = m.as_str();
                    let token = self.special_tokens_encoder[piece];
                    ret.push(token);
                    start = m.end();
                }
                None => break,
            }
        }

        ret
    }

    pub(crate) fn new(
        num_vocab: usize,
        decoder: HashMap<Rank, Vec<u8>>,
        special_tokens_encoder: HashMap<String, Rank>,
        pattern: &str,
        escape_non_printable: bool,
    ) -> Self {
        let regex = Regex::new(pattern).unwrap();

        let special_regex = {
            let parts = special_tokens_encoder
                .keys()
                .map(|s| fancy_regex::escape(s))
                .collect::<Vec<_>>();
            Regex::new(&parts.join("|")).unwrap()
        };

        let encoder: HashMap<Vec<u8>, Rank> =
            decoder.iter().map(|(k, v)| (v.clone(), *k)).collect();

        assert_eq!(
            encoder.len(),
            decoder.len(),
            "Encoder and decoder must be of equal length; maybe you had duplicate token indices in your encoder?"
        );

        let special_tokens_decoder: HashMap<Rank, Vec<u8>> = special_tokens_encoder
            .iter()
            .map(|(k, v)| (*v, k.as_bytes().to_vec()))
            .collect();

        // Clone because I don't know how to tell Rust I'm not going to change the map

        Self {
            num_vocab,
            encoder,
            special_tokens_encoder,
            decoder,
            special_tokens_decoder,
            regex,
            special_regex,
            escape_non_printable,
        }
    }

    pub fn encode_with_special_tokens(&self, text: &str) -> Vec<Rank> {
        let allowed_special = self
            .special_tokens_encoder
            .keys()
            .map(|s| s.as_str())
            .collect();
        self.encode(text, &allowed_special)
    }

    pub fn get_special_tokens(&self) -> (Vec<Rank>, Vec<Vec<u8>>) {
        (
            self.special_tokens_decoder.keys().cloned().collect(),
            self.special_tokens_decoder.values().cloned().collect(),
        )
    }

    pub fn get_split_regex(&self) -> String {
        self.regex.to_string()
    }

    pub fn get_vocabs(&self) -> (Vec<Rank>, Vec<Vec<u8>>) {
        (
            self.decoder.keys().cloned().collect(),
            self.decoder.values().cloned().collect(),
        )
    }
}

pub fn load_merge_rules(path: &str) -> Result<HashMap<Rank, Vec<u8>>, Box<dyn std::error::Error>> {
    // Read the entire file as a UTF-8 string
    let contents = fs::read_to_string(path)?;

    let mut ret = HashMap::new();

    for (line_number, line) in contents.lines().enumerate() {
        let line = line.trim();
        // Skip empty lines
        if line.is_empty() {
            continue;
        }

        // Expect two parts: base64-encoded token and rank
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            return Err(format!(
                "Error parsing line {}: expected two parts, got {} (line: {:?})",
                line_number,
                parts.len(),
                line
            )
            .into());
        }

        let b64_token = parts[0];
        let rank_str = parts[1];

        // Decode base64 token
        let decoded_token = general_purpose::STANDARD
            .decode(b64_token)
            .map_err(|e| format!("Error decoding base64 at line {}: {}", line_number, e))?;

        // Parse rank into i32
        let rank = rank_str
            .parse::<Rank>()
            .map_err(|e| format!("Error parsing rank at line {}: {}", line_number, e))?;

        // Insert into the HashMap
        ret.insert(rank, decoded_token);
    }

    Ok(ret)
}

/// Generate the 256-entry “byte-level” maps.
///
///  * `enc[byte] -> char`  (stage-2 encoding)
///  * `dec[char] -> byte`  (stage-2 decoding)
///
/// The algorithm is identical to OpenAI-tiktoken’s `bytes_to_unicode()`.
///
/// Printable ranges kept as-is:
///   1.  '!' (0x21) .. '~' (0x7E)
///   2.  '¡' (0xA1) .. '¬' (0xAC)
///   3.  '®' (0xAE) .. 'ÿ' (0xFF)
///
/// Everything else (control bytes, space, TAB, …) is
/// remapped to the BMP starting at U+0100.
fn build_tables() -> ([char; 256], HashMap<char, u8>) {
    // Step 1: collect the “safe” byte values we keep unchanged
    let mut bs: Vec<u8> = (b'!'..=b'~').collect(); // 0x21–0x7E
    bs.extend(0xA1..=0xAC); // 0xA1–0xAC
    bs.extend(0xAE..=0xFF); // 0xAE–0xFF

    // cs will hold the *Unicode code points* corresponding to bs
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();

    // Step 2: assign code points ≥ 0x100 to the remaining bytes
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n); // U+0100, U+0101, …
            n += 1;
        }
    }

    // Convert to char
    let cs: Vec<char> = cs.into_iter().map(|u| char::from_u32(u).unwrap()).collect();

    // Zip into the forward & reverse tables
    let mut enc = ['\0'; 256];
    let mut dec = HashMap::with_capacity(256);
    for (b, ch) in bs.into_iter().zip(cs.into_iter()) {
        enc[b as usize] = ch;
        dec.insert(ch, b);
    }
    (enc, dec)
}

/// Encode a byte slice with the Qwen/GPT byte-level mapping.
pub fn escape_non_printable(bytes: &[u8]) -> String {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    bytes.iter().map(|&b| TABLES.0[b as usize]).collect()
}

/// Decode a stage-2 string back to raw bytes.
pub fn unescape_non_printable(s: &str) -> Option<Vec<u8>> {
    static TABLES: once_cell::sync::Lazy<([char; 256], HashMap<char, u8>)> =
        once_cell::sync::Lazy::new(build_tables);

    let mut out = Vec::with_capacity(s.len());
    for ch in s.chars() {
        match TABLES.1.get(&ch) {
            Some(&b) => out.push(b),
            None => return None, // invalid symbol
        }
    }
    Some(out)
}
