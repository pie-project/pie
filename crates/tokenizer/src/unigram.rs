use rustc_hash::FxHashMap;

#[derive(Debug)]
pub(crate) struct UnigramScores {
    pieces: FxHashMap<Box<str>, (u32, f32)>,
    max_len: usize,
    unk_id: u32,
}

impl UnigramScores {
    pub(crate) fn new(pieces: &[(String, f32)], unk_id: u32) -> anyhow::Result<Self> {
        anyhow::ensure!(!pieces.is_empty(), "Unigram vocabulary is empty");
        anyhow::ensure!(
            (unk_id as usize) < pieces.len(),
            "unk_id {unk_id} is outside a {}-piece vocabulary",
            pieces.len()
        );
        let mut max_len = 0usize;
        let mut map: FxHashMap<Box<str>, (u32, f32)> =
            FxHashMap::with_capacity_and_hasher(pieces.len(), Default::default());
        for (index, (piece, score)) in pieces.iter().enumerate() {
            if piece.is_empty() {
                continue;
            }
            max_len = max_len.max(piece.len());
            let id = u32::try_from(index)
                .map_err(|_| anyhow::anyhow!("Unigram vocabulary is longer than a u32 counts"))?;
            map.entry(piece.as_str().into()).or_insert((id, *score));
        }
        Ok(Self {
            pieces: map,
            max_len,
            unk_id,
        })
    }

    pub(crate) fn scores_by_id(&self, vocab: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; vocab];
        for (id, score) in self.pieces.values() {
            if let Some(cell) = out.get_mut(*id as usize) {
                *cell = *score;
            }
        }
        out
    }

    pub(crate) fn unk_id(&self) -> u32 {
        self.unk_id
    }

    pub(crate) fn from_scores(pieces: &[(String, f32)], unk_id: u32) -> anyhow::Result<Self> {
        Self::new(pieces, unk_id)
    }

    pub(crate) fn encode(&self, text: &str, ids: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        let bytes = text.len();
        let mut best = vec![f32::NEG_INFINITY; bytes + 1];
        let mut from = vec![(usize::MAX, 0u32); bytes + 1];
        best[0] = 0.0;

        let unk_penalty = self
            .pieces
            .values()
            .map(|(_, score)| *score)
            .fold(f32::INFINITY, f32::min)
            - 10.0;

        for at in 0..bytes {
            if best[at] == f32::NEG_INFINITY || !text.is_char_boundary(at) {
                continue;
            }
            let rest = &text[at..];
            let mut reached = false;
            let ceiling = self.max_len.min(rest.len());
            for end in 1..=ceiling {
                if !rest.is_char_boundary(end) {
                    continue;
                }
                let Some((id, score)) = self.pieces.get(&rest[..end]) else {
                    continue;
                };
                reached = true;
                let candidate = best[at] + score;
                if candidate > best[at + end] {
                    best[at + end] = candidate;
                    from[at + end] = (at, *id);
                }
            }
            let step = rest.chars().next().map_or(0, char::len_utf8);
            if step > 0 {
                let candidate = best[at] + unk_penalty;
                if candidate > best[at + step] {
                    best[at + step] = candidate;
                    from[at + step] = (at, self.unk_id);
                }
            }
            let _ = reached;
        }

        let mut walk = Vec::new();
        let mut at = bytes;
        while at > 0 {
            let (start, id) = from[at];
            if start == usize::MAX {
                walk.clear();
                walk.extend(text.chars().map(|_| self.unk_id));
                break;
            }
            walk.push(id);
            at = start;
        }
        walk.reverse();
        ids.extend(walk);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unigram_every_case() {
        the_walk_is_exact_and_not_greedy();
        the_walk_takes_the_long_piece_when_it_scores_better();
        an_unreachable_character_is_one_unk_and_the_walk_continues();
        a_multibyte_character_is_one_unk();
        an_unk_outside_the_vocabulary_is_refused_at_build();
    }

    fn the_walk_is_exact_and_not_greedy() {
        let pieces = vec![
            ("<unk>".to_string(), -100.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
            ("ab".to_string(), -9.0),
        ];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("ab", &mut ids);
        assert_eq!(ids, vec![1, 2], "-1 + -1 beats -9");
    }

    fn the_walk_takes_the_long_piece_when_it_scores_better() {
        let pieces = vec![
            ("<unk>".to_string(), -100.0),
            ("a".to_string(), -5.0),
            ("b".to_string(), -5.0),
            ("ab".to_string(), -1.0),
        ];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("ab", &mut ids);
        assert_eq!(ids, vec![3]);
    }

    fn an_unreachable_character_is_one_unk_and_the_walk_continues() {
        let pieces = vec![
            ("<unk>".to_string(), -100.0),
            ("a".to_string(), -1.0),
            ("b".to_string(), -1.0),
        ];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("azb", &mut ids);
        assert_eq!(ids, vec![1, 0, 2], "`z` is one unk between two pieces");
    }

    fn a_multibyte_character_is_one_unk() {
        let pieces = vec![("<unk>".to_string(), -100.0), ("a".to_string(), -1.0)];
        let table = UnigramScores::new(&pieces, 0).expect("a vocabulary");
        let mut ids = Vec::new();
        table.encode("a\u{4e16}a", &mut ids);
        assert_eq!(ids, vec![1, 0, 1]);
    }

    fn an_unk_outside_the_vocabulary_is_refused_at_build() {
        let pieces = vec![("a".to_string(), -1.0)];
        let why = UnigramScores::new(&pieces, 7).expect_err("no such piece");
        assert!(why.to_string().contains("unk_id 7"), "{why}");
    }
}
