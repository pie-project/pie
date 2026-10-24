//! The journal of a persistent disk tier: which slots of the kept slot files
//! hold which sealed pages, so that the next run finds them again.
//!
//! A header names what the slots were written for (the cache domain, the
//! page size and the slot count); a journal whose header does not match is
//! started afresh. Then records, each summed: a page put in a slot, or slots
//! given up. A record that does not sum is where the journal ends: a run that
//! died mid-append loses at most what it was appending. Opening compacts the
//! journal to the pages still held.
//!
//! The slots' bytes are the engine's, written before the record that names
//! them; a slot is given up, durably, before anything else is written there.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use super::hash::Hash256;

const MAGIC: &[u8; 8] = b"piekvj02";
const PUT: u8 = 1;
const GIVEN_UP: u8 = 2;

/// What the slots were written for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub domain: Hash256,
    pub page_size: u32,
    pub slots: u32,
}

/// A sealed page a slot holds: its key, the key of the page before it (none
/// at token 0), its tokens' hashes, and the cluster retention key it was
/// kept under: swap takes the lowest first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageRecord {
    pub page: Hash256,
    pub parent: Option<Hash256>,
    pub tokens: Vec<Option<Hash256>>,
    pub retention: i64,
}

#[derive(Debug)]
pub struct Journal {
    file: File,
    path: PathBuf,
}

impl Journal {
    /// The journal at `path`, and the pages it holds by slot. Unless `kept`
    /// (the slot files hold what the last run left) and its header is
    /// `header`, it starts empty.
    pub fn open(
        path: &Path,
        header: Header,
        kept: bool,
    ) -> io::Result<(Self, BTreeMap<u32, PageRecord>)> {
        let held = if kept {
            match std::fs::read(path) {
                Ok(bytes) => replay(&bytes, header),
                Err(error) if error.kind() == io::ErrorKind::NotFound => BTreeMap::new(),
                Err(error) => return Err(error),
            }
        } else {
            BTreeMap::new()
        };
        let mut bytes = encode_header(header);
        for (&slot, record) in &held {
            bytes.extend(encode_record(&put(slot, record)));
        }
        let tmp = path.with_extension("journal.tmp");
        {
            let mut file = File::create(&tmp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        if let Some(dir) = path.parent() {
            sync_dir(dir)?;
        }
        let file = OpenOptions::new().append(true).open(path)?;
        Ok((
            Self {
                file,
                path: path.to_path_buf(),
            },
            held,
        ))
    }

    /// Records that `slot` holds `record`; durable after [`Self::sync`].
    pub fn put(&mut self, slot: u32, record: &PageRecord) -> io::Result<()> {
        self.file.write_all(&encode_record(&put(slot, record)))
    }

    /// Records that `slots` hold nothing any more, durably: only then may
    /// anything else be written there.
    pub fn give_up(&mut self, slots: &[u32]) -> io::Result<()> {
        let mut body = vec![GIVEN_UP];
        body.extend((slots.len() as u32).to_le_bytes());
        for slot in slots {
            body.extend(slot.to_le_bytes());
        }
        self.file.write_all(&encode_record(&body))?;
        self.sync()
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// Removes the journal, so that the next run trusts none of the slots:
    /// what follows a write that failed.
    pub fn abandon(self) -> io::Result<()> {
        drop(self.file);
        std::fs::remove_file(&self.path)
    }
}

fn put(slot: u32, record: &PageRecord) -> Vec<u8> {
    let mut body = vec![PUT];
    body.extend(slot.to_le_bytes());
    body.extend(record.page);
    put_hash(&mut body, record.parent.as_ref());
    body.extend((record.tokens.len() as u32).to_le_bytes());
    for token in &record.tokens {
        put_hash(&mut body, token.as_ref());
    }
    body.extend(record.retention.to_le_bytes());
    body
}

fn put_hash(body: &mut Vec<u8>, hash: Option<&Hash256>) {
    match hash {
        Some(hash) => {
            body.push(1);
            body.extend(hash);
        }
        None => body.push(0),
    }
}

fn sum(bytes: &[u8]) -> [u8; 8] {
    let mut out = [0; 8];
    out.copy_from_slice(&blake3::hash(bytes).as_bytes()[..8]);
    out
}

fn encode_header(header: Header) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    bytes.extend(header.domain);
    bytes.extend(header.page_size.to_le_bytes());
    bytes.extend(header.slots.to_le_bytes());
    let summed = sum(&bytes);
    bytes.extend(summed);
    bytes
}

fn encode_record(body: &[u8]) -> Vec<u8> {
    let mut bytes = (body.len() as u32).to_le_bytes().to_vec();
    bytes.extend(body);
    bytes.extend(sum(body));
    bytes
}

/// The pages `bytes` hold by slot, as far as they sum; none unless the
/// header is `header`.
fn replay(bytes: &[u8], header: Header) -> BTreeMap<u32, PageRecord> {
    let mut held = BTreeMap::new();
    let expected = encode_header(header);
    if !bytes.starts_with(&expected) {
        if !bytes.is_empty() {
            tracing::info!(target: "pie::kv", "disk kv journal is for another load; starting afresh");
        }
        return held;
    }
    let mut reader = &bytes[expected.len()..];
    while let Some(body) = next_record(&mut reader) {
        let mut body = body;
        let Some(tag) = take_u8(&mut body) else {
            break;
        };
        let applied = match tag {
            PUT => decode_put(&mut body).map(|(slot, record)| {
                held.insert(slot, record);
            }),
            GIVEN_UP => take_u32(&mut body).and_then(|n| {
                for _ in 0..n {
                    held.remove(&take_u32(&mut body)?);
                }
                Some(())
            }),
            _ => None,
        };
        if applied.is_none() || !body.is_empty() {
            break;
        }
    }
    held
}

fn next_record<'a>(reader: &mut &'a [u8]) -> Option<&'a [u8]> {
    let len = take_u32(reader)? as usize;
    if reader.len() < len + 8 {
        return None;
    }
    let (body, rest) = reader.split_at(len);
    let (summed, rest) = rest.split_at(8);
    if sum(body) != summed {
        return None;
    }
    *reader = rest;
    Some(body)
}

fn decode_put(body: &mut &[u8]) -> Option<(u32, PageRecord)> {
    let slot = take_u32(body)?;
    let page = take_hash(body)?;
    let parent = take_maybe_hash(body)?;
    let n = take_u32(body)?;
    let tokens = (0..n)
        .map(|_| take_maybe_hash(body))
        .collect::<Option<Vec<_>>>()?;
    let mut retention = [0; 8];
    body.read_exact(&mut retention).ok()?;
    Some((
        slot,
        PageRecord {
            page,
            parent,
            tokens,
            retention: i64::from_le_bytes(retention),
        },
    ))
}

fn take_u8(body: &mut &[u8]) -> Option<u8> {
    let mut byte = [0; 1];
    body.read_exact(&mut byte).ok()?;
    Some(byte[0])
}

fn take_u32(body: &mut &[u8]) -> Option<u32> {
    let mut bytes = [0; 4];
    body.read_exact(&mut bytes).ok()?;
    Some(u32::from_le_bytes(bytes))
}

fn take_hash(body: &mut &[u8]) -> Option<Hash256> {
    let mut hash = [0; 32];
    body.read_exact(&mut hash).ok()?;
    Some(hash)
}

fn take_maybe_hash(body: &mut &[u8]) -> Option<Option<Hash256>> {
    match take_u8(body)? {
        0 => Some(None),
        1 => take_hash(body).map(Some),
        _ => None,
    }
}

#[cfg(unix)]
fn sync_dir(dir: &Path) -> io::Result<()> {
    File::open(dir)?.sync_all()
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) -> io::Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> Header {
        Header {
            domain: [3; 32],
            page_size: 16,
            slots: 8,
        }
    }

    fn record(page: u8, parent: Option<u8>) -> PageRecord {
        PageRecord {
            page: [page; 32],
            parent: parent.map(|p| [p; 32]),
            tokens: vec![Some([page; 32]), None],
            retention: i64::from(page),
        }
    }

    #[test]
    fn a_journal_gives_back_what_it_holds_until_a_slot_is_given_up() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g0.journal");
        {
            let (mut journal, held) = Journal::open(&path, header(), true).unwrap();
            assert!(held.is_empty());
            journal.put(1, &record(1, None)).unwrap();
            journal.put(4, &record(2, Some(1))).unwrap();
            journal.put(5, &record(3, Some(2))).unwrap();
            journal.give_up(&[5]).unwrap();
        }
        let (_, held) = Journal::open(&path, header(), true).unwrap();
        assert_eq!(
            held.into_iter().collect::<Vec<_>>(),
            vec![(1, record(1, None)), (4, record(2, Some(1)))]
        );
        // Opened again, compacted: the same.
        let (_, held) = Journal::open(&path, header(), true).unwrap();
        assert_eq!(held.len(), 2);
    }

    #[test]
    fn a_journal_for_another_load_or_unkept_slots_holds_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g0.journal");
        Journal::open(&path, header(), true)
            .unwrap()
            .0
            .put(1, &record(1, None))
            .unwrap();
        let other = Header {
            page_size: 32,
            ..header()
        };
        assert!(Journal::open(&path, other, true).unwrap().1.is_empty());
        Journal::open(&path, header(), true)
            .unwrap()
            .0
            .put(1, &record(1, None))
            .unwrap();
        assert!(
            Journal::open(&path, header(), false).unwrap().1.is_empty(),
            "slots the engine took afresh read as kept"
        );
    }

    #[test]
    fn a_torn_tail_ends_the_journal_where_it_stops_summing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("g0.journal");
        {
            let (mut journal, _) = Journal::open(&path, header(), true).unwrap();
            journal.put(1, &record(1, None)).unwrap();
            journal.put(2, &record(2, Some(1))).unwrap();
        }
        let len = std::fs::metadata(&path).unwrap().len();
        let file = OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(len - 3).unwrap();
        let (_, held) = Journal::open(&path, header(), true).unwrap();
        assert_eq!(
            held.into_iter().collect::<Vec<_>>(),
            vec![(1, record(1, None))]
        );
    }
}
