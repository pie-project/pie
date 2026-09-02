//! The `.zt` reading algorithm (spec §8) and validation summary (spec §3.6).
//!
//! Two entry points onto the same rules: [`image`] over a complete in-memory
//! file, which is what the conformance corpus and the fuzz targets drive, and
//! an internal one over an opened container, which reads only the footer and
//! the manifest blob, so opening a 100 GB checkpoint to plan against it touches
//! two ranges, not a hundred gigabytes.
//!
//! Canonical form (§6.4) is decided here too, as a pure function of a manifest
//! and of where the file put things. The reader supplies both; nothing here
//! opens a path. [`read::canonical_violations`](crate::read::canonical_violations)
//! is the entry point that does.

use xxhash_rust::xxh3::xxh3_64;

use crate::error::{Error, Result, Rule};
use crate::format::cbor;
use crate::format::{
    align_up, check_name, check_shape, check_shard_name, Manifest, ALIGN_CANONICAL, ALIGN_FLOOR,
    FOOTER_LEN, MAGIC, MAX_MANIFEST_LEN, MIN_FILE_LEN, VERSION,
};
use crate::provide::store::Store;
use crate::vocab::Vocabulary;

/// What the footer points at.
pub(crate) struct Footer {
    offset: u64,
    length: u64,
    hash: u64,
}

/// A validated `.zt` file: its manifest and where the manifest blob lies
/// (absent for a data shard), and every byte range it occupies.
pub(crate) struct Parsed {
    pub manifest: Option<(Manifest, Placement)>,
    pub occupied: Vec<(u64, u64)>,
}

/// Checks the frame every container shares (minimum size, header magic,
/// footer magic, supported version) and returns the footer bytes. `read`
/// fetches a range of the file.
fn check_frame(file_len: u64, read: impl Fn(u64, u64) -> Result<Vec<u8>>) -> Result<Vec<u8>> {
    if file_len < MIN_FILE_LEN {
        return Err(Error::reject(
            Rule::FileTooSmall,
            format!("file shorter than {MIN_FILE_LEN} B"),
        ));
    }
    if read(0, MAGIC.len() as u64)? != MAGIC {
        return Err(Error::reject(Rule::HeaderMagic, "bad header magic"));
    }
    let footer = read(file_len - FOOTER_LEN, FOOTER_LEN)?;
    if footer[32..40] != MAGIC {
        return Err(Error::reject(Rule::FooterMagic, "bad footer magic"));
    }
    let version = u32::from_le_bytes(footer[24..28].try_into().unwrap());
    if version != VERSION {
        return Err(Error::reject(
            Rule::Version,
            format!("unsupported version {version}; this build reads {VERSION}"),
        ));
    }
    Ok(footer)
}

/// `None` where a footer describes a data shard (spec §7.2): no manifest,
/// and the other fields must be zero.
fn parse_footer(footer: &[u8]) -> Result<Option<Footer>> {
    let offset = u64::from_le_bytes(footer[0..8].try_into().unwrap());
    let length = u64::from_le_bytes(footer[8..16].try_into().unwrap());
    let hash = u64::from_le_bytes(footer[16..24].try_into().unwrap());
    if length == 0 {
        if offset != 0 || hash != 0 {
            return Err(Error::reject(
                Rule::ManifestBounds,
                "data shard footer must zero manifest offset and hash",
            ));
        }
        return Ok(None);
    }
    if length > MAX_MANIFEST_LEN {
        return Err(Error::reject(Rule::ManifestTooLarge, "manifest over 1 GiB"));
    }
    if offset % ALIGN_FLOOR != 0 || offset < ALIGN_FLOOR {
        return Err(Error::reject(
            Rule::BlobAlignment,
            "manifest blob is misaligned",
        ));
    }
    Ok(Some(Footer {
        offset,
        length,
        hash,
    }))
}

fn manifest_bounds(footer: &Footer, file_len: u64) -> Result<(u64, u64)> {
    let data_end = file_len - FOOTER_LEN;
    let end = footer
        .offset
        .checked_add(footer.length)
        .filter(|&e| e <= data_end)
        .ok_or_else(|| Error::reject(Rule::ManifestBounds, "manifest outside data region"))?;
    Ok((end, data_end))
}

fn decode_manifest(bytes: &[u8], expected_hash: u64) -> Result<Manifest> {
    if xxh3_64(bytes) != expected_hash {
        return Err(Error::reject(Rule::ManifestHash, "manifest hash mismatch"));
    }
    Manifest::from_value(cbor::decode(bytes)?)
}

fn frame_ranges(mut ranges: Vec<(u64, u64)>, file_len: u64) -> Vec<(u64, u64)> {
    ranges.push((0, MAGIC.len() as u64));
    ranges.push((file_len - FOOTER_LEN, FOOTER_LEN));
    ranges.sort_unstable();
    ranges.dedup();
    ranges
}

/// Validates a complete in-memory `.zt` file image and returns its manifest
/// (`None` for a data shard).
pub fn image(buf: &[u8], vocab: &Vocabulary) -> Result<Option<Manifest>> {
    let file_len = buf.len() as u64;
    let footer = check_frame(file_len, |at, n| Ok(buf[at as usize..(at + n) as usize].to_vec()))?;
    let Some(footer) = parse_footer(&footer)? else {
        return Ok(None);
    };
    let (manifest_end, data_end) = manifest_bounds(&footer, file_len)?;
    let manifest = decode_manifest(
        &buf[footer.offset as usize..manifest_end as usize],
        footer.hash,
    )?;
    validate_manifest(&manifest, data_end, (footer.offset, footer.length), vocab)?;
    Ok(Some(manifest))
}

/// Where a file put the things canonical form has an opinion about.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Placement {
    pub manifest_at: u64,
    pub manifest_len: u64,
    pub file_len: u64,
}

/// Checks a manifest and its placement against canonical form (spec §6.4) and
/// returns every rule they break, in rule order. An empty list means the file
/// is canonical.
///
/// Blob sharing (rule 3) is judged by digest and length rather than by
/// comparing payloads. Rule 4 guarantees every object carries one.
pub(crate) fn canonical_violations(manifest: &Manifest, at: &Placement) -> Vec<String> {
    let Placement {
        manifest_at,
        manifest_len,
        file_len,
    } = *at;

    let mut bad = Vec::new();

    if !manifest.shards.is_empty() {
        bad.push(format!(
            "rule 6: canonical form is single-file, but the manifest declares {} shard(s)",
            manifest.shards.len()
        ));
    }

    for name in manifest.objects.keys() {
        if !name.is_ascii() && !unicode_normalization::is_nfc(name) {
            bad.push(format!("rule 5: object name {name:?} is not in NFC"));
        }
    }

    let mut cursor = MAGIC.len() as u64;
    let mut placed: std::collections::BTreeMap<u64, u64> = Default::default();
    let mut by_content: std::collections::BTreeMap<(Vec<u8>, u64), u64> = Default::default();
    for (name, object) in &manifest.objects {
        let blob = &object.blob;
        let label = format!("{name:?}");
        if blob.encoding.is_some() {
            bad.push(format!("rule 4: {label} is stored under an encoding"));
        }
        match &blob.digest {
            None => bad.push(format!("rule 4: {label} carries no digest")),
            Some(d) if d.algorithm != "xxh3" => {
                bad.push(format!("rule 4: {label} has a {d} digest, not xxh3"));
            }
            Some(_) => {}
        }
        if blob.blocks.is_some() {
            bad.push(format!("rule 4: {label} carries block digests"));
        }
        if blob.shard.is_some() {
            continue;
        }

        let offset = blob.offset;
        if let Some(len) = placed.get(&offset) {
            if *len != blob.length {
                bad.push(format!(
                    "rule 3: {label} shares offset {offset} with a blob of a different length"
                ));
            }
            continue;
        }
        if let Some(d) = &blob.digest {
            let key = (d.value.clone(), blob.length);
            if let Some(&first) = by_content.get(&key) {
                bad.push(format!(
                    "rule 3: {label} repeats the content already at offset {first} \
                     instead of sharing that blob"
                ));
            } else {
                by_content.insert(key, offset);
            }
        }

        let expected = match align_up(cursor, ALIGN_CANONICAL) {
            Some(e) => e,
            None => {
                bad.push(format!("rule 2: {label} places a blob past the end of u64"));
                break;
            }
        };
        if !offset.is_multiple_of(ALIGN_CANONICAL) {
            bad.push(format!(
                "rule 2: {label} is at offset {offset}, which is not a multiple of \
                 {ALIGN_CANONICAL}"
            ));
        } else if offset > expected {
            bad.push(format!(
                "rule 1: {} bytes before {label} at {offset} belong to nothing; \
                 canonical form packs blobs with no gaps",
                offset - expected
            ));
        } else if offset < expected {
            bad.push(format!(
                "rule 3: {label} is at {offset}, before the {expected} that its place in \
                 name order gives it"
            ));
        }
        placed.insert(offset, blob.length);
        cursor = offset + blob.length;
    }

    if let Some(expected) = align_up(cursor, ALIGN_CANONICAL) {
        if manifest_at != expected {
            bad.push(format!(
                "rule 1: the manifest is at {manifest_at}, but the blobs end at {expected}; \
                 the space between belongs to nothing"
            ));
        }
    }
    if manifest_at + manifest_len + FOOTER_LEN != file_len {
        bad.push(format!(
            "rule 3: the footer does not immediately follow the manifest \
             (manifest ends at {}, file is {file_len})",
            manifest_at + manifest_len
        ));
    }
    bad
}

/// Opens a `.zt` store: reads the footer and the manifest blob, validates
/// both, and reports every occupied range.
pub(crate) fn store(store: &Store, vocab: &Vocabulary) -> Result<Parsed> {
    let file_len = store.len();
    let footer = check_frame(file_len, |at, n| store.read(at, n))?;
    let Some(footer) = parse_footer(&footer)? else {
        return Ok(Parsed {
            manifest: None,
            occupied: frame_ranges(Vec::new(), file_len),
        });
    };
    let (_, data_end) = manifest_bounds(&footer, file_len)?;
    let manifest = decode_manifest(&store.read(footer.offset, footer.length)?, footer.hash)?;
    let occupied = validate_manifest(&manifest, data_end, (footer.offset, footer.length), vocab)?;
    let placement = Placement {
        manifest_at: footer.offset,
        manifest_len: footer.length,
        file_len,
    };
    Ok(Parsed {
        manifest: Some((manifest, placement)),
        occupied: frame_ranges(occupied, file_len),
    })
}

/// Every metadata rule of §3.6. Returns the blob ranges that live in this
/// file, manifest blob included.
pub(crate) fn validate_manifest(
    manifest: &Manifest,
    data_end: u64,
    manifest_blob: (u64, u64),
    vocab: &Vocabulary,
) -> Result<Vec<(u64, u64)>> {
    for (name, shard) in &manifest.shards {
        check_shard_name(name)?;
        if shard.size < MIN_FILE_LEN {
            return Err(Error::reject(
                Rule::Schema,
                format!(
                    "shard {name:?}: size {} below minimum file size",
                    shard.size
                ),
            ));
        }
    }

    let mut refs_by_shard: std::collections::BTreeMap<Option<&str>, Vec<(u64, u64)>> =
        std::collections::BTreeMap::new();
    refs_by_shard.insert(None, vec![manifest_blob]);

    for (name, obj) in &manifest.objects {
        check_name(name)?;
        check_shape(&obj.shape).map_err(|e| e.at(name))?;

        let b = &obj.blob;
        let shard = b.shard.as_deref();
        if b.offset % ALIGN_FLOOR != 0 || b.offset < ALIGN_FLOOR {
            return Err(Error::reject(
                Rule::BlobAlignment,
                format!("{name:?}: offset {} misaligned", b.offset),
            ));
        }
        let region_end = match shard {
            None => data_end,
            Some(s) => manifest
                .shards
                .get(s)
                .ok_or_else(|| {
                    Error::reject(
                        Rule::ShardRef,
                        format!("{name:?}: shard {s:?} not in table"),
                    )
                })?
                .size
                .checked_sub(FOOTER_LEN)
                .ok_or_else(|| Error::reject(Rule::BlobBounds, "shard smaller than footer"))?,
        };
        b.offset
            .checked_add(b.length)
            .filter(|&e| e <= region_end)
            .ok_or_else(|| {
                Error::reject(
                    Rule::BlobBounds,
                    format!("{name:?}: blob outside data region"),
                )
            })?;
        if b.length > 0 {
            refs_by_shard
                .entry(shard)
                .or_default()
                .push((b.offset, b.length));
        }

        match &obj.layout {
            None => {
                let expected = obj
                    .canonical_size()
                    .map_err(|e| e.at(name))?;
                if b.decoded_size() != expected {
                    return Err(Error::reject(
                        Rule::Size,
                        format!(
                            "{name:?}: decoded size {} != the {expected} its shape and type require",
                            b.decoded_size()
                        ),
                    ));
                }
            }
            Some(layout) => {
                if let Some(profile) = vocab.layout(layout) {
                    profile.validate(name, obj)?;
                }
            }
        }
    }

    for (shard, refs) in &mut refs_by_shard {
        refs.sort_unstable();
        refs.dedup();
        for w in refs.windows(2) {
            let (a_off, a_len) = w[0];
            let (b_off, _) = w[1];
            if a_off + a_len > b_off {
                let location = match shard {
                    None => "this file".to_string(),
                    Some(s) => format!("shard {s:?}"),
                };
                return Err(Error::reject(
                    Rule::BlobOverlap,
                    format!(
                        "blobs at {a_off} (+{a_len}) and {b_off} in {location} \
                         partially overlap"
                    ),
                ));
            }
        }
    }
    Ok(refs_by_shard.remove(&None).unwrap_or_default())
}
