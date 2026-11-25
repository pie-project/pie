//! What a code object asks of the HIP runtime beyond its own image.
//!
//! A kernel whose metadata names a hostcall, printf or heap buffer among
//! its hidden arguments (a `printf`, an `assert`, a device `malloc`) has the
//! runtime allocate that buffer, and start a listener thread for a hostcall,
//! at the kernel's first launch. A kernel with a private segment (a spill,
//! a private array) or a dynamic stack has ROCr grow the queue's scratch
//! when a dispatch of it first needs more than the queue holds, from ROCr's
//! own thread as the device reaches it. Either is some tens of MiB on a
//! large device, taken after the load, outside every ledger and at a time
//! no fire chooses. A load refuses such a code object by the kernel's name
//! instead.

use super::msgpack::{self, Mp};

/// `NT_AMDGPU_METADATA`: the note whose descriptor is the code object's
/// MessagePack metadata.
const NT_AMDGPU_METADATA: u32 = 32;
const SHT_NOTE: u32 = 7;

/// The hidden arguments whose buffers the runtime allocates at first launch.
const LATE_BUFFERS: &[(&str, &str)] = &[
    ("hidden_hostcall_buffer", "hostcall"),
    ("hidden_printf_buffer", "printf"),
    ("hidden_heap_v1", "heap"),
];

/// What a kernel asks the runtime for late.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ask {
    /// A hostcall, printf or heap buffer, allocated at its first launch.
    Buffer(&'static str),
    /// Scratch, this many bytes a lane, grown when a dispatch first needs it.
    Scratch(u64),
    /// A dynamic stack in scratch, grown likewise.
    DynamicStack,
}

impl std::fmt::Display for Ask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ask::Buffer(buffer) => write!(
                f,
                "a {buffer} buffer, which it allocates at the kernel's first launch"
            ),
            Ask::Scratch(bytes) => write!(
                f,
                "scratch ({bytes} bytes a lane), which ROCr grows when a dispatch first needs it"
            ),
            Ask::DynamicStack => write!(
                f,
                "a dynamic stack in scratch, which ROCr grows when a dispatch first needs it"
            ),
        }
    }
}

/// Refuses `image` where one of its kernels asks the runtime for memory it
/// allocates late (a buffer at first launch, or scratch), naming the kernel
/// and what it asks. An image this cannot read (not an ELF code object, or
/// one that carries no kernel metadata) is refused too: what it asks is not
/// known. An offload bundle is unbundled first ([`unbundle`]).
pub fn refuse_late_buffers(image: &[u8]) -> Result<(), String> {
    refuse(&asked(image)?)
}

fn refuse(asked: &[(String, Ask)]) -> Result<(), String> {
    if asked.is_empty() {
        return Ok(());
    }
    Err(asked
        .iter()
        .map(|(kernel, ask)| format!("kernel {kernel} asks the HIP runtime for {ask}"))
        .collect::<Vec<_>>()
        .join("; "))
}

/// Each kernel of `image` and the late memory it asks for.
fn asked(image: &[u8]) -> Result<Vec<(String, Ask)>, String> {
    if !elf64(image) {
        return Err("the image is not an ELF code object, so what its kernels ask is not known".into());
    }
    let mut asked = Vec::new();
    let mut described = false;
    for meta in notes(image).filter_map(msgpack::read) {
        let Some(Mp::Arr(kernels)) = meta.get("amdhsa.kernels") else {
            continue;
        };
        described = true;
        for kernel in kernels {
            let name = kernel.get(".name").and_then(Mp::text).unwrap_or("?");
            let args = match kernel.get(".args") {
                Some(Mp::Arr(args)) => args.as_slice(),
                _ => &[],
            };
            for arg in args {
                let kind = arg.get(".value_kind").and_then(Mp::text);
                if let Some(&(_, buffer)) = LATE_BUFFERS.iter().find(|(k, _)| Some(*k) == kind) {
                    asked.push((name.to_owned(), Ask::Buffer(buffer)));
                }
            }
            let private = kernel
                .get(".private_segment_fixed_size")
                .and_then(Mp::int)
                .and_then(|bytes| u64::try_from(bytes).ok())
                .unwrap_or(0);
            if private > 0 {
                asked.push((name.to_owned(), Ask::Scratch(private)));
            }
            if kernel.get(".uses_dynamic_stack").is_some_and(Mp::truth) {
                asked.push((name.to_owned(), Ask::DynamicStack));
            }
        }
    }
    if !described {
        return Err("the code object carries no AMDGPU kernel metadata, so what its kernels ask is not known".into());
    }
    Ok(asked)
}

/// Whether `image` is an ELF64 little-endian image, as an AMDGPU code
/// object is.
fn elf64(image: &[u8]) -> bool {
    image.starts_with(b"\x7fELF") && image.get(4) == Some(&2) && image.get(5) == Some(&1)
}

/// The code object for `arch` (`gfx942`) out of a clang offload bundle,
/// compressed (`CCOB`, zlib or zstd) or not; an ELF image is its own code
/// object. A bundle with no entry for `arch` is refused by name.
#[cfg(feature = "rocm")]
pub fn unbundle(image: &[u8], arch: &str) -> Result<Vec<u8>, String> {
    if elf64(image) {
        return Ok(image.to_vec());
    }
    let le = |bytes: &[u8], at: usize, n: usize| -> Option<u64> {
        let slice = bytes.get(at..at.checked_add(n)?)?;
        Some(slice.iter().rev().fold(0, |acc, &b| acc << 8 | u64::from(b)))
    };
    let bundle = if image.starts_with(b"CCOB") {
        // The compressed bundle's header: version, method, then (from
        // version 3) 64-bit total and uncompressed sizes, and a hash.
        let version = le(image, 4, 2).ok_or("a CCOB header cut short")?;
        let method = le(image, 6, 2).ok_or("a CCOB header cut short")?;
        let (header, size) = match version {
            2 => (24, le(image, 12, 4)),
            3 => (32, le(image, 16, 8)),
            other => return Err(format!("a CCOB bundle of version {other}, which this reads not")),
        };
        let size = usize::try_from(size.ok_or("a CCOB header cut short")?)
            .map_err(|_| "a CCOB bundle past the address space")?;
        let packed = image.get(header..).ok_or("a CCOB bundle with no payload")?;
        let unpacked = match method {
            0 => miniz_oxide::inflate::decompress_to_vec_zlib(packed)
                .map_err(|why| format!("a zlib CCOB bundle does not inflate: {why:?}"))?,
            1 => zstd::bulk::decompress(packed, size)
                .map_err(|why| format!("a zstd CCOB bundle does not inflate: {why}"))?,
            other => return Err(format!("a CCOB bundle compressed by method {other}, which this reads not")),
        };
        if unpacked.len() != size {
            return Err(format!("a CCOB bundle inflated to {} bytes, not the {size} it states", unpacked.len()));
        }
        unpacked
    } else {
        image.to_vec()
    };
    const MAGIC: &[u8] = b"__CLANG_OFFLOAD_BUNDLE__";
    if !bundle.starts_with(MAGIC) {
        return Err("the image is neither an ELF code object nor an offload bundle".into());
    }
    let cut = || "an offload bundle's entry table cut short".to_string();
    let entries = le(&bundle, MAGIC.len(), 8).ok_or_else(cut)?;
    let mut at = MAGIC.len() + 8;
    for _ in 0..entries {
        let offset = le(&bundle, at, 8).ok_or_else(cut)?;
        let size = le(&bundle, at + 8, 8).ok_or_else(cut)?;
        let id_len = usize::try_from(le(&bundle, at + 16, 8).ok_or_else(cut)?).map_err(|_| cut())?;
        let id = bundle.get(at + 24..at + 24 + id_len).ok_or_else(cut)?;
        at += 24 + id_len;
        // `hipv4-amdgcn-amd-amdhsa--gfx942[:features]`
        let target = std::str::from_utf8(id)
            .ok()
            .and_then(|id| id.rsplit_once("--"))
            .map(|(_, target)| target.split(':').next().unwrap_or_default());
        if target == Some(arch) {
            let (offset, size) = (usize::try_from(offset).map_err(|_| cut())?, usize::try_from(size).map_err(|_| cut())?);
            return bundle
                .get(offset..offset.checked_add(size).ok_or_else(cut)?)
                .map(<[u8]>::to_vec)
                .ok_or_else(cut);
        }
    }
    Err(format!("the offload bundle holds no code object for {arch}"))
}

/// The descriptors of `image`'s AMDGPU metadata notes: an ELF64
/// little-endian image's note sections, walked note by note.
fn notes(image: &[u8]) -> impl Iterator<Item = &[u8]> {
    let le = |at: usize, n: usize| -> Option<u64> {
        let bytes = image.get(at..at.checked_add(n)?)?;
        Some(
            bytes
                .iter()
                .rev()
                .fold(0, |acc, &b| acc << 8 | u64::from(b)),
        )
    };
    let sections = elf64(image)
        .then(|| {
            let table = usize::try_from(le(0x28, 8)?).ok()?;
            let entry = usize::try_from(le(0x3a, 2)?).ok()?;
            let count = usize::try_from(le(0x3c, 2)?).ok()?;
            Some((table, entry, count))
        })
        .flatten();
    let mut descs = Vec::new();
    if let Some((table, entry, count)) = sections {
        for index in 0..count {
            let Some(header) = index
                .checked_mul(entry)
                .and_then(|at| at.checked_add(table))
            else {
                break;
            };
            if le(header + 4, 4) != Some(u64::from(SHT_NOTE)) {
                continue;
            }
            let (Some(offset), Some(size)) = (le(header + 0x18, 8), le(header + 0x20, 8)) else {
                continue;
            };
            let (Ok(offset), Ok(size)) = (usize::try_from(offset), usize::try_from(size)) else {
                continue;
            };
            let Some(section) = offset
                .checked_add(size)
                .and_then(|end| image.get(offset..end))
            else {
                continue;
            };
            descs.extend(note_descs(section));
        }
    }
    descs.into_iter()
}

/// The metadata descriptors of one note section.
fn note_descs(section: &[u8]) -> Vec<&[u8]> {
    let word = |at: usize| -> Option<usize> {
        let bytes: [u8; 4] = section.get(at..at.checked_add(4)?)?.try_into().ok()?;
        usize::try_from(u32::from_le_bytes(bytes)).ok()
    };
    let pad = |n: usize| n.div_ceil(4) * 4;
    let mut descs = Vec::new();
    let mut at = 0;
    while let (Some(name), Some(desc), Some(kind)) = (word(at), word(at + 4), word(at + 8)) {
        let desc_at = at + 12 + pad(name);
        let Some(bytes) = desc_at
            .checked_add(desc)
            .and_then(|end| section.get(desc_at..end))
        else {
            break;
        };
        if kind == NT_AMDGPU_METADATA as usize {
            descs.push(bytes);
        }
        at = desc_at + pad(desc);
    }
    descs
}
