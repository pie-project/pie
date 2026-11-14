//! What a code object asks of the HIP runtime beyond its own image.
//!
//! A kernel whose metadata names a hostcall, printf or heap buffer among
//! its hidden arguments (a `printf`, an `assert`, a device `malloc`) has the
//! runtime allocate that buffer, and start a listener thread for a hostcall,
//! at the kernel's first launch: some tens of MiB on a large device, after
//! the load, outside every ledger. A load refuses such a code object by the
//! kernel's name instead.

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

/// Refuses `image` where one of its kernels asks the runtime for a buffer
/// it allocates at first launch, naming the kernel and the buffer. An image
/// that is not an ELF code object, or carries no metadata, asks nothing.
pub fn refuse_late_buffers(image: &[u8]) -> Result<(), String> {
    let asked = late_buffers(image);
    if asked.is_empty() {
        return Ok(());
    }
    Err(asked
        .iter()
        .map(|(kernel, buffer)| {
            format!(
                "kernel {kernel} asks the HIP runtime for a {buffer} buffer, which it \
                 allocates at the kernel's first launch"
            )
        })
        .collect::<Vec<_>>()
        .join("; "))
}

/// Each kernel of `image` and the late buffer it asks for.
fn late_buffers(image: &[u8]) -> Vec<(String, &'static str)> {
    let mut asked = Vec::new();
    for meta in notes(image).filter_map(msgpack::read) {
        let Some(Mp::Arr(kernels)) = meta.get("amdhsa.kernels") else {
            continue;
        };
        for kernel in kernels {
            let name = kernel.get(".name").and_then(Mp::text).unwrap_or("?");
            let Some(Mp::Arr(args)) = kernel.get(".args") else {
                continue;
            };
            for arg in args {
                let kind = arg.get(".value_kind").and_then(Mp::text);
                if let Some(&(_, buffer)) = LATE_BUFFERS.iter().find(|(k, _)| Some(*k) == kind) {
                    asked.push((name.to_owned(), buffer));
                }
            }
        }
    }
    asked
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
    let elf64 =
        image.starts_with(b"\x7fELF") && image.get(4) == Some(&2) && image.get(5) == Some(&1);
    let sections = elf64
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

#[cfg(test)]
mod tests {
    use super::{NT_AMDGPU_METADATA, SHT_NOTE, refuse_late_buffers};

    /// MessagePack for a string.
    fn s(out: &mut Vec<u8>, text: &str) {
        assert!(text.len() < 32);
        out.push(0xa0 | text.len() as u8);
        out.extend_from_slice(text.as_bytes());
    }

    /// The metadata of one kernel `name` whose one argument is `kind`.
    fn metadata(name: &str, kind: &str) -> Vec<u8> {
        let mut m = vec![0x81];
        s(&mut m, "amdhsa.kernels");
        m.extend_from_slice(&[0x91, 0x82]);
        s(&mut m, ".name");
        s(&mut m, name);
        s(&mut m, ".args");
        m.extend_from_slice(&[0x91, 0x81]);
        s(&mut m, ".value_kind");
        s(&mut m, kind);
        m
    }

    /// An ELF64 image whose one section is a note carrying `desc` as AMDGPU
    /// metadata.
    fn image(desc: &[u8]) -> Vec<u8> {
        let mut note = Vec::new();
        note.extend_from_slice(&7u32.to_le_bytes());
        note.extend_from_slice(&(desc.len() as u32).to_le_bytes());
        note.extend_from_slice(&NT_AMDGPU_METADATA.to_le_bytes());
        note.extend_from_slice(b"AMDGPU\0\0");
        note.extend_from_slice(desc);
        while note.len() % 4 != 0 {
            note.push(0);
        }
        let mut elf = vec![0u8; 64];
        elf[..6].copy_from_slice(b"\x7fELF\x02\x01");
        let note_at = elf.len();
        elf.extend_from_slice(&note);
        let table = elf.len();
        elf[0x28..0x30].copy_from_slice(&(table as u64).to_le_bytes());
        elf[0x3a..0x3c].copy_from_slice(&64u16.to_le_bytes());
        elf[0x3c..0x3e].copy_from_slice(&1u16.to_le_bytes());
        let mut header = vec![0u8; 64];
        header[4..8].copy_from_slice(&SHT_NOTE.to_le_bytes());
        header[0x18..0x20].copy_from_slice(&(note_at as u64).to_le_bytes());
        header[0x20..0x28].copy_from_slice(&(note.len() as u64).to_le_bytes());
        elf.extend_from_slice(&header);
        elf
    }

    #[test]
    fn a_kernel_asking_a_late_buffer_is_refused_by_name() {
        let refused = refuse_late_buffers(&image(&metadata("pull", "hidden_hostcall_buffer")));
        assert_eq!(
            refused,
            Err(
                "kernel pull asks the HIP runtime for a hostcall buffer, which it allocates at \
                 the kernel's first launch"
                    .to_owned()
            )
        );
        assert!(refuse_late_buffers(&image(&metadata("say", "hidden_printf_buffer"))).is_err());
        assert!(refuse_late_buffers(&image(&metadata("grow", "hidden_heap_v1"))).is_err());
    }

    #[test]
    fn a_kernel_asking_nothing_late_loads() {
        assert_eq!(
            refuse_late_buffers(&image(&metadata("add", "global_buffer"))),
            Ok(())
        );
        assert_eq!(
            refuse_late_buffers(&image(&metadata("add", "hidden_block_count_x"))),
            Ok(())
        );
        assert_eq!(refuse_late_buffers(b"not an elf"), Ok(()));
        assert_eq!(refuse_late_buffers(&[]), Ok(()));
    }
}
