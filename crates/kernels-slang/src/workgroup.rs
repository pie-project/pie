//! The static workgroup memory (`groupshared`) a kernel's SPIR-V module
//! declares: what a device's `maxComputeSharedMemorySize` must hold for the
//! kernel to run, read from the module itself.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::encode::{Encode, Fire};

const MAGIC: u32 = 0x0723_0203;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const STORAGE_WORKGROUP: u32 = 4;
const MAX_DEPTH: u32 = 32;

/// The bytes of workgroup memory the SPIR-V module `code` declares: every
/// `Workgroup` variable's type, laid naturally (elements packed, the
/// layout a driver lays them by at least). `None` where the module is not
/// SPIR-V or a variable's size is not static.
#[must_use]
pub fn declared(code: &[u8]) -> Option<u32> {
    if !code.len().is_multiple_of(4) {
        return None;
    }
    let words: Vec<u32> = code
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    if words.len() < 5 || words[0] != MAGIC {
        return None;
    }
    let mut types: HashMap<u32, (u32, Vec<u32>)> = HashMap::new();
    let mut constants: HashMap<u32, u32> = HashMap::new();
    let mut pointers: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut shared: Vec<u32> = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let op = words[i] & 0xffff;
        if count == 0 || i + count > words.len() {
            return None;
        }
        let w = &words[i..i + count];
        match op {
            OP_TYPE_BOOL | OP_TYPE_INT | OP_TYPE_FLOAT | OP_TYPE_VECTOR | OP_TYPE_MATRIX
            | OP_TYPE_ARRAY | OP_TYPE_STRUCT
                if count >= 2 =>
            {
                types.insert(w[1], (op, w[2..].to_vec()));
            }
            OP_TYPE_POINTER if count >= 4 => {
                pointers.insert(w[1], (w[2], w[3]));
            }
            OP_CONSTANT if count >= 4 => {
                constants.insert(w[2], w[3]);
            }
            OP_VARIABLE if count >= 4 && w[3] == STORAGE_WORKGROUP => {
                shared.push(w[1]);
            }
            _ => {}
        }
        i += count;
    }
    let mut total = 0u32;
    for pointer in shared {
        let &(class, pointee) = pointers.get(&pointer)?;
        if class != STORAGE_WORKGROUP {
            return None;
        }
        total = total.checked_add(size_of(pointee, &types, &constants, 0)?)?;
    }
    Some(total)
}

fn size_of(
    ty: u32,
    types: &HashMap<u32, (u32, Vec<u32>)>,
    constants: &HashMap<u32, u32>,
    depth: u32,
) -> Option<u32> {
    if depth > MAX_DEPTH {
        return None;
    }
    let (op, operands) = types.get(&ty)?;
    let of = |inner: u32| size_of(inner, types, constants, depth + 1);
    match *op {
        OP_TYPE_BOOL => Some(4),
        OP_TYPE_INT | OP_TYPE_FLOAT => Some(operands.first()?.div_ceil(8)),
        OP_TYPE_VECTOR | OP_TYPE_MATRIX => of(*operands.first()?)?.checked_mul(*operands.get(1)?),
        OP_TYPE_ARRAY => of(*operands.first()?)?.checked_mul(*constants.get(operands.get(1)?)?),
        OP_TYPE_STRUCT => operands
            .iter()
            .try_fold(0u32, |sum, &member| sum.checked_add(of(member)?)),
        _ => None,
    }
}

/// The workgroup memory the module built under `key` declares, read once.
#[must_use]
pub fn of_key(key: &str) -> Option<u32> {
    static READ: OnceLock<HashMap<&'static str, Option<u32>>> = OnceLock::new();
    READ.get_or_init(|| {
        crate::sources::SPIRV
            .iter()
            .map(|(key, _, code)| (*key, declared(code)))
            .collect()
    })
    .get(key)
    .copied()
    .flatten()
}

/// The workgroup memory `fire`'s kernel declares, where its SPIR-V module
/// is built in.
#[must_use]
pub fn of_fire(fire: &Fire) -> Option<u32> {
    of_key(crate::sources::key(fire.entrypoint, fire.tier)?)
}

/// Whether `fire`'s kernel fits the workgroup memory `ctx`'s device gives
/// a workgroup ([`Encode::workgroup_limit`]): a tactic whose kernel does
/// not is passed over for the next where one is chosen among several.
#[must_use]
pub fn fits(ctx: &dyn Encode, fire: &Fire) -> bool {
    match (ctx.workgroup_limit(), of_fire(fire)) {
        (Some(limit), Some(bytes)) => bytes <= limit,
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every shipped module's workgroup memory reads statically.
    #[test]
    fn every_shipped_module_states_its_workgroup_memory() {
        for (key, _, code) in crate::sources::SPIRV {
            assert!(declared(code).is_some(), "{key}: workgroup memory unread");
        }
    }

    /// The tiled gemm declares its two 32 x 32 f32 tiles: 8 KiB.
    #[test]
    fn the_tiled_gemm_declares_its_tiles() {
        if crate::sources::SPIRV.is_empty() {
            return;
        }
        assert_eq!(of_key("linear_gemm_tiled_bf16"), Some(8192));
    }
}
