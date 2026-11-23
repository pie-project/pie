use std::collections::HashMap;

const MAGIC: u32 = 0x0723_0203;

const OP_EXECUTION_MODE: u32 = 16;
const OP_TYPE_BOOL: u32 = 20;
const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_LOAD: u32 = 61;
const OP_ACCESS_CHAIN: u32 = 65;
const OP_IN_BOUNDS_ACCESS_CHAIN: u32 = 66;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;

const MODE_LOCAL_SIZE: u32 = 17;
const DECORATION_ARRAY_STRIDE: u32 = 6;
const DECORATION_MATRIX_STRIDE: u32 = 7;
const DECORATION_NON_WRITABLE: u32 = 24;
const DECORATION_BINDING: u32 = 33;
const DECORATION_OFFSET: u32 = 35;
const STORAGE_PUSH_CONSTANT: u32 = 9;
const STORAGE_UNIFORM: u32 = 2;
const STORAGE_STORAGE_BUFFER: u32 = 12;
const DECORATION_BLOCK: u32 = 2;
const DECORATION_BUFFER_BLOCK: u32 = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declared {
    pub local: [u32; 3],

    pub bindings: u32,
    pub used: Vec<bool>,

    pub writable: Vec<bool>,

    pub push_bytes: u32,

    /// The fire block's uniform (`PIE_PARAMS_FIRED`), where the module reads
    /// some of its scalars.
    pub fire: Option<Fire>,
}

/// The scalars a module reads from the fire block: the uniform's binding,
/// its bytes (the scalars', as pushed), and the byte runs of the scalars it
/// reads there and never from the push block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fire {
    pub binding: u32,
    pub bytes: u32,
    pub words: Vec<(u32, u32)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Malformed {
    Truncated,
    NotSpirv,
    ZeroLengthInstruction,
    NoLocalSize,
    /// A binding the module declares as something other than a storage
    /// buffer, which every layout here binds it as.
    NotStorage {
        binding: u32,
    },
    /// A second uniform block, or one whose size the parser cannot read:
    /// the fire block is one (`PIE_PARAMS_FIRED`).
    Fire {
        binding: u32,
    },
}

impl std::fmt::Display for Malformed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => f.write_str("the SPIR-V is truncated"),
            Self::NotSpirv => f.write_str("the bytes are not SPIR-V"),
            Self::ZeroLengthInstruction => f.write_str("a zero-length SPIR-V instruction"),
            Self::NoLocalSize => f.write_str("the module states no LocalSize"),
            Self::NotStorage { binding } => write!(
                f,
                "binding {binding} is not a storage buffer, and its pipeline layout binds one"
            ),
            Self::Fire { binding } => write!(
                f,
                "binding {binding} is a uniform block beside another, or of a size not read; \
                 a kernel reads its fire block from one (`PIE_PARAMS_FIRED`)"
            ),
        }
    }
}

pub fn words(code: &[u8]) -> Result<Vec<u32>, Malformed> {
    if !code.len().is_multiple_of(4) {
        return Err(Malformed::Truncated);
    }
    Ok(code
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect())
}

pub fn declared(words: &[u32]) -> Result<Declared, Malformed> {
    if words.len() < 5 {
        return Err(Malformed::Truncated);
    }
    if words[0] != MAGIC {
        return Err(Malformed::NotSpirv);
    }
    let mut local = None;
    let mut highest: Option<u32> = None;
    let mut bound: Vec<(u32, u32)> = Vec::new();
    let mut non_writable: Vec<u32> = Vec::new();
    let mut constants: HashMap<u32, u32> = HashMap::new();
    let mut pointers: HashMap<u32, u32> = HashMap::new();
    let mut push_variables: Vec<u32> = Vec::new();
    let mut blocks: Vec<u32> = Vec::new();
    let mut push_variable: Option<u32> = None;
    // Each access chain's base and first index (a constant's id), and each
    // whole load's pointer (no index).
    let mut chains: Vec<(u32, Option<u32>)> = Vec::new();
    // Each variable's pointer type and storage class; types decorated
    // BufferBlock (a storage buffer in the Uniform class, older SPIR-V).
    let mut variables: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut buffer_blocks: Vec<u32> = Vec::new();
    let mut offsets: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    let mut types: HashMap<u32, (u32, Vec<u32>)> = HashMap::new();
    let mut strides: HashMap<u32, (Option<u32>, Option<u32>)> = HashMap::new();

    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let op = words[i] & 0xffff;
        if count == 0 {
            return Err(Malformed::ZeroLengthInstruction);
        }
        let end = i.checked_add(count).ok_or(Malformed::Truncated)?;
        if end > words.len() {
            return Err(Malformed::Truncated);
        }
        let w = &words[i..end];
        match op {
            OP_EXECUTION_MODE if count >= 6 && w[2] == MODE_LOCAL_SIZE => {
                local = Some([w[3], w[4], w[5]]);
            }
            OP_DECORATE if count >= 3 && w[2] == DECORATION_NON_WRITABLE => {
                non_writable.push(w[1]);
            }
            OP_DECORATE if count >= 3 && w[2] == DECORATION_BUFFER_BLOCK => {
                buffer_blocks.push(w[1]);
            }
            OP_DECORATE if count >= 3 && w[2] == DECORATION_BLOCK => {
                blocks.push(w[1]);
            }
            OP_DECORATE if count >= 4 => match w[2] {
                DECORATION_BINDING => {
                    highest = Some(highest.map_or(w[3], |h| h.max(w[3])));
                    bound.push((w[1], w[3]));
                }
                DECORATION_ARRAY_STRIDE => strides.entry(w[1]).or_default().0 = Some(w[3]),
                DECORATION_MATRIX_STRIDE => strides.entry(w[1]).or_default().1 = Some(w[3]),
                _ => {}
            },
            OP_CONSTANT if count >= 4 => {
                constants.insert(w[2], w[3]);
            }
            OP_TYPE_POINTER if count >= 4 => {
                pointers.insert(w[1], w[3]);
            }
            OP_VARIABLE if count >= 4 && w[3] == STORAGE_PUSH_CONSTANT => {
                push_variables.push(w[1]);
                push_variable = Some(w[2]);
            }
            OP_ACCESS_CHAIN | OP_IN_BOUNDS_ACCESS_CHAIN if count >= 5 => {
                chains.push((w[3], Some(w[4])));
            }
            OP_LOAD if count >= 4 => {
                chains.push((w[3], None));
            }
            OP_VARIABLE if count >= 4 => {
                variables.insert(w[2], (w[1], w[3]));
            }
            OP_TYPE_INT
            | OP_TYPE_FLOAT
            | OP_TYPE_BOOL
            | OP_TYPE_VECTOR
            | OP_TYPE_MATRIX
            | OP_TYPE_ARRAY
            | OP_TYPE_RUNTIME_ARRAY
            | OP_TYPE_STRUCT
                if count >= 2 =>
            {
                types.insert(w[1], (op, w[2..].to_vec()));
            }
            OP_MEMBER_DECORATE if count >= 5 && w[3] == DECORATION_OFFSET => {
                offsets.entry(w[1]).or_default().push((w[2], w[4]));
            }
            _ => {}
        }
        i = end;
    }

    let graph = Graph {
        types: &types,
        strides: &strides,
        constants: &constants,
        offsets: &offsets,
    };
    let bindings = highest.map_or(0, |h| h + 1);
    let mut used = vec![false; bindings as usize];
    let mut writable = vec![false; bindings as usize];
    let mut fire_at: Option<(u32, u32, u32)> = None;
    for &(variable, binding) in &bound {
        let var = variables.get(&variable).copied();
        let storage = var.is_none_or(|(pointer, class)| {
            class == STORAGE_STORAGE_BUFFER
                || (class == STORAGE_UNIFORM
                    && pointers
                        .get(&pointer)
                        .is_some_and(|block| buffer_blocks.contains(block)))
        });
        let uniform = var.and_then(|(pointer, class)| {
            let block = *pointers.get(&pointer)?;
            (class == STORAGE_UNIFORM && blocks.contains(&block)).then_some(block)
        });
        if let Some(block) = uniform {
            let bytes = size_of(block, &graph, 0).ok_or(Malformed::Fire { binding })?;
            if fire_at.is_some() {
                return Err(Malformed::Fire { binding });
            }
            fire_at = Some((variable, binding, bytes));
            used[binding as usize] = true;
            continue;
        }
        if !storage {
            return Err(Malformed::NotStorage { binding });
        }
        used[binding as usize] = true;
        if !non_writable.contains(&variable) {
            writable[binding as usize] = true;
        }
    }

    let push = if push_variables.is_empty() {
        None
    } else {
        push_block(words, &pointers, &offsets)
    };
    let push_bytes = push
        .and_then(|block| size_of(block, &graph, 0))
        .unwrap_or(0);

    // The members a variable's chains reach; every member where one loads it
    // whole.
    let members = |variable: u32, all: usize| -> Vec<u32> {
        let mut reached: Vec<u32> = Vec::new();
        for &(base, index) in &chains {
            if base != variable {
                continue;
            }
            match index.and_then(|id| constants.get(&id)) {
                Some(&member) => reached.push(member),
                None => reached.extend(0..all as u32),
            }
        }
        reached.sort_unstable();
        reached.dedup();
        reached
    };
    // The fire block's fields lie as the push block's; a module that reads
    // every scalar from the fire block keeps no push block, and the fire
    // block's own struct gives them.
    let fire_struct = fire_at.and_then(|(variable, _, _)| {
        variables
            .get(&variable)
            .and_then(|&(pointer, _)| pointers.get(&pointer).copied())
    });
    let fire = match (fire_at, push.or(fire_struct)) {
        (Some((variable, binding, bytes)), Some(block)) => {
            let layout = offsets.get(&block).cloned().unwrap_or_default();
            let (_, fields) = types.get(&block).cloned().unwrap_or((0, Vec::new()));
            let pushed = push_variable.map_or_else(Vec::new, |v| members(v, fields.len()));
            let words = members(variable, fields.len())
                .into_iter()
                .filter(|member| !pushed.contains(member))
                .map(|member| {
                    let at = layout
                        .iter()
                        .find(|&&(m, _)| m == member)
                        .map(|&(_, at)| at)
                        .ok_or(Malformed::Fire { binding })?;
                    let ty = *fields
                        .get(member as usize)
                        .ok_or(Malformed::Fire { binding })?;
                    let len = size_of(ty, &graph, 0).ok_or(Malformed::Fire { binding })?;
                    Ok((at, len))
                })
                .collect::<Result<Vec<_>, Malformed>>()?;
            Some(Fire {
                binding,
                bytes,
                words,
            })
        }
        (Some((_, binding, _)), None) => return Err(Malformed::Fire { binding }),
        (None, _) => None,
    };

    Ok(Declared {
        local: local.ok_or(Malformed::NoLocalSize)?,
        bindings,
        used,
        writable,
        push_bytes,
        fire,
    })
}

fn push_block(
    words: &[u32],
    pointers: &HashMap<u32, u32>,
    offsets: &HashMap<u32, Vec<(u32, u32)>>,
) -> Option<u32> {
    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let op = words[i] & 0xffff;
        if count == 0 {
            return None;
        }
        let end = i.checked_add(count)?;
        if end > words.len() {
            return None;
        }
        if op == OP_TYPE_POINTER && count >= 4 && words[i + 2] == STORAGE_PUSH_CONSTANT {
            let block = pointers.get(&words[i + 1]).copied()?;
            if offsets.contains_key(&block) {
                return Some(block);
            }
        }
        i = end;
    }
    None
}

struct Graph<'a> {
    types: &'a HashMap<u32, (u32, Vec<u32>)>,
    strides: &'a HashMap<u32, (Option<u32>, Option<u32>)>,
    constants: &'a HashMap<u32, u32>,
    offsets: &'a HashMap<u32, Vec<(u32, u32)>>,
}

const MAX_TYPE_DEPTH: u32 = 32;

fn size_of(ty: u32, graph: &Graph<'_>, depth: u32) -> Option<u32> {
    if depth > MAX_TYPE_DEPTH {
        return None;
    }
    let (op, operands) = graph.types.get(&ty)?;
    match *op {
        OP_TYPE_BOOL => Some(4),
        OP_TYPE_INT | OP_TYPE_FLOAT => Some(operands.first()?.div_ceil(8)),
        OP_TYPE_VECTOR => {
            let element = size_of(*operands.first()?, graph, depth + 1)?;
            element.checked_mul(*operands.get(1)?)
        }
        OP_TYPE_MATRIX => {
            let columns = *operands.get(1)?;
            let stride = graph
                .strides
                .get(&ty)
                .and_then(|s| s.1)
                .or_else(|| size_of(*operands.first()?, graph, depth + 1))?;
            stride.checked_mul(columns)
        }
        OP_TYPE_ARRAY => {
            let length = *graph.constants.get(operands.get(1)?)?;
            let stride = graph
                .strides
                .get(&ty)
                .and_then(|s| s.0)
                .or_else(|| size_of(*operands.first()?, graph, depth + 1))?;
            stride.checked_mul(length)
        }
        OP_TYPE_RUNTIME_ARRAY => None,
        OP_TYPE_STRUCT => {
            let members = graph.offsets.get(&ty)?;
            let mut total = 0u32;
            for &(member, at) in members {
                let member_ty = *operands.get(member as usize)?;
                let size = size_of(member_ty, graph, depth + 1)?;
                total = total.max(at.checked_add(size)?);
            }
            Some(total)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_shipped_module_binds_storage_buffers_only() {
        assert!(!kernels_slang::SPIRV.is_empty());
        for (key, _, code) in kernels_slang::SPIRV {
            let words = words(code).unwrap_or_else(|why| panic!("{key}: {why}"));
            declared(&words).unwrap_or_else(|why| panic!("{key}: {why}"));
        }
    }
}
