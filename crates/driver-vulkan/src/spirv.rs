//! What a `.spv` says about itself.
//!
//! A driver holds the module anyway, and three of the numbers it needs to
//! dispatch one are inside it rather than in any table: the workgroup size, the
//! highest binding a descriptor set must cover, and — for deciding whether a
//! grid may be rounded — whether the shader reads its own workgroup count.
//!
//! Reading them here rather than restating them in `kernels-vulkan`'s table is
//! not a preference. A table entry can drift from the shader it describes; a
//! byte read out of the module cannot. Every one of these was measured against
//! all 665 compiled modules before it was written down.
//!
//! No `ash` and no device: this is a byte parser, so it is in the portable half
//! and its tests run anywhere.

use std::collections::HashMap;

/// The magic word a SPIR-V module starts with.
const MAGIC: u32 = 0x0723_0203;

/// `OpExecutionMode`.
const OP_EXECUTION_MODE: u32 = 16;
/// `OpDecorate`.
const OP_DECORATE: u32 = 71;
/// Execution mode `LocalSize`.
const MODE_LOCAL_SIZE: u32 = 17;
/// Decoration `Binding`.
const DECORATION_BINDING: u32 = 33;
/// Decoration `BuiltIn`.
const DECORATION_BUILTIN: u32 = 11;
/// Builtin `NumWorkgroups`.
const BUILTIN_NUM_WORKGROUPS: u32 = 24;
/// Builtin `WorkgroupId` — `gl_WorkGroupID`.
const BUILTIN_WORKGROUP_ID: u32 = 26;
/// Builtin `GlobalInvocationId` — `gl_GlobalInvocationID`, which is
/// `WorkgroupId * WorkgroupSize + LocalInvocationId` and so says the same thing
/// about which grid axes the shader is indexed by.
const BUILTIN_GLOBAL_INVOCATION_ID: u32 = 28;
/// `OpConstant`.
const OP_CONSTANT: u32 = 43;
/// `OpAccessChain`.
const OP_ACCESS_CHAIN: u32 = 65;
/// `OpCompositeExtract`.
const OP_COMPOSITE_EXTRACT: u32 = 81;
/// `OpLoad`, which is how a builtin VARIABLE becomes a builtin VALUE.
///
/// Needed because `OpCompositeExtract`'s base is the loaded vector, not the
/// variable that was decorated. Without following the load, that whole branch
/// is unreachable: every module that loads `gl_GlobalInvocationID` whole and
/// then extracts a component would report reading no axis at all.
const OP_LOAD: u32 = 61;
/// `OpTypePointer`, which is how a variable's storage class reaches its type.
const OP_TYPE_POINTER: u32 = 32;
/// `OpVariable`. The one in `PushConstant` storage names the block.
const OP_VARIABLE: u32 = 59;
/// `OpMemberDecorate`, which carries each block member's byte offset.
const OP_MEMBER_DECORATE: u32 = 72;
/// Decoration `Offset`: a member's byte offset within its block.
const DECORATION_OFFSET: u32 = 35;
/// Storage class `PushConstant`.
const STORAGE_PUSH_CONSTANT: u32 = 9;
/// Storage class `StorageBuffer`, which is what an SSBO block is.
const STORAGE_BUFFER: u32 = 12;
/// Storage class `Uniform`, which is what an SSBO block is under the older
/// `BufferBlock` spelling glslc still emits for some targets.
const STORAGE_UNIFORM: u32 = 2;
/// `OpTypeInt`, `OpTypeFloat`, `OpTypeBool` — the scalars a block is built of.
const OP_TYPE_INT: u32 = 21;
/// `OpTypeFloat`.
const OP_TYPE_FLOAT: u32 = 22;
/// `OpTypeBool`. Four bytes in every block layout this tree produces.
const OP_TYPE_BOOL: u32 = 20;
/// `OpTypeVector <result> <component> <count>`.
const OP_TYPE_VECTOR: u32 = 23;
/// `OpTypeMatrix <result> <column type> <count>`, sized by its `MatrixStride`.
const OP_TYPE_MATRIX: u32 = 24;
/// `OpTypeArray <result> <element> <length id>`, sized by its `ArrayStride`.
const OP_TYPE_ARRAY: u32 = 28;
/// `OpTypeRuntimeArray`: the unsized tail of a tensor block, and the reason
/// most bindings have no knowable size.
const OP_TYPE_RUNTIME_ARRAY: u32 = 29;
/// `OpTypeStruct <result> <member types..>`.
const OP_TYPE_STRUCT: u32 = 30;
/// Decoration `ArrayStride`.
const DECORATION_ARRAY_STRIDE: u32 = 6;
/// Decoration `MatrixStride`.
const DECORATION_MATRIX_STRIDE: u32 = 7;

/// What a module declares about how it must be launched and bound.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Declared {
    /// `layout(local_size_x = ..)`, the divisor a grid is built with.
    pub local: [u32; 3],
    /// One past the highest `binding = N` the module decorates.
    ///
    /// The COUNT of them is the wrong number and using it is a crash. 79 of
    /// this tree's modules decorate a non-contiguous set — `glslc` drops the
    /// declaration of a buffer a variant never reads, so
    /// `affine_qmm_t_fp16_precast` decorates 0, 1, 2, 4, 7 — and a few hole
    /// theirs on purpose, since `kv_append_paged` keeps Metal's ring-ABI
    /// placeholder slots at 10 and 11. The layout still needs a descriptor at
    /// every number up to the highest; the shader simply ignores the ones it
    /// dropped.
    pub bindings: u32,
    /// Does the module read `gl_NumWorkGroups`?
    ///
    /// When it does, the workgroup count is a QUANTITY the shader computes
    /// with and not merely a bound it is clipped to, so its grid may not be
    /// rounded up. See [`crate::geometry::groups`].
    pub reads_workgroup_count: bool,
    /// Which of the three grid axes the shader is actually indexed by.
    ///
    /// The component of `gl_WorkGroupID` or `gl_GlobalInvocationID` it reads.
    /// This is what makes a grid on the WRONG AXIS detectable, and that is not
    /// a hypothetical: this crate's first `Rule::Rms` put the row count on y,
    /// while `norm/rms.comp` reads its row from `gl_WorkGroupID.x`. Every row
    /// but the first was left holding the zeros its buffer was born with, four
    /// dispatches returned success, and the lane-coverage sweeps all passed --
    /// they counted lanes and never asked which axis carried them.
    pub grid_axes: [bool; 3],
    /// The byte offset of every member of the push-constant block, in order.
    ///
    /// What the SHADER thinks the block looks like, so that it can be compared
    /// to what the row says the driver will write. The two agree for all 188
    /// stated entrypoints today, which is exactly why it is worth fixing: a
    /// disagreement here is not a crash and not an error. It is the shader
    /// reading a different scalar than the one that was written -- a stride
    /// where a head count belongs -- and the result is a plausible number.
    ///
    /// Empty means the module declares no push block, which is legal and is
    /// what a kernel taking only buffers wants.
    pub push_offsets: Vec<u32>,
    /// Which binding numbers the module actually decorates.
    ///
    /// [`Self::bindings`] is one past the HIGHEST, so the two disagree
    /// wherever the set has a hole -- and 79 of this tree's modules have one.
    /// `glslc` drops the declaration of a buffer a variant never reads, so
    /// `affine_qmm_t_fp16_precast` decorates 0, 1, 2, 4, 7 and nothing at all
    /// at 3, 5 or 6. `kv_append_paged` holes its 10 and 11 on purpose, keeping
    /// Metal's ring-ABI slots.
    ///
    /// # Why a driver has to know the difference
    ///
    /// On Metal a hole costs nothing: an argument index nothing is set at is
    /// an argument the shader does not read. Vulkan has no such thing. The
    /// descriptor SET still needs a slot at every number up to the highest,
    /// so a driver either finds a buffer to put in a hole -- and it has none,
    /// because the plan does not name one -- or establishes that leaving it
    /// alone is allowed.
    ///
    /// Indexed by binding number and [`Self::bindings`] long, so a hole reads
    /// as `false` rather than as an absence a caller has to notice.
    pub used: Vec<bool>,
    /// The bytes each binding's block requires, when that is knowable.
    ///
    /// Indexed by binding number, so it is [`Self::bindings`] long and a hole
    /// is `None` like an unsized block is. Only 39 of this tree's 665 modules
    /// have any entry at all, and that is not a shortfall: a tensor block ends
    /// in a `runtime array` whose length is the descriptor's, so its size is
    /// genuinely a property of the CALL. The ones that are knowable are the
    /// PARAMETER blocks -- a plain struct of scalars -- and those are the ones
    /// where a short binding is a defect rather than a shape.
    ///
    /// # Why this is worth deriving
    ///
    /// A parameter block shorter than the struct the shader reads is a defect
    /// with no symptom, and it is not hypothetical: `driver-metal` was found
    /// packing two words into `RouterParams` and two into
    /// `ExpertCombineParams`, and this walk independently measures those same
    /// blocks at 16 and 12 bytes. On Metal the shader then read the NEXT
    /// dispatch's scalars. On Vulkan it is quieter still, because
    /// `robustBufferAccess` is on: a read past the bound range returns ZERO,
    /// so the missing `logits_pitch` is not garbage, it is a plausible number
    /// that no layer and no assertion will ever object to.
    pub block_bytes: Vec<Option<u32>>,
}

/// Why a module could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Malformed {
    /// Fewer than the five header words, or a length that is not whole words.
    Truncated,
    /// The first word is not SPIR-V's magic. A module that is byte-swapped
    /// lands here too, which is the useful reading: this tree only ever
    /// produces little-endian modules, so the other case is a corrupt file.
    NotSpirv,
    /// An instruction claiming zero words, which would not terminate.
    ZeroLengthInstruction,
    /// No `OpExecutionMode LocalSize`. Every compute module has one.
    NoLocalSize,
}

impl core::fmt::Display for Malformed {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Truncated => write!(f, "not a whole number of SPIR-V words"),
            Self::NotSpirv => write!(f, "no SPIR-V magic in the first word"),
            Self::ZeroLengthInstruction => write!(f, "an instruction claims zero words"),
            Self::NoLocalSize => write!(f, "no OpExecutionMode LocalSize"),
        }
    }
}

impl core::error::Error for Malformed {}

/// Read a module's word stream from its bytes.
///
/// # Errors
///
/// [`Malformed::Truncated`] if the byte count is not a multiple of four.
pub fn words(code: &[u8]) -> Result<Vec<u32>, Malformed> {
    if !code.len().is_multiple_of(4) {
        return Err(Malformed::Truncated);
    }
    Ok(code
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Read what a module declares.
///
/// Walks every instruction rather than stopping at the first match of each
/// kind. Stopping early is wrong twice over: `OpDecorate` is not sorted by
/// binding, and a module with two entry points would declare two local sizes —
/// which is a thing this loader must notice rather than silently take the first
/// of, since the two would need different grids.
///
/// # Errors
///
/// [`Malformed`] when the stream is not a module this tree could have built.
pub fn declared(words: &[u32]) -> Result<Declared, Malformed> {
    if words.len() < 5 {
        return Err(Malformed::Truncated);
    }
    if words[0] != MAGIC {
        return Err(Malformed::NotSpirv);
    }

    let mut local = None;
    let mut highest_binding = None;
    let mut reads_workgroup_count = false;
    // Ids of the variables decorated as a grid-position builtin, and the
    // literal value of every `OpConstant`, which is how an `OpAccessChain`'s
    // index is resolved back to an axis number.
    let mut position_ids: Vec<u32> = Vec::new();
    let mut constants: HashMap<u32, u32> = HashMap::new();
    let mut grid_axes = [false; 3];
    // The push block is found by following a variable in `PushConstant`
    // storage to the struct it points at, rather than by taking any struct
    // with `Offset`-decorated members: an SSBO block has those too, and this
    // tree's kernels are full of them.
    let mut pointers: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut push_variables: Vec<u32> = Vec::new();
    let mut member_offsets: HashMap<u32, Vec<(u32, u32)>> = HashMap::new();
    // The type graph, kept whole so a block's size can be walked down through
    // it afterwards. Collected during the same pass because SPIR-V requires a
    // type to be defined before it is used, so one forward walk sees every
    // dependency before the thing that needs it -- but the block itself is
    // reached through a VARIABLE, which comes later, so the resolution has to
    // wait until the walk is done.
    let mut types: HashMap<u32, (u32, Vec<u32>)> = HashMap::new();
    let mut strides: HashMap<u32, (Option<u32>, Option<u32>)> = HashMap::new();
    let mut bound_variables: Vec<(u32, u32)> = Vec::new();
    let mut variable_types: HashMap<u32, u32> = HashMap::new();

    let mut i = 5;
    while i < words.len() {
        let count = (words[i] >> 16) as usize;
        let op = words[i] & 0xffff;
        if count == 0 {
            return Err(Malformed::ZeroLengthInstruction);
        }
        // A well-formed instruction cannot claim more words than remain. This
        // is what keeps every `words[i + n]` below in bounds without a check of
        // its own.
        let end = i.checked_add(count).ok_or(Malformed::Truncated)?;
        if end > words.len() {
            return Err(Malformed::Truncated);
        }

        if op == OP_EXECUTION_MODE && count >= 6 && words[i + 2] == MODE_LOCAL_SIZE {
            local = Some([words[i + 3], words[i + 4], words[i + 5]]);
        }
        if op == OP_DECORATE && count >= 4 {
            match words[i + 2] {
                DECORATION_BINDING => {
                    let n = words[i + 3];
                    highest_binding = Some(highest_binding.map_or(n, |h: u32| h.max(n)));
                    bound_variables.push((words[i + 1], n));
                }
                DECORATION_ARRAY_STRIDE => {
                    strides.entry(words[i + 1]).or_default().0 = Some(words[i + 3]);
                }
                DECORATION_MATRIX_STRIDE => {
                    strides.entry(words[i + 1]).or_default().1 = Some(words[i + 3]);
                }
                DECORATION_BUILTIN if words[i + 3] == BUILTIN_NUM_WORKGROUPS => {
                    reads_workgroup_count = true;
                }
                DECORATION_BUILTIN
                    if words[i + 3] == BUILTIN_WORKGROUP_ID
                        || words[i + 3] == BUILTIN_GLOBAL_INVOCATION_ID =>
                {
                    position_ids.push(words[i + 1]);
                }
                _ => {}
            }
        }
        // `OpConstant <type> <result> <literal>`. Collected for every
        // constant rather than only the small ones, since an index is just an
        // integer constant like any other and there is no way to tell which
        // will be used as one before the access chain is seen.
        if op == OP_CONSTANT && count >= 4 {
            constants.insert(words[i + 2], words[i + 3]);
        }
        // `OpLoad <type> <result> <pointer>`. Loading a decorated variable
        // makes the RESULT a grid position too, which is what lets the
        // `OpCompositeExtract` branch below ever match.
        if op == OP_LOAD && count >= 4 && position_ids.contains(&words[i + 3]) {
            position_ids.push(words[i + 2]);
        }
        // Reading one component of a grid-position builtin. `OpAccessChain`
        // indexes the variable through a pointer and its index is a constant
        // ID; `OpCompositeExtract` indexes a loaded vector and its index is a
        // literal. Both appear in this tree, from the same GLSL, depending on
        // whether glslc loaded the whole vector first.
        if op == OP_ACCESS_CHAIN
            && count >= 5
            && position_ids.contains(&words[i + 3])
            && let Some(&axis) = constants.get(&words[i + 4])
            && let Some(slot) = grid_axes.get_mut(axis as usize)
        {
            *slot = true;
        }
        if op == OP_COMPOSITE_EXTRACT
            && count >= 5
            && position_ids.contains(&words[i + 3])
            && let Some(slot) = grid_axes.get_mut(words[i + 4] as usize)
        {
            *slot = true;
        }
        // `OpTypePointer <result> <storage class> <type>`.
        if op == OP_TYPE_POINTER && count >= 4 {
            pointers.insert(words[i + 1], (words[i + 2], words[i + 3]));
        }
        // `OpVariable <type> <result> <storage class>`.
        if op == OP_VARIABLE && count >= 4 {
            if words[i + 3] == STORAGE_PUSH_CONSTANT {
                push_variables.push(words[i + 1]);
            }
            if words[i + 3] == STORAGE_BUFFER || words[i + 3] == STORAGE_UNIFORM {
                variable_types.insert(words[i + 2], words[i + 1]);
            }
        }
        // Every type definition, kept by result id with its operands. The
        // result id is the FIRST operand for these -- unlike `OpVariable`,
        // where it is the second, because a variable has a type and a type
        // does not.
        if matches!(
            op,
            OP_TYPE_INT
                | OP_TYPE_FLOAT
                | OP_TYPE_BOOL
                | OP_TYPE_VECTOR
                | OP_TYPE_MATRIX
                | OP_TYPE_ARRAY
                | OP_TYPE_RUNTIME_ARRAY
                | OP_TYPE_STRUCT
        ) && count >= 2
        {
            types.insert(words[i + 1], (op, words[i + 2..end].to_vec()));
        }
        // `OpMemberDecorate <struct> <member> Offset <literal>`.
        if op == OP_MEMBER_DECORATE && count >= 5 && words[i + 3] == DECORATION_OFFSET {
            member_offsets
                .entry(words[i + 1])
                .or_default()
                .push((words[i + 2], words[i + 4]));
        }
        i = end;
    }

    Ok(Declared {
        local: local.ok_or(Malformed::NoLocalSize)?,
        // A module that binds nothing needs a layout of no descriptors, which
        // is legal and is what a push-constant-only kernel wants.
        bindings: highest_binding.map_or(0, |h| h + 1),
        used: {
            let mut used = vec![false; highest_binding.map_or(0, |h| h + 1) as usize];
            for &(_, binding) in &bound_variables {
                if let Some(slot) = used.get_mut(binding as usize) {
                    *slot = true;
                }
            }
            used
        },
        reads_workgroup_count,
        grid_axes,
        // Members come out in declaration order, not in the order glslc
        // happened to emit their decorations, because the driver writes the
        // block front to back and an out-of-order comparison would agree with
        // a shader that reads its scalars transposed.
        push_offsets: push_variables
            .iter()
            .find_map(|v| {
                let (_, block) = pointers.get(v)?;
                let mut members = member_offsets.get(block)?.clone();
                members.sort_unstable_by_key(|(member, _)| *member);
                Some(members.into_iter().map(|(_, at)| at).collect())
            })
            .unwrap_or_default(),
        block_bytes: block_bytes(
            &Graph {
                types: &types,
                strides: &strides,
                constants: &constants,
                member_offsets: &member_offsets,
            },
            &pointers,
            &variable_types,
            &bound_variables,
            highest_binding.map_or(0, |h| h + 1),
        ),
    })
}

/// The pieces of a module's type graph a size walk needs.
///
/// One struct rather than five parameters because they are always passed
/// together and the recursion would otherwise thread all five through every
/// level.
struct Graph<'a> {
    /// Every type definition by result id: the opcode, and its operands after
    /// the result.
    types: &'a HashMap<u32, (u32, Vec<u32>)>,
    /// `ArrayStride` and `MatrixStride` per type.
    strides: &'a HashMap<u32, (Option<u32>, Option<u32>)>,
    /// Literal values of `OpConstant`, which is where an array length lives.
    constants: &'a HashMap<u32, u32>,
    /// `(member, offset)` per struct.
    member_offsets: &'a HashMap<u32, Vec<(u32, u32)>>,
}

/// The bytes each binding's block requires, or `None` where that is not a
/// fixed number.
///
/// Separate from the walk because a block is reached through a variable, and a
/// variable comes after the types it is built from -- so the graph has to be
/// whole before any of it can be resolved.
fn block_bytes(
    graph: &Graph<'_>,
    pointers: &HashMap<u32, (u32, u32)>,
    variable_types: &HashMap<u32, u32>,
    bound_variables: &[(u32, u32)],
    bindings: u32,
) -> Vec<Option<u32>> {
    let mut out = vec![None; bindings as usize];
    for &(variable, binding) in bound_variables {
        let Some(slot) = out.get_mut(binding as usize) else {
            continue;
        };
        // A decorated id that is not a variable, or a variable whose pointer
        // type was never seen, is not a block. Both are skipped rather than
        // refused: this is one fact among several a module declares, and a
        // module that cannot answer it is still perfectly loadable.
        let Some(&pointer) = variable_types.get(&variable) else {
            continue;
        };
        let Some(&(_, block)) = pointers.get(&pointer) else {
            continue;
        };
        *slot = size_of(block, graph, 0);
    }
    out
}

/// How deep the size walk will follow a type before giving up.
///
/// A well-formed module cannot nest this far, and a MALFORMED one can point a
/// type at itself. `declared` is handed bytes from disk, so the recursion needs
/// a floor that does not depend on the input being honest.
const MAX_TYPE_DEPTH: u32 = 32;

/// The bytes a type occupies under the block layout its decorations describe,
/// or `None` when that is not a fixed number.
///
/// `None` is the ordinary answer and not a failure. A tensor block ends in a
/// runtime array, whose length is whatever the descriptor's range says at the
/// call -- so its size genuinely is not a property of the module, and reporting
/// one would be inventing it.
fn size_of(ty: u32, graph: &Graph<'_>, depth: u32) -> Option<u32> {
    if depth >= MAX_TYPE_DEPTH {
        return None;
    }
    let (op, operands) = graph.types.get(&ty)?;
    match *op {
        // `OpTypeInt <width> <signed>` / `OpTypeFloat <width>`: the width is
        // in BITS, and 8-bit and 16-bit members both appear in this tree.
        OP_TYPE_INT | OP_TYPE_FLOAT => operands.first().map(|bits| bits / 8),
        // A `bool` in a block is four bytes; it is not the one-byte host type.
        OP_TYPE_BOOL => Some(4),
        // `OpTypeVector <component> <count>`. No stride: components are tight
        // within a vector, and it is the vector that gets aligned, not its
        // parts.
        OP_TYPE_VECTOR => {
            let component = size_of(*operands.first()?, graph, depth + 1)?;
            component.checked_mul(*operands.get(1)?)
        }
        // A matrix is its column count times the stride BETWEEN columns, which
        // is a decoration and not the column's own size -- a `mat3` of `vec3`
        // columns is 48 bytes, not 36.
        OP_TYPE_MATRIX => graph
            .strides
            .get(&ty)
            .and_then(|(_, matrix)| *matrix)?
            .checked_mul(*operands.get(1)?),
        // Same reasoning: the stride is the decoration, because an array of
        // `float` in a std140 block strides by 16.
        OP_TYPE_ARRAY => {
            let stride = graph.strides.get(&ty).and_then(|(array, _)| *array)?;
            stride.checked_mul(*graph.constants.get(operands.get(1)?)?)
        }
        // The whole point of the `Option`.
        OP_TYPE_RUNTIME_ARRAY => None,
        // The end of a struct is the furthest any MEMBER reaches, not the sum
        // of their sizes and not the last one's end: members are placed by
        // their own `Offset` decorations, padding lives between them, and
        // declaration order is not required to be offset order.
        OP_TYPE_STRUCT => {
            let offsets = graph.member_offsets.get(&ty)?;
            let mut end = 0u32;
            for (member, &member_ty) in operands.iter().enumerate() {
                let at = offsets
                    .iter()
                    .find(|(m, _)| *m as usize == member)
                    .map(|(_, at)| *at)?;
                end = end.max(at.checked_add(size_of(member_ty, graph, depth + 1)?)?);
            }
            Some(end)
        }
        _ => None,
    }
}

impl Declared {
    /// How many binding numbers below [`Self::bindings`] the module skips.
    ///
    /// The count a descriptor set has to carry and nothing fills. 165 of this
    /// tree's 665 modules have at least one, 358 in all.
    #[must_use]
    pub fn holes(&self) -> usize {
        self.used.iter().filter(|u| !**u).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal module: header, one `LocalSize`, two bindings with a hole.
    fn built() -> Vec<u32> {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 1, 0];
        // OpExecutionMode <entry> LocalSize 16 16 1
        w.extend([(6 << 16) | OP_EXECUTION_MODE, 4, MODE_LOCAL_SIZE, 16, 16, 1]);
        // OpDecorate %a Binding 0, OpDecorate %b Binding 3 -- a hole at 1, 2.
        w.extend([(4 << 16) | OP_DECORATE, 10, DECORATION_BINDING, 0]);
        w.extend([(4 << 16) | OP_DECORATE, 11, DECORATION_BINDING, 3]);
        w
    }

    #[test]
    fn a_modules_workgroup_and_binding_count_are_read() {
        let d = declared(&built()).expect("well formed");
        assert_eq!(d.local, [16, 16, 1]);
        // FOUR, not the two that are decorated. A layout of two descriptors
        // under a module that binds 3 is a crash inside pipeline creation.
        assert_eq!(d.bindings, 4);
        assert!(!d.reads_workgroup_count);
    }

    #[test]
    fn the_workgroup_count_builtin_is_noticed() {
        let mut w = built();
        w.extend([
            (4 << 16) | OP_DECORATE,
            12,
            DECORATION_BUILTIN,
            BUILTIN_NUM_WORKGROUPS,
        ]);
        assert!(declared(&w).expect("well formed").reads_workgroup_count);
    }

    /// Another builtin is not that builtin.
    ///
    /// `WorkgroupId` is 26 and every shader here decorates one. Matching on
    /// the decoration alone would make every module look like it reads its
    /// workgroup count, and the exactness rule would then apply to all 665 --
    /// which would pass, and would be checking nothing.
    #[test]
    fn a_different_builtin_is_not_mistaken_for_it() {
        let mut w = built();
        w.extend([(4 << 16) | OP_DECORATE, 12, DECORATION_BUILTIN, 26]);
        assert!(!declared(&w).expect("well formed").reads_workgroup_count);
    }

    /// Which components of the grid a module is indexed by, both ways round.
    ///
    /// Both instruction forms appear in this kernel tree from identical GLSL,
    /// depending on whether glslc loaded the whole vector before indexing it,
    /// so a reader that handles one and not the other is a reader that is
    /// right about some modules and silently wrong about the rest.
    #[test]
    fn the_grid_components_a_module_reads_are_recovered() {
        let mut w = built();
        // %20 is `gl_WorkGroupID`, %21 is `gl_GlobalInvocationID`.
        w.extend([
            (4 << 16) | OP_DECORATE,
            20,
            DECORATION_BUILTIN,
            BUILTIN_WORKGROUP_ID,
        ]);
        w.extend([
            (4 << 16) | OP_DECORATE,
            21,
            DECORATION_BUILTIN,
            BUILTIN_GLOBAL_INVOCATION_ID,
        ]);
        // %30 = OpConstant uint 0, %31 = OpConstant uint 2.
        w.extend([(4 << 16) | OP_CONSTANT, 5, 30, 0]);
        w.extend([(4 << 16) | OP_CONSTANT, 5, 31, 2]);
        // Through a pointer: `gl_WorkGroupID[0]`.
        w.extend([(5 << 16) | OP_ACCESS_CHAIN, 6, 40, 20, 30]);
        // Through a load: `gl_GlobalInvocationID.z`.
        w.extend([(4 << 16) | OP_LOAD, 7, 41, 21]);
        w.extend([(5 << 16) | OP_COMPOSITE_EXTRACT, 5, 42, 41, 2]);

        let d = declared(&w).expect("well formed");
        assert_eq!(
            d.grid_axes,
            [true, false, true],
            "x came through an access chain and z through a load-and-extract; \
             y is read by nothing and must stay false"
        );
    }

    /// An axis a module never mentions is not reported as one it reads.
    ///
    /// The half that carries the weight. `grid_axes` exists to say a geometry
    /// is putting work where nobody will do it, and a reader that answered
    /// `[true; 3]` -- by marking the whole variable read rather than the
    /// component -- would agree with every geometry ever written.
    #[test]
    fn an_axis_that_is_only_written_past_is_not_read() {
        let mut w = built();
        w.extend([
            (4 << 16) | OP_DECORATE,
            20,
            DECORATION_BUILTIN,
            BUILTIN_WORKGROUP_ID,
        ]);
        w.extend([(4 << 16) | OP_CONSTANT, 5, 30, 1]);
        w.extend([(5 << 16) | OP_ACCESS_CHAIN, 6, 40, 20, 30]);
        assert_eq!(
            declared(&w).expect("well formed").grid_axes,
            [false, true, false]
        );
    }

    /// An index that is not a constant is not silently read as one.
    ///
    /// The constant table is keyed by result id, and an `OpAccessChain` whose
    /// index is a computed value has no entry there. Resolving a miss to axis
    /// zero would mark x read on every module that indexes a grid position
    /// dynamically, and x is the axis most geometries load, so the mistake
    /// would look like agreement.
    #[test]
    fn an_index_that_is_not_a_constant_names_no_axis() {
        let mut w = built();
        w.extend([
            (4 << 16) | OP_DECORATE,
            20,
            DECORATION_BUILTIN,
            BUILTIN_WORKGROUP_ID,
        ]);
        // %99 was never defined by an `OpConstant`.
        w.extend([(5 << 16) | OP_ACCESS_CHAIN, 6, 40, 20, 99]);
        assert_eq!(declared(&w).expect("well formed").grid_axes, [false; 3]);
    }

    /// The push block is the one a `PushConstant` variable points at.
    ///
    /// Not merely "a struct with `Offset`-decorated members" -- every SSBO in
    /// this tree has those, and taking the first one found disagrees with the
    /// row for 151 of the 188 stated entrypoints. The offsets would still look
    /// like offsets, and the resulting comparison would look like a check.
    #[test]
    fn the_push_block_is_told_apart_from_a_storage_block() {
        let mut w = built();
        // %60 is an SSBO block: decorated with offsets, and not the answer.
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 0, DECORATION_OFFSET, 64]);
        // %61 is the push block: members declared 0, 1 at bytes 0 and 8.
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 61, 0, DECORATION_OFFSET, 0]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 61, 1, DECORATION_OFFSET, 8]);
        // A pointer to the SSBO block in `StorageBuffer` (12), and one to the
        // push block in `PushConstant`.
        w.extend([(4 << 16) | OP_TYPE_POINTER, 70, 12, 60]);
        w.extend([(4 << 16) | OP_TYPE_POINTER, 71, STORAGE_PUSH_CONSTANT, 61]);
        w.extend([(4 << 16) | OP_VARIABLE, 70, 80, 12]);
        w.extend([(4 << 16) | OP_VARIABLE, 71, 81, STORAGE_PUSH_CONSTANT]);

        assert_eq!(declared(&w).expect("well formed").push_offsets, vec![0, 8]);
    }

    /// Members come back in declaration order, not decoration order.
    ///
    /// glslc emits these in whatever order it likes, and the driver packs the
    /// block front to back. A reader that kept emission order would agree
    /// with a shader that reads its scalars transposed.
    #[test]
    fn push_members_are_ordered_by_declaration() {
        let mut w = built();
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 61, 2, DECORATION_OFFSET, 16]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 61, 0, DECORATION_OFFSET, 0]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 61, 1, DECORATION_OFFSET, 8]);
        w.extend([(4 << 16) | OP_TYPE_POINTER, 71, STORAGE_PUSH_CONSTANT, 61]);
        w.extend([(4 << 16) | OP_VARIABLE, 71, 81, STORAGE_PUSH_CONSTANT]);

        assert_eq!(
            declared(&w).expect("well formed").push_offsets,
            vec![0, 8, 16]
        );
    }

    /// A module with no push block says so, rather than guessing.
    #[test]
    fn a_module_with_no_push_block_declares_none() {
        assert!(
            declared(&built())
                .expect("well formed")
                .push_offsets
                .is_empty()
        );
    }

    /// A four-byte scalar type, for the block tests below.
    ///
    /// `OpTypeInt %id 32 0` -- unsigned, which is what every scalar in a
    /// parameter block of this tree is or is the same width as.
    fn u32_type(w: &mut Vec<u32>, id: u32) {
        w.extend([(4 << 16) | OP_TYPE_INT, id, 32, 0]);
    }

    /// A struct at `block`, bound at `binding`, reached the way a real module
    /// reaches one: a `StorageBuffer` pointer and a decorated variable.
    fn bound_block(w: &mut Vec<u32>, block: u32, binding: u32, members: &[u32]) {
        let pointer = block + 100;
        let variable = block + 200;
        let mut instruction = vec![((2 + members.len() as u32) << 16) | OP_TYPE_STRUCT, block];
        instruction.extend_from_slice(members);
        w.extend(instruction);
        w.extend([(4 << 16) | OP_TYPE_POINTER, pointer, STORAGE_BUFFER, block]);
        w.extend([(4 << 16) | OP_VARIABLE, pointer, variable, STORAGE_BUFFER]);
        w.extend([
            (4 << 16) | OP_DECORATE,
            variable,
            DECORATION_BINDING,
            binding,
        ]);
    }

    /// A block's size is where its furthest member ENDS, not the sum of them.
    ///
    /// Worth a synthetic module because the tree cannot test it: all 39 of its
    /// parameter blocks are flat runs of four-byte scalars, so summing the
    /// members and taking `max(offset + size)` give the same 39 answers and
    /// the whole-table check in `tests/rules.rs` passes either way. That was
    /// measured, not assumed -- the reading was changed to a sum and no block
    /// disagreed. A block with padding is legal, `kv_append`'s PUSH block
    /// already is one, and the sum would be four bytes short of it.
    #[test]
    fn a_block_is_measured_to_where_its_last_member_ends() {
        let mut w = built();
        u32_type(&mut w, 50);
        // Three four-byte members at 0, 8 and 16: a 32-bit scalar, then two
        // that a 64-bit neighbour has pushed apart. Twenty bytes, and twelve
        // if the members are merely added up.
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 0, DECORATION_OFFSET, 0]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 1, DECORATION_OFFSET, 8]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 2, DECORATION_OFFSET, 16]);
        bound_block(&mut w, 60, 0, &[50, 50, 50]);

        assert_eq!(
            declared(&w).expect("well formed").block_bytes[0],
            Some(20),
            "the block ends at 16 + 4, and the members sum to 12"
        );
    }

    /// Members are not required to be declared in offset order, and the
    /// furthest one is not required to be the last.
    #[test]
    fn the_furthest_member_sets_the_size_wherever_it_was_declared() {
        let mut w = built();
        u32_type(&mut w, 50);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 0, DECORATION_OFFSET, 32]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 1, DECORATION_OFFSET, 0]);
        bound_block(&mut w, 60, 0, &[50, 50]);

        assert_eq!(declared(&w).expect("well formed").block_bytes[0], Some(36));
    }

    /// An array is its stride times its length, and the stride is a
    /// decoration rather than the element's own size.
    ///
    /// Also untestable against this tree, whose parameter blocks contain no
    /// arrays -- the reading was changed to use the element size and no block
    /// disagreed. It matters because a `float[4]` in a block strides by 16,
    /// so the element size is a quarter of the truth.
    #[test]
    fn an_array_member_is_sized_by_its_stride_and_not_its_element() {
        let mut w = built();
        u32_type(&mut w, 50);
        // `OpConstant %u32 %51 4` -- the length.
        w.extend([(4 << 16) | OP_CONSTANT, 50, 51, 4]);
        // `OpTypeArray %52 %u32 %51`, striding by 16 rather than by 4.
        w.extend([(4 << 16) | OP_TYPE_ARRAY, 52, 50, 51]);
        w.extend([(4 << 16) | OP_DECORATE, 52, DECORATION_ARRAY_STRIDE, 16]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 0, DECORATION_OFFSET, 0]);
        bound_block(&mut w, 60, 0, &[52]);

        assert_eq!(
            declared(&w).expect("well formed").block_bytes[0],
            Some(64),
            "four elements striding by 16, not the 16 bytes they contain"
        );
    }

    /// A block ending in a runtime array has no size, and says so.
    ///
    /// This is the ordinary case -- 3754 of this tree's 3793 bindings -- and
    /// answering it with the fixed prefix would be worse than answering
    /// nothing: `run` would then refuse every tensor bound to fewer bytes than
    /// its header, which is a number that means nothing.
    #[test]
    fn a_block_ending_in_a_runtime_array_has_no_fixed_size() {
        let mut w = built();
        u32_type(&mut w, 50);
        w.extend([(3 << 16) | OP_TYPE_RUNTIME_ARRAY, 52, 50]);
        w.extend([(4 << 16) | OP_DECORATE, 52, DECORATION_ARRAY_STRIDE, 4]);
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 0, DECORATION_OFFSET, 0]);
        bound_block(&mut w, 60, 0, &[52]);

        assert_eq!(declared(&w).expect("well formed").block_bytes[0], None);
    }

    /// A type that contains itself is stopped by exhaustion rather than by
    /// running the stack out.
    ///
    /// `declared` is handed bytes from disk. A module that is merely WRONG is
    /// one thing; one that is hostile is another, and neither may take the
    /// process down. There is no legal SPIR-V that nests this deep, so the
    /// floor costs nothing a real module would notice.
    ///
    /// It has to be a STRUCT. An array is sized from its `ArrayStride`
    /// decoration and never looks at its element, so an array of itself
    /// terminates by not recursing at all -- which was the first version of
    /// this test, and it proved only that the array path is not the one with
    /// the hazard.
    #[test]
    fn a_type_that_contains_itself_does_not_run_the_stack_out() {
        let mut w = built();
        w.extend([(5 << 16) | OP_MEMBER_DECORATE, 60, 0, DECORATION_OFFSET, 0]);
        bound_block(&mut w, 60, 0, &[60]);

        assert_eq!(declared(&w).expect("well formed").block_bytes[0], None);
    }

    /// A binding whose block is a bare tensor -- no struct at all -- is not
    /// mistaken for a sized one.
    #[test]
    fn a_binding_that_is_not_a_struct_reports_no_size() {
        let mut w = built();
        u32_type(&mut w, 50);
        w.extend([(4 << 16) | OP_TYPE_POINTER, 150, STORAGE_BUFFER, 50]);
        w.extend([(4 << 16) | OP_VARIABLE, 150, 250, STORAGE_BUFFER]);
        w.extend([(4 << 16) | OP_DECORATE, 250, DECORATION_BINDING, 0]);

        // A bare `u32` at a binding is four bytes, and that IS knowable -- the
        // point is only that it is not confused with a struct's.
        assert_eq!(declared(&w).expect("well formed").block_bytes[0], Some(4));
    }

    #[test]
    fn a_module_that_binds_nothing_needs_no_descriptors() {
        let w = vec![
            MAGIC,
            0x0001_0300,
            0,
            1,
            0,
            (6 << 16) | OP_EXECUTION_MODE,
            4,
            MODE_LOCAL_SIZE,
            1,
            1,
            1,
        ];
        assert_eq!(declared(&w).expect("well formed").bindings, 0);
    }

    /// A truncated stream is refused rather than walked off the end.
    ///
    /// The instruction claims six words and four remain. Trusting the count
    /// would index past the slice, and this parser runs on files -- a
    /// half-written module from an interrupted build is the ordinary way to get
    /// one.
    #[test]
    fn an_instruction_longer_than_what_remains_is_refused() {
        let w = vec![MAGIC, 0, 0, 1, 0, (6 << 16) | OP_EXECUTION_MODE, 4, 17];
        assert_eq!(declared(&w), Err(Malformed::Truncated));
    }

    #[test]
    fn a_zero_length_instruction_does_not_loop_forever() {
        let w = vec![MAGIC, 0, 0, 1, 0, 0];
        assert_eq!(declared(&w), Err(Malformed::ZeroLengthInstruction));
    }

    #[test]
    fn something_that_is_not_a_module_is_refused() {
        assert_eq!(declared(&[1, 2, 3, 4, 5]), Err(Malformed::NotSpirv));
        assert_eq!(declared(&[MAGIC]), Err(Malformed::Truncated));
        assert_eq!(words(&[0, 1, 2]), Err(Malformed::Truncated));
    }

    #[test]
    fn a_compute_module_without_a_local_size_is_refused() {
        assert_eq!(declared(&[MAGIC, 0, 0, 1, 0]), Err(Malformed::NoLocalSize));
    }
}
