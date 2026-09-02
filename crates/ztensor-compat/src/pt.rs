//! PyTorch `.pt` / `.bin` → zTensor object model projection.
//!
//! A torch checkpoint is a ZIP: `<root>/data.pkl` (a pickle stream
//! describing the state dict) plus `<root>/data/<key>` raw storages.
//!
//! The pickle stream is evaluated by a restricted VM that executes **no
//! code**: it recognizes exactly the reconstruction patterns torch uses
//! (`persistent_load` storage tuples, `torch._utils._rebuild_tensor_v2`,
//! `collections.OrderedDict`) and materializes everything else as opaque.
//! Anything that would require reinterpreting bytes to "make work" is a loud
//! refusal: an unknown storage dtype, or a non-contiguous stride.
//!
//! This module is feature-gated (`pickle`): parsing pickle at all is a
//! larger attack surface than any other format here, and enabling it is an
//! explicit choice.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;

use ztensor::provide::{Catalog, Decode};
use ztensor::provide::{Entry, Location, Payload};
use ztensor::{Error, Leaf, Result, Store, StoreId};

use crate::project::Projection;

fn bad(detail: impl Into<String>) -> Error {
    Error::InvalidInput(format!("pt: {}", detail.into()))
}

// =======================================================================
// Restricted pickle VM
// =======================================================================

const MAX_PICKLE_BYTES: usize = 256 << 20;
const MAX_STACK: usize = 10_000_000;
const MAX_MEMO: usize = 10_000_000;
const MAX_OPCODES: usize = 50_000_000;
const MAX_ITEMS: usize = 1_000_000;
const MAX_DEPTH: usize = 128;
/// Longest string kept in the memo table (longer ones become opaque).
const MAX_MEMO_STR: usize = 4096;

#[derive(Debug, Clone)]
struct TensorRef {
    storage_key: String,
    /// Byte offset within the storage.
    byte_offset: u64,
    shape: Vec<u64>,
    dtype: Leaf,
}

#[derive(Debug, Clone)]
enum Val {
    None,
    #[allow(dead_code)]
    Bool(bool),
    Int(i64),
    #[allow(dead_code)]
    Float(f64),
    #[allow(dead_code)]
    Bytes(Vec<u8>),
    Str(String),
    Tuple(Vec<Val>),
    List(Vec<Val>),
    Dict(Vec<(Val, Val)>),
    Global {
        module: String,
        name: String,
    },
    Storage {
        key: String,
        dtype: Leaf,
    },
    Tensor(Box<TensorRef>),
    Mark,
    Opaque,
}

fn storage_dtype(name: &str) -> Option<Leaf> {
    Some(match name {
        "DoubleStorage" => Leaf::F64,
        "FloatStorage" => Leaf::F32,
        "HalfStorage" => Leaf::F16,
        "BFloat16Storage" => Leaf::BF16,
        "LongStorage" => Leaf::I64,
        "IntStorage" => Leaf::I32,
        "ShortStorage" => Leaf::I16,
        "CharStorage" => Leaf::I8,
        "ByteStorage" => Leaf::U8,
        "BoolStorage" => Leaf::Bool,
        "Float8_e4m3fnStorage" => Leaf::E4M3,
        "Float8_e5m2Storage" => Leaf::E5M2,
        _ => return None,
    })
}

struct Vm<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Val>,
    /// Stack indices of live MARK sentinels.
    marks: Vec<usize>,
    memo: BTreeMap<u32, Val>,
    /// A refusal encountered mid-stream (e.g., a non-contiguous tensor):
    /// recorded here so the failure is loud, never a silently dropped
    /// tensor.
    refusal: Option<Error>,
}

impl<'a> Vm<'a> {
    fn byte(&mut self) -> Result<u8> {
        let b = *self
            .data
            .get(self.pos)
            .ok_or_else(|| bad("pickle stream truncated"))?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.data.len())
            .ok_or_else(|| bad("pickle stream truncated"))?;
        let s = &self.data[self.pos..end];
        self.pos = end;
        Ok(s)
    }

    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn sized(&mut self, n: usize) -> Result<&'a [u8]> {
        if n > MAX_PICKLE_BYTES {
            return Err(bad("pickle object exceeds size limit"));
        }
        self.take(n)
    }

    fn line(&mut self) -> Result<&'a [u8]> {
        let start = self.pos;
        while self.pos < self.data.len() && self.data[self.pos] != b'\n' {
            self.pos += 1;
        }
        if self.pos >= self.data.len() {
            return Err(bad("pickle stream truncated"));
        }
        let line = &self.data[start..self.pos];
        self.pos += 1;
        Ok(line)
    }

    fn pop(&mut self) -> Val {
        self.stack.pop().unwrap_or(Val::None)
    }

    /// Pops back to the most recent MARK. `marks` tracks mark positions so
    /// this is O(popped items). A linear rescan would make N mark-less
    /// POP_MARKs quadratic.
    fn pop_to_mark(&mut self) -> Vec<Val> {
        match self.marks.pop() {
            Some(i) if i < self.stack.len() => {
                let items = self.stack.split_off(i + 1);
                self.stack.pop();
                items
            }
            _ => Vec::new(),
        }
    }

    fn push_str(&mut self, bytes: &[u8]) {
        self.stack
            .push(Val::Str(String::from_utf8_lossy(bytes).into_owned()));
    }

    fn execute(&mut self) -> Result<()> {
        let mut ops = 0usize;
        while self.pos < self.data.len() {
            ops += 1;
            if ops > MAX_OPCODES {
                return Err(bad("pickle opcode limit exceeded"));
            }
            let op = self.byte()?;
            match op {
                0x80 => {
                    self.byte()?; // PROTO
                }
                0x95 => {
                    let n = self.u64()?; // FRAME
                    if self.pos as u64 + n > self.data.len() as u64 {
                        return Err(bad("FRAME length exceeds stream"));
                    }
                }
                0x2e => break, // STOP
                0x28 => {
                    self.marks.push(self.stack.len()); // MARK
                    self.stack.push(Val::Mark);
                }
                0x29 => self.stack.push(Val::Tuple(Vec::new())), // EMPTY_TUPLE
                0x5d => self.stack.push(Val::List(Vec::new())),  // EMPTY_LIST
                0x7d => self.stack.push(Val::Dict(Vec::new())),  // EMPTY_DICT
                0x4e => self.stack.push(Val::None),              // NONE
                0x88 => self.stack.push(Val::Bool(true)),        // NEWTRUE
                0x89 => self.stack.push(Val::Bool(false)),       // NEWFALSE
                0x4a => {
                    let v = self.u32()? as i32;
                    self.stack.push(Val::Int(v as i64)); // BININT
                }
                0x4b => {
                    let v = self.byte()?;
                    self.stack.push(Val::Int(v as i64)); // BININT1
                }
                0x4d => {
                    let v = self.u16()?;
                    self.stack.push(Val::Int(v as i64)); // BININT2
                }
                0x8a => {
                    // LONG1
                    let n = self.byte()? as usize;
                    let bytes = self.take(n)?;
                    self.stack.push(Val::Int(long_from_le(bytes)));
                }
                0x47 => {
                    // BINFLOAT (big-endian f64)
                    let v = f64::from_be_bytes(self.take(8)?.try_into().unwrap());
                    self.stack.push(Val::Float(v));
                }
                0x58 => {
                    let n = self.u32()? as usize;
                    let b = self.sized(n)?;
                    self.push_str(b); // BINUNICODE
                }
                0x8c => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?;
                    self.push_str(b); // SHORT_BINUNICODE
                }
                0x8d => {
                    let n = self.u64()? as usize;
                    let b = self.sized(n)?;
                    self.push_str(b); // BINUNICODE8
                }
                0x55 => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?;
                    self.push_str(b); // SHORT_BINSTRING
                }
                0x43 => {
                    let n = self.byte()? as usize;
                    let b = self.take(n)?.to_vec();
                    self.stack.push(Val::Bytes(b)); // SHORT_BINBYTES
                }
                0x44 => {
                    let n = self.u32()? as usize;
                    let b = self.sized(n)?.to_vec();
                    self.stack.push(Val::Bytes(b)); // BINBYTES
                }
                0x8e | 0x96 => {
                    let n = self.u64()? as usize;
                    let b = self.sized(n)?.to_vec();
                    self.stack.push(Val::Bytes(b)); // BINBYTES8 / BYTEARRAY8
                }
                0x63 => {
                    // GLOBAL
                    let module = String::from_utf8_lossy(self.line()?).into_owned();
                    let name = String::from_utf8_lossy(self.line()?).into_owned();
                    self.stack.push(Val::Global { module, name });
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.pop();
                    let module = self.pop();
                    match (module, name) {
                        (Val::Str(module), Val::Str(name)) => {
                            self.stack.push(Val::Global { module, name })
                        }
                        _ => self.stack.push(Val::Opaque),
                    }
                }
                0x85 => {
                    let a = self.pop();
                    self.stack.push(Val::Tuple(vec![a])); // TUPLE1
                }
                0x86 => {
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(Val::Tuple(vec![a, b])); // TUPLE2
                }
                0x87 => {
                    let c = self.pop();
                    let b = self.pop();
                    let a = self.pop();
                    self.stack.push(Val::Tuple(vec![a, b, c])); // TUPLE3
                }
                0x74 => {
                    let items = self.pop_to_mark();
                    self.stack.push(Val::Tuple(items)); // TUPLE
                }
                0x6c => {
                    let items = self.pop_to_mark();
                    self.stack.push(Val::List(items)); // LIST
                }
                0x64 => {
                    let items = self.pop_to_mark();
                    self.stack.push(Val::Dict(pairs(items))); // DICT
                }
                0x52 | 0x81 => {
                    // REDUCE / NEWOBJ
                    let args = self.pop();
                    let callable = self.pop();
                    let v = self.reduce(callable, args)?;
                    self.stack.push(v);
                }
                0x92 => {
                    // NEWOBJ_EX
                    let _kwargs = self.pop();
                    let args = self.pop();
                    let callable = self.pop();
                    let v = self.reduce(callable, args)?;
                    self.stack.push(v);
                }
                0x51 => {
                    // BINPERSID
                    let pid = self.pop();
                    let v = self.persistent_load(pid);
                    self.stack.push(v);
                }
                0x62 => {
                    // BUILD
                    let state = self.pop();
                    let obj = self.pop();
                    self.stack.push(build(obj, state));
                }
                0x73 => {
                    // SETITEM
                    let value = self.pop();
                    let key = self.pop();
                    if let Some(Val::Dict(entries)) = self.stack.last_mut() {
                        if entries.len() >= MAX_ITEMS {
                            return Err(bad("dict size limit exceeded"));
                        }
                        entries.push((key, value));
                    }
                }
                0x75 => {
                    // SETITEMS
                    let items = self.pop_to_mark();
                    if let Some(Val::Dict(entries)) = self.stack.last_mut() {
                        if entries.len() + items.len() / 2 > MAX_ITEMS {
                            return Err(bad("dict size limit exceeded"));
                        }
                        entries.extend(pairs(items));
                    }
                }
                0x61 => {
                    // APPEND
                    let value = self.pop();
                    if let Some(Val::List(list)) = self.stack.last_mut() {
                        if list.len() >= MAX_ITEMS {
                            return Err(bad("list size limit exceeded"));
                        }
                        list.push(value);
                    }
                }
                0x65 => {
                    // APPENDS
                    let items = self.pop_to_mark();
                    if let Some(Val::List(list)) = self.stack.last_mut() {
                        if list.len() + items.len() > MAX_ITEMS {
                            return Err(bad("list size limit exceeded"));
                        }
                        list.extend(items);
                    }
                }
                0x71 => {
                    let idx = self.byte()? as u32;
                    self.memoize(idx); // BINPUT
                }
                0x72 => {
                    let idx = self.u32()?;
                    self.memoize(idx); // LONG_BINPUT
                }
                0x94 => {
                    let idx = self.memo.len() as u32;
                    self.memoize(idx); // MEMOIZE
                }
                0x68 => {
                    let idx = self.byte()? as u32;
                    self.memo_get(idx); // BINGET
                }
                0x6a => {
                    let idx = self.u32()?;
                    self.memo_get(idx); // LONG_BINGET
                }
                0x30 => {
                    self.pop(); // POP
                }
                0x31 => {
                    self.pop_to_mark(); // POP_MARK
                }
                0x49 => {
                    // INT (text)
                    let line = String::from_utf8_lossy(self.line()?).into_owned();
                    let s = line.trim();
                    let v = match s {
                        "00" => Val::Bool(false),
                        "01" => Val::Bool(true),
                        _ => Val::Int(s.parse().unwrap_or(0)),
                    };
                    self.stack.push(v);
                }
                0x91 | 0x90 => {
                    // FROZENSET / ADDITEMS: irrelevant to state dicts
                    let items = self.pop_to_mark();
                    if op == 0x91 {
                        self.stack.push(Val::Tuple(items));
                    }
                }
                0x97 => self.stack.push(Val::Opaque), // NEXT_BUFFER
                0x98 => {}                            // READONLY_BUFFER
                other => {
                    return Err(bad(format!(
                        "unsupported pickle opcode 0x{other:02x} at {}",
                        self.pos - 1
                    )))
                }
            }
            if self.stack.len() > MAX_STACK || self.memo.len() > MAX_MEMO {
                return Err(bad("pickle stack/memo limit exceeded"));
            }
        }
        match self.refusal.take() {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn memoize(&mut self, idx: u32) {
        if let Some(v) = self.stack.last() {
            // Only scalars are memoized. Containers (including tuples,
            // which torch nests freely) would let `T <- (T, T)` double the
            // heap every few opcodes; state dicts never need container
            // back-references.
            let stored = match v {
                Val::Int(n) => Val::Int(*n),
                Val::Bool(b) => Val::Bool(*b),
                Val::Float(f) => Val::Float(*f),
                Val::None => Val::None,
                Val::Str(s) if s.len() <= MAX_MEMO_STR => Val::Str(s.clone()),
                Val::Global { module, name } => Val::Global {
                    module: module.clone(),
                    name: name.clone(),
                },
                Val::Storage { key, dtype } => Val::Storage {
                    key: key.clone(),
                    dtype: *dtype,
                },
                _ => Val::Opaque,
            };
            self.memo.insert(idx, stored);
        }
    }

    fn memo_get(&mut self, idx: u32) {
        let v = self.memo.get(&idx).cloned().unwrap_or(Val::None);
        self.stack.push(v);
    }

    fn reduce(&mut self, callable: Val, args: Val) -> Result<Val> {
        let Val::Global { module, name } = &callable else {
            return Ok(Val::Opaque);
        };
        if module == "collections" && name == "OrderedDict" {
            return Ok(Val::Dict(Vec::new()));
        }
        if module == "torch._utils" && name.starts_with("_rebuild_tensor") {
            if let Val::Tuple(items) = &args {
                return self.rebuild_tensor(items);
            }
        }
        Ok(Val::Opaque)
    }

    /// `_rebuild_tensor_v2(storage, storage_offset, size, stride, ...)`.
    /// Non-contiguous strides are refused: the storage bytes of such a
    /// tensor are not its row-major bytes, and reading them as dense would
    /// silently return wrong data.
    fn rebuild_tensor(&mut self, args: &[Val]) -> Result<Val> {
        if args.len() < 4 {
            return Ok(Val::Opaque);
        }
        let Val::Storage { key, dtype } = &args[0] else {
            return Ok(Val::Opaque);
        };
        let offset_elems = match &args[1] {
            Val::Int(v) if *v >= 0 => *v as u64,
            _ => return Ok(Val::Opaque),
        };
        let (Some(shape), Some(stride)) = (dims(&args[2]), dims(&args[3])) else {
            return Ok(Val::Opaque);
        };

        // Contiguity: stride[i] == product(shape[i+1..]), dims of size ≤ 1
        // exempt (their stride is arbitrary).
        let mut expected = 1u64;
        for (i, &dim) in shape.iter().enumerate().rev() {
            if dim > 1 && stride.get(i) != Some(&expected) {
                self.refusal = Some(Error::Unsupported(format!(
                    "pt: tensor on storage {key:?} is not contiguous \
                     (shape {shape:?}, stride {stride:?}); refusing to read it as dense"
                )));
                return Ok(Val::Opaque);
            }
            expected = expected.saturating_mul(dim.max(1));
        }

        let Some(byte_offset) = dtype.width().and_then(|w| offset_elems.checked_mul(w)) else {
            self.refusal = Some(bad("storage offset overflows"));
            return Ok(Val::Opaque);
        };
        Ok(Val::Tensor(Box::new(TensorRef {
            storage_key: key.clone(),
            byte_offset,
            shape,
            dtype: *dtype,
        })))
    }

    /// torch persistent id: `("storage", <StorageType>, key, location, numel)`.
    fn persistent_load(&mut self, pid: Val) -> Val {
        let Val::Tuple(items) = &pid else {
            return Val::Opaque;
        };
        if items.len() < 5 {
            return Val::Opaque;
        }
        let (Val::Str(tag), Val::Global { name, .. }, Val::Str(key), _, Val::Int(numel)) =
            (&items[0], &items[1], &items[2], &items[3], &items[4])
        else {
            return Val::Opaque;
        };
        if tag != "storage" || *numel < 0 {
            return Val::Opaque;
        }
        match storage_dtype(name) {
            Some(dtype) => Val::Storage {
                key: key.clone(),
                dtype,
            },
            None => {
                self.refusal = Some(Error::Unsupported(format!(
                    "pt: storage type {name:?} has no registered projection"
                )));
                Val::Opaque
            }
        }
    }
}

fn dims(v: &Val) -> Option<Vec<u64>> {
    match v {
        Val::Tuple(items) | Val::List(items) => items
            .iter()
            .map(|d| match d {
                Val::Int(v) if *v >= 0 => Some(*v as u64),
                _ => None,
            })
            .collect(),
        _ => None,
    }
}

fn build(obj: Val, state: Val) -> Val {
    match (obj, state) {
        (Val::Dict(mut entries), Val::Dict(more)) => {
            entries.extend(more);
            Val::Dict(entries)
        }
        (_, state) => state,
    }
}

fn pairs(items: Vec<Val>) -> Vec<(Val, Val)> {
    let mut out = Vec::with_capacity(items.len() / 2);
    let mut it = items.into_iter();
    while let (Some(k), Some(v)) = (it.next(), it.next()) {
        out.push((k, v));
    }
    out
}

fn long_from_le(bytes: &[u8]) -> i64 {
    if bytes.is_empty() {
        return 0;
    }
    let n = bytes.len().min(8);
    let mut v = 0i64;
    for (i, &b) in bytes[..n].iter().enumerate() {
        v |= (b as i64) << (i * 8);
    }
    if bytes[n - 1] & 0x80 != 0 && n < 8 {
        v |= !0i64 << (n * 8);
    }
    v
}

/// Walks the unpickled tree and collects named tensors. Duplicate names
/// are rejected (a genuine Python dict cannot produce them).
fn collect_tensors(
    prefix: &str,
    v: &Val,
    out: &mut BTreeMap<String, TensorRef>,
    depth: usize,
) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(bad("structure nesting too deep"));
    }
    match v {
        Val::Tensor(t) => {
            if out.insert(prefix.to_string(), (**t).clone()).is_some() {
                return Err(bad(format!("duplicate tensor name {prefix:?}")));
            }
        }
        Val::Dict(entries) => {
            for (k, val) in entries {
                let name = match k {
                    Val::Str(s) if prefix.is_empty() => s.clone(),
                    Val::Str(s) => format!("{prefix}.{s}"),
                    _ => prefix.to_string(),
                };
                collect_tensors(&name, val, out, depth + 1)?;
            }
        }
        Val::List(items) | Val::Tuple(items) => {
            for (i, item) in items.iter().enumerate() {
                let name = if prefix.is_empty() {
                    i.to_string()
                } else {
                    format!("{prefix}.{i}")
                };
                collect_tensors(&name, item, out, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}

// =======================================================================
// Container
// =======================================================================

enum StorageLoc {
    /// Stored entry: absolute range in the file.
    Stored { offset: u64, length: u64 },
    /// Compressed entry: inflated on demand, cached whole.
    Compressed { zip_index: usize, length: u64 },
}

/// Tensors that live inside a compressed storage.
///
/// A `.pt` tensor is a window into a storage, and a compressed storage has no
/// window until it is inflated, so these have no address and the whole
/// storage is cached once rather than inflated per tensor.
struct Compressed {
    archive: std::sync::Mutex<zip::ZipArchive<File>>,
    cache: std::sync::Mutex<BTreeMap<usize, Vec<u8>>>,
    /// Keyed by the `key` in [`Payload::Opaque`]:
    /// (zip index, storage length, byte offset within the storage).
    slices: Vec<(usize, u64, u64)>,
}

impl Compressed {
    fn ensure_cached(&self, zip_index: usize, length: u64) -> Result<()> {
        if self
            .cache
            .lock()
            .expect("pt cache lock")
            .contains_key(&zip_index)
        {
            return Ok(());
        }
        let mut archive = self.archive.lock().expect("pt archive lock");
        let mut entry = archive
            .by_index(zip_index)
            .map_err(|e| bad(format!("storage {zip_index}: {e}")))?;
        let cap = crate::safe::alloc_size("pt storage", length)?;
        let mut bytes = Vec::with_capacity(cap);
        std::io::Read::take(&mut entry, length + 1)
            .read_to_end(&mut bytes)
            .map_err(|e| bad(format!("storage {zip_index}: {e}")))?;
        if bytes.len() as u64 != length {
            return Err(bad(format!(
                "storage {zip_index} decompressed size mismatch"
            )));
        }
        self.cache
            .lock()
            .expect("pt cache lock")
            .insert(zip_index, bytes);
        Ok(())
    }
}

impl Decode for Compressed {
    fn decode(&self, key: u64, decoded_len: u64) -> Result<Vec<u8>> {
        let (zip_index, length, offset) = *self
            .slices
            .get(key as usize)
            .ok_or_else(|| bad(format!("no compressed tensor {key}")))?;
        self.ensure_cached(zip_index, length)?;
        let cache = self.cache.lock().expect("pt cache lock");
        let storage = &cache[&zip_index];
        let (start, end) = crate::safe::range("pt tensor", offset, decoded_len, storage.len())?;
        Ok(storage[start..end].to_vec())
    }
}

pub(crate) fn project(store: &Store) -> Result<Projection> {
    let mut archive = zip::ZipArchive::new(File::open(store.path())?)
        .map_err(|e| bad(format!("not a ZIP archive: {e}")))?;

    // Locate the pickle.
    let pickle_name = (0..archive.len())
        .filter_map(|i| archive.by_index_raw(i).ok().map(|e| e.name().to_string()))
        .find(|n| n.ends_with("data.pkl"))
        .ok_or_else(|| bad("no data.pkl entry"))?;
    let prefix = format!(
        "{}data/",
        pickle_name.strip_suffix("data.pkl").unwrap_or_default()
    );

    // The pickle is decompressed here, so its *decompressed* size must be
    // capped independently of what the ZIP declares.
    let mut pickle = Vec::new();
    let mut entry = archive
        .by_name(&pickle_name)
        .map_err(|e| bad(format!("{pickle_name}: {e}")))?;
    std::io::Read::take(&mut entry, MAX_PICKLE_BYTES as u64 + 1).read_to_end(&mut pickle)?;
    if pickle.len() > MAX_PICKLE_BYTES {
        return Err(bad("data.pkl exceeds the 256 MiB limit"));
    }
    drop(entry);

    // Run the VM and collect tensors.
    let mut vm = Vm {
        data: &pickle,
        pos: 0,
        stack: Vec::new(),
        marks: Vec::new(),
        memo: BTreeMap::new(),
        refusal: None,
    };
    vm.execute()?;
    let mut tensors = BTreeMap::new();
    for v in &vm.stack {
        collect_tensors("", v, &mut tensors, 0)?;
    }
    if tensors.is_empty() {
        return Err(bad("no tensors found"));
    }

    // Locate each referenced storage entry.
    let mut storages: BTreeMap<String, StorageLoc> = BTreeMap::new();
    for t in tensors.values() {
        let key = &t.storage_key;
        if storages.contains_key(key) {
            continue;
        }
        let entry_name = format!("{prefix}{key}");
        let zip_index = archive
            .index_for_name(&entry_name)
            .ok_or_else(|| bad(format!("storage entry {entry_name:?} missing")))?;
        let entry = archive
            .by_index_raw(zip_index)
            .map_err(|e| bad(format!("storage entry {entry_name:?}: {e}")))?;
        let loc = if entry.compression() == zip::CompressionMethod::Stored {
            let start = entry.data_start();
            let size = entry.size();
            if crate::safe::add("pt storage", start, size)? > store.len() {
                return Err(bad(format!("storage {entry_name:?} extends past file")));
            }
            StorageLoc::Stored {
                offset: start,
                length: size,
            }
        } else {
            StorageLoc::Compressed {
                zip_index,
                length: entry.size(),
            }
        };
        drop(entry);
        storages.insert(key.clone(), loc);
    }

    // Build the catalog, bounds-checking every tensor against its storage.
    let mut catalog = Catalog::new();
    let mut slices: Vec<(usize, u64, u64)> = Vec::new();
    for (name, t) in &tensors {
        let elems = crate::safe::product("pt shape", &t.shape)?;
        let byte_len = t
            .dtype
            .size(elems)
            .ok_or_else(|| bad("size not computable"))?;
        let storage = &storages[&t.storage_key];
        let storage_len = match storage {
            StorageLoc::Stored { length, .. } | StorageLoc::Compressed { length, .. } => *length,
        };
        t.byte_offset
            .checked_add(byte_len)
            .filter(|&e| e <= storage_len)
            .ok_or_else(|| bad(format!("tensor {name:?} extends past its storage")))?;

        let payload = match storage {
            StorageLoc::Stored { offset, .. } => Payload::At(Location {
                store: StoreId(0),
                offset: offset + t.byte_offset,
                len: byte_len,
            }),
            StorageLoc::Compressed { zip_index, length } => {
                slices.push((*zip_index, *length, t.byte_offset));
                Payload::Opaque {
                    store: StoreId(0),
                    key: slices.len() as u64 - 1,
                    decoded_len: byte_len,
                }
            }
        };

        catalog.insert(
            name.clone(),
            Entry {
                shape: t.shape.clone(),
                term: Some(t.dtype.into()),
                layout: None,
                attributes: None,
                payload,
                digest: None,
                blocks: None,
            },
        );
    }

    // Storages sit behind ZIP local headers and torch packs several tensors
    // into one storage, so nothing here can prove a tensor has its pages to
    // itself: occupancy stays unstated.
    let projection = Projection::new(catalog);
    Ok(if slices.is_empty() {
        projection
    } else {
        projection.with_decoder(Box::new(Compressed {
            archive: std::sync::Mutex::new(archive),
            cache: std::sync::Mutex::new(BTreeMap::new()),
            slices,
        }))
    })
}
