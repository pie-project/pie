# zTensor Container Format, Version 2

**Status:** Draft 2 · **File extension:** `.zt` · **Footer version integer:** `2`

---

## 1. Introduction

zTensor v2 is a container format for tensor data. It is designed around one
premise: a format survives by separating what must never change from what is
allowed to die.

The format is organized in three layers with explicit stability contracts:

| Layer | Contents | Stability contract |
| --- | --- | --- |
| **L0 — Container** | Magic, footer, blob heap, alignment floor, byte order | **Frozen.** Never changes. A major revision changes only what the footer points to. |
| **L1 — Manifest** | Manifest schema (deterministic CBOR): objects, parts, blob references, shards | Gated by the footer version integer. Evolves rarely; unknown fields are ignored. |
| **L2 — Vocabulary** | Layout profiles, logical types, encoding profiles, digest algorithms | **Deliberately mortal.** Namespaced, versioned identifiers managed by a registry. |

Everything in the file is a **blob**: an aligned, contiguous, unnamed byte
range. Tensor data is blobs. The manifest itself is a blob. The footer is a
fixed-size trailer whose only job is to point at the root manifest blob and
protect it with a hash.

### 1.1 Design principles

1. **Container and semantics are fully separated.** The container knows
   nothing about tensors. It stores aligned blobs and one manifest.
2. **Everything is a blob.** One mechanism; no special cases.
3. **Validation is not optional.** Every invariant in this document is a MUST,
   and readers MUST reject files that violate them. There are no
   silent-fallback rules.
4. **The format owns only the minimal verifiable contract.** Values derivable
   from the data (e.g., actual alignment) are not stored. Policy knobs
   (e.g., placement) belong to writers, not to the format.

### 1.2 Conventions

The key words MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be
interpreted as described in RFC 2119.

All multi-byte integers in the container (L0) are **little-endian**. All
tensor element data is little-endian. The only exception is CBOR's internal
length encoding, which is big-endian per RFC 8949 and handled by any CBOR
library.

---

## 2. Container (L0)

### 2.1 File structure

```text
+----------------------------------------+  offset 0
| Magic (8 bytes)                        |
+----------------------------------------+
| Blob heap                              |
|   blob, blob, ..., manifest blob       |  every blob offset ≡ 0 (mod 4096)
|   (padding between blobs = 0x00)       |
+----------------------------------------+
| Footer (40 bytes)                      |
+----------------------------------------+  EOF
```

A file MUST be at least 48 bytes long (8-byte magic + 40-byte footer).

### 2.2 Magic

The 8-byte magic appears at offset 0 and again at the end of the footer:

```text
89 5A 54 32 0D 0A 1A 0A        (0x89 'Z' 'T' '2' CR LF SUB LF)
```

Following the PNG convention, the high-bit first byte, the CRLF pair, and the
trailing LF detect 7-bit stripping, line-ending translation, and other
transmission corruption.

### 2.3 Footer

The footer occupies the last 40 bytes of the file:

| Offset from EOF | Size | Field | Description |
| --- | --- | --- | --- |
| −40 | 8 | `manifest_offset` | u64 LE. Absolute offset of the manifest blob. `0` if no manifest (data shard, see §7). |
| −32 | 8 | `manifest_length` | u64 LE. Byte length of the manifest blob. `0` if no manifest. |
| −24 | 8 | `manifest_hash` | u64 LE. XXH3-64 of the manifest bytes. `0` if no manifest. |
| −16 | 4 | `version` | u32 LE. This document defines version `2`. Gates the L1 schema. |
| −12 | 4 | `reserved` | u32 LE. Writers MUST write `0`; readers MUST ignore it. Padding, not an evolution channel — `version` is the only one. |
| −8 | 8 | `magic` | Same 8 bytes as §2.2. |

`manifest_hash` is an integrity check against corruption, not a cryptographic
commitment; XXH3-64 is acceptable to freeze into L0 because its role is
error detection. Cryptographic identity is provided by digests (§6) and
canonical form (§6.3), which are algorithm-agile.

### 2.4 Blobs and alignment

- A blob is a contiguous byte range identified by `(offset, length)`.
- Every blob offset MUST satisfy `offset % 4096 == 0` and `offset >= 4096`.
- Every blob MUST satisfy `offset + length <= file_size − 40` (blobs may not
  overlap the footer).
- Blob references within a file, plus the manifest blob, MUST be pairwise
  **identical or disjoint**: two references are valid iff they have exactly
  equal `(offset, length)` or do not overlap at all. Partial overlap MUST be
  rejected. Identical references are deliberate — they enable weight tying
  (e.g., input/output embeddings sharing one blob) and intra-file dedup,
  while the dangerous case (one range validated under one interpretation,
  read under another) remains impossible. (Zero-length blobs are exempt from
  this check but not from the alignment check.)
- All bytes between blobs, and between the magic and the first blob, MUST be
  written as `0x00`. Readers MAY ignore padding content.

**4096 is a floor, not a ceiling.** Writers MAY place blobs at any coarser
alignment (16 KiB, 64 KiB, ...); a multiple of a coarser power of two is
automatically a multiple of 4096, so stricter placement is always valid.
Because alignment is observable from the offsets themselves, the actual
alignment used by a writer is **not** stored in the file. See §6.3 for the
canonical placement (64 KiB).

### 2.5 Append and generations

The container is append-only. To amend a file, a writer appends new blobs, a
new manifest blob, and a new footer at the end. Prior manifest blobs remain
in the file as unreferenced blobs; the file's meaning is always defined by
the footer at EOF. Writers MUST NOT truncate or overwrite existing bytes.

Files SHOULD nevertheless be treated as immutable artifacts once published.
Canonical files (§6.3) contain exactly one manifest and no unreferenced
blobs.

A file whose footer is not at EOF — e.g., after a crashed append — is simply
invalid; there is no reader-side recovery (§10). Durable publication is a
transport concern: write to a temporary name, then rename atomically
(Appendix D).

---

## 3. Manifest (L1)

### 3.1 Encoding

The manifest blob is a single CBOR map encoded in the **Core Deterministic
Encoding** profile of RFC 8949 §4.2.1 (definite lengths, shortest-form
integers, bytewise-lexicographic key order).

Readers MUST reject a manifest that is not in deterministic encoding:
non-shortest heads, indefinite lengths, unsorted or duplicate map keys, and
non-canonical floats (NaN other than `0xf9 0x7e00`, or a float wider than
the value requires). Anything weaker is unsound, not merely lenient: a
reader that accepts non-canonical floats will accept two NaN-payload map
keys as distinct — duplicate keys in disguise. Readers MUST reject a
manifest whose `manifest_length` exceeds **1 GiB**, before parsing.

Readers MUST ignore map keys they do not recognize, at every level. This is
the L1 minor-evolution mechanism; the footer `version` integer is the major
one.

An `attributes` field MUST be a map whose keys are text strings obeying the
name rules of §3.5. Attribute values (at every level) MUST NOT use CBOR
tags; permitted types are integers, text strings, byte strings, booleans,
null, floats, arrays, and maps. Nesting is limited to 32 levels **measured
from the manifest root**, so a reader needs one depth counter rather than a
per-value one. The manifest blob itself is always stored raw; it is never
compressed or encoded.

### 3.2 Root schema

Shown in CBOR diagnostic notation:

```text
{
  "attributes": { ... },            ; optional, arbitrary file metadata
  "shards": { 1: {...}, 2: {...} }, ; optional, see §7; absent ⇒ single file
  "objects": {                      ; required in a root manifest
    "layer1.weight": <object>,
    "layer1.bias":   <object>
  }
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `objects` | map | Yes | Named object definitions. |
| `shards` | map | No | Shard table (§7), keyed by integer shard index ≥ 1. When absent, the file is self-contained. |
| `attributes` | map | No | Arbitrary key-value metadata for the whole file. |

### 3.3 Object

```text
{
  "shape": [4096, 4096],
  "layout": "dense",                ; or a namespaced profile id, §5
  "attributes": { ... },            ; optional
  "parts": { "data": <part> }
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `shape` | array of u64 | Yes | Logical dimensions. Rank MUST be ≤ 64. The element count is the product of dimensions (empty shape ⇒ 1, scalar). The product MUST NOT overflow u64. |
| `layout` | string | Yes | `"dense"` (core, §5.1) or a namespaced layout profile id (§5.2). |
| `parts` | map | Yes | One entry per data blob, keyed by role name. MUST be non-empty — an object with no bytes has no meaning, and layouts cannot state that rule for layouts a reader does not know. |
| `attributes` | map | No | Per-object metadata. Layout profiles define which keys they require. |

### 3.4 Part

```text
{
  "dtype": "bf16",
  "type": "...",                    ; optional logical type, §4.2
  "blob": [0, 65536, 33554432],     ; [shard, offset, length]
  "encoding": "zt.zstd-seekable/1", ; optional; absent ⇒ raw
  "decoded_length": 67108864,       ; required iff encoding is present
  "digest": "xxh3:9f2c..."          ; optional, over decoded bytes
}
```

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `dtype` | string | Yes | Storage type, one of the 12 primitives (§4.1). |
| `type` | string | No | Logical type (§4.2). Absent ⇒ logical type equals `dtype`. |
| `blob` | `[u64, u64, u64]` | Yes | `[shard_index, offset, length]`. Shard index `0` always refers to the containing file; the reference is subject to §2.4 within the referenced shard. A blob is contiguous within a single file and never spans shards. |
| `encoding` | string | No | Absent ⇒ raw bytes. Otherwise a namespaced encoding profile id (§5.3). |
| `decoded_length` | u64 | Cond. | Decoded byte count. MUST be present iff `encoding` is present. Readers MUST verify the decoded output is exactly this long. |
| `digest` | string | No | `"<algorithm>:<lowercase hex>"`, computed over the **decoded** (logical) bytes. Registered algorithms: `xxh3` (64-bit), `sha256`. |

The *decoded size* of a part is `length` when `encoding` is absent, else
`decoded_length`.

Digests cover decoded bytes because a digest identifies **content**, not its
encoding; re-encoding a blob does not change its digest. Transport integrity
of encoded bytes is the encoding profile's responsibility (e.g., zstd frame
checksums, §C).

### 3.5 Names

Object names, part names, and attribute keys MUST be non-empty, valid UTF-8
strings of at most 1024 bytes containing no `U+0000`. That is the entire
reader-side check. Writers SHOULD emit names in Unicode Normalization
Form C, and canonical form (§6.3) requires it — but readers do not verify
normalization: NFC verification needs Unicode tables, which would break the
minimal-reader contract (§11). Normalization is a writer duty, not a reader
check.

### 3.6 Validation summary

A reader MUST reject a file if any of the following fail:

1. Footer: magic, `version` supported.
2. Manifest: bounds (`manifest_offset`/`length` inside the file, off the
   footer), size cap, XXH3-64 match, deterministic-CBOR parse, no duplicate
   keys.
3. Every blob reference: shard index is `0` or a key of `shards`,
   `offset % 4096 == 0`,
   `offset >= 4096`, `offset + length` within the referenced shard's data
   region.
4. Blob references are grouped by the file they point into (shard index)
   and checked per group: within each file, references — plus the manifest
   blob for shard 0 — are pairwise identical-or-disjoint; any partial
   overlap is rejected (§2.4). References into a shard are checked from the
   root manifest alone, without opening the shard.
5. Layout rules for every object whose layout the reader interprets
   (e.g., the dense size equation, §5.1).
6. Name rules (§3.5); shape rank and overflow rules (§3.3); attribute value
   constraints (§3.1).

---

## 4. Type system (L2)

### 4.1 Storage types (`dtype`)

A closed set of 12 primitives. The storage layer's only responsibilities are
**byte width** and **endianness**; therefore no storage type may have invalid
bit patterns.

| Category | Types | Width |
| --- | --- | --- |
| Float | `f64`, `f32`, `f16`, `bf16` | 8, 4, 2, 2 |
| Signed int | `i64`, `i32`, `i16`, `i8` | 8, 4, 2, 1 |
| Unsigned int | `u64`, `u32`, `u16`, `u8` | 8, 4, 2, 1 |

There is no `bool` storage type; `bool` is a logical type over `u8`
(Appendix A). This is deliberate: a type with invalid bit patterns
(`0x02`..`0xFF`) does not belong in the layer whose contract is "any byte
pattern is valid."

### 4.2 Logical types (`type`)

An open, registered set giving meaning to raw storage elements. When absent,
the logical type equals the storage type and no interpretation is needed.
Each registered logical type specifies its required `dtype`, its **size
function** — the decoded byte size for a given logical element count — and,
for packed sub-byte types, the bit order. A plain type has
`size(n) = n × width(dtype)`; a compound type like `complex64` has
`size(n) = 2n × width`; a packed type like `f4_e2m1` has `size(n) = ⌈n/2⌉`.
See Appendix A.

**Unknown logical types are an error.** A reader asked to decode a part whose
`type` it does not recognize MUST refuse. Raw structural access (bytes plus
metadata) MAY be offered through an explicitly structural API, but a reader
MUST NOT silently reinterpret unknown-typed data as its storage type. Two
conforming readers must either agree on a tensor or agree on an error.

---

## 5. Layouts (L2)

An object's `layout` selects how its parts combine into a tensor.

### 5.1 Core layout: `dense`

The only layout defined by this document.

- Exactly one part, named `"data"`.
- Row-major (C-contiguous) element order.
- The part's decoded size MUST equal the logical type's size function
  applied to `num_elements(shape)` (for a part with no `type`, that is
  `num_elements × width(dtype)`). Readers MUST verify this equation.

### 5.2 Layout profiles

Every other layout — sparse, quantized, anything future — is a **profile**:
a separately published mini-specification identified by a namespaced,
versioned id such as `zt.sparse_csr/1` or `pie.paged_kv/1`. Profiles live
beside this document under `spec/profiles/`; this core specification
defines the mechanism and nothing else.

A layout profile MUST completely specify: its required and optional parts;
each part's permitted `dtype`/`type`; how each part's decoded size derives
from `shape` and the object's `attributes`; the meaning of every attribute
key it uses; and its validation rules. A profile that cannot be implemented
from its text alone is not a profile. The test is whether two independent
implementations, working from the text, produce interchangeable files.

The `zt.` namespace is reserved for profiles blessed by the zTensor registry.
Vendors MUST use their own namespace prefix. Version suffixes (`/1`, `/2`)
are mandatory; any semantic change requires a new version.

A reader encountering an unknown layout MUST NOT interpret the object; it
MAY expose the object structurally (shape, attributes, raw parts).

The registry at the time of writing: `zt.sparse_csr/1`, `zt.sparse_coo/1`,
`zt.quant_group/1`, `zt.mx/1`, the `gguf.<type>/1` family, and the encoding
profile `zt.zstd-seekable/1`.

#### Parametric and opaque profiles

There are two kinds of profile, and choosing the wrong one is the common
way to write a bad one.

A **parametric** profile describes a space, not a scheme: its attributes
determine the decoder, and two different attribute sets under the same
profile are two different schemes. `zt.quant_group/1` is one — affine
group quantization, where the bit width, group size, packing order, scale
form and zero-point form are each stated. A parametric profile MUST make
every parameter its decoder needs a required attribute; anything left
unstated will be inferred from something incidental, which is how a file
that happens to use a 32-element group comes to be read as the one scheme
that used to have 32-element groups.

An **opaque** profile does not describe the payload's internal structure
at all. It names an authoritative external definition and preserves the
bytes verbatim. The `gguf.<type>/1` family is opaque: a ggml block struct
interleaves scales with data, and `q4_k` nests a second level of scales
quantized to six bits inside a super-block — structure no attribute set
describes without inventing a layout language. An opaque profile MUST
still carry the constants a reader needs to validate sizes without
knowing the layout (for the `gguf` family, `elems_per_block` and
`block_bytes`), and its version suffix carries real weight: an external
definition that changes under a stable name is exactly what the suffix
distinguishes.

Which kind a profile is settled by one question: **can the attributes
alone determine the decoder?** If they cannot, do not force it. A profile
parameterized past what its payload actually admits is readable only by
implementations of the invented language, which is worse than an opaque
profile with an accurate name.

### 5.3 Encoding profiles

The core defines exactly one encoding: **raw** (the absence of the
`encoding` field). Compression algorithms are the most obviously mortal
component of any container, so they live in L2 like layouts:
namespaced, versioned encoding profiles, published under
`spec/profiles/` like layouts.

A reader encountering an unknown encoding MUST refuse to decode the part
(structural access to the encoded bytes MAY be offered).

---

## 6. Integrity, identity, and canonical form

### 6.1 Integrity ladder

Cheapest first; each rung is independent:

1. **Structural** — footer magic and bounds checks (always on).
2. **Manifest** — XXH3-64 in the footer (always on).
3. **Transport** — encoding-profile checksums, e.g. zstd frame checksums
   (on whenever the encoding is used).
4. **Content** — per-part `digest` over decoded bytes (optional; verify
   mode).
5. **Model identity / provenance** — whole-file hash of a canonical file,
   optionally signed (§6.4).

### 6.2 What is deliberately not protected

Padding bytes are written as zero but not covered by any digest;
unreferenced blobs (superseded generations, §2.5) are not covered. Canonical
form closes both gaps by forbidding them.

### 6.3 Canonical form

A file is **canonical** iff all of the following hold:

1. Exactly one manifest blob; no unreferenced blobs.
2. Every blob offset ≡ 0 (mod **65536**), assigned by packing blobs at
   successive 64 KiB boundaries starting at offset 65536, in the order of
   rule 3.
3. Blobs are ordered by the bytewise-lexicographic `(object name, part
   name)` of their first reference. Parts with byte-identical decoded
   content MUST share a single blob (aliased references, §2.4); a writer
   that finds sharing candidates by hashing MUST confirm equality on the
   bytes themselves, since a hash collision would otherwise alias two
   different tensors onto one blob.
   The manifest blob comes last; the footer immediately follows it.
4. All parts use raw encoding (no `encoding` field) and every part carries
   an `xxh3` digest.
5. All names are in Unicode Normalization Form C.
6. The model is single-file (no `shards` table). A canonical multi-file
   profile additionally requires a fixed shard-partition policy and is
   deferred.

Because the manifest encoding is deterministic and nothing in a canonical
file depends on time, randomness, or writer identity, **identical tensors
produce bit-identical files**. The hash of a canonical file is therefore a
stable identity for the model. Canonical form is the RECOMMENDED
distribution format; non-canonical files are still fully conforming.

The 64 KiB canonical alignment guarantees that no two blobs share a page on
any known page size (4 KiB x86/ARM, 16 KiB Apple Silicon, 64 KiB ARM64
distributions): every tensor can be independently memory-mapped, registered,
and evicted (`madvise`) with exact page ranges, on every platform, with no
fallback path. Average cost is 32 KiB per tensor — noise at weight-file
scale.

### 6.4 Signing

Signing is out-of-band: a detached signature over the bytes (or a
cryptographic hash) of a canonical file. For multi-file models, signing the
root file suffices **if** the root's shard digests (§7) use a cryptographic
algorithm (`sha256`), since the root manifest then commits to every shard
byte (one-way Merkle direction).

---

## 7. Multi-file models (shards)

Sharding is not a feature bolted onto the format; it is the general case of
the blob reference, with the single file as the degenerate case.

### 7.1 Shard table

```text
"shards": {
  1: { "size": 8589934592, "digest": "sha256:..." },
  2: { "size": 8589934592, "digest": "sha256:..." }
}
```

- The shard table is a CBOR map from **unsigned integer shard index** to a
  shard identity. Keys MUST be ≥ 1.
- Shard index `0` denotes the containing file by definition and MUST NOT
  appear as a key. It needs no identity entry: its manifest is protected by
  the footer hash, and a whole-file self-digest would be circular.
- Every entry MUST carry `size` (exact file size in bytes) and `digest`
  (whole-file). Because the table contains no names, the digest **is** the
  shard's identity; it is therefore required, not optional. `xxh3` is the
  minimum; distribution files intended for signing SHOULD use `sha256`
  (§6.4).
- When `shards` is absent, every blob reference MUST use shard index `0`.

The manifest stores shard **identity**, never shard **location**: no file
names, paths, or URLs appear anywhere in the format. A file name is owned by
the filesystem, is not verifiable by the format, and would couple model
identity to naming (a rename would change or break the model). Resolution of
a shard index to bytes is entirely the transport's concern (Appendix D gives
the RECOMMENDED conventions); because identity lives in the content, a
resolver can always fall back to locating a file by size and digest.

### 7.2 Data shards

A data shard is a minimal container: magic, blob heap, and a footer with
`manifest_offset = manifest_length = manifest_hash = 0`. It carries no
manifest and describes nothing; all meaning lives in the root manifest.

Knowledge flows one way — the root knows its shards; shards do not know
their root. This avoids circular hashing and keeps multi-file output
deterministic.

A shard need not be a data shard: **any** `.zt` file, including one with its
own manifest, may serve as a shard of another model. The referencing
manifest treats it purely as a blob heap identified by size and digest.
This is how overlay models work: a LoRA root can reference the base model
file's blobs directly and store only its own deltas.

### 7.3 Verification ladder

Cheapest first: (1) shard footer magic and version; (2) actual file size
equals the root's `size`; (3) whole-file `digest` match (deep verify /
signed distribution).

Placement of blobs across shards is entirely the writer's policy. Writers
MAY shard by layer, by pipeline stage, or by size threshold; the container
imposes nothing.

---

## 8. Reading a file

```text
1. Require file_size ≥ 48.
2. Read the last 40 bytes. Verify magic and version.
3. If manifest_length == 0: this is a data shard; only structural access
   is possible. Stop.
4. Verify manifest bounds and manifest_length ≤ 1 GiB.
   Read the manifest bytes; verify XXH3-64 == manifest_hash.
5. Parse deterministic CBOR; reject duplicate keys.
6. Run the validation summary (§3.6) over the whole manifest, including the
   global non-overlap check.
7. To load a part: resolve the shard, seek to offset, read length bytes,
   decode per the encoding profile (verifying decoded_length exactly),
   verify digest if requested, interpret as dtype/type elements.
```

For remote access, steps 2 and 4 are exactly two range requests (the footer
is fixed-size).

### 8.1 Security rules

- **No code execution.** No field of this format is ever evaluated.
- **Bounds before bytes.** All §3.6 checks precede any data read.
- **Decompression limits.** `decoded_length` is enforced as an exact output
  size, pre-allocated and verified; a short or long decode is an error, never
  a silent zero-fill.
- **Caps.** Manifest ≤ 1 GiB; implementations SHOULD additionally cap total
  decoded allocation per read call.

---

## 9. Versioning policy

- **L0 is frozen.** The magic, footer layout, alignment floor, and byte
  order defined here never change. A hypothetical future major version
  changes the footer `version` integer and may point the footer at a
  different root structure — the container skeleton survives.
- **L1 minor evolution:** new OPTIONAL manifest fields; readers MUST ignore
  unknown fields. **L1 major evolution:** increment the footer `version`.
- **L2 evolves by registry:** new logical types, layout profiles, and
  encoding profiles are added without touching this document. Existing
  profile versions are immutable; semantic changes require a new `/n`.

---

## 10. Non-goals

This format deliberately does not attempt to be:

- **A database.** No random access over billions of objects, no query
  language, no in-place mutation. The manifest is a monolith by design; if
  you need an index over millions of entries, you need a different tool.
- **A partial-update store.** Amendment is generational append (§2.5), not
  in-place writes.
- **A crash-recovery journal.** A file whose footer is not at EOF is
  invalid, full stop. Durability comes from atomic publication (write to a
  temporary name, rename into place — Appendix D), not from reader-side
  recovery scans.
- **An encryption container.** Encryption belongs to the transport or
  filesystem layer.
- **A compute or graph description.** This format stores bytes and their
  interpretation, never behavior. (This is the tombstone on pickle's grave.)

Scope discipline is a futureproofing feature: these boundaries are what keep
the format small enough to remain correct.

---

## 11. Conformance

A conforming implementation is one that passes the **conformance corpus**:
a versioned set of golden files — valid files with their expected decoded
contents, and invalid files that MUST be rejected with the corresponding
§3.6 rule. The corpus and a minimal reference reader (a few hundred lines,
no dependencies beyond CBOR and XXH3) are normative companions to this
document: where prose and corpus disagree, the corpus is the bug tracker's
problem, but implementations follow the corpus until resolved.

The mandatory-to-implement surface is deliberately small: the `dense`
layout, raw encoding, and the 12 storage types. Everything beyond that is
optional to support but mandatory to refuse cleanly (§4.2, §5.2, §5.3).

---

## Appendix A — Registered logical types (initial)

| `type` | `dtype` | Size for n elements | Notes |
| --- | --- | --- | --- |
| `bool` | `u8` | n | Values MUST be `0x00` or `0x01`; readers MUST reject others when decoding or verifying (a raw structural read of the bytes is exempt). |
| `f8_e4m3fn` | `u8` | n | OCP / NVIDIA FP8 |
| `f8_e5m2` | `u8` | n | OCP FP8 |
| `f8_e4m3fnuz` | `u8` | n | AMD FP8 |
| `f8_e5m2fnuz` | `u8` | n | AMD FP8 |
| `f8_e8m0` | `u8` | n | OCP MX block-scale exponent |
| `f4_e2m1` | `u8` | ⌈n/2⌉ | OCP MXFP4 element. Packed two per byte, low nibble first; the final odd nibble, if any, MUST be zero (checked when decoding or verifying). |
| `complex64` | `f32` | 8n | Interleaved `[real, imag]` |
| `complex128` | `f64` | 16n | Interleaved `[real, imag]` |

A registry entry defines its required `dtype`, its size function, and — for
packed sub-byte types — the bit order. The `dense` layout applies the size
function to `num_elements(shape)`; other layout profiles state which element
count each part's size function receives (e.g., the reserved `zt.mx/1`
profile gives the `scales` part one `f8_e8m0` element per 32-element block).

## Appendix B — Recommended conventions (non-normative)

- **Shard resolution (positional):** for a root file named `<stem>.zt`,
  shard index `k` resolves to a sibling file `<stem>-<k as 5 digits>.zt`
  (root `model.zt` → shards `model-00001.zt`, `model-00002.zt`, ...).
  Renaming the family together (`model` → `llama`) keeps it resolvable.
- **Shard resolution (content-addressed):** a store MAY resolve by digest
  instead, e.g. `blobs/xxh3/<hex>`; shard references are then
  location-independent, which also enables sharing shards across models.
- **Shard resolution (fallback):** if a convention lookup fails, a resolver
  MAY scan the directory for a file matching the expected size, then verify
  its digest.
- **Writer alignment flag:** reference writer default is canonical placement
  (64 KiB). A `--align` option may lower placement to any multiple of 4096
  for space-sensitive, non-distribution files (e.g., small adapters).
- **Advisory attributes:** writers MAY record `"alignment": <n>` in file
  `attributes`; it is informational only and readers MUST NOT trust it over
  the observable offsets.
- **Atomic publication:** write to a temporary name in the destination
  directory, `fsync`, then `rename(2)` into place. A crashed write never
  leaves a partial file under a published name.
