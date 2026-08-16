//! The ingest aspect: a foreign vocabulary in, this crate's own out.
//!
//! [`contract`](crate::contract) is this module's mirror. It answers "how
//! does a row's checkpoint become tensors a device binds" -- zt to GPU. This
//! one answers the question one layer below: how does a checkpoint written in
//! somebody else's vocabulary become a zt artifact at all.
//!
//! For HuggingFace the answer is "it already is one", which is why this
//! module names only GGUF. That identity is an accident worth stating: the
//! catalog's rows are spelled HuggingFace-shaped, so a HuggingFace checkpoint
//! passes through and matches. `tests/a_row_identifies_from_its_own_names.rs`
//! turns the accident into a decision, and this module is what a vocabulary
//! that does NOT get the accident has to go through.
//!
//! # Dispatched on what the file says, not on what it looks like
//!
//! `crate::shared::weight_names::wire` has to sniff: it picks a family by
//! looking for a tensor only one of them publishes, in an order that matters,
//! with qwen3.5 explicitly bailing when gemma-4's tensor is present. Nothing
//! here needs that, because a GGUF declares `general.architecture` in its
//! key-value block. The dispatch is a string match on a fact the file states
//! about itself.
//!
//! # Five arms, and the others refusing
//!
//! `qwen2`, `qwen3`, `qwen3moe`, `llama` and `gemma3` are here. That the rest
//! are missing is not a stub: they are REFUSED, by name, with the reason --
//! which is the same shape the family contract table already uses ("no MLX
//! authoring pass exists..."), and it is an honest answer where a silent
//! partial map would not be. Adding an architecture is a module in its
//! generation and an arm here.
//!
//! # Why an arm answers five questions and not one
//!
//! A rename was enough for `qwen2` and `qwen3`, and both were CHECKED to be:
//! a BF16 GGUF and the safetensors release it came from agree bit for bit.
//! The others prove a rename is not the general case, and they fail
//! differently. `llama` reorders the rows of Q and K and publishes a
//! `rope_freqs.weight` that is not a weight. `gemma3` publishes every norm as
//! `w + 1`, because llama.cpp folds the constant its kernel would otherwise
//! add and pie's kernel adds its own. `qwen3moe` publishes its experts
//! JOINED: one `[E, I, H]` tensor where the safetensors release has `E`
//! separate ones.
//!
//! So a pass states a name, which tensors the converter DERIVED, which rows
//! regroup, which constants were folded, and which tensors are a stack of
//! many -- all five together, because a family that answered one and was
//! forgotten by another is a model that loads, serves, and answers slightly
//! wrong. Every one of these failures is invisible to a shape check, which is
//! the only check that runs.
//!
//! # An architecture that is refused for a measured reason: `gpt-oss`
//!
//! Not every GGUF can be ingested, and the one that cannot is worth naming
//! here because it looks like it can. gpt-oss's GGUF is a mixture like
//! `qwen3moe`, and the arm it would need is not the same arm.
//!
//! Measured on `ggml-org/gpt-oss-20b-GGUF`, against the artifact the
//! published safetensors import to:
//!
//! * The experts are MXFP4, and llama.cpp's MXFP4 block is **self-contained**
//!   -- 17 bytes holding one E8M0 scale byte and 16 bytes of codes. The
//!   safetensors release splits the same numbers into a `_blocks` tensor of
//!   16-byte rows and a `_scales` tensor of single bytes, which is the pair
//!   the contract in `crate::gpt_oss` declares. Getting from one to the other
//!   means addressing INSIDE a block, and the placement algebra refuses that
//!   by construction: see `ByteScale::Blocked` in
//!   `model_loader::contract::compile`, where only whole blocks have
//!   addresses because half a block is codes with no scale to read them by.
//! * GGUF keeps `ffn_gate_exps` and `ffn_up_exps` apart where the checkpoint
//!   fuses them into one `gate_up_proj` with gate on even rows and up on odd.
//!   That half IS within reach -- a destination row is 90 blocks of 16 bytes
//!   laid end to end, so the interleave is 32 x 5760 runs and stays under
//!   `MAX_RUNS`. It never gets its turn, because the split above comes first.
//! * That file also requantizes attention and the embeddings to Q8_0, so it
//!   is not a lossless carrier of the checkpoint pie can already import.
//!
//! ## What it would actually take, since "refused by the algebra" misleads
//!
//! The first bullet is about the AFFINE fragment, and reads as though the
//! algebra were short a node. It is not, and someone who sets out to add one
//! will find the node already written.
//!
//! `Expr::Repack` is precisely this case -- the table in
//! `model_loader::contract` files it as **placement priced as a kernel**, may
//! pad, and may not reinterpret an element, which a block split does not. The
//! cost ladder is therefore behaving correctly rather than obstructing:
//! `Transmute` to bytes and `Slice` the 17 apart WOULD denote the right
//! tensors, and `infer` is right to push it off the free row, because as a
//! placement it is 16.6 M runs against a `MAX_RUNS` of 1 M. That number is
//! not a budget to raise. The innermost axis is 17 bytes and the cut is
//! inside it, so contiguity breaks at every block and the run list is the
//! size of the data. A transform whose run list is O(data) is not a
//! placement; run lists are how the executor says "memcpy".
//!
//! So the missing piece is not algebra but an IMPLEMENTATION of the node.
//! `TILE_MAP_REPACK` is set in no mask in `model_loader::plan::passes::tile`
//! -- not `HOST_TILE_MAP_MASK`, not `CONVERT_TILE_MAP_MASK`, and not CUDA's,
//! whose doc records dropping the claim because advertising a transform with
//! no implementation made plans compile and then die at execution naming only
//! a kind. `RepackLayout::MarlinMxfp4Weight` and its scale twin exist as
//! types with no executor behind them; the kernels were the deleted C++
//! `transcode_engine.hpp`'s and were never ported. `walk.rs` says it plainly:
//! `Repack` is the host executor's standing refusal, and an import runs on
//! the host.
//!
//! That is the honest size of it: a host Repack kernel and a wider mask, in
//! the loader, for a node the tree deliberately does not implement.
//!
//! ## And the carrier is why it would not pay
//!
//! Four gpt-oss GGUFs were checked by reading their headers over range
//! requests, against pie's own `openai--gpt-oss-20b.zt`, which holds every
//! attention projection, `embed_tokens` and `lm_head` at BF16:
//!
//! | file | the 98 non-expert weights |
//! |------|---------------------------|
//! | `ggml-org/gpt-oss-20b-GGUF` MXFP4 | Q8_0 |
//! | `bartowski/...-MXFP4-Experimental` | Q8_0 |
//! | `lmstudio-community/gpt-oss-20b-GGUF` | Q8_0 |
//! | `unsloth/gpt-oss-20b-GGUF` F16 | F16 |
//!
//! Every one of them carries the experts as MXFP4 and downgrades everything
//! else -- `token_embd.weight` and `output.weight` included. There is no
//! lossless carrier to import, so the kernel above would be built to produce
//! a WORSE artifact than `pie model import openai/gpt-oss-20b` already
//! produces. That, and not the algebra, is the reason to leave this alone.
//!
//! ## The way around the kernel, and why it is worse still
//!
//! There is a second route, and it is much cheaper than a host Repack: do
//! not split the block, DECODE it. `walk.rs` already decodes eight GGUF
//! schemes, and a 17-byte MXFP4 block -- one E8M0 exponent, then 32 E2M1
//! nibbles -- is structurally the smallest of them. The nibble table and the
//! exponent are already in `model_loader::codec`, and MXFP4 to BF16 is
//! exact, so it would be a decoder and no arithmetic risk.
//!
//! It is closed today by one line rather than by a policy:
//! `QuantScheme::Mxfp4E2M1E8M0` answers `None` to `block_layout`, because
//! inside pie MXFP4 is not an opaque block at all -- it is the split
//! `_blocks`/`_scales` pair, which is why the safetensors release is CARRIED
//! rather than decoded. Opening it means a distinct scheme, not a layout on
//! that one.
//!
//! The reason not to is a size. Measured on `openai--gpt-oss-20b.zt`:
//!
//! | | packed | decoded to BF16 |
//! |---|--------|-----------------|
//! | expert `_blocks` + `_scales` | 9.46 GiB | 35.60 GiB |
//! | everything else | 3.37 GiB | 3.37 GiB |
//! | artifact | **12.82 GiB** | **38.97 GiB** |
//!
//! Three times the artifact, to hold the same numbers, with the 98 weights
//! around them at Q8_0 instead of BF16. So the cheap route loses on BOTH
//! counts the expensive one only lost on one of -- and pie already writes
//! the 12.82 GiB column from the HuggingFace release. A GGUF gpt-oss saves
//! about 26 GiB of download and spends it back on disk three times over.
//!
//! Which is why there is no `gpt-oss` arm. `qwen3moe` needed a new `Ingest`
//! shape too, and got it, because a slab of a BF16 stack is a contiguous run
//! -- it stayed on the free row, where this cannot.
//!
//! # An architecture refused for the opposite reason: `gemma4`
//!
//! `gpt-oss` is refused because its weights cannot be reached. `gemma4` is
//! refused although its weights are the EASIEST of any file measured here --
//! which is why the reason has to be written down, or the next reader will
//! see how short the map is and write it.
//!
//! Measured on `unsloth/gemma-4-26B-A4B-it-GGUF` BF16, range-read against
//! `unsloth/gemma-4-26B-A4B-it` at the same tensors:
//!
//! * The text tower is a **pure rename**. `attn_q`, `ffn_gate_up_exps` and
//!   `ffn_down_exps` are byte-identical to `self_attn.q_proj.weight`,
//!   `experts.gate_up_proj` and `experts.down_proj`. The mixture arrives
//!   already fused as `[E, 2I, H]` in the checkpoint's own order, so this
//!   family does not even need [`Ingest::Unstack`] that `qwen3moe` did.
//! * **Nothing is folded into the norms.** `attn_norm`, `ffn_norm`,
//!   `post_ffw_norm_2` and `output_norm` all differ from their HuggingFace
//!   counterparts by exactly `0.0`, where `gemma3` differs by exactly `1.0`.
//!   Both files widen to F32 what the checkpoint holds as BF16 and change no
//!   value. This agrees with the forward pass -- gemma-4 norms are
//!   `NormVariant::Plain` where gemma-3's are `NormVariant::Gemma` -- so
//!   there is no constant for a converter to fold. Copying `gemma3`'s
//!   [`crate::gemma_3::import::folded_constant`] into a gemma-4 arm on the
//!   strength of the shared family name would be wrong by one everywhere,
//!   which is the exact failure that function's own documentation exists to
//!   prevent, running backwards.
//!
//! What refuses it is the tower. Every gemma-4 row this build ships states
//! `vision: Some(..)`, and the E-series states an audio tower besides;
//! llama.cpp puts neither in the model file. They live in a SEPARATE
//! `mmproj-*.gguf` whose `general.architecture` is `clip` and whose
//! `general.type` is `mmproj` -- 1411 tensors for `gemma-4-E2B`, under an `a.`
//! and `v.` naming scheme this module has never seen, carrying per-tensor
//! `input_max`/`output_min` calibration scalars that no HuggingFace
//! checkpoint has a name for at all.
//!
//! So an arm here would import cleanly and produce an artifact that matches
//! no row, every time, because the half of the model the row asks for is in a
//! file this pass was not given. Refusing at the door says that once. The
//! text map above is the head start for whoever brings the tower.

use model_loader::checkpoint::Attributes;
use model_loader::error::Error;

/// What one GGUF tensor becomes in the artifact.
///
/// Five answers and not one, because llama.cpp writes tensors that are
/// neither a weight under another name nor a weight pie can use, writes
/// others whose values are its own kernel's rather than pie's, and joins
/// tensors the artifact holds apart.
#[derive(Clone, Debug, PartialEq)]
pub enum Ingest {
    /// The same bytes under the artifact's name.
    Rename(String),
    /// The artifact's name, and rows regrouped into pie's order first.
    ///
    /// `heads` is how many equal row groups the tensor divides into, each of
    /// which is regrouped on its own. The transform itself is the loader's --
    /// this says which tensors need it and at what granularity, which is the
    /// only part that is a fact about the family.
    Unpermute { name: String, heads: u32 },
    /// The artifact's name, and a constant the converter folded in taken back
    /// out: the artifact holds `stored + by`.
    ///
    /// `by` rather than a flag because the family states the constant. Gemma's
    /// is `-1.0` and nothing else in this crate needs one, but a number that
    /// is named can be checked against a file and a flag cannot.
    Debias { name: String, by: f32 },
    /// One stacked tensor taken apart into the per-instance tensors the
    /// artifact holds, one for each index along the leading axis.
    ///
    /// `each` is a name template with a single `{}` for that index. A
    /// template rather than a list because the count is the SOURCE's leading
    /// extent, which the family does not know and the file states: a map that
    /// carried its own expert count would be a second answer to a question
    /// the checkpoint has already answered, and the two can disagree.
    ///
    /// llama.cpp stacks a mixture's experts -- `blk.3.ffn_gate_exps.weight`
    /// is `[E, I, H]` where HuggingFace publishes `E` separate `[I, H]`
    /// tensors. Taking them apart rather than re-fusing them is what makes
    /// the artifact the same artifact the safetensors release imports to,
    /// and the fused form the contract wants is a join it already builds.
    Unstack { each: String },
    /// Not a weight: the converter computed it, and pie computes its own.
    ///
    /// Distinct from an unmapped name, which is an error. This one is a
    /// decision, and dropping it is the whole of it.
    Drop,
}

impl Ingest {
    /// The artifact name, or `None` for a tensor that does not reach the
    /// artifact under one.
    ///
    /// `None` covers both a tensor that reaches no name and one that reaches
    /// many: an unstacked tensor's template is not a name, and handing it out
    /// as one would put a literal `{}` in an artifact.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Rename(name) | Self::Unpermute { name, .. } | Self::Debias { name, .. } => {
                Some(name)
            }
            Self::Unstack { .. } | Self::Drop => None,
        }
    }
}

/// One family's ingest pass, as the five questions the dispatch asks it.
///
/// A struct of function pointers rather than five matches on `architecture`,
/// so a family cannot answer one question and be forgotten by another --
/// which would mean a tensor renamed but not regrouped, or renamed but not
/// unfolded, and a model that serves slightly wrong answers.
struct Pass {
    /// The artifact's name for a GGUF name, or `None` if the map has none.
    name: fn(&str) -> Option<String>,
    /// Tensors the converter derived, which pie derives for itself.
    derived: fn(&str) -> bool,
    /// How many row groups a tensor regroups by, or `None` for rows already
    /// in pie's order.
    regroup: fn(&Attributes, &str) -> Option<u32>,
    /// The constant the converter folded into a tensor, to be taken back out,
    /// or `None` for a tensor whose values are already pie's.
    folded: fn(&str) -> Option<f32>,
    /// Whether the name is one tensor here and many in the artifact, so its
    /// mapped name is a `{}` template rather than a name.
    stacked: fn(&str) -> bool,
}

fn pass_for(architecture: &str) -> Result<Pass, Error> {
    match architecture {
        "qwen2" => Ok(Pass {
            name: crate::qwen_2::import::hf_name,
            derived: |_| false,
            regroup: |_, _| None,
            folded: |_| None,
            stacked: |_| false,
        }),
        "qwen3" => Ok(Pass {
            name: crate::qwen_3::import::hf_name,
            derived: |_| false,
            regroup: |_, _| None,
            folded: |_| None,
            stacked: |_| false,
        }),
        "qwen3moe" => Ok(Pass {
            name: crate::qwen_3::import_moe::hf_name,
            derived: |_| false,
            regroup: |_, _| None,
            folded: |_| None,
            stacked: crate::qwen_3::import_moe::is_stacked,
        }),
        "gemma3" => Ok(Pass {
            name: crate::gemma_3::import::hf_name,
            derived: |_| false,
            regroup: |_, _| None,
            folded: crate::gemma_3::import::folded_constant,
            stacked: |_| false,
        }),
        "llama" => Ok(Pass {
            name: crate::llama_2::import::hf_name,
            derived: crate::llama_2::import::is_derived,
            regroup: crate::llama_2::import::regroup_heads,
            folded: |_| None,
            stacked: |_| false,
        }),
        other => Err(Error::Contract(format!(
            "this is a `{other}` GGUF, and pie has no GGUF ingest pass for it. \
             Import the HuggingFace release instead, or add \
             `crates/model/src/<generation>/import.rs` and an arm in \
             `crates/model/src/ingest.rs`"
        ))),
    }
}

/// This checkpoint's tensors, as the artifact will hold them.
///
/// Takes the whole key-value block rather than the architecture string alone
/// because a regrouping is stated in head counts, and those are facts the
/// file carries about itself. A family that needs none simply does not read
/// them.
///
/// # Errors
///
/// No pass for this architecture, or a tensor the pass has no name for.
pub fn gguf_ingest(attributes: &Attributes, names: &[&str]) -> Result<Vec<Ingest>, Error> {
    let architecture = attributes.architecture().unwrap_or_default();
    let pass = pass_for(architecture)?;
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        if (pass.derived)(name) {
            out.push(Ingest::Drop);
            continue;
        }
        let Some(renamed) = (pass.name)(name) else {
            return Err(Error::Contract(format!(
                "`{architecture}` GGUF ingest has no name for `{name}`; the map in \
                 `crates/model/src/*/import.rs` predates this checkpoint"
            )));
        };
        let regroup = (pass.regroup)(attributes, name);
        let folded = (pass.folded)(name);
        let stacked = (pass.stacked)(name);
        // No family needs two, and the enum cannot say two. Refusing is the
        // difference between finding that out here, by name, and finding it
        // out as a tensor that was silently only half converted.
        if u8::from(regroup.is_some()) + u8::from(folded.is_some()) + u8::from(stacked) > 1 {
            return Err(Error::Contract(format!(
                "`{architecture}` GGUF ingest wants more than one of regroup, unfold and \
                 unstack for `{name}`; `Ingest` states one transform per tensor, so this \
                 needs a composition the enum does not have"
            )));
        }
        out.push(match (regroup, folded, stacked) {
            (Some(heads), _, _) => Ingest::Unpermute {
                name: renamed,
                heads,
            },
            (_, Some(by), _) => Ingest::Debias { name: renamed, by },
            (_, _, true) => Ingest::Unstack { each: renamed },
            _ => Ingest::Rename(renamed),
        });
    }
    Ok(out)
}

/// This checkpoint's tensors, renamed into the artifact's vocabulary.
///
/// Returns one entry per input name, in the order given, so a caller can zip
/// it against whatever it holds them in. A name the family's map does not
/// know is an error rather than a passthrough: an unrenamed `blk.3.attn_q`
/// beside a renamed one is an artifact with a hole exactly where the map ran
/// out, and the catalog would report it as a missing tensor without ever
/// saying that the map was the reason.
pub fn gguf_rename(architecture: &str, names: &[&str]) -> Result<Vec<String>, Error> {
    let attributes = Attributes::from_pairs([(
        "general.architecture".to_string(),
        model_loader::checkpoint::Attribute::Text(architecture.to_string()),
    )]);
    gguf_ingest(&attributes, names)?
        .into_iter()
        .map(|ingest| match ingest {
            Ingest::Rename(name) | Ingest::Unpermute { name, .. } | Ingest::Debias { name, .. } => {
                Ok(name)
            }
            Ingest::Unstack { each } => Err(Error::Contract(format!(
                "`{each}` is a template, not a name: this tensor reaches the artifact as \
                 one per instance. Ask `gguf_ingest`"
            ))),
            Ingest::Drop => Err(Error::Contract(
                "a dropped tensor has no name; ask `gguf_ingest`".to_string(),
            )),
        })
        .collect()
}

/// Whether this build can ingest a GGUF written for `architecture`.
///
/// So a caller can say what will happen before it does 12 GB of work.
#[must_use]
pub fn can_ingest_gguf(architecture: &str) -> bool {
    gguf_rename(architecture, &[]).is_ok()
}

#[cfg(test)]
mod tests {
    use super::{Ingest, can_ingest_gguf, gguf_ingest, gguf_rename};
    use model_loader::checkpoint::Attributes;

    #[test]
    fn a_qwen2_gguf_is_renamed_in_the_order_it_was_given() {
        let names = [
            "token_embd.weight",
            "blk.0.attn_q.bias",
            "output_norm.weight",
        ];
        assert_eq!(
            gguf_rename("qwen2", &names).unwrap(),
            [
                "model.embed_tokens.weight",
                "model.layers.0.self_attn.q_proj.bias",
                "model.norm.weight",
            ]
        );
    }

    /// An architecture with no pass is named in the refusal.
    #[test]
    fn an_architecture_with_no_pass_is_refused_by_name() {
        let why = gguf_rename("phi3", &["token_embd.weight"])
            .unwrap_err()
            .to_string();
        assert!(why.contains("`phi3`"), "{why}");
        assert!(!can_ingest_gguf("phi3"));
        for supported in ["qwen2", "qwen3", "qwen3moe", "llama", "gemma3"] {
            assert!(can_ingest_gguf(supported), "{supported}");
        }
    }

    /// gpt-oss is refused, and that is a decision with a measurement behind
    /// it rather than a gap.
    ///
    /// Its GGUF holds MXFP4 experts as self-contained 17-byte blocks where
    /// the safetensors release splits the same numbers into `_blocks` and
    /// `_scales`. Only whole blocks have byte addresses -- see
    /// `ByteScale::Blocked` -- so the split is outside the placement algebra
    /// by construction, and no name map can stand in for it. A rename-only
    /// arm would pass every shape check and read codes as though they were
    /// the scales that price them.
    #[test]
    fn gpt_oss_is_refused_because_its_blocks_cannot_be_taken_apart() {
        assert!(!can_ingest_gguf("gpt-oss"));
        let why = gguf_rename("gpt-oss", &["blk.0.ffn_gate_exps.weight"])
            .unwrap_err()
            .to_string();
        assert!(why.contains("`gpt-oss`"), "{why}");
    }

    /// The refusal above rests on a loader fact, so the loader is asked.
    ///
    /// The module doc says the missing piece for gpt-oss is not a node in the
    /// algebra -- `Expr::Repack` already denotes "placement priced as a
    /// kernel" -- but an implementation of that node, which nothing in this
    /// tree has. That is a claim about `model_loader`, written in `model`,
    /// and the two crates are free to move apart.
    ///
    /// So it is checked rather than asserted in prose. The day a host Repack
    /// lands, this fails, and the paragraph explaining why gpt-oss cannot be
    /// ingested is the thing that needs rereading -- the carrier measurement
    /// beneath it would then be the whole of the answer.
    ///
    /// `TILE_MAP_REBLOCK` rides along as the negative control: the host does
    /// implement that one, so a mask of zero, or a constant that stopped
    /// naming anything, cannot pass this by accident.
    #[test]
    fn no_target_advertises_the_repack_this_refusal_rests_on() {
        use model_loader::plan::{
            CONVERT_TILE_MAP_MASK, HOST_TILE_MAP_MASK, TILE_MAP_REBLOCK, TILE_MAP_REPACK,
        };

        assert_eq!(
            HOST_TILE_MAP_MASK & TILE_MAP_REPACK,
            0,
            "the host executor implements Repack now; re-read the gpt-oss section"
        );
        assert_eq!(
            CONVERT_TILE_MAP_MASK & TILE_MAP_REPACK,
            0,
            "an import may carry a Repack now; re-read the gpt-oss section"
        );
        assert_ne!(
            CONVERT_TILE_MAP_MASK & TILE_MAP_REBLOCK,
            0,
            "the control: a convert does carry Reblock, so these masks do name transforms"
        );
    }

    /// And the cheap way around it is shut too, by a `None`.
    ///
    /// The other half of the same paragraph. Decoding an MXFP4 block instead
    /// of splitting it needs no Repack at all -- only a block layout and a
    /// decoder, next to the eight `walk.rs` already has. What stops it is
    /// that `Mxfp4E2M1E8M0` reports no block layout, because in pie MXFP4 is
    /// the split `_blocks`/`_scales` pair rather than an opaque block.
    ///
    /// That is a `None` and not a refusal, so nothing would complain if it
    /// became a `Some`. This is the complaint. If it fires, the thing to
    /// re-read is the size table: decoding trades 12.82 GiB of artifact for
    /// 38.97 GiB holding the same numbers.
    ///
    /// `GgufQ8_0` is the negative control -- the scheme this file's own
    /// measurement found gpt-oss GGUFs demoting everything else to, and one
    /// the loader does decode, so a `block_layout` that had stopped
    /// answering at all cannot pass this by accident.
    #[test]
    fn mxfp4_is_not_a_block_the_loader_can_decode_either() {
        use model_loader::types::QuantScheme;

        assert_eq!(
            QuantScheme::Mxfp4E2M1E8M0.block_layout(),
            None,
            "MXFP4 decodes as a GGUF block now; re-read the gpt-oss section, \
             which prices that route at 3x the artifact"
        );
        assert_eq!(
            QuantScheme::GgufQ8_0.block_layout(),
            Some((32, 34)),
            "the control: the loader does describe blocks, so this is MXFP4's \
             own answer and not an empty table"
        );
    }

    /// gemma4 is refused too, and for a reason nothing about its weights
    /// would suggest.    ///
    /// Its text tower is the shortest map any file measured here would need:
    /// a pure rename, experts already fused as `[E, 2I, H]`, and -- unlike
    /// `gemma3` -- no constant folded into any norm. What refuses it is that
    /// every gemma-4 row states a vision tower and llama.cpp ships none in
    /// the model file. The tower is a separate `mmproj` GGUF whose
    /// architecture is `clip`. An arm would import cleanly and match no row,
    /// every time, so the refusal is at the door and says so.
    #[test]
    fn gemma4_is_refused_because_its_towers_are_in_another_file() {
        assert!(!can_ingest_gguf("gemma4"));
        assert!(can_ingest_gguf("gemma3"), "not a typo for the arm that is");
        let why = gguf_rename("gemma4", &["blk.0.attn_q.weight"])
            .unwrap_err()
            .to_string();
        assert!(why.contains("`gemma4`"), "{why}");
    }

    /// A qwen3moe GGUF renames what it can and unstacks its experts.
    ///
    /// The router is in this list next to the three expert tensors on
    /// purpose: `ffn_gate_inp` leads with the expert count exactly as they
    /// do, and is one tensor on both sides. It is the tensor a rule written
    /// against extents rather than names would take apart.
    #[test]
    fn a_qwen3moe_gguf_renames_its_weights_and_unstacks_its_experts() {
        let attributes = Attributes::from_pairs([(
            "general.architecture".to_string(),
            model_loader::checkpoint::Attribute::Text("qwen3moe".to_string()),
        )]);
        let got = gguf_ingest(
            &attributes,
            &[
                "blk.0.attn_q.weight",
                "blk.0.ffn_gate_inp.weight",
                "blk.0.ffn_gate_exps.weight",
                "blk.0.ffn_up_exps.weight",
                "blk.0.ffn_down_exps.weight",
                "output_norm.weight",
            ],
        )
        .unwrap();
        assert_eq!(
            got,
            [
                Ingest::Rename("model.layers.0.self_attn.q_proj.weight".into()),
                Ingest::Rename("model.layers.0.mlp.gate.weight".into()),
                Ingest::Unstack {
                    each: "model.layers.0.mlp.experts.{}.gate_proj.weight".into()
                },
                Ingest::Unstack {
                    each: "model.layers.0.mlp.experts.{}.up_proj.weight".into()
                },
                Ingest::Unstack {
                    each: "model.layers.0.mlp.experts.{}.down_proj.weight".into()
                },
                Ingest::Rename("model.norm.weight".into()),
            ]
        );
        // A template is not a name, and asking for one says so rather than
        // handing back a string with a `{}` in it.
        assert!(gguf_rename("qwen3moe", &["blk.0.ffn_up_exps.weight"]).is_err());
    }

    /// A gemma3 GGUF renames its weights and unfolds its norms.
    ///
    /// The pair in this list is the whole point: two tensors of the same
    /// generation, one of which carries a constant llama.cpp added and one of
    /// which does not. A pass that answered `Rename` for both would produce a
    /// model that is wrong by one in every norm, and nothing downstream --
    /// shape identification, the build, the bind table -- can see that.
    #[test]
    fn a_gemma3_gguf_renames_its_weights_and_unfolds_its_norms() {
        let attributes = Attributes::from_pairs([(
            "general.architecture".to_string(),
            model_loader::checkpoint::Attribute::Text("gemma3".to_string()),
        )]);
        let got = gguf_ingest(
            &attributes,
            &["blk.0.attn_q.weight", "blk.0.ffn_norm.weight"],
        )
        .unwrap();
        assert_eq!(
            got,
            [
                Ingest::Rename("model.layers.0.self_attn.q_proj.weight".to_string()),
                Ingest::Debias {
                    name: "model.layers.0.pre_feedforward_layernorm.weight".to_string(),
                    by: -1.0,
                },
            ]
        );
    }

    /// A llama GGUF renames, regroups and drops in one pass.
    ///
    /// All three outcomes in one call, in the order given, because the
    /// caller zips this against the tensors it holds and a pass that
    /// silently shortened the list would misalign every entry after the
    /// first drop.
    #[test]
    fn a_llama_gguf_renames_regroups_and_drops_in_order() {
        let attributes = Attributes::from_pairs([
            (
                "general.architecture".to_string(),
                model_loader::checkpoint::Attribute::Text("llama".to_string()),
            ),
            (
                "llama.attention.head_count".to_string(),
                model_loader::checkpoint::Attribute::Uint(32),
            ),
            (
                "llama.attention.head_count_kv".to_string(),
                model_loader::checkpoint::Attribute::Uint(8),
            ),
        ]);
        let names = [
            "token_embd.weight",
            "blk.0.attn_q.weight",
            "rope_freqs.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
        ];
        assert_eq!(
            gguf_ingest(&attributes, &names).unwrap(),
            [
                Ingest::Rename("model.embed_tokens.weight".to_string()),
                Ingest::Unpermute {
                    name: "model.layers.0.self_attn.q_proj.weight".to_string(),
                    heads: 32
                },
                Ingest::Drop,
                Ingest::Unpermute {
                    name: "model.layers.0.self_attn.k_proj.weight".to_string(),
                    heads: 8
                },
                Ingest::Rename("model.layers.0.self_attn.v_proj.weight".to_string()),
            ]
        );
    }

    /// A tensor the map does not know stops the import and names itself.
    #[test]
    fn an_unmapped_tensor_is_an_error_and_not_a_passthrough() {
        let why = gguf_rename("qwen2", &["blk.0.ffn_gate_inp.weight"])
            .unwrap_err()
            .to_string();
        assert!(why.contains("ffn_gate_inp"), "{why}");
    }
}
