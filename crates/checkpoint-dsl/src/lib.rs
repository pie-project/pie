//! The checkpoint-provenance authoring eDSL over the typed load contract —
//! the edge `model-dsl` is to `model-ir`, on the supply side. A family's
//! import calls [`Builder`]'s four verbs, each line is checked against the
//! open source as it is said, and [`Builder::build`] is the
//! [`ModelContract`] that `checkpoint`'s `infer`/`compile`/`executor` verify
//! and run. This crate knows no model family and traces no forward pass.
//!
//! WHAT A WEIGHT IS IS NOT SAID HERE. [`Weight`] is the noun the two
//! authoring surfaces share — the forward text declares one and the recorder
//! interns it into `Trace::params`; this crate only reads one. And the
//! contract language itself is `checkpoint`'s: [`Expr`] and [`ModelContract`]
//! are the typed IR, this crate is the pen that writes in it.

use checkpoint::contract::{Expr, ModelContract, Scales, TensorContract, TensorType};
use checkpoint::types::{
    Axis, DType, Encoding, QuantGranularity, QuantScheme, QuantSpec, RepackLayout, ScaleForm,
    TILED_BAND,
};
use model_dsl::{Dtype, Shard, Weight};

/// Why a read refused: the checkpoint lacks the name, states it in terms no
/// reader here can name, or holds it in a representation the declared one is
/// not decoded from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    Missing(String),

    Illegible {
        name: String,
        detail: String,
    },

    Incompatible {
        name: String,
        stored: Encoding,
        want: Encoding,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Missing(name) => write!(
                f,
                "this model reads a plane called `{name}` and the checkpoint \
                 holds no tensor under that name"
            ),
            Self::Illegible { name, detail } => write!(
                f,
                "`{name}`: this checkpoint is stated in terms no reader here \
                 can name ({detail})"
            ),
            Self::Incompatible { name, stored, want } => write!(
                f,
                "`{name}` is stored {stored:?} and this model wants {want:?}; \
                 one quantization is not decoded into another on the way in"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// One checkpoint being read into one [`ModelContract`].
///
/// Each `read*` line states where one declared weight's bytes come from and
/// is checked against the open source as it is said — a missing name, an
/// illegible representation or a refused conversion faults on the line that
/// asked, not four stages later. [`build`](Builder::build) is what the lines
/// add up to.
pub struct Builder<'a> {
    src: &'a ztensor::Source,
    tp: u32,
    tensors: Vec<TensorContract>,
}

impl<'a> Builder<'a> {
    /// Over an open checkpoint, for a deployment `tp` ranks wide.
    ///
    /// `tp` feeds [`read_own`](Builder::read_own) alone — a native artifact
    /// stores whole weights and each rank reads its band. The other verbs
    /// read foreign checkpoints, which are imported whole: they refuse
    /// `tp > 1`.
    ///
    /// "The other verbs" is a statement about what a verb DOES and not about
    /// which one was called. Each of them first asks
    /// [`holds_the_landed_plane`](Builder::holds_the_landed_plane), and a
    /// file that already holds the weight under its own name is read own —
    /// so one `Builder` reads a foreign checkpoint and a native artifact in
    /// the same pass, weight by weight, which is exactly what an artifact
    /// holding SOME landed planes and copying the rest through requires.
    #[must_use]
    pub fn new(src: &'a ztensor::Source, tp: u32) -> Builder<'a> {
        Builder {
            src,
            tp,
            tensors: Vec::new(),
        }
    }

    /// Read `w` from the checkpoint tensor named `from`.
    ///
    /// **A FAMILY IMPORT NAMES A WEIGHT ONCE, WHATEVER IT IS STORED AS.** A
    /// bf16 projection is one stored tensor; an MLX affine-U4 projection is
    /// three — `q_proj.weight` holding the codes eight to a `u32` word,
    /// `q_proj.scales` and `q_proj.biases` holding one bf16 each per
    /// sixty-four codes — and an import that had to know which case it was in
    /// would be two imports pretending to be one. So the call site says the
    /// logical name and this verb says how many tensors that is.
    ///
    /// **THE TRIPLET IS REQUIRED, NOT PREFERRED.** A weight declared `U4g64`
    /// against a checkpoint that ships bf16 could instead let the LOADER
    /// encode — which is a real and wanted path through `read_own`, and a
    /// disaster through `models::identify`: the u4 SKU would claim every bf16
    /// checkpoint of its own family, and being listed ahead of the bf16 row
    /// (which it must be, see `qwen_3::IMPORTS`) it would claim it first. An
    /// import states what the FILE holds, so a missing `.scales` is a miss
    /// and the next row gets its turn.
    pub fn read(&mut self, w: &Weight, from: impl Into<String>) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = planes(self.src, w, from)?;
        self.tensors.extend(read);
        Ok(())
    }

    /// Read `w` from several checkpoint tensors, concatenated — a bank this
    /// text states whole and the checkpoint ships in parts: a gate and an up,
    /// a q and a k and a v.
    ///
    /// The parts join on the weight's own cut axis, which is the axis they
    /// were split along. For a packed representation each part carries its
    /// own triplet and the companions join at the same seams, because a group
    /// belongs to the row it scales and rows are what a pack seam separates.
    pub fn read_concat(
        &mut self,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
    ) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = planes_fused(self.src, w, parts)?;
        self.tensors.extend(read);
        Ok(())
    }

    /// Read `w` from one MLX affine triplet and LAND IT DENSE: the codes
    /// scaled and offset per group at load, so what the device holds is the
    /// weight's declared dtype and no kernel needs an affine point.
    ///
    /// The one verb whose weight and file genuinely disagree, which is why
    /// `stored` is an argument: `read` asks the weight because a text
    /// declares what the device holds, and here the device holds bf16 while
    /// the file holds a triplet — so the import states the file's own width,
    /// as an import states everything about the file. The decode is IN THE
    /// PLAN — a per-block [`Expr::Scale`] over the codes and a per-block
    /// [`Expr::Bias`] over its answer, with the companions declared internal
    /// — rather than a host loop outside it, which is the thing the contract
    /// algebra exists to stop.
    pub fn read_dequant(
        &mut self,
        w: &Weight,
        from: impl Into<String>,
        stored: Dtype,
    ) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = dequant_planes(self.src, w, vec![from.into()], stored)?;
        self.tensors.extend(read);
        Ok(())
    }

    /// [`read_dequant`](Builder::read_dequant) over parts, joined at the
    /// weight's declared seams — [`read_concat`](Builder::read_concat)'s
    /// rule, landed dense.
    pub fn read_dequant_concat(
        &mut self,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
        stored: Dtype,
    ) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = dequant_planes(self.src, w, parts.into_iter().collect(), stored)?;
        self.tensors.extend(read);
        Ok(())
    }

    /// **READ AN MLX AFFINE TRIPLET AND RELAY IT** (§J4b) — the same three
    /// planes [`read`](Builder::read) binds, put into the m16n8k16 fragment
    /// order `kernels_cuda::linear::tiled` reads before they are written.
    ///
    /// The one verb that states an [`Expr::Repack`], and it is an IMPORT
    /// verb: no device mask admits one, so a contract carrying this compiles
    /// at the conversion target and is refused, with the layout and the
    /// command named, on any serving load that reaches it. What runs it is
    /// `pie model import`, once per weight.
    ///
    /// `w` must be declared [`Dtype::U4g64tiled`] — the declaration is what
    /// says which order the artifact will hold, and stating a repack over a
    /// weight declared row-major would write bytes no reader of that
    /// declaration can read.
    ///
    /// # Errors
    ///
    /// [`read`](Builder::read)'s, plus a weight not declared tiled.
    pub fn read_repack(&mut self, w: &Weight, from: impl Into<String>) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = tiled_planes(self.src, w, vec![from.into()])?;
        self.tensors.extend(read);
        Ok(())
    }

    /// [`read_repack`](Builder::read_repack) over parts, joined at the
    /// weight's declared seams before the relabelling —
    /// [`read_concat`](Builder::read_concat)'s rule, relaid.
    ///
    /// The join comes FIRST and the repack second, which is the only order
    /// that means anything: the fragment map is a function of the whole
    /// bank's column count, so repacking two legs and concatenating the
    /// answers would interleave two lane maps.
    ///
    /// # Errors
    ///
    /// [`read_repack`](Builder::read_repack)'s.
    pub fn read_repack_concat(
        &mut self,
        w: &Weight,
        parts: impl IntoIterator<Item = String>,
    ) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = tiled_planes(self.src, w, parts.into_iter().collect())?;
        self.tensors.extend(read);
        Ok(())
    }

    /// Read `w` via a stated expression — the door for the reads with
    /// arithmetic in them: a fold taken back out with [`Expr::bias`], a
    /// stored rank squeezed away, a conv kernel transmuted to the matmul bank
    /// it already is.
    ///
    /// Stated, not trusted: every source the expression names must exist and
    /// agree on one stored representation, and the conversion between stored
    /// and declared walks the same ladder [`read`](Builder::read) walks.
    pub fn read_expr(&mut self, w: &Weight, expr: Expr) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = declare(self.src, w, expr)?;
        self.tensors.push(read);
        Ok(())
    }

    /// [`read_expr`](Builder::read_expr), for an expression that must ASK THE
    /// CHECKPOINT before it can be written down.
    ///
    /// **EVERY OTHER VERB HERE CONSULTS
    /// [`holds_the_landed_plane`](Builder::holds_the_landed_plane) FIRST, AND
    /// `read_expr` CONSULTS IT TOO — ONE STEP TOO LATE.** It takes an `Expr`
    /// already made, so the making happens at the CALL SITE, which is before
    /// this builder is reached. For an expression assembled out of names and
    /// constants that costs nothing and the ordering never shows. For one
    /// that reads the source to decide its own shape it is fatal: the helper
    /// returns [`Error::Missing`] for a plane that is not under the source's
    /// spelling, the `?` fires at the call site, and the import fails on a
    /// weight whose landed plane was sitting right there.
    ///
    /// Two helpers in the model texts are of that kind and four call sites
    /// use them — `squeezed`, which asks which of the two depthwise conv1d
    /// spellings a file uses, and `flattened`, which reads the stored extents
    /// to check an element count. Each one takes `&Source` and each one can
    /// answer `Missing`, which is exactly the answer a promoted artifact
    /// gives and exactly the answer this verb exists to not ask for.
    ///
    /// So the expression arrives as a THUNK and is forced on the far side of
    /// the check. Nothing else differs, which is why this is a sibling and
    /// not a replacement: a call site whose expression cannot fail says so by
    /// using [`read_expr`](Builder::read_expr).
    ///
    /// # Errors
    ///
    /// The builder's own, and [`read_expr`](Builder::read_expr)'s.
    pub fn read_derived(
        &mut self,
        w: &Weight,
        build: impl FnOnce() -> Result<Expr, Error>,
    ) -> Result<(), Error> {
        if self.holds_the_landed_plane(w) {
            return self.read_own(w);
        }
        self.whole_checkpoint(w)?;
        let read = declare(self.src, w, build()?)?;
        self.tensors.push(read);
        Ok(())
    }

    /// Read `w` from the tensor of its own name — the native artifact's
    /// spelling, written by `pie model import` out of these same contracts.
    ///
    /// This is the one verb that bands: a weight declared cut is read as
    /// `tp` bands of the stored whole, each rank claiming its own. A
    /// checkpoint that ships a bank's codes brings the shipped companion
    /// planes along under the same banding; one that stores the bytes
    /// unquantized against a quantized declaration leaves the encode to the
    /// loader, stated as an honest cast.
    pub fn read_own(&mut self, w: &Weight) -> Result<(), Error> {
        let read = resolve(self.src, claim(w, self.tp))?;
        self.tensors.extend(read);
        Ok(())
    }

    /// A contract this language cannot say, stated raw — interleaved rows,
    /// strided legs, planes checked with [`stored_encoding`] and typed with
    /// [`extents`]/[`grouped`]/[`divided`]. What comes through this door is
    /// exactly as bound as the rest — the plan compiler checks every entry
    /// the same way — it is only said without the sugar.
    pub fn push(&mut self, tensor: TensorContract) {
        self.tensors.push(tensor);
    }

    /// [`push`](Builder::push), several at a time.
    pub fn extend(&mut self, tensors: impl IntoIterator<Item = TensorContract>) {
        self.tensors.extend(tensors);
    }

    /// The contract the lines added up to.
    #[must_use]
    pub fn build(self) -> ModelContract {
        ModelContract {
            alignment: ALIGNMENT,
            tensors: self.tensors,

            groups: Vec::new(),
        }
    }

    /// **DOES THE FILE ALREADY HOLD THIS WEIGHT'S LANDED PLANE?** (§M-4a) —
    /// the one question every foreign verb asks before it does anything, and
    /// the door `pie model import` was widened to make reachable.
    ///
    /// A checkpoint spells a weight the way its own format spells it:
    /// `model.layers.7.self_attn.q_proj.weight`, `blk.7.attn_q.weight`. A
    /// tensor sitting under `layer.7.qg_proj` — the name THIS TEXT declares —
    /// is therefore not a checkpoint's tensor at all. It is an artifact
    /// `pie model import` wrote out of this very contract: the fusion has been
    /// performed, the codes have been lifted out of their words, the repack
    /// has run, and the plane is the one the engine binds. So the read is
    /// [`read_own`](Builder::read_own) and there is nothing left to convert.
    ///
    /// **ASKED HERE AND NOT IN SEVEN FAMILY TEXTS.** Two families wrote this
    /// arm by hand — `qwen_3::import`'s `projection` and `kimi_k3`'s
    /// `expert_bank` — over the handful of weights whose transform they knew
    /// import would take. §M-4a promotes the WHOLE landing, so the property
    /// has to hold for every weight a text reads, and the texts read them
    /// through some two hundred `b.read*` lines. The verbs are the one place
    /// all two hundred pass through, so the arm goes here and the property is
    /// structural rather than remembered.
    ///
    /// **AND IT IS ASKED BEFORE [`whole_checkpoint`](Builder::whole_checkpoint).**
    /// That assert says a foreign checkpoint is imported whole, which is why
    /// the foreign verbs refuse `tp > 1`; `read_own` is the tp-aware verb and
    /// carries no such refusal. A file holding this plane under its own name
    /// is not the foreign case, so the assert is not the one that applies —
    /// which is the two-door shape the assert was written against.
    ///
    /// **A NAME, WHICH IS A BELIEF AND NOT A CHECK.** Nothing here proves the
    /// tensor was written by this build from this contract; a checkpoint that
    /// happened to publish `layer.7.qg_proj` would be taken at its word. The
    /// collision is implausible in every format this tree reads — the two
    /// spellings above share no prefix with a declared plane name — and the
    /// honest answer to it is §M-4c's stamp, which makes the artifact SAY what
    /// it is for instead of being recognised by its names. Until that lands
    /// this is the fact the artifact states, and it is the fact the two
    /// hand-written arms this replaces were already resting on.
    fn holds_the_landed_plane(&self, w: &Weight) -> bool {
        self.src.get(&w.name).is_some()
    }

    /// The foreign verbs read a checkpoint nothing has banded, so a sharded
    /// deployment cannot import — the rule the six families each stated as
    /// their own assert, now stated once.
    ///
    /// **AND IT IS A REFUSAL AND NOT AN ASSERT**, which it was. `models::imports`
    /// publishes every row with the width its catalog row names, and `ImportFn`
    /// takes that width as an argument, so a caller reaching tp > 1 here has
    /// done nothing a type or a doc forbade — it walked the public table and
    /// handed a published row its own published number. A panic is the answer
    /// to a caller that broke an invariant it could see; this one could not.
    ///
    /// It went unnoticed because the witness sniff refused first: every arm
    /// was gated on a name, a stranger checkpoint failed the gate, and no
    /// contract was ever built at a sharded width. The moment the arms started
    /// being CHOSEN BY BUILDING THEM, every row reached this line, and
    /// `every_import_row_reads_the_checkpoint_it_is_handed` — which hands all
    /// of them one tensor no model reads — stopped being a list of refusals
    /// and became a panic on the first tp2 row. It would have panicked on a
    /// real sharded import all along.
    ///
    /// [`Error::Illegible`] rather than a new variant: from the caller's side
    /// this IS the file being unreadable by this row, and the row that can
    /// read it at that width does not exist yet. §M-4's ruling is that a
    /// sharded artifact gets BUILT — degree in the stamp, rank cut at load —
    /// and when the banded import lands it is this refusal that it replaces.
    fn whole_checkpoint(&self, w: &Weight) -> Result<(), Error> {
        if self.tp != 1 {
            return Err(Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "an import states the WHOLE checkpoint and this contract is \
                     built for {} ranks; nothing has banded the file it is \
                     reading, so there is no rank {}'s share of `{}` in it to \
                     land",
                    self.tp,
                    self.tp - 1,
                    w.name,
                ),
            });
        }
        Ok(())
    }
}

/// How many codes one group of a packed bank shares a scale with.
///
/// A NUMBER PER SCHEME, NOT A NUMBER FOR THE TREE. It was `const GROUP: u32 =
/// 32` — mxfp4's, back when mxfp4 was the only packed weight here — and every
/// site that blocked an axis or sized a scales plane read it. MLX's affine U4
/// groups sixty-four, so a single constant would have declared a scales plane
/// twice as long as the checkpoint ships and failed at the byte count, which
/// is late and far from the sentence that was wrong.
fn group_of(dtype: Dtype) -> u32 {
    match dtype {
        Dtype::Mxfp4 => 32,
        // Both affine widths group sixty-four CODES, not sixty-four bytes:
        // the group is a property of the scheme and the width is a property
        // of the code. See `dtype::Dtype::U8g64`.
        Dtype::U4g64 | Dtype::U8g64 | Dtype::U2g64 | Dtype::U4g64tiled => 64,
        Dtype::U4g32 | Dtype::U2g32 => 32,
        Dtype::U2g128 => 128,
        other => panic!("`{other:?}` blocks no axis; only a packed bank has groups"),
    }
}

/// How many affine codes the checkpoint packs into one `u32` word — MLX's own
/// packing, least-significant code first, which is a fact the loader reads off
/// `QuantScheme::MlxAffineU4` and this file only has to count with.
///
/// **THE COUNT IS THE WIDTH'S, NOT THE SCHEME'S**, and it had been a constant
/// while four bits was the only width MLX wrote here. A u32 holds eight
/// four-bit codes and FOUR eight-bit ones, so a router gate stored `[32, 720]`
/// unpacks to `[32, 2880]` through this and to `[32, 5760]` through the
/// constant — a bank twice as wide as the checkpoint holds, caught late and
/// far from the sentence that was wrong.
fn word_codes(dtype: Dtype) -> i64 {
    let bits = i64::try_from(dtype.bits()).expect("a code width inside i64");
    assert!(
        bits > 0 && 32 % bits == 0,
        "`{dtype:?}` is {bits} bits wide, which does not divide a 32-bit word"
    );
    32 / bits
}

const ALIGNMENT: u32 = 256;

struct Claim {
    pub name: String,
    pub shape: Vec<i64>,
    pub bands: Option<(u32, Vec<i64>)>,
    pub encoding: Encoding,
    pub scales: Option<Scales>,
}


#[must_use]
fn claim(w: &Weight, tp: u32) -> Claim {
    let (encoding, scales) = match w.dtype {
        Dtype::Mxfp4 | Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => {
            (grouped(w), Some(scaling(w)))
        }
        Dtype::Bf16
        | Dtype::F16
        | Dtype::F32
        | Dtype::I32
        | Dtype::U32
        | Dtype::U8
        | Dtype::I8
        | Dtype::E4m3
        | Dtype::E8m0
        | Dtype::E5m2
        | Dtype::I64
        | Dtype::I16
        | Dtype::U64
        | Dtype::U16
        | Dtype::Bool => (encoding(w.dtype), None),
        // **A SELF-CONTAINED TERM CLAIMS ONE TENSOR AND NO COMPANIONS.** Its
        // factors live inside its payload, so there are no `.scales` and no
        // `.biases` to pair in — which is what puts it on the plain arm's
        // side of this match with a quantized encoding, and what makes
        // serving it AS STORED a claim with nothing on either side of it.
        Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k => (encoding(w.dtype), None),
        Dtype::E2m1 => panic!(
            "`Dtype::E2m1` names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
        // Served, but no load contract declares one yet — the same
        // statement `model_dsl::Weight::planes` makes, one crate over.
        Dtype::Nvfp4 | Dtype::E4m3row | Dtype::E4m3tile128 => panic!(
            "a {:?} weight is served but no load contract declares one",
            w.dtype
        ),
    };
    Claim {
        name: w.name.clone(),
        shape: banded_rows(w, whole(w, tp)),
        bands: banding(w, tp),
        encoding,
        scales,
    }
}

/// **A TILED AFFINE WEIGHT CLAIMS THE RECTANGLE IT WAS REPACKED INTO**, which
/// is its output columns rounded up to a whole mma band (§J4b).
///
/// Every other row claims the shape the text declared, and this one cannot:
/// `model_dsl::Weight::planes` publishes the PADDED rectangle — the tail is
/// zero codes beside zero factors, which decodes to a zero weight the kernel
/// does not store — and a contract that claimed the unpadded one would
/// declare a tensor shorter than the plane the trace interned. The engine
/// checks an arriving tensor against `plane_bytes`, so the two shapes must be
/// the same shape, and [`TILED_BAND`] is the one number that says so.
///
/// The companions follow for free: `interned` divides the LAST axis to size
/// them, so a `[rows, k]` that has been banded gives a `[rows, groups]` that
/// has been banded too.
fn banded_rows(w: &Weight, shape: Vec<i64>) -> Vec<i64> {
    if w.dtype != Dtype::U4g64tiled {
        return shape;
    }
    let mut shape = shape;
    let band = i64::from(TILED_BAND);
    let rows = shape.first_mut().unwrap_or_else(|| {
        panic!("`{}` is a tiled affine weight declared with no rows", w.name)
    });
    *rows = (*rows + band - 1) / band * band;
    shape
}

/// One claim, checked against the source and stated as the one, two or
/// three tensors the checkpoint stores it as.
fn resolve(src: &ztensor::Source, claim: Claim) -> Result<Vec<TensorContract>, Error> {
    let mut tensors = Vec::new();
    let Claim {
        name,
        shape,
        bands,
        encoding,
        scales,
    } = claim;
    let stored = stored_encoding(src, &name)?;
    let read = banded(&name, bands.as_ref());
    let expr = ladder(&name, read, &stored, &encoding)?;
    tensors.push(match &encoding {
        Encoding::Raw(_) => {
            TensorContract::new(name.clone(), expr, shape.clone(), encoding.clone())
        }
        Encoding::Quant(_) => TensorContract::inferred(name.clone(), expr, encoding.clone()),
    });
    match (&stored, scales) {
        // The checkpoint SHIPPED both planes: the second one is bytes on
        // disk, so the contract says where they are.
        (Encoding::Quant(_), Some(pairing)) => {
            tensors.extend(interned(&name, &shape, bands, pairing));
        }
        // The checkpoint ships the weight unquantized and the model wants
        // it quantized, so `ladder` above put an honest
        // `Cast { to: Quant(..) }` in the expression and the LOADER
        // encodes on the way in. It also publishes the scales plane the
        // codes cannot be read without, under `<w>.scales` — the one name
        // this tree binds an mxfp4 second plane by, the one
        // `model_dsl::scales_name` writes and `Weight::planes` interns
        // into `Trace::params`. That accord is settled in the loader
        // (`plan::build::ScaleLayout::for_encode`'s MXFP4 arm carries the
        // ruling and `executor::walk`'s
        // `an_expert_bank_encodes_to_the_same_bytes_as_the_rows_it_stacks`
        // proves the bytes and the name), which is why there is nothing
        // to declare here: an entry of our own would be a SECOND producer
        // for a plane that already has exactly one.
        //
        // THIS ARM WAS A REFUSAL — `Error::Incompatible`, "one
        // quantization is not decoded into another on the way in", which
        // is a sentence about the Quant-to-Quant case wrongly applied to
        // this one. It made runtime quantization unreachable through
        // `Model::load` at all, against M18's own ruling that a declared
        // encoding the checkpoint does not hold is exactly what makes the
        // loader cast or encode. The Quant-to-Quant refusal is unaffected:
        // it lives in `ladder`, where it belongs.
        (Encoding::Raw(_), Some(_)) => {}
        (Encoding::Quant(_), None) | (Encoding::Raw(_), None) => {}
    }
    Ok(tensors)
}
fn declare(
    src: &ztensor::Source,
    w: &Weight,
    expr: Expr,
) -> Result<TensorContract, Error> {
    let want = encoding(w.dtype);
    let stored = agreed(src, &w.name, &expr)?;
    let expr = ladder(&w.name, expr, &stored, &want)?;
    Ok(match &stored {
        Encoding::Raw(_) => TensorContract::new(w.name.clone(), expr, extents(w), want),
        Encoding::Quant(_) => TensorContract::inferred(w.name.clone(), expr, want),
    })
}

fn copy(
    src: &ztensor::Source,
    w: &Weight,
    from: impl Into<String>,
) -> Result<TensorContract, Error> {
    declare(src, w, Expr::src(from))
}

fn fused(
    src: &ztensor::Source,
    w: &Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<TensorContract, Error> {
    let legs = parts.into_iter().map(Expr::src).collect();
    declare(src, w, Expr::concat(pack_axis(w), legs))
}

fn planes(
    src: &ztensor::Source,
    w: &Weight,
    from: impl Into<String>,
) -> Result<Vec<TensorContract>, Error> {
    let from = from.into();
    match w.dtype {
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => affine_planes(src, w, vec![from]),
        _ => Ok(vec![copy(src, w, from)?]),
    }
}

fn planes_fused(
    src: &ztensor::Source,
    w: &Weight,
    parts: impl IntoIterator<Item = String>,
) -> Result<Vec<TensorContract>, Error> {
    let parts: Vec<String> = parts.into_iter().collect();
    match w.dtype {
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => affine_planes(src, w, parts),
        _ => Ok(vec![fused(src, w, parts)?]),
    }
}

/// The codes, the scales and the biases of one MLX affine-U4 bank, read out of
/// the `<stem>.weight` / `<stem>.scales` / `<stem>.biases` triplet each part
/// names.
fn affine_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, Error> {
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, w, part)?;
        legs.push(unpacked.clone());
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(w))));
        scales.push(model_dsl::scales_name(stem));
        biases.push(model_dsl::biases_name(stem));
    }
    holds_the_declared_rectangle(w, axis, &legs)?;
    let pairing = scaling(w);
    let counted = divided(
        &extents(w),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    Ok(vec![
        // `inferred`, as `gpt_oss`'s bank planes are: the transmute above
        // already stated the logical shape of every leg, and a Quant
        // declaration that also predicted the joined shape would state the
        // same rectangle twice in two arithmetics.
        TensorContract::inferred(w.name.clone(), joined(axis, codes), grouped(w)),
        factors(
            src,
            model_dsl::scales_name(&w.name),
            &scales,
            counted.clone(),
            axis,
        )?
        .scaling(pairing),
        // `offsetting` and not a second `scaling`: the two companions complete
        // ONE attachment, and it is the zero-point entry that says which
        // weight it centres. A biases plane that named nothing would land as
        // a bound tensor no kernel reaches, and the codes beside it would
        // dequantize around zero — right spread, wrong centre, no NaN to
        // notice it by.
        factors(src, model_dsl::biases_name(&w.name), &biases, counted, axis)?
            .offsetting(w.name.clone()),
    ])
}

/// **THE SAME TRIPLET, RELAID** — [`affine_planes`]'s three entries with an
/// [`Expr::Repack`] on the end of each.
///
/// The codes take [`RepackLayout::TiledAffineU4Weight`] and both companions
/// take [`RepackLayout::TiledAffineFactor`], and every one of them declares
/// the SAME target rows: the weight's own, rounded up to a whole
/// [`TILED_BAND`]. That is the rectangle `model_dsl::Weight::planes` publishes
/// and the rectangle [`claim`] claims, so the three statements of it are one
/// number.
///
/// **THE PADDING IS ZERO CODES BESIDE ZERO FACTORS**, which decodes to a zero
/// weight — the kernels walk a padded band with the others, accumulate zero,
/// and the epilogue does not write those columns. So the tail is not a value
/// anybody has to know about downstream; it is a column that is not there.
fn tiled_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
) -> Result<Vec<TensorContract>, Error> {
    if w.dtype != Dtype::U4g64tiled {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!(
                "it is declared {:?} and this verb states a repack; a relabelled plane \
                 is declared `U4g64tiled`, because the declaration is what says which \
                 order the artifact holds",
                w.dtype
            ),
        });
    }
    let axis = if parts.len() > 1 { pack_axis(w) } else { 0 };
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, w, part)?;
        legs.push(unpacked.clone());
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(w))));
        scales.push(model_dsl::scales_name(stem));
        biases.push(model_dsl::biases_name(stem));
    }
    // Against the flat rectangle, which is what the legs join into; the band
    // order below is a relabelling of that same shape and not another one.
    holds_the_declared_rectangle(w, axis, &legs)?;
    let pairing = scaling(w);
    // The row-major rectangle the legs join into, and the banded one they are
    // relaid into. `divided` sizes the companions off whichever it is handed,
    // so the factor planes band exactly as the codes do.
    let flat = extents(w);
    let landed = banded_rows(w, flat.clone());
    let counted = divided(&landed, pairing.channel_axis, pairing.group_size, &w.name);
    let factor_target = |shape: Vec<i64>| TensorType::new(shape, encoding(Dtype::Bf16));
    Ok(vec![
        TensorContract::inferred(
            w.name.clone(),
            joined(axis, codes).repack(
                RepackLayout::TiledAffineU4Weight,
                TensorType::new(landed, grouped(w)),
            ),
            grouped(w),
        ),
        relaid(
            src,
            model_dsl::scales_name(&w.name),
            &scales,
            counted.clone(),
            axis,
            factor_target(counted.clone()),
        )?
        .scaling(pairing),
        relaid(
            src,
            model_dsl::biases_name(&w.name),
            &biases,
            counted.clone(),
            axis,
            factor_target(counted),
        )?
        .offsetting(w.name.clone()),
    ])
}

/// [`factors`] with the relabelling on the end — one companion plane, joined
/// at its seams and then put into band order.
fn relaid(
    src: &ztensor::Source,
    name: String,
    legs: &[String],
    shape: Vec<i64>,
    axis: u8,
    to: TensorType,
) -> Result<TensorContract, Error> {
    let plane = factors(src, name.clone(), legs, shape.clone(), axis)?;
    Ok(TensorContract::new(
        name,
        plane.expr.repack(RepackLayout::TiledAffineFactor, to),
        shape,
        encoding(Dtype::Bf16),
    ))
}

/// The same triplet, landed DENSE: the codes joined and decoded through a
/// per-block scale and a per-block bias, the companions declared INTERNAL —
/// read at load, bound by nothing. The stored representation is the
/// import's statement (`stored`), because the weight's own dtype is the
/// dense landing; everything the affine helpers key off a weight's dtype is
/// therefore keyed off a stand-in wearing the file's.
fn dequant_planes(
    src: &ztensor::Source,
    w: &Weight,
    parts: Vec<String>,
    stored: Dtype,
) -> Result<Vec<TensorContract>, Error> {
    let file = Weight {
        dtype: stored,
        ..w.clone()
    };
    let axis = if parts.len() > 1 { pack_axis(&file) } else { 0 };
    let mut codes = Vec::new();
    let mut scales = Vec::new();
    let mut biases = Vec::new();
    let mut legs = Vec::new();
    for part in &parts {
        let stem = part
            .strip_suffix(".weight")
            .ok_or_else(|| Error::Illegible {
                name: w.name.clone(),
                detail: format!(
                    "`{part}` holds MLX affine codes, whose scales and biases \
                     are named beside a `.weight`, and it does not end in one"
                ),
            })?;
        let unpacked = unpacked_extents(src, &file, part)?;
        legs.push(unpacked.clone());
        codes.push(Expr::src(part.clone()).transmute(TensorType::new(unpacked, grouped(&file))));
        scales.push(model_dsl::scales_name(stem));
        biases.push(model_dsl::biases_name(stem));
    }
    // Against `file`, whose extents are `w`'s: the two differ in dtype alone,
    // and the rectangle a dequant lands is the one the file stores.
    holds_the_declared_rectangle(&file, axis, &legs)?;
    let pairing = scaling(&file);
    let counted = divided(
        &extents(&file),
        pairing.channel_axis,
        pairing.group_size,
        &w.name,
    );
    let scales_name = model_dsl::scales_name(&w.name);
    let biases_name = model_dsl::biases_name(&w.name);
    // Two kernel nodes, two contracts: the byte-run lowering carries at most
    // one kernel at an expression's root, so the scale lands an internal
    // intermediate and the bias reads it back as a leaf.
    let scaled_name = format!("{}.scaled", w.name);
    let scaled = joined(axis, codes).scale_per_block(Expr::out(scales_name.clone()));
    let decoded = Expr::out(scaled_name.clone()).bias_per_block(Expr::out(biases_name.clone()));
    Ok(vec![
        factors(src, scales_name, &scales, counted.clone(), axis)?.internal(),
        factors(src, biases_name, &biases, counted, axis)?.internal(),
        TensorContract::inferred(scaled_name, scaled, encoding(w.dtype)).internal(),
        TensorContract::inferred(w.name.clone(), decoded, encoding(w.dtype)),
    ])
}

/// One companion plane of an affine bank, joined across the parts and brought
/// to bf16.
///
/// The cast is not a formality: mlx-community ships some conversions with F16
/// scales and biases and others with BF16, and this tree reads a bank's
/// factors as bf16 in one spelling. `ladder` is what makes that a stated
/// conversion rather than a reinterpretation of the bytes.
fn factors(
    src: &ztensor::Source,
    name: String,
    legs: &[String],
    shape: Vec<i64>,
    axis: u8,
) -> Result<TensorContract, Error> {
    let expr = joined(axis, legs.iter().cloned().map(Expr::src).collect());
    let stored = agreed(src, &name, &expr)?;
    let want = encoding(Dtype::Bf16);
    let expr = ladder(&name, expr, &stored, &want)?;
    Ok(TensorContract::new(name, expr, shape, want))
}

fn joined(axis: u8, mut legs: Vec<Expr>) -> Expr {
    if legs.len() == 1 {
        legs.pop().expect("one leg")
    } else {
        Expr::concat(axis, legs)
    }
}

/// The logical shape of a stored affine-U4 plane: what the checkpoint holds,
/// with its contracted axis multiplied back out of the words it was packed
/// into.
///
/// Read off the FILE and not off the declaration, because the declaration is
/// the whole joined bank and this is one of its legs — a fused `gate_up` is
/// one `Weight` and two stored tensors, and neither leg's width is derivable
/// from the sum without assuming they are equal, which for a qkv pack they are
/// not.
fn unpacked_extents(src: &ztensor::Source, w: &Weight, name: &str) -> Result<Vec<i64>, Error> {
    let Some(tensor) = src.get(name) else {
        return Err(Error::Missing(name.to_string()));
    };
    let stored = stored_encoding(src, name)?;
    if stored != Encoding::Raw(DType::U32) {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!(
                "`{name}` is stored {stored:?}, and MLX affine codes are read \
                 as raw u32 words of {} codes each",
                word_codes(w.dtype),
            ),
        });
    }
    let mut dims: Vec<i64> = tensor
        .shape()
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect();
    let Some(words) = dims.last_mut() else {
        return Err(Error::Illegible {
            name: w.name.clone(),
            detail: format!("`{name}` is a scalar and a bank has a contracted axis"),
        });
    };
    *words *= word_codes(w.dtype);
    Ok(dims)
}

/// **THE LEGS JOIN INTO THE RECTANGLE THIS TEXT DECLARED, OR THIS ROW DOES NOT
/// READ THIS FILE.**
///
/// A packed bank's contract is [`TensorContract::inferred`] — the transmute in
/// [`affine_planes`] already stated each leg's logical shape, so declaring the
/// joined rectangle a second time would state it in two arithmetics. The cost
/// of inferring it was that NOTHING compared the file's widths to the ones the
/// model text asks for, and a contract that never looks at a width is a
/// contract that cannot miss on one.
///
/// **WHICH MADE `identify` READ BANKS IT WAS NOT WRITTEN FOR, SILENTLY.** The
/// A3B width-invariance fleet is where it surfaced: `mini-l5-e16-k8` and
/// `mini-l5-e64-k8` hold the same 227 tensor NAMES and differ only in the
/// routed bank's leading axis, so the sixteen-expert row claimed the
/// sixty-four-expert carve — first in the walk — and served a bank of 64 as a
/// bank of 16: forty-eight experts the router could never reach, and not one
/// error anywhere. The dense path never had this hole, because [`declare`]
/// states [`extents`] on a raw plane and a bf16 row that misread a width fails
/// on the byte count. This is that same guarantee, for the planes the packing
/// made inferred.
///
/// It is checked against the JOINED shape and not per leg, because a fused
/// bank's legs are not each the declaration: `gate_up` is one `Weight` and two
/// stored tensors that meet at [`pack_axis`], and a qkv pack's legs are not
/// even the same width as each other. So the legs must agree everywhere off
/// the seam and sum to the declaration on it.
///
/// The contract is always built at `tp == 1` — [`Builder::read`] and its
/// siblings pass through `whole_checkpoint` first — so [`extents`] here is the
/// whole model's rectangle, and no sharding can make an honest file disagree
/// with an honest declaration.
fn holds_the_declared_rectangle(w: &Weight, axis: u8, legs: &[Vec<i64>]) -> Result<(), Error> {
    let declared = extents(w);
    let refuse = |detail: String| Error::Illegible {
        name: w.name.clone(),
        detail,
    };
    let Some(first) = legs.first() else {
        return Ok(());
    };
    let mut joined = first.clone();
    let at = axis as usize;
    for leg in &legs[1..] {
        if leg.len() != joined.len() {
            return Err(refuse(format!(
                "its stored parts are rank {} and rank {}, and parts that join \
                 into one bank have one rank",
                joined.len(),
                leg.len(),
            )));
        }
        for (i, (into, part)) in joined.iter_mut().zip(leg).enumerate() {
            if i == at {
                *into += *part;
            } else if *into != *part {
                return Err(refuse(format!(
                    "its stored parts differ at axis {i} ({into} against \
                     {part}), which is not the axis {at} they join on, so they \
                     are not two halves of one rectangle"
                )));
            }
        }
    }
    if joined != declared {
        return Err(refuse(format!(
            "the file stores it {joined:?} and this text declares it \
             {declared:?}; a text reads the widths it states, so this row is \
             not the one that reads this checkpoint"
        )));
    }
    Ok(())
}

/// The companion planes the checkpoint SHIPPED beside `of`, said as
/// declarations of their own.
///
/// **A SCHEME'S SECOND PLANE IS NOT ALWAYS ITS LAST.** This answered one
/// contract, because mxfp4 has one companion: a byte of exponent per block.
/// MLX's affine U4 has two — `code * scale + bias` reads a scale AND an
/// offset — and they are declared as siblings rather than one wide tensor
/// because that is how MLX ships them and how
/// `plan::build::QuantAttachment` binds them: the zero point of a shipped
/// triplet is a tensor the contract states in its own right, and only an
/// encode the loader performs has an id to record on the attachment instead.
///
/// The form is what says how many there are. `Scales::form` already carries
/// the scheme's whole answer to "what do these numbers mean", so asking it
/// "and how many tensors is that" costs nothing and cannot drift from the
/// pairing the same call site built.
fn interned(
    of: &str,
    shape: &[i64],
    bands: Option<(u32, Vec<i64>)>,
    pairing: Scales,
) -> Vec<TensorContract> {
    seams_clear_the_blocked_axis(of, bands.as_ref(), pairing.channel_axis, pairing.group_size);
    let declared = divided(shape, pairing.channel_axis, pairing.group_size, of);
    let plane = |name: String, dtype: Dtype, shape: Vec<i64>| {
        let expr = banded(&name, bands.as_ref());
        TensorContract::new(name, expr, shape, encoding(dtype))
    };
    match pairing.form {
        ScaleForm::RawE8M0 => {
            vec![plane(model_dsl::scales_name(of), Dtype::E8m0, declared).scaling(pairing)]
        }
        ScaleForm::Bf16AffineFactors => vec![
            plane(model_dsl::scales_name(of), Dtype::Bf16, declared.clone()).scaling(pairing),
            plane(model_dsl::biases_name(of), Dtype::Bf16, declared).offsetting(of),
        ],
        other => panic!(
            "`{of}` pairs its codes with {other:?} scales, and no family here \
             declares a bank in those terms"
        ),
    }
}

fn agreed(src: &ztensor::Source, name: &str, expr: &Expr) -> Result<Encoding, Error> {
    let mut read: Vec<(&str, Encoding)> = Vec::new();
    for source in expr.sources() {
        let stored = stored_encoding(src, source)?;
        read.push((source, stored));
    }
    let Some((whose, first)) = read.first() else {
        return Err(Error::Illegible {
            name: name.to_string(),
            detail: "it is built from no checkpoint tensor at all".to_string(),
        });
    };
    for (source, seen) in &read {
        if seen != first {
            return Err(Error::Illegible {
                name: name.to_string(),
                detail: format!(
                    "`{whose}` is stored {first:?} and `{source}` is stored \
                     {seen:?}; one weight is not read out of two representations"
                ),
            });
        }
    }
    Ok(first.clone())
}

/// How the checkpoint stores `name`, read off the file's own header — the
/// fact every conversion decision here starts from.
pub fn stored_encoding(src: &ztensor::Source, name: &str) -> Result<Encoding, Error> {
    let Some(tensor) = src.get(name) else {
        return Err(Error::Missing(name.to_string()));
    };
    let illegible = |why: &dyn std::fmt::Display| Error::Illegible {
        name: name.to_string(),
        detail: why.to_string(),
    };
    let part = tensor.part("data").map_err(|why| illegible(&why))?;
    checkpoint::file::encoding_of(&tensor, &part).map_err(|why| illegible(&why))
}

fn ladder(name: &str, expr: Expr, stored: &Encoding, want: &Encoding) -> Result<Expr, Error> {
    match (stored, want) {
        (s, w) if s == w => Ok(expr),
        (Encoding::Raw(_), Encoding::Raw(_)) => Ok(expr.cast(want.clone())),
        // **PACKED CODES ARE NOT VALUES TO QUANTIZE.** A `Raw -> Quant` rung
        // is the loader encoding on the way in, which is a real path — kimi
        // declares mxfp4 expert banks over a bf16 checkpoint and means exactly
        // that. It is nonsense for a checkpoint that already ships the codes:
        // MLX writes them as `u32` words, and encoding those integers as if
        // they were weights lands a bank whose every element is a code read as
        // a number, with no name in the plan being wrong. Such a file is bound
        // by naming its three planes — `contract::planes` — so this rung is a
        // refusal and not a conversion.
        (Encoding::Raw(DType::U32), Encoding::Quant(_)) => Err(Error::Illegible {
            name: name.to_string(),
            detail: "it is stored as raw u32 words and this model wants it \
                     quantized; a checkpoint that already ships packed codes is \
                     read by naming its planes, not by encoding its words"
                .to_string(),
        }),
        (Encoding::Raw(_), Encoding::Quant(_)) => Ok(expr.cast(want.clone())),
        (Encoding::Quant(_), Encoding::Raw(_)) => Ok(expr.cast(want.clone())),
        (Encoding::Quant(_), Encoding::Quant(_)) => Err(Error::Incompatible {
            name: name.to_string(),
            stored: stored.clone(),
            want: want.clone(),
        }),
    }
}

fn banding(w: &Weight, tp: u32) -> Option<(u32, Vec<i64>)> {
    match &w.shard {
        Shard::Replicated => None,
        Shard::Cut { axis, segments } => Some((
            *axis,
            segments
                .iter()
                .map(|segment| leg_extent(*segment, tp, &w.name))
                .collect(),
        )),
    }
}

fn banded(name: &str, bands: Option<&(u32, Vec<i64>)>) -> Expr {
    let Some((axis, extents)) = bands else {
        return Expr::src(name);
    };
    let at = as_axis(*axis, name);
    match extents.as_slice() {
        [] => panic!("`{name}` is cut at no seam at all"),
        [_lone] => Expr::src(name).shard(at),
        many => {
            let mut start = 0;
            let legs = many
                .iter()
                .map(|extent| {
                    let leg = Expr::src(name).slice(at, start, *extent).shard(at);
                    start += *extent;
                    leg
                })
                .collect();
            Expr::concat(at, legs)
        }
    }
}

/// `w`'s declared shape, as the signed extents a contract states.
pub fn extents(w: &Weight) -> Vec<i64> {
    w.shape
        .iter()
        .map(|extent| i64::try_from(*extent).expect("an extent no i64 holds"))
        .collect()
}

fn whole(w: &Weight, tp: u32) -> Vec<i64> {
    let mut dims = extents(w);
    match &w.shard {
        Shard::Replicated => dims,
        Shard::Cut { axis, segments } => {
            let at = *axis as usize;
            let dim = dims.get_mut(at).unwrap_or_else(|| {
                panic!("`{}` is {:?} and its cut names axis {at}", w.name, w.shape)
            });
            let seams: u64 = segments.iter().sum();
            assert_eq!(
                u64::try_from(*dim).expect("an extent no u64 holds"),
                seams,
                "`{}`: its segments sum to {seams} and its axis {at} is {dim}",
                w.name,
            );
            *dim = dim
                .checked_mul(i64::from(tp))
                .unwrap_or_else(|| panic!("`{}` is {tp} times wider than an i64", w.name));
            dims
        }
    }
}

/// `shape` with `axis` counted in `group`-code blocks — the shape of a scales
/// plane, derived from the bank it reads.
pub fn divided(shape: &[i64], axis: u32, group: u32, name: &str) -> Vec<i64> {
    let mut dims = shape.to_vec();
    let at = axis as usize;
    let extent = *dims
        .get(at)
        .unwrap_or_else(|| panic!("`{name}` is {shape:?} and its blocks count along axis {at}"));
    let width = i64::from(group);
    assert!(
        extent % width == 0,
        "`{name}` contracts over {extent}, which is not a whole number of \
         {group}-code blocks",
    );
    dims[at] = extent / width;
    dims
}

fn seams_clear_the_blocked_axis(
    name: &str,
    bands: Option<&(u32, Vec<i64>)>,
    channel: u32,
    group: u32,
) {
    let Some((axis, extents)) = bands else {
        return;
    };
    assert!(
        extents.len() < 2 || *axis != channel,
        "`{name}` is cut at {} seams along axis {axis}, which is the axis its \
         scales count in {group}-code blocks",
        extents.len(),
    );
}

/// How `w`'s codes are paired with the numbers that read them.
///
/// The scheme decides both halves and they are not independent: an mxfp4 block
/// is thirty-two codes under one exponent byte, an MLX affine group is
/// sixty-four codes under one bf16 scale and one bf16 offset. Reading the
/// group width off one scheme and the form off another would produce a pairing
/// no kernel implements and no checkpoint ships.
pub fn scaling(w: &Weight) -> Scales {
    let form = match w.dtype {
        Dtype::Mxfp4 => ScaleForm::RawE8M0,
        Dtype::U4g64
        | Dtype::U8g64
        | Dtype::U4g32
        | Dtype::U4g64tiled
        | Dtype::U2g32
        | Dtype::U2g64
        | Dtype::U2g128 => ScaleForm::Bf16AffineFactors,
        other => panic!(
            "`{}` is {other:?}, which pairs with nothing; only a packed bank \
             has scales",
            w.name
        ),
    };
    Scales {
        of: w.name.clone(),
        granularity: QuantGranularity::PerGroup,
        group_size: group_of(w.dtype),
        channel_axis: u32::from(channel_axis(w)),
        form,
    }
}

/// `w`'s quantized encoding with its blocked axis stated — the channel axis
/// is the bank's own last, because a rank is not a fact about a scheme.
pub fn grouped(w: &Weight) -> Encoding {
    match encoding(w.dtype) {
        Encoding::Quant(spec) => Encoding::Quant(QuantSpec {
            channel_axis: Some(Axis(channel_axis(w))),
            ..spec
        }),
        Encoding::Raw(dtype) => panic!(
            "`{}` is {dtype:?}, which groups nothing; only a quantized bank \
             has a blocked axis",
            w.name
        ),
    }
}

fn channel_axis(w: &Weight) -> u8 {
    let last = w
        .shape
        .len()
        .checked_sub(1)
        .unwrap_or_else(|| panic!("`{}` is a bank and has no contracted axis", w.name));
    u8::try_from(last).expect("an axis inside a shape")
}

fn pack_axis(w: &Weight) -> u8 {
    match &w.shard {
        Shard::Replicated => panic!("`{}` is replicated and has no cut axis", w.name),
        Shard::Cut { axis, .. } => as_axis(*axis, &w.name),
    }
}

fn leg_extent(segment: u64, tp: u32, name: &str) -> i64 {
    let whole = segment
        .checked_mul(u64::from(tp))
        .unwrap_or_else(|| panic!("`{name}`: a segment of {segment} is not {tp} times anything"));
    i64::try_from(whole).expect("an extent no i64 holds")
}

fn as_axis(axis: u32, name: &str) -> u8 {
    u8::try_from(axis)
        .unwrap_or_else(|_| panic!("`{name}` is cut on axis {axis}, which is no axis"))
}

/// The stored representation each weight [`Dtype`] declares -- raw of
/// itself for everything that stores itself verbatim, and the quantization
/// spec of the scheme for a packed bank's codes.
///
/// # Panics
///
/// On a [`Dtype::Quant`] whose term no self-contained scheme in this tree's
/// checkpoint vocabulary has -- see [`checkpoint::spec_of_term`].
pub fn encoding(dtype: Dtype) -> Encoding {
    match dtype {
        Dtype::Bf16 => Encoding::Raw(DType::Bf16),
        Dtype::F16 => Encoding::Raw(DType::F16),
        Dtype::F32 => Encoding::Raw(DType::F32),
        Dtype::I32 => Encoding::Raw(DType::I32),
        Dtype::U32 => Encoding::Raw(DType::U32),
        Dtype::U8 => Encoding::Raw(DType::U8),
        Dtype::I8 => Encoding::Raw(DType::I8),
        Dtype::E4m3 => Encoding::Raw(DType::E4m3),
        Dtype::E8m0 => Encoding::Raw(DType::E8m0),
        // The six the checkpoint vocabulary brought when the two dtype enums
        // merged. Each stores itself verbatim, so each is `Raw` of itself —
        // the same answer every row above gives.
        Dtype::E5m2 => Encoding::Raw(DType::E5m2),
        Dtype::I64 => Encoding::Raw(DType::I64),
        Dtype::I16 => Encoding::Raw(DType::I16),
        Dtype::U64 => Encoding::Raw(DType::U64),
        Dtype::U16 => Encoding::Raw(DType::U16),
        Dtype::Bool => Encoding::Raw(DType::Bool),
        Dtype::Mxfp4 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::Mxfp4E2M1E8M0,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        }),
        // MLX's affine U4, in the loader's own vocabulary: sixty-four codes
        // under one bf16 scale and one bf16 offset, dequantized to bf16. The
        // channel axis is stated by whoever holds the shape —
        // `grouped` — because a rank is not a fact about a scheme.
        Dtype::U4g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
        }),
        // **THE SAME SCHEME AT TWICE THE WIDTH.** `MlxAffineU4` names the
        // arithmetic — affine codes, sixty-four to a group, one bf16 scale and
        // one bf16 offset apiece — and `bits_per_element` has always been the
        // field that says how wide a code is, which is why an eight-bit MLX
        // bank needs no scheme of its own. `Landing::affine_point_of` reports
        // this number to the engine and `kernels_metal::linear::quant` stamps
        // a point at both widths, so the two travel the whole way down as one
        // path with a number in it. See `dtype::Dtype::U8g64` for the
        // checkpoint that mixes them.
        Dtype::U8g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 8,
            group_size: 64,
            channel_axis: None,
        }),
        // **AND THE SAME SCHEME AT HALF THE GROUP** — `U8g64`'s argument one
        // spec field over: `group_size` has always been the number that says
        // how many codes share a scale, and a 160-wide table row can only
        // group by thirty-two (see `dtype::Dtype::U4g32`).
        Dtype::U4g32 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 32,
            channel_axis: None,
        }),
        // **AND THE SAME SCHEME AGAIN, WITH ITS BYTES SOMEWHERE ELSE**
        // (§J4b). A repack moves no value: `U4g64tiled` holds `U4g64`'s codes
        // under `U4g64`'s factors, in the order a tensor-core lane's fragment
        // reads them. So the stored ENCODING is `U4g64`'s, letter for letter,
        // and the layout is said where the other placement facts are said —
        // on the `Dtype`, which is what selected this arm. A spec that
        // differed here would be a scheme claiming a permutation changed the
        // arithmetic.
        Dtype::U4g64tiled => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 4,
            group_size: 64,
            channel_axis: None,
        }),
        // **AND THE SAME SCHEME AT TWO BITS, OVER ITS THREE GROUPS.** The two
        // fields that have carried every MLX affine row — `bits_per_element`
        // and `group_size` — carry these as well; nothing about the arithmetic
        // is new, which is the whole reason `MlxAffineU4` did not have to grow
        // a sibling. A DQ checkpoint spends its bits per tensor and writes all
        // three groups, so all three are spellable (`dtype::Dtype::U2g32`).
        Dtype::U2g32 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 2,
            group_size: 32,
            channel_axis: None,
        }),
        Dtype::U2g64 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 2,
            group_size: 64,
            channel_axis: None,
        }),
        Dtype::U2g128 => Encoding::Quant(QuantSpec {
            scheme: QuantScheme::MlxAffineU4,
            logical_dtype: DType::Bf16,
            bits_per_element: 2,
            group_size: 128,
            channel_axis: None,
        }),
        Dtype::E2m1 => panic!(
            "`Dtype::E2m1` names a kv-page quantization scheme, not a stored \
             weight plane; no load contract declares one"
        ),
        // Served, but no load contract declares one yet — the same
        // statement `model_dsl::Weight::planes` makes, one crate over.
        Dtype::Nvfp4 | Dtype::E4m3row | Dtype::E4m3tile128 => panic!(
            "a {:?} weight is served but no load contract declares one",
            dtype
        ),
        // **THE TERM IS THE DECLARATION AND THE SCHEME IS THE LOOKUP.** The
        // rows above spell a scheme's numbers out because their `Dtype`
        // variant is a NAME and the numbers had to come from somewhere; this
        // one carries the arithmetic itself, so the only thing left to find
        // is which of the checkpoint's block schemes has it. `qnf` is that
        // door, and asking it here is what keeps this file from being a
        // second place a `q4_k` is defined.
        Dtype::U2g16k
        | Dtype::I3g16k
        | Dtype::U4g32k
        | Dtype::U5g32k
        | Dtype::I6g16k => Encoding::Quant(
            checkpoint::spec_of_term(dtype.repr()).unwrap_or_else(|| {
                panic!(
                    "`{dtype}` is not the arithmetic of any self-contained scheme this \
                     checkpoint vocabulary knows; a term whose factors live in \
                     companion planes is declared by the variant that names them"
                )
            }),
        ),
    }
}
