#pragma once

/// Authoring a load contract from C++ (architecture.md §12 row 12).
///
/// A contract is the *program* half of `plan = compile(source, program,
/// target)`. It names every tensor the driver wants, and for each one gives an
/// expression over the checkpoint's tensors that produces it. The loader
/// type-checks the expression against what the files actually contain, lowers
/// it to storage instructions, and schedules them; it does not decide *what* to
/// build, which is the whole point.
///
/// The driver already holds every fact this needs. `llama_like`'s
/// `Hq = num_attention_heads * head_dim` and its `must(engine, p +
/// "self_attn.o_proj.weight")` are exactly the shape and the name that used to
/// be re-derived inside the loader's `arch/` passes from a second, independent
/// reading of `config.json`. Two readings that can disagree is one too many.
///
/// # The algebra
///
/// Fourteen node kinds, mirroring `crate::contract::Expr` one for one. They are
/// grouped by the invariant each one preserves, because that -- not the name --
/// is what tells you which of them to reach for:
///
/// | Node                             | Preserves            | Meaning                                     |
/// |----------------------------------|----------------------|---------------------------------------------|
/// | `src(name)`                      | --                   | A tensor as it appears in the checkpoint    |
/// | `out(name)`                      | --                   | An earlier `define` in this contract        |
/// | `fill(value, shape, enc)`        | --                   | A constant tensor; a `concat` leg that pads |
/// | `slice(x, axis, start, len)`     | bytes, values, type  | A contiguous band along one axis            |
/// | `stride(x, axis, start, len, k)` | bytes, values, type  | Every `k`-th element along one axis         |
/// | `gather(x, axis, {i, j, k})`     | bytes, values, type  | The elements at those indices, in order     |
/// | `concat({a, b, c}, axis)`        | bytes, values, type  | The parts joined, in the order given        |
/// | `shard(x, axis)`                 | bytes, values, type  | This rank's slice of a TP-split axis        |
/// | `transmute(x, shape, enc)`       | bytes                | Same bytes under a new type; one `-1` is inferred |
/// | `scale(x, factor)`               | type                 | Multiply every element                      |
/// | `cast(x, enc)`                   | values               | The same values in another representation   |
/// | `repack(x, spec, out)`           | values, type         | An opaque kernel-specific relayout          |
/// | `index_src("a.{}.w")`            | --                   | A checkpoint tensor named for this instance |
/// | `select(x, axis, stride, len)`   | bytes, values, type  | This instance's band along one axis         |
///
/// The five that preserve all three move bytes and nothing else: what comes out
/// is a rearrangement of what went in, and every one of them compiles to byte
/// runs the loader can read straight out of the checkpoint.
///
/// `transmute` is free as well — it renames bytes without moving them. The
/// other three cost a kernel. `repack` preserves everything the five do except
/// placement, and it is held to that: it may pad, and it may not reinterpret an
/// element. It is a relayout priced as a kernel, not an escape hatch for
/// whatever a relayout could be made to do.
///
/// `slice`, `stride` and `gather` are three rungs of one cost ladder, and the
/// author picks the rung. A band leaves every run in the source intact; a
/// stride cuts the source into one run per element; a gather is what is left
/// when no arithmetic describes the order. The loader refuses a list that a
/// cheaper node could express, so the ladder cannot be climbed by accident.
/// For the same reason neither `stride` nor `gather` may touch a quantized
/// axis, where a `slice` may as long as it lands on group boundaries.
///
/// No model has needed `gather` yet. It is the rung that makes the other two
/// the cheap ends of one operation rather than a pair of special cases, so it
/// stays; reach for it only when no arithmetic describes the order you want.
///
/// `shard` is the only node that means different things on different ranks, and
/// it is resolved before anything downstream of it runs — which is why a
/// contract is written once for the whole TP group rather than once per rank.
///
/// `index_src` and `select` are `shard`'s siblings on a second axis: they mean
/// different things for different *instances* of a `group`, and are resolved
/// the same way and at the same point. The two axes are independent -- one
/// picks a tensor, the other a slice of it -- so a sharded group needs no
/// special care from either side. Both are an error outside a group, where
/// there is no instance for them to stand for.
///
/// `transmute` and `cast` are duals and are easy to confuse, so: a transmute
/// keeps the *bytes* and gives them a new name, a cast keeps the *values* and
/// rewrites them. One is free, the other runs a converter. Rust spells the
/// same pair `transmute` and `as`.
///
/// `scale` and `cast` are the two nodes that change a value, and they change it
/// in different ways: `scale` by arithmetic the author asked for, `cast` only
/// by the error of the representation it lands in.
///
/// # Shapes and encodings are both predictions
///
/// `define(name, expr, encoding)` states what the tensor should *be*, and both
/// halves of that are claims about the expression, checked against what it
/// actually yields. A driver whose model of the checkpoint is wrong gets a
/// compile error rather than a plausible-looking wrong buffer.
///
/// A conversion is never inferred from a declaration that disagrees. Ask for it
/// with `cast`, so that a line that runs a kernel does not look like a line that
/// does not.
///
/// The shape half may be declined: omitting `expect` is the honest declaration
/// for a packed quantized weight, whose on-disk extents belong to the quantizer
/// that produced the file and not to the model.
///
/// # Ownership
///
/// The builder owns every name, shape and node; `view()` lends a borrowed POD
/// snapshot. The builder must outlive the `pie_loader_compile_contract` call and
/// must not be mutated while a view is outstanding.
///
/// # Example
///
/// ```
/// pie_loader::ModelContract c;
/// auto q = c.src("model.layers.0.self_attn.q_proj.weight");
/// auto k = c.src("model.layers.0.self_attn.k_proj.weight");
/// auto v = c.src("model.layers.0.self_attn.v_proj.weight");
/// c.define("model.layers.0.self_attn.qkv_proj.weight",
///          c.shard(c.concat({q, k, v}, 0), 0),
///          pie_loader::raw(PieLoaderDType::BF16))
///     .expect({(Hq + 2 * Hkv) / tp_size, hidden});
/// ```

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "pie_loader.h"

namespace pie_loader {

/// A node in the expression graph being built.
///
/// Opaque on purpose: it is an index into the builder that produced it, so
/// mixing handles from two builders is meaningless. Copying is free and sharing
/// is free — naming the same handle twice builds one node, not two, which is how
/// a `concat` of three `slice`s of one `src` reads the source once.
class Node {
public:
    Node() = default;

private:
    friend class ModelContract;
    explicit Node(std::uint32_t index) : index_(index) {}
    std::uint32_t index_ = PIE_LOADER_NO_NODE;
};

/// A raw (unquantized) encoding.
inline PieLoaderEncodingSpec raw(PieLoaderDType dtype) {
    PieLoaderEncodingSpec spec{};
    spec.kind = static_cast<std::uint32_t>(PieLoaderEncodingKind::Raw);
    spec.dtype = static_cast<std::uint32_t>(dtype);
    spec.quant.scheme = static_cast<std::uint32_t>(PieLoaderQuantScheme::None);
    spec.quant.logical_dtype = static_cast<std::uint32_t>(PieLoaderDType::BF16);
    spec.quant.channel_axis = PIE_LOADER_NO_AXIS;
    return spec;
}

/// A quantization spec with everything defaulted.
///
/// `bits_per_element` and `group_size` left at `0` ask for the scheme's own
/// default, which is how the loader states them internally too — spelling out a
/// value the scheme already implies is a second place for the two to disagree.
inline PieLoaderQuantSpecView quant_spec(PieLoaderQuantScheme scheme, PieLoaderDType logical) {
    PieLoaderQuantSpecView spec{};
    spec.scheme = static_cast<std::uint32_t>(scheme);
    spec.logical_dtype = static_cast<std::uint32_t>(logical);
    spec.bits_per_element = 0;
    spec.group_size = 0;
    spec.channel_axis = PIE_LOADER_NO_AXIS;
    return spec;
}

/// A quantized encoding.
inline PieLoaderEncodingSpec quantized(PieLoaderQuantSpecView quant) {
    PieLoaderEncodingSpec spec{};
    spec.kind = static_cast<std::uint32_t>(PieLoaderEncodingKind::Quant);
    spec.dtype = static_cast<std::uint32_t>(PieLoaderDType::BF16);
    spec.quant = quant;
    return spec;
}

class ModelContract {
public:
    /// Handle to the tensor just defined, so that `expect` can be chained.
    class Defined {
    public:
        /// Declare the per-rank shape this tensor should have.
        ///
        /// Already divided by the TP size: a column-sharded projection on a
        /// 4-way group declares `hidden / 4`, not `hidden`. Declaring the
        /// model-wide shape reports a mismatch on every sharded weight.
        Defined& expect(std::vector<std::int64_t> shape) {
            owner_->set_shape(group_, index_, std::move(shape));
            return *this;
        }

        /// Declare that this tensor holds the scales for `of`.
        ///
        /// Only for scales the checkpoint shipped. Scales the loader writes
        /// while quantizing are paired at creation and need nothing here.
        ///
        /// The loader used to work this pairing out for itself, by matching
        /// `{name}_scale_inv` and then `{base}.scale` over the finished tensor
        /// table with `group_size` hardcoded to 128. `dsv4_block_scales_to_fp32`
        /// had already found it properly -- it takes a `.scale` tensor, looks up
        /// `<base>.weight`, and publishes only if that companion is really
        /// FP8-E4M3 -- and then dropped it. This is where that finding is kept.
        ///
        /// `of` may name a declaration made later -- this pairs two published
        /// tensors rather than feeding one into the other -- but the loader
        /// rejects a contract whose scales name nothing at all.
        Defined& scaling(std::string of, PieLoaderQuantGranularity granularity,
                         std::uint32_t group_size, std::uint32_t channel_axis,
                         PieLoaderScaleForm form) {
            owner_->set_scales(group_, index_, std::move(of), granularity, group_size,
                               channel_axis, form);
            return *this;
        }

        /// Keep this tensor out of the driver's bind table.
        ///
        /// The algebra has no `let`: using a subexpression twice, or feeding
        /// one entry into another's `scale_per_block` factors, means giving it
        /// a name. Left public, such a name is also a runtime weight -- it
        /// lands in the persistent arena and stays there for the process's
        /// lifetime, because an arena view reclaims nothing when erased -- and
        /// the driver gets a bind entry no kernel will ask for.
        ///
        /// Say this and it is a name only: still resolved by later
        /// expressions, never finalized, and a temporary the memory planner
        /// may reuse.
        Defined& internal() {
            owner_->set_internal(group_, index_);
            return *this;
        }

    private:
        friend class ModelContract;
        // The slot is named, not pointed at: a group's declarations live in a
        // vector of their own, and a pointer into it would dangle the moment
        // the next `define` grew it.
        Defined(ModelContract* owner, std::size_t group, std::size_t index)
            : owner_(owner), group_(group), index_(index) {}
        ModelContract* owner_;
        std::size_t group_;
        std::size_t index_;
    };

    /// A set of interchangeable declarations, being built.
    ///
    /// Every `define` on this handle describes *one* instance; the loader
    /// compiles all `arity` of them and requires them to differ only in which
    /// bytes they read. The expressions may use `index_src` and `select`, which
    /// stand for the instance and are an error anywhere else.
    ///
    /// The driver decides what a group is for. Running its plan `arity` times
    /// into `arity` destinations makes it resident; running it on demand into a
    /// bounded set of slots is streaming. Neither word appears in a contract.
    class Group {
    public:
        Defined define(std::string name, Node expr, PieLoaderEncodingSpec encoding) {
            return owner_->define_in(index_, std::move(name), expr, encoding);
        }

    private:
        friend class ModelContract;
        Group(ModelContract* owner, std::size_t index) : owner_(owner), index_(index) {}
        ModelContract* owner_;
        std::size_t index_;
    };

    /// Byte alignment every materialized buffer must satisfy. `1` is unaligned.
    ModelContract& align(std::uint32_t alignment) {
        alignment_ = alignment;
        return *this;
    }

    /// A tensor as it appears in the checkpoint.
    Node src(std::string name) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Src);
        node.name = store_name(std::move(name));
        return push(node);
    }

    /// A tensor whose *name* carries the group instance.
    ///
    /// `template_name` holds exactly one `{}`, which is replaced by the index
    /// in decimal. For checkpoints that store each instance as its own tensor:
    /// `"model.layers.3.mlp.experts.{}.gate_proj.weight"`.
    ///
    /// The loader substitutes; it never parses. An author who knows the naming
    /// scheme writes it down once, and the loader is spared a per-architecture
    /// plugin whose job is to rediscover it.
    Node index_src(std::string template_name) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::SrcIndexed);
        node.name = store_name(std::move(template_name));
        return push(node);
    }

    /// The band this instance occupies: `len` along `axis`, starting at
    /// `index * stride`.
    ///
    /// For checkpoints that store every instance in one fused bank. The type is
    /// the same whatever the index is, which is what makes the instances
    /// interchangeable by construction rather than by inspection.
    ///
    /// A `slice` whose start is the only thing the index decides. Reach for it
    /// exactly there: an axis whose extent is the arity.
    Node select(Node src, std::uint8_t axis, std::int64_t stride, std::int64_t len) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Select);
        node.src = src.index_;
        node.axis = axis;
        // The stride rides in `start`, which is the field a `Slice` would have
        // used; an index-parametric start *is* a stride.
        node.start = stride;
        node.len = len;
        return push(node);
    }

    /// A tensor defined earlier in this contract.
    ///
    /// The loader materializes it once and the second reader borrows the
    /// result, which is what makes a fused QKV and a separately-published Q view
    /// cost one copy rather than two.
    Node out(std::string name) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Out);
        node.name = store_name(std::move(name));
        return push(node);
    }

    /// A contiguous band of `len` along `axis`, starting at `start`.
    Node slice(Node src, std::uint8_t axis, std::int64_t start, std::int64_t len) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Slice);
        node.src = src.index_;
        node.axis = axis;
        node.start = start;
        node.len = len;
        return push(node);
    }

    /// A strided run: `len` elements along `axis`, every `step`-th one. `step`
    /// must be at least 2 -- a contiguous run is a `slice`.
    Node stride(Node src, std::uint8_t axis, std::int64_t start, std::int64_t len,
                std::int64_t step) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Stride);
        node.src = src.index_;
        node.axis = axis;
        node.start = start;
        node.len = len;
        node.step = step;
        return push(node);
    }

    /// An explicit selection: the elements at `indices` along `axis`, in the
    /// order given.
    ///
    /// The third and most general placement. `slice` is a run, `stride` is an
    /// arithmetic progression, and this is a list nobody can compute -- and the
    /// loader holds the three apart, refusing a list that either of the cheaper
    /// two could say. Duplicates are allowed; a quantized axis is not.
    ///
    /// Consecutive indices lower to a single byte run, so a permutation of
    /// contiguous blocks costs one run per block rather than one per element.
    Node gather(Node src, std::uint8_t axis, std::vector<std::int64_t> indices) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Gather);
        node.src = src.index_;
        node.axis = axis;
        node.indices = store_shape(std::move(indices));
        return push(node);
    }

    /// Concatenation along `axis`, in the order given.
    Node concat(std::initializer_list<Node> parts, std::uint8_t axis) {
        return concat(std::vector<Node>(parts), axis);
    }

    Node concat(const std::vector<Node>& parts, std::uint8_t axis) {
        std::vector<std::uint32_t> indices;
        indices.reserve(parts.size());
        for (const Node& part : parts) {
            indices.push_back(part.index_);
        }
        const std::vector<std::uint32_t>& stored = part_lists_.emplace_back(std::move(indices));
        PieLoaderExprNode node = blank(PieLoaderExprKind::Concat);
        node.axis = axis;
        node.parts = {stored.empty() ? nullptr : stored.data(), stored.size()};
        return push(node);
    }

    /// The same bytes under a new type: new extents, a new encoding, or both.
    ///
    /// Nothing moves, so the byte count must balance. At most one extent may be
    /// `-1`, and it is solved in elements of the type being named -- `{-1}` over
    /// 64 bytes is 32 elements as bf16 and 128 as a 4-bit code.
    ///
    /// Two further rules follow from the fact that this node only renames:
    ///
    /// * If the element *width* changes, `src` must be a whole tensor (a `src`
    ///   or an `out`). An element offset into a partial view stops meaning
    ///   anything once the element size changes under it.
    /// * If both sides are quantized, the shape from the channel axis onward
    ///   must be unchanged -- only leading axes may regroup. A quantized
    ///   tensor's bytes are packed for the block shape it was quantized at, so
    ///   `{128,128} -> {256,64}` balances but is not a rename.
    Node transmute(Node src, std::vector<std::int64_t> out_shape,
                PieLoaderEncodingSpec out_encoding) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Transmute);
        node.src = src.index_;
        node.out_shape = store_shape(std::move(out_shape));
        node.out_encoding = out_encoding;
        return push(node);
    }

    /// A constant tensor: the third leaf, and the one with no source.
    ///
    /// Zero-extension is written as what it is -- `concat({x, fill(...)}, axis)`
    /// -- rather than as a `pad` node that was a `concat` with a leg it could not
    /// name. The padding therefore has a type, and `concat` checks it against the
    /// data the way it checks every other leg, which a `pad` could not be wrong
    /// about at all.
    ///
    /// One constant is admissible today, and the loader says so rather than
    /// implying it: a fill is realized by zeroing the destination and never
    /// writing there, so `value` must be one that a run of zero bytes denotes.
    /// That rules out `E8M0`, whose zero byte means `2^-127`, and every
    /// quantized encoding -- a code word means nothing without the block scale
    /// beside it, and which code reads as zero is scheme-specific.
    ///
    /// `shape` must give every extent. There is no operand for a `-1` to be
    /// solved against, and the rank is what tells a real fill apart from a node
    /// built by zeroing the struct and never populating it.
    Node fill(float value, std::vector<std::int64_t> shape, PieLoaderEncodingSpec encoding) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Fill);
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        node.fill_bits = bits;
        node.out_shape = store_shape(std::move(shape));
        node.out_encoding = encoding;
        return push(node);
    }

    /// This rank's slice of a TP-split axis.
    ///
    /// The only node whose meaning depends on the rank, and the only one that
    /// needs to: everything downstream sees an expression that means the same
    /// thing everywhere.
    Node shard(Node src, std::uint8_t axis) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Shard);
        node.src = src.index_;
        node.axis = axis;
        return push(node);
    }

    /// A named hardware layout, applied to whatever `src` selects.
    ///
    /// The escape hatch, and only for what genuinely escapes: `layout` names a
    /// kernel and nothing else. Which rows and columns the kernel sees is
    /// `src`'s business -- use `slice`, `shard` and `stride` for that -- and the
    /// loader derives the kernel's geometry from `src`'s inferred type, so a
    /// contract cannot disagree with the tensor it is reading.
    ///
    /// `out_shape` may be *larger* than what `src` holds on either of the last
    /// two axes; the kernel zero-fills the difference, which is how a Marlin
    /// tile size is met. It may not be smaller: a truncation is a `slice`.
    Node repack(Node src, PieLoaderRepackLayout layout, std::vector<std::int64_t> out_shape,
                PieLoaderEncodingSpec out_encoding) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Repack);
        node.src = src.index_;
        node.repack_layout = static_cast<std::uint32_t>(layout);
        node.out_shape = store_shape(std::move(out_shape));
        node.out_encoding = out_encoding;
        return push(node);
    }

    /// The same values in a different representation.
    ///
    /// The exact dual of `transmute`, and the pair is worth reading together: a
    /// transmute preserves the *bytes* and renames them, a cast preserves the
    /// *values* and rewrites them. One is free; this one is a kernel. Rust
    /// spells the same distinction `transmute` and `as`.
    ///
    /// One node covers all three directions, because they are one question --
    /// what representation, not what operation: raw to raw is a numeric cast,
    /// raw to quantized is load-time encoding (and publishes the scale tensor
    /// the encoder computes), quantized to raw is decoding. Shape is preserved
    /// in every direction; a cast has an opinion about elements and none about
    /// placement.
    ///
    /// Re-encoding one quantized scheme directly as another is refused: no
    /// kernel does it. Declare the decoded tensor -- `internal()`, if the
    /// driver has no use for it -- and cast that.
    Node cast(Node src, PieLoaderEncodingSpec to) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Cast);
        node.src = src.index_;
        node.out_encoding = to;
        return push(node);
    }

    /// Multiply every element by a constant.
    ///
    /// The only node that changes a value rather than moving one, and the one
    /// to reach for whenever a bind step would otherwise copy a tensor to the
    /// host, do arithmetic on it and upload the result -- arithmetic done that
    /// way is invisible to the plan, so it is re-done on every load and the
    /// unscaled original stays resident.
    ///
    /// Shape and encoding are preserved, so a scale that also narrows is
    /// written by declaring the narrower type on `define`: the planner derives
    /// the cast from the declaration, and no author spells it. The operand must
    /// be a `src` node -- scaling one leg of a `concat` would have to materialize
    /// that leg first, and the planner rejects it rather than doing so quietly.
    ///
    /// The factor is carried as an fp32 bit pattern. The loader rejects one
    /// that is not finite, and rejects zero -- zero is what this field reads as
    /// when a node is built without setting it, and a tensor scaled to nothing
    /// would otherwise load, cache and run.
    Node scale(Node src, float factor) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Scale);
        node.src = src.index_;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &factor, sizeof(bits));
        node.scale_factor_bits = bits;
        return push(node);
    }

    /// Multiply by a tensor of factors, one per block.
    ///
    /// **The factors' shape is what says how big a block is**, per axis: a
    /// `[256, 512]` payload with `[2, 4]` factors is blocked 128x128, with
    /// `[256, 4]` it is 1x128, and with `[256, 1]` it is one factor per row.
    /// Nothing else states the blocking, so nothing else can contradict it --
    /// this used to take a `group` and an `axis` beside `by`, which said the
    /// same thing twice and, being one number, could only say it about one
    /// axis at all.
    ///
    /// This is dequantization written as a declaration. Over a `src` the
    /// planner reinterprets through `transmute` -- the checkpoint stores packed
    /// codes as bytes, and the contract is what says they are a quantization
    /// scheme -- and the result is the scheme's logical type, so `define`
    /// declares the dtype the driver wants to compute in and nothing else is
    /// spelled.
    ///
    /// `by` must be a node reading a tensor an *earlier* `define` published,
    /// not a fresh `src`. Scales are published anyway, because a driver reads
    /// them for whatever consumes the weight; scaling by that name keeps one
    /// copy of them and makes the pairing something the loader checks rather
    /// than something it guesses from a suffix.
    ///
    /// The loader checks equal rank and that every axis of the payload is a
    /// whole number of blocks, so an author who names the wrong tensor or
    /// shards one side and not the other is told which. Factors shaped exactly
    /// like the payload are refused: that groups nothing, and a factor per
    /// element is an elementwise product.
    Node scale_per_block(Node src, Node by) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Scale);
        node.src = src.index_;
        node.scale_by = by.index_;
        return push(node);
    }

    /// Publish `expr` under `name`, in the encoding the driver wants it in.
    Defined define(std::string name, Node expr, PieLoaderEncodingSpec encoding) {
        return define_in(kNoGroup, std::move(name), expr, encoding);
    }

    /// Open a set of `arity` interchangeable declarations.
    ///
    /// `name` is what the driver asks for the group by; it is not a tensor
    /// name and never reaches the checkpoint.
    Group group(std::string name, std::uint32_t arity) {
        groups_.push_back(GroupState{std::move(name), arity, {}});
        return Group(this, groups_.size() - 1);
    }

    std::size_t size() const { return tensors_.size(); }
    bool empty() const { return tensors_.empty(); }

    /// A borrowed POD snapshot for `PieLoaderContractRequest::contract`.
    /// Invalidated by any later call that mutates the builder.
    PieLoaderModelContractView view() const {
        PieLoaderModelContractView v{};
        v.alignment = alignment_;
        v.nodes = {nodes_.empty() ? nullptr : nodes_.data(), nodes_.size()};
        v.tensors = {tensors_.empty() ? nullptr : tensors_.data(), tensors_.size()};
        groups_view_.clear();
        groups_view_.reserve(groups_.size());
        for (const GroupState& group : groups_) {
            groups_view_.push_back(PieLoaderGroupContractView{
                .name = {reinterpret_cast<const std::uint8_t*>(group.name.data()),
                         group.name.size()},
                .arity = group.arity,
                .tensors = {group.tensors.empty() ? nullptr : group.tensors.data(),
                            group.tensors.size()},
            });
        }
        v.groups = {groups_view_.empty() ? nullptr : groups_view_.data(),
                    groups_view_.size()};
        return v;
    }

private:
    static constexpr std::size_t kNoGroup = static_cast<std::size_t>(-1);

    struct GroupState {
        std::string name;
        std::uint32_t arity;
        std::vector<PieLoaderTensorContractView> tensors;
    };

    /// The declarations a `Defined` refers to: the contract's, or one group's.
    std::vector<PieLoaderTensorContractView>& slots(std::size_t group) {
        return group == kNoGroup ? tensors_ : groups_[group].tensors;
    }

    Defined define_in(std::size_t group, std::string name, Node expr,
                      PieLoaderEncodingSpec encoding) {
        const std::string& stored = names_.emplace_back(std::move(name));
        std::vector<PieLoaderTensorContractView>& into = slots(group);
        into.push_back(PieLoaderTensorContractView{
            .name = {reinterpret_cast<const std::uint8_t*>(stored.data()), stored.size()},
            .root = expr.index_,
            .shape = {nullptr, 0},
            .encoding = encoding,
            .scales = {},
            .visibility = PieLoaderVisibility::Public,
        });
        return Defined(this, group, into.size() - 1);
    }

    static PieLoaderExprNode blank(PieLoaderExprKind kind) {
        PieLoaderExprNode node{};
        node.kind = static_cast<std::uint32_t>(kind);
        node.src = PIE_LOADER_NO_NODE;
        node.step = 1;
        node.out_encoding = raw(PieLoaderDType::BF16);
        node.repack_layout = static_cast<std::uint32_t>(PieLoaderRepackLayout::None);
        return node;
    }

    Node push(const PieLoaderExprNode& node) {
        nodes_.push_back(node);
        return Node(static_cast<std::uint32_t>(nodes_.size() - 1));
    }

    void set_shape(std::size_t group, std::size_t tensor, std::vector<std::int64_t> shape) {
        slots(group)[tensor].shape = store_shape(std::move(shape));
    }

    void set_scales(std::size_t group, std::size_t tensor, std::string of,
                    PieLoaderQuantGranularity granularity, std::uint32_t group_size,
                    std::uint32_t channel_axis, PieLoaderScaleForm form) {
        const std::string& stored = names_.emplace_back(std::move(of));
        slots(group)[tensor].scales = PieLoaderScalesView{
            .of = {reinterpret_cast<const std::uint8_t*>(stored.data()), stored.size()},
            .granularity = static_cast<std::uint32_t>(granularity),
            .group_size = group_size,
            .channel_axis = channel_axis,
            .form = static_cast<std::uint32_t>(form),
        };
    }

    void set_internal(std::size_t group, std::size_t tensor) {
        slots(group)[tensor].visibility = PieLoaderVisibility::Internal;
    }

    // `deque` rather than `vector` for the backing stores: the POD nodes point
    // into them, and a vector would dangle every previously-handed-out pointer
    // when it grew. `nodes_` and `tensors_` are vectors because nothing points
    // *into* them — the view borrows them whole, once, at the end.
    PieLoaderBytes store_name(std::string name) {
        const std::string& stored = names_.emplace_back(std::move(name));
        return {reinterpret_cast<const std::uint8_t*>(stored.data()), stored.size()};
    }

    PieLoaderI64Slice store_shape(std::vector<std::int64_t> shape) {
        const std::vector<std::int64_t>& stored = shapes_.emplace_back(std::move(shape));
        return {stored.empty() ? nullptr : stored.data(), stored.size()};
    }

    std::deque<std::string> names_;
    std::deque<std::vector<std::int64_t>> shapes_;
    std::deque<std::vector<std::uint32_t>> part_lists_;
    std::vector<PieLoaderExprNode> nodes_;
    std::vector<PieLoaderTensorContractView> tensors_;
    // `deque`, because a `Defined` outlives the push that follows it and the
    // group it names must not move.
    std::deque<GroupState> groups_;
    // Rebuilt by `view()`, which is why it is mutable: a view is a snapshot,
    // and the builder's own storage is the group states above.
    mutable std::vector<PieLoaderGroupContractView> groups_view_;
    std::uint32_t alignment_ = 256;
};

}  // namespace pie_loader
