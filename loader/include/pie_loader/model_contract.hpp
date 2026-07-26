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
/// Ten node kinds, mirroring `crate::contract::Expr` one for one:
///
/// | Node                          | Meaning                                          |
/// |-------------------------------|--------------------------------------------------|
/// | `src(name)`                   | A tensor as it appears in the checkpoint          |
/// | `out(name)`                   | An earlier `define` in this contract              |
/// | `slice(x, axis, start, len)`  | A contiguous or strided run along one axis        |
/// | `cat({a, b, c}, axis)`        | Concatenation, in the order given                 |
/// | `reshape(x, shape)`           | Same bytes, new extents; one `-1` is inferred     |
/// | `pad(x, axis, before, after)` | Grow an axis with zeros                           |
/// | `shard(x, axis)`              | This rank's slice of a TP-split axis              |
/// | `repack(x, spec, out)`        | An opaque kernel-specific relayout                |
/// | `quantize(x, spec)`           | Encode into a quantized format                    |
/// | `bitcast(x, out)`             | Reinterpret bytes as another type                 |
///
/// `shard` is the only node that means different things on different ranks, and
/// it is resolved before anything downstream of it runs — which is why a
/// contract is written once for the whole TP group rather than once per rank.
///
/// # Shapes are optional, encodings are not
///
/// `define(name, expr, encoding)` states what the tensor should *be*; the loader
/// inserts whatever cast, decode or encode reaches it. The shape is different:
/// it is a prediction the loader checks against what the expression actually
/// yields, so `expect(shape)` turns a driver whose model of the checkpoint is
/// wrong into a compile error instead of a plausible-looking wrong buffer.
///
/// A prediction may be declined. Omitting `expect` is the honest declaration for
/// a packed quantized weight, whose on-disk extents belong to the quantizer that
/// produced the file and not to the model.
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
///          c.shard(c.cat({q, k, v}, 0), 0),
///          pie_loader::raw(PieLoaderDType::BF16))
///     .expect({(Hq + 2 * Hkv) / tp_size, hidden});
/// ```

#include <cstddef>
#include <cstdint>
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
/// a `cat` of three `slice`s of one `src` reads the source once.
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
    spec.quant.scale_dtype = PIE_LOADER_NO_DTYPE;
    spec.quant.zero_point_dtype = PIE_LOADER_NO_DTYPE;
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
    spec.scale_dtype = PIE_LOADER_NO_DTYPE;
    spec.zero_point_dtype = PIE_LOADER_NO_DTYPE;
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
            owner_->set_shape(index_, std::move(shape));
            return *this;
        }

    private:
        friend class ModelContract;
        Defined(ModelContract* owner, std::size_t index) : owner_(owner), index_(index) {}
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

    /// A contiguous run of `len` along `axis`, starting at `start`.
    Node slice(Node src, std::uint8_t axis, std::int64_t start, std::int64_t len) {
        return strided_slice(src, axis, start, len, 1);
    }

    /// A strided run: `len` elements, every `step`-th one.
    Node strided_slice(Node src, std::uint8_t axis, std::int64_t start, std::int64_t len,
                       std::int64_t step) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Slice);
        node.src = src.index_;
        node.axis = axis;
        node.start = start;
        node.len = len;
        node.step = step;
        return push(node);
    }

    /// Concatenation along `axis`, in the order given.
    Node cat(std::initializer_list<Node> parts, std::uint8_t axis) {
        return cat(std::vector<Node>(parts), axis);
    }

    Node cat(const std::vector<Node>& parts, std::uint8_t axis) {
        std::vector<std::uint32_t> indices;
        indices.reserve(parts.size());
        for (const Node& part : parts) {
            indices.push_back(part.index_);
        }
        const std::vector<std::uint32_t>& stored = part_lists_.emplace_back(std::move(indices));
        PieLoaderExprNode node = blank(PieLoaderExprKind::Cat);
        node.axis = axis;
        node.parts = {stored.empty() ? nullptr : stored.data(), stored.size()};
        return push(node);
    }

    /// Same bytes, new extents. At most one extent may be `-1`.
    Node reshape(Node src, std::vector<std::int64_t> shape) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Reshape);
        node.src = src.index_;
        node.shape = store_shape(std::move(shape));
        return push(node);
    }

    /// Grow `axis` with zeros.
    Node pad(Node src, std::uint8_t axis, std::int64_t before, std::int64_t after) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Pad);
        node.src = src.index_;
        node.axis = axis;
        node.before = before;
        node.after = after;
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

    /// An opaque kernel-specific relayout.
    ///
    /// The type checker cannot see through a repack, so `out_shape` and
    /// `out_encoding` are taken on trust and are the driver's responsibility.
    Node repack(Node src, PieLoaderRepackSpecView spec, std::vector<std::int64_t> out_shape,
                PieLoaderEncodingSpec out_encoding) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Repack);
        node.src = src.index_;
        node.repack = spec;
        node.out_shape = store_shape(std::move(out_shape));
        node.out_encoding = out_encoding;
        return push(node);
    }

    /// Encode into a quantized format.
    Node quantize(Node src, PieLoaderQuantSpecView spec) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Quantize);
        node.src = src.index_;
        node.quant = spec;
        return push(node);
    }

    /// Reinterpret the same bytes as another type. Also opaque to the checker.
    Node bitcast(Node src, std::vector<std::int64_t> out_shape,
                 PieLoaderEncodingSpec out_encoding) {
        PieLoaderExprNode node = blank(PieLoaderExprKind::Bitcast);
        node.src = src.index_;
        node.out_shape = store_shape(std::move(out_shape));
        node.out_encoding = out_encoding;
        return push(node);
    }

    /// Publish `expr` under `name`, in the encoding the driver wants it in.
    Defined define(std::string name, Node expr, PieLoaderEncodingSpec encoding) {
        const std::string& stored = names_.emplace_back(std::move(name));
        tensors_.push_back(PieLoaderTensorContractView{
            .name = {reinterpret_cast<const std::uint8_t*>(stored.data()), stored.size()},
            .root = expr.index_,
            .shape = {nullptr, 0},
            .encoding = encoding,
        });
        return Defined(this, tensors_.size() - 1);
    }

    std::size_t size() const { return tensors_.size(); }
    bool empty() const { return tensors_.empty(); }

    /// A borrowed POD snapshot for `PieLoaderContractRequest::contract`.
    /// Invalidated by any later call that mutates the builder.
    PieLoaderModelContractView view() const {
        PieLoaderModelContractView v{};
        v.abi_version = kAbiVersion;
        v.alignment = alignment_;
        v.nodes = {nodes_.empty() ? nullptr : nodes_.data(), nodes_.size()};
        v.tensors = {tensors_.empty() ? nullptr : tensors_.data(), tensors_.size()};
        return v;
    }

    /// Attach a block shape to `spec`, backed by this contract's storage.
    ///
    /// `PieLoaderQuantSpecView::block_shape` is a borrowed slice, so an author
    /// cannot hand it a local: the contract has to own the numbers for as long
    /// as the view is live. Everything else on the spec is a scalar the author
    /// can just assign.
    PieLoaderQuantSpecView with_block_shape(PieLoaderQuantSpecView spec,
                                            std::vector<std::int64_t> block_shape) {
        spec.block_shape = store_shape(std::move(block_shape));
        return spec;
    }

private:
    static constexpr std::uint32_t kAbiVersion = 1;

    static PieLoaderExprNode blank(PieLoaderExprKind kind) {
        PieLoaderExprNode node{};
        node.kind = static_cast<std::uint32_t>(kind);
        node.src = PIE_LOADER_NO_NODE;
        node.step = 1;
        node.out_encoding = raw(PieLoaderDType::BF16);
        node.repack.layout = static_cast<std::uint32_t>(PieLoaderRepackLayout::None);
        node.repack.row_map = static_cast<std::uint32_t>(PieLoaderRowMap::Identity);
        node.quant = quant_spec(PieLoaderQuantScheme::None, PieLoaderDType::BF16);
        return node;
    }

    Node push(const PieLoaderExprNode& node) {
        nodes_.push_back(node);
        return Node(static_cast<std::uint32_t>(nodes_.size() - 1));
    }

    void set_shape(std::size_t tensor, std::vector<std::int64_t> shape) {
        tensors_[tensor].shape = store_shape(std::move(shape));
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
    std::uint32_t alignment_ = 256;
};

}  // namespace pie_loader
