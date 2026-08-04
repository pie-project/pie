#pragma once

/// What DeepSeek-V4 binds (`model/registry.cpp` row).
///
/// The only family with its own tensor-parallel shard-axis rule: its experts
/// are named `.ffn.experts.w1/w2/w3` rather than `.mlp.experts.gate/up/down`,
/// and the intermediate dim is split within each expert so every rank computes
/// a partial expert output that an all-reduce combines.

#include <stdexcept>

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

inline ShardAxis dsv4_shard_axis(std::string_view name) {
    // Routed experts: shard the intermediate dim within each expert. w1/w3 on
    // axis 0 (gate/up out dim), w2 on axis 1 (down in dim). Each rank computes
    // a partial expert output; the results are combined by an all-reduce.
    // Weights only. A companion scale reaches here already rewritten to the
    // weight it scales -- `ContractBuilder::shard_axis` strips the suffix
    // before consulting this -- so listing scales again is how the two lists
    // drift apart.
    if (contains(name, ".ffn.experts.")) {
        if (ends_with_any(name, {".w1.weight", ".w3.weight"})) {
            return std::uint8_t{0};
        }
        if (ends_with(name, ".w2.weight")) {
            return std::uint8_t{1};
        }
    }
    if (ends_with_any(name, {".shared_experts.w1.weight",
                             ".shared_experts.w3.weight"})) {
        return std::uint8_t{0};
    }
    if (ends_with(name, ".shared_experts.w2.weight")) {
        return std::uint8_t{1};
    }
    // Inside the FFN, replication is never the answer, so falling through to
    // it is a bug rather than a default. Every projection here is split and
    // all-reduced; a name this function does not recognise -- a checkpoint
    // variant that spells an expert differently, say -- would otherwise be
    // replicated silently while the forward went on sharding around it, and
    // the model would answer plausibly and wrongly. Say so instead.
    //
    // Scales never reach here: `ContractBuilder::shard_axis` rewrites them to
    // the weight they scale first.
    if (contains(name, ".ffn.") && ends_with(name, ".weight") &&
        !contains(name, ".gate.") && !contains(name, "_norm.") &&
        !contains(name, "layernorm")) {
        throw std::runtime_error(
            "deepseek_v4: no sharding decision for FFN tensor '" +
            std::string(name) +
            "'; add it to dsv4_shard_axis rather than letting it replicate");
    }
    // Everything outside the FFN is replicated, which avoids TP communication
    // in the main path.
    return std::nullopt;
}

/// Decode the block scales that ride beside DeepSeek-V4's FP8 weights.
///
/// The checkpoint stores one byte per tile of the weight, and that byte is OCP
/// Microscaling's E8M0: it denotes `2^(b - 127)`. The FP8 GEMM wants fp32.
/// Both halves of that are sayable -- `Bitcast` names how the bytes are to be
/// read, and the declaration names the type wanted -- so the planner inserts
/// the cast itself.
///
/// `QuantScheme::Mxfp4E2M1E8M0` cannot name this pairing: that symbol bundles
/// the element format together with the scale format, and E8M0 beside
/// FP8-E4M3 is a combination it has no name for. Which is why the driver used
/// to copy every scale to the host, run `ldexpf` over it and upload the result
/// during bind -- arithmetic the algebra has no vocabulary for, done outside
/// the loader entirely.
inline void dsv4_block_scales_to_fp32(ContractBuilder& b) {
    constexpr std::string_view kSuffix = ".scale";
    for (const SourceTensor& raw : b.tensors()) {
        if (!contract_detail::ends_with(raw.name, kSuffix) ||
            !contract_detail::is_raw(raw.encoding, PieLoaderDType::U8)) {
            continue;
        }
        // Only a companion to a **block-FP8** weight is an fp32-bound E8M0
        // exponent. A `.scale` beside anything else is some other convention,
        // and guessing is how a scale tensor gets silently reinterpreted.
        //
        // DeepSeek-V4 ships exactly two quantizations, and only one of them
        // belongs here. The dense and shared paths store F8E4M3 weights with
        // 128x128 block scales, which the FP8 GEMM wants as fp32 -- 50 tensors
        // on both minis. The routed experts store **packed MXFP4**: an `I8`
        // tensor holding two E2M1 nibbles per byte, whose E8M0 scales
        // (`[rows, cols/32]`) `launch_dequant_mxfp4_to_bf16` consumes as raw
        // bytes -- 144 tensors. Widening this guard to I8 hands that kernel
        // fp32 words to read as exponents, and the routed experts come out
        // four orders of magnitude too large.
        const std::string weight =
            std::string(raw.name.substr(0, raw.name.size() - kSuffix.size())) + ".weight";
        const SourceTensor* companion = b.find(weight);
        if (companion == nullptr ||
            !contract_detail::is_raw(companion->encoding, PieLoaderDType::F8E4M3)) {
            continue;
        }
        std::vector<std::int64_t> shape = contract_detail::shape_of(raw);
        // Declaring F32 over the transmute was the original bug, and the
        // reason is worth keeping: the transmute says how to *read* the stored
        // bytes -- one E8M0 exponent each -- so F32 over it relabels a 1-byte
        // element as a 4-byte one, and the algebra rejects it. That is how
        // this was found the first time a real DeepSeek-V4 checkpoint was
        // compiled.
        //
        // Spelling the widening as a `cast` fixes the type and cannot survive
        // tensor parallelism. A cast is a kernel, so it leaves the affine
        // fragment that `contract::compile` lowers into byte runs, and a
        // `Shard` can no longer slice it: nested under the shard, compile
        // refuses it; hoisted above, the executor casts compact sources only;
        // routed through an `internal()` tensor, a strided slice of a computed
        // buffer is refused in turn. At TP=1 there is no shard, so all three
        // stay hidden.
        //
        // The widening does not belong here at all. `scaling(..., F32Factors)`
        // below is the request, and `LoadPlanExecutor` answers it with
        // `convert_block_scale_to_f32` once the tensor is materialised,
        // writing the `.f32` companion the FP8 GEMM binds to. So this pass
        // publishes the exponent bytes and nothing more -- a rename and a
        // shard, which shards at any degree.
        //
        // U8 rather than E8M0, and that part is not cosmetic: the expansion
        // dispatches on the tensor's dtype and knows only UINT8 and BF16, so
        // an E8M0 tensor falls through its `return` and the GEMM is handed
        // exponent bytes it refuses by name. U8 is what every other E8M0 scale
        // in the tree is declared as.
        auto [expr, local] = b.shard(b.contract().src(std::string(raw.name)),
                                     shape, b.shard_axis(raw.name));
        auto defined = b.define(b.output_name(raw.name), expr,
                                pie_loader::raw(PieLoaderDType::U8), std::move(local));
        // The pairing this loop just established, stated rather than dropped.
        // The loader used to rediscover it by appending `_scale_inv` and then
        // `.scale` to every F8E4M3 tensor's name, with `group_size` hardcoded to
        // 128 -- a guess about the checkpoint made three layers away from the
        // only code that checks it. Here both shapes are in hand, so the block
        // size is read off them instead of assumed.
        const std::vector<std::int64_t> weight_shape = contract_detail::shape_of(*companion);
        if (defined.has_value() && !shape.empty() && shape.back() > 0 &&
            !weight_shape.empty()) {
            const std::int64_t block = weight_shape.back() / shape.back();
            if (block > 0) {
                defined->scaling(b.output_name(weight), PieLoaderQuantGranularity::PerGroup,
                                 static_cast<std::uint32_t>(block), 0,
                                 PieLoaderScaleForm::F32Factors);
            }
        }
        b.consume(raw.id);
    }
}

/// Dequantize DeepSeek-V4's routed experts and stack them, at load time.
///
/// The forward pass wants two dense bf16 slabs per layer -- `[E, 2I, H]` gate
/// over up, and `[E, H, I]` down -- because a batched GEMM over experts wants
/// one base pointer and a stride, not 3E separate tensors. The checkpoint
/// stores the other thing: per expert, `w1`/`w2`/`w3` as packed MXFP4 (an `I8`
/// tensor holding two E2M1 nibbles per byte) beside E8M0 block scales.
///
/// Both halves of the gap are sayable. `Bitcast` names the packed bytes as the
/// quantization they are; `Concat` and `Reshape` build the slab; and
/// `scale_per_block` multiplies each group of 32 elements by its own factor,
/// which over a quantized operand *is* the dequantization. Sharding sits
/// inside all of it, so each rank dequantizes only the slice it keeps.
///
/// The order matters: the concatenation is *inside* the scale, not outside.
/// One scale node over the whole slab is one instruction per slab whose packed
/// input is a temporary the memory planner can reuse; E separate scales
/// concatenated afterwards would not typecheck, and would have kept the packed
/// originals resident besides.
///
/// This ran during bind until the algebra could state it: a loop calling
/// `launch_dequant_mxfp4_to_bf16` 3E times per layer, into slabs allocated
/// outside the storage arena, with the packed originals left resident because
/// every loaded weight is a view into that one arena and dropping a view
/// reclaims nothing.
inline void dsv4_bf16_expert_stacks(ContractBuilder& b) {
    using contract_detail::fail;
    using contract_detail::is_raw;
    using contract_detail::shape_of;
    constexpr std::int64_t kGroup = 32;

    // The layer and expert counts are read off the names that are actually
    // present rather than off the config: a `Component` build holds a slice of
    // the checkpoint, and a config-driven loop would ask for tensors that this
    // process was never given.
    for (std::uint32_t layer = 0;; ++layer) {
        const std::string ffn = std::string(b.decoder_layer_prefix()) +
                                std::to_string(layer) + ".ffn.";
        if (b.find(b.source_name(ffn + "experts.0.w1.weight")) == nullptr) {
            break;
        }
        std::vector<Node> gate_up, gate_up_scales, down, down_scales;
        std::vector<std::uint32_t> consumed;
        std::int64_t local_inter = 0;
        std::int64_t hidden = 0;

        for (std::uint32_t expert = 0;; ++expert) {
            const std::string ep = ffn + "experts." + std::to_string(expert) + ".";
            if (b.find(b.source_name(ep + "w1.weight")) == nullptr) {
                break;
            }
            const SourceTensor* parts[6] = {
                b.find(b.source_name(ep + "w1.weight")),
                b.find(b.source_name(ep + "w1.scale")),
                b.find(b.source_name(ep + "w3.weight")),
                b.find(b.source_name(ep + "w3.scale")),
                b.find(b.source_name(ep + "w2.weight")),
                b.find(b.source_name(ep + "w2.scale")),
            };
            for (const SourceTensor* part : parts) {
                if (part == nullptr) {
                    fail("deepseek_v4 expert stack: " + ep + " is missing a weight or scale");
                }
            }
            // Packed nibbles are stored as `I8`, two elements per byte. A
            // checkpoint that stores its experts some other way is not this
            // pass's to rewrite, and stacking it anyway would hand the GEMM
            // garbage -- so leave the whole checkpoint alone rather than half
            // of it, and let `author_dense_contract` publish the experts as
            // they are.
            if (!is_raw(parts[0]->encoding, PieLoaderDType::I8) ||
                !is_raw(parts[4]->encoding, PieLoaderDType::I8)) {
                return;
            }

            // `w1`/`w3` are `[I_full, H/2]` packed; `w2` is `[H, I_full/2]`.
            // The logical shapes the transmutes declare unpack the last axis.
            const std::vector<std::int64_t> up_raw = shape_of(*parts[0]);
            const std::vector<std::int64_t> down_raw = shape_of(*parts[4]);
            if (up_raw.size() != 2 || down_raw.size() != 2) {
                fail("deepseek_v4 expert stack: " + ep + " expects rank-2 expert weights");
            }
            const std::int64_t inter_full = up_raw[0];
            const std::int64_t h = up_raw[1] * 2;
            const std::int64_t inter = b.local_extent(inter_full);
            if (h % kGroup != 0 || inter % kGroup != 0) {
                fail("deepseek_v4 expert stack: " + ep +
                     " expects both expert dims to be a multiple of 32");
            }
            if (local_inter != 0 && (inter != local_inter || h != hidden)) {
                fail("deepseek_v4 expert stack: " + ep + " disagrees with its siblings on shape");
            }
            local_inter = inter;
            hidden = h;

            // Every leg is declared rank 3 with a leading 1, so that the
            // outer concatenation over axis 0 is a stack. `Bitcast` is what
            // carries the rank lift: it already says "read these bytes as this
            // shape and this type", so nothing has to reshape a packed tensor
            // afterwards -- and reshaping one is not meaningful anyway, since
            // the byte layout is a function of the shape it was packed for.
            //
            // `w1`/`w3` shard the out dim and `w2` the in dim -- the same
            // split `dsv4_shard_axis` states, applied to the logical shapes
            // rather than the packed ones, which is why it is spelled out
            // here.
            auto packed = [&](const SourceTensor& raw, std::vector<std::int64_t> shape,
                              std::uint8_t axis) {
                return b.shard(b.contract().transmute(
                                   b.contract().src(std::string(raw.name)), shape,
                                   mxfp4_encoding(b.contract(), static_cast<std::uint8_t>(2))),
                               shape, ShardAxis{axis})
                    .first;
            };
            auto factors = [&](const SourceTensor& raw, std::vector<std::int64_t> shape,
                               std::uint8_t axis) {
                return b.shard(b.contract().transmute(b.contract().src(std::string(raw.name)),
                                                    shape,
                                                    pie_loader::raw(PieLoaderDType::E8M0)),
                               shape, ShardAxis{axis})
                    .first;
            };

            gate_up.push_back(
                b.contract().concat({packed(*parts[0], {1, inter_full, h}, 1),
                                  packed(*parts[2], {1, inter_full, h}, 1)},
                                 1));
            gate_up_scales.push_back(
                b.contract().concat({factors(*parts[1], {1, inter_full, h / kGroup}, 1),
                                  factors(*parts[3], {1, inter_full, h / kGroup}, 1)},
                                 1));
            down.push_back(packed(*parts[4], {1, down_raw[0], inter_full}, 2));
            down_scales.push_back(
                factors(*parts[5], {1, down_raw[0], inter_full / kGroup}, 2));
            for (const SourceTensor* part : parts) {
                consumed.push_back(part->id);
            }
        }
        if (gate_up.empty()) {
            continue;
        }
        const auto experts = static_cast<std::int64_t>(gate_up.size());

        // Named but not bound. `scale_per_block` takes its factors by output
        // name -- a scale is a tensor the contract declared, not a companion
        // the lowering guesses at from a suffix -- and the stacked slab is
        // dequantized here, so no kernel ever reads these again. Left public
        // they would sit in the persistent arena for the process's lifetime,
        // one E8M0 byte per 32 weights, and appear in the driver's bind table
        // under a name nothing asks for.
        const std::string gu_scale = ffn + "experts.gate_up.scale";
        const std::string dn_scale = ffn + "experts.down.scale";
        if (auto gu = b.define(gu_scale, b.contract().concat(gate_up_scales, 0),
                               pie_loader::raw(PieLoaderDType::E8M0),
                               std::vector<std::int64_t>{experts, 2 * local_inter,
                                                         hidden / kGroup})) {
            gu->internal();
        }
        if (auto dn = b.define(dn_scale, b.contract().concat(down_scales, 0),
                               pie_loader::raw(PieLoaderDType::E8M0),
                               std::vector<std::int64_t>{experts, hidden,
                                                         local_inter / kGroup})) {
            dn->internal();
        }
        b.define(ffn + "experts.gate_up.weight",
                 b.contract().scale_per_block(b.contract().concat(gate_up, 0),
                                              b.contract().out(gu_scale)),
                 pie_loader::raw(PieLoaderDType::BF16),
                 std::vector<std::int64_t>{experts, 2 * local_inter, hidden});
        b.define(ffn + "experts.down.weight",
                 b.contract().scale_per_block(b.contract().concat(down, 0),
                                              b.contract().out(dn_scale)),
                 pie_loader::raw(PieLoaderDType::BF16),
                 std::vector<std::int64_t>{experts, hidden, local_inter});
        for (std::uint32_t id : consumed) {
            b.consume(id);
        }
    }
}

/// The same experts, as a group per layer, for a driver that will page them.
///
/// `dsv4_bf16_expert_stacks` says "every expert of this layer, concatenated".
/// This says "one expert of this layer", once, with the expert index left
/// standing -- which is the same sentence with the outer `concat` removed and
/// each `src` replaced by the `index_src` it was a member of. Everything that
/// made the stack correct is still here in the same order: the transmutes that
/// name packed nibbles as MXFP4, the shard that keeps only this rank's slice,
/// and the `scale_per_block` that dequantizes. A group's plan is a whole plan,
/// so all of it runs on the page-in path; nothing had to be reduced to plain
/// copies to be streamable.
///
/// One group per layer rather than one for the model, because a name template
/// carries a single `{}` and this checkpoint spells an expert
/// `layers.<L>.ffn.experts.<E>.w1`. That is the contract vocabulary declining
/// to grow a second index, and it costs nothing: the driver's slab spans the
/// groups.
///
/// The declared names carry no layer and no expert -- there is one plan, and
/// it is the same plan for every instance of every layer. The driver asks a
/// group by name and gets back `gate_up.weight` and `down.weight`, which is
/// what the per-expert GEMM path wants anyway.
inline void dsv4_streamed_expert_groups(ContractBuilder& b) {
    using contract_detail::fail;
    using contract_detail::is_raw;
    using contract_detail::shape_of;
    constexpr std::int64_t kGroup = 32;

    for (std::uint32_t layer = 0;; ++layer) {
        const std::string ffn = std::string(b.decoder_layer_prefix()) +
                                std::to_string(layer) + ".ffn.";
        if (b.find(b.source_name(ffn + "experts.0.w1.weight")) == nullptr) {
            break;
        }

        // Instance 0 is the shape oracle. It has to be: the group's plan is
        // compiled at index 0 and every other instance is then required to
        // match it, so a shape read anywhere else would be a shape the loader
        // is about to reject anyway.
        const SourceTensor* proto[6] = {
            b.find(b.source_name(ffn + "experts.0.w1.weight")),
            b.find(b.source_name(ffn + "experts.0.w1.scale")),
            b.find(b.source_name(ffn + "experts.0.w3.weight")),
            b.find(b.source_name(ffn + "experts.0.w3.scale")),
            b.find(b.source_name(ffn + "experts.0.w2.weight")),
            b.find(b.source_name(ffn + "experts.0.w2.scale")),
        };
        for (const SourceTensor* part : proto) {
            if (part == nullptr) {
                fail("deepseek_v4 expert group: " + ffn +
                     "experts.0 is missing a weight or scale");
            }
        }
        if (!is_raw(proto[0]->encoding, PieLoaderDType::I8) ||
            !is_raw(proto[4]->encoding, PieLoaderDType::I8)) {
            // Not packed MXFP4. Leave the whole checkpoint alone rather than
            // half of it, exactly as the stacking pass does.
            return;
        }

        const std::vector<std::int64_t> up_raw = shape_of(*proto[0]);
        const std::vector<std::int64_t> down_raw = shape_of(*proto[4]);
        if (up_raw.size() != 2 || down_raw.size() != 2) {
            fail("deepseek_v4 expert group: " + ffn +
                 "experts.0 expects rank-2 expert weights");
        }
        const std::int64_t inter_full = up_raw[0];
        const std::int64_t hidden = up_raw[1] * 2;
        const std::int64_t inter = b.local_extent(inter_full);
        if (hidden % kGroup != 0 || inter % kGroup != 0) {
            fail("deepseek_v4 expert group: " + ffn +
                 "experts.0 expects both expert dims to be a multiple of 32");
        }

        // Count the experts, and claim every tensor all of them read. A group
        // reads through a template, so the sources it consumes are not named
        // by any node and `author_dense_contract` would otherwise publish
        // every one of them.
        std::uint32_t experts = 0;
        std::vector<std::uint32_t> consumed;
        for (;; ++experts) {
            const std::string ep =
                ffn + "experts." + std::to_string(experts) + ".";
            const SourceTensor* w1 = b.find(b.source_name(ep + "w1.weight"));
            if (w1 == nullptr) break;
            for (const char* suffix : {"w1.weight", "w1.scale", "w3.weight",
                                       "w3.scale", "w2.weight", "w2.scale"}) {
                const SourceTensor* part = b.find(b.source_name(ep + suffix));
                if (part == nullptr) {
                    fail("deepseek_v4 expert group: " + ep +
                         " is missing a weight or scale");
                }
                consumed.push_back(part->id);
            }
        }
        if (experts == 0) continue;

        auto& c = b.contract();
        auto packed = [&](const std::string& tmpl, std::vector<std::int64_t> shape,
                          std::uint8_t axis) {
            return b.shard(c.transmute(c.index_src(b.source_name(ffn + tmpl)), shape,
                                       mxfp4_encoding(c, static_cast<std::uint8_t>(2))),
                           shape, ShardAxis{axis})
                .first;
        };
        auto factors = [&](const std::string& tmpl, std::vector<std::int64_t> shape,
                           std::uint8_t axis) {
            return b.shard(c.transmute(c.index_src(b.source_name(ffn + tmpl)), shape,
                                       pie_loader::raw(PieLoaderDType::E8M0)),
                           shape, ShardAxis{axis})
                .first;
        };

        // Named the way every bound tensor is named -- prefix stripped --
        // because the driver looks a group up beside the weights it binds, and
        // one naming convention for both is one less thing to get wrong.
        auto group = c.group(b.output_name(ffn + "experts"), experts);
        // Same structure as the stack, one rank lower: with no expert axis to
        // stack along, `w1`/`w3` concatenate along the intermediate dim they
        // were already sharded on, and the scale groups run along the last
        // axis as before.
        group.define("gate_up.scale",
                     c.concat({factors("experts.{}.w1.scale",
                                       {inter_full, hidden / kGroup}, 0),
                               factors("experts.{}.w3.scale",
                                       {inter_full, hidden / kGroup}, 0)},
                              0),
                     pie_loader::raw(PieLoaderDType::E8M0))
            .expect({2 * inter, hidden / kGroup})
            .internal();
        group.define("down.scale",
                     factors("experts.{}.w2.scale",
                             {down_raw[0], inter_full / kGroup}, 1),
                     pie_loader::raw(PieLoaderDType::E8M0))
            .expect({down_raw[0], inter / kGroup})
            .internal();
        group.define(
                 "gate_up.weight",
                 c.scale_per_block(
                     c.concat({packed("experts.{}.w1.weight", {inter_full, hidden}, 0),
                               packed("experts.{}.w3.weight", {inter_full, hidden}, 0)},
                              0),
                     c.out("gate_up.scale")),
                 pie_loader::raw(PieLoaderDType::BF16))
            .expect({2 * inter, hidden});
        group.define("down.weight",
                     c.scale_per_block(
                         packed("experts.{}.w2.weight", {down_raw[0], inter_full}, 1),
                         c.out("down.scale")),
                     pie_loader::raw(PieLoaderDType::BF16))
            .expect({down_raw[0], inter});

        for (std::uint32_t id : consumed) {
            b.consume(id);
        }
    }
}

}  // namespace contract_detail

/// DeepSeek-V4.
///
/// The routed experts are handed to the driver one of two ways, and this
/// contract picks which. Dequantized and stacked -- `dsv4_bf16_expert_stacks`
/// -- is what the batched expert GEMM wants and is the default. Left packed is
/// the fallback: the forward pass dequantizes a slice per step instead, which
/// is correct and slow, and is worth taking only when the caller asks for it
/// to hold memory down.
///
/// The device rule `resolve_mxfp4_moe` applies is not this family's rule.
/// That rule asks whether there are native MXFP4 GEMM kernels to fall back
/// *from*, and DeepSeek-V4 has none -- so on a device without them it would
/// answer `RoutedDecode`, picking the per-step path on grounds that say
/// nothing about this family. The choice here is between eager and per-step,
/// so anything short of an explicit `RoutedDecode` takes the eager path.
inline void author_deepseek_v4_contract(ContractBuilder& b) {
    // Ask where the layers are rather than declare it. This family's released
    // checkpoints name them `layers.<L>.`, not the HF `model.layers.<L>.` the
    // default assumes, and the two expert passes below select by that prefix --
    // so with it wrong they matched nothing and the forward pass's packed
    // fallback quietly covered for them, dequantizing every routed expert on
    // every layer of every step. That is the failure `decoder_layer_prefix_any_of`
    // was written for, and the same silence would have hidden it here.
    b.decoder_layer_prefix_any_of({"model.layers.", "layers."});
    b.shard_axis_fn(contract_detail::dsv4_shard_axis);
    b.decide_mxfp4_moe(b.mxfp4_moe_request() == Mxfp4MoeRequest::RoutedDecode
                           ? Mxfp4MoePolicy::RoutedDecode
                           : Mxfp4MoePolicy::EagerBf16);
    if (b.stream_routed_experts()) {
        // Streaming and stacking are alternatives, not layers: a stack is one
        // slab per layer holding every expert, which is exactly the residency
        // the slab is there to avoid.
        contract_detail::dsv4_streamed_expert_groups(b);
    } else if (b.mxfp4_moe() == Mxfp4MoePolicy::EagerBf16) {
        contract_detail::dsv4_bf16_expert_stacks(b);
    }
    contract_detail::dsv4_block_scales_to_fp32(b);
    author_dense_contract(b);
}
}  // namespace pie_cuda_driver::model
