#pragma once

/// What CSM binds (`model/registry.cpp` row).
///
/// One family-wide fact: every kernel in the backbone, the depth decoder and
/// the Mimi codec reads bf16, and the published checkpoints ship fp32.

#include "model/contract.hpp"

namespace pie_cuda_driver::model {
namespace contract_detail {

/// Ask for bf16 everywhere, because that is the only thing CSM's kernels read.
///
/// The published checkpoints are fp32 throughout, and every tensor `bind_csm`
/// resolves goes through the same accessor -- there is no CSM weight that
/// wants to stay fp32. So the narrowing is a property of the family, and the
/// right place to say it is once, here.
///
/// The narrowing is written as a `cast`, which is what it is: the values are
/// preserved and their representation is not, so it costs a kernel. Saying it
/// on the expression rather than leaving it implicit in a declaration that
/// happens to disagree is what makes that cost visible here (`spec.md` §3).
/// It also means the fp32 bytes are never resident -- the bind used to
/// allocate a bf16 copy of every weight while the fp32 original stayed
/// loaded, so peak memory was three times the bf16 model.
inline void csm_bf16_weights(ContractBuilder& b) {
    for (const SourceTensor& raw : b.tensors()) {
        if (!is_raw(raw.encoding, PieLoaderDType::F32)) {
            continue;
        }
        auto [expr, local] =
            b.shard(b.contract().src(std::string(raw.name)), shape_of(raw),
                    b.shard_axis(raw.name));
        const PieLoaderEncodingSpec bf16 = pie_loader::raw(PieLoaderDType::BF16);
        b.define(b.output_name(raw.name), b.contract().cast(expr, bf16), bf16,
                 std::move(local));
        b.consume(raw.id);
    }
}

}  // namespace contract_detail

/// CSM: a dense backbone plus a depth decoder and the Mimi codec, all bf16.
inline void author_csm_contract(ContractBuilder& b) {
    contract_detail::csm_bf16_weights(b);
    author_dense_contract(b);
}
}  // namespace pie_cuda_driver::model
