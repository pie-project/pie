#include "model/weight_store.hpp"

namespace pie_cuda_driver {

const DeviceTensor& WeightStore::get(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("weight store: tensor not loaded: " + name);
    }
    return it->second.tensor;
}

const TensorRecord& WeightStore::record(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) {
        throw std::runtime_error("weight store: tensor not loaded: " + name);
    }
    return it->second;
}

void WeightStore::move_from(WeightStore& other) noexcept {
    tensors_.swap(other.tensors_);
    quant_meta_.swap(other.quant_meta_);
    finalized_ = other.finalized_;
    other.finalized_ = false;
}

TensorDecl WeightStore::default_spec_for(
    const std::string& name,
    const DeviceTensor& tensor)
{
    TensorDecl spec;
    spec.name = name;
    spec.dtype = tensor.dtype();
    spec.shape = tensor.shape();
    spec.layout = TensorLayoutKind::Dense;
    spec.ownership = tensor.owns_memory()
        ? TensorOwnershipKind::Owned
        : TensorOwnershipKind::BorrowedView;
    spec.parallel = TensorParallelKind::Replicated;
    return spec;
}

std::pair<WeightStore::iterator, bool> WeightStore::emplace(
    std::string name,
    DeviceTensor tensor)
{
    ensure_mutable();
    if (name.empty()) {
        throw std::runtime_error("weight store: empty tensor name");
    }
    TensorRecord rec;
    rec.spec = default_spec_for(name, tensor);
    rec.tensor = std::move(tensor);
    rec.has_spec = true;
    return tensors_.emplace(std::move(name), std::move(rec));
}

void WeightStore::insert(std::string name, DeviceTensor tensor) {
    ensure_mutable();
    if (name.empty()) {
        throw std::runtime_error("weight store: empty tensor name");
    }
    TensorRecord rec;
    rec.spec = default_spec_for(name, tensor);
    rec.tensor = std::move(tensor);
    rec.has_spec = true;
    auto [it, inserted] = tensors_.emplace(std::move(name), std::move(rec));
    if (!inserted) {
        throw std::runtime_error(
            "weight store: tensor already registered: " + it->first);
    }
}

void WeightStore::insert(std::string name, DeviceTensor tensor, TensorDecl spec) {
    ensure_mutable();
    if (name.empty()) {
        throw std::runtime_error("weight store: empty tensor name");
    }
    if (spec.name != name) {
        throw std::runtime_error(
            "weight store: TensorDecl name mismatch for '" + name + "'");
    }
    if (tensor.dtype() != spec.dtype || tensor.shape() != spec.shape) {
        throw std::runtime_error(
            "weight store: tensor does not match TensorDecl for '" + name + "'");
    }
    TensorRecord rec;
    rec.spec = std::move(spec);
    rec.tensor = std::move(tensor);
    rec.has_spec = true;
    auto [it, inserted] = tensors_.emplace(std::move(name), std::move(rec));
    if (!inserted) {
        throw std::runtime_error(
            "weight store: tensor already registered: " + it->first);
    }
}

void WeightStore::replace(std::string name, DeviceTensor tensor) {
    ensure_mutable();
    if (name.empty()) {
        throw std::runtime_error("weight store: empty tensor name");
    }
    validate_erase_allowed(name);
    quant_meta_.erase(name);
    TensorRecord rec;
    rec.spec = default_spec_for(name, tensor);
    rec.tensor = std::move(tensor);
    rec.has_spec = true;
    tensors_.insert_or_assign(std::move(name), std::move(rec));
}

void WeightStore::replace(std::string name, DeviceTensor tensor, TensorDecl spec) {
    ensure_mutable();
    if (name.empty()) {
        throw std::runtime_error("weight store: empty tensor name");
    }
    if (spec.name != name) {
        throw std::runtime_error(
            "weight store: TensorDecl name mismatch for '" + name + "'");
    }
    if (tensor.dtype() != spec.dtype || tensor.shape() != spec.shape) {
        throw std::runtime_error(
            "weight store: tensor does not match TensorDecl for '" + name + "'");
    }
    validate_erase_allowed(name);
    quant_meta_.erase(name);
    TensorRecord rec;
    rec.spec = std::move(spec);
    rec.tensor = std::move(tensor);
    rec.has_spec = true;
    tensors_.insert_or_assign(std::move(name), std::move(rec));
}

WeightStore::iterator WeightStore::erase(iterator it) {
    ensure_mutable();
    if (it == tensors_.end()) return it;
    validate_erase_allowed(it->first);
    const DeviceTensor* erased = &it->second.tensor;
    quant_meta_.erase(it->first);
    erase_quant_meta_refs_to(erased);
    return tensors_.erase(it);
}

std::size_t WeightStore::erase(const std::string& name) {
    ensure_mutable();
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return 0;
    erase(it);
    return 1;
}

std::uint64_t WeightStore::total_bytes() const noexcept {
    std::uint64_t bytes = 0;
    for (const auto& [_, record] : tensors_) {
        const bool owned =
            record.has_spec
                ? record.spec.ownership == TensorOwnershipKind::Owned
                : record.tensor.owns_memory();
        if (owned && record.tensor.owns_memory()) {
            bytes += record.tensor.nbytes();
        }
    }
    return bytes;
}

void WeightStore::set_quant_meta(const std::string& name, QuantMeta meta) {
    ensure_mutable();
    if (tensors_.find(name) == tensors_.end()) {
        throw std::runtime_error(
            "weight store: quantized tensor '" + name + "' not registered");
    }
    if (!meta.scale) {
        throw std::runtime_error(
            "weight store: quant metadata for '" + name +
            "' has null scale tensor");
    }
    auto [it, inserted] = quant_meta_.emplace(name, std::move(meta));
    if (!inserted) {
        throw std::runtime_error(
            "weight store: quant metadata already attached to '" + name + "'");
    }
}

std::optional<QuantMeta> WeightStore::quant_meta(
    const std::string& name) const
{
    auto it = quant_meta_.find(name);
    if (it == quant_meta_.end()) return std::nullopt;
    QuantMeta meta = it->second;
    if (!meta.scale_name.empty()) {
        meta.scale = &get(meta.scale_name);
    }
    if (!meta.zero_point_name.empty()) {
        meta.zero_point = &get(meta.zero_point_name);
    }
    return meta;
}

bool WeightStore::owns_tensor_handle(
    const DeviceTensor* tensor) const noexcept
{
    if (!tensor) return false;
    for (const auto& [_, candidate] : tensors_) {
        if (&candidate.tensor == tensor) return true;
    }
    return false;
}

void WeightStore::erase_quant_meta_refs_to(const DeviceTensor* tensor) {
    if (!tensor) return;
    for (auto it = quant_meta_.begin(); it != quant_meta_.end();) {
        if (it->second.scale == tensor || it->second.zero_point == tensor) {
            it = quant_meta_.erase(it);
        } else {
            ++it;
        }
    }
}

void WeightStore::validate_quant_metadata() const {
    for (const auto& [name, meta] : quant_meta_) {
        if (tensors_.find(name) == tensors_.end()) {
            throw std::runtime_error(
                "weight store: quant metadata references missing weight '" +
                name + "'");
        }
        const DeviceTensor* scale = meta.scale;
        if (!meta.scale_name.empty()) {
            auto scale_it = tensors_.find(meta.scale_name);
            if (scale_it == tensors_.end()) {
                throw std::runtime_error(
                    "weight store: quant metadata for '" + name +
                    "' references missing scale tensor '" +
                    meta.scale_name + "'");
            }
            scale = &scale_it->second.tensor;
        }
        if (!owns_tensor_handle(scale)) {
            throw std::runtime_error(
                "weight store: quant metadata for '" + name +
                "' references an unregistered scale tensor");
        }
        const DeviceTensor* zero_point = meta.zero_point;
        if (!meta.zero_point_name.empty()) {
            auto zp_it = tensors_.find(meta.zero_point_name);
            if (zp_it == tensors_.end()) {
                throw std::runtime_error(
                    "weight store: quant metadata for '" + name +
                    "' references missing zero-point tensor '" +
                    meta.zero_point_name + "'");
            }
            zero_point = &zp_it->second.tensor;
        }
        if (zero_point && !owns_tensor_handle(zero_point)) {
            throw std::runtime_error(
                "weight store: quant metadata for '" + name +
                "' references an unregistered zero-point tensor");
        }
        if (meta.kind == QuantMeta::Kind::PerGroup && meta.group_size <= 0) {
            throw std::runtime_error(
                "weight store: per-group quant metadata for '" + name +
                "' has invalid group_size");
        }
    }
}

void WeightStore::validate_tensor_records() const {
    for (const auto& [name, record] : tensors_) {
        if (!record.has_spec) {
            throw std::runtime_error(
                "weight store: tensor '" + name + "' has no TensorDecl");
        }
        if (record.spec.name != name) {
            throw std::runtime_error(
                "weight store: tensor key/spec mismatch for '" + name + "'");
        }
        if (record.spec.dtype != record.tensor.dtype() ||
            record.spec.shape != record.tensor.shape()) {
            throw std::runtime_error(
                "weight store: tensor '" + name +
                "' no longer matches its TensorDecl");
        }
        if (record.spec.ownership == TensorOwnershipKind::Temporary) {
            throw std::runtime_error(
                "weight store: temporary tensor survived finalization: '" +
                name + "'");
        }
        if (record.spec.ownership == TensorOwnershipKind::Owned) {
            if (!record.tensor.empty() && !record.tensor.owns_memory()) {
                throw std::runtime_error(
                    "weight store: owned tensor '" + name +
                    "' does not own its backing allocation");
            }
        } else {
            if (record.spec.backing_tensor.empty()) {
                throw std::runtime_error(
                    "weight store: view/alias tensor '" + name +
                    "' has no backing tensor");
            }
            if (record.spec.backing_tensor == name) {
                throw std::runtime_error(
                    "weight store: view/alias tensor '" + name +
                    "' cannot use itself as backing storage");
            }
            auto backing = tensors_.find(record.spec.backing_tensor);
            if (backing == tensors_.end()) {
                throw std::runtime_error(
                    "weight store: view/alias tensor '" + name +
                    "' references missing backing tensor '" +
                    record.spec.backing_tensor + "'");
            }
            if (record.tensor.owns_memory()) {
                throw std::runtime_error(
                    "weight store: view/alias tensor '" + name +
                    "' unexpectedly owns memory");
            }
        }
    }
}

void WeightStore::validate_erase_allowed(const std::string& name) const {
    for (const auto& [candidate_name, record] : tensors_) {
        if (candidate_name == name || !record.has_spec) continue;
        if ((record.spec.ownership == TensorOwnershipKind::BorrowedView ||
             record.spec.ownership == TensorOwnershipKind::Alias) &&
            record.spec.backing_tensor == name) {
            throw std::runtime_error(
                "weight store: cannot erase backing tensor '" + name +
                "' while view/alias '" + candidate_name + "' is registered");
        }
    }
}

void WeightStore::ensure_mutable() const {
    if (finalized_) {
        throw std::runtime_error(
            "weight store: cannot mutate finalized store");
    }
}

void WeightStore::finalize() {
    if (finalized_) return;
    validate_tensor_records();
    validate_quant_metadata();
    finalized_ = true;
}

}  // namespace pie_cuda_driver
