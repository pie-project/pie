#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "loader/tensor_spec.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

class WeightStoreBuilder;

// Per-weight metadata for quantized tensors. Lives beside the materialized
// tensor store; absent entries mean "use the raw bf16/fp16/fp32 path".
//
// The pointers reference DeviceTensors registered separately under their own
// names in the same WeightStore. QuantMeta does not own those tensors.
struct QuantMeta {
    enum class Kind { PerTensor, PerChannel, PerGroup };
    Kind kind = Kind::PerTensor;
    std::string scale_name;
    std::string zero_point_name;
    const DeviceTensor* scale = nullptr;
    const DeviceTensor* zero_point = nullptr;
    int group_size = 0;
    int channel_axis = 0;
};

struct TensorRecord {
    TensorDecl spec;
    DeviceTensor tensor;
    bool has_spec = false;

    DType dtype() const noexcept { return tensor.dtype(); }
    const std::vector<std::int64_t>& shape() const noexcept { return tensor.shape(); }
    std::size_t numel() const noexcept { return tensor.numel(); }
    std::size_t nbytes() const noexcept { return tensor.nbytes(); }
    void* data() noexcept { return tensor.data(); }
    const void* data() const noexcept { return tensor.data(); }

    operator DeviceTensor&() noexcept { return tensor; }
    operator const DeviceTensor&() const noexcept { return tensor; }

    TensorRecord& operator=(DeviceTensor&& replacement) noexcept {
        tensor = std::move(replacement);
        if (has_spec) {
            spec.dtype = tensor.dtype();
            spec.shape = tensor.shape();
            spec.ownership = tensor.owns_memory()
                ? TensorOwnershipKind::Owned
                : TensorOwnershipKind::BorrowedView;
        }
        return *this;
    }
};

class WeightStore {
public:
    using Map = std::unordered_map<std::string, TensorRecord>;
    using QuantMap = std::unordered_map<std::string, QuantMeta>;
    using iterator = Map::iterator;
    using const_iterator = Map::const_iterator;

    WeightStore() = default;
    WeightStore(const WeightStore&) = delete;
    WeightStore& operator=(const WeightStore&) = delete;
    WeightStore(WeightStore&& other) noexcept { move_from(other); }
    WeightStore& operator=(WeightStore&& other) noexcept {
        if (this != &other) {
            tensors_.clear();
            quant_meta_.clear();
            finalized_ = false;
            move_from(other);
        }
        return *this;
    }

    std::size_t size() const noexcept { return tensors_.size(); }
    bool empty() const noexcept { return tensors_.empty(); }

    const_iterator begin() const noexcept { return tensors_.begin(); }
    const_iterator end() const noexcept { return tensors_.end(); }

    const_iterator find(const std::string& name) const {
        return tensors_.find(name);
    }

    const DeviceTensor& at(const std::string& name) const {
        return tensors_.at(name).tensor;
    }

    const DeviceTensor& get(const std::string& name) const;
    const TensorRecord& record(const std::string& name) const;

    std::uint64_t total_bytes() const noexcept;

    std::optional<QuantMeta> quant_meta(const std::string& name) const;
    void validate_quant_metadata() const;
    const QuantMap& quant_meta_map() const noexcept { return quant_meta_; }
    bool finalized() const noexcept { return finalized_; }
    std::size_t erase_runtime_weight(const std::string& name);

private:
    friend class WeightStoreBuilder;

    iterator begin_mut() noexcept { return tensors_.begin(); }
    iterator end_mut() noexcept { return tensors_.end(); }
    iterator find_mut(const std::string& name) { return tensors_.find(name); }
    DeviceTensor& at_mut(const std::string& name) {
        ensure_mutable();
        return tensors_.at(name).tensor;
    }
    void reserve(std::size_t n) {
        ensure_mutable();
        tensors_.reserve(n);
    }

    std::pair<iterator, bool> emplace(std::string name, DeviceTensor tensor);
    iterator erase(iterator it);
    std::size_t erase(const std::string& name);
    void insert(std::string name, DeviceTensor tensor);
    void insert(std::string name, DeviceTensor tensor, TensorDecl spec);
    void replace(std::string name, DeviceTensor tensor);
    void replace(std::string name, DeviceTensor tensor, TensorDecl spec);
    void set_quant_meta(const std::string& name, QuantMeta meta);

    bool owns_tensor_handle(const DeviceTensor* tensor) const noexcept;
    void erase_quant_meta_refs_to(const DeviceTensor* tensor);
    void validate_tensor_records() const;
    void validate_erase_allowed(const std::string& name) const;
    void ensure_mutable() const;
    void finalize();
    void move_from(WeightStore& other) noexcept;
    static TensorDecl default_spec_for(
        const std::string& name,
        const DeviceTensor& tensor);

    Map tensors_;
    QuantMap quant_meta_;
    bool finalized_ = false;
};

class WeightStoreBuilder {
public:
    explicit WeightStoreBuilder(WeightStore& store) noexcept
        : store_(store) {}

    void reserve(std::size_t n) { store_.reserve(n); }
    void finalize() { store_.finalize(); }

    WeightStore::iterator find(const std::string& name) {
        return store_.find_mut(name);
    }
    WeightStore::const_iterator find(const std::string& name) const {
        return store_.find(name);
    }
    WeightStore::iterator begin() noexcept { return store_.begin_mut(); }
    WeightStore::iterator end() noexcept { return store_.end_mut(); }
    WeightStore::const_iterator begin() const noexcept { return store_.begin(); }
    WeightStore::const_iterator end() const noexcept { return store_.end(); }

    const DeviceTensor& get(const std::string& name) const {
        return store_.get(name);
    }
    DeviceTensor& at(const std::string& name) {
        return store_.at_mut(name);
    }

    void insert(std::string name, DeviceTensor tensor) {
        store_.insert(std::move(name), std::move(tensor));
    }
    void insert(std::string name, DeviceTensor tensor, TensorDecl spec) {
        store_.insert(std::move(name), std::move(tensor), std::move(spec));
    }
    void replace(std::string name, DeviceTensor tensor) {
        store_.replace(std::move(name), std::move(tensor));
    }
    void replace(std::string name, DeviceTensor tensor, TensorDecl spec) {
        store_.replace(std::move(name), std::move(tensor), std::move(spec));
    }
    std::size_t erase(const std::string& name) {
        return store_.erase(name);
    }
    void set_quant_meta(const std::string& name, QuantMeta meta) {
        store_.set_quant_meta(name, std::move(meta));
    }

private:
    WeightStore& store_;
};

}  // namespace pie_cuda_driver
