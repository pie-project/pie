#pragma once

#include "common.cuh"
#include "tensor.hpp"
#include <stdexcept>
#include <string>
#include <algorithm>
#include <cstdint>
#include <vector>

inline size_t align_up(size_t size, size_t alignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

class StackAllocator {
public:
    explicit StackAllocator(size_t total_size_bytes)
        : buffer_(total_size_bytes),
          offset_(0) {
        // Pre-allocate capacity to avoid frequent std::vector reallocations
        offset_stack_.reserve(256);
    }

    StackAllocator(const StackAllocator&) = delete;
    StackAllocator& operator=(const StackAllocator&) = delete;
    StackAllocator(StackAllocator&&) = delete;
    StackAllocator& operator=(StackAllocator&&) = delete;

    /**
     * @brief Resets the allocator, clearing the offset and the history.
     */
    void reset() {
        offset_ = 0;
        offset_stack_.clear();
    }

    /**
     * @brief Allocates a Tensor view, saving the previous state for deallocation.
     */
    template<typename T>
    Tensor<T> allocate(size_t count = 1) {
        constexpr size_t alignment = std::max(alignof(T), (size_t)256);

        // **FIX**: Store the current offset before this allocation.
        offset_stack_.push_back(offset_);

        size_t aligned_offset = align_up(offset_, alignment);
        size_t bytes_to_alloc = count * sizeof(T);

        if (aligned_offset + bytes_to_alloc > buffer_.size()) {
            offset_stack_.pop_back(); // Revert state before throwing
            throw std::runtime_error("StackAllocator out of memory.");
        }

        offset_ = aligned_offset + bytes_to_alloc;

        return Tensor<T>(buffer_, aligned_offset, count);
    }

    /**
     * @brief Allocates and copies data, saving the previous state.
     */
    template<typename T>
    Tensor<T> allocate_and_copy_async(const std::vector<T>& host_vector, cudaStream_t stream) {
        if (host_vector.empty()) {
            // Gracefully return a zero-sized tensor without consuming allocator state.
            return Tensor<T>(buffer_, offset_, 0);
        }

        size_t count = host_vector.size();
        constexpr size_t alignment = std::max(alignof(T), (size_t)256);

        // **FIX**: Store the current offset before this allocation.
        offset_stack_.push_back(offset_);

        size_t aligned_offset = align_up(offset_, alignment);
        size_t bytes_to_alloc = count * sizeof(T);

        if (aligned_offset + bytes_to_alloc > buffer_.size()) {
            offset_stack_.pop_back(); // Revert state before throwing
            throw std::runtime_error("StackAllocator::allocate_and_copy_async out of memory.");
        }

        T* device_ptr = reinterpret_cast<T*>(buffer_.data() + aligned_offset);
        offset_ = aligned_offset + bytes_to_alloc;

        CUDA_CHECK(cudaMemcpyAsync(device_ptr, host_vector.data(), bytes_to_alloc, cudaMemcpyHostToDevice, stream));

        return Tensor<T>(buffer_, aligned_offset, count);
    }

    /**
     * @brief Allocates the rest of the buffer, saving the previous state.
     */
    Tensor<uint8_t> allocate_rest() {
        // **FIX**: Store the current offset before this allocation.
        offset_stack_.push_back(offset_);

        constexpr size_t alignment = 256;
        size_t aligned_offset = align_up(offset_, alignment);
        size_t remaining_bytes = buffer_.size() - aligned_offset;

        if (remaining_bytes == 0) {
            return Tensor<uint8_t>(buffer_, aligned_offset, 0);
        }

        offset_ = buffer_.size();
        return Tensor<uint8_t>(buffer_, aligned_offset, remaining_bytes);
    }

    /**
     * @brief Deallocates by restoring the allocator's state from before the allocation.
     */
    template<typename T>
    void deallocate(const Tensor<T>& tensor) {
        if (offset_stack_.empty()) {
            // This can happen for a default-constructed tensor that was never allocated.
            if (tensor.size() == 0) return;
            throw std::runtime_error("StackAllocator deallocation error: Unbalanced deallocation call.");
        }

        // The offset to restore to is the one we saved before this block was allocated.
        size_t previous_offset = offset_stack_.back();

        // Sanity Check: Ensure the tensor being freed was indeed at the top of the stack.
        // This check is now robust because it uses the true current offset.
        uint8_t* start_ptr = reinterpret_cast<uint8_t*>(tensor.data());
        size_t bytes = tensor.size() * sizeof(T);
        uint8_t* end_ptr = start_ptr + bytes;
        uint8_t* expected_end_ptr = buffer_.data() + offset_;

        if (end_ptr != expected_end_ptr && tensor.size() != 0) {
             throw std::runtime_error(
                "StackAllocator deallocation error: The provided tensor does not match "
                "the most recent allocation (LIFO violation)."
            );
        }

        // **FIX**: Rewind the offset to its actual previous state and pop from the stack.
        offset_ = previous_offset;
        offset_stack_.pop_back();
    }

    size_t get_used_size() const { return offset_; }
    size_t get_total_size() const { return buffer_.size(); }

private:
    ByteTensor buffer_;
    size_t offset_;
    // **FIX**: This stack stores the `offset_` value from *before* each allocation.
    std::vector<size_t> offset_stack_;
};