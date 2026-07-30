#pragma once

// C++ ownership layer over the forward crate's generated C ABI, modeled on
// `loader/include/pie_loader/plan.hpp` but much smaller: tracing has no
// checkpoint handle, no diagnostics list and no verification pass, so what
// remains is exactly the part C++ has to supply — RAII over storage that
// belongs to Rust, and typed accessors over the POD the driver walks.
//
// `pie_forward.h` is generated from the Rust definitions by
// `cargo run -p pie-forward-cbindgen`, so nothing here restates a struct or
// an enum; a hand-written copy is the drift the loader's header replaced
// (`loader/architecture.md` §9).
//
// One ownership difference from the loader: the plan *header* is caller
// storage (`pie_forward_trace_llama_like` takes `PieForwardPlan*`, not
// `PieForwardPlan**`), so this class holds the POD by value and Rust owns
// only the arena behind `owner`. `pie_forward_release` frees that arena and
// resets the header to empty, which is why the destructor can call it
// unconditionally.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "pie_forward.h"

namespace pie_forward {

/// An owned traced form.
///
/// Move-only: the arena behind `plan_.owner` must be released exactly once,
/// which matches how `pie_loader::LoadPlan` treats its plan pointer.
class ForwardPlan {
  public:
    /// A contiguous run of value ids (an op's inputs or outputs).
    struct IdSpan {
        const std::uint32_t* data = nullptr;
        std::size_t size = 0;
        const std::uint32_t* begin() const noexcept { return data; }
        const std::uint32_t* end() const noexcept { return data + size; }
        std::uint32_t operator[](std::size_t i) const { return data[i]; }
    };

    ForwardPlan() = default;
    ForwardPlan(const ForwardPlan&) = delete;
    ForwardPlan& operator=(const ForwardPlan&) = delete;
    ForwardPlan(ForwardPlan&& other) noexcept
        : plan_(std::exchange(other.plan_, PieForwardPlan{})) {}
    ForwardPlan& operator=(ForwardPlan&& other) noexcept {
        if (this != &other) {
            reset();
            plan_ = std::exchange(other.plan_, PieForwardPlan{});
        }
        return *this;
    }
    ~ForwardPlan() { reset(); }

    void reset() noexcept {
        // Release empties the header (owner becomes null), so this is safe
        // on a default-constructed or moved-from instance.
        pie_forward_release(&plan_);
    }

    /// Trace the llama_like family. The only way in, mirroring
    /// `pie_loader::LoadPlan::compile`: the facts are the whole request, and
    /// none of them is a model's name.
    static ForwardPlan trace_llama_like(const PieForwardLlamaLikeFacts& facts) {
        PieForwardPlan raw{};
        const PieForwardStatus status = pie_forward_trace_llama_like(&facts, &raw);
        if (status != PieForwardStatus::Ok) {
            throw std::runtime_error(
                "forward plan: trace failed (" + status_name(status) + ")");
        }
        return ForwardPlan(raw);
    }

    explicit operator bool() const noexcept { return plan_.owner != nullptr; }

    const PieForwardPlan& view() const {
        if (plan_.owner == nullptr) throw std::runtime_error("forward plan: empty");
        return plan_;
    }

    std::size_t op_count() const { return view().ops.len; }
    const PieForwardOp& op(std::size_t i) const {
        const PieForwardPlan& plan = view();
        if (i >= plan.ops.len) throw std::runtime_error("forward plan: op out of range");
        return plan.ops.ptr[i];
    }

    std::size_t value_count() const { return view().values.len; }
    const PieForwardValue& value(std::uint32_t id) const {
        const PieForwardPlan& plan = view();
        if (id >= plan.values.len) {
            throw std::runtime_error("forward plan: value id out of range");
        }
        return plan.values.ptr[id];
    }

    IdSpan inputs(const PieForwardOp& op) const { return ids(op.inputs); }
    IdSpan outputs(const PieForwardOp& op) const { return ids(op.outputs); }

    /// A name-table entry as a view into the plan's blob; valid for the
    /// plan's lifetime (the strings are not NUL-terminated — see
    /// `PieForwardBytes` in the generated header).
    std::string_view name(std::uint32_t index) const {
        const PieForwardPlan& plan = view();
        if (index >= plan.names.len) {
            throw std::runtime_error("forward plan: name index out of range");
        }
        const PieForwardName entry = plan.names.ptr[index];
        return std::string_view(
            reinterpret_cast<const char*>(plan.name_bytes.ptr) + entry.offset,
            entry.len);
    }

    /// The op's weight name, or empty for the kinds that reference none.
    std::string_view weight_name(const PieForwardOp& op) const {
        if (op.weight_name == PIE_FORWARD_NO_NAME) return {};
        return name(op.weight_name);
    }

    std::string_view family() const { return name(view().family); }
    std::uint64_t compiler_version() const { return view().compiler_version; }

    static std::string status_name(PieForwardStatus status) {
        switch (status) {
        case PieForwardStatus::Ok: return "ok";
        case PieForwardStatus::InvalidArgument: return "invalid argument";
        }
        return "unknown status";
    }

  private:
    explicit ForwardPlan(const PieForwardPlan& plan) noexcept : plan_(plan) {}

    IdSpan ids(PieForwardIdRange range) const {
        const PieForwardPlan& plan = view();
        if (static_cast<std::size_t>(range.offset) + range.len > plan.value_ids.len) {
            throw std::runtime_error("forward plan: id range out of range");
        }
        return IdSpan{plan.value_ids.ptr + range.offset, range.len};
    }

    PieForwardPlan plan_ = PieForwardPlan{};
};

}  // namespace pie_forward
