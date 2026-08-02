#pragma once

// One-step frames, for tests that were written against the v13 launch shape.
//
// ABI v14 (Project Venus) replaced `PieLaunchDesc` with a sealed frame: the
// lane roster and the WorkingSet page translation are stated once on the
// `PieFrameDesc`, and each `PieStepDesc` references the roster by index
// through `roster_rows`. A test that used to post one batch now posts a
// one-step frame whose roster is that batch's instances and whose
// `roster_rows` is the identity permutation over them.
//
// That is all this header does. It exists so the migration reads as the shape
// change it is, rather than as one copy of the same bookkeeping per call site
// — and so that a test asserting on a *rejection* keeps asserting on the
// rejection it named, not on a frame this wrapper built wrong.
//
// Copies are deleted because the frame points into the object: `steps` names
// the embedded step and `roster_rows` names the embedded vector, so a copied
// frame would describe the original's storage.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <pie_driver_abi.h>

namespace pie::metal::test {

class OneStepFrame {
  public:
    OneStepFrame(const std::uint64_t* instance_ids, std::size_t count, const PieStepDesc& step)
        : rows_(count), step_(step) {
        for (std::size_t i = 0; i < count; ++i) {
            rows_[i] = static_cast<std::uint32_t>(i);
        }
        step_.roster_rows = {.ptr = rows_.empty() ? nullptr : rows_.data(), .len = rows_.size()};
        frame_.abi_version = PIE_DRIVER_ABI_VERSION;
        frame_.instance_ids = {.ptr = instance_ids, .len = count};
        frame_.steps = {.ptr = &step_, .len = 1};
    }

    OneStepFrame(const OneStepFrame&) = delete;
    OneStepFrame& operator=(const OneStepFrame&) = delete;

    const PieFrameDesc* desc() const { return &frame_; }

    /// The frame's admission demand. Mirrored onto the step at expansion, so a
    /// test that needs a KV high-water sets it here rather than on the step.
    void set_required_kv_pages(std::uint32_t pages) { frame_.required_kv_pages = pages; }

    /// The ABI version moved to the frame in v14, so a test that checks the
    /// version rejection sets it here.
    void set_abi_version(std::uint32_t version) { frame_.abi_version = version; }

  private:
    std::vector<std::uint32_t> rows_;
    PieStepDesc step_{};
    PieFrameDesc frame_{};
};

}  // namespace pie::metal::test
