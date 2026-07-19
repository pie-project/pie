#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace pie::metal {

class LinearStateSlots {
  public:
    void resize(std::size_t count) {
        steps_.assign(std::max<std::size_t>(count, 1), 0);
    }

    void reset_all() {
        std::fill(steps_.begin(), steps_.end(), 0);
    }

    void reset(std::uint32_t slot) {
        if (slot < steps_.size()) steps_[slot] = 0;
    }

    void copy(std::uint32_t src, std::uint32_t dst) {
        if (src < steps_.size() && dst < steps_.size()) {
            steps_[dst] = steps_[src];
        }
    }

    int& at(std::uint32_t slot) {
        return steps_[slot < steps_.size() ? slot : 0];
    }

  private:
    std::vector<int> steps_{0};
};

}  // namespace pie::metal
