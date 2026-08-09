#pragma once
#include <string>

namespace pie_cuda_driver {
// The planner passes the format through to the byte-per-page functions and
// never inspects it, so an opaque name is the whole of what it needs.
class KvCacheFormat {
public:
    std::string name;
};
}  // namespace pie_cuda_driver
