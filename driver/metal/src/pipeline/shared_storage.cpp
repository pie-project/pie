#include "pipeline/shared_storage.hpp"

#if !defined(__APPLE__)

namespace pie::metal::pipeline {

SharedStorage make_platform_shared_storage(std::size_t size) {
    return make_host_shared_storage(size);
}

}  // namespace pie::metal::pipeline

#endif
