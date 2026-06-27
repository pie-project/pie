#include "ops/compiled.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>

#include <mlx/mlx.h>

namespace pie_metal_driver::ops {

namespace mx = mlx::core;

std::vector<Tensor> compiled(const std::string& key,
                             const std::vector<Tensor>& inputs,
                             CompiledRegion fn) {
    (void)key;  // diagnostics only; the fn pointer is the cache identity.
    using CompiledFn =
        std::function<std::vector<mx::array>(const std::vector<mx::array>&)>;

    // One compiled instance per region fn, created on first use and replayed
    // thereafter. Keyed on the fn pointer: distinct regions / variants are
    // distinct fns => distinct compiled instances. mx::compile handles
    // shape-keyed retracing internally (decode/prefill), and the fn-pointer
    // compile path is stable across shape alternation (the std::function path
    // is not -- see compiled.hpp).
    static std::mutex mu;
    static std::unordered_map<std::uintptr_t, CompiledFn> cache;

    const auto id = reinterpret_cast<std::uintptr_t>(fn);
    CompiledFn f;
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(id);
        if (it == cache.end()) {
            it = cache.emplace(id, mx::compile(fn)).first;
        }
        f = it->second;
    }
    // Replay outside the lock: the compiled callable is reentrant and the heavy
    // work is graph construction, not cache access.
    return f(inputs);
}

}  // namespace pie_metal_driver::ops
