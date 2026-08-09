#pragma once
// Stub for csrc/src/config.hpp. Only `cache_dir()` is referenced by
// planner_profile_cache.cpp, and it is an INPUT to the logic under test (the
// engine publishes it at load time) rather than part of it -- so standing in
// for it is not stubbing away anything being tested. The real header pulls in
// toml++ and the whole engine config surface.
#include <string>
namespace pie_cuda_driver {
inline std::string& mutable_cache_dir() { static std::string d; return d; }
inline const std::string& cache_dir() { return mutable_cache_dir(); }
}
