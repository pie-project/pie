#pragma once
//
// Golden-file loader for the PTIR cross-backend cert. Parses echo's golden
// vector files (interface/sampling-ir/tests/golden-ptir/*.txt), the SAME files
// charlie's ptir_golden_exec_test.cu consumes. Extracts the container hex +
// identity hash, the host_put / input tensors, and the expected `take` results
// in Rust `Debug` form. Pure host C++.

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ptir_metal::golden {

inline std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}

inline std::vector<std::uint8_t> hex_to_bytes(const std::string& h) {
    std::vector<std::uint8_t> b;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2)
        b.push_back(static_cast<std::uint8_t>(std::stoul(h.substr(i, 2), nullptr, 16)));
    return b;
}

// Substring between the first '[' and its matching ']' (inclusive of contents).
inline std::string bracket(const std::string& s) {
    std::size_t a = s.find('[');
    std::size_t b = s.find(']', a);
    if (a == std::string::npos || b == std::string::npos) return "";
    return s.substr(a + 1, b - a - 1);
}

inline std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        std::string t = trim(tok);
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

inline std::vector<float> parse_f32(const std::string& listbody) {
    std::vector<float> v;
    for (auto& t : split_commas(listbody)) v.push_back(std::stof(t));
    return v;
}
inline std::vector<std::int32_t> parse_i32(const std::string& listbody) {
    std::vector<std::int32_t> v;
    for (auto& t : split_commas(listbody)) v.push_back(static_cast<std::int32_t>(std::stol(t)));
    return v;
}
// Rust Debug bools: "false"/"true" -> 1-byte 0/1 (the (B) host-Bool wire form).
inline std::vector<std::uint8_t> parse_bool(const std::string& listbody) {
    std::vector<std::uint8_t> v;
    for (auto& t : split_commas(listbody)) v.push_back(t == "true" ? 1 : 0);
    return v;
}

struct Golden {
    std::string name;
    std::string container_hex;
    std::string sidecar_hex;
    std::uint64_t hash = 0;
    std::vector<std::string> lines;  // all raw lines, for ad-hoc queries

    // First line whose "key" (before ':') equals `key`, value trimmed.
    std::string value_of(const std::string& key) const {
        for (auto& ln : lines) {
            auto c = ln.find(':');
            if (c == std::string::npos) continue;
            if (trim(ln.substr(0, c)) == key) return trim(ln.substr(c + 1));
        }
        return "";
    }
    // First line that contains `needle`.
    std::string line_with(const std::string& needle) const {
        for (auto& ln : lines)
            if (ln.find(needle) != std::string::npos) return ln;
        return "";
    }
};

inline bool load(const std::string& path, Golden& g) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        g.lines.push_back(line);
        auto c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = trim(line.substr(0, c)), v = trim(line.substr(c + 1));
        if (k == "name") g.name = v;
        else if (k == "container") g.container_hex = v;
        else if (k == "sidecar") g.sidecar_hex = v;
        else if (k == "hash") g.hash = std::stoull(v, nullptr, 16);
    }
    return !g.container_hex.empty();
}

}  // namespace ptir_metal::golden
