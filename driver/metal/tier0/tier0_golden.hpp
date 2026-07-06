#pragma once
//
// Golden replay parser for the tier-0 interp cert. Parses echo's golden files
// (interface/sampling-ir/tests/golden-ptir/*.txt) into a container+sidecar +
// seeds + an ordered action list (set-inputs / step-assert / take-assert /
// host-put), replayed against the tier-0 Instance.

#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "runner.hpp"

namespace tier0::golden {

inline std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
inline std::vector<std::uint8_t> hex_to_bytes(const std::string& h) {
    std::vector<std::uint8_t> b;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2)
        b.push_back((std::uint8_t)std::stoul(h.substr(i, 2), nullptr, 16));
    return b;
}
inline std::string bracket(const std::string& s) {
    std::size_t a = s.find('[');
    std::size_t b = s.find(']', a);
    if (a == std::string::npos || b == std::string::npos) return "";
    return s.substr(a + 1, b - a - 1);
}
inline std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) { auto x = trim(t); if (!x.empty()) out.push_back(x); }
    return out;
}

// Parse a "TYPE([...])" typed list (F32/I32/U32/Bool) starting at/after `eq`.
inline Val parse_typed(const std::string& s) {
    std::string body = bracket(s);
    auto toks = split_commas(body);
    if (s.find("F32(") != std::string::npos) {
        std::vector<float> v;
        for (auto& t : toks) v.push_back(std::stof(t));
        return Val::make_f32(std::move(v));
    }
    P::DType dt = s.find("I32(") != std::string::npos ? P::DType::I32
                : s.find("U32(") != std::string::npos ? P::DType::U32
                                                      : P::DType::Bool;
    std::vector<std::int64_t> v;
    for (auto& t : toks) {
        if (t == "true") v.push_back(1);
        else if (t == "false") v.push_back(0);
        else v.push_back((std::int64_t)std::stoll(t));
    }
    return Val::make_int(dt, std::move(v));
}

struct Seed { std::uint32_t chan; Val val; };
struct Action {
    enum Kind { SetInputs, Step, Take, HostPut } kind;
    // SetInputs
    bool has_logits = false;
    std::vector<float> logits;
    // Step
    bool exp_committed = false, exp_miss = false;
    std::uint32_t miss_chan = 0, miss_phase = 0;
    // Take / HostPut
    std::uint32_t chan = 0;
    bool exp_wouldblock = false;
    Val val;
};

struct Golden {
    std::string name, container_hex, sidecar_hex;
    std::uint64_t hash = 0;
    std::vector<Seed> seeds;
    std::vector<Action> actions;
};

inline bool load(const std::string& path, Golden& g) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    while (std::getline(f, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;
        auto colon = t.find(':');
        std::string key = colon == std::string::npos ? "" : trim(t.substr(0, colon));
        std::string val = colon == std::string::npos ? "" : trim(t.substr(colon + 1));
        if (key == "name") g.name = val;
        else if (key == "container") g.container_hex = val;
        else if (key == "sidecar") g.sidecar_hex = val;
        else if (key == "hash") g.hash = std::stoull(val, nullptr, 16);
        else if (t.rfind("seed chan=", 0) == 0) {
            Seed s;
            s.chan = (std::uint32_t)std::stoul(t.substr(std::string("seed chan=").size()));
            s.val = parse_typed(t);
            g.seeds.push_back(std::move(s));
        } else if (t.rfind("inputs ", 0) == 0) {
            Action a; a.kind = Action::SetInputs;
            if (t.find("logits: Some(F32(") != std::string::npos) {
                a.has_logits = true;
                std::size_t p = t.find("logits: Some(F32(");
                Val v = parse_typed(t.substr(p));
                a.logits = v.f;
            }
            g.actions.push_back(std::move(a));
        } else if (t.rfind("step ", 0) == 0) {
            Action a; a.kind = Action::Step;
            a.exp_committed = t.find("committed=true") != std::string::npos;
            std::size_t m = t.find("missed=Some((");
            if (m != std::string::npos) {
                a.exp_miss = true;
                std::size_t p = m + std::string("missed=Some((").size();
                a.miss_chan = (std::uint32_t)std::stoul(t.substr(p));
                std::size_t comma = t.find(',', p);
                a.miss_phase = (std::uint32_t)std::stoul(t.substr(comma + 1));
            }
            g.actions.push_back(std::move(a));
        } else if (t.rfind("take chan=", 0) == 0) {
            Action a; a.kind = Action::Take;
            a.chan = (std::uint32_t)std::stoul(t.substr(std::string("take chan=").size()));
            if (t.find("ERR WouldBlock") != std::string::npos) a.exp_wouldblock = true;
            else a.val = parse_typed(t);
            g.actions.push_back(std::move(a));
        } else if (t.rfind("host_put chan=", 0) == 0) {
            Action a; a.kind = Action::HostPut;
            a.chan = (std::uint32_t)std::stoul(t.substr(std::string("host_put chan=").size()));
            a.val = parse_typed(t);
            g.actions.push_back(std::move(a));
        }
    }
    return !g.container_hex.empty();
}

}  // namespace tier0::golden
