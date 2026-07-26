// The driver's declared contract, checked against the plan the loader builds.
//
// This is the test that makes §12 row 7b-4b real. `declare_model_contract`
// states what a family will bind *before* the load; `pie_loader_verify` compares
// that with the plan. The two are computed by different code in different
// languages from the same `config.json` facts, so agreement is evidence and
// disagreement is a bug in one of them — which is exactly what the old
// `covered_contract_count != runtime_tensor_count` (both `view.tensors.len`)
// could never tell anyone.
//
// This file covers the *builder* — that it lends back exactly what it was given
// and that its borrowed pointers stay valid — and, when `PIE_TEST_SNAPSHOT`
// points at a real checkpoint, it *exports* the declaration that checkpoint's
// family produces.
//
// The export exists because these C++ test binaries cannot link the loader (it
// is an rlib consumed by the worker), so the comparison against a compiled plan
// has to run on the Rust side. Writing the real declaration out, rather than
// transcribing the formulas into the Rust test, is what keeps the two from
// drifting: `loader/src/ffi/tests.rs::a_declared_contract_matches_a_real_plan`
// reads this file and knows nothing about any particular family, so declaring a
// new one in `model_contracts.hpp` needs no Rust change at all.

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "pie_loader.h"
#include "pie_loader/tensor_contract.hpp"

#include "loader/model_contracts.hpp"
#include "model/config.hpp"

namespace {

int failures = 0;

void check(bool ok, std::string_view what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

std::string_view view_of(const pie_loader::PieLoaderBytes& b) {
    return {reinterpret_cast<const char*>(b.ptr), b.len};
}

// -- the builder itself ------------------------------------------------------

void test_builder_lends_what_it_was_given() {
    pie_loader::TensorContract c;
    c.demand("a", {2, 3}).demand("b").demand_optional("c", {4});

    const auto s = c.slice();
    check(s.len == 3, "three demands");
    check(c.size() == 3, "size agrees with the slice");

    check(view_of(s.ptr[0].name) == "a", "first name");
    check(s.ptr[0].shape.len == 2 && s.ptr[0].shape.ptr[1] == 3, "first shape");
    check(!s.ptr[0].optional, "first is required");

    // A shapeless demand must be an empty slice, not a slice of one zero: the
    // loader reads `len == 0` as "unstated".
    check(view_of(s.ptr[1].name) == "b", "second name");
    check(s.ptr[1].shape.len == 0 && s.ptr[1].shape.ptr == nullptr, "second is shapeless");

    check(s.ptr[2].optional, "third is optional");
}

// The reason the builder stores names in a `deque`. With a `vector`, growing
// past the initial capacity would move every string and leave the already-built
// POD demands pointing at freed memory — a use-after-free the loader would
// dereference, and one that only appears above some tensor count.
void test_pointers_survive_growth() {
    pie_loader::TensorContract c;
    std::vector<std::string> expected;
    for (int i = 0; i < 512; ++i) {
        expected.push_back("model.layers." + std::to_string(i) + ".weight");
        c.demand(expected.back(), {static_cast<std::int64_t>(i + 1)});
    }
    const auto s = c.slice();
    check(s.len == expected.size(), "all demands present after growth");
    bool all_intact = true;
    for (std::size_t i = 0; i < s.len; ++i) {
        if (view_of(s.ptr[i].name) != expected[i] ||
            s.ptr[i].shape.len != 1 ||
            s.ptr[i].shape.ptr[0] != static_cast<std::int64_t>(i + 1)) {
            all_intact = false;
        }
    }
    check(all_intact, "every name and shape still readable after growth");
}

// -- the real declaration ----------------------------------------------------

nlohmann::json contract_to_json(const pie_loader::TensorContract& c) {
    nlohmann::json out = nlohmann::json::array();
    const auto s = c.slice();
    for (std::size_t i = 0; i < s.len; ++i) {
        nlohmann::json shape = nlohmann::json::array();
        for (std::size_t d = 0; d < s.ptr[i].shape.len; ++d) shape.push_back(s.ptr[i].shape.ptr[d]);
        out.push_back({{"name", std::string(view_of(s.ptr[i].name))},
                       {"shape", std::move(shape)},
                       {"optional", s.ptr[i].optional}});
    }
    return out;
}

// Declares the snapshot's family at several TP sizes and writes the result to
// `PIE_TEST_CONTRACT_DUMP`. An empty array is a legitimate answer — it means the
// family has not been declared yet — and the Rust side treats it as such.
void export_declaration(const std::string& snapshot, const std::string& dest) {
    const auto cfg = std::filesystem::path(snapshot) / "config.json";
    const pie_cuda_driver::HfConfig hf = pie_cuda_driver::parse_hf_config(cfg);

    nlohmann::json by_tp = nlohmann::json::object();
    for (std::uint32_t tp : {1u, 2u, 4u}) {
        by_tp[std::to_string(tp)] =
            contract_to_json(pie_cuda_driver::declare_model_contract(hf, tp));
    }
    nlohmann::json doc = {{"model_type", hf.model_type}, {"by_tp_size", std::move(by_tp)}};

    std::ofstream out(dest);
    check(out.good(), "contract dump destination is writable");
    out << doc.dump(2) << "\n";
    std::cout << "declared " << doc["by_tp_size"]["1"].size() << " tensor(s) for "
              << hf.model_type << " -> " << dest << "\n";
}

}  // namespace

int main() {
    test_builder_lends_what_it_was_given();
    test_pointers_survive_growth();

    const char* snapshot = std::getenv("PIE_TEST_SNAPSHOT");
    const char* dump = std::getenv("PIE_TEST_CONTRACT_DUMP");
    if (snapshot != nullptr && dump != nullptr) {
        export_declaration(snapshot, dump);
    }

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "model_contract_test: OK\n";
    return 0;
}
