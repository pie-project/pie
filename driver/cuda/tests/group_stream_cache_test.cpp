// The slab, against the residency it replaces.
//
// Every other test of this feature checks a claim about structure: that the
// loader proves instances interchangeable, that the eviction rule frees the
// right slot, that the contract declares one expert instead of E. None of them
// moves a byte, and the claim that matters is about bytes -- that an expert
// paged into a slot is the *same weight* the stacked contract would have put
// in the slab, after the same dequantize, at the same TP rank.
//
// So this authors the same checkpoint twice, once stacked and once as groups,
// materializes the stack on the device, and then pages every expert through a
// cache too small to hold them and compares. The stack is the oracle because
// it is the path in production today: if streaming disagrees with it, streaming
// is wrong, whatever the shapes say.
//
// The cache is deliberately given two slots for eight instances. A slab that
// fits everything would exercise the fill path and nothing else; thrashing is
// where the interesting failures live -- a stale pointer surviving an eviction,
// a source binding read at the wrong instance's offset, a refill landing at a
// different address than the first fill.

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "pie_loader.h"
#include "pie_loader/checkpoint_source.hpp"
#include "pie_loader/plan.hpp"
#include "pie_loader/request.hpp"
#include "pie_loader/source_checkpoint.hpp"

#include "cuda_check.hpp"
#include "loader/group_stream_cache.hpp"
#include "loader/load_plan.hpp"
#include "loader/load_plan_executor.hpp"
#include "loader/rust_author.hpp"
#include "model/facts.hpp"
#include "model/weight_store.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

constexpr std::int64_t kLayers = 2;
constexpr std::int64_t kExperts = 4;
constexpr std::int64_t kHidden = 128;
constexpr std::int64_t kInter = 64;
constexpr std::int64_t kGroup = 32;

struct Entry {
    std::string name;
    std::vector<std::int64_t> shape;
    std::string dtype;
};

/// A checkpoint whose every tensor holds different bytes.
///
/// The zero-filled writer the contract tests use would pass this test for the
/// wrong reason: with all experts identical, reading the wrong one is
/// indistinguishable from reading the right one, which is exactly the bug a
/// cache keyed on (group, instance) can have. So each tensor is filled from its
/// own name.
///
/// The scales are the constrained part. An E8M0 byte is an exponent, and the
/// dequantize multiplies by `2^(b-127)`; left uniform-random it would produce
/// infinities on one expert and denormals on the next, and a bf16 comparison
/// against the stack would then be comparing NaNs. Keeping them near 127 keeps
/// every product finite while still differing per expert.
std::filesystem::path write_checkpoint(const std::vector<Entry>& entries) {
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "pie_group_stream_cache_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);

    nlohmann::json header = nlohmann::json::object();
    std::uint64_t offset = 0;
    for (const Entry& e : entries) {
        std::uint64_t bytes = (e.dtype == "BF16") ? 2 : 1;
        for (std::int64_t extent : e.shape) {
            bytes *= static_cast<std::uint64_t>(extent);
        }
        header[e.name] = {{"dtype", e.dtype},
                          {"shape", e.shape},
                          {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    const std::string text = header.dump();
    std::ofstream out(dir / "model.safetensors", std::ios::binary);
    const std::uint64_t len = text.size();
    out.write(reinterpret_cast<const char*>(&len), sizeof(len));
    out.write(text.data(), static_cast<std::streamsize>(text.size()));

    for (const Entry& e : entries) {
        std::uint64_t count = (e.dtype == "BF16") ? 2 : 1;
        for (std::int64_t extent : e.shape) {
            count *= static_cast<std::uint64_t>(extent);
        }
        std::uint64_t seed = 1469598103934665603ull;
        for (const char c : e.name) {
            seed = (seed ^ static_cast<std::uint8_t>(c)) * 1099511628211ull;
        }
        const bool is_scale = e.name.size() > 6 &&
            e.name.compare(e.name.size() - 6, 6, ".scale") == 0;
        std::vector<char> data(static_cast<std::size_t>(count));
        for (std::size_t i = 0; i < data.size(); ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const auto byte = static_cast<std::uint8_t>(seed >> 33);
            data[i] = static_cast<char>(
                is_scale ? static_cast<std::uint8_t>(124 + (byte % 7)) : byte);
        }
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    }
    return dir;
}

std::vector<Entry> dsv4_entries() {
    std::vector<Entry> entries{
        {"embed.weight", {16, kHidden}, "BF16"},
        {"norm.weight", {kHidden}, "BF16"},
        {"head.weight", {16, kHidden}, "BF16"},
    };
    for (std::int64_t l = 0; l < kLayers; ++l) {
        const std::string ffn = "layers." + std::to_string(l) + ".ffn.";
        entries.push_back({ffn + "gate.weight", {kExperts, kHidden}, "BF16"});
        for (std::int64_t e = 0; e < kExperts; ++e) {
            const std::string ep = ffn + "experts." + std::to_string(e) + ".";
            for (const char* half : {"w1", "w3"}) {
                entries.push_back({ep + half + ".weight", {kInter, kHidden / 2}, "I8"});
                entries.push_back({ep + half + ".scale", {kInter, kHidden / kGroup}, "U8"});
            }
            entries.push_back({ep + "w2.weight", {kHidden, kInter / 2}, "I8"});
            entries.push_back({ep + "w2.scale", {kHidden, kInter / kGroup}, "U8"});
        }
    }
    return entries;
}

pie_loader::LoadPlan compile_dsv4(
    const pie_loader::Checkpoint& checkpoint,
    const pie_loader::DeviceTarget& target,
    bool streamed)
{
    // The boot's own path: the C++ author this test used to drive was
    // harvested (`plan/model-in-rust.md` §8-5), so it states the same facts
    // through the same request entry the driver boots through. The claim
    // under test is unchanged -- stacked and streamed are two answers to one
    // request, differing only in `stream_routed_experts`.
    namespace model = pie_cuda_driver::model;
    // The facts this test states, in the form the request carries them: a
    // `pie.model/1` document. It used to build a `ModelFacts` struct, which
    // stopped crossing the ABI — `ModelFacts::from_descriptor` projects these
    // names on the far side, so writing them here is writing the same fields
    // one layer out.
    const std::string descriptor = std::string(R"({"version":"pie.model/1",)") +
                                   R"("model_type":"deepseek_v4",)" +
                                   R"("quant_method":"",)" +
                                   R"("num_hidden_layers":)" +
                                   std::to_string(kLayers) + R"(,)" +
                                   R"("num_experts":)" +
                                   std::to_string(kExperts) + R"(})";
    return pie_cuda_driver::prepare_load_plan_rust_author(
               checkpoint, descriptor, target, "", model::Mxfp4MoeRequest::Auto,
               model::Component::Full, streamed, nullptr)
        .plan;
}

std::vector<std::uint8_t> read_back(const pie_cuda_driver::DeviceTensor& t) {
    std::vector<std::uint8_t> host(t.nbytes());
    CUDA_CHECK(cudaMemcpy(host.data(), t.data(), host.size(),
                          cudaMemcpyDeviceToHost));
    return host;
}

std::vector<std::uint8_t> read_back(const void* device, std::size_t bytes) {
    std::vector<std::uint8_t> host(bytes);
    CUDA_CHECK(cudaMemcpy(host.data(), device, bytes, cudaMemcpyDeviceToHost));
    return host;
}

/// A slot's expert, byte for byte, against the same expert in the stack.
void test_a_paged_expert_equals_the_stacked_one() {
    using namespace pie_cuda_driver;

    const std::filesystem::path dir = write_checkpoint(dsv4_entries());
    std::string open_error;
    pie_loader::Checkpoint checkpoint =
        pie_loader::Checkpoint::open(dir.string(), &open_error);
    check(static_cast<bool>(checkpoint), "the checkpoint opens: " + open_error);
    if (!checkpoint) return;

    const auto target = cuda_device_target();

    // The oracle: the resident path, materialized.
    const pie_loader::LoadPlan stacked =
        compile_dsv4(checkpoint, target, /*streamed=*/false);
    WeightStore resident;
    {
        pie_loader::CheckpointSource loader(stacked.view());
        WeightStoreBuilder builder(resident);
        LoadPlanExecutor executor(loader, builder, {});
        executor.execute(stacked.view());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    check(resident.find("layers.0.ffn.experts.gate_up.weight") != resident.end(),
          "the stacked contract published a slab to compare against");
    if (resident.find("layers.0.ffn.experts.gate_up.weight") == resident.end()) {
        return;
    }

    const pie_loader::LoadPlan streamed =
        compile_dsv4(checkpoint, target, /*streamed=*/true);
    const auto groups = streamed.view().groups;
    check(groups.len == static_cast<std::size_t>(kLayers),
          "one group per layer");
    if (groups.len != static_cast<std::size_t>(kLayers)) return;

    // Two slots for eight instances: every sweep evicts.
    const std::uint64_t slot_bytes = groups.ptr[0].plan->memory.persistent_bytes;
    pie_loader::CheckpointSource loader(streamed.view());
    GroupStreamCache cache(loader, groups, 2 * slot_bytes);
    check(cache.num_slots() == 2, "the budget bought two slots");
    check(!cache.fully_resident(), "eight instances do not fit in two slots");

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Two sweeps. The second is the one that can go wrong: every slot now
    // holds an earlier instance, so every fill is a refill over live bytes.
    std::size_t compared = 0;
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (std::size_t g = 0; g < groups.len; ++g) {
            const std::string ffn = "layers." + std::to_string(g) + ".ffn.";
            const auto gate_up = read_back(resident.get(ffn + "experts.gate_up.weight"));
            const auto down = read_back(resident.get(ffn + "experts.down.weight"));
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                const WeightStore& slot =
                    cache.ensure_resident(g, e, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));

                const auto& slot_gu = slot.get("gate_up.weight");
                const auto& slot_dn = slot.get("down.weight");
                const auto host_gu = read_back(slot_gu.data(), slot_gu.nbytes());
                const auto host_dn = read_back(slot_dn.data(), slot_dn.nbytes());

                const std::size_t gu_off = e * host_gu.size();
                const std::size_t dn_off = e * host_dn.size();
                check(gu_off + host_gu.size() <= gate_up.size() &&
                          dn_off + host_dn.size() <= down.size(),
                      "the stack is E times a slot");
                if (gu_off + host_gu.size() > gate_up.size()) return;

                const std::string at = " (layer " + std::to_string(g) +
                                       " expert " + std::to_string(e) +
                                       " sweep " + std::to_string(sweep) + ")";
                check(std::memcmp(host_gu.data(), gate_up.data() + gu_off,
                                  host_gu.size()) == 0,
                      "the slot's gate_up is the stack's" + at);
                check(std::memcmp(host_dn.data(), down.data() + dn_off,
                                  host_dn.size()) == 0,
                      "the slot's down is the stack's" + at);
                ++compared;
                cache.end_batch();
            }
        }
    }
    check(compared == 2 * kLayers * kExperts, "every instance was compared");
    check(cache.evictions() > 0, "a slab this small must have evicted");
    // Nothing was resident twice: with the pin released after each expert and
    // a strictly cycling access order, two slots can never hold the next one.
    check(cache.stats().misses == compared, "each access was a real page-in");
    check(cache.stats().bytes_paged_in > 0, "bytes actually moved");

    CUDA_CHECK(cudaStreamDestroy(stream));
}

/// A cache big enough for the group stops paging, and says so.
///
/// This is the case that lets streaming be left on by default: when the slab
/// holds everything, every sweep is all hits, `fully_resident` is true, and
/// the driver keeps CUDA graph capture. If this ever regressed to misses,
/// turning the feature on would cost throughput on cards that never needed it.
///
/// The paging happens at construction rather than on the first sweep. A slab
/// that fits has no residency decision left to make, so `place_all` makes it
/// once and `all_placed` lets the forward drop the routed-set readback it
/// would otherwise need every layer -- which is the whole steady-state cost of
/// streaming. That is why this asserts zero misses rather than one sweep of
/// them, and why it checks `all_placed`.
void test_a_slab_that_fits_stops_missing() {
    using namespace pie_cuda_driver;

    const std::filesystem::path dir = write_checkpoint(dsv4_entries());
    std::string open_error;
    pie_loader::Checkpoint checkpoint =
        pie_loader::Checkpoint::open(dir.string(), &open_error);
    if (!checkpoint) return;

    const auto target = cuda_device_target();
    const pie_loader::LoadPlan plan =
        compile_dsv4(checkpoint, target, /*streamed=*/true);
    const auto groups = plan.view().groups;
    if (groups.len == 0) return;

    const std::uint64_t slot_bytes = groups.ptr[0].plan->memory.persistent_bytes;
    pie_loader::CheckpointSource loader(plan.view());
    GroupStreamCache cache(loader, groups, slot_bytes * kLayers * kExperts);
    check(cache.fully_resident(), "the whole group fits");
    check(cache.num_slots() == kLayers * kExperts, "a slot per instance");

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    for (int sweep = 0; sweep < 2; ++sweep) {
        for (std::size_t g = 0; g < groups.len; ++g) {
            for (std::uint32_t e = 0; e < kExperts; ++e) {
                cache.ensure_resident(g, e, stream);
            }
        }
        cache.end_batch();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    check(cache.stats().misses == 0,
          "construction placed the group, so no sweep paged");
    check(cache.stats().hits == static_cast<std::uint64_t>(2 * kLayers * kExperts),
          "both sweeps hit every time");
    check(cache.evictions() == 0, "nothing was evicted");
    check(cache.all_placed(),
          "a slab that fits reports the group placed, so the forward can stop "
          "reading back the routed set");
    for (std::size_t g = 0; g < groups.len; ++g) {
        for (std::uint32_t e = 0; e < kExperts; ++e) {
            check(cache.store_of(g, e) != nullptr,
                  "every instance resolves to a slot without a page-in");
        }
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

/// A slot's store owns nothing, and says so.
///
/// The two tests above exercise `External` only in passing -- they would still
/// pass if the kind were spelled some other way that happened not to throw.
/// What it actually asserts is worth pinning directly: every tensor in a slot
/// points into a slab the cache allocated and will free, so the store must
/// hold no allocation of its own and must contribute nothing to the memory
/// accounting. Were these `Owned`, the slab would be freed once per slot on
/// top of once by the cache, and would be counted `arity` times over.
void test_a_slot_store_owns_nothing() {
    using namespace pie_cuda_driver;

    const std::filesystem::path dir = write_checkpoint(dsv4_entries());
    std::string open_error;
    pie_loader::Checkpoint checkpoint =
        pie_loader::Checkpoint::open(dir.string(), &open_error);
    if (!checkpoint) return;

    const auto target = cuda_device_target();
    const pie_loader::LoadPlan plan =
        compile_dsv4(checkpoint, target, /*streamed=*/true);
    const auto groups = plan.view().groups;
    if (groups.len == 0) return;

    const std::uint64_t slot_bytes = groups.ptr[0].plan->memory.persistent_bytes;
    pie_loader::CheckpointSource loader(plan.view());
    GroupStreamCache cache(loader, groups, slot_bytes * 2);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    const WeightStore& store = cache.ensure_resident(0, 0, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Finalization is where the invariant is enforced, and reaching this far
    // means it ran: `ensure_resident` finalizes before it hands the store back.
    check(store.total_bytes() == 0,
          "a slot's store is not charged for the slab it was lent");

    std::size_t seen = 0;
    const std::uint8_t* slab_lo = nullptr;
    for (const auto& [name, record] : store) {
        if (record.tensor.empty()) continue;
        ++seen;
        check(record.has_spec &&
                  record.spec.ownership ==
                      TensorOwnershipKind::External,
              "'" + name + "' is declared External");
        check(!record.tensor.owns_memory(),
              "'" + name + "' does not own the slab it points into");
        const auto* base = static_cast<const std::uint8_t*>(record.tensor.data());
        if (slab_lo == nullptr || base < slab_lo) slab_lo = base;
    }
    check(seen > 0, "the slot published tensors at all");
    // And they point into one slot, not wherever: every byte a slot's tensors
    // touch is inside the stride the cache priced it at.
    for (const auto& [name, record] : store) {
        if (record.tensor.empty()) continue;
        const auto* base = static_cast<const std::uint8_t*>(record.tensor.data());
        const std::uint64_t end =
            static_cast<std::uint64_t>(base - slab_lo) + record.tensor.nbytes();
        check(end <= slot_bytes, "'" + name + "' stays inside its slot");
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "group_stream_cache_test: no CUDA device; skipping\n";
        return 0;
    }
    try {
        test_a_paged_expert_equals_the_stacked_one();
        test_a_slab_that_fits_stops_missing();
        test_a_slot_store_owns_nothing();
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        ++failures;
    }
    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cerr << "group_stream_cache_test: OK\n";
    return 0;
}
