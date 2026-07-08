// PTIR (thrust-3) stage-program dispatcher — the nvcc-compiled impl behind the
// CUDA-free `ptir_dispatch.hpp` façade. Includes the tier-0 runtime (device
// kernels) here, isolated from the host `.cpp` translation units.

#include "ptir/ptir_dispatch.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ptir/program_runtime.hpp"

#include "ptir/descriptor_resolve.hpp"

namespace pie_cuda_driver::ptir {

struct PtirDispatch::Impl {
    // C3 hash-keyed decode cache (container+sidecar → Trace, first-fire-of-hash).
    PtirProgramCache cache;
    // The GLOBAL device channel registry (W0.1): channel storage keyed by global
    // id, shared across instances so multi-pass channels resolve one device cell.
    DeviceChannelRegistry channels;
    // Persistent per-instance execution contexts, keyed by the wire instance id
    // (cross-fire channel VIEW state survives). unique_ptr: PtirInstance owns its
    // view + runner, kept stable across rehashes.
    std::unordered_map<std::uint64_t, std::unique_ptr<PtirInstance>> instances;
    // `out_resp.ptir_output_*` staging (per-program CSR of committed outputs).
    std::vector<std::uint64_t> out_channels;
    std::vector<std::uint8_t>  out_blob;
    std::vector<std::uint32_t> out_lens;
    std::vector<std::uint32_t> out_indptr;
};

PtirDispatch::PtirDispatch() : impl_(std::make_unique<Impl>()) {}
PtirDispatch::~PtirDispatch() = default;

namespace {

// Slice program `p` out of a (channels, blob, lens, indptr) SoA into
// ChannelValue[]. The blob offset for an entry is the running Σ of prior lens
// (the blob is concatenated across all programs' entries in entry order).
// `chans` is now the GLOBAL channel id per entry (W0.2 re-key).
std::vector<ChannelValue> read_channel_values(
    const pie_driver::PieSlice<std::uint64_t>& chans,
    const pie_driver::PieSlice<std::uint8_t>& blob,
    const pie_driver::PieSlice<std::uint32_t>& lens,
    const pie_driver::PieSlice<std::uint32_t>& indptr,
    std::size_t p) {
    std::vector<ChannelValue> out;
    if (indptr.size() < p + 2) return out;
    const std::uint32_t lo = indptr.data()[p];
    const std::uint32_t hi = indptr.data()[p + 1];
    std::size_t off = 0;
    for (std::uint32_t i = 0; i < lo; ++i) off += lens.data()[i];
    for (std::uint32_t e = lo; e < hi; ++e) {
        ChannelValue cv;
        cv.channel = chans.data()[e];
        const std::uint32_t n = lens.data()[e];
        cv.bytes.assign(blob.data() + off, blob.data() + off + n);
        off += n;
        out.push_back(std::move(cv));
    }
    return out;
}

// Slice program `p`'s dense-index → global-id channel map out of the wire's
// `ptir_program_channel_ids` CSR (W0.2). `channel_ids[i]` is dense channel i's
// global id; empty ⇒ legacy (the view falls back to dense == global).
std::vector<std::uint64_t> read_channel_ids(
    const pie_driver::PieSlice<std::uint64_t>& ids,
    const pie_driver::PieSlice<std::uint32_t>& indptr,
    std::size_t p) {
    std::vector<std::uint64_t> out;
    if (indptr.size() < p + 2) return out;
    const std::uint32_t lo = indptr.data()[p];
    const std::uint32_t hi = indptr.data()[p + 1];
    for (std::uint32_t e = lo; e < hi; ++e) out.push_back(ids.data()[e]);
    return out;
}

}  // namespace

bool PtirDispatch::run(const pie_driver::PieForwardRequestView& view,
                       pie_driver::PieForwardResponseView& out_resp,
                       const void* logits, std::uint32_t vocab, cudaStream_t stream) {
    Impl& s = *impl_;
    const bool has_releases = !view.ptir_release_channel_ids.empty() ||
                              !view.ptir_release_instance_ids.empty();
    if (view.ptir_program_hashes.empty() && !has_releases) return false;
    const std::size_t n_prog = view.ptir_program_hashes.size();

    // Env-gated fire trace (§6.2 e2e bring-up): proves the ptir-carrier req
    // reached the executor hook (vs being gated by forward_prepare/plan upstream).
    if (std::getenv("PIE_PTIR_TRACE")) {
        std::fprintf(stderr, "[ptir-hook] FIRED: n_prog=%zu vocab=%u logits=%p\n",
                     n_prog, vocab, logits);
    }

    s.out_channels.clear();
    s.out_blob.clear();
    s.out_lens.clear();
    s.out_indptr.assign(1, 0);  // per-program output CSR, leading 0

    for (std::size_t p = 0; p < n_prog; ++p) {
        // Decode (first fire ships container+sidecar; steady-state empty ⇒ cache).
        const std::uint32_t blo = view.ptir_program_bytes_indptr.data()[p];
        const std::uint32_t bhi = view.ptir_program_bytes_indptr.data()[p + 1];
        const std::uint32_t slo = view.ptir_program_sidecar_indptr.data()[p];
        const std::uint32_t shi = view.ptir_program_sidecar_indptr.data()[p + 1];
        std::string derr;
        const Trace* trace = s.cache.get_or_decode(
            view.ptir_program_hashes.data()[p],
            view.ptir_program_bytes.data() + blo, static_cast<std::size_t>(bhi - blo),
            view.ptir_program_sidecar_bytes.data() + slo,
            static_cast<std::size_t>(shi - slo), &derr);
        if (trace == nullptr) {
            std::fprintf(stderr, "[pie-driver-cuda] ptir decode failed (program %zu): %s\n",
                         p, derr.c_str());
            s.out_indptr.push_back(static_cast<std::uint32_t>(s.out_channels.size()));
            continue;
        }

        // Persistent instance by wire id: first sighting → instantiate (binding
        // the channel VIEW onto the global registry via channel_ids + applying
        // D2 seeds); every fire → apply per-fire host-puts.
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto channel_ids = read_channel_ids(
            view.ptir_program_channel_ids, view.ptir_program_channel_ids_indptr, p);
        auto it = s.instances.find(iid);
        if (it == s.instances.end()) {
            auto seeds = read_channel_values(
                view.ptir_program_seed_channels, view.ptir_program_seed_blob,
                view.ptir_program_seed_lens, view.ptir_program_seed_indptr, p);
            std::string ierr;
            auto inst = std::make_unique<PtirInstance>(*trace, &s.channels, channel_ids,
                                                       seeds, &ierr);
            if (!inst->ok()) {
                std::fprintf(stderr, "[pie-driver-cuda] ptir instance %zu bind failed: %s\n",
                             p, ierr.c_str());
                s.out_indptr.push_back(static_cast<std::uint32_t>(s.out_channels.size()));
                continue;
            }
            it = s.instances.emplace(iid, std::move(inst)).first;
        }
        auto host_puts = read_channel_values(
            view.ptir_program_host_put_channels, view.ptir_program_host_put_blob,
            view.ptir_program_host_put_lens, view.ptir_program_host_put_indptr, p);

        FireInputs fin;
        fin.logits = logits;
        fin.vocab = vocab;
        fin.stream = stream;
        it->second->fire(host_puts, fin);
        auto outs = it->second->harvest_outputs();
        if (std::getenv("PIE_PTIR_TRACE"))
            std::fprintf(stderr, "[ptir-hook] program %zu: harvested %zu output(s)%s\n",
                         p, outs.size(), outs.empty() ? " (NONE — no committed READER channel)" : "");
        for (auto& kv : outs) {
            s.out_channels.push_back(kv.first);
            s.out_blob.insert(s.out_blob.end(), kv.second.begin(), kv.second.end());
            s.out_lens.push_back(static_cast<std::uint32_t>(kv.second.size()));
        }
        s.out_indptr.push_back(static_cast<std::uint32_t>(s.out_channels.size()));
    }

    // Release markers (W0.3): free the device storage of dropped channels + the
    // views of dropped instances (also fixes the pre-existing instance-map leak).
    // Applied AFTER this request's fires so a channel referenced this fire is not
    // freed mid-use.
    for (std::size_t i = 0; i < view.ptir_release_instance_ids.size(); ++i)
        s.instances.erase(view.ptir_release_instance_ids.data()[i]);
    for (std::size_t i = 0; i < view.ptir_release_channel_ids.size(); ++i)
        s.channels.release(view.ptir_release_channel_ids.data()[i]);

    out_resp.ptir_output_channels = pie_driver::PieSlice<std::uint64_t>{
        s.out_channels.data(), s.out_channels.size()};
    out_resp.ptir_output_blob = pie_driver::PieSlice<std::uint8_t>{
        s.out_blob.data(), s.out_blob.size()};
    out_resp.ptir_output_lens = pie_driver::PieSlice<std::uint32_t>{
        s.out_lens.data(), s.out_lens.size()};
    out_resp.ptir_output_indptr = pie_driver::PieSlice<std::uint32_t>{
        s.out_indptr.data(), s.out_indptr.size()};
    return true;
}

bool PtirDispatch::resolve_descriptors(const pie_driver::PieForwardRequestView& view,
                                       std::uint32_t page_size, FireGeometry& out,
                                       std::string* err) {
    if (err) err->clear();
    if (view.ptir_program_hashes.empty()) return false;
    Impl& s = *impl_;
    const std::size_t n_prog = view.ptir_program_hashes.size();

    for (std::size_t p = 0; p < n_prog; ++p) {
        // Decode (first fire ships container+sidecar; steady-state empty ⇒ cache).
        const std::uint32_t blo = view.ptir_program_bytes_indptr.data()[p];
        const std::uint32_t bhi = view.ptir_program_bytes_indptr.data()[p + 1];
        const std::uint32_t slo = view.ptir_program_sidecar_indptr.data()[p];
        const std::uint32_t shi = view.ptir_program_sidecar_indptr.data()[p + 1];
        std::string derr;
        const Trace* trace = s.cache.get_or_decode(
            view.ptir_program_hashes.data()[p],
            view.ptir_program_bytes.data() + blo, static_cast<std::size_t>(bhi - blo),
            view.ptir_program_sidecar_bytes.data() + slo,
            static_cast<std::size_t>(shi - slo), &derr);
        if (trace == nullptr) continue;

        // Only a DEVICE-GEOMETRY program (a descriptor port binds a channel)
        // needs pre-forward resolution; host-known/const-port programs are wire-
        // prefilled by the runtime's `map_geometry`.
        bool device_geometry = false;
        for (const PortBinding& pb : trace->ports)
            if (!pb.is_const) { device_geometry = true; break; }
        if (!device_geometry) continue;

        // Get-or-build the instance (applying D2 seeds on its first fire) so its
        // descriptor channels hold fire 0's seeded / prior-fire-produced values.
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto channel_ids = read_channel_ids(
            view.ptir_program_channel_ids, view.ptir_program_channel_ids_indptr, p);
        auto it = s.instances.find(iid);
        if (it == s.instances.end()) {
            auto seeds = read_channel_values(
                view.ptir_program_seed_channels, view.ptir_program_seed_blob,
                view.ptir_program_seed_lens, view.ptir_program_seed_indptr, p);
            std::string ierr;
            auto inst = std::make_unique<PtirInstance>(*trace, &s.channels, channel_ids,
                                                       seeds, &ierr);
            if (!inst->ok()) {
                if (err) *err = "ptir: resolve_descriptors instance bind failed: " + ierr;
                return false;
            }
            it = s.instances.emplace(iid, std::move(inst)).first;
        }
        // Apply any per-fire host-puts BEFORE resolving (they may fill descriptor
        // channels host-side, e.g. a granted fresh-page list).
        auto host_puts = read_channel_values(
            view.ptir_program_host_put_channels, view.ptir_program_host_put_blob,
            view.ptir_program_host_put_lens, view.ptir_program_host_put_indptr, p);
        for (const ChannelValue& hp : host_puts) {
            const std::uint32_t slot = s.channels.slot_for(hp.channel);
            if (slot != DeviceChannelRegistry::kBadSlot)
                s.channels.host_feed(slot, hp.bytes.data(), hp.bytes.size());
        }

        // The device mirror of the host's map_geometry (W1.1). A not-ready
        // descriptor channel fails the fire (W1.6).
        return resolve_fire_geometry(*trace, it->second->view(), page_size, out, err);
    }
    return false;  // no device-geometry program in this request
}

}  // namespace pie_cuda_driver::ptir
