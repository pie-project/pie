#pragma once

// PTIR tier-0 stage-runner (docs/ptir/thrust-3-programs.md P4.2, overview §7.1).
// Walks a validated trace launch-by-launch: for each stage it (1) launches a
// readiness-check kernel that reads the channel bits word and ANDs the result
// into the pass-wide commit flag, (2) resolves each value to a device buffer and
// launches one prebuilt tier-0 kernel per op (tier0_launch.hpp), (3) writes each
// `put` to the channel's PENDING cell. After the last stage an end-of-pass
// predicated commit-bump publishes puts / consumes takes only if the pass was
// ready (pass-atomic, T3/T4). A miss discards effects and is reported by the
// dispatcher as a terminal failed launch.
//
// This is the tier-0 "interpret" backend (overview §7.3): correct on day one,
// the golden model every other tier diffs against echo's host reference. Tier 1
// (P5) fuses these launches per stage; the readiness/commit semantics are
// identical.
//
// The synchronous test path blocks between passes. Production uses
// `launch_pass_async`; projected host indices preserve stream-ordered run-ahead.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "ptir/channels.hpp"
#include "ptir/channel_registry.hpp"
#include "ptir/tier0_launch.hpp"
#include "ptir/trace.hpp"

namespace pie_cuda_driver::ptir {

// Per-fire inputs the runner binds into the trace (intrinsics + host tensors).
struct FireInputs {
    const void* logits = nullptr;     // Intrinsic(Logits) base [rows, vocab], row-major
    std::uint32_t vocab = 0;
    const void* row_seeds = nullptr;  // gumbel per-row seed buffer (u32 [rows])
    std::unordered_map<std::uint32_t, const void*> host_inputs;  // host_key → device ptr
    cudaStream_t stream = nullptr;
    // Stage-2 MTP: the row base (within the `logits` buffer) of this fire's K MTP
    // DRAFT rows — an Intrinsic(MtpLogits) `[K, vocab]` matrix reads
    // `[mtp_draft_row .. mtp_draft_row + K)`. The draft logits live in DRAFT ROWS
    // of the same `ws.logits` base (not a separate buffer); only the row differs,
    // exactly mirroring the sampling-IR resolver (runtime.cpp:135). `-1` = unset ⇒
    // MtpLogits falls back to the base (row 0), same as Logits (safe no-op until
    // the MTP head / draft layout writes the K rows and sets this).
    int mtp_draft_row = -1;
};

// Result of one pass: whether the pass committed (all stages ready).
struct PassResult {
    bool committed = false;
    bool ok = true;            // false → an op/dtype was uncovered by tier-0
    std::string error;
};

class Tier0Runner {
  public:
    explicit Tier0Runner(const Trace& trace) : trace_(&trace) {
        cudaMalloc(&d_commit_, sizeof(std::uint32_t));
    }
    ~Tier0Runner() {
        if (d_commit_) cudaFree(d_commit_);
        free_baked_lists();
    }
    Tier0Runner(const Tier0Runner&) = delete;
    Tier0Runner& operator=(const Tier0Runner&) = delete;

    // Bind this runner's channel view (W0.1): the dense→global-slot map onto the
    // shared device channel registry. Must be called (by `PtirInstance`) before
    // the first `run_pass`. Bakes the pass's static readiness / commit slot lists
    // (trace + view are both fixed now) into persistent device arrays, so the hot
    // `run_pass` path launches the ring kernels with no per-pass malloc/upload.
    void bind_view(ChannelView* view) {
        view_ = view;
        bake_static_lists();
    }

    ChannelView& view() { ensure_channels(); return *view_; }

    // Back-compat accessor (driver tests): the channel VIEW exposes the same
    // seed_cell / host_feed / host_take / committed_full / read_committed surface
    // the old per-instance `ChannelArena` did. Auto-inits standalone channels if
    // no shared view was bound (test convenience).
    ChannelView& arena() { ensure_channels(); return *view_; }

    std::uint32_t* commit_device_flag() const noexcept { return d_commit_; }
    const std::vector<std::uint32_t>& commit_taken_slots() const noexcept {
        return host_commit_taken_slots_;
    }
    const std::vector<std::uint32_t>& commit_put_slots() const noexcept {
        return host_commit_put_slots_;
    }
    void apply_host_commit(bool committed) {
        ensure_channels();
        if (!committed) return;
        view_->apply_host_commit(host_commit_taken_slots_, host_commit_put_slots_);
    }

    // Standalone channel storage (tests / single-instance use): own a private
    // registry, register an endpoint per trace channel under identity (1-based)
    // global ids, and bind a view over them, so `run_pass` works without an
    // external shared registry. Production registers endpoints through the
    // driver entry and binds the shared registry via `bind_view` instead.
    void init_standalone_channels() {
        standalone_reg_ = std::make_unique<DeviceChannelRegistry>();
        std::vector<std::uint64_t> ids(trace_->channels.size());
        for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = i + 1;  // 0 = null sentinel
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const Channel& decl = trace_->channels[i];
            PieChannelDesc desc{};
            desc.abi_version = PIE_DRIVER_ABI_VERSION;
            desc.channel_id = ids[i];
            desc.shape = {decl.type.shape.dims.data(), decl.type.shape.dims.size()};
            desc.dtype = static_cast<std::uint8_t>(decl.type.dtype);
            desc.host_role = decl.host_reader
                ? PIE_CHANNEL_HOST_ROLE_READER
                : (decl.host_visible ? PIE_CHANNEL_HOST_ROLE_WRITER
                                     : PIE_CHANNEL_HOST_ROLE_NONE);
            desc.seeded = decl.has_seed ? 1 : 0;
            desc.extern_dir = PIE_CHANNEL_EXTERN_NONE;
            desc.capacity = decl.capacity;
            // Nonzero placeholder wait ids: standalone runs have no waker
            // table; the notifies that would target them never fire here.
            desc.reader_wait_id = 2 * ids[i] - 1;
            desc.writer_wait_id = 2 * ids[i];
            PieChannelEndpointBinding binding{};
            std::string register_err;
            if (!standalone_reg_->register_endpoint(desc, &binding, &register_err)) {
                std::fprintf(stderr,
                             "[tier0] standalone channel %zu registration failed: %s\n",
                             i, register_err.c_str());
            }
        }
        std::string err;
        standalone_view_.bind(standalone_reg_.get(), trace_->channels, ids, &err);
        bind_view(&standalone_view_);
    }

    // Run one pass of the trace over `in`. Returns commit status; on `!committed`
    // the channel rings are unchanged and the caller treats the fire as failed.
    PassResult run_pass(const FireInputs& in) {
        PassResult res;
        cudaStream_t s = in.stream;
        ensure_channels();   // lazily binds the view + bakes the static lists

        // pass_commit := 1 (structural predicate seed; stages AND into it).
        std::uint32_t one = 1;
        cudaMemcpyAsync(d_commit_, &one, sizeof(one), cudaMemcpyHostToDevice, s);

        std::vector<void*> scratch;   // per-pass intermediate buffers to free

        launch_pass_body(in, scratch, res);

        // ── end-of-pass predicated commit bump (pass-atomic, §7.1) ──
        if (res.ok) {
            // The taken/put slot lists were deduped + remapped to registry slots at
            // bind (Register/SPSC semantics, T4: a channel consumed by multiple ops
            // advances head once; multiple puts publish once, last wins).
            k_commit_bump<<<1, 1, 0, s>>>(view_->d_full(), view_->d_head(), view_->d_tail(),
                                          view_->d_cap1(), d_commit_taken_, n_commit_taken_,
                                          d_commit_put_, n_commit_put_, d_commit_);
            std::uint32_t committed = 0;
            cudaMemcpyAsync(&committed, d_commit_, sizeof(committed),
                            cudaMemcpyDeviceToHost, s);
            free_scratch_async(scratch, s);
            cudaStreamSynchronize(s);
            res.committed = committed != 0;
            view_->sync_host_rings();
        } else {
            free_scratch_async(scratch, s);
            cudaStreamSynchronize(s);
        }
        val_ptr_.clear();
        val_type_.clear();
        return res;
    }

    PassResult launch_pass_async(const FireInputs& in, std::vector<void*>& scratch) {
        PassResult res;
        cudaStream_t s = in.stream;
        ensure_channels();

        std::uint32_t one = 1;
        cudaMemcpyAsync(d_commit_, &one, sizeof(one), cudaMemcpyHostToDevice, s);
        launch_pass_body(in, scratch, res);
        if (res.ok) {
            k_commit_bump<<<1, 1, 0, s>>>(view_->d_full(), view_->d_head(), view_->d_tail(),
                                          view_->d_cap1(), d_commit_taken_, n_commit_taken_,
                                          d_commit_put_, n_commit_put_, d_commit_);
        } else {
            std::uint32_t zero = 0;
            cudaMemcpyAsync(d_commit_, &zero, sizeof(zero), cudaMemcpyHostToDevice, s);
        }
        free_scratch_async(scratch, s);
        val_ptr_.clear();
        val_type_.clear();
        return res;
    }

  private:
    // A stage's baked readiness slot arrays: the "need full" (take/read/peek) and
    // "need empty" (leading put) channel sets, already remapped to registry slots
    // and resident on device (uploaded once at bind by `bake_static_lists`).
    struct StageChannelLists {
        std::uint32_t* d_full = nullptr;  std::uint32_t n_full = 0;
        std::uint32_t* d_empty = nullptr; std::uint32_t n_empty = 0;
    };

    // Lazily provision standalone channels if no shared view was bound (test
    // convenience; a no-op once `bind_view` / `init_standalone_channels` ran).
    void ensure_channels() {
        if (view_ == nullptr) init_standalone_channels();
    }

    // Collect this stage's channel direction requirements from its ops + puts.
    // `desc_taken` = channels already consumed by a descriptor port this pass
    // (loop-carried): a put to such a channel reuses the vacated cell, so it is
    // NOT a leading put and needs no empty check.
    void collect_stage_channels(const Stage& st, const std::vector<std::uint32_t>& desc_taken,
                                std::vector<std::uint32_t>& need_full,
                                std::vector<std::uint32_t>& need_empty,
                                std::vector<std::uint32_t>& taken) {
        std::vector<std::uint8_t> is_full(trace_->channels.size(), 0);
        std::vector<std::uint8_t> is_taken(trace_->channels.size(), 0);
        std::vector<std::uint8_t> ext_taken(trace_->channels.size(), 0);
        for (std::uint32_t c : desc_taken) if (c < ext_taken.size()) ext_taken[c] = 1;
        for (const Op& op : st.ops) {
            for (ValueId a : op.args) {
                const Value* v = trace_->value(a);
                if (!v) continue;
                if (v->source == ValueSource::ChannelTake) { is_full[v->channel] = 1; is_taken[v->channel] = 1; }
                else if (v->source == ValueSource::ChannelRead) { is_full[v->channel] = 1; }
            }
        }
        // Takes/reads whose result is unused (e.g. §6.2 klen/kvm drain) aren't in
        // any op's args — the translator records them explicitly.
        for (ChannelId c : st.takes) if (c < is_full.size()) { is_full[c] = 1; is_taken[c] = 1; }
        for (ChannelId c : st.reads) if (c < is_full.size()) is_full[c] = 1;
        for (std::size_t c = 0; c < is_full.size(); ++c) {
            if (is_full[c]) need_full.push_back((std::uint32_t)c);
            if (is_taken[c]) taken.push_back((std::uint32_t)c);
        }
        // A put channel that is not take/read-first (in-stage or via a descriptor
        // consume) needs its pending cell empty.
        for (const ChannelPut& p : st.puts)
            if (!is_full[p.channel] && !ext_taken[p.channel]) need_empty.push_back(p.channel);
    }

    // Launch a stage's readiness check against its BAKED slot arrays (uploaded
    // once at bind). The arrays are persistent, so there is no per-pass malloc /
    // upload / sync — the kernel just ANDs into the pass commit flag on stream s.
    void launch_readiness_baked(const StageChannelLists& l, cudaStream_t s) {
        if (l.n_full == 0 && l.n_empty == 0) return;
        k_stage_readiness<<<1, 1, 0, s>>>(view_->d_full(), view_->d_head(), view_->d_tail(),
                                          view_->d_cap1(),
                                          l.d_full, l.n_full, l.d_empty, l.n_empty, d_commit_);
    }

    // Bake the pass's static readiness / commit slot lists (called from bind_view,
    // once the trace + view are fixed). The descriptor + per-stage full/empty
    // channel sets and the deduped commit taken/put sets depend ONLY on the trace
    // and the (now-bound) dense→slot view — never on per-fire `FireInputs` — so
    // they are remapped to registry slots + uploaded to persistent device arrays
    // here, removing the per-pass malloc/upload/free churn from `run_pass`.
    void bake_static_lists() {
        free_baked_lists();

        // Descriptor phase: non-const ports need their channel full; consuming
        // (token-family) ports also advance its head at commit (§5.1).
        std::vector<std::uint32_t> desc_full, desc_taken;
        for (const PortBinding& pb : trace_->ports) {
            if (pb.is_const) continue;
            desc_full.push_back(pb.channel);
            if (port_consumes(pb.port)) desc_taken.push_back(pb.channel);
        }
        upload_slots(view_->to_slots(desc_full), desc_lists_.d_full, desc_lists_.n_full);

        // Per-stage readiness + the pass-wide commit taken/put accumulation.
        std::vector<std::uint32_t> pass_taken = desc_taken, pass_put;
        stage_lists_.assign(trace_->stages.size(), StageChannelLists{});
        for (std::size_t i = 0; i < trace_->stages.size(); ++i) {
            const Stage& st = trace_->stages[i];
            std::vector<std::uint32_t> need_full, need_empty, taken;
            collect_stage_channels(st, desc_taken, need_full, need_empty, taken);
            upload_slots(view_->to_slots(need_full), stage_lists_[i].d_full, stage_lists_[i].n_full);
            upload_slots(view_->to_slots(need_empty), stage_lists_[i].d_empty, stage_lists_[i].n_empty);
            for (auto c : taken) pass_taken.push_back(c);
            for (const ChannelPut& p : st.puts) pass_put.push_back(p.channel);
        }

        // Register/SPSC semantics (T4): a channel consumed by multiple ops advances
        // its head exactly once; multiple puts publish once (last wins). Dedup,
        // then remap dense ids → shared registry slots for the commit-bump kernel.
        auto dedup = [](std::vector<std::uint32_t>& v) {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        };
        dedup(pass_taken);
        dedup(pass_put);
        host_commit_taken_slots_ = view_->to_slots(pass_taken);
        host_commit_put_slots_ = view_->to_slots(pass_put);
        upload_slots(host_commit_taken_slots_, d_commit_taken_, n_commit_taken_);
        upload_slots(host_commit_put_slots_, d_commit_put_, n_commit_put_);
    }

    // Upload a host slot list into a persistent device array (freeing any prior).
    static void upload_slots(const std::vector<std::uint32_t>& slots,
                             std::uint32_t*& d, std::uint32_t& n) {
        if (d) { cudaFree(d); d = nullptr; }
        n = (std::uint32_t)slots.size();
        if (n) {
            cudaMalloc(&d, (std::size_t)n * sizeof(std::uint32_t));
            cudaMemcpy(d, slots.data(), (std::size_t)n * sizeof(std::uint32_t),
                       cudaMemcpyHostToDevice);
        }
    }

    void free_baked_lists() {
        auto f = [](std::uint32_t*& d) { if (d) { cudaFree(d); d = nullptr; } };
        f(desc_lists_.d_full); f(desc_lists_.d_empty);
        desc_lists_.n_full = desc_lists_.n_empty = 0;
        for (StageChannelLists& l : stage_lists_) { f(l.d_full); f(l.d_empty); }
        stage_lists_.clear();
        f(d_commit_taken_); n_commit_taken_ = 0;
        f(d_commit_put_);   n_commit_put_ = 0;
    }

    bool run_stage(const Stage& st, const FireInputs& in, std::vector<void*>& scratch, std::string& err) {
        // Resolve every value referenced in this stage (roots first, on demand).
        for (const Op& op : st.ops) {
            for (ValueId a : op.args)
                if (!resolve_root(a, in, scratch, err)) return false;

            LaunchOp lo;
            if (!build_launch(op, in, scratch, lo, err)) return false;
            if (lo.code != OpCode::Reshape) {
                if (!launch_op(lo)) { err = "tier-0 uncovered op/dtype: " + std::string(op_name(op.code)); return false; }
            }
        }
        // Resolve values that outputs/puts reference but no op consumed (e.g. a
        // channel value put straight through).
        for (const ChannelPut& p : st.puts)
            if (!resolve_root(p.value, in, scratch, err)) return false;
        for (const Output& o : st.outputs)
            if (!resolve_root(o.value, in, scratch, err)) return false;
        return true;
    }

    void launch_pass_body(const FireInputs& in,
                          std::vector<void*>& scratch,
                          PassResult& res) {
        cudaStream_t s = in.stream;
        launch_readiness_baked(desc_lists_, s);
        for (std::size_t i = 0; i < trace_->stages.size(); ++i) {
            const Stage& st = trace_->stages[i];
            launch_readiness_baked(stage_lists_[i], s);
            if (!run_stage(st, in, scratch, res.error)) { res.ok = false; break; }
            for (const ChannelPut& p : st.puts) {
                auto it = val_ptr_.find(p.value);
                if (it == val_ptr_.end()) {
                    res.ok = false;
                    res.error = "put of unresolved value";
                    break;
                }
                cudaMemcpyAsync(view_->pending_cell(p.channel), it->second,
                                view_->cell_bytes(p.channel), cudaMemcpyDeviceToDevice, s);
            }
            if (!res.ok) break;
        }
    }

    static void free_scratch_async(const std::vector<void*>& scratch, cudaStream_t stream) {
        for (void* p : scratch) {
            cudaFreeAsync(p, stream);
        }
    }

    static void* alloc_scratch(std::size_t bytes,
                               cudaStream_t stream,
                               std::vector<void*>& scratch) {
        void* d = nullptr;
        cudaMallocAsync(&d, bytes, stream);
        scratch.push_back(d);
        return d;
    }

    // Materialize a root value (Const/Intrinsic/HostInput/Channel take-read) to a
    // device pointer, recorded in val_ptr_. Op results are recorded by build_launch.
    bool resolve_root(ValueId id, const FireInputs& in, std::vector<void*>& scratch, std::string& err) {
        if (val_ptr_.count(id)) return true;
        const Value* v = trace_->value(id);
        if (!v) { err = "unknown value id"; return false; }
        val_type_[id] = v->type;
        switch (v->source) {
            case ValueSource::OpResult:
                return true;  // defined by its producing op (build_launch)
            case ValueSource::Const: {
                std::size_t bytes = v->type.shape.numel() * dtype_size(v->type.dtype);
                if (bytes == 0) bytes = dtype_size(v->type.dtype);
                void* d = alloc_scratch(bytes, in.stream, scratch);
                // Broadcast the scalar literal across numel (Const tensors in the
                // examples are small trace-known vectors; tier-0 fills element 0
                // and relies on codegen const-fold for larger ones — here we fill
                // the whole buffer with the literal for scalar consts).
                std::uint64_t n = v->type.shape.numel() == 0 ? 1 : v->type.shape.numel();
                std::vector<std::uint8_t> host(bytes);
                for (std::uint64_t i = 0; i < n; ++i)
                    std::memcpy(host.data() + i * dtype_size(v->type.dtype), &v->lit.bits, dtype_size(v->type.dtype));
                cudaMemcpy(d, host.data(), bytes, cudaMemcpyHostToDevice);
                val_ptr_[id] = d;
                return true;
            }
            case ValueSource::Intrinsic: {
                if (v->intrinsic == Intrinsic::Logits || v->intrinsic == Intrinsic::MtpLogits) {
                    if (in.logits == nullptr) { err = "logits intrinsic unbound"; return false; }
                    // MtpLogits reads the K DRAFT rows at `mtp_draft_row` within the
                    // same `logits` base (a `[K, vocab]` matrix; the consuming op
                    // strides K rows from here). Logits reads the base (row 0). `-1`
                    // ⇒ MtpLogits falls back to the base (draft layout not yet set).
                    if (v->intrinsic == Intrinsic::MtpLogits && in.mtp_draft_row >= 0) {
                        const std::size_t elt = dtype_size(v->type.dtype);
                        val_ptr_[id] = const_cast<std::uint8_t*>(
                            static_cast<const std::uint8_t*>(in.logits)) +
                            static_cast<std::size_t>(in.mtp_draft_row) *
                                static_cast<std::size_t>(in.vocab) * elt;
                    } else {
                        val_ptr_[id] = const_cast<void*>(in.logits);
                    }
                    return true;
                }
                err = "tier-0: intrinsic not yet bound"; return false;
            }
            case ValueSource::HostInput: {
                auto it = in.host_inputs.find(v->host_key);
                if (it == in.host_inputs.end()) { err = "host input missing (late-bind miss)"; return false; }
                val_ptr_[id] = const_cast<void*>(it->second);
                return true;
            }
            case ValueSource::ChannelTake:
            case ValueSource::ChannelRead:
                val_ptr_[id] = view_->committed_cell(v->channel);
                return true;
        }
        err = "unhandled value source"; return false;
    }

    // Fill a LaunchOp from an op + its resolved operands; allocate result buffers.
    bool build_launch(const Op& op, const FireInputs& in, std::vector<void*>& scratch,
                      LaunchOp& lo, std::string& err) {
        lo.code = op.code;
        lo.stream = in.stream;
        lo.row_seeds = in.row_seeds;
        lo.rng_stream = op.imm;   // Gumbel uses imm as the stream salt
        for (ValueId a : op.args) {
            auto it = val_ptr_.find(a);
            if (it == val_ptr_.end()) { err = "operand unresolved"; return false; }
            lo.in.push_back(it->second);
        }
        const TensorType& rt = op.result_type;
        lo.out_dtype = rt.dtype;
        lo.numel = rt.shape.numel() == 0 ? 1 : rt.shape.numel();
        lo.rows = rt.shape.rows();
        lo.len = rt.shape.row_len();

        // Primary operand dtype/shape drives math dtype + row-local decomposition.
        DType prim = rt.dtype;
        Shape prim_shape = rt.shape;
        if (!op.args.empty()) {
            const Value* v0 = trace_->value(op.args[0]);
            if (v0) { prim = v0->type.dtype; prim_shape = v0->type.shape; }
        }
        lo.elem_dtype = prim;

        // Scalar-broadcast flags: an elementwise operand of numel 1 against a
        // wider result broadcasts (index 0). Covers `mul(logits, scalar)` etc.
        std::uint64_t out_n = rt.shape.numel() == 0 ? 1 : rt.shape.numel();
        auto operand_numel = [&](std::size_t k) -> std::uint64_t {
            if (k >= op.args.size()) return out_n;
            const Value* v = trace_->value(op.args[k]);
            if (!v) return out_n;
            std::uint64_t n = v->type.shape.numel();
            return n == 0 ? 1 : n;
        };
        if (op.args.size() >= 2) {
            lo.a_scalar = (operand_numel(0) == 1 && out_n > 1) ? 1 : 0;
            lo.b_scalar = (operand_numel(1) == 1 && out_n > 1) ? 1 : 0;
        }
        // select: broadcast the a/b operands (args[1], args[2]); cond is full.
        // Its element dtype is a/b's (= result dtype), NOT cond's (bool).
        if (op.code == OpCode::Select && op.args.size() == 3) {
            const Value* va = trace_->value(op.args[1]);
            if (va) lo.elem_dtype = va->type.dtype;
            lo.a_scalar = (operand_numel(1) == 1 && out_n > 1) ? 1 : 0;
            lo.b_scalar = (operand_numel(2) == 1 && out_n > 1) ? 1 : 0;
        }

        // Family-specific fixups.
        switch (op.code) {
            case OpCode::Reshape:                       // alias: result ptr = operand ptr
                val_ptr_[op.result_id] = lo.in.empty() ? nullptr : const_cast<void*>(lo.in[0]);
                val_type_[op.result_id] = rt;
                return true;
            case OpCode::ReduceSum: case OpCode::ReduceMax: case OpCode::ReduceArgmax:
            case OpCode::CumSum: case OpCode::CumProd:
            case OpCode::Softmax: case OpCode::LogSoftmax: case OpCode::L2Norm:
            case OpCode::RankLe: case OpCode::PivotThreshold:
                lo.rows = prim_shape.rows(); lo.len = prim_shape.row_len();
                lo.elem_dtype = prim;
                break;
            case OpCode::Gather: {                      // axis-0 generalized (§4)
                // src rank-1 → element gather (out[i]=src[idx[i]]); src rank≥2 →
                // ROW gather (out[i,:]=src[idx[i],:]). The container folds both
                // under tag 0x60; route by the src operand's rank.
                lo.elem_dtype = prim;
                const Value* src = op.args.empty() ? nullptr : trace_->value(op.args[0]);
                const Value* idx = op.args.size() < 2 ? nullptr : trace_->value(op.args[1]);
                if (src && src->type.shape.rank() >= 2) {
                    lo.code = OpCode::GatherRow;
                    lo.rows = idx ? (std::uint32_t)(idx->type.shape.numel() == 0 ? 1 : idx->type.shape.numel()) : 1;
                    lo.len = src->type.shape.row_len();   // product of src.dims[1..]
                }
                break;
            }
            case OpCode::ScatterSet: {                  // copy base → out, then scatter
                lo.n_scatter = 0;
                if (op.args.size() >= 2) { const Value* iv = trace_->value(op.args[1]);
                    if (iv) lo.n_scatter = (std::uint32_t)(iv->type.shape.numel() == 0 ? 1 : iv->type.shape.numel()); }
                lo.elem_dtype = prim; break;
            }
            case OpCode::Broadcast: {
                lo.rows = rt.shape.rows(); lo.len = rt.shape.row_len();
                lo.elem_dtype = rt.dtype;
                // General same-rank broadcast meta: [tdims(4), sstride(4)].
                std::uint32_t R = rt.shape.rank();
                if (R >= 1 && R <= 4) {
                    std::uint32_t meta[8] = {1,1,1,1, 0,0,0,0};
                    for (std::uint32_t d = 0; d < R; ++d) meta[d] = rt.shape.dims[d];
                    const Value* sv = op.args.empty() ? nullptr : trace_->value(op.args[0]);
                    std::uint32_t sd[4] = {1,1,1,1};
                    if (sv) {
                        // LEFT-align (echo infer.rs can_broadcast_to): src dim k →
                        // target position k; trailing positions padded with 1. So
                        // reduce([B,V])→[B] broadcasts [B]→[B,V] as out[i,j]=src[i].
                        std::uint32_t sr = sv->type.shape.rank();
                        for (std::uint32_t k = 0; k < sr && k < 4; ++k) sd[k] = sv->type.shape.dims[k];
                    }
                    std::uint32_t st = 1;
                    for (int d = (int)R - 1; d >= 0; --d) {
                        meta[4 + d] = (sd[d] == 1 && meta[d] > 1) ? 0u : st;
                        st *= sd[d];
                    }
                    auto* d_meta = static_cast<std::uint32_t*>(
                        alloc_scratch(sizeof(meta), in.stream, scratch));
                    cudaMemcpyAsync(d_meta, meta, sizeof(meta), cudaMemcpyHostToDevice, in.stream);
                    lo.bcast_meta = d_meta; lo.bcast_rank = R;
                }
                break;
            }
            case OpCode::Transpose:
                lo.rows = prim_shape.rows(); lo.len = prim_shape.row_len(); lo.elem_dtype = prim; break;
            case OpCode::TopK: case OpCode::SortDesc:
                // rows/len are the INPUT shape ([n] or [m,n]) — the axis top_k
                // scans — NOT the [k] result. k = the immediate.
                lo.rows = prim_shape.rows(); lo.len = prim_shape.row_len();
                lo.k = op.predicate.payload ? op.predicate.payload : op.imm;
                if (op.code == OpCode::SortDesc) lo.k = lo.len;
                break;
            case OpCode::Matmul:  // rows=M, len=K, k=N encoded in imm (N)
                lo.k = op.imm; break;
            case OpCode::MaskApplyPacked:              // rows/len from result [rows,V]; k = mask words/row
                lo.rows = rt.shape.rows(); lo.len = rt.shape.row_len();
                lo.k = (lo.len + 31) / 32; lo.elem_dtype = DType::F32; break;
            case OpCode::Rng:                          // ambient seed: stream=imm, kind→gumbel flag
                lo.rows = rt.shape.rows(); lo.len = rt.shape.row_len();
                lo.rng_stream = op.imm; lo.bcast_mode = (op.rng_kind == RngKind::Gumbel) ? 1 : 0;
                lo.row_seeds = in.row_seeds; break;
            case OpCode::RngKeyed:                     // state=in[0]; kind→gumbel flag
                lo.bcast_mode = (op.rng_kind == RngKind::Gumbel) ? 1 : 0; break;
            case OpCode::GumbelNoise:
                lo.rows = rt.shape.rows(); lo.len = rt.shape.row_len();
                lo.rng_stream = op.imm; lo.row_seeds = in.row_seeds; break;
            default: break;
        }
        if (op.code == OpCode::RankLe || op.code == OpCode::PivotThreshold)
            lo.k = op.predicate.payload ? op.predicate.payload : op.imm;

        // Allocate result buffer(s) (Reshape returned above).
        std::size_t out_bytes = lo.numel * dtype_size(rt.dtype);
        if (op.code == OpCode::TopK) out_bytes = (std::size_t)lo.rows * lo.k * sizeof(float);
        if (out_bytes == 0) out_bytes = dtype_size(rt.dtype);
        void* d_out = alloc_scratch(out_bytes, in.stream, scratch);
        lo.out = d_out;
        val_ptr_[op.result_id] = d_out;
        val_type_[op.result_id] = rt;

        if (op.code == OpCode::ScatterSet && !lo.in.empty()) {   // out starts as base copy
            cudaMemcpyAsync(d_out, lo.in[0], out_bytes, cudaMemcpyDeviceToDevice, in.stream);
        }
        if (op.code == OpCode::TopK) {   // second result = indices
            void* d_idx = alloc_scratch(
                (std::size_t)lo.rows * lo.k * sizeof(std::uint32_t),
                in.stream, scratch);
            lo.out2 = d_idx;
            if (op.result_count > 1) val_ptr_[op.result_id + 1] = d_idx;
        }
        return true;
    }

    const Trace* trace_ = nullptr;
    ChannelView* view_ = nullptr;               // dense→global-slot view (W0.1); not owned
    std::unique_ptr<DeviceChannelRegistry> standalone_reg_;  // owned only in standalone mode
    ChannelView standalone_view_;               // owned only in standalone mode
    std::uint32_t* d_commit_ = nullptr;         // pass-ephemeral commit flag
    // Baked-at-bind static slot lists (see bake_static_lists): descriptor-phase +
    // per-stage readiness arrays and the deduped commit-bump taken/put arrays,
    // resident on device so run_pass launches the ring kernels with no malloc.
    StageChannelLists desc_lists_;
    std::vector<StageChannelLists> stage_lists_;
    std::uint32_t* d_commit_taken_ = nullptr; std::uint32_t n_commit_taken_ = 0;
    std::uint32_t* d_commit_put_ = nullptr;   std::uint32_t n_commit_put_ = 0;
    std::vector<std::uint32_t> host_commit_taken_slots_;
    std::vector<std::uint32_t> host_commit_put_slots_;
    std::unordered_map<ValueId, void*> val_ptr_;
    std::unordered_map<ValueId, TensorType> val_type_;
};

}  // namespace pie_cuda_driver::ptir
