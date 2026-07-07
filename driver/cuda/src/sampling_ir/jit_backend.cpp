// SamplingIrBackend — codegen + JIT facade behind echo's IProgramBackend.
// See jit_backend.hpp for the design.

#include "sampling_ir/jit_backend.hpp"

#include <algorithm>
#include <exception>

#include "sampling_ir/codegen.hpp"
#include "sampling_ir/ir.hpp"
#include "sampling_ir/program_identity.hpp"
#include "sampling_ir/reader.hpp"

namespace pie_cuda_driver::sampling_ir {

namespace {

jit::ScalarType to_jit_dtype(DType d) {
    switch (d) {
        case DType::F32: return jit::ScalarType::F32;
        case DType::I32: return jit::ScalarType::I32;
        case DType::U32: return jit::ScalarType::U32;
        case DType::Bool: return jit::ScalarType::U32;  // 1B value, hint only
    }
    return jit::ScalarType::U32;
}

BindingClass to_binding_class(BufferClass cls) {
    switch (cls) {
        case BufferClass::IntrinsicLogits: return BindingClass::Intrinsic;
        case BufferClass::HostSubmit: return BindingClass::HostSubmit;
        case BufferClass::HostLate: return BindingClass::HostLate;
        default: return BindingClass::Const;  // unreachable for inputs
    }
}

// Map charlie's codegen launch shape onto the JIT's per-fire grid mode so the
// JIT recomputes grid.x from num_rows each fire (M>1). Custom dims bake (Fixed).
jit::GridShape to_jit_grid_shape(LaunchShape s) {
    switch (s) {
        case LaunchShape::OneBlockPerRow: return jit::GridShape::OneBlockPerRow;
        case LaunchShape::GridStrideOverVocab: return jit::GridShape::GridStrideOverVocab;
        case LaunchShape::GridStrideOverLen: return jit::GridShape::GridStrideOverLen;
        case LaunchShape::Custom: return jit::GridShape::Fixed;
    }
    return jit::GridShape::Fixed;
}

// Translate one codegen KernelArg into the JIT's arg taxonomy.
jit::KernelArg to_jit_arg(const KernelArg& a) {
    switch (a.kind) {
        case KernelArgKind::Buffer:
            return jit::KernelArg::buffer_arg(a.buffer);
        case KernelArgKind::Scalar:
            switch (a.scalar_dtype) {
                case DType::F32: return jit::KernelArg::f32(a.scalar.f32);
                case DType::I32: return jit::KernelArg::i32(a.scalar.i32);
                case DType::U32: return jit::KernelArg::u32(a.scalar.u32);
                case DType::Bool: return jit::KernelArg::u32(a.scalar.u32 ? 1u : 0u);
            }
            return jit::KernelArg::u32(a.scalar.u32);
        case KernelArgKind::Param: {
            // prng_offset is 64-bit; row/vocab counts are 32-bit.
            const auto ty = (a.param_index ==
                             static_cast<std::uint32_t>(ParamSlot::PrngOffset))
                                ? jit::ScalarType::U64
                                : jit::ScalarType::U32;
            return jit::KernelArg::param_arg(a.param_index, ty);
        }
    }
    return jit::KernelArg::u32(0);
}

// Map the codegen-facing manifest (Logits | HostTensor) → in-memory per-slot
// bindings for decode_v4 (mirrors lower_bytecode_v4's mapping). One Binding per
// input slot, in slot order.
std::vector<Binding> manifest_to_slot_bindings(const ProgramManifest& manifest) {
    std::vector<Binding> slots;
    slots.reserve(manifest.size());
    for (const InputBind& b : manifest) {
        Binding bd;
        if (b.kind == BindKind::Logits) {
            bd.tag = BindingTag::Intrinsic;
            // Propagate the intrinsic KIND (Logits | MtpLogits | MtpDrafts) — the
            // manifest carries it (executor.cpp sets b.intrinsic_kind from wire
            // kind 2=MtpLogits / 3=MtpDrafts). Hardcoding Logits here dropped it
            // for EVERY fire → the resolver read the sampled row for MtpLogits
            // (drafts = target-greedy, not the produce'd MTP rows) and mis-typed
            // MtpDrafts as a logits row → the (b) A/B zero-cascade.
            bd.intrinsic = b.intrinsic_kind;
        } else {
            bd.tag = BindingTag::Host;
            bd.host_key = b.host_key;
            bd.host_avail = b.ready;
        }
        slots.push_back(bd);
    }
    return slots;
}

}  // namespace

SamplingIrBackend::SamplingIrBackend(bool batched_lowering)
    : engine_(), batched_lowering_(batched_lowering) {}

ProgramHandle SamplingIrBackend::get_or_compile(std::span<const std::uint8_t> bytecode) {
    return get_or_compile(bytecode, ProgramManifest{});
}

ProgramHandle SamplingIrBackend::get_or_compile(std::span<const std::uint8_t> bytecode,
                                                const ProgramManifest& manifest) {
    last_error_.clear();
    if (bytecode.empty()) {
        last_error_ = "empty bytecode";
        return kInvalidProgram;
    }

    // #11 cost contract: the program IDENTITY (== this dedup key minus the
    // batched bit) is the single shared key for #10 distinct-count + echo's
    // M-batch grouping + #11 compile-dedup (see program_identity.hpp). v4
    // bytecode is binding-free, so the manifest is folded in — distinct bindings
    // ⇒ distinct compiled program.
    const std::uint64_t identity = program_identity_hash(bytecode, manifest);
    // The M=1-vs-batched lowering yields distinct PTX artifacts, so fold the
    // process-constant batched bit on top for the COMPILE cache key (NOT the
    // shared identity, which is pre-lowering — see program_identity.hpp banner).
    const std::uint64_t keyed =
        batched_lowering_ ? (identity ^ 0x9E3779B97F4A7C15ULL) : identity;
    const ProgramHandle handle = keyed ? keyed : 1;  // reserve 0 = kInvalidProgram
    if (programs_.find(handle) != programs_.end()) return handle;

    // 1. Decode the IR -> Program (input/output binding metadata: host keys,
    //    kinds). The version routing (v4 binding-free via `manifest` vs v2/v3
    //    self-binding) lives in decode_program, shared with the thread-safe
    //    prefetch path.
    Program program;
    std::string derr;
    if (!decode_program(bytecode, manifest, program, &derr)) {
        last_error_ = derr;
        return kInvalidProgram;
    }
    return compile_decoded(program, keyed);
}

ProgramHandle SamplingIrBackend::compile_decoded(const Program& program,
                                                 std::uint64_t keyed) {
    const ProgramHandle handle = keyed ? keyed : 1;  // reserve 0 = kInvalidProgram

    // 2. Lower IR -> KernelDAG (batched with M=1 fallback) via lower_with_fallback,
    //    shared with prefetch_compile.
    bool batched_mode = false;
    LowerResult lr = lower_with_fallback(program, batched_mode);
    if (!lr.ok) {
        last_error_ = "lower failed: " + lr.error;
        return kInvalidProgram;
    }

    // 3. Adapt the codegen KernelDAG -> jit::KernelDAG (shared with prefetch via
    //    build_jdag).
    jit::KernelDAG jdag = build_jdag(lr, keyed);

    // 4. NVRTC-compile + allocate via the JIT engine.
    jit::CompiledProgram* prog = nullptr;
    try {
        prog = &engine_.get_or_compile(jdag);
    } catch (const std::exception& e) {
        last_error_ = std::string("jit compile failed: ") + e.what();
        return kInvalidProgram;
    }

    // 5. Build the declared interface + per-fire binding maps.
    Entry entry;
    entry.prog = prog;
    entry.batched_mode = batched_mode;
    std::size_t n_out = 0;
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls == BufferClass::Output)
            n_out = std::max<std::size_t>(n_out, static_cast<std::size_t>(b.output_index) + 1);
    entry.output_buffers.assign(n_out, 0);
    entry.iface.outputs.resize(n_out);

    const Slot* slot0 = program.slots.empty() ? nullptr : &program.slots.front();
    for (const BufferDecl& b : lr.dag.buffers) {
        // Per-row byte stride for the custom-batch replay slicing. The intrinsic
        // logits is bf16 (`Phys::BF16`), but its BufferDecl.dtype is a non-allocated
        // F32 placeholder (codegen.cpp), so byte_size() would over-count — use
        // elem_count*2. Host/output/row-seed buffers carry their true dtype.
        if (b.cls != BufferClass::Intermediate) {
            // IntrinsicLogits is bf16 on device (dtype is an F32 placeholder, so
            // byte_size() would over-count) → elem_count*2. The MtpDrafts
            // intrinsic is the exception: a real I32 [k] buffer, so use its true
            // byte_size() (elem_count*4), not the bf16 stride.
            entry.row_stride_bytes[b.id] =
                (b.cls == BufferClass::IntrinsicLogits && b.dtype != DType::I32)
                    ? static_cast<std::uint64_t>(b.elem_count) * 2u
                    : b.byte_size();
        }
        switch (b.cls) {
            case BufferClass::IntrinsicLogits:
            case BufferClass::HostSubmit:
            case BufferClass::HostLate: {
                InputDecl in;
                in.input_id = b.input_id;
                in.cls = to_binding_class(b.cls);
                // Bridge the manifest intrinsic (ir::Intrinsic, stamped onto
                // BufferDecl.intrinsic_kind from the program manifest) to the
                // runtime IntrinsicKind echo's resolver reads: MtpLogits selects
                // ws.logits[mtp_draft_row], Logits the sampled row.
                in.intrinsic = (b.intrinsic_kind == Intrinsic::MtpLogits)
                                   ? IntrinsicKind::MtpLogits
                               : (b.intrinsic_kind == Intrinsic::MtpDrafts)
                                   ? IntrinsicKind::MtpDrafts
                                   : IntrinsicKind::Logits;
                in.host_key = (b.input_id < program.inputs.size())
                                  ? program.inputs[b.input_id].binding.host_key
                                  : 0;
                in.elem_count = b.elem_count;
                entry.iface.inputs.push_back(in);
                entry.input_to_buffer[b.input_id] = b.id;
                break;
            }
            case BufferClass::Output: {
                DeclaredOutput od;
                od.value_id = (slot0 && b.output_index < slot0->outputs.size())
                                  ? slot0->outputs[b.output_index].value
                                  : 0;
                od.cls = static_cast<OutputClass>(b.output_kind);
                od.elem_count = b.elem_count;
                entry.iface.outputs[b.output_index] = od;
                entry.output_buffers[b.output_index] = b.id;
                break;
            }
            case BufferClass::Intermediate:
                break;  // JIT-owned scratch; not part of the interface
            case BufferClass::RowSeed:
                // Ambient per-row seed S (Model B Op::Rng): external, bind-only,
                // bound per-fire from LaunchArgs.row_seeds. NOT an Op::Input, so it
                // is not part of the declared interface echo resolves by input_id;
                // codegen reads rowseed[r] (seed_eff_stream). One per RNG program.
                entry.row_seed_buffer = b.id;
                entry.has_row_seed = true;
                break;
        }
    }

    programs_.emplace(handle, std::move(entry));
    return handle;
}

bool SamplingIrBackend::decode_program(std::span<const std::uint8_t> bytecode,
                                       const ProgramManifest& manifest,
                                       Program& out, std::string* err) const {
    // Route by the bytecode's AUTHORITATIVE PSIR version field (the "PSIR" magic
    // then a little-endian u16 at offset 4), NOT by `manifest.empty()`. The old
    // emptiness proxy mis-routed a v4 program that arrived with an empty manifest
    // to the v2/v3 `decode`, which rejects version 4 (`reader.cpp` BadVersion) —
    // the carrier→driver consolidation seam. v3/v2 self-bind (`decode`); v4 is
    // binding-free and takes per-slot bindings from `manifest` (`decode_v4`). NB:
    // v4 still binds one slot per input (decode_v4 asserts size==n_inputs), so a
    // v4 program WITH inputs still needs its manifest populated by the carrier.
    std::uint16_t psir_version = 0;
    if (bytecode.size() >= 6 && bytecode[0] == 'P' && bytecode[1] == 'S' &&
        bytecode[2] == 'I' && bytecode[3] == 'R') {
        psir_version = static_cast<std::uint16_t>(
            bytecode[4] | (static_cast<std::uint16_t>(bytecode[5]) << 8));
    }
    DecodeError derr;
    const bool decoded =
        psir_version >= 4
            ? decode_v4(bytecode.data(), bytecode.size(),
                        manifest_to_slot_bindings(manifest), out, &derr)
            : decode(bytecode.data(), bytecode.size(), out, &derr);
    if (!decoded) {
        if (err) *err = "decode failed: " + derr.detail;
        return false;
    }
    return true;
}

LowerResult SamplingIrBackend::lower_with_fallback(const Program& program,
                                                   bool& batched_mode) const {
    // Batched lowering emits one grid=num_rows kernel; FALLBACK to M=1 when the
    // batched emit rejects an op (Gather/GatherRow/Scatter*/SortDesc are M=1-only,
    // e.g. mirostat's gather) so custom samplers stay green while standard samplers
    // ride the batched fast path. "always-batched" is the safe production default.
    batched_mode = batched_lowering_;
    LowerResult lr = lower(program, LowerOptions{/*batched=*/batched_lowering_});
    if (batched_lowering_ && !lr.ok) {
        LowerResult m1 = lower(program, LowerOptions{/*batched=*/false});
        if (m1.ok) {
            lr = std::move(m1);
            batched_mode = false;  // M=1 per-row fallback
        }
    }
    return lr;
}

jit::KernelDAG SamplingIrBackend::build_jdag(const LowerResult& lr,
                                             std::uint64_t keyed) const {
    // Adapt codegen KernelDAG -> jit::KernelDAG. Grid is recomputed per fire from
    // num_rows (Param 0) via the launch shape, so M>1 batches scale in a single
    // launch; vocab for grid-stride comes from the launch param table.
    std::uint32_t vocab = 0;
    for (const BufferDecl& b : lr.dag.buffers)
        if (b.cls == BufferClass::IntrinsicLogits) vocab = b.elem_count;

    jit::KernelDAG jdag;
    jdag.hash = keyed;
    jdag.buffers.reserve(lr.dag.buffers.size());
    for (const BufferDecl& b : lr.dag.buffers) {
        jit::BufferDecl jb;
        jb.id = b.id;
        jb.size_bytes = b.byte_size();  // per-row; JIT scales by num_rows if batched
        jb.dtype = to_jit_dtype(b.dtype);
        jb.external = (b.cls != BufferClass::Intermediate);
        jb.batched = b.batched;  // batched lowering: per-row size, JIT scales by num_rows
        jdag.buffers.push_back(jb);
    }
    jdag.kernels.reserve(lr.dag.kernels.size());
    for (const KernelDesc& k : lr.dag.kernels) {
        std::uint32_t len = 0;
        if (k.shape == LaunchShape::GridStrideOverLen &&
            k.len_buffer < lr.dag.buffers.size()) {
            len = lr.dag.buffers[k.len_buffer].elem_count;  // per-row element count
        }
        // Baked dims (M=1 / Custom fallback); dynamic shapes recompute grid.x/fire.
        const LaunchDims d = compute_launch_dims(k.shape, /*num_rows=*/1, vocab,
                                                 len, k.custom_grid, k.custom_block);
        jit::KernelDef jk;
        jk.name = k.entry_name;
        jk.source = k.source;
        jk.grid = {d.grid_x, 1, 1};
        jk.block = {d.block_x, 1, 1};
        jk.shared_bytes = k.shared_bytes;
        jk.grid_shape = to_jit_grid_shape(k.shape);
        jk.per_row_len = len;
        jk.args.reserve(k.args.size());
        for (const KernelArg& a : k.args) jk.args.push_back(to_jit_arg(a));
        jdag.kernels.push_back(std::move(jk));
    }
    return jdag;
}

void SamplingIrBackend::prefetch_compile(std::span<const std::uint8_t> bytecode,
                                         const ProgramManifest& manifest) {
    // Fire-and-forget, ANY THREAD (called at admission, off the context thread):
    // warm the off-context PTX cache so the later context-thread get_or_compile
    // finds the entry Ready (cache hit) -> TTFT win. Idempotent + dedup'd by the
    // SAME identity key as get_or_compile/#10 (engine_.prefetch_compile keys on
    // jdag.hash). Touches NO mutable backend state (programs_/last_error_), so it
    // races nothing: decode_program (const, errors via out-param) + lower_with_
    // fallback (pure) + build_jdag (pure) + the mu_-guarded JitEngine. The
    // programs_ membership check is intentionally skipped (it's not thread-safe
    // against the context thread); a redundant prefetch re-lowers (cheap) but the
    // engine's compile-cache dedups the expensive NVRTC step.
    if (bytecode.empty()) return;
    const std::uint64_t identity = program_identity_hash(bytecode, manifest);
    const std::uint64_t keyed =
        batched_lowering_ ? (identity ^ 0x9E3779B97F4A7C15ULL) : identity;
    Program program;
    if (!decode_program(bytecode, manifest, program, /*err=*/nullptr)) return;
    bool batched_mode = false;
    LowerResult lr = lower_with_fallback(program, batched_mode);
    if (!lr.ok) return;
    jit::KernelDAG jdag = build_jdag(lr, keyed);
    engine_.prefetch_compile(jdag);
}

const ProgramInterface& SamplingIrBackend::interface(ProgramHandle program) {
    static const ProgramInterface kEmpty;
    auto it = programs_.find(program);
    return it == programs_.end() ? kEmpty : it->second.iface;
}

void SamplingIrBackend::bind_external(Entry& e, const LaunchArgs& args, int row) {
    const auto offset = [&](jit::BufferId id) -> CUdeviceptr {
        auto sit = e.row_stride_bytes.find(id);
        const std::uint64_t stride = (sit == e.row_stride_bytes.end()) ? 0 : sit->second;
        return static_cast<CUdeviceptr>(static_cast<std::uint64_t>(row) * stride);
    };
    // Resolved external inputs (Const is inlined by codegen -> skip), sliced to row.
    for (const ResolvedInput& ri : args.inputs) {
        if (ri.cls == BindingClass::Const || ri.device_ptr == nullptr) continue;
        auto bit = e.input_to_buffer.find(ri.input_id);
        if (bit == e.input_to_buffer.end()) continue;
        engine_.bind_buffer(*e.prog, bit->second,
                            reinterpret_cast<CUdeviceptr>(ri.device_ptr) + offset(bit->second));
    }
    // Declared output slots, sliced to row.
    const std::size_t n = std::min(args.output_ptrs.size(), e.output_buffers.size());
    for (std::size_t i = 0; i < n; ++i) {
        engine_.bind_buffer(*e.prog, e.output_buffers[i],
                            reinterpret_cast<CUdeviceptr>(args.output_ptrs[i]) +
                                offset(e.output_buffers[i]));
    }
    // Ambient per-row RNG seed (Model B): RowSeed is not an Op::Input, so it is
    // delivered out-of-band via LaunchArgs.row_seeds. Programs with no Op::Rng
    // declare no RowSeed buffer (has_row_seed=false) and ignore it.
    if (e.has_row_seed && args.row_seeds != nullptr) {
        engine_.bind_buffer(*e.prog, e.row_seed_buffer,
                            reinterpret_cast<CUdeviceptr>(args.row_seeds) +
                                offset(e.row_seed_buffer));
    }
}

void SamplingIrBackend::launch(ProgramHandle program, const LaunchArgs& args,
                               cudaStream_t stream) {
    auto it = programs_.find(program);
    if (it == programs_.end()) {
        throw std::runtime_error("SamplingIrBackend::launch: unknown ProgramHandle");
    }
    Entry& e = it->second;
    const CUstream st = reinterpret_cast<CUstream>(stream);

    // Launch-context scalars, indexed by ParamSlot (0=rows, 1=vocab, 2=prng).
    const auto fire = [&](int num_rows) {
        std::vector<std::uint64_t> params(3, 0);
        params[static_cast<std::size_t>(ParamSlot::NumRows)] =
            static_cast<std::uint64_t>(num_rows);
        params[static_cast<std::size_t>(ParamSlot::VocabSize)] =
            static_cast<std::uint64_t>(args.vocab_size);
        params[static_cast<std::size_t>(ParamSlot::PrngOffset)] = args.prng_offset;
        engine_.launch(*e.prog, st, params);
    };

    if (!e.batched_mode && args.num_rows > 1) {
        // Custom (M=1-only: Gather/Scatter/SortDesc — e.g. mirostat) program over N
        // rows. The batched emit path rejected it, so replay the single-row program
        // once per row, slicing each external binding to row r — token-identical to
        // N separate num_rows=1 fires (same per-row seed/column axis), at the cost
        // of N launches instead of one grid=N. The deferred custom-batch path
        // (#10/#11 prerequisite): standard samplers stay on the one-launch batched
        // fast path; only batched-unsupported customs take this loop. Each launch
        // captures its row's pointers at enqueue, so the sequential rebinds are safe.
        for (int r = 0; r < args.num_rows; ++r) {
            bind_external(e, args, r);
            fire(/*num_rows=*/1);
        }
        return;
    }

    // Batched fast path (or M=1 at num_rows<=1): bind buffer bases (row 0 = no
    // offset) and issue one launch; the batched kernel strides per row internally.
    bind_external(e, args, /*row=*/0);
    fire(args.num_rows);
}

bool SamplingIrBackend::program_is_batched(ProgramHandle program) const {
    auto it = programs_.find(program);
    return it != programs_.end() && it->second.batched_mode;
}

}  // namespace pie_cuda_driver::sampling_ir

extern "C" void pie_sampling_ir_prefetch_trampoline(
    void* backend_ctx, const std::uint8_t* bytecode, std::size_t bytecode_len,
    const std::uint8_t* binds_kind, const std::uint32_t* binds_key,
    std::size_t binds_len) {
    using namespace pie_cuda_driver::sampling_ir;
    auto* backend = static_cast<IProgramBackend*>(backend_ctx);
    if (backend == nullptr || bytecode == nullptr || bytecode_len == 0) return;

    // Reconstruct the manifest EXACTLY as the submit path does (executor.cpp:3239-
    // 3250): ready = SubmitBound for every bind (default), so the identity hash
    // matches the real fire -> cache HIT. kind 1 = host tensor (carries key); kind
    // 0/2 = the logits intrinsic (2 = MtpLogits draft row); key/intrinsic default.
    ProgramManifest manifest;
    manifest.reserve(binds_len);
    for (std::size_t i = 0; i < binds_len; ++i) {
        InputBind b;  // defaults: Logits, host_key=0, SubmitBound, Intrinsic::Logits
        const std::uint32_t bk = binds_kind ? binds_kind[i] : 0u;
        if (bk == 1u) {  // KIND_TENSOR
            b.kind = BindKind::HostTensor;
            b.host_key = binds_key ? binds_key[i] : 0u;
            b.ready = HostAvailability::SubmitBound;
        } else {  // KIND_LOGITS (0) / KIND_MTP_LOGITS (2) / KIND_MTP_DRAFTS (3)
            b.kind = BindKind::Logits;
            if (bk == 2u) b.intrinsic_kind = Intrinsic::MtpLogits;
            else if (bk == 3u) b.intrinsic_kind = Intrinsic::MtpDrafts;
        }
        manifest.push_back(b);
    }
    backend->prefetch_compile(
        std::span<const std::uint8_t>(bytecode, bytecode_len), manifest);
}
