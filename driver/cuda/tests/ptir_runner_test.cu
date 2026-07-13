// PTIR tier-0 stage-runner test.
//
// Drives Tier0Runner over hand-built traces on the live GPU, verifying the
// readiness → predicated-commit → epoch-ring-bump machinery (overview §7.1):
//   1. loop-carried counter channel ping-pongs and commits across passes;
//   2. a readiness MISS (take on an empty channel) discards the pass (dummy-run,
//      no commit) and recovers once the channel is host-fed;
//   3. an argmax epilogue binds the logits intrinsic, runs a reduce op, and
//      publishes the sampled token into a channel;
//   4. a capacity-1 output channel exerts back-pressure (producer blocks until
//      the cell is drained).
//
// Standalone (no driver lib); needs a GPU.
//   nvcc -std=c++17 -arch=sm_89 --extended-lambda --expt-relaxed-constexpr \
//        -I../src tests/ptir_runner_test.cu -o ptir_runner_test && ./ptir_runner_test

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "support/host_eval.hpp"
#include "pipeline/tier0/tier0_runner.hpp"

using namespace pie_cuda_driver::pipeline;

namespace {
int g_pass = 0, g_fail = 0;
float* dev_ls(const std::vector<float>& h) {
    float* d = nullptr; cudaMalloc(&d, h.size() * sizeof(float));
    cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
    return d;
}
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

Value mk_value(ValueId id, TensorType ty, ValueSource src) {
    Value v; v.id = id; v.type = ty; v.source = src; return v;
}

// ── Trace 1: loop-carried counter, ctr := ctr + 1 each pass ──
void test_counter_pingpong() {
    std::printf("[counter ping-pong]\n");
    Trace t;
    Channel ctr; ctr.id = 0; ctr.type = {Shape::vec(1), DType::U32}; ctr.capacity = 1; ctr.has_seed = true;
    t.channels = {ctr};

    TensorType u32s{Shape::vec(1), DType::U32};
    t.values.push_back(mk_value(0, u32s, ValueSource::ChannelTake));  // v0 = ctr.take()
    t.values.back().channel = 0;
    Value c1 = mk_value(1, u32s, ValueSource::Const); c1.lit = Literal::u32(1); t.values.push_back(c1);  // v1 = 1
    t.values.push_back(mk_value(2, u32s, ValueSource::OpResult));      // v2 = add(v0, v1)

    Op add; add.code = OpCode::Add; add.args = {0, 1}; add.result_type = u32s; add.result_id = 2;
    Stage ep; ep.kind = StageKind::Epilogue; ep.ops = {add}; ep.puts = {{0, 2}};
    t.stages = {ep};

    Tier0Runner runner(t);
    std::uint32_t seed = 5;
    runner.arena().seed_cell(0, &seed, sizeof(seed));

    FireInputs in;
    for (std::uint32_t step = 0; step < 3; ++step) {
        PassResult r = runner.run_pass(in);
        std::uint32_t got = 0;
        runner.arena().read_committed(0, &got, sizeof(got));
        expect(r.ok && r.committed && got == seed + step + 1,
               "pass " + std::to_string(step) + " ctr==" + std::to_string(seed + step + 1) +
               " (got " + std::to_string(got) + ")");
    }
}

// ── Trace 2: readiness miss on an empty channel → dummy-run, then recover ──
void test_readiness_miss() {
    std::printf("[readiness miss + recovery]\n");
    Trace t;
    Channel ctr; ctr.id = 0; ctr.type = {Shape::vec(1), DType::U32}; ctr.capacity = 1; ctr.has_seed = true;
    Channel late; late.id = 1; late.type = {Shape::vec(1), DType::U32}; late.capacity = 1; late.has_seed = false;
    t.channels = {ctr, late};

    TensorType u32s{Shape::vec(1), DType::U32};
    Value v0 = mk_value(0, u32s, ValueSource::ChannelTake); v0.channel = 0; t.values.push_back(v0);
    Value v1 = mk_value(1, u32s, ValueSource::ChannelTake); v1.channel = 1; t.values.push_back(v1);
    t.values.push_back(mk_value(2, u32s, ValueSource::OpResult));   // v2 = add(ctr, late)

    Op add; add.code = OpCode::Add; add.args = {0, 1}; add.result_type = u32s; add.result_id = 2;
    Stage ep; ep.kind = StageKind::Epilogue; ep.ops = {add}; ep.puts = {{0, 2}};
    t.stages = {ep};

    Tier0Runner runner(t);
    std::uint32_t seed = 5; runner.arena().seed_cell(0, &seed, sizeof(seed));

    FireInputs in;
    PassResult r1 = runner.run_pass(in);
    std::uint32_t got = 0; runner.arena().read_committed(0, &got, sizeof(got));
    expect(r1.ok && !r1.committed && got == 5, "miss: no commit, ctr stays 5 (got " + std::to_string(got) + ")");

    std::uint32_t feed = 10; runner.arena().host_feed(1, &feed, sizeof(feed));
    PassResult r2 = runner.run_pass(in);
    runner.arena().read_committed(0, &got, sizeof(got));
    expect(r2.ok && r2.committed && got == 15, "recover: ctr==15 (got " + std::to_string(got) + ")");
}

// ── Trace 3: argmax epilogue — bind logits intrinsic, publish sampled token ──
void test_argmax_epilogue() {
    std::printf("[argmax epilogue]\n");
    const std::uint32_t V = 64;
    Trace t;
    Channel tok; tok.id = 0; tok.type = {Shape::vec(1), DType::U32}; tok.capacity = 1; tok.has_seed = false;
    tok.host_visible = true;
    t.channels = {tok};

    TensorType logits_ty{Shape::mat(1, V), DType::F32};
    TensorType tok_ty{Shape::vec(1), DType::U32};
    Value v0 = mk_value(0, logits_ty, ValueSource::Intrinsic); v0.intrinsic = Intrinsic::Logits; t.values.push_back(v0);
    t.values.push_back(mk_value(1, tok_ty, ValueSource::OpResult));   // v1 = argmax(logits)

    Op am; am.code = OpCode::ReduceArgmax; am.args = {0}; am.result_type = tok_ty; am.result_id = 1;
    Stage ep; ep.kind = StageKind::Epilogue; ep.ops = {am}; ep.puts = {{0, 1}};
    ep.outputs = {{1, OutputKind::Token}};
    t.stages = {ep};

    Tier0Runner runner(t);

    // logits with argmax at a known column, changing per pass.
    float* d_logits = nullptr; cudaMalloc(&d_logits, V * sizeof(float));
    for (std::uint32_t step = 0; step < 3; ++step) {
        std::vector<float> logits(V, 0.1f);
        std::uint32_t want = (step * 17 + 5) % V;
        logits[want] = 9.0f;
        cudaMemcpy(d_logits, logits.data(), V * sizeof(float), cudaMemcpyHostToDevice);
        FireInputs in; in.logits = d_logits; in.vocab = V;
        PassResult r = runner.run_pass(in);
        std::uint32_t got = 0; runner.arena().host_take(0, &got, sizeof(got));   // host harvests out.take()
        expect(r.ok && r.committed && got == want,
               "pass " + std::to_string(step) + " token==" + std::to_string(want) +
               " (got " + std::to_string(got) + ")");
    }
    cudaFree(d_logits);
}

// ── Trace 4: capacity-1 output channel back-pressure ──
void test_backpressure() {
    std::printf("[capacity-1 back-pressure]\n");
    Trace t;
    Channel out; out.id = 0; out.type = {Shape::vec(1), DType::U32}; out.capacity = 1; out.has_seed = false;
    out.host_visible = true;
    t.channels = {out};

    TensorType u32s{Shape::vec(1), DType::U32};
    Value c = mk_value(0, u32s, ValueSource::Const); c.lit = Literal::u32(42); t.values.push_back(c);

    // Pure producer: put a constant into `out` each pass (no take → first op is put).
    Stage ep; ep.kind = StageKind::Epilogue; ep.ops = {}; ep.puts = {{0, 0}};
    t.stages = {ep};

    Tier0Runner runner(t);
    FireInputs in;
    PassResult r1 = runner.run_pass(in);
    expect(r1.ok && r1.committed, "pass 0 commits (ring has room)");
    // Second pass without draining: ring now full (1 unconsumed ≥ capacity 1) →
    // need_empty fails → back-pressure, no commit.
    PassResult r2 = runner.run_pass(in);
    expect(r2.ok && !r2.committed, "pass 1 back-pressured (producer blocks until drained)");
    // Host drains the committed cell (out.take()) → a slot frees → producer resumes.
    std::uint32_t got = 0;
    runner.arena().host_take(0, &got, sizeof(got));
    expect(got == 42, "drained value == 42 (got " + std::to_string(got) + ")");
    PassResult r3 = runner.run_pass(in);
    expect(r3.ok && r3.committed, "pass 2 commits after drain (back-pressure released)");
}

// ── Trace 5: rank-2 gather via the runner routes to ROW gather (§4 axis-0) ──
void test_gather_row_routing() {
    std::printf("[rank-2 gather → row gather]\n");
    Trace t;
    Channel src; src.id = 0; src.type = {Shape::mat(3, 4), DType::U32}; src.capacity = 1; src.has_seed = true;
    Channel idx; idx.id = 1; idx.type = {Shape::vec(2), DType::U32}; idx.capacity = 1; idx.has_seed = true;
    Channel out; out.id = 2; out.type = {Shape::mat(2, 4), DType::U32}; out.capacity = 1; out.host_visible = true;
    t.channels = {src, idx, out};

    TensorType srct{Shape::mat(3, 4), DType::U32}, idxt{Shape::vec(2), DType::U32}, outt{Shape::mat(2, 4), DType::U32};
    Value v0 = mk_value(0, srct, ValueSource::ChannelTake); v0.channel = 0; t.values.push_back(v0);
    Value v1 = mk_value(1, idxt, ValueSource::ChannelTake); v1.channel = 1; t.values.push_back(v1);
    t.values.push_back(mk_value(2, outt, ValueSource::OpResult));   // v2 = gather(src, idx)

    Op g; g.code = OpCode::Gather; g.args = {0, 1}; g.result_type = outt; g.result_id = 2;
    Stage ep; ep.kind = StageKind::Epilogue; ep.ops = {g}; ep.puts = {{2, 2}};
    t.stages = {ep};

    Tier0Runner runner(t);
    std::vector<std::uint32_t> srcv{0,1,2,3, 4,5,6,7, 8,9,10,11};
    std::vector<std::uint32_t> idxv{2, 0};   // rows 2 then 0
    runner.arena().seed_cell(0, srcv.data(), srcv.size() * 4);
    runner.arena().seed_cell(1, idxv.data(), idxv.size() * 4);

    FireInputs in;
    PassResult r = runner.run_pass(in);
    std::vector<std::uint32_t> got(8, 0);
    runner.arena().host_take(2, got.data(), got.size() * 4);
    std::vector<std::uint32_t> want{8,9,10,11, 0,1,2,3};
    bool ok = r.ok && r.committed && got == want;
    expect(ok, "gather([3,4], [2,0]) row-gathers to [[8,9,10,11],[0,1,2,3]]");
}

// ── Trace 6: log_softmax EXPANSION (reduce_max/bcast/sub/exp/reduce_sum/log/
//    bcast/sub) over [2,8] via the runner — isolates beam's cand computation ──
void test_log_softmax_expansion() {
    std::printf("[log_softmax expansion]\n");
    const std::uint32_t B = 2, V = 8;
    Trace t;
    Channel out; out.id = 0; out.type = {Shape::mat(B, V), DType::F32}; out.capacity = 1; out.host_visible = true;
    t.channels = {out};
    TensorType matt{Shape::mat(B, V), DType::F32}, vect{Shape::vec(B), DType::F32};
    // v0 = intrinsic logits [B,V]; expansion ids v1..v8; put v8
    Value lg = mk_value(0, matt, ValueSource::Intrinsic); lg.intrinsic = Intrinsic::Logits; t.values.push_back(lg);
    for (ValueId id = 1; id <= 8; ++id)
        t.values.push_back(mk_value(id, (id==1||id==5||id==6) ? vect : matt, ValueSource::OpResult));
    Stage ep; ep.kind = StageKind::Epilogue;
    auto op = [&](OpCode c, std::vector<ValueId> a, ValueId r, TensorType ty) { Op o; o.code=c; o.args=a; o.result_id=r; o.result_type=ty; ep.ops.push_back(o); };
    op(OpCode::ReduceMax, {0}, 1, vect);           // m
    op(OpCode::Broadcast, {1}, 2, matt);           // mb
    op(OpCode::Sub, {0, 2}, 3, matt);              // c
    op(OpCode::Exp, {3}, 4, matt);                 // e
    op(OpCode::ReduceSum, {4}, 5, vect);           // s
    op(OpCode::Log, {5}, 6, vect);                 // l
    op(OpCode::Broadcast, {6}, 7, matt);           // lb
    op(OpCode::Sub, {3, 7}, 8, matt);              // result
    ep.puts = {{0, 8}};
    t.stages = {ep};

    std::vector<float> logits(B * V, 0.f); logits[3] = 8.f; logits[V + 5] = 7.f;
    float* d_logits = dev_ls(logits);
    Tier0Runner runner(t);
    FireInputs in; in.logits = d_logits; in.vocab = V;
    PassResult r = runner.run_pass(in);
    std::vector<float> got(B * V); runner.arena().host_take(0, got.data(), got.size() * 4);
    const auto maximum =
        host_eval::reduce(RedKind::Max, logits, B, V);
    std::vector<float> centered(logits.size());
    std::vector<float> exponentials(logits.size());
    for (std::uint32_t row = 0; row < B; ++row) {
        for (std::uint32_t column = 0; column < V; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * V + column;
            centered[index] = logits[index] - maximum[row];
            exponentials[index] = std::exp(centered[index]);
        }
    }
    const auto sum =
        host_eval::reduce(RedKind::Sum, exponentials, B, V);
    std::vector<float> want(logits.size());
    for (std::uint32_t row = 0; row < B; ++row) {
        for (std::uint32_t column = 0; column < V; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * V + column;
            want[index] = centered[index] - std::log(sum[row]);
        }
    }
    bool ok = r.ok && r.committed;
    for (std::size_t i = 0; ok && i < got.size(); ++i)
        ok = std::fabs(got[i] - want[i]) <= 1e-4f + 1e-4f * std::fabs(want[i]);
    expect(ok, "log_softmax expansion matches fused (got[3]=" + std::to_string(got[3]) + " want=" + std::to_string(want[3]) + ")");
    if (!ok) std::printf("        got: [%g %g %g %g ...] want: [%g %g %g %g ...]\n", got[0],got[1],got[2],got[3], want[0],want[1],want[2],want[3]);
    cudaFree(d_logits);
}

void test_lazy_packed_bool_pull() {
    std::printf("[lazy packed-bool host pull]\n");
    std::uint8_t* mirror = nullptr;
    std::uint64_t* words = nullptr;
    cudaHostAlloc(&mirror, 2, cudaHostAllocDefault);
    cudaHostAlloc(&words, 4 * sizeof(std::uint64_t), cudaHostAllocDefault);
    std::memset(mirror, 0, 2);
    std::memset(words, 0, 4 * sizeof(std::uint64_t));
    mirror[0] = 0b10000101;
    words[1] = 1;

    std::uint8_t* cells = nullptr;
    std::uint8_t* full = nullptr;
    std::uint32_t* commit = nullptr;
    cudaMalloc(&cells, 16);
    cudaMalloc(&full, kMaxRing);
    cudaMalloc(&commit, sizeof(std::uint32_t));
    cudaMemset(cells, 0, 16);
    cudaMemset(full, 0, kMaxRing);
    const std::uint32_t one = 1;
    cudaMemcpy(commit, &one, sizeof(one), cudaMemcpyHostToDevice);

    const DeviceHostChannelTicket ticket{
        .slot = 0,
        .flags = kTicketConsume | kTicketHostWriter |
                 kTicketPackedBool | kTicketRequireInput,
        .expected_head = 0,
        .expected_tail = kNoChannelTicket,
        .words = words,
        .mirror = mirror,
        .cells = cells,
        .cap1 = 2,
        .wire_bytes = 1,
        .native_bytes = 8,
    };
    DeviceHostChannelTicket* uploaded =
        launch_pull_validate_host_channels({ticket}, full, commit, nullptr);
    cudaDeviceSynchronize();
    std::uint8_t native[8]{};
    std::uint32_t committed = 0;
    cudaMemcpy(native, cells, sizeof(native), cudaMemcpyDeviceToHost);
    cudaMemcpy(&committed, commit, sizeof(committed), cudaMemcpyDeviceToHost);
    expect(
        committed == 1 &&
            native[0] == 1 && native[1] == 0 && native[2] == 1 &&
            native[7] == 1,
        "published packed bool is acquired and unpacked on device");
    cudaFree(uploaded);

    words[1] = 0;
    cudaMemcpy(commit, &one, sizeof(one), cudaMemcpyHostToDevice);
    uploaded = launch_pull_validate_host_channels(
        {ticket}, full, commit, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&committed, commit, sizeof(committed), cudaMemcpyDeviceToHost);
    expect(committed == 0, "withheld writer value clears commit without blocking");

    cudaFree(uploaded);
    DeviceHostChannelTicket later_ticket = ticket;
    later_ticket.expected_head = 1;
    words[0] = 0;
    words[1] = 2;
    cudaMemcpy(commit, &one, sizeof(one), cudaMemcpyHostToDevice);
    uploaded = launch_pull_validate_host_channels(
        {later_ticket}, full, commit, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&committed, commit, sizeof(committed), cudaMemcpyDeviceToHost);
    expect(
        committed == 0,
        "later fire ticket cannot steal a retried predecessor's entry");
    cudaFree(uploaded);

    const DeviceHostChannelTicket publish_ticket{
        .slot = 0,
        .flags = kTicketPublish,
        .expected_head = kNoChannelTicket,
        .expected_tail = 1,
        .words = words,
        .mirror = mirror,
        .cells = cells,
        .cap1 = 2,
        .wire_bytes = 1,
        .native_bytes = 8,
    };
    words[0] = 0;
    words[1] = 1;
    cudaMemcpy(commit, &one, sizeof(one), cudaMemcpyHostToDevice);
    uploaded = launch_pull_validate_host_channels(
        {publish_ticket}, full, commit, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&committed, commit, sizeof(committed), cudaMemcpyDeviceToHost);
    expect(committed == 0, "full host-reader ring retries at device commit");
    cudaFree(uploaded);

    words[0] = 1;
    cudaMemcpy(commit, &one, sizeof(one), cudaMemcpyHostToDevice);
    uploaded = launch_pull_validate_host_channels(
        {publish_ticket}, full, commit, nullptr);
    cudaDeviceSynchronize();
    cudaMemcpy(&committed, commit, sizeof(committed), cudaMemcpyDeviceToHost);
    expect(committed == 1, "reader capacity released before execution commits");
    cudaFree(uploaded);

    cudaFree(commit);
    cudaFree(full);
    cudaFree(cells);
    cudaFreeHost(words);
    cudaFreeHost(mirror);
}

}  // namespace

int main() {
    cudaDeviceProp prop{}; int dev = 0;
    cudaGetDevice(&dev); cudaGetDeviceProperties(&prop, dev);
    std::printf("PTIR tier-0 stage-runner — device: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

    test_counter_pingpong();
    test_readiness_miss();
    test_argmax_epilogue();
    test_backpressure();
    test_gather_row_routing();
    test_log_softmax_expansion();
    test_lazy_packed_bool_pull();

    std::printf("\n==== runner: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
