// Differential oracle for src/store/recurrent_state_cache.rs.
//
// Builds the REAL store/recurrent_state_cache.cpp and drives it over a grid of
// hybrid layer stacks, slot counts and geometries, transcribing:
//
//   * the allocation sequence (which buffers, in what order, how large),
//   * every stream operation reset/reset_slot/copy issues, with its exact
//     offset, pitch, width and row count,
//   * the byte offsets the per-(layer, slot) accessors hand out,
//   * the optional tiers (verify hidden stash, buffered-activation pool),
//   * every message it throws.
//
// The stream operations are the point. A `cudaMemset2DAsync` with the wrong
// pitch zeroes another layer's recurrent state and returns cudaSuccess; the
// model then produces slightly wrong tokens thousands of steps later. Nothing
// about it is checkable after the fact, so it is checked before the fact.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "cuda_runtime.h"
#include "store/recurrent_state_cache.hpp"

using pie_cuda_driver::RecurrentStateCache;

namespace {

const char kUnit = '\037';

std::string join(const std::vector<std::string>& parts) {
    std::string out;
    for (std::size_t i = 0; i < parts.size(); ++i) {
        if (i) out += kUnit;
        out += parts[i];
    }
    return out;
}

void emit(const std::string& id, const std::string& body) {
    std::printf("%s|%s\n", id.c_str(), body.c_str());
}

// Drain the recorder's log into one field and clear it, so each reported
// column holds exactly the operations that column's call issued.
std::string drain() {
    std::string out = join(oracle_cuda::log());
    oracle_cuda::reset_log();
    return out;
}

std::string layers_label(const std::vector<bool>& linear) {
    std::string s;
    for (bool b : linear) s += (b ? 'L' : '.');
    return s.empty() ? "-" : s;
}

// Byte offsets of every (layer, slot) accessor, plus the messages the
// out-of-range ones throw. Rendered relative to the buffer each pointer lands
// in, so a wrong layer stride shows as a wrong offset rather than as an
// indistinguishable address.
// Every scalar accessor the generated forward bodies read off the live
// cache, in one field per case. Appended to the grid cases rather than
// given a section of their own so every geometry in the sweep pins them —
// the strides are where a conv_dim/conv_kernel transposition becomes a
// number, and the geometry section varies exactly those.
std::string scalar_dims(RecurrentStateCache& c) {
    return "dims cd=" + std::to_string(c.conv_dim()) +
           " ck=" + std::to_string(c.conv_kernel()) +
           " vh=" + std::to_string(c.v_heads()) +
           " kd=" + std::to_string(c.head_k_dim()) +
           " vd=" + std::to_string(c.head_v_dim()) +
           " hs=" + std::to_string(c.hidden_size()) +
           " nl=" + std::to_string(c.num_layers()) +
           " ms=" + std::to_string(c.max_slots()) +
           " bf16=" + std::to_string((int)c.recurrent_state_bf16()) +
           " css=" + std::to_string(c.conv_slot_stride_bytes()) +
           " rsf=" + std::to_string(c.recurrent_slot_stride_floats()) +
           " rsb=" + std::to_string(c.recurrent_slot_stride_bytes()) +
           " vst=" + std::to_string(c.verify_stash_max_tokens()) +
           " vsh=" + std::to_string(c.verify_stash_hidden());
}

std::string accessors(RecurrentStateCache& c) {
    std::vector<std::string> rows;
    const int nl = c.num_layers();
    for (int layer = -1; layer <= nl; ++layer) {
        for (int slot = -1; slot <= c.max_slots(); ++slot) {
            std::string row = "L" + std::to_string(layer) + "/S" +
                              std::to_string(slot) + " ";
            try {
                row += "conv=" + oracle_cuda::where(c.conv_state(layer, slot));
            } catch (const std::exception& e) {
                row += std::string("conv!") + e.what();
            }
            try {
                row += " rec=" +
                       oracle_cuda::where(c.recurrent_state_raw(layer, slot));
            } catch (const std::exception& e) {
                row += std::string(" rec!") + e.what();
            }
            try {
                row += " recf=" +
                       oracle_cuda::where(c.recurrent_state(layer, slot));
            } catch (const std::exception& e) {
                row += std::string(" recf!") + e.what();
            }
            if (layer == 0) {
                try {
                    row += " mtp=" +
                           oracle_cuda::where(c.mtp_pending_hidden(slot));
                } catch (const std::exception& e) {
                    row += std::string(" mtp!") + e.what();
                }
            }
            rows.push_back(row);
        }
    }
    return join(rows);
}

std::string shape_report(RecurrentStateCache& c) {
    return "layers=" + std::to_string(c.num_layers()) +
           " slots=" + std::to_string(c.max_slots()) +
           " convdim=" + std::to_string(c.conv_dim()) +
           " convk=" + std::to_string(c.conv_kernel()) +
           " vh=" + std::to_string(c.v_heads()) +
           " kd=" + std::to_string(c.head_k_dim()) +
           " vd=" + std::to_string(c.head_v_dim()) +
           " hidden=" + std::to_string(c.hidden_size()) +
           " bf16=" + std::to_string((int)c.recurrent_state_bf16()) +
           " convstride=" + std::to_string(c.conv_slot_stride_bytes()) +
           " recfloats=" + std::to_string(c.recurrent_slot_stride_floats()) +
           " recstride=" + std::to_string(c.recurrent_slot_stride_bytes()) +
           " frozen=" + std::to_string((int)c.verify_frozen());
}

// One full case: construct, then exercise every operation in a fixed order.
void run_case(const std::string& id,
              const std::vector<bool>& linear,
              int conv_dim, int conv_kernel,
              int v_heads, int head_k_dim, int head_v_dim,
              int hidden_size, int max_slots,
              bool force_bf16 = false)
{
    oracle_cuda::reset_case();
    RecurrentStateCache c =
        force_bf16
            ? RecurrentStateCache::allocate_bf16_recurrent(
                  linear, conv_dim, conv_kernel, v_heads, head_k_dim,
                  head_v_dim, max_slots)
            : RecurrentStateCache::allocate(
                  linear, conv_dim, conv_kernel, v_heads, head_k_dim,
                  head_v_dim, hidden_size, max_slots);
    const std::string ctor = drain();

    std::vector<std::string> fields;
    fields.push_back(shape_report(c));
    fields.push_back(ctor);

    c.reset();
    fields.push_back(drain());

    // Slot ids around every boundary, including the invalid ones.
    for (int slot : {-1, 0, 1, c.max_slots() - 1, c.max_slots()}) {
        std::string f = "slot" + std::to_string(slot) + " ";
        try {
            c.reset_slot(slot);
            f += drain();
        } catch (const std::exception& e) {
            oracle_cuda::reset_log();
            f += std::string("!") + e.what();
        }
        fields.push_back(f);
    }

    for (auto pair : std::vector<std::pair<int, int>>{
             {0, 0}, {0, 1}, {1, 0}, {-1, 0}, {0, -1},
             {0, c.max_slots()}, {c.max_slots() - 1, 0}}) {
        std::string f = "cp" + std::to_string(pair.first) + "->" +
                        std::to_string(pair.second) + " ";
        try {
            c.copy_slot_d2d(pair.first, pair.second);
            f += drain();
        } catch (const std::exception& e) {
            oracle_cuda::reset_log();
            f += std::string("!") + e.what();
        }
        std::string g = " lin ";
        try {
            c.copy_linear_state_slot_d2d(pair.first, pair.second);
            g += drain();
        } catch (const std::exception& e) {
            oracle_cuda::reset_log();
            g += std::string("!") + e.what();
        }
        fields.push_back(f + g);
    }

    // The device-predicated reset. Null arrays and a zero count must issue
    // nothing at all, which is a different transcript from "issue with count
    // zero" -- the C++ returns before touching the kernel.
    const std::int32_t ids[4] = {0, 1, -1, 2};
    const std::uint8_t fresh[4] = {1, 0, 1, 1};
    for (int n : {0, 1, 4, -1}) {
        c.reset_slots_if_fresh(ids, fresh, n);
        fields.push_back("fresh" + std::to_string(n) + " " + drain());
    }
    c.reset_slots_if_fresh(nullptr, fresh, 4);
    fields.push_back("freshnull " + drain());
    c.reset_slots_if_fresh(ids, nullptr, 4);
    fields.push_back("freshnullf " + drain());

    fields.push_back(accessors(c));

    c.set_verify_frozen(true);
    fields.push_back("frozen=" + std::to_string((int)c.verify_frozen()));
    c.set_verify_frozen(false);

    fields.push_back(scalar_dims(c));

    emit(id, join(fields));
}

// The two optional tiers, swept separately: they are configured after
// construction and each has its own set of reject-and-return-silently rules.
void run_tiers(const std::string& id,
               const std::vector<bool>& linear,
               int max_slots,
               int stash_tokens, int stash_hidden,
               int pool_tokens, int pool_hidden, int pool_slots)
{
    oracle_cuda::reset_case();
    RecurrentStateCache c = RecurrentStateCache::allocate(
        linear, 64, 4, 2, 8, 16, 32, max_slots);
    oracle_cuda::reset_log();

    std::vector<std::string> fields;

    c.configure_verify_hidden_stash(stash_tokens, stash_hidden);
    fields.push_back("stash " + drain());
    fields.push_back(
        "on=" + std::to_string((int)c.verify_hidden_stash_enabled()) +
        " tok=" + std::to_string(c.verify_stash_max_tokens()) +
        " hid=" + std::to_string(c.verify_stash_hidden()));
    {
        std::vector<std::string> rows;
        for (int i = -1; i <= c.num_layers(); ++i) {
            rows.push_back("s" + std::to_string(i) + "=" +
                           oracle_cuda::where(c.verify_hidden_stash_layer(i)));
        }
        fields.push_back(join(rows));
    }

    c.configure_rs_buffer_pool(pool_tokens, pool_hidden, pool_slots);
    fields.push_back("pool " + drain());
    fields.push_back("on=" + std::to_string((int)c.rs_buffer_pool_enabled()) +
                     " tok=" + std::to_string(c.rs_buffer_page_tokens()) +
                     " hid=" + std::to_string(c.rs_buffer_hidden()) +
                     " slots=" + std::to_string(c.rs_buffer_num_slots()));
    {
        std::vector<std::string> rows;
        for (int i = -1; i <= c.num_layers(); ++i) {
            for (int s = -1; s <= pool_slots; ++s) {
                rows.push_back("p" + std::to_string(i) + "/" +
                               std::to_string(s) + "=" +
                               oracle_cuda::where(c.rs_buffer_slab(i, s)));
            }
        }
        fields.push_back(join(rows));
    }

    fields.push_back(scalar_dims(c));

    emit(id, join(fields));
}

// The PIE_RS_STASH_TOKENS cap, swept separately.
//
// The C++ reads it with getenv on EVERY call (not through a function-static),
// so the value can be changed between cases in one process. It is parsed with
// std::atoi, which returns 0 for anything unreadable -- and 0 is exactly the
// value the guard ignores, so a typo silently leaves the stash at its full
// prefill width.
void run_stash_cap(const std::string& id, const char* value, int max_tokens) {
    oracle_cuda::reset_case();
    if (value == nullptr) {
        unsetenv("PIE_RS_STASH_TOKENS");
    } else {
        setenv("PIE_RS_STASH_TOKENS", value, 1);
    }
    RecurrentStateCache c = RecurrentStateCache::allocate(
        {true, false, true}, 64, 4, 2, 8, 16, 32, 2);
    oracle_cuda::reset_log();
    c.configure_verify_hidden_stash(max_tokens, 5);
    emit(id, drain() + kUnit + "tok=" +
                 std::to_string(c.verify_stash_max_tokens()) + " hid=" +
                 std::to_string(c.verify_stash_hidden()) + " on=" +
                 std::to_string((int)c.verify_hidden_stash_enabled()));
    unsetenv("PIE_RS_STASH_TOKENS");
}

}  // namespace

int main() {
    std::printf("bf16default|%d\n",
                (int)RecurrentStateCache::recurrent_state_bf16_default());

    // ---- 1. hybrid layer patterns ------------------------------------------
    //
    // The dense compaction is the single most dangerous piece of arithmetic in
    // this file: layer 7 of a stack whose linear layers are 0, 3, 7 lives at
    // index 2. Every interleaving shape gets swept, including the two
    // degenerate ones (all-linear, none-linear).
    const std::vector<std::vector<bool>> kPatterns = {
        {},
        {false},
        {true},
        {true, true, true, true},
        {false, false, false},
        {true, false, true, false, true, false},
        {false, true, false, true, false, true},
        {false, false, true, true, false, false, true},
        {true, false, false, false, false, false, false, true},
    };
    for (const auto& p : kPatterns) {
        for (int slots : {0, 1, 4}) {
            run_case("pat/" + layers_label(p) + "/" + std::to_string(slots),
                     p, 128, 4, 2, 8, 16, 64, slots);
        }
    }

    // ---- 2. geometry -------------------------------------------------------
    //
    // conv_dim/conv_kernel and head_k_dim/head_v_dim are each a pair of ints
    // that transpose silently. Asymmetric values throughout so a swap moves
    // the strides.
    const std::vector<bool> mixed = {true, false, true, false, true};
    for (int conv_dim : {0, 1, 96, 4096}) {
        for (int conv_kernel : {0, 1, 4}) {
            run_case("geo/c" + std::to_string(conv_dim) + "x" +
                         std::to_string(conv_kernel),
                     mixed, conv_dim, conv_kernel, 3, 8, 16, 32, 3);
        }
    }
    for (int v_heads : {0, 1, 5}) {
        for (int kd : {0, 8, 128}) {
            for (int vd : {0, 16, 64}) {
                run_case("geo/v" + std::to_string(v_heads) + "/" +
                             std::to_string(kd) + "/" + std::to_string(vd),
                         mixed, 96, 4, v_heads, kd, vd, 32, 2);
            }
        }
    }

    // ---- 3. the MTP tier ---------------------------------------------------
    //
    // hidden_size 0 means "no tier"; a negative one is clamped to 0 rather
    // than rejected, which is the only place a negative argument is accepted.
    for (int hidden : {-8, 0, 1, 2048}) {
        for (int slots : {1, 3}) {
            run_case("mtp/" + std::to_string(hidden) + "/" +
                         std::to_string(slots),
                     mixed, 96, 4, 2, 8, 16, hidden, slots);
        }
    }

    // ---- 4. negative and degenerate slot counts ----------------------------
    for (int slots : {-4, -1, 0, 1, 2}) {
        run_case("slots/" + std::to_string(slots), mixed, 96, 4, 2, 8, 16, 32,
                 slots);
    }

    // ---- 5. the forced-bf16 constructor ------------------------------------
    //
    // `allocate_bf16_recurrent` ignores hidden_size (it passes 0) and forces
    // bf16 storage. With the env switch gone the default is already bf16, so
    // its re-allocation branch is dead -- the transcript shows whether a
    // second slab is ever allocated.
    for (const auto& p : kPatterns) {
        run_case("bf16/" + layers_label(p), p, 96, 4, 2, 8, 16, 2048, 3, true);
    }

    // ---- 6. the optional tiers ---------------------------------------------
    for (const auto& p : {std::vector<bool>{}, std::vector<bool>{false, false},
                          mixed}) {
        for (int st : {0, 1, 7}) {
            for (int sh : {0, 5}) {
                run_tiers("tier/" + layers_label(p) + "/" +
                              std::to_string(st) + "x" + std::to_string(sh),
                          p, 2, st, sh, 3, 4, 2);
            }
        }
        for (int pt : {0, 3}) {
            for (int ph : {0, 4}) {
                for (int ps : {0, 1, 5}) {
                    run_tiers("tier/" + layers_label(p) + "/pool" +
                                  std::to_string(pt) + "x" +
                                  std::to_string(ph) + "x" +
                                  std::to_string(ps),
                              p, 2, 6, 8, pt, ph, ps);
                }
            }
        }
    }

    // ---- 7. the stash token cap -------------------------------------------
    const char* kCaps[] = {nullptr, "", "0", "-1", "1", "7", "256", "8192",
                           "99999", "abc", "12x", " 7", "+9", "0007",
                           "2147483648"};
    for (const char* cap : kCaps) {
        for (int mt : {0, 8, 8192}) {
            run_stash_cap(std::string("cap/") +
                              (cap == nullptr ? "<unset>"
                                              : (cap[0] == 0 ? "<empty>" : cap)) +
                              "/" + std::to_string(mt),
                          cap, mt);
        }
    }

    return 0;
}
