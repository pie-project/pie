#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "batch_schedule.hpp"
#include "batch/compose.hpp"
#include "batch/forward.hpp"

using namespace pie::metal;

namespace {
int pass = 0, fail = 0;
void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    ok ? ++pass : ++fail;
}
}  // namespace

int main() {
    std::printf("[paged batch validation]\n");
    const uint32_t tokens[] = {10, 11};
    const uint32_t qo[] = {0, 1, 2};
    // Request 0 uses pages {7,2}; request 1 shares page 7.  Physical CSR
    // order is intentionally non-monotone and shared.
    const uint32_t pi[] = {0, 2, 3};
    const uint32_t pages[] = {7, 2, 7};
    const uint32_t last[] = {32, 1};
    const uint32_t slots[] = {0, 1};
    const uint8_t flags[] = {0, 1};
    BatchSchedule s = build_batch_schedule(tokens, 2, qo, pi, last, slots, flags, 3, 32);
    const std::vector<uint32_t> pos = {31, 0};
    const std::vector<uint32_t> page_vec(pages, pages + 3);
    std::vector<uint32_t> wp = {7, 7}, wo = {31, 0};
    std::string err;
    expect(validate_paged_batch(s, pos, page_vec, wp, wo, 8, 2, &err),
           "ordered physical CSR/shared prefix and matching explicit writes validate (" + err + ")");

    wp[1] = 2;
    expect(!validate_paged_batch(s, pos, page_vec, wp, wo, 8, 2, &err) &&
               err.find("disagrees") != std::string::npos,
           "write page that disagrees with CSR is rejected before dispatch");
    wp[1] = 7;
    wo[0] = 32;
    expect(!validate_paged_batch(s, pos, page_vec, wp, wo, 8, 2, &err) &&
               err.find("exceeds") != std::string::npos,
           "out-of-page write offset is rejected before dispatch");
    wo[0] = 31;

    std::vector<pie::metal::batch::MemberForwardDesc> descs(2);
    descs[0].token_ids = {1, 2, 3};
    descs[0].readout_local_indices = {2};
    descs[1].token_ids = {4};
    descs[1].readout_local_indices = {0};
    std::uint32_t rejected_row = 999;
    std::uint32_t accepted_row = 999;
    int rejected_callbacks = 0;
    int accepted_callbacks = 0;
    std::vector<pie::metal::batch::PtirCommandCallbacks> callbacks(2);
    callbacks[0].set_logits_row = [&](std::uint32_t row) {
        rejected_row = row;
    };
    callbacks[0].post_forward =
        [&](pie::metal::StepEncoder&) { ++rejected_callbacks; };
    callbacks[1].set_logits_row = [&](std::uint32_t row) {
        accepted_row = row;
    };
    callbacks[1].post_forward =
        [&](pie::metal::StepEncoder&) { ++accepted_callbacks; };
    const auto compacted =
        pie::metal::batch::compact_ptir_callbacks(
            descs, {1}, {0}, callbacks);
    expect(
        compacted.size() == 1 && rejected_row == 999 &&
            accepted_row == 0 && rejected_callbacks == 0 &&
            accepted_callbacks == 0,
        "invalid-first/valid-second compacts callbacks and rebases logits to row zero");
    {
        std::vector<pie::metal::batch::MemberForwardDesc> multi(1);
        multi[0].token_ids = {4, 5, 6};
        multi[0].readout_local_indices = {0, 2};
        std::vector<std::uint32_t> attributed_rows;
        std::vector<pie::metal::batch::PtirCommandCallbacks>
            multi_callbacks(1);
        multi_callbacks[0].set_logits_rows =
            [&](const std::vector<std::uint32_t>& rows) {
                attributed_rows = rows;
            };
        const auto compacted_multi =
            pie::metal::batch::compact_ptir_callbacks(
                multi, {0}, {7}, multi_callbacks);
        expect(
            compacted_multi.size() == 1 &&
                attributed_rows ==
                    std::vector<std::uint32_t>({7, 9}),
            "Context callback compaction preserves ragged multi-row "
            "attribution");
    }
    std::vector<uint32_t> bad_pages = {7, 9, 7};
    expect(!validate_paged_batch(s, pos, bad_pages, wp, wo, 8, 2, &err) &&
               err.find("physical page") != std::string::npos,
           "out-of-pool CSR page is rejected before dispatch");

    BatchSchedule over_cap = s;
    over_cap.N = 65;
    expect(!validate_paged_batch_capacity(over_cap, 64, 4, &err) &&
               err.find("capacity") != std::string::npos,
           "over-cap (>64 token) paged prompt rejects before command encoding");

    {
        pie_native::ptir::FireGeometry geometry;
        geometry.token_ids = {1, 2};
        geometry.position_ids = {0, 1};
        geometry.qo_indptr = {0, 2};
        geometry.kv_page_indices = {0};
        geometry.kv_page_indptr = {0, 1};
        geometry.kv_last_page_lens = {2};
        geometry.sampling_indices = {1};
        geometry.sampling_indptr = {0, 1};
        geometry.mask = {
            1, 0, 0, 0,
            1, 1, 0, 0,
        };
        geometry.has_mask = true;
        geometry.structured_mask = {
            .kind =
                pie_native::ptir::StructuredMaskKind::SlidingWindow,
            .key_len = 4,
            .window = 2,
        };
        pie::metal::batch::MemberForwardDesc desc;
        pie_native::LaunchView empty;
        err.clear();
        expect(
            pie::metal::batch::build_member_forward_desc(
                empty, 0, 1, false, 32, &geometry, desc, err) &&
                desc.has_attention_mask &&
                desc.attention_mask_stride == 4 &&
                desc.attention_mask == geometry.mask &&
                desc.structured_mask.kind ==
                    pie_native::ptir::StructuredMaskKind::SlidingWindow,
            "resolved dense/structured fallback mask survives member "
            "composition exactly (" +
                err + ")");
        geometry.mask.clear();
        geometry.has_mask = false;
        err.clear();
        expect(
            !pie::metal::batch::build_member_forward_desc(
                empty, 0, 1, false, 32, &geometry, desc, err) &&
                err.find("no dense fallback") != std::string::npos,
            "structured masks without a dense fallback reject explicitly");
    }
    {
        pie_native::ptir::FireGeometry geometry;
        geometry.token_ids = {3, 5};
        geometry.position_ids = {7, 7};
        geometry.qo_indptr = {0, 1, 2};
        geometry.kv_page_indices = {5, 6, 0, 5, 6, 0};
        geometry.kv_page_indptr = {0, 3, 6};
        geometry.kv_last_page_lens = {3, 3};
        geometry.sampling_indices = {0, 0};
        geometry.sampling_indptr = {0, 1, 2};
        geometry.w_page = {6, 6};
        geometry.w_off = {2, 2};
        geometry.has_write_desc = true;
        geometry.mask = std::vector<std::uint8_t>(24, 1);
        geometry.has_mask = true;
        pie::metal::batch::MemberForwardDesc desc;
        const std::uint32_t folded_slots[] = {0, 1};
        const std::uint8_t reset_flags[] = {
            PIE_RS_FLAG_RESET,
            PIE_RS_FLAG_RESET,
        };
        const std::uint32_t buffered_activation_slots[] = {3, 2};
        const std::uint32_t buffered_activation_indptr[] = {0, 2};
        PieLaunchDesc launch{};
        launch.rs_slot_ids = {
            .ptr = folded_slots,
            .len = 2,
        };
        launch.rs_slot_flags = {
            .ptr = reset_flags,
            .len = 2,
        };
        launch.rs_buffer_slot_ids = {
            .ptr = buffered_activation_slots,
            .len = 2,
        };
        launch.rs_buffer_slot_indptr = {
            .ptr = buffered_activation_indptr,
            .len = 2,
        };
        auto owned =
            pie::metal::batch::OwnedLaunchView::capture(launch);
        const pie_native::LaunchView context_view = owned.view();
        err.clear();
        const bool composed =
            pie::metal::batch::build_member_forward_desc(
                context_view,
                0,
                1,
                true,
                4,
                &geometry,
                desc,
                err);
        const std::uint32_t missing_folded_slots[] = {0};
        const std::uint8_t missing_folded_flags[] = {
            PIE_RS_FLAG_RESET};
        PieLaunchDesc missing_folded_launch = launch;
        missing_folded_launch.rs_slot_ids = {
            .ptr = missing_folded_slots,
            .len = 1,
        };
        missing_folded_launch.rs_slot_flags = {
            .ptr = missing_folded_flags,
            .len = 1,
        };
        auto missing_folded_owned =
            pie::metal::batch::OwnedLaunchView::capture(
                missing_folded_launch);
        pie::metal::batch::MemberForwardDesc rejected_desc;
        std::string missing_folded_error;
        const bool buffered_never_substitutes =
            !pie::metal::batch::build_member_forward_desc(
                missing_folded_owned.view(),
                0,
                1,
                true,
                4,
                &geometry,
                rejected_desc,
                missing_folded_error) &&
            missing_folded_error.find("exactly one folded") !=
                std::string::npos;
        std::string positions_error;
        const bool request_local_positions =
            composed &&
            pie::metal::batch::validate_request_local_positions(
                desc, &positions_error);
        auto collapsed = desc;
        collapsed.qo_indptr = {0, 2};
        std::unordered_map<
            std::uint32_t,
            pie::metal::batch::LinearSequenceState>
            states;
        bool independent_rs =
            composed &&
            desc.request_rs_slot_ids ==
                std::vector<std::uint32_t>({0, 1}) &&
            desc.request_rs_reset ==
                std::vector<std::uint8_t>({1, 1}) &&
            desc.request_rs_read ==
                std::vector<std::uint8_t>({0, 0}) &&
            desc.request_rs_write ==
                std::vector<std::uint8_t>({1, 1}) &&
            pie::metal::batch::validate_paged_request_state(
                states, desc, 0, &err) &&
            pie::metal::batch::validate_paged_request_state(
                states, desc, 1, &err);
        if (independent_rs) {
            pie::metal::batch::commit_paged_request_state(
                states, desc, 0);
            pie::metal::batch::commit_paged_request_state(
                states, desc, 1);
            independent_rs =
                states.size() == 2 &&
                states[0].resident_pages ==
                    std::vector<std::uint32_t>({5, 6, 0}) &&
                states[1].resident_pages ==
                    std::vector<std::uint32_t>({5, 6, 0}) &&
                states[0].resident_next_position == 8 &&
                states[1].resident_next_position == 8;
        }
        auto continuation = desc;
        continuation.position_ids = {8, 8};
        continuation.request_rs_reset = {0, 0};
        continuation.request_rs_read = {1, 1};
        if (independent_rs) {
            independent_rs =
                pie::metal::batch::validate_paged_request_state(
                    states, continuation, 0, &err) &&
                pie::metal::batch::validate_paged_request_state(
                    states, continuation, 1, &err);
        }
        expect(
            composed && buffered_never_substitutes &&
                request_local_positions && independent_rs &&
                !pie::metal::batch::validate_request_local_positions(
                    collapsed, nullptr) &&
                desc.qo_indptr == geometry.qo_indptr &&
                desc.kv_page_indptr ==
                    geometry.kv_page_indptr &&
                desc.kv_last_page_lens ==
                    geometry.kv_last_page_lens &&
                desc.sampling_indptr ==
                    geometry.sampling_indptr &&
                desc.readout_local_indices ==
                    std::vector<std::uint32_t>({0, 0}) &&
                desc.row_count == 2 &&
                desc.kv_last_page_len == 0,
            "B=2 folded RS slots drive multi-request hybrid state while "
            "nonempty buffered activation slots never substitute for missing "
            "folded IDs (" +
                err + positions_error + ")");
    }

    std::printf("\n==== paged_batch_validation_test: %d passed, %d failed ====\n", pass, fail);
    return fail == 0 ? 0 : 1;
}
