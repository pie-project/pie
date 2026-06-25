#pragma once
// batch_schedule_view.hpp — thin adapter: PieForwardRequestView → BatchSchedule.
//
// Keeps batch_schedule.hpp schema-free (pure, standalone-unit-testable); THIS header is the
// one place that depends on the schema, so alpha's entry-marshaling includes it and calls
// build_batch_schedule(view, page_size) directly, then reads `.req_of_token` / `.spans` /
// `.is_pure_decode` to populate the IO CSR buffers + ForwardGraphKey (the seam alpha proposed
// in #mac: beta owns CSR interpretation, alpha owns IO-buffer population + decoder feed).

#include "batch_schedule.hpp"
#include "pie_driver_abi/view.hpp"

namespace pie_metal_driver::raw_metal {

// Interpret the marshaled CSR view into the batch plan. page_size = runtime kv_page_size (32).
inline BatchSchedule build_batch_schedule(const pie_driver::PieForwardRequestView& v, int page_size) {
    return build_batch_schedule(
        v.token_ids.data(),         int(v.token_ids.size()),
        v.qo_indptr.data(),
        v.kv_page_indptr.data(),
        v.kv_last_page_lens.data(),
        v.rs_slot_ids.size()   ? v.rs_slot_ids.data()   : nullptr,
        v.rs_slot_flags.size() ? v.rs_slot_flags.data() : nullptr,
        int(v.qo_indptr.size()),
        page_size);
}

}  // namespace pie_metal_driver::raw_metal
