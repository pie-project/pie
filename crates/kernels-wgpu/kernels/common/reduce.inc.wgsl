const PIE_REDUCE_LANES = 256u;

var<workgroup> pie_partials: array<f32, PIE_REDUCE_LANES>;

fn pie_workgroup_sum(lane: u32, lanes: u32, v: f32) -> f32 {
//#if defined(PIE_SUBGROUP)

    let s = subgroupAdd(v);
    let width = max(1u, u32(subgroupAdd(1.0)));
    pie_partials[lane] = 0.0;
    workgroupBarrier();

    if (lane == subgroupBroadcastFirst(lane)) {
        pie_partials[lane] = s;
    }
    workgroupBarrier();
    var total = 0.0;
    for (var i = 0u; i < lanes; i = i + width) {
        total = total + pie_partials[i];
    }
    workgroupBarrier();
    return total;
//#else
    pie_partials[lane] = v;
    workgroupBarrier();

    var stride = lanes >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride && lane + stride < lanes) {
            pie_partials[lane] = pie_partials[lane] + pie_partials[lane + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let total = pie_partials[0];
    workgroupBarrier();
    return total;
//#endif
}

fn pie_workgroup_max(lane: u32, lanes: u32, v: f32) -> f32 {
//#if defined(PIE_SUBGROUP)
    let s = subgroupMax(v);
    let width = max(1u, u32(subgroupAdd(1.0)));

    pie_partials[lane] = -3.4028235e38;
    workgroupBarrier();
    if (lane == subgroupBroadcastFirst(lane)) {
        pie_partials[lane] = s;
    }
    workgroupBarrier();
    var total = -3.4028235e38;
    for (var i = 0u; i < lanes; i = i + width) {
        total = max(total, pie_partials[i]);
    }
    workgroupBarrier();
    return total;
//#else
    pie_partials[lane] = v;
    workgroupBarrier();
    var stride = lanes >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride && lane + stride < lanes) {
            pie_partials[lane] = max(pie_partials[lane], pie_partials[lane + stride]);
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }
    let total = pie_partials[0];
    workgroupBarrier();
    return total;
//#endif
}

fn pie_inv_rms(lane: u32, lanes: u32, sum_sq: f32, axis_size: u32, eps: f32) -> f32 {
    let total = pie_workgroup_sum(lane, lanes, sum_sq);
    return inverseSqrt(total / f32(axis_size) + eps);
}
