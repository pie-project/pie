fn pie_subgroup_sum32(v: f32) -> f32 {
    var x = v;
    x = x + subgroupShuffleXor(x, 16u);
    x = x + subgroupShuffleXor(x, 8u);
    x = x + subgroupShuffleXor(x, 4u);
    x = x + subgroupShuffleXor(x, 2u);
    x = x + subgroupShuffleXor(x, 1u);
    return x;
}

fn pie_subgroup_sum16(v: f32) -> f32 {
    var x = v;
    x = x + subgroupShuffleXor(x, 8u);
    x = x + subgroupShuffleXor(x, 4u);
    x = x + subgroupShuffleXor(x, 2u);
    x = x + subgroupShuffleXor(x, 1u);
    return x;
}
