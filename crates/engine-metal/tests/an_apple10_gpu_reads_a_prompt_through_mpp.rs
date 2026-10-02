use kernels_metal::DeviceInfo;
use kernels_metal::tuning::DeviceTuning;

fn tuned(apple_family: u32) -> DeviceTuning {
    DeviceTuning::of(DeviceInfo {
        apple_family,
        gpu_core_count: 0,
    })
}

#[test]
fn an_apple10_gpu_reads_a_prompt_through_mpp() {
    let apple10 = tuned(10);
    assert!(
        apple10.qmm_mpp,
        "an Apple10 GPU multiplies prompt rows through the MPP matmul"
    );
    assert!(
        !apple10.qmm_mpp_packed,
        "and over the bank as it lies, so the weight tier is not held twice"
    );
    assert!(
        apple10.sdpa_mpp,
        "and its prompt rows attend through the MPP matmul too"
    );
    for older in [0, 7, 8, 9] {
        let older = tuned(older);
        assert!(
            !older.qmm_mpp && !older.sdpa_mpp,
            "an older family keeps the kernels it was measured on"
        );
    }
}
