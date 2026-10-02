use kernels_metal::DeviceInfo;
use kernels_metal::tuning::DeviceTuning;

fn tuned(apple_family: u32) -> DeviceTuning {
    DeviceTuning::of(DeviceInfo {
        apple_family,
        gpu_core_count: 0,
    })
}

#[test]
fn an_apple10_gpu_multiplies_prompt_rows_through_mpp() {
    let apple10 = tuned(10);
    assert!(
        apple10.qmm_mpp,
        "an Apple10 GPU reads a prompt through the MPP matmul"
    );
    assert!(
        !apple10.qmm_mpp_packed,
        "and over the bank as it lies, so the weight tier is not held twice"
    );
    for older in [0, 7, 8, 9] {
        assert!(
            !tuned(older).qmm_mpp,
            "family {older} keeps the tiled matmul it was measured on"
        );
    }
}
