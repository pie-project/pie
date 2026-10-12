use kernels_metal::DeviceInfo;
use kernels_metal::tuning::DeviceTuning;

fn tuned(apple_family: u32, metal4: bool) -> DeviceTuning {
    DeviceTuning::of(DeviceInfo {
        apple_family,
        gpu_core_count: 0,
        metal4,
    })
}

#[test]
fn an_apple10_gpu_reads_a_prompt_through_mpp() {
    let apple10 = tuned(10, true);
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
    assert_eq!(
        (apple10.gdn_scan_lanes, apple10.gdn_scan_rows),
        (8, 2),
        "and its recurrent scan takes the staged shape, which is twice as fast there"
    );
    let apple8 = tuned(8, true);
    assert!(
        apple8.sdpa_mpp && !apple8.qmm_mpp,
        "an Apple8 GPU attends through MPP and keeps the tiled matmul, which is faster there"
    );
    for unmeasured in [0, 7, 9] {
        let unmeasured = tuned(unmeasured, true);
        assert!(
            !unmeasured.qmm_mpp && !unmeasured.sdpa_mpp,
            "a family nobody measured keeps the kernels it had"
        );
    }
    for family in [8, 10] {
        let before_metal4 = tuned(family, false);
        assert!(
            !before_metal4.qmm_mpp && !before_metal4.sdpa_mpp,
            "family {family} without Metal 4 cannot build the MPP kernels and keeps the old ones"
        );
    }
}
