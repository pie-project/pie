//! Each queue holds a wave save area the driver allocates, which HIP's
//! free figure does not count. Where KFD's list of this process's queues is
//! readable (the host's PID namespace), HIP's rule for how many it made (one
//! utility queue, a hardware queue a stream up to its limit, a copy queue a
//! copying stream) agrees with it at 1, 4 and 15 streams: the rule is what
//! every load counts by. Skips with no HIP device, and where the
//! list is another namespace's.
#![cfg(feature = "rocm")]

use kernels_rocm::hip::{self, kfd};

#[test]
fn the_driver_holds_a_save_area_a_queue() {
    let Ok(count) = hip::device_count() else {
        eprintln!("no HIP runtime; skipping");
        return;
    };
    if count == 0 {
        eprintln!("no HIP device; skipping");
        return;
    }
    hip::set_device(0).unwrap();
    let mut streams = Vec::new();
    for want in [1u32, 4, 15] {
        while streams.len() < want as usize {
            streams.push(hip::Stream::new().unwrap());
        }
        let areas = kfd::save_areas(want).unwrap();
        let Some(listed) = kfd::listed_queues() else {
            eprintln!("KFD's queue list is another PID namespace's; skipping");
            return;
        };
        assert_eq!(
            listed, areas.queues,
            "at {want} streams KFD lists {listed} queues and HIP's rule counts {}",
            areas.queues
        );
        assert!(areas.area() > 0, "a queue holds a save area: {areas}");
        // KFD counts every queue's save area as this process's.
        let held = kfd::process_vram().expect("KFD's list for this process reads");
        assert!(
            held >= areas.bytes(),
            "KFD counts {held} bytes for this process, under its queues' {} of save areas",
            areas.bytes()
        );
        eprintln!("{want} streams: {areas}");
    }
}
