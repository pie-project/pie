//! A mounted adapter directory is read once a plane file, and resolved into
//! one plane a bank, sliced by layer and padded by orientation; what it
//! cannot resolve is refused by name. Where the planes land is the bind's
//! (the pool's `Adapter` slots), never this reader's.

use std::path::{Path, PathBuf};

use engine_cuda::BankSeat;
use engine_cuda::blob::{Adapters, Blobs, Layout};

const LAYERS: u64 = 3;
const BANK_RANK: u64 = 8;
const HIDDEN: u64 = 16;
const ELEM: u64 = 2;

fn scratch(what: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or(0);
    let at = std::env::temp_dir().join(format!("pie-blob-{what}-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&at).expect("a scratch directory");
    at
}

fn seats() -> Vec<BankSeat> {
    let slot = BANK_RANK * HIDDEN * ELEM;
    (0..LAYERS)
        .flat_map(|layer| {
            [
                BankSeat {
                    name: format!("layer.{layer}.lora_a"),
                    slot,
                    rows: BANK_RANK,
                    cols: HIDDEN,
                    elem: ELEM,
                },
                BankSeat {
                    name: format!("layer.{layer}.lora_b"),
                    slot,
                    rows: HIDDEN,
                    cols: BANK_RANK,
                    elem: ELEM,
                },
            ]
        })
        .collect()
}

fn write_adapter(mount: &Path, name: &str, rank: u64, layouts: (Layout, Layout)) {
    let dir = mount.join(name);
    std::fs::create_dir_all(&dir).expect("an adapter directory");
    let spell = |layout: Layout| match layout {
        Layout::RankMajor => "rank_major",
        Layout::OutMajor => "out_major",
    };
    std::fs::write(
        dir.join("adapter.toml"),
        format!(
            "rank = {rank}\n\n\
             [[plane]]\nrole = \"lora_a\"\nfile = \"a.bin\"\nlayout = \"{}\"\n\n\
             [[plane]]\nrole = \"lora_b\"\nfile = \"b.bin\"\nlayout = \"{}\"\n",
            spell(layouts.0),
            spell(layouts.1)
        ),
    )
    .expect("a manifest");
    let elements = (LAYERS * rank * HIDDEN) as usize;
    let ramp: Vec<u8> = (0..elements)
        .flat_map(|at| ((at as u16) | 0x0100).to_le_bytes())
        .collect();
    std::fs::write(dir.join("a.bin"), &ramp).expect("an A plane");
    std::fs::write(dir.join("b.bin"), &ramp).expect("a B plane");
}

fn mounted(what: &str) -> (PathBuf, Adapters) {
    let mount = scratch(what);
    write_adapter(&mount, "alice-v2", 4, (Layout::RankMajor, Layout::OutMajor));
    let mut adapters = Adapters::default();
    adapters.mount(Some(mount.clone()));
    (mount, adapters)
}

#[test]
fn a_shared_adapter_is_read_once_and_resolved_by_bank_every_case() {
    eight_threads_asking_for_one_blob_read_it_once();
    the_resolver_slices_per_layer_and_pads_per_orientation();
    the_refusals_fire_by_name();
}

fn eight_threads_asking_for_one_blob_read_it_once() {
    let at = scratch("flight").join("plane.bin");
    std::fs::write(&at, vec![7u8; 1 << 16]).expect("a plane");
    let blobs = Blobs::default();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let blobs = &blobs;
                let at = &at;
                scope.spawn(move || blobs.open(at, "plane").expect("the read"))
            })
            .collect();
        let held: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("a thread"))
            .collect();
        assert_eq!(held.len(), 8);
        for blob in &held {
            assert_eq!(blob.bytes.len(), 1 << 16);
            assert_eq!(blob.fingerprint, held[0].fingerprint);
        }
    });

    assert_eq!(blobs.loads(), 1, "one read, seven waiters");

    let again = blobs.open(&at, "plane").expect("a second generation");
    assert_eq!(blobs.loads(), 2);
    assert_eq!(again.bytes.len(), 1 << 16);
}

fn the_resolver_slices_per_layer_and_pads_per_orientation() {
    let (_mount, adapters) = mounted("slice");
    let seats = seats();
    let (built, fingerprint) = adapters
        .planes("alice-v2", &seats)
        .expect("the resolver reads a well-formed adapter");

    assert_eq!(
        built.len(),
        (2 * LAYERS) as usize,
        "one plane per bank, and the banks are per layer"
    );
    assert_ne!(fingerprint, 0, "the identity's content half is recorded");
    for (name, plane) in &built {
        assert_eq!(
            plane.len() as u64,
            BANK_RANK * HIDDEN * ELEM,
            "`{name}` is one whole plane, which is what an `Adapter` slot holds"
        );
    }

    let source = |element: usize| ((element as u16) | 0x0100).to_le_bytes();
    let rank = 4usize;
    let hidden = HIDDEN as usize;
    let bank_rank = BANK_RANK as usize;

    let a = &built
        .iter()
        .find(|(name, _)| name == "layer.1.lora_a")
        .expect("layer 1's A")
        .1;
    for row in 0..bank_rank {
        for col in 0..hidden {
            let at = (row * hidden + col) * 2;
            let want = match row < rank {
                true => source(hidden * rank + row * hidden + col),
                false => [0, 0],
            };
            assert_eq!(&a[at..at + 2], &want, "A row {row} col {col} of layer 1");
        }
    }

    let b = &built
        .iter()
        .find(|(name, _)| name == "layer.1.lora_b")
        .expect("layer 1's B")
        .1;
    for row in 0..hidden {
        for col in 0..bank_rank {
            let at = (row * bank_rank + col) * 2;
            let want = match col < rank {
                true => source(hidden * rank + row * rank + col),
                false => [0, 0],
            };
            assert_eq!(&b[at..at + 2], &want, "B row {row} col {col} of layer 1");
        }
    }
}

fn the_refusals_fire_by_name() {
    let seats = seats();

    let bare = Adapters::default();
    let said = bare
        .planes("alice", &seats)
        .expect_err("nothing is mounted")
        .to_string();
    assert!(
        said.contains("no shared adapter directory mounted"),
        "{said}"
    );

    let mount = scratch("refusals");
    write_adapter(&mount, "alice", 4, (Layout::RankMajor, Layout::OutMajor));
    let mut adapters = Adapters::default();
    adapters.mount(Some(mount.clone()));

    let said = adapters
        .planes("nobody", &seats)
        .expect_err("no such adapter")
        .to_string();
    assert!(said.contains("is not a directory in the mount"), "{said}");

    let said = adapters
        .planes("../elsewhere", &seats)
        .expect_err("a traversal")
        .to_string();
    assert!(said.contains("leaves the mount"), "{said}");

    std::fs::create_dir_all(mount.join("mute")).expect("a directory");
    let said = adapters
        .planes("mute", &seats)
        .expect_err("no manifest")
        .to_string();
    assert!(said.contains("adapter.toml"), "{said}");

    write_adapter(&mount, "short", 4, (Layout::RankMajor, Layout::OutMajor));
    std::fs::write(mount.join("short").join("a.bin"), vec![0u8; 8]).expect("a truncated plane");
    let said = adapters
        .planes("short", &seats)
        .expect_err("a plane that is not the banks' size")
        .to_string();
    assert!(said.contains("carries 8 bytes"), "{said}");
    assert!(said.contains("want 384"), "{said}");

    write_adapter(&mount, "flipped", 4, (Layout::RankMajor, Layout::RankMajor));
    let said = adapters
        .planes("flipped", &seats)
        .expect_err("a rank-major B")
        .to_string();
    assert!(said.contains("rank-major [rank, hidden]"), "{said}");
    assert!(said.contains("out-major [hidden, rank]"), "{said}");
    assert!(said.contains("refused rather than repacked"), "{said}");

    write_adapter(&mount, "wide", 32, (Layout::RankMajor, Layout::OutMajor));
    let said = adapters
        .planes("wide", &seats)
        .expect_err("a rank the bank cannot seat")
        .to_string();
    assert!(said.contains("is rank 32"), "{said}");
    assert!(said.contains("seats rank 8"), "{said}");

    std::fs::create_dir_all(mount.join("stray")).expect("a directory");
    std::fs::write(
        mount.join("stray").join("adapter.toml"),
        "rank = 4\n\n[[plane]]\nrole = \"ia3_l\"\nfile = \"l.bin\"\nlayout = \"rank_major\"\n",
    )
    .expect("a manifest");
    std::fs::write(mount.join("stray").join("l.bin"), vec![0u8; 4]).expect("a plane");
    let said = adapters
        .planes("stray", &seats)
        .expect_err("a role with no bank")
        .to_string();
    assert!(said.contains("this load declares no bank"), "{said}");
    std::fs::remove_dir_all(&mount).expect("the mount goes");
}
