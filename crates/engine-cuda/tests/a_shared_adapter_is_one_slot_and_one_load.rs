use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use engine_cuda::blob::{Adapters, Blobs, Layout, Source};
use engine_cuda::{AdapterPlane, BankSeat};

const LAYERS: u64 = 3;
const BANK_RANK: u64 = 8;
const HIDDEN: u64 = 16;
const ELEM: u64 = 2;

fn scratch(what: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or(0);
    let at = std::env::temp_dir().join(format!(
        "pie-blob-{what}-{}-{nanos}",
        std::process::id()
    ));
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
                    adapters: 4,
                    slot,
                    rows: BANK_RANK,
                    cols: HIDDEN,
                    elem: ELEM,
                },
                BankSeat {
                    name: format!("layer.{layer}.lora_b"),
                    adapters: 4,
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
    let mut adapters = Adapters::new(2);
    adapters.mount(Some(mount.clone()));
    (mount, adapters)
}

#[derive(Default)]
struct Landings {
    calls: AtomicU64,
    planes: AtomicU64,
}

impl Landings {
    fn land(&self) -> impl FnOnce(u32, &[AdapterPlane<'_>]) -> engine_cuda::Result<()> + '_ {
        move |_slot, planes| {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.planes
                .fetch_add(planes.len() as u64, Ordering::Relaxed);
            Ok(())
        }
    }

    fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

#[test]
fn a_shared_adapter_is_one_slot_and_one_load_every_case() {
    two_instances_of_one_blob_share_one_slot_and_one_landing();
    a_byte_seeded_instance_gets_a_slot_of_its_own();
    a_released_slot_keeps_its_bytes_and_pressure_takes_the_oldest();
    every_slot_pinned_is_refused_and_nothing_live_is_evicted();
    a_rewritten_adapter_is_a_new_identity_and_the_old_one_stays();
    eight_threads_asking_for_one_blob_read_it_once();
    the_resolver_slices_per_layer_and_pads_per_orientation();
    the_refusals_fire_by_name();
    a_load_with_no_banks_seats_nothing_and_says_so();
}

fn two_instances_of_one_blob_share_one_slot_and_one_landing() {
    let (_mount, mut adapters) = mounted("shared");
    let seats = seats();
    let landings = Landings::default();

    let first = adapters
        .bind(Source::Shared { name: "alice-v2" }, &seats, landings.land())
        .expect("the first bind lands");
    let second = adapters
        .bind(Source::Shared { name: "alice-v2" }, &seats, landings.land())
        .expect("the second bind joins it");
    let third = adapters
        .bind(
            Source::Shared {
                name: "/alice-v2",
            },
            &seats,
            landings.land(),
        )
        .expect("a leading slash is the same adapter");

    assert_eq!(first.slot, second.slot, "one blob, one slot");
    assert_eq!(first.slot, third.slot, "and the name is the same name");
    assert!(first.landed, "the first bind is the one that pays");
    assert!(!second.landed, "the second joins what is already there");
    assert!(!third.landed);
    assert_eq!(landings.calls(), 1, "one blob, one landing");
    assert_eq!(
        adapters.blobs().loads(),
        2,
        "one read per plane FILE, and nothing re-read"
    );
    assert_eq!(adapters.slots().refs(first.slot), 3, "three live binds");
    assert_eq!(
        adapters.slots().resident().len(),
        1,
        "and one occupied slot in the whole table"
    );
}

fn a_byte_seeded_instance_gets_a_slot_of_its_own() {
    let (_mount, mut adapters) = mounted("own");
    let seats = seats();
    let landings = Landings::default();
    let slot = usize::try_from(seats[0].slot).expect("a slot fits this host");
    let bytes = vec![0u8; slot];
    let planes: Vec<AdapterPlane<'_>> = seats
        .iter()
        .map(|seat| AdapterPlane {
            bank: seat.name.as_str(),
            bytes: &bytes,
        })
        .collect();

    let shared = adapters
        .bind(Source::Shared { name: "alice-v2" }, &seats, landings.land())
        .expect("the file binds");
    let own = adapters
        .bind(
            Source::Own {
                instance: 7,
                planes: &planes,
            },
            &seats,
            landings.land(),
        )
        .expect("and so do the caller's own bytes");

    assert_ne!(shared.slot, own.slot, "a private adapter shares nothing");
    assert!(shared.shared);
    assert!(!own.shared);
    assert!(own.landed, "its bytes are its own and it pays for them");
    assert_eq!(landings.calls(), 2);
    let again = adapters
        .bind(
            Source::Own {
                instance: 7,
                planes: &planes,
            },
            &seats,
            landings.land(),
        )
        .expect("re-binding one instance");
    assert_eq!(again.slot, own.slot);
    assert!(!again.landed);
}

fn a_released_slot_keeps_its_bytes_and_pressure_takes_the_oldest() {
    let mount = scratch("lru");
    for name in ["alice", "bob", "carol"] {
        write_adapter(&mount, name, 4, (Layout::RankMajor, Layout::OutMajor));
    }
    let mut adapters = Adapters::new(2);
    adapters.mount(Some(mount));
    let seats = seats();
    let landings = Landings::default();

    let alice = adapters
        .bind(Source::Shared { name: "alice" }, &seats, landings.land())
        .expect("alice");
    let bob = adapters
        .bind(Source::Shared { name: "bob" }, &seats, landings.land())
        .expect("bob");
    assert_ne!(alice.slot, bob.slot);
    adapters.release(alice);
    adapters.release(bob);

    let alice_again = adapters
        .bind(Source::Shared { name: "alice" }, &seats, landings.land())
        .expect("alice returns");
    assert_eq!(alice_again.slot, alice.slot, "her bytes are still there");
    assert!(!alice_again.landed, "and she does not re-pay the H2D");
    assert_eq!(landings.calls(), 2, "two landings so far, not three");
    adapters.release(alice_again);

    let carol = adapters
        .bind(Source::Shared { name: "carol" }, &seats, landings.land())
        .expect("carol");
    assert_eq!(carol.slot, bob.slot, "the least recently used slot goes");
    assert!(carol.landed);
    assert_eq!(landings.calls(), 3);

    let alice_third = adapters
        .bind(Source::Shared { name: "alice" }, &seats, landings.land())
        .expect("alice is still there");
    assert_eq!(alice_third.slot, alice.slot);
    assert!(!alice_third.landed);
}

fn every_slot_pinned_is_refused_and_nothing_live_is_evicted() {
    let mount = scratch("pinned");
    for name in ["alice", "bob", "carol"] {
        write_adapter(&mount, name, 4, (Layout::RankMajor, Layout::OutMajor));
    }
    let mut adapters = Adapters::new(2);
    adapters.mount(Some(mount));
    let seats = seats();
    let landings = Landings::default();

    let alice = adapters
        .bind(Source::Shared { name: "alice" }, &seats, landings.land())
        .expect("alice");
    let bob = adapters
        .bind(Source::Shared { name: "bob" }, &seats, landings.land())
        .expect("bob");
    let refused = adapters
        .bind(Source::Shared { name: "carol" }, &seats, landings.land())
        .expect_err("a third identity at a two-seat table, both pinned");
    let said = refused.to_string();
    assert!(
        said.contains("pinned by a live bind"),
        "the refusal says what is wrong: {said}"
    );
    assert!(
        said.contains("concurrent residency, not the catalog"),
        "and why capacity is not the fix: {said}"
    );
    assert_eq!(landings.calls(), 2, "the refusal landed nothing");
    assert_eq!(
        adapters.slots().refs(alice.slot),
        1,
        "and evicted nothing that was held"
    );
    assert_eq!(adapters.slots().refs(bob.slot), 1);

    adapters.release(bob);
    let carol = adapters
        .bind(Source::Shared { name: "carol" }, &seats, landings.land())
        .expect("carol, once a seat is free");
    assert_eq!(carol.slot, bob.slot);
}

fn a_rewritten_adapter_is_a_new_identity_and_the_old_one_stays() {
    let mount = scratch("rewrite");
    write_adapter(&mount, "alice", 4, (Layout::RankMajor, Layout::OutMajor));
    let mut adapters = Adapters::new(2);
    adapters.mount(Some(mount.clone()));
    let seats = seats();
    let landings = Landings::default();

    let held = adapters
        .bind(Source::Shared { name: "alice" }, &seats, landings.land())
        .expect("the first version");

    std::thread::sleep(std::time::Duration::from_millis(10));
    write_adapter(&mount, "alice", 8, (Layout::RankMajor, Layout::OutMajor));

    let fresh = adapters
        .bind(Source::Shared { name: "alice" }, &seats, landings.land())
        .expect("the second version");
    assert_ne!(
        fresh.slot, held.slot,
        "no fire in flight observes an adapter changing"
    );
    assert!(fresh.landed);
    assert_eq!(landings.calls(), 2);
    assert_eq!(adapters.slots().refs(held.slot), 1, "the old one is pinned");
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
            "`{name}` is one whole slot, which is what `register_adapter` takes"
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
            assert_eq!(
                &a[at..at + 2],
                &want,
                "A row {row} col {col} of layer 1"
            );
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
            assert_eq!(
                &b[at..at + 2],
                &want,
                "B row {row} col {col} of layer 1"
            );
        }
    }
}

fn the_refusals_fire_by_name() {
    let seats = seats();
    let landings = Landings::default();

    let mut bare = Adapters::new(2);
    let said = bare
        .bind(Source::Shared { name: "alice" }, &seats, landings.land())
        .expect_err("nothing is mounted")
        .to_string();
    assert!(said.contains("no shared adapter directory mounted"), "{said}");

    let mount = scratch("refusals");
    write_adapter(&mount, "alice", 4, (Layout::RankMajor, Layout::OutMajor));
    let mut adapters = Adapters::new(2);
    adapters.mount(Some(mount.clone()));

    let said = adapters
        .bind(Source::Shared { name: "nobody" }, &seats, landings.land())
        .expect_err("no such adapter")
        .to_string();
    assert!(said.contains("is not a directory in the mount"), "{said}");

    let said = adapters
        .bind(
            Source::Shared {
                name: "../elsewhere",
            },
            &seats,
            landings.land(),
        )
        .expect_err("a traversal")
        .to_string();
    assert!(said.contains("leaves the mount"), "{said}");

    std::fs::create_dir_all(mount.join("mute")).expect("a directory");
    let said = adapters
        .bind(Source::Shared { name: "mute" }, &seats, landings.land())
        .expect_err("no manifest")
        .to_string();
    assert!(said.contains("adapter.toml"), "{said}");

    write_adapter(&mount, "short", 4, (Layout::RankMajor, Layout::OutMajor));
    std::fs::write(mount.join("short").join("a.bin"), vec![0u8; 8]).expect("a truncated plane");
    let said = adapters
        .bind(Source::Shared { name: "short" }, &seats, landings.land())
        .expect_err("a plane that is not the banks' size")
        .to_string();
    assert!(said.contains("carries 8 bytes"), "{said}");
    assert!(said.contains("want 384"), "{said}");

    write_adapter(
        &mount,
        "flipped",
        4,
        (Layout::RankMajor, Layout::RankMajor),
    );
    let said = adapters
        .bind(Source::Shared { name: "flipped" }, &seats, landings.land())
        .expect_err("a rank-major B")
        .to_string();
    assert!(said.contains("rank-major [rank, hidden]"), "{said}");
    assert!(said.contains("out-major [hidden, rank]"), "{said}");
    assert!(said.contains("refused rather than repacked"), "{said}");

    write_adapter(&mount, "wide", 32, (Layout::RankMajor, Layout::OutMajor));
    let said = adapters
        .bind(Source::Shared { name: "wide" }, &seats, landings.land())
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
        .bind(Source::Shared { name: "stray" }, &seats, landings.land())
        .expect_err("a role with no bank")
        .to_string();
    assert!(said.contains("this load declares no bank"), "{said}");

    assert_eq!(landings.calls(), 0);
    assert!(
        adapters.slots().resident().is_empty(),
        "a refused landing holds no slot"
    );
}

fn a_load_with_no_banks_seats_nothing_and_says_so() {
    let (_mount, mut adapters) = mounted("bankless");
    let mut bankless = Adapters::new(0);
    bankless.mount(adapters.vfs().root().map(Path::to_path_buf));
    let landings = Landings::default();
    let said = bankless
        .bind(Source::Shared { name: "alice-v2" }, &[], landings.land())
        .expect_err("no bank, no seat")
        .to_string();
    assert!(said.contains("0 adapter slots"), "{said}");
    let seats = seats();
    adapters
        .bind(Source::Shared { name: "alice-v2" }, &seats, landings.land())
        .expect("a load with banks binds");
}
