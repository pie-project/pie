//! One point, compiled on demand, cached by everything that changes it.
//!
//! # The shape, and where it came from
//!
//! `kernels-cuda/src/jit/{root,cache}.rs` had this and nothing else did. CUDA
//! reached it first because NVRTC compiles ONE instantiation on demand — the
//! body names a template-id and the compiler lowers it — so nothing had to be
//! enumerated ahead of time and no list of points existed to go stale.
//!
//! The other three backends each arrived at a list instead, and each paid for
//! it differently:
//!
//! * **wgpu** already expands its sources at run time and never needed one;
//!   the 261-row lattice tables beside it were a lookup for a name `format!`
//!   builds.
//! * **Metal** compiles a whole library at run time and picks an entry by
//!   name, so every point had to be stamped into the source ahead of time by
//!   an `instantiate_*` macro and listed again in Rust.
//! * **Vulkan** compiles at BUILD time, so its set has to be known before the
//!   binary exists — which is what put `slangc` in the build graph.
//!
//! None of those is a fact about the shading language. They are three answers
//! to *"when is a point compiled"*, and this module is the fourth being made
//! the only one.
//!
//! # What a backend still owns
//!
//! Two methods. [`Backend::compile`] turns a [`Root`] and a point into an
//! image, and [`Backend::load`] turns an image into something launchable.
//! Everything around them — the key, the disk cache, the once-per-process slot,
//! the two-thread race — is written here once.
//!
//! # Why the key is long
//!
//! Every term is something that changes the image, and dropping any one makes
//! a stale one loadable. `kernels-cuda`'s `Root::key` said this first and the
//! list is unchanged: the point, the options (one root compiles several ways
//! under `-D` alone), the architecture, the source text, the header bytes, and
//! the compiler floor.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

/// What a backend needs to be able to say about compiling one point.
///
/// Separate from [`crate::routine::Backend`], which is about BINDING: that one
/// answers *"what does an argument look like on this plane"* and this one
/// answers *"how does a point become something that runs"*. A backend that
/// only reads the signature table implements the first and not the second,
/// which is what lets `model-ir` depend on a kernels crate without owning a
/// shader toolchain.
pub trait Compiles: Copy + 'static {
    /// The header set a root's `#include`s resolve against.
    ///
    /// Opaque here. CUDA's is a three-way choice over carried `.cuh` text,
    /// Metal's is the same over `.metal`, and wgpu's is the source tree its
    /// own expander reaches. What this module needs of it is only that it can
    /// be digested into the key.
    type Headers: Copy + 'static;

    /// The oldest compiler that may build a root.
    ///
    /// In the key because a raised floor means the cached image came from a
    /// compiler this root now refuses.
    type Toolchain: Copy + core::fmt::Display + 'static;

    /// What a launch is issued against: a `CUfunction`, a pipeline state, a
    /// compute pipeline.
    type Entry: Send + Sync + 'static;

    /// This backend's compiler failure, as a value.
    type Error: Clone + Send + Sync + 'static;

    /// The architecture images are built for, or `None` where there is no
    /// device to ask.
    ///
    /// In the key: an image is per-architecture on every backend that has the
    /// concept, and the backends that do not answer a constant.
    fn arch() -> Option<&'static str>;

    /// A digest of `headers`, and a short tag naming which set it is.
    ///
    /// Two values rather than one because the tag reaches a human reading a
    /// cache path and the digest reaches nobody.
    fn headers_key(headers: Self::Headers) -> (&'static str, u64);

    /// Whether this backend will stand behind `point` at all, asked BEFORE the
    /// cache.
    ///
    /// Default: yes. A backend whose compiler refuses an unknown point — CUDA,
    /// where NVRTC fails to lower a template-id that names nothing — has
    /// nothing to add here.
    ///
    /// # Why this is not just a check inside `compile`
    ///
    /// [`load`] reads the disk cache first and calls [`Compiles::compile`]
    /// only on a miss, which is the whole point of having a cache. So a check
    /// that lives in `compile` runs exactly once per point per machine, and
    /// every run after the first takes the cached image without it. If the
    /// check is the only thing standing between a misspelled point and a
    /// plausible-looking image, that image outlives the check.
    ///
    /// Vulkan is where this bites: `-DPIE_ENTRYPOINT=<name>` is passed and NO
    /// SHADER READS IT, so a name nothing declares compiles to a perfectly
    /// valid module. `kernels-vulkan` answers this against the census its
    /// build writes from the tree's `// pie:instantiate` lines.
    ///
    /// # Errors
    ///
    /// The refusal to hand back, when the point is not one this backend has.
    fn admits(_root: &Root<Self>, _point: &str) -> Result<(), Self::Error> {
        Ok(())
    }

    /// Compile `point` out of `root`, returning the image and the name the
    /// image knows it by.
    ///
    /// The second half is not decorative: NVRTC lowers a template-id to a
    /// mangled symbol, and an image looked up by the SOURCE spelling would
    /// miss. A backend whose names survive compilation returns `point`.
    ///
    /// # Errors
    ///
    /// Whatever the backend's compiler refused.
    fn compile(
        root: &Root<Self>,
        point: &str,
        arch: &str,
    ) -> Result<(Vec<u8>, String), Self::Error>;

    /// Load `image` and resolve `mangled` in it.
    ///
    /// # Errors
    ///
    /// Whatever the backend's loader refused.
    fn load(root: &Root<Self>, image: &[u8], mangled: &str) -> Result<Self::Entry, Self::Error>;
}

/// One translation unit: the source a point is compiled out of, and everything
/// that changes what compiling it produces.
///
/// A `&'static` bundle rather than an owned one because every field is either
/// carried in the binary or a compile-time constant, and a root is named by a
/// body — `Fire::at`'s first argument resolves to one.
pub struct Root<B: Compiles> {
    /// The root's name, which reaches a diagnostic and the cache path.
    pub name: &'static str,
    /// The source, handed to the backend's compiler.
    pub text: &'static str,
    /// The path a diagnostic names, relative to the kernel tree.
    pub file: &'static str,
    /// Compiler options this root needs and the others must not have.
    ///
    /// Part of the key, and not decoratively: on CUDA one XQA root compiles
    /// five ways by `-D` alone, and on the shader planes this is where a
    /// lattice point's `-DPIE_GROUP=64 -DPIE_BITS=4` lives — which is what
    /// lets a point be COMPOSED rather than looked up in a table.
    pub options: &'static [&'static str],
    /// Which carried header set its includes resolve against.
    pub headers: B::Headers,
    /// The oldest compiler that may build it.
    pub floor: B::Toolchain,
}

impl<B: Compiles> Root<B> {
    /// The cache key for one point of this root.
    ///
    /// See the module doc for why each term is here. `kernels-cuda`'s
    /// `Root::key` is the original and this is it, with the two backend-shaped
    /// terms asked of the backend.
    #[must_use]
    pub fn key(&self, point: &str, arch: &str) -> String {
        let (tag, headers) = B::headers_key(self.headers);
        format!(
            "jit/{}/{arch}/{}/floor>={}/{tag}/r{:016x}/h{headers:016x}/i{:016x}",
            self.name,
            self.options.join(","),
            self.floor,
            fnv1a64(self.text.as_bytes()),
            fnv1a64(point.as_bytes()),
        )
    }
}

/// One point's slot: compiled at most once, in this process.
type Slot<B> = OnceLock<Result<Entry<B>, <B as Compiles>::Error>>;

/// A compiled point, held for the life of the process.
///
/// Images are never unloaded. CUDA states the reason and it generalises: a
/// driver reuses handle addresses after an unload, so anything memoised by
/// handle would answer for a different kernel. Eviction wants those memos
/// re-keyed first.
pub struct Entry<B: Compiles> {
    /// What a launch is issued against.
    pub entry: B::Entry,
    /// The name the image knew the point by, kept for diagnostics.
    pub mangled: String,
}

/// The entry point for one point of `root`, compiling it on first ask.
///
/// # Errors
///
/// [`Compiles::Error`] when there is no device, the compiler refuses the root,
/// or the point is not in the image.
///
/// # Panics
///
/// Never. A poisoned slot map is taken over rather than propagated: the map
/// holds only leaked slot pointers, so a panic mid-insert leaves it usable.
pub fn resolve<B: Compiles>(
    root: &Root<B>,
    point: &str,
    no_device: B::Error,
) -> Result<&'static Entry<B>, B::Error> {
    let Some(arch) = B::arch() else {
        return Err(no_device);
    };
    let key = root.key(point, arch);
    slot::<B>(&key)
        .get_or_init(|| load(root, point, &key, arch))
        .as_ref()
        .map_err(Clone::clone)
}

/// Compile (or read back) one point and load it.
fn load<B: Compiles>(
    root: &Root<B>,
    point: &str,
    key: &str,
    arch: &str,
) -> Result<Entry<B>, B::Error> {
    let started = std::time::Instant::now();
    // Before the cache, not inside `compile`: see `Compiles::admits`. A point
    // this backend does not have must refuse on every run, not only on the run
    // that would have compiled it.
    B::admits(root, point)?;
    let (image, mangled, compiled) = match read_disk(key) {
        Some((image, mangled)) => (image, mangled, false),
        None => {
            let (image, mangled) = B::compile(root, point, arch)?;
            write_disk(key, &image, &mangled);
            (image, mangled, true)
        }
    };
    let entry = B::load(root, &image, &mangled)?;
    // NO `tracing`, BECAUSE THIS CRATE HAS NO DEPENDENCIES AND THAT IS THE
    // DESIGN -- `Cargo.toml` says a row must be writable next to the `.cu` it
    // describes without dragging a graph along. `kernels-cuda`'s cache logged
    // an `info!` here, and what it was FOR is the two numbers below: whether
    // the point was compiled or read back, and how long it took. Behind an
    // env var so the default costs nothing and a bring-up can still see it.
    if std::env::var_os("PIE_TRACE_JIT").is_some() {
        eprintln!(
            "[jit] {} {point} arch={arch} {} {:.1}ms",
            root.name,
            if compiled { "compiled" } else { "cached" },
            started.elapsed().as_secs_f64() * 1e3,
        );
    }
    Ok(Entry { entry, mangled })
}

/// The slot for a key, created empty if this is the first ask.
///
/// The map lock is held only across the lookup, never across a compile: two
/// threads wanting two different points must not serialise on each other, and
/// two wanting the SAME one meet in the slot's `OnceLock` instead.
///
/// # One map, not one per backend, and why that is not a shortcut
///
/// A `static` inside a generic `fn` is ONE item in Rust, shared by every
/// instantiation -- unlike a C++ template's, which is per-specialisation. So
/// the obvious trick does not work and the map has to be type-erased: the key
/// carries the backend's `TypeId` and the value is downcast back on the way
/// out. Two backends cannot collide, because a `TypeId` is in the key and the
/// downcast would fail if it somehow were not.
fn slot<B: Compiles>(key: &str) -> &'static Slot<B> {
    /// One erased slot: a `Slot<B>` for a `B` this map cannot name, because
    /// the map is one item shared by every instantiation and the paragraph
    /// above says why it has to be.
    type Erased = &'static (dyn Any + Send + Sync);
    /// The map itself, spelled once. Its own name is here rather than inline
    /// because a five-deep generic on one line is a type `clippy` refuses to
    /// let a reader parse, and it is not wrong: the shape is a lock around a
    /// map from a backend-and-point pair to an erased slot, which is four
    /// facts, and a name is where four facts belong.
    type Slots = OnceLock<Mutex<HashMap<(TypeId, String), Erased>>>;
    static SLOTS: Slots = OnceLock::new();
    let mut map = SLOTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let erased = *map
        .entry((TypeId::of::<B>(), key.to_owned()))
        .or_insert_with(|| {
            // Leaked on purpose. Images are never unloaded, so a slot is live for
            // the rest of the process either way, and leaking is what lets the
            // borrow escape the lock.
            let fresh: &'static Slot<B> = Box::leak(Box::new(OnceLock::new()));
            fresh
        });
    erased
        .downcast_ref::<Slot<B>>()
        .expect("the `TypeId` in the key is this backend's")
}

// ── The disk cache ────────────────────────────────────────────────────────
//
// `kernels-cuda/src/jit/cache.rs`'s, verbatim in shape: a length-prefixed key,
// a length-prefixed mangled name, then the image. The key is stored INSIDE the
// file as well as being the path, so a hash collision reads as a miss rather
// than as somebody else's kernel.

/// Where an image for `key` would live, or `None` when there is no cache.
fn disk_path(key: &str) -> Option<std::path::PathBuf> {
    let base = std::env::var_os("PIE_JIT_CACHE")
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::var_os("XDG_CACHE_HOME").map(std::path::PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| std::path::PathBuf::from(h).join(".cache")))?;
    Some(
        base.join("pie")
            .join(format!("{:016x}.image", fnv1a64(key.as_bytes()))),
    )
}

/// The image and mangled name stored for `key`, if the stored key matches.
fn read_disk(key: &str) -> Option<(Vec<u8>, String)> {
    let bytes = std::fs::read(disk_path(key)?).ok()?;
    let (stored, rest) = take_str(&bytes)?;
    if stored != key {
        return None;
    }
    let (mangled, image) = take_str(rest)?;
    Some((image.to_vec(), mangled.to_owned()))
}

/// Store `image` under `key`. Every failure is silent: a cache that cannot be
/// written is a slow start, not a broken one.
fn write_disk(key: &str, image: &[u8], mangled: &str) {
    let Some(path) = disk_path(key) else { return };
    let Some(parent) = path.parent() else { return };
    if std::fs::create_dir_all(parent).is_err() {
        return;
    }
    let mut out = Vec::with_capacity(image.len() + mangled.len() + key.len() + 8);
    put_str(&mut out, key);
    put_str(&mut out, mangled);
    out.extend_from_slice(image);
    // Written beside the target and renamed, so a reader never sees a partial
    // file: two processes may compile the same point at once.
    let staging = path.with_extension(format!("image.{}", std::process::id()));
    if std::fs::write(&staging, &out).is_ok() && std::fs::rename(&staging, &path).is_err() {
        let _ = std::fs::remove_file(&staging);
    }
}

/// A length-prefixed string off the front of `bytes`, and what follows it.
fn take_str(bytes: &[u8]) -> Option<(&str, &[u8])> {
    let (len, rest) = bytes.split_at_checked(4)?;
    let len = u32::from_le_bytes(len.try_into().ok()?) as usize;
    let (text, tail) = rest.split_at_checked(len)?;
    Some((core::str::from_utf8(text).ok()?, tail))
}

/// [`take_str`]'s inverse.
fn put_str(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&u32::try_from(text.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

/// FNV-1a over 64 bits, which is what the CUDA cache keyed on.
///
/// Not a cryptographic choice and does not need to be: a collision reads as a
/// cache miss, because the key is stored in the file and compared.
#[must_use]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    const BASIS: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x1000_0000_01b3;
    let mut hash = BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

/// A composed point name, leaked so a body may hand it to `Fire::at`.
///
/// # Why leaking is not a leak
///
/// A point is one member of a finite lattice — `gs × bits × bm × bn` is 54 —
/// and each is interned at most once, so the set stops growing after the first
/// fire of each. `kernels-cuda::jit::symbol` is the original and this is it,
/// moved here because all four backends compose names now.
///
/// It is what replaces the tables. `QMM_T[qmm_point(gs, bits, bm, bn)?]` folded
/// four numbers into an index into 54 rows written by hand; the name IS the
/// four numbers, so there is nothing to fold.
#[must_use]
pub fn symbol(name: &str) -> &'static str {
    static INTERNED: OnceLock<Mutex<HashMap<String, &'static str>>> = OnceLock::new();
    let mut map = INTERNED
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(found) = map.get(name) {
        return found;
    }
    let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
    map.insert(name.to_owned(), leaked);
    leaked
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// How many times the fake backend was asked to compile each point.
    ///
    /// PER POINT, not a single counter. Tests in one binary run in parallel and
    /// a shared count is a race that reads as a flake: the fixture that asks
    /// "was this compiled once" would see another test's compile land between
    /// its two reads. A point name is unique per test and per run, so counting
    /// against it is exact whatever else is running.
    fn compiles(point: &str) -> usize {
        COMPILES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(point)
            .copied()
            .unwrap_or(0)
    }

    static COMPILES: OnceLock<Mutex<HashMap<String, usize>>> = OnceLock::new();

    #[derive(Clone, Copy)]
    struct Fake;

    impl Compiles for Fake {
        type Headers = u64;
        type Toolchain = u32;
        type Entry = String;
        type Error = String;

        fn arch() -> Option<&'static str> {
            Some("test")
        }
        fn headers_key(headers: Self::Headers) -> (&'static str, u64) {
            ("fake", headers)
        }
        fn compile(
            root: &Root<Self>,
            point: &str,
            _arch: &str,
        ) -> Result<(Vec<u8>, String), Self::Error> {
            *COMPILES
                .get_or_init(|| Mutex::new(HashMap::new()))
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .entry(point.to_owned())
                .or_insert(0) += 1;
            if point.contains("nope") {
                return Err(format!("{} has no `{point}`", root.name));
            }
            Ok((point.as_bytes().to_vec(), point.to_owned()))
        }
        fn load(_: &Root<Self>, image: &[u8], mangled: &str) -> Result<Self::Entry, Self::Error> {
            let text = core::str::from_utf8(image).map_err(|e| e.to_string())?;
            Ok(format!("{text}@{mangled}"))
        }
    }

    fn root(text: &'static str, options: &'static [&'static str]) -> Root<Fake> {
        Root {
            name: "t",
            text,
            file: "t.x",
            options,
            headers: 7,
            floor: 1,
        }
    }

    /// A point name no other run has used.
    ///
    /// THE DISK CACHE IS REAL AND THAT IS THE POINT OF DOING IT THIS WAY. It
    /// keys on `HOME`, so a first run writes images a second run reads back --
    /// which is exactly what it is for, and exactly what makes "was this
    /// compiled" untestable if a fixture reuses a name. Setting the cache
    /// directory to somewhere unwritable was the other candidate and it loses
    /// twice: `set_var` races the other tests in this binary, and it would
    /// test a path production never takes.
    fn fresh(what: &str) -> String {
        static N: AtomicUsize = AtomicUsize::new(0);
        format!(
            "{what}_{}_{}",
            std::process::id(),
            N.fetch_add(1, Ordering::SeqCst)
        )
    }

    #[test]
    fn a_point_is_compiled_once_and_answered_from_then_on() {
        let r = root("body", &[]);
        let point = fresh("once");
        let a = resolve(&r, &point, "no device".to_owned()).expect("compiles");
        let b = resolve(&r, &point, "no device".to_owned()).expect("cached");
        assert_eq!(a.entry, format!("{point}@{point}"));
        assert!(std::ptr::eq(a, b), "the second ask is the same slot");
        assert_eq!(compiles(&point), 1);
    }

    /// THE KEY IS WHY A LATTICE CAN BE COMPOSED. Two points of one root differ
    /// only in their `-D`s, and if the key did not carry the options the second
    /// would be served the first's image -- which is the whole failure mode
    /// that made every backend enumerate its points ahead of time.
    #[test]
    fn two_points_of_one_root_that_differ_only_in_options_do_not_share_a_slot() {
        let gs32 = root("body", &["-DPIE_GROUP=32"]);
        let gs64 = root("body", &["-DPIE_GROUP=64"]);
        assert_ne!(gs32.key("p", "test"), gs64.key("p", "test"));

        let point = fresh("opt");
        resolve(&gs32, &point, "no device".to_owned()).expect("compiles");
        resolve(&gs64, &point, "no device".to_owned()).expect("compiles");
        assert_eq!(compiles(&point), 2, "two keys, two compiles");
    }

    /// Every term that changes the image is in the key. Dropping any one makes
    /// a stale image loadable, which is a wrong kernel and not a slow one.
    #[test]
    fn the_key_moves_when_anything_that_changes_the_image_does() {
        let base = root("body", &["-D=1"]);
        let key = base.key("p", "sm_89");
        assert_ne!(key, base.key("q", "sm_89"), "the point");
        assert_ne!(key, base.key("p", "sm_90"), "the architecture");
        assert_ne!(
            key,
            root("other", &["-D=1"]).key("p", "sm_89"),
            "the source"
        );
        assert_ne!(
            key,
            root("body", &["-D=2"]).key("p", "sm_89"),
            "the options"
        );
        let mut headers = root("body", &["-D=1"]);
        headers.headers = 8;
        assert_ne!(key, headers.key("p", "sm_89"), "the header set");
        let mut floor = root("body", &["-D=1"]);
        floor.floor = 2;
        assert_ne!(key, floor.key("p", "sm_89"), "the compiler floor");
    }

    /// A refusal is remembered, so a point that cannot compile is not retried
    /// on every fire of it. `kernels-cuda`'s `said` memoised the LOG line for
    /// the same reason; the slot memoises the answer.
    #[test]
    fn a_refusal_is_answered_from_the_slot_too() {
        let r = root("body", &[]);
        let point = fresh("nope");
        assert!(resolve(&r, &point, "no device".to_owned()).is_err());
        assert!(resolve(&r, &point, "no device".to_owned()).is_err());
        assert_eq!(compiles(&point), 1, "refusals are not retried");
    }

    /// A composed name is interned once, which is what lets a body hand
    /// `Fire::at` a `&'static str` it built out of four numbers.
    #[test]
    fn a_composed_point_name_interns_to_one_address() {
        let a = symbol(&format!("affine_qmm_t_bfloat16_gs_{}_b_{}", 64, 4));
        let b = symbol("affine_qmm_t_bfloat16_gs_64_b_4");
        assert_eq!(a, b);
        assert!(std::ptr::eq(a.as_ptr(), b.as_ptr()), "interned, not copied");
    }
}
