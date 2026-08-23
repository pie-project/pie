use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub trait Compiles: Copy + 'static {
    type Headers: Copy + 'static;

    type Toolchain: Copy + core::fmt::Display + 'static;

    type Entry: Send + Sync + 'static;

    type Error: Clone + Send + Sync + 'static;

    fn arch() -> Option<&'static str>;

    fn headers_key(headers: Self::Headers) -> (&'static str, u64);

    fn admits(_root: &Root<Self>, _point: &str) -> Result<(), Self::Error> {
        Ok(())
    }

    fn compile(
        root: &Root<Self>,
        point: &str,
        arch: &str,
    ) -> Result<(Vec<u8>, String), Self::Error>;

    fn load(root: &Root<Self>, image: &[u8], mangled: &str) -> Result<Self::Entry, Self::Error>;
}

pub struct Root<B: Compiles> {
    pub name: &'static str,

    pub text: &'static str,

    pub file: &'static str,

    pub options: &'static [&'static str],

    pub headers: B::Headers,

    pub floor: B::Toolchain,
}

impl<B: Compiles> Root<B> {
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

type Slot<B> = OnceLock<Result<Entry<B>, <B as Compiles>::Error>>;

pub struct Entry<B: Compiles> {
    pub entry: B::Entry,

    pub mangled: String,
}

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

fn load<B: Compiles>(
    root: &Root<B>,
    point: &str,
    key: &str,
    arch: &str,
) -> Result<Entry<B>, B::Error> {
    let started = std::time::Instant::now();

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

fn slot<B: Compiles>(key: &str) -> &'static Slot<B> {
    type Erased = &'static (dyn Any + Send + Sync);

    type Slots = OnceLock<Mutex<HashMap<(TypeId, String), Erased>>>;
    static SLOTS: Slots = OnceLock::new();
    let mut map = SLOTS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let erased = *map
        .entry((TypeId::of::<B>(), key.to_owned()))
        .or_insert_with(|| {
            let fresh: &'static Slot<B> = Box::leak(Box::new(OnceLock::new()));
            fresh
        });
    erased
        .downcast_ref::<Slot<B>>()
        .expect("the `TypeId` in the key is this backend's")
}

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

fn read_disk(key: &str) -> Option<(Vec<u8>, String)> {
    let bytes = std::fs::read(disk_path(key)?).ok()?;
    let (stored, rest) = take_str(&bytes)?;
    if stored != key {
        return None;
    }
    let (mangled, image) = take_str(rest)?;
    Some((image.to_vec(), mangled.to_owned()))
}

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

    let staging = path.with_extension(format!("image.{}", std::process::id()));
    if std::fs::write(&staging, &out).is_ok() && std::fs::rename(&staging, &path).is_err() {
        let _ = std::fs::remove_file(&staging);
    }
}

fn take_str(bytes: &[u8]) -> Option<(&str, &[u8])> {
    let (len, rest) = bytes.split_at_checked(4)?;
    let len = u32::from_le_bytes(len.try_into().ok()?) as usize;
    let (text, tail) = rest.split_at_checked(len)?;
    Some((core::str::from_utf8(text).ok()?, tail))
}

fn put_str(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&u32::try_from(text.len()).unwrap_or(u32::MAX).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

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
