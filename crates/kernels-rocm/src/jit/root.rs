use crate::source::{self, LIBRARY};

/// The options every unit compiles under, after `--offload-arch`. No
/// contraction of a*b+c into an fma: a kernel rounds as its reference does,
/// and a fused path says so in its own text with `fmaf`.
/// A unit is also a header other units include, so it carries `#pragma
/// once`, which clang flags in the file it compiles first.
pub const OPTIONS: [&str; 4] = [
    "-std=c++17",
    "-O3",
    "-ffp-contract=off",
    "-Wno-pragma-once-outside-header",
];

/// A unit hiprtc compiles: one `.hip` file under `kernels/`, with every
/// other carried file in scope as a header.
#[derive(Clone, Copy, Debug)]
pub struct Root {
    pub name: &'static str,
    pub text: &'static str,
    pub file: &'static str,
}

impl Root {
    #[must_use]
    pub fn of(file: &'static str) -> Option<Self> {
        let text = source::text_of(file)?;
        let name = file.strip_suffix(".hip")?;
        Some(Self { name, text, file })
    }

    /// The cache key of one instantiation of this unit for `arch`: it moves
    /// when the unit, any carried header, or the instantiation does.
    #[must_use]
    pub fn key(&self, instantiation: &str, arch: &str) -> String {
        format!(
            "{}/i{:016x}",
            self.key_prefix(arch),
            source::fnv1a64(instantiation.as_bytes()),
        )
    }

    fn key_prefix(&self, arch: &str) -> &'static str {
        use std::collections::HashMap;
        use std::sync::{Mutex, OnceLock};

        static PREFIXES: OnceLock<Mutex<HashMap<(usize, String), &'static str>>> = OnceLock::new();
        let identity = (self.text.as_ptr() as usize, arch.to_owned());
        let mut cache = PREFIXES
            .get_or_init(|| Mutex::new(HashMap::new()))
            .lock()
            .expect("lock poisoned");
        if let Some(found) = cache.get(&identity) {
            return found;
        }
        let prefix: &'static str = Box::leak(
            format!(
                "jit/{}/{arch}/{}/r{:016x}/h{:016x}",
                self.name,
                OPTIONS.join(","),
                source::fnv1a64(self.text.as_bytes()),
                source::digest(LIBRARY),
            )
            .into_boxed_str(),
        );
        cache.insert(identity, prefix);
        prefix
    }
}
