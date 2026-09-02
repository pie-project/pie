//! Unit-carrying value types: [`Duration`] and [`ByteSize`].
//!
//!
//! A duration or a size is written with its unit -- `"50ms"`, `"4GiB"` --
//! rather than carried in the field name. A unit in the name and a number in
//! the value can disagree silently, cannot be read without knowing the schema,
//! and drifts: `_us` beside `_secs` beside `_s`, and `_mb` beside `_gb`.

/// A duration written with its unit: `"50ms"`, `"120s"`, `"2m"`.
///
/// Accepted units: `ns`, `us`, `ms`, `s`, `m`, `h`. A bare number is refused
/// rather than assumed to be seconds -- assuming is how `_us` and `_secs` came
/// to live in one table.
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct Duration(std::time::Duration);

impl Duration {
    pub const fn from_millis(ms: u64) -> Self {
        Self(std::time::Duration::from_millis(ms))
    }
    pub const fn from_secs(s: u64) -> Self {
        Self(std::time::Duration::from_secs(s))
    }
    pub fn as_micros(&self) -> u64 {
        self.0.as_micros().min(u64::MAX as u128) as u64
    }
    pub fn as_secs(&self) -> u64 {
        self.0.as_secs()
    }
    pub fn as_secs_f64(&self) -> f64 {
        self.0.as_secs_f64()
    }
}

fn parse_duration(text: &str) -> std::result::Result<Duration, String> {
    let t = text.trim();
    let split = t
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .ok_or_else(|| format!("duration {t:?} has no unit; write one of 120s, 50ms, 2m"))?;
    let (value, unit) = t.split_at(split);
    let value: f64 = value
        .parse()
        .map_err(|_| format!("duration {t:?} has an unparseable number"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("duration {t:?} must be finite and non-negative"));
    }
    let nanos = match unit.trim() {
        "ns" => value,
        "us" | "\u{b5}s" => value * 1e3,
        "ms" => value * 1e6,
        "s" => value * 1e9,
        "m" => value * 6e10,
        "h" => value * 3.6e12,
        other => {
            return Err(format!(
                "duration {t:?} has unknown unit {other:?}; use ns, us, ms, s, m, h"
            ));
        }
    };
    Ok(Duration(std::time::Duration::from_nanos(nanos as u64)))
}

impl<'de> Deserialize<'de> for Duration {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        parse_duration(&raw).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Duration {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let us = self.as_micros();
        if us.is_multiple_of(1_000_000) {
            s.serialize_str(&format!("{}s", us / 1_000_000))
        } else if us.is_multiple_of(1_000) {
            s.serialize_str(&format!("{}ms", us / 1_000))
        } else {
            s.serialize_str(&format!("{us}us"))
        }
    }
}

/// A byte size written with its unit: `"256MiB"`, `"4GiB"`.
///
/// Binary units only (`B`, `KiB`, `MiB`, `GiB`, `TiB`), because that is what
/// the `_mb`/`_gb` fields this replaces always meant -- each multiplied by
/// 1024*1024, never 1000*1000.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByteSize(u64);

impl ByteSize {
    pub const fn from_mib(mib: u64) -> Self {
        Self(mib * 1024 * 1024)
    }
    pub const fn as_bytes(&self) -> u64 {
        self.0
    }
    pub const fn as_mib(&self) -> u64 {
        self.0 / (1024 * 1024)
    }
    pub fn as_gib_f64(&self) -> f64 {
        self.0 as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

fn parse_byte_size(text: &str) -> std::result::Result<ByteSize, String> {
    let t = text.trim();
    let split = t
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .ok_or_else(|| format!("size {t:?} has no unit; write one of 512B, 256MiB, 4GiB"))?;
    let (value, unit) = t.split_at(split);
    let value: f64 = value
        .parse()
        .map_err(|_| format!("size {t:?} has an unparseable number"))?;
    if !value.is_finite() || value < 0.0 {
        return Err(format!("size {t:?} must be finite and non-negative"));
    }
    let scale: f64 = match unit.trim() {
        "B" => 1.0,
        "KiB" => 1024.0,
        "MiB" => 1024.0 * 1024.0,
        "GiB" => 1024.0 * 1024.0 * 1024.0,
        "TiB" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => {
            return Err(format!(
                "size {t:?} has unknown unit {other:?}; use B, KiB, MiB, GiB, TiB \
                 (binary units only)"
            ));
        }
    };
    Ok(ByteSize((value * scale) as u64))
}

impl<'de> Deserialize<'de> for ByteSize {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        parse_byte_size(&raw).map_err(serde::de::Error::custom)
    }
}

impl Serialize for ByteSize {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        const GIB: u64 = 1024 * 1024 * 1024;
        const MIB: u64 = 1024 * 1024;
        const KIB: u64 = 1024;
        let b = self.0;
        let text = if b.is_multiple_of(GIB) && b != 0 {
            format!("{}GiB", b / GIB)
        } else if b.is_multiple_of(MIB) && b != 0 {
            format!("{}MiB", b / MIB)
        } else if b.is_multiple_of(KIB) && b != 0 {
            format!("{}KiB", b / KIB)
        } else {
            format!("{b}B")
        };
        s.serialize_str(&text)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bare_number_is_refused_rather_than_assumed() {
        // Assuming a unit is how `_us` and `_secs` came to live in one table.
        let err = parse_duration("120").unwrap_err();
        assert!(err.contains("has no unit"), "got: {err}");
        let err = parse_byte_size("256").unwrap_err();
        assert!(err.contains("has no unit"), "got: {err}");
    }

    #[test]
    fn decimal_units_are_refused_for_sizes() {
        // The `_mb`/`_gb` fields this replaces always meant MiB/GiB -- each
        // multiplied by 1024*1024, never 1000*1000. Accepting "MB" would let a
        // config mean 5% less than it says.
        let err = parse_byte_size("256MB").unwrap_err();
        assert!(err.contains("binary units only"), "got: {err}");
    }
}
