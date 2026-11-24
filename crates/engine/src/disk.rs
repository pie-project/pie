//! Slot files below host memory, one per rank and kind. On a host they sit
//! on its filesystem, unlinked once open so they never outlive the process,
//! or kept, for a persistent tier. In a browser they sit in the origin's
//! private file system, which can do neither.

use std::io;
#[cfg(unix)]
use std::os::unix::fs::FileExt;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod opfs;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{kept_bytes, room};
#[cfg(target_arch = "wasm32")]
pub use opfs::room;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(target_arch = "wasm32")]
use web_std::opfs::File;

/// Slot offsets and lengths are multiples of this, as direct I/O asks.
pub const ALIGN: usize = 4096;

#[derive(Debug)]
pub struct SlotFile {
    file: File,
    stride: usize,
    slots: u32,
    kept: bool,
}

/// The stride and the slots `budget` holds of `slot_bytes` each; `None` when
/// that is none.
fn geometry(slot_bytes: u64, budget: u64) -> io::Result<Option<(usize, u32)>> {
    let stride = usize::try_from(slot_bytes)
        .map_err(io::Error::other)?
        .next_multiple_of(ALIGN);
    let slots = u32::try_from(budget.checked_div(stride as u64).unwrap_or(0)).unwrap_or(u32::MAX);
    Ok((slot_bytes != 0 && slots != 0).then_some((stride, slots)))
}

impl SlotFile {
    #[must_use]
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// Whether the slots hold what the last run left: a kept file whose
    /// meta matched.
    #[must_use]
    pub fn kept(&self) -> bool {
        self.kept
    }

    /// Makes every slot written so far durable.
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// The bytes one slot takes in the file and in a page's stage.
    #[must_use]
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Writes one slot's `stage`, an `ALIGN`-aligned buffer of `stride` bytes.
    pub fn write(&self, slot: u32, stage: &[u8]) -> io::Result<()> {
        self.file.write_all_at(stage, self.offset(slot, stage)?)
    }

    /// Reads slot `slot` into `stage`.
    pub fn read(&self, slot: u32, stage: &mut [u8]) -> io::Result<()> {
        self.file.read_exact_at(stage, self.offset(slot, stage)?)
    }

    /// Writes `stage`, an `ALIGN`-aligned buffer of a multiple of `ALIGN`
    /// bytes, at byte `at` (a multiple of `ALIGN`) within the slots: a span
    /// of a slot, for an engine that lays several of its own in one.
    pub fn write_span(&self, at: u64, stage: &[u8]) -> io::Result<()> {
        self.file.write_all_at(stage, self.span(at, stage)?)
    }

    /// Reads the span at byte `at` into `stage`, as [`Self::write_span`].
    pub fn read_span(&self, at: u64, stage: &mut [u8]) -> io::Result<()> {
        self.file.read_exact_at(stage, self.span(at, stage)?)
    }

    fn span(&self, at: u64, stage: &[u8]) -> io::Result<u64> {
        let len = u64::from(self.slots) * self.stride as u64;
        let align = ALIGN as u64;
        if at % align != 0
            || stage.len() % ALIGN != 0
            || stage.as_ptr().align_offset(ALIGN) != 0
            || at.checked_add(stage.len() as u64).is_none_or(|end| end > len)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} bytes at byte {at} is not an aligned span of the file's {} x {}",
                    stage.len(),
                    self.slots,
                    self.stride
                ),
            ));
        }
        Ok(at)
    }

    fn offset(&self, slot: u32, stage: &[u8]) -> io::Result<u64> {
        if slot >= self.slots
            || stage.len() != self.stride
            || stage.as_ptr().align_offset(ALIGN) != 0
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} bytes at slot {slot} is not one aligned slot of the file's {} x {}",
                    stage.len(),
                    self.slots,
                    self.stride
                ),
            ));
        }
        Ok(u64::from(slot) * self.stride as u64)
    }
}

/// The room the disk under a tier's directory gives it, as the platform
/// states it: [`room`] reads it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Room {
    /// The bytes free on the filesystem.
    pub free: u64,
    /// The bytes of `free` the platform keeps back from this process.
    pub reserve: u64,
    /// What `reserve` is, named by the rule that states it.
    pub reserve_is: String,
}

impl Room {
    /// The bytes an unstated tier may take: the free space less the
    /// platform's reserve.
    #[must_use]
    pub fn for_us(&self) -> u64 {
        self.free.saturating_sub(self.reserve)
    }

    /// No figure: the filesystem states no free space, so an unstated tier
    /// comes to nothing.
    #[must_use]
    pub fn unstated(why: &str) -> Self {
        Self {
            free: 0,
            reserve: 0,
            reserve_is: format!("nothing ({why})"),
        }
    }

    /// A filesystem's statvfs figures in bytes (`f_bfree`, `f_bavail`,
    /// each times `f_frsize`): what lies between them is its reserve for
    /// privileged processes, which this one is not given.
    #[must_use]
    pub fn statvfs(free: u64, available: u64) -> Self {
        Self {
            free,
            reserve: free.saturating_sub(available),
            reserve_is: "the filesystem's reserve for privileged processes (statvfs f_bfree \
                         less f_bavail)"
                .to_string(),
        }
    }

    /// Android's: the filesystem's figures as [`Self::statvfs`], less the
    /// platform's low-storage threshold on a filesystem of `total` bytes
    /// (StorageManager's getStorageLowBytes), which apps are told to stop
    /// writing at.
    #[must_use]
    pub fn android(total: u64, free: u64, available: u64, low: &LowStorage) -> Self {
        let root = free.saturating_sub(available);
        let threshold = low.bytes(total);
        Self {
            free,
            reserve: root.saturating_add(threshold),
            reserve_is: format!(
                "{root} bytes of the filesystem's reserve for privileged processes (statvfs \
                 f_bfree less f_bavail) and {threshold} bytes of {}: min({}% of the {total} \
                 bytes, {} bytes)",
                low.source, low.percent, low.max_bytes
            ),
        }
    }

    /// Windows': GetDiskFreeSpaceExW's free bytes and its
    /// FreeBytesAvailableToCaller; what lies between is withheld from this
    /// user (a quota).
    #[must_use]
    pub fn windows(free: u64, available: u64) -> Self {
        Self {
            free,
            reserve: free.saturating_sub(available),
            reserve_is: "what the volume withholds from this user (GetDiskFreeSpaceExW's free \
                         bytes less FreeBytesAvailableToCaller)"
                .to_string(),
        }
    }
}

impl std::fmt::Display for Room {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} bytes free, less {} bytes kept as {}: {} bytes for the tier",
            self.free,
            self.reserve,
            self.reserve_is,
            self.for_us()
        )
    }
}

/// Android's low-storage threshold: min(total x `percent` / 100,
/// `max_bytes`), from the global settings `sys_storage_threshold_percentage`
/// and `sys_storage_threshold_max_bytes`, or AOSP's defaults for them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LowStorage {
    pub percent: u64,
    pub max_bytes: u64,
    /// Where the figures came from, for the boot line.
    pub source: &'static str,
}

impl LowStorage {
    /// AOSP's `DEFAULT_THRESHOLD_PERCENTAGE`.
    pub const DEFAULT_PERCENT: u64 = 5;
    /// AOSP's `DEFAULT_THRESHOLD_MAX_BYTES`: 500 MiB.
    pub const DEFAULT_MAX_BYTES: u64 = 500 << 20;

    /// The threshold on a filesystem of `total` bytes, as
    /// getStorageLowBytes computes it.
    #[must_use]
    pub fn bytes(&self, total: u64) -> u64 {
        (total.saturating_mul(self.percent) / 100).min(self.max_bytes)
    }

    /// From the text of `settings_global.xml` (`<setting ... name="..."
    /// value="..." />`) when it is given and readable as text; a setting it
    /// lacks takes AOSP's default, as the platform does. Binary (ABX) or
    /// absent settings: the defaults, named as such.
    #[must_use]
    pub fn from_settings(xml: Option<&str>) -> Self {
        let value = |name: &str| -> Option<u64> {
            let xml = xml?;
            let at = xml.find(&format!("name=\"{name}\""))?;
            let rest = &xml[at..xml[at..].find('>').map_or(xml.len(), |end| at + end)];
            let value = &rest[rest.find("value=\"")? + 7..];
            value[..value.find('"')?].trim().parse().ok()
        };
        let percent = value("sys_storage_threshold_percentage");
        let max_bytes = value("sys_storage_threshold_max_bytes");
        Self {
            percent: percent.unwrap_or(Self::DEFAULT_PERCENT),
            max_bytes: max_bytes.unwrap_or(Self::DEFAULT_MAX_BYTES),
            source: if percent.is_some() || max_bytes.is_some() {
                "Android's low-storage threshold (global settings)"
            } else {
                "Android's default low-storage threshold"
            },
        }
    }
}

/// What can be parked, each on the slot file's direct I/O boundary: the
/// declared concurrency's state (`seats` sequences, `state_bytes` each),
/// kv (`seat_pages` pages of `page_bytes`: the seats at the declared
/// context) and the guest state the device holds live.
#[must_use]
pub fn derived_need(
    seats: u64,
    state_bytes: u64,
    seat_pages: u64,
    page_bytes: u64,
    guest_bytes: u64,
) -> u64 {
    let slot = |bytes: u64| bytes.next_multiple_of(ALIGN as u64);
    seats
        .saturating_mul(slot(state_bytes))
        .saturating_add(seat_pages.saturating_mul(slot(page_bytes)))
        .saturating_add(slot(guest_bytes))
}

/// The disk tier's bytes derived on their own: [`derived_need`] within the
/// `free` bytes the tier may take ([`Room::for_us`]); past them it is
/// seated to them, sparse. Never a share of the host tier (a host budget
/// below one chunk seated no disk tier at all), a fraction of the free
/// space, nor the device pool (a hidden multiple of the device budget:
/// 62 GB at 2 GiB and 125 GB at 5 GiB on one model and mix).
#[must_use]
pub fn derived_budget(
    seats: u64,
    state_bytes: u64,
    seat_pages: u64,
    page_bytes: u64,
    guest_bytes: u64,
    free: u64,
) -> u64 {
    derived_need(seats, state_bytes, seat_pages, page_bytes, guest_bytes).min(free)
}

#[cfg(test)]
mod budget_tests {
    use super::*;

    /// Derived from what can be parked alone: the seats' state and their
    /// kv at the declared context; the free space bounds it whole, never a
    /// quarter of it, and nothing of the device pool enters it.
    #[test]
    fn the_unstated_disk_tier_is_derived_from_the_load_within_the_free_space() {
        let page = ALIGN as u64;
        assert_eq!(derived_budget(2, page, 10, page, 0, u64::MAX), 12 * page);
        assert_eq!(derived_budget(0, 0, 40, page, 0, u64::MAX), 40 * page);
        assert_eq!(
            derived_budget(0, 0, 40, page, 3 * page, u64::MAX),
            43 * page,
            "the guest state the device holds live has a home too"
        );
        assert_eq!(
            derived_budget(0, 0, 40, page, 0, 20 * page),
            20 * page,
            "the free space whole, not a quarter of it"
        );
    }

    /// Unix: the room is f_bavail; what lies between f_bfree and it is the
    /// filesystem's reserve, kept and named.
    #[test]
    fn the_unix_room_is_what_unprivileged_processes_may_take() {
        let room = Room::statvfs(100 << 30, 95 << 30);
        assert_eq!(
            (room.free, room.reserve, room.for_us()),
            (100 << 30, 5 << 30, 95 << 30)
        );
        assert!(room.to_string().contains("f_bavail"), "{room}");
        let odd = Room::statvfs(10, 20);
        assert_eq!((odd.reserve, odd.for_us()), (0, 10), "never more than free");
        assert_eq!(Room::unstated("no statvfs").for_us(), 0);
    }

    /// Windows: FreeBytesAvailableToCaller is the room.
    #[test]
    fn the_windows_room_is_what_the_caller_may_take() {
        let room = Room::windows(50 << 30, 20 << 30);
        assert_eq!((room.reserve, room.for_us()), (30 << 30, 20 << 30));
    }

    /// Android: min(total x 5%, 500 MiB) by default, below f_bavail; the
    /// phone that gave the tier all 16.5 GB of its free /data now keeps
    /// the platform's threshold back.
    #[test]
    fn android_keeps_its_low_storage_threshold() {
        let defaults = LowStorage::from_settings(None);
        assert_eq!(
            (defaults.percent, defaults.max_bytes, defaults.source),
            (5, 500 << 20, "Android's default low-storage threshold")
        );
        // 256 GB /data: 5% is 12.8 GB, past the 500 MiB cap.
        assert_eq!(defaults.bytes(256_000_000_000), 500 << 20);
        // 4 GB: 5% is 200 MB, under the cap.
        assert_eq!(defaults.bytes(4_000_000_000), 200_000_000);
        let free = 16_500_000_000;
        let room = Room::android(256_000_000_000, free, free - (64 << 20), &defaults);
        assert_eq!(room.reserve, (64 << 20) + (500 << 20));
        assert_eq!(room.for_us(), free - (64 << 20) - (500 << 20));
        assert!(
            room.to_string()
                .contains("Android's default low-storage threshold"),
            "{room}"
        );

        let xml = r#"<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<settings version="-1">
  <setting id="31" name="sys_storage_threshold_percentage" value="10" package="android" defaultValue="10" defaultSysSet="true" />
  <setting id="32" name="sys_storage_threshold_max_bytes" value="1073741824" package="android" />
  <setting id="33" name="adb_enabled" value="1" package="android" />
</settings>"#;
        let stated = LowStorage::from_settings(Some(xml));
        assert_eq!((stated.percent, stated.max_bytes), (10, 1 << 30));
        assert_eq!(
            stated.source,
            "Android's low-storage threshold (global settings)"
        );
        assert_eq!(stated.bytes(8_000_000_000), 800_000_000);
        assert_eq!(stated.bytes(256_000_000_000), 1 << 30);

        let one = LowStorage::from_settings(Some(
            r#"<setting id="1" name="sys_storage_threshold_percentage" value="2" />"#,
        ));
        assert_eq!(
            (one.percent, one.max_bytes),
            (2, 500 << 20),
            "a missing setting is AOSP's default"
        );
        let binary = LowStorage::from_settings(Some("ABX\0\u{1}garbage"));
        assert_eq!(binary, defaults);
    }
}

/// Windows reads and writes at an offset through `seek_read`/`seek_write`,
/// which may stop short: the whole slot is looped through, as Unix's
/// `read_exact_at`/`write_all_at` do.
#[cfg(windows)]
trait FileExt {
    fn write_all_at(&self, buf: &[u8], at: u64) -> io::Result<()>;
    fn read_exact_at(&self, buf: &mut [u8], at: u64) -> io::Result<()>;
}

#[cfg(windows)]
impl FileExt for File {
    fn write_all_at(&self, mut buf: &[u8], mut at: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_write(buf, at)? {
                0 => return Err(io::ErrorKind::WriteZero.into()),
                n => (buf, at) = (&buf[n..], at + n as u64),
            }
        }
        Ok(())
    }

    fn read_exact_at(&self, mut buf: &mut [u8], mut at: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt as _;
        while !buf.is_empty() {
            match self.seek_read(buf, at)? {
                0 => return Err(io::ErrorKind::UnexpectedEof.into()),
                n => (buf, at) = (&mut buf[n..], at + n as u64),
            }
        }
        Ok(())
    }
}

/// A heap buffer of `len` bytes that starts on an `ALIGN` boundary: the
/// stage one page goes through on an engine whose kv lives in host memory.
#[derive(Debug, Default)]
pub struct Aligned {
    bytes: Vec<u8>,
    at: usize,
    len: usize,
}

impl Aligned {
    #[must_use]
    pub fn zeroed(len: usize) -> Self {
        let bytes = vec![0u8; len + ALIGN];
        let at = bytes.as_ptr().align_offset(ALIGN);
        Self { bytes, at, len }
    }

    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes[self.at..self.at + self.len]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.bytes[self.at..self.at + self.len]
    }
}
