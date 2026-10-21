//! Sealed pages' bytes between workers over a plain TCP stream: the inline
//! backend every engine can speak. A request names a run of pages by key
//! from token 0 and the page to start from; the answer is the pages' bytes
//! back to back, as the source engine lays out a host slot. Nothing is
//! encoded: the bytes go from one worker's host tier to the socket and from
//! the socket to the other's, in chunks, so the source reads the next while
//! the link carries this one and the asker lands the one before.
//!
//! Request: `PIEB`, u32 version, u32 key count, the keys (32 bytes each),
//! u32 first page. Answer: u8 status; 0 then u32 pages, u64 bytes per page,
//! u32 length and the source's description of its layout (opaque here), the
//! slots (below), and the bytes; 1 then u32 length and a UTF-8 reason. A
//! source that fails mid-answer closes the stream short.
//!
//! A put (to a pool) is `PIEP`, u32 version, u32 key count, the keys, u64
//! bytes per page, u32 length and the layout, the slots, then the bytes;
//! the answer is a status as above.
//!
//! Slots: u32 per page (0 when the source has none), then that many 32-byte
//! hashes a page, page after page: each token's chained hash. A page's key
//! names it, and a sequence goes on from the hash of its last token, so a
//! worker that keeps a run it did not compute needs them.
//!
//! An answer's bytes come page after page, or, where it names groups, in
//! parts: after the slots, u32 groups (0: page after page), and per group
//! u32 spans and each span's u64 offset and u64 length within a page. A
//! part is u32 group, u32 first page, u32 pages, then each of those pages'
//! spans of the group, page after page. The parts come in any order, each
//! (group, page) once; the answer ends with the last. A source that reads
//! its pages out a part at a time (a layer's planes for the pages a fire
//! filled, as the layer is computed) sends each as it has it.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use bytes::BytesMut;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::mpsc;

const MAGIC: &[u8; 4] = b"PIEB";
/// A put: pages to keep, by key.
const PUT: &[u8; 4] = b"PIEP";
const VERSION: u32 = 5;
/// More keys than one sequence could hold is not a request.
const MAX_KEYS: u32 = 1 << 20;
/// More groups, spans a group or slots a page than a model's layers and
/// planes could need is not an answer.
const MAX_GROUPS: u32 = 1 << 16;
const MAX_SPANS: u32 = 1 << 16;
const MAX_SLOTS_PER_PAGE: u32 = 1 << 16;
const SOCKET_BUFFER: usize = 16 << 20;
/// What the asker reads into before handing a chunk on.
const RECV_CHUNK_BYTES: usize = 32 << 20;

pub type Key = [u8; 32];

/// Pages' bytes, back to back, what the source says of their layout, and
/// their tokens' hashes, page after page (empty when it has none).
#[derive(Debug, Clone, Default)]
pub struct Pages {
    pub data: Vec<u8>,
    pub count: usize,
    pub meta: Vec<u8>,
    pub slots: Vec<Key>,
}

impl Pages {
    pub fn page_bytes(&self) -> usize {
        self.data.len().checked_div(self.count).unwrap_or(0)
    }
}

/// Parts of a page, each its spans (offset, length) within it; together
/// they cover the page once.
pub type Groups = Vec<Vec<(u64, u64)>>;

/// An answer as it comes: how many pages of how many bytes, what the source
/// says of their layout, and the bytes in chunks: of whole pages, or, with
/// `groups`, each chunk one [`part`].
pub struct Stream {
    pub count: usize,
    pub page_bytes: usize,
    pub meta: Vec<u8>,
    /// The pages' tokens' hashes, page after page (empty when none).
    pub slots: Vec<Key>,
    /// Empty: page after page.
    pub groups: Groups,
    pub chunks: mpsc::Receiver<Result<Vec<u8>, String>>,
}

const PART_HEAD: usize = 12;

/// A part of a grouped answer as a chunk carries it: group `group` of pages
/// `first..first + pages`, `bytes` each page's spans of it, page after page.
pub fn part(group: u32, first: u32, pages: u32, bytes: &[u8]) -> Vec<u8> {
    let mut chunk = Vec::with_capacity(PART_HEAD + bytes.len());
    for word in [group, first, pages] {
        chunk.extend_from_slice(&word.to_le_bytes());
    }
    chunk.extend_from_slice(bytes);
    chunk
}

/// Whether `groups` cover a page of `page_bytes` once, each byte by one span.
fn tiles(groups: &Groups, page_bytes: usize) -> bool {
    let mut spans: Vec<(u64, u64)> = groups.iter().flatten().copied().collect();
    spans.sort_unstable();
    let mut at = 0u64;
    for (offset, len) in spans {
        if offset != at || len == 0 {
            return false;
        }
        at += len;
    }
    at == page_bytes as u64
}

/// Which (group, page) cells of a grouped answer have come.
struct Cells {
    seen: Vec<bool>,
    count: usize,
    left: usize,
    /// Bytes of one page's spans of each group.
    group_bytes: Vec<usize>,
}

impl Cells {
    fn new(groups: &Groups, count: usize) -> Self {
        Self {
            seen: vec![false; groups.len() * count],
            count,
            left: groups.len() * count,
            group_bytes: groups
                .iter()
                .map(|group| group.iter().map(|(_, len)| *len as usize).sum())
                .collect(),
        }
    }

    /// A part's (group, first page, pages) from its head, and its bytes'
    /// length; marked come.
    fn take(&mut self, head: &[u8]) -> Result<(usize, usize, usize, usize), String> {
        let word = |at: usize| u32::from_le_bytes(head[at..at + 4].try_into().unwrap()) as usize;
        let (group, first, pages) = (word(0), word(4), word(8));
        let Some(&group_bytes) = self.group_bytes.get(group) else {
            return Err(format!("a part of group {group}, past the answer's groups"));
        };
        if first + pages > self.count || pages == 0 {
            return Err(format!(
                "a part of pages {first}+{pages}, past the answer's"
            ));
        }
        for page in first..first + pages {
            let cell = &mut self.seen[group * self.count + page];
            if *cell {
                return Err(format!("group {group} of page {page} came twice"));
            }
            *cell = true;
        }
        self.left -= pages;
        Ok((group, first, pages, pages * group_bytes))
    }

    fn whole(&self) -> bool {
        self.left == 0
    }
}

impl Stream {
    /// The same answer page after page: a grouped one is laid back out as
    /// its parts come, and goes on whole once its last is in.
    pub fn paged(self) -> Result<Stream, String> {
        if self.groups.is_empty() {
            return Ok(self);
        }
        if !tiles(&self.groups, self.page_bytes) {
            return Err("the answer's groups do not cover its pages once".into());
        }
        let Stream {
            count,
            page_bytes,
            meta,
            slots,
            groups,
            mut chunks,
        } = self;
        let (tx, paged) = mpsc::channel(1);
        tokio::spawn(async move {
            let mut pages = vec![0u8; count * page_bytes];
            let mut cells = Cells::new(&groups, count);
            while !cells.whole() {
                let chunk = match chunks.recv().await {
                    Some(Ok(chunk)) => chunk,
                    Some(Err(why)) => {
                        let _ = tx.send(Err(why)).await;
                        return;
                    }
                    None => {
                        let _ = tx
                            .send(Err("the answer stopped before its last part".into()))
                            .await;
                        return;
                    }
                };
                let taken = if chunk.len() >= PART_HEAD {
                    cells.take(&chunk[..PART_HEAD])
                } else {
                    Err("a part shorter than its head".into())
                };
                let (group, first, n, len) = match taken {
                    Ok(part) if part.3 == chunk.len() - PART_HEAD => part,
                    Ok(_) => {
                        let _ = tx
                            .send(Err("a part's bytes are not its pages'".into()))
                            .await;
                        return;
                    }
                    Err(why) => {
                        let _ = tx.send(Err(why)).await;
                        return;
                    }
                };
                let mut bytes = &chunk[PART_HEAD..PART_HEAD + len];
                for page in first..first + n {
                    for &(offset, len) in &groups[group] {
                        let at = page * page_bytes + offset as usize;
                        let (this, rest) = bytes.split_at(len as usize);
                        pages[at..at + len as usize].copy_from_slice(this);
                        bytes = rest;
                    }
                }
            }
            let _ = tx.send(Ok(pages)).await;
        });
        Ok(Stream {
            count,
            page_bytes,
            meta,
            slots,
            groups: Groups::new(),
            chunks: paged,
        })
    }

    /// The whole answer at once.
    pub async fn collect(self) -> Result<Pages, String> {
        let mut this = self.paged()?;
        let mut data = Vec::with_capacity(this.count * this.page_bytes);
        while let Some(chunk) = this.chunks.recv().await {
            data.extend_from_slice(&chunk?);
        }
        if data.len() != this.count * this.page_bytes {
            return Err(format!(
                "the answer stopped at {} of {} bytes",
                data.len(),
                this.count * this.page_bytes
            ));
        }
        Ok(Pages {
            data,
            count: this.count,
            meta: this.meta,
            slots: this.slots,
        })
    }

    /// A whole answer, one chunk.
    pub fn of(pages: Pages) -> Stream {
        let (tx, chunks) = mpsc::channel(1);
        let page_bytes = pages.page_bytes();
        let _ = tx.try_send(Ok(pages.data));
        Stream {
            count: pages.count,
            page_bytes,
            meta: pages.meta,
            slots: pages.slots,
            groups: Groups::new(),
            chunks,
        }
    }
}

pub type Answer = Pin<Box<dyn Future<Output = Result<Stream, String>> + Send>>;

pub type Taken = Pin<Box<dyn Future<Output = Result<(), String>> + Send>>;

/// What a worker serves: the bytes of pages `keys[from..]`; and, where it
/// keeps what others send it, pages to keep.
pub trait Source: Send + Sync + 'static {
    fn read(&self, keys: Vec<Key>, from: u32) -> Answer;

    fn write(&self, keys: Vec<Key>, pages: Pages) -> Taken {
        let _ = (keys, pages);
        Box::pin(async { Err("this worker keeps no pages it is sent".to_string()) })
    }
}

/// The slots as the link carries them: per page, then the hashes. Each of
/// `count` pages names as many.
fn put_slots(out: &mut Vec<u8>, slots: &[Key], count: usize) -> Result<(), String> {
    let per_page = slots.len().checked_div(count).unwrap_or(0);
    if per_page * count != slots.len() {
        return Err(format!(
            "{} slots do not divide among {count} pages",
            slots.len()
        ));
    }
    out.extend_from_slice(&(per_page as u32).to_le_bytes());
    for slot in slots {
        out.extend_from_slice(slot);
    }
    Ok(())
}

fn put_groups(out: &mut Vec<u8>, groups: &Groups) {
    out.extend_from_slice(&(groups.len() as u32).to_le_bytes());
    for group in groups {
        out.extend_from_slice(&(group.len() as u32).to_le_bytes());
        for (offset, len) in group {
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&len.to_le_bytes());
        }
    }
}

async fn read_groups(stream: &mut TcpStream) -> std::io::Result<Groups> {
    let count = stream.read_u32_le().await?;
    if count > MAX_GROUPS {
        return Err(std::io::Error::other("too many groups"));
    }
    let mut groups = Groups::with_capacity(count as usize);
    for _ in 0..count {
        let spans = stream.read_u32_le().await?;
        if spans > MAX_SPANS {
            return Err(std::io::Error::other("too many spans a group"));
        }
        let mut group = Vec::with_capacity(spans as usize);
        for _ in 0..spans {
            group.push((stream.read_u64_le().await?, stream.read_u64_le().await?));
        }
        groups.push(group);
    }
    Ok(groups)
}

async fn read_slots(stream: &mut TcpStream, count: usize) -> std::io::Result<Vec<Key>> {
    let per_page = stream.read_u32_le().await?;
    if per_page > MAX_SLOTS_PER_PAGE {
        return Err(std::io::Error::other("too many slots a page"));
    }
    let per_page = per_page as usize;
    // As many as one sequence's keys, whatever the header says.
    if per_page.saturating_mul(count) > MAX_KEYS as usize {
        return Err(std::io::Error::other("more slots than a sequence holds"));
    }
    let mut slots = vec![[0u8; 32]; per_page * count];
    for slot in &mut slots {
        stream.read_exact(slot).await?;
    }
    Ok(slots)
}

fn tune(stream: &TcpStream) {
    let _ = stream.set_nodelay(true);
    let sock = socket2::SockRef::from(stream);
    let _ = sock.set_send_buffer_size(SOCKET_BUFFER);
    let _ = sock.set_recv_buffer_size(SOCKET_BUFFER);
}

/// Serves `source` on `listener` for as long as the task lives.
pub async fn serve(listener: TcpListener, source: Arc<dyn Source>) {
    loop {
        let (stream, _) = match listener.accept().await {
            Ok(accepted) => accepted,
            Err(error) => {
                tracing::warn!(%error, "bulk accept error");
                continue;
            }
        };
        tune(&stream);
        let source = source.clone();
        tokio::spawn(async move {
            if let Err(error) = answer(stream, &*source).await {
                tracing::debug!(%error, "bulk link closed");
            }
        });
    }
}

async fn answer(mut stream: TcpStream, source: &dyn Source) -> std::io::Result<()> {
    loop {
        let mut magic = [0u8; 4];
        match stream.read_exact(&mut magic).await {
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error),
        }
        let put = &magic == PUT;
        if !(put || &magic == MAGIC) || stream.read_u32_le().await? != VERSION {
            return Err(std::io::Error::other("not a bulk request"));
        }
        let n = stream.read_u32_le().await?;
        if n > MAX_KEYS {
            return Err(std::io::Error::other("too many keys"));
        }
        let mut keys = vec![[0u8; 32]; n as usize];
        for key in &mut keys {
            stream.read_exact(key).await?;
        }
        if put {
            let page_bytes = stream.read_u64_le().await? as usize;
            let mut meta = vec![0u8; stream.read_u32_le().await? as usize];
            stream.read_exact(&mut meta).await?;
            let count = keys.len();
            let slots = read_slots(&mut stream, count).await?;
            let mut data = vec![0u8; page_bytes * count];
            stream.read_exact(&mut data).await?;
            let taken = source
                .write(
                    keys,
                    Pages {
                        data,
                        count,
                        meta,
                        slots,
                    },
                )
                .await;
            let mut status = Vec::new();
            match taken {
                Ok(()) => status.push(0),
                Err(why) => {
                    status.push(1);
                    status.extend_from_slice(&(why.len() as u32).to_le_bytes());
                    status.extend_from_slice(why.as_bytes());
                }
            }
            stream.write_all(&status).await?;
            stream.flush().await?;
            continue;
        }
        let from = stream.read_u32_le().await?;
        match source.read(keys, from).await {
            Ok(mut answer) => {
                let mut head = Vec::with_capacity(17 + answer.meta.len());
                head.push(0);
                head.extend_from_slice(&(answer.count as u32).to_le_bytes());
                head.extend_from_slice(&(answer.page_bytes as u64).to_le_bytes());
                head.extend_from_slice(&(answer.meta.len() as u32).to_le_bytes());
                head.extend_from_slice(&answer.meta);
                put_slots(&mut head, &answer.slots, answer.count).map_err(std::io::Error::other)?;
                put_groups(&mut head, &answer.groups);
                stream.write_all(&head).await?;
                // A grouped answer's parts carry their heads; the asker
                // counts what came.
                let mut sent = 0usize;
                while let Some(chunk) = answer.chunks.recv().await {
                    let chunk = chunk.map_err(std::io::Error::other)?;
                    sent += chunk.len();
                    stream.write_all(&chunk).await?;
                }
                if answer.groups.is_empty() && sent != answer.count * answer.page_bytes {
                    return Err(std::io::Error::other("the source stopped short"));
                }
            }
            Err(why) => {
                let mut refusal = Vec::with_capacity(5 + why.len());
                refusal.push(1);
                refusal.extend_from_slice(&(why.len() as u32).to_le_bytes());
                refusal.extend_from_slice(why.as_bytes());
                stream.write_all(&refusal).await?;
            }
        }
        stream.flush().await?;
    }
}

/// The pages `keys[from..]` at the worker serving `addr`, as they come.
pub async fn fetch_stream(addr: &str, keys: &[Key], from: u32) -> Result<Stream, String> {
    let mut stream = TcpStream::connect(addr)
        .await
        .map_err(|e| format!("connect {addr}: {e}"))?;
    tune(&stream);
    let mut request = Vec::with_capacity(16 + keys.len() * 32);
    request.extend_from_slice(MAGIC);
    request.extend_from_slice(&VERSION.to_le_bytes());
    request.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for key in keys {
        request.extend_from_slice(key);
    }
    request.extend_from_slice(&from.to_le_bytes());
    let io = |e: std::io::Error| format!("{addr}: {e}");
    stream.write_all(&request).await.map_err(io)?;
    if stream.read_u8().await.map_err(io)? != 0 {
        let len = stream.read_u32_le().await.map_err(io)? as usize;
        let mut why = vec![0u8; len];
        stream.read_exact(&mut why).await.map_err(io)?;
        return Err(String::from_utf8_lossy(&why).into_owned());
    }
    let count = stream.read_u32_le().await.map_err(io)? as usize;
    let page_bytes = stream.read_u64_le().await.map_err(io)? as usize;
    let mut meta = vec![0u8; stream.read_u32_le().await.map_err(io)? as usize];
    stream.read_exact(&mut meta).await.map_err(io)?;
    let slots = read_slots(&mut stream, count).await.map_err(io)?;
    let groups = read_groups(&mut stream).await.map_err(io)?;
    let (tx, chunks) = mpsc::channel(4);
    let addr = addr.to_string();
    if !groups.is_empty() {
        let mut cells = Cells::new(&groups, count);
        tokio::spawn(async move {
            while !cells.whole() {
                let mut head = [0u8; PART_HEAD];
                let read = match stream.read_exact(&mut head).await {
                    Ok(_) => cells.take(&head),
                    Err(error) => Err(format!("{addr}: {error}")),
                };
                let chunk = match read {
                    Ok((.., len)) => {
                        let mut chunk = vec![0u8; PART_HEAD + len];
                        chunk[..PART_HEAD].copy_from_slice(&head);
                        match stream.read_exact(&mut chunk[PART_HEAD..]).await {
                            Ok(_) => Ok(chunk),
                            Err(error) => Err(format!("{addr}: {error}")),
                        }
                    }
                    Err(why) => Err(why),
                };
                let failed = chunk.is_err();
                if tx.send(chunk).await.is_err() || failed {
                    return;
                }
            }
        });
        return Ok(Stream {
            count,
            page_bytes,
            meta,
            slots,
            groups,
            chunks,
        });
    }
    let total = count * page_bytes;
    let chunk = RECV_CHUNK_BYTES
        .checked_div(page_bytes)
        .map_or(0, |pages| pages.max(1) * page_bytes);
    tokio::spawn(async move {
        let mut left = total;
        while left > 0 {
            let want = chunk.min(left);
            let mut buf = BytesMut::with_capacity(want);
            while buf.len() < want {
                match stream.read_buf(&mut buf).await {
                    Ok(0) => {
                        let _ = tx
                            .send(Err(format!("{addr} closed {left} bytes short")))
                            .await;
                        return;
                    }
                    Ok(_) => {}
                    Err(error) => {
                        let _ = tx.send(Err(format!("{addr}: {error}"))).await;
                        return;
                    }
                }
            }
            left -= want;
            if tx.send(Ok(buf.into())).await.is_err() {
                return;
            }
        }
    });
    Ok(Stream {
        count,
        page_bytes,
        meta,
        slots,
        groups,
        chunks,
    })
}

/// Sends `pages`, one per key, to be kept by the worker serving `addr`.
pub async fn put(addr: &str, keys: &[Key], pages: &Pages) -> Result<(), String> {
    if keys.len() != pages.count {
        return Err(format!("{} keys for {} pages", keys.len(), pages.count));
    }
    let io = |e: std::io::Error| format!("{addr}: {e}");
    let mut stream = TcpStream::connect(addr).await.map_err(io)?;
    tune(&stream);
    let mut head = Vec::with_capacity(24 + keys.len() * 32 + pages.meta.len());
    head.extend_from_slice(PUT);
    head.extend_from_slice(&VERSION.to_le_bytes());
    head.extend_from_slice(&(keys.len() as u32).to_le_bytes());
    for key in keys {
        head.extend_from_slice(key);
    }
    head.extend_from_slice(&(pages.page_bytes() as u64).to_le_bytes());
    head.extend_from_slice(&(pages.meta.len() as u32).to_le_bytes());
    head.extend_from_slice(&pages.meta);
    put_slots(&mut head, &pages.slots, pages.count)?;
    stream.write_all(&head).await.map_err(io)?;
    stream.write_all(&pages.data).await.map_err(io)?;
    if stream.read_u8().await.map_err(io)? == 0 {
        return Ok(());
    }
    let len = stream.read_u32_le().await.map_err(io)? as usize;
    let mut why = vec![0u8; len];
    stream.read_exact(&mut why).await.map_err(io)?;
    Err(String::from_utf8_lossy(&why).into_owned())
}

/// The bytes of pages `keys[from..]` at the worker serving `addr`.
pub async fn fetch(addr: &str, keys: &[Key], from: u32) -> Result<Pages, String> {
    fetch_stream(addr, keys, from).await?.collect().await
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;

    impl Source for Echo {
        fn read(&self, keys: Vec<Key>, from: u32) -> Answer {
            Box::pin(async move {
                if from as usize > keys.len() {
                    return Err("past the run".to_string());
                }
                let (tx, chunks) = mpsc::channel(2);
                let count = keys.len() - from as usize;
                // Two slots a page, each the page's key.
                let slots = keys[from as usize..]
                    .iter()
                    .flat_map(|key| [*key, *key])
                    .collect();
                tokio::spawn(async move {
                    for key in &keys[from as usize..] {
                        let _ = tx.send(Ok(key.to_vec())).await;
                    }
                });
                Ok(Stream {
                    count,
                    page_bytes: 32,
                    meta: b"echo".to_vec(),
                    slots,
                    groups: Groups::new(),
                    chunks,
                })
            })
        }
    }

    /// Pages of 8 bytes, page `p` byte `b` holding `10 * p + b`, sent as two
    /// groups (bytes 4.. then ..4): as two fires would, the first page's
    /// groups, then the rest's.
    struct Halves;

    impl Source for Halves {
        fn read(&self, keys: Vec<Key>, _from: u32) -> Answer {
            Box::pin(async move {
                let count = keys.len();
                let groups: Groups = vec![vec![(4, 4)], vec![(0, 4)]];
                let bytes = |group: usize, pages: std::ops::Range<usize>| -> Vec<u8> {
                    let (offset, len) = groups[group][0];
                    pages
                        .flat_map(|p| {
                            (offset..offset + len).map(move |b| (10 * p as u64 + b) as u8)
                        })
                        .collect()
                };
                let parts = [
                    part(0, 0, 1, &bytes(0, 0..1)),
                    part(1, 0, 1, &bytes(1, 0..1)),
                    part(0, 1, count as u32 - 1, &bytes(0, 1..count)),
                    part(1, 1, count as u32 - 1, &bytes(1, 1..count)),
                ];
                let (tx, chunks) = mpsc::channel(8);
                for chunk in parts {
                    let _ = tx.try_send(Ok(chunk));
                }
                Ok(Stream {
                    count,
                    page_bytes: 8,
                    meta: Vec::new(),
                    slots: Vec::new(),
                    groups,
                    chunks,
                })
            })
        }
    }

    #[tokio::test]
    async fn an_answer_sent_in_parts_comes_back_page_after_page() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(serve(listener, Arc::new(Halves)));
        let pages = fetch(&addr, &[[1u8; 32], [2u8; 32], [3u8; 32]], 0)
            .await
            .unwrap();
        let want: Vec<u8> = (0..3u8)
            .flat_map(|p| (0..8u8).map(move |b| 10 * p + b))
            .collect();
        assert_eq!(pages.data, want);
    }

    #[test]
    fn groups_must_cover_a_page_once() {
        assert!(tiles(&vec![vec![(4, 4)], vec![(0, 4)]], 8));
        assert!(!tiles(&vec![vec![(0, 4)], vec![(2, 6)]], 8), "an overlap");
        assert!(!tiles(&vec![vec![(0, 4)]], 8), "a gap");
    }

    #[tokio::test]
    async fn a_source_that_keeps_nothing_refuses_a_put() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(serve(listener, Arc::new(Echo)));
        let pages = Pages {
            data: vec![7u8; 64],
            count: 2,
            meta: Vec::new(),
            slots: vec![[9u8; 32]; 4],
        };
        let refused = put(&addr, &[[1u8; 32], [2u8; 32]], &pages)
            .await
            .unwrap_err();
        assert!(refused.contains("keeps no pages"), "{refused}");
        // The link answers a fetch after a refused put.
        assert_eq!(fetch(&addr, &[[5u8; 32]], 0).await.unwrap().count, 1);
    }

    #[tokio::test]
    async fn pages_cross_whole_and_a_refusal_crosses_as_its_reason() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        tokio::spawn(serve(listener, Arc::new(Echo)));
        let keys = [[1u8; 32], [2u8; 32], [3u8; 32]];
        let pages = fetch(&addr, &keys, 1).await.unwrap();
        assert_eq!(pages.count, 2);
        assert_eq!(pages.page_bytes(), 32);
        assert_eq!(&pages.data[..32], &[2u8; 32]);
        assert_eq!(&pages.data[32..], &[3u8; 32]);
        assert_eq!(pages.meta, b"echo");
        assert_eq!(pages.slots, [[2u8; 32], [2u8; 32], [3u8; 32], [3u8; 32]]);
        assert_eq!(fetch(&addr, &keys, 9).await.unwrap_err(), "past the run");
    }
}
