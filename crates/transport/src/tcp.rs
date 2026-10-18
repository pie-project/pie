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
//! u32 length and the source's description of its layout (opaque here), and
//! the bytes; 1 then u32 length and a UTF-8 reason. A source that fails
//! mid-answer closes the stream short.
//!
//! A put (to a pool) is `PIEP`, u32 version, u32 key count, the keys, u64
//! bytes per page, u32 length and the layout, then the bytes; the answer is
//! a status as above.

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
const VERSION: u32 = 3;
/// More keys than one sequence could hold is not a request.
const MAX_KEYS: u32 = 1 << 20;
const SOCKET_BUFFER: usize = 16 << 20;
/// What the asker reads into before handing a chunk on.
const RECV_CHUNK_BYTES: usize = 32 << 20;

pub type Key = [u8; 32];

/// Pages' bytes, back to back, and what the source says of their layout.
#[derive(Debug, Clone, Default)]
pub struct Pages {
    pub data: Vec<u8>,
    pub count: usize,
    pub meta: Vec<u8>,
}

impl Pages {
    pub fn page_bytes(&self) -> usize {
        self.data.len().checked_div(self.count).unwrap_or(0)
    }
}

/// An answer as it comes: how many pages of how many bytes, what the source
/// says of their layout, and the bytes in chunks of whole pages.
pub struct Stream {
    pub count: usize,
    pub page_bytes: usize,
    pub meta: Vec<u8>,
    pub chunks: mpsc::Receiver<Result<Vec<u8>, String>>,
}

impl Stream {
    /// The whole answer at once.
    pub async fn collect(mut self) -> Result<Pages, String> {
        let mut data = Vec::with_capacity(self.count * self.page_bytes);
        while let Some(chunk) = self.chunks.recv().await {
            data.extend_from_slice(&chunk?);
        }
        if data.len() != self.count * self.page_bytes {
            return Err(format!(
                "the answer stopped at {} of {} bytes",
                data.len(),
                self.count * self.page_bytes
            ));
        }
        Ok(Pages {
            data,
            count: self.count,
            meta: self.meta,
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
            let mut data = vec![0u8; page_bytes * keys.len()];
            stream.read_exact(&mut data).await?;
            let count = keys.len();
            let taken = source.write(keys, Pages { data, count, meta }).await;
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
                stream.write_all(&head).await?;
                let mut sent = 0usize;
                while let Some(chunk) = answer.chunks.recv().await {
                    let chunk = chunk.map_err(std::io::Error::other)?;
                    sent += chunk.len();
                    stream.write_all(&chunk).await?;
                }
                if sent != answer.count * answer.page_bytes {
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
    let (tx, chunks) = mpsc::channel(4);
    let total = count * page_bytes;
    let chunk = RECV_CHUNK_BYTES
        .checked_div(page_bytes)
        .map_or(0, |pages| pages.max(1) * page_bytes);
    let addr = addr.to_string();
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
                tokio::spawn(async move {
                    for key in &keys[from as usize..] {
                        let _ = tx.send(Ok(key.to_vec())).await;
                    }
                });
                Ok(Stream {
                    count,
                    page_bytes: 32,
                    meta: b"echo".to_vec(),
                    chunks,
                })
            })
        }
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
        assert_eq!(fetch(&addr, &keys, 9).await.unwrap_err(), "past the run");
    }
}
