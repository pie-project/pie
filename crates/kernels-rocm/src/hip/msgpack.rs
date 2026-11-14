//! MessagePack, as much as AMD's code objects and hipBLASLt's logic files
//! use: a value read whole, then walked by key.

/// `bytes` read as one value; none where it is not whole MessagePack.
pub(crate) fn read(bytes: &[u8]) -> Option<Mp> {
    Reader { b: bytes, at: 0 }.value()
}

#[derive(Debug)]
pub(crate) enum Mp {
    Nil,
    Bool(bool),
    Int(i64),
    Float,
    Str(String),
    Bin,
    Arr(Vec<Mp>),
    Map(Vec<(Mp, Mp)>),
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let s = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }
    fn be(&mut self, n: usize) -> Option<u64> {
        Some(self.take(n)?.iter().fold(0, |a, &x| a << 8 | u64::from(x)))
    }
    fn str(&mut self, n: usize) -> Option<Mp> {
        Some(Mp::Str(String::from_utf8_lossy(self.take(n)?).into_owned()))
    }
    fn arr(&mut self, n: usize) -> Option<Mp> {
        (0..n)
            .map(|_| self.value())
            .collect::<Option<_>>()
            .map(Mp::Arr)
    }
    fn map(&mut self, n: usize) -> Option<Mp> {
        (0..n)
            .map(|_| Some((self.value()?, self.value()?)))
            .collect::<Option<_>>()
            .map(Mp::Map)
    }
    fn value(&mut self) -> Option<Mp> {
        let t = self.take(1)?[0];
        Some(match t {
            0x00..=0x7f => Mp::Int(i64::from(t)),
            0x80..=0x8f => self.map(usize::from(t & 15))?,
            0x90..=0x9f => self.arr(usize::from(t & 15))?,
            0xa0..=0xbf => self.str(usize::from(t & 31))?,
            0xe0..=0xff => Mp::Int(i64::from(t as i8)),
            0xc0 => Mp::Nil,
            0xc2 => Mp::Bool(false),
            0xc3 => Mp::Bool(true),
            0xc4..=0xc6 => {
                let n = self.be(1 << (t - 0xc4))? as usize;
                self.take(n)?;
                Mp::Bin
            }
            0xca => {
                self.take(4)?;
                Mp::Float
            }
            0xcb => {
                self.take(8)?;
                Mp::Float
            }
            0xcc..=0xcf => Mp::Int(self.be(1 << (t - 0xcc))? as i64),
            0xd0 => Mp::Int(i64::from(self.be(1)? as u8 as i8)),
            0xd1 => Mp::Int(i64::from(self.be(2)? as u16 as i16)),
            0xd2 => Mp::Int(i64::from(self.be(4)? as u32 as i32)),
            0xd3 => Mp::Int(self.be(8)? as i64),
            0xd9..=0xdb => {
                let n = self.be(1 << (t - 0xd9))? as usize;
                self.str(n)?
            }
            0xdc | 0xdd => {
                let n = self.be(2 << (t - 0xdc))? as usize;
                self.arr(n)?
            }
            0xde | 0xdf => {
                let n = self.be(2 << (t - 0xde))? as usize;
                self.map(n)?
            }
            _ => return None,
        })
    }
}

impl Mp {
    pub(crate) fn get(&self, key: &str) -> Option<&Mp> {
        let Mp::Map(kv) = self else { return None };
        kv.iter()
            .find(|(k, _)| matches!(k, Mp::Str(s) if s == key))
            .map(|(_, v)| v)
    }
    pub(crate) fn at(&self, path: &[&str]) -> Option<&Mp> {
        path.iter().try_fold(self, |v, k| v.get(k))
    }
    pub(crate) fn int(&self) -> Option<i64> {
        match self {
            Mp::Int(v) => Some(*v),
            Mp::Bool(b) => Some(i64::from(*b)),
            _ => None,
        }
    }
    pub(crate) fn truth(&self) -> bool {
        matches!(self, Mp::Bool(true))
    }
    pub(crate) fn text(&self) -> Option<&str> {
        match self {
            Mp::Str(s) => Some(s),
            _ => None,
        }
    }
    pub(crate) fn ints(&self) -> Vec<i64> {
        match self {
            Mp::Arr(v) => v.iter().filter_map(Mp::int).collect(),
            _ => Vec::new(),
        }
    }
}
