//! A `.npy` file read in slices (`pread`, no mapping): little-endian `u2`
//! and `f4` arrays in C order, as the prep writes them.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

type R<T> = std::result::Result<T, String>;

pub struct Npy {
    file: File,
    name: String,
    pub shape: Vec<usize>,
    /// Bytes an element: 2 (`<u2`) or 4 (`<f4`, `<u4`).
    pub elem: usize,
    data: u64,
}

impl Npy {
    pub fn open(path: &Path) -> R<Npy> {
        let name = path.display().to_string();
        let file = File::open(path).map_err(|e| format!("{name}: {e}"))?;
        let mut head = [0u8; 10];
        file.read_exact_at(&mut head, 0)
            .map_err(|e| format!("{name}: {e}"))?;
        if &head[..6] != b"\x93NUMPY" {
            return Err(format!("{name}: not a .npy file"));
        }
        let (hlen, at) = if head[6] == 1 {
            (u16::from_le_bytes([head[8], head[9]]) as usize, 10u64)
        } else {
            let mut b = [0u8; 4];
            file.read_exact_at(&mut b, 8)
                .map_err(|e| format!("{name}: {e}"))?;
            (u32::from_le_bytes(b) as usize, 12u64)
        };
        let mut h = vec![0u8; hlen];
        file.read_exact_at(&mut h, at)
            .map_err(|e| format!("{name}: {e}"))?;
        let h = String::from_utf8_lossy(&h).to_string();
        let descr = field(&h, "descr").ok_or_else(|| format!("{name}: no descr"))?;
        let elem = match descr.trim_matches(|c| c == '\'' || c == '"') {
            "<u2" | "<i2" => 2,
            "<f4" | "<u4" | "<i4" => 4,
            d => return Err(format!("{name}: dtype {d}")),
        };
        if h.contains("'fortran_order': True") {
            return Err(format!("{name}: fortran order"));
        }
        let s = h
            .find("'shape':")
            .ok_or_else(|| format!("{name}: no shape"))?;
        let rest = &h[s..];
        let (a, b) = (rest.find('(').unwrap(), rest.find(')').unwrap());
        let shape = rest[a + 1..b]
            .split(',')
            .filter(|t| !t.trim().is_empty())
            .map(|t| {
                t.trim()
                    .parse::<usize>()
                    .map_err(|e| format!("{name}: {e}"))
            })
            .collect::<R<Vec<_>>>()?;
        Ok(Npy {
            file,
            name,
            shape,
            elem,
            data: at + hlen as u64,
        })
    }

    /// Elements `at .. at + n` (flat C order), one a word (a 16-bit element
    /// zero-extended, a 32-bit one as its bits).
    pub fn words(&self, at: usize, n: usize, out: &mut Vec<u32>) -> R<()> {
        let mut b = vec![0u8; n * self.elem];
        self.file
            .read_exact_at(&mut b, self.data + (at * self.elem) as u64)
            .map_err(|e| format!("{}: {e}", self.name))?;
        if self.elem == 2 {
            out.extend(
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| u16::from_le_bytes(*c) as u32),
            );
        } else {
            out.extend(b.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)));
        }
        Ok(())
    }

    /// Elements an index of axis 0.
    pub fn stride0(&self) -> usize {
        self.shape[1..].iter().product()
    }

    /// The `[i0][r0..r1][c0..c1][..]` block of a 3- or 4-D array as words,
    /// C order (`i0` only on 4-D... arrays with a leading axis: pass `None`
    /// for a 3-D array).
    pub fn block(
        &self,
        lead: Option<usize>,
        rows: (usize, usize),
        cols: (usize, usize),
    ) -> R<Vec<u32>> {
        let sh = if lead.is_some() {
            &self.shape[1..]
        } else {
            &self.shape[..]
        };
        let inner: usize = sh[2..].iter().product();
        let base = lead.map_or(0, |i| i * self.stride0());
        let mut out = Vec::with_capacity((rows.1 - rows.0) * (cols.1 - cols.0) * inner);
        for r in rows.0..rows.1 {
            self.words(
                base + (r * sh[1] + cols.0) * inner,
                (cols.1 - cols.0) * inner,
                &mut out,
            )?;
        }
        Ok(out)
    }
}

fn field(h: &str, key: &str) -> Option<String> {
    let at = h.find(&format!("'{key}':"))?;
    let rest = h[at + key.len() + 3..].trim_start();
    let end = rest.find(',')?;
    Some(rest[..end].trim().to_string())
}
