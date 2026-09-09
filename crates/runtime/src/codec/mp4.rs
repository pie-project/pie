use super::y4m::frame_rate;

const MOVIE_TIMESCALE: u32 = 1000;

struct Writer {
    buf: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn open(&mut self, kind: &[u8; 4]) -> usize {
        let at = self.buf.len();
        self.buf.extend_from_slice(&[0, 0, 0, 0]);
        self.buf.extend_from_slice(kind);
        at
    }

    fn close(&mut self, at: usize) {
        let size = (self.buf.len() - at) as u32;
        self.buf[at..at + 4].copy_from_slice(&size.to_be_bytes());
    }

    fn full(&mut self, version: u8, flags: u32) {
        self.u32(((version as u32) << 24) | (flags & 0x00ff_ffff));
    }

    fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_be_bytes());
    }

    fn zeros(&mut self, n: usize) {
        self.buf.resize(self.buf.len() + n, 0);
    }

    fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }

    fn unity_matrix(&mut self) {
        for v in [0x0001_0000u32, 0, 0, 0, 0x0001_0000, 0, 0, 0, 0x4000_0000] {
            self.u32(v);
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sample {
    pub avcc: Vec<u8>,
    pub is_sync: bool,
}

#[derive(Debug, Clone)]
pub struct Track {
    pub width: u32,
    pub height: u32,
    pub timescale: u32,
    pub sample_delta: u32,
    pub sps: Vec<u8>,
    pub pps: Vec<u8>,
}

fn avcc_payload(sps: &[u8], pps: &[u8]) -> Result<Vec<u8>, String> {
    if sps.len() < 4 {
        return Err(format!("avcC: SPS is {} bytes, need at least 4", sps.len()));
    }
    if pps.is_empty() {
        return Err("avcC: no PPS".to_string());
    }
    let mut v = Vec::with_capacity(11 + sps.len() + pps.len());
    v.push(1);
    v.push(sps[1]);
    v.push(sps[2]);
    v.push(sps[3]);
    v.push(0xff);
    v.push(0xe1);
    v.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    v.extend_from_slice(sps);
    v.push(1);
    v.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    v.extend_from_slice(pps);
    Ok(v)
}

pub fn mux(track: &Track, samples: &[Sample]) -> Result<Vec<u8>, String> {
    if samples.is_empty() {
        return Err("mp4: no samples to mux".to_string());
    }
    if track.timescale == 0 || track.sample_delta == 0 {
        return Err(format!(
            "mp4: degenerate timing (timescale {}, delta {})",
            track.timescale, track.sample_delta
        ));
    }
    if track.width > u16::MAX as u32 || track.height > u16::MAX as u32 {
        return Err(format!(
            "mp4: {}x{} exceeds the container's 16-bit frame size",
            track.width, track.height
        ));
    }
    let avcc = avcc_payload(&track.sps, &track.pps)?;

    let mut w = Writer::new();

    let b = w.open(b"ftyp");
    w.bytes(b"isom");
    w.u32(512);
    for brand in [b"isom", b"iso2", b"avc1", b"mp41"] {
        w.bytes(brand);
    }
    w.close(b);

    let mdat = w.open(b"mdat");
    let data_offset = w.buf.len();
    let mut sizes = Vec::with_capacity(samples.len());
    for s in samples {
        sizes.push(s.avcc.len() as u32);
        w.bytes(&s.avcc);
    }
    w.close(mdat);

    let n = samples.len() as u32;
    let media_duration = (n as u64) * track.sample_delta as u64;
    let movie_duration = media_duration * MOVIE_TIMESCALE as u64 / track.timescale.max(1) as u64;

    let moov = w.open(b"moov");

    let mvhd = w.open(b"mvhd");
    w.full(0, 0);
    w.u32(0);
    w.u32(0);
    w.u32(MOVIE_TIMESCALE);
    w.u32(movie_duration.min(u32::MAX as u64) as u32);
    w.u32(0x0001_0000);
    w.u16(0x0100);
    w.u16(0);
    w.zeros(8);
    w.unity_matrix();
    w.zeros(24);
    w.u32(2);
    w.close(mvhd);

    let trak = w.open(b"trak");

    let tkhd = w.open(b"tkhd");
    w.full(0, 0x7);
    w.u32(0);
    w.u32(0);
    w.u32(1);
    w.u32(0);
    w.u32(movie_duration.min(u32::MAX as u64) as u32);
    w.zeros(8);
    w.u16(0);
    w.u16(0);
    w.u16(0);
    w.u16(0);
    w.unity_matrix();
    w.u32(track.width << 16);
    w.u32(track.height << 16);
    w.close(tkhd);

    let mdia = w.open(b"mdia");

    let mdhd = w.open(b"mdhd");
    w.full(0, 0);
    w.u32(0);
    w.u32(0);
    w.u32(track.timescale);
    w.u32(media_duration.min(u32::MAX as u64) as u32);
    w.u16(0x55c4); // language "und", packed 5-bit ISO-639-2
    w.u16(0);
    w.close(mdhd);

    let hdlr = w.open(b"hdlr");
    w.full(0, 0);
    w.u32(0);
    w.bytes(b"vide");
    w.zeros(12);
    w.bytes(b"pie video\0");
    w.close(hdlr);

    let minf = w.open(b"minf");

    let vmhd = w.open(b"vmhd");
    w.full(0, 1);
    w.u16(0);
    w.zeros(6);
    w.close(vmhd);

    let dinf = w.open(b"dinf");
    let dref = w.open(b"dref");
    w.full(0, 0);
    w.u32(1);
    let url = w.open(b"url ");
    w.full(0, 1); // "the media is in this same file"
    w.close(url);
    w.close(dref);
    w.close(dinf);

    let stbl = w.open(b"stbl");

    let stsd = w.open(b"stsd");
    w.full(0, 0);
    w.u32(1);
    let avc1 = w.open(b"avc1");
    w.zeros(6);
    w.u16(1);
    w.u16(0);
    w.u16(0);
    w.zeros(12);
    w.u16(track.width as u16);
    w.u16(track.height as u16);
    w.u32(0x0048_0000);
    w.u32(0x0048_0000);
    w.u32(0);
    w.u16(1);
    let name = b"pie NVENC H.264";
    w.bytes(&[name.len() as u8]);
    w.bytes(name);
    w.zeros(31 - name.len());
    w.u16(0x0018);
    w.u16(0xffff);
    let avcc_box = w.open(b"avcC");
    w.bytes(&avcc);
    w.close(avcc_box);
    w.close(avc1);
    w.close(stsd);

    let stts = w.open(b"stts");
    w.full(0, 0);
    w.u32(1);
    w.u32(n);
    w.u32(track.sample_delta);
    w.close(stts);

    if !samples.iter().all(|s| s.is_sync) {
        let stss = w.open(b"stss");
        w.full(0, 0);
        let syncs: Vec<u32> = samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.is_sync)
            .map(|(i, _)| i as u32 + 1)
            .collect();
        w.u32(syncs.len() as u32);
        for s in syncs {
            w.u32(s);
        }
        w.close(stss);
    }

    let stsc = w.open(b"stsc");
    w.full(0, 0);
    w.u32(1);
    w.u32(1);
    w.u32(n);
    w.u32(1);
    w.close(stsc);

    let stsz = w.open(b"stsz");
    w.full(0, 0);
    w.u32(0);
    w.u32(n);
    for s in &sizes {
        w.u32(*s);
    }
    w.close(stsz);

    let stco = w.open(b"stco");
    w.full(0, 0);
    w.u32(1);
    w.u32(data_offset as u32);
    w.close(stco);

    w.close(stbl);
    w.close(minf);
    w.close(mdia);
    w.close(trak);
    w.close(moov);

    Ok(w.buf)
}

pub fn nal_units(data: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut e = starts.get(k + 1).map_or(data.len(), |&n| n - 3);
        while e > s && data[e - 1] == 0 {
            e -= 1;
        }
        if e > s {
            out.push(&data[s..e]);
        }
    }
    out
}

pub fn sample_from_annexb(
    annexb: &[u8],
    sps: &mut Option<Vec<u8>>,
    pps: &mut Option<Vec<u8>>,
) -> Sample {
    let mut avcc = Vec::with_capacity(annexb.len());
    let mut is_sync = false;
    for nal in nal_units(annexb) {
        match nal[0] & 0x1f {
            7 => {
                if sps.is_none() {
                    *sps = Some(nal.to_vec());
                }
            }
            8 => {
                if pps.is_none() {
                    *pps = Some(nal.to_vec());
                }
            }
            9 | 12 => {}
            kind => {
                if kind == 5 {
                    is_sync = true;
                }
                avcc.extend_from_slice(&(nal.len() as u32).to_be_bytes());
                avcc.extend_from_slice(nal);
            }
        }
    }
    Sample { avcc, is_sync }
}

pub fn mux_annexb(
    width: u32,
    height: u32,
    fps: f32,
    pictures: &[Vec<u8>],
) -> Result<Vec<u8>, String> {
    let mut sps = None;
    let mut pps = None;
    let samples: Vec<Sample> = pictures
        .iter()
        .map(|p| sample_from_annexb(p, &mut sps, &mut pps))
        .collect();
    let (num, den) = frame_rate(fps);
    let track = Track {
        width,
        height,
        timescale: num,
        sample_delta: den,
        sps: sps.ok_or("mp4: the stream carries no SPS")?,
        pps: pps.ok_or("mp4: the stream carries no PPS")?,
    };
    mux(&track, &samples)
}

#[derive(Debug, Clone)]
pub struct Node {
    pub kind: [u8; 4],
    pub offset: usize,
    pub size: usize,
    pub body: usize,
    pub children: Vec<Node>,
}

impl Node {
    pub fn name(&self) -> &str {
        std::str::from_utf8(&self.kind).unwrap_or("????")
    }
}

fn container_preamble(kind: &[u8; 4]) -> Option<usize> {
    match kind {
        b"moov" | b"trak" | b"edts" | b"mdia" | b"minf" | b"dinf" | b"stbl" | b"udta" | b"mvex"
        | b"moof" | b"traf" => Some(0),
        b"stsd" | b"dref" => Some(8),
        b"avc1" | b"avc3" | b"hvc1" | b"hev1" => Some(78),
        _ => None,
    }
}

pub fn parse_boxes(data: &[u8]) -> Result<Vec<Node>, String> {
    parse_at(data, 0, data.len())
}

fn parse_at(data: &[u8], from: usize, to: usize) -> Result<Vec<Node>, String> {
    let mut out = Vec::new();
    let mut at = from;
    while at + 8 <= to {
        let size = u32::from_be_bytes(data[at..at + 4].try_into().unwrap()) as usize;
        let mut kind = [0u8; 4];
        kind.copy_from_slice(&data[at + 4..at + 8]);
        let size = match size {
            0 => to - at,
            1 => {
                if at + 16 > to {
                    return Err(format!(
                        "box '{}' claims a 64-bit size it cannot hold",
                        String::from_utf8_lossy(&kind)
                    ));
                }
                u64::from_be_bytes(data[at + 8..at + 16].try_into().unwrap()) as usize
            }
            n => n,
        };
        if size < 8 || at + size > to {
            return Err(format!(
                "box '{}' at {at} claims {size} bytes, {} remain",
                String::from_utf8_lossy(&kind),
                to - at
            ));
        }
        let body = at + 8;
        let children = match container_preamble(&kind) {
            Some(pre) if body + pre <= at + size => parse_at(data, body + pre, at + size)?,
            _ => Vec::new(),
        };
        out.push(Node {
            kind,
            offset: at,
            size,
            body,
            children,
        });
        at += size;
    }
    Ok(out)
}

pub fn find<'a>(nodes: &'a [Node], path: &str) -> Option<&'a Node> {
    let mut here: &'a [Node] = nodes;
    let mut found: Option<&'a Node> = None;
    for step in path.split('/') {
        let node = here.iter().find(|n| n.name() == step)?;
        here = &node.children;
        found = Some(node);
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(pictures: usize) -> Vec<Vec<u8>> {
        let sps = [0x67u8, 0x64, 0x00, 0x28, 0xac, 0xd9, 0x40, 0x78];
        let pps = [0x68u8, 0xeb, 0xe3, 0xcb, 0x22, 0xc0];
        (0..pictures)
            .map(|i| {
                let mut v = Vec::new();
                v.extend_from_slice(&[0, 0, 0, 1, 9, 0x10]);
                if i == 0 {
                    v.extend_from_slice(&[0, 0, 0, 1]);
                    v.extend_from_slice(&sps);
                    v.extend_from_slice(&[0, 0, 0, 1]);
                    v.extend_from_slice(&pps);
                }
                v.extend_from_slice(&[0, 0, 0, 1]);
                v.push(if i == 0 { 0x65 } else { 0x41 });
                v.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef, i as u8 + 1]);
                v
            })
            .collect()
    }

    #[test]
    fn mp4_every_case() {
        nal_units_ignore_start_code_length();
        parameter_sets_leave_the_sample_and_land_in_avcc();
        every_box_the_structure_needs_is_present_and_well_sized();
        stco_points_at_the_first_sample_and_stts_states_the_rate();
        a_stream_without_parameter_sets_is_refused_by_name();
        a_truncated_box_length_is_caught_rather_than_walked_past();
    }

    fn nal_units_ignore_start_code_length() {
        let s = [0u8, 0, 0, 1, 0xaa, 0xbb, 0, 0, 1, 0xcc];
        let n = nal_units(&s);
        assert_eq!(n.len(), 2);
        assert_eq!(n[0], &[0xaa, 0xbb]);
        assert_eq!(n[1], &[0xcc]);
    }

    fn parameter_sets_leave_the_sample_and_land_in_avcc() {
        let pics = synthetic(3);
        let mp4 = mux_annexb(64, 64, 25.0, &pics).expect("mux");
        let tree = parse_boxes(&mp4).expect("parse");
        let avcc = find(&tree, "moov/trak/mdia/minf/stbl/stsd/avc1/avcC").expect("avcC");
        let payload = &mp4[avcc.body..avcc.offset + avcc.size];
        assert_eq!(payload[0], 1, "configurationVersion");
        assert_eq!(payload[1], 0x64, "profile from SPS[1]");
        assert_eq!(payload[3], 0x28, "level from SPS[3]");
        assert_eq!(payload[4], 0xff, "lengthSizeMinusOne = 3");

        let stsz = find(&tree, "moov/trak/mdia/minf/stbl/stsz").expect("stsz");
        let body = &mp4[stsz.body..];
        assert_eq!(
            u32::from_be_bytes(body[4..8].try_into().unwrap()),
            0,
            "sample_size table"
        );
        assert_eq!(
            u32::from_be_bytes(body[8..12].try_into().unwrap()),
            3,
            "sample_count"
        );
        for i in 0..3 {
            let at = 12 + i * 4;
            assert_eq!(u32::from_be_bytes(body[at..at + 4].try_into().unwrap()), 10);
        }
    }

    fn every_box_the_structure_needs_is_present_and_well_sized() {
        let mp4 = mux_annexb(64, 48, 30.0, &synthetic(8)).expect("mux");
        let tree = parse_boxes(&mp4).expect("parse");
        assert_eq!(tree[0].name(), "ftyp");
        assert_eq!(tree[1].name(), "mdat");
        assert_eq!(tree[2].name(), "moov");
        assert_eq!(tree.iter().map(|n| n.size).sum::<usize>(), mp4.len());
        for path in [
            "moov/mvhd",
            "moov/trak/tkhd",
            "moov/trak/mdia/mdhd",
            "moov/trak/mdia/hdlr",
            "moov/trak/mdia/minf/vmhd",
            "moov/trak/mdia/minf/dinf/dref",
            "moov/trak/mdia/minf/stbl/stsd/avc1/avcC",
            "moov/trak/mdia/minf/stbl/stts",
            "moov/trak/mdia/minf/stbl/stss",
            "moov/trak/mdia/minf/stbl/stsc",
            "moov/trak/mdia/minf/stbl/stsz",
            "moov/trak/mdia/minf/stbl/stco",
        ] {
            assert!(find(&tree, path).is_some(), "missing {path}");
        }
        let stss = find(&tree, "moov/trak/mdia/minf/stbl/stss").unwrap();
        let body = &mp4[stss.body..];
        assert_eq!(u32::from_be_bytes(body[4..8].try_into().unwrap()), 1);
        assert_eq!(u32::from_be_bytes(body[8..12].try_into().unwrap()), 1);
    }

    fn stco_points_at_the_first_sample_and_stts_states_the_rate() {
        let pics = synthetic(4);
        let mp4 = mux_annexb(64, 64, 23.976, &pics).expect("mux");
        let tree = parse_boxes(&mp4).expect("parse");

        let mdat = tree.iter().find(|n| n.name() == "mdat").unwrap();
        let stco = find(&tree, "moov/trak/mdia/minf/stbl/stco").unwrap();
        let offset = u32::from_be_bytes(mp4[stco.body + 8..stco.body + 12].try_into().unwrap());
        assert_eq!(offset as usize, mdat.body, "chunk offset is mdat's payload");

        let stts = find(&tree, "moov/trak/mdia/minf/stbl/stts").unwrap();
        let b = &mp4[stts.body..];
        assert_eq!(
            u32::from_be_bytes(b[4..8].try_into().unwrap()),
            1,
            "one run"
        );
        assert_eq!(
            u32::from_be_bytes(b[8..12].try_into().unwrap()),
            4,
            "four samples"
        );
        assert_eq!(
            u32::from_be_bytes(b[12..16].try_into().unwrap()),
            1000,
            "delta"
        );
        let mdhd = find(&tree, "moov/trak/mdia/mdhd").unwrap();
        let m = &mp4[mdhd.body..];
        assert_eq!(
            u32::from_be_bytes(m[12..16].try_into().unwrap()),
            23_976,
            "timescale"
        );
    }

    fn a_stream_without_parameter_sets_is_refused_by_name() {
        let err = mux_annexb(16, 16, 25.0, &[vec![0, 0, 0, 1, 0x41, 0x00]]).unwrap_err();
        assert!(err.contains("SPS"), "{err}");
    }

    fn a_truncated_box_length_is_caught_rather_than_walked_past() {
        let mut mp4 = mux_annexb(16, 16, 25.0, &synthetic(1)).expect("mux");
        let n = mp4.len();
        mp4[0..4].copy_from_slice(&((n + 16) as u32).to_be_bytes());
        assert!(parse_boxes(&mp4).is_err());
    }
}
