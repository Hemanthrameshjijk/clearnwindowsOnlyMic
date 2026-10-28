//! Minimal `.npz` reader (numpy `savez`): stored or deflated entries, little-endian f4/f8/i4/i8 arrays.
//! Everything is widened to f64; callers narrow with [`Npy::f32s`].

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

pub struct Npy {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Npy {
    pub fn f32s(&self) -> Vec<f32> {
        self.data.iter().map(|&v| v as f32).collect()
    }
    pub fn scalar(&self) -> f64 {
        self.data[0]
    }
}

pub struct Npz(HashMap<String, Npy>);

impl Npz {
    pub fn open(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&bytes).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn get(&self, key: &str) -> Result<&Npy, String> {
        self.0.get(key).ok_or_else(|| format!("npz key '{key}' missing"))
    }
    pub fn has(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn scalar(&self, key: &str) -> Result<f64, String> {
        Ok(self.get(key)?.scalar())
    }
    pub fn f32s(&self, key: &str) -> Result<Vec<f32>, String> {
        Ok(self.get(key)?.f32s())
    }
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }

    pub fn parse(b: &[u8]) -> Result<Self, String> {
        let u16at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]) as usize;
        let u32at = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
        let eocd = (0..b.len().saturating_sub(21))
            .rev()
            .find(|&i| b[i..i + 4] == [0x50, 0x4b, 0x05, 0x06])
            .ok_or("not a zip (no end-of-central-directory)")?;
        let n_entries = u16at(eocd + 10);
        let mut cd = u32at(eocd + 16);
        let mut out = HashMap::new();
        for _ in 0..n_entries {
            if b[cd..cd + 4] != [0x50, 0x4b, 0x01, 0x02] {
                return Err("bad central directory".into());
            }
            let method = u16at(cd + 10);
            let csize = u32at(cd + 20);
            let name_len = u16at(cd + 28);
            let extra_len = u16at(cd + 30);
            let comment_len = u16at(cd + 32);
            let local = u32at(cd + 42);
            let name = String::from_utf8_lossy(&b[cd + 46..cd + 46 + name_len]).to_string();
            cd += 46 + name_len + extra_len + comment_len;
            let data_start = local + 30 + u16at(local + 26) + u16at(local + 28);
            let raw = &b[data_start..data_start + csize];
            let npy_bytes = match method {
                0 => raw.to_vec(),
                8 => {
                    let mut v = Vec::new();
                    flate2::read::DeflateDecoder::new(raw).read_to_end(&mut v).map_err(|e| e.to_string())?;
                    v
                }
                m => return Err(format!("zip method {m} unsupported")),
            };
            out.insert(name.trim_end_matches(".npy").to_string(), parse_npy(&npy_bytes)?);
        }
        Ok(Npz(out))
    }
}

fn parse_npy(b: &[u8]) -> Result<Npy, String> {
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        return Err("bad npy magic".into());
    }
    let (hlen, hstart) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[hstart..hstart + hlen]).map_err(|e| e.to_string())?;
    let descr = header.split("'descr':").nth(1).and_then(|s| s.split('\'').nth(1)).ok_or("npy: no descr")?;
    if header.contains("'fortran_order': True") {
        return Err("npy: fortran order unsupported".into());
    }
    let shape_txt = header
        .split("'shape':")
        .nth(1)
        .and_then(|s| s.split('(').nth(1))
        .and_then(|s| s.split(')').next())
        .ok_or("npy: no shape")?;
    let shape: Vec<usize> = shape_txt.split(',').filter_map(|t| t.trim().parse().ok()).collect();
    let n: usize = shape.iter().product();
    let d = &b[hstart + hlen..];
    let data: Vec<f64> = match descr {
        "<f4" => d.chunks_exact(4).take(n).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
        "<f8" => d.chunks_exact(8).take(n).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        "<i8" => d.chunks_exact(8).take(n).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
        "<i4" => d.chunks_exact(4).take(n).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64).collect(),
        other => return Err(format!("npy dtype {other} unsupported")),
    };
    if data.len() != n {
        return Err("npy: truncated".into());
    }
    Ok(Npy { shape, data })
}
