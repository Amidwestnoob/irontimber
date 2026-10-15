//! Minimal memory-mapped GGUF v3 reader.
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U8(u8), I8(i8), U16(u16), I16(i16), U32(u32), I32(i32), F32(f32), Bool(bool),
    Str(String), Arr(Vec<Value>), U64(u64), I64(i64), F64(f64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            Value::U8(v) => v as u64, Value::I8(v) => v as u64, Value::U16(v) => v as u64,
            Value::I16(v) => v as u64, Value::U32(v) => v as u64, Value::I32(v) => v as u64,
            Value::U64(v) => v, Value::I64(v) => v as u64, Value::Bool(b) => b as u64,
            _ => return None,
        })
    }
    pub fn as_i64(&self) -> Option<i64> {
        Some(match *self {
            Value::I8(v) => v as i64, Value::I16(v) => v as i64, Value::I32(v) => v as i64,
            Value::I64(v) => v, _ => return self.as_u64().map(|v| v as i64),
        })
    }
    pub fn as_f32(&self) -> Option<f32> {
        match *self { Value::F32(v) => Some(v), Value::F64(v) => Some(v as f32), _ => self.as_i64().map(|v| v as f32) }
    }
    pub fn as_str(&self) -> Option<&str> { if let Value::Str(s) = self { Some(s) } else { None } }
    pub fn as_arr(&self) -> Option<&[Value]> { if let Value::Arr(a) = self { Some(a) } else { None } }
    pub fn as_bool(&self) -> Option<bool> { if let Value::Bool(b) = *self { Some(b) } else { None } }
}

/// ggml tensor types we understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GgmlType { F32, F16, BF16, PQ2_0, PTQ1_0, Q8_0, Q4_K, Q6_K, Other(u32) }

impl GgmlType {
    pub fn from_u32(t: u32) -> Self {
        match t { 0 => Self::F32, 1 => Self::F16, 30 => Self::BF16, 142 => Self::PQ2_0, 143 => Self::PTQ1_0, 8 => Self::Q8_0, 12 => Self::Q4_K, 14 => Self::Q6_K, o => Self::Other(o) }
    }
    /// (elements per block, bytes per block)
    pub fn block(&self) -> (usize, usize) {
        match self {
            Self::F32 => (1, 4), Self::F16 | Self::BF16 => (1, 2),
            Self::PQ2_0 => (128, 34), Self::PTQ1_0 => (128, 28), Self::Q8_0 => (32, 34), Self::Q4_K => (256, 144), Self::Q6_K => (256, 210),
            Self::Other(t) => panic!("unsupported ggml type {t}"),
        }
    }
    pub fn to_u32(&self) -> u32 { match *self { Self::F32 => 0, Self::F16 => 1, Self::BF16 => 30, Self::PQ2_0 => 142, Self::PTQ1_0 => 143, Self::Q8_0 => 8, Self::Q4_K => 12, Self::Q6_K => 14, Self::Other(t) => t } }
    pub fn row_bytes(&self, n: usize) -> usize { let (b, s) = self.block(); assert!(n % b == 0); n / b * s }
}

#[derive(Debug, Clone)]
pub struct TensorInfo { pub name: String, pub dims: Vec<usize>, pub ty: GgmlType, pub offset: usize }

impl TensorInfo {
    pub fn nelem(&self) -> usize { self.dims.iter().product() }
    pub fn nbytes(&self) -> usize { self.ty.row_bytes(self.dims[0]) * self.dims[1..].iter().product::<usize>() }
}

pub struct Gguf {
    pub mmap: Mmap,
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
    pub tensor_order: Vec<String>,
    pub data_offset: usize,
    /// KV pairs in file order as raw (type tag + value) bytes, for lossless rewriting.
    pub kv_raw: Vec<(String, Vec<u8>)>,
}

struct Cur<'a> { b: &'a [u8], p: usize }
impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        if n > self.b.len().saturating_sub(self.p) { return Err("gguf: unexpected EOF".into()); }
        let s = &self.b[self.p..self.p + n]; self.p += n; Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> { Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap())) }
    fn u64(&mut self) -> Result<u64, String> { Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap())) }
    fn str(&mut self) -> Result<String, String> { let n = self.u64()? as usize; Ok(String::from_utf8_lossy(self.take(n)?).into_owned()) }
    fn val(&mut self, t: u32) -> Result<Value, String> {
        Ok(match t {
            0 => Value::U8(self.take(1)?[0]), 1 => Value::I8(self.take(1)?[0] as i8),
            2 => Value::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => Value::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => Value::U32(self.u32()?), 5 => Value::I32(self.u32()? as i32),
            6 => Value::F32(f32::from_bits(self.u32()?)), 7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            9 => { let et = self.u32()?; let n = self.u64()? as usize; if et == 9 || n > self.b.len().saturating_sub(self.p) { return Err("invalid GGUF array length/type".into()); } let mut v = Vec::with_capacity(n); for _ in 0..n { v.push(self.val(et)?); } Value::Arr(v) }
            10 => Value::U64(self.u64()?), 11 => Value::I64(self.u64()? as i64), 12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(format!("gguf: bad value type {t}")),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let f = File::open(path.as_ref()).map_err(|e| format!("open {}: {e}", path.as_ref().display()))?;
        let mmap = unsafe { Mmap::map(&f) }.map_err(|e| e.to_string())?;
        let mut c = Cur { b: &mmap, p: 0 };
        if c.take(4)? != b"GGUF" { return Err("not a GGUF file".into()); }
        let ver = c.u32()?;
        if ver < 2 { return Err(format!("unsupported gguf version {ver}")); }
        let nt = c.u64()? as usize; let nkv = c.u64()? as usize;
        let mut kv = HashMap::new();
        let mut kv_raw = Vec::new();
        for _ in 0..nkv { let k = c.str()?; let s0 = c.p; let t = c.u32()?; let v = c.val(t)?; kv_raw.push((k.clone(), c.b[s0..c.p].to_vec())); if kv.insert(k, v).is_some() { return Err("duplicate metadata key".into()); } }
        let mut tensors = HashMap::new(); let mut order = Vec::new();
        for _ in 0..nt {
            let name = c.str()?; let nd = c.u32()? as usize;
            if !(1..=4).contains(&nd) { return Err("invalid tensor rank".into()); }
            let mut dims = Vec::new(); for _ in 0..nd { dims.push(c.u64()? as usize); }
            if dims.iter().any(|&d| d == 0) || dims.iter().try_fold(1usize, |a,&b| a.checked_mul(b)).is_none() { return Err("invalid tensor dimensions".into()); }
            let ty = GgmlType::from_u32(c.u32()?); let offset = c.u64()? as usize;
            order.push(name.clone());
            if tensors.insert(name.clone(), TensorInfo { name, dims, ty, offset }).is_some() { return Err("duplicate tensor name".into()); }
        }
        let align = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        if align == 0 || !align.is_power_of_two() || align > 65536 { return Err("invalid GGUF alignment".into()); }
        let data_offset = c.p.div_ceil(align) * align;
        let g = Gguf { mmap, kv, tensors, tensor_order: order, data_offset, kv_raw };
        for t in g.tensors.values() {
            if let GgmlType::Other(_) = t.ty { continue; }
            let (block, bytes) = t.ty.block();
            if t.dims[0] % block != 0 { return Err(format!("tensor {} quantization alignment", t.name)); }
            let end = (t.nelem()/block).checked_mul(bytes).and_then(|n| n.checked_add(t.offset)).and_then(|n| n.checked_add(g.data_offset));
            if end.map_or(true, |n| n > g.mmap.len()) { return Err(format!("tensor {} out of bounds", t.name)); }
        }
        Ok(g)
    }
    pub fn get(&self, k: &str) -> Option<&Value> { self.kv.get(k) }
    pub fn u(&self, k: &str) -> Result<usize, String> { self.get(k).and_then(|v| v.as_u64()).map(|v| v as usize).ok_or_else(|| format!("missing key {k}")) }
    pub fn f(&self, k: &str) -> Result<f32, String> { self.get(k).and_then(|v| v.as_f32()).ok_or_else(|| format!("missing key {k}")) }
    pub fn tensor(&self, name: &str) -> Result<&TensorInfo, String> { self.tensors.get(name).ok_or_else(|| format!("missing tensor {name}")) }
    pub fn data(&self, t: &TensorInfo) -> &[u8] { let s = self.data_offset + t.offset; &self.mmap[s..s + t.nbytes()] }
    pub fn tensor_data(&self, name: &str) -> Result<(&TensorInfo, &[u8]), String> { let t = self.tensor(name)?; Ok((t, self.data(t))) }
}

/// Raw KV value bytes for a u32 (type tag 4).
pub fn raw_u32(v: u32) -> Vec<u8> { let mut b = 4u32.to_le_bytes().to_vec(); b.extend_from_slice(&v.to_le_bytes()); b }

/// Write a GGUF v3 file: `kvs` are (key, raw type+value bytes); tensors are copied verbatim from their source files
/// (optionally renamed). Data is aligned to `align` (from general.alignment, default 32).
pub fn write_gguf(path: impl AsRef<Path>, kvs: &[(String, Vec<u8>)], tensors: &[(&Gguf, &TensorInfo, String)]) -> Result<u64, String> {
    use std::io::{BufWriter, Write};
    let align = 32usize;
    if kvs.iter().any(|(k, v)| k == "general.alignment" && v != &raw_u32(32)) { return Err("only general.alignment=32 supported".into()); }
    // Refuse to truncate an input, existing model or earlier conversion receipt.
    let f = std::fs::OpenOptions::new().write(true).create_new(true).open(path.as_ref()).map_err(|e| e.to_string())?;
    let mut w = BufWriter::with_capacity(1 << 24, f);
    let mut hdr: Vec<u8> = Vec::new();
    let ps = |h: &mut Vec<u8>, s: &str| { h.extend_from_slice(&(s.len() as u64).to_le_bytes()); h.extend_from_slice(s.as_bytes()); };
    hdr.extend_from_slice(b"GGUF"); hdr.extend_from_slice(&3u32.to_le_bytes());
    hdr.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); hdr.extend_from_slice(&(kvs.len() as u64).to_le_bytes());
    for (k, v) in kvs { ps(&mut hdr, k); hdr.extend_from_slice(v); }
    let mut off = 0usize; let mut offs = Vec::new();
    for (_, t, name) in tensors {
        ps(&mut hdr, name); hdr.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims { hdr.extend_from_slice(&(*d as u64).to_le_bytes()); }
        hdr.extend_from_slice(&t.ty.to_u32().to_le_bytes()); hdr.extend_from_slice(&(off as u64).to_le_bytes());
        offs.push(off); off = (off + t.nbytes()).div_ceil(align) * align;
    }
    hdr.resize(hdr.len().div_ceil(align) * align, 0);
    w.write_all(&hdr).map_err(|e| e.to_string())?;
    let mut pos = 0usize;
    for ((g, t, _), o) in tensors.iter().zip(offs) {
        if pos < o { w.write_all(&vec![0u8; o - pos]).map_err(|e| e.to_string())?; pos = o; }
        let d = g.data(t); w.write_all(d).map_err(|e| e.to_string())?; pos += d.len();
    }
    if pos < off { w.write_all(&vec![0u8; off - pos]).map_err(|e| e.to_string())?; }
    w.flush().map_err(|e| e.to_string())?;
    Ok((hdr.len() + off) as u64)
}
