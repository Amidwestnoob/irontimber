//! Quant formats: PQ2_0 / PTQ1_0 (Prism, group 128), BF16/F16, and ggml's Q8_K
//! activation quantisation (used so the CPU reference reproduces ggml-cpu numerics).
use half::{bf16, f16};

pub const QK: usize = 128; // PQ2_0 / PTQ1_0 group
pub const PQ2_BYTES: usize = 34;
pub const PTQ1_BYTES: usize = 28;
pub const QK_K: usize = 256;

#[inline] pub fn f16_to_f32(lo: u8, hi: u8) -> f32 { f16::from_bits(u16::from_le_bytes([lo, hi])).to_f32() }
#[inline] pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 { bf16::from_bits(u16::from_le_bytes([lo, hi])).to_f32() }
/// ggml's fp32->bf16 rounding (round-to-nearest-even, NaN quieted).
#[inline] pub fn f32_to_bf16_round(x: f32) -> f32 {
    let u = x.to_bits();
    if (u & 0x7fffffff) > 0x7f800000 { return f32::from_bits((u | 0x00400000) & 0xffff0000); }
    let r = (u + (0x7fff + ((u >> 16) & 1))) >> 16;
    f32::from_bits(r << 16)
}

/// 2-bit code c in {0,1,2,3} -> value c-1. Both PQ2_0 and PTQ1_0 decode into this code space.
#[inline] pub fn pq2_code(qs: &[u8], j: usize) -> u8 { (qs[j / 4] >> ((j % 4) * 2)) & 3 }

/// Decode one PQ2_0 block into codes (value = code - 1) and its scale.
pub fn pq2_block_codes(b: &[u8], codes: &mut [u8; QK]) -> f32 {
    let d = f16_to_f32(b[0], b[1]);
    let qs = &b[2..34];
    for j in 0..QK { codes[j] = pq2_code(qs, j); }
    d
}

const PTQ_STAGES: [usize; 3] = [32, 16, 8];
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
#[inline] fn trit(byte: u8, n: usize) -> u8 { let q = byte.wrapping_mul(POW3[n]); ((q as u16 * 3) >> 8) as u8 }

/// Decode one PTQ1_0 block (qs[24], qh[2], d f16) into codes (value = code - 1), exactly
/// following ggml dequantize_row_ptq1_0's element order.
pub fn ptq1_block_codes(b: &[u8], codes: &mut [u8; QK]) -> f32 {
    let qs = &b[0..24]; let qh = &b[24..26]; let d = f16_to_f32(b[26], b[27]);
    let mut o = 0; let mut j = 0;
    for &c in PTQ_STAGES.iter() {
        while j + c <= qs.len() {
            for n in 0..5 { for m in 0..c { codes[o] = trit(qs[j + m], n); o += 1; } }
            j += c;
        }
    }
    for n in 0..4 { for h in 0..2 { codes[o] = trit(qh[h], n); o += 1; } }
    debug_assert_eq!(o, QK);
    d
}

pub fn dequant_pq2_0(src: &[u8], out: &mut [f32]) {
    let mut c = [0u8; QK];
    for (i, b) in src.chunks_exact(PQ2_BYTES).enumerate() {
        let d = pq2_block_codes(b, &mut c);
        for j in 0..QK { out[i * QK + j] = (c[j] as i32 - 1) as f32 * d; }
    }
}
pub fn dequant_ptq1_0(src: &[u8], out: &mut [f32]) {
    let mut c = [0u8; QK];
    for (i, b) in src.chunks_exact(PTQ1_BYTES).enumerate() {
        let d = ptq1_block_codes(b, &mut c);
        for j in 0..QK { out[i * QK + j] = (c[j] as i32 - 1) as f32 * d; }
    }
}

/// ggml nearest_int: round half to even.
#[inline] pub fn nearest_int(x: f32) -> i32 { x.round_ties_even() as i32 }

/// ggml block_q8_K (d f32, qs[256] i8). bsums are recomputed where needed.
#[derive(Clone)]
pub struct Q8K { pub d: Vec<f32>, pub qs: Vec<i8>, /// per-128 sums of qs
    pub sum128: Vec<i32> }

/// quantize_row_q8_K_ref
pub fn quantize_q8k(x: &[f32]) -> Q8K {
    assert!(x.len() % QK_K == 0);
    let nb = x.len() / QK_K;
    let mut d = vec![0f32; nb]; let mut qs = vec![0i8; x.len()];
    for i in 0..nb {
        let xb = &x[i * QK_K..(i + 1) * QK_K];
        let (mut amax, mut max) = (0f32, 0f32);
        for &v in xb { if v.abs() > amax { amax = v.abs(); max = v; } }
        if amax == 0.0 { continue; }
        let iscale = -127.0f32 / max;
        for j in 0..QK_K { qs[i * QK_K + j] = nearest_int(iscale * xb[j]).min(127) as i8; }
        d[i] = 1.0 / iscale;
    }
    let sum128 = qs.chunks_exact(QK).map(|c| c.iter().map(|&v| v as i32).sum()).collect();
    Q8K { d, qs, sum128 }
}

/// Repacked 2-bit ternary row-major matrix: code words (16 codes/u32, element j of a
/// row at bit 2*(j%16) of word j/16) plus one f32 scale per 128 elements. This is the
/// single layout both the CPU reference and the Vulkan kernels consume; PQ2_0 and
/// PTQ1_0 both repack into it losslessly.
#[derive(Clone)]
pub struct Packed2 { pub rows: usize, pub cols: usize, pub words: Vec<u32>, pub scales: Vec<f32> }

impl Packed2 {
    pub fn words_per_row(&self) -> usize { self.cols / 16 }
    pub fn blocks_per_row(&self) -> usize { self.cols / QK }
    pub fn from_ggml(ty: crate::gguf::GgmlType, data: &[u8], rows: usize, cols: usize) -> Self {
        use crate::gguf::GgmlType::*;
        assert!(cols % QK == 0);
        let (bs, f): (usize, fn(&[u8], &mut [u8; QK]) -> f32) = match ty {
            PQ2_0 => (PQ2_BYTES, pq2_block_codes), PTQ1_0 => (PTQ1_BYTES, ptq1_block_codes),
            t => panic!("Packed2 from {t:?}"),
        };
        let nblk = rows * cols / QK;
        let mut words = vec![0u32; rows * cols / 16];
        let mut scales = vec![0f32; nblk];
        let mut c = [0u8; QK];
        for (i, b) in data.chunks_exact(bs).take(nblk).enumerate() {
            scales[i] = f(b, &mut c);
            for w in 0..8 {
                let mut v = 0u32;
                for k in 0..16 { v |= (c[w * 16 + k] as u32) << (2 * k); }
                words[i * 8 + w] = v;
            }
        }
        Packed2 { rows, cols, words, scales }
    }
    pub fn row_f32(&self, r: usize, out: &mut [f32]) {
        let wpr = self.words_per_row();
        for j in 0..self.cols {
            let c = (self.words[r * wpr + j / 16] >> (2 * (j % 16))) & 3;
            out[j] = (c as i32 - 1) as f32 * self.scales[r * self.blocks_per_row() + j / QK];
        }
    }
    /// ggml-exact dot of row r with a Q8_K-quantised activation (ggml_vec_dot_pq2_0_q8_K_generic order).
    #[inline]
    pub fn dot_q8k(&self, r: usize, y: &Q8K) -> f32 {
        let wpr = self.words_per_row(); let bpr = self.blocks_per_row();
        let w = &self.words[r * wpr..(r + 1) * wpr];
        let mut sumf = 0f32;
        for b in 0..bpr {
            let q8 = &y.qs[b * QK..(b + 1) * QK];
            let mut s: i32 = 0;
            for k in 0..8 {
                let word = w[b * 8 + k];
                let q = &q8[k * 16..k * 16 + 16];
                for e in 0..16 { s += ((word >> (2 * e)) & 3) as i32 * q[e] as i32; }
            }
            let sumi = s - y.sum128[b];
            sumf += (self.scales[r * bpr + b] * y.d[b / 2]) * sumi as f32;
        }
        sumf
    }
    /// plain f32 dot (used by the f32-activation mode and by GPU parity tests)
    pub fn dot_f32(&self, r: usize, x: &[f32]) -> f32 {
        let wpr = self.words_per_row(); let bpr = self.blocks_per_row();
        let w = &self.words[r * wpr..(r + 1) * wpr];
        let mut sum = 0f32;
        for b in 0..bpr {
            let mut s = 0f32;
            for k in 0..8 { let word = w[b * 8 + k]; let xb = &x[b * QK + k * 16..];
                for e in 0..16 { s += (((word >> (2 * e)) & 3) as f32 - 1.0) * xb[e]; } }
            sum += s * self.scales[r * bpr + b];
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pq2_codes_map() {
        let mut b = [0u8; 34];
        b[0..2].copy_from_slice(&f16::from_f32(0.5).to_bits().to_le_bytes());
        b[2] = 0b11_10_01_00; // elements 0..3 = codes 0,1,2,3 -> -1,0,1,2
        let mut o = [0f32; 128];
        dequant_pq2_0(&b, &mut o);
        assert_eq!(&o[0..4], &[-0.5, 0.0, 0.5, 1.0]);
        assert!(o[4..].iter().all(|&v| v == -0.5));
    }
    #[test]
    fn q8k_roundtrip_bounds() {
        let x: Vec<f32> = (0..512).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
        let q = quantize_q8k(&x);
        for i in 0..512 { assert!((q.qs[i] as f32 * q.d[i / 256] - x[i]).abs() <= q.d[i / 256].abs() * 0.5 + 1e-6); }
    }
    #[test]
    fn bf16_round() { assert_eq!(f32_to_bf16_round(1.0), 1.0); assert_eq!(f32_to_bf16_round(1.00390625), 1.0); assert_eq!(f32_to_bf16_round(1.01171875), 1.015625); }
}

// ---------------- ggml k-quants used by the grafted MTP head (Q8_0, Q6_K, Q4_K) ----------------
/// ggml dequantize_row_q8_0: block 32 = d f16, qs[32] i8.
pub fn dequant_q8_0(src: &[u8], out: &mut [f32]) {
    for (i, b) in src.chunks_exact(34).enumerate() { let d = f16_to_f32(b[0], b[1]); for j in 0..32 { out[i * 32 + j] = (b[2 + j] as i8) as f32 * d; } }
}
/// ggml dequantize_row_q6_K: block 256 = ql[128], qh[64], scales[16] i8, d f16.
pub fn dequant_q6_k(src: &[u8], out: &mut [f32]) {
    for (i, b) in src.chunks_exact(210).enumerate() {
        let d = f16_to_f32(b[208], b[209]); let y = &mut out[i * 256..(i + 1) * 256];
        for n in 0..2 {
            let ql = &b[n * 64..]; let qh = &b[128 + n * 32..]; let sc = &b[192 + n * 8..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;
                y[n * 128 + l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[n * 128 + l + 32] = d * (sc[is + 2] as i8) as f32 * q2 as f32;
                y[n * 128 + l + 64] = d * (sc[is + 4] as i8) as f32 * q3 as f32;
                y[n * 128 + l + 96] = d * (sc[is + 6] as i8) as f32 * q4 as f32;
            }
        }
    }
}
fn scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 { (q[j] & 63, q[j + 4] & 63) } else { ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)) }
}
/// ggml dequantize_row_q4_K: block 256 = d f16, dmin f16, scales[12], qs[128].
pub fn dequant_q4_k(src: &[u8], out: &mut [f32]) {
    for (i, b) in src.chunks_exact(144).enumerate() {
        let d = f16_to_f32(b[0], b[1]); let mn = f16_to_f32(b[2], b[3]); let sc = &b[4..16]; let q = &b[16..144];
        let y = &mut out[i * 256..(i + 1) * 256];
        for (jj, is) in (0..4).map(|k| (k, 2 * k)) {
            let (s1, m1) = scale_min_k4(is, sc); let (s2, m2) = scale_min_k4(is + 1, sc);
            let (d1, mm1, d2, mm2) = (d * s1 as f32, mn * m1 as f32, d * s2 as f32, mn * m2 as f32);
            let qq = &q[jj * 32..jj * 32 + 32];
            for l in 0..32 { y[jj * 64 + l] = d1 * (qq[l] & 0xF) as f32 - mm1; }
            for l in 0..32 { y[jj * 64 + 32 + l] = d2 * (qq[l] >> 4) as f32 - mm2; }
        }
    }
}

/// GPU/CPU layout for the MTP head's int-quant matrices (Q8_0 and Q6_K repack into it losslessly):
/// row-major int8 values packed 4 per u32 (element j of a row at byte j%4 of word j/4) and one f32
/// scale per 16 elements. value = scale * q, with scale = d (Q8_0) or d*sc (Q6_K, exact in f32).
#[derive(Clone)]
pub struct Q8G16 { pub rows: usize, pub cols: usize, pub words: Vec<u32>, pub scales: Vec<f32> }
impl Q8G16 {
    pub fn from_ggml(ty: crate::gguf::GgmlType, data: &[u8], rows: usize, cols: usize) -> Self {
        use crate::gguf::GgmlType::*;
        assert!(cols % 256 == 0 || (ty == Q8_0 && cols % 32 == 0));
        let n = rows * cols; let mut q = vec![0i8; n]; let mut scales = vec![0f32; n / 16];
        match ty {
            Q8_0 => for (i, b) in data.chunks_exact(34).take(n / 32).enumerate() {
                let d = f16_to_f32(b[0], b[1]); scales[2 * i] = d; scales[2 * i + 1] = d;
                for j in 0..32 { q[i * 32 + j] = b[2 + j] as i8; }
            },
            Q6_K => for (i, b) in data.chunks_exact(210).take(n / 256).enumerate() {
                let d = f16_to_f32(b[208], b[209]);
                for g in 0..16 { scales[i * 16 + g] = d * (b[192 + g] as i8) as f32; }
                for nn in 0..2 {
                    let ql = &b[nn * 64..]; let qh = &b[128 + nn * 32..]; let o = i * 256 + nn * 128;
                    for l in 0..32 {
                        q[o + l] = (((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i32 - 32) as i8;
                        q[o + l + 32] = (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32) as i8;
                        q[o + l + 64] = (((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32) as i8;
                        q[o + l + 96] = (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32) as i8;
                    }
                }
            },
            t => panic!("Q8G16 from {t:?}"),
        }
        let words = q.chunks_exact(4).map(|c| u32::from_le_bytes([c[0] as u8, c[1] as u8, c[2] as u8, c[3] as u8])).collect();
        Q8G16 { rows, cols, words, scales }
    }
    #[inline] pub fn q(&self, i: usize) -> i8 { (self.words[i / 4] >> (8 * (i % 4))) as u8 as i8 }
    pub fn row_f32(&self, r: usize, out: &mut [f32]) { for j in 0..self.cols { let i = r * self.cols + j; out[j] = self.scales[i / 16] * self.q(i) as f32; } }
    /// f32 reference dot (per-16 group sums, then scaled) - same grouping as the GPU kernel.
    pub fn dot(&self, r: usize, x: &[f32]) -> f32 {
        let mut s = 0f32;
        for g in 0..self.cols / 16 { let b = r * self.cols + g * 16; let mut t = 0f32; for j in 0..16 { t += self.q(b + j) as f32 * x[g * 16 + j]; } s += t * self.scales[b / 16]; }
        s
    }
}
