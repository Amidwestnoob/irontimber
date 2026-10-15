//! Prism "normalized-sylvester-walsh-hadamard" activation transform.
//! Folded weights expect x' = H_b · (s ⊙ P x) per block of b (=1024) features, where P is
//! the optional GDN grouped-head permutation and s the explicit ±1 sign vector for the width.
//! The latent token embedding needs the inverse: h = s ⊙ (H_b z).

/// In-place normalised Sylvester-Walsh-Hadamard transform on each `block`-sized chunk.
pub fn fwht_blocks(x: &mut [f32], block: usize) {
    assert!(block.is_power_of_two() && x.len() % block == 0);
    let scale = 1.0 / (block as f32).sqrt();
    for c in x.chunks_exact_mut(block) {
        let mut h = 1;
        while h < block {
            for i in (0..block).step_by(h * 2) {
                for j in i..i + h { let a = c[j]; let b = c[j + h]; c[j] = a + b; c[j + h] = a - b; }
            }
            h *= 2;
        }
        for v in c.iter_mut() { *v *= scale; }
    }
}

/// tiled [hd, nk, rep] -> grouped [hd, rep, nk] (ggml_permute(x,0,2,1,3) + cont)
pub fn gdn_group_perm(x: &[f32], hd: usize, nk: usize, rep: usize) -> Vec<f32> {
    let mut o = vec![0f32; x.len()];
    for k in 0..nk { for r in 0..rep { for i in 0..hd {
        o[i + hd * (r + rep * k)] = x[i + hd * (k + nk * r)];
    } } }
    o
}

#[derive(Clone, Debug)]
pub struct Hadamard { pub block: usize, pub signs: std::collections::HashMap<usize, Vec<f32>> }

impl Hadamard {
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut y = x.to_vec();
        if let Some(s) = self.signs.get(&x.len()) { for (v, s) in y.iter_mut().zip(s) { *v *= s; } }
        fwht_blocks(&mut y, self.block);
        y
    }
    pub fn inverse(&self, z: &[f32]) -> Vec<f32> {
        let mut y = z.to_vec();
        fwht_blocks(&mut y, self.block);
        if let Some(s) = self.signs.get(&z.len()) { for (v, s) in y.iter_mut().zip(s) { *v *= s; } }
        y
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fwht_matches_explicit_matrix() {
        let n = 64; let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut y = x.clone(); fwht_blocks(&mut y, n);
        for r in 0..n {
            let mut s = 0f64;
            for c in 0..n { let sg = if (r & c).count_ones() % 2 == 1 { -1.0 } else { 1.0 }; s += sg * x[c] as f64; }
            assert!((s / (n as f64).sqrt() - y[r] as f64).abs() < 1e-5);
        }
    }
    #[test]
    fn inverse_roundtrip() {
        let mut signs = std::collections::HashMap::new();
        signs.insert(256usize, (0..256).map(|i| if i % 3 == 0 { -1.0 } else { 1.0 }).collect());
        let h = Hadamard { block: 128, signs };
        let x: Vec<f32> = (0..256).map(|i| i as f32 - 100.0).collect();
        let r = h.inverse(&h.forward(&x));
        for i in 0..256 { assert!((r[i] - x[i]).abs() < 1e-3); }
    }
}
