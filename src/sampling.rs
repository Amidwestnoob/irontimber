//! Token sampling with llama.cpp-style semantics and order:
//! repeat penalty -> top-k -> top-p -> min-p -> temperature -> seeded draw.
//! `temperature <= 0` (or `top_k == 1`) is exact greedy (argmax, lowest index on ties).
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplerParams {
    pub temperature: f32,
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    /// u64::MAX (or -1 in JSON) = random seed.
    pub seed: u64,
}
impl Default for SamplerParams {
    /// llama-server defaults.
    fn default() -> Self { Self { temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1.0, repeat_last_n: 64, seed: u64::MAX } }
}
impl SamplerParams {
    pub fn greedy() -> Self { Self { temperature: 0.0, top_k: 1, top_p: 1.0, min_p: 0.0, repeat_penalty: 1.0, repeat_last_n: 0, seed: 0 } }
    pub fn is_greedy(&self) -> bool { (self.temperature <= 0.0 || self.top_k == 1) && self.repeat_penalty == 1.0 }
    /// Read llama-server request fields (missing ones keep the current value).
    pub fn apply_json(&mut self, j: &serde_json::Value) {
        let f = |k: &str| j.get(k).and_then(|v| v.as_f64());
        if let Some(v) = f("temperature") { self.temperature = v as f32; }
        if let Some(v) = f("top_k") { self.top_k = v as i32; }
        if let Some(v) = f("top_p") { self.top_p = v as f32; }
        if let Some(v) = f("min_p") { self.min_p = v as f32; }
        if let Some(v) = f("repeat_penalty") { self.repeat_penalty = v as f32; }
        if let Some(v) = f("repeat_last_n") { self.repeat_last_n = v as i32; }
        if let Some(v) = j.get("seed") { if let Some(s) = v.as_u64() { self.seed = s; } else if v.as_i64().map_or(false, |s| s < 0) { self.seed = u64::MAX; } }
    }
}

/// splitmix64: tiny, seedable, identical on every platform.
#[derive(Clone, Debug)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Rng(seed) }
    pub fn next_u64(&mut self) -> u64 { self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15); let mut z = self.0; z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9); z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb); z ^ (z >> 31) }
    /// Uniform in [0, 1).
    pub fn f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
}

pub struct Sampler { pub p: SamplerParams, rng: Rng, pub seed_used: u64 }

pub fn argmax(v: &[f32]) -> u32 { let mut b = 0; for i in 1..v.len() { if v[i] > v[b] { b = i; } } b as u32 }

impl Sampler {
    pub fn new(p: SamplerParams) -> Self {
        let seed = if p.seed == u64::MAX { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) } else { p.seed };
        Sampler { p, rng: Rng::new(seed), seed_used: seed }
    }
    /// Pick the next token from `logits`; `history` = prompt + generated ids (for the repeat penalty).
    pub fn sample(&mut self, logits: &[f32], history: &[u32]) -> u32 {
        let p = &self.p;
        let mut l: Vec<f32>;
        let lg: &[f32] = if p.repeat_penalty != 1.0 && p.repeat_last_n != 0 {
            l = logits.to_vec();
            let n = if p.repeat_last_n < 0 { history.len() } else { (p.repeat_last_n as usize).min(history.len()) };
            let mut seen = std::collections::HashSet::new();
            for &t in &history[history.len() - n..] {
                if (t as usize) < l.len() && seen.insert(t) { let x = &mut l[t as usize]; *x = if *x <= 0.0 { *x * p.repeat_penalty } else { *x / p.repeat_penalty }; }
            }
            &l
        } else { logits };
        if p.temperature <= 0.0 || p.top_k == 1 { return argmax(lg); }
        // candidates sorted by logit desc (ties: lower id first), truncated to top_k
        let k = if p.top_k <= 0 { lg.len() } else { (p.top_k as usize).min(lg.len()) };
        let mut c: Vec<(u32, f32)> = lg.iter().enumerate().map(|(i, &v)| (i as u32, v)).collect();
        let cmp = |a: &(u32, f32), b: &(u32, f32)| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0));
        if k < c.len() { c.select_nth_unstable_by(k - 1, cmp); c.truncate(k); }
        c.sort_by(cmp);
        // softmax at temperature 1 for top-p / min-p (llama.cpp applies temperature last)
        let probs = |c: &[(u32, f32)], t: f32| -> Vec<f64> { let m = c[0].1 as f64; let e: Vec<f64> = c.iter().map(|x| ((x.1 as f64 - m) / t as f64).exp()).collect(); let s: f64 = e.iter().sum(); e.into_iter().map(|x| x / s).collect() };
        if p.top_p < 1.0 {
            let pr = probs(&c, 1.0); let mut cum = 0.0; let mut keep = c.len();
            for (i, q) in pr.iter().enumerate() { cum += q; if cum >= p.top_p as f64 { keep = i + 1; break; } }
            c.truncate(keep.max(1));
        }
        if p.min_p > 0.0 {
            let pr = probs(&c, 1.0); let thr = pr[0] * p.min_p as f64;
            let keep = pr.iter().take_while(|&&q| q >= thr).count();
            c.truncate(keep.max(1));
        }
        let pr = probs(&c, p.temperature);
        let r = self.rng.f64(); let mut cum = 0.0;
        for (i, q) in pr.iter().enumerate() { cum += q; if r < cum { return c[i].0; } }
        c[c.len() - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn logits() -> Vec<f32> { (0..1000).map(|i| ((i * 7919 % 1000) as f32 / 100.0) - 5.0).collect() }
    #[test] fn greedy_is_argmax() {
        let l = logits(); let mut s = Sampler::new(SamplerParams::greedy());
        assert_eq!(s.sample(&l, &[]), argmax(&l));
        let mut p = SamplerParams::default(); p.top_k = 1; p.seed = 3; assert_eq!(Sampler::new(p).sample(&l, &[]), argmax(&l));
        let mut p = SamplerParams::default(); p.temperature = 0.0; assert_eq!(Sampler::new(p).sample(&l, &[]), argmax(&l));
    }
    #[test] fn seeded_is_deterministic() {
        let l = logits(); let mut p = SamplerParams::default(); p.seed = 42; p.temperature = 1.5; p.top_k = 0; p.top_p = 1.0; p.min_p = 0.0;
        let a: Vec<u32> = { let mut s = Sampler::new(p.clone()); (0..200).map(|_| s.sample(&l, &[])).collect() };
        let b: Vec<u32> = { let mut s = Sampler::new(p.clone()); (0..200).map(|_| s.sample(&l, &[])).collect() };
        assert_eq!(a, b); p.seed = 43;
        let c: Vec<u32> = { let mut s = Sampler::new(p); (0..200).map(|_| s.sample(&l, &[])).collect() };
        assert_ne!(a, c);
    }
    #[test] fn top_k_restricts() {
        let l = logits(); let mut p = SamplerParams::default(); p.seed = 1; p.top_k = 5; p.top_p = 1.0; p.min_p = 0.0; p.temperature = 5.0;
        let mut idx: Vec<u32> = (0..1000).collect(); idx.sort_by(|&a, &b| l[b as usize].partial_cmp(&l[a as usize]).unwrap().then(a.cmp(&b)));
        let top: std::collections::HashSet<u32> = idx[..5].iter().copied().collect();
        let mut s = Sampler::new(p); for _ in 0..500 { assert!(top.contains(&s.sample(&l, &[]))); }
    }
    #[test] fn top_p_and_min_p_restrict() {
        let mut l = vec![-20.0f32; 100]; l[3] = 2.0; l[7] = 1.0; l[9] = 0.0; // probs ~ .665 .245 .090
        let mut p = SamplerParams::default(); p.seed = 9; p.top_k = 0; p.min_p = 0.0; p.top_p = 0.8;
        let mut s = Sampler::new(p.clone()); for _ in 0..300 { let t = s.sample(&l, &[]); assert!(t == 3 || t == 7); }
        p.top_p = 1.0; p.min_p = 0.3; // keep p >= 0.3*0.665 = 0.2 -> {3, 7}
        let mut s = Sampler::new(p); for _ in 0..300 { let t = s.sample(&l, &[]); assert!(t == 3 || t == 7); }
    }
    #[test] fn frequencies_match_softmax() {
        let l = vec![1.0f32, 0.0, -1.0, 0.5]; let mut p = SamplerParams::default(); p.seed = 7; p.top_k = 0; p.top_p = 1.0; p.min_p = 0.0; p.temperature = 0.7;
        let mut s = Sampler::new(p); let n = 200_000; let mut cnt = [0usize; 4];
        for _ in 0..n { cnt[s.sample(&l, &[]) as usize] += 1; }
        let e: Vec<f64> = l.iter().map(|&x| (x as f64 / 0.7).exp()).collect(); let z: f64 = e.iter().sum();
        for i in 0..4 { let f = cnt[i] as f64 / n as f64; assert!((f - e[i] / z).abs() < 0.005, "token {i}: {f} vs {}", e[i] / z); }
    }
    #[test] fn repeat_penalty_applies() {
        let l = vec![2.0f32, 1.9, -1.0]; let mut p = SamplerParams::greedy(); p.repeat_penalty = 1.2; p.repeat_last_n = 64;
        let mut s = Sampler::new(p.clone()); assert_eq!(s.sample(&l, &[0]), 1); // 2.0/1.2 < 1.9
        assert_eq!(s.sample(&l, &[2]), 0);
        let l2 = vec![-1.0f32, -1.1]; assert_eq!(Sampler::new(p).sample(&l2, &[0]), 1); // negative logits are multiplied
    }
}
