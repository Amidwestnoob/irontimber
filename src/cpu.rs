//! CPU reference forward pass for qwen35 hybrid models, reading weights straight from the mmap.
//! `Numerics::Ggml` reproduces ggml-cpu's arithmetic (Q8_K activations for ternary matmuls,
//! bf16 activations for BF16 weights, f16 KV cache) so logits can be compared tightly with
//! llama-prism's CPU build. `Numerics::F32` is the exact-math reference used for GPU kernels.
use crate::gguf::{Gguf, GgmlType};
use crate::hadamard::{gdn_group_perm, Hadamard};
use crate::model::{load_hadamard, Config, WRef};
use crate::quant::{self, quantize_q8k, Q8K};
use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Numerics { Ggml, F32 }

pub fn rmsnorm(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let ss: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let s = 1.0 / (ss + eps).sqrt();
    x.iter().zip(w).map(|(a, b)| a * s * b).collect()
}
#[inline] pub fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
#[inline] pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
#[inline] pub fn softplus(x: f32) -> f32 { if x > 20.0 { x } else { (1.0 + x.exp()).ln() } }
#[inline] fn f16r(x: f32) -> f32 { half::f16::from_f32(x).to_f32() }

/// y = W x for a ternary / bf16 / f32 weight. Rows in parallel.
pub fn matvec(w: &WRef, x: &[f32], num: Numerics) -> Vec<f32> {
    assert_eq!(x.len(), w.cols, "matvec dim");
    let rb = w.row_bytes();
    match w.ty {
        GgmlType::PQ2_0 | GgmlType::PTQ1_0 => {
            let q: Option<Q8K> = if num == Numerics::Ggml { Some(quantize_q8k(x)) } else { None };
            (0..w.rows).into_par_iter().with_min_len(64).map(|r| {
                let row = &w.data[r * rb..(r + 1) * rb];
                ternary_row_dot(w.ty, row, x, q.as_ref())
            }).collect()
        }
        GgmlType::BF16 | GgmlType::F16 | GgmlType::F32 => {
            let xs: Vec<f32> = if num == Numerics::Ggml && w.ty == GgmlType::BF16 { x.iter().map(|&v| quant::f32_to_bf16_round(v)).collect() } else { x.to_vec() };
            (0..w.rows).into_par_iter().map(|r| {
                let row = w.row_f32(r);
                if num == Numerics::Ggml { let mut s = 0f64; for i in 0..row.len() { s += (row[i] * xs[i]) as f64; } s as f32 }
                else { row.iter().zip(&xs).map(|(a, b)| a * b).sum() }
            }).collect()
        }
        t => panic!("matvec {t:?}"),
    }
}

fn ternary_row_dot(ty: GgmlType, row: &[u8], x: &[f32], q: Option<&Q8K>) -> f32 {
    let (bs, dec): (usize, fn(&[u8], &mut [u8; 128]) -> f32) = match ty {
        GgmlType::PQ2_0 => (quant::PQ2_BYTES, quant::pq2_block_codes),
        _ => (quant::PTQ1_BYTES, quant::ptq1_block_codes),
    };
    let mut c = [0u8; 128];
    let mut sum = 0f32;
    for (b, blk) in row.chunks_exact(bs).enumerate() {
        let d = dec(blk, &mut c);
        match q {
            Some(q) => {
                let q8 = &q.qs[b * 128..b * 128 + 128];
                let mut s = 0i32;
                for j in 0..128 { s += c[j] as i32 * q8[j] as i32; }
                sum += (d * q.d[b / 2]) * (s - q.sum128[b]) as f32;
            }
            None => {
                let xb = &x[b * 128..b * 128 + 128];
                let mut s = 0f32;
                for j in 0..128 { s += (c[j] as f32 - 1.0) * xb[j]; }
                sum += s * d;
            }
        }
    }
    sum
}

pub struct Layer<'a> {
    pub attn_norm: Vec<f32>, pub post_norm: Vec<f32>,
    pub ffn_gate: WRef<'a>, pub ffn_up: WRef<'a>, pub ffn_down: WRef<'a>,
    pub kind: Kind<'a>,
}
pub enum Kind<'a> {
    Gdn { qkv: WRef<'a>, gate: WRef<'a>, conv: Vec<f32>, dt: Vec<f32>, a: Vec<f32>, beta: WRef<'a>, alpha: WRef<'a>, norm: Vec<f32>, out: WRef<'a> },
    Attn { q: WRef<'a>, k: WRef<'a>, v: WRef<'a>, o: WRef<'a>, q_norm: Vec<f32>, k_norm: Vec<f32> },
}

pub struct Model<'a> {
    pub cfg: Config, pub g: &'a Gguf, pub had: Option<Hadamard>,
    pub tok_embd: WRef<'a>, pub output_norm: Vec<f32>, pub output: WRef<'a>, pub layers: Vec<Layer<'a>>,
}

impl<'a> Model<'a> {
    pub fn load(g: &'a Gguf, n_layers: Option<usize>) -> Result<Self, String> {
        let mut cfg = Config::from_gguf(g)?;
        if let Some(n) = n_layers { cfg = cfg.with_layers(n); }
        let w = |n: &str| WRef::new(g, &cfg, n);
        let v = |n: &str| -> Result<Vec<f32>, String> { Ok(WRef::new(g, &cfg, n)?.f32_vec()) };
        let mut layers = Vec::new();
        for il in 0..cfg.n_layer {
            let p = |s: &str| format!("blk.{il}.{s}");
            let kind = if cfg.is_recurrent(il) {
                Kind::Gdn { qkv: w(&p("attn_qkv.weight"))?, gate: w(&p("attn_gate.weight"))?, conv: v(&p("ssm_conv1d.weight"))?,
                    dt: v(&p("ssm_dt.bias"))?, a: v(&p("ssm_a"))?, beta: w(&p("ssm_beta.weight"))?, alpha: w(&p("ssm_alpha.weight"))?,
                    norm: v(&p("ssm_norm.weight"))?, out: w(&p("ssm_out.weight"))? }
            } else {
                Kind::Attn { q: w(&p("attn_q.weight"))?, k: w(&p("attn_k.weight"))?, v: w(&p("attn_v.weight"))?, o: w(&p("attn_output.weight"))?,
                    q_norm: v(&p("attn_q_norm.weight"))?, k_norm: v(&p("attn_k_norm.weight"))? }
            };
            layers.push(Layer { attn_norm: v(&p("attn_norm.weight"))?, post_norm: v(&p("post_attention_norm.weight"))?,
                ffn_gate: w(&p("ffn_gate.weight"))?, ffn_up: w(&p("ffn_up.weight"))?, ffn_down: w(&p("ffn_down.weight"))?, kind });
        }
        let output = if g.tensors.contains_key("output.weight") { w("output.weight")? } else { w("token_embd.weight")? };
        Ok(Model { had: load_hadamard(g)?, tok_embd: w("token_embd.weight")?, output_norm: v("output_norm.weight")?, output, layers, cfg, g })
    }

    /// Hadamard-transform the activation if the weight was folded.
    pub fn prep(&self, w: &WRef, x: &[f32]) -> Vec<f32> {
        match (&self.had, w.hadamard) { (Some(h), true) => h.forward(x), _ => x.to_vec() }
    }
    pub fn embed(&self, tok: u32) -> Vec<f32> {
        let z = self.tok_embd.row_f32(tok as usize);
        let inv = self.cfg.had_inverse.iter().any(|n| n == "token_embd.weight");
        match (&self.had, inv) { (Some(h), true) => h.inverse(&z), _ => z }
    }
}

/// Recurrent + KV state for one sequence.
#[derive(Clone)]
pub struct State {
    pub pos: usize,
    /// output-normed hidden state of the last processed token (llama.cpp "h_nextn"; MTP head input)
    pub h: Vec<f32>,
    pub conv: Vec<Vec<f32>>,   // per layer: [(k-1) * conv_dim], oldest first
    pub ssm: Vec<Vec<f32>>,    // per layer: [vheads][S*S], M[j][i] = S[i][j]
    pub k: Vec<Vec<f32>>, pub v: Vec<Vec<f32>>, // per layer: [pos][kv_heads*head_dim]
}
impl State {
    pub fn new(cfg: &Config) -> Self {
        let n = cfg.n_layer;
        State { pos: 0, h: vec![0.0; cfg.n_embd], conv: (0..n).map(|_| vec![0.0; (cfg.conv_k - 1) * cfg.conv_dim()]).collect(),
            ssm: (0..n).map(|_| vec![0.0; cfg.ssm_vheads * cfg.ssm_state * cfg.ssm_state]).collect(),
            k: vec![Vec::new(); n], v: vec![Vec::new(); n] }
    }
}

/// neox-style (I)MRoPE for text positions (t=h=w=pos), first n_rot dims of each head.
pub fn rope(x: &mut [f32], head_dim: usize, n_rot: usize, pos: usize, base: f32, sections: [usize; 4]) {
    let half = n_rot / 2;
    let theta_scale = base.powf(-2.0 / n_rot as f32);
    let sect: usize = sections.iter().sum();
    for h in x.chunks_exact_mut(head_dim) {
        let mut theta = pos as f32;
        for i in 0..half {
            // imrope: sectors beyond 3*sections use the "e" position (0 for text)
            let sector = if sect > 0 { i % sect } else { 0 };
            let use_e = sect > 0 && !((sector % 3 == 1 && sector < 3 * sections[1]) || (sector % 3 == 2 && sector < 3 * sections[2]) || (sector % 3 == 0 && sector < 3 * sections[0]));
            let th = if use_e { 0.0 } else { theta };
            let (s, c) = th.sin_cos();
            let (x0, x1) = (h[i], h[i + half]);
            h[i] = x0 * c - x1 * s; h[i + half] = x0 * s + x1 * c;
            theta *= theta_scale;
        }
    }
}

pub fn l2norm(x: &mut [f32], eps: f32) { let s: f32 = x.iter().map(|v| v * v).sum(); let sc = 1.0 / s.sqrt().max(eps); for v in x { *v *= sc; } }

impl<'a> Model<'a> {
    /// Process one token; returns logits if `want_logits`.
    pub fn forward_token(&self, st: &mut State, tok: u32, want_logits: bool, num: Numerics, mut tap: Option<&mut dyn FnMut(&str, usize, &[f32])>) -> Option<Vec<f32>> {
        let c = &self.cfg;
        let mut x = self.embed(tok);
        if let Some(t) = tap.as_mut() { t("embd", 0, &x); }
        for (il, l) in self.layers.iter().enumerate() {
            let xn = rmsnorm(&x, &l.attn_norm, c.eps);
            let a = match &l.kind {
                Kind::Gdn { .. } => self.gdn(il, l, st, &xn, num),
                Kind::Attn { .. } => self.attn(il, l, st, &xn, num),
            };
            for i in 0..x.len() { x[i] += a[i]; }
            let hn = rmsnorm(&x, &l.post_norm, c.eps);
            let hx = self.prep(&l.ffn_gate, &hn);
            let (g, u) = rayon::join(|| matvec(&l.ffn_gate, &hx, num), || matvec(&l.ffn_up, &hx, num));
            let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
            let d = matvec(&l.ffn_down, &self.prep(&l.ffn_down, &act), num);
            for i in 0..x.len() { x[i] += d[i]; }
            if let Some(t) = tap.as_mut() { t("l_out", il, &x); }
        }
        st.pos += 1;
        let xn = rmsnorm(&x, &self.output_norm, c.eps);
        st.h.copy_from_slice(&xn);
        if !want_logits { return None; }
        Some(matvec(&self.output, &self.prep(&self.output, &xn), num))
    }

    fn gdn(&self, il: usize, l: &Layer, st: &mut State, xn: &[f32], num: Numerics) -> Vec<f32> {
        let c = &self.cfg;
        let Kind::Gdn { qkv, gate, conv, dt, a, beta, alpha, norm, out } = &l.kind else { unreachable!() };
        let hx = self.prep(qkv, xn);
        let (mixed, z) = rayon::join(|| matvec(qkv, &hx, num), || matvec(gate, &self.prep(gate, xn), num));
        let b_raw = matvec(beta, &self.prep(beta, xn), num);
        let al = matvec(alpha, &self.prep(alpha, xn), num);
        let cd = c.conv_dim(); let kk = c.conv_k;
        // causal depthwise conv over [state..., x_t]
        let cs = &mut st.conv[il];
        let mut co = vec![0f32; cd];
        for ch in 0..cd {
            let mut s = 0f32;
            for i in 0..kk { let v = if i < kk - 1 { cs[i * cd + ch] } else { mixed[ch] }; s += v * conv[ch * kk + i]; }
            co[ch] = silu(s);
        }
        // shift state
        for i in 0..kk - 2 { for ch in 0..cd { cs[i * cd + ch] = cs[(i + 1) * cd + ch]; } }
        cs[(kk - 2) * cd..(kk - 1) * cd].copy_from_slice(&mixed);
        let sd = c.ssm_state; let nk = c.ssm_groups; let nv = c.ssm_vheads; let qk = c.qk_dim();
        for h in 0..2 * nk { l2norm(&mut co[h * sd..(h + 1) * sd], c.eps); }
        let scale = 1.0 / (sd as f32).sqrt();
        let ssm = &mut st.ssm[il];
        let mut o = vec![0f32; c.ssm_inner];
        o.par_chunks_mut(sd).zip(ssm.par_chunks_mut(sd * sd)).enumerate().for_each(|(h, (oh, m))| {
            let kh = h % nk;
            let q = &co[kh * sd..(kh + 1) * sd]; let k = &co[qk + kh * sd..qk + (kh + 1) * sd];
            let v = &co[2 * qk + h * sd..2 * qk + (h + 1) * sd];
            let bt = sigmoid(b_raw[h]); let g = softplus(al[h] + dt[h]) * a[h]; let eg = g.exp();
            for j in 0..sd {
                let row = &mut m[j * sd..(j + 1) * sd];
                let mut s = 0f32;
                for i in 0..sd { row[i] *= eg; s += row[i] * k[i]; }
                let delta = (v[j] - s) * bt;
                let mut acc = 0f32;
                for i in 0..sd { row[i] += k[i] * delta; acc += row[i] * q[i]; }
                oh[j] = acc * scale;
            }
        });
        // gated rmsnorm per head
        let hv = c.head_v_dim();
        for h in 0..nv {
            let n = rmsnorm(&o[h * hv..(h + 1) * hv], norm, c.eps);
            for j in 0..hv { o[h * hv + j] = n[j] * silu(z[h * hv + j]); }
        }
        let o = if c.gdn_v_grouped && out.hadamard { gdn_group_perm(&o, hv, nk, nv / nk) } else { o };
        matvec(out, &self.prep(out, &o), num)
    }

    fn attn(&self, il: usize, l: &Layer, st: &mut State, xn: &[f32], num: Numerics) -> Vec<f32> {
        let c = &self.cfg;
        let Kind::Attn { q, k, v, o, q_norm, k_norm } = &l.kind else { unreachable!() };
        let hx = self.prep(q, xn);
        let qf = matvec(q, &hx, num); let mut kc = matvec(k, &self.prep(k, xn), num); let vc = matvec(v, &self.prep(v, xn), num);
        let hd = c.head_dim; let nh = c.n_head; let nkv = c.n_head_kv;
        let mut qc = vec![0f32; nh * hd]; let mut gate = vec![0f32; nh * hd];
        for h in 0..nh {
            let n = rmsnorm(&qf[h * 2 * hd..h * 2 * hd + hd], q_norm, c.eps);
            qc[h * hd..(h + 1) * hd].copy_from_slice(&n);
            gate[h * hd..(h + 1) * hd].copy_from_slice(&qf[h * 2 * hd + hd..(h + 1) * 2 * hd]);
        }
        for h in 0..nkv { let n = rmsnorm(&kc[h * hd..(h + 1) * hd], k_norm, c.eps); kc[h * hd..(h + 1) * hd].copy_from_slice(&n); }
        rope(&mut qc, hd, c.n_rot, st.pos, c.rope_base, c.rope_sections);
        rope(&mut kc, hd, c.n_rot, st.pos, c.rope_base, c.rope_sections);
        let (kc, vc): (Vec<f32>, Vec<f32>) = if num == Numerics::Ggml { (kc.iter().map(|&x| f16r(x)).collect(), vc.iter().map(|&x| f16r(x)).collect()) } else { (kc, vc) };
        st.k[il].extend_from_slice(&kc); st.v[il].extend_from_slice(&vc);
        let n = st.pos + 1; let kvd = nkv * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let (ks, vs) = (&st.k[il], &st.v[il]);
        let mut out = vec![0f32; nh * hd];
        out.par_chunks_mut(hd).enumerate().for_each(|(h, oh)| {
            let kh = h / (nh / nkv);
            let qh = &qc[h * hd..(h + 1) * hd];
            let mut sc: Vec<f32> = (0..n).map(|p| { let kp = &ks[p * kvd + kh * hd..p * kvd + (kh + 1) * hd]; qh.iter().zip(kp).map(|(a, b)| a * b).sum::<f32>() * scale }).collect();
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0f32; for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
            for p in 0..n { let w = sc[p] / sum; let vp = &vs[p * kvd + kh * hd..]; for j in 0..hd { oh[j] += w * vp[j]; } }
        });
        for i in 0..out.len() { out[i] *= sigmoid(gate[i]); }
        matvec(o, &self.prep(o, &out), num)
    }
}
