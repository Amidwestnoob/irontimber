//! Multi-token prediction (Qwen3.5/3.8 NextN head) speculative decoding, greedy, draft length 1.
//!
//! Head (llama-prism `graph_mtp` for qwen35): for MTP position q the input pair is (h_{q-1}, x_q), where h is
//! the trunk's output-normed hidden state ("h_nextn") of the previous token:
//!   e = rmsnorm(embed(x_q), enorm); hn = rmsnorm(h_{q-1}, hnorm); x = eh_proj [e; hn]
//!   x += attn(rmsnorm(x, attn_norm))      (gated full attention with its own KV cache, IMRoPE at position q)
//!   x += ffn(rmsnorm(x, post_norm))       (SwiGLU)
//!   logits = output(rmsnorm(x, shared_head_norm))  -> draft for x_{q+1}
//! The grafted head is unrotated (no Hadamard folding) and stored as Q8_0/Q6_K (repacked to `Q8G16`), with an
//! optional unrotated Q4_K copy of the embedding (`nextn.embed_tokens`, used when present, as llama-prism does).
//!
//! Loop (identical output to plain greedy by construction): verify [x, d] in one trunk pass with the recurrent
//! state checkpointed after x; accept if argmax_0 == d (emit d and argmax_1), else roll back to the checkpoint and
//! emit argmax_0. The MTP KV cache only ever holds confirmed pairs (catch-up runs 1 or 2 rows per step).
use crate::cpu::{self, rmsnorm, rope, sigmoid, silu, Model, Numerics};
use crate::gguf::Gguf;
use crate::model::WRef;
use crate::quant::Q8G16;
use crate::sampling::argmax;
use rayon::prelude::*;

pub struct MtpHead<'a> {
    pub il: usize,
    pub eh: Q8G16, pub q: Q8G16, pub k: Q8G16, pub v: Q8G16, pub o: Q8G16, pub gate: Q8G16, pub up: Q8G16, pub down: Q8G16,
    pub enorm: Vec<f32>, pub hnorm: Vec<f32>, pub attn_norm: Vec<f32>, pub post_norm: Vec<f32>, pub q_norm: Vec<f32>, pub k_norm: Vec<f32>, pub head_norm: Vec<f32>,
    pub embd: Option<WRef<'a>>,
}

/// Fail closed before Vulkan construction/repacking. A compatible graph is not proof of trained quality;
/// provenance and full trunk equivalence are separately attested by the review/conversion tools.
pub fn validate_compatibility(head: &Gguf, trunk: &Gguf) -> Result<(), String> {
    let c = crate::model::Config::from_gguf(trunk)?;
    if head.u("qwen35.nextn_predict_layers")? != 1 || head.u("qwen35.block_count")?.checked_sub(1) != Some(c.n_layer) {
        return Err("MTP/trunk layer identity mismatch".into());
    }
    for key in head.kv.keys().chain(trunk.kv.keys()) {
        if key == "qwen35.block_count" || key == "qwen35.nextn_predict_layers" { continue; }
        if key == "general.architecture" || key.starts_with("qwen35.") || key.starts_with("tokenizer.") || key.starts_with("prism.hadamard.") {
            if head.get(key) != trunk.get(key) { return Err(format!("MTP/trunk metadata mismatch: {key}")); }
        }
    }
    MtpHead::validate_layout(head, &c)
}

impl<'a> MtpHead<'a> {
    pub fn validate_layout(g: &Gguf, c: &crate::model::Config) -> Result<(), String> {
        use crate::gguf::GgmlType;
        if g.get("general.architecture").and_then(|v|v.as_str()) != Some("qwen35") || g.u("qwen35.nextn_predict_layers")? != 1 { return Err("expected qwen35 single NextN head".into()); }
        let il = g.u("qwen35.block_count")?.checked_sub(1).ok_or("invalid head block count")?;
        let p = |s: &str| format!("blk.{il}.{s}");
        let e=c.n_embd; let q=c.n_head*c.head_dim; let kv=c.n_head_kv*c.head_dim;
        let matrices = [("nextn.eh_proj.weight",2*e,e),("attn_q.weight",e,2*q),("attn_k.weight",e,kv),("attn_v.weight",e,kv),("attn_output.weight",q,e),("ffn_gate.weight",e,c.n_ff),("ffn_up.weight",e,c.n_ff),("ffn_down.weight",c.n_ff,e)];
        for (s,cols,rows) in matrices {
            let t=g.tensor(&p(s))?;
            if t.dims != [cols,rows] || !matches!(t.ty,GgmlType::Q8_0|GgmlType::Q6_K) || rows%4!=0 || cols%256!=0 { return Err(format!("unsupported MTP matrix shape/type: {}",t.name)); }
        }
        for (s,n) in [("nextn.enorm.weight",e),("nextn.hnorm.weight",e),("attn_norm.weight",e),("post_attention_norm.weight",e),("attn_q_norm.weight",c.head_dim),("attn_k_norm.weight",c.head_dim),("nextn.shared_head_norm.weight",e)] {
            let t=g.tensor(&p(s))?;
            if t.dims != [n] || t.ty != GgmlType::F32 { return Err(format!("unsupported MTP norm shape/type: {}",t.name)); }
            if g.data(t).chunks_exact(4).any(|b| !f32::from_le_bytes(b.try_into().unwrap()).is_finite()) { return Err(format!("nonfinite MTP norm: {}",t.name)); }
        }
        if let Some(t)=g.tensors.get(&p("nextn.embed_tokens.weight")) {
            if t.dims != [e,c.n_vocab] || t.ty != GgmlType::Q4_K { return Err("unsupported MTP embedding shape/type".into()); }
        }
        if g.tensors.contains_key(&p("nextn.shared_head_head.weight")) { return Err("separate MTP LM head unsupported".into()); }
        for key in ["prism.hadamard.weight_names","prism.hadamard.inverse_weight_names"] {
            if g.get(key).and_then(|v|v.as_arr()).map_or(false,|a|a.iter().any(|v|v.as_str().map_or(false,|s|s.starts_with(&format!("blk.{il}."))))) { return Err("rotated MTP head is unsupported".into()); }
        }
        Ok(())
    }
    /// Load the NextN block from `g` (the model itself or a head-only sidecar). None if the file has no MTP block.
    pub fn load(g: &'a Gguf, cfg: &crate::model::Config) -> Result<Option<Self>, String> {
        let nn = g.u("qwen35.nextn_predict_layers").unwrap_or(0);
        if nn == 0 { return Ok(None); }
        if nn != 1 { return Err("only a single MTP block is supported".into()); }
        Self::validate_layout(g, cfg)?;
        let il = g.u("qwen35.block_count")? - 1;
        let p = |s: &str| format!("blk.{il}.{s}");
        let m = |s: &str| -> Result<Q8G16, String> { let (t, d) = g.tensor_data(&p(s))?; let w = Q8G16::from_ggml(t.ty, d, t.dims[1..].iter().product(), t.dims[0]); if w.scales.iter().any(|s|!s.is_finite()) { return Err(format!("nonfinite MTP scale: {}",t.name)); } Ok(w) };
        let v = |s: &str| -> Result<Vec<f32>, String> { Ok(WRef::new(g, cfg, &p(s))?.f32_vec()) };
        let embd = if std::env::var("BONSAI_MTP_EMBD").map(|v| v == "main").unwrap_or(false) { None } else { g.tensors.get(&p("nextn.embed_tokens.weight")).map(|_| WRef::new(g, cfg, &p("nextn.embed_tokens.weight"))).transpose()? };
        let head_norm = if g.tensors.contains_key(&p("nextn.shared_head_norm.weight")) { v("nextn.shared_head_norm.weight")? } else { WRef::new(g, cfg, "output_norm.weight")?.f32_vec() };
        if g.tensors.contains_key(&p("nextn.shared_head_head.weight")) { return Err("separate MTP LM head (nextn.shared_head_head) not supported; head uses output.weight".into()); }
        Ok(Some(MtpHead { il, eh: m("nextn.eh_proj.weight")?, q: m("attn_q.weight")?, k: m("attn_k.weight")?, v: m("attn_v.weight")?, o: m("attn_output.weight")?,
            gate: m("ffn_gate.weight")?, up: m("ffn_up.weight")?, down: m("ffn_down.weight")?,
            enorm: v("nextn.enorm.weight")?, hnorm: v("nextn.hnorm.weight")?, attn_norm: v("attn_norm.weight")?, post_norm: v("post_attention_norm.weight")?,
            q_norm: v("attn_q_norm.weight")?, k_norm: v("attn_k_norm.weight")?, head_norm, embd }))
    }
    pub fn embed(&self, m: &Model, tok: u32) -> Vec<f32> { match &self.embd { Some(w) => w.row_f32(tok as usize), None => m.embed(tok) } }
    pub fn bytes(&self) -> usize { [&self.eh, &self.q, &self.k, &self.v, &self.o, &self.gate, &self.up, &self.down].iter().map(|w| w.words.len() * 4 + w.scales.len() * 4).sum() }
}

pub fn mv(w: &Q8G16, x: &[f32]) -> Vec<f32> { assert_eq!(x.len(), w.cols); (0..w.rows).into_par_iter().map(|r| w.dot(r, x)).collect() }

/// CPU MTP KV cache (slot = position - 1).
#[derive(Default, Clone)]
pub struct CpuMtpState { pub k: Vec<f32>, pub v: Vec<f32> }

/// One MTP step on the CPU (f32 reference): returns (logits, output-normed hidden).
pub fn cpu_step(m: &Model, hd: &MtpHead, st: &mut CpuMtpState, h: &[f32], tok: u32, pos: usize, num: Numerics) -> Vec<f32> {
    let c = &m.cfg; let e = c.n_embd;
    let en = rmsnorm(&hd.embed(m, tok), &hd.enorm, c.eps); let hn = rmsnorm(h, &hd.hnorm, c.eps);
    let mut cat = en; cat.extend_from_slice(&hn);
    let mut x = mv(&hd.eh, &cat);
    let xn = rmsnorm(&x, &hd.attn_norm, c.eps);
    let qf = mv(&hd.q, &xn); let mut kc = mv(&hd.k, &xn); let vc = mv(&hd.v, &xn);
    let (hdim, nh, nkv) = (c.head_dim, c.n_head, c.n_head_kv);
    let mut qc = vec![0f32; nh * hdim]; let mut gate = vec![0f32; nh * hdim];
    for h in 0..nh {
        qc[h * hdim..(h + 1) * hdim].copy_from_slice(&rmsnorm(&qf[h * 2 * hdim..h * 2 * hdim + hdim], &hd.q_norm, c.eps));
        gate[h * hdim..(h + 1) * hdim].copy_from_slice(&qf[h * 2 * hdim + hdim..(h + 1) * 2 * hdim]);
    }
    for h in 0..nkv { let n = rmsnorm(&kc[h * hdim..(h + 1) * hdim], &hd.k_norm, c.eps); kc[h * hdim..(h + 1) * hdim].copy_from_slice(&n); }
    rope(&mut qc, hdim, c.n_rot, pos, c.rope_base, c.rope_sections);
    rope(&mut kc, hdim, c.n_rot, pos, c.rope_base, c.rope_sections);
    let kvd = nkv * hdim; let slot = pos - 1;
    st.k.truncate(slot * kvd); st.v.truncate(slot * kvd);
    assert_eq!(st.k.len(), slot * kvd, "MTP KV must be contiguous");
    st.k.extend_from_slice(&kc); st.v.extend_from_slice(&vc);
    let n = slot + 1; let scale = 1.0 / (hdim as f32).sqrt();
    let mut out = vec![0f32; nh * hdim];
    out.par_chunks_mut(hdim).enumerate().for_each(|(h, oh)| {
        let kh = h / (nh / nkv); let qh = &qc[h * hdim..(h + 1) * hdim];
        let mut sc: Vec<f32> = (0..n).map(|p| st.k[p * kvd + kh * hdim..p * kvd + (kh + 1) * hdim].iter().zip(qh).map(|(a, b)| a * b).sum::<f32>() * scale).collect();
        let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max); let mut sum = 0f32;
        for s in sc.iter_mut() { *s = (*s - mx).exp(); sum += *s; }
        for p in 0..n { let w = sc[p] / sum; let vp = &st.v[p * kvd + kh * hdim..]; for j in 0..hdim { oh[j] += w * vp[j]; } }
    });
    for i in 0..out.len() { out[i] *= sigmoid(gate[i]); }
    let a = mv(&hd.o, &out); for i in 0..e { x[i] += a[i]; }
    let hn2 = rmsnorm(&x, &hd.post_norm, c.eps);
    let (g, u) = rayon::join(|| mv(&hd.gate, &hn2), || mv(&hd.up, &hn2));
    let act: Vec<f32> = g.iter().zip(&u).map(|(g, u)| silu(*g) * u).collect();
    let d = mv(&hd.down, &act); for i in 0..e { x[i] += d[i]; }
    let xo = rmsnorm(&x, &hd.head_norm, c.eps);
    cpu::matvec(&m.output, &m.prep(&m.output, &xo), num)
}

#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct MtpStats {
    pub drafted: usize, pub accepted: usize, pub steps: usize,
    pub draft_calls: usize, pub verify_calls: usize, pub single_forwards: usize,
    pub timed_decode_forwards: usize, pub rejected: usize, pub terminal_rollbacks: usize,
    pub prefill_seconds: f64, pub generation_seconds: f64,
    pub draft_seconds: f64, pub verify_seconds: f64, pub rollback_seconds: f64, pub single_seconds: f64,
}
impl MtpStats { pub fn rate(&self) -> f64 { if self.drafted == 0 { 0.0 } else { self.accepted as f64 / self.drafted as f64 } } }

/// Backend hooks for the speculative loop (CPU reference or Vulkan).
pub trait MtpBackend {
    /// Trunk prefill of the prompt + MTP catch-up over its pairs; returns the greedy next token.
    fn mtp_prefill(&mut self, prompt: &[u32]) -> u32;
    /// Final remaining prediction: one trunk token, no unused draft or extra cache slot.
    fn mtp_single(&mut self, x: u32) -> u32;
    /// Run the MTP head over `toks` (paired with the pending trunk h rows, same count); returns the draft.
    fn mtp_draft(&mut self, toks: &[u32]) -> u32;
    /// Trunk pass over [x, d] with the recurrent state checkpointed after x; returns (argmax after x, argmax after d).
    fn mtp_verify(&mut self, x: u32, d: u32) -> (u32, u32);
    /// Draft rejected: restore the checkpoint (state after x) and keep only x's h row pending.
    fn mtp_reject(&mut self);
    /// Drop every per-request state (trunk KV, conv/SSM, checkpoints, MTP KV/history, pending h rows).
    fn reset_all(&mut self);
}

/// Greedy generation with MTP drafting. `on_tok` returns false to stop. Output equals plain greedy.
pub fn greedy_mtp(b: &mut dyn MtpBackend, prompt: &[u32], n: usize, is_eog: &dyn Fn(u32) -> bool, on_tok: &mut dyn FnMut(u32) -> bool) -> (Vec<u32>, MtpStats) {
    use std::time::Instant;
    let mut out = vec![]; let mut st = MtpStats::default();
    let mut emit = |t: u32, out: &mut Vec<u32>| -> bool { out.push(t); on_tok(t) && !is_eog(t) && out.len() < n };
    if n == 0 { return (out, st); }
    assert!(!prompt.is_empty(), "empty MTP prompt");
    let t = Instant::now();
    let mut x = b.mtp_prefill(prompt);
    st.prefill_seconds = t.elapsed().as_secs_f64();
    let generation = Instant::now();
    if !emit(x, &mut out) { st.generation_seconds = generation.elapsed().as_secs_f64(); return (out, st); }
    let mut pending = vec![x];
    loop {
        if n - out.len() == 1 {
            let t = Instant::now(); let a = b.mtp_single(x);
            st.single_seconds += t.elapsed().as_secs_f64(); st.single_forwards += 1; st.timed_decode_forwards += 1;
            emit(a, &mut out); break;
        }
        let t = Instant::now(); let d = b.mtp_draft(&pending);
        st.draft_seconds += t.elapsed().as_secs_f64(); st.draft_calls += 1;
        let t = Instant::now(); let (a0, a1) = b.mtp_verify(x, d);
        st.verify_seconds += t.elapsed().as_secs_f64(); st.drafted += 1; st.steps += 1; st.verify_calls += 1; st.timed_decode_forwards += 2;
        if a0 == d {
            st.accepted += 1;
            if !emit(d, &mut out) {
                let t = Instant::now(); b.mtp_reject(); st.rollback_seconds += t.elapsed().as_secs_f64(); st.terminal_rollbacks += 1;
                break;
            }
            if !emit(a1, &mut out) { break; }
            x = a1; pending = vec![a0, a1];
        } else {
            let t = Instant::now(); b.mtp_reject(); st.rollback_seconds += t.elapsed().as_secs_f64(); st.rejected += 1;
            if !emit(a0, &mut out) { break; }
            x = a0; pending = vec![a0];
        }
    }
    st.generation_seconds = generation.elapsed().as_secs_f64();
    (out, st)
}

/// CPU reference backend.
pub struct CpuMtp<'m, 'a> { pub m: &'m Model<'a>, pub hd: &'m MtpHead<'a>, pub num: Numerics, pub st: cpu::State, pub ck: Option<cpu::State>, pub ms: CpuMtpState, pub hpend: Vec<Vec<f32>>, pub mpos: usize }
impl<'m, 'a> CpuMtp<'m, 'a> {
    pub fn new(m: &'m Model<'a>, hd: &'m MtpHead<'a>, num: Numerics) -> Self { CpuMtp { m, hd, num, st: cpu::State::new(&m.cfg), ck: None, ms: CpuMtpState::default(), hpend: vec![], mpos: 1 } }
}
impl MtpBackend for CpuMtp<'_, '_> {
    fn mtp_single(&mut self, x: u32) -> u32 { argmax(&self.m.forward_token(&mut self.st, x, true, self.num, None).unwrap()) }
    fn mtp_prefill(&mut self, prompt: &[u32]) -> u32 {
        self.reset_all();
        let mut hs = vec![]; let mut lg = None;
        for (i, &t) in prompt.iter().enumerate() { lg = self.m.forward_token(&mut self.st, t, i + 1 == prompt.len(), self.num, None); hs.push(self.st.h.clone()); }
        for q in 1..prompt.len() { cpu_step(self.m, self.hd, &mut self.ms, &hs[q - 1], prompt[q], q, self.num); }
        self.mpos = prompt.len(); self.hpend = vec![hs.pop().unwrap()];
        argmax(&lg.unwrap())
    }
    fn mtp_draft(&mut self, toks: &[u32]) -> u32 {
        assert_eq!(toks.len(), self.hpend.len());
        let mut lg = vec![];
        for (i, &t) in toks.iter().enumerate() { lg = cpu_step(self.m, self.hd, &mut self.ms, &self.hpend[i], t, self.mpos, self.num); self.mpos += 1; }
        argmax(&lg)
    }
    fn mtp_verify(&mut self, x: u32, d: u32) -> (u32, u32) {
        let l0 = self.m.forward_token(&mut self.st, x, true, self.num, None).unwrap(); let h0 = self.st.h.clone();
        self.ck = Some(self.st.clone());
        let l1 = self.m.forward_token(&mut self.st, d, true, self.num, None).unwrap(); let h1 = self.st.h.clone();
        self.hpend = vec![h0, h1];
        (argmax(&l0), argmax(&l1))
    }
    fn mtp_reject(&mut self) { self.st = self.ck.take().unwrap(); self.hpend.truncate(1); }
    fn reset_all(&mut self) { self.st = cpu::State::new(&self.m.cfg); self.ck = None; self.ms = CpuMtpState::default(); self.hpend.clear(); self.mpos = 1; }
}
