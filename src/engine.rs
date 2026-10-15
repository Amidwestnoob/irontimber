//! Backends behind one trait, plus the shared generation loop (greedy or sampled, streamed per token).
use crate::cpu;
use crate::sampling::{argmax, Sampler, SamplerParams};
use crate::tokenizer::Tokenizer;
use std::time::Instant;

/// Anything that can eat tokens and optionally produce logits for the last one.
pub trait Engine {
    fn eval(&mut self, toks: &[u32], want_logits: bool) -> Option<Vec<f32>>;
    fn eval_argmax(&mut self, toks: &[u32]) -> u32 { argmax(&self.eval(toks, true).unwrap()) }
    fn reset(&mut self);
    fn profile(&self) {}
    /// Opt-in review telemetry; not an inference or sampling input.
    fn review_snapshot(&self) -> serde_json::Value { serde_json::Value::Null }
}
#[cfg(feature = "vulkan")]
impl Engine for crate::vk::VkEngine<'_> {
    fn review_snapshot(&self) -> serde_json::Value { self.review_identity() }
    fn eval(&mut self, toks: &[u32], want: bool) -> Option<Vec<f32>> { self.eval_tokens(toks, want) }
    fn reset(&mut self) { crate::vk::VkEngine::reset(self) }
    fn profile(&self) { self.ctx.print_profile() }
    fn eval_argmax(&mut self, toks: &[u32]) -> u32 { if std::env::var("BONSAI_HOST_ARGMAX").is_ok() { argmax(&self.eval_tokens(toks, true).unwrap()) } else { crate::vk::VkEngine::eval_argmax(self, toks) } }
}
pub struct CpuEngine<'a> { pub m: cpu::Model<'a>, pub st: cpu::State, pub num: cpu::Numerics }
impl Engine for CpuEngine<'_> {
    fn eval(&mut self, toks: &[u32], want: bool) -> Option<Vec<f32>> {
        let mut r = None;
        for (i, &t) in toks.iter().enumerate() { r = self.m.forward_token(&mut self.st, t, want && i + 1 == toks.len(), self.num, None); }
        r
    }
    fn reset(&mut self) { self.st = cpu::State::new(&self.m.cfg); }
}

/// Incremental detokenizer: emits only complete UTF-8 text (multi-byte chars split across tokens are held back).
#[derive(Default)]
pub struct TextStream { pending: Vec<u8> }
impl TextStream {
    pub fn push(&mut self, tk: &Tokenizer, id: u32) -> String {
        self.pending.extend(tk.token_bytes(id));
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => { out.push_str(s); self.pending.clear(); break; }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    self.pending.drain(..valid);
                    if let Some(n) = e.error_len() { out.push('\u{fffd}'); self.pending.drain(..n); }
                    else { break; }
                }
            }
        }
        out
    }
    pub fn flush(&mut self) -> String { let s = String::from_utf8_lossy(&self.pending).into_owned(); self.pending.clear(); s }
}

#[derive(Clone, Debug)]
pub struct GenRequest { pub prompt: Vec<u32>, pub n_predict: usize, pub sampler: SamplerParams, pub stop: Vec<String>, pub ignore_eos: bool }
#[derive(Clone, Debug, Default)]
pub struct GenResult { pub tokens: Vec<u32>, pub text: String, pub stop_type: &'static str, pub stopping_word: String, pub prompt_ms: f64, pub predicted_ms: f64, pub seed: u64, pub decode_forwards: usize,
    /// r012: prompt tokens reused from the prompt cache (0 = full prefill).
    pub cached_tokens: usize,
    /// Present iff this request ran the MTP speculative path.
    pub mtp: Option<crate::mtp::MtpStats> }

/// An engine the server can hold for its whole lifetime. `mtp_backend` exposes the speculative
/// path when an MTP head was loaded at start-up (weights/pipelines/device buffers are reused).
pub trait ServeEngine: Engine {
    fn mtp_backend(&mut self) -> Option<&mut dyn crate::mtp::MtpBackend> { None }
    fn as_engine(&mut self) -> &mut dyn Engine;
}
impl ServeEngine for CpuEngine<'_> { fn as_engine(&mut self) -> &mut dyn Engine { self } }
#[cfg(feature = "vulkan")]
impl ServeEngine for crate::vk::VkEngine<'_> {
    fn mtp_backend(&mut self) -> Option<&mut dyn crate::mtp::MtpBackend> { if self.has_mtp() { Some(self) } else { None } }
    fn as_engine(&mut self) -> &mut dyn Engine { self }
}
/// CPU reference trunk + MTP head, usable both as a plain engine and as the speculative backend.
impl Engine for crate::mtp::CpuMtp<'_, '_> {
    fn eval(&mut self, toks: &[u32], want: bool) -> Option<Vec<f32>> {
        let mut r = None;
        for (i, &t) in toks.iter().enumerate() { r = self.m.forward_token(&mut self.st, t, want && i + 1 == toks.len(), self.num, None); }
        r
    }
    fn reset(&mut self) { crate::mtp::MtpBackend::reset_all(self) }
}
impl ServeEngine for crate::mtp::CpuMtp<'_, '_> {
    fn mtp_backend(&mut self) -> Option<&mut dyn crate::mtp::MtpBackend> { Some(self) }
    fn as_engine(&mut self) -> &mut dyn Engine { self }
}

/// Per-token post-processing shared by plain and MTP generation: terminal EOG, stop strings with
/// held-back partial matches, cancellation. `push` returns false when generation must stop.
struct Emitter<'a> { tk: &'a Tokenizer, r: &'a GenRequest, res: GenResult, ts: TextStream, emitted: usize }
impl Emitter<'_> {
    fn push(&mut self, t: u32, on_piece: &mut dyn FnMut(u32, &str) -> bool) -> bool {
        let (r, res) = (self.r, &mut self.res);
        res.tokens.push(t);
        // Native return_tokens includes the terminal EOG token, but not its textual spelling.
        if !r.ignore_eos && self.tk.is_eog(t) { res.stop_type = "eos"; on_piece(t, ""); return false; }
        let piece = self.ts.push(self.tk, t);
        res.text.push_str(&piece);
        if let Some((pos, w)) = r.stop.iter().filter(|w| !w.is_empty()).filter_map(|w| res.text.find(w.as_str()).map(|p| (p, w.clone()))).min_by_key(|x| x.0) {
            let emit = res.text[self.emitted..pos].to_string();
            res.text.truncate(pos); res.stop_type = "word"; res.stopping_word = w;
            on_piece(t, &emit); return false;
        }
        // Never send a suffix that could become a stop word in a later token.
        let hold = res.text.char_indices().filter_map(|(i, _)| {
            let suffix = &res.text[i..];
            r.stop.iter().any(|w| w.starts_with(suffix)).then_some(suffix.len())
        }).max().unwrap_or(0);
        let safe = res.text.len() - hold;
        let emit = res.text[self.emitted..safe].to_string(); self.emitted = safe;
        if !on_piece(t, &emit) { res.stop_type = "cancel"; return false; }
        true
    }
    fn finish(mut self, on_piece: &mut dyn FnMut(u32, &str) -> bool) -> GenResult {
        let tail = self.ts.flush();
        if self.res.stop_type != "word" && self.res.stop_type != "cancel" {
            self.res.text.push_str(&tail);
            // Incomplete stop prefixes are ordinary text on EOS/length exhaustion.
            if self.emitted < self.res.text.len() { on_piece(u32::MAX, &self.res.text[self.emitted..]); }
        }
        self.res
    }
}

/// Run one generation. `on_piece(token, text_piece)` is called for every generated token (streaming);
/// returning false cancels. Greedy requests use the GPU argmax path (4-byte readback), unchanged from before.
pub fn generate(eng: &mut dyn Engine, tk: &Tokenizer, r: &GenRequest, on_piece: &mut dyn FnMut(u32, &str) -> bool) -> GenResult {
    generate_cached(eng, tk, r, None, on_piece)
}

/// r012 prompt cache (opt-in). `cache` holds exactly the tokens already fed through the engine since its last
/// reset. If the new prompt strictly extends them, the engine is not reset and only the suffix is evaluated
/// (KV and recurrent state are reused; the hybrid model's GDN state cannot be rewound, so only exact
/// extensions hit). Otherwise the engine is reset as before. On return `cache` = tokens fed this time.
/// The returned `cached_tokens` count is reported to clients as prompt_tokens_details.cached_tokens.
pub fn generate_cached(eng: &mut dyn Engine, tk: &Tokenizer, r: &GenRequest, cache: Option<&mut Vec<u32>>, on_piece: &mut dyn FnMut(u32, &str) -> bool) -> GenResult {
    let mut fed: Vec<u32> = vec![];
    let hit = match &cache { Some(c) if !c.is_empty() && r.prompt.len() > c.len() && r.prompt.starts_with(c) => c.len(), _ => 0 };
    if hit == 0 { eng.reset(); } else { fed.extend_from_slice(&r.prompt[..hit]); }
    let mut res = generate_inner(eng, tk, r, hit, &mut fed, on_piece);
    res.cached_tokens = hit;
    if let Some(c) = cache { *c = fed; }
    res
}

fn generate_inner(eng: &mut dyn Engine, tk: &Tokenizer, r: &GenRequest, skip: usize, fed: &mut Vec<u32>, on_piece: &mut dyn FnMut(u32, &str) -> bool) -> GenResult {
    let mut s = Sampler::new(r.sampler.clone());
    if r.n_predict == 0 { return GenResult { stop_type: "limit", seed: s.seed_used, ..Default::default() }; }
    let greedy = r.sampler.is_greedy();
    let mut hist = r.prompt.clone();
    let t0 = Instant::now();
    let todo = &r.prompt[skip..]; fed.extend_from_slice(todo);
    let mut next = if greedy { eng.eval_argmax(todo) } else { let l = eng.eval(todo, true).unwrap(); s.sample(&l, &hist) };
    let prompt_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t1 = Instant::now();
    let mut em = Emitter { tk, r, res: GenResult { stop_type: "limit", seed: s.seed_used, prompt_ms, ..Default::default() }, ts: TextStream::default(), emitted: 0 };
    for i in 0..r.n_predict {
        let t = next;
        hist.push(t);
        if !em.push(t, on_piece) { break; }
        if i + 1 == r.n_predict { break; }
        em.res.decode_forwards += 1; fed.push(t);
        next = if greedy { eng.eval_argmax(&[t]) } else { let l = eng.eval(&[t], true).unwrap(); s.sample(&l, &hist) };
    }
    let predicted_ms = t1.elapsed().as_secs_f64() * 1e3;
    let mut res = em.finish(on_piece);
    res.predicted_ms = predicted_ms;
    res
}

/// Greedy generation through the MTP speculative loop, with exactly the same emission semantics as
/// `generate` (EOS, stop strings, cancellation). Stopping on an accepted draft rolls back the second
/// verified token (r009 terminal-rollback semantics). All per-request trunk, recurrent, KV and MTP
/// head state is reset by `mtp_prefill`, so nothing leaks between requests.
pub fn generate_mtp(b: &mut dyn crate::mtp::MtpBackend, tk: &Tokenizer, r: &GenRequest, on_piece: &mut dyn FnMut(u32, &str) -> bool) -> Result<GenResult, String> {
    if !r.sampler.is_greedy() { return Err("mtp requires greedy decoding (temperature 0 or top_k 1, repeat_penalty 1)".into()); }
    let seed = Sampler::new(r.sampler.clone()).seed_used;
    if r.n_predict == 0 { b.reset_all(); return Ok(GenResult { stop_type: "limit", seed, mtp: Some(Default::default()), ..Default::default() }); }
    let mut em = Emitter { tk, r, res: GenResult { stop_type: "limit", seed, ..Default::default() }, ts: TextStream::default(), emitted: 0 };
    let eog = |t: u32| !r.ignore_eos && tk.is_eog(t);
    let (_toks, st) = crate::mtp::greedy_mtp(b, &r.prompt, r.n_predict, &eog, &mut |t| em.push(t, on_piece));
    let mut res = em.finish(on_piece);
    res.prompt_ms = st.prefill_seconds * 1e3; res.predicted_ms = st.generation_seconds * 1e3;
    res.decode_forwards = st.timed_decode_forwards; res.mtp = Some(st);
    Ok(res)
}
