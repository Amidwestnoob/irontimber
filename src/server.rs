//! llama-server-compatible HTTP API (pure Rust, tiny_http): GET /health, /props, /v1/models;
//! POST /tokenize, /detokenize, /completion (+ "stream": SSE), /v1/chat/completions (+ "stream": SSE).
//! Defaults to 127.0.0.1:18081 so it never collides with llama-prism on 18080.
//! Requests are served one at a time (single GPU context), like llama-server with one slot.
use crate::chat::ChatTemplate;
use crate::engine::{generate_mtp, GenRequest, GenResult, ServeEngine};
use crate::sampling::SamplerParams;
use crate::tokenizer::Tokenizer;
use serde_json::{json, Value};
use std::io::Write;

/// `mtp_default`: whether greedy requests that do not set `"mtp"` use the speculative path
/// (only meaningful when an MTP head was loaded at start-up).
pub struct ServerCfg { pub addr: String, pub model_name: String, pub n_ctx: usize, pub mtp_default: bool,
    /// r012: reuse engine state when a prompt strictly extends the previous fed sequence (opt-in, --prompt-cache on).
    pub prompt_cache: bool }

/// Resolve the per-request MTP choice. Explicit `"mtp": true` on a sampled request or on a server
/// without a head is a 400; the server default applies to greedy requests only.
pub fn resolve_mtp(j: &Value, r: &GenRequest, have_head: bool, default_on: bool) -> Result<bool, String> {
    match j.get("mtp") {
        None => Ok(have_head && default_on && r.sampler.is_greedy()),
        Some(Value::Bool(false)) => Ok(false),
        Some(Value::Bool(true)) => {
            if !have_head { return Err("mtp requested but the server was started without an MTP head (--mtp-head)".into()); }
            if !r.sampler.is_greedy() { return Err("mtp supports greedy decoding only (temperature 0 or top_k 1, repeat_penalty 1); sampled + mtp is not supported".into()); }
            Ok(true)
        }
        Some(_) => Err("mtp must be boolean".into()),
    }
}
fn mtp_json(res: &GenResult) -> Value {
    match &res.mtp {
        None => json!({"enabled": false}),
        Some(s) => json!({"enabled": true, "drafted": s.drafted, "accepted": s.accepted, "acceptance": s.rate(), "rejected": s.rejected, "verify_calls": s.verify_calls,
                          "single_forwards": s.single_forwards, "terminal_rollbacks": s.terminal_rollbacks, "draft_ms": s.draft_seconds * 1e3, "verify_ms": s.verify_seconds * 1e3, "rollback_ms": s.rollback_seconds * 1e3}),
    }
}

fn token_ids(v: &Value, tk: &Tokenizer) -> Result<Vec<u32>, String> {
    v.as_array().ok_or("tokens must be an array")?.iter().map(|v| {
        let id = v.as_u64().ok_or("token IDs must be unsigned integers")?;
        if id >= tk.tokens.len() as u64 { return Err("token ID outside vocabulary".into()); }
        Ok(id as u32)
    }).collect()
}

/// Validate before resetting state or writing SSE headers. The first prediction comes
/// from prefill, hence N emitted tokens require at most N-1 additional forwards.
pub fn validate_context(r: &mut GenRequest, j: &Value, n_ctx: usize) -> Result<(), String> {
    if r.prompt.len() > n_ctx { return Err("prompt exceeds context capacity".into()); }
    let available = n_ctx - r.prompt.len() + 1;
    let explicit = ["n_predict", "max_tokens", "max_completion_tokens"].iter()
        .any(|k| j.get(*k).and_then(Value::as_i64).map_or(false, |n| n >= 0));
    if !explicit { r.n_predict = r.n_predict.min(available); }
    if r.n_predict > available { return Err("requested generation exceeds context capacity".into()); }
    Ok(())
}

fn timings(r: &GenResult, n_prompt: usize) -> Value {
    // Output throughput is not speculative work throughput: rejected verification
    // rows still cost a forward, but never count as delivered predictions.
    let pn = n_prompt as f64; let gn = r.tokens.len().saturating_sub(1) as f64;
    json!({"prompt_n": n_prompt, "prompt_ms": r.prompt_ms, "prompt_per_token_ms": r.prompt_ms / pn.max(1.0), "prompt_per_second": pn / (r.prompt_ms / 1e3).max(1e-9),
           "predicted_n": r.tokens.len(), "predicted_ms": r.predicted_ms, "predicted_per_token_ms": r.predicted_ms / gn.max(1.0), "predicted_per_second": gn / (r.predicted_ms / 1e3).max(1e-9), "decode_forward_n": r.decode_forwards, "decode_output_n": gn, "first_prediction_in_prefill": true})
}
fn gen_settings(r: &GenRequest, seed: u64) -> Value {
    let p = &r.sampler;
    json!({"n_predict": r.n_predict, "temperature": p.temperature, "top_k": p.top_k, "top_p": p.top_p, "min_p": p.min_p, "repeat_penalty": p.repeat_penalty, "repeat_last_n": p.repeat_last_n, "seed": seed, "stop": r.stop, "ignore_eos": r.ignore_eos})
}
fn now() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

/// Build a GenRequest from a llama-server style body. `chat` selects OpenAI field names.
pub fn parse_request(j: &Value, tk: &Tokenizer, tmpl: Option<&ChatTemplate>, chat: bool) -> Result<GenRequest, String> {
    if !j.is_object() { return Err("request must be an object".into()); }
    let prompt: Vec<u32> = if chat {
        let extra = j.get("chat_template_kwargs").cloned().unwrap_or(json!({}));
        let t = tmpl.ok_or("model has no chat template")?;
        tk.encode(&t.render(j.get("messages").ok_or("missing messages")?, true, &extra)?, true, true)
    } else if let Some(ids) = j.get("prompt_ids") {
        token_ids(ids, tk)?
    } else {
        match j.get("prompt") {
            Some(v @ Value::Array(_)) => token_ids(v, tk)?,
            Some(Value::String(s)) => tk.encode(s, true, true),
            _ => return Err("missing prompt".into()),
        }
    };
    if prompt.is_empty() { return Err("empty prompt".into()); }
    for key in ["temperature", "top_p", "min_p", "repeat_penalty"] {
        if let Some(v) = j.get(key) {
            let n = v.as_f64().ok_or_else(|| format!("{key} must be numeric"))?;
            if !(n as f32).is_finite() || (key == "repeat_penalty" && n <= 0.0)
                || (["top_p", "min_p"].contains(&key) && !(0.0..=1.0).contains(&n)) {
                return Err(format!("invalid {key}"));
            }
        }
    }
    for key in ["top_k", "repeat_last_n"] {
        if let Some(v) = j.get(key) {
            let n = v.as_i64().ok_or_else(|| format!("{key} must be an integer"))?;
            if i32::try_from(n).is_err() { return Err(format!("{key} outside i32 range")); }
        }
    }
    for key in ["n_predict", "max_tokens", "max_completion_tokens"] {
        if let Some(v) = j.get(key) {
            if v.as_i64().map_or(true, |n| n < -1) { return Err(format!("invalid {key}")); }
        }
    }
    if let Some(v) = j.get("seed") {
        if v.as_u64().is_none() && v.as_i64() != Some(-1) { return Err("invalid seed".into()); }
    }
    for key in ["stream", "ignore_eos", "cache_prompt"] {
        if j.get(key).map_or(false, |v| !v.is_boolean()) { return Err(format!("{key} must be boolean")); }
    }
    if let Some(v) = j.get("stop") {
        if !v.is_string() && !v.as_array().map_or(false, |a| a.iter().all(Value::is_string)) { return Err("stop must be a string or string array".into()); }
    }
    let mut sampler = SamplerParams::default(); sampler.apply_json(j);
    let n = j.get("n_predict").or(j.get("max_tokens")).or(j.get("max_completion_tokens")).and_then(|v| v.as_i64()).unwrap_or(-1);
    let stop = match j.get("stop") { Some(Value::String(s)) => vec![s.clone()], Some(Value::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(), _ => vec![] };
    Ok(GenRequest { prompt, n_predict: if n < 0 { 4096 } else { n as usize }, sampler, stop, ignore_eos: j.get("ignore_eos").and_then(|v| v.as_bool()).unwrap_or(false) })
}

fn chunk(w: &mut dyn Write, b: &[u8]) -> bool { w.write_all(format!("{:x}\r\n", b.len()).as_bytes()).is_ok() && w.write_all(b).is_ok() && w.write_all(b"\r\n").is_ok() && w.flush().is_ok() }
fn sse(w: &mut dyn Write, v: &Value) -> bool { chunk(w, format!("data: {}\n\n", v).as_bytes()) }

pub fn serve(cfg: &ServerCfg, tk: &Tokenizer, tmpl: Option<&ChatTemplate>, eng: &mut dyn ServeEngine) {
    let have_head = eng.mtp_backend().is_some();
    let srv = tiny_http::Server::http(&cfg.addr).unwrap_or_else(|e| panic!("bind {}: {e}", cfg.addr));
    eprintln!("bonsai serving on http://{} (mtp head loaded: {have_head}, mtp default: {})", cfg.addr, have_head && cfg.mtp_default);
    let mut nreq: u64 = 0;
    let mut pcache: Vec<u32> = vec![];
    let mut last: (usize, usize, String) = (0, 0, String::new()); // /slots: (n_prompt, n_decoded, stop_type) of the last task
    for mut req in srv.incoming_requests() {
        nreq += 1;
        let url = req.url().split('?').next().unwrap_or("").to_string();
        let mut body = String::new();
        use std::io::Read;
        if req.as_reader().take(1_048_577).read_to_string(&mut body).is_err() || body.len() > 1_048_576 {
            respond(req, 400, json!({"error": {"message": "invalid UTF-8 or request body exceeds 1 MiB"}})); continue;
        }
        let j: Value = if body.trim().is_empty() { json!({}) } else { match serde_json::from_str(&body) { Ok(v) => v, Err(e) => { respond(req, 400, json!({"error": {"code": 400, "message": format!("invalid JSON: {e}"), "type": "invalid_request_error"}})); continue; } } };
        let stream = j.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
        match url.as_str() {
            "/health" | "/v1/health" => respond(req, 200, json!({"status": "ok"})),
            "/props" => respond(req, 200, json!({"default_generation_settings": {"n_ctx": cfg.n_ctx, "params": SamplerParams::default()}, "total_slots": 1, "mtp": {"head_loaded": have_head, "default": have_head && cfg.mtp_default}, "model_path": cfg.model_name, "chat_template": tmpl.map(|_| "gguf").unwrap_or("")})),
            // r012: llama-server-shaped /slots. The server is single-threaded, so a reply is only possible while idle.
            "/slots" => respond(req, 200, json!([{"id": 0, "id_task": nreq.saturating_sub(1), "n_ctx": cfg.n_ctx, "speculative": have_head && cfg.mtp_default, "is_processing": false,
                "params": SamplerParams::default(), "next_token": [{"has_next_token": false, "has_new_line": false, "n_remain": -1, "n_decoded": last.1, "stopping_word": ""}],
                "prompt_tokens_last": last.0, "stop_type_last": last.2}])),
            "/v1/models" | "/models" => respond(req, 200, json!({"object": "list", "data": [{"id": cfg.model_name, "object": "model", "created": now(), "owned_by": "bonsai"}]})),
            "/tokenize" => { let add = j.get("add_special").and_then(|v| v.as_bool()).unwrap_or(false); respond(req, 200, json!({"tokens": tk.encode(j.get("content").and_then(|v| v.as_str()).unwrap_or(""), add, true)})) }
            "/detokenize" => { match token_ids(j.get("tokens").unwrap_or(&Value::Null), tk) {
                Ok(ids) => respond(req, 200, json!({"content": tk.decode(&ids)})),
                Err(e) => respond(req, 400, json!({"error": {"message": e}})),
            } }
            "/completion" | "/completions" | "/v1/chat/completions" | "/chat/completions" => {
                let chat = url.ends_with("chat/completions");
                let rr = parse_request(&j, tk, tmpl, chat).and_then(|mut r| { validate_context(&mut r, &j, cfg.n_ctx)?; let m = resolve_mtp(&j, &r, have_head, cfg.mtp_default)?; Ok((r, m)) });
                let (r, use_mtp) = match rr { Ok(r) => r, Err(e) => { respond(req, 400, json!({"error": {"code": 400, "message": e, "type": "invalid_request_error"}})); continue; } };
                let use_cache = cfg.prompt_cache && !use_mtp && j.get("cache_prompt").and_then(|v| v.as_bool()).unwrap_or(true);
                if !use_cache { pcache.clear(); }
                let id = format!("chatcmpl-bonsai-{nreq}"); let created = now(); let model = cfg.model_name.clone();
                if stream {
                    let mut w = req.into_writer();
                    let _ = w.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n");
                    let mut n = 0usize; let np = r.prompt.len();
                    if chat { sse(&mut *w, &json!({"choices": [{"finish_reason": null, "index": 0, "delta": {"role": "assistant", "content": null}}], "created": created, "id": id, "model": model, "system_fingerprint": "bonsai", "object": "chat.completion.chunk"})); }
                    let mut cb = |t: u32, piece: &str| -> bool {
                        if t != u32::MAX { n += 1; }
                        let toks: Vec<u32> = if t == u32::MAX { vec![] } else { vec![t] };
                        if chat { if piece.is_empty() { return true; } sse(&mut *w, &json!({"choices": [{"finish_reason": null, "index": 0, "delta": {"content": piece}}], "created": created, "id": id, "model": model, "system_fingerprint": "bonsai", "object": "chat.completion.chunk"})) }
                        else { sse(&mut *w, &json!({"index": 0, "content": piece, "tokens": toks, "stop": false, "id_slot": 0, "tokens_predicted": n, "tokens_evaluated": np})) }
                    };
                    let res = run(eng, tk, &r, use_mtp, if use_cache { Some(&mut pcache) } else { None }, &mut cb);
                    last = (np, res.tokens.len(), res.stop_type.to_string());
                    if chat {
                        let fr = if res.stop_type == "limit" { "length" } else { "stop" };
                        sse(&mut *w, &json!({"choices": [{"finish_reason": fr, "index": 0, "delta": {}}], "created": created, "id": id, "model": model, "system_fingerprint": "bonsai", "object": "chat.completion.chunk",
                            "usage": {"completion_tokens": res.tokens.len(), "prompt_tokens": np, "total_tokens": np + res.tokens.len(), "prompt_tokens_details": {"cached_tokens": res.cached_tokens}}, "timings": timings(&res, np - res.cached_tokens), "mtp": mtp_json(&res)}));
                        chunk(&mut *w, b"data: [DONE]\n\n");
                    } else {
                        let mut fin = final_completion(&r, &res, &model); fin["content"] = json!(""); fin["tokens"] = json!([]);
                        sse(&mut *w, &fin);
                    }
                    let _ = w.write_all(b"0\r\n\r\n"); let _ = w.flush(); drop(w);
                } else {
                    let res = run(eng, tk, &r, use_mtp, if use_cache { Some(&mut pcache) } else { None }, &mut |_, _| true);
                    let np = r.prompt.len(); last = (np, res.tokens.len(), res.stop_type.to_string());
                    let v = if chat {
                        let fr = if res.stop_type == "limit" { "length" } else { "stop" };
                        json!({"choices": [{"finish_reason": fr, "index": 0, "message": {"role": "assistant", "content": res.text}}], "created": created, "model": model, "system_fingerprint": "bonsai", "object": "chat.completion",
                               "usage": {"completion_tokens": res.tokens.len(), "prompt_tokens": np, "total_tokens": np + res.tokens.len(), "prompt_tokens_details": {"cached_tokens": res.cached_tokens}}, "id": id, "timings": timings(&res, np - res.cached_tokens), "tokens": res.tokens, "mtp": mtp_json(&res)})
                    } else { final_completion(&r, &res, &model) };
                    respond(req, 200, v);
                }
            }
            _ => respond(req, 404, json!({"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}})),
        }
    }
}

fn final_completion(r: &GenRequest, res: &GenResult, model: &str) -> Value {
    json!({"index": 0, "content": res.text, "tokens": res.tokens, "id_slot": 0, "stop": true, "model": model, "tokens_predicted": res.tokens.len(), "tokens_evaluated": r.prompt.len(),
           "generation_settings": gen_settings(r, res.seed), "prompt": "", "has_new_line": res.text.contains('\n'), "truncated": false, "stop_type": res.stop_type, "stopping_word": res.stopping_word, "tokens_cached": res.cached_tokens,
           "timings": timings(res, r.prompt.len() - res.cached_tokens), "mtp": mtp_json(res)})
}

fn run(eng: &mut dyn ServeEngine, tk: &Tokenizer, r: &GenRequest, use_mtp: bool, cache: Option<&mut Vec<u32>>, cb: &mut dyn FnMut(u32, &str) -> bool) -> GenResult {
    let trace = std::env::var_os("BONSAI_REVIEW_TRACE").is_some();
    if trace { eprintln!("[api-review] {}", json!({"event":"request_start", "pid":std::process::id(), "mtp":use_mtp, "prompt":r.prompt, "budget":r.n_predict, "engine":eng.review_snapshot()})); }
    let start = std::time::Instant::now();
    let res = if use_mtp {
        generate_mtp(eng.mtp_backend().expect("validated MTP head"), tk, r, cb).expect("validated greedy MTP request")
    } else { crate::engine::generate_cached(eng.as_engine(), tk, r, cache, cb) };
    if trace { eprintln!("[api-review] {}", json!({"event":"request_end", "pid":std::process::id(), "mtp":use_mtp, "tokens":res.tokens, "stop_type":res.stop_type, "wall_ms":start.elapsed().as_secs_f64()*1e3, "timings":timings(&res,r.prompt.len()), "stats":res.mtp, "engine":eng.review_snapshot()})); }
    res
}

fn respond(req: tiny_http::Request, code: u16, v: Value) {
    let _ = req.respond(tiny_http::Response::from_string(v.to_string()).with_status_code(code).with_header(tiny_http::Header::from_bytes("Content-Type", "application/json; charset=utf-8").unwrap()));
}
