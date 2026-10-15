//! End-to-end HTTP API tests: spawns `bonsai serve` (CPU backend, 2 layers, so it runs anywhere) and checks
//! llama-server response shapes, greedy == CLI greedy, seeded sampling determinism, and SSE streaming.
use serde_json::{json, Value};
use std::process::{Child, Command};

struct Srv(Child, String);
impl Drop for Srv { fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); } }
fn start(port: u16) -> Option<Srv> {
    let model = std::env::var("BONSAI_MODEL").ok()?;
    let child = Command::new(env!("CARGO_BIN_EXE_bonsai")).args(["serve", "--model", &model, "--backend", "cpu", "--numerics", "f32", "--layers", "2", "--host", &format!("127.0.0.1:{port}")]).stderr(std::process::Stdio::null()).spawn().unwrap();
    let url = format!("http://127.0.0.1:{port}");
    for _ in 0..600 { if ureq::get(&format!("{url}/health")).call().is_ok() { return Some(Srv(child, url)); } std::thread::sleep(std::time::Duration::from_millis(200)); }
    panic!("server did not start");
}
fn post(url: &str, body: Value) -> Value { ureq::post(url).send_json(body).unwrap().into_json().unwrap() }
fn sse(url: &str, body: Value) -> Vec<String> {
    let r = ureq::post(url).send_json(body).unwrap();
    assert!(r.header("Content-Type").unwrap_or("").starts_with("text/event-stream"));
    r.into_string().unwrap().split("\n\n").filter_map(|e| e.strip_prefix("data: ").map(String::from)).collect()
}

#[test]
fn http_api_matches_llama_server_shape_and_semantics() {
    let Some(s) = start(18697) else { eprintln!("BONSAI_MODEL not set; skipped"); return };
    let u = &s.1;
    assert_eq!(post(&format!("{u}/tokenize"), json!({"content": "Hello world", "add_special": true}))["tokens"].as_array().unwrap().len() > 0, true);
    // greedy completion == CLI greedy
    let c = post(&format!("{u}/completion"), json!({"prompt": "The capital of France is", "n_predict": 8, "temperature": 0}));
    for k in ["content", "tokens", "stop", "stop_type", "tokens_predicted", "tokens_evaluated", "generation_settings", "timings"] { assert!(c.get(k).is_some(), "missing {k}"); }
    for k in ["prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second"] { assert!(c["timings"].get(k).is_some(), "missing timings.{k}"); }
    let model = std::env::var("BONSAI_MODEL").unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_bonsai")).args(["run", "--model", &model, "--backend", "cpu", "--numerics", "f32", "--layers", "2", "--prompt", "The capital of France is", "--n-predict", "8", "--json"]).output().unwrap();
    let cli: Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(c["tokens"], cli["greedy"], "server greedy != CLI greedy");
    // r012 /slots: llama-server shape, idle, reflects the last task
    let sl: Value = ureq::get(&format!("{u}/slots")).call().unwrap().into_json().unwrap();
    assert_eq!(sl[0]["id"], 0); assert_eq!(sl[0]["is_processing"], false); assert!(sl[0]["n_ctx"].as_u64().unwrap() > 0);
    assert_eq!(sl[0]["next_token"][0]["n_decoded"], c["tokens_predicted"]);
    // seeded sampling: deterministic per seed
    let body = json!({"prompt": "Once upon a time", "n_predict": 12, "temperature": 1.0, "top_k": 50, "top_p": 0.95, "min_p": 0.0, "seed": 1234});
    let a = post(&format!("{u}/completion"), body.clone()); let b = post(&format!("{u}/completion"), body.clone());
    assert_eq!(a["tokens"], b["tokens"]); assert_eq!(a["generation_settings"]["seed"], 1234);
    // streaming /completion: concatenated pieces == non-stream content, last event has stop=true + timings
    let mut sb = body.clone(); sb["stream"] = json!(true);
    let ev = sse(&format!("{u}/completion"), sb);
    let parsed: Vec<Value> = ev.iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    let text: String = parsed.iter().map(|v| v["content"].as_str().unwrap_or("")).collect();
    assert_eq!(text, a["content"].as_str().unwrap());
    assert_eq!(parsed.last().unwrap()["stop"], true); assert!(parsed.last().unwrap().get("timings").is_some());
    // chat, non-stream and stream
    let msgs = json!([{"role": "user", "content": "Say hi."}]);
    let cc = post(&format!("{u}/v1/chat/completions"), json!({"messages": msgs, "max_tokens": 6, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(cc["object"], "chat.completion"); assert!(cc["choices"][0]["message"]["content"].is_string()); assert!(cc["usage"]["completion_tokens"].as_u64().unwrap() > 0);
    let ev = sse(&format!("{u}/v1/chat/completions"), json!({"messages": msgs, "max_tokens": 6, "temperature": 0, "stream": true, "chat_template_kwargs": {"enable_thinking": false}}));
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = ev[..ev.len() - 1].iter().map(|e| serde_json::from_str(e).unwrap()).collect();
    assert!(chunks.iter().all(|c| c["object"] == "chat.completion.chunk"));
    let st: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert_eq!(st, cc["choices"][0]["message"]["content"].as_str().unwrap());
    assert!(chunks.last().unwrap()["choices"][0]["finish_reason"].is_string());
    // stop strings
    let st = post(&format!("{u}/completion"), json!({"prompt": "The capital of France is", "n_predict": 8, "temperature": 0, "stop": [c["content"].as_str().unwrap().chars().nth(2).map(|ch| ch.to_string()).unwrap_or("x".into())]}));
    assert!(st["stop_type"] == "word" || st["stop_type"] == "limit");
}

/// r012 prompt cache: a prompt that strictly extends the previously fed sequence reuses state, reports
/// cached_tokens, and yields exactly the same greedy tokens as a full uncached prefill.
#[test]
fn prompt_cache_extension_is_greedy_identical() {
    let Some(model) = std::env::var("BONSAI_MODEL").ok() else { eprintln!("BONSAI_MODEL not set; skipped"); return };
    let port = 18698u16;
    let child = Command::new(env!("CARGO_BIN_EXE_bonsai")).args(["serve", "--model", &model, "--backend", "cpu", "--numerics", "f32", "--layers", "2", "--prompt-cache", "on", "--host", &format!("127.0.0.1:{port}")]).stderr(std::process::Stdio::null()).spawn().unwrap();
    let u = format!("http://127.0.0.1:{port}");
    let mut up = false; for _ in 0..600 { if ureq::get(&format!("{u}/health")).call().is_ok() { up = true; break; } std::thread::sleep(std::time::Duration::from_millis(200)); }
    let _g = Srv(child, u.clone()); assert!(up);
    let p1: Vec<u64> = post(&format!("{u}/tokenize"), json!({"content": "The capital of France is", "add_special": true}))["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let a = post(&format!("{u}/completion"), json!({"prompt": p1, "n_predict": 6, "temperature": 0}));
    assert_eq!(a["tokens_cached"], 0);
    let gen: Vec<u64> = a["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let mut p2 = p1.clone(); p2.extend_from_slice(&gen[..gen.len() - 1]); p2.extend_from_slice(&gen[gen.len() - 1..]); p2.extend_from_slice(&p1[1..]);
    let b = post(&format!("{u}/completion"), json!({"prompt": p2, "n_predict": 6, "temperature": 0}));
    assert_eq!(b["tokens_cached"].as_u64().unwrap() as usize, p1.len() + gen.len() - 1, "cache should cover prompt + all fed generated tokens");
    let c = post(&format!("{u}/completion"), json!({"prompt": p2, "n_predict": 6, "temperature": 0, "cache_prompt": false}));
    assert_eq!(c["tokens_cached"], 0);
    assert_eq!(b["tokens"], c["tokens"], "cached continuation != full prefill");
}
