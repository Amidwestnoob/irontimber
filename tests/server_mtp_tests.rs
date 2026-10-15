//! Persistent MTP API (r010): one server process loads trunk + MTP head once and serves many requests.
//! Proves (a) per-request mtp on/off gives identical output, (b) streaming/stop/EOS/cancel semantics hold
//! with MTP, (c) no per-request state leaks: request B is identical whether or not A (plain, MTP,
//! stopped early by a stop string, or a client-cancelled stream) ran before it.
//! Needs BONSAI_MODEL and BONSAI_MTP_HEAD; skipped otherwise.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::process::{Child, Command};

struct Srv(Child, String);
impl Drop for Srv { fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); } }
fn backend_args(vk: bool) -> Vec<&'static str> {
    if vk { vec!["--backend", "vulkan", "--device", "HAWAII", "--ctx", "8192", "--kv", "f16"] } else { vec!["--backend", "cpu", "--numerics", "f32", "--layers", "2", "--ctx", "256"] }
}
fn start(port: u16, vk: bool, extra: &[&str]) -> Option<Srv> {
    let model = std::env::var("BONSAI_MODEL").ok()?; let head = std::env::var("BONSAI_MTP_HEAD").ok()?;
    let mut args = vec!["serve", "--model", &model]; args.extend(backend_args(vk));
    let host = format!("127.0.0.1:{port}"); args.extend(["--host", &host, "--mtp", "--mtp-head", &head]); args.extend(extra);
    let child = Command::new(env!("CARGO_BIN_EXE_bonsai")).args(&args).stderr(std::process::Stdio::null()).spawn().unwrap();
    let url = format!("http://127.0.0.1:{port}");
    for _ in 0..1500 { if ureq::get(&format!("{url}/health")).call().is_ok() { return Some(Srv(child, url)); } std::thread::sleep(std::time::Duration::from_millis(200)); }
    panic!("server did not start");
}
fn post(url: &str, body: Value) -> Value { ureq::post(url).send_json(body).unwrap().into_json().unwrap() }
fn post_err(url: &str, body: Value) -> (u16, Value) {
    match ureq::post(url).send_json(body) { Err(ureq::Error::Status(c, r)) => (c, r.into_json().unwrap()), Ok(r) => (r.status(), r.into_json().unwrap()), Err(e) => panic!("{e}") }
}
fn sse(url: &str, body: Value) -> Vec<String> {
    let r = ureq::post(url).send_json(body).unwrap();
    r.into_string().unwrap().split("\n\n").filter_map(|e| e.strip_prefix("data: ").map(String::from)).collect()
}
/// Open a stream, read the first event, then drop the connection (client cancellation).
fn sse_cancel(url: &str, body: Value) {
    let r = ureq::post(url).send_json(body).unwrap();
    let mut rd = BufReader::new(r.into_reader()); let mut line = String::new();
    while rd.read_line(&mut line).unwrap() > 0 { if line.starts_with("data: ") { break; } line.clear(); }
}
const CHAT: &str = "Name three primary colors.";
fn prompts() -> Vec<Value> {
    vec![json!({"prompt": "The capital of France is"}), json!({"prompt": "def fibonacci(n):\n    \"\"\""}),
         json!({"prompt": "Write a detailed explanation of how a transformer language model processes a prompt. "})]
}
fn with(mut b: Value, extra: Value) -> Value { for (k, v) in extra.as_object().unwrap() { b[k] = v.clone(); } b }

fn suite(port: u16, vk: bool) {
    let Some(s) = start(port, vk, &[]) else { eprintln!("BONSAI_MODEL/BONSAI_MTP_HEAD unset: skipped"); return };
    let u = &s.1; let comp = format!("{u}/completion"); let chat = format!("{u}/v1/chat/completions");
    let props = ureq::get(&format!("{u}/props")).call().unwrap().into_json::<Value>().unwrap();
    assert_eq!(props["mtp"], json!({"head_loaded": true, "default": true}));

    // Request B on a fresh process: the reference for every isolation check below.
    let b_body = json!({"prompt": "The capital of France is", "n_predict": 10, "temperature": 0});
    let b_mtp_ref = post(&comp, with(b_body.clone(), json!({"mtp": true})));
    let b_plain_ref = post(&comp, with(b_body.clone(), json!({"mtp": false})));
    assert_eq!(b_mtp_ref["tokens"], b_plain_ref["tokens"], "mtp vs plain differ on B");
    assert_eq!(b_mtp_ref["mtp"]["enabled"], true); assert_eq!(b_plain_ref["mtp"]["enabled"], false);
    assert!(b_mtp_ref["mtp"]["drafted"].as_u64().unwrap() > 0);

    // (a) per-request on/off parity, non-stream + stream, completion + chat
    for p in prompts() {
        let body = with(p, json!({"n_predict": 9, "temperature": 0}));
        let on = post(&comp, with(body.clone(), json!({"mtp": true}))); let off = post(&comp, with(body.clone(), json!({"mtp": false})));
        assert_eq!(on["tokens"], off["tokens"]); assert_eq!(on["content"], off["content"]); assert_eq!(on["stop_type"], off["stop_type"]);
        let dflt = post(&comp, body.clone()); assert_eq!(dflt["mtp"]["enabled"], true, "server default on"); assert_eq!(dflt["tokens"], off["tokens"]);
        let ev = sse(&comp, with(body.clone(), json!({"mtp": true, "stream": true})));
        let parsed: Vec<Value> = ev.iter().map(|e| serde_json::from_str(e).unwrap()).collect();
        let text: String = parsed.iter().map(|v| v["content"].as_str().unwrap_or("")).collect();
        let toks: Vec<Value> = parsed.iter().flat_map(|v| if v["stop"] == true { vec![] } else { v["tokens"].as_array().cloned().unwrap_or_default() }).collect();
        assert_eq!(text, off["content"].as_str().unwrap()); assert_eq!(json!(toks), off["tokens"], "streamed MTP token order");
        assert_eq!(parsed.last().unwrap()["stop"], true); assert_eq!(parsed.last().unwrap()["mtp"]["enabled"], true);
    }
    let msgs = json!([{"role": "user", "content": CHAT}]);
    let cb = json!({"messages": msgs, "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}});
    let con = post(&chat, with(cb.clone(), json!({"mtp": true}))); let coff = post(&chat, with(cb.clone(), json!({"mtp": false})));
    assert_eq!(con["tokens"], coff["tokens"]); assert_eq!(con["choices"][0]["message"]["content"], coff["choices"][0]["message"]["content"]);
    let ev = sse(&chat, with(cb.clone(), json!({"mtp": true, "stream": true})));
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let st: String = ev[..ev.len() - 1].iter().map(|e| serde_json::from_str::<Value>(e).unwrap()).filter_map(|c| c["choices"][0]["delta"]["content"].as_str().map(String::from)).collect();
    assert_eq!(st, coff["choices"][0]["message"]["content"].as_str().unwrap());

    // (b) stop strings: same truncation, stop_type and streamed text with MTP; ignore_eos honoured
    let full = b_plain_ref["content"].as_str().unwrap().to_string();
    let stop_at: String = full.chars().skip(4).take(2).collect();
    if !stop_at.trim().is_empty() {
        let sb = with(b_body.clone(), json!({"stop": [stop_at]}));
        let son = post(&comp, with(sb.clone(), json!({"mtp": true}))); let soff = post(&comp, with(sb.clone(), json!({"mtp": false})));
        assert_eq!(son["content"], soff["content"]); assert_eq!(son["stop_type"], soff["stop_type"]); assert_eq!(son["tokens"], soff["tokens"]);
        let ev = sse(&comp, with(sb, json!({"mtp": true, "stream": true})));
        let text: String = ev.iter().map(|e| serde_json::from_str::<Value>(e).unwrap()["content"].as_str().unwrap_or("").to_string()).collect();
        assert_eq!(text, soff["content"].as_str().unwrap());
    }
    // every budget 1..=7 (odd/even, final single forward) identical
    for n in 1..=7 { let b = with(b_body.clone(), json!({"n_predict": n}));
        assert_eq!(post(&comp, with(b.clone(), json!({"mtp": true})))["tokens"], post(&comp, with(b, json!({"mtp": false})))["tokens"], "n={n}"); }

    // (c) isolation: run each A variant, then B with and without MTP; must equal fresh-process B
    let long_a = with(prompts()[2].clone(), json!({"n_predict": 24, "temperature": 0}));
    let variants: Vec<(&str, Box<dyn Fn()>)> = vec![
        ("plain A", Box::new(|| { post(&comp, with(long_a.clone(), json!({"mtp": false}))); })),
        ("mtp A", Box::new(|| { post(&comp, with(long_a.clone(), json!({"mtp": true}))); })),
        ("sampled A", Box::new(|| { post(&comp, with(long_a.clone(), json!({"temperature": 1.0, "seed": 7}))); })),
        ("mtp A stopped by stop string", Box::new(|| { post(&comp, with(long_a.clone(), json!({"mtp": true, "stop": [" "]}))); })),
        ("mtp A cancelled stream", Box::new(|| { sse_cancel(&comp, with(long_a.clone(), json!({"mtp": true, "stream": true}))); })),
        ("plain A cancelled stream", Box::new(|| { sse_cancel(&comp, with(long_a.clone(), json!({"mtp": false, "stream": true}))); })),
        ("mtp chat A", Box::new(|| { post(&chat, with(cb.clone(), json!({"mtp": true, "max_tokens": 20}))); })),
    ];
    for (name, a) in &variants {
        for m in [true, false] {
            a();
            let b = post(&comp, with(b_body.clone(), json!({"mtp": m})));
            assert_eq!(b["tokens"], b_plain_ref["tokens"], "B (mtp={m}) changed after {name}");
            if m { assert_eq!(b["mtp"]["accepted"], b_mtp_ref["mtp"]["accepted"], "MTP acceptance changed after {name}: head state leaked"); }
        }
    }

    // errors: explicit mtp + sampling, non-boolean
    let (c, e) = post_err(&comp, with(b_body.clone(), json!({"mtp": true, "temperature": 0.8})));
    assert_eq!(c, 400); assert!(e["error"]["message"].as_str().unwrap().contains("greedy"));
    assert_eq!(post_err(&comp, with(b_body.clone(), json!({"mtp": "yes"}))).0, 400);
    // server default applies to greedy only: sampled request without "mtp" runs plain
    assert_eq!(post(&comp, with(b_body.clone(), json!({"temperature": 0.8, "seed": 3})))["mtp"]["enabled"], false);
    drop(s);

    // --mtp-default off: requests without "mtp" run plain; explicit true still works
    let s = start(port + 1, vk, &["--mtp-default", "off"]).unwrap(); let comp = format!("{}/completion", s.1);
    let d = post(&comp, b_body.clone()); assert_eq!(d["mtp"]["enabled"], false); assert_eq!(d["tokens"], b_plain_ref["tokens"]);
    assert_eq!(post(&comp, with(b_body.clone(), json!({"mtp": true})))["tokens"], b_plain_ref["tokens"]);
}

#[test] fn persistent_mtp_api_cpu() { suite(18611, false); }
#[cfg(feature = "vulkan")]
#[test] fn persistent_mtp_api_vulkan() { suite(18613, true); }

#[test]
fn plain_server_rejects_explicit_mtp() {
    let Ok(model) = std::env::var("BONSAI_MODEL") else { return };
    let child = Command::new(env!("CARGO_BIN_EXE_bonsai")).args(["serve", "--model", &model, "--backend", "cpu", "--numerics", "f32", "--layers", "2", "--host", "127.0.0.1:18615"]).stderr(std::process::Stdio::null()).spawn().unwrap();
    let s = Srv(child, "http://127.0.0.1:18615".into());
    for _ in 0..1500 { if ureq::get(&format!("{}/health", s.1)).call().is_ok() { break; } std::thread::sleep(std::time::Duration::from_millis(200)); }
    let (c, e) = post_err(&format!("{}/completion", s.1), json!({"prompt": "Hi", "n_predict": 2, "temperature": 0, "mtp": true}));
    assert_eq!(c, 400); assert!(e["error"]["message"].as_str().unwrap().contains("MTP head"));
    assert_eq!(post(&format!("{}/completion", s.1), json!({"prompt": "Hi", "n_predict": 2, "temperature": 0, "mtp": false}))["mtp"]["enabled"], false);
}
