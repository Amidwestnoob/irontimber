//! parity-bench: black-box client benchmark of a llama-server-compatible endpoint (Prism or Rust Bonsai).
//! Same requests to both; writes <out>/<label>.json. `parity-bench compare a.json b.json` prints the matrix.
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

fn agent() -> ureq::Agent { ureq::AgentBuilder::new().timeout(Duration::from_secs(1800)).build() }
fn post(url: &str, b: &Value) -> Result<Value, String> { agent().post(url).send_json(b.clone()).map_err(|e| e.to_string())?.into_json().map_err(|e| e.to_string()) }
/// Streamed request: (ttft_ms, total_ms, concatenated content, tokens, final event).
fn stream(url: &str, b: &Value, cancel_after: Option<usize>) -> Result<(f64, f64, String, Vec<Value>, Value), String> {
    let t = Instant::now(); let mut b = b.clone(); b["stream"] = json!(true);
    let r = agent().post(url).send_json(b).map_err(|e| e.to_string())?;
    let rd = BufReader::new(r.into_reader()); let (mut ttft, mut text, mut toks, mut last, mut n) = (-1.0, String::new(), vec![], Value::Null, 0);
    for line in rd.lines() {
        let line = line.map_err(|e| e.to_string())?; let Some(d) = line.strip_prefix("data: ") else { continue };
        if d == "[DONE]" { break; }
        let v: Value = serde_json::from_str(d).map_err(|e| e.to_string())?;
        let c = v["content"].as_str().or(v["choices"][0]["delta"]["content"].as_str()).unwrap_or("");
        let has = !c.is_empty() || v["tokens"].as_array().map_or(false, |a| !a.is_empty());
        if has && ttft < 0.0 { ttft = t.elapsed().as_secs_f64() * 1e3; }
        text.push_str(c); if let Some(a) = v["tokens"].as_array() { if v["stop"] != true { toks.extend(a.iter().cloned()); } }
        n += 1; last = v; if cancel_after.map_or(false, |k| n >= k) { break; }
    }
    Ok((ttft, t.elapsed().as_secs_f64() * 1e3, text, toks, last))
}
fn filler(n_words: usize) -> String {
    let base = "The history of computing spans mechanical calculators, vacuum tubes, transistors and integrated circuits. Each generation reduced cost and increased speed, enabling new applications in science, business and communication. ";
    let mut s = String::new(); let mut i = 0; while s.split_whitespace().count() < n_words { s.push_str(&format!("Section {i}. {base}")); i += 1; } s
}
fn body(p: &str, n: usize) -> Value { json!({"prompt": p, "n_predict": n, "temperature": 0, "top_k": 1, "cache_prompt": false, "return_tokens": true, "mtp": false}) }

fn run(url: &str, label: &str, out: &str, long_targets: &[usize], extra: &Value) -> i32 {
    let comp = format!("{url}/completion");
    let mk = |p: &str, n: usize| { let mut b = body(p, n); if let Some(o) = extra.as_object() { for (k, v) in o { b[k] = v.clone(); } } b };
    let mut res = json!({"label": label, "url": url});
    res["props"] = agent().get(&format!("{url}/props")).call().ok().and_then(|r| r.into_json::<Value>().ok()).map(|mut p| { p.as_object_mut().map(|o| o.remove("chat_template")); p }).unwrap_or(Value::Null);
    for ep in ["/health", "/metrics", "/slots", "/v1/models", "/props"] { res["endpoints"][ep] = json!(match agent().get(&format!("{url}{ep}")).call() { Ok(r) => r.status(), Err(ureq::Error::Status(c, _)) => c, Err(_) => 0 }); }
    // warm-up
    let _ = post(&comp, &mk("Hello", 4));
    // 1) standard + quality prompts: greedy tokens (non-stream) and streamed TTFT/total
    let qa: Vec<(&str, &str, &str)> = vec![
        ("capital", "The capital of France is", ""), ("code", "def fibonacci(n):\n    \"\"\"", ""),
        ("long129", &*Box::leak(("Write a detailed explanation of how a transformer language model processes a prompt, step by step, covering tokenization, embeddings, attention, feed-forward layers and sampling. ".repeat(4)).into_boxed_str()), ""),
        ("qa_japan", "Q: What is the capital of Japan?\nA:", "Tokyo"), ("qa_h2o", "Q: What is the chemical formula of water?\nA:", "H2O"),
        ("qa_math", "Q: What is 12 times 12?\nA:", "144"), ("qa_planet", "Q: Which planet is known as the Red Planet?\nA:", "Mars"),
        ("qa_author", "Q: Who wrote the play Romeo and Juliet?\nA:", "Shakespeare"),
        ("code_rev", "# Python function that returns the reverse of a string s\ndef reverse(s):\n    return", "[::-1]"),
        ("code_max", "// C function returning the larger of two ints\nint max(int a, int b) {\n    return", "?"),
    ];
    let mut rows = vec![];
    for (name, p, expect) in &qa {
        let ns = post(&comp, &mk(p, 64)); let st = stream(&comp, &mk(p, 64), None);
        let (ns, st) = match (ns, st) { (Ok(a), Ok(b)) => (a, b), (a, b) => { rows.push(json!({"name": name, "error": format!("{:?} {:?}", a.err(), b.err())})); continue } };
        let t = &ns["timings"]; let ptoks = t["prompt_n"].as_f64().unwrap_or(0.0); let gen = ns["tokens"].as_array().map_or(0, |a| a.len());
        rows.push(json!({"name": name, "prompt_n": t["prompt_n"], "prefill_tps": t["prompt_per_second"], "prompt_ms": t["prompt_ms"], "gen_n": gen, "predicted_ms": t["predicted_ms"],
            "decode_tps": (gen.saturating_sub(1)) as f64 / (t["predicted_ms"].as_f64().unwrap_or(1.0) / 1e3).max(1e-9), "server_predicted_per_second": t["predicted_per_second"],
            "ttft_ms": st.0, "stream_total_ms": st.1, "stream_text_eq_nonstream": st.2 == ns["content"].as_str().unwrap_or("#"),
            "expect": expect, "pass": expect.is_empty() || ns["content"].as_str().unwrap_or("").contains(expect), "content": ns["content"], "tokens": ns["tokens"], "prompt_tokens_est": ptoks}));
        eprintln!("{label} {name} done");
    }
    res["prompts"] = json!(rows);
    // 2) long context: ~N-token prompts, 16 generated
    let mut lc = vec![];
    for &n in long_targets {
        let p = filler(n * 3 / 4 * 100 / 100); let t0 = Instant::now();
        let r = stream(&comp, &mk(&p, 16), None);
        let v = match r { Ok((ttft, total, text, _, last)) => { let tt = &last["timings"]; json!({"target": n, "prompt_n": tt["prompt_n"], "prefill_tps": tt["prompt_per_second"], "ttft_ms": ttft, "total_ms": total, "decode_tps": tt["predicted_per_second"], "text": text}) }
                          Err(e) => json!({"target": n, "error": e, "elapsed_ms": t0.elapsed().as_secs_f64() * 1e3}) };
        eprintln!("{label} long {n}: {v}"); lc.push(v);
    }
    res["long_context"] = json!(lc);
    // 3) cancel mid-stream, then back-to-back identical requests
    let base = post(&comp, &mk("The capital of France is", 32)).ok();
    let c = stream(&comp, &mk(&filler(300), 200), Some(4)).is_ok();
    let t = Instant::now(); let after = post(&comp, &mk("The capital of France is", 32)).ok(); let after_ms = t.elapsed().as_secs_f64() * 1e3;
    let b2b: Vec<Value> = (0..3).map(|_| post(&comp, &mk("The capital of France is", 32)).map(|v| v["tokens"].clone()).unwrap_or(Value::Null)).collect();
    res["cancel_restart"] = json!({"cancel_ok": c, "after_cancel_identical": base.as_ref().map(|b| &b["tokens"]) == after.as_ref().map(|a| &a["tokens"]), "after_cancel_ms": after_ms, "back_to_back_identical": b2b.windows(2).all(|w| w[0] == w[1] && !w[0].is_null())});
    // 4) concurrency: 2 simultaneous 64-token requests
    let t = Instant::now(); let mk2 = mk("Q: Name three primary colors.\nA:", 64);
    let hs: Vec<_> = (0..2).map(|_| { let u = comp.clone(); let b = mk2.clone(); std::thread::spawn(move || { let s = Instant::now(); let ok = post(&u, &b).is_ok(); (ok, s.elapsed().as_secs_f64() * 1e3) }) }).collect();
    let r: Vec<(bool, f64)> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    let single = { let s = Instant::now(); let _ = post(&comp, &mk2); s.elapsed().as_secs_f64() * 1e3 };
    res["concurrency"] = json!({"two_parallel_ms": r.iter().map(|x| x.1).collect::<Vec<_>>(), "all_ok": r.iter().all(|x| x.0), "wall_ms": t.elapsed().as_secs_f64() * 1e3 - single, "single_ms": single});
    // 5) chat endpoint smoke
    res["chat"] = post(&format!("{url}/v1/chat/completions"), &json!({"messages": [{"role": "user", "content": "Name three primary colors."}], "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}, "mtp": false}))
        .map(|v| json!({"content": v["choices"][0]["message"]["content"], "usage": v["usage"]})).unwrap_or_else(|e| json!({"error": e}));
    std::fs::create_dir_all(out).unwrap();
    std::fs::write(format!("{out}/{label}.json"), serde_json::to_string_pretty(&res).unwrap()).unwrap();
    0
}

fn compare(a: &Value, b: &Value) -> String {
    let mut s = format!("| prompt | prompt tok | {la} prefill tok/s | {lb} prefill tok/s | {la} decode tok/s | {lb} decode tok/s | {la} TTFT ms | {lb} TTFT ms | {la} total ms | {lb} total ms | exact match | {la} pass | {lb} pass |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n", la = a["label"].as_str().unwrap(), lb = b["label"].as_str().unwrap());
    let f = |v: &Value| v.as_f64().map(|x| format!("{x:.2}")).unwrap_or("-".into());
    let empty = vec![];
    for ra in a["prompts"].as_array().unwrap_or(&empty) {
        let Some(rb) = b["prompts"].as_array().and_then(|x| x.iter().find(|r| r["name"] == ra["name"])) else { continue };
        let (ta, tb) = (ra["tokens"].as_array(), rb["tokens"].as_array());
        let common = match (ta, tb) { (Some(x), Some(y)) => x.iter().zip(y).take_while(|(p, q)| p == q).count(), _ => 0 };
        let em = if ta == tb { "yes".to_string() } else { format!("first {common} tok") };
        s += &format!("| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n", ra["name"].as_str().unwrap(), ra["prompt_n"], f(&ra["prefill_tps"]), f(&rb["prefill_tps"]), f(&ra["decode_tps"]), f(&rb["decode_tps"]), f(&ra["ttft_ms"]), f(&rb["ttft_ms"]), f(&ra["stream_total_ms"]), f(&rb["stream_total_ms"]), em, ra["pass"], rb["pass"]);
    }
    s += "\nLong context:\n\n| target | side | prompt tok | prefill tok/s | TTFT ms | decode tok/s | note |\n|---|---|---|---|---|---|---|\n";
    for (v, side) in [(a, "a"), (b, "b")] { for r in v["long_context"].as_array().unwrap_or(&empty) {
        s += &format!("| {} | {} | {} | {} | {} | {} | {} |\n", r["target"], v["label"].as_str().unwrap_or(side), r["prompt_n"], f(&r["prefill_tps"]), f(&r["ttft_ms"]), f(&r["decode_tps"]), r["error"].as_str().unwrap_or(""));
    } }
    for k in ["cancel_restart", "concurrency", "endpoints", "chat"] { s += &format!("\n{k}: {} = {}\n         {} = {}\n", a["label"], a[k], b["label"], b[k]); }
    s
}

/// r012 quality set: chat questions with an unambiguous short answer, greedy, thinking off. Records content (for
/// exact cross-server comparison) and whether every expected substring (case-insensitive, any-of per '|') appears.
const QUALITY: &[(&str, &str, &str)] = &[
    ("cap_fr", "What is the capital of France? Answer in one word.", "paris"),
    ("cap_jp", "What is the capital of Japan? Answer in one word.", "tokyo"),
    ("h2o", "What is the chemical formula of water?", "h2o|h₂o"),
    ("arith", "What is 17 * 23? Reply with just the number.", "391"),
    ("arith2", "What is 144 divided by 12? Reply with just the number.", "12"),
    ("planet", "Which planet is known as the Red Planet? One word.", "mars"),
    ("author", "Who wrote Romeo and Juliet? Give just the name.", "shakespeare"),
    ("boil", "At sea level, at what temperature in Celsius does water boil? Just the number.", "100"),
    ("days", "How many days are in a leap year? Just the number.", "366"),
    ("py_len", "In Python, which built-in function returns the number of items in a list? One word.", "len"),
    ("opposite", "What is the opposite of 'hot'? One word.", "cold"),
    ("sky", "What color is a clear daytime sky? One word.", "blue"),
    ("prime", "Is 97 a prime number? Answer yes or no.", "yes"),
    ("continent", "On which continent is Egypt? One word.", "africa"),
    ("sql", "Write the SQL keyword used to remove rows from a table. One word.", "delete"),
    ("translate", "Translate 'thank you' into Spanish. Just the translation.", "gracias"),
];
fn quality(url: &str, label: &str, out: &str) -> i32 {
    let mut rows = vec![]; let mut pass = 0;
    for (name, q, exp) in QUALITY {
        let b = json!({"messages": [{"role": "user", "content": q}], "max_tokens": 48, "temperature": 0, "top_k": 1, "chat_template_kwargs": {"enable_thinking": false}});
        let r = post(&format!("{url}/v1/chat/completions"), &b);
        let content = r.as_ref().ok().and_then(|v| v["choices"][0]["message"]["content"].as_str().map(String::from)).unwrap_or_default();
        let ok = exp.split('|').any(|e| content.to_lowercase().contains(e)); pass += ok as i32;
        rows.push(json!({"name": name, "q": q, "expect": exp, "content": content, "ok": ok, "error": r.err()}));
    }
    let res = json!({"label": label, "url": url, "pass": pass, "n": QUALITY.len(), "rows": rows});
    std::fs::write(format!("{out}/quality_{label}.json"), serde_json::to_string_pretty(&res).unwrap()).unwrap();
    println!("quality {label}: {pass}/{}", QUALITY.len()); 0
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let g = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned());
    match a.first().map(|s| s.as_str()) {
        Some("compare") => { let l = |p: &str| serde_json::from_str::<Value>(&std::fs::read_to_string(p).unwrap()).unwrap(); print!("{}", compare(&l(&a[1]), &l(&a[2]))); }
        Some("run") => { let lt: Vec<usize> = g("--long").map(|s| s.split(',').filter(|x| !x.is_empty()).map(|x| x.parse().unwrap()).collect()).unwrap_or_default();
            let extra: Value = g("--extra").map(|s| serde_json::from_str(&s).unwrap()).unwrap_or(json!({}));
            std::process::exit(run(&g("--url").unwrap(), &g("--label").unwrap(), &g("--out").unwrap(), &lt, &extra)); }
        Some("quality") => std::process::exit(quality(&g("--url").unwrap(), &g("--label").unwrap(), &g("--out").unwrap())),
        _ => { eprintln!("parity-bench quality --url U --label L --out DIR | run --url U --label L --out DIR [--long 2000,4000,7000] [--extra JSON] | compare a.json b.json"); std::process::exit(2) }
    }
}
