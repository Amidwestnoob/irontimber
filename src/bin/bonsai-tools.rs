//! bonsai-tools: the whole test/bench/report harness in Rust (replaces tools/*.py and tools/dell_*.sh).
//!   compare A.json B.json [tol]          per-position top-10 logits + greedy comparison (exit 1 on FAIL)
//!   vs-prism --model M --url U --out DIR [--n-predict N] [--ctx C] [--port P]
//!   api-shape --url PRISM --bonsai-url URL   compare llama-server JSON key structure (prism vs bonsai)
//!   dell-run [MODEL]                     full W9100 run -> report/<ts>/ (VRAM gate 7.4 GiB; else 8 layers, labelled)
//!   quick [MODEL]                        8-layer rust vs glsl A/B + per-kernel profile
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

fn bonsai_bin() -> PathBuf {
    let me = std::env::current_exe().unwrap();
    let p = me.with_file_name("bonsai");
    if p.exists() { p } else { PathBuf::from("bonsai") }
}
fn load(p: &Path) -> Value { serde_json::from_str(&fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap() }

struct Log { out: PathBuf, buf: String }
impl Log {
    fn line(&mut self, file: &str, s: &str) {
        println!("{s}");
        let mut f = fs::OpenOptions::new().create(true).append(true).open(self.out.join(file)).unwrap();
        writeln!(f, "{s}").unwrap();
        if file == "log.txt" || file == "summary.txt" { self.buf.push_str(s); self.buf.push('\n'); }
    }
}

// ---------- compare ----------
fn compare(a: &Value, b: &Value, tol: f64) -> (bool, String) {
    let mut out = String::new(); let mut ok = true; let mut worst = 0f64; let mut top1 = 0; let mut n = 0;
    let bpos: std::collections::HashMap<i64, &Value> = b["positions"].as_array().map(|v| v.iter().map(|p| (p["pos"].as_i64().unwrap(), p)).collect()).unwrap_or_default();
    let top = |p: &Value| -> Vec<(i64, f64)> { p["top"].as_array().unwrap().iter().map(|t| (t[0].as_i64().unwrap(), t[1].as_f64().unwrap())).collect() };
    for pa in a["positions"].as_array().into_iter().flatten() {
        let Some(pb) = bpos.get(&pa["pos"].as_i64().unwrap()) else { continue };
        let (ta, tb) = (top(pa), top(pb));
        n += 1; if ta[0].0 == tb[0].0 { top1 += 1; }
        let common: Vec<(f64, f64)> = ta.iter().filter_map(|(t, v)| tb.iter().find(|x| x.0 == *t).map(|x| (*v, x.1))).collect();
        let d = if common.is_empty() { 99.0 } else { common.iter().map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) };
        worst = worst.max(d);
        let _ = writeln!(out, "pos {:3}: top1 {:>6} vs {:>6}  top10-overlap {:2}/10  max|dlogit| {:.4}", pa["pos"], ta[0].0, tb[0].0, common.len(), d);
        if ta[0].0 != tb[0].0 || d > tol { ok = false; }
    }
    let g = |v: &Value| -> Vec<i64> { v["greedy"].as_array().map(|a| a.iter().filter_map(|x| x.as_i64()).collect()).unwrap_or_default() };
    let (ga, gb) = (g(a), g(b));
    let _ = writeln!(out, "greedy A {:?}\ngreedy B {:?}\ntop1 agreement {top1}/{n}, worst top-10 logit diff {worst:.4} (tol {tol})", ga, gb);
    let pass = ok && ga == gb && n > 0;
    out.push_str(if pass { "PASS\n" } else { "FAIL\n" });
    (pass, out)
}

// ---------- http ----------
fn agent() -> ureq::Agent { ureq::AgentBuilder::new().timeout(Duration::from_secs(3600)).build() }
fn post(url: &str, body: &Value) -> Result<Value, String> {
    agent().post(url).send_json(body.clone()).map_err(|e| e.to_string())?.into_json::<Value>().map_err(|e| e.to_string())
}
fn post_sse(url: &str, body: &Value) -> Result<Vec<String>, String> {
    let s = agent().post(url).send_json(body.clone()).map_err(|e| e.to_string())?.into_string().map_err(|e| e.to_string())?;
    Ok(s.split("\n\n").filter_map(|e| e.trim_start().strip_prefix("data: ").map(|x| x.trim().to_string())).collect())
}
struct Serve(std::process::Child, String);
impl Drop for Serve { fn drop(&mut self) { let _ = self.0.kill(); let _ = self.0.wait(); } }
fn start_serve(model: &str, extra: &[&str], port: u16, errlog: &Path) -> Result<Serve, String> {
    assert!(port != 18080 && port != 18081, "18080/18081 are protected service ports");
    let mut args = vec!["serve", "--model", model]; args.extend_from_slice(extra);
    let host = format!("127.0.0.1:{port}"); args.extend_from_slice(&["--host", &host]);
    let child = Command::new(bonsai_bin()).args(&args).stdout(Stdio::null()).stderr(fs::File::create(errlog).unwrap()).spawn().map_err(|e| e.to_string())?;
    let mut s = Serve(child, format!("http://{host}"));
    for _ in 0..1800 {
        if agent().get(&format!("{}/health", s.1)).timeout(Duration::from_secs(2)).call().is_ok() { return Ok(s); }
        if let Ok(Some(st)) = s.0.try_wait() { return Err(format!("bonsai serve exited ({st}); see {}", errlog.display())); }
        std::thread::sleep(Duration::from_millis(500));
    }
    Err("bonsai serve did not become healthy".into())
}
fn tokenize(model: &str, prompt: &str, chat: bool) -> Vec<u32> {
    let mut a = vec!["tokenize", "--model", model, "--prompt", prompt]; if chat { a.extend_from_slice(&["--chat", "--no-think"]); }
    let o = Command::new(bonsai_bin()).args(&a).stderr(Stdio::null()).output().unwrap();
    serde_json::from_slice(&o.stdout).unwrap()
}

// ---------- vs-prism ----------
fn vs_prism(model: &str, url: &str, out: &Path, np: usize, ctx: usize, port: u16) -> bool {
    let long = "Write a detailed explanation of how a transformer language model processes a prompt, step by step, covering tokenization, embeddings, attention, feed-forward layers and sampling. ".repeat(4);
    let prompts: Vec<(&str, Option<String>)> = vec![("capital", Some("The capital of France is".into())), ("code", Some("def fibonacci(n):\n    \"\"\"".into())), ("chat", None), ("long", Some(long))];
    let srv = match start_serve(model, &["--backend", "vulkan", "--ctx", &ctx.to_string()], port, &out.join("bonsai_serve.err")) { Ok(s) => s, Err(e) => { println!("{e}"); return false } };
    let mut rows = vec![]; let mut allok = true;
    for (name, text) in &prompts {
        let ids = match text { Some(t) => tokenize(model, t, false), None => tokenize(model, "Name three primary colors.", true) };
        let tok_ok: Value = match text { Some(t) => match post(&format!("{url}/tokenize"), &json!({"content": t, "add_special": true})) { Ok(v) => json!(v["tokens"] == json!(ids)), Err(e) => json!(format!("n/a ({e})")) }, None => Value::Null };
        let pr = match post(&format!("{url}/completion"), &json!({"prompt": ids, "n_predict": np, "temperature": 0, "top_k": 1, "cache_prompt": false, "return_tokens": true})) { Ok(v) => v, Err(e) => { println!("prism error: {e}"); allok = false; continue } };
        let bo = match post(&format!("{}/completion", srv.1), &json!({"prompt_ids": ids, "prompt": "", "n_predict": np, "temperature": 0})) { Ok(v) => v, Err(e) => { println!("bonsai error: {e}"); allok = false; continue } };
        let (pt, bt) = (pr["tokens"].as_array().cloned().unwrap_or_default(), bo["tokens"].as_array().cloned().unwrap_or_default());
        let k = pt.len().min(bt.len()); let mut m = 0; while m < k && pt[m] == bt[m] { m += 1; }
        let ok = pt == bt && k > 0; allok &= ok;
        let f = |v: &Value, k: &str| v["timings"][k].as_f64().unwrap_or(0.0);
        let row = json!({"prompt": name, "n_prompt": ids.len(), "tokenizer_match": tok_ok, "greedy_match_prefix": m, "compared": k, "identical": ok,
            "prism_pp_tps": f(&pr, "prompt_per_second"), "prism_gen_tps": f(&pr, "predicted_per_second"), "bonsai_pp_tps": f(&bo, "prompt_per_second"), "bonsai_gen_tps": f(&bo, "predicted_per_second"),
            "prism_text": pr["content"].as_str().unwrap_or("").chars().take(200).collect::<String>(), "bonsai_text": bo["content"].as_str().unwrap_or("").chars().take(200).collect::<String>()});
        println!("{row}"); rows.push(row);
    }
    drop(srv);
    fs::write(out.join("vs_prism.json"), serde_json::to_string_pretty(&rows).unwrap()).unwrap();
    let mut md = String::from("# bonsai (Rust/Vulkan) vs llama-prism on W9100\n\n| prompt | n_prompt | tok match | greedy identical | match prefix | prism PP t/s | bonsai PP t/s | prism gen t/s | bonsai gen t/s |\n|---|---|---|---|---|---|---|---|---|\n");
    for r in &rows {
        let g = |k: &str| r[k].as_f64().unwrap_or(0.0);
        let _ = writeln!(md, "| {} | {} | {} | {} | {}/{} | {:.2} | {:.2} | {:.2} | {:.2} |", r["prompt"].as_str().unwrap(), r["n_prompt"], r["tokenizer_match"], r["identical"], r["greedy_match_prefix"], r["compared"], g("prism_pp_tps"), g("bonsai_pp_tps"), g("prism_gen_tps"), g("bonsai_gen_tps"));
    }
    let _ = writeln!(md, "\nOverall greedy parity: {}", if allok { "PASS" } else { "FAIL" });
    fs::write(out.join("report.md"), md).unwrap();
    println!("{}", if allok { "PASS" } else { "FAIL" });
    allok
}

// ---------- api-shape ----------
fn shape(v: &Value, pre: &str, out: &mut BTreeSet<String>) {
    match v {
        Value::Object(m) => for (k, x) in m { let p = if pre.is_empty() { k.clone() } else { format!("{pre}.{k}") }; out.insert(p.clone()); shape(x, &p, out); },
        Value::Array(a) => if let Some(x) = a.first() { shape(x, &format!("{pre}[]"), out) },
        _ => {}
    }
}
fn keys(v: &Value) -> BTreeSet<String> { let mut s = BTreeSet::new(); shape(v, "", &mut s); s }
fn sse_keys(ev: &[String]) -> (BTreeSet<String>, bool) {
    let mut s = BTreeSet::new(); let done = ev.last().map(|e| e == "[DONE]").unwrap_or(false);
    for e in ev { if let Ok(v) = serde_json::from_str::<Value>(e) { shape(&v, "", &mut s); } }
    (s, done)
}
/// Every key prism returns that bonsai also must return (we ignore prism-only extras listed as optional).
fn api_shape(prism: &str, bonsai: &str) -> (bool, String) {
    const OPTIONAL: &[&str] = &["completion_probabilities", "prompt", "has_new_line", "truncated", "tokens_cached", "timings.cache_n", "timings.draft", "timings.draft_n", "timings.draft_n_accepted", "generation_settings.", "id_slot", "stop_word", "model", "system_fingerprint", "usage.prompt_tokens_details", "__verbose", "choices[].logprobs", "choices[].delta.reasoning_content", "choices[].message.reasoning_content", "id", "created", "choices[].message.tool_calls"];
    let opt = |k: &str| OPTIONAL.iter().any(|o| k == *o || (o.ends_with('.') && k.starts_with(o)) || k.starts_with(&format!("{o}.")));
    let ids = json!([760, 6511, 314, 9338, 369]);
    let cases: Vec<(&str, &str, Value, bool)> = vec![
        ("/completion", "/completion", json!({"prompt": ids, "n_predict": 4, "temperature": 0}), false),
        ("/completion stream", "/completion", json!({"prompt": ids, "n_predict": 4, "temperature": 0, "stream": true}), true),
        ("/v1/chat/completions", "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}), false),
        ("/v1/chat/completions stream", "/v1/chat/completions", json!({"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 4, "temperature": 0, "stream": true, "chat_template_kwargs": {"enable_thinking": false}}), true),
        ("/health", "/health", Value::Null, false),
    ];
    let mut rep = String::new(); let mut all = true;
    for (name, path, body, stream) in cases {
        let (pk, bk, extra) = if path == "/health" {
            let g = |u: &str| agent().get(&format!("{u}/health")).call().ok().and_then(|r| r.into_json::<Value>().ok()).unwrap_or(Value::Null);
            (keys(&g(prism)), keys(&g(bonsai)), String::new())
        } else if stream {
            let (p, pd) = sse_keys(&post_sse(&format!("{prism}{path}"), &body).unwrap_or_default());
            let (b, bd) = sse_keys(&post_sse(&format!("{bonsai}{path}"), &body).unwrap_or_default());
            if path.contains("chat") && pd != bd { all = false; }
            (p, b, format!(" [DONE]: prism {pd} bonsai {bd}"))
        } else {
            (keys(&post(&format!("{prism}{path}"), &body).unwrap_or(Value::Null)), keys(&post(&format!("{bonsai}{path}"), &body).unwrap_or(Value::Null)), String::new())
        };
        let missing: Vec<&String> = pk.iter().filter(|k| !bk.contains(*k) && !opt(k)).collect();
        let ok = !pk.is_empty() && missing.is_empty(); all &= ok;
        let _ = writeln!(rep, "{name}: {} (prism {} keys, bonsai {} keys{extra}){}", if ok { "OK" } else { "MISMATCH" }, pk.len(), bk.len(), if missing.is_empty() { String::new() } else { format!(" missing {:?}", missing) });
    }
    rep.push_str(if all { "API SHAPE PASS\n" } else { "API SHAPE FAIL\n" });
    (all, rep)
}

// ---------- bench-long: repeated full-model long-prompt prefill, bonsai vs prism ----------
fn stats(v: &[f64]) -> (f64, f64, f64, f64) {
    let n = v.len() as f64; let m = v.iter().sum::<f64>() / n;
    let sd = if v.len() > 1 { (v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1.0)).sqrt() } else { 0.0 };
    (m, sd, v.iter().cloned().fold(f64::MAX, f64::min), v.iter().cloned().fold(f64::MIN, f64::max))
}
fn bench_long(model: &str, prism: &str, out: &Path, reps: usize, port: u16, label: &str) -> String {
    let long = "Write a detailed explanation of how a transformer language model processes a prompt, step by step, covering tokenization, embeddings, attention, feed-forward layers and sampling. ".repeat(4);
    let ids = tokenize(model, &long, false);
    let srv = match start_serve(model, &["--backend", "vulkan", "--ctx", "2048"], port, &out.join(format!("bench_long_{label}.err"))) { Ok(s) => s, Err(e) => return format!("{label}: serve failed: {e}\n") };
    let body = json!({"prompt": ids, "prompt_ids": ids, "n_predict": 8, "temperature": 0, "top_k": 1, "cache_prompt": false});
    let _ = post(&format!("{}/completion", srv.1), &body); // warm-up (pipeline/cache), not counted
    let (mut b, mut p, mut bg) = (vec![], vec![], vec![]);
    for _ in 0..reps {
        if let Ok(v) = post(&format!("{}/completion", srv.1), &body) { b.push(v["timings"]["prompt_per_second"].as_f64().unwrap_or(0.0)); bg.push(v["timings"]["predicted_per_second"].as_f64().unwrap_or(0.0)); }
        if let Ok(v) = post(&format!("{prism}/completion"), &body) { p.push(v["timings"]["prompt_per_second"].as_f64().unwrap_or(0.0)); }
    }
    drop(srv);
    let f = |n: &str, v: &[f64]| { if v.is_empty() { return format!("{n}: no data\n"); } let (m, sd, lo, hi) = stats(v); format!("{n}: mean {m:.2} sd {sd:.2} min {lo:.2} max {hi:.2} n={} runs {:?}\n", v.len(), v.iter().map(|x| (x * 100.0).round() / 100.0).collect::<Vec<_>>()) };
    let r = format!("[{label}] long prompt {} tok, {reps} reps\n{}{}{}", ids.len(), f("  bonsai prefill tok/s", &b), f("  prism  prefill tok/s", &p), f("  bonsai gen(8) tok/s", &bg));
    fs::write(out.join(format!("bench_long_{label}.txt")), &r).unwrap();
    r
}

// ---------- dell-run / quick ----------
fn detect_model(arg: Option<String>) -> String {
    if let Some(m) = arg { return m; }
    for e in fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Ok(c) = fs::read(e.path().join("cmdline")) else { continue };
        let a: Vec<String> = c.split(|b| *b == 0).map(|s| String::from_utf8_lossy(s).into_owned()).collect();
        if !a.first().map(|x| x.contains("llama-server")).unwrap_or(false) { continue; }
        if let Some(i) = a.iter().position(|x| x == "-m" || x == "--model") { if let Some(m) = a.get(i + 1) { return m.clone(); } }
    }
    panic!("no model given and no running llama-server found");
}
fn vram_free_mib() -> Option<u64> {
    for e in fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let d = e.path().join("device");
        let r = |f: &str| fs::read_to_string(d.join(f)).ok()?.trim().parse::<u64>().ok();
        if let (Some(t), Some(u)) = (r("mem_info_vram_total"), r("mem_info_vram_used")) { return Some((t - u) / 1048576); }
    }
    None
}
fn rand_ids(n: usize, seed: u64) -> String {
    let mut s = seed; (0..n).map(|_| { s = s.wrapping_add(0x9E3779B97F4A7C15); let mut z = s; z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9); z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB); z ^= z >> 31; (100 + z % 149_900).to_string() }).collect::<Vec<_>>().join(",")
}
/// Run `bonsai run ...` with env, stdout -> json file, stderr -> err file; returns parsed JSON (Null on failure).
fn brun(out: &Path, name: &str, env: &[(&str, &str)], args: &[&str]) -> Value {
    let o = Command::new(bonsai_bin()).arg("run").args(args).envs(env.iter().cloned()).stderr(fs::File::create(out.join(format!("{name}.err"))).unwrap()).output().unwrap();
    fs::write(out.join(format!("{name}.json")), &o.stdout).unwrap();
    serde_json::from_slice(&o.stdout).unwrap_or(Value::Null)
}
fn tps(v: &Value, k: &str) -> f64 { v[k].as_f64().unwrap_or(0.0) }
fn prof_line(out: &Path, name: &str, re: &str) -> String { fs::read_to_string(out.join(format!("{name}.err"))).unwrap_or_default().lines().filter(|l| l.contains(re)).take(1).collect() }

fn dell_run(model: String, quick: bool) -> i32 {
    let root = std::env::current_dir().unwrap();
    let ts = String::from_utf8(Command::new("date").arg("+%Y%m%d-%H%M%S").output().unwrap().stdout).unwrap().trim().to_string();
    let out = root.join("report").join(if quick { format!("quick-{ts}") } else { ts }); fs::create_dir_all(&out).unwrap();
    let mut log = Log { out: out.clone(), buf: String::new() };
    let prism = std::env::var("PRISM_URL").unwrap_or("http://127.0.0.1:18080".into());
    let np: usize = std::env::var("N_PREDICT").ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let ctx: usize = std::env::var("CTX").ok().and_then(|s| s.parse().ok()).unwrap_or(2048);
    let port: u16 = std::env::var("BONSAI_PORT").ok().and_then(|s| s.parse().ok()).unwrap_or(18091);
    if std::env::var("BONSAI_VK_DEVICE").is_err() {
        let vi = Command::new("vulkaninfo").arg("--summary").output().map(|o| String::from_utf8_lossy(&o.stdout).into_owned()).unwrap_or_default();
        let d = ["W9100", "HAWAII", "Hawaii"].into_iter().find(|d| vi.contains(d)).unwrap_or("W9100");
        std::env::set_var("BONSAI_VK_DEVICE", d);
        fs::write(out.join("vulkaninfo.txt"), vi).unwrap();
    }
    let vram = vram_free_mib();
    log.line("env.txt", &format!("model: {model}\nbonsai: {}\nVRAM free: {:?} MiB\nBONSAI_VK_DEVICE={}", bonsai_bin().display(), vram, std::env::var("BONSAI_VK_DEVICE").unwrap()));
    let m = model.as_str(); let mut fails = 0;
    let p1 = "760,6511,314,9338,369,13,1049,374,1100,2014,0,220,17,15";
    let p7: String = p1.split(',').take(7).collect::<Vec<_>>().join(",");
    let p17 = rand_ids(17, 1); let p128 = rand_ids(128, 1); let p80: String = p128.split(',').take(80).collect::<Vec<_>>().join(",");

    if !quick {
        log.line("log.txt", "== GPU kernels vs Rust CPU f32 reference on the W9100 (first 8 layers)");
        brun(&out, "cpu_f32_L8", &[], &["--model", m, "--backend", "cpu", "--numerics", "f32", "--layers", "8", "--ids", p1, "--n-predict", "4", "--json", "--logits-all"]);
        brun(&out, "vk_L8", &[], &["--model", m, "--backend", "vulkan", "--layers", "8", "--ctx", "256", "--ids", p1, "--n-predict", "4", "--json", "--logits-all"]);
        let (ok, r) = compare(&load(&out.join("cpu_f32_L8.json")), &load(&out.join("vk_L8.json")), 0.02); fails += !ok as i32; log.line("gpu_vs_cpu_L8.txt", &r);
        log.line("log.txt", "== every rust-gpu kernel vs its GLSL twin on the W9100 (tol 1e-4)");
        let o = Command::new(bonsai_bin()).arg("kernel-parity").stderr(Stdio::null()).output().unwrap();
        let s = String::from_utf8_lossy(&o.stdout).into_owned(); fails += !o.status.success() as i32; log.line("kernel_parity_w9100.txt", &s);
    }
    log.line("log.txt", "== 8-layer speed A/B (kernels rust|glsl, small-prompt mode, tile threshold)");
    for (k, s, tm, pn, p) in [("rust", "mv1", "22", "P7", &p7), ("glsl", "mv1", "22", "P7", &p7), ("rust", "mv1", "22", "P17", &p17), ("glsl", "mv1", "22", "P17", &p17), ("rust", "mv1", "22", "P128", &p128), ("glsl", "mv1", "22", "P128", &p128)] {
        let name = format!("ab_{k}_{s}_{tm}_{pn}");
        let j = brun(&out, &name, &[("BONSAI_KERNELS", k), ("BONSAI_SMALL", s), ("BONSAI_TILE_MIN", tm)], &["--model", m, "--backend", "vulkan", "--layers", "8", "--ctx", "512", "--ids", p, "--n-predict", "33", "--json"]);
        log.line("ab_L8.txt", &format!("L8 kernels={k} small={s} tile_min={tm} prompt={pn}: prefill {:.1} tok/s, gen {:.2} tok/s, greedy {:?}", tps(&j, "pp_tps"), tps(&j, "gen_tps"), j["greedy"].as_array().map(|a| a.iter().take(8).cloned().collect::<Vec<_>>())));
    }
    log.line("log.txt", "== per-kernel GPU time, 8 layers, 128-token prefill + 8 decode steps (BONSAI_PROFILE)");
    for k in ["rust", "glsl"] { brun(&out, &format!("profile_{k}_L8"), &[("BONSAI_PROFILE", "1"), ("BONSAI_KERNELS", k)], &["--model", m, "--backend", "vulkan", "--layers", "8", "--ctx", "512", "--ids", &p128, "--n-predict", "9", "--json"]); }
    for kn in ["matmul_t2 ", "matvec_t2_1", "gdn ", "gdn_p", "hadamard", "argmax", "rmsnorm", "matvec_f32", "conv", "attn"] {
        log.line("profile_compare.txt", &format!("rust: {}\nglsl: {}", prof_line(&out, "profile_rust_L8", kn), prof_line(&out, "profile_glsl_L8", kn)));
    }
    if quick { println!("report dir: {}", out.display()); return 0; }

    log.line("log.txt", "== batched 80-token prefill logits (tiled) vs Rust CPU f32 (8 layers)");
    brun(&out, "cpu_f32_L8_p80", &[], &["--model", m, "--backend", "cpu", "--numerics", "f32", "--layers", "8", "--ids", &p80, "--n-predict", "2", "--json", "--logits-all"]);
    brun(&out, "vk_L8_p80_batched", &[], &["--model", m, "--backend", "vulkan", "--layers", "8", "--ctx", "256", "--ids", &p80, "--n-predict", "2", "--json", "--logits-batched"]);
    let (ok, r) = compare(&load(&out.join("cpu_f32_L8_p80.json")), &load(&out.join("vk_L8_p80_batched.json")), 0.02); fails += !ok as i32; log.line("prefill_tiled_vs_cpu_L8.txt", &r);

    let full = vram.map(|v| v >= 7578).unwrap_or(false) || std::env::var("FORCE_FULL").is_ok();
    let layers_arg: Vec<&str> = if full { vec![] } else { vec!["--layers", "8"] };
    let label = if full { "FULL 64-layer" } else { "8-layer ONLY (VRAM gate < 7.4 GiB) — numbers are NOT full-model; any full-model figure is an EXTRAPOLATION" };
    log.line("log.txt", &format!("== {label}"));
    let mut a = vec!["--model", m, "--backend", "cpu", "--numerics", "f32", "--ids", p1, "--n-predict", "4", "--json", "--logits-all"]; a.extend(&layers_arg);
    brun(&out, "cpu_f32_full", &[], &a);
    let mut a = vec!["--model", m, "--backend", "vulkan", "--ctx", "256", "--ids", p1, "--n-predict", "4", "--json", "--logits-all"]; a.extend(&layers_arg);
    brun(&out, "vk_full", &[], &a);
    let (ok, r) = compare(&load(&out.join("cpu_f32_full.json")), &load(&out.join("vk_full.json")), 0.1); fails += !ok as i32; log.line("gpu_vs_cpu_full.txt", &r);
    let ch = tokenize(m, "Name three primary colors.", true).iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");
    let mut a = vec!["--model", m, "--backend", "vulkan", "--ctx", "256", "--ids", &ch, "--n-predict", "16", "--json"]; a.extend(&layers_arg);
    let jb = brun(&out, "vk_full_batched", &[], &a); a.push("--logits-all");
    let jp = brun(&out, "vk_full_pertoken", &[], &a);
    let ok = jb["greedy"] == jp["greedy"] && !jb["greedy"].is_null(); fails += !ok as i32;
    log.line("batched_vs_pertoken_full.txt", &format!("batched {}\npertoken {}\n{}", jb["greedy"], jp["greedy"], if ok { "PASS" } else { "FAIL" }));

    if full {
        log.line("log.txt", &format!("== full model: bonsai (W9100) vs llama-prism ({prism})"));
        fails += !vs_prism(m, &prism, &out, np, ctx, port) as i32;
        log.line("log.txt", "== API shape vs llama-prism + sampling smoke (bonsai serve, full model)");
        match start_serve(m, &["--backend", "vulkan", "--ctx", "512"], port, &out.join("bonsai_serve_api.err")) {
            Ok(s) => {
                let (ok, r) = api_shape(&prism, &s.1); fails += !ok as i32; log.line("api_shape.txt", &r);
                let body = json!({"prompt": "Once upon a time", "n_predict": 24, "temperature": 0.8, "top_k": 40, "top_p": 0.95, "min_p": 0.05, "repeat_penalty": 1.1, "seed": 42});
                let (x, y) = (post(&format!("{}/completion", s.1), &body), post(&format!("{}/completion", s.1), &body));
                let ok = matches!((&x, &y), (Ok(a), Ok(b)) if a["tokens"] == b["tokens"]); fails += !ok as i32;
                log.line("sampling_smoke.txt", &format!("seed 42 run A: {:?}\nseed 42 run B: {:?}\nseeded determinism: {}", x.as_ref().map(|v| v["content"].clone()), y.as_ref().map(|v| v["content"].clone()), if ok { "PASS" } else { "FAIL" }));
            }
            Err(e) => { fails += 1; log.line("api_shape.txt", &format!("serve failed: {e}")); }
        }
    } else {
        log.line("log.txt", "SKIP vs-prism/api-shape full-model comparison (VRAM gate)");
    }
    log.line("summary.txt", &format!("failures: {fails}\nreport dir: {}", out.display()));
    fs::write(out.join("log_all.txt"), &log.buf).unwrap();
    if fails == 0 { 0 } else { 1 }
}

fn arg(a: &[String], k: &str) -> Option<String> { a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned()) }
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let code = match a.first().map(|s| s.as_str()) {
        Some("compare") => { let (ok, r) = compare(&load(Path::new(&a[1])), &load(Path::new(&a[2])), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(0.25)); print!("{r}"); !ok as i32 }
        Some("vs-prism") => {
            let out = PathBuf::from(arg(&a, "--out").unwrap_or(".".into())); fs::create_dir_all(&out).unwrap();
            !vs_prism(&arg(&a, "--model").expect("--model"), &arg(&a, "--url").unwrap_or("http://127.0.0.1:18080".into()), &out, arg(&a, "--n-predict").and_then(|s| s.parse().ok()).unwrap_or(64), arg(&a, "--ctx").and_then(|s| s.parse().ok()).unwrap_or(2048), arg(&a, "--port").and_then(|s| s.parse().ok()).unwrap_or(18091)) as i32
        }
        Some("api-shape") => { let (ok, r) = api_shape(&arg(&a, "--url").unwrap_or("http://127.0.0.1:18080".into()), &arg(&a, "--bonsai-url").expect("--bonsai-url")); print!("{r}"); !ok as i32 }
        Some("bench-long") => {
            let out = PathBuf::from(arg(&a, "--out").unwrap_or(".".into())); fs::create_dir_all(&out).unwrap();
            // A just-killed GPU process frees its VRAM asynchronously: wait up to 60 s before applying the gate.
            for _ in 0..60 { if vram_free_mib().map(|v| v >= 7578).unwrap_or(false) { break; } std::thread::sleep(Duration::from_secs(1)); }
            if vram_free_mib().map(|v| v < 7578).unwrap_or(true) && std::env::var("FORCE_FULL").is_err() { println!("SKIP: VRAM gate (< 7.4 GiB free: {:?} MiB)", vram_free_mib()); 3 } else {
            print!("{}", bench_long(&detect_model(arg(&a, "--model")), &arg(&a, "--url").unwrap_or("http://127.0.0.1:18080".into()), &out, arg(&a, "--repeats").and_then(|s| s.parse().ok()).unwrap_or(5), arg(&a, "--port").and_then(|s| s.parse().ok()).unwrap_or(18091), &arg(&a, "--label").unwrap_or("default".into()))); 0 }
        }
        // graft <base.gguf> <mtp.gguf> <out.gguf>: base trunk + the MTP block of mtp.gguf (renumbered to blk.<base_layers>).
        // sidecar <mtp.gguf> <out.gguf> [--no-embed]: head-only GGUF (no tokenizer, no trunk) for `run --mtp-head`.
        Some(c @ ("graft" | "sidecar")) => {
            use bonsai::gguf::{Gguf, raw_u32, write_gguf};
            let graft = c == "graft";
            let base = if graft { Some(Gguf::open(&a[1]).unwrap_or_else(|e| panic!("{e}"))) } else { None };
            let mg = Gguf::open(&a[if graft { 2 } else { 1 }]).unwrap_or_else(|e| panic!("{e}"));
            let out = &a[if graft { 3 } else { 2 }];
            let no_embed = a.iter().any(|x| x == "--no-embed");
            bonsai::mtp::validate_compatibility(&mg,base.as_ref().unwrap_or(&mg)).expect("incompatible converter input");
            let mbc = mg.u("qwen35.block_count").unwrap(); assert_eq!(mg.u("qwen35.nextn_predict_layers").unwrap(), 1, "mtp file has no single NextN block");
            let mil = mbc - 1; let src_pfx = format!("blk.{mil}.");
            let (kv_src, tl) = match &base { Some(b) => (b, b.u("qwen35.block_count").unwrap()), None => (&mg, mil) };
            assert_eq!(tl, mil, "trunk layer counts differ");
            let mut kvs: Vec<(String, Vec<u8>)> = Vec::new(); let mut have_nn = false;
            for (k, v) in &kv_src.kv_raw {
                // Retain tokenizer identity so sidecars can be validated against their trunk.
                if k == "qwen35.block_count" { kvs.push((k.clone(), raw_u32((tl + 1) as u32))); continue; }
                if k == "qwen35.nextn_predict_layers" { kvs.push((k.clone(), raw_u32(1))); have_nn = true; continue; }
                kvs.push((k.clone(), v.clone()));
            }
            if !have_nn { kvs.push(("qwen35.nextn_predict_layers".into(), raw_u32(1))); }
            if graft { for (k,v) in &mg.kv_raw { if k.starts_with("graft.") && !kvs.iter().any(|(n,_)|n==k) { kvs.push((k.clone(),v.clone())); } } }
            let mut ts: Vec<(&Gguf, &bonsai::gguf::TensorInfo, String)> = Vec::new();
            if let Some(b) = &base { for n in &b.tensor_order { ts.push((b, &b.tensors[n], n.clone())); } }
            for n in &mg.tensor_order {
                if let Some(rest) = n.strip_prefix(&src_pfx) {
                    if no_embed && rest == "nextn.embed_tokens.weight" { continue; }
                    ts.push((&mg, &mg.tensors[n], format!("blk.{tl}.{rest}")));
                }
            }
            let bytes = write_gguf(out, &kvs, &ts).unwrap_or_else(|e| panic!("{e}"));
            println!("wrote {out}: {bytes} bytes, {} tensors ({} head)", ts.len(), ts.iter().filter(|t| t.2.starts_with(&format!("blk.{tl}."))).count());
            let chk = Gguf::open(out).unwrap_or_else(|e| panic!("reopen: {e}"));
            let mut bad = 0; for (g, t, n) in &ts { if chk.tensor_data(n).map(|(_, d)| d != g.data(t)).unwrap_or(true) { bad += 1; } }
            println!("verify: {} tensors byte-identical, {bad} mismatched", ts.len() - bad); (bad != 0) as i32
        }
        Some("gguf-info") => {
            let g = bonsai::gguf::Gguf::open(&a[1]).unwrap_or_else(|e| panic!("{e}"));
            let filt = a.get(2).cloned().unwrap_or_default();
            let mut keys: Vec<_> = g.kv.keys().cloned().collect(); keys.sort();
            for k in keys { if k.contains(&filt) || filt.is_empty() { let v = format!("{:?}", g.kv[&k]); if v.len() < 200 { println!("kv {k} = {v}"); } else { println!("kv {k} = <{} chars>", v.len()); } } }
            for n in &g.tensor_order { if n.contains(&filt) || filt.is_empty() { let t = &g.tensors[n]; println!("t {n} {:?} {:?} off {}", t.dims, t.ty, t.offset); } }
            0
        }
        Some("cmp-ternary") => {
            // Do two GGUFs carry the same ternary values (e.g. PQ2_0 vs PTQ1_0 packings)? Compares the
            // lossless Packed2 repack (codes + scales) of every ternary tensor present in both.
            use bonsai::gguf::{Gguf, GgmlType}; use bonsai::quant::Packed2;
            let (ga, gb) = (Gguf::open(&a[1]).unwrap(), Gguf::open(&a[2]).unwrap());
            let (mut same, mut diff, mut skip) = (0, 0, 0);
            for n in &ga.tensor_order {
                let ta = &ga.tensors[n]; let Some(tb) = gb.tensors.get(n) else { skip += 1; continue };
                if ta.dims != tb.dims { diff+=1; println!("DIFF {n} dimensions"); continue; }
                let tern = |t: GgmlType| matches!(t, GgmlType::PQ2_0 | GgmlType::PTQ1_0);
                if !tern(ta.ty) || !tern(tb.ty) { if ta.ty == tb.ty && ga.data(ta) == gb.data(tb) { same += 1; } else if ta.ty == tb.ty { diff += 1; println!("DIFF {n} (raw {:?})", ta.ty); } else { skip += 1; } continue; }
                let rows: usize = ta.dims[1..].iter().product();
                let (pa, pb) = (Packed2::from_ggml(ta.ty, ga.data(ta), rows, ta.dims[0]), Packed2::from_ggml(tb.ty, gb.data(tb), rows, tb.dims[0]));
                if pa.words == pb.words && pa.scales == pb.scales { same += 1; } else {
                    let wd = pa.words.iter().zip(&pb.words).filter(|(x, y)| x != y).count(); let sd = pa.scales.iter().zip(&pb.scales).filter(|(x, y)| x != y).count();
                    diff += 1; println!("DIFF {n} {:?} vs {:?}: words differ {wd}/{}, scales differ {sd}/{}", ta.ty, tb.ty, pa.words.len(), pa.scales.len());
                }
            }
            println!("identical {same}, different {diff}, skipped {skip}");
            (diff > 0 || skip > 0) as i32
        }
        Some("mtp-api-bench") => mtp_api_bench(&arg(&a, "--url").unwrap_or("http://127.0.0.1:18093".into()), Path::new(&arg(&a, "--out").expect("--out")), arg(&a, "--n-predict").map(|s| s.parse().unwrap()).unwrap_or(64), arg(&a, "--reps").map(|s| s.parse().unwrap()).unwrap_or(3)),
        Some("mtp-api-isolation") => mtp_api_isolation(&arg(&a, "--url").unwrap_or("http://127.0.0.1:18093".into()), Path::new(&arg(&a, "--out").expect("--out"))),
        Some("dell-run") => dell_run(detect_model(a.get(1).cloned()), false),
        Some("quick") => dell_run(detect_model(a.get(1).cloned()), true),
        _ => { eprintln!("usage: bonsai-tools compare|vs-prism|api-shape|dell-run|quick ..."); 2 }
    };
    std::process::exit(code);
}

// ---------- warm MTP vs plain through one persistent server ----------
fn mtp_api_bench(url: &str, out: &Path, np: usize, reps: usize) -> i32 {
    std::fs::create_dir_all(out).unwrap();
    let long = "Write a detailed explanation of how a transformer language model processes a prompt, step by step, covering tokenization, embeddings, attention, feed-forward layers and sampling. ".repeat(4);
    let chat = |m: &str| json!({"messages": [{"role": "user", "content": m}], "chat_template_kwargs": {"enable_thinking": false}});
    let cases: Vec<(&str, &str, Value)> = vec![
        ("capital", "/completion", json!({"prompt": "The capital of France is"})),
        ("code", "/completion", json!({"prompt": "def fibonacci(n):\n    \"\"\""})),
        ("chat", "/v1/chat/completions", chat("Name three primary colors.")),
        ("long", "/completion", json!({"prompt": long})),
        ("chat-explain", "/v1/chat/completions", chat("Explain in two sentences why the sky is blue.")),
        ("chat-list", "/v1/chat/completions", chat("List five uses for a paperclip.")),
    ];
    let props = agent().get(&format!("{url}/props")).call().map(|r| r.into_json::<Value>().unwrap()).unwrap_or(Value::Null);
    let mut rows = vec![]; let mut md = String::from("| case | prompt tok | gen tok | identical | plain total ms | mtp total ms | plain prefill ms | mtp prefill ms | plain decode tok/s | mtp decode tok/s | decode speedup | total speedup | acceptance |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    let mut all_ok = true;
    for (name, path, base) in &cases {
        let mut body = base.clone(); body["temperature"] = json!(0); body["n_predict"] = json!(np); body["max_tokens"] = json!(np); body["ignore_eos"] = json!(false);
        let run = |m: bool| -> (Value, f64) { let mut b = body.clone(); b["mtp"] = json!(m); let t = Instant::now(); let v = post(&format!("{url}{path}"), &b).unwrap_or_else(|e| panic!("{name}: {e}")); (v, t.elapsed().as_secs_f64() * 1e3) };
        run(false); run(true); // warm both paths
        let mut samples: Vec<Value> = vec![]; let mut ref_toks = Value::Null; let mut ident = true;
        for r in 0..reps { for m in [false, true] {
            let (v, wall) = run(m);
            if ref_toks.is_null() { ref_toks = v["tokens"].clone(); } else if v["tokens"] != ref_toks { ident = false; }
            let t = &v["timings"]; let n = v["tokens"].as_array().map(|a| a.len()).unwrap_or(0);
            samples.push(json!({"rep": r, "mtp": m, "wall_ms": wall, "prompt_n": t["prompt_n"], "prompt_ms": t["prompt_ms"], "predicted_ms": t["predicted_ms"], "gen_tokens": n,
                "decode_tps": (n.saturating_sub(1)) as f64 / (t["predicted_ms"].as_f64().unwrap() / 1e3).max(1e-9), "mtp_stats": v["mtp"], "tokens": v["tokens"]}));
        } }
        let med = |m: bool, k: &str| -> f64 { let mut x: Vec<f64> = samples.iter().filter(|s| s["mtp"] == m).map(|s| s[k].as_f64().unwrap()).collect(); x.sort_by(|a, b| a.partial_cmp(b).unwrap()); x[x.len() / 2] };
        let acc: (u64, u64) = samples.iter().filter(|s| s["mtp"] == true).fold((0, 0), |a, s| (a.0 + s["mtp_stats"]["accepted"].as_u64().unwrap_or(0), a.1 + s["mtp_stats"]["drafted"].as_u64().unwrap_or(0)));
        let rate = acc.0 as f64 / (acc.1 as f64).max(1.0);
        all_ok &= ident;
        md += &format!("| {name} | {} | {} | {} | {:.0} | {:.0} | {:.0} | {:.0} | {:.2} | {:.2} | {:.2}x | {:.2}x | {:.1}% |\n", samples[0]["prompt_n"], samples[0]["gen_tokens"], if ident { "yes" } else { "NO" },
            med(false, "wall_ms"), med(true, "wall_ms"), med(false, "prompt_ms"), med(true, "prompt_ms"), med(false, "decode_tps"), med(true, "decode_tps"),
            med(true, "decode_tps") / med(false, "decode_tps"), med(false, "wall_ms") / med(true, "wall_ms"), rate * 100.0);
        rows.push(json!({"case": name, "identical": ident, "acceptance": rate, "samples": samples}));
    }
    md += &format!("\nMedians of {reps} warm reps per mode (after one warm-up each), n_predict {np}, one persistent server process. Overall identical: {all_ok}\n");
    std::fs::write(out.join("mtp_api_bench.json"), serde_json::to_string_pretty(&json!({"url": url, "props": props, "n_predict": np, "reps": reps, "cases": rows})).unwrap()).unwrap();
    std::fs::write(out.join("mtp_api_bench.md"), &md).unwrap();
    print!("{md}"); if all_ok { 0 } else { 1 }
}

/// Request-B-after-A isolation on a live persistent server. Must be run against a freshly started server.
fn mtp_api_isolation(url: &str, out: &Path) -> i32 {
    std::fs::create_dir_all(out).unwrap();
    let comp = format!("{url}/completion");
    let b = |m: bool| post(&comp, &json!({"prompt": "The capital of France is", "n_predict": 24, "temperature": 0, "mtp": m})).unwrap();
    let (rm, rp) = (b(true), b(false));
    let mut log = format!("fresh B: mtp tokens == plain tokens: {}\nB tokens {}\n", rm["tokens"] == rp["tokens"], rp["tokens"]);
    let mut ok = rm["tokens"] == rp["tokens"];
    let long = json!({"prompt": "Write a detailed explanation of how a transformer language model processes a prompt. ", "n_predict": 48, "temperature": 0});
    let with = |x: Value| { let mut v = long.clone(); for (k, y) in x.as_object().unwrap() { v[k] = y.clone(); } v };
    let cancel = |body: Value| { use std::io::BufRead; let r = agent().post(&comp).send_json(body).unwrap(); let mut rd = std::io::BufReader::new(r.into_reader()); let mut l = String::new();
        while rd.read_line(&mut l).unwrap() > 0 { if l.starts_with("data: ") { break; } l.clear(); } };
    let steps: Vec<(&str, Box<dyn Fn()>)> = vec![
        ("plain A", Box::new(|| { post(&comp, &with(json!({"mtp": false}))).unwrap(); })),
        ("mtp A", Box::new(|| { post(&comp, &with(json!({"mtp": true}))).unwrap(); })),
        ("sampled A", Box::new(|| { post(&comp, &with(json!({"temperature": 0.9, "seed": 11}))).unwrap(); })),
        ("mtp A stopped by stop string", Box::new(|| { post(&comp, &with(json!({"mtp": true, "stop": [" the"]}))).unwrap(); })),
        ("mtp A cancelled stream", Box::new(|| cancel(with(json!({"mtp": true, "stream": true}))))),
        ("plain A cancelled stream", Box::new(|| cancel(with(json!({"mtp": false, "stream": true}))))),
    ];
    for (name, a) in &steps { for m in [true, false] {
        a(); let r = b(m);
        let same = r["tokens"] == rp["tokens"] && (!m || r["mtp"]["accepted"] == rm["mtp"]["accepted"]);
        ok &= same; log += &format!("{name} -> B(mtp={m}): identical={same} accepted={} drafted={}\n", r["mtp"]["accepted"], r["mtp"]["drafted"]);
    } }
    log += if ok { "PASS\n" } else { "FAIL\n" };
    std::fs::write(out.join("mtp_api_isolation.txt"), &log).unwrap(); print!("{log}"); if ok { 0 } else { 1 }
}
