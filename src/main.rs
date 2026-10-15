//! bonsai CLI: tokenize | run | serve
use bonsai::{chat::ChatTemplate, cpu, gguf::Gguf, tokenizer::Tokenizer};
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

pub struct Args { pub cmd: String, pub kv: HashMap<String, String>, pub flags: Vec<String> }
fn parse() -> Args {
    let mut a = std::env::args().skip(1);
    let cmd = a.next().unwrap_or_else(|| "help".into());
    let mut kv = HashMap::new(); let mut flags = vec![]; let rest: Vec<String> = a.collect();
    let mut i = 0;
    while i < rest.len() {
        let k = rest[i].trim_start_matches("--").to_string();
        if i + 1 < rest.len() && !rest[i + 1].starts_with("--") { kv.insert(k, rest[i + 1].clone()); i += 2; } else { flags.push(k); i += 1; }
    }
    Args { cmd, kv, flags }
}

use bonsai::engine::{CpuEngine, GenRequest, generate};
use bonsai::sampling::{argmax, SamplerParams};

fn top(v: &[f32], k: usize) -> Vec<(u32, f32)> { let mut i: Vec<usize> = (0..v.len()).collect(); i.select_nth_unstable_by(k, |a, b| v[*b].partial_cmp(&v[*a]).unwrap()); i.truncate(k); i.sort_by(|a, b| v[*b].partial_cmp(&v[*a]).unwrap()); i.into_iter().map(|j| (j as u32, v[j])).collect() }

fn main() {
    let a = parse();
    for k in ["mtp-default", "mtp-head"] {
        assert!(!a.flags.iter().any(|f| f == k), "--{k} requires a value");
    }
    if a.flags.iter().any(|f| f == "mtp") {
        assert!(a.cmd=="run"||a.cmd=="serve","--mtp applies to run|serve");
        if let Some(v)=a.kv.get("mtp-default") { assert!(v=="on"||v=="off","--mtp-default on|off, got {v}"); }
        if a.cmd=="run" { for k in ["temp","top-k","top-p","min-p","repeat-penalty","repeat-last-n","seed","stop"] { assert!(!a.kv.contains_key(k),"--mtp does not support --{k}; use plain generation"); }
        for k in ["stream","logits-all","logits-batched"] { assert!(!a.flags.iter().any(|f|f==k),"--mtp does not support --{k}"); } }
    } else { assert!(!a.kv.contains_key("mtp-head"),"--mtp-head requires --mtp"); assert!(!a.kv.contains_key("mtp-default"),"--mtp-default requires --mtp"); }
    if let Some(v) = a.kv.get("kv") { if v != "f16" && v != "f32" { eprintln!("--kv must be f16 or f32"); std::process::exit(2); } std::env::set_var("BONSAI_KV", v); }
    if a.cmd == "help" { eprintln!("bonsai tokenize|run|serve --model M [--prompt P|--ids 1,2] [--chat] [--no-think] [--n-predict N] [--backend cpu|vulkan] [--layers N] [--numerics ggml|f32] [--json] [--logits-all|--logits-batched] [--stream] [--temp T --top-k K --top-p P --min-p M --repeat-penalty R --repeat-last-n N --seed S --stop STR] [--kv f32|f16] [--ctx N] [--prompt-cache on|off] | serve [--host 127.0.0.1:18081] | kernel-parity"); return; }
    #[cfg(feature = "vulkan")]
    if a.cmd == "prefill-gqa-bench" { print!("{}",bonsai::vk::kparity::prefill_gqa_bench(a.kv.get("device").map(|s|s.as_str())).expect("GQA progressive screen/parity"));return; }
    #[cfg(feature = "vulkan")]
    if a.cmd == "prefill-attn-bench" {
        for (ctx,nt) in [(16,1),(16,3),(16,7),(16,8),(16,22),(16,43),(16,64),(500,8),(2000,8),(7000,8)] {
            print!("{}",bonsai::vk::kparity::prefill_attention_bench(a.kv.get("device").map(|s|s.as_str()),ctx,nt,3).expect("prefill attention screen/parity"));
        }
        return;
    }
    #[cfg(feature = "vulkan")]
    if a.cmd == "prefill-bench" {
        let nt:u32=a.kv.get("ntok").map(|s|s.parse().unwrap()).unwrap_or(64);
        let shapes=if let (Some(m),Some(k))=(a.kv.get("m"),a.kv.get("k")) {vec![(m.parse().unwrap(),k.parse().unwrap())]} else {vec![(17408,5120),(5120,17408),(12288,5120)]};
        let iters=a.kv.get("iters").map(|s|s.parse().unwrap()).unwrap_or(5);
        for (m,k) in shapes {
            print!("{}",bonsai::vk::kparity::prefill_bench(a.kv.get("device").map(|s|s.as_str()),m,k,nt,iters).expect("prefill screen/parity"));
        }
        return;
    }
    #[cfg(feature = "vulkan")]
    if a.cmd == "mv-bench" {
        let g = |k: &str, d: u32| a.kv.get(k).map(|s| s.parse().unwrap()).unwrap_or(d);
        for (m, k) in [(17408, 5120), (5120, 17408), (12288, 5120)] {
            print!("{}", bonsai::vk::kparity::mv_bench(a.kv.get("device").map(|s| s.as_str()), m, k, g("iters", 200), g("ntok", 1)).unwrap());
        }
        return;
    }
    if a.cmd == "attn-bench" {
        let it: u32 = a.kv.get("iters").map(|s| s.parse().unwrap()).unwrap_or(20);
        for (ctx, nt) in [(16, 1), (500, 1), (2000, 1), (4000, 1), (7000, 1), (2000, 2), (2000, 64), (6000, 64)] {
            print!("{}", bonsai::vk::kparity::attn_bench(a.kv.get("device").map(|s| s.as_str()), ctx, nt, it).unwrap_or_else(|e| format!("{e}\n")));
        }
        return;
    }
    if a.cmd == "kernel-parity" {
        let (r, ok) = bonsai::vk::kparity::run(a.kv.get("device").map(|s| s.as_str()), 1e-4).unwrap();
        print!("{r}"); std::process::exit(if ok { 0 } else { 1 });
    }
    let model = a.kv.get("model").cloned().unwrap_or_else(|| std::env::var("BONSAI_MODEL").expect("--model"));
    let g = Gguf::open(&model).unwrap();
    let tk = Tokenizer::from_gguf(&g).unwrap();
    let prompt_ids = |a: &Args| -> Vec<u32> {
        if let Some(ids) = a.kv.get("ids") { return ids.split(',').map(|s| s.trim().parse().unwrap()).collect(); }
        let p = a.kv.get("prompt").cloned().unwrap_or_default();
        if a.flags.iter().any(|f| f == "chat") {
            let t = ChatTemplate::from_gguf(&g).unwrap();
            let extra = if a.flags.iter().any(|f| f == "no-think") { serde_json::json!({"enable_thinking": false}) } else { serde_json::json!({}) };
            let s = t.render(&serde_json::json!([{"role":"user","content": p}]), true, &extra).unwrap();
            tk.encode(&s, true, true)
        } else { tk.encode(&p, true, a.flags.iter().any(|f| f == "parse-special")) }
    };
    match a.cmd.as_str() {
        "tokenize" => { let ids = prompt_ids(&a); println!("{}", serde_json::to_string(&ids).unwrap()); }
        "detokenize" => { let ids = prompt_ids(&a); print!("{}", tk.decode(&ids)); }
        "chat-format" => { let t = ChatTemplate::from_gguf(&g).unwrap(); let p = a.kv.get("prompt").cloned().unwrap_or_default();
            let extra = if a.flags.iter().any(|f| f == "no-think") { serde_json::json!({"enable_thinking": false}) } else { serde_json::json!({}) };
            print!("{}", t.render(&serde_json::json!([{"role":"user","content": p}]), true, &extra).unwrap()); }
        "run" | "serve" => {
            if a.cmd == "run" {
                let ids=prompt_ids(&a); let n:usize=a.kv.get("n-predict").map(|s|s.parse().expect("invalid --n-predict")).unwrap_or(32);
                let ctx:usize=a.kv.get("ctx").map(|s|s.parse().expect("invalid --ctx")).unwrap_or(4096);
                assert!(!ids.is_empty() && ids.iter().all(|&t|(t as usize)<tk.tokens.len()),"empty or out-of-vocabulary prompt");
                assert!(ids.len()<=ctx && n.saturating_sub(1)<=ctx-ids.len(),"requested generation exceeds context capacity");
            }
            let layers = a.kv.get("layers").map(|s| s.parse().unwrap());
            let backend = a.kv.get("backend").map(|s| s.as_str()).unwrap_or("cpu");
            if a.cmd == "run" && a.flags.iter().any(|f| f == "mtp") { run_mtp(&a, &g, &tk, layers, backend, &prompt_ids(&a)); return; }
            if a.cmd == "serve" && a.flags.iter().any(|f| f == "mtp") { serve_mtp(&a, &g, &tk, layers, backend, &model); return; }
            let mut eng: Box<dyn bonsai::engine::ServeEngine> = match backend {
                "cpu" => {
                    let num = if a.kv.get("numerics").map(|s| s.as_str()) == Some("f32") { cpu::Numerics::F32 } else { cpu::Numerics::Ggml };
                    if let Some(t) = a.kv.get("threads") { rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().ok(); }
                    let m = cpu::Model::load(&g, layers).unwrap(); let st = cpu::State::new(&m.cfg);
                    Box::new(CpuEngine { m, st, num })
                }
                #[cfg(feature = "vulkan")]
                "vulkan" | "vk" => Box::new(bonsai::vk::VkEngine::new(&g, layers, a.kv.get("ctx").map(|s| s.parse().unwrap()).unwrap_or(4096), a.kv.get("device").map(|s| s.as_str())).unwrap()),
                b => panic!("unknown backend {b}"),
            };
            if a.cmd == "serve" {
                let tmpl = ChatTemplate::from_gguf(&g);
                let name = std::path::Path::new(&model).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
                let cfg = bonsai::server::ServerCfg { addr: a.kv.get("host").cloned().unwrap_or("127.0.0.1:18081".into()), model_name: name, n_ctx: a.kv.get("ctx").map(|s| s.parse().unwrap()).unwrap_or(4096), mtp_default: false, prompt_cache: a.kv.get("prompt-cache").map_or(false, |v| v == "on") };
                bonsai::server::serve(&cfg, &tk, tmpl.as_ref(), eng.as_mut()); return;
            }
            // Sampled generation (any of --temp/--top-k/--top-p/--min-p/--repeat-penalty/--seed): stream text as it is produced.
            let sampled = ["temp", "top-k", "top-p", "min-p", "repeat-penalty", "seed"].iter().any(|k| a.kv.contains_key(*k));
            if sampled || a.flags.iter().any(|f| f == "stream") {
                let mut p = if sampled { SamplerParams::default() } else { SamplerParams::greedy() };
                let gf = |k: &str| a.kv.get(k).map(|s| s.parse::<f64>().unwrap());
                if let Some(v) = gf("temp") { p.temperature = v as f32; } if let Some(v) = gf("top-k") { p.top_k = v as i32; } if let Some(v) = gf("top-p") { p.top_p = v as f32; }
                if let Some(v) = gf("min-p") { p.min_p = v as f32; } if let Some(v) = gf("repeat-penalty") { p.repeat_penalty = v as f32; } if let Some(v) = gf("repeat-last-n") { p.repeat_last_n = v as i32; }
                if let Some(v) = a.kv.get("seed") { p.seed = v.parse().unwrap(); }
                let r = GenRequest { prompt: prompt_ids(&a), n_predict: a.kv.get("n-predict").map(|s| s.parse().unwrap()).unwrap_or(32), sampler: p, stop: a.kv.get("stop").map(|s| vec![s.clone()]).unwrap_or_default(), ignore_eos: false };
                let json = a.flags.iter().any(|f| f == "json");
                let res = generate(eng.as_engine(), &tk, &r, &mut |_, piece| { if !json { print!("{piece}"); std::io::stdout().flush().ok(); } true });
                let pp = r.prompt.len() as f64 / (res.prompt_ms / 1e3); let gt = res.decode_forwards as f64 / (res.predicted_ms / 1e3).max(1e-9);
                if json { println!("{}", serde_json::json!({"backend": backend, "prompt_ids": r.prompt, "tokens": res.tokens, "greedy": res.tokens, "text": res.text, "seed": res.seed, "stop_type": res.stop_type, "pp_tokens": r.prompt.len(), "pp_tps": pp, "gen_tokens": res.tokens.len(), "gen_tps": gt})); }
                else { eprintln!("\n[prompt {} tok, {:.2} tok/s | gen {} tok, {:.2} tok/s | seed {}]", r.prompt.len(), pp, res.tokens.len(), gt, res.seed); }
                return;
            }
            let ids = prompt_ids(&a);
            let n_pred: usize = a.kv.get("n-predict").map(|s| s.parse().unwrap()).unwrap_or(32);
            if n_pred == 0 { println!("{}",serde_json::json!({"backend":backend,"prompt_ids":ids,"greedy":[],"text":"","gen_tokens":0,"pp_seconds":0.0,"gen_seconds":0.0,"timed_decode_forwards":0})); return; }
            let json = a.flags.iter().any(|f| f == "json");
            let logits_batched = a.flags.iter().any(|f| f == "logits-batched");
            let logits_all = logits_batched || a.flags.iter().any(|f| f == "logits-all");
            let t0 = Instant::now();
            let mut positions = vec![];
            let mut logits = if logits_batched {
                let lg = eng.as_engine().eval(&ids, true).unwrap(); positions.push(serde_json::json!({"pos": ids.len() - 1, "top": top(&lg, 10)})); lg
            } else if logits_all {
                let mut l = None;
                for (i, &t) in ids.iter().enumerate() { let lg = eng.as_engine().eval(&[t], true).unwrap(); positions.push(serde_json::json!({"pos": i, "top": top(&lg, 10)})); l = Some(lg); }
                l.unwrap()
            } else { vec![] };
            let mut next = if logits_all { argmax(&logits) } else { eng.as_engine().eval_argmax(&ids) };
            let t_pp = t0.elapsed().as_secs_f64();
            let mut gen = vec![]; let t1 = Instant::now();
            for i in 0..n_pred {
                let nt = next; gen.push(nt);
                if !json { print!("{}", tk.decode(&[nt])); std::io::stdout().flush().ok(); }
                if tk.is_eog(nt) || i + 1 == n_pred { break; }
                if logits_all { logits = eng.as_engine().eval(&[nt], true).unwrap(); next = argmax(&logits); positions.push(serde_json::json!({"pos": ids.len() + i, "top": top(&logits, 10)})); }
                else { next = eng.as_engine().eval_argmax(&[nt]); }
            }
            let t_gen = t1.elapsed().as_secs_f64();
            let pp_tps = ids.len() as f64 / t_pp; let gen_tps = if gen.len() > 1 { (gen.len() - 1) as f64 / t_gen } else { 0.0 };
            if json { println!("{}", serde_json::json!({"backend": backend, "prompt_ids": ids, "greedy": gen, "text": tk.decode(&gen.iter().copied().filter(|&t|!tk.is_eog(t)).collect::<Vec<_>>()), "positions": positions, "pp_tokens": ids.len(), "pp_seconds": t_pp, "pp_tps": pp_tps, "gen_tokens": gen.len(), "gen_seconds": t_gen, "gen_tps": gen_tps,"timed_decode_forwards":gen.len().saturating_sub(1),"first_prediction_in_prefill":true,"seconds_prefill_plus_gen":t_pp+t_gen})); }
            eng.as_engine().profile();
            if !json { eprintln!("\n[prompt {} tok, {:.2} tok/s | gen {} tok, {:.2} tok/s]", ids.len(), pp_tps, gen.len(), gen_tps); }
        }
        c => panic!("unknown command {c}"),
    }
}


/// `run --mtp [--mtp-head sidecar.gguf]`: greedy generation with MTP speculative decoding; JSON includes acceptance.
fn run_mtp(a: &Args, g: &Gguf, tk: &Tokenizer, layers: Option<usize>, backend: &str, ids: &[u32]) {
    use bonsai::mtp::*;
    let hg_own; let hg: &Gguf = match a.kv.get("mtp-head") { Some(p) => { hg_own = Gguf::open(p).unwrap(); &hg_own } None => g };
    validate_compatibility(hg,g).expect("incompatible MTP head");
    let n_pred: usize = a.kv.get("n-predict").map(|s| s.parse().unwrap()).unwrap_or(32);
    let eog = |t: u32| tk.is_eog(t);
    let t0 = Instant::now();
    let (gen, st, load_s);
    match backend {
        "cpu" => {
            let num = if a.kv.get("numerics").map(|s| s.as_str()) == Some("f32") { cpu::Numerics::F32 } else { cpu::Numerics::Ggml };
            let m = cpu::Model::load(g, layers).unwrap();
            let hd = MtpHead::load(hg, &m.cfg).unwrap().expect("model has no MTP head (use --mtp-head)");
            load_s = t0.elapsed().as_secs_f64();
            let mut b = CpuMtp::new(&m, &hd, num);
            (gen, st) = greedy_mtp(&mut b, ids, n_pred, &eog, &mut |_| true);
        }
        #[cfg(feature = "vulkan")]
        "vulkan" | "vk" => {
            let mut e = bonsai::vk::VkEngine::new_with_mtp(g, layers, a.kv.get("ctx").map(|s| s.parse().unwrap()).unwrap_or(4096), a.kv.get("device").map(|s| s.as_str()), Some(hg)).unwrap();
            load_s = t0.elapsed().as_secs_f64();
            (gen, st) = greedy_mtp(&mut e, ids, n_pred, &eog, &mut |_| true);
            e.ctx.print_profile();
        }
        b => panic!("--mtp: unknown backend {b}"),
    }
    let secs = t0.elapsed().as_secs_f64() - load_s;
    println!("{}", serde_json::json!({"backend": backend, "mtp": true, "prompt_ids": ids, "greedy": gen, "text": tk.decode(&gen.iter().copied().filter(|&t|!tk.is_eog(t)).collect::<Vec<_>>()), "gen_tokens": gen.len(), "seconds_prefill_plus_gen": secs,
        "pp_seconds":st.prefill_seconds,"gen_seconds":st.generation_seconds,"timed_decode_forwards":st.timed_decode_forwards,"first_prediction_in_prefill":true,"load_seconds":load_s,"mtp_stats":st,
        "mtp_drafted": st.drafted, "mtp_accepted": st.accepted, "mtp_acceptance": st.rate(), "tokens_per_trunk_pass": if st.steps + st.single_forwards > 0 { (gen.len() as f64 - 1.0) / (st.steps + st.single_forwards) as f64 } else { 0.0 }}));
}

/// `serve --mtp [--mtp-head sidecar.gguf] [--mtp-default on|off]`: persistent server with the trunk and MTP head
/// loaded once; each greedy request may choose `"mtp": true|false` (default from --mtp-default, "on").
fn serve_mtp(a: &Args, g: &Gguf, tk: &Tokenizer, layers: Option<usize>, backend: &str, model: &str) {
    use bonsai::mtp::*;
    let hg_own; let hg: &Gguf = match a.kv.get("mtp-head") { Some(p) => { hg_own = Gguf::open(p).unwrap(); &hg_own } None => g };
    validate_compatibility(hg, g).expect("incompatible MTP head");
    let mtp_default = match a.kv.get("mtp-default").map(|s| s.as_str()) { None | Some("on") => true, Some("off") => false, Some(v) => panic!("--mtp-default on|off, got {v}") };
    let n_ctx: usize = a.kv.get("ctx").map(|s| s.parse().unwrap()).unwrap_or(4096);
    let tmpl = ChatTemplate::from_gguf(g);
    let name = std::path::Path::new(model).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let cfg = bonsai::server::ServerCfg { addr: a.kv.get("host").cloned().unwrap_or("127.0.0.1:18081".into()), model_name: name, n_ctx, mtp_default, prompt_cache: a.kv.get("prompt-cache").map_or(false, |v| v == "on") };
    let t0 = Instant::now();
    match backend {
        "cpu" => {
            let num = if a.kv.get("numerics").map(|s| s.as_str()) == Some("f32") { cpu::Numerics::F32 } else { cpu::Numerics::Ggml };
            if let Some(t) = a.kv.get("threads") { rayon::ThreadPoolBuilder::new().num_threads(t.parse().unwrap()).build_global().ok(); }
            let m = cpu::Model::load(g, layers).unwrap();
            let hd = MtpHead::load(hg, &m.cfg).unwrap().expect("no MTP head (use --mtp-head)");
            let mut b = CpuMtp::new(&m, &hd, num);
            eprintln!("loaded trunk + MTP head in {:.1}s", t0.elapsed().as_secs_f64());
            bonsai::server::serve(&cfg, tk, tmpl.as_ref(), &mut b);
        }
        #[cfg(feature = "vulkan")]
        "vulkan" | "vk" => {
            let mut e = bonsai::vk::VkEngine::new_with_mtp(g, layers, n_ctx, a.kv.get("device").map(|s| s.as_str()), Some(hg)).unwrap();
            eprintln!("loaded trunk + MTP head in {:.1}s", t0.elapsed().as_secs_f64());
            bonsai::server::serve(&cfg, tk, tmpl.as_ref(), &mut e);
        }
        b => panic!("--mtp: unknown backend {b}"),
    }
}
