//! Tests that need the real GGUF (set BONSAI_MODEL; skipped otherwise).
use bonsai::{chat::ChatTemplate, gguf::Gguf, tokenizer::Tokenizer};
fn model() -> Option<Gguf> { std::env::var("BONSAI_MODEL").ok().map(|p| Gguf::open(p).unwrap()) }

#[test]
fn tokenizer_matches_llama_prism() {
    let Some(g) = model() else { eprintln!("BONSAI_MODEL unset: skipped"); return };
    let tk = Tokenizer::from_gguf(&g).unwrap();
    let data = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/tokenizer_cases.jsonl")).unwrap();
    let mut n = 0;
    for line in data.lines() {
        let v: serde_json::Value = serde_json::from_str(line).unwrap();
        let want: Vec<u32> = v["ids"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect();
        let text = v["text"].as_str().unwrap();
        // llama-tokenize parses special tokens by default
        assert_eq!(tk.encode(text, true, true), want, "text {text:?}");
        assert_eq!(tk.decode(&want), text, "decode roundtrip {text:?}");
        n += 1;
    }
    assert!(n >= 25);
}

#[test]
fn chat_template_renders() {
    let Some(g) = model() else { return };
    let t = ChatTemplate::from_gguf(&g).unwrap();
    let s = t.render(&serde_json::json!([{"role":"user","content":"Hi"}]), true, &serde_json::json!({"enable_thinking": false})).unwrap();
    assert_eq!(s, "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let s = t.render(&serde_json::json!([{"role":"system","content":"Be brief."},{"role":"user","content":"Hi"}]), true, &serde_json::json!({})).unwrap();
    assert!(s.starts_with("<|im_start|>system\nReasoning effort is set to xhigh."), "{s}");
    assert!(s.ends_with("<|im_start|>assistant\n<think>\n"));
}

#[test]
fn config_and_hadamard_metadata() {
    let Some(g) = model() else { return };
    let c = bonsai::model::Config::from_gguf(&g).unwrap();
    assert_eq!((c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_head_kv, c.head_dim), (64, 5120, 17408, 24, 4, 256));
    assert_eq!((c.ssm_state, c.ssm_groups, c.ssm_vheads, c.ssm_inner, c.conv_k), (128, 16, 48, 6144, 4));
    assert_eq!((0..64).filter(|&i| !c.is_recurrent(i)).count(), 16);
    let h = bonsai::model::load_hadamard(&g).unwrap().unwrap();
    assert_eq!(h.block, 1024);
    let mut w: Vec<_> = h.signs.keys().copied().collect(); w.sort();
    assert_eq!(w, vec![5120, 6144, 17408]);
}
