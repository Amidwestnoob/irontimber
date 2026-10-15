//! Independent review regressions. Fake Engine tests exercise production generation, not model accuracy.
use bonsai::{engine::{Engine, GenRequest, generate, TextStream}, sampling::SamplerParams, tokenizer::Tokenizer, gguf::Gguf};
use serde_json::json;
fn tokenizer() -> Tokenizer { Tokenizer::from_gguf(&Gguf::open(std::env::var("BONSAI_MODEL").expect("review tests require BONSAI_MODEL")).unwrap()).unwrap() }
struct Script { tokens: Vec<u32>, at: usize, calls: usize }
impl Engine for Script {
    fn eval(&mut self, _: &[u32], _: bool) -> Option<Vec<f32>> { panic!("greedy only") }
    fn eval_argmax(&mut self, _: &[u32]) -> u32 { self.calls += 1; let t = self.tokens[self.at]; self.at += 1; t }
    fn reset(&mut self) { self.at = 0; }
}
fn byte(tk: &Tokenizer, b: u8) -> u32 { (0..tk.tokens.len() as u32).find(|&t| tk.token_bytes(t) == [b]).unwrap() }
#[test] fn invalid_token_ids_rejected() {
    let tk = tokenizer();
    for p in [json!([4294967296u64]), json!([248320]), json!([-1]), json!(["bad"])] {
        assert!(bonsai::server::parse_request(&json!({"prompt_ids":p}), &tk, None, false).is_err(), "accepted {p}");
    }
}
#[test] fn malformed_budget_and_sampler_rejected() {
    let tk = tokenizer();
    for (key, value) in [("n_predict", json!(1.5)), ("max_tokens", json!("4")), ("repeat_penalty", json!(0)), ("temperature", json!(1e100)), ("top_p", json!(2.0)), ("top_k", json!(1.5)), ("cache_prompt", json!(null)), ("cache_prompt", json!("false"))] {
        let mut v = json!({"prompt":"Hi"}); v[key] = value;
        assert!(bonsai::server::parse_request(&v, &tk, None, false).is_err(), "accepted {v}");
    }
}
#[test] fn split_stop_never_leaks_to_stream() {
    let tk = tokenizer();
    let mut e = Script { tokens: b"STOPx".iter().map(|&b| byte(&tk,b)).collect(), at:0, calls:0 };
    let r = GenRequest { prompt:vec![1], n_predict:5, sampler:SamplerParams::greedy(), stop:vec!["STOP".into()], ignore_eos:false };
    let mut text = String::new();
    let res = generate(&mut e, &tk, &r, &mut |_,p| { text.push_str(p); true });
    assert_eq!(res.stop_type,"word"); assert_eq!(res.text,""); assert_eq!(text,res.text);
}
#[test] fn zero_budget_does_no_forward() {
    let tk = tokenizer();
    let mut e = Script { tokens:vec![1], at:0,calls:0 };
    let r = GenRequest { prompt:vec![1], n_predict:0, sampler:SamplerParams::greedy(), stop:vec![], ignore_eos:false };
    let res = generate(&mut e,&tk,&r,&mut |_,_| true);
    assert!(res.tokens.is_empty()); assert_eq!(e.calls,0);
}
#[test] fn utf8_byte_boundaries() {
    let tk = tokenizer(); let mut s = TextStream::default(); let mut out = String::new();
    for b in "A😀é中".bytes() { out.push_str(&s.push(&tk,byte(&tk,b))); }
    out.push_str(&s.flush()); assert_eq!(out,"A😀é中");
}
#[test] fn context_budget_validation() {
    let tk = tokenizer();
    for (j, ok) in [(json!({"prompt_ids":[1,2,3],"n_predict":2}),true), (json!({"prompt_ids":[1,2,3],"n_predict":3}),false), (json!({"prompt_ids":[1,2,3,4,5],"n_predict":0}),false)] {
        let mut r = bonsai::server::parse_request(&j,&tk,None,false).unwrap();
        assert_eq!(bonsai::server::validate_context(&mut r,&j,4).is_ok(),ok);
    }
    let j=json!({"prompt_ids":[1,2,3,4]}); let mut r=bonsai::server::parse_request(&j,&tk,None,false).unwrap();
    bonsai::server::validate_context(&mut r,&j,4).unwrap(); assert_eq!(r.n_predict,1);
}
#[test] fn eos_is_returned_but_not_spelled_and_forward_count_is_exact() {
    let tk=tokenizer(); let first=byte(&tk,b'A');
    let mut e=Script { tokens:vec![first,tk.eos],at:0,calls:0 };
    let r=GenRequest { prompt:vec![1],n_predict:4,sampler:SamplerParams::greedy(),stop:vec![],ignore_eos:false };
    let mut text=String::new(); let mut tokens=Vec::new();
    let res=generate(&mut e,&tk,&r,&mut |t,p| {text.push_str(p);if t!=u32::MAX {tokens.push(t);} true});
    assert_eq!(res.tokens,vec![first,tk.eos]); assert_eq!(tokens,res.tokens);
    assert_eq!(res.text,"A"); assert_eq!(text,"A"); assert_eq!(res.decode_forwards,1); assert_eq!(e.calls,2);
}
#[test] fn incomplete_stop_prefix_flushes_on_limit() {
    let tk = tokenizer();
    let mut e = Script { tokens:b"ST".iter().map(|&b| byte(&tk,b)).collect(),at:0,calls:0 };
    let r = GenRequest { prompt:vec![1],n_predict:2,sampler:SamplerParams::greedy(),stop:vec!["STOP".into()],ignore_eos:false };
    let mut text=String::new(); let res=generate(&mut e,&tk,&r,&mut |_,p| {text.push_str(p);true});
    assert_eq!(res.text,"ST"); assert_eq!(text,res.text);
}
