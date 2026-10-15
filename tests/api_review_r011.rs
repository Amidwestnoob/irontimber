//! Independent r011 strict selection / pre-open validation regressions.
use bonsai::{engine::GenRequest,sampling::SamplerParams,server::resolve_mtp};
use serde_json::json;
fn r() -> GenRequest { GenRequest { prompt:vec![1], n_predict:2, sampler:SamplerParams::greedy(),stop:vec![],ignore_eos:false } }
#[test] fn null_is_not_boolean() { assert!(resolve_mtp(&json!({"mtp":null}),&r(),true,true).is_err()); }
#[test] fn other_non_booleans_rejected() { for v in [json!(0),json!(1),json!("true"),json!([]),json!({})] { assert!(resolve_mtp(&json!({"mtp":v}),&r(),true,true).is_err()); } }
#[test] fn explicit_selection() { for d in [false,true] { for m in [false,true] { assert_eq!(resolve_mtp(&json!({"mtp":m}),&r(),true,d).unwrap(),m); } } }
#[test] fn omitted_uses_default() { for d in [false,true] { assert_eq!(resolve_mtp(&json!({}),&r(),true,d).unwrap(),d); } }
#[test] fn absent_head() { assert!(resolve_mtp(&json!({"mtp":true}),&r(),false,true).is_err()); assert!(!resolve_mtp(&json!({}),&r(),false,true).unwrap()); }
#[test] fn sampled_default_and_explicit() { let mut r=r();r.sampler.temperature=0.8;r.sampler.top_k=40;assert!(!resolve_mtp(&json!({}),&r,true,true).unwrap());assert!(resolve_mtp(&json!({"mtp":true}),&r,true,true).is_err()); }
#[test] fn preopen_invalid_defaults() {
 for args in [vec!["--mtp","--mtp-default","bad"],vec!["--mtp","--mtp-default"],vec!["--mtp-default","on"],vec!["--mtp","--mtp-head"]] {
  let p=std::process::Command::new(env!("CARGO_BIN_EXE_bonsai")).args(["serve","--model","/nonexistent-review-model","--backend","vulkan"]).args(&args).output().unwrap();
  let s=String::from_utf8_lossy(&p.stderr);assert!(!p.status.success());assert!(s.contains("--mtp"),"{s}");assert!(!s.contains("No such file")&&!s.contains("[vk]"),"{s}");
 }
}
