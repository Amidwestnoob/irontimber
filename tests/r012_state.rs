//! Additional independent full-model r012 terminal and cache-state proofs.
#[cfg(feature="vulkan")]
mod gpu {
use bonsai::{gguf::Gguf,mtp::{self,MtpBackend},vk::VkEngine,tokenizer::Tokenizer};
fn near(a:&[f32],b:&[f32],label:&str){assert_eq!(a.len(),b.len());let mut worst=0f32;for(&x,&y)in a.iter().zip(b){assert!(x.is_finite()&&y.is_finite());worst=worst.max((x-y).abs()/(1.+x.abs().max(y.abs())));}eprintln!("{label}: {} active elements max_scaled={worst}",a.len());assert!(worst<1e-4);}
#[test]
fn terminal_draft_cancel_and_cache_active_state(){
 let g=Gguf::open(std::env::var("BONSAI_MODEL").unwrap()).unwrap();let h=Gguf::open(std::env::var("BONSAI_MTP_HEAD").unwrap()).unwrap();let tk=Tokenizer::from_gguf(&g).unwrap();
 let mut b=VkEngine::new_with_mtp(&g,None,8192,Some("HAWAII"),Some(&h)).unwrap();assert!(b.kv16);assert_eq!(b.ctx.props.device_id,0x67a0);
 let mut prompt=tk.encode("The capital of France is",true,true);
 // Find a real accepted draft, never replace the head's prediction with a fabricated one.
 let mut accepted=None;
 for _ in 0..16{let x=b.mtp_prefill(&prompt);let d=b.mtp_draft(&[x]);let(a,_)=b.mtp_verify(x,d);if a==d && d!=x{accepted=Some((x,d));break;}prompt.push(x);}
 let(x,d)=accepted.expect("no real accepted draft in search window");let mut fed=prompt.clone();fed.push(x);
 // Match the plain generation dispatch history: prefill prompt, then feed x.
 // A single batched prefill of prompt+x is a different floating-point computation.
 b.reset();b.eval_argmax(&prompt);b.eval_argmax(&[x]);let expected=b.review_state(false);
 for cancel in [false,true]{let mut count=0;let(out,st)=mtp::greedy_mtp(&mut b,&prompt,8,&|t|!cancel&&t==d,&mut |_|{count+=1;!cancel||count<2});assert_eq!(out,vec![x,d]);assert_eq!(st.accepted,1);assert_eq!(st.terminal_rollbacks,1);assert_eq!(b.pos,fed.len());near(&b.review_state(false),&expected,"terminal rollback active KV/SSM/conv");let head=b.review_state(true);b.mtp_prefill(&fed);near(&head,&b.review_state(true),"terminal rollback active head");eprintln!("terminal cancel={cancel} tokens={out:?} stats={st:?}");}
 // Prefix-cache continuation, with the recurrent state carried across chunk boundaries,
 // must match fresh full evaluation in both logits and every active cache/state element.
 let mut prefix=tk.encode("The capital of France is",true,true);let suffix=tk.encode(" Paris. The capital of Japan is",false,true);
 b.reset();b.eval_argmax(&prefix);let next=b.eval_argmax(&suffix);let hit=b.review_state(false);prefix.extend(suffix);b.reset();let full=b.eval_argmax(&prefix);assert_eq!(next,full);near(&hit,&b.review_state(false),"prefix continuation versus reset full active state");
}
#[test]
fn small_context_scratch_regression_not_acceptance_substitute(){
 let g=Gguf::open(std::env::var("BONSAI_MODEL").unwrap()).unwrap();
 let mut b=VkEngine::new_with_mtp(&g,None,1,Some("HAWAII"),None).unwrap();assert!(b.kv16);assert_eq!(b.ctx.props.device_id,0x67a0);
 let a=b.eval_argmax(&[760]);assert_eq!(b.pos,1);b.reset();assert_eq!(b.eval_argmax(&[760]),a);assert!(b.ctx.completed_dispatches.get()>0);
 eprintln!("small context scratch regression full64 ctx=1 token={a}; full8192 acceptance is a separate test");
}
}
