//! Independent head validation, real CPU/W9100 state and boundary regressions.
use bonsai::{gguf::{Gguf,Value,GgmlType},mtp::{self,MtpBackend,MtpHead,CpuMtp},cpu,model::Config,sampling::argmax};
fn files()->(Gguf,Gguf){(Gguf::open(std::env::var("BONSAI_MODEL").expect("baseline required")).unwrap(),Gguf::open(std::env::var("BONSAI_MTP_HEAD").expect("head required")).unwrap())}
#[test]fn compatible_head(){let(g,h)=files();mtp::validate_compatibility(&h,&g).unwrap();}
#[test]fn wrong_head_shape(){let(g,mut h)=files();h.tensors.get_mut("blk.64.nextn.eh_proj.weight").unwrap().dims.swap(0,1);assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn wrong_head_quant(){let(g,mut h)=files();h.tensors.get_mut("blk.64.attn_k.weight").unwrap().ty=GgmlType::F32;assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn missing_head_norm(){let(g,mut h)=files();h.tensors.remove("blk.64.nextn.hnorm.weight");assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn wrong_head_layer(){let(g,mut h)=files();h.kv.insert("qwen35.block_count".into(),Value::U32(64));assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn wrong_tokenizer(){let(g,mut h)=files();h.kv.insert("tokenizer.ggml.eos_token_id".into(),Value::U32(0));assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn missing_tokenizer(){let(g,mut h)=files();h.kv.remove("tokenizer.ggml.tokens");assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn wrong_hadamard(){let(g,mut h)=files();h.kv.insert("prism.hadamard.block_size".into(),Value::U32(512));assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn wrong_rope(){let(g,mut h)=files();h.kv.insert("qwen35.rope.freq_base".into(),Value::F32(10.));assert!(mtp::validate_compatibility(&h,&g).is_err());}
#[test]fn zero_head_layers(){let(g,mut h)=files();h.kv.insert("qwen35.block_count".into(),Value::U32(0));assert!(MtpHead::validate_layout(&h,&Config::from_gguf(&g).unwrap()).is_err());}
#[cfg(feature="vulkan")]
#[test]fn invalid_head_before_device_dispatch(){let(g,mut h)=files();h.tensors.get_mut("blk.64.attn_q.weight").unwrap().dims[1]=1;let e=bonsai::vk::VkEngine::new_with_mtp(&g,Some(4),256,Some("THIS_DEVICE_MUST_NOT_BE_QUERIED"),Some(&h));let msg=e.err().expect("bad head accepted");assert!(msg.contains("shape/type"),"{msg}");}
fn near(a:&[f32],b:&[f32],label:&str){assert_eq!(a.len(),b.len(),"{label}");let mut abs=0f32;let mut rel=0f32;for(&x,&y)in a.iter().zip(b){assert!(x.is_finite()&&y.is_finite());abs=abs.max((x-y).abs());rel=rel.max((x-y).abs()/(1.+x.abs().max(y.abs())));}eprintln!("state {label}: elements={} max_abs={abs} max_scaled={rel}",a.len());assert!(rel<1e-4,"{label}: {rel}");}
fn state(s:&cpu::State)->Vec<f32>{s.conv.iter().chain(&s.ssm).chain(&s.k).chain(&s.v).flatten().copied().collect()}
#[test]fn cpu_rejection_accepted_chain_and_reset(){let(g,h)=files();let m=cpu::Model::load(&g,Some(2)).unwrap();let hd=MtpHead::load(&h,&m.cfg).unwrap().unwrap();let mut b=CpuMtp::new(&m,&hd,cpu::Numerics::F32);let p=vec![760,6511,314];let x=b.mtp_prefill(&p);let mut expected=b.st.clone();let a=argmax(&m.forward_token(&mut expected,x,true,cpu::Numerics::F32,None).unwrap());let draft=b.mtp_draft(&[x]);let wrong=(a+1)%m.cfg.n_vocab as u32;let(a0,_)=b.mtp_verify(x,wrong);assert_eq!(a0,a);b.mtp_reject();assert_eq!(b.st.pos,expected.pos);near(&state(&b.st),&state(&expected),"CPU synthetic reject active trunk");b.mtp_draft(&[a]);let mk=b.ms.k.clone();let mv=b.ms.v.clone();let mut p2=p.clone();p2.push(x);b.mtp_prefill(&p2);b.mtp_draft(&[a]);near(&mk,&b.ms.k,"CPU rejected head K catchup");near(&mv,&b.ms.v,"CPU rejected head V catchup");let next=argmax(&m.forward_token(&mut expected,a,true,cpu::Numerics::F32,None).unwrap());let _=m.forward_token(&mut expected,next,true,cpu::Numerics::F32,None);b.mtp_verify(a,next);near(&state(&b.st),&state(&expected),"CPU controlled accepted chain");let(o,_)=mtp::greedy_mtp(&mut b,&p,7,&|_|false,&mut |_|true);let(o2,_)=mtp::greedy_mtp(&mut b,&p,7,&|_|false,&mut |_|true);assert_eq!(o,o2);eprintln!("CPU actual head draft={draft}; rejected probe explicitly forced wrong={wrong}");}
#[cfg(feature="vulkan")]
#[test]fn gpu_state_rollback_context_and_reset(){
 let(g,h)=files();
 let mut b=bonsai::vk::VkEngine::new_with_mtp(&g,None,8192,Some("HAWAII"),Some(&h)).unwrap();assert_eq!(b.ctx.props.device_id,0x67a0,"not W9100");assert!(b.kv16,"r012 acceptance requires BONSAI_KV=f16");
 let p=vec![760,6511,314];let x=b.eval_argmax(&p);let a=b.eval_argmax(&[x]);let expected=b.review_state(false);b.reset();assert_eq!(b.mtp_prefill(&p),x);let draft=b.mtp_draft(&[x]);let wrong=(a+1)%b.cpu.cfg.n_vocab as u32;let(a0,_)=b.mtp_verify(x,wrong);assert_eq!(a0,a);b.mtp_reject();assert_eq!(b.pos,p.len()+1);near(&b.review_state(false),&expected,"W9100 synthetic reject active conv/SSM/KV");
 b.mtp_draft(&[a]);let mh=b.review_state(true);let mut p2=p.clone();p2.push(x);b.mtp_prefill(&p2);b.mtp_draft(&[a]);near(&b.review_state(true),&mh,"W9100 rejected head cache catchup");
 b.reset();b.eval_argmax(&p2);let next=b.eval_argmax(&[a]);b.eval_argmax(&[next]);let accepted=b.review_state(false);
 b.mtp_prefill(&p2);b.mtp_draft(&[a]);let(v,_)=b.mtp_verify(a,next);assert_eq!(v,next);near(&b.review_state(false),&accepted,"W9100 controlled accepted chain");
 for (prompt,n) in [(p.clone(),0),(p.clone(),1),(p.clone(),2),(p.clone(),3),(p.clone(),7),(vec![760;80],5),(vec![760;190],3),(vec![760;191],2),(vec![760;192],1)]{
  b.reset();let mut plain=vec![];if n>0{let mut t=b.eval_argmax(&prompt);for i in 0..n{plain.push(t);if i+1<n{t=b.eval_argmax(&[t]);}}}
  let(out,stats)=mtp::greedy_mtp(&mut b,&prompt,n,&|_|false,&mut |_|true);assert_eq!(out,plain,"prompt={} budget={n}",prompt.len());let(out2,_)=mtp::greedy_mtp(&mut b,&prompt,n,&|_|false,&mut |_|true);assert_eq!(out,out2);eprintln!("W9100 boundary prompt={} budget={n} actual_stats={stats:?}",prompt.len());
 }
 eprintln!("W9100 full_layers={} actual_draft={draft}; forced rejection wrong={wrong}; completed_dispatches={} q8={}",b.cpu.cfg.n_layer,b.ctx.completed_dispatches.get(),b.ctx.completed_q8.get());assert!(b.ctx.completed_q8.get()>0);
}
