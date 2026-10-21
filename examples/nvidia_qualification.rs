//! Real desktop-only qualification. Run under the external shared GPU lock.
//! No persistent grids, no process-exit lifetime substitution, no fixture skips.
#[cfg(not(feature="vulkan"))]
fn main(){panic!("nvidia_qualification requires the vulkan feature");}
#[cfg(feature="vulkan")]
fn main(){gpu::main();}
#[cfg(feature="vulkan")]
mod gpu {
use bonsai::{gguf::Gguf,tokenizer::Tokenizer,vk::VkEngine,mtp::{self,MtpBackend},engine::{self,GenRequest},sampling::SamplerParams};
use std::{time::{Instant,Duration},process::Command};
const UUID:&str="GPU-32ce3f05-a5de-4c1a-01f8-4027e0cd5f6b";
fn memory(label:&str)->u64 {
 let o=Command::new("nvidia-smi").args(["-i",UUID,"--query-gpu=uuid,memory.used,memory.free","--format=csv,noheader,nounits"]).output().unwrap();assert!(o.status.success());
 let s=String::from_utf8(o.stdout).unwrap();let v:Vec<_>=s.trim().split(',').map(str::trim).collect();assert_eq!(v[0],UUID);
 let used=v[1].parse::<u64>().unwrap()*(1<<20);let free=v[2].parse::<u64>().unwrap()*(1<<20);assert!(free>=1<<30,"VRAM reserve lost");
 println!("MEMORY {}",serde_json::json!({"label":label,"pid":std::process::id(),"device_uuid":UUID,"used":used,"free":free}));used
}
fn released(base:u64,label:&str){for _ in 0..50{if memory(label)<=base+(128<<20){return;}std::thread::sleep(Duration::from_millis(100));}panic!("same-process memory did not return within 128MiB");}
fn load<'a>(g:&'a Gguf,h:Option<&'a Gguf>,nv:bool)->VkEngine<'a>{
 std::env::set_var("BONSAI_NV",if nv{"1"}else{"0"});let b=VkEngine::new_with_mtp(g,None,8192,Some("NVIDIA"),h).unwrap();
 assert_eq!(b.ctx.props.vendor_id,0x10de);assert_eq!(b.cpu.cfg.n_layer,64);assert!(b.kv16);assert_eq!(b.ctx.nv,nv);memory("loaded");println!("IDENTITY {}",b.review_identity());b
}
fn near(a:&[f32],b:&[f32],label:&str){assert_eq!(a.len(),b.len());let mut worst=0f32;for(&x,&y)in a.iter().zip(b){assert!(x.is_finite()&&y.is_finite());worst=worst.max((x-y).abs()/(1.+x.abs().max(y.abs())));}println!("STATE label={label:?} elements={} max_scaled={worst} tolerance=1e-4",a.len());assert!(worst<1e-4,"{label}");}
fn request(p:Vec<u32>,n:usize)->GenRequest{GenRequest{prompt:p,n_predict:n,sampler:SamplerParams{temperature:0.,..Default::default()},stop:vec![],ignore_eos:true}}
fn cases(tk:&Tokenizer)->Vec<GenRequest>{
 let texts=["The capital of France is","Write a short story about a lighthouse.","What is the result of 17 plus 25?","Explain photosynthesis in one sentence.","Translate hello into French.","List three prime numbers.","A Rust function to add two integers:","The opposite of hot is","Name the largest ocean.","Complete the sequence: 2, 4, 6,","A recipe for tea begins with","Summarize why exercise matters.","Unicode: café, 東京, 🌲. Explain."];
 let mut v:Vec<_>=texts.iter().enumerate().map(|(i,t)|request(tk.encode(t,true,true),[1,2,3,7,12][i%5])).collect();
 let seed=tk.encode("The forest has tall trees and a river. ",true,true);for n in [64,65,129]{v.push(request(seed.iter().copied().cycle().take(n).collect(),7));}v
}
fn run(b:&mut VkEngine,tk:&Tokenizer,r:&GenRequest,mtp:bool,cancel:bool)->engine::GenResult{
 let mut stream=Vec::new();let mut text=String::new();let mut cb=|t:u32,s:&str|{if t!=u32::MAX{stream.push(t);}text.push_str(s);!cancel||stream.len()<2};
 let out=if mtp{engine::generate_mtp(b,tk,r,&mut cb).unwrap()}else{engine::generate(b,tk,r,&mut cb)};
 assert_eq!(stream,out.tokens,"complete streaming token history");assert_eq!(text,out.text,"streamed text");if cancel&&r.n_predict>2{assert_eq!(out.stop_type,"cancel");}
 out
}
fn parity(g:&Gguf,h:&Gguf,tk:&Tokenizer,soak:bool){
 let cs=cases(tk);let base=memory("parity-baseline");let mut gold=Vec::new();
 {let mut b=load(g,Some(h),false);for mtp in [false,true]{for(i,r)in cs.iter().enumerate(){let o=run(&mut b,tk,r,mtp,false);println!("REFERENCE {}",serde_json::json!({"case":i,"mtp":mtp,"input_ids":r.prompt,"output_ids":o.tokens,"text":o.text}));gold.push((o.tokens,o.text));}}}
 released(base,"reference-dropped");let mut b=load(g,Some(h),true);let mut n=0;let start=Instant::now();let mut low=u64::MAX;let mut high=0;
 loop{for mtp in [false,true]{for(i,r)in cs.iter().enumerate(){let o=run(&mut b,tk,r,mtp,false);let j=i+if mtp{cs.len()}else{0};assert_eq!(o.tokens,gold[j].0,"complete history case={i} mtp={mtp}");assert_eq!(o.text,gold[j].1);n+=1;let m=memory("mixed-request");low=low.min(m);high=high.max(m);println!("PARITY {}",serde_json::json!({"request":n,"case":i,"mtp":mtp,"input_ids":r.prompt,"output_ids":o.tokens,"elapsed_seconds":start.elapsed().as_secs_f64()}));}
 }if !soak||start.elapsed().as_secs()>=600{break;}}
 assert!(high-low<=128<<20,"soak memory growth");println!("SUMMARY {}",serde_json::json!({"mode":if soak{"soak"}else{"parity"},"cases":cs.len()*2,"requests":n,"seconds":start.elapsed().as_secs_f64(),"errors":0,"history_mismatches":0,"memory_min":low,"memory_max":high,"engine":b.review_identity()}));
 drop(b);released(base,"parity-dropped");
}
fn lifecycle(g:&Gguf,h:&Gguf,tk:&Tokenizer){
 let p=tk.encode("The capital of France is",true,true);
 for mtp in [false,true]{
  {let mut warm=load(g,if mtp{Some(h)}else{None},true);let _=run(&mut warm,tk,&request(p.clone(),4),mtp,false);}
  std::thread::sleep(Duration::from_millis(500));let base=memory("warmed-baseline");let mut histories=Vec::new();
  for cycle in 0..3{let mut b=load(g,if mtp{Some(h)}else{None},true);let o=run(&mut b,tk,&request(p.clone(),4),mtp,false);assert!(b.ctx.completed_dispatches.get()>0);if mtp{assert!(b.ctx.completed_q8.get()>0);}println!("LIFECYCLE {}",serde_json::json!({"pid":std::process::id(),"cycle":cycle,"mtp":mtp,"output_ids":o.tokens,"engine":b.review_identity(),"baseline":base,"tolerance_bytes":128<<20}));histories.push(o.tokens);drop(b);released(base,"cycle-dropped");}
  assert!(histories.windows(2).all(|w|w[0]==w[1]));
 }
}
fn state(g:&Gguf,h:&Gguf,tk:&Tokenizer){
 let mut b=load(g,Some(h),true);let p=vec![760,6511,314];let x=b.eval_argmax(&p);let a=b.eval_argmax(&[x]);let expected=b.review_state(false);
 b.reset();assert_eq!(b.mtp_prefill(&p),x);let draft=b.mtp_draft(&[x]);let wrong=(a+1)%b.cpu.cfg.n_vocab as u32;let(a0,_)=b.mtp_verify(x,wrong);assert_eq!(a0,a);b.mtp_reject();assert_eq!(b.pos,p.len()+1);near(&b.review_state(false),&expected,"synthetic rejection active trunk");
 b.mtp_draft(&[a]);let mh=b.review_state(true);let mut p2=p.clone();p2.push(x);b.mtp_prefill(&p2);b.mtp_draft(&[a]);near(&b.review_state(true),&mh,"rejected head cache catchup");
 b.reset();b.eval_argmax(&p2);let next=b.eval_argmax(&[a]);b.eval_argmax(&[next]);let accepted=b.review_state(false);b.mtp_prefill(&p2);b.mtp_draft(&[a]);let(v,_)=b.mtp_verify(a,next);assert_eq!(v,next);near(&b.review_state(false),&accepted,"controlled accepted chain");
 for(prompt,n)in [(p.clone(),0),(p.clone(),1),(p.clone(),2),(p.clone(),3),(p.clone(),7),(vec![760;80],5),(vec![760;190],3),(vec![760;191],2),(vec![760;192],1)]{let r=request(prompt,n);let plain=run(&mut b,tk,&r,false,false);let out=run(&mut b,tk,&r,true,false);assert_eq!(out.tokens,plain.tokens);let again=run(&mut b,tk,&r,true,false);assert_eq!(again.tokens,out.tokens);}
 let mut prompt=tk.encode("The capital of France is",true,true);let mut found=None;
 for _ in 0..16{let x=b.mtp_prefill(&prompt);let d=b.mtp_draft(&[x]);let(a,_)=b.mtp_verify(x,d);if a==d&&d!=x{found=Some((x,d));break;}prompt.push(x);}
 let(x,d)=found.expect("real accepted draft required");let mut fed=prompt.clone();fed.push(x);b.reset();b.eval_argmax(&prompt);b.eval_argmax(&[x]);let expected=b.review_state(false);
 for cancel in [false,true]{let mut count=0;let(out,st)=mtp::greedy_mtp(&mut b,&prompt,8,&|t|!cancel&&t==d,&mut |_|{count+=1;!cancel||count<2});assert_eq!(out,vec![x,d]);assert_eq!(st.accepted,1);assert_eq!(st.terminal_rollbacks,1);assert_eq!(b.pos,fed.len());near(&b.review_state(false),&expected,"terminal rollback");let head=b.review_state(true);b.mtp_prefill(&fed);near(&head,&b.review_state(true),"terminal head rollback");println!("TERMINAL cancel={cancel} stats={st:?}");}
 let mut cache=Vec::new();let r=request(tk.encode("The capital of France is",true,true),3);let _=engine::generate_cached(&mut b,tk,&r,Some(&mut cache),&mut |_,_|true);let mut ext=cache.clone();ext.extend(tk.encode(" and Japan is",false,true));let r=request(ext,7);let hit=engine::generate_cached(&mut b,tk,&r,Some(&mut cache),&mut |_,_|true);assert!(hit.cached_tokens>0);let fresh=run(&mut b,tk,&r,false,false);assert_eq!(hit.tokens,fresh.tokens);
 for mtp in [false,true]{let r=request(p.clone(),7);run(&mut b,tk,&r,mtp,true);let recovery=run(&mut b,tk,&r,mtp,false);let again=run(&mut b,tk,&r,mtp,false);assert_eq!(recovery.tokens,again.tokens);}
 println!("STATE_PASS draft={draft} identity={}",b.review_identity());
}
fn context(g:&Gguf,h:&Gguf,tk:&Tokenizer){
 let seed=tk.encode("The forest has tall trees and a river. ",true,true);let full:Vec<_>=seed.iter().copied().cycle().take(8192).collect();
 for mtp in [false,true]{let mut b=load(g,if mtp{Some(h)}else{None},true);let r=request(tk.encode("The capital of France is",true,true),4);let gold=run(&mut b,tk,&r,mtp,false).tokens;b.reset();
 let t=Instant::now();let last=if mtp{b.mtp_prefill(&full)}else{b.eval_argmax(&full)};assert_eq!(b.pos,8192);memory("full8192");let before=b.ctx.completed_dispatches.get();
 let overflow=std::panic::catch_unwind(std::panic::AssertUnwindSafe(||b.eval_argmax(&[last])));assert!(overflow.is_err());assert_eq!(b.ctx.completed_dispatches.get(),before,"overflow must fail before dispatch");assert_eq!(b.pos,8192);
 let out=run(&mut b,tk,&r,mtp,false);assert_eq!(out.tokens,gold);println!("CONTEXT_PASS {}",serde_json::json!({"mtp":mtp,"capacity":8192,"last":last,"seconds":t.elapsed().as_secs_f64(),"overflow_rejected_before_dispatch":true,"recovery_history":out.tokens,"engine":b.review_identity()}));}
}
pub fn main(){
 assert_eq!(std::fs::read_to_string("/proc/sys/kernel/hostname").unwrap().trim(),"drew-Desktop","desktop only; never enumerate Rig GPU");
 assert!(std::env::var("BONSAI_GPU_OWNER").is_ok(),"external shared exclusive owner required");assert_eq!(std::env::var("BONSAI_KV").unwrap(),"f16");
 let mode=std::env::args().nth(1).expect("kernels|state|context|lifecycle|parity|soak");memory("phase-start");
 if mode=="kernels"{let(r,ok)=bonsai::vk::kparity::run(Some("NVIDIA"),1e-4).unwrap();println!("{r}");assert!(ok);bonsai::vk::nvidia_oracles::run().unwrap();return;}
 let g=Gguf::open(std::env::var("BONSAI_MODEL").expect("actual model required")).unwrap();let h=Gguf::open(std::env::var("BONSAI_MTP_HEAD").expect("actual head required")).unwrap();let tk=Tokenizer::from_gguf(&g).unwrap();
 match mode.as_str(){"state"=>state(&g,&h,&tk),"context"=>context(&g,&h,&tk),"lifecycle"=>lifecycle(&g,&h,&tk),"parity"=>parity(&g,&h,&tk,false),"soak"=>parity(&g,&h,&tk,true),_=>panic!("unknown phase")};memory("phase-end");
}
}
