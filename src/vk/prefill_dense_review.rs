//! Actual device screen including temporary expansion cost, with full output comparison.
use super::*;

pub fn screen(dev:Option<&str>,m:u32,k:u32,nt:u32,iters:u32,accum:bool)->Result<String,String>{
 if !matches!(nt,22|43|64|128|256|512)||m%4!=0||m==0||m>17408||k==0||k>17408||k%128!=0||iters==0||iters>20{return Err("dense screen geometry rejected before device".into());}
 let mut rng=Rng(0xabcdef12345678);let wo=((k+m)*nt+63)/64*64;let an=wo as usize+m as usize*k as usize;
 let mut a=vec![0f32;an];for x in &mut a[..wo as usize]{*x=rng.f();}
 let w:Vec<u32>=(0..m as usize*k as usize/16).map(|_|rng.u()).collect();
 let sc:Vec<f32>=(0..m as usize*k as usize/128).map(|_|half::f16::from_f32(rng.f().abs()*0.05+0.01).to_f32()).collect();
 let s=side(dev,true,&a,&[0.;64],&w,&sc)?;let yo=k*nt;let ab:Vec<u8>=a[..wo as usize].iter().flat_map(|v|v.to_le_bytes()).collect();
 // Screen one bounded expansion block before the complete sequence.
 let count=(m*(k/16)).min(65536);let t=std::time::Instant::now();s.c.begin();s.c.dispatch("prefill_expand",s.set,&[m,k,wo,0,count],(count.div_ceil(256),1,1));s.c.submit_wait();let expand_probe=t.elapsed().as_secs_f64();
 if expand_probe>0.05{return Err(format!("expansion fixed-block screen {expand_probe}s"));}
 s.c.begin();s.c.expand_prefill(s.set,m,k,wo);s.c.submit_wait();
 // Check every expanded word against independent CPU dequantization, in bounded host chunks.
 for start in (0..m as usize*k as usize).step_by(262144){let n=(m as usize*k as usize-start).min(262144);let mut raw=vec![0u8;n*4];s.c.download(&s.arena,(wo as usize+start)as u64*4,&mut raw);
  for(j,b)in raw.chunks_exact(4).enumerate(){let i=start+j;let row=i%m as usize;let col=i/m as usize;let d=sc[row*(k as usize/128)+col/128];let code=(w[row*(k as usize/16)+col/16]>>(2*(col%16)))&3;let expected=(code as f32-1.)*d;if f32::from_le_bytes(b.try_into().unwrap()).to_bits()!=expected.to_bits(){return Err(format!("dense expansion word {i} mismatch"));}}
 }
 let mut reference=vec![];let mut text=String::new();
 for pipe in ["matmul_t2","matmul_dense64"]{
  let pc=[m,k,nt,0,k,yo,m,accum as u32,wo];
  let mut previous=0.;let mut screened=64u32.min(nt);
  loop {if previous>0. && previous*(screened.div_ceil(64)as f64)>0.05{return Err("dense predicted expansion screen".into());}
   let mut p=pc;p[2]=screened;let t=std::time::Instant::now();s.c.begin();s.c.dispatch(pipe,s.set,&p,(m.div_ceil(64),screened.div_ceil(64),1));s.c.submit_wait();previous=t.elapsed().as_secs_f64();if previous>0.05{return Err(format!("{pipe} screen at nt={screened}: {previous}s"));}if screened==nt{break;}screened=(screened*2).min(nt);
  }
  s.c.upload(&s.arena,0,&ab);s.c.begin();s.c.dispatch(pipe,s.set,&pc,(m.div_ceil(64),nt.div_ceil(64),1));s.c.submit_wait();
  let mut raw=vec![0u8;m as usize*nt as usize*4];s.c.download(&s.arena,yo as u64*4,&mut raw);let result:Vec<f32>=raw.chunks_exact(4).map(|b|f32::from_le_bytes(b.try_into().unwrap())).collect();let mut diff=0;
  if reference.is_empty(){reference=result;}else{for(x,y)in reference.iter().zip(result){if x.to_bits()!=y.to_bits(){diff+=1;}}if diff>0{return Err(format!("dense full output differs in {diff} words"));}}
  let mut times=vec![];let mut expansion=vec![];
  for i in 0..20+iters {s.c.upload(&s.arena,0,&ab);let t=std::time::Instant::now();s.c.begin();if pipe=="matmul_dense64"{s.c.expand_prefill(s.set,m,k,wo);}s.c.submit_wait();let es=t.elapsed().as_secs_f64();let t=std::time::Instant::now();s.c.begin();s.c.dispatch(pipe,s.set,&pc,(m.div_ceil(64),nt.div_ceil(64),1));s.c.submit_wait();let sec=t.elapsed().as_secs_f64();if sec>0.05{return Err(format!("{pipe} warm/measure screen {sec}s"));}if i>=20{times.push(sec);expansion.push(es);}}
  text+=&format!("{}\n",serde_json::json!({"kernel":pipe,"m":m,"k":k,"nt":nt,"accumulate":accum,"seconds":times,"expansion_seconds":expansion,"expansion_probe_seconds":expand_probe,"expanded_words_checked":m*k,"output_words_checked":m*nt,"differing_words":diff,"completed_dispatches":s.c.completed_dispatches.get()}));
 }Ok(text)
}
