//! Independent f64 softmax oracle for r012 f16 packing and split-K tails.
use super::Ctx;
pub fn run()->Result<(),String> {
 let c=Ctx::new(Some("HAWAII"),8)?;assert_eq!(c.props.device_id,0x67a0);assert!(c.rust);
 let (nh,nkv,hd)=(24usize,4usize,256usize);let kvd=nkv*hd;let qts=nh*hd*2;
 for (pos,nt) in [(0usize,1usize),(0,64),(63,2),(64,2),(65,1),(2047,2),(4095,2),(8190,2)] {
  let total=pos+nt;let cn=64usize;let sm=total.div_ceil(cn);
  let qo=0;let kf=nt*qts;let vf=kf+total*kvd;let kh=vf+total*kvd;let vh=kh+total*kvd/2;let oo=vh+total*kvd/2;let scr=(oo+nt*nh*hd+63)/64*64;let len=scr+nt*nh*sm*260;
  let mut a=vec![0f32;len];for (i,x) in a[..kh].iter_mut().enumerate(){*x=((i*17%1009) as f32-504.0)/503.0;}
  let arena=c.buffer((len*4)as u64,false)?;let params=c.buffer(16,false)?;let set=c.set([(&arena,0,arena.size),(&params,0,16),(&params,0,16),(&params,0,16)]);
  let raw:Vec<u8>=a.iter().flat_map(|v|v.to_le_bytes()).collect();c.upload(&arena,0,&raw);
  for (src,dst) in [(kf,kh),(vf,vh)] {for start in (0..total).step_by(64){let n=(total-start).min(64)*kvd;c.begin();c.dispatch("kv_pack",set,&[(src+start*kvd)as u32,(dst+start*kvd/2)as u32,n as u32],((n/2).div_ceil(64) as u32,1,1));c.submit_wait();}}
  let mut packed=vec![0u8;total*kvd*2];c.download(&arena,kh as u64*4,&mut packed);
  // HAWAII PackHalf2x16 uses round-toward-zero, not half::from_f32's RNE.
  // Assert the exact RTZ mapping independently for every active cache element.
  let trunc=|v:f32| {let h=half::f16::from_f32(v);if h.to_f32().abs()>v.abs(){half::f16::from_bits(h.to_bits()-1)}else{h}};
  let mut rne_differences=0;
  for(i,b)in packed.chunks_exact(2).enumerate(){let got=u16::from_le_bytes(b.try_into().unwrap());assert_eq!(got,trunc(a[kf+i]).to_bits(),"RTZ half conversion at {i}");rne_differences+=usize::from(got!=half::f16::from_f32(a[kf+i]).to_bits());}
  eprintln!("r012 packed K elements={} exact_RTZ=true RNE_differences={rne_differences}",total*kvd);
  c.begin();c.dispatch("attn_s_h",set,&[qo as u32,qts as u32,kh as u32,vh as u32,kvd as u32,pos as u32,hd as u32,nh as u32,nkv as u32,scr as u32,cn as u32,sm as u32,(1.0f32/16.0).to_bits()],(nh as u32,nt as u32,sm as u32));c.dispatch("attn_comb",set,&[qo as u32,qts as u32,pos as u32,nh as u32,scr as u32,cn as u32,sm as u32,oo as u32,(nh*hd)as u32],(nh as u32,nt as u32,1));c.submit_wait();
  let mut out=vec![0u8;nt*nh*hd*4];c.download(&arena,oo as u64*4,&mut out);let got=super::decode_kv_readback(&out,false);
  let hval=|x:f32|trunc(x).to_f64();let mut worst=0f64;
  for t in 0..nt {for h in 0..nh {let n=pos+t+1;let kvh=h/(nh/nkv);let qb=t*qts+h*512;let mut scores=Vec::with_capacity(n);
   for p in 0..n {scores.push((0..hd).map(|d|a[qb+d]as f64*hval(a[kf+p*kvd+kvh*hd+d])).sum::<f64>()/16.0);}
   let max=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);for s in &mut scores{*s=(*s-max).exp();}let sum:f64=scores.iter().sum();
   for d in 0..hd{let want=(0..n).map(|p|scores[p]*hval(a[vf+p*kvd+kvh*hd+d])).sum::<f64>()/sum/(1.0+(-(a[qb+hd+d]as f64)).exp());let value=got[(t*nh+h)*hd+d]as f64;assert!(value.is_finite());worst=worst.max((want-value).abs());}
  }}
  eprintln!("r012 f16 independent oracle pos={pos} nt={nt} max_abs={worst} completed_dispatches={}",c.completed_dispatches.get());assert!(worst<2e-6);
  // RAII releases both buffers at this iteration's end, including error unwinding.
 }
 Ok(())
}
