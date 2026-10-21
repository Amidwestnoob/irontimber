//! NVIDIA adaptation of the accepted r012 f64 oracle. Original W9100/RTZ
//! acceptance in r012_parity.rs is byte-identical. Tolerances are unchanged.
use super::Ctx;
pub fn run() -> Result<(),String> {
 let c=Ctx::new(Some("NVIDIA"),32)?;assert!(c.nv);assert_eq!(c.props.vendor_id,0x10de);
 let (nh,nkv,hd)=(24usize,4usize,256usize);let kvd=nkv*hd;let qts=nh*hd*2;
 for (pos,nt) in [(0usize,1usize),(0,64),(63,2),(64,2),(65,1),(2047,2),(4095,2),(8190,2)] {
  let total=pos+nt;let cn=64usize;let sm=total.div_ceil(cn);
  let kf=nt*qts;let vf=kf+total*kvd;let kh=vf+total*kvd;let vh=kh+total*kvd/2;let oo=vh+total*kvd/2;let scr=(oo+nt*nh*hd+63)/64*64;let len=scr+nt*nh*sm*260;
  let mut a=vec![0f32;len];for(i,x)in a[..kh].iter_mut().enumerate(){*x=((i*17%1009)as f32-504.)/503.;}
  let arena=c.buffer((len*4)as u64,false)?;let params=c.buffer(16,false)?;let set=c.set([(&arena,0,arena.size),(&params,0,16),(&params,0,16),(&params,0,16)]);
  c.upload(&arena,0,&a.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<_>>());
  for(src,dst)in [(kf,kh),(vf,vh)]{for start in (0..total).step_by(64){let n=(total-start).min(64)*kvd;c.begin();c.dispatch("kv_pack",set,&[(src+start*kvd)as u32,(dst+start*kvd/2)as u32,n as u32],((n/2).div_ceil(64)as u32,1,1));c.submit_wait();}}
  let mut packed=vec![0;total*kvd*2];c.download(&arena,kh as u64*4,&mut packed);
  for(i,b)in packed.chunks_exact(2).enumerate(){assert_eq!(u16::from_le_bytes(b.try_into().unwrap()),half::f16::from_f32(a[kf+i]).to_bits(),"NVIDIA exact RNE packing");}
  let mut outputs=Vec::new();
  for(split,combine)in [("attn_s_h","attn_comb"),("attn_s_h2","attn_comb2")]{
   c.begin();c.dispatch(split,set,&[0,qts as u32,kh as u32,vh as u32,kvd as u32,pos as u32,hd as u32,nh as u32,nkv as u32,scr as u32,cn as u32,sm as u32,(1f32/16.).to_bits()],(nh as u32,nt as u32,sm as u32));
   c.dispatch(combine,set,&[0,qts as u32,pos as u32,nh as u32,scr as u32,cn as u32,sm as u32,oo as u32,(nh*hd)as u32],(nh as u32,nt as u32,1));c.submit_wait();
   let mut out=vec![0;nt*nh*hd*4];c.download(&arena,oo as u64*4,&mut out);outputs.push(out);
  }
  assert_eq!(outputs[0],outputs[1],"full split-K output bytes differ");let got=super::decode_kv_readback(&outputs[1],false);
  let hval=|x:f32|half::f16::from_f32(x).to_f64();let mut worst=0f64;
  for t in 0..nt{for h in 0..nh{let n=pos+t+1;let kvh=h/(nh/nkv);let qb=t*qts+h*512;
   let mut scores:Vec<f64>=(0..n).map(|p|(0..hd).map(|d|a[qb+d]as f64*hval(a[kf+p*kvd+kvh*hd+d])).sum::<f64>()/16.).collect();
   let max=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);for s in &mut scores{*s=(*s-max).exp();}let sum:f64=scores.iter().sum();
   for d in 0..hd{let want=(0..n).map(|p|scores[p]*hval(a[vf+p*kvd+kvh*hd+d])).sum::<f64>()/sum/(1.+(-(a[qb+hd+d]as f64)).exp());let value=got[(t*nh+h)*hd+d]as f64;assert!(value.is_finite());worst=worst.max((want-value).abs());}
  }}
  eprintln!("NVIDIA independent f64 oracle pos={pos} nt={nt} exact_RNE=true bitwise_base=true max_abs={worst} completed_dispatches={}",c.completed_dispatches.get());assert!(worst<2e-6);
 }
 Ok(())
}
