//! Profile-gated GQA sharing: two heads x four queries, or six heads x one.
//! Same per-query K/dot, reduction tree, V order and 260-word partial ABI.
//! Actual 7k profile has14.4% attention including combine, not short-profile0.4%.
use super::*;
macro_rules! gqa {($name:ident,$entry:literal,$H:literal,$Q:literal,$T:literal,[$($t:literal),*])=>{
#[spirv(compute(threads(64),entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],#[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] u4:&[UVec4],#[spirv(storage_buffer,descriptor_set=0,binding=0)] au:&[u32],
 #[spirv(workgroup)] qv:&mut[Vec4;64*$T],#[spirv(workgroup)] sc:&mut[f32;256*$T],#[spirv(workgroup)] red:&mut[f32;64*$T]){
 use spirv_std::float::f16x2_to_vec2 as h2;
 let(qo,qts,kco,vco,kvd,pos0)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5));
 let(nh,nkv,scr,cn,smax,nt)=(pu(pc,7),pu(pc,8),pu(pc,9),pu(pc,10),pu(pc,11),pu(pc,13));let scale=pf(pc,12);
 if cn>256 || nkv==0 || nh/nkv!=$H*(6/$H) || nh%nkv!=0 {return;}
 let(h0,t0,s,l)=(wg.x*$H,wg.y*$Q,wg.z,lid.x);let p0=s*cn;let nmax=pos0+(t0+$Q).min(nt);
 if p0>=nmax {return;}let p1=(p0+cn).min(nmax);let kh=h0/(nh/nkv);
 unroll!([$($t),*],|t|{let qt=t0+t%$Q;let h=h0+t/$Q;let qb=(qo+qt*qts+h*512)/4;s!(qv,t*64+l,if qt<nt {g!(a4,qb+l)}else{Vec4::ZERO});});bar();
 let mut mx=[-1e30f32;$T];let mut p=p0+l;
 while p<p1 {
  let kb=(kco+(p*kvd+kh*256)/2)/4;let mut d=[Vec4::ZERO;$T];let mut i=0;
  while i<32 {let w=g!(u4,kb+i);let(x0,x1,x2,x3)=(h2(w.x),h2(w.y),h2(w.z),h2(w.w));let k0=Vec4::new(x0.x,x0.y,x1.x,x1.y);let k1=Vec4::new(x2.x,x2.y,x3.x,x3.y);
   unroll!([$($t),*],|t|{s!(d,t,g!(d,t)+(k0*g!(qv,t*64+2*i)+k1*g!(qv,t*64+2*i+1)));});i+=1;
  }
  unroll!([$($t),*],|t|{if t0+t%$Q<nt && p<pos0+t0+t%$Q+1 {let v=g!(d,t).dot(Vec4::ONE)*scale;s!(sc,t*256+p-p0,v);s!(mx,t,g!(mx,t).max(v));}});p+=64;
 }
 unroll!([$($t),*],|t|{s!(red,t*64+l,g!(mx,t));});bar();
 unroll!([5,4,3,2,1,0],|sh|{let st=1u32<<sh;if l<st {unroll!([$($t),*],|t|{s!(red,t*64+l,g!(red,t*64+l).max(g!(red,t*64+l+st)));});}bar();});
 let mut m=[0f32;$T];unroll!([$($t),*],|t|{s!(m,t,g!(red,t*64));});bar();let mut sum=[0f32;$T];p=p0+l;
 while p<p1 {unroll!([$($t),*],|t|{if t0+t%$Q<nt && p<pos0+t0+t%$Q+1 {let e=(g!(sc,t*256+p-p0)-g!(m,t)).exp();s!(sc,t*256+p-p0,e);s!(sum,t,g!(sum,t)+e);}});p+=64;}
 unroll!([$($t),*],|t|{s!(red,t*64+l,g!(sum,t));});bar();
 unroll!([5,4,3,2,1,0],|sh|{let st=1u32<<sh;if l<st {unroll!([$($t),*],|t|{s!(red,t*64+l,g!(red,t*64+l)+g!(red,t*64+l+st));});}bar();});
 let mut o=[Vec4::ZERO;$T];p=p0;
 while p<p1 {let vb=vco+(p*kvd+kh*256)/2+2*l;let(y0,y1)=(h2(g!(au,vb)),h2(g!(au,vb+1)));let v=Vec4::new(y0.x,y0.y,y1.x,y1.y);
  unroll!([$($t),*],|t|{if t0+t%$Q<nt && p<pos0+t0+t%$Q+1 {s!(o,t,g!(o,t)+v*g!(sc,t*256+p-p0));}});p+=1;
 }
 unroll!([$($t),*],|t|{let qt=t0+t%$Q;let h=h0+t/$Q;if qt<nt && p0<pos0+qt+1 {let pb=scr+((qt*nh+h)*smax+s)*260;if l==0 {s!(a,pb,g!(m,t));s!(a,pb+1,g!(red,t*64));}let ob=pb+4+l*4;let v=g!(o,t);s!(a,ob,v.x);s!(a,ob+1,v.y);s!(a,ob+2,v.z);s!(a,ob+3,v.w);}});
}
};}
gqa!(attn_gqa8,"attn_gqa8",2,4,8,[0,1,2,3,4,5,6,7]);
gqa!(attn_gqa6,"attn_gqa6",6,1,6,[0,1,2,3,4,5]);
// 6-head tile requires13.5KiB LDS; smaller head-sharing groups increase resident
// one-wave workgroups while retaining the exact reduction and260-word ABI.
gqa!(attn_gqa3,"attn_gqa3",3,1,3,[0,1,2]);
gqa!(attn_gqa2,"attn_gqa2",2,1,2,[0,1]);
