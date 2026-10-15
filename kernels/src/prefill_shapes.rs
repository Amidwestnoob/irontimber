//! Resource-shape experiments: preserve r012 scalar accumulation and quantization.
use super::*;
macro_rules! shape {($name:ident,$entry:literal,$threads:literal,$rows:literal,$toks:literal,$rq:literal,$tq:literal,$kc:literal,[$($kk:literal),*])=>{
#[spirv(compute(threads($threads),entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
#[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
#[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
#[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
#[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
#[spirv(workgroup)] wf:&mut[[f32;4];$kc*$rq],#[spirv(workgroup)] xf:&mut[[f32;4];$kc*$tq]){
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l%$tq;let ty=l/$tq;let r0=wg.x*$rows;let t0=wg.y*$toks;let mut acc=[Vec4::ZERO;4];let mut base=0;
 while base<k {
  let mut u=l;while u<$tq*($kc/4){let tq=u%$tq;let kv=u/$tq;let mut x=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=t0+tq*4+c;s!(x,c,if t<nt{g!(a4,(xo+t*xs+base)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(x,0),g!(x,1),g!(x,2),g!(x,3));
   s!(xf,(kv*4)*$tq+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*$tq+tq,[x0.y,x1.y,x2.y,x3.y]);s!(xf,(kv*4+2)*$tq+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*$tq+tq,[x0.w,x1.w,x2.w,x3.w]);u+=$threads;
  }
  let mut u=l;while u<$rq*($kc/4){let rq=u%$rq;let jg=u/$rq;let wi=jg/4;let jq=jg%4;let mut wd=[0u32;4];let mut dd=[0f32;4];
   unroll!([0,1,2,3],|c|{let r=(r0+rq*4+c).min(m-1);s!(wd,c,g!(wq,r*(k/16)+base/16+wi));s!(dd,c,g!(wsc,r*(k/128)+(base+wi*16)/128));});
   unroll!([0,1,2,3],|jj|{let j=jq*4+jj;let f=|c:u32|(((g!(wd,c)>>(2*j))&3)as f32-1.)*g!(dd,c);s!(wf,(wi*16+j)*$rq+rq,[f(0),f(1),f(2),f(3)]);});u+=$threads;
  }
  bar();unroll!([$($kk),*],|kk|{let w=g!(wf,kk*$rq+ty);let x=g!(xf,kk*$tq+tx);let w=Vec4::new(w[0],w[1],w[2],w[3]);unroll!([0,1,2,3],|c|{s!(acc,c,g!(acc,c)+w*x[c as usize]);});});bar();base+=$kc;
 }
 unroll!([0,1,2,3],|r|{let row=r0+ty*4+r;if row<m{unroll!([0,1,2,3],|c|{let t=t0+tx*4+c;if t<nt{let i=yo+t*ys+row;let v=get_c(g!(acc,c),r);s!(a,i,if add!=0{g!(a,i)+v}else{v});}});}});
}
};}
shape!(matmul_pf128x32,"matmul_pf128x32",256,128,32,32,8,32,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
shape!(matmul_pf64x32,"matmul_pf64x32",128,64,32,16,8,32,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
shape!(matmul_pf64k64,"matmul_pf64k64",256,64,64,16,16,64,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63]);
