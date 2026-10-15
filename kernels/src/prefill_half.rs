//! Lossless f16 weight LDS storage: production scales originate from GGUF f16.
//! Activations and accumulation stay f32. K16 reduces LDS to 6KiB/workgroup.
//! Host rejects nonrepresentable/overflow/subnormal scales before this path is used.
use super::*;
#[spirv(compute(threads(256),entry_point_name="matmul_pf_h16"))]
pub fn matmul_pf_h16(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
#[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
#[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
#[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
#[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
#[spirv(workgroup)] wf:&mut[spirv_std::glam::UVec2;256],#[spirv(workgroup)] xf:&mut[[f32;4];256]){
 use spirv_std::float::{vec2_to_f16x2 as pack,f16x2_to_vec2 as unpack};use spirv_std::glam::{Vec2,UVec2};
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l&15;let ty=l>>4;let r0=wg.x*64;let t0=wg.y*64;let mut acc=[Vec4::ZERO;4];let mut base=0;
 while base<k{
  if l<64 {let tq=l&15;let kv=l>>4;let mut x=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=t0+tq*4+c;s!(x,c,if t<nt{g!(a4,(xo+t*xs+base)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(x,0),g!(x,1),g!(x,2),g!(x,3));
   s!(xf,(kv*4)*16+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*16+tq,[x0.y,x1.y,x2.y,x3.y]);s!(xf,(kv*4+2)*16+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*16+tq,[x0.w,x1.w,x2.w,x3.w]);
  }else if l<128 {let u=l-64;let rq=u&15;let jg=u>>4;let mut wd=[0u32;4];let mut dd=[0f32;4];
   unroll!([0,1,2,3],|c|{let r=(r0+rq*4+c).min(m-1);s!(wd,c,g!(wq,r*(k/16)+base/16));s!(dd,c,g!(wsc,r*(k/128)+base/128));});
   unroll!([0,1,2,3],|jj|{let j=jg*4+jj;let f=|c:u32|(((g!(wd,c)>>(2*j))&3)as f32-1.)*g!(dd,c);s!(wf,j*16+rq,UVec2::new(pack(Vec2::new(f(0),f(1))),pack(Vec2::new(f(2),f(3)))));});
  }
  bar();unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],|kk|{let wa=g!(wf,kk*16+ty);let w0=unpack(wa.x);let w1=unpack(wa.y);let w=Vec4::new(w0.x,w0.y,w1.x,w1.y);let x=g!(xf,kk*16+tx);unroll!([0,1,2,3],|c|{s!(acc,c,g!(acc,c)+w*x[c as usize]);});});bar();base+=16;
 }
 unroll!([0,1,2,3],|r|{let row=r0+ty*4+r;if row<m{unroll!([0,1,2,3],|c|{let t=t0+tx*4+c;if t<nt{let i=yo+t*ys+row;let v=get_c(g!(acc,c),r);s!(a,i,if add!=0{g!(a,i)+v}else{v});}});}});
}
