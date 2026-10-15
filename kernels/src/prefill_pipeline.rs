//! Astra pingpong16: load next tile while computing previous; one barrier per16steps.
//! Total LDS remains16KiB, like original column32; no reassociation.
use super::*;
#[spirv(compute(threads(256),entry_point_name="matmul_col_pipe"))]
pub fn matmul_col_pipe(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[UVec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[Vec4],
 #[spirv(workgroup)] wf:&mut[[f32;4];512],#[spirv(workgroup)] xf:&mut[[f32;4];512]) {
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l&15;let ty=l>>4;let row0=wg.x*64;let tok0=wg.y*64;let padded=(m+3)/4*4;
 let mut acc=[Vec4::ZERO;4];let mut kc=0;
 while kc<=k {
  let wb=((kc/16)&1)*256;let rb=256-wb;
  if kc<k {if l<64 {
   let tq=l&15;let kv=l>>4;let mut xv=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
   s!(xf,wb+kv*4*16+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,wb+(kv*4+1)*16+tq,[x0.y,x1.y,x2.y,x3.y]);
   s!(xf,wb+(kv*4+2)*16+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,wb+(kv*4+3)*16+tq,[x0.w,x1.w,x2.w,x3.w]);
  }else if l<128 {
   let u=l-64;let rq=u&15;let wi=0;let jq=u>>4;
   // A complete padded row quad is always in bounds, including m%4 != 0.
   let r=(row0+rq*4).min(padded-4);
   let wd=g!(wq,((kc/16+wi)*padded+r)/4);let dd=g!(wsc,((kc/128)*padded+r)/4);
   unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);let f=|c:u32|(((comp(wd,c)>>sh)&3)as f32-1.)*get_c(dd,c);
    s!(wf,wb+(wi*16+jq*4+j)*16+rq,[f(0),f(1),f(2),f(3)]);
   });
  }}
  if kc>0 {
  unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],|kk|{
   let wa=g!(wf,rb+kk*16+ty);let xa=g!(xf,rb+kk*16+tx);let w=Vec4::new(wa[0],wa[1],wa[2],wa[3]);
   s!(acc,0,g!(acc,0)+w*xa[0]);s!(acc,1,g!(acc,1)+w*xa[1]);s!(acc,2,g!(acc,2)+w*xa[2]);s!(acc,3,g!(acc,3)+w*xa[3]);
  });}
  bar();kc+=16;
 }
 unroll!([0,1,2,3],|r|{let row=row0+ty*4+r;if row<m {unroll!([0,1,2,3],|c|{let t=tok0+tx*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}});
}
