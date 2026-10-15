//! Temporary lossless f32 weight expansion, shared across all token tiles in a batch.
//! Original PQ2 words/scales and decode paths are unchanged.
use super::*;
#[spirv(compute(threads(256),entry_point_name="prefill_expand"))]
pub fn prefill_expand(#[spirv(global_invocation_id)] id:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32]){
 let(m,k,off,start,count)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4));if id.x>=count{return;}let i=start+id.x;if i>=m*(k/16){return;}let row=i%m;let q=i/m;
 let word=g!(wq,row*(k/16)+q);let d=g!(wsc,row*(k/128)+q/8);
 unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],|j|{s!(a,off+(q*16+j)*m+row,(((word>>(2*j))&3)as f32-1.)*d);});
}
#[spirv(compute(threads(256),entry_point_name="matmul_dense64"))]
pub fn matmul_dense64(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(workgroup)] wf:&mut[[f32;4];512],#[spirv(workgroup)] xf:&mut[[f32;4];512]){
 let(m,k,nt,xo,xs,yo,ys,add,wo)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7),pu(pc,8));
 let l=lid.x;let tx=l&15;let ty=l>>4;let row0=wg.x*64;let tok0=wg.y*64;
 let(mut c0,mut c1,mut c2,mut c3)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);let mut kc=0;
 while kc<k {
  let mut i=l;while i<512 {let rq=i%16;let kk=i/16;let r=row0+rq*4;let w=if r+3<m {g!(a4,(wo+(kc+kk)*m+r)/4)}else{Vec4::new(if r<m {g!(a,wo+(kc+kk)*m+r)}else{0.},if r+1<m {g!(a,wo+(kc+kk)*m+r+1)}else{0.},if r+2<m {g!(a,wo+(kc+kk)*m+r+2)}else{0.},0.)};s!(wf,i,[w.x,w.y,w.z,w.w]);i+=256;}
  if l<128 {let tq=l&15;let kv=l>>4;let mut xv=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
   s!(xf,kv*4*16+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*16+tq,[x0.y,x1.y,x2.y,x3.y]);
   s!(xf,(kv*4+2)*16+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*16+tq,[x0.w,x1.w,x2.w,x3.w]);
  }bar();
  unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],|kk|{
   let wa=g!(wf,kk*16+ty);let w=Vec4::new(wa[0],wa[1],wa[2],wa[3]);let x=g!(xf,kk*16+tx);
   c0+=w*x[0];c1+=w*x[1];c2+=w*x[2];c3+=w*x[3];
  });bar();kc+=32;
 }
 let acc=[c0,c1,c2,c3];unroll!([0,1,2,3],|r|{let row=row0+ty*4+r;if row<m {unroll!([0,1,2,3],|c|{let t=tok0+tx*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}});
}
