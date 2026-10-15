//! 8x8 register tiles with independent SSA accumulators, not array-typed loop phis.
//! r1 ACO showed physical VGPR v219 versus v61 for r012; this addresses that mechanism.
use super::*;
macro_rules! register_tile {($name:ident,$entry:literal,$THREADS:literal,$Q:literal,$TX:literal)=>{
#[spirv(compute(threads($THREADS),entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
 #[spirv(workgroup)] wf:&mut[[f32;4];32*$Q],#[spirv(workgroup)] xf:&mut[[f32;4];32*$Q]){
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l%$TX;let ty=l/$TX;let row0=wg.x*($Q*4);let tok0=wg.y*($Q*4);let wpr=k/16;let bpr=k/128;
 let(mut c0,mut c1,mut c2,mut c3,mut c4,mut c5,mut c6,mut c7)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);
 let(mut c8,mut c9,mut c10,mut c11,mut c12,mut c13,mut c14,mut c15)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);
 let mut kc=0;
 while kc<k {
  let mut u=l;
  while u<8*$Q {
   let tq=u%$Q;let kv=u/$Q;let mut xv=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
   s!(xf,kv*4*$Q+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*$Q+tq,[x0.y,x1.y,x2.y,x3.y]);
   s!(xf,(kv*4+2)*$Q+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*$Q+tq,[x0.w,x1.w,x2.w,x3.w]);
   let wi=kv/4;let jq=kv%4;let mut wd=[0u32;4];let mut dd=[0f32;4];
   unroll!([0,1,2,3],|c|{let row=(row0+tq*4+c).min(m-1);s!(wd,c,g!(wq,row*wpr+kc/16+wi));s!(dd,c,g!(wsc,row*bpr+kc/128));});
   unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);let f=|c:u32|(((g!(wd,c)>>sh)&3)as f32-1.)*g!(dd,c);s!(wf,(wi*16+jq*4+j)*$Q+tq,[f(0),f(1),f(2),f(3)]);});
   u+=$THREADS;
  }
  bar();
  unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],|kk|{
   let wa=g!(wf,kk*$Q+ty);let wb=g!(wf,kk*$Q+ty+$Q/2);
   let w0=Vec4::new(wa[0],wa[1],wa[2],wa[3]);let w1=Vec4::new(wb[0],wb[1],wb[2],wb[3]);
   let xa=g!(xf,kk*$Q+tx);let xb=g!(xf,kk*$Q+tx+$Q/2);
   c0+=w0*xa[0];c1+=w0*xa[1];c2+=w0*xa[2];c3+=w0*xa[3];
   c4+=w0*xb[0];c5+=w0*xb[1];c6+=w0*xb[2];c7+=w0*xb[3];
   c8+=w1*xa[0];c9+=w1*xa[1];c10+=w1*xa[2];c11+=w1*xa[3];
   c12+=w1*xb[0];c13+=w1*xb[1];c14+=w1*xb[2];c15+=w1*xb[3];
  });
  bar();kc+=32;
 }
 let acc=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15];
 unroll!([0,1],|rr|{unroll!([0,1],|tt|{unroll!([0,1,2,3],|r|{
  let row=row0+(ty+rr*$Q/2)*4+r;
  if row<m {unroll!([0,1,2,3],|c|{let t=tok0+(tx+tt*$Q/2)*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,rr*8+tt*4+c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}
 });});});
}
};}
register_tile!(matmul_rb64,"matmul_rb64",64,16,8);
register_tile!(matmul_rb128,"matmul_rb128",256,32,16);
