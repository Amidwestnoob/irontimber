//! 32-accumulator column tiles without the retained r7/r14 one-trip kk loop.
//! CPU retained-ISA audit found 32-copy preheader + backedge sets and128 VGPRs
//! even at U32. Fully static expansion removes that extra induction/phi layer.
//! Same ascending K, decode, masks and contraction; no split-K approximation.
use super::*;
macro_rules! direct {($name:ident,$entry:literal,$RQ:literal,$TQ:literal,$RR:literal,$TT:literal,$TH:literal,$TX:literal,$KT:literal,[$($kk:literal),*])=>{
#[spirv(compute(threads($TH),entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],#[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[UVec4],#[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[Vec4],
 #[spirv(workgroup)] wf:&mut[[f32;4];$KT*$RQ],#[spirv(workgroup)] xf:&mut[[f32;4];$KT*$TQ]){
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l%$TX;let ty=l/$TX;let row0=wg.x*($RQ*4);let tok0=wg.y*($TQ*4);let padded=m.div_ceil(4)*4;
 let(mut c0,mut c1,mut c2,mut c3,mut c4,mut c5,mut c6,mut c7)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);let mut kc=0;
 if k==0 {return;}
 loop {
  if l<($KT/4)*$TQ {let tq=l%$TQ;let kv=l/$TQ;let mut xv=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
   s!(xf,kv*4*$TQ+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*$TQ+tq,[x0.y,x1.y,x2.y,x3.y]);
   s!(xf,(kv*4+2)*$TQ+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*$TQ+tq,[x0.w,x1.w,x2.w,x3.w]);
  }
  if l<($KT/4)*$RQ {let rq=l%$RQ;let wi=(l/$RQ)%($KT/16);let jq=l/($RQ*($KT/16));let r=(row0+rq*4).min(padded-4);
   let wd=g!(wq,((kc/16+wi)*padded+r)/4);let dd=g!(wsc,((kc/128)*padded+r)/4);
   unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);let f=|c:u32|(((comp(wd,c)>>sh)&3)as f32-1.)*get_c(dd,c);s!(wf,(wi*16+jq*4+j)*$RQ+rq,[f(0),f(1),f(2),f(3)]);});
  }bar();
  unroll!([$($kk),*],|kk|{
   let wa=g!(wf,kk*$RQ+ty);let w=Vec4::new(wa[0],wa[1],wa[2],wa[3]);let x=g!(xf,kk*$TQ+tx);
   c0+=w*x[0];c1+=w*x[1];c2+=w*x[2];c3+=w*x[3];
   let wb=g!(wf,kk*$RQ+ty+$RR*16);let w2=Vec4::new(wb[0],wb[1],wb[2],wb[3]);let y=g!(xf,kk*$TQ+tx+$TT*16);
   c4+=w2*y[0];c5+=w2*y[1];c6+=w2*y[2];c7+=w2*y[3];
  });bar();kc+=$KT;if kc>=k {break;}
 }
 let acc=[c0,c1,c2,c3,c4,c5,c6,c7];
 unroll!([0,1],|half|{unroll!([0,1,2,3],|r|{let row=row0+(ty+half*$RR*16)*4+r;if row<m {unroll!([0,1,2,3],|c|{let t=tok0+(tx+half*$TT*16)*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,half*4+c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}});});
}
};}
direct!(matmul_col_dt,"matmul_col_dt",16,32,0,1,256,16,32,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
direct!(matmul_col_dr,"matmul_col_dr",32,16,1,0,256,16,32,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
direct!(matmul_col_dr16,"matmul_col_dr16",32,16,1,0,256,16,16,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
// Eight waves/32 accumulators, unlike retained16-accumulator512-thread tiles.
// Prior dr16 used64 registers: test that specific occupancy/geometry opportunity.
direct!(matmul_col_dr512,"matmul_col_dr512",32,32,1,0,512,32,16,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
