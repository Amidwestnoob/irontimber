//! 128x128/64-accumulator test with post-tested outer loop and column loads.
//! Retained rb128 still uses164 VGPRs/161 moves; c2 proved post-test loop
//! eliminates duplicate live accumulators. This tests that specific codegen gap.
use super::*;
macro_rules! square {($name:ident,$entry:literal,$U:literal,[$($kk:literal),*])=>{
#[spirv(compute(threads(256),entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],#[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[UVec4],#[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[Vec4],
 #[spirv(workgroup)] wf:&mut[[f32;4];1024],#[spirv(workgroup)] xf:&mut[[f32;4];1024]){
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l&15;let ty=l>>4;let row0=wg.x*128;let tok0=wg.y*128;let padded=m.div_ceil(4)*4;
 let(mut c0,mut c1,mut c2,mut c3,mut c4,mut c5,mut c6,mut c7)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);
 let(mut c8,mut c9,mut c10,mut c11,mut c12,mut c13,mut c14,mut c15)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);let mut kc=0;
 if k==0 {return;} loop {
  let tq=l%32;let kv=l/32;let mut xv=[Vec4::ZERO;4];
  unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
  let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
  s!(xf,kv*4*32+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*32+tq,[x0.y,x1.y,x2.y,x3.y]);
  s!(xf,(kv*4+2)*32+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*32+tq,[x0.w,x1.w,x2.w,x3.w]);
  let rq=l%32;let wi=(l/32)%2;let jq=l/64;let r=(row0+rq*4).min(padded-4);
  let wd=g!(wq,((kc/16+wi)*padded+r)/4);let dd=g!(wsc,((kc/128)*padded+r)/4);
  unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);let f=|c:u32|(((comp(wd,c)>>sh)&3)as f32-1.)*get_c(dd,c);s!(wf,(wi*16+jq*4+j)*32+rq,[f(0),f(1),f(2),f(3)]);});bar();
  let mut base=0;loop {
  unroll!([$($kk),*],|ki|{let kk=base+ki;
   let wa=g!(wf,kk*32+ty);let wb=g!(wf,kk*32+ty+16);let w0=Vec4::new(wa[0],wa[1],wa[2],wa[3]);let w1=Vec4::new(wb[0],wb[1],wb[2],wb[3]);let xa=g!(xf,kk*32+tx);let xb=g!(xf,kk*32+tx+16);
   c0+=w0*xa[0];c1+=w0*xa[1];c2+=w0*xa[2];c3+=w0*xa[3];
   c4+=w0*xb[0];c5+=w0*xb[1];c6+=w0*xb[2];c7+=w0*xb[3];
   c8+=w1*xa[0];c9+=w1*xa[1];c10+=w1*xa[2];c11+=w1*xa[3];
   c12+=w1*xb[0];c13+=w1*xb[1];c14+=w1*xb[2];c15+=w1*xb[3];
  });base+=$U;if base>=32 {break;}
  }bar();kc+=32;if kc>=k {break;}
 }
 let acc=[c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15];
 unroll!([0,1],|rr|{unroll!([0,1],|tt|{unroll!([0,1,2,3],|r|{let row=row0+(ty+rr*16)*4+r;if row<m {unroll!([0,1,2,3],|c|{let t=tok0+(tx+tt*16)*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,rr*8+tt*4+c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}});});});
}
};}
square!(matmul_col_sq,"matmul_col_sq",32,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
square!(matmul_col_sq4,"matmul_col_sq4",4,[0,1,2,3]);
