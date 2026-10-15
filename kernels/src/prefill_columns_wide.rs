//! 64row x128token column-packed prefill,32 independent accumulator scalars/lane.
//! Short four-step loop bounds live LDS operands; unlike rejected64accumulator tiles.
use super::*;
macro_rules! tile {($name:ident,$entry:literal,$U:literal,[$($u:literal),*])=>{
#[spirv(compute(threads(256),entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],#[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[UVec4],#[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[Vec4],
 #[spirv(workgroup)] wf:&mut[[f32;4];512],#[spirv(workgroup)] xf:&mut[[f32;4];1024]){
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l&15;let ty=l>>4;let row0=wg.x*64;let tok0=wg.y*128;let padded=m.div_ceil(4)*4;
 let(mut c0,mut c1,mut c2,mut c3,mut c4,mut c5,mut c6,mut c7)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);let mut kc=0;
 while kc<k {
  let tq=l&31;let kv=l>>5;let mut xv=[Vec4::ZERO;4];
  unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
  let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
  s!(xf,kv*4*32+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*32+tq,[x0.y,x1.y,x2.y,x3.y]);
  s!(xf,(kv*4+2)*32+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*32+tq,[x0.w,x1.w,x2.w,x3.w]);
  if l<128 {let rq=l&15;let wi=(l>>4)&1;let jq=l>>5;let r=(row0+rq*4).min(padded-4);
   let wd=g!(wq,((kc/16+wi)*padded+r)/4);let dd=g!(wsc,((kc/128)*padded+r)/4);
   unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);let f=|c:u32|(((comp(wd,c)>>sh)&3)as f32-1.)*get_c(dd,c);s!(wf,(wi*16+jq*4+j)*16+rq,[f(0),f(1),f(2),f(3)]);});
  }bar();let mut kk=0;
  while kk<32 {unroll!([$($u),*],|u|{let wa=g!(wf,(kk+u)*16+ty);let w=Vec4::new(wa[0],wa[1],wa[2],wa[3]);let x=g!(xf,(kk+u)*32+tx);let y=g!(xf,(kk+u)*32+tx+16);
   c0+=w*x[0];c1+=w*x[1];c2+=w*x[2];c3+=w*x[3];c4+=w*y[0];c5+=w*y[1];c6+=w*y[2];c7+=w*y[3];
  });kk+=$U;}bar();kc+=32;
 }
 let acc=[c0,c1,c2,c3,c4,c5,c6,c7];unroll!([0,1],|tt|{unroll!([0,1,2,3],|r|{let row=row0+ty*4+r;if row<m {unroll!([0,1,2,3],|c|{let t=tok0+(tx+tt*16)*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,tt*4+c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}});});
}
};}
tile!(matmul_col128u4,"matmul_col128u4",4,[0,1,2,3]);
tile!(matmul_col128u8,"matmul_col128u8",8,[0,1,2,3,4,5,6,7]);
tile!(matmul_col128u32,"matmul_col128u32",32,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
