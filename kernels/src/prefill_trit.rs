//! Astra: lossless radix3 four-weight lookup. Codes must be0..2, checked on CPU.
//! 512rows x16tokens; four-code combinations reuse two nine-entry pair tables.
//! 25,344bytes LDS; full scale128 grouping, no activation quantization.
use super::*;
#[spirv(compute(threads(256),entry_point_name="matmul_trit4"))]
pub fn matmul_trit4(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
 #[spirv(workgroup)] pairs:&mut[Vec4;288],#[spirv(workgroup)] tab:&mut[Vec4;1296]) {
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tq=l&3;let ry=l>>2;let row0=wg.x*512;let tok0=wg.y*16;let padded=m.div_ceil(4)*4;
 let mut acc=[Vec4::ZERO;8];let mut kb=0;
 while kb<k {
  let mut partial=[Vec4::ZERO;8];let mut wi=0;
  while wi<8 {
   let mut u=l;
   while u<288 {
    let tquad=u&3;let code=(u>>2)%9;let pair=(u>>2)/9;
    let mut xx=[0f32;4];let mut yy=[0f32;4];
    unroll!([0,1,2,3],|c|{let t=tok0+tquad*4+c;s!(xx,c,if t<nt {g!(a,xo+t*xs+kb+wi*16+pair*2)}else{0.});s!(yy,c,if t<nt {g!(a,xo+t*xs+kb+wi*16+pair*2+1)}else{0.});});
    let x=Vec4::new(xx[0],xx[1],xx[2],xx[3]);let y=Vec4::new(yy[0],yy[1],yy[2],yy[3]);
    s!(pairs,u,x*((code%3)as f32-1.)+y*((code/3)as f32-1.));u+=256;
   }
   bar();u=l;
   while u<1296 {
    let tquad=u&3;let code=(u>>2)%81;let group=(u>>2)/81;
    s!(tab,u,g!(pairs,(group*18+code%9)*4+tquad)+g!(pairs,(group*18+9+code/9)*4+tquad));u+=256;
   }
   bar();
   unroll!([0,1,2,3,4,5,6,7],|r|{let row=(row0+ry+r*64).min(m-1);let wd=g!(wq,(kb/16+wi)*padded+row);let mut v=g!(partial,r);
    unroll!([0,1,2,3],|p|{v+=g!(tab,p*324+((wd>>(p*8))&255)*4+tq);});s!(partial,r,v);
   });bar();wi+=1;
  }
  unroll!([0,1,2,3,4,5,6,7],|r|{let row=(row0+ry+r*64).min(m-1);s!(acc,r,g!(acc,r)+g!(partial,r)*g!(wsc,(kb/128)*padded+row));});kb+=128;
 }
 unroll!([0,1,2,3,4,5,6,7],|r|{let row=row0+ry+r*64;let v=g!(acc,r);if row<m {unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;if t<nt {let yi=yo+t*ys+row;s!(a,yi,if add!=0 {g!(a,yi)+get_c(v,c)}else{get_c(v,c)});}});}});
}
