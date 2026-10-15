//! Astra radix3 LUT with token-contiguous scalar LDS to avoid vec4 random-lookup bank conflicts.
//! Same lossless four-trit encoding and scale128 reassociation as prefill_trit.
use super::*;
#[spirv(compute(threads(256),entry_point_name="matmul_trits"))]
pub fn matmul_trits(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],#[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
 #[spirv(workgroup)] pairs:&mut[f32;1152],#[spirv(workgroup)] tab:&mut[f32;5184]) {
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l&15;let ry=l>>4;let row0=wg.x*512;let t=wg.y*16+tx;let padded=m.div_ceil(4)*4;
 let mut acc=[0f32;32];let mut kb=0;
 while kb<k {
  let mut partial=[0f32;32];let mut wi=0;
  while wi<8 {
   let mut u=l;
   while u<1152 {let tok=wg.y*16+(u&15);let code=(u>>4)%9;let pair=(u>>4)/9;
    let x=if tok<nt {g!(a,xo+tok*xs+kb+wi*16+pair*2)}else{0.};let y=if tok<nt {g!(a,xo+tok*xs+kb+wi*16+pair*2+1)}else{0.};
    s!(pairs,u,x*((code%3)as f32-1.)+y*((code/3)as f32-1.));u+=256;
   }bar();u=l;
   while u<5184 {let tok=u&15;let code=(u>>4)%81;let group=(u>>4)/81;
    s!(tab,u,g!(pairs,(group*18+code%9)*16+tok)+g!(pairs,(group*18+9+code/9)*16+tok));u+=256;
   }bar();
   unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],|r|{
    let row=(row0+ry+r*16).min(m-1);let wd=g!(wq,(kb/16+wi)*padded+row);let mut v=g!(partial,r);
    unroll!([0,1,2,3],|p|{v+=g!(tab,p*1296+((wd>>(p*8))&255)*16+tx);});s!(partial,r,v);
   });bar();wi+=1;
  }
  unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],|r|{let row=(row0+ry+r*16).min(m-1);s!(acc,r,g!(acc,r)+g!(partial,r)*g!(wsc,(kb/128)*padded+row));});kb+=128;
 }
 unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],|r|{let row=row0+ry+r*16;if row<m && t<nt {let yi=yo+t*ys+row;let v=g!(acc,r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});
}
