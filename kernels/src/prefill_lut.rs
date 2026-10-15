//! Astra pair-LUT prefill: full original PQ2 code space (including code3=2).
//! For each 128-weight scale block, sum two-term activation combinations then scale.
//! This reassociates f32 arithmetic; it does not requantize inputs or weights.
//! 64 rows x32 tokens, 256 lanes, 16KiB LDS, eight outputs/lane. Original weight ABI.
use super::*;
#[spirv(compute(threads(256),entry_point_name="matmul_lut2"))]
pub fn matmul_lut2(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
 #[spirv(workgroup)] tab:&mut[f32;4096]) {
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let t=wg.y*32+(l&31);let ry=l>>5;let row0=wg.x*64;
 let mut acc=[0f32;8];let mut kb=0;
 while kb<k {
  let mut partial=[0f32;8];let mut wi=0;
  while wi<8 {
   // One lane generates all16 code combinations for one pair and one token.
   let x=if t<nt {g!(a,xo+t*xs+kb+wi*16+ry*2)}else{0.};
   let y=if t<nt {g!(a,xo+t*xs+kb+wi*16+ry*2+1)}else{0.};
   let base=ry*512+(l&31);
   unroll!([0,1,2,3],|b|{let v=match b {0=>-y,1=>0.,2=>y,_=>y+y};
    s!(tab,base+(b*4)*32,v-x);s!(tab,base+(b*4+1)*32,v);
    s!(tab,base+(b*4+2)*32,v+x);s!(tab,base+(b*4+3)*32,v+(x+x));
   });
   bar();
   unroll!([0,1,2,3,4,5,6,7],|r|{let row=(row0+ry+r*8).min(m-1);let wd=g!(wq,row*(k/16)+kb/16+wi);
    let mut v=g!(partial,r);
    unroll!([0,1,2,3,4,5,6,7],|p|{v+=g!(tab,p*512+((wd>>(p*4))&15)*32+(l&31));});
    s!(partial,r,v);
   });
   bar();wi+=1;
  }
  unroll!([0,1,2,3,4,5,6,7],|r|{let row=(row0+ry+r*8).min(m-1);let d=g!(wsc,row*(k/128)+kb/128);s!(acc,r,g!(acc,r)+g!(partial,r)*d);});
  kb+=128;
 }
 if t<nt {unroll!([0,1,2,3,4,5,6,7],|r|{let row=row0+ry+r*8;if row<m {let yi=yo+t*ys+row;let v=g!(acc,r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}
}
