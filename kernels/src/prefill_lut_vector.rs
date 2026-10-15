//! Pair-LUT with vec4 token reads: amortizes code/address arithmetic across four outputs.
//! Same full PQ2 code space and block128 reassociation as matmul_lut2;16KiB LDS.
use super::*;
#[spirv(compute(threads(256),entry_point_name="matmul_lutv"))]
pub fn matmul_lutv(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
 #[spirv(workgroup)] tab:&mut[Vec4;1024]) {
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tq=l&7;let ry=l>>3;let row0=wg.x*64;let tok0=wg.y*32+tq*4;
 let(mut a0,mut a1)=(Vec4::ZERO,Vec4::ZERO);let mut kb=0;
 while kb<k {
  let(mut c0,mut c1)=(Vec4::ZERO,Vec4::ZERO);let mut wi=0;
  while wi<8 {
   let pair=(l>>3)&7;let b=l>>6;let mut xx=[0f32;4];let mut yy=[0f32;4];
   unroll!([0,1,2,3],|c|{let t=tok0+c;s!(xx,c,if t<nt {g!(a,xo+t*xs+kb+wi*16+pair*2)}else{0.});s!(yy,c,if t<nt {g!(a,xo+t*xs+kb+wi*16+pair*2+1)}else{0.});});
   let x=Vec4::new(xx[0],xx[1],xx[2],xx[3]);let y=Vec4::new(yy[0],yy[1],yy[2],yy[3]);
   let v=match b {0=>-y,1=>Vec4::ZERO,2=>y,_=>y+y};let base=pair*128+b*32+tq;
   s!(tab,base,v-x);s!(tab,base+8,v);s!(tab,base+16,v+x);s!(tab,base+24,v+(x+x));
   bar();
   let w0=g!(wq,(row0+ry).min(m-1)*(k/16)+kb/16+wi);
   let w1=g!(wq,(row0+ry+32).min(m-1)*(k/16)+kb/16+wi);
   unroll!([0,1,2,3,4,5,6,7],|p|{
    c0+=g!(tab,p*128+((w0>>(p*4))&15)*8+tq);
    c1+=g!(tab,p*128+((w1>>(p*4))&15)*8+tq);
   });bar();wi+=1;
  }
  a0+=c0*g!(wsc,(row0+ry).min(m-1)*(k/128)+kb/128);
  a1+=c1*g!(wsc,(row0+ry+32).min(m-1)*(k/128)+kb/128);kb+=128;
 }
 unroll!([0,1],|r|{let row=row0+ry+r*32;let v=if r==0 {a0}else{a1};if row<m {unroll!([0,1,2,3],|c|{let t=tok0+c;if t<nt {let yi=yo+t*ys+row;s!(a,yi,if add!=0 {g!(a,yi)+get_c(v,c)}else{get_c(v,c)});}});}});
}
