//! Astra: bank-separated LDS operands. Same 64x64 tile and ascending-k arithmetic.
//! Component-major storage removes the four-word bank stride of each lane's quad.
use super::*;
#[spirv(compute(threads(256),entry_point_name="matmul_colsoa"))]
pub fn matmul_colsoa(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[UVec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[Vec4],
 #[spirv(workgroup)] wf:&mut[f32;2048],#[spirv(workgroup)] xf:&mut[f32;2048]) {
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l&15;let ty=l>>4;let row0=wg.x*64;let tok0=wg.y*64;let padded=(m+3)/4*4;
 let(mut c0,mut c1,mut c2,mut c3)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);let mut kc=0;
 while kc<k {
  if l<128 {
   let tq=l&15;let kv=l>>4;
   unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;let x=if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO};
    s!(xf,(kv*4)*64+c*16+tq,x.x);s!(xf,(kv*4+1)*64+c*16+tq,x.y);
    s!(xf,(kv*4+2)*64+c*16+tq,x.z);s!(xf,(kv*4+3)*64+c*16+tq,x.w);
   });
  }else{
   let u=l-128;let rq=u&15;let wi=(u>>4)&1;let jq=u>>5;let r=(row0+rq*4).min(padded-4);
   let wd=g!(wq,((kc/16+wi)*padded+r)/4);let dd=g!(wsc,((kc/128)*padded+r)/4);
   unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);unroll!([0,1,2,3],|c|{
    s!(wf,(wi*16+jq*4+j)*64+c*16+rq,(((comp(wd,c)>>sh)&3)as f32-1.)*get_c(dd,c));
   });});
  }
  bar();
  unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],|kk|{
   let w=Vec4::new(g!(wf,kk*64+ty),g!(wf,kk*64+16+ty),g!(wf,kk*64+32+ty),g!(wf,kk*64+48+ty));
   c0+=w*g!(xf,kk*64+tx);c1+=w*g!(xf,kk*64+16+tx);c2+=w*g!(xf,kk*64+32+tx);c3+=w*g!(xf,kk*64+48+tx);
  });bar();kc+=32;
 }
 let acc=[c0,c1,c2,c3];
 unroll!([0,1,2,3],|r|{let row=row0+ty*4+r;if row<m {unroll!([0,1,2,3],|c|{let t=tok0+tx*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}});
}
