//! Keep64x64x32 geometry and lossless column layout; shorten inner unroll/live operands.
//!16 independent accumulator scalars, not rejected32/64-scalar register tiles.
use super::*;
#[inline(always)] fn mac(w:Vec4,x:f32,a:Vec4,fma:bool)->Vec4 {if fma {Vec4::new(w.x.mul_add(x,a.x),w.y.mul_add(x,a.y),w.z.mul_add(x,a.z),w.w.mul_add(x,a.w))}else{a+w*x}}
macro_rules! short {($name:ident,$entry:literal,$FMA:literal,$U:literal,[$($u:literal),*])=>{
#[spirv(compute(threads(256),entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],#[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[UVec4],#[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[Vec4],
 #[spirv(workgroup)] wf:&mut[[f32;4];512],#[spirv(workgroup)] xf:&mut[[f32;4];512]){
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l&15;let ty=l>>4;let row0=wg.x*64;let tok0=wg.y*64;let padded=m.div_ceil(4)*4;
 let(mut c0,mut c1,mut c2,mut c3)=(Vec4::ZERO,Vec4::ZERO,Vec4::ZERO,Vec4::ZERO);let mut kc=0;
 while kc<k {
  if l<128 {let tq=l&15;let kv=l>>4;let mut xv=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
   s!(xf,kv*4*16+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*16+tq,[x0.y,x1.y,x2.y,x3.y]);
   s!(xf,(kv*4+2)*16+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*16+tq,[x0.w,x1.w,x2.w,x3.w]);
  }else{let u=l-128;let rq=u&15;let wi=(u>>4)&1;let jq=u>>5;let r=(row0+rq*4).min(padded-4);
   let wd=g!(wq,((kc/16+wi)*padded+r)/4);let dd=g!(wsc,((kc/128)*padded+r)/4);
   unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);let f=|c:u32|(((comp(wd,c)>>sh)&3)as f32-1.)*get_c(dd,c);s!(wf,(wi*16+jq*4+j)*16+rq,[f(0),f(1),f(2),f(3)]);});
  }bar();let mut kk=0;
  while kk<32 {unroll!([$($u),*],|u|{let wa=g!(wf,(kk+u)*16+ty);let w=Vec4::new(wa[0],wa[1],wa[2],wa[3]);let x=g!(xf,(kk+u)*16+tx);c0=mac(w,x[0],c0,$FMA);c1=mac(w,x[1],c1,$FMA);c2=mac(w,x[2],c2,$FMA);c3=mac(w,x[3],c3,$FMA);});kk+=$U;}
  bar();kc+=32;
 }
 let acc=[c0,c1,c2,c3];unroll!([0,1,2,3],|r|{let row=row0+ty*4+r;if row<m {unroll!([0,1,2,3],|c|{let t=tok0+tx*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}});
}
};}
short!(matmul_col_u4,"matmul_col_u4",false,4,[0,1,2,3]);
short!(matmul_col_u8,"matmul_col_u8",false,8,[0,1,2,3,4,5,6,7]);
short!(matmul_col_u16,"matmul_col_u16",false,16,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
short!(matmul_col_fma8,"matmul_col_fma8",true,8,[0,1,2,3,4,5,6,7]);
short!(matmul_col_fma32,"matmul_col_fma32",true,32,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
