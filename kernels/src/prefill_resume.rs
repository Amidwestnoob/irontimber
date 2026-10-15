//! Continuation: 8x8 register blocking and a bounded 8-step inner loop.
//! Each output preserves increasing-k f32 multiply/add. No quant/cache changes.
use super::*;
macro_rules! resume_tile { ($name:ident,$entry:literal,$THREADS:literal,$QUADS:literal,$TX:literal,$RX:literal,$KUNROLL:literal,[$($kk:literal),*]) => {
#[spirv(compute(threads($THREADS), entry_point_name=$entry))]
pub fn $name(#[spirv(local_invocation_id)] lid:UVec3,#[spirv(workgroup_id)] wg:UVec3,#[spirv(push_constant)] pc:&PC,
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a:&mut[f32],
 #[spirv(storage_buffer,descriptor_set=0,binding=0)] a4:&[Vec4],
 #[spirv(storage_buffer,descriptor_set=0,binding=2)] wq:&[u32],
 #[spirv(storage_buffer,descriptor_set=0,binding=3)] wsc:&[f32],
 #[spirv(workgroup)] wf:&mut[[f32;4];32*$QUADS],
 #[spirv(workgroup)] xf:&mut[[f32;4];32*$QUADS]) {
 let(m,k,nt,xo,xs,yo,ys,add)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
 let l=lid.x;let tx=l%$TX;let ty=l/$TX;let row0=wg.x*($QUADS*4);let tok0=wg.y*($QUADS*4);
 let wpr=k/16;let bpr=k/128;let mut acc=[Vec4::ZERO;4*$RX*$RX];let mut kc=0;
 while kc<k {
  let mut u=l;
  while u<8*$QUADS {
   let tq=u%$QUADS;let kv=u/$QUADS;
   let mut xv=[Vec4::ZERO;4];
   unroll!([0,1,2,3],|c|{let t=tok0+tq*4+c;s!(xv,c,if t<nt {g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
   let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
   s!(xf,(kv*4)*$QUADS+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*$QUADS+tq,[x0.y,x1.y,x2.y,x3.y]);
   s!(xf,(kv*4+2)*$QUADS+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*$QUADS+tq,[x0.w,x1.w,x2.w,x3.w]);
   let rq=tq;let wi=kv/4;let jq=kv%4;let mut wd=[0u32;4];let mut dd=[0f32;4];
   unroll!([0,1,2,3],|c|{let row=(row0+rq*4+c).min(m-1);s!(wd,c,g!(wq,row*wpr+kc/16+wi));s!(dd,c,g!(wsc,row*bpr+kc/128));});
   unroll!([0,1,2,3],|j|{let sh=2*(jq*4+j);let f=|c:u32|(((g!(wd,c)>>sh)&3)as f32-1.)*g!(dd,c);s!(wf,(wi*16+jq*4+j)*$QUADS+rq,[f(0),f(1),f(2),f(3)]);});
   u+=$THREADS;
  }
  bar();let mut kb=0;
  while kb<32 {
   unroll!([$($kk),*],|kk|{
    unroll!([0,1],|rr|{if rr<$RX {let wa=g!(wf,(kb+kk)*$QUADS+ty+rr*($QUADS/$RX));let w=Vec4::new(wa[0],wa[1],wa[2],wa[3]);
     unroll!([0,1],|tt|{if tt<$RX {let xa=g!(xf,(kb+kk)*$QUADS+tx+tt*($QUADS/$RX));
      s!(acc,(rr*$RX+tt)*4,g!(acc,(rr*$RX+tt)*4)+w*xa[0]);
      s!(acc,(rr*$RX+tt)*4+1,g!(acc,(rr*$RX+tt)*4+1)+w*xa[1]);
      s!(acc,(rr*$RX+tt)*4+2,g!(acc,(rr*$RX+tt)*4+2)+w*xa[2]);
      s!(acc,(rr*$RX+tt)*4+3,g!(acc,(rr*$RX+tt)*4+3)+w*xa[3]);
     }});
    }});
   });kb+=$KUNROLL;
  }
  bar();kc+=32;
 }
 unroll!([0,1],|rr|{if rr<$RX {unroll!([0,1],|tt|{if tt<$RX {unroll!([0,1,2,3],|r|{
  let row=row0+(ty+rr*($QUADS/$RX))*4+r;
  if row<m {unroll!([0,1,2,3],|c|{let t=tok0+(tx+tt*($QUADS/$RX))*4+c;if t<nt {let yi=yo+t*ys+row;let v=get_c(g!(acc,(rr*$RX+tt)*4+c),r);s!(a,yi,if add!=0 {g!(a,yi)+v}else{v});}});}
 });}});}});
}
};}
resume_tile!(matmul_r64,"matmul_r64",64,16,8,2,8,[0,1,2,3,4,5,6,7]);
resume_tile!(matmul_r128,"matmul_r128",256,32,16,2,8,[0,1,2,3,4,5,6,7]);
resume_tile!(matmul_ru8,"matmul_ru8",256,16,16,1,8,[0,1,2,3,4,5,6,7]);
