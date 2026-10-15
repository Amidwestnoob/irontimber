//! Independent prefill-only tile experiments. Decode and MTP kernels unchanged.
//! Same quantization, per-k ascending accumulation, boundary/accumulate handling.
//! Explicit FMA reduces rounding to one step (tolerance derived/tested by harness).
use super::*;
#[inline(always)] fn fma4(w: Vec4, x: f32, a: Vec4) -> Vec4 {
    Vec4::new(w.x.mul_add(x,a.x),w.y.mul_add(x,a.y),w.z.mul_add(x,a.z),w.w.mul_add(x,a.w))
}
macro_rules! pf_tile { ($name:ident,$entry:literal,$ROWS:literal,$RQUADS:literal,$RT:literal,[$($r:literal),*]) => {
#[spirv(compute(threads(256), entry_point_name = $entry))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] wf: &mut [[f32;4]; 32*$RQUADS],
    #[spirv(workgroup)] xf: &mut [[f32;4]; 512]) {
    let (m,k,ntok,xo,xs,yo,ys,accum)=(pu(pc,0),pu(pc,1),pu(pc,2),pu(pc,3),pu(pc,4),pu(pc,5),pu(pc,6),pu(pc,7));
    let l=lid.x;let tx=l&15;let ty=l>>4;let row0=wg.x*$ROWS;let tok0=wg.y*64;
    let wpr=k>>4;let bpr=k>>7;let mut acc=[Vec4::ZERO;4*$RT];let mut kc=0;
    while kc<k {
        // All threads load X; redundant same-value stores avoided: exactly 128 writers.
        if l<128 {
            let tq=l&15;let kv=l>>4;let mut xv=[Vec4::ZERO;4];
            unroll!([0,1,2,3],|c| {let t=tok0+tq*4+c;s!(xv,c,if t<ntok{g!(a4,(xo+t*xs+kc)/4+kv)}else{Vec4::ZERO});});
            let(x0,x1,x2,x3)=(g!(xv,0),g!(xv,1),g!(xv,2),g!(xv,3));
            s!(xf,(kv*4)*16+tq,[x0.x,x1.x,x2.x,x3.x]);s!(xf,(kv*4+1)*16+tq,[x0.y,x1.y,x2.y,x3.y]);
            s!(xf,(kv*4+2)*16+tq,[x0.z,x1.z,x2.z,x3.z]);s!(xf,(kv*4+3)*16+tq,[x0.w,x1.w,x2.w,x3.w]);
        }
        // General cooperative decode: thread covers four consecutive k, four rows.
        let mut u=l;
        while u<($RQUADS*8) {
            let rq=u%$RQUADS;let jg=u/$RQUADS;let wi=jg/4;let jq=jg%4;
            let mut wd=[0u32;4];let mut dd=[0f32;4];
            unroll!([0,1,2,3],|c|{let row=(row0+rq*4+c).min(m-1);s!(wd,c,g!(wq,row*wpr+(kc>>4)+wi));s!(dd,c,g!(wsc,row*bpr+(kc>>7)));});
            unroll!([0,1,2,3],|jj|{let j=jq*4+jj;let f=|c:u32|(((g!(wd,c)>>(2*j))&3)as f32-1.)*g!(dd,c);s!(wf,(wi*16+j)*$RQUADS+rq,[f(0),f(1),f(2),f(3)]);});
            u+=256;
        }
        bar();
        unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31],|kk|{
            let xa=g!(xf,kk*16+tx);
            unroll!([$($r),*],|rq|{let wa=g!(wf,kk*$RQUADS+ty+rq*16);let w=Vec4::new(wa[0],wa[1],wa[2],wa[3]);
                s!(acc,rq*4+0,fma4(w,xa[0],g!(acc,rq*4+0)));s!(acc,rq*4+1,fma4(w,xa[1],g!(acc,rq*4+1)));
                s!(acc,rq*4+2,fma4(w,xa[2],g!(acc,rq*4+2)));s!(acc,rq*4+3,fma4(w,xa[3],g!(acc,rq*4+3)));
            });
        });
        bar();kc+=32;
    }
    unroll!([$($r),*],|rq|{unroll!([0,1,2,3],|r|{let row=row0+(ty+rq*16)*4+r;
        if row<m {unroll!([0,1,2,3],|c|{let t=tok0+tx*4+c;if t<ntok {let yi=yo+t*ys+row;let v=get_c(g!(acc,rq*4+c),r);s!(a,yi,if accum!=0 {g!(a,yi)+v}else{v});}});}
    });});
}
};}
pf_tile!(matmul_pf64,"matmul_pf64",64,16,1,[0]);
pf_tile!(matmul_pf128,"matmul_pf128",128,32,2,[0,1]);
