//! Rust (rust-gpu) ports of the GLSL compute kernels in ../shaders. One SPIR-V module per entry
//! point (multimodule build). Bindings and push constants mirror shaders/common.glsl exactly:
//! 0 = arena f32 (also viewed as Vec4), 1 = params f32, 2 = ternary code words, 3 = scales.
#![no_std]
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]
use spirv_std::arch::IndexUnchecked;
use spirv_std::glam::{UVec3, UVec4, Vec4};
use spirv_std::num_traits::Float;
use spirv_std::spirv;

#[repr(C)]
pub struct PC { pub u: [u32; 32] }

const NONE: u32 = 0xffff_ffff;

macro_rules! g { ($s:expr, $i:expr) => { *unsafe { $s.index_unchecked(($i) as usize) } } }
macro_rules! s { ($s:expr, $i:expr, $v:expr) => {{ let __v = $v; let __i = ($i) as usize; *unsafe { $s.index_unchecked_mut(__i) } = __v; }} }
/// Compile-time loop unrolling (rust-gpu has no loop-unroll hint): `unroll!([0, 1, 2], |i| { ... })`
/// pastes the body once per literal, so per-thread arrays index with constants and stay in registers.
macro_rules! unroll { ([$($n:literal),*], |$v:ident| $body:block) => { $( { #[allow(unused)] let $v: u32 = $n; $body } )* }; }
pub mod prefill;
pub mod prefill_shapes;
pub mod prefill_occupancy;
pub mod prefill_half;
pub mod prefill_attention;
pub mod prefill_resume;
pub mod prefill_register;
pub mod prefill_columns;
pub mod prefill_columns_k16;
pub mod prefill_pipeline;
pub mod prefill_banked;
pub mod prefill_wave;
pub mod prefill_lut;
pub mod prefill_lut_vector;
pub mod prefill_trit;
pub mod prefill_trit_scalar;
pub mod prefill_columns_small;
pub mod prefill_attention_small;
pub mod prefill_dense;
pub mod prefill_columns_wide;
pub mod prefill_direct;
pub mod prefill_tail;
pub mod prefill_square;
pub mod prefill_swizzle;
pub mod prefill_gqa;
pub mod prefill_columns_group;
pub mod prefill_columns_short;
pub mod prefill_columns_k64;
#[inline(always)] fn bar() { spirv_std::arch::workgroup_memory_barrier_with_group_sync() }
/// Workgroup barrier that also makes this workgroup's storage-buffer writes visible to its own
/// threads (GLSL `memoryBarrierBuffer(); barrier();`), at workgroup scope.
#[inline(always)] fn bar_all() {
    use spirv_std::memory::{Scope, Semantics};
    spirv_std::arch::control_barrier::<{ Scope::Workgroup as u32 }, { Scope::Workgroup as u32 }, { Semantics::UNIFORM_MEMORY.bits() | Semantics::WORKGROUP_MEMORY.bits() | Semantics::ACQUIRE_RELEASE.bits() }>()
}
#[inline(always)] fn pu(pc: &PC, i: usize) -> u32 { g!(pc.u, i) }
#[inline(always)] fn pf(pc: &PC, i: usize) -> f32 { f32::from_bits(g!(pc.u, i)) }
#[inline(always)] fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
#[inline(always)] fn sigm(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
#[inline(always)] fn codes(word: u32, sh: u32) -> Vec4 {
    Vec4::new(((word >> sh) & 3) as f32, ((word >> (sh + 2)) & 3) as f32, ((word >> (sh + 4)) & 3) as f32, ((word >> (sh + 6)) & 3) as f32)
}
#[inline(always)] fn comp(v: UVec4, i: u32) -> u32 { if i == 0 { v.x } else if i == 1 { v.y } else if i == 2 { v.z } else { v.w } }

/// Greedy argmax over a logit row; ties -> lowest index.
#[spirv(compute(threads(256)))]
pub fn argmax(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(workgroup)] bv: &mut [f32; 256], #[spirv(workgroup)] bi: &mut [u32; 256]) {
    let (off, n, dst, l) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), lid.x);
    let mut best = -3.0e38f32; let mut idx = 0xffff_ffffu32;
    let mut i = l; while i < n { let v = g!(a, off + i); if v > best { best = v; idx = i; } i += 256; }
    s!(bv, l, best); s!(bi, l, idx); bar();
    unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| {
        let st = 1u32 << sh;
        if l < st { let o = g!(bv, l + st); let oi = g!(bi, l + st); if o > g!(bv, l) || (o == g!(bv, l) && oi < g!(bi, l)) { s!(bv, l, o); s!(bi, l, oi); } }
        bar();
    });
    if l == 0 { s!(a, dst, f32::from_bits(g!(bi, 0))); }
}

/// Causal GQA attention for one (head, token) + sigmoid output gate.
#[spirv(compute(threads(256)))]
pub fn attn(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(workgroup)] qv: &mut [f32; 256], #[spirv(workgroup)] red: &mut [f32; 256]) {
    let (qo, qts, kco, vco, kvd, pos0, hd) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6));
    let (nh, nkv, oo, ots, scr, ctx) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12)); let scale = pf(pc, 13);
    let (h, t, l) = (wg.x, wg.y, lid.x);
    let kh = h / (nh / nkv); let n = pos0 + t + 1;
    let qb = qo + t * qts + h * 2 * hd;
    if l < hd { s!(qv, l, g!(a, qb + l)); }
    bar();
    let sb = scr + (t * nh + h) * ctx;
    let mut mx = -1e30f32;
    let mut p = l;
    while p < n {
        let kb = kco + p * kvd + kh * hd; let mut sc = 0.0f32;
        let mut i = 0; while i < hd { sc += g!(qv, i) * g!(a, kb + i); i += 1; }
        sc *= scale; s!(a, sb + p, sc); mx = mx.max(sc); p += 256;
    }
    s!(red, l, mx); bar();
    unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l).max(g!(red, l + st))); } bar(); });
    mx = g!(red, 0); bar();
    let mut sum = 0.0f32;
    p = l; while p < n { let e = (g!(a, sb + p) - mx).exp(); s!(a, sb + p, e); sum += e; p += 256; }
    s!(red, l, sum); bar_all();
    unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
    let inv = 1.0 / g!(red, 0);
    if l < hd {
        let mut o = 0.0f32;
        p = 0; while p < n { o += g!(a, sb + p) * g!(a, vco + p * kvd + kh * hd + l); p += 1; }
        o *= inv; o *= sigm(g!(a, qb + hd + l));
        s!(a, oo + t * ots + h * hd + l, o);
    }
}

/// Causal depthwise conv (K<=4) + SiLU, carrying K-1 state. Thread per channel.
#[spirv(compute(threads(256)))]
pub fn conv(#[spirv(global_invocation_id)] gid: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32]) {
    let (cd, ntok, io, is_, oo, os, wo, so, k) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7), pu(pc, 8));
    let ch = gid.x;
    if ch >= cd { return; }
    let mut st = [0.0f32; 3];
    let mut i = 0; while i < 3 { s!(st, i, if i < k - 1 { g!(a, so + i * cd + ch) } else { 0.0 }); i += 1; }
    let mut t = 0;
    while t < ntok {
        let x = g!(a, io + t * is_ + ch);
        let mut sm = 0.0f32;
        i = 0; while i < k - 1 { sm += g!(st, i) * g!(prm, wo + ch * k + i); i += 1; }
        sm += x * g!(prm, wo + ch * k + k - 1);
        s!(a, oo + t * os + ch, silu(sm));
        i = 0; while i + 1 < k - 1 { s!(st, i, g!(st, i + 1)); i += 1; }
        s!(st, k - 2, x);
        t += 1;
    }
    i = 0; while i < k - 1 { s!(a, so + i * cd + ch, g!(st, i)); i += 1; }
}

/// mode 0: a+b, 1: silu(a)*b, 2: copy. wg.y = token.
#[spirv(compute(threads(256)))]
pub fn ewise(#[spirv(global_invocation_id)] gid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32]) {
    let (mode, n, ao, as_, bo, bs, oo, os) = (pu(pc, 0), pu(pc, 1), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7), pu(pc, 8));
    let (i, t) = (gid.x, wg.y);
    if i >= n { return; }
    let x = g!(a, ao + t * as_ + i);
    let r = if mode == 0 { x + g!(a, bo + t * bs + i) } else if mode == 1 { silu(x) * g!(a, bo + t * bs + i) } else { x };
    s!(a, oo + t * os + i, r);
}

/// Prism 1024-block Hadamard transform (forward with optional group permutation, or inverse).
#[spirv(compute(threads(256)))]
pub fn hadamard(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] v: &mut [f32; 1024]) {
    let (io, its, oo, ots, so, inv) = (pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6));
    let (hd, nk, rep) = (pu(pc, 7), pu(pc, 8), pu(pc, 9));
    let (blk, t, l) = (wg.x, wg.y, lid.x);
    unroll!([0, 1, 2, 3], |q| {
        let i = l + 256 * q;
        let gi = blk * 1024 + i; let mut src = gi;
        if hd != 0 { let i0 = gi % hd; let q = gi / hd; let r = q % rep; let k = q / rep; src = i0 + hd * (k + nk * r); }
        let mut x = g!(a, io + t * its + src);
        if inv == 0 && so != NONE { x *= g!(prm, so + gi); }
        s!(v, i, x);
    });
    bar();
    // 10 butterfly stages, unrolled so h is a constant power of two (p / h, p % h become shifts).
    unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], |sh| {
        let h = 1u32 << sh;
        unroll!([0, 1], |q| {
            let p = l + 256 * q;
            let i = ((p >> sh) << (sh + 1)) + (p & (h - 1)); let j = i + h;
            let x = g!(v, i); let y = g!(v, j); s!(v, i, x + y); s!(v, j, x - y);
        });
        bar();
    });
    unroll!([0, 1, 2, 3], |q| {
        let i = l + 256 * q;
        let gi = blk * 1024 + i;
        let mut x = g!(v, i) * 0.03125;
        if inv != 0 && so != NONE { x *= g!(prm, so + gi); }
        s!(a, oo + t * ots + gi, x);
    });
}

/// Small dense f32 matvec from Params (ssm_alpha / ssm_beta).
#[spirv(compute(threads(64)))]
pub fn matvec_f32(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 64]) {
    let (k, xo, xs, yo, ys, wo) = (pu(pc, 1), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (row, t, l) = (wg.x, wg.y, lid.x);
    let mut sm = 0.0f32;
    let mut i = l; while i < k { sm += g!(prm, wo + row * k + i) * g!(a, xo + t * xs + i); i += 64; }
    s!(red, l, sm); bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
    if l == 0 { s!(a, yo + t * ys + row, g!(red, 0)); }
}

/// Row norms. mode 0: rms*w, 1: rms*w*silu(z), 2: l2.
#[spirv(compute(threads(256)))]
pub fn rmsnorm(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 256]) {
    let (n, io, irs, its, oo, ors, ots) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6));
    let (wo, mode, zo, zrs, zts) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11)); let eps = pf(pc, 12);
    let (r, t, l) = (wg.x, wg.y, lid.x);
    let ib = io + t * its + r * irs; let ob = oo + t * ots + r * ors;
    let mut sm = 0.0f32;
    let mut i = l; while i < n { let v = g!(a, ib + i); sm += v * v; i += 256; }
    s!(red, l, sm); bar();
    unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
    let tot = g!(red, 0);
    let sc = if mode == 2 { 1.0 / tot.sqrt().max(eps) } else { 1.0 / (tot / n as f32 + eps).sqrt() };
    i = l;
    while i < n {
        let mut v = g!(a, ib + i) * sc;
        if wo != NONE { v *= g!(prm, wo + i); }
        if mode == 1 { v *= silu(g!(a, zo + t * zts + r * zrs + i)); }
        s!(a, ob + i, v); i += 256;
    }
}

/// neox-style (I)MRoPE, in place.
#[spirv(compute(threads(64)))]
pub fn rope(#[spirv(global_invocation_id)] gid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32]) {
    let (off, hs, ts, nh, pos0, nrot) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 5), pu(pc, 6));
    let tsc = pf(pc, 7); let (s0, s1, s2, s3) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11));
    let half = nrot / 2; let (gi, t) = (gid.x, wg.y);
    if gi >= nh * half { return; }
    let h = gi / half; let i = gi % half;
    let sect = s0 + s1 + s2 + s3;
    let sector = if sect > 0 { i % sect } else { 0 };
    let use_e = sect > 0 && !((sector % 3 == 1 && sector < 3 * s1) || (sector % 3 == 2 && sector < 3 * s2) || (sector % 3 == 0 && sector < 3 * s0));
    let p = if use_e { 0.0 } else { (pos0 + t) as f32 };
    let th = p * tsc.powf(i as f32);
    let (c, sn) = (th.cos(), th.sin());
    let b = off + t * ts + h * hs;
    let x0 = g!(a, b + i); let x1 = g!(a, b + i + half);
    s!(a, b + i, x0 * c - x1 * sn); s!(a, b + i + half, x0 * sn + x1 * c);
}

/// Gated DeltaNet recurrence (decode), one workgroup per value head, thread j owns column j.
#[spirv(compute(threads(128)))]
pub fn gdn(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [f32; 128], #[spirv(workgroup)] ks: &mut [f32; 128]) {
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, j) = (wg.x, lid.x); let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd;
    let mut t = 0;
    while t < ntok {
        let cb = co + t * cs;
        s!(qs, j, g!(a, cb + kh * sd + j));
        s!(ks, j, g!(a, cb + qk + kh * sd + j));
        let vj = g!(a, cb + 2 * qk + h * sd + j);
        bar();
        let beta = sigm(g!(a, bo + t * bas + h));
        let x = g!(a, alo + t * bas + h) + g!(prm, dto + h);
        let gg = g!(prm, ao + h) * (if x > 20.0 { x } else { (1.0 + x.exp()).ln() });
        let eg = gg.exp();
        let mut sm = 0.0f32;
        let mut i = 0; while i < sd { unroll!([0, 1, 2, 3], |u| { let ii = i + u; let m = g!(a, sb + ii * sd + j) * eg; s!(a, sb + ii * sd + j, m); sm += m * g!(ks, ii); }); i += 4; }
        let delta = (vj - sm) * beta;
        let mut acc = 0.0f32;
        i = 0; while i < sd { unroll!([0, 1, 2, 3], |u| { let ii = i + u; let m = g!(a, sb + ii * sd + j) + g!(ks, ii) * delta; s!(a, sb + ii * sd + j, m); acc += m * g!(qs, ii); }); i += 4; }
        s!(a, oo + t * os + h * sd + j, acc * scale);
        bar();
        t += 1;
    }
}

/// Prefill Gated DeltaNet (S=128): state in registers for the whole chunk.
#[spirv(compute(threads(256)))]
pub fn gdn_p(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [[f32; 4]; 32], #[spirv(workgroup)] ks: &mut [[f32; 4]; 32],
    #[spirv(workgroup)] r1: &mut [[f32; 4]; 64], #[spirv(workgroup)] r2: &mut [[f32; 4]; 64]) {
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, l) = (wg.x, lid.x); let rq = l & 3; let j = wg.y * 64 + (l >> 2); let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd + rq * 32 * sd + j;
    // Thread state: 32 rows of column j as 8 independent Vec4 SSA values (rust-gpu otherwise carries a
    // [f32; 32] as one array-typed OpPhi across the token loop).
    let ld = |i: u32| Vec4::new(g!(a, sb + (4 * i) * sd), g!(a, sb + (4 * i + 1) * sd), g!(a, sb + (4 * i + 2) * sd), g!(a, sb + (4 * i + 3) * sd));
    let (mut s0, mut s1, mut s2, mut s3, mut s4, mut s5, mut s6, mut s7) = (ld(0), ld(1), ld(2), ld(3), ld(4), ld(5), ld(6), ld(7));
    macro_rules! all8 { (|$s:ident, $i:ident| $b:block) => {
        { let $i = 0u32; let $s = &mut s0; $b } { let $i = 1u32; let $s = &mut s1; $b } { let $i = 2u32; let $s = &mut s2; $b } { let $i = 3u32; let $s = &mut s3; $b }
        { let $i = 4u32; let $s = &mut s4; $b } { let $i = 5u32; let $s = &mut s5; $b } { let $i = 6u32; let $s = &mut s6; $b } { let $i = 7u32; let $s = &mut s7; $b } } }
    let v4 = |x: [f32; 4]| Vec4::new(x[0], x[1], x[2], x[3]);
    // do-while shape: rust-gpu lowers `while` to a header test plus a second exit test, which gives the
    // loop-carried state an extra phi edge; ACO then copies all 32 state registers every token.
    let mut t = 0;
    if ntok > 0 { loop {
        let cb = co + t * cs;
        if l < 128 { unsafe { *qs.index_unchecked_mut((l >> 2) as usize).index_unchecked_mut((l & 3) as usize) = g!(a, cb + kh * sd + l);
                              *ks.index_unchecked_mut((l >> 2) as usize).index_unchecked_mut((l & 3) as usize) = g!(a, cb + qk + kh * sd + l); } }
        let vj = g!(a, cb + 2 * qk + h * sd + j);
        let beta = sigm(g!(a, bo + t * bas + h));
        let x = g!(a, alo + t * bas + h) + g!(prm, dto + h);
        let eg = (g!(prm, ao + h) * (if x > 20.0 { x } else { (1.0 + x.exp()).ln() })).exp();
        bar();
        let mut sm = Vec4::ZERO;
        all8!(|s, i| { *s = *s * eg; sm += *s * v4(g!(ks, rq * 8 + i)); });
        unsafe { *r1.index_unchecked_mut((l >> 2) as usize).index_unchecked_mut(rq as usize) = sm.dot(Vec4::ONE); }
        bar();
        let delta = (vj - v4(g!(r1, l >> 2)).dot(Vec4::ONE)) * beta;
        let mut o = Vec4::ZERO;
        all8!(|s, i| { let kv = v4(g!(ks, rq * 8 + i)); *s = *s + kv * delta; o += *s * v4(g!(qs, rq * 8 + i)); });
        unsafe { *r2.index_unchecked_mut((l >> 2) as usize).index_unchecked_mut(rq as usize) = o.dot(Vec4::ONE); }
        bar();
        if rq == 0 { s!(a, oo + t * os + h * sd + j, v4(g!(r2, l >> 2)).dot(Vec4::ONE) * scale); }
        bar();
        t += 1;
        if t >= ntok { break; }
    } }
    all8!(|s, i| { let v = *s; s!(a, sb + (4 * i) * sd, v.x); s!(a, sb + (4 * i + 1) * sd, v.y); s!(a, sb + (4 * i + 2) * sd, v.z); s!(a, sb + (4 * i + 3) * sd, v.w); });
}

/// Decode ternary matvec: one 64-lane wave per R=8 rows (R is a compile-time const here; the
/// GLSL version takes it as a specialization constant, which rust-gpu cannot use as an array size).
const MV_R: u32 = 8;
#[spirv(compute(threads(64)))]
pub fn matvec_t2_1(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq4: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 512]) {
    let (m, k, ntok, mut xo, xs, mut yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let lane = lid.x;
    let mut r0 = (wg.y * nwg.x + wg.x) * MV_R;
    // Short-prompt mode (ntok > 1): grid x = token, y = row group (rows shared by the tokens stay cached).
    if ntok > 1 { let t = wg.x; r0 = wg.y * MV_R; xo += t * xs; yo += t * ys; }
    let qpr = k >> 6; let bpr = k >> 7; let xb = xo / 4;
    let mut acc = [0.0f32; MV_R as usize];
    let mut qi = lane;
    while qi < qpr {
        let mut x = [Vec4::ZERO; 16];
        unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], |i| { s!(x, i, g!(a4, xb + qi * 16 + i)); });
        let mut xs = [0.0f32; 4];
        unroll!([0,1,2,3], |wd| { s!(xs, wd, (g!(x, wd * 4) + g!(x, wd * 4 + 1) + g!(x, wd * 4 + 2) + g!(x, wd * 4 + 3)).dot(Vec4::ONE)); });
        unroll!([0,1,2,3,4,5,6,7], |r| {
            let row = (r0 + r).min(m - 1);
            let w4 = g!(wq4, row * qpr + qi);
            let d = g!(wsc, row * bpr + (qi >> 1));
            let mut sm = 0.0f32;
            unroll!([0,1,2,3], |wd| {
                let word = comp(w4, wd);
                let mut tt = 0.0f32;
                unroll!([0,1,2,3], |v| { tt += codes(word, v * 8).dot(g!(x, wd * 4 + v)); });
                sm += tt - g!(xs, wd);
            });
            s!(acc, r, g!(acc, r) + sm * d);
        });
        qi += 64;
    }
    unroll!([0,1,2,3,4,5,6,7], |r| { s!(red, r * 64 + lane, g!(acc, r)); });
    bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh;
        if lane < st { unroll!([0,1,2,3,4,5,6,7], |r| { s!(red, r * 64 + lane, g!(red, r * 64 + lane) + g!(red, r * 64 + lane + st)); }); }
        bar(); });
    if lane < MV_R && r0 + lane < m { let yi = yo + r0 + lane; let v = g!(red, lane * 64); s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v }); }
}

/// Batched ternary matvec (small token counts): 4 rows x 8 tokens per wave, grid z = token tile.
#[spirv(compute(threads(64)))]
pub fn matvec_t2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq4: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 2048]) {
    const R: u32 = 4; const T: u32 = 8;
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let lane = lid.x;
    let r0 = (wg.y * nwg.x + wg.x) * R;
    let qpr = k >> 6; let bpr = k >> 7;
    let t0 = wg.z * T; if t0 >= ntok { return; }
    let nt = (ntok - t0).min(T);
    let mut acc = [0.0f32; 32];
    let mut qi = lane;
    while qi < qpr {
        let mut w = [UVec4::ZERO; 4]; let mut d = [0.0f32; 4];
        unroll!([0,1,2,3], |r| { let row = (r0 + r).min(m - 1); s!(w, r, g!(wq4, row * qpr + qi)); s!(d, r, g!(wsc, row * bpr + (qi >> 1))); });
        // Decode each row's 16 codes of word wd once, then reuse them for every token in the tile.
        unroll!([0,1,2,3], |wd| {
            let mut c = [Vec4::ZERO; 16];
            unroll!([0,1,2,3], |r| {
                let word = comp(g!(w, r), wd);
                s!(c, r * 4, codes(word, 0)); s!(c, r * 4 + 1, codes(word, 8)); s!(c, r * 4 + 2, codes(word, 16)); s!(c, r * 4 + 3, codes(word, 24));
            });
            unroll!([0,1,2,3,4,5,6,7], |tt| { if tt < nt {
                let b4 = (xo + (t0 + tt) * xs) / 4 + qi * 16 + wd * 4;
                let x0 = g!(a4, b4); let x1 = g!(a4, b4 + 1); let x2 = g!(a4, b4 + 2); let x3 = g!(a4, b4 + 3);
                let sx = (x0 + x1 + x2 + x3).dot(Vec4::ONE);
                unroll!([0,1,2,3], |r| {
                    let v = g!(c, r * 4).dot(x0) + g!(c, r * 4 + 1).dot(x1) + g!(c, r * 4 + 2).dot(x2) + g!(c, r * 4 + 3).dot(x3) - sx;
                    s!(acc, r * T + tt, g!(acc, r * T + tt) + v * g!(d, r));
                });
            } });
        });
        qi += 64;
    }
    unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], |i| { s!(red, i * 64 + lane, g!(acc, i)); });
    bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh;
        if lane < st { unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], |i| { s!(red, i * 64 + lane, g!(red, i * 64 + lane) + g!(red, i * 64 + lane + st)); }); }
        bar(); });
    if lane < R * T {
        let r = lane / T; let tt = lane % T;
        if r0 + r < m && tt < nt { let yi = yo + (t0 + tt) * ys + r0 + r; let v = g!(red, lane * 64); s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v }); }
    }
}

#[inline(always)] fn get_c(v: Vec4, c: u32) -> f32 { if c == 0 { v.x } else if c == 1 { v.y } else if c == 2 { v.z } else { v.w } }

/// Tiled batched ternary matmul: 64 rows x 64 tokens per 256-thread workgroup, K-chunks of 32.
/// Shared tiles are [[f32; 4]; 512] (Ws[k][row/4][row%4], Xs[k][tok/4][tok%4]): scalar element stores like the
/// GLSL kernel, and whole 16-byte array loads in the inner loop (so the driver emits ds_read_b128).
#[spirv(compute(threads(256)))]
pub fn matmul_t2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] wf: &mut [[f32; 4]; 512], #[spirv(workgroup)] xf: &mut [[f32; 4]; 512]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let tx = l & 15; let ty = l >> 4;
    let row0 = wg.x * 64; let tok0 = wg.y * 64;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut acc = [Vec4::ZERO; 4];
    let mut kc = 0;
    while kc < k {
        if l < 128 {
            // X: thread = (token quad tq, k-vec kv); load 4 tokens x 4 k, store 4 whole [f32;4] (4 tokens) per k.
            let tq = l & 15; let kv = l >> 4;
            let mut xv = [Vec4::ZERO; 4];
            unroll!([0,1,2,3], |c| { let t = tok0 + tq * 4 + c; s!(xv, c, if t < ntok { g!(a4, (xo + t * xs + kc) / 4 + kv) } else { Vec4::ZERO }); });
            let (x0, x1, x2, x3) = (g!(xv, 0), g!(xv, 1), g!(xv, 2), g!(xv, 3));
            s!(xf, (kv * 4) * 16 + tq, [x0.x, x1.x, x2.x, x3.x]); s!(xf, (kv * 4 + 1) * 16 + tq, [x0.y, x1.y, x2.y, x3.y]);
            s!(xf, (kv * 4 + 2) * 16 + tq, [x0.z, x1.z, x2.z, x3.z]); s!(xf, (kv * 4 + 3) * 16 + tq, [x0.w, x1.w, x2.w, x3.w]);
        } else {
            // W: thread = (row quad rq, word wi, j-group jg of 4); decode 4 rows, store whole [f32;4] (4 rows) per k.
            let u = l - 128; let rq = u & 15; let wi = (u >> 4) & 1; let jg = u >> 5;
            let mut wd = [0u32; 4]; let mut dd = [0.0f32; 4];
            unroll!([0,1,2,3], |c| { let row = (row0 + rq * 4 + c).min(m - 1); s!(wd, c, g!(wq, row * wpr + (kc >> 4) + wi)); s!(dd, c, g!(wsc, row * bpr + (kc >> 7))); });
            unroll!([0,1,2,3], |jj| {
                let jx = jg * 4 + jj;
                let f = |c: u32| (((g!(wd, c) >> (2 * jx)) & 3) as f32 - 1.0) * g!(dd, c);
                s!(wf, (wi * 16 + jx) * 16 + rq, [f(0), f(1), f(2), f(3)]);
            });
        }
        bar();
        unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], |kk| {
            // acc[c] = 4 rows of token c: per k, one weight Vec4 (rows) times 4 scalar activations.
            let wa = g!(wf, kk * 16 + ty); let xa = g!(xf, kk * 16 + tx); // whole 16-byte rows
            let w = Vec4::new(wa[0], wa[1], wa[2], wa[3]);
            s!(acc, 0, g!(acc, 0) + w * xa[0]); s!(acc, 1, g!(acc, 1) + w * xa[1]);
            s!(acc, 2, g!(acc, 2) + w * xa[2]); s!(acc, 3, g!(acc, 3) + w * xa[3]);
        });
        bar();
        kc += 32;
    }
    unroll!([0,1,2,3], |r| {
        let row = row0 + ty * 4 + r;
        if row < m {
            unroll!([0,1,2,3], |c| {
                let t = tok0 + tx * 4 + c;
                if t < ntok { let yi = yo + t * ys + row; let v = get_c(g!(acc, c), r); s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v }); }
            });
        }
    });
}

/// Small-tile batched ternary matmul for short prompts: 128 rows x 8 tokens per 256-thread
/// workgroup (thread = 4 rows x 1 token); weights decoded once per 8 tokens into shared memory.
#[spirv(compute(threads(256)))]
pub fn matmul_t2s(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] wf: &mut [Vec4; 1024], #[spirv(workgroup)] xf: &mut [f32; 256]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let tx = l & 7; let ty = l >> 3;
    let row0 = wg.x * 128; let tok0 = wg.y * 8;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut acc = Vec4::ZERO;
    let mut kc = 0;
    while kc < k {
        {
            let rg = l >> 3; let wi = (l >> 2) & 1; let jq = l & 3;
            let mut wd = [0u32; 4]; let mut dd = [0.0f32; 4];
            unroll!([0,1,2,3], |c| { let row = (row0 + rg * 4 + c).min(m - 1); s!(wd, c, g!(wq, row * wpr + (kc >> 4) + wi)); s!(dd, c, g!(wsc, row * bpr + (kc >> 7))); });
            unroll!([0,1,2,3], |jj| {
                let j = jq * 4 + jj;
                let f = |c: u32| (((g!(wd, c) >> (2 * j)) & 3) as f32 - 1.0) * g!(dd, c);
                s!(wf, (wi * 16 + j) * 32 + rg, Vec4::new(f(0), f(1), f(2), f(3)));
            });
        }
        let tl = l >> 5; let kk = l & 31; let t = tok0 + tl;
        s!(xf, kk * 8 + tl, if t < ntok { g!(a, xo + t * xs + kc + kk) } else { 0.0 });
        bar();
        unroll!([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], |kk| { acc += g!(wf, kk * 32 + ty) * g!(xf, kk * 8 + tx); });
        bar();
        kc += 32;
    }
    let t = tok0 + tx;
    if t < ntok {
        unroll!([0,1,2,3], |r| {
            let row = row0 + ty * 4 + r;
            if row < m { let yi = yo + t * ys + row; let v = get_c(acc, r); s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v }); }
        });
    }
}

/// Int8-group matvec for the grafted MTP head (Q8_0 / Q6_K repacked to `Q8G16`: 16 int8 per UVec4,
/// one f32 scale per 16). One 64-lane wave per 4 rows; grid x = token, y = row group.
#[inline(always)] fn i8x4(w: u32) -> Vec4 { Vec4::new(((w << 24) as i32 >> 24) as f32, ((w << 16) as i32 >> 24) as f32, ((w << 8) as i32 >> 24) as f32, (w as i32 >> 24) as f32) }
#[spirv(compute(threads(64)))]
pub fn matvec_q8(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq4: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 256]) {
    let (m, k, _ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let lane = lid.x; let t = wg.x; let r0 = wg.y * 4;
    let xb = (xo + t * xs) / 4; let yo = yo + t * ys;
    let gpr = k >> 4;
    let mut acc = [0.0f32; 4];
    let mut g = lane;
    while g < gpr {
        let x0 = g!(a4, xb + g * 4); let x1 = g!(a4, xb + g * 4 + 1); let x2 = g!(a4, xb + g * 4 + 2); let x3 = g!(a4, xb + g * 4 + 3);
        unroll!([0,1,2,3], |r| {
            let row = (r0 + r).min(m - 1); let w = g!(wq4, row * gpr + g);
            let d = i8x4(w.x).dot(x0) + i8x4(w.y).dot(x1) + i8x4(w.z).dot(x2) + i8x4(w.w).dot(x3);
            s!(acc, r, g!(acc, r) + d * g!(wsc, row * gpr + g));
        });
        g += 64;
    }
    unroll!([0,1,2,3], |r| { s!(red, r * 64 + lane, g!(acc, r)); });
    bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh;
        if lane < st { unroll!([0,1,2,3], |r| { s!(red, r * 64 + lane, g!(red, r * 64 + lane) + g!(red, r * 64 + lane + st)); }); }
        bar(); });
    if lane < 4 && r0 + lane < m { let yi = yo + r0 + lane; let v = g!(red, lane * 64); s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v }); }
}

/// r012 decode matvec family: lanes take W consecutive u32 code words (16*W weights) per step, rows per
/// wave = R. Every row length here (K = 5120 / 6144 / 17408 -> 320 / 384 / 1088 words) splits evenly over
/// 64 lanes for W = 1; x loads span 4*W KiB per step (L1-resident) and are reused by R rows. Tokens
/// (ntok <= 4, MTP verify) loop inside the wave.
macro_rules! mv_c { ($name:ident, $R:literal, $W:literal, [$($r:literal),*], [$($w:literal),*]) => {
#[spirv(compute(threads(64)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 2048]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let lane = lid.x; let rn: u32 = $R;
    let r0 = (wg.y * nwg.x + wg.x) * rn;
    let wpr = k >> 4; let bpr = k >> 7; let steps = wpr / $W;
    let mut t = 0u32;
    while t < ntok {
        let xb = (xo + t * xs) / 4;
        let mut acc = [0.0f32; $R];
        let mut st = lane;
        while st < steps {
            unroll!([$($w),*], |ww| {
                let wi = st * $W + ww;
                let x0 = g!(a4, xb + wi * 4); let x1 = g!(a4, xb + wi * 4 + 1); let x2 = g!(a4, xb + wi * 4 + 2); let x3 = g!(a4, xb + wi * 4 + 3);
                let xsum = (x0 + x1 + x2 + x3).dot(Vec4::ONE);
                unroll!([$($r),*], |rr| {
                    let row = (r0 + rr).min(m - 1);
                    let word = g!(wq, row * wpr + wi);
                    let d = g!(wsc, row * bpr + (wi >> 3));
                    let tt = codes(word, 0).dot(x0) + codes(word, 8).dot(x1) + codes(word, 16).dot(x2) + codes(word, 24).dot(x3);
                    s!(acc, rr, g!(acc, rr) + (tt - xsum) * d);
                });
            });
            st += 64;
        }
        unroll!([$($r),*], |rr| { s!(red, rr * 64 + lane, g!(acc, rr)); });
        bar();
        // reduce R x 64 partials: each lane sums (R*64/64) = R consecutive partials of one row slice
        let per = rn; let row = (lane * per) >> 6; let base = lane * per;
        let mut s = 0.0f32; let mut i = 0u32; while i < per { s += g!(red, base + i); i += 1; }
        bar();
        s!(red, lane, s);
        bar();
        let lpr = 64 / rn; // lanes per row
        if lane < rn && r0 + lane < m {
            let mut v = 0.0f32; let mut j = 0u32; while j < lpr { v += g!(red, lane * lpr + j); j += 1; }
            let yi = yo + t * ys + r0 + lane; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
        }
        let _ = row;
        bar();
        t += 1;
    }
}
}; }
mv_c!(matvec_c8, 8, 1, [0,1,2,3,4,5,6,7], [0]);


/// r012 multi-token decode matvec: 8 rows per wave, T tokens share every weight/scale load (MTP verify,
/// short prompts). Grid: (row groups, y, token tiles of T); tokens past ntok are skipped.
macro_rules! mv_t { ($name:ident, $T:literal, [$($t:literal),*]) => {
#[spirv(compute(threads(64)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 2048]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let lane = lid.x;
    let r0 = (wg.y * nwg.x + wg.x) * 8;
    let t0 = wg.z * $T; let nt = (ntok - t0).min($T);
    let wpr = k >> 4; let bpr = k >> 7;
    let mut acc = [0.0f32; 8 * $T];
    let mut wi = lane;
    while wi < wpr {
        let mut xv = [Vec4::ZERO; 4 * $T]; let mut xsum = [0.0f32; $T];
        unroll!([$($t),*], |tt| {
            let tk = (t0 + tt).min(ntok - 1); let xb = (xo + tk * xs) / 4 + wi * 4;
            unroll!([0,1,2,3], |v| { s!(xv, tt * 4 + v, g!(a4, xb + v)); });
            s!(xsum, tt, (g!(xv, tt * 4) + g!(xv, tt * 4 + 1) + g!(xv, tt * 4 + 2) + g!(xv, tt * 4 + 3)).dot(Vec4::ONE));
        });
        unroll!([0,1,2,3,4,5,6,7], |rr| {
            let row = (r0 + rr).min(m - 1);
            let word = g!(wq, row * wpr + wi);
            let d = g!(wsc, row * bpr + (wi >> 3));
            let c0 = codes(word, 0); let c1 = codes(word, 8); let c2 = codes(word, 16); let c3 = codes(word, 24);
            unroll!([$($t),*], |tt| {
                let v = c0.dot(g!(xv, tt * 4)) + c1.dot(g!(xv, tt * 4 + 1)) + c2.dot(g!(xv, tt * 4 + 2)) + c3.dot(g!(xv, tt * 4 + 3));
                s!(acc, rr * $T + tt, g!(acc, rr * $T + tt) + (v - g!(xsum, tt)) * d);
            });
        });
        wi += 64;
    }
    // same reduction as matvec_c8, once per token
    unroll!([$($t),*], |tt| {
        unroll!([0,1,2,3,4,5,6,7], |rr| { s!(red, rr * 64 + lane, g!(acc, rr * $T + tt)); });
        bar();
        let mut s = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |i| { s += g!(red, lane * 8 + i); });
        bar();
        s!(red, lane, s);
        bar();
        if lane < 8 && r0 + lane < m && tt < nt {
            let mut v = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |j| { v += g!(red, lane * 8 + j); });
            let yi = yo + (t0 + tt) * ys + r0 + lane; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
        }
        bar();
    });
}
}; }
mv_t!(matvec_c8x2, 2, [0, 1]);
mv_t!(matvec_c8x4, 4, [0, 1, 2, 3]);

/// r012 split-K ("flash-decoding") causal GQA attention, head_dim 256. One wave per (q head h, token t,
/// position chunk s of C positions). Lane = position for scores (vec4 K loads, q broadcast from LDS),
/// lane = 4 dims for the weighted V sum (coalesced 1 KiB V rows). Writes unnormalised partials
/// (max, sum, o[256]) to scratch; `attn_comb` merges them and applies the sigmoid output gate.
/// pc: 0 qo, 1 qts, 2 kco, 3 vco, 4 kvd, 5 pos0, 6 hd(=256), 7 nh, 8 nkv, 9 scr, 10 C, 11 smax, 12 scale
#[spirv(compute(threads(64)))]
pub fn attn_s(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(workgroup)] qv: &mut [Vec4; 64], #[spirv(workgroup)] sc: &mut [f32; 1024], #[spirv(workgroup)] red: &mut [f32; 64]) {
    let (qo, qts, kco, vco, kvd, pos0) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5));
    let (nh, nkv, scr, cn, smax) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11)); let scale = pf(pc, 12);
    let (h, t, s, l) = (wg.x, wg.y, wg.z, lid.x);
    let n = pos0 + t + 1; let p0 = s * cn; if p0 >= n { return; }
    let p1 = (p0 + cn).min(n);
    let kh = h / (nh / nkv);
    let qb = (qo + t * qts + h * 512) / 4; // q row: [q(256) | gate(256)] per head
    s!(qv, l, g!(a4, qb + l));
    bar();
    let mut mx = -1e30f32;
    let mut p = p0 + l;
    while p < p1 {
        let kb = (kco + p * kvd + kh * 256) / 4;
        let mut d = Vec4::ZERO;
        let mut i = 0u32; while i < 64 { d += g!(a4, kb + i) * g!(qv, i); i += 1; }
        let v = d.dot(Vec4::ONE) * scale;
        s!(sc, p - p0, v); mx = mx.max(v);
        p += 64;
    }
    s!(red, l, mx); bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l).max(g!(red, l + st))); } bar(); });
    let m = g!(red, 0); bar();
    let mut sum = 0.0f32;
    p = p0 + l; while p < p1 { let e = (g!(sc, p - p0) - m).exp(); s!(sc, p - p0, e); sum += e; p += 64; }
    s!(red, l, sum); bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
    let tot = g!(red, 0);
    let mut o = Vec4::ZERO;
    p = p0; while p < p1 { o += g!(a4, (vco + p * kvd + kh * 256) / 4 + l) * g!(sc, p - p0); p += 1; }
    let pb = scr + ((t * nh + h) * smax + s) * 260; // 260: m, sum, pad, pad, o[256] (vec4 aligned)
    if l == 0 { s!(a, pb, m); s!(a, pb + 1, tot); }
    let ob = pb + 4 + l * 4; s!(a, ob, o.x); s!(a, ob + 1, o.y); s!(a, ob + 2, o.z); s!(a, ob + 3, o.w);
}

/// Merge `attn_s` partials for (head h, token t) and apply the sigmoid gate.
/// pc: 0 qo, 1 qts, 2 pos0, 3 nh, 4 scr, 5 C, 6 smax, 7 oo, 8 ots
#[spirv(compute(threads(64)))]
pub fn attn_comb(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32]) {
    let (qo, qts, pos0, nh, scr, cn, smax, oo, ots) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7), pu(pc, 8));
    let (h, t, l) = (wg.x, wg.y, lid.x);
    let n = pos0 + t + 1; let ns = (n + cn - 1) / cn;
    let base = scr + (t * nh + h) * smax * 260;
    let mut m = -1e30f32; let mut s = 0u32; while s < ns { m = m.max(g!(a, base + s * 260)); s += 1; }
    let mut den = 0.0f32; let mut o = [0.0f32; 4];
    s = 0; while s < ns {
        let pb = base + s * 260; let w = (g!(a, pb) - m).exp(); den += g!(a, pb + 1) * w;
        unroll!([0, 1, 2, 3], |j| { s!(o, j, g!(o, j) + g!(a, pb + 4 + l * 4 + j) * w); });
        s += 1;
    }
    unroll!([0, 1, 2, 3], |j| {
        let d = l * 4 + j;
        s!(a, oo + t * ots + h * 256 + d, g!(o, j) / den * sigm(g!(a, qo + t * qts + h * 512 + 256 + d)));
    });
}

/// r012 f16 KV: pack n f32 at `src` into n/2 u32 (f16x2, packHalf2x16) at `dst`. pc: 0 src, 1 dst, 2 n (even)
#[spirv(compute(threads(64)))]
pub fn kv_pack(#[spirv(global_invocation_id)] gid: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &[f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32]) {
    let (src, dst, n) = (pu(pc, 0), pu(pc, 1), pu(pc, 2));
    let i = gid.x + gid.y * 65535 * 64; if i * 2 >= n { return; }
    s!(au, dst + i, spirv_std::float::vec2_to_f16x2(spirv_std::glam::Vec2::new(g!(a, src + 2 * i), g!(a, src + 2 * i + 1))));
}

/// `attn_s` over an f16 (f16x2-packed) K/V cache. kco/vco are u32-element offsets, kvd counted in halves.
#[spirv(compute(threads(64)))]
pub fn attn_s_h(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] u4: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(workgroup)] qv: &mut [Vec4; 64], #[spirv(workgroup)] sc: &mut [f32; 1024], #[spirv(workgroup)] red: &mut [f32; 64]) {
    use spirv_std::float::f16x2_to_vec2 as h2;
    let (qo, qts, kco, vco, kvd, pos0) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5));
    let (nh, nkv, scr, cn, smax) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11)); let scale = pf(pc, 12);
    let (h, t, s, l) = (wg.x, wg.y, wg.z, lid.x);
    let n = pos0 + t + 1; let p0 = s * cn; if p0 >= n { return; }
    let p1 = (p0 + cn).min(n);
    let kh = h / (nh / nkv);
    let qb = (qo + t * qts + h * 512) / 4;
    s!(qv, l, g!(a4, qb + l));
    bar();
    let mut mx = -1e30f32;
    let mut p = p0 + l;
    while p < p1 {
        let kb = (kco + (p * kvd + kh * 256) / 2) / 4; // 32 UVec4 = 256 halves
        let mut d = Vec4::ZERO;
        let mut i = 0u32; while i < 32 {
            let w = g!(u4, kb + i); let (x0, x1, x2, x3) = (h2(w.x), h2(w.y), h2(w.z), h2(w.w));
            d += Vec4::new(x0.x, x0.y, x1.x, x1.y) * g!(qv, 2 * i) + Vec4::new(x2.x, x2.y, x3.x, x3.y) * g!(qv, 2 * i + 1);
            i += 1; }
        let v = d.dot(Vec4::ONE) * scale;
        s!(sc, p - p0, v); mx = mx.max(v);
        p += 64;
    }
    s!(red, l, mx); bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l).max(g!(red, l + st))); } bar(); });
    let m = g!(red, 0); bar();
    let mut sum = 0.0f32;
    p = p0 + l; while p < p1 { let e = (g!(sc, p - p0) - m).exp(); s!(sc, p - p0, e); sum += e; p += 64; }
    s!(red, l, sum); bar();
    unroll!([5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
    let tot = g!(red, 0);
    let mut o = Vec4::ZERO;
    p = p0; while p < p1 {
        let vb = vco + (p * kvd + kh * 256) / 2 + 2 * l; let (y0, y1) = (h2(g!(au, vb)), h2(g!(au, vb + 1)));
        o += Vec4::new(y0.x, y0.y, y1.x, y1.y) * g!(sc, p - p0); p += 1; }
    let pb = scr + ((t * nh + h) * smax + s) * 260;
    if l == 0 { s!(a, pb, m); s!(a, pb + 1, tot); }
    let ob = pb + 4 + l * 4; s!(a, ob, o.x); s!(a, ob + 1, o.y); s!(a, ob + 2, o.z); s!(a, ob + 3, o.w);
}
