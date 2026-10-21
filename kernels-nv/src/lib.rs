//! NVIDIA/Ampere-only rust-gpu kernels (built separately by tools/spvbuild-nv with CooperativeMatrixKHR + Float16).
//! Bindings and push constants mirror ../kernels (0 = arena f32, 2 = 2-bit code words, 3 = per-128 scales).
#![no_std]
#![feature(asm_experimental_arch)]
use core::arch::asm;
use spirv_std::arch::IndexUnchecked;
use spirv_std::float::vec2_to_f16x2;
use spirv_std::glam::{UVec3, UVec4, Vec2, Vec4};
use spirv_std::spirv;
#[allow(unused_imports)] use spirv_std::num_traits::Float;

#[repr(C)]
pub struct PC { pub u: [u32; 32] }
macro_rules! g { ($s:expr, $i:expr) => { *unsafe { $s.index_unchecked(($i) as usize) } } }
macro_rules! s { ($s:expr, $i:expr, $v:expr) => {{ let __v = $v; let __i = ($i) as usize; *unsafe { $s.index_unchecked_mut(__i) } = __v; }} }
macro_rules! unroll { ([$($n:literal),*], |$v:ident| $body:block) => { $( { #[allow(unused)] let $v: u32 = $n; $body } )* }; }
#[inline(always)] fn bar() { spirv_std::arch::workgroup_memory_barrier_with_group_sync() }
#[inline(always)] fn pu(pc: &PC, i: usize) -> u32 { g!(pc.u, i) }
/// Exact f16 -> f32 by integer bit manipulation. The NVIDIA compiler folds UnpackHalf2x16(PackHalf2x16(x)) to x,
/// which silently zeroed the lo half of the split activation; integer decoding cannot be folded that way.
#[inline(always)] fn h2f(h: u32) -> f32 {
    let sgn = (h & 0x8000) << 16; let e = (h >> 10) & 0x1f; let mt = h & 0x3ff;
    if e == 0 { let v = mt as f32 * (1.0 / 16777216.0); if sgn != 0 { -v } else { v } }
    else { f32::from_bits(sgn | ((e + 112) << 23) | (mt << 13)) }
}
#[inline(always)] fn unpack_exact(p: u32) -> Vec2 { Vec2::new(h2f(p & 0xffff), h2f(p >> 16)) }

/// PQ2_0 prefill GEMM on cooperative matrices: 64 rows x 64 tokens per 128-thread workgroup (4 warps x 32x32),
/// K chunks of 64. Weights decode exactly to f16 ((code-1)*d, d is f16 in the file); activations are split x = hi + lo
/// (two f16) so the f16 x f16 -> f32 MMA reproduces ~f32 products. C accumulates in shared f32.
#[spirv(compute(threads(128)))]
pub fn matmul_cm(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2048], #[spirv(workgroup)] xh: &mut [u32; 2048], #[spirv(workgroup)] xl: &mut [u32; 2048],
    #[spirv(workgroup)] cs: &mut [f32; 4096]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 1) * 32; let wc = (w >> 1) * 32;
    let row0 = wg.x * 64; let tok0 = wg.y * 64;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut i = 0u32; while i < 32 { s!(cs, l + i * 128, 0.0); i += 1; }
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1], |p| {
            let idx = l + p * 128; let r = idx >> 2; let wi = idx & 3;
            let row = (row0 + r).min(m - 1);
            let word = g!(wq, row * wpr + (kc >> 4) + wi); let d = g!(wsc, row * bpr + (kc >> 7));
            unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
                let v0 = (((word >> (4 * q)) & 3) as f32 - 1.0) * d; let v1 = (((word >> (4 * q + 2)) & 3) as f32 - 1.0) * d;
                s!(ws, r * 32 + wi * 8 + q, vec2_to_f16x2(Vec2::new(v0, v1)));
            });
        });
        {
            let t = l >> 1; let half = l & 1; let tt = tok0 + t;
            unroll!([0, 1, 2, 3, 4, 5, 6, 7], |v| {
                let x = if tt < ntok { g!(a4, (xo + tt * xs + kc) / 4 + half * 8 + v) } else { Vec4::ZERO };
                let i0 = t * 32 + half * 16 + v * 2;
                let h0 = vec2_to_f16x2(Vec2::new(x.x, x.y)); let h1 = vec2_to_f16x2(Vec2::new(x.z, x.w));
                s!(xh, i0, h0); s!(xh, i0 + 1, h1);
                let b0 = unpack_exact(h0); let b1 = unpack_exact(h1);
                s!(xl, i0, vec2_to_f16x2(Vec2::new(x.x - b0.x, x.y - b0.y))); s!(xl, i0 + 1, vec2_to_f16x2(Vec2::new(x.z - b1.x, x.w - b1.y)));
            });
        }
        bar();
        macro_rules! ap { ($i:literal, $kk:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 32 + $kk * 8) as usize) } } }
        macro_rules! hp { ($j:literal, $kk:literal) => { unsafe { xh.index_unchecked(((wc + $j * 16) * 32 + $kk * 8) as usize) } } }
        #[allow(unused_macros)] macro_rules! lp { ($j:literal, $kk:literal) => { unsafe { xl.index_unchecked(((wc + $j * 16) * 32 + $kk * 8) as usize) } } }
        macro_rules! cp { ($i:literal, $j:literal) => { unsafe { cs.index_unchecked_mut(((wc + $j * 16) * 64 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
        "%f16 = OpTypeFloat 16",
        "%f32 = OpTypeFloat 32",
        "%u32 = OpTypeInt 32 0",
        "%c0 = OpConstant %u32 0",
        "%c1 = OpConstant %u32 1",
        "%c2 = OpConstant %u32 2",
        "%c3 = OpConstant %u32 3",
        "%c16 = OpConstant %u32 16",
        "%s32 = OpConstant %u32 32",
        "%s64 = OpConstant %u32 64",
        "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
        "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
        "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
        "%C00_0 = OpCooperativeMatrixLoadKHR %tC {c00} %c1 %s64",
        "%C01_0 = OpCooperativeMatrixLoadKHR %tC {c01} %c1 %s64",
        "%C10_0 = OpCooperativeMatrixLoadKHR %tC {c10} %c1 %s64",
        "%C11_0 = OpCooperativeMatrixLoadKHR %tC {c11} %c1 %s64",
        "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %s32",
        "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %s32",
        "%H0_0 = OpCooperativeMatrixLoadKHR %tB {h00} %c1 %s32",
        "%L0_0 = OpCooperativeMatrixLoadKHR %tB {l00} %c1 %s32",
        "%H1_0 = OpCooperativeMatrixLoadKHR %tB {h10} %c1 %s32",
        "%L1_0 = OpCooperativeMatrixLoadKHR %tB {l10} %c1 %s32",
        "%D00_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L0_0 %C00_0",
        "%C00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H0_0 %D00_0",
        "%D01_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L1_0 %C01_0",
        "%C01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H1_0 %D01_0",
        "%D10_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L0_0 %C10_0",
        "%C10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H0_0 %D10_0",
        "%D11_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L1_0 %C11_0",
        "%C11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H1_0 %D11_0",
        "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %s32",
        "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %s32",
        "%H0_1 = OpCooperativeMatrixLoadKHR %tB {h01} %c1 %s32",
        "%L0_1 = OpCooperativeMatrixLoadKHR %tB {l01} %c1 %s32",
        "%H1_1 = OpCooperativeMatrixLoadKHR %tB {h11} %c1 %s32",
        "%L1_1 = OpCooperativeMatrixLoadKHR %tB {l11} %c1 %s32",
        "%D00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L0_1 %C00_1",
        "%C00_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H0_1 %D00_1",
        "%D01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L1_1 %C01_1",
        "%C01_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H1_1 %D01_1",
        "%D10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L0_1 %C10_1",
        "%C10_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H0_1 %D10_1",
        "%D11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L1_1 %C11_1",
        "%C11_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H1_1 %D11_1",
        "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %s32",
        "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %s32",
        "%H0_2 = OpCooperativeMatrixLoadKHR %tB {h02} %c1 %s32",
        "%L0_2 = OpCooperativeMatrixLoadKHR %tB {l02} %c1 %s32",
        "%H1_2 = OpCooperativeMatrixLoadKHR %tB {h12} %c1 %s32",
        "%L1_2 = OpCooperativeMatrixLoadKHR %tB {l12} %c1 %s32",
        "%D00_2 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %L0_2 %C00_2",
        "%C00_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H0_2 %D00_2",
        "%D01_2 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %L1_2 %C01_2",
        "%C01_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H1_2 %D01_2",
        "%D10_2 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %L0_2 %C10_2",
        "%C10_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H0_2 %D10_2",
        "%D11_2 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %L1_2 %C11_2",
        "%C11_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H1_2 %D11_2",
        "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %s32",
        "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %s32",
        "%H0_3 = OpCooperativeMatrixLoadKHR %tB {h03} %c1 %s32",
        "%L0_3 = OpCooperativeMatrixLoadKHR %tB {l03} %c1 %s32",
        "%H1_3 = OpCooperativeMatrixLoadKHR %tB {h13} %c1 %s32",
        "%L1_3 = OpCooperativeMatrixLoadKHR %tB {l13} %c1 %s32",
        "%D00_3 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %L0_3 %C00_3",
        "%C00_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H0_3 %D00_3",
        "%D01_3 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %L1_3 %C01_3",
        "%C01_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H1_3 %D01_3",
        "%D10_3 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %L0_3 %C10_3",
        "%C10_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H0_3 %D10_3",
        "%D11_3 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %L1_3 %C11_3",
        "%C11_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H1_3 %D11_3",
        "OpCooperativeMatrixStoreKHR {c00} %C00_4 %c1 %s64",
        "OpCooperativeMatrixStoreKHR {c01} %C01_4 %c1 %s64",
        "OpCooperativeMatrixStoreKHR {c10} %C10_4 %c1 %s64",
        "OpCooperativeMatrixStoreKHR {c11} %C11_4 %c1 %s64",
        c00 = in(reg) cp!(0,0),
        c01 = in(reg) cp!(0,1),
        c10 = in(reg) cp!(1,0),
        c11 = in(reg) cp!(1,1),
        a00 = in(reg) ap!(0,0),
        a10 = in(reg) ap!(1,0),
        h00 = in(reg) hp!(0,0),
        l00 = in(reg) lp!(0,0),
        h10 = in(reg) hp!(1,0),
        l10 = in(reg) lp!(1,0),
        a01 = in(reg) ap!(0,1),
        a11 = in(reg) ap!(1,1),
        h01 = in(reg) hp!(0,1),
        l01 = in(reg) lp!(0,1),
        h11 = in(reg) hp!(1,1),
        l11 = in(reg) lp!(1,1),
        a02 = in(reg) ap!(0,2),
        a12 = in(reg) ap!(1,2),
        h02 = in(reg) hp!(0,2),
        l02 = in(reg) lp!(0,2),
        h12 = in(reg) hp!(1,2),
        l12 = in(reg) lp!(1,2),
        a03 = in(reg) ap!(0,3),
        a13 = in(reg) ap!(1,3),
        h03 = in(reg) hp!(0,3),
        l03 = in(reg) lp!(0,3),
        h13 = in(reg) hp!(1,3),
        l13 = in(reg) lp!(1,3)
    );
        }
        bar();
        kc += 64;
    }
    let mut i = 0u32;
    while i < 32 {
        let e = l + i * 128; let t = tok0 + (e >> 6); let row = row0 + (e & 63);
        if t < ntok && row < m { let yi = yo + t * ys + row; let v = g!(cs, e); s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v }); }
        i += 1;
    }
}

/// Same tiling, single f16 activation (hi only): faster, lower precision (A/B option).
#[spirv(compute(threads(128)))]
pub fn matmul_cm1(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2048], #[spirv(workgroup)] xh: &mut [u32; 2048], #[spirv(workgroup)] xl: &mut [u32; 2048],
    #[spirv(workgroup)] cs: &mut [f32; 4096]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 1) * 32; let wc = (w >> 1) * 32;
    let row0 = wg.x * 64; let tok0 = wg.y * 64;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut i = 0u32; while i < 32 { s!(cs, l + i * 128, 0.0); i += 1; }
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1], |p| {
            let idx = l + p * 128; let r = idx >> 2; let wi = idx & 3;
            let row = (row0 + r).min(m - 1);
            let word = g!(wq, row * wpr + (kc >> 4) + wi); let d = g!(wsc, row * bpr + (kc >> 7));
            unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
                let v0 = (((word >> (4 * q)) & 3) as f32 - 1.0) * d; let v1 = (((word >> (4 * q + 2)) & 3) as f32 - 1.0) * d;
                s!(ws, r * 32 + wi * 8 + q, vec2_to_f16x2(Vec2::new(v0, v1)));
            });
        });
        {
            let t = l >> 1; let half = l & 1; let tt = tok0 + t;
            unroll!([0, 1, 2, 3, 4, 5, 6, 7], |v| {
                let x = if tt < ntok { g!(a4, (xo + tt * xs + kc) / 4 + half * 8 + v) } else { Vec4::ZERO };
                let i0 = t * 32 + half * 16 + v * 2;
                let h0 = vec2_to_f16x2(Vec2::new(x.x, x.y)); let h1 = vec2_to_f16x2(Vec2::new(x.z, x.w));
                s!(xh, i0, h0); s!(xh, i0 + 1, h1);
                
            });
        }
        bar();
        macro_rules! ap { ($i:literal, $kk:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 32 + $kk * 8) as usize) } } }
        macro_rules! hp { ($j:literal, $kk:literal) => { unsafe { xh.index_unchecked(((wc + $j * 16) * 32 + $kk * 8) as usize) } } }
        #[allow(unused_macros)] macro_rules! lp { ($j:literal, $kk:literal) => { unsafe { xl.index_unchecked(((wc + $j * 16) * 32 + $kk * 8) as usize) } } }
        macro_rules! cp { ($i:literal, $j:literal) => { unsafe { cs.index_unchecked_mut(((wc + $j * 16) * 64 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
        "%f16 = OpTypeFloat 16",
        "%f32 = OpTypeFloat 32",
        "%u32 = OpTypeInt 32 0",
        "%c0 = OpConstant %u32 0",
        "%c1 = OpConstant %u32 1",
        "%c2 = OpConstant %u32 2",
        "%c3 = OpConstant %u32 3",
        "%c16 = OpConstant %u32 16",
        "%s32 = OpConstant %u32 32",
        "%s64 = OpConstant %u32 64",
        "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
        "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
        "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
        "%C00_0 = OpCooperativeMatrixLoadKHR %tC {c00} %c1 %s64",
        "%C01_0 = OpCooperativeMatrixLoadKHR %tC {c01} %c1 %s64",
        "%C10_0 = OpCooperativeMatrixLoadKHR %tC {c10} %c1 %s64",
        "%C11_0 = OpCooperativeMatrixLoadKHR %tC {c11} %c1 %s64",
        "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %s32",
        "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %s32",
        "%H0_0 = OpCooperativeMatrixLoadKHR %tB {h00} %c1 %s32",
        "%H1_0 = OpCooperativeMatrixLoadKHR %tB {h10} %c1 %s32",
        "%C00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H0_0 %C00_0",
        "%C01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H1_0 %C01_0",
        "%C10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H0_0 %C10_0",
        "%C11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H1_0 %C11_0",
        "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %s32",
        "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %s32",
        "%H0_1 = OpCooperativeMatrixLoadKHR %tB {h01} %c1 %s32",
        "%H1_1 = OpCooperativeMatrixLoadKHR %tB {h11} %c1 %s32",
        "%C00_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H0_1 %C00_1",
        "%C01_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H1_1 %C01_1",
        "%C10_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H0_1 %C10_1",
        "%C11_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H1_1 %C11_1",
        "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %s32",
        "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %s32",
        "%H0_2 = OpCooperativeMatrixLoadKHR %tB {h02} %c1 %s32",
        "%H1_2 = OpCooperativeMatrixLoadKHR %tB {h12} %c1 %s32",
        "%C00_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H0_2 %C00_2",
        "%C01_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H1_2 %C01_2",
        "%C10_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H0_2 %C10_2",
        "%C11_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H1_2 %C11_2",
        "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %s32",
        "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %s32",
        "%H0_3 = OpCooperativeMatrixLoadKHR %tB {h03} %c1 %s32",
        "%H1_3 = OpCooperativeMatrixLoadKHR %tB {h13} %c1 %s32",
        "%C00_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H0_3 %C00_3",
        "%C01_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H1_3 %C01_3",
        "%C10_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H0_3 %C10_3",
        "%C11_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H1_3 %C11_3",
        "OpCooperativeMatrixStoreKHR {c00} %C00_4 %c1 %s64",
        "OpCooperativeMatrixStoreKHR {c01} %C01_4 %c1 %s64",
        "OpCooperativeMatrixStoreKHR {c10} %C10_4 %c1 %s64",
        "OpCooperativeMatrixStoreKHR {c11} %C11_4 %c1 %s64",
        c00 = in(reg) cp!(0,0),
        c01 = in(reg) cp!(0,1),
        c10 = in(reg) cp!(1,0),
        c11 = in(reg) cp!(1,1),
        a00 = in(reg) ap!(0,0),
        a10 = in(reg) ap!(1,0),
        h00 = in(reg) hp!(0,0),
        h10 = in(reg) hp!(1,0),
        a01 = in(reg) ap!(0,1),
        a11 = in(reg) ap!(1,1),
        h01 = in(reg) hp!(0,1),
        h11 = in(reg) hp!(1,1),
        a02 = in(reg) ap!(0,2),
        a12 = in(reg) ap!(1,2),
        h02 = in(reg) hp!(0,2),
        h12 = in(reg) hp!(1,2),
        a03 = in(reg) ap!(0,3),
        a13 = in(reg) ap!(1,3),
        h03 = in(reg) hp!(0,3),
        h13 = in(reg) hp!(1,3)
    );
        }
        bar();
        kc += 64;
    }
    let mut i = 0u32;
    while i < 32 {
        let e = l + i * 128; let t = tok0 + (e >> 6); let row = row0 + (e & 63);
        if t < ntok && row < m { let yi = yo + t * ys + row; let v = g!(cs, e); s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v }); }
        i += 1;
    }
}

/// v2 PQ2_0 prefill GEMM: 128 rows x 128 tokens per 256-thread workgroup (8 warps, each 32 rows x 64 tokens = 2x4
/// 16x16 f32 accumulators held in Function-storage cooperative-matrix variables across the K loop), K chunks of 32.
/// Exact f16 weights, hi+lo f16 activations. Stores whole 16x16 tiles straight to Y (column-major, stride ys):
/// requires m % 128 == 0 and Y to have room for ntok rounded up to 128 tokens (the engine's maxt-sized buffers).
/// accum != 0 seeds the accumulators from Y.
#[spirv(compute(threads(256)))]
pub fn matmul_cm2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2048], #[spirv(workgroup)] xh: &mut [u32; 2048], #[spirv(workgroup)] xl: &mut [u32; 2048]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut kc = 0u32;
    while kc < k {
        {
            let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
            let word = g!(wq, row * wpr + (kc >> 4) + wi); let d = g!(wsc, row * bpr + (kc >> 7));
            unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
                let v0 = (((word >> (4 * q)) & 3) as f32 - 1.0) * d; let v1 = (((word >> (4 * q + 2)) & 3) as f32 - 1.0) * d;
                s!(ws, r * 16 + wi * 8 + q, vec2_to_f16x2(Vec2::new(v0, v1)));
            });
        }
        {
            let t = l >> 1; let half = l & 1; let tt = tok0 + t;
            unroll!([0, 1, 2, 3], |v| {
                let x = if tt < ntok { g!(a4, (xo + tt * xs + kc) / 4 + half * 4 + v) } else { Vec4::ZERO };
                let i0 = t * 16 + half * 8 + v * 2;
                let h0 = vec2_to_f16x2(Vec2::new(x.x, x.y)); let h1 = vec2_to_f16x2(Vec2::new(x.z, x.w));
                s!(xh, i0, h0); s!(xh, i0 + 1, h1);
                let b0 = unpack_exact(h0); let b1 = unpack_exact(h1);
                s!(xl, i0, vec2_to_f16x2(Vec2::new(x.x - b0.x, x.y - b0.y))); s!(xl, i0 + 1, vec2_to_f16x2(Vec2::new(x.z - b1.x, x.w - b1.y)));
            });
        }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 32 >= k { 1 } else { 0 };
        macro_rules! ap { ($i:literal, $kk:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 16 + $kk * 8) as usize) } } }
        macro_rules! hp { ($j:literal, $kk:literal) => { unsafe { xh.index_unchecked(((wc + $j * 16) * 16 + $kk * 8) as usize) } } }
        macro_rules! lp { ($j:literal, $kk:literal) => { unsafe { xl.index_unchecked(((wc + $j * 16) * 16 + $kk * 8) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f16 = OpTypeFloat 16",
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%C00_0 = OpLoad %tC %v00",
            "%C01_0 = OpLoad %tC %v01",
            "%C02_0 = OpLoad %tC %v02",
            "%C03_0 = OpLoad %tC %v03",
            "%C10_0 = OpLoad %tC %v10",
            "%C11_0 = OpLoad %tC %v11",
            "%C12_0 = OpLoad %tC %v12",
            "%C13_0 = OpLoad %tC %v13",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c16",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c16",
            "%H0_0 = OpCooperativeMatrixLoadKHR %tB {h00} %c1 %c16",
            "%L0_0 = OpCooperativeMatrixLoadKHR %tB {l00} %c1 %c16",
            "%H1_0 = OpCooperativeMatrixLoadKHR %tB {h10} %c1 %c16",
            "%L1_0 = OpCooperativeMatrixLoadKHR %tB {l10} %c1 %c16",
            "%H2_0 = OpCooperativeMatrixLoadKHR %tB {h20} %c1 %c16",
            "%L2_0 = OpCooperativeMatrixLoadKHR %tB {l20} %c1 %c16",
            "%H3_0 = OpCooperativeMatrixLoadKHR %tB {h30} %c1 %c16",
            "%L3_0 = OpCooperativeMatrixLoadKHR %tB {l30} %c1 %c16",
            "%D00_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L0_0 %C00_0",
            "%C00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H0_0 %D00_0",
            "%D01_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L1_0 %C01_0",
            "%C01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H1_0 %D01_0",
            "%D02_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L2_0 %C02_0",
            "%C02_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H2_0 %D02_0",
            "%D03_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L3_0 %C03_0",
            "%C03_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H3_0 %D03_0",
            "%D10_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L0_0 %C10_0",
            "%C10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H0_0 %D10_0",
            "%D11_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L1_0 %C11_0",
            "%C11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H1_0 %D11_0",
            "%D12_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L2_0 %C12_0",
            "%C12_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H2_0 %D12_0",
            "%D13_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L3_0 %C13_0",
            "%C13_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H3_0 %D13_0",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c16",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c16",
            "%H0_1 = OpCooperativeMatrixLoadKHR %tB {h01} %c1 %c16",
            "%L0_1 = OpCooperativeMatrixLoadKHR %tB {l01} %c1 %c16",
            "%H1_1 = OpCooperativeMatrixLoadKHR %tB {h11} %c1 %c16",
            "%L1_1 = OpCooperativeMatrixLoadKHR %tB {l11} %c1 %c16",
            "%H2_1 = OpCooperativeMatrixLoadKHR %tB {h21} %c1 %c16",
            "%L2_1 = OpCooperativeMatrixLoadKHR %tB {l21} %c1 %c16",
            "%H3_1 = OpCooperativeMatrixLoadKHR %tB {h31} %c1 %c16",
            "%L3_1 = OpCooperativeMatrixLoadKHR %tB {l31} %c1 %c16",
            "%D00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L0_1 %C00_1",
            "%C00_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H0_1 %D00_1",
            "%D01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L1_1 %C01_1",
            "%C01_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H1_1 %D01_1",
            "%D02_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L2_1 %C02_1",
            "%C02_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H2_1 %D02_1",
            "%D03_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L3_1 %C03_1",
            "%C03_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H3_1 %D03_1",
            "%D10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L0_1 %C10_1",
            "%C10_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H0_1 %D10_1",
            "%D11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L1_1 %C11_1",
            "%C11_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H1_1 %D11_1",
            "%D12_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L2_1 %C12_1",
            "%C12_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H2_1 %D12_1",
            "%D13_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L3_1 %C13_1",
            "%C13_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H3_1 %D13_1",
            "OpStore %v00 %C00_2",
            "OpStore %v01 %C01_2",
            "OpStore %v02 %C02_2",
            "OpStore %v03 %C03_2",
            "OpStore %v10 %C10_2",
            "OpStore %v11 %C11_2",
            "OpStore %v12 %C12_2",
            "OpStore %v13 %C13_2",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %C00_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y01} %C01_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y02} %C02_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y03} %C03_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %C10_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %C11_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %C12_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %C13_2 %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            h00 = in(reg) hp!(0,0),
            l00 = in(reg) lp!(0,0),
            h10 = in(reg) hp!(1,0),
            l10 = in(reg) lp!(1,0),
            h20 = in(reg) hp!(2,0),
            l20 = in(reg) lp!(2,0),
            h30 = in(reg) hp!(3,0),
            l30 = in(reg) lp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            h01 = in(reg) hp!(0,1),
            l01 = in(reg) lp!(0,1),
            h11 = in(reg) hp!(1,1),
            l11 = in(reg) lp!(1,1),
            h21 = in(reg) hp!(2,1),
            l21 = in(reg) lp!(2,1),
            h31 = in(reg) hp!(3,1),
            l31 = in(reg) lp!(3,1)
        );
        }
        bar();
        kc += 32;
    }
}

/// int8 tensor-core PQ2_0 prefill GEMM (Prism-like numerics): weights as exact int8 codes {-1,0,1,2} with the f32 group
/// scale, activations quantized in-kernel to int8 per (token, 128-wide K group) with d = max|x|/127. Per K group:
/// int32 = sum over 128 of codes*q (4 x 16x16x32 s8 MMAs), then acc += float(int32) * dw[row] * dx[tok].
/// 256 threads, 128 rows x 128 tokens, 8 warps x (32 rows x 64 tokens). Same tiling/store contract as matmul_cm2,
/// plus k % 128 == 0.
#[spirv(compute(threads(256)))]
pub fn matmul_cm8(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4096], #[spirv(workgroup)] xq: &mut [u32; 4096],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 128], #[spirv(workgroup)] red: &mut [f32; 256]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut kc = 0u32;
    while kc < k {
        {
            let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
            unroll!([0, 1, 2, 3], |q| {
                let word = g!(wq, row * wpr + (kc >> 4) + wi * 4 + q);
                unroll!([0, 1, 2, 3], |b| {
                    let c = (word >> (8 * b)) & 0xff;
                    let sp = (c & 3) | (((c >> 2) & 3) << 8) | (((c >> 4) & 3) << 16) | (((c >> 6) & 3) << 24);
                    s!(ws, r * 32 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
                });
            });
            if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        }
        let t = l >> 1; let half = l & 1; let tt = tok0 + t; let live = tt < ntok;
        let base = (xo + tt * xs + kc) / 4 + half * 16;
        let mut mx = 0.0f32;
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| {
            if live { let x = g!(a4, base + v); mx = mx.max(x.x.abs()).max(x.y.abs()).max(x.z.abs()).max(x.w.abs()); }
        });
        s!(red, l, mx);
        bar();
        let mx = mx.max(g!(red, l ^ 1));
        let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
        if half == 0 { s!(dxs, t, mx / 127.0); }
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| {
            let x = if live { g!(a4, base + v) } else { Vec4::ZERO };
            let q0 = ((x.x * inv).round() as i32) as u32 & 0xff; let q1 = ((x.y * inv).round() as i32) as u32 & 0xff;
            let q2 = ((x.z * inv).round() as i32) as u32 & 0xff; let q3 = ((x.w * inv).round() as i32) as u32 & 0xff;
            s!(xq, t * 32 + half * 16 + v, q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
        });
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        macro_rules! ap { ($i:literal, $ks:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 32 + $ks * 8) as usize) } } }
        macro_rules! bp { ($j:literal, $ks:literal) => { unsafe { xq.index_unchecked(((wc + $j * 16) * 32 + $ks * 8) as usize) } } }
        macro_rules! dwp { ($i:literal) => { unsafe { dws.index_unchecked((wr + $i * 16) as usize) } } }
        macro_rules! dxp { ($j:literal) => { unsafe { dxs.index_unchecked((wc + $j * 16) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%c32 = OpConstant %u32 32",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c3 %c16 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c3 %c32 %c16 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c3 %c16 %c16 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c32",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c32",
            "%B0_0 = OpCooperativeMatrixLoadKHR %tB {b00} %c1 %c32",
            "%B1_0 = OpCooperativeMatrixLoadKHR %tB {b10} %c1 %c32",
            "%B2_0 = OpCooperativeMatrixLoadKHR %tB {b20} %c1 %c32",
            "%B3_0 = OpCooperativeMatrixLoadKHR %tB {b30} %c1 %c32",
            "%I00_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c32",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c32",
            "%B0_1 = OpCooperativeMatrixLoadKHR %tB {b01} %c1 %c32",
            "%B1_1 = OpCooperativeMatrixLoadKHR %tB {b11} %c1 %c32",
            "%B2_1 = OpCooperativeMatrixLoadKHR %tB {b21} %c1 %c32",
            "%B3_1 = OpCooperativeMatrixLoadKHR %tB {b31} %c1 %c32",
            "%I00_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B0_1 %I00_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B1_1 %I01_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B2_1 %I02_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B3_1 %I03_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B0_1 %I10_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B1_1 %I11_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B2_1 %I12_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B3_1 %I13_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %c32",
            "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %c32",
            "%B0_2 = OpCooperativeMatrixLoadKHR %tB {b02} %c1 %c32",
            "%B1_2 = OpCooperativeMatrixLoadKHR %tB {b12} %c1 %c32",
            "%B2_2 = OpCooperativeMatrixLoadKHR %tB {b22} %c1 %c32",
            "%B3_2 = OpCooperativeMatrixLoadKHR %tB {b32} %c1 %c32",
            "%I00_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B0_2 %I00_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B1_2 %I01_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B2_2 %I02_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B3_2 %I03_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B0_2 %I10_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B1_2 %I11_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B2_2 %I12_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B3_2 %I13_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %c32",
            "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %c32",
            "%B0_3 = OpCooperativeMatrixLoadKHR %tB {b03} %c1 %c32",
            "%B1_3 = OpCooperativeMatrixLoadKHR %tB {b13} %c1 %c32",
            "%B2_3 = OpCooperativeMatrixLoadKHR %tB {b23} %c1 %c32",
            "%B3_3 = OpCooperativeMatrixLoadKHR %tB {b33} %c1 %c32",
            "%I00_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B0_3 %I00_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B1_3 %I01_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B2_3 %I02_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B3_3 %I03_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B0_3 %I10_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B1_3 %I11_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B2_3 %I12_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B3_3 %I13_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW0 = OpCooperativeMatrixLoadKHR %tC {dw0} %c1 %c0",
            "%SW1 = OpCooperativeMatrixLoadKHR %tC {dw1} %c1 %c0",
            "%SX0 = OpCooperativeMatrixLoadKHR %tC {dx0} %c0 %c0",
            "%SX1 = OpCooperativeMatrixLoadKHR %tC {dx1} %c0 %c0",
            "%SX2 = OpCooperativeMatrixLoadKHR %tC {dx2} %c0 %c0",
            "%SX3 = OpCooperativeMatrixLoadKHR %tC {dx3} %c0 %c0",
            "%F00 = OpConvertSToF %tC %I00_4",
            "%P00 = OpFMul %tC %SW0 %SX0",
            "%G00 = OpFMul %tC %F00 %P00",
            "%O00 = OpLoad %tC %v00",
            "%N00 = OpFAdd %tC %O00 %G00",
            "OpStore %v00 %N00",
            "%F01 = OpConvertSToF %tC %I01_4",
            "%P01 = OpFMul %tC %SW0 %SX1",
            "%G01 = OpFMul %tC %F01 %P01",
            "%O01 = OpLoad %tC %v01",
            "%N01 = OpFAdd %tC %O01 %G01",
            "OpStore %v01 %N01",
            "%F02 = OpConvertSToF %tC %I02_4",
            "%P02 = OpFMul %tC %SW0 %SX2",
            "%G02 = OpFMul %tC %F02 %P02",
            "%O02 = OpLoad %tC %v02",
            "%N02 = OpFAdd %tC %O02 %G02",
            "OpStore %v02 %N02",
            "%F03 = OpConvertSToF %tC %I03_4",
            "%P03 = OpFMul %tC %SW0 %SX3",
            "%G03 = OpFMul %tC %F03 %P03",
            "%O03 = OpLoad %tC %v03",
            "%N03 = OpFAdd %tC %O03 %G03",
            "OpStore %v03 %N03",
            "%F10 = OpConvertSToF %tC %I10_4",
            "%P10 = OpFMul %tC %SW1 %SX0",
            "%G10 = OpFMul %tC %F10 %P10",
            "%O10 = OpLoad %tC %v10",
            "%N10 = OpFAdd %tC %O10 %G10",
            "OpStore %v10 %N10",
            "%F11 = OpConvertSToF %tC %I11_4",
            "%P11 = OpFMul %tC %SW1 %SX1",
            "%G11 = OpFMul %tC %F11 %P11",
            "%O11 = OpLoad %tC %v11",
            "%N11 = OpFAdd %tC %O11 %G11",
            "OpStore %v11 %N11",
            "%F12 = OpConvertSToF %tC %I12_4",
            "%P12 = OpFMul %tC %SW1 %SX2",
            "%G12 = OpFMul %tC %F12 %P12",
            "%O12 = OpLoad %tC %v12",
            "%N12 = OpFAdd %tC %O12 %G12",
            "OpStore %v12 %N12",
            "%F13 = OpConvertSToF %tC %I13_4",
            "%P13 = OpFMul %tC %SW1 %SX3",
            "%G13 = OpFMul %tC %F13 %P13",
            "%O13 = OpLoad %tC %v13",
            "%N13 = OpFAdd %tC %O13 %G13",
            "OpStore %v13 %N13",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %N00 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y01} %N01 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y02} %N02 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y03} %N03 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %N10 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %N11 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %N12 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %N13 %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            b00 = in(reg) bp!(0,0),
            b10 = in(reg) bp!(1,0),
            b20 = in(reg) bp!(2,0),
            b30 = in(reg) bp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            b01 = in(reg) bp!(0,1),
            b11 = in(reg) bp!(1,1),
            b21 = in(reg) bp!(2,1),
            b31 = in(reg) bp!(3,1),
            a02 = in(reg) ap!(0,2),
            a12 = in(reg) ap!(1,2),
            b02 = in(reg) bp!(0,2),
            b12 = in(reg) bp!(1,2),
            b22 = in(reg) bp!(2,2),
            b32 = in(reg) bp!(3,2),
            a03 = in(reg) ap!(0,3),
            a13 = in(reg) ap!(1,3),
            b03 = in(reg) bp!(0,3),
            b13 = in(reg) bp!(1,3),
            b23 = in(reg) bp!(2,3),
            b33 = in(reg) bp!(3,3),
            dw0 = in(reg) dwp!(0),
            dw1 = in(reg) dwp!(1),
            dx0 = in(reg) dxp!(0),
            dx1 = in(reg) dxp!(1),
            dx2 = in(reg) dxp!(2),
            dx3 = in(reg) dxp!(3)
        );
        }
        bar();
        kc += 128;
    }
}

/// matmul_cm2 tiling with single-f16 activations and BK=64 (half the MMAs, half the barriers per K): weights exact in
/// f16, activations rounded once to f16 (rel. error ~5e-4 per product, far below Prism's q8_1 activations), f32 accumulate.
#[spirv(compute(threads(256)))]
pub fn matmul_cm3(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4096], #[spirv(workgroup)] xh: &mut [u32; 4096]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut kc = 0u32;
    while kc < k {
        {
            let r = l >> 1; let row = (row0 + r).min(m - 1);
            let d = g!(wsc, row * bpr + (kc >> 7));
            unroll!([0, 1], |u| {
                let wi = (l & 1) * 2 + u;
                let word = g!(wq, row * wpr + (kc >> 4) + wi);
                unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
                    let v0 = (((word >> (4 * q)) & 3) as f32 - 1.0) * d; let v1 = (((word >> (4 * q + 2)) & 3) as f32 - 1.0) * d;
                    s!(ws, r * 32 + wi * 8 + q, vec2_to_f16x2(Vec2::new(v0, v1)));
                });
            });
        }
        {
            let t = l >> 1; let half = l & 1; let tt = tok0 + t;
            unroll!([0, 1, 2, 3, 4, 5, 6, 7], |v| {
                let x = if tt < ntok { g!(a4, (xo + tt * xs + kc) / 4 + half * 8 + v) } else { Vec4::ZERO };
                let i0 = t * 32 + half * 16 + v * 2;
                s!(xh, i0, vec2_to_f16x2(Vec2::new(x.x, x.y))); s!(xh, i0 + 1, vec2_to_f16x2(Vec2::new(x.z, x.w)));
            });
        }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 64 >= k { 1 } else { 0 };
        macro_rules! ap { ($i:literal, $kk:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 32 + $kk * 8) as usize) } } }
        macro_rules! hp { ($j:literal, $kk:literal) => { unsafe { xh.index_unchecked(((wc + $j * 16) * 32 + $kk * 8) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f16 = OpTypeFloat 16",
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%c32 = OpConstant %u32 32",
            "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%C00_0 = OpLoad %tC %v00",
            "%C01_0 = OpLoad %tC %v01",
            "%C02_0 = OpLoad %tC %v02",
            "%C03_0 = OpLoad %tC %v03",
            "%C10_0 = OpLoad %tC %v10",
            "%C11_0 = OpLoad %tC %v11",
            "%C12_0 = OpLoad %tC %v12",
            "%C13_0 = OpLoad %tC %v13",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c32",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c32",
            "%H0_0 = OpCooperativeMatrixLoadKHR %tB {h00} %c1 %c32",
            "%H1_0 = OpCooperativeMatrixLoadKHR %tB {h10} %c1 %c32",
            "%H2_0 = OpCooperativeMatrixLoadKHR %tB {h20} %c1 %c32",
            "%H3_0 = OpCooperativeMatrixLoadKHR %tB {h30} %c1 %c32",
            "%C00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H0_0 %C00_0",
            "%C01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H1_0 %C01_0",
            "%C02_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H2_0 %C02_0",
            "%C03_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H3_0 %C03_0",
            "%C10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H0_0 %C10_0",
            "%C11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H1_0 %C11_0",
            "%C12_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H2_0 %C12_0",
            "%C13_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H3_0 %C13_0",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c32",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c32",
            "%H0_1 = OpCooperativeMatrixLoadKHR %tB {h01} %c1 %c32",
            "%H1_1 = OpCooperativeMatrixLoadKHR %tB {h11} %c1 %c32",
            "%H2_1 = OpCooperativeMatrixLoadKHR %tB {h21} %c1 %c32",
            "%H3_1 = OpCooperativeMatrixLoadKHR %tB {h31} %c1 %c32",
            "%C00_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H0_1 %C00_1",
            "%C01_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H1_1 %C01_1",
            "%C02_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H2_1 %C02_1",
            "%C03_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H3_1 %C03_1",
            "%C10_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H0_1 %C10_1",
            "%C11_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H1_1 %C11_1",
            "%C12_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H2_1 %C12_1",
            "%C13_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H3_1 %C13_1",
            "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %c32",
            "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %c32",
            "%H0_2 = OpCooperativeMatrixLoadKHR %tB {h02} %c1 %c32",
            "%H1_2 = OpCooperativeMatrixLoadKHR %tB {h12} %c1 %c32",
            "%H2_2 = OpCooperativeMatrixLoadKHR %tB {h22} %c1 %c32",
            "%H3_2 = OpCooperativeMatrixLoadKHR %tB {h32} %c1 %c32",
            "%C00_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H0_2 %C00_2",
            "%C01_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H1_2 %C01_2",
            "%C02_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H2_2 %C02_2",
            "%C03_3 = OpCooperativeMatrixMulAddKHR %tC %A0_2 %H3_2 %C03_2",
            "%C10_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H0_2 %C10_2",
            "%C11_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H1_2 %C11_2",
            "%C12_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H2_2 %C12_2",
            "%C13_3 = OpCooperativeMatrixMulAddKHR %tC %A1_2 %H3_2 %C13_2",
            "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %c32",
            "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %c32",
            "%H0_3 = OpCooperativeMatrixLoadKHR %tB {h03} %c1 %c32",
            "%H1_3 = OpCooperativeMatrixLoadKHR %tB {h13} %c1 %c32",
            "%H2_3 = OpCooperativeMatrixLoadKHR %tB {h23} %c1 %c32",
            "%H3_3 = OpCooperativeMatrixLoadKHR %tB {h33} %c1 %c32",
            "%C00_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H0_3 %C00_3",
            "%C01_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H1_3 %C01_3",
            "%C02_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H2_3 %C02_3",
            "%C03_4 = OpCooperativeMatrixMulAddKHR %tC %A0_3 %H3_3 %C03_3",
            "%C10_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H0_3 %C10_3",
            "%C11_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H1_3 %C11_3",
            "%C12_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H2_3 %C12_3",
            "%C13_4 = OpCooperativeMatrixMulAddKHR %tC %A1_3 %H3_3 %C13_3",
            "OpStore %v00 %C00_4",
            "OpStore %v01 %C01_4",
            "OpStore %v02 %C02_4",
            "OpStore %v03 %C03_4",
            "OpStore %v10 %C10_4",
            "OpStore %v11 %C11_4",
            "OpStore %v12 %C12_4",
            "OpStore %v13 %C13_4",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %C00_4 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y01} %C01_4 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y02} %C02_4 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y03} %C03_4 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %C10_4 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %C11_4 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %C12_4 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %C13_4 %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            h00 = in(reg) hp!(0,0),
            h10 = in(reg) hp!(1,0),
            h20 = in(reg) hp!(2,0),
            h30 = in(reg) hp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            h01 = in(reg) hp!(0,1),
            h11 = in(reg) hp!(1,1),
            h21 = in(reg) hp!(2,1),
            h31 = in(reg) hp!(3,1),
            a02 = in(reg) ap!(0,2),
            a12 = in(reg) ap!(1,2),
            h02 = in(reg) hp!(0,2),
            h12 = in(reg) hp!(1,2),
            h22 = in(reg) hp!(2,2),
            h32 = in(reg) hp!(3,2),
            a03 = in(reg) ap!(0,3),
            a13 = in(reg) ap!(1,3),
            h03 = in(reg) hp!(0,3),
            h13 = in(reg) hp!(1,3),
            h23 = in(reg) hp!(2,3),
            h33 = in(reg) hp!(3,3)
        );
        }
        bar();
        kc += 64;
    }
}


/// Stores only whole 16-token column tiles below ntok (the caller handles ntok % 16 separately), so it never writes past ntok.
/// matmul_cm2 with shared rows padded to 20 words (conflict-free cooperative-matrix loads) and register prefetch of the next K chunk (global loads overlap the MMAs). v2 PQ2_0 prefill GEMM: 128 rows x 128 tokens per 256-thread workgroup (8 warps, each 32 rows x 64 tokens = 2x4
/// 16x16 f32 accumulators held in Function-storage cooperative-matrix variables across the K loop), K chunks of 32.
/// Exact f16 weights, hi+lo f16 activations. Stores whole 16x16 tiles straight to Y (column-major, stride ys):
/// requires m % 128 == 0 and Y to have room for ntok rounded up to 128 tokens (the engine's maxt-sized buffers).
/// accum != 0 seeds the accumulators from Y.
#[spirv(compute(threads(256)))]
pub fn matmul_cm4(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2560], #[spirv(workgroup)] xh: &mut [u32; 2560], #[spirv(workgroup)] xl: &mut [u32; 2560]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let t = l >> 1; let half = l & 1; let tt = tok0 + t; let live = tt < ntok;
    let mut word = g!(wq, row * wpr + wi); let mut d = g!(wsc, row * bpr);
    let xb = (xo + tt * xs) / 4 + half * 4;
    let z = Vec4::ZERO;
    let mut x0 = if live { g!(a4, xb) } else { z }; let mut x1 = if live { g!(a4, xb + 1) } else { z };
    let mut x2 = if live { g!(a4, xb + 2) } else { z }; let mut x3 = if live { g!(a4, xb + 3) } else { z };
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
            let v0 = (((word >> (4 * q)) & 3) as f32 - 1.0) * d; let v1 = (((word >> (4 * q + 2)) & 3) as f32 - 1.0) * d;
            s!(ws, r * 20 + wi * 8 + q, vec2_to_f16x2(Vec2::new(v0, v1)));
        });
        unroll!([0, 1, 2, 3], |v| {
            let x = if v == 0 { x0 } else if v == 1 { x1 } else if v == 2 { x2 } else { x3 };
            let i0 = t * 20 + half * 8 + v * 2;
            let h0 = vec2_to_f16x2(Vec2::new(x.x, x.y)); let h1 = vec2_to_f16x2(Vec2::new(x.z, x.w));
            s!(xh, i0, h0); s!(xh, i0 + 1, h1);
            let b0 = unpack_exact(h0); let b1 = unpack_exact(h1);
            s!(xl, i0, vec2_to_f16x2(Vec2::new(x.x - b0.x, x.y - b0.y))); s!(xl, i0 + 1, vec2_to_f16x2(Vec2::new(x.z - b1.x, x.w - b1.y)));
        });
        bar();
        let kn = kc + 32;
        if kn < k {
            word = g!(wq, row * wpr + (kn >> 4) + wi); d = g!(wsc, row * bpr + (kn >> 7));
            let b = xb + kn / 4;
            if live { x0 = g!(a4, b); x1 = g!(a4, b + 1); x2 = g!(a4, b + 2); x3 = g!(a4, b + 3); }
        }
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 32 >= k { 1 } else { 0 };
        macro_rules! ap { ($i:literal, $kk:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! hp { ($j:literal, $kk:literal) => { unsafe { xh.index_unchecked(((wc + $j * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! lp { ($j:literal, $kk:literal) => { unsafe { xl.index_unchecked(((wc + $j * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        unsafe {
            asm!(
            "%f16 = OpTypeFloat 16",
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16", "%c20 = OpConstant %u32 20",
            "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%C00_0 = OpLoad %tC %v00",
            "%C01_0 = OpLoad %tC %v01",
            "%C02_0 = OpLoad %tC %v02",
            "%C03_0 = OpLoad %tC %v03",
            "%C10_0 = OpLoad %tC %v10",
            "%C11_0 = OpLoad %tC %v11",
            "%C12_0 = OpLoad %tC %v12",
            "%C13_0 = OpLoad %tC %v13",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c20",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c20",
            "%H0_0 = OpCooperativeMatrixLoadKHR %tB {h00} %c1 %c20",
            "%L0_0 = OpCooperativeMatrixLoadKHR %tB {l00} %c1 %c20",
            "%H1_0 = OpCooperativeMatrixLoadKHR %tB {h10} %c1 %c20",
            "%L1_0 = OpCooperativeMatrixLoadKHR %tB {l10} %c1 %c20",
            "%H2_0 = OpCooperativeMatrixLoadKHR %tB {h20} %c1 %c20",
            "%L2_0 = OpCooperativeMatrixLoadKHR %tB {l20} %c1 %c20",
            "%H3_0 = OpCooperativeMatrixLoadKHR %tB {h30} %c1 %c20",
            "%L3_0 = OpCooperativeMatrixLoadKHR %tB {l30} %c1 %c20",
            "%D00_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L0_0 %C00_0",
            "%C00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H0_0 %D00_0",
            "%D01_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L1_0 %C01_0",
            "%C01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H1_0 %D01_0",
            "%D02_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L2_0 %C02_0",
            "%C02_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H2_0 %D02_0",
            "%D03_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L3_0 %C03_0",
            "%C03_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H3_0 %D03_0",
            "%D10_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L0_0 %C10_0",
            "%C10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H0_0 %D10_0",
            "%D11_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L1_0 %C11_0",
            "%C11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H1_0 %D11_0",
            "%D12_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L2_0 %C12_0",
            "%C12_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H2_0 %D12_0",
            "%D13_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L3_0 %C13_0",
            "%C13_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H3_0 %D13_0",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c20",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c20",
            "%H0_1 = OpCooperativeMatrixLoadKHR %tB {h01} %c1 %c20",
            "%L0_1 = OpCooperativeMatrixLoadKHR %tB {l01} %c1 %c20",
            "%H1_1 = OpCooperativeMatrixLoadKHR %tB {h11} %c1 %c20",
            "%L1_1 = OpCooperativeMatrixLoadKHR %tB {l11} %c1 %c20",
            "%H2_1 = OpCooperativeMatrixLoadKHR %tB {h21} %c1 %c20",
            "%L2_1 = OpCooperativeMatrixLoadKHR %tB {l21} %c1 %c20",
            "%H3_1 = OpCooperativeMatrixLoadKHR %tB {h31} %c1 %c20",
            "%L3_1 = OpCooperativeMatrixLoadKHR %tB {l31} %c1 %c20",
            "%D00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L0_1 %C00_1",
            "%C00_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H0_1 %D00_1",
            "%D01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L1_1 %C01_1",
            "%C01_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H1_1 %D01_1",
            "%D02_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L2_1 %C02_1",
            "%C02_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H2_1 %D02_1",
            "%D03_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L3_1 %C03_1",
            "%C03_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H3_1 %D03_1",
            "%D10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L0_1 %C10_1",
            "%C10_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H0_1 %D10_1",
            "%D11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L1_1 %C11_1",
            "%C11_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H1_1 %D11_1",
            "%D12_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L2_1 %C12_1",
            "%C12_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H2_1 %D12_1",
            "%D13_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L3_1 %C13_1",
            "%C13_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H3_1 %D13_1",
            "OpStore %v00 %C00_2",
            "OpStore %v01 %C01_2",
            "OpStore %v02 %C02_2",
            "OpStore %v03 %C03_2",
            "OpStore %v10 %C10_2",
            "OpStore %v11 %C11_2",
            "OpStore %v12 %C12_2",
            "OpStore %v13 %C13_2",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1", "OpBranchConditional %ok0 %s0 %e0", "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %C00_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %C10_2 %c1 {ys}",
            "OpBranch %e0", "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1", "OpBranchConditional %ok1 %s1 %e1", "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %C01_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %C11_2 %c1 {ys}",
            "OpBranch %e1", "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1", "OpBranchConditional %ok2 %s2 %e2", "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %C02_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %C12_2 %c1 {ys}",
            "OpBranch %e2", "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1", "OpBranchConditional %ok3 %s3 %e3", "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %C03_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %C13_2 %c1 {ys}",
            "OpBranch %e3", "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, ok0 = in(reg) okj(0), ok1 = in(reg) okj(1), ok2 = in(reg) okj(2), ok3 = in(reg) okj(3),
            last = in(reg) last,
            ys = in(reg) ys,
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            h00 = in(reg) hp!(0,0),
            l00 = in(reg) lp!(0,0),
            h10 = in(reg) hp!(1,0),
            l10 = in(reg) lp!(1,0),
            h20 = in(reg) hp!(2,0),
            l20 = in(reg) lp!(2,0),
            h30 = in(reg) hp!(3,0),
            l30 = in(reg) lp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            h01 = in(reg) hp!(0,1),
            l01 = in(reg) lp!(0,1),
            h11 = in(reg) hp!(1,1),
            l11 = in(reg) lp!(1,1),
            h21 = in(reg) hp!(2,1),
            l21 = in(reg) lp!(2,1),
            h31 = in(reg) hp!(3,1),
            l31 = in(reg) lp!(3,1)
        );
        }
        bar();
        kc += 32;
    }
}

/// MTP-head (int8 weights, f32 scale per 16) prefill GEMM on tensor cores: matmul_cm4 with an int8 -> f16 weight decode
/// (weights rounded to f16: draft-only, outputs stay verified by the trunk).
#[spirv(compute(threads(256)))]
pub fn matmul_q8c(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2560], #[spirv(workgroup)] xh: &mut [u32; 2560], #[spirv(workgroup)] xl: &mut [u32; 2560]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let t = l >> 1; let half = l & 1; let tt = tok0 + t; let live = tt < ntok;
    let k4 = k >> 2; let gpr = k >> 4; let wb = row * k4 + wi * 4;
    let mut q0 = g!(wq, wb); let mut q1 = g!(wq, wb + 1); let mut q2 = g!(wq, wb + 2); let mut q3 = g!(wq, wb + 3); let mut d = g!(wsc, row * gpr + wi);
    let xb = (xo + tt * xs) / 4 + half * 4;
    let z = Vec4::ZERO;
    let mut x0 = if live { g!(a4, xb) } else { z }; let mut x1 = if live { g!(a4, xb + 1) } else { z };
    let mut x2 = if live { g!(a4, xb + 2) } else { z }; let mut x3 = if live { g!(a4, xb + 3) } else { z };
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
            let word = if q < 2 { q0 } else if q < 4 { q1 } else if q < 6 { q2 } else { q3 }; let sh = (q & 1) * 16;
            let v0 = (((word << (24 - sh)) as i32 >> 24) as f32) * d; let v1 = (((word << (16 - sh)) as i32 >> 24) as f32) * d;
            s!(ws, r * 20 + wi * 8 + q, vec2_to_f16x2(Vec2::new(v0, v1)));
        });
        unroll!([0, 1, 2, 3], |v| {
            let x = if v == 0 { x0 } else if v == 1 { x1 } else if v == 2 { x2 } else { x3 };
            let i0 = t * 20 + half * 8 + v * 2;
            let h0 = vec2_to_f16x2(Vec2::new(x.x, x.y)); let h1 = vec2_to_f16x2(Vec2::new(x.z, x.w));
            s!(xh, i0, h0); s!(xh, i0 + 1, h1);
            let b0 = unpack_exact(h0); let b1 = unpack_exact(h1);
            s!(xl, i0, vec2_to_f16x2(Vec2::new(x.x - b0.x, x.y - b0.y))); s!(xl, i0 + 1, vec2_to_f16x2(Vec2::new(x.z - b1.x, x.w - b1.y)));
        });
        bar();
        let kn = kc + 32;
        if kn < k {
            let wn = wb + (kn >> 2); q0 = g!(wq, wn); q1 = g!(wq, wn + 1); q2 = g!(wq, wn + 2); q3 = g!(wq, wn + 3); d = g!(wsc, row * gpr + (kn >> 4) + wi);
            let b = xb + kn / 4;
            if live { x0 = g!(a4, b); x1 = g!(a4, b + 1); x2 = g!(a4, b + 2); x3 = g!(a4, b + 3); }
        }
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 32 >= k { 1 } else { 0 };
        macro_rules! ap { ($i:literal, $kk:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! hp { ($j:literal, $kk:literal) => { unsafe { xh.index_unchecked(((wc + $j * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! lp { ($j:literal, $kk:literal) => { unsafe { xl.index_unchecked(((wc + $j * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        unsafe {
            asm!(
            "%f16 = OpTypeFloat 16",
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16", "%c20 = OpConstant %u32 20",
            "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%C00_0 = OpLoad %tC %v00",
            "%C01_0 = OpLoad %tC %v01",
            "%C02_0 = OpLoad %tC %v02",
            "%C03_0 = OpLoad %tC %v03",
            "%C10_0 = OpLoad %tC %v10",
            "%C11_0 = OpLoad %tC %v11",
            "%C12_0 = OpLoad %tC %v12",
            "%C13_0 = OpLoad %tC %v13",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c20",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c20",
            "%H0_0 = OpCooperativeMatrixLoadKHR %tB {h00} %c1 %c20",
            "%L0_0 = OpCooperativeMatrixLoadKHR %tB {l00} %c1 %c20",
            "%H1_0 = OpCooperativeMatrixLoadKHR %tB {h10} %c1 %c20",
            "%L1_0 = OpCooperativeMatrixLoadKHR %tB {l10} %c1 %c20",
            "%H2_0 = OpCooperativeMatrixLoadKHR %tB {h20} %c1 %c20",
            "%L2_0 = OpCooperativeMatrixLoadKHR %tB {l20} %c1 %c20",
            "%H3_0 = OpCooperativeMatrixLoadKHR %tB {h30} %c1 %c20",
            "%L3_0 = OpCooperativeMatrixLoadKHR %tB {l30} %c1 %c20",
            "%D00_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L0_0 %C00_0",
            "%C00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H0_0 %D00_0",
            "%D01_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L1_0 %C01_0",
            "%C01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H1_0 %D01_0",
            "%D02_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L2_0 %C02_0",
            "%C02_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H2_0 %D02_0",
            "%D03_0 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %L3_0 %C03_0",
            "%C03_1 = OpCooperativeMatrixMulAddKHR %tC %A0_0 %H3_0 %D03_0",
            "%D10_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L0_0 %C10_0",
            "%C10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H0_0 %D10_0",
            "%D11_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L1_0 %C11_0",
            "%C11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H1_0 %D11_0",
            "%D12_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L2_0 %C12_0",
            "%C12_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H2_0 %D12_0",
            "%D13_0 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %L3_0 %C13_0",
            "%C13_1 = OpCooperativeMatrixMulAddKHR %tC %A1_0 %H3_0 %D13_0",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c20",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c20",
            "%H0_1 = OpCooperativeMatrixLoadKHR %tB {h01} %c1 %c20",
            "%L0_1 = OpCooperativeMatrixLoadKHR %tB {l01} %c1 %c20",
            "%H1_1 = OpCooperativeMatrixLoadKHR %tB {h11} %c1 %c20",
            "%L1_1 = OpCooperativeMatrixLoadKHR %tB {l11} %c1 %c20",
            "%H2_1 = OpCooperativeMatrixLoadKHR %tB {h21} %c1 %c20",
            "%L2_1 = OpCooperativeMatrixLoadKHR %tB {l21} %c1 %c20",
            "%H3_1 = OpCooperativeMatrixLoadKHR %tB {h31} %c1 %c20",
            "%L3_1 = OpCooperativeMatrixLoadKHR %tB {l31} %c1 %c20",
            "%D00_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L0_1 %C00_1",
            "%C00_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H0_1 %D00_1",
            "%D01_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L1_1 %C01_1",
            "%C01_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H1_1 %D01_1",
            "%D02_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L2_1 %C02_1",
            "%C02_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H2_1 %D02_1",
            "%D03_1 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %L3_1 %C03_1",
            "%C03_2 = OpCooperativeMatrixMulAddKHR %tC %A0_1 %H3_1 %D03_1",
            "%D10_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L0_1 %C10_1",
            "%C10_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H0_1 %D10_1",
            "%D11_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L1_1 %C11_1",
            "%C11_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H1_1 %D11_1",
            "%D12_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L2_1 %C12_1",
            "%C12_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H2_1 %D12_1",
            "%D13_1 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %L3_1 %C13_1",
            "%C13_2 = OpCooperativeMatrixMulAddKHR %tC %A1_1 %H3_1 %D13_1",
            "OpStore %v00 %C00_2",
            "OpStore %v01 %C01_2",
            "OpStore %v02 %C02_2",
            "OpStore %v03 %C03_2",
            "OpStore %v10 %C10_2",
            "OpStore %v11 %C11_2",
            "OpStore %v12 %C12_2",
            "OpStore %v13 %C13_2",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1", "OpBranchConditional %ok0 %s0 %e0", "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %C00_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %C10_2 %c1 {ys}",
            "OpBranch %e0", "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1", "OpBranchConditional %ok1 %s1 %e1", "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %C01_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %C11_2 %c1 {ys}",
            "OpBranch %e1", "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1", "OpBranchConditional %ok2 %s2 %e2", "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %C02_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %C12_2 %c1 {ys}",
            "OpBranch %e2", "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1", "OpBranchConditional %ok3 %s3 %e3", "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %C03_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %C13_2 %c1 {ys}",
            "OpBranch %e3", "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, ok0 = in(reg) okj(0), ok1 = in(reg) okj(1), ok2 = in(reg) okj(2), ok3 = in(reg) okj(3),
            last = in(reg) last,
            ys = in(reg) ys,
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            h00 = in(reg) hp!(0,0),
            l00 = in(reg) lp!(0,0),
            h10 = in(reg) hp!(1,0),
            l10 = in(reg) lp!(1,0),
            h20 = in(reg) hp!(2,0),
            l20 = in(reg) lp!(2,0),
            h30 = in(reg) hp!(3,0),
            l30 = in(reg) lp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            h01 = in(reg) hp!(0,1),
            l01 = in(reg) lp!(0,1),
            h11 = in(reg) hp!(1,1),
            l11 = in(reg) lp!(1,1),
            h21 = in(reg) hp!(2,1),
            l21 = in(reg) lp!(2,1),
            h31 = in(reg) hp!(3,1),
            l31 = in(reg) lp!(3,1)
        );
        }
        bar();
        kc += 32;
    }
}

#[doc(hidden)]
#[spirv(compute(threads(256)))]
pub fn matmul_dbga(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2560], #[spirv(workgroup)] xh: &mut [u32; 2560], #[spirv(workgroup)] xl: &mut [u32; 2560]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let t = l >> 1; let half = l & 1; let tt = tok0 + t; let live = tt < ntok;
    let mut word = g!(wq, row * wpr + wi); let mut d = g!(wsc, row * bpr);
    let xb = (xo + tt * xs) / 4 + half * 4;
    let z = Vec4::ZERO;
    let mut x0 = if live { g!(a4, xb) } else { z }; let mut x1 = if live { g!(a4, xb + 1) } else { z };
    let mut x2 = if live { g!(a4, xb + 2) } else { z }; let mut x3 = if live { g!(a4, xb + 3) } else { z };
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
            let v0 = (((word >> (4 * q)) & 3) as f32 - 1.0) * d; let v1 = (((word >> (4 * q + 2)) & 3) as f32 - 1.0) * d;
            s!(ws, r * 20 + wi * 8 + q, vec2_to_f16x2(Vec2::new(v0, v1)));
        });
        unroll!([0, 1, 2, 3], |v| {
            let x = if v == 0 { x0 } else if v == 1 { x1 } else if v == 2 { x2 } else { x3 };
            let i0 = t * 20 + half * 8 + v * 2;
            let h0 = vec2_to_f16x2(Vec2::new(x.x, x.y)); let h1 = vec2_to_f16x2(Vec2::new(x.z, x.w));
            s!(xh, i0, h0); s!(xh, i0 + 1, h1);
            let b0 = unpack_exact(h0); let b1 = unpack_exact(h1);
            s!(xl, i0, vec2_to_f16x2(Vec2::new(x.x - b0.x, x.y - b0.y))); s!(xl, i0 + 1, vec2_to_f16x2(Vec2::new(x.z - b1.x, x.w - b1.y)));
        });
        bar();
        let kn = kc + 32;
        if kn < k {
            word = g!(wq, row * wpr + (kn >> 4) + wi); d = g!(wsc, row * bpr + (kn >> 7));
            let b = xb + kn / 4;
            if live { x0 = g!(a4, b); x1 = g!(a4, b + 1); x2 = g!(a4, b + 2); x3 = g!(a4, b + 3); }
        }
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 32 >= k { 1 } else { 0 };
        macro_rules! ap { ($i:literal, $kk:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! hp { ($j:literal, $kk:literal) => { unsafe { xh.index_unchecked(((wc + $j * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! lp { ($j:literal, $kk:literal) => { unsafe { xl.index_unchecked(((wc + $j * 16) * 20 + $kk * 8) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        unsafe {
            asm!(
            "%f16 = OpTypeFloat 16",
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16", "%c20 = OpConstant %u32 20",
            "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%C00_0 = OpLoad %tC %v00",
            "%C01_0 = OpLoad %tC %v01",
            "%C02_0 = OpLoad %tC %v02",
            "%C03_0 = OpLoad %tC %v03",
            "%C10_0 = OpLoad %tC %v10",
            "%C11_0 = OpLoad %tC %v11",
            "%C12_0 = OpLoad %tC %v12",
            "%C13_0 = OpLoad %tC %v13",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c20",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c20",
            "%H0_0 = OpCooperativeMatrixLoadKHR %tB {h00} %c1 %c20",
            "%L0_0 = OpCooperativeMatrixLoadKHR %tB {l00} %c1 %c20",
            "%H1_0 = OpCooperativeMatrixLoadKHR %tB {h10} %c1 %c20",
            "%L1_0 = OpCooperativeMatrixLoadKHR %tB {l10} %c1 %c20",
            "%H2_0 = OpCooperativeMatrixLoadKHR %tB {h20} %c1 %c20",
            "%L2_0 = OpCooperativeMatrixLoadKHR %tB {l20} %c1 %c20",
            "%H3_0 = OpCooperativeMatrixLoadKHR %tB {h30} %c1 %c20",
            "%L3_0 = OpCooperativeMatrixLoadKHR %tB {l30} %c1 %c20",
            "%D00_0 = OpCopyObject %tC %C00_0",
            "%C00_1 = OpCopyObject %tC %D00_0",
            "%D01_0 = OpCopyObject %tC %C01_0",
            "%C01_1 = OpCopyObject %tC %D01_0",
            "%D02_0 = OpCopyObject %tC %C02_0",
            "%C02_1 = OpCopyObject %tC %D02_0",
            "%D03_0 = OpCopyObject %tC %C03_0",
            "%C03_1 = OpCopyObject %tC %D03_0",
            "%D10_0 = OpCopyObject %tC %C10_0",
            "%C10_1 = OpCopyObject %tC %D10_0",
            "%D11_0 = OpCopyObject %tC %C11_0",
            "%C11_1 = OpCopyObject %tC %D11_0",
            "%D12_0 = OpCopyObject %tC %C12_0",
            "%C12_1 = OpCopyObject %tC %D12_0",
            "%D13_0 = OpCopyObject %tC %C13_0",
            "%C13_1 = OpCopyObject %tC %D13_0",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c20",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c20",
            "%H0_1 = OpCooperativeMatrixLoadKHR %tB {h01} %c1 %c20",
            "%L0_1 = OpCooperativeMatrixLoadKHR %tB {l01} %c1 %c20",
            "%H1_1 = OpCooperativeMatrixLoadKHR %tB {h11} %c1 %c20",
            "%L1_1 = OpCooperativeMatrixLoadKHR %tB {l11} %c1 %c20",
            "%H2_1 = OpCooperativeMatrixLoadKHR %tB {h21} %c1 %c20",
            "%L2_1 = OpCooperativeMatrixLoadKHR %tB {l21} %c1 %c20",
            "%H3_1 = OpCooperativeMatrixLoadKHR %tB {h31} %c1 %c20",
            "%L3_1 = OpCooperativeMatrixLoadKHR %tB {l31} %c1 %c20",
            "%D00_1 = OpCopyObject %tC %C00_1",
            "%C00_2 = OpCopyObject %tC %D00_1",
            "%D01_1 = OpCopyObject %tC %C01_1",
            "%C01_2 = OpCopyObject %tC %D01_1",
            "%D02_1 = OpCopyObject %tC %C02_1",
            "%C02_2 = OpCopyObject %tC %D02_1",
            "%D03_1 = OpCopyObject %tC %C03_1",
            "%C03_2 = OpCopyObject %tC %D03_1",
            "%D10_1 = OpCopyObject %tC %C10_1",
            "%C10_2 = OpCopyObject %tC %D10_1",
            "%D11_1 = OpCopyObject %tC %C11_1",
            "%C11_2 = OpCopyObject %tC %D11_1",
            "%D12_1 = OpCopyObject %tC %C12_1",
            "%C12_2 = OpCopyObject %tC %D12_1",
            "%D13_1 = OpCopyObject %tC %C13_1",
            "%C13_2 = OpCopyObject %tC %D13_1",
            "OpStore %v00 %C00_2",
            "OpStore %v01 %C01_2",
            "OpStore %v02 %C02_2",
            "OpStore %v03 %C03_2",
            "OpStore %v10 %C10_2",
            "OpStore %v11 %C11_2",
            "OpStore %v12 %C12_2",
            "OpStore %v13 %C13_2",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1", "OpBranchConditional %ok0 %s0 %e0", "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %C00_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %C10_2 %c1 {ys}",
            "OpBranch %e0", "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1", "OpBranchConditional %ok1 %s1 %e1", "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %C01_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %C11_2 %c1 {ys}",
            "OpBranch %e1", "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1", "OpBranchConditional %ok2 %s2 %e2", "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %C02_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %C12_2 %c1 {ys}",
            "OpBranch %e2", "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1", "OpBranchConditional %ok3 %s3 %e3", "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %C03_2 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %C13_2 %c1 {ys}",
            "OpBranch %e3", "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, ok0 = in(reg) okj(0), ok1 = in(reg) okj(1), ok2 = in(reg) okj(2), ok3 = in(reg) okj(3),
            last = in(reg) last,
            ys = in(reg) ys,
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            h00 = in(reg) hp!(0,0),
            l00 = in(reg) lp!(0,0),
            h10 = in(reg) hp!(1,0),
            l10 = in(reg) lp!(1,0),
            h20 = in(reg) hp!(2,0),
            l20 = in(reg) lp!(2,0),
            h30 = in(reg) hp!(3,0),
            l30 = in(reg) lp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            h01 = in(reg) hp!(0,1),
            l01 = in(reg) lp!(0,1),
            h11 = in(reg) hp!(1,1),
            l11 = in(reg) lp!(1,1),
            h21 = in(reg) hp!(2,1),
            l21 = in(reg) lp!(2,1),
            h31 = in(reg) hp!(3,1),
            l31 = in(reg) lp!(3,1)
        );
        }
        bar();
        kc += 32;
    }
}

#[inline(always)] fn sigm(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
#[inline(always)] fn pf(pc: &PC, i: usize) -> f32 { f32::from_bits(g!(pc.u, i)) }

/// NVIDIA tensor-core causal GQA attention over the f16 KV cache (head_dim 256), replacing attn_s_h + attn_comb on the
/// cooperative-matrix path. Workgroup = (head, 64-query block), 4 warps x 16 queries. Two passes over 32-key blocks
/// (pass 1: row max of S = QK^T; pass 2: P = exp(S - max) in f16, O += P V with f32 accumulators held across blocks), so
/// O never needs a per-row rescale. K/V tiles are loaded straight from the cache; Q is staged in shared as f16. O goes
/// to the scratch rows, then O / sum * sigmoid(gate) is written to the output like attn_comb.
/// pc: 0 qo, 1 qts, 2 kco, 3 vco, 4 kvd (halves), 5 pos0, 6 -, 7 nh, 8 nkv, 9 scr, 10 nt, 11 oo, 12 ots, 13 scale
#[spirv(compute(threads(128)))]
pub fn attn_fa(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(workgroup)] qs: &mut [u32; 8448], #[spirv(workgroup)] ss: &mut [f32; 2048], #[spirv(workgroup)] ps: &mut [u32; 1280],
    #[spirv(workgroup)] red: &mut [f32; 128]) {
    let (qo, qts, kco, vco, kvd, pos0) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5));
    let (nh, nkv, scr, nt, oo, ots) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12));
    let scale = pf(pc, 13);
    let h = wg.x; let qb = wg.y * 64; let l = lid.x; let w = l >> 5; let ln = l & 31;
    let kh = h / (nh / nkv);
    {
        let r = l >> 1; let half = l & 1; let t = (qb + r).min(nt - 1);
        let base = (qo + t * qts + h * 512) / 4 + half * 32;
        let mut v = 0u32;
        while v < 32 {
            let x = g!(a4, base + v); let i = r * 132 + half * 64 + v * 2;
            s!(qs, i, vec2_to_f16x2(Vec2::new(x.x, x.y))); s!(qs, i + 1, vec2_to_f16x2(Vec2::new(x.z, x.w)));
            v += 1;
        }
    }
    bar();
    let qlast = (qb + 64).min(nt) - 1; let nblk = (pos0 + qlast + 1 + 31) / 32;
    let r = ln >> 1; let ch = (ln & 1) * 16; let qrow = w * 16 + r; let qpos = pos0 + qb + qrow;
    let sb = w * 512; let pb = w * 320; let ks = kvd / 2; let os = nh * 256;
    macro_rules! qp { ($kk:literal) => { unsafe { qs.index_unchecked((w * 16 * 132 + $kk * 8) as usize) } } }
    macro_rules! sp { ($j:literal) => { unsafe { ss.index_unchecked_mut((sb + $j * 16) as usize) } as *mut f32 } }
    macro_rules! pp { ($kk:literal) => { unsafe { ps.index_unchecked((pb + $kk * 8) as usize) } } }
    macro_rules! op { ($d:literal) => { unsafe { a.index_unchecked_mut((scr + (qb + w * 16) * os + h * 256 + $d * 16) as usize) } as *mut f32 } }
    let mut m = -1e30f32;
    let mut pass = 0u32; let mut sum = 0.0f32;
    while pass < 2 {
        let mut b = 0u32;
        while b < nblk {
            let key0 = b * 32;
            macro_rules! kp { ($j:literal, $kk:literal) => { unsafe { au.index_unchecked((kco + ((key0 + $j * 16) * kvd + kh * 256) / 2 + $kk * 8) as usize) } } }
            unsafe {
                asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%A0 = OpCooperativeMatrixLoadKHR %tA {q0} %c0 %c132",
                "%B0_0 = OpCooperativeMatrixLoadKHR %tB {k00} %c1 {ks}",
                "%S0_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B0_0 %z",
                "%B1_0 = OpCooperativeMatrixLoadKHR %tB {k10} %c1 {ks}",
                "%S1_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B1_0 %z",
                "%A1 = OpCooperativeMatrixLoadKHR %tA {q1} %c0 %c132",
                "%B0_1 = OpCooperativeMatrixLoadKHR %tB {k01} %c1 {ks}",
                "%S0_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B0_1 %S0_1",
                "%B1_1 = OpCooperativeMatrixLoadKHR %tB {k11} %c1 {ks}",
                "%S1_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B1_1 %S1_1",
                "%A2 = OpCooperativeMatrixLoadKHR %tA {q2} %c0 %c132",
                "%B0_2 = OpCooperativeMatrixLoadKHR %tB {k02} %c1 {ks}",
                "%S0_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B0_2 %S0_2",
                "%B1_2 = OpCooperativeMatrixLoadKHR %tB {k12} %c1 {ks}",
                "%S1_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B1_2 %S1_2",
                "%A3 = OpCooperativeMatrixLoadKHR %tA {q3} %c0 %c132",
                "%B0_3 = OpCooperativeMatrixLoadKHR %tB {k03} %c1 {ks}",
                "%S0_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B0_3 %S0_3",
                "%B1_3 = OpCooperativeMatrixLoadKHR %tB {k13} %c1 {ks}",
                "%S1_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B1_3 %S1_3",
                "%A4 = OpCooperativeMatrixLoadKHR %tA {q4} %c0 %c132",
                "%B0_4 = OpCooperativeMatrixLoadKHR %tB {k04} %c1 {ks}",
                "%S0_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B0_4 %S0_4",
                "%B1_4 = OpCooperativeMatrixLoadKHR %tB {k14} %c1 {ks}",
                "%S1_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B1_4 %S1_4",
                "%A5 = OpCooperativeMatrixLoadKHR %tA {q5} %c0 %c132",
                "%B0_5 = OpCooperativeMatrixLoadKHR %tB {k05} %c1 {ks}",
                "%S0_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B0_5 %S0_5",
                "%B1_5 = OpCooperativeMatrixLoadKHR %tB {k15} %c1 {ks}",
                "%S1_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B1_5 %S1_5",
                "%A6 = OpCooperativeMatrixLoadKHR %tA {q6} %c0 %c132",
                "%B0_6 = OpCooperativeMatrixLoadKHR %tB {k06} %c1 {ks}",
                "%S0_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B0_6 %S0_6",
                "%B1_6 = OpCooperativeMatrixLoadKHR %tB {k16} %c1 {ks}",
                "%S1_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B1_6 %S1_6",
                "%A7 = OpCooperativeMatrixLoadKHR %tA {q7} %c0 %c132",
                "%B0_7 = OpCooperativeMatrixLoadKHR %tB {k07} %c1 {ks}",
                "%S0_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B0_7 %S0_7",
                "%B1_7 = OpCooperativeMatrixLoadKHR %tB {k17} %c1 {ks}",
                "%S1_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B1_7 %S1_7",
                "%A8 = OpCooperativeMatrixLoadKHR %tA {q8} %c0 %c132",
                "%B0_8 = OpCooperativeMatrixLoadKHR %tB {k08} %c1 {ks}",
                "%S0_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B0_8 %S0_8",
                "%B1_8 = OpCooperativeMatrixLoadKHR %tB {k18} %c1 {ks}",
                "%S1_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B1_8 %S1_8",
                "%A9 = OpCooperativeMatrixLoadKHR %tA {q9} %c0 %c132",
                "%B0_9 = OpCooperativeMatrixLoadKHR %tB {k09} %c1 {ks}",
                "%S0_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B0_9 %S0_9",
                "%B1_9 = OpCooperativeMatrixLoadKHR %tB {k19} %c1 {ks}",
                "%S1_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B1_9 %S1_9",
                "%A10 = OpCooperativeMatrixLoadKHR %tA {q10} %c0 %c132",
                "%B0_10 = OpCooperativeMatrixLoadKHR %tB {k010} %c1 {ks}",
                "%S0_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B0_10 %S0_10",
                "%B1_10 = OpCooperativeMatrixLoadKHR %tB {k110} %c1 {ks}",
                "%S1_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B1_10 %S1_10",
                "%A11 = OpCooperativeMatrixLoadKHR %tA {q11} %c0 %c132",
                "%B0_11 = OpCooperativeMatrixLoadKHR %tB {k011} %c1 {ks}",
                "%S0_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B0_11 %S0_11",
                "%B1_11 = OpCooperativeMatrixLoadKHR %tB {k111} %c1 {ks}",
                "%S1_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B1_11 %S1_11",
                "%A12 = OpCooperativeMatrixLoadKHR %tA {q12} %c0 %c132",
                "%B0_12 = OpCooperativeMatrixLoadKHR %tB {k012} %c1 {ks}",
                "%S0_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B0_12 %S0_12",
                "%B1_12 = OpCooperativeMatrixLoadKHR %tB {k112} %c1 {ks}",
                "%S1_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B1_12 %S1_12",
                "%A13 = OpCooperativeMatrixLoadKHR %tA {q13} %c0 %c132",
                "%B0_13 = OpCooperativeMatrixLoadKHR %tB {k013} %c1 {ks}",
                "%S0_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B0_13 %S0_13",
                "%B1_13 = OpCooperativeMatrixLoadKHR %tB {k113} %c1 {ks}",
                "%S1_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B1_13 %S1_13",
                "%A14 = OpCooperativeMatrixLoadKHR %tA {q14} %c0 %c132",
                "%B0_14 = OpCooperativeMatrixLoadKHR %tB {k014} %c1 {ks}",
                "%S0_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B0_14 %S0_14",
                "%B1_14 = OpCooperativeMatrixLoadKHR %tB {k114} %c1 {ks}",
                "%S1_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B1_14 %S1_14",
                "%A15 = OpCooperativeMatrixLoadKHR %tA {q15} %c0 %c132",
                "%B0_15 = OpCooperativeMatrixLoadKHR %tB {k015} %c1 {ks}",
                "%S0_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B0_15 %S0_15",
                "%B1_15 = OpCooperativeMatrixLoadKHR %tB {k115} %c1 {ks}",
                "%S1_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B1_15 %S1_15",
                "OpCooperativeMatrixStoreKHR {s0} %S0_16 %c0 %c32",
                "OpCooperativeMatrixStoreKHR {s1} %S1_16 %c0 %c32",
                ks = in(reg) ks,
                q0 = in(reg) qp!(0),
                q1 = in(reg) qp!(1),
                q2 = in(reg) qp!(2),
                q3 = in(reg) qp!(3),
                q4 = in(reg) qp!(4),
                q5 = in(reg) qp!(5),
                q6 = in(reg) qp!(6),
                q7 = in(reg) qp!(7),
                q8 = in(reg) qp!(8),
                q9 = in(reg) qp!(9),
                q10 = in(reg) qp!(10),
                q11 = in(reg) qp!(11),
                q12 = in(reg) qp!(12),
                q13 = in(reg) qp!(13),
                q14 = in(reg) qp!(14),
                q15 = in(reg) qp!(15),
                k00 = in(reg) kp!(0,0),
                k01 = in(reg) kp!(0,1),
                k02 = in(reg) kp!(0,2),
                k03 = in(reg) kp!(0,3),
                k04 = in(reg) kp!(0,4),
                k05 = in(reg) kp!(0,5),
                k06 = in(reg) kp!(0,6),
                k07 = in(reg) kp!(0,7),
                k08 = in(reg) kp!(0,8),
                k09 = in(reg) kp!(0,9),
                k010 = in(reg) kp!(0,10),
                k011 = in(reg) kp!(0,11),
                k012 = in(reg) kp!(0,12),
                k013 = in(reg) kp!(0,13),
                k014 = in(reg) kp!(0,14),
                k015 = in(reg) kp!(0,15),
                k10 = in(reg) kp!(1,0),
                k11 = in(reg) kp!(1,1),
                k12 = in(reg) kp!(1,2),
                k13 = in(reg) kp!(1,3),
                k14 = in(reg) kp!(1,4),
                k15 = in(reg) kp!(1,5),
                k16 = in(reg) kp!(1,6),
                k17 = in(reg) kp!(1,7),
                k18 = in(reg) kp!(1,8),
                k19 = in(reg) kp!(1,9),
                k110 = in(reg) kp!(1,10),
                k111 = in(reg) kp!(1,11),
                k112 = in(reg) kp!(1,12),
                k113 = in(reg) kp!(1,13),
                k114 = in(reg) kp!(1,14),
                k115 = in(reg) kp!(1,15),
                s0 = in(reg) sp!(0),
                s1 = in(reg) sp!(1)
            );
            }
            bar();
            if pass == 0 {
                let mut j = 0u32;
                while j < 16 { if key0 + ch + j <= qpos { m = m.max(g!(ss, sb + r * 32 + ch + j) * scale); } j += 1; }
            } else {
                let mut j = 0u32;
                while j < 16 {
                    let p0 = key0 + ch + j;
                    let e0 = if p0 <= qpos { (g!(ss, sb + r * 32 + ch + j) * scale - m).exp() } else { 0.0 };
                    let e1 = if p0 + 1 <= qpos { (g!(ss, sb + r * 32 + ch + j + 1) * scale - m).exp() } else { 0.0 };
                    sum += e0 + e1;
                    s!(ps, pb + r * 20 + (ch + j) / 2, vec2_to_f16x2(Vec2::new(e0, e1)));
                    j += 2;
                }
                bar();
                let last: u32 = if b + 1 == nblk { 1 } else { 0 };
                macro_rules! vp { ($d:literal, $kk:literal) => { unsafe { au.index_unchecked((vco + ((key0 + $kk * 16) * kvd + kh * 256) / 2 + $d * 8) as usize) } } }
                unsafe {
                    asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%pC = OpTypePointer Function %tC",
                "%last = OpIEqual %bool {last} %c1",
                "%v0 = OpVariable %pC Function %z",
                "%v1 = OpVariable %pC Function %z",
                "%v2 = OpVariable %pC Function %z",
                "%v3 = OpVariable %pC Function %z",
                "%v4 = OpVariable %pC Function %z",
                "%v5 = OpVariable %pC Function %z",
                "%v6 = OpVariable %pC Function %z",
                "%v7 = OpVariable %pC Function %z",
                "%v8 = OpVariable %pC Function %z",
                "%v9 = OpVariable %pC Function %z",
                "%v10 = OpVariable %pC Function %z",
                "%v11 = OpVariable %pC Function %z",
                "%v12 = OpVariable %pC Function %z",
                "%v13 = OpVariable %pC Function %z",
                "%v14 = OpVariable %pC Function %z",
                "%v15 = OpVariable %pC Function %z",
                "%P0 = OpCooperativeMatrixLoadKHR %tA {p0} %c0 %c20",
                "%P1 = OpCooperativeMatrixLoadKHR %tA {p1} %c0 %c20",
                "%O0_0 = OpLoad %tC %v0",
                "%V0_0 = OpCooperativeMatrixLoadKHR %tB {v00} %c0 {ks}",
                "%O0_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V0_0 %O0_0",
                "%V0_1 = OpCooperativeMatrixLoadKHR %tB {v01} %c0 {ks}",
                "%O0_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V0_1 %O0_1",
                "OpStore %v0 %O0_2",
                "%O1_0 = OpLoad %tC %v1",
                "%V1_0 = OpCooperativeMatrixLoadKHR %tB {v10} %c0 {ks}",
                "%O1_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V1_0 %O1_0",
                "%V1_1 = OpCooperativeMatrixLoadKHR %tB {v11} %c0 {ks}",
                "%O1_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V1_1 %O1_1",
                "OpStore %v1 %O1_2",
                "%O2_0 = OpLoad %tC %v2",
                "%V2_0 = OpCooperativeMatrixLoadKHR %tB {v20} %c0 {ks}",
                "%O2_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V2_0 %O2_0",
                "%V2_1 = OpCooperativeMatrixLoadKHR %tB {v21} %c0 {ks}",
                "%O2_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V2_1 %O2_1",
                "OpStore %v2 %O2_2",
                "%O3_0 = OpLoad %tC %v3",
                "%V3_0 = OpCooperativeMatrixLoadKHR %tB {v30} %c0 {ks}",
                "%O3_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V3_0 %O3_0",
                "%V3_1 = OpCooperativeMatrixLoadKHR %tB {v31} %c0 {ks}",
                "%O3_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V3_1 %O3_1",
                "OpStore %v3 %O3_2",
                "%O4_0 = OpLoad %tC %v4",
                "%V4_0 = OpCooperativeMatrixLoadKHR %tB {v40} %c0 {ks}",
                "%O4_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V4_0 %O4_0",
                "%V4_1 = OpCooperativeMatrixLoadKHR %tB {v41} %c0 {ks}",
                "%O4_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V4_1 %O4_1",
                "OpStore %v4 %O4_2",
                "%O5_0 = OpLoad %tC %v5",
                "%V5_0 = OpCooperativeMatrixLoadKHR %tB {v50} %c0 {ks}",
                "%O5_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V5_0 %O5_0",
                "%V5_1 = OpCooperativeMatrixLoadKHR %tB {v51} %c0 {ks}",
                "%O5_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V5_1 %O5_1",
                "OpStore %v5 %O5_2",
                "%O6_0 = OpLoad %tC %v6",
                "%V6_0 = OpCooperativeMatrixLoadKHR %tB {v60} %c0 {ks}",
                "%O6_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V6_0 %O6_0",
                "%V6_1 = OpCooperativeMatrixLoadKHR %tB {v61} %c0 {ks}",
                "%O6_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V6_1 %O6_1",
                "OpStore %v6 %O6_2",
                "%O7_0 = OpLoad %tC %v7",
                "%V7_0 = OpCooperativeMatrixLoadKHR %tB {v70} %c0 {ks}",
                "%O7_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V7_0 %O7_0",
                "%V7_1 = OpCooperativeMatrixLoadKHR %tB {v71} %c0 {ks}",
                "%O7_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V7_1 %O7_1",
                "OpStore %v7 %O7_2",
                "%O8_0 = OpLoad %tC %v8",
                "%V8_0 = OpCooperativeMatrixLoadKHR %tB {v80} %c0 {ks}",
                "%O8_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V8_0 %O8_0",
                "%V8_1 = OpCooperativeMatrixLoadKHR %tB {v81} %c0 {ks}",
                "%O8_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V8_1 %O8_1",
                "OpStore %v8 %O8_2",
                "%O9_0 = OpLoad %tC %v9",
                "%V9_0 = OpCooperativeMatrixLoadKHR %tB {v90} %c0 {ks}",
                "%O9_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V9_0 %O9_0",
                "%V9_1 = OpCooperativeMatrixLoadKHR %tB {v91} %c0 {ks}",
                "%O9_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V9_1 %O9_1",
                "OpStore %v9 %O9_2",
                "%O10_0 = OpLoad %tC %v10",
                "%V10_0 = OpCooperativeMatrixLoadKHR %tB {v100} %c0 {ks}",
                "%O10_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V10_0 %O10_0",
                "%V10_1 = OpCooperativeMatrixLoadKHR %tB {v101} %c0 {ks}",
                "%O10_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V10_1 %O10_1",
                "OpStore %v10 %O10_2",
                "%O11_0 = OpLoad %tC %v11",
                "%V11_0 = OpCooperativeMatrixLoadKHR %tB {v110} %c0 {ks}",
                "%O11_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V11_0 %O11_0",
                "%V11_1 = OpCooperativeMatrixLoadKHR %tB {v111} %c0 {ks}",
                "%O11_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V11_1 %O11_1",
                "OpStore %v11 %O11_2",
                "%O12_0 = OpLoad %tC %v12",
                "%V12_0 = OpCooperativeMatrixLoadKHR %tB {v120} %c0 {ks}",
                "%O12_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V12_0 %O12_0",
                "%V12_1 = OpCooperativeMatrixLoadKHR %tB {v121} %c0 {ks}",
                "%O12_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V12_1 %O12_1",
                "OpStore %v12 %O12_2",
                "%O13_0 = OpLoad %tC %v13",
                "%V13_0 = OpCooperativeMatrixLoadKHR %tB {v130} %c0 {ks}",
                "%O13_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V13_0 %O13_0",
                "%V13_1 = OpCooperativeMatrixLoadKHR %tB {v131} %c0 {ks}",
                "%O13_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V13_1 %O13_1",
                "OpStore %v13 %O13_2",
                "%O14_0 = OpLoad %tC %v14",
                "%V14_0 = OpCooperativeMatrixLoadKHR %tB {v140} %c0 {ks}",
                "%O14_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V14_0 %O14_0",
                "%V14_1 = OpCooperativeMatrixLoadKHR %tB {v141} %c0 {ks}",
                "%O14_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V14_1 %O14_1",
                "OpStore %v14 %O14_2",
                "%O15_0 = OpLoad %tC %v15",
                "%V15_0 = OpCooperativeMatrixLoadKHR %tB {v150} %c0 {ks}",
                "%O15_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V15_0 %O15_0",
                "%V15_1 = OpCooperativeMatrixLoadKHR %tB {v151} %c0 {ks}",
                "%O15_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V15_1 %O15_1",
                "OpStore %v15 %O15_2",
                "OpBranchConditional %last %l1 %m1",
                "%l1 = OpLabel",
                "OpCooperativeMatrixStoreKHR {o0} %O0_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o1} %O1_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o2} %O2_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o3} %O3_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o4} %O4_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o5} %O5_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o6} %O6_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o7} %O7_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o8} %O8_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o9} %O9_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o10} %O10_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o11} %O11_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o12} %O12_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o13} %O13_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o14} %O14_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o15} %O15_2 %c0 {os}",
                "OpBranch %m1",
                "%m1 = OpLabel",
                last = in(reg) last,
                ks = in(reg) ks,
                os = in(reg) os,
                p0 = in(reg) pp!(0),
                p1 = in(reg) pp!(1),
                v00 = in(reg) vp!(0,0),
                v01 = in(reg) vp!(0,1),
                v10 = in(reg) vp!(1,0),
                v11 = in(reg) vp!(1,1),
                v20 = in(reg) vp!(2,0),
                v21 = in(reg) vp!(2,1),
                v30 = in(reg) vp!(3,0),
                v31 = in(reg) vp!(3,1),
                v40 = in(reg) vp!(4,0),
                v41 = in(reg) vp!(4,1),
                v50 = in(reg) vp!(5,0),
                v51 = in(reg) vp!(5,1),
                v60 = in(reg) vp!(6,0),
                v61 = in(reg) vp!(6,1),
                v70 = in(reg) vp!(7,0),
                v71 = in(reg) vp!(7,1),
                v80 = in(reg) vp!(8,0),
                v81 = in(reg) vp!(8,1),
                v90 = in(reg) vp!(9,0),
                v91 = in(reg) vp!(9,1),
                v100 = in(reg) vp!(10,0),
                v101 = in(reg) vp!(10,1),
                v110 = in(reg) vp!(11,0),
                v111 = in(reg) vp!(11,1),
                v120 = in(reg) vp!(12,0),
                v121 = in(reg) vp!(12,1),
                v130 = in(reg) vp!(13,0),
                v131 = in(reg) vp!(13,1),
                v140 = in(reg) vp!(14,0),
                v141 = in(reg) vp!(14,1),
                v150 = in(reg) vp!(15,0),
                v151 = in(reg) vp!(15,1),
                o0 = in(reg) op!(0),
                o1 = in(reg) op!(1),
                o2 = in(reg) op!(2),
                o3 = in(reg) op!(3),
                o4 = in(reg) op!(4),
                o5 = in(reg) op!(5),
                o6 = in(reg) op!(6),
                o7 = in(reg) op!(7),
                o8 = in(reg) op!(8),
                o9 = in(reg) op!(9),
                o10 = in(reg) op!(10),
                o11 = in(reg) op!(11),
                o12 = in(reg) op!(12),
                o13 = in(reg) op!(13),
                o14 = in(reg) op!(14),
                o15 = in(reg) op!(15)
            );
                }
            }
            bar();
            b += 1;
        }
        if pass == 0 { s!(red, l, m); bar(); m = m.max(g!(red, l ^ 1)); bar(); }
        pass += 1;
    }
    s!(red, l, sum); bar();
    let den = sum + g!(red, l ^ 1);
    if ln & 1 == 0 { s!(ss, qrow, den); }
    { use spirv_std::memory::{Scope, Semantics};
      spirv_std::arch::control_barrier::<{ Scope::Workgroup as u32 }, { Scope::Workgroup as u32 }, { Semantics::UNIFORM_MEMORY.bits() | Semantics::WORKGROUP_MEMORY.bits() | Semantics::ACQUIRE_RELEASE.bits() }>(); }
    let rr = l >> 1; let t = qb + rr;
    if t < nt {
        let dn = g!(ss, rr); let half = (l & 1) * 128;
        let mut d = 0u32;
        while d < 128 {
            let dd = half + d;
            let o = g!(a, scr + t * os + h * 256 + dd);
            s!(a, oo + t * ots + h * 256 + dd, o / dn * sigm(g!(a, qo + t * qts + h * 512 + 256 + dd)));
            d += 1;
        }
    }
}

/// Single-pass (online-softmax) variant of attn_fa: the f32 O accumulators are rescaled per 32-key block by a
/// per-row factor broadcast with a stride-0 cooperative-matrix load. Same push constants and grid as attn_fa.
#[spirv(compute(threads(128)))]
pub fn attn_fa1(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(workgroup)] qs: &mut [u32; 8448], #[spirv(workgroup)] ss: &mut [f32; 2048], #[spirv(workgroup)] ps: &mut [u32; 1280],
    #[spirv(workgroup)] red: &mut [f32; 128], #[spirv(workgroup)] alph: &mut [f32; 64]) {
    let (qo, qts, kco, vco, kvd, pos0) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5));
    let (nh, nkv, scr, nt, oo, ots) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12));
    let scale = pf(pc, 13);
    let h = wg.x; let qb = wg.y * 64; let l = lid.x; let w = l >> 5; let ln = l & 31;
    let kh = h / (nh / nkv);
    {
        let r = l >> 1; let half = l & 1; let t = (qb + r).min(nt - 1);
        let base = (qo + t * qts + h * 512) / 4 + half * 32;
        let mut v = 0u32;
        while v < 32 {
            let x = g!(a4, base + v); let i = r * 132 + half * 64 + v * 2;
            s!(qs, i, vec2_to_f16x2(Vec2::new(x.x, x.y))); s!(qs, i + 1, vec2_to_f16x2(Vec2::new(x.z, x.w)));
            v += 1;
        }
    }
    bar();
    let qlast = (qb + 64).min(nt) - 1; let nblk = (pos0 + qlast + 1 + 31) / 32;
    let r = ln >> 1; let ch = (ln & 1) * 16; let qrow = w * 16 + r; let qpos = pos0 + qb + qrow;
    let sb = w * 512; let pb = w * 320; let ks = kvd / 2; let os = nh * 256;
    macro_rules! qp { ($kk:literal) => { unsafe { qs.index_unchecked((w * 16 * 132 + $kk * 8) as usize) } } }
    macro_rules! sp { ($j:literal) => { unsafe { ss.index_unchecked_mut((sb + $j * 16) as usize) } as *mut f32 } }
    macro_rules! pp { ($kk:literal) => { unsafe { ps.index_unchecked((pb + $kk * 8) as usize) } } }
    macro_rules! op { ($d:literal) => { unsafe { a.index_unchecked_mut((scr + (qb + w * 16) * os + h * 256 + $d * 16) as usize) } as *mut f32 } }
    let mut m = -1e30f32; let mut sum = 0.0f32;
    let mut b = 0u32;
    while b < nblk {
        let key0 = b * 32;
        macro_rules! kp { ($j:literal, $kk:literal) => { unsafe { au.index_unchecked((kco + ((key0 + $j * 16) * kvd + kh * 256) / 2 + $kk * 8) as usize) } } }
        unsafe {
            asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%A0 = OpCooperativeMatrixLoadKHR %tA {q0} %c0 %c132",
                "%B0_0 = OpCooperativeMatrixLoadKHR %tB {k00} %c1 {ks}",
                "%S0_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B0_0 %z",
                "%B1_0 = OpCooperativeMatrixLoadKHR %tB {k10} %c1 {ks}",
                "%S1_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B1_0 %z",
                "%A1 = OpCooperativeMatrixLoadKHR %tA {q1} %c0 %c132",
                "%B0_1 = OpCooperativeMatrixLoadKHR %tB {k01} %c1 {ks}",
                "%S0_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B0_1 %S0_1",
                "%B1_1 = OpCooperativeMatrixLoadKHR %tB {k11} %c1 {ks}",
                "%S1_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B1_1 %S1_1",
                "%A2 = OpCooperativeMatrixLoadKHR %tA {q2} %c0 %c132",
                "%B0_2 = OpCooperativeMatrixLoadKHR %tB {k02} %c1 {ks}",
                "%S0_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B0_2 %S0_2",
                "%B1_2 = OpCooperativeMatrixLoadKHR %tB {k12} %c1 {ks}",
                "%S1_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B1_2 %S1_2",
                "%A3 = OpCooperativeMatrixLoadKHR %tA {q3} %c0 %c132",
                "%B0_3 = OpCooperativeMatrixLoadKHR %tB {k03} %c1 {ks}",
                "%S0_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B0_3 %S0_3",
                "%B1_3 = OpCooperativeMatrixLoadKHR %tB {k13} %c1 {ks}",
                "%S1_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B1_3 %S1_3",
                "%A4 = OpCooperativeMatrixLoadKHR %tA {q4} %c0 %c132",
                "%B0_4 = OpCooperativeMatrixLoadKHR %tB {k04} %c1 {ks}",
                "%S0_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B0_4 %S0_4",
                "%B1_4 = OpCooperativeMatrixLoadKHR %tB {k14} %c1 {ks}",
                "%S1_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B1_4 %S1_4",
                "%A5 = OpCooperativeMatrixLoadKHR %tA {q5} %c0 %c132",
                "%B0_5 = OpCooperativeMatrixLoadKHR %tB {k05} %c1 {ks}",
                "%S0_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B0_5 %S0_5",
                "%B1_5 = OpCooperativeMatrixLoadKHR %tB {k15} %c1 {ks}",
                "%S1_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B1_5 %S1_5",
                "%A6 = OpCooperativeMatrixLoadKHR %tA {q6} %c0 %c132",
                "%B0_6 = OpCooperativeMatrixLoadKHR %tB {k06} %c1 {ks}",
                "%S0_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B0_6 %S0_6",
                "%B1_6 = OpCooperativeMatrixLoadKHR %tB {k16} %c1 {ks}",
                "%S1_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B1_6 %S1_6",
                "%A7 = OpCooperativeMatrixLoadKHR %tA {q7} %c0 %c132",
                "%B0_7 = OpCooperativeMatrixLoadKHR %tB {k07} %c1 {ks}",
                "%S0_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B0_7 %S0_7",
                "%B1_7 = OpCooperativeMatrixLoadKHR %tB {k17} %c1 {ks}",
                "%S1_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B1_7 %S1_7",
                "%A8 = OpCooperativeMatrixLoadKHR %tA {q8} %c0 %c132",
                "%B0_8 = OpCooperativeMatrixLoadKHR %tB {k08} %c1 {ks}",
                "%S0_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B0_8 %S0_8",
                "%B1_8 = OpCooperativeMatrixLoadKHR %tB {k18} %c1 {ks}",
                "%S1_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B1_8 %S1_8",
                "%A9 = OpCooperativeMatrixLoadKHR %tA {q9} %c0 %c132",
                "%B0_9 = OpCooperativeMatrixLoadKHR %tB {k09} %c1 {ks}",
                "%S0_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B0_9 %S0_9",
                "%B1_9 = OpCooperativeMatrixLoadKHR %tB {k19} %c1 {ks}",
                "%S1_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B1_9 %S1_9",
                "%A10 = OpCooperativeMatrixLoadKHR %tA {q10} %c0 %c132",
                "%B0_10 = OpCooperativeMatrixLoadKHR %tB {k010} %c1 {ks}",
                "%S0_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B0_10 %S0_10",
                "%B1_10 = OpCooperativeMatrixLoadKHR %tB {k110} %c1 {ks}",
                "%S1_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B1_10 %S1_10",
                "%A11 = OpCooperativeMatrixLoadKHR %tA {q11} %c0 %c132",
                "%B0_11 = OpCooperativeMatrixLoadKHR %tB {k011} %c1 {ks}",
                "%S0_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B0_11 %S0_11",
                "%B1_11 = OpCooperativeMatrixLoadKHR %tB {k111} %c1 {ks}",
                "%S1_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B1_11 %S1_11",
                "%A12 = OpCooperativeMatrixLoadKHR %tA {q12} %c0 %c132",
                "%B0_12 = OpCooperativeMatrixLoadKHR %tB {k012} %c1 {ks}",
                "%S0_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B0_12 %S0_12",
                "%B1_12 = OpCooperativeMatrixLoadKHR %tB {k112} %c1 {ks}",
                "%S1_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B1_12 %S1_12",
                "%A13 = OpCooperativeMatrixLoadKHR %tA {q13} %c0 %c132",
                "%B0_13 = OpCooperativeMatrixLoadKHR %tB {k013} %c1 {ks}",
                "%S0_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B0_13 %S0_13",
                "%B1_13 = OpCooperativeMatrixLoadKHR %tB {k113} %c1 {ks}",
                "%S1_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B1_13 %S1_13",
                "%A14 = OpCooperativeMatrixLoadKHR %tA {q14} %c0 %c132",
                "%B0_14 = OpCooperativeMatrixLoadKHR %tB {k014} %c1 {ks}",
                "%S0_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B0_14 %S0_14",
                "%B1_14 = OpCooperativeMatrixLoadKHR %tB {k114} %c1 {ks}",
                "%S1_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B1_14 %S1_14",
                "%A15 = OpCooperativeMatrixLoadKHR %tA {q15} %c0 %c132",
                "%B0_15 = OpCooperativeMatrixLoadKHR %tB {k015} %c1 {ks}",
                "%S0_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B0_15 %S0_15",
                "%B1_15 = OpCooperativeMatrixLoadKHR %tB {k115} %c1 {ks}",
                "%S1_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B1_15 %S1_15",
                "OpCooperativeMatrixStoreKHR {s0} %S0_16 %c0 %c32",
                "OpCooperativeMatrixStoreKHR {s1} %S1_16 %c0 %c32",
                ks = in(reg) ks,
                q0 = in(reg) qp!(0),
                q1 = in(reg) qp!(1),
                q2 = in(reg) qp!(2),
                q3 = in(reg) qp!(3),
                q4 = in(reg) qp!(4),
                q5 = in(reg) qp!(5),
                q6 = in(reg) qp!(6),
                q7 = in(reg) qp!(7),
                q8 = in(reg) qp!(8),
                q9 = in(reg) qp!(9),
                q10 = in(reg) qp!(10),
                q11 = in(reg) qp!(11),
                q12 = in(reg) qp!(12),
                q13 = in(reg) qp!(13),
                q14 = in(reg) qp!(14),
                q15 = in(reg) qp!(15),
                k00 = in(reg) kp!(0,0),
                k01 = in(reg) kp!(0,1),
                k02 = in(reg) kp!(0,2),
                k03 = in(reg) kp!(0,3),
                k04 = in(reg) kp!(0,4),
                k05 = in(reg) kp!(0,5),
                k06 = in(reg) kp!(0,6),
                k07 = in(reg) kp!(0,7),
                k08 = in(reg) kp!(0,8),
                k09 = in(reg) kp!(0,9),
                k010 = in(reg) kp!(0,10),
                k011 = in(reg) kp!(0,11),
                k012 = in(reg) kp!(0,12),
                k013 = in(reg) kp!(0,13),
                k014 = in(reg) kp!(0,14),
                k015 = in(reg) kp!(0,15),
                k10 = in(reg) kp!(1,0),
                k11 = in(reg) kp!(1,1),
                k12 = in(reg) kp!(1,2),
                k13 = in(reg) kp!(1,3),
                k14 = in(reg) kp!(1,4),
                k15 = in(reg) kp!(1,5),
                k16 = in(reg) kp!(1,6),
                k17 = in(reg) kp!(1,7),
                k18 = in(reg) kp!(1,8),
                k19 = in(reg) kp!(1,9),
                k110 = in(reg) kp!(1,10),
                k111 = in(reg) kp!(1,11),
                k112 = in(reg) kp!(1,12),
                k113 = in(reg) kp!(1,13),
                k114 = in(reg) kp!(1,14),
                k115 = in(reg) kp!(1,15),
                s0 = in(reg) sp!(0),
                s1 = in(reg) sp!(1)
            );
        }
        bar();
        let mut lm = -1e30f32;
        let mut j = 0u32;
        while j < 16 { if key0 + ch + j <= qpos { lm = lm.max(g!(ss, sb + r * 32 + ch + j) * scale); } j += 1; }
        s!(red, l, lm); bar();
        let mn = m.max(lm).max(g!(red, l ^ 1));
        let alpha = (m - mn).exp(); m = mn; sum *= alpha;
        j = 0;
        while j < 16 {
            let p0 = key0 + ch + j;
            let e0 = if p0 <= qpos { (g!(ss, sb + r * 32 + ch + j) * scale - m).exp() } else { 0.0 };
            let e1 = if p0 + 1 <= qpos { (g!(ss, sb + r * 32 + ch + j + 1) * scale - m).exp() } else { 0.0 };
            sum += e0 + e1;
            s!(ps, pb + r * 20 + (ch + j) / 2, vec2_to_f16x2(Vec2::new(e0, e1)));
            j += 2;
        }
        if ln & 1 == 0 { s!(alph, w * 16 + r, alpha); }
        bar();
        let last: u32 = if b + 1 == nblk { 1 } else { 0 };
        macro_rules! vp { ($d:literal, $kk:literal) => { unsafe { au.index_unchecked((vco + ((key0 + $kk * 16) * kvd + kh * 256) / 2 + $d * 8) as usize) } } }
        unsafe {
            asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%pC = OpTypePointer Function %tC",
                "%last = OpIEqual %bool {last} %c1",
                "%v0 = OpVariable %pC Function %z",
                "%v1 = OpVariable %pC Function %z",
                "%v2 = OpVariable %pC Function %z",
                "%v3 = OpVariable %pC Function %z",
                "%v4 = OpVariable %pC Function %z",
                "%v5 = OpVariable %pC Function %z",
                "%v6 = OpVariable %pC Function %z",
                "%v7 = OpVariable %pC Function %z",
                "%v8 = OpVariable %pC Function %z",
                "%v9 = OpVariable %pC Function %z",
                "%v10 = OpVariable %pC Function %z",
                "%v11 = OpVariable %pC Function %z",
                "%v12 = OpVariable %pC Function %z",
                "%v13 = OpVariable %pC Function %z",
                "%v14 = OpVariable %pC Function %z",
                "%v15 = OpVariable %pC Function %z",
                "%AL = OpCooperativeMatrixLoadKHR %tC {al} %c1 %c0", "%P0 = OpCooperativeMatrixLoadKHR %tA {p0} %c0 %c20",
                "%P1 = OpCooperativeMatrixLoadKHR %tA {p1} %c0 %c20",
                "%Q0 = OpLoad %tC %v0", "%O0_0 = OpFMul %tC %Q0 %AL",
                "%V0_0 = OpCooperativeMatrixLoadKHR %tB {v00} %c0 {ks}",
                "%O0_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V0_0 %O0_0",
                "%V0_1 = OpCooperativeMatrixLoadKHR %tB {v01} %c0 {ks}",
                "%O0_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V0_1 %O0_1",
                "OpStore %v0 %O0_2",
                "%Q1 = OpLoad %tC %v1", "%O1_0 = OpFMul %tC %Q1 %AL",
                "%V1_0 = OpCooperativeMatrixLoadKHR %tB {v10} %c0 {ks}",
                "%O1_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V1_0 %O1_0",
                "%V1_1 = OpCooperativeMatrixLoadKHR %tB {v11} %c0 {ks}",
                "%O1_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V1_1 %O1_1",
                "OpStore %v1 %O1_2",
                "%Q2 = OpLoad %tC %v2", "%O2_0 = OpFMul %tC %Q2 %AL",
                "%V2_0 = OpCooperativeMatrixLoadKHR %tB {v20} %c0 {ks}",
                "%O2_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V2_0 %O2_0",
                "%V2_1 = OpCooperativeMatrixLoadKHR %tB {v21} %c0 {ks}",
                "%O2_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V2_1 %O2_1",
                "OpStore %v2 %O2_2",
                "%Q3 = OpLoad %tC %v3", "%O3_0 = OpFMul %tC %Q3 %AL",
                "%V3_0 = OpCooperativeMatrixLoadKHR %tB {v30} %c0 {ks}",
                "%O3_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V3_0 %O3_0",
                "%V3_1 = OpCooperativeMatrixLoadKHR %tB {v31} %c0 {ks}",
                "%O3_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V3_1 %O3_1",
                "OpStore %v3 %O3_2",
                "%Q4 = OpLoad %tC %v4", "%O4_0 = OpFMul %tC %Q4 %AL",
                "%V4_0 = OpCooperativeMatrixLoadKHR %tB {v40} %c0 {ks}",
                "%O4_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V4_0 %O4_0",
                "%V4_1 = OpCooperativeMatrixLoadKHR %tB {v41} %c0 {ks}",
                "%O4_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V4_1 %O4_1",
                "OpStore %v4 %O4_2",
                "%Q5 = OpLoad %tC %v5", "%O5_0 = OpFMul %tC %Q5 %AL",
                "%V5_0 = OpCooperativeMatrixLoadKHR %tB {v50} %c0 {ks}",
                "%O5_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V5_0 %O5_0",
                "%V5_1 = OpCooperativeMatrixLoadKHR %tB {v51} %c0 {ks}",
                "%O5_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V5_1 %O5_1",
                "OpStore %v5 %O5_2",
                "%Q6 = OpLoad %tC %v6", "%O6_0 = OpFMul %tC %Q6 %AL",
                "%V6_0 = OpCooperativeMatrixLoadKHR %tB {v60} %c0 {ks}",
                "%O6_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V6_0 %O6_0",
                "%V6_1 = OpCooperativeMatrixLoadKHR %tB {v61} %c0 {ks}",
                "%O6_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V6_1 %O6_1",
                "OpStore %v6 %O6_2",
                "%Q7 = OpLoad %tC %v7", "%O7_0 = OpFMul %tC %Q7 %AL",
                "%V7_0 = OpCooperativeMatrixLoadKHR %tB {v70} %c0 {ks}",
                "%O7_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V7_0 %O7_0",
                "%V7_1 = OpCooperativeMatrixLoadKHR %tB {v71} %c0 {ks}",
                "%O7_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V7_1 %O7_1",
                "OpStore %v7 %O7_2",
                "%Q8 = OpLoad %tC %v8", "%O8_0 = OpFMul %tC %Q8 %AL",
                "%V8_0 = OpCooperativeMatrixLoadKHR %tB {v80} %c0 {ks}",
                "%O8_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V8_0 %O8_0",
                "%V8_1 = OpCooperativeMatrixLoadKHR %tB {v81} %c0 {ks}",
                "%O8_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V8_1 %O8_1",
                "OpStore %v8 %O8_2",
                "%Q9 = OpLoad %tC %v9", "%O9_0 = OpFMul %tC %Q9 %AL",
                "%V9_0 = OpCooperativeMatrixLoadKHR %tB {v90} %c0 {ks}",
                "%O9_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V9_0 %O9_0",
                "%V9_1 = OpCooperativeMatrixLoadKHR %tB {v91} %c0 {ks}",
                "%O9_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V9_1 %O9_1",
                "OpStore %v9 %O9_2",
                "%Q10 = OpLoad %tC %v10", "%O10_0 = OpFMul %tC %Q10 %AL",
                "%V10_0 = OpCooperativeMatrixLoadKHR %tB {v100} %c0 {ks}",
                "%O10_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V10_0 %O10_0",
                "%V10_1 = OpCooperativeMatrixLoadKHR %tB {v101} %c0 {ks}",
                "%O10_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V10_1 %O10_1",
                "OpStore %v10 %O10_2",
                "%Q11 = OpLoad %tC %v11", "%O11_0 = OpFMul %tC %Q11 %AL",
                "%V11_0 = OpCooperativeMatrixLoadKHR %tB {v110} %c0 {ks}",
                "%O11_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V11_0 %O11_0",
                "%V11_1 = OpCooperativeMatrixLoadKHR %tB {v111} %c0 {ks}",
                "%O11_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V11_1 %O11_1",
                "OpStore %v11 %O11_2",
                "%Q12 = OpLoad %tC %v12", "%O12_0 = OpFMul %tC %Q12 %AL",
                "%V12_0 = OpCooperativeMatrixLoadKHR %tB {v120} %c0 {ks}",
                "%O12_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V12_0 %O12_0",
                "%V12_1 = OpCooperativeMatrixLoadKHR %tB {v121} %c0 {ks}",
                "%O12_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V12_1 %O12_1",
                "OpStore %v12 %O12_2",
                "%Q13 = OpLoad %tC %v13", "%O13_0 = OpFMul %tC %Q13 %AL",
                "%V13_0 = OpCooperativeMatrixLoadKHR %tB {v130} %c0 {ks}",
                "%O13_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V13_0 %O13_0",
                "%V13_1 = OpCooperativeMatrixLoadKHR %tB {v131} %c0 {ks}",
                "%O13_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V13_1 %O13_1",
                "OpStore %v13 %O13_2",
                "%Q14 = OpLoad %tC %v14", "%O14_0 = OpFMul %tC %Q14 %AL",
                "%V14_0 = OpCooperativeMatrixLoadKHR %tB {v140} %c0 {ks}",
                "%O14_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V14_0 %O14_0",
                "%V14_1 = OpCooperativeMatrixLoadKHR %tB {v141} %c0 {ks}",
                "%O14_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V14_1 %O14_1",
                "OpStore %v14 %O14_2",
                "%Q15 = OpLoad %tC %v15", "%O15_0 = OpFMul %tC %Q15 %AL",
                "%V15_0 = OpCooperativeMatrixLoadKHR %tB {v150} %c0 {ks}",
                "%O15_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V15_0 %O15_0",
                "%V15_1 = OpCooperativeMatrixLoadKHR %tB {v151} %c0 {ks}",
                "%O15_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V15_1 %O15_1",
                "OpStore %v15 %O15_2",
                "OpBranchConditional %last %l1 %m1",
                "%l1 = OpLabel",
                "OpCooperativeMatrixStoreKHR {o0} %O0_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o1} %O1_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o2} %O2_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o3} %O3_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o4} %O4_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o5} %O5_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o6} %O6_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o7} %O7_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o8} %O8_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o9} %O9_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o10} %O10_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o11} %O11_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o12} %O12_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o13} %O13_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o14} %O14_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o15} %O15_2 %c0 {os}",
                "OpBranch %m1",
                "%m1 = OpLabel",
                last = in(reg) last, al = in(reg) unsafe { alph.index_unchecked((w * 16) as usize) },
                ks = in(reg) ks,
                os = in(reg) os,
                p0 = in(reg) pp!(0),
                p1 = in(reg) pp!(1),
                v00 = in(reg) vp!(0,0),
                v01 = in(reg) vp!(0,1),
                v10 = in(reg) vp!(1,0),
                v11 = in(reg) vp!(1,1),
                v20 = in(reg) vp!(2,0),
                v21 = in(reg) vp!(2,1),
                v30 = in(reg) vp!(3,0),
                v31 = in(reg) vp!(3,1),
                v40 = in(reg) vp!(4,0),
                v41 = in(reg) vp!(4,1),
                v50 = in(reg) vp!(5,0),
                v51 = in(reg) vp!(5,1),
                v60 = in(reg) vp!(6,0),
                v61 = in(reg) vp!(6,1),
                v70 = in(reg) vp!(7,0),
                v71 = in(reg) vp!(7,1),
                v80 = in(reg) vp!(8,0),
                v81 = in(reg) vp!(8,1),
                v90 = in(reg) vp!(9,0),
                v91 = in(reg) vp!(9,1),
                v100 = in(reg) vp!(10,0),
                v101 = in(reg) vp!(10,1),
                v110 = in(reg) vp!(11,0),
                v111 = in(reg) vp!(11,1),
                v120 = in(reg) vp!(12,0),
                v121 = in(reg) vp!(12,1),
                v130 = in(reg) vp!(13,0),
                v131 = in(reg) vp!(13,1),
                v140 = in(reg) vp!(14,0),
                v141 = in(reg) vp!(14,1),
                v150 = in(reg) vp!(15,0),
                v151 = in(reg) vp!(15,1),
                o0 = in(reg) op!(0),
                o1 = in(reg) op!(1),
                o2 = in(reg) op!(2),
                o3 = in(reg) op!(3),
                o4 = in(reg) op!(4),
                o5 = in(reg) op!(5),
                o6 = in(reg) op!(6),
                o7 = in(reg) op!(7),
                o8 = in(reg) op!(8),
                o9 = in(reg) op!(9),
                o10 = in(reg) op!(10),
                o11 = in(reg) op!(11),
                o12 = in(reg) op!(12),
                o13 = in(reg) op!(13),
                o14 = in(reg) op!(14),
                o15 = in(reg) op!(15)
            );
        }
        bar();
        b += 1;
    }
    s!(red, l, sum); bar();
    let den = sum + g!(red, l ^ 1);
    if ln & 1 == 0 { s!(ss, qrow, den); }
    { use spirv_std::memory::{Scope, Semantics};
      spirv_std::arch::control_barrier::<{ Scope::Workgroup as u32 }, { Scope::Workgroup as u32 }, { Semantics::UNIFORM_MEMORY.bits() | Semantics::WORKGROUP_MEMORY.bits() | Semantics::ACQUIRE_RELEASE.bits() }>(); }
    let rr = l >> 1; let t = qb + rr;
    if t < nt {
        let dn = g!(ss, rr); let half = (l & 1) * 128;
        let mut d = 0u32;
        while d < 128 {
            let dd = half + d;
            let o = g!(a, scr + t * os + h * 256 + dd);
            s!(a, oo + t * ots + h * 256 + dd, o / dn * sigm(g!(a, qo + t * qts + h * 512 + 256 + dd)));
            d += 1;
        }
    }
}

#[inline(always)] fn rnd(v: f32) -> f32 { if v >= 0.0 { (v + 0.5).floor() } else { (v - 0.5).ceil() } } // roundf: ties away from zero (as Prism / Rust)

/// Per-chunk activation quantization for matmul_cm9 (Prism q8-style, one scale per token per 128-wide K group):
/// q = round(x / d), d = max|x| / 127. One warp per (group, token). pc: 0 xo, 1 xs, 2 k, 3 qo (u32), 4 so (f32)
#[spirv(compute(threads(32)))]
pub fn quant_x(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32],
    #[spirv(workgroup)] red: &mut [f32; 32]) {
    let (xo, xs, k, qo, so) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4));
    let (g, t, l) = (wg.x, wg.y, lid.x);
    // q8 layout: within each 16-wide K group the 4x4 (word, byte) grid is transposed: word w holds K = w, w+4, w+8, w+12
    let xb = xo + t * xs + g * 128 + (l >> 2) * 16 + (l & 3);
    let x = Vec4::new(g!(a, xb), g!(a, xb + 4), g!(a, xb + 8), g!(a, xb + 12));
    s!(red, l, x.x.abs().max(x.y.abs()).max(x.z.abs()).max(x.w.abs()));
    bar();
    unroll!([4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l).max(g!(red, l + st))); } bar(); });
    let mx = g!(red, 0);
    let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
    let q0 = (rnd(x.x * inv) as i32) as u32 & 0xff; let q1 = (rnd(x.y * inv) as i32) as u32 & 0xff;
    let q2 = (rnd(x.z * inv) as i32) as u32 & 0xff; let q3 = (rnd(x.w * inv) as i32) as u32 & 0xff;
    s!(au, qo + t * (k / 4) + g * 32 + l, q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
    if l == 0 { s!(a, so + t * (k / 128) + g, mx / 127.0); }
}

/// int8 tensor-core PQ2_0 prefill GEMM over pre-quantized activations (quant_x): exact int8 weight codes {-1,0,1,2},
/// per 128-K group: int32 = codes . q (4 x 16x16x32 s8 MMAs), acc += float(int32) * dw[row] * dx[token] (f32).
/// 128 rows x 128 tokens / 256 threads; shared rows padded to 36 words; stores whole 16-token tiles below ntok only.
/// pc: 0 m, 1 k, 2 ntok, 3 qo (u32 offset of q), 4 so (f32 offset of scales), 5 yo, 6 ys, 7 accum
#[spirv(compute(threads(256)))]
pub fn matmul_cm9(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 4608],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 128]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3], |q| {
            let word = g!(wq, row * wpr + (kc >> 4) + wi * 4 + q);
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        let xb = qo + tt * k4 + (kc >> 2) + half * 16;
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| {
            s!(xq, r * 36 + half * 16 + v, if live { g!(au, xb + v) } else { 0 });
        });
        if half == 0 { s!(dxs, r, if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 }); }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        macro_rules! ap { ($i:literal, $ks:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! bp { ($j:literal, $ks:literal) => { unsafe { xq.index_unchecked(((wc + $j * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! dwp { ($i:literal) => { unsafe { dws.index_unchecked((wr + $i * 16) as usize) } } }
        macro_rules! dxp { ($j:literal) => { unsafe { dxs.index_unchecked((wc + $j * 16) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c3 %c16 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c3 %c32 %c16 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c3 %c16 %c16 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c36",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c36",
            "%B0_0 = OpCooperativeMatrixLoadKHR %tB {b00} %c1 %c36",
            "%B1_0 = OpCooperativeMatrixLoadKHR %tB {b10} %c1 %c36",
            "%B2_0 = OpCooperativeMatrixLoadKHR %tB {b20} %c1 %c36",
            "%B3_0 = OpCooperativeMatrixLoadKHR %tB {b30} %c1 %c36",
            "%I00_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c36",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c36",
            "%B0_1 = OpCooperativeMatrixLoadKHR %tB {b01} %c1 %c36",
            "%B1_1 = OpCooperativeMatrixLoadKHR %tB {b11} %c1 %c36",
            "%B2_1 = OpCooperativeMatrixLoadKHR %tB {b21} %c1 %c36",
            "%B3_1 = OpCooperativeMatrixLoadKHR %tB {b31} %c1 %c36",
            "%I00_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B0_1 %I00_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B1_1 %I01_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B2_1 %I02_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B3_1 %I03_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B0_1 %I10_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B1_1 %I11_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B2_1 %I12_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B3_1 %I13_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %c36",
            "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %c36",
            "%B0_2 = OpCooperativeMatrixLoadKHR %tB {b02} %c1 %c36",
            "%B1_2 = OpCooperativeMatrixLoadKHR %tB {b12} %c1 %c36",
            "%B2_2 = OpCooperativeMatrixLoadKHR %tB {b22} %c1 %c36",
            "%B3_2 = OpCooperativeMatrixLoadKHR %tB {b32} %c1 %c36",
            "%I00_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B0_2 %I00_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B1_2 %I01_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B2_2 %I02_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B3_2 %I03_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B0_2 %I10_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B1_2 %I11_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B2_2 %I12_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B3_2 %I13_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %c36",
            "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %c36",
            "%B0_3 = OpCooperativeMatrixLoadKHR %tB {b03} %c1 %c36",
            "%B1_3 = OpCooperativeMatrixLoadKHR %tB {b13} %c1 %c36",
            "%B2_3 = OpCooperativeMatrixLoadKHR %tB {b23} %c1 %c36",
            "%B3_3 = OpCooperativeMatrixLoadKHR %tB {b33} %c1 %c36",
            "%I00_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B0_3 %I00_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B1_3 %I01_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B2_3 %I02_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B3_3 %I03_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B0_3 %I10_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B1_3 %I11_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B2_3 %I12_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B3_3 %I13_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW0 = OpCooperativeMatrixLoadKHR %tC {dw0} %c1 %c0",
            "%SW1 = OpCooperativeMatrixLoadKHR %tC {dw1} %c1 %c0",
            "%SX0 = OpCooperativeMatrixLoadKHR %tC {dx0} %c0 %c0",
            "%SX1 = OpCooperativeMatrixLoadKHR %tC {dx1} %c0 %c0",
            "%SX2 = OpCooperativeMatrixLoadKHR %tC {dx2} %c0 %c0",
            "%SX3 = OpCooperativeMatrixLoadKHR %tC {dx3} %c0 %c0",
            "%F00 = OpConvertSToF %tC %I00_4",
            "%P00 = OpFMul %tC %SW0 %SX0",
            "%G00 = OpFMul %tC %F00 %P00",
            "%O00 = OpLoad %tC %v00",
            "%N00 = OpFAdd %tC %O00 %G00",
            "OpStore %v00 %N00",
            "%F01 = OpConvertSToF %tC %I01_4",
            "%P01 = OpFMul %tC %SW0 %SX1",
            "%G01 = OpFMul %tC %F01 %P01",
            "%O01 = OpLoad %tC %v01",
            "%N01 = OpFAdd %tC %O01 %G01",
            "OpStore %v01 %N01",
            "%F02 = OpConvertSToF %tC %I02_4",
            "%P02 = OpFMul %tC %SW0 %SX2",
            "%G02 = OpFMul %tC %F02 %P02",
            "%O02 = OpLoad %tC %v02",
            "%N02 = OpFAdd %tC %O02 %G02",
            "OpStore %v02 %N02",
            "%F03 = OpConvertSToF %tC %I03_4",
            "%P03 = OpFMul %tC %SW0 %SX3",
            "%G03 = OpFMul %tC %F03 %P03",
            "%O03 = OpLoad %tC %v03",
            "%N03 = OpFAdd %tC %O03 %G03",
            "OpStore %v03 %N03",
            "%F10 = OpConvertSToF %tC %I10_4",
            "%P10 = OpFMul %tC %SW1 %SX0",
            "%G10 = OpFMul %tC %F10 %P10",
            "%O10 = OpLoad %tC %v10",
            "%N10 = OpFAdd %tC %O10 %G10",
            "OpStore %v10 %N10",
            "%F11 = OpConvertSToF %tC %I11_4",
            "%P11 = OpFMul %tC %SW1 %SX1",
            "%G11 = OpFMul %tC %F11 %P11",
            "%O11 = OpLoad %tC %v11",
            "%N11 = OpFAdd %tC %O11 %G11",
            "OpStore %v11 %N11",
            "%F12 = OpConvertSToF %tC %I12_4",
            "%P12 = OpFMul %tC %SW1 %SX2",
            "%G12 = OpFMul %tC %F12 %P12",
            "%O12 = OpLoad %tC %v12",
            "%N12 = OpFAdd %tC %O12 %G12",
            "OpStore %v12 %N12",
            "%F13 = OpConvertSToF %tC %I13_4",
            "%P13 = OpFMul %tC %SW1 %SX3",
            "%G13 = OpFMul %tC %F13 %P13",
            "%O13 = OpLoad %tC %v13",
            "%N13 = OpFAdd %tC %O13 %G13",
            "OpStore %v13 %N13",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1",
            "OpBranchConditional %ok0 %s0 %e0",
            "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %N00 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %N10 %c1 {ys}",
            "OpBranch %e0",
            "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1",
            "OpBranchConditional %ok1 %s1 %e1",
            "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %N01 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %N11 %c1 {ys}",
            "OpBranch %e1",
            "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1",
            "OpBranchConditional %ok2 %s2 %e2",
            "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %N02 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %N12 %c1 {ys}",
            "OpBranch %e2",
            "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1",
            "OpBranchConditional %ok3 %s3 %e3",
            "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %N03 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %N13 %c1 {ys}",
            "OpBranch %e3",
            "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            ok0 = in(reg) okj(0),
            ok1 = in(reg) okj(1),
            ok2 = in(reg) okj(2),
            ok3 = in(reg) okj(3),
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            b00 = in(reg) bp!(0,0),
            b10 = in(reg) bp!(1,0),
            b20 = in(reg) bp!(2,0),
            b30 = in(reg) bp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            b01 = in(reg) bp!(0,1),
            b11 = in(reg) bp!(1,1),
            b21 = in(reg) bp!(2,1),
            b31 = in(reg) bp!(3,1),
            a02 = in(reg) ap!(0,2),
            a12 = in(reg) ap!(1,2),
            b02 = in(reg) bp!(0,2),
            b12 = in(reg) bp!(1,2),
            b22 = in(reg) bp!(2,2),
            b32 = in(reg) bp!(3,2),
            a03 = in(reg) ap!(0,3),
            a13 = in(reg) ap!(1,3),
            b03 = in(reg) bp!(0,3),
            b13 = in(reg) bp!(1,3),
            b23 = in(reg) bp!(2,3),
            b33 = in(reg) bp!(3,3),
            dw0 = in(reg) dwp!(0),
            dw1 = in(reg) dwp!(1),
            dx0 = in(reg) dxp!(0),
            dx1 = in(reg) dxp!(1),
            dx2 = in(reg) dxp!(2),
            dx3 = in(reg) dxp!(3)
        );
        }
        bar();
        kc += 128;
    }
}

/// matmul_cm9 tiled 64 rows x 256 tokens (half the per-token weight decode work).
#[spirv(compute(threads(256)))]
pub fn matmul_c9w(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2304], #[spirv(workgroup)] xq: &mut [u32; 9216],
    #[spirv(workgroup)] dws: &mut [f32; 64], #[spirv(workgroup)] dxs: &mut [f32; 256]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 1) * 32; let wc = (w >> 1) * 64;
    let row0 = wg.x * 64; let tok0 = wg.y * 256;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 2; let wi = l & 3; let row = (row0 + r).min(m - 1);
    let tt = tok0 + l; let live = tt < ntok;
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1], |q| {
            let word = g!(wq, row * wpr + (kc >> 4) + wi * 2 + q);
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 8 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        let xb = qo + tt * k4 + (kc >> 2);
        let mut v = 0u32;
        while v < 32 { let q4 = if live { g!(au, xb + v) } else { 0 }; s!(xq, l * 36 + v, q4); v += 1; }
        s!(dxs, l, if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 });
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        macro_rules! ap { ($i:literal, $ks:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! bp { ($j:literal, $ks:literal) => { unsafe { xq.index_unchecked(((wc + $j * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! dwp { ($i:literal) => { unsafe { dws.index_unchecked((wr + $i * 16) as usize) } } }
        macro_rules! dxp { ($j:literal) => { unsafe { dxs.index_unchecked((wc + $j * 16) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c3 %c16 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c3 %c32 %c16 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c3 %c16 %c16 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c36",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c36",
            "%B0_0 = OpCooperativeMatrixLoadKHR %tB {b00} %c1 %c36",
            "%B1_0 = OpCooperativeMatrixLoadKHR %tB {b10} %c1 %c36",
            "%B2_0 = OpCooperativeMatrixLoadKHR %tB {b20} %c1 %c36",
            "%B3_0 = OpCooperativeMatrixLoadKHR %tB {b30} %c1 %c36",
            "%I00_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c36",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c36",
            "%B0_1 = OpCooperativeMatrixLoadKHR %tB {b01} %c1 %c36",
            "%B1_1 = OpCooperativeMatrixLoadKHR %tB {b11} %c1 %c36",
            "%B2_1 = OpCooperativeMatrixLoadKHR %tB {b21} %c1 %c36",
            "%B3_1 = OpCooperativeMatrixLoadKHR %tB {b31} %c1 %c36",
            "%I00_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B0_1 %I00_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B1_1 %I01_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B2_1 %I02_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B3_1 %I03_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B0_1 %I10_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B1_1 %I11_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B2_1 %I12_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B3_1 %I13_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %c36",
            "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %c36",
            "%B0_2 = OpCooperativeMatrixLoadKHR %tB {b02} %c1 %c36",
            "%B1_2 = OpCooperativeMatrixLoadKHR %tB {b12} %c1 %c36",
            "%B2_2 = OpCooperativeMatrixLoadKHR %tB {b22} %c1 %c36",
            "%B3_2 = OpCooperativeMatrixLoadKHR %tB {b32} %c1 %c36",
            "%I00_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B0_2 %I00_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B1_2 %I01_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B2_2 %I02_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B3_2 %I03_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B0_2 %I10_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B1_2 %I11_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B2_2 %I12_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B3_2 %I13_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %c36",
            "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %c36",
            "%B0_3 = OpCooperativeMatrixLoadKHR %tB {b03} %c1 %c36",
            "%B1_3 = OpCooperativeMatrixLoadKHR %tB {b13} %c1 %c36",
            "%B2_3 = OpCooperativeMatrixLoadKHR %tB {b23} %c1 %c36",
            "%B3_3 = OpCooperativeMatrixLoadKHR %tB {b33} %c1 %c36",
            "%I00_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B0_3 %I00_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B1_3 %I01_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B2_3 %I02_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B3_3 %I03_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B0_3 %I10_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B1_3 %I11_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B2_3 %I12_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B3_3 %I13_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW0 = OpCooperativeMatrixLoadKHR %tC {dw0} %c1 %c0",
            "%SW1 = OpCooperativeMatrixLoadKHR %tC {dw1} %c1 %c0",
            "%SX0 = OpCooperativeMatrixLoadKHR %tC {dx0} %c0 %c0",
            "%SX1 = OpCooperativeMatrixLoadKHR %tC {dx1} %c0 %c0",
            "%SX2 = OpCooperativeMatrixLoadKHR %tC {dx2} %c0 %c0",
            "%SX3 = OpCooperativeMatrixLoadKHR %tC {dx3} %c0 %c0",
            "%F00 = OpConvertSToF %tC %I00_4",
            "%P00 = OpFMul %tC %SW0 %SX0",
            "%G00 = OpFMul %tC %F00 %P00",
            "%O00 = OpLoad %tC %v00",
            "%N00 = OpFAdd %tC %O00 %G00",
            "OpStore %v00 %N00",
            "%F01 = OpConvertSToF %tC %I01_4",
            "%P01 = OpFMul %tC %SW0 %SX1",
            "%G01 = OpFMul %tC %F01 %P01",
            "%O01 = OpLoad %tC %v01",
            "%N01 = OpFAdd %tC %O01 %G01",
            "OpStore %v01 %N01",
            "%F02 = OpConvertSToF %tC %I02_4",
            "%P02 = OpFMul %tC %SW0 %SX2",
            "%G02 = OpFMul %tC %F02 %P02",
            "%O02 = OpLoad %tC %v02",
            "%N02 = OpFAdd %tC %O02 %G02",
            "OpStore %v02 %N02",
            "%F03 = OpConvertSToF %tC %I03_4",
            "%P03 = OpFMul %tC %SW0 %SX3",
            "%G03 = OpFMul %tC %F03 %P03",
            "%O03 = OpLoad %tC %v03",
            "%N03 = OpFAdd %tC %O03 %G03",
            "OpStore %v03 %N03",
            "%F10 = OpConvertSToF %tC %I10_4",
            "%P10 = OpFMul %tC %SW1 %SX0",
            "%G10 = OpFMul %tC %F10 %P10",
            "%O10 = OpLoad %tC %v10",
            "%N10 = OpFAdd %tC %O10 %G10",
            "OpStore %v10 %N10",
            "%F11 = OpConvertSToF %tC %I11_4",
            "%P11 = OpFMul %tC %SW1 %SX1",
            "%G11 = OpFMul %tC %F11 %P11",
            "%O11 = OpLoad %tC %v11",
            "%N11 = OpFAdd %tC %O11 %G11",
            "OpStore %v11 %N11",
            "%F12 = OpConvertSToF %tC %I12_4",
            "%P12 = OpFMul %tC %SW1 %SX2",
            "%G12 = OpFMul %tC %F12 %P12",
            "%O12 = OpLoad %tC %v12",
            "%N12 = OpFAdd %tC %O12 %G12",
            "OpStore %v12 %N12",
            "%F13 = OpConvertSToF %tC %I13_4",
            "%P13 = OpFMul %tC %SW1 %SX3",
            "%G13 = OpFMul %tC %F13 %P13",
            "%O13 = OpLoad %tC %v13",
            "%N13 = OpFAdd %tC %O13 %G13",
            "OpStore %v13 %N13",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1",
            "OpBranchConditional %ok0 %s0 %e0",
            "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %N00 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %N10 %c1 {ys}",
            "OpBranch %e0",
            "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1",
            "OpBranchConditional %ok1 %s1 %e1",
            "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %N01 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %N11 %c1 {ys}",
            "OpBranch %e1",
            "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1",
            "OpBranchConditional %ok2 %s2 %e2",
            "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %N02 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %N12 %c1 {ys}",
            "OpBranch %e2",
            "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1",
            "OpBranchConditional %ok3 %s3 %e3",
            "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %N03 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %N13 %c1 {ys}",
            "OpBranch %e3",
            "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            ok0 = in(reg) okj(0),
            ok1 = in(reg) okj(1),
            ok2 = in(reg) okj(2),
            ok3 = in(reg) okj(3),
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            b00 = in(reg) bp!(0,0),
            b10 = in(reg) bp!(1,0),
            b20 = in(reg) bp!(2,0),
            b30 = in(reg) bp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            b01 = in(reg) bp!(0,1),
            b11 = in(reg) bp!(1,1),
            b21 = in(reg) bp!(2,1),
            b31 = in(reg) bp!(3,1),
            a02 = in(reg) ap!(0,2),
            a12 = in(reg) ap!(1,2),
            b02 = in(reg) bp!(0,2),
            b12 = in(reg) bp!(1,2),
            b22 = in(reg) bp!(2,2),
            b32 = in(reg) bp!(3,2),
            a03 = in(reg) ap!(0,3),
            a13 = in(reg) ap!(1,3),
            b03 = in(reg) bp!(0,3),
            b13 = in(reg) bp!(1,3),
            b23 = in(reg) bp!(2,3),
            b33 = in(reg) bp!(3,3),
            dw0 = in(reg) dwp!(0),
            dw1 = in(reg) dwp!(1),
            dx0 = in(reg) dxp!(0),
            dx1 = in(reg) dxp!(1),
            dx2 = in(reg) dxp!(2),
            dx3 = in(reg) dxp!(3)
        );
        }
        bar();
        kc += 128;
    }
}

/// matmul_cm9 with register prefetch of the next 128-K chunk (overlaps global loads with the int8 MMAs).
#[spirv(compute(threads(256)))]
pub fn matmul_c9p(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 4608],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 128]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let wb = row * wpr + wi * 4; let xb0 = qo + tt * k4 + half * 16;
    let mut w0 = g!(wq, wb); let mut w1 = g!(wq, wb + 1); let mut w2 = g!(wq, wb + 2); let mut w3 = g!(wq, wb + 3);
    let mut dwv = g!(wsc, row * bpr); let mut dxv = if live { g!(a, so + tt * bpr) } else { 0.0 };
    let mut xr = [0u32; 16];
    unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| { s!(xr, v, if live { g!(au, xb0 + v) } else { 0 }); });
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3], |q| {
            let word = if q == 0 { w0 } else if q == 1 { w1 } else if q == 2 { w2 } else { w3 };
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, dwv); }
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| { s!(xq, r * 36 + half * 16 + v, g!(xr, v)); });
        if half == 0 { s!(dxs, r, dxv); }
        bar();
        let kn = kc + 128;
        if kn < k {
            let wn = wb + (kn >> 4); w0 = g!(wq, wn); w1 = g!(wq, wn + 1); w2 = g!(wq, wn + 2); w3 = g!(wq, wn + 3);
            dwv = g!(wsc, row * bpr + (kn >> 7));
            if live { dxv = g!(a, so + tt * bpr + (kn >> 7)); let xn = xb0 + (kn >> 2);
                unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| { s!(xr, v, g!(au, xn + v)); }); }
        }
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        macro_rules! ap { ($i:literal, $ks:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! bp { ($j:literal, $ks:literal) => { unsafe { xq.index_unchecked(((wc + $j * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! dwp { ($i:literal) => { unsafe { dws.index_unchecked((wr + $i * 16) as usize) } } }
        macro_rules! dxp { ($j:literal) => { unsafe { dxs.index_unchecked((wc + $j * 16) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c3 %c16 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c3 %c32 %c16 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c3 %c16 %c16 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c36",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c36",
            "%B0_0 = OpCooperativeMatrixLoadKHR %tB {b00} %c1 %c36",
            "%B1_0 = OpCooperativeMatrixLoadKHR %tB {b10} %c1 %c36",
            "%B2_0 = OpCooperativeMatrixLoadKHR %tB {b20} %c1 %c36",
            "%B3_0 = OpCooperativeMatrixLoadKHR %tB {b30} %c1 %c36",
            "%I00_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c36",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c36",
            "%B0_1 = OpCooperativeMatrixLoadKHR %tB {b01} %c1 %c36",
            "%B1_1 = OpCooperativeMatrixLoadKHR %tB {b11} %c1 %c36",
            "%B2_1 = OpCooperativeMatrixLoadKHR %tB {b21} %c1 %c36",
            "%B3_1 = OpCooperativeMatrixLoadKHR %tB {b31} %c1 %c36",
            "%I00_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B0_1 %I00_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B1_1 %I01_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B2_1 %I02_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B3_1 %I03_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B0_1 %I10_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B1_1 %I11_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B2_1 %I12_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B3_1 %I13_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %c36",
            "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %c36",
            "%B0_2 = OpCooperativeMatrixLoadKHR %tB {b02} %c1 %c36",
            "%B1_2 = OpCooperativeMatrixLoadKHR %tB {b12} %c1 %c36",
            "%B2_2 = OpCooperativeMatrixLoadKHR %tB {b22} %c1 %c36",
            "%B3_2 = OpCooperativeMatrixLoadKHR %tB {b32} %c1 %c36",
            "%I00_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B0_2 %I00_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B1_2 %I01_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B2_2 %I02_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B3_2 %I03_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B0_2 %I10_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B1_2 %I11_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B2_2 %I12_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B3_2 %I13_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %c36",
            "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %c36",
            "%B0_3 = OpCooperativeMatrixLoadKHR %tB {b03} %c1 %c36",
            "%B1_3 = OpCooperativeMatrixLoadKHR %tB {b13} %c1 %c36",
            "%B2_3 = OpCooperativeMatrixLoadKHR %tB {b23} %c1 %c36",
            "%B3_3 = OpCooperativeMatrixLoadKHR %tB {b33} %c1 %c36",
            "%I00_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B0_3 %I00_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B1_3 %I01_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B2_3 %I02_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B3_3 %I03_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B0_3 %I10_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B1_3 %I11_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B2_3 %I12_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B3_3 %I13_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW0 = OpCooperativeMatrixLoadKHR %tC {dw0} %c1 %c0",
            "%SW1 = OpCooperativeMatrixLoadKHR %tC {dw1} %c1 %c0",
            "%SX0 = OpCooperativeMatrixLoadKHR %tC {dx0} %c0 %c0",
            "%SX1 = OpCooperativeMatrixLoadKHR %tC {dx1} %c0 %c0",
            "%SX2 = OpCooperativeMatrixLoadKHR %tC {dx2} %c0 %c0",
            "%SX3 = OpCooperativeMatrixLoadKHR %tC {dx3} %c0 %c0",
            "%F00 = OpConvertSToF %tC %I00_4",
            "%P00 = OpFMul %tC %SW0 %SX0",
            "%G00 = OpFMul %tC %F00 %P00",
            "%O00 = OpLoad %tC %v00",
            "%N00 = OpFAdd %tC %O00 %G00",
            "OpStore %v00 %N00",
            "%F01 = OpConvertSToF %tC %I01_4",
            "%P01 = OpFMul %tC %SW0 %SX1",
            "%G01 = OpFMul %tC %F01 %P01",
            "%O01 = OpLoad %tC %v01",
            "%N01 = OpFAdd %tC %O01 %G01",
            "OpStore %v01 %N01",
            "%F02 = OpConvertSToF %tC %I02_4",
            "%P02 = OpFMul %tC %SW0 %SX2",
            "%G02 = OpFMul %tC %F02 %P02",
            "%O02 = OpLoad %tC %v02",
            "%N02 = OpFAdd %tC %O02 %G02",
            "OpStore %v02 %N02",
            "%F03 = OpConvertSToF %tC %I03_4",
            "%P03 = OpFMul %tC %SW0 %SX3",
            "%G03 = OpFMul %tC %F03 %P03",
            "%O03 = OpLoad %tC %v03",
            "%N03 = OpFAdd %tC %O03 %G03",
            "OpStore %v03 %N03",
            "%F10 = OpConvertSToF %tC %I10_4",
            "%P10 = OpFMul %tC %SW1 %SX0",
            "%G10 = OpFMul %tC %F10 %P10",
            "%O10 = OpLoad %tC %v10",
            "%N10 = OpFAdd %tC %O10 %G10",
            "OpStore %v10 %N10",
            "%F11 = OpConvertSToF %tC %I11_4",
            "%P11 = OpFMul %tC %SW1 %SX1",
            "%G11 = OpFMul %tC %F11 %P11",
            "%O11 = OpLoad %tC %v11",
            "%N11 = OpFAdd %tC %O11 %G11",
            "OpStore %v11 %N11",
            "%F12 = OpConvertSToF %tC %I12_4",
            "%P12 = OpFMul %tC %SW1 %SX2",
            "%G12 = OpFMul %tC %F12 %P12",
            "%O12 = OpLoad %tC %v12",
            "%N12 = OpFAdd %tC %O12 %G12",
            "OpStore %v12 %N12",
            "%F13 = OpConvertSToF %tC %I13_4",
            "%P13 = OpFMul %tC %SW1 %SX3",
            "%G13 = OpFMul %tC %F13 %P13",
            "%O13 = OpLoad %tC %v13",
            "%N13 = OpFAdd %tC %O13 %G13",
            "OpStore %v13 %N13",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1",
            "OpBranchConditional %ok0 %s0 %e0",
            "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %N00 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %N10 %c1 {ys}",
            "OpBranch %e0",
            "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1",
            "OpBranchConditional %ok1 %s1 %e1",
            "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %N01 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %N11 %c1 {ys}",
            "OpBranch %e1",
            "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1",
            "OpBranchConditional %ok2 %s2 %e2",
            "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %N02 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %N12 %c1 {ys}",
            "OpBranch %e2",
            "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1",
            "OpBranchConditional %ok3 %s3 %e3",
            "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %N03 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %N13 %c1 {ys}",
            "OpBranch %e3",
            "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            ok0 = in(reg) okj(0),
            ok1 = in(reg) okj(1),
            ok2 = in(reg) okj(2),
            ok3 = in(reg) okj(3),
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            b00 = in(reg) bp!(0,0),
            b10 = in(reg) bp!(1,0),
            b20 = in(reg) bp!(2,0),
            b30 = in(reg) bp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            b01 = in(reg) bp!(0,1),
            b11 = in(reg) bp!(1,1),
            b21 = in(reg) bp!(2,1),
            b31 = in(reg) bp!(3,1),
            a02 = in(reg) ap!(0,2),
            a12 = in(reg) ap!(1,2),
            b02 = in(reg) bp!(0,2),
            b12 = in(reg) bp!(1,2),
            b22 = in(reg) bp!(2,2),
            b32 = in(reg) bp!(3,2),
            a03 = in(reg) ap!(0,3),
            a13 = in(reg) ap!(1,3),
            b03 = in(reg) bp!(0,3),
            b13 = in(reg) bp!(1,3),
            b23 = in(reg) bp!(2,3),
            b33 = in(reg) bp!(3,3),
            dw0 = in(reg) dwp!(0),
            dw1 = in(reg) dwp!(1),
            dx0 = in(reg) dxp!(0),
            dx1 = in(reg) dxp!(1),
            dx2 = in(reg) dxp!(2),
            dx3 = in(reg) dxp!(3)
        );
        }
        bar();
        kc += 128;
    }
}

#[spirv(compute(threads(256)))]
pub fn matmul_c9a(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 4608],
    #[spirv(workgroup)] dws: &mut [f32; 2048], #[spirv(workgroup)] dxs: &mut [f32; 128]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3], |q| {
            let word = g!(wq, row * wpr + (kc >> 4) + wi * 4 + q);
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { let dv = g!(wsc, row * bpr + (kc >> 7)); let mut c = 0u32; while c < 16 { s!(dws, (r >> 4) * 256 + c * 16 + (r & 15), dv); c += 1; } }
        let xb = qo + tt * k4 + (kc >> 2) + half * 16;
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| {
            s!(xq, r * 36 + half * 16 + v, if live { g!(au, xb + v) } else { 0 });
        });
        if half == 0 { s!(dxs, r, if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 }); }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        macro_rules! ap { ($i:literal, $ks:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! bp { ($j:literal, $ks:literal) => { unsafe { xq.index_unchecked(((wc + $j * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! dwp { ($i:literal) => { unsafe { dws.index_unchecked(((wr + $i * 16) * 16) as usize) } } }
        macro_rules! dxp { ($j:literal) => { unsafe { dxs.index_unchecked((wc + $j * 16) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c3 %c16 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c3 %c32 %c16 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c3 %c16 %c16 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c36",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c36",
            "%B0_0 = OpCooperativeMatrixLoadKHR %tB {b00} %c1 %c36",
            "%B1_0 = OpCooperativeMatrixLoadKHR %tB {b10} %c1 %c36",
            "%B2_0 = OpCooperativeMatrixLoadKHR %tB {b20} %c1 %c36",
            "%B3_0 = OpCooperativeMatrixLoadKHR %tB {b30} %c1 %c36",
            "%I00_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c36",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c36",
            "%B0_1 = OpCooperativeMatrixLoadKHR %tB {b01} %c1 %c36",
            "%B1_1 = OpCooperativeMatrixLoadKHR %tB {b11} %c1 %c36",
            "%B2_1 = OpCooperativeMatrixLoadKHR %tB {b21} %c1 %c36",
            "%B3_1 = OpCooperativeMatrixLoadKHR %tB {b31} %c1 %c36",
            "%I00_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B0_1 %I00_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B1_1 %I01_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B2_1 %I02_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B3_1 %I03_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B0_1 %I10_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B1_1 %I11_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B2_1 %I12_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B3_1 %I13_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %c36",
            "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %c36",
            "%B0_2 = OpCooperativeMatrixLoadKHR %tB {b02} %c1 %c36",
            "%B1_2 = OpCooperativeMatrixLoadKHR %tB {b12} %c1 %c36",
            "%B2_2 = OpCooperativeMatrixLoadKHR %tB {b22} %c1 %c36",
            "%B3_2 = OpCooperativeMatrixLoadKHR %tB {b32} %c1 %c36",
            "%I00_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B0_2 %I00_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B1_2 %I01_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B2_2 %I02_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B3_2 %I03_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B0_2 %I10_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B1_2 %I11_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B2_2 %I12_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B3_2 %I13_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %c36",
            "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %c36",
            "%B0_3 = OpCooperativeMatrixLoadKHR %tB {b03} %c1 %c36",
            "%B1_3 = OpCooperativeMatrixLoadKHR %tB {b13} %c1 %c36",
            "%B2_3 = OpCooperativeMatrixLoadKHR %tB {b23} %c1 %c36",
            "%B3_3 = OpCooperativeMatrixLoadKHR %tB {b33} %c1 %c36",
            "%I00_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B0_3 %I00_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B1_3 %I01_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B2_3 %I02_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B3_3 %I03_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B0_3 %I10_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B1_3 %I11_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B2_3 %I12_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B3_3 %I13_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW0 = OpCooperativeMatrixLoadKHR %tC {dw0} %c1 %c16",
            "%SW1 = OpCooperativeMatrixLoadKHR %tC {dw1} %c1 %c16",
            "%SX0 = OpCooperativeMatrixLoadKHR %tC {dx0} %c0 %c0",
            "%SX1 = OpCooperativeMatrixLoadKHR %tC {dx1} %c0 %c0",
            "%SX2 = OpCooperativeMatrixLoadKHR %tC {dx2} %c0 %c0",
            "%SX3 = OpCooperativeMatrixLoadKHR %tC {dx3} %c0 %c0",
            "%F00 = OpConvertSToF %tC %I00_4",
            "%P00 = OpFMul %tC %SW0 %SX0",
            "%G00 = OpFMul %tC %F00 %P00",
            "%O00 = OpLoad %tC %v00",
            "%N00 = OpFAdd %tC %O00 %G00",
            "OpStore %v00 %N00",
            "%F01 = OpConvertSToF %tC %I01_4",
            "%P01 = OpFMul %tC %SW0 %SX1",
            "%G01 = OpFMul %tC %F01 %P01",
            "%O01 = OpLoad %tC %v01",
            "%N01 = OpFAdd %tC %O01 %G01",
            "OpStore %v01 %N01",
            "%F02 = OpConvertSToF %tC %I02_4",
            "%P02 = OpFMul %tC %SW0 %SX2",
            "%G02 = OpFMul %tC %F02 %P02",
            "%O02 = OpLoad %tC %v02",
            "%N02 = OpFAdd %tC %O02 %G02",
            "OpStore %v02 %N02",
            "%F03 = OpConvertSToF %tC %I03_4",
            "%P03 = OpFMul %tC %SW0 %SX3",
            "%G03 = OpFMul %tC %F03 %P03",
            "%O03 = OpLoad %tC %v03",
            "%N03 = OpFAdd %tC %O03 %G03",
            "OpStore %v03 %N03",
            "%F10 = OpConvertSToF %tC %I10_4",
            "%P10 = OpFMul %tC %SW1 %SX0",
            "%G10 = OpFMul %tC %F10 %P10",
            "%O10 = OpLoad %tC %v10",
            "%N10 = OpFAdd %tC %O10 %G10",
            "OpStore %v10 %N10",
            "%F11 = OpConvertSToF %tC %I11_4",
            "%P11 = OpFMul %tC %SW1 %SX1",
            "%G11 = OpFMul %tC %F11 %P11",
            "%O11 = OpLoad %tC %v11",
            "%N11 = OpFAdd %tC %O11 %G11",
            "OpStore %v11 %N11",
            "%F12 = OpConvertSToF %tC %I12_4",
            "%P12 = OpFMul %tC %SW1 %SX2",
            "%G12 = OpFMul %tC %F12 %P12",
            "%O12 = OpLoad %tC %v12",
            "%N12 = OpFAdd %tC %O12 %G12",
            "OpStore %v12 %N12",
            "%F13 = OpConvertSToF %tC %I13_4",
            "%P13 = OpFMul %tC %SW1 %SX3",
            "%G13 = OpFMul %tC %F13 %P13",
            "%O13 = OpLoad %tC %v13",
            "%N13 = OpFAdd %tC %O13 %G13",
            "OpStore %v13 %N13",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1",
            "OpBranchConditional %ok0 %s0 %e0",
            "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %N00 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %N10 %c1 {ys}",
            "OpBranch %e0",
            "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1",
            "OpBranchConditional %ok1 %s1 %e1",
            "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %N01 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %N11 %c1 {ys}",
            "OpBranch %e1",
            "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1",
            "OpBranchConditional %ok2 %s2 %e2",
            "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %N02 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %N12 %c1 {ys}",
            "OpBranch %e2",
            "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1",
            "OpBranchConditional %ok3 %s3 %e3",
            "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %N03 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %N13 %c1 {ys}",
            "OpBranch %e3",
            "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            ok0 = in(reg) okj(0),
            ok1 = in(reg) okj(1),
            ok2 = in(reg) okj(2),
            ok3 = in(reg) okj(3),
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            b00 = in(reg) bp!(0,0),
            b10 = in(reg) bp!(1,0),
            b20 = in(reg) bp!(2,0),
            b30 = in(reg) bp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            b01 = in(reg) bp!(0,1),
            b11 = in(reg) bp!(1,1),
            b21 = in(reg) bp!(2,1),
            b31 = in(reg) bp!(3,1),
            a02 = in(reg) ap!(0,2),
            a12 = in(reg) ap!(1,2),
            b02 = in(reg) bp!(0,2),
            b12 = in(reg) bp!(1,2),
            b22 = in(reg) bp!(2,2),
            b32 = in(reg) bp!(3,2),
            a03 = in(reg) ap!(0,3),
            a13 = in(reg) ap!(1,3),
            b03 = in(reg) bp!(0,3),
            b13 = in(reg) bp!(1,3),
            b23 = in(reg) bp!(2,3),
            b33 = in(reg) bp!(3,3),
            dw0 = in(reg) dwp!(0),
            dw1 = in(reg) dwp!(1),
            dx0 = in(reg) dxp!(0),
            dx1 = in(reg) dxp!(1),
            dx2 = in(reg) dxp!(2),
            dx3 = in(reg) dxp!(3)
        );
        }
        bar();
        kc += 128;
    }
}

#[spirv(compute(threads(256)))]
pub fn matmul_c9b(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 4608],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 2048]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3], |q| {
            let word = g!(wq, row * wpr + (kc >> 4) + wi * 4 + q);
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        let xb = qo + tt * k4 + (kc >> 2) + half * 16;
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| {
            s!(xq, r * 36 + half * 16 + v, if live { g!(au, xb + v) } else { 0 });
        });
        if half == 0 { let dv = if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 }; let mut c = 0u32; while c < 16 { s!(dxs, (r >> 4) * 256 + c * 16 + (r & 15), dv); c += 1; } }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        macro_rules! ap { ($i:literal, $ks:literal) => { unsafe { ws.index_unchecked(((wr + $i * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! bp { ($j:literal, $ks:literal) => { unsafe { xq.index_unchecked(((wc + $j * 16) * 36 + $ks * 8) as usize) } } }
        macro_rules! dwp { ($i:literal) => { unsafe { dws.index_unchecked((wr + $i * 16) as usize) } } }
        macro_rules! dxp { ($j:literal) => { unsafe { dxs.index_unchecked(((wc + $j * 16) * 16) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c3 = OpConstant %u32 3",
            "%c16 = OpConstant %u32 16",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c3 %c16 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c3 %c32 %c16 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c3 %c16 %c16 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v00 = OpVariable %pC Function %z",
            "%v01 = OpVariable %pC Function %z",
            "%v02 = OpVariable %pC Function %z",
            "%v03 = OpVariable %pC Function %z",
            "%v10 = OpVariable %pC Function %z",
            "%v11 = OpVariable %pC Function %z",
            "%v12 = OpVariable %pC Function %z",
            "%v13 = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
            "OpStore %v00 %y00",
            "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
            "OpStore %v01 %y01",
            "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
            "OpStore %v02 %y02",
            "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
            "OpStore %v03 %y03",
            "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
            "OpStore %v10 %y10",
            "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
            "OpStore %v11 %y11",
            "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
            "OpStore %v12 %y12",
            "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
            "OpStore %v13 %y13",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c36",
            "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c36",
            "%B0_0 = OpCooperativeMatrixLoadKHR %tB {b00} %c1 %c36",
            "%B1_0 = OpCooperativeMatrixLoadKHR %tB {b10} %c1 %c36",
            "%B2_0 = OpCooperativeMatrixLoadKHR %tB {b20} %c1 %c36",
            "%B3_0 = OpCooperativeMatrixLoadKHR %tB {b30} %c1 %c36",
            "%I00_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B0_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B1_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B2_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B3_0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c36",
            "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c36",
            "%B0_1 = OpCooperativeMatrixLoadKHR %tB {b01} %c1 %c36",
            "%B1_1 = OpCooperativeMatrixLoadKHR %tB {b11} %c1 %c36",
            "%B2_1 = OpCooperativeMatrixLoadKHR %tB {b21} %c1 %c36",
            "%B3_1 = OpCooperativeMatrixLoadKHR %tB {b31} %c1 %c36",
            "%I00_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B0_1 %I00_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B1_1 %I01_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B2_1 %I02_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B3_1 %I03_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B0_1 %I10_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B1_1 %I11_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B2_1 %I12_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B3_1 %I13_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_2 = OpCooperativeMatrixLoadKHR %tA {a02} %c0 %c36",
            "%A1_2 = OpCooperativeMatrixLoadKHR %tA {a12} %c0 %c36",
            "%B0_2 = OpCooperativeMatrixLoadKHR %tB {b02} %c1 %c36",
            "%B1_2 = OpCooperativeMatrixLoadKHR %tB {b12} %c1 %c36",
            "%B2_2 = OpCooperativeMatrixLoadKHR %tB {b22} %c1 %c36",
            "%B3_2 = OpCooperativeMatrixLoadKHR %tB {b32} %c1 %c36",
            "%I00_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B0_2 %I00_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B1_2 %I01_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B2_2 %I02_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_3 = OpCooperativeMatrixMulAddKHR %tI %A0_2 %B3_2 %I03_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B0_2 %I10_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B1_2 %I11_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B2_2 %I12_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_3 = OpCooperativeMatrixMulAddKHR %tI %A1_2 %B3_2 %I13_2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A0_3 = OpCooperativeMatrixLoadKHR %tA {a03} %c0 %c36",
            "%A1_3 = OpCooperativeMatrixLoadKHR %tA {a13} %c0 %c36",
            "%B0_3 = OpCooperativeMatrixLoadKHR %tB {b03} %c1 %c36",
            "%B1_3 = OpCooperativeMatrixLoadKHR %tB {b13} %c1 %c36",
            "%B2_3 = OpCooperativeMatrixLoadKHR %tB {b23} %c1 %c36",
            "%B3_3 = OpCooperativeMatrixLoadKHR %tB {b33} %c1 %c36",
            "%I00_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B0_3 %I00_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I01_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B1_3 %I01_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I02_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B2_3 %I02_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I03_4 = OpCooperativeMatrixMulAddKHR %tI %A0_3 %B3_3 %I03_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I10_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B0_3 %I10_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I11_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B1_3 %I11_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I12_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B2_3 %I12_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I13_4 = OpCooperativeMatrixMulAddKHR %tI %A1_3 %B3_3 %I13_3 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW0 = OpCooperativeMatrixLoadKHR %tC {dw0} %c1 %c0",
            "%SW1 = OpCooperativeMatrixLoadKHR %tC {dw1} %c1 %c0",
            "%SX0 = OpCooperativeMatrixLoadKHR %tC {dx0} %c0 %c16",
            "%SX1 = OpCooperativeMatrixLoadKHR %tC {dx1} %c0 %c16",
            "%SX2 = OpCooperativeMatrixLoadKHR %tC {dx2} %c0 %c16",
            "%SX3 = OpCooperativeMatrixLoadKHR %tC {dx3} %c0 %c16",
            "%F00 = OpConvertSToF %tC %I00_4",
            "%P00 = OpFMul %tC %SW0 %SX0",
            "%G00 = OpFMul %tC %F00 %P00",
            "%O00 = OpLoad %tC %v00",
            "%N00 = OpFAdd %tC %O00 %G00",
            "OpStore %v00 %N00",
            "%F01 = OpConvertSToF %tC %I01_4",
            "%P01 = OpFMul %tC %SW0 %SX1",
            "%G01 = OpFMul %tC %F01 %P01",
            "%O01 = OpLoad %tC %v01",
            "%N01 = OpFAdd %tC %O01 %G01",
            "OpStore %v01 %N01",
            "%F02 = OpConvertSToF %tC %I02_4",
            "%P02 = OpFMul %tC %SW0 %SX2",
            "%G02 = OpFMul %tC %F02 %P02",
            "%O02 = OpLoad %tC %v02",
            "%N02 = OpFAdd %tC %O02 %G02",
            "OpStore %v02 %N02",
            "%F03 = OpConvertSToF %tC %I03_4",
            "%P03 = OpFMul %tC %SW0 %SX3",
            "%G03 = OpFMul %tC %F03 %P03",
            "%O03 = OpLoad %tC %v03",
            "%N03 = OpFAdd %tC %O03 %G03",
            "OpStore %v03 %N03",
            "%F10 = OpConvertSToF %tC %I10_4",
            "%P10 = OpFMul %tC %SW1 %SX0",
            "%G10 = OpFMul %tC %F10 %P10",
            "%O10 = OpLoad %tC %v10",
            "%N10 = OpFAdd %tC %O10 %G10",
            "OpStore %v10 %N10",
            "%F11 = OpConvertSToF %tC %I11_4",
            "%P11 = OpFMul %tC %SW1 %SX1",
            "%G11 = OpFMul %tC %F11 %P11",
            "%O11 = OpLoad %tC %v11",
            "%N11 = OpFAdd %tC %O11 %G11",
            "OpStore %v11 %N11",
            "%F12 = OpConvertSToF %tC %I12_4",
            "%P12 = OpFMul %tC %SW1 %SX2",
            "%G12 = OpFMul %tC %F12 %P12",
            "%O12 = OpLoad %tC %v12",
            "%N12 = OpFAdd %tC %O12 %G12",
            "OpStore %v12 %N12",
            "%F13 = OpConvertSToF %tC %I13_4",
            "%P13 = OpFMul %tC %SW1 %SX3",
            "%G13 = OpFMul %tC %F13 %P13",
            "%O13 = OpLoad %tC %v13",
            "%N13 = OpFAdd %tC %O13 %G13",
            "OpStore %v13 %N13",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "%ok0 = OpIEqual %bool {ok0} %c1",
            "OpBranchConditional %ok0 %s0 %e0",
            "%s0 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y00} %N00 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y10} %N10 %c1 {ys}",
            "OpBranch %e0",
            "%e0 = OpLabel",
            "%ok1 = OpIEqual %bool {ok1} %c1",
            "OpBranchConditional %ok1 %s1 %e1",
            "%s1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y01} %N01 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y11} %N11 %c1 {ys}",
            "OpBranch %e1",
            "%e1 = OpLabel",
            "%ok2 = OpIEqual %bool {ok2} %c1",
            "OpBranchConditional %ok2 %s2 %e2",
            "%s2 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y02} %N02 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y12} %N12 %c1 {ys}",
            "OpBranch %e2",
            "%e2 = OpLabel",
            "%ok3 = OpIEqual %bool {ok3} %c1",
            "OpBranchConditional %ok3 %s3 %e3",
            "%s3 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y03} %N03 %c1 {ys}",
            "OpCooperativeMatrixStoreKHR {y13} %N13 %c1 {ys}",
            "OpBranch %e3",
            "%e3 = OpLabel",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa,
            last = in(reg) last,
            ys = in(reg) ys,
            ok0 = in(reg) okj(0),
            ok1 = in(reg) okj(1),
            ok2 = in(reg) okj(2),
            ok3 = in(reg) okj(3),
            y00 = in(reg) yp!(0,0),
            y01 = in(reg) yp!(0,1),
            y02 = in(reg) yp!(0,2),
            y03 = in(reg) yp!(0,3),
            y10 = in(reg) yp!(1,0),
            y11 = in(reg) yp!(1,1),
            y12 = in(reg) yp!(1,2),
            y13 = in(reg) yp!(1,3),
            a00 = in(reg) ap!(0,0),
            a10 = in(reg) ap!(1,0),
            b00 = in(reg) bp!(0,0),
            b10 = in(reg) bp!(1,0),
            b20 = in(reg) bp!(2,0),
            b30 = in(reg) bp!(3,0),
            a01 = in(reg) ap!(0,1),
            a11 = in(reg) ap!(1,1),
            b01 = in(reg) bp!(0,1),
            b11 = in(reg) bp!(1,1),
            b21 = in(reg) bp!(2,1),
            b31 = in(reg) bp!(3,1),
            a02 = in(reg) ap!(0,2),
            a12 = in(reg) ap!(1,2),
            b02 = in(reg) bp!(0,2),
            b12 = in(reg) bp!(1,2),
            b22 = in(reg) bp!(2,2),
            b32 = in(reg) bp!(3,2),
            a03 = in(reg) ap!(0,3),
            a13 = in(reg) ap!(1,3),
            b03 = in(reg) bp!(0,3),
            b13 = in(reg) bp!(1,3),
            b23 = in(reg) bp!(2,3),
            b33 = in(reg) bp!(3,3),
            dw0 = in(reg) dwp!(0),
            dw1 = in(reg) dwp!(1),
            dx0 = in(reg) dxp!(0),
            dx1 = in(reg) dxp!(1),
            dx2 = in(reg) dxp!(2),
            dx3 = in(reg) dxp!(3)
        );
        }
        bar();
        kc += 128;
    }
}

/// NVIDIA variant of gdn_p (Gated DeltaNet prefill, S=128): same math and token order, but 8 row-groups x 32 columns
/// per 256-thread workgroup (4 workgroups per head instead of 2), so twice the workgroups and half the serial state
/// work per thread per token. Same push constants as gdn_p; grid (heads, 4).
#[spirv(compute(threads(256)))]
pub fn gdn_p2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [[f32; 4]; 32], #[spirv(workgroup)] ks: &mut [[f32; 4]; 32],
    #[spirv(workgroup)] r1: &mut [[f32; 8]; 32], #[spirv(workgroup)] r2: &mut [[f32; 8]; 32]) {
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, l) = (wg.x, lid.x); let rq = l & 7; let jl = l >> 3; let j = wg.y * 32 + jl; let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd + rq * 16 * sd + j;
    let ld = |i: u32| Vec4::new(g!(a, sb + (4 * i) * sd), g!(a, sb + (4 * i + 1) * sd), g!(a, sb + (4 * i + 2) * sd), g!(a, sb + (4 * i + 3) * sd));
    let (mut s0, mut s1, mut s2, mut s3) = (ld(0), ld(1), ld(2), ld(3));
    macro_rules! all4 { (|$s:ident, $i:ident| $b:block) => {
        { let $i = 0u32; let $s = &mut s0; $b } { let $i = 1u32; let $s = &mut s1; $b } { let $i = 2u32; let $s = &mut s2; $b } { let $i = 3u32; let $s = &mut s3; $b } } }
    let v4 = |x: [f32; 4]| Vec4::new(x[0], x[1], x[2], x[3]);
    let sum8 = |x: [f32; 8]| x[0] + x[1] + x[2] + x[3] + x[4] + x[5] + x[6] + x[7];
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
        all4!(|s, i| { *s = *s * eg; sm += *s * v4(g!(ks, rq * 4 + i)); });
        unsafe { *r1.index_unchecked_mut(jl as usize).index_unchecked_mut(rq as usize) = sm.dot(Vec4::ONE); }
        bar();
        let delta = (vj - sum8(g!(r1, jl))) * beta;
        let mut o = Vec4::ZERO;
        all4!(|s, i| { let kv = v4(g!(ks, rq * 4 + i)); *s = *s + kv * delta; o += *s * v4(g!(qs, rq * 4 + i)); });
        unsafe { *r2.index_unchecked_mut(jl as usize).index_unchecked_mut(rq as usize) = o.dot(Vec4::ONE); }
        bar();
        if rq == 0 { s!(a, oo + t * os + h * sd + j, sum8(g!(r2, jl)) * scale); }
        bar();
        t += 1;
        if t >= ntok { break; }
    } }
    all4!(|s, i| { let v = *s; s!(a, sb + (4 * i) * sd, v.x); s!(a, sb + (4 * i + 1) * sd, v.y); s!(a, sb + (4 * i + 2) * sd, v.z); s!(a, sb + (4 * i + 3) * sd, v.w); });
}

/// NVIDIA tiled variant of matvec_f32 (ssm_alpha / ssm_beta over a prefill chunk): 16 rows x 16 tokens per 256-thread
/// workgroup, K staged through shared in 64-wide chunks, so each weight/activation load is reused 16x.
/// Same push constants as matvec_f32 (0 rows, 1 k, 2 ntok, 3 xo, 4 xs, 5 yo, 6 ys, 7 wo); k % 64 == 0. Grid (rows/16, ntok/16).
#[spirv(compute(threads(256)))]
pub fn matvec_f32t(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] wsh: &mut [f32; 1040], #[spirv(workgroup)] xsh: &mut [f32; 1040]) {
    let (m, k, ntok, xo, xs, yo, ys, wo) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let rr = l >> 4; let tt = l & 15;
    let row0 = wg.x * 16; let tok0 = wg.y * 16;
    let mut sm = 0.0f32;
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3], |q| {
            let idx = q * 256 + l; let r = idx >> 6; let c = idx & 63;
            let row = (row0 + r).min(m - 1); let tok = (tok0 + r).min(ntok - 1);
            s!(wsh, r * 65 + c, g!(prm, wo + row * k + kc + c));
            s!(xsh, r * 65 + c, g!(a, xo + tok * xs + kc + c));
        });
        bar();
        let mut c = 0u32;
        while c < 64 { sm += g!(wsh, rr * 65 + c) * g!(xsh, tt * 65 + c); c += 1; }
        bar();
        kc += 64;
    }
    if row0 + rr < m && tok0 + tt < ntok { s!(a, yo + (tok0 + tt) * ys + row0 + rr, sm); }
}

/// gdn_p2 with subgroup-shuffle reductions (the 8 row-groups of a column are adjacent lanes) and q/k staged for 16
/// tokens at a time: 2 workgroup barriers per 16 tokens instead of 4 per token. Same math, same push constants/grid.
#[spirv(compute(threads(256)))]
pub fn gdn_p3(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [Vec4; 512], #[spirv(workgroup)] ks: &mut [Vec4; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, l) = (wg.x, lid.x); let rq = l & 7; let j = wg.y * 32 + (l >> 3); let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd + rq * 16 * sd + j;
    let ld = |i: u32| Vec4::new(g!(a, sb + (4 * i) * sd), g!(a, sb + (4 * i + 1) * sd), g!(a, sb + (4 * i + 2) * sd), g!(a, sb + (4 * i + 3) * sd));
    let (mut s0, mut s1, mut s2, mut s3) = (ld(0), ld(1), ld(2), ld(3));
    macro_rules! all4 { (|$s:ident, $i:ident| $b:block) => {
        { let $i = 0u32; let $s = &mut s0; $b } { let $i = 1u32; let $s = &mut s1; $b } { let $i = 2u32; let $s = &mut s2; $b } { let $i = 3u32; let $s = &mut s3; $b } } }
    let red8 = |v: f32| -> f32 { let v = v + sx(v, 1); let v = v + sx(v, 2); v + sx(v, 4) };
    let dt = g!(prm, dto + h); let aa = g!(prm, ao + h);
    let mut tb = 0u32;
    while tb < ntok {
        unroll!([0, 1], |c| {
            let idx = l + 256 * c; let tk = idx >> 5; let d4 = idx & 31; let t = tb + tk;
            let cb = (co + t.min(ntok - 1) * cs) / 4;
            s!(qs, idx, g!(a4, cb + (kh * sd) / 4 + d4)); s!(ks, idx, g!(a4, cb + (qk + kh * sd) / 4 + d4));
        });
        bar();
        let nb = (ntok - tb).min(16);
        let mut u = 0u32;
        while u < nb {
            let t = tb + u; let cb = co + t * cs;
            let vj = g!(a, cb + 2 * qk + h * sd + j);
            let beta = sigm(g!(a, bo + t * bas + h));
            let x = g!(a, alo + t * bas + h) + dt;
            let eg = (aa * (if x > 20.0 { x } else { (1.0 + x.exp()).ln() })).exp();
            let kb = u * 32 + rq * 4;
            let mut sm = Vec4::ZERO;
            all4!(|s, i| { *s = *s * eg; sm += *s * g!(ks, kb + i); });
            let delta = (vj - red8(sm.dot(Vec4::ONE))) * beta;
            let mut o = Vec4::ZERO;
            all4!(|s, i| { *s = *s + g!(ks, kb + i) * delta; o += *s * g!(qs, kb + i); });
            let ov = red8(o.dot(Vec4::ONE));
            if rq == 0 { s!(a, oo + t * os + h * sd + j, ov * scale); }
            u += 1;
        }
        bar();
        tb += 16;
    }
    all4!(|s, i| { let v = *s; s!(a, sb + (4 * i) * sd, v.x); s!(a, sb + (4 * i + 1) * sd, v.y); s!(a, sb + (4 * i + 2) * sd, v.z); s!(a, sb + (4 * i + 3) * sd, v.w); });
}

/// matmul_cm9 with double-buffered shared memory: 64-wide K steps, the next step's weights (decoded to int8) and
/// activations are written to the other buffer before this step's MMAs, so one barrier per step. int32 partials live
/// across the two steps of a 128-K scale group and are folded into the f32 accumulators at the group end.
/// Same push constants / grid / store contract as matmul_cm9.
#[spirv(compute(threads(256)))]
pub fn matmul_cm10(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 5120], #[spirv(workgroup)] xq: &mut [u32; 5120],
    #[spirv(workgroup)] dws: &mut [f32; 256], #[spirv(workgroup)] dxs: &mut [f32; 256]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok;
    let nst = k >> 6;
    // stage(step): decode weights + copy q8 activations of K step `st` into buffer st & 1, scales into slot (st >> 1) & 1
    macro_rules! stage { ($st:expr) => {{
        let st: u32 = $st; let kc = st * 64; let bo = (st & 1) * 2560; let so2 = ((st >> 1) & 1) * 128;
        unroll!([0, 1], |q| {
            let word = g!(wq, row * wpr + (kc >> 4) + wi * 2 + q);
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, bo + r * 20 + wi * 8 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        let xb = qo + tt * k4 + (kc >> 2) + wi * 8;
        unroll!([0, 1, 2, 3, 4, 5, 6, 7], |v| { s!(xq, bo + r * 20 + wi * 8 + v, if live { g!(au, xb + v) } else { 0 }); });
        if st & 1 == 0 {
            if wi == 0 { s!(dws, so2 + r, g!(wsc, row * bpr + (kc >> 7))); }
            else { s!(dxs, so2 + r, if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 }); }
        }
    }} }
    stage!(0);
    bar();
    let mut st = 0u32;
    while st < nst {
        if st + 1 < nst { stage!(st + 1); }
        let bo = (st & 1) * 2560; let so2 = ((st >> 1) & 1) * 128;
        let fa: u32 = if st == 0 && accum != 0 { 1 } else { 0 };
        let ge: u32 = st & 1;
        let last: u32 = if st + 1 == nst { 1 } else { 0 };
        let okj = |j: u32| -> u32 { if tok0 + wc + j * 16 + 16 <= ntok { 1 } else { 0 } };
        macro_rules! ap { ($i:literal, $ks:literal) => { unsafe { ws.index_unchecked((bo + (wr + $i * 16) * 20 + $ks * 8) as usize) } } }
        macro_rules! bp { ($j:literal, $ks:literal) => { unsafe { xq.index_unchecked((bo + (wc + $j * 16) * 20 + $ks * 8) as usize) } } }
        macro_rules! dwp { ($i:literal) => { unsafe { dws.index_unchecked((so2 + wr + $i * 16) as usize) } } }
        macro_rules! dxp { ($j:literal) => { unsafe { dxs.index_unchecked((so2 + wc + $j * 16) as usize) } } }
        macro_rules! yp { ($i:literal, $j:literal) => { unsafe { a.index_unchecked_mut((yo + (tok0 + wc + $j * 16) * ys + row0 + wr + $i * 16) as usize) } as *mut f32 } }
        unsafe {
            asm!(
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%s8 = OpTypeInt 8 1",
                "%s32 = OpTypeInt 32 1",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%tA = OpTypeCooperativeMatrixKHR %s8 %c3 %c16 %c32 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %s8 %c3 %c32 %c16 %c1",
                "%tI = OpTypeCooperativeMatrixKHR %s32 %c3 %c16 %c16 %c2",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%pC = OpTypePointer Function %tC",
                "%pI = OpTypePointer Function %tI",
                "%z = OpConstantNull %tC",
                "%zI = OpConstantNull %tI",
                "%fa = OpIEqual %bool {fa} %c1",
                "%ge = OpIEqual %bool {ge} %c1",
                "%last = OpIEqual %bool {last} %c1",
                "%v00 = OpVariable %pC Function %z",
                "%u00 = OpVariable %pI Function %zI",
                "%v01 = OpVariable %pC Function %z",
                "%u01 = OpVariable %pI Function %zI",
                "%v02 = OpVariable %pC Function %z",
                "%u02 = OpVariable %pI Function %zI",
                "%v03 = OpVariable %pC Function %z",
                "%u03 = OpVariable %pI Function %zI",
                "%v10 = OpVariable %pC Function %z",
                "%u10 = OpVariable %pI Function %zI",
                "%v11 = OpVariable %pC Function %z",
                "%u11 = OpVariable %pI Function %zI",
                "%v12 = OpVariable %pC Function %z",
                "%u12 = OpVariable %pI Function %zI",
                "%v13 = OpVariable %pC Function %z",
                "%u13 = OpVariable %pI Function %zI",
                "OpBranchConditional %fa %l0 %m0",
                "%l0 = OpLabel",
                "%y00 = OpCooperativeMatrixLoadKHR %tC {y00} %c1 {ys}",
                "OpStore %v00 %y00",
                "%y01 = OpCooperativeMatrixLoadKHR %tC {y01} %c1 {ys}",
                "OpStore %v01 %y01",
                "%y02 = OpCooperativeMatrixLoadKHR %tC {y02} %c1 {ys}",
                "OpStore %v02 %y02",
                "%y03 = OpCooperativeMatrixLoadKHR %tC {y03} %c1 {ys}",
                "OpStore %v03 %y03",
                "%y10 = OpCooperativeMatrixLoadKHR %tC {y10} %c1 {ys}",
                "OpStore %v10 %y10",
                "%y11 = OpCooperativeMatrixLoadKHR %tC {y11} %c1 {ys}",
                "OpStore %v11 %y11",
                "%y12 = OpCooperativeMatrixLoadKHR %tC {y12} %c1 {ys}",
                "OpStore %v12 %y12",
                "%y13 = OpCooperativeMatrixLoadKHR %tC {y13} %c1 {ys}",
                "OpStore %v13 %y13",
                "OpBranch %m0",
                "%m0 = OpLabel",
                "%I00_0 = OpLoad %tI %u00",
                "%I01_0 = OpLoad %tI %u01",
                "%I02_0 = OpLoad %tI %u02",
                "%I03_0 = OpLoad %tI %u03",
                "%I10_0 = OpLoad %tI %u10",
                "%I11_0 = OpLoad %tI %u11",
                "%I12_0 = OpLoad %tI %u12",
                "%I13_0 = OpLoad %tI %u13",
                "%A0_0 = OpCooperativeMatrixLoadKHR %tA {a00} %c0 %c20",
                "%A1_0 = OpCooperativeMatrixLoadKHR %tA {a10} %c0 %c20",
                "%B0_0 = OpCooperativeMatrixLoadKHR %tB {b00} %c1 %c20",
                "%B1_0 = OpCooperativeMatrixLoadKHR %tB {b10} %c1 %c20",
                "%B2_0 = OpCooperativeMatrixLoadKHR %tB {b20} %c1 %c20",
                "%B3_0 = OpCooperativeMatrixLoadKHR %tB {b30} %c1 %c20",
                "%I00_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B0_0 %I00_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I01_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B1_0 %I01_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I02_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B2_0 %I02_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I03_1 = OpCooperativeMatrixMulAddKHR %tI %A0_0 %B3_0 %I03_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I10_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B0_0 %I10_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I11_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B1_0 %I11_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I12_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B2_0 %I12_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I13_1 = OpCooperativeMatrixMulAddKHR %tI %A1_0 %B3_0 %I13_0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%A0_1 = OpCooperativeMatrixLoadKHR %tA {a01} %c0 %c20",
                "%A1_1 = OpCooperativeMatrixLoadKHR %tA {a11} %c0 %c20",
                "%B0_1 = OpCooperativeMatrixLoadKHR %tB {b01} %c1 %c20",
                "%B1_1 = OpCooperativeMatrixLoadKHR %tB {b11} %c1 %c20",
                "%B2_1 = OpCooperativeMatrixLoadKHR %tB {b21} %c1 %c20",
                "%B3_1 = OpCooperativeMatrixLoadKHR %tB {b31} %c1 %c20",
                "%I00_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B0_1 %I00_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I01_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B1_1 %I01_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I02_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B2_1 %I02_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I03_2 = OpCooperativeMatrixMulAddKHR %tI %A0_1 %B3_1 %I03_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I10_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B0_1 %I10_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I11_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B1_1 %I11_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I12_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B2_1 %I12_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "%I13_2 = OpCooperativeMatrixMulAddKHR %tI %A1_1 %B3_1 %I13_1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
                "OpBranchConditional %ge %lg %ln",
                "%ln = OpLabel",
                "OpStore %u00 %I00_2",
                "OpStore %u01 %I01_2",
                "OpStore %u02 %I02_2",
                "OpStore %u03 %I03_2",
                "OpStore %u10 %I10_2",
                "OpStore %u11 %I11_2",
                "OpStore %u12 %I12_2",
                "OpStore %u13 %I13_2",
                "OpBranch %mm",
                "%lg = OpLabel",
                "%SW0 = OpCooperativeMatrixLoadKHR %tC {dw0} %c1 %c0",
                "%SW1 = OpCooperativeMatrixLoadKHR %tC {dw1} %c1 %c0",
                "%SX0 = OpCooperativeMatrixLoadKHR %tC {dx0} %c0 %c0",
                "%SX1 = OpCooperativeMatrixLoadKHR %tC {dx1} %c0 %c0",
                "%SX2 = OpCooperativeMatrixLoadKHR %tC {dx2} %c0 %c0",
                "%SX3 = OpCooperativeMatrixLoadKHR %tC {dx3} %c0 %c0",
                "%F00 = OpConvertSToF %tC %I00_2",
                "%P00 = OpFMul %tC %SW0 %SX0",
                "%G00 = OpFMul %tC %F00 %P00",
                "%O00 = OpLoad %tC %v00",
                "%N00 = OpFAdd %tC %O00 %G00",
                "OpStore %v00 %N00",
                "OpStore %u00 %zI",
                "%F01 = OpConvertSToF %tC %I01_2",
                "%P01 = OpFMul %tC %SW0 %SX1",
                "%G01 = OpFMul %tC %F01 %P01",
                "%O01 = OpLoad %tC %v01",
                "%N01 = OpFAdd %tC %O01 %G01",
                "OpStore %v01 %N01",
                "OpStore %u01 %zI",
                "%F02 = OpConvertSToF %tC %I02_2",
                "%P02 = OpFMul %tC %SW0 %SX2",
                "%G02 = OpFMul %tC %F02 %P02",
                "%O02 = OpLoad %tC %v02",
                "%N02 = OpFAdd %tC %O02 %G02",
                "OpStore %v02 %N02",
                "OpStore %u02 %zI",
                "%F03 = OpConvertSToF %tC %I03_2",
                "%P03 = OpFMul %tC %SW0 %SX3",
                "%G03 = OpFMul %tC %F03 %P03",
                "%O03 = OpLoad %tC %v03",
                "%N03 = OpFAdd %tC %O03 %G03",
                "OpStore %v03 %N03",
                "OpStore %u03 %zI",
                "%F10 = OpConvertSToF %tC %I10_2",
                "%P10 = OpFMul %tC %SW1 %SX0",
                "%G10 = OpFMul %tC %F10 %P10",
                "%O10 = OpLoad %tC %v10",
                "%N10 = OpFAdd %tC %O10 %G10",
                "OpStore %v10 %N10",
                "OpStore %u10 %zI",
                "%F11 = OpConvertSToF %tC %I11_2",
                "%P11 = OpFMul %tC %SW1 %SX1",
                "%G11 = OpFMul %tC %F11 %P11",
                "%O11 = OpLoad %tC %v11",
                "%N11 = OpFAdd %tC %O11 %G11",
                "OpStore %v11 %N11",
                "OpStore %u11 %zI",
                "%F12 = OpConvertSToF %tC %I12_2",
                "%P12 = OpFMul %tC %SW1 %SX2",
                "%G12 = OpFMul %tC %F12 %P12",
                "%O12 = OpLoad %tC %v12",
                "%N12 = OpFAdd %tC %O12 %G12",
                "OpStore %v12 %N12",
                "OpStore %u12 %zI",
                "%F13 = OpConvertSToF %tC %I13_2",
                "%P13 = OpFMul %tC %SW1 %SX3",
                "%G13 = OpFMul %tC %F13 %P13",
                "%O13 = OpLoad %tC %v13",
                "%N13 = OpFAdd %tC %O13 %G13",
                "OpStore %v13 %N13",
                "OpStore %u13 %zI",
                "OpBranchConditional %last %l1 %m1",
                "%l1 = OpLabel",
                "%ok0 = OpIEqual %bool {ok0} %c1",
                "OpBranchConditional %ok0 %s0 %e0",
                "%s0 = OpLabel",
                "OpCooperativeMatrixStoreKHR {y00} %N00 %c1 {ys}",
                "OpCooperativeMatrixStoreKHR {y10} %N10 %c1 {ys}",
                "OpBranch %e0",
                "%e0 = OpLabel",
                "%ok1 = OpIEqual %bool {ok1} %c1",
                "OpBranchConditional %ok1 %s1 %e1",
                "%s1 = OpLabel",
                "OpCooperativeMatrixStoreKHR {y01} %N01 %c1 {ys}",
                "OpCooperativeMatrixStoreKHR {y11} %N11 %c1 {ys}",
                "OpBranch %e1",
                "%e1 = OpLabel",
                "%ok2 = OpIEqual %bool {ok2} %c1",
                "OpBranchConditional %ok2 %s2 %e2",
                "%s2 = OpLabel",
                "OpCooperativeMatrixStoreKHR {y02} %N02 %c1 {ys}",
                "OpCooperativeMatrixStoreKHR {y12} %N12 %c1 {ys}",
                "OpBranch %e2",
                "%e2 = OpLabel",
                "%ok3 = OpIEqual %bool {ok3} %c1",
                "OpBranchConditional %ok3 %s3 %e3",
                "%s3 = OpLabel",
                "OpCooperativeMatrixStoreKHR {y03} %N03 %c1 {ys}",
                "OpCooperativeMatrixStoreKHR {y13} %N13 %c1 {ys}",
                "OpBranch %e3",
                "%e3 = OpLabel",
                "OpBranch %m1",
                "%m1 = OpLabel",
                "OpBranch %mm",
                "%mm = OpLabel",
                fa = in(reg) fa,
                ge = in(reg) ge,
                last = in(reg) last,
                ys = in(reg) ys,
                ok0 = in(reg) okj(0),
                ok1 = in(reg) okj(1),
                ok2 = in(reg) okj(2),
                ok3 = in(reg) okj(3),
                y00 = in(reg) yp!(0,0),
                y01 = in(reg) yp!(0,1),
                y02 = in(reg) yp!(0,2),
                y03 = in(reg) yp!(0,3),
                y10 = in(reg) yp!(1,0),
                y11 = in(reg) yp!(1,1),
                y12 = in(reg) yp!(1,2),
                y13 = in(reg) yp!(1,3),
                a00 = in(reg) ap!(0,0),
                a10 = in(reg) ap!(1,0),
                b00 = in(reg) bp!(0,0),
                b10 = in(reg) bp!(1,0),
                b20 = in(reg) bp!(2,0),
                b30 = in(reg) bp!(3,0),
                a01 = in(reg) ap!(0,1),
                a11 = in(reg) ap!(1,1),
                b01 = in(reg) bp!(0,1),
                b11 = in(reg) bp!(1,1),
                b21 = in(reg) bp!(2,1),
                b31 = in(reg) bp!(3,1),
                dw0 = in(reg) dwp!(0),
                dw1 = in(reg) dwp!(1),
                dx0 = in(reg) dxp!(0),
                dx1 = in(reg) dxp!(1),
                dx2 = in(reg) dxp!(2),
                dx3 = in(reg) dxp!(3)
            );
        }
        bar();
        st += 1;
    }
}

/// NVIDIA fused [rmsnorm] + 1024-block Hadamard + q8 quantization (feeds matmul_cm9 without a quant_x pass).
/// grid (n/1024, ntok), 256 threads. pc: 0 n, 1 io, 2 its, 3 xno (norm output, written when donorm), 4 oo, 5 ots,
/// 6 wo (norm weight), 7 so (signs or NONE), 8 qo, 9 sco, 10 donorm, 11 eps. Norm math/order = rmsnorm mode 0.
#[spirv(compute(threads(256)))]
pub fn nhq(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] v: &mut [f32; 1024], #[spirv(workgroup)] red: &mut [f32; 256]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (n, io, its, xno, oo, ots, wo, so) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (qo, sco, donorm) = (pu(pc, 8), pu(pc, 9), pu(pc, 10)); let eps = pf(pc, 11);
    let (blk, t, l) = (wg.x, wg.y, lid.x);
    let ib = io + t * its;
    let mut sc = 1.0f32;
    if donorm != 0 {
        let mut sm = 0.0f32;
        let mut i = l; while i < n { let x = g!(a, ib + i); sm += x * x; i += 256; }
        s!(red, l, sm); bar();
        unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
        sc = 1.0 / (g!(red, 0) / n as f32 + eps).sqrt();
    }
    unroll!([0, 1, 2, 3], |q| {
        let i = l + 256 * q; let gi = blk * 1024 + i;
        let mut x = g!(a, ib + gi);
        if donorm != 0 { x = x * sc * g!(prm, wo + gi); s!(a, xno + t * n + gi, x); }
        if so != u32::MAX { x *= g!(prm, so + gi); }
        s!(v, i, x);
    });
    bar();
    unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], |sh| {
        let h = 1u32 << sh;
        unroll!([0, 1], |q| {
            let p = l + 256 * q;
            let i = ((p >> sh) << (sh + 1)) + (p & (h - 1)); let j = i + h;
            let x = g!(v, i); let y = g!(v, j); s!(v, i, x + y); s!(v, j, x - y);
        });
        bar();
    });
    // contiguous 4 per thread: warp w owns the 128-group w of this block
    let b0 = l * 4;
    let x0 = g!(v, b0) * 0.03125; let x1 = g!(v, b0 + 1) * 0.03125; let x2 = g!(v, b0 + 2) * 0.03125; let x3 = g!(v, b0 + 3) * 0.03125;
    let ob = oo + t * ots + blk * 1024 + b0;
    s!(a, ob, x0); s!(a, ob + 1, x1); s!(a, ob + 2, x2); s!(a, ob + 3, x3);
    let mut mx = x0.abs().max(x1.abs()).max(x2.abs()).max(x3.abs());
    mx = mx.max(sx(mx, 1)); mx = mx.max(sx(mx, 2)); mx = mx.max(sx(mx, 4)); mx = mx.max(sx(mx, 8)); mx = mx.max(sx(mx, 16));
    let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
    let pb = (l >> 2) * 16 + (l & 3); // K-transposed q8 layout (see quant_x)
    let p0 = g!(v, pb) * 0.03125; let p1 = g!(v, pb + 4) * 0.03125; let p2 = g!(v, pb + 8) * 0.03125; let p3 = g!(v, pb + 12) * 0.03125;
    let q0 = (rnd(p0 * inv) as i32) as u32 & 0xff; let q1 = (rnd(p1 * inv) as i32) as u32 & 0xff;
    let q2 = (rnd(p2 * inv) as i32) as u32 & 0xff; let q3 = (rnd(p3 * inv) as i32) as u32 & 0xff;
    s!(au, qo + t * (n / 4) + blk * 256 + l, q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
    if (l & 31) == 0 { s!(a, sco + t * (n / 128) + blk * 8 + (l >> 5), mx / 127.0); }
}

/// coop-matrix2 workgroup-scope variant of matmul_cm9 (VK_NV_cooperative_matrix2 cooperativeMatrixWorkgroupScope):
/// one 128x128 s32 tile per workgroup (A 128x32, B 32x128 s8), driver-scheduled across the 8 warps.
/// Full 128-token tiles only (host sends the tail to matmul_cm9). Same pc / shared layout as matmul_cm9.
#[spirv(compute(threads(256)))]
pub fn matmul_cm11(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 4608],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 128]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let diag = pu(pc, 8); // timing diagnostics only (host passes 0): bit0 skip global->shared fill, bit1 skip MMA
    let mut kc = 0u32;
    while kc < k {
        if diag & 1 == 0 {
        unroll!([0, 1, 2, 3], |q| {
            let word = g!(wq, row * wpr + (kc >> 4) + wi * 4 + q);
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        let xb = qo + tt * k4 + (kc >> 2) + half * 16;
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |v| {
            s!(xq, r * 36 + half * 16 + v, if live { g!(au, xb + v) } else { 0 });
        });
        if half == 0 { s!(dxs, r, if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 }); }
        }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        if diag & 2 == 0 { unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%c128 = OpConstant %u32 128",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c128 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c128 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c128 %c128 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c128 %c128 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c36",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 %c36",
            "%I0 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c36",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 %c36",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A2 = OpCooperativeMatrixLoadKHR %tA {a2} %c0 %c36",
            "%B2 = OpCooperativeMatrixLoadKHR %tB {b2} %c1 %c36",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A2 %B2 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A3 = OpCooperativeMatrixLoadKHR %tA {a3} %c0 %c36",
            "%B3 = OpCooperativeMatrixLoadKHR %tB {b3} %c1 %c36",
            "%I3 = OpCooperativeMatrixMulAddKHR %tI %A3 %B3 %I2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I3",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ys = in(reg) ys,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((0 * 8) as usize),
            b0 = in(reg) xq.index_unchecked((0 * 8) as usize),
            a1 = in(reg) ws.index_unchecked((1 * 8) as usize),
            b1 = in(reg) xq.index_unchecked((1 * 8) as usize),
            a2 = in(reg) ws.index_unchecked((2 * 8) as usize),
            b2 = in(reg) xq.index_unchecked((2 * 8) as usize),
            a3 = in(reg) ws.index_unchecked((3 * 8) as usize),
            b3 = in(reg) xq.index_unchecked((3 * 8) as usize),
            dw = in(reg) dws.index_unchecked(0),
            dx = in(reg) dxs.index_unchecked(0),
            );
        } }
        bar();
        kc += 128;
    }
}

/// matmul_cm11 with the int8 activation (B) matrices loaded straight from global memory (workgroup-scope
/// OpCooperativeMatrixLoadKHR, column-major, stride k/4 words); only the decoded weights and scales go through shared.
#[spirv(compute(threads(256)))]
pub fn matmul_cm12(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], 
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 128]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let diag = pu(pc, 8); // timing diagnostics only (host passes 0): bit0 skip global->shared fill, bit1 skip MMA
    let mut kc = 0u32;
    while kc < k {
        if diag & 1 == 0 {
        unroll!([0, 1, 2, 3], |q| {
            let word = g!(wq, row * wpr + (kc >> 4) + wi * 4 + q);
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        if half == 0 { s!(dxs, r, if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 }); }
        }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        if diag & 2 == 0 { unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%c128 = OpConstant %u32 128",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c128 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c128 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c128 %c128 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c128 %c128 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c36",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 {k4}",
            "%I0 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c36",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 {k4}",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A2 = OpCooperativeMatrixLoadKHR %tA {a2} %c0 %c36",
            "%B2 = OpCooperativeMatrixLoadKHR %tB {b2} %c1 {k4}",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A2 %B2 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A3 = OpCooperativeMatrixLoadKHR %tA {a3} %c0 %c36",
            "%B3 = OpCooperativeMatrixLoadKHR %tB {b3} %c1 {k4}",
            "%I3 = OpCooperativeMatrixMulAddKHR %tI %A3 %B3 %I2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I3",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ys = in(reg) ys, k4 = in(reg) k4,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((0 * 8) as usize),
            b0 = in(reg) au.index_unchecked((qo + tok0 * k4 + (kc >> 2) + 0 * 8) as usize),
            a1 = in(reg) ws.index_unchecked((1 * 8) as usize),
            b1 = in(reg) au.index_unchecked((qo + tok0 * k4 + (kc >> 2) + 1 * 8) as usize),
            a2 = in(reg) ws.index_unchecked((2 * 8) as usize),
            b2 = in(reg) au.index_unchecked((qo + tok0 * k4 + (kc >> 2) + 2 * 8) as usize),
            a3 = in(reg) ws.index_unchecked((3 * 8) as usize),
            b3 = in(reg) au.index_unchecked((qo + tok0 * k4 + (kc >> 2) + 3 * 8) as usize),
            dw = in(reg) dws.index_unchecked(0),
            dx = in(reg) dxs.index_unchecked(0),
            );
        } }
        bar();
        kc += 128;
    }
}

/// matmul_cm11 with a coalesced, 16-byte-vectorised global->shared fill (requires qo and the weight rows 16-byte aligned).
#[spirv(compute(threads(256)))]
pub fn matmul_cm13(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] av: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wv: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 4608],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 128]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 128;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let diag = pu(pc, 8); // timing diagnostics only (host passes 0): bit0 skip global->shared fill, bit1 skip MMA; needs qo % 4 == 0
    let mut kc = 0u32;
    while kc < k {
        if diag & 1 == 0 {
        // coalesced 16-byte loads: weights one uvec4 (32 codes of row r) per thread; activations 4 uvec4 per thread,
        // 8 consecutive threads cover one token's 128 bytes.
        let wd = g!(wv, (row * wpr + (kc >> 4) + wi * 4) >> 2);
        unroll!([0, 1, 2, 3], |q| {
            let word = if q == 0 { wd.x } else if q == 1 { wd.y } else if q == 2 { wd.z } else { wd.w };
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        unroll!([0, 1, 2, 3], |j| {
            let idx = l + 256 * j; let tk = idx >> 3; let part = idx & 7; let t2 = tok0 + tk;
            let v = if t2 < ntok { g!(av, (qo + t2 * k4 + (kc >> 2) + part * 4) >> 2) } else { UVec4::ZERO };
            let o = tk * 36 + part * 4;
            s!(xq, o, v.x); s!(xq, o + 1, v.y); s!(xq, o + 2, v.z); s!(xq, o + 3, v.w);
        });
        if half == 0 { s!(dxs, r, if live { g!(a, so + tt * bpr + (kc >> 7)) } else { 0.0 }); }
        }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        if diag & 2 == 0 { unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%c128 = OpConstant %u32 128",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c128 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c128 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c128 %c128 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c128 %c128 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c36",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 %c36",
            "%I0 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c36",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 %c36",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A2 = OpCooperativeMatrixLoadKHR %tA {a2} %c0 %c36",
            "%B2 = OpCooperativeMatrixLoadKHR %tB {b2} %c1 %c36",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A2 %B2 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A3 = OpCooperativeMatrixLoadKHR %tA {a3} %c0 %c36",
            "%B3 = OpCooperativeMatrixLoadKHR %tB {b3} %c1 %c36",
            "%I3 = OpCooperativeMatrixMulAddKHR %tI %A3 %B3 %I2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I3",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ys = in(reg) ys,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((0 * 8) as usize),
            b0 = in(reg) xq.index_unchecked((0 * 8) as usize),
            a1 = in(reg) ws.index_unchecked((1 * 8) as usize),
            b1 = in(reg) xq.index_unchecked((1 * 8) as usize),
            a2 = in(reg) ws.index_unchecked((2 * 8) as usize),
            b2 = in(reg) xq.index_unchecked((2 * 8) as usize),
            a3 = in(reg) ws.index_unchecked((3 * 8) as usize),
            b3 = in(reg) xq.index_unchecked((3 * 8) as usize),
            dw = in(reg) dws.index_unchecked(0),
            dx = in(reg) dxs.index_unchecked(0),
            );
        } }
        bar();
        kc += 128;
    }
}

/// matmul_cm13 with a 128-row x 64-token tile (half the accumulators and activation smem) so two workgroups fit per SM.
#[spirv(compute(threads(256)))]
pub fn matmul_cm14(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] av: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wv: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 2304],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 64]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 64;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let diag = pu(pc, 8); // timing diagnostics only (host passes 0): bit0 skip global->shared fill, bit1 skip MMA; needs qo % 4 == 0
    let mut kc = 0u32;
    while kc < k {
        if diag & 1 == 0 {
        // coalesced 16-byte loads: weights one uvec4 (32 codes of row r) per thread; activations 4 uvec4 per thread,
        // 8 consecutive threads cover one token's 128 bytes.
        let wd = g!(wv, (row * wpr + (kc >> 4) + wi * 4) >> 2);
        unroll!([0, 1, 2, 3], |q| {
            let word = if q == 0 { wd.x } else if q == 1 { wd.y } else if q == 2 { wd.z } else { wd.w };
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        unroll!([0, 1], |j| {
            let idx = l + 256 * j; let tk = idx >> 3; let part = idx & 7; let t2 = tok0 + tk;
            let v = if t2 < ntok { g!(av, (qo + t2 * k4 + (kc >> 2) + part * 4) >> 2) } else { UVec4::ZERO };
            let o = tk * 36 + part * 4;
            s!(xq, o, v.x); s!(xq, o + 1, v.y); s!(xq, o + 2, v.z); s!(xq, o + 3, v.w);
        });
        if l < 64 { let t3 = tok0 + l; s!(dxs, l, if t3 < ntok { g!(a, so + t3 * bpr + (kc >> 7)) } else { 0.0 }); }
        }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        if diag & 2 == 0 { unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%c128 = OpConstant %u32 128",
            "%c64 = OpConstant %u32 64",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c128 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c64 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c128 %c64 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c128 %c64 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c36",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 %c36",
            "%I0 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c36",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 %c36",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A2 = OpCooperativeMatrixLoadKHR %tA {a2} %c0 %c36",
            "%B2 = OpCooperativeMatrixLoadKHR %tB {b2} %c1 %c36",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A2 %B2 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A3 = OpCooperativeMatrixLoadKHR %tA {a3} %c0 %c36",
            "%B3 = OpCooperativeMatrixLoadKHR %tB {b3} %c1 %c36",
            "%I3 = OpCooperativeMatrixMulAddKHR %tI %A3 %B3 %I2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I3",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ys = in(reg) ys,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((0 * 8) as usize),
            b0 = in(reg) xq.index_unchecked((0 * 8) as usize),
            a1 = in(reg) ws.index_unchecked((1 * 8) as usize),
            b1 = in(reg) xq.index_unchecked((1 * 8) as usize),
            a2 = in(reg) ws.index_unchecked((2 * 8) as usize),
            b2 = in(reg) xq.index_unchecked((2 * 8) as usize),
            a3 = in(reg) ws.index_unchecked((3 * 8) as usize),
            b3 = in(reg) xq.index_unchecked((3 * 8) as usize),
            dw = in(reg) dws.index_unchecked(0),
            dx = in(reg) dxs.index_unchecked(0),
            );
        } }
        bar();
        kc += 128;
    }
}

/// matmul_cm14 with a 128-row x 32-token tile.
#[spirv(compute(threads(256)))]
pub fn matmul_cm15(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] av: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wv: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 1152],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 32]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 32;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let diag = pu(pc, 8); // timing diagnostics only (host passes 0): bit0 skip global->shared fill, bit1 skip MMA; needs qo % 4 == 0
    let mut kc = 0u32;
    while kc < k {
        if diag & 1 == 0 {
        // coalesced 16-byte loads: weights one uvec4 (32 codes of row r) per thread; activations 4 uvec4 per thread,
        // 8 consecutive threads cover one token's 128 bytes.
        let wd = g!(wv, (row * wpr + (kc >> 4) + wi * 4) >> 2);
        unroll!([0, 1, 2, 3], |q| {
            let word = if q == 0 { wd.x } else if q == 1 { wd.y } else if q == 2 { wd.z } else { wd.w };
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, g!(wsc, row * bpr + (kc >> 7))); }
        unroll!([0], |j| {
            let idx = l + 256 * j; let tk = idx >> 3; let part = idx & 7; let t2 = tok0 + tk;
            let v = if t2 < ntok { g!(av, (qo + t2 * k4 + (kc >> 2) + part * 4) >> 2) } else { UVec4::ZERO };
            let o = tk * 36 + part * 4;
            s!(xq, o, v.x); s!(xq, o + 1, v.y); s!(xq, o + 2, v.z); s!(xq, o + 3, v.w);
        });
        if l < 32 { let t3 = tok0 + l; s!(dxs, l, if t3 < ntok { g!(a, so + t3 * bpr + (kc >> 7)) } else { 0.0 }); }
        }
        bar();
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        if diag & 2 == 0 { unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%c128 = OpConstant %u32 128",
            "%c64 = OpConstant %u32 64",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c128 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c32 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c128 %c32 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c128 %c32 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c36",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 %c36",
            "%I0 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c36",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 %c36",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A2 = OpCooperativeMatrixLoadKHR %tA {a2} %c0 %c36",
            "%B2 = OpCooperativeMatrixLoadKHR %tB {b2} %c1 %c36",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A2 %B2 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A3 = OpCooperativeMatrixLoadKHR %tA {a3} %c0 %c36",
            "%B3 = OpCooperativeMatrixLoadKHR %tB {b3} %c1 %c36",
            "%I3 = OpCooperativeMatrixMulAddKHR %tI %A3 %B3 %I2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I3",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ys = in(reg) ys,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((0 * 8) as usize),
            b0 = in(reg) xq.index_unchecked((0 * 8) as usize),
            a1 = in(reg) ws.index_unchecked((1 * 8) as usize),
            b1 = in(reg) xq.index_unchecked((1 * 8) as usize),
            a2 = in(reg) ws.index_unchecked((2 * 8) as usize),
            b2 = in(reg) xq.index_unchecked((2 * 8) as usize),
            a3 = in(reg) ws.index_unchecked((3 * 8) as usize),
            b3 = in(reg) xq.index_unchecked((3 * 8) as usize),
            dw = in(reg) dws.index_unchecked(0),
            dx = in(reg) dxs.index_unchecked(0),
            );
        } }
        bar();
        kc += 128;
    }
}

/// matmul_cm14 with register-staged prefetch of the next K chunk (global loads overlap the MMAs).
#[spirv(compute(threads(256)))]
pub fn matmul_cm16(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] av: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wv: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 4608], #[spirv(workgroup)] xq: &mut [u32; 2304],
    #[spirv(workgroup)] dws: &mut [f32; 128], #[spirv(workgroup)] dxs: &mut [f32; 64]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 128; let tok0 = wg.y * 64;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let diag = pu(pc, 8); // timing diagnostics only (host passes 0): bit0 skip global->shared fill, bit1 skip MMA; needs qo % 4 == 0
    // register-staged prefetch: chunk kc+128's global loads are issued before chunk kc's MMAs.
    macro_rules! ld { ($kc:expr) => {{ let kc: u32 = $kc;
        let wd = g!(wv, (row * wpr + (kc >> 4) + wi * 4) >> 2);
        let dw = if wi == 0 { g!(wsc, row * bpr + (kc >> 7)) } else { 0.0 };
        let i0 = l; let t0 = tok0 + (i0 >> 3); let i1 = l + 256; let t1 = tok0 + (i1 >> 3);
        let v0 = if t0 < ntok { g!(av, (qo + t0 * k4 + (kc >> 2) + (i0 & 7) * 4) >> 2) } else { UVec4::ZERO };
        let v1 = if t1 < ntok { g!(av, (qo + t1 * k4 + (kc >> 2) + (i1 & 7) * 4) >> 2) } else { UVec4::ZERO };
        let t3 = tok0 + l; let dx = if l < 64 && t3 < ntok { g!(a, so + t3 * bpr + (kc >> 7)) } else { 0.0 };
        (wd, dw, v0, v1, dx)
    }} }
    let mut st = ld!(0);
    let mut kc = 0u32;
    while kc < k {
        let (wd, dw, v0, v1, dx) = st;
        unroll!([0, 1, 2, 3], |q| {
            let word = if q == 0 { wd.x } else if q == 1 { wd.y } else if q == 2 { wd.z } else { wd.w };
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, dw); }
        { let o = (l >> 3) * 36 + (l & 7) * 4; s!(xq, o, v0.x); s!(xq, o + 1, v0.y); s!(xq, o + 2, v0.z); s!(xq, o + 3, v0.w); }
        { let i1 = l + 256; let o = (i1 >> 3) * 36 + (i1 & 7) * 4; s!(xq, o, v1.x); s!(xq, o + 1, v1.y); s!(xq, o + 2, v1.z); s!(xq, o + 3, v1.w); }
        if l < 64 { s!(dxs, l, dx); }
        bar();
        if kc + 128 < k { st = ld!(kc + 128); }
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        { unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%c128 = OpConstant %u32 128",
            "%c64 = OpConstant %u32 64",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c128 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c64 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c128 %c64 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c128 %c64 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c36",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 %c36",
            "%I0 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c36",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 %c36",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A2 = OpCooperativeMatrixLoadKHR %tA {a2} %c0 %c36",
            "%B2 = OpCooperativeMatrixLoadKHR %tB {b2} %c1 %c36",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A2 %B2 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A3 = OpCooperativeMatrixLoadKHR %tA {a3} %c0 %c36",
            "%B3 = OpCooperativeMatrixLoadKHR %tB {b3} %c1 %c36",
            "%I3 = OpCooperativeMatrixMulAddKHR %tI %A3 %B3 %I2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I3",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ys = in(reg) ys,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((0 * 8) as usize),
            b0 = in(reg) xq.index_unchecked((0 * 8) as usize),
            a1 = in(reg) ws.index_unchecked((1 * 8) as usize),
            b1 = in(reg) xq.index_unchecked((1 * 8) as usize),
            a2 = in(reg) ws.index_unchecked((2 * 8) as usize),
            b2 = in(reg) xq.index_unchecked((2 * 8) as usize),
            a3 = in(reg) ws.index_unchecked((3 * 8) as usize),
            b3 = in(reg) xq.index_unchecked((3 * 8) as usize),
            dw = in(reg) dws.index_unchecked(0),
            dx = in(reg) dxs.index_unchecked(0),
            );
        } }
        bar();
        kc += 128;
    }
}

/// matmul_cm16 with a 64-row x 64-token tile and 128 threads (3 workgroups per SM).
#[spirv(compute(threads(128)))]
pub fn matmul_cm17(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] av: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wv: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 2304], #[spirv(workgroup)] xq: &mut [u32; 2304],
    #[spirv(workgroup)] dws: &mut [f32; 64], #[spirv(workgroup)] dxs: &mut [f32; 64]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let wr = (w & 3) * 32; let wc = (w >> 2) * 64;
    let row0 = wg.x * 64; let tok0 = wg.y * 64;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let r = l >> 1; let wi = l & 1; let row = (row0 + r).min(m - 1);
    let tt = tok0 + r; let live = tt < ntok; let half = l & 1;
    let diag = pu(pc, 8); // timing diagnostics only (host passes 0): bit0 skip global->shared fill, bit1 skip MMA; needs qo % 4 == 0
    // register-staged prefetch: chunk kc+128's global loads are issued before chunk kc's MMAs.
    macro_rules! ld { ($kc:expr) => {{ let kc: u32 = $kc;
        let wd = g!(wv, (row * wpr + (kc >> 4) + wi * 4) >> 2);
        let dw = if wi == 0 { g!(wsc, row * bpr + (kc >> 7)) } else { 0.0 };
        macro_rules! lv { ($i:expr) => {{ let ii: u32 = $i; let tq = tok0 + (ii >> 3); if tq < ntok { g!(av, (qo + tq * k4 + (kc >> 2) + (ii & 7) * 4) >> 2) } else { UVec4::ZERO } }} }
        let v0 = lv!(l); let v1 = lv!(l + 128); let v2 = lv!(l + 256); let v3 = lv!(l + 384);
        let t3 = tok0 + l; let dx = if l < 64 && t3 < ntok { g!(a, so + t3 * bpr + (kc >> 7)) } else { 0.0 };
        (wd, dw, v0, v1, v2, v3, dx)
    }} }
    let mut st = ld!(0);
    let mut kc = 0u32;
    while kc < k {
        let (wd, dw, v0, v1, v2, v3, dx) = st;
        unroll!([0, 1, 2, 3], |q| {
            let word = if q == 0 { wd.x } else if q == 1 { wd.y } else if q == 2 { wd.z } else { wd.w };
            unroll!([0, 1, 2, 3], |b| {
                let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                s!(ws, r * 36 + wi * 16 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
            });
        });
        if wi == 0 { s!(dws, r, dw); }
        macro_rules! sv { ($i:expr, $v:expr) => {{ let ii: u32 = $i; let v: UVec4 = $v; let o = (ii >> 3) * 36 + (ii & 7) * 4; s!(xq, o, v.x); s!(xq, o + 1, v.y); s!(xq, o + 2, v.z); s!(xq, o + 3, v.w); }} }
        sv!(l, v0); sv!(l + 128, v1); sv!(l + 256, v2); sv!(l + 384, v3);
        if l < 64 { s!(dxs, l, dx); }
        bar();
        if kc + 128 < k { st = ld!(kc + 128); }
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let last: u32 = if kc + 128 >= k { 1 } else { 0 };
        { unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c32 = OpConstant %u32 32",
            "%c36 = OpConstant %u32 36",
            "%c128 = OpConstant %u32 128",
            "%c64 = OpConstant %u32 64",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c64 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c64 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c64 %c64 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c64 %c64 %c2",
            "%pC = OpTypePointer Function %tC",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%v = OpVariable %pC Function %z",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c36",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 %c36",
            "%I0 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %zI MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c36",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 %c36",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A2 = OpCooperativeMatrixLoadKHR %tA {a2} %c0 %c36",
            "%B2 = OpCooperativeMatrixLoadKHR %tB {b2} %c1 %c36",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A2 %B2 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%A3 = OpCooperativeMatrixLoadKHR %tA {a3} %c0 %c36",
            "%B3 = OpCooperativeMatrixLoadKHR %tB {b3} %c1 %c36",
            "%I3 = OpCooperativeMatrixMulAddKHR %tI %A3 %B3 %I2 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I3",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ys = in(reg) ys,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((0 * 8) as usize),
            b0 = in(reg) xq.index_unchecked((0 * 8) as usize),
            a1 = in(reg) ws.index_unchecked((1 * 8) as usize),
            b1 = in(reg) xq.index_unchecked((1 * 8) as usize),
            a2 = in(reg) ws.index_unchecked((2 * 8) as usize),
            b2 = in(reg) xq.index_unchecked((2 * 8) as usize),
            a3 = in(reg) ws.index_unchecked((3 * 8) as usize),
            b3 = in(reg) xq.index_unchecked((3 * 8) as usize),
            dw = in(reg) dws.index_unchecked(0),
            dx = in(reg) dxs.index_unchecked(0),
            );
        } }
        bar();
        kc += 128;
    }
}

/// Two-stage shared-memory pipeline, BK=64: chunk kc+64 is fetched to registers and written to the other smem stage
/// while chunk kc runs on the tensor cores (one barrier per 64-K step). int32 partials span the 128-K scale group,
/// the f32 epilogue runs every second step. 128-row x 64-token workgroup-scope tile, 256 threads, ~31 KB smem.
/// Same pc as matmul_cm9 (+ pc 8 unused).
#[spirv(compute(threads(256)))]
pub fn matmul_cm18(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] av: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wv: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] ws: &mut [u32; 5120], #[spirv(workgroup)] xq: &mut [u32; 2560],
    #[spirv(workgroup)] dws: &mut [f32; 256], #[spirv(workgroup)] dxs: &mut [f32; 128]) {
    let (m, k, ntok, qo, so, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x;
    let row0 = wg.x * 128; let tok0 = wg.y * 64;
    let wpr = k >> 4; let bpr = k >> 7; let k4 = k >> 2;
    let row = (row0 + (l & 127)).min(m - 1);
    let tk = l >> 2; let part = l & 3; let tq = tok0 + tk;
    macro_rules! ld { ($kc:expr) => {{ let kc: u32 = $kc;
        let wd = if l < 128 { g!(wv, (row * wpr + (kc >> 4)) >> 2) } else { UVec4::ZERO };
        let dw = if l < 128 { g!(wsc, row * bpr + (kc >> 7)) } else { 0.0 };
        let xv = if tq < ntok { g!(av, (qo + tq * k4 + (kc >> 2) + part * 4) >> 2) } else { UVec4::ZERO };
        let t3 = tok0 + (l & 63); let dx = if l >= 128 && l < 192 && t3 < ntok { g!(a, so + t3 * bpr + (kc >> 7)) } else { 0.0 };
        (wd, dw, xv, dx)
    }} }
    macro_rules! put { ($st:expr, $v:expr) => {{ let st: u32 = $st; let (wd, dw, xv, dx) = $v;
        if l < 128 {
            unroll!([0, 1, 2, 3], |q| {
                let word: u32 = if q == 0 { wd.x } else if q == 1 { wd.y } else if q == 2 { wd.z } else { wd.w };
                unroll!([0, 1, 2, 3], |b| {
                    let sp = (word >> (2 * b)) & 0x0303_0303; // codes b, b+4, b+8, b+12 (K-transposed 16-groups, matches q8 layout)
                    s!(ws, st * 2560 + l * 20 + q * 4 + b, ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080);
                });
            });
            s!(dws, st * 128 + l, dw);
        } else if l < 192 { s!(dxs, st * 64 + (l - 128), dx); }
        let o = st * 1280 + tk * 20 + part * 4; s!(xq, o, xv.x); s!(xq, o + 1, xv.y); s!(xq, o + 2, xv.z); s!(xq, o + 3, xv.w);
    }} }
    let v0 = ld!(0); put!(0, v0);
    bar();
    let mut kc = 0u32;
    while kc < k {
        let st = (kc >> 6) & 1;
        let more = kc + 64 < k;
        let nx = if more { ld!(kc + 64) } else { (UVec4::ZERO, 0.0, UVec4::ZERO, 0.0) };
        let fa: u32 = if kc == 0 && accum != 0 { 1 } else { 0 };
        let ep: u32 = if (kc & 64) != 0 { 1 } else { 0 };
        let last: u32 = if kc + 64 >= k { 1 } else { 0 };
        unsafe {
            asm!(
            "%f32 = OpTypeFloat 32",
            "%u32 = OpTypeInt 32 0",
            "%s8 = OpTypeInt 8 1",
            "%s32 = OpTypeInt 32 1",
            "%bool = OpTypeBool",
            "%c0 = OpConstant %u32 0",
            "%c1 = OpConstant %u32 1",
            "%c2 = OpConstant %u32 2",
            "%c20 = OpConstant %u32 20",
            "%c32 = OpConstant %u32 32",
            "%c64 = OpConstant %u32 64",
            "%c128 = OpConstant %u32 128",
            "%tA = OpTypeCooperativeMatrixKHR %s8 %c2 %c128 %c32 %c0",
            "%tB = OpTypeCooperativeMatrixKHR %s8 %c2 %c32 %c64 %c1",
            "%tI = OpTypeCooperativeMatrixKHR %s32 %c2 %c128 %c64 %c2",
            "%tC = OpTypeCooperativeMatrixKHR %f32 %c2 %c128 %c64 %c2",
            "%pC = OpTypePointer Function %tC",
            "%pI = OpTypePointer Function %tI",
            "%z = OpConstantNull %tC",
            "%zI = OpConstantNull %tI",
            "%fa = OpIEqual %bool {fa} %c1",
            "%last = OpIEqual %bool {last} %c1",
            "%ep = OpIEqual %bool {ep} %c1",
            "%v = OpVariable %pC Function %z",
            "%vi = OpVariable %pI Function %zI",
            "OpBranchConditional %fa %l0 %m0",
            "%l0 = OpLabel",
            "%y0 = OpCooperativeMatrixLoadKHR %tC {y} %c1 {ys}",
            "OpStore %v %y0",
            "OpBranch %m0",
            "%m0 = OpLabel",
            "%A0 = OpCooperativeMatrixLoadKHR %tA {a0} %c0 %c20",
            "%B0 = OpCooperativeMatrixLoadKHR %tB {b0} %c1 %c20",
            "%A1 = OpCooperativeMatrixLoadKHR %tA {a1} %c0 %c20",
            "%B1 = OpCooperativeMatrixLoadKHR %tB {b1} %c1 %c20",
            "%I0 = OpLoad %tI %vi",
            "%I1 = OpCooperativeMatrixMulAddKHR %tI %A0 %B0 %I0 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "%I2 = OpCooperativeMatrixMulAddKHR %tI %A1 %B1 %I1 MATRIX_A_SIGNED_COMPONENTS_KHR|MATRIX_B_SIGNED_COMPONENTS_KHR|MATRIX_C_SIGNED_COMPONENTS_KHR|MATRIX_RESULT_SIGNED_COMPONENTS_KHR",
            "OpBranchConditional %ep %l2 %l3",
            "%l2 = OpLabel",
            "%SW = OpCooperativeMatrixLoadKHR %tC {dw} %c1 %c0",
            "%SX = OpCooperativeMatrixLoadKHR %tC {dx} %c0 %c0",
            "%F = OpConvertSToF %tC %I2",
            "%P = OpFMul %tC %SW %SX",
            "%G = OpFMul %tC %F %P",
            "%O = OpLoad %tC %v",
            "%N = OpFAdd %tC %O %G",
            "OpStore %v %N",
            "OpStore %vi %zI",
            "OpBranchConditional %last %l1 %m1",
            "%l1 = OpLabel",
            "OpCooperativeMatrixStoreKHR {y} %N %c1 {ys}",
            "OpBranch %m1",
            "%m1 = OpLabel",
            "OpBranch %m2",
            "%l3 = OpLabel",
            "OpStore %vi %I2",
            "OpBranch %m2",
            "%m2 = OpLabel",
            fa = in(reg) fa, last = in(reg) last, ep = in(reg) ep, ys = in(reg) ys,
            y = in(reg) a.index_unchecked_mut((yo + tok0 * ys + row0) as usize) as *mut f32,
            a0 = in(reg) ws.index_unchecked((st * 2560) as usize),
            a1 = in(reg) ws.index_unchecked((st * 2560 + 8) as usize),
            b0 = in(reg) xq.index_unchecked((st * 1280) as usize),
            b1 = in(reg) xq.index_unchecked((st * 1280 + 8) as usize),
            dw = in(reg) dws.index_unchecked((st * 128) as usize),
            dx = in(reg) dxs.index_unchecked((st * 64) as usize),
            );
        }
        if more { put!(st ^ 1, nx); }
        bar();
        kc += 64;
    }
}

/// ssm_beta and ssm_alpha projections in one pass (replaces two matvec_f32t): 2*nv <= 96 rows x 64 tokens per 256-thread
/// workgroup, each thread 6 rows x 4 tokens; activations read once per 64 tokens. Per-output summation order is
/// identical to matvec_f32t (sequential over K), so results are bit-identical.
/// pc: 0 nv, 1 k, 2 ntok, 3 xo, 4 xs, 5 yb, 6 ya, 7 ys, 8 wb, 9 wa. k % 64 == 0. Grid (1, ntok/64 rounded up).
#[spirv(compute(threads(256)))]
pub fn matvec_ab(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] wsh: &mut [f32; 6240], #[spirv(workgroup)] xsh: &mut [f32; 4160]) {
    let (nv, k, ntok, xo, xs, yb, ya, ys) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (wb, wa) = (pu(pc, 8), pu(pc, 9));
    let rows = 2 * nv;
    let l = lid.x; let rr = l & 15; let tt = l >> 4;
    let tok0 = wg.y * 64;
    let mut acc = [0.0f32; 24];
    let mut kc = 0u32;
    while kc < k {
        let mut q = 0u32;
        while q < 24 {
            let idx = q * 256 + l; let r = idx >> 6; let c = idx & 63;
            let rc = r.min(rows - 1);
            let src = if rc < nv { wb + rc * k } else { wa + (rc - nv) * k };
            s!(wsh, r * 65 + c, g!(prm, src + kc + c));
            q += 1;
        }
        unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], |q| {
            let idx = q * 256 + l; let r = idx >> 6; let c = idx & 63;
            let tok = (tok0 + r).min(ntok - 1);
            s!(xsh, r * 65 + c, g!(a, xo + tok * xs + kc + c));
        });
        bar();
        let mut c = 0u32;
        while c < 64 {
            let x0 = g!(xsh, tt * 65 + c); let x1 = g!(xsh, (tt + 16) * 65 + c);
            let x2 = g!(xsh, (tt + 32) * 65 + c); let x3 = g!(xsh, (tt + 48) * 65 + c);
            unroll!([0, 1, 2, 3, 4, 5], |i| {
                let w = g!(wsh, (rr + 16 * i) * 65 + c);
                acc[(i * 4) as usize] += w * x0; acc[(i * 4 + 1) as usize] += w * x1; acc[(i * 4 + 2) as usize] += w * x2; acc[(i * 4 + 3) as usize] += w * x3;
            });
            c += 1;
        }
        bar();
        kc += 64;
    }
    unroll!([0, 1, 2, 3, 4, 5], |i| {
        let r = rr + 16 * i;
        unroll!([0, 1, 2, 3], |j| {
            let t = tok0 + tt + 16 * j;
            if r < rows && t < ntok {
                let v = acc[(i * 4 + j) as usize];
                if r < nv { s!(a, yb + t * ys + r, v); } else { s!(a, ya + t * ys + r - nv, v); }
            }
        });
    });
}

/// attn_fa1 without the 33 KB shared Q tile: Q is converted to f16 once into the head's O scratch slot and its
/// coop-matrix fragments are loaded from there (L1/L2) each key block, so ~2.5x more warps fit per SM.
#[spirv(compute(threads(128)))]
pub fn attn_fa2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] aw: &mut [u32],
    #[spirv(workgroup)] ss: &mut [f32; 2048], #[spirv(workgroup)] ps: &mut [u32; 1280],
    #[spirv(workgroup)] red: &mut [f32; 128], #[spirv(workgroup)] alph: &mut [f32; 64]) {
    let (qo, qts, kco, vco, kvd, pos0) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5));
    let (nh, nkv, scr, nt, oo, ots) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12));
    let scale = pf(pc, 13);
    let h = wg.x; let qb = wg.y * 64; let l = lid.x; let w = l >> 5; let ln = l & 31;
    let kh = h / (nh / nkv);
    {
        let r = l >> 1; let half = l & 1; let t = (qb + r).min(nt - 1);
        let base = (qo + t * qts + h * 512) / 4 + half * 32;
        // f16 Q goes to this head's (still unused) O scratch slot for the same row; only this workgroup touches it
        let qd = scr + (qb + r) * (nh * 256) + h * 256 + half * 64;
        let mut v = 0u32;
        while v < 32 {
            let x = g!(a4, base + v);
            s!(aw, qd + v * 2, vec2_to_f16x2(Vec2::new(x.x, x.y))); s!(aw, qd + v * 2 + 1, vec2_to_f16x2(Vec2::new(x.z, x.w)));
            v += 1;
        }
    }
    bar();
    let qlast = (qb + 64).min(nt) - 1; let nblk = (pos0 + qlast + 1 + 31) / 32;
    let r = ln >> 1; let ch = (ln & 1) * 16; let qrow = w * 16 + r; let qpos = pos0 + qb + qrow;
    let sb = w * 512; let pb = w * 320; let ks = kvd / 2; let os = nh * 256;
    macro_rules! qp { ($kk:literal) => { unsafe { au.index_unchecked((scr + (qb + w * 16) * os + h * 256 + $kk * 8) as usize) } } }
    macro_rules! sp { ($j:literal) => { unsafe { ss.index_unchecked_mut((sb + $j * 16) as usize) } as *mut f32 } }
    macro_rules! pp { ($kk:literal) => { unsafe { ps.index_unchecked((pb + $kk * 8) as usize) } } }
    macro_rules! op { ($d:literal) => { unsafe { a.index_unchecked_mut((scr + (qb + w * 16) * os + h * 256 + $d * 16) as usize) } as *mut f32 } }
    let mut m = -1e30f32; let mut sum = 0.0f32;
    let mut b = 0u32;
    while b < nblk {
        let key0 = b * 32;
        macro_rules! kp { ($j:literal, $kk:literal) => { unsafe { au.index_unchecked((kco + ((key0 + $j * 16) * kvd + kh * 256) / 2 + $kk * 8) as usize) } } }
        unsafe {
            asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%A0 = OpCooperativeMatrixLoadKHR %tA {q0} %c0 {qst}",
                "%B0_0 = OpCooperativeMatrixLoadKHR %tB {k00} %c1 {ks}",
                "%S0_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B0_0 %z",
                "%B1_0 = OpCooperativeMatrixLoadKHR %tB {k10} %c1 {ks}",
                "%S1_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B1_0 %z",
                "%A1 = OpCooperativeMatrixLoadKHR %tA {q1} %c0 {qst}",
                "%B0_1 = OpCooperativeMatrixLoadKHR %tB {k01} %c1 {ks}",
                "%S0_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B0_1 %S0_1",
                "%B1_1 = OpCooperativeMatrixLoadKHR %tB {k11} %c1 {ks}",
                "%S1_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B1_1 %S1_1",
                "%A2 = OpCooperativeMatrixLoadKHR %tA {q2} %c0 {qst}",
                "%B0_2 = OpCooperativeMatrixLoadKHR %tB {k02} %c1 {ks}",
                "%S0_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B0_2 %S0_2",
                "%B1_2 = OpCooperativeMatrixLoadKHR %tB {k12} %c1 {ks}",
                "%S1_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B1_2 %S1_2",
                "%A3 = OpCooperativeMatrixLoadKHR %tA {q3} %c0 {qst}",
                "%B0_3 = OpCooperativeMatrixLoadKHR %tB {k03} %c1 {ks}",
                "%S0_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B0_3 %S0_3",
                "%B1_3 = OpCooperativeMatrixLoadKHR %tB {k13} %c1 {ks}",
                "%S1_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B1_3 %S1_3",
                "%A4 = OpCooperativeMatrixLoadKHR %tA {q4} %c0 {qst}",
                "%B0_4 = OpCooperativeMatrixLoadKHR %tB {k04} %c1 {ks}",
                "%S0_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B0_4 %S0_4",
                "%B1_4 = OpCooperativeMatrixLoadKHR %tB {k14} %c1 {ks}",
                "%S1_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B1_4 %S1_4",
                "%A5 = OpCooperativeMatrixLoadKHR %tA {q5} %c0 {qst}",
                "%B0_5 = OpCooperativeMatrixLoadKHR %tB {k05} %c1 {ks}",
                "%S0_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B0_5 %S0_5",
                "%B1_5 = OpCooperativeMatrixLoadKHR %tB {k15} %c1 {ks}",
                "%S1_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B1_5 %S1_5",
                "%A6 = OpCooperativeMatrixLoadKHR %tA {q6} %c0 {qst}",
                "%B0_6 = OpCooperativeMatrixLoadKHR %tB {k06} %c1 {ks}",
                "%S0_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B0_6 %S0_6",
                "%B1_6 = OpCooperativeMatrixLoadKHR %tB {k16} %c1 {ks}",
                "%S1_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B1_6 %S1_6",
                "%A7 = OpCooperativeMatrixLoadKHR %tA {q7} %c0 {qst}",
                "%B0_7 = OpCooperativeMatrixLoadKHR %tB {k07} %c1 {ks}",
                "%S0_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B0_7 %S0_7",
                "%B1_7 = OpCooperativeMatrixLoadKHR %tB {k17} %c1 {ks}",
                "%S1_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B1_7 %S1_7",
                "%A8 = OpCooperativeMatrixLoadKHR %tA {q8} %c0 {qst}",
                "%B0_8 = OpCooperativeMatrixLoadKHR %tB {k08} %c1 {ks}",
                "%S0_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B0_8 %S0_8",
                "%B1_8 = OpCooperativeMatrixLoadKHR %tB {k18} %c1 {ks}",
                "%S1_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B1_8 %S1_8",
                "%A9 = OpCooperativeMatrixLoadKHR %tA {q9} %c0 {qst}",
                "%B0_9 = OpCooperativeMatrixLoadKHR %tB {k09} %c1 {ks}",
                "%S0_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B0_9 %S0_9",
                "%B1_9 = OpCooperativeMatrixLoadKHR %tB {k19} %c1 {ks}",
                "%S1_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B1_9 %S1_9",
                "%A10 = OpCooperativeMatrixLoadKHR %tA {q10} %c0 {qst}",
                "%B0_10 = OpCooperativeMatrixLoadKHR %tB {k010} %c1 {ks}",
                "%S0_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B0_10 %S0_10",
                "%B1_10 = OpCooperativeMatrixLoadKHR %tB {k110} %c1 {ks}",
                "%S1_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B1_10 %S1_10",
                "%A11 = OpCooperativeMatrixLoadKHR %tA {q11} %c0 {qst}",
                "%B0_11 = OpCooperativeMatrixLoadKHR %tB {k011} %c1 {ks}",
                "%S0_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B0_11 %S0_11",
                "%B1_11 = OpCooperativeMatrixLoadKHR %tB {k111} %c1 {ks}",
                "%S1_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B1_11 %S1_11",
                "%A12 = OpCooperativeMatrixLoadKHR %tA {q12} %c0 {qst}",
                "%B0_12 = OpCooperativeMatrixLoadKHR %tB {k012} %c1 {ks}",
                "%S0_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B0_12 %S0_12",
                "%B1_12 = OpCooperativeMatrixLoadKHR %tB {k112} %c1 {ks}",
                "%S1_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B1_12 %S1_12",
                "%A13 = OpCooperativeMatrixLoadKHR %tA {q13} %c0 {qst}",
                "%B0_13 = OpCooperativeMatrixLoadKHR %tB {k013} %c1 {ks}",
                "%S0_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B0_13 %S0_13",
                "%B1_13 = OpCooperativeMatrixLoadKHR %tB {k113} %c1 {ks}",
                "%S1_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B1_13 %S1_13",
                "%A14 = OpCooperativeMatrixLoadKHR %tA {q14} %c0 {qst}",
                "%B0_14 = OpCooperativeMatrixLoadKHR %tB {k014} %c1 {ks}",
                "%S0_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B0_14 %S0_14",
                "%B1_14 = OpCooperativeMatrixLoadKHR %tB {k114} %c1 {ks}",
                "%S1_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B1_14 %S1_14",
                "%A15 = OpCooperativeMatrixLoadKHR %tA {q15} %c0 {qst}",
                "%B0_15 = OpCooperativeMatrixLoadKHR %tB {k015} %c1 {ks}",
                "%S0_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B0_15 %S0_15",
                "%B1_15 = OpCooperativeMatrixLoadKHR %tB {k115} %c1 {ks}",
                "%S1_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B1_15 %S1_15",
                "OpCooperativeMatrixStoreKHR {s0} %S0_16 %c0 %c32",
                "OpCooperativeMatrixStoreKHR {s1} %S1_16 %c0 %c32",
                ks = in(reg) ks,
                qst = in(reg) os,
                q0 = in(reg) qp!(0),
                q1 = in(reg) qp!(1),
                q2 = in(reg) qp!(2),
                q3 = in(reg) qp!(3),
                q4 = in(reg) qp!(4),
                q5 = in(reg) qp!(5),
                q6 = in(reg) qp!(6),
                q7 = in(reg) qp!(7),
                q8 = in(reg) qp!(8),
                q9 = in(reg) qp!(9),
                q10 = in(reg) qp!(10),
                q11 = in(reg) qp!(11),
                q12 = in(reg) qp!(12),
                q13 = in(reg) qp!(13),
                q14 = in(reg) qp!(14),
                q15 = in(reg) qp!(15),
                k00 = in(reg) kp!(0,0),
                k01 = in(reg) kp!(0,1),
                k02 = in(reg) kp!(0,2),
                k03 = in(reg) kp!(0,3),
                k04 = in(reg) kp!(0,4),
                k05 = in(reg) kp!(0,5),
                k06 = in(reg) kp!(0,6),
                k07 = in(reg) kp!(0,7),
                k08 = in(reg) kp!(0,8),
                k09 = in(reg) kp!(0,9),
                k010 = in(reg) kp!(0,10),
                k011 = in(reg) kp!(0,11),
                k012 = in(reg) kp!(0,12),
                k013 = in(reg) kp!(0,13),
                k014 = in(reg) kp!(0,14),
                k015 = in(reg) kp!(0,15),
                k10 = in(reg) kp!(1,0),
                k11 = in(reg) kp!(1,1),
                k12 = in(reg) kp!(1,2),
                k13 = in(reg) kp!(1,3),
                k14 = in(reg) kp!(1,4),
                k15 = in(reg) kp!(1,5),
                k16 = in(reg) kp!(1,6),
                k17 = in(reg) kp!(1,7),
                k18 = in(reg) kp!(1,8),
                k19 = in(reg) kp!(1,9),
                k110 = in(reg) kp!(1,10),
                k111 = in(reg) kp!(1,11),
                k112 = in(reg) kp!(1,12),
                k113 = in(reg) kp!(1,13),
                k114 = in(reg) kp!(1,14),
                k115 = in(reg) kp!(1,15),
                s0 = in(reg) sp!(0),
                s1 = in(reg) sp!(1)
            );
        }
        bar();
        let mut lm = -1e30f32;
        let mut j = 0u32;
        while j < 16 { if key0 + ch + j <= qpos { lm = lm.max(g!(ss, sb + r * 32 + ch + j) * scale); } j += 1; }
        s!(red, l, lm); bar();
        let mn = m.max(lm).max(g!(red, l ^ 1));
        let alpha = (m - mn).exp(); m = mn; sum *= alpha;
        j = 0;
        while j < 16 {
            let p0 = key0 + ch + j;
            let e0 = if p0 <= qpos { (g!(ss, sb + r * 32 + ch + j) * scale - m).exp() } else { 0.0 };
            let e1 = if p0 + 1 <= qpos { (g!(ss, sb + r * 32 + ch + j + 1) * scale - m).exp() } else { 0.0 };
            sum += e0 + e1;
            s!(ps, pb + r * 20 + (ch + j) / 2, vec2_to_f16x2(Vec2::new(e0, e1)));
            j += 2;
        }
        if ln & 1 == 0 { s!(alph, w * 16 + r, alpha); }
        bar();
        let last: u32 = if b + 1 == nblk { 1 } else { 0 };
        macro_rules! vp { ($d:literal, $kk:literal) => { unsafe { au.index_unchecked((vco + ((key0 + $kk * 16) * kvd + kh * 256) / 2 + $d * 8) as usize) } } }
        unsafe {
            asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%pC = OpTypePointer Function %tC",
                "%last = OpIEqual %bool {last} %c1",
                "%v0 = OpVariable %pC Function %z",
                "%v1 = OpVariable %pC Function %z",
                "%v2 = OpVariable %pC Function %z",
                "%v3 = OpVariable %pC Function %z",
                "%v4 = OpVariable %pC Function %z",
                "%v5 = OpVariable %pC Function %z",
                "%v6 = OpVariable %pC Function %z",
                "%v7 = OpVariable %pC Function %z",
                "%v8 = OpVariable %pC Function %z",
                "%v9 = OpVariable %pC Function %z",
                "%v10 = OpVariable %pC Function %z",
                "%v11 = OpVariable %pC Function %z",
                "%v12 = OpVariable %pC Function %z",
                "%v13 = OpVariable %pC Function %z",
                "%v14 = OpVariable %pC Function %z",
                "%v15 = OpVariable %pC Function %z",
                "%AL = OpCooperativeMatrixLoadKHR %tC {al} %c1 %c0", "%P0 = OpCooperativeMatrixLoadKHR %tA {p0} %c0 %c20",
                "%P1 = OpCooperativeMatrixLoadKHR %tA {p1} %c0 %c20",
                "%Q0 = OpLoad %tC %v0", "%O0_0 = OpFMul %tC %Q0 %AL",
                "%V0_0 = OpCooperativeMatrixLoadKHR %tB {v00} %c0 {ks}",
                "%O0_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V0_0 %O0_0",
                "%V0_1 = OpCooperativeMatrixLoadKHR %tB {v01} %c0 {ks}",
                "%O0_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V0_1 %O0_1",
                "OpStore %v0 %O0_2",
                "%Q1 = OpLoad %tC %v1", "%O1_0 = OpFMul %tC %Q1 %AL",
                "%V1_0 = OpCooperativeMatrixLoadKHR %tB {v10} %c0 {ks}",
                "%O1_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V1_0 %O1_0",
                "%V1_1 = OpCooperativeMatrixLoadKHR %tB {v11} %c0 {ks}",
                "%O1_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V1_1 %O1_1",
                "OpStore %v1 %O1_2",
                "%Q2 = OpLoad %tC %v2", "%O2_0 = OpFMul %tC %Q2 %AL",
                "%V2_0 = OpCooperativeMatrixLoadKHR %tB {v20} %c0 {ks}",
                "%O2_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V2_0 %O2_0",
                "%V2_1 = OpCooperativeMatrixLoadKHR %tB {v21} %c0 {ks}",
                "%O2_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V2_1 %O2_1",
                "OpStore %v2 %O2_2",
                "%Q3 = OpLoad %tC %v3", "%O3_0 = OpFMul %tC %Q3 %AL",
                "%V3_0 = OpCooperativeMatrixLoadKHR %tB {v30} %c0 {ks}",
                "%O3_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V3_0 %O3_0",
                "%V3_1 = OpCooperativeMatrixLoadKHR %tB {v31} %c0 {ks}",
                "%O3_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V3_1 %O3_1",
                "OpStore %v3 %O3_2",
                "%Q4 = OpLoad %tC %v4", "%O4_0 = OpFMul %tC %Q4 %AL",
                "%V4_0 = OpCooperativeMatrixLoadKHR %tB {v40} %c0 {ks}",
                "%O4_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V4_0 %O4_0",
                "%V4_1 = OpCooperativeMatrixLoadKHR %tB {v41} %c0 {ks}",
                "%O4_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V4_1 %O4_1",
                "OpStore %v4 %O4_2",
                "%Q5 = OpLoad %tC %v5", "%O5_0 = OpFMul %tC %Q5 %AL",
                "%V5_0 = OpCooperativeMatrixLoadKHR %tB {v50} %c0 {ks}",
                "%O5_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V5_0 %O5_0",
                "%V5_1 = OpCooperativeMatrixLoadKHR %tB {v51} %c0 {ks}",
                "%O5_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V5_1 %O5_1",
                "OpStore %v5 %O5_2",
                "%Q6 = OpLoad %tC %v6", "%O6_0 = OpFMul %tC %Q6 %AL",
                "%V6_0 = OpCooperativeMatrixLoadKHR %tB {v60} %c0 {ks}",
                "%O6_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V6_0 %O6_0",
                "%V6_1 = OpCooperativeMatrixLoadKHR %tB {v61} %c0 {ks}",
                "%O6_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V6_1 %O6_1",
                "OpStore %v6 %O6_2",
                "%Q7 = OpLoad %tC %v7", "%O7_0 = OpFMul %tC %Q7 %AL",
                "%V7_0 = OpCooperativeMatrixLoadKHR %tB {v70} %c0 {ks}",
                "%O7_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V7_0 %O7_0",
                "%V7_1 = OpCooperativeMatrixLoadKHR %tB {v71} %c0 {ks}",
                "%O7_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V7_1 %O7_1",
                "OpStore %v7 %O7_2",
                "%Q8 = OpLoad %tC %v8", "%O8_0 = OpFMul %tC %Q8 %AL",
                "%V8_0 = OpCooperativeMatrixLoadKHR %tB {v80} %c0 {ks}",
                "%O8_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V8_0 %O8_0",
                "%V8_1 = OpCooperativeMatrixLoadKHR %tB {v81} %c0 {ks}",
                "%O8_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V8_1 %O8_1",
                "OpStore %v8 %O8_2",
                "%Q9 = OpLoad %tC %v9", "%O9_0 = OpFMul %tC %Q9 %AL",
                "%V9_0 = OpCooperativeMatrixLoadKHR %tB {v90} %c0 {ks}",
                "%O9_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V9_0 %O9_0",
                "%V9_1 = OpCooperativeMatrixLoadKHR %tB {v91} %c0 {ks}",
                "%O9_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V9_1 %O9_1",
                "OpStore %v9 %O9_2",
                "%Q10 = OpLoad %tC %v10", "%O10_0 = OpFMul %tC %Q10 %AL",
                "%V10_0 = OpCooperativeMatrixLoadKHR %tB {v100} %c0 {ks}",
                "%O10_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V10_0 %O10_0",
                "%V10_1 = OpCooperativeMatrixLoadKHR %tB {v101} %c0 {ks}",
                "%O10_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V10_1 %O10_1",
                "OpStore %v10 %O10_2",
                "%Q11 = OpLoad %tC %v11", "%O11_0 = OpFMul %tC %Q11 %AL",
                "%V11_0 = OpCooperativeMatrixLoadKHR %tB {v110} %c0 {ks}",
                "%O11_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V11_0 %O11_0",
                "%V11_1 = OpCooperativeMatrixLoadKHR %tB {v111} %c0 {ks}",
                "%O11_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V11_1 %O11_1",
                "OpStore %v11 %O11_2",
                "%Q12 = OpLoad %tC %v12", "%O12_0 = OpFMul %tC %Q12 %AL",
                "%V12_0 = OpCooperativeMatrixLoadKHR %tB {v120} %c0 {ks}",
                "%O12_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V12_0 %O12_0",
                "%V12_1 = OpCooperativeMatrixLoadKHR %tB {v121} %c0 {ks}",
                "%O12_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V12_1 %O12_1",
                "OpStore %v12 %O12_2",
                "%Q13 = OpLoad %tC %v13", "%O13_0 = OpFMul %tC %Q13 %AL",
                "%V13_0 = OpCooperativeMatrixLoadKHR %tB {v130} %c0 {ks}",
                "%O13_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V13_0 %O13_0",
                "%V13_1 = OpCooperativeMatrixLoadKHR %tB {v131} %c0 {ks}",
                "%O13_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V13_1 %O13_1",
                "OpStore %v13 %O13_2",
                "%Q14 = OpLoad %tC %v14", "%O14_0 = OpFMul %tC %Q14 %AL",
                "%V14_0 = OpCooperativeMatrixLoadKHR %tB {v140} %c0 {ks}",
                "%O14_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V14_0 %O14_0",
                "%V14_1 = OpCooperativeMatrixLoadKHR %tB {v141} %c0 {ks}",
                "%O14_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V14_1 %O14_1",
                "OpStore %v14 %O14_2",
                "%Q15 = OpLoad %tC %v15", "%O15_0 = OpFMul %tC %Q15 %AL",
                "%V15_0 = OpCooperativeMatrixLoadKHR %tB {v150} %c0 {ks}",
                "%O15_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V15_0 %O15_0",
                "%V15_1 = OpCooperativeMatrixLoadKHR %tB {v151} %c0 {ks}",
                "%O15_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V15_1 %O15_1",
                "OpStore %v15 %O15_2",
                "OpBranchConditional %last %l1 %m1",
                "%l1 = OpLabel",
                "OpCooperativeMatrixStoreKHR {o0} %O0_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o1} %O1_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o2} %O2_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o3} %O3_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o4} %O4_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o5} %O5_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o6} %O6_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o7} %O7_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o8} %O8_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o9} %O9_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o10} %O10_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o11} %O11_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o12} %O12_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o13} %O13_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o14} %O14_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o15} %O15_2 %c0 {os}",
                "OpBranch %m1",
                "%m1 = OpLabel",
                last = in(reg) last, al = in(reg) unsafe { alph.index_unchecked((w * 16) as usize) },
                ks = in(reg) ks,
                os = in(reg) os,
                p0 = in(reg) pp!(0),
                p1 = in(reg) pp!(1),
                v00 = in(reg) vp!(0,0),
                v01 = in(reg) vp!(0,1),
                v10 = in(reg) vp!(1,0),
                v11 = in(reg) vp!(1,1),
                v20 = in(reg) vp!(2,0),
                v21 = in(reg) vp!(2,1),
                v30 = in(reg) vp!(3,0),
                v31 = in(reg) vp!(3,1),
                v40 = in(reg) vp!(4,0),
                v41 = in(reg) vp!(4,1),
                v50 = in(reg) vp!(5,0),
                v51 = in(reg) vp!(5,1),
                v60 = in(reg) vp!(6,0),
                v61 = in(reg) vp!(6,1),
                v70 = in(reg) vp!(7,0),
                v71 = in(reg) vp!(7,1),
                v80 = in(reg) vp!(8,0),
                v81 = in(reg) vp!(8,1),
                v90 = in(reg) vp!(9,0),
                v91 = in(reg) vp!(9,1),
                v100 = in(reg) vp!(10,0),
                v101 = in(reg) vp!(10,1),
                v110 = in(reg) vp!(11,0),
                v111 = in(reg) vp!(11,1),
                v120 = in(reg) vp!(12,0),
                v121 = in(reg) vp!(12,1),
                v130 = in(reg) vp!(13,0),
                v131 = in(reg) vp!(13,1),
                v140 = in(reg) vp!(14,0),
                v141 = in(reg) vp!(14,1),
                v150 = in(reg) vp!(15,0),
                v151 = in(reg) vp!(15,1),
                o0 = in(reg) op!(0),
                o1 = in(reg) op!(1),
                o2 = in(reg) op!(2),
                o3 = in(reg) op!(3),
                o4 = in(reg) op!(4),
                o5 = in(reg) op!(5),
                o6 = in(reg) op!(6),
                o7 = in(reg) op!(7),
                o8 = in(reg) op!(8),
                o9 = in(reg) op!(9),
                o10 = in(reg) op!(10),
                o11 = in(reg) op!(11),
                o12 = in(reg) op!(12),
                o13 = in(reg) op!(13),
                o14 = in(reg) op!(14),
                o15 = in(reg) op!(15)
            );
        }
        bar();
        b += 1;
    }
    s!(red, l, sum); bar();
    let den = sum + g!(red, l ^ 1);
    if ln & 1 == 0 { s!(ss, qrow, den); }
    { use spirv_std::memory::{Scope, Semantics};
      spirv_std::arch::control_barrier::<{ Scope::Workgroup as u32 }, { Scope::Workgroup as u32 }, { Semantics::UNIFORM_MEMORY.bits() | Semantics::WORKGROUP_MEMORY.bits() | Semantics::ACQUIRE_RELEASE.bits() }>(); }
    let rr = l >> 1; let t = qb + rr;
    if t < nt {
        let dn = g!(ss, rr); let half = (l & 1) * 128;
        let mut d = 0u32;
        while d < 128 {
            let dd = half + d;
            let o = g!(a, scr + t * os + h * 256 + dd);
            s!(a, oo + t * ots + h * 256 + dd, o / dn * sigm(g!(a, qo + t * qts + h * 512 + 256 + dd)));
            d += 1;
        }
    }
}

/// attn_fa2 with Q kept register-resident: the 16 f16 Q fragments are loaded once (first key block) into
/// Function-scope coop matrices, so there is neither a 33 KB smem Q tile nor a per-block Q reload.
#[spirv(compute(threads(128)))]
pub fn attn_fa3(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] aw: &mut [u32],
    #[spirv(workgroup)] ss: &mut [f32; 2048], #[spirv(workgroup)] ps: &mut [u32; 1280],
    #[spirv(workgroup)] red: &mut [f32; 128], #[spirv(workgroup)] alph: &mut [f32; 64]) {
    let (qo, qts, kco, vco, kvd, pos0) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5));
    let (nh, nkv, scr, nt, oo, ots) = (pu(pc, 7), pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12));
    let scale = pf(pc, 13);
    let h = wg.x; let qb = wg.y * 64; let l = lid.x; let w = l >> 5; let ln = l & 31;
    let kh = h / (nh / nkv);
    {
        let r = l >> 1; let half = l & 1; let t = (qb + r).min(nt - 1);
        let base = (qo + t * qts + h * 512) / 4 + half * 32;
        // f16 Q goes to this head's (still unused) O scratch slot for the same row; only this workgroup touches it
        let qd = scr + (qb + r) * (nh * 256) + h * 256 + half * 64;
        let mut v = 0u32;
        while v < 32 {
            let x = g!(a4, base + v);
            s!(aw, qd + v * 2, vec2_to_f16x2(Vec2::new(x.x, x.y))); s!(aw, qd + v * 2 + 1, vec2_to_f16x2(Vec2::new(x.z, x.w)));
            v += 1;
        }
    }
    bar();
    let qlast = (qb + 64).min(nt) - 1; let nblk = (pos0 + qlast + 1 + 31) / 32;
    let r = ln >> 1; let ch = (ln & 1) * 16; let qrow = w * 16 + r; let qpos = pos0 + qb + qrow;
    let sb = w * 512; let pb = w * 320; let ks = kvd / 2; let os = nh * 256;
    macro_rules! qp { ($kk:literal) => { unsafe { au.index_unchecked((scr + (qb + w * 16) * os + h * 256 + $kk * 8) as usize) } } }
    macro_rules! sp { ($j:literal) => { unsafe { ss.index_unchecked_mut((sb + $j * 16) as usize) } as *mut f32 } }
    macro_rules! pp { ($kk:literal) => { unsafe { ps.index_unchecked((pb + $kk * 8) as usize) } } }
    macro_rules! op { ($d:literal) => { unsafe { a.index_unchecked_mut((scr + (qb + w * 16) * os + h * 256 + $d * 16) as usize) } as *mut f32 } }
    let mut m = -1e30f32; let mut sum = 0.0f32;
    let mut b = 0u32;
    while b < nblk {
        let key0 = b * 32;
        macro_rules! kp { ($j:literal, $kk:literal) => { unsafe { au.index_unchecked((kco + ((key0 + $j * 16) * kvd + kh * 256) / 2 + $kk * 8) as usize) } } }
        unsafe {
            asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%pA = OpTypePointer Function %tA",
                "%first = OpIEqual %bool {first} %c1",
                "%qv0 = OpVariable %pA Function",
                "%qv1 = OpVariable %pA Function",
                "%qv2 = OpVariable %pA Function",
                "%qv3 = OpVariable %pA Function",
                "%qv4 = OpVariable %pA Function",
                "%qv5 = OpVariable %pA Function",
                "%qv6 = OpVariable %pA Function",
                "%qv7 = OpVariable %pA Function",
                "%qv8 = OpVariable %pA Function",
                "%qv9 = OpVariable %pA Function",
                "%qv10 = OpVariable %pA Function",
                "%qv11 = OpVariable %pA Function",
                "%qv12 = OpVariable %pA Function",
                "%qv13 = OpVariable %pA Function",
                "%qv14 = OpVariable %pA Function",
                "%qv15 = OpVariable %pA Function",
                "OpBranchConditional %first %lq %mq",
                "%lq = OpLabel",
                "%L0 = OpCooperativeMatrixLoadKHR %tA {q0} %c0 {qst}", "OpStore %qv0 %L0",
                "%L1 = OpCooperativeMatrixLoadKHR %tA {q1} %c0 {qst}", "OpStore %qv1 %L1",
                "%L2 = OpCooperativeMatrixLoadKHR %tA {q2} %c0 {qst}", "OpStore %qv2 %L2",
                "%L3 = OpCooperativeMatrixLoadKHR %tA {q3} %c0 {qst}", "OpStore %qv3 %L3",
                "%L4 = OpCooperativeMatrixLoadKHR %tA {q4} %c0 {qst}", "OpStore %qv4 %L4",
                "%L5 = OpCooperativeMatrixLoadKHR %tA {q5} %c0 {qst}", "OpStore %qv5 %L5",
                "%L6 = OpCooperativeMatrixLoadKHR %tA {q6} %c0 {qst}", "OpStore %qv6 %L6",
                "%L7 = OpCooperativeMatrixLoadKHR %tA {q7} %c0 {qst}", "OpStore %qv7 %L7",
                "%L8 = OpCooperativeMatrixLoadKHR %tA {q8} %c0 {qst}", "OpStore %qv8 %L8",
                "%L9 = OpCooperativeMatrixLoadKHR %tA {q9} %c0 {qst}", "OpStore %qv9 %L9",
                "%L10 = OpCooperativeMatrixLoadKHR %tA {q10} %c0 {qst}", "OpStore %qv10 %L10",
                "%L11 = OpCooperativeMatrixLoadKHR %tA {q11} %c0 {qst}", "OpStore %qv11 %L11",
                "%L12 = OpCooperativeMatrixLoadKHR %tA {q12} %c0 {qst}", "OpStore %qv12 %L12",
                "%L13 = OpCooperativeMatrixLoadKHR %tA {q13} %c0 {qst}", "OpStore %qv13 %L13",
                "%L14 = OpCooperativeMatrixLoadKHR %tA {q14} %c0 {qst}", "OpStore %qv14 %L14",
                "%L15 = OpCooperativeMatrixLoadKHR %tA {q15} %c0 {qst}", "OpStore %qv15 %L15",
                "OpBranch %mq",
                "%mq = OpLabel",
                "%A0 = OpLoad %tA %qv0",
                "%B0_0 = OpCooperativeMatrixLoadKHR %tB {k00} %c1 {ks}",
                "%S0_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B0_0 %z",
                "%B1_0 = OpCooperativeMatrixLoadKHR %tB {k10} %c1 {ks}",
                "%S1_1 = OpCooperativeMatrixMulAddKHR %tC %A0 %B1_0 %z",
                "%A1 = OpLoad %tA %qv1",
                "%B0_1 = OpCooperativeMatrixLoadKHR %tB {k01} %c1 {ks}",
                "%S0_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B0_1 %S0_1",
                "%B1_1 = OpCooperativeMatrixLoadKHR %tB {k11} %c1 {ks}",
                "%S1_2 = OpCooperativeMatrixMulAddKHR %tC %A1 %B1_1 %S1_1",
                "%A2 = OpLoad %tA %qv2",
                "%B0_2 = OpCooperativeMatrixLoadKHR %tB {k02} %c1 {ks}",
                "%S0_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B0_2 %S0_2",
                "%B1_2 = OpCooperativeMatrixLoadKHR %tB {k12} %c1 {ks}",
                "%S1_3 = OpCooperativeMatrixMulAddKHR %tC %A2 %B1_2 %S1_2",
                "%A3 = OpLoad %tA %qv3",
                "%B0_3 = OpCooperativeMatrixLoadKHR %tB {k03} %c1 {ks}",
                "%S0_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B0_3 %S0_3",
                "%B1_3 = OpCooperativeMatrixLoadKHR %tB {k13} %c1 {ks}",
                "%S1_4 = OpCooperativeMatrixMulAddKHR %tC %A3 %B1_3 %S1_3",
                "%A4 = OpLoad %tA %qv4",
                "%B0_4 = OpCooperativeMatrixLoadKHR %tB {k04} %c1 {ks}",
                "%S0_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B0_4 %S0_4",
                "%B1_4 = OpCooperativeMatrixLoadKHR %tB {k14} %c1 {ks}",
                "%S1_5 = OpCooperativeMatrixMulAddKHR %tC %A4 %B1_4 %S1_4",
                "%A5 = OpLoad %tA %qv5",
                "%B0_5 = OpCooperativeMatrixLoadKHR %tB {k05} %c1 {ks}",
                "%S0_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B0_5 %S0_5",
                "%B1_5 = OpCooperativeMatrixLoadKHR %tB {k15} %c1 {ks}",
                "%S1_6 = OpCooperativeMatrixMulAddKHR %tC %A5 %B1_5 %S1_5",
                "%A6 = OpLoad %tA %qv6",
                "%B0_6 = OpCooperativeMatrixLoadKHR %tB {k06} %c1 {ks}",
                "%S0_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B0_6 %S0_6",
                "%B1_6 = OpCooperativeMatrixLoadKHR %tB {k16} %c1 {ks}",
                "%S1_7 = OpCooperativeMatrixMulAddKHR %tC %A6 %B1_6 %S1_6",
                "%A7 = OpLoad %tA %qv7",
                "%B0_7 = OpCooperativeMatrixLoadKHR %tB {k07} %c1 {ks}",
                "%S0_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B0_7 %S0_7",
                "%B1_7 = OpCooperativeMatrixLoadKHR %tB {k17} %c1 {ks}",
                "%S1_8 = OpCooperativeMatrixMulAddKHR %tC %A7 %B1_7 %S1_7",
                "%A8 = OpLoad %tA %qv8",
                "%B0_8 = OpCooperativeMatrixLoadKHR %tB {k08} %c1 {ks}",
                "%S0_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B0_8 %S0_8",
                "%B1_8 = OpCooperativeMatrixLoadKHR %tB {k18} %c1 {ks}",
                "%S1_9 = OpCooperativeMatrixMulAddKHR %tC %A8 %B1_8 %S1_8",
                "%A9 = OpLoad %tA %qv9",
                "%B0_9 = OpCooperativeMatrixLoadKHR %tB {k09} %c1 {ks}",
                "%S0_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B0_9 %S0_9",
                "%B1_9 = OpCooperativeMatrixLoadKHR %tB {k19} %c1 {ks}",
                "%S1_10 = OpCooperativeMatrixMulAddKHR %tC %A9 %B1_9 %S1_9",
                "%A10 = OpLoad %tA %qv10",
                "%B0_10 = OpCooperativeMatrixLoadKHR %tB {k010} %c1 {ks}",
                "%S0_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B0_10 %S0_10",
                "%B1_10 = OpCooperativeMatrixLoadKHR %tB {k110} %c1 {ks}",
                "%S1_11 = OpCooperativeMatrixMulAddKHR %tC %A10 %B1_10 %S1_10",
                "%A11 = OpLoad %tA %qv11",
                "%B0_11 = OpCooperativeMatrixLoadKHR %tB {k011} %c1 {ks}",
                "%S0_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B0_11 %S0_11",
                "%B1_11 = OpCooperativeMatrixLoadKHR %tB {k111} %c1 {ks}",
                "%S1_12 = OpCooperativeMatrixMulAddKHR %tC %A11 %B1_11 %S1_11",
                "%A12 = OpLoad %tA %qv12",
                "%B0_12 = OpCooperativeMatrixLoadKHR %tB {k012} %c1 {ks}",
                "%S0_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B0_12 %S0_12",
                "%B1_12 = OpCooperativeMatrixLoadKHR %tB {k112} %c1 {ks}",
                "%S1_13 = OpCooperativeMatrixMulAddKHR %tC %A12 %B1_12 %S1_12",
                "%A13 = OpLoad %tA %qv13",
                "%B0_13 = OpCooperativeMatrixLoadKHR %tB {k013} %c1 {ks}",
                "%S0_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B0_13 %S0_13",
                "%B1_13 = OpCooperativeMatrixLoadKHR %tB {k113} %c1 {ks}",
                "%S1_14 = OpCooperativeMatrixMulAddKHR %tC %A13 %B1_13 %S1_13",
                "%A14 = OpLoad %tA %qv14",
                "%B0_14 = OpCooperativeMatrixLoadKHR %tB {k014} %c1 {ks}",
                "%S0_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B0_14 %S0_14",
                "%B1_14 = OpCooperativeMatrixLoadKHR %tB {k114} %c1 {ks}",
                "%S1_15 = OpCooperativeMatrixMulAddKHR %tC %A14 %B1_14 %S1_14",
                "%A15 = OpLoad %tA %qv15",
                "%B0_15 = OpCooperativeMatrixLoadKHR %tB {k015} %c1 {ks}",
                "%S0_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B0_15 %S0_15",
                "%B1_15 = OpCooperativeMatrixLoadKHR %tB {k115} %c1 {ks}",
                "%S1_16 = OpCooperativeMatrixMulAddKHR %tC %A15 %B1_15 %S1_15",
                "OpCooperativeMatrixStoreKHR {s0} %S0_16 %c0 %c32",
                "OpCooperativeMatrixStoreKHR {s1} %S1_16 %c0 %c32",
                first = in(reg) (if b == 0 { 1u32 } else { 0u32 }),
                ks = in(reg) ks,
                qst = in(reg) os,
                q0 = in(reg) qp!(0),
                q1 = in(reg) qp!(1),
                q2 = in(reg) qp!(2),
                q3 = in(reg) qp!(3),
                q4 = in(reg) qp!(4),
                q5 = in(reg) qp!(5),
                q6 = in(reg) qp!(6),
                q7 = in(reg) qp!(7),
                q8 = in(reg) qp!(8),
                q9 = in(reg) qp!(9),
                q10 = in(reg) qp!(10),
                q11 = in(reg) qp!(11),
                q12 = in(reg) qp!(12),
                q13 = in(reg) qp!(13),
                q14 = in(reg) qp!(14),
                q15 = in(reg) qp!(15),
                k00 = in(reg) kp!(0,0),
                k01 = in(reg) kp!(0,1),
                k02 = in(reg) kp!(0,2),
                k03 = in(reg) kp!(0,3),
                k04 = in(reg) kp!(0,4),
                k05 = in(reg) kp!(0,5),
                k06 = in(reg) kp!(0,6),
                k07 = in(reg) kp!(0,7),
                k08 = in(reg) kp!(0,8),
                k09 = in(reg) kp!(0,9),
                k010 = in(reg) kp!(0,10),
                k011 = in(reg) kp!(0,11),
                k012 = in(reg) kp!(0,12),
                k013 = in(reg) kp!(0,13),
                k014 = in(reg) kp!(0,14),
                k015 = in(reg) kp!(0,15),
                k10 = in(reg) kp!(1,0),
                k11 = in(reg) kp!(1,1),
                k12 = in(reg) kp!(1,2),
                k13 = in(reg) kp!(1,3),
                k14 = in(reg) kp!(1,4),
                k15 = in(reg) kp!(1,5),
                k16 = in(reg) kp!(1,6),
                k17 = in(reg) kp!(1,7),
                k18 = in(reg) kp!(1,8),
                k19 = in(reg) kp!(1,9),
                k110 = in(reg) kp!(1,10),
                k111 = in(reg) kp!(1,11),
                k112 = in(reg) kp!(1,12),
                k113 = in(reg) kp!(1,13),
                k114 = in(reg) kp!(1,14),
                k115 = in(reg) kp!(1,15),
                s0 = in(reg) sp!(0),
                s1 = in(reg) sp!(1)
            );
        }
        bar();
        let mut lm = -1e30f32;
        let mut j = 0u32;
        while j < 16 { if key0 + ch + j <= qpos { lm = lm.max(g!(ss, sb + r * 32 + ch + j) * scale); } j += 1; }
        s!(red, l, lm); bar();
        let mn = m.max(lm).max(g!(red, l ^ 1));
        let alpha = (m - mn).exp(); m = mn; sum *= alpha;
        j = 0;
        while j < 16 {
            let p0 = key0 + ch + j;
            let e0 = if p0 <= qpos { (g!(ss, sb + r * 32 + ch + j) * scale - m).exp() } else { 0.0 };
            let e1 = if p0 + 1 <= qpos { (g!(ss, sb + r * 32 + ch + j + 1) * scale - m).exp() } else { 0.0 };
            sum += e0 + e1;
            s!(ps, pb + r * 20 + (ch + j) / 2, vec2_to_f16x2(Vec2::new(e0, e1)));
            j += 2;
        }
        if ln & 1 == 0 { s!(alph, w * 16 + r, alpha); }
        bar();
        let last: u32 = if b + 1 == nblk { 1 } else { 0 };
        macro_rules! vp { ($d:literal, $kk:literal) => { unsafe { au.index_unchecked((vco + ((key0 + $kk * 16) * kvd + kh * 256) / 2 + $d * 8) as usize) } } }
        unsafe {
            asm!(
                "%f16 = OpTypeFloat 16",
                "%f32 = OpTypeFloat 32",
                "%u32 = OpTypeInt 32 0",
                "%bool = OpTypeBool",
                "%c0 = OpConstant %u32 0",
                "%c1 = OpConstant %u32 1",
                "%c2 = OpConstant %u32 2",
                "%c3 = OpConstant %u32 3",
                "%c16 = OpConstant %u32 16",
                "%c20 = OpConstant %u32 20",
                "%c32 = OpConstant %u32 32",
                "%c132 = OpConstant %u32 132",
                "%tA = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c0",
                "%tB = OpTypeCooperativeMatrixKHR %f16 %c3 %c16 %c16 %c1",
                "%tC = OpTypeCooperativeMatrixKHR %f32 %c3 %c16 %c16 %c2",
                "%z = OpConstantNull %tC",
                "%pC = OpTypePointer Function %tC",
                "%last = OpIEqual %bool {last} %c1",
                "%v0 = OpVariable %pC Function %z",
                "%v1 = OpVariable %pC Function %z",
                "%v2 = OpVariable %pC Function %z",
                "%v3 = OpVariable %pC Function %z",
                "%v4 = OpVariable %pC Function %z",
                "%v5 = OpVariable %pC Function %z",
                "%v6 = OpVariable %pC Function %z",
                "%v7 = OpVariable %pC Function %z",
                "%v8 = OpVariable %pC Function %z",
                "%v9 = OpVariable %pC Function %z",
                "%v10 = OpVariable %pC Function %z",
                "%v11 = OpVariable %pC Function %z",
                "%v12 = OpVariable %pC Function %z",
                "%v13 = OpVariable %pC Function %z",
                "%v14 = OpVariable %pC Function %z",
                "%v15 = OpVariable %pC Function %z",
                "%AL = OpCooperativeMatrixLoadKHR %tC {al} %c1 %c0", "%P0 = OpCooperativeMatrixLoadKHR %tA {p0} %c0 %c20",
                "%P1 = OpCooperativeMatrixLoadKHR %tA {p1} %c0 %c20",
                "%Q0 = OpLoad %tC %v0", "%O0_0 = OpFMul %tC %Q0 %AL",
                "%V0_0 = OpCooperativeMatrixLoadKHR %tB {v00} %c0 {ks}",
                "%O0_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V0_0 %O0_0",
                "%V0_1 = OpCooperativeMatrixLoadKHR %tB {v01} %c0 {ks}",
                "%O0_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V0_1 %O0_1",
                "OpStore %v0 %O0_2",
                "%Q1 = OpLoad %tC %v1", "%O1_0 = OpFMul %tC %Q1 %AL",
                "%V1_0 = OpCooperativeMatrixLoadKHR %tB {v10} %c0 {ks}",
                "%O1_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V1_0 %O1_0",
                "%V1_1 = OpCooperativeMatrixLoadKHR %tB {v11} %c0 {ks}",
                "%O1_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V1_1 %O1_1",
                "OpStore %v1 %O1_2",
                "%Q2 = OpLoad %tC %v2", "%O2_0 = OpFMul %tC %Q2 %AL",
                "%V2_0 = OpCooperativeMatrixLoadKHR %tB {v20} %c0 {ks}",
                "%O2_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V2_0 %O2_0",
                "%V2_1 = OpCooperativeMatrixLoadKHR %tB {v21} %c0 {ks}",
                "%O2_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V2_1 %O2_1",
                "OpStore %v2 %O2_2",
                "%Q3 = OpLoad %tC %v3", "%O3_0 = OpFMul %tC %Q3 %AL",
                "%V3_0 = OpCooperativeMatrixLoadKHR %tB {v30} %c0 {ks}",
                "%O3_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V3_0 %O3_0",
                "%V3_1 = OpCooperativeMatrixLoadKHR %tB {v31} %c0 {ks}",
                "%O3_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V3_1 %O3_1",
                "OpStore %v3 %O3_2",
                "%Q4 = OpLoad %tC %v4", "%O4_0 = OpFMul %tC %Q4 %AL",
                "%V4_0 = OpCooperativeMatrixLoadKHR %tB {v40} %c0 {ks}",
                "%O4_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V4_0 %O4_0",
                "%V4_1 = OpCooperativeMatrixLoadKHR %tB {v41} %c0 {ks}",
                "%O4_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V4_1 %O4_1",
                "OpStore %v4 %O4_2",
                "%Q5 = OpLoad %tC %v5", "%O5_0 = OpFMul %tC %Q5 %AL",
                "%V5_0 = OpCooperativeMatrixLoadKHR %tB {v50} %c0 {ks}",
                "%O5_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V5_0 %O5_0",
                "%V5_1 = OpCooperativeMatrixLoadKHR %tB {v51} %c0 {ks}",
                "%O5_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V5_1 %O5_1",
                "OpStore %v5 %O5_2",
                "%Q6 = OpLoad %tC %v6", "%O6_0 = OpFMul %tC %Q6 %AL",
                "%V6_0 = OpCooperativeMatrixLoadKHR %tB {v60} %c0 {ks}",
                "%O6_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V6_0 %O6_0",
                "%V6_1 = OpCooperativeMatrixLoadKHR %tB {v61} %c0 {ks}",
                "%O6_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V6_1 %O6_1",
                "OpStore %v6 %O6_2",
                "%Q7 = OpLoad %tC %v7", "%O7_0 = OpFMul %tC %Q7 %AL",
                "%V7_0 = OpCooperativeMatrixLoadKHR %tB {v70} %c0 {ks}",
                "%O7_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V7_0 %O7_0",
                "%V7_1 = OpCooperativeMatrixLoadKHR %tB {v71} %c0 {ks}",
                "%O7_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V7_1 %O7_1",
                "OpStore %v7 %O7_2",
                "%Q8 = OpLoad %tC %v8", "%O8_0 = OpFMul %tC %Q8 %AL",
                "%V8_0 = OpCooperativeMatrixLoadKHR %tB {v80} %c0 {ks}",
                "%O8_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V8_0 %O8_0",
                "%V8_1 = OpCooperativeMatrixLoadKHR %tB {v81} %c0 {ks}",
                "%O8_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V8_1 %O8_1",
                "OpStore %v8 %O8_2",
                "%Q9 = OpLoad %tC %v9", "%O9_0 = OpFMul %tC %Q9 %AL",
                "%V9_0 = OpCooperativeMatrixLoadKHR %tB {v90} %c0 {ks}",
                "%O9_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V9_0 %O9_0",
                "%V9_1 = OpCooperativeMatrixLoadKHR %tB {v91} %c0 {ks}",
                "%O9_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V9_1 %O9_1",
                "OpStore %v9 %O9_2",
                "%Q10 = OpLoad %tC %v10", "%O10_0 = OpFMul %tC %Q10 %AL",
                "%V10_0 = OpCooperativeMatrixLoadKHR %tB {v100} %c0 {ks}",
                "%O10_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V10_0 %O10_0",
                "%V10_1 = OpCooperativeMatrixLoadKHR %tB {v101} %c0 {ks}",
                "%O10_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V10_1 %O10_1",
                "OpStore %v10 %O10_2",
                "%Q11 = OpLoad %tC %v11", "%O11_0 = OpFMul %tC %Q11 %AL",
                "%V11_0 = OpCooperativeMatrixLoadKHR %tB {v110} %c0 {ks}",
                "%O11_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V11_0 %O11_0",
                "%V11_1 = OpCooperativeMatrixLoadKHR %tB {v111} %c0 {ks}",
                "%O11_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V11_1 %O11_1",
                "OpStore %v11 %O11_2",
                "%Q12 = OpLoad %tC %v12", "%O12_0 = OpFMul %tC %Q12 %AL",
                "%V12_0 = OpCooperativeMatrixLoadKHR %tB {v120} %c0 {ks}",
                "%O12_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V12_0 %O12_0",
                "%V12_1 = OpCooperativeMatrixLoadKHR %tB {v121} %c0 {ks}",
                "%O12_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V12_1 %O12_1",
                "OpStore %v12 %O12_2",
                "%Q13 = OpLoad %tC %v13", "%O13_0 = OpFMul %tC %Q13 %AL",
                "%V13_0 = OpCooperativeMatrixLoadKHR %tB {v130} %c0 {ks}",
                "%O13_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V13_0 %O13_0",
                "%V13_1 = OpCooperativeMatrixLoadKHR %tB {v131} %c0 {ks}",
                "%O13_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V13_1 %O13_1",
                "OpStore %v13 %O13_2",
                "%Q14 = OpLoad %tC %v14", "%O14_0 = OpFMul %tC %Q14 %AL",
                "%V14_0 = OpCooperativeMatrixLoadKHR %tB {v140} %c0 {ks}",
                "%O14_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V14_0 %O14_0",
                "%V14_1 = OpCooperativeMatrixLoadKHR %tB {v141} %c0 {ks}",
                "%O14_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V14_1 %O14_1",
                "OpStore %v14 %O14_2",
                "%Q15 = OpLoad %tC %v15", "%O15_0 = OpFMul %tC %Q15 %AL",
                "%V15_0 = OpCooperativeMatrixLoadKHR %tB {v150} %c0 {ks}",
                "%O15_1 = OpCooperativeMatrixMulAddKHR %tC %P0 %V15_0 %O15_0",
                "%V15_1 = OpCooperativeMatrixLoadKHR %tB {v151} %c0 {ks}",
                "%O15_2 = OpCooperativeMatrixMulAddKHR %tC %P1 %V15_1 %O15_1",
                "OpStore %v15 %O15_2",
                "OpBranchConditional %last %l1 %m1",
                "%l1 = OpLabel",
                "OpCooperativeMatrixStoreKHR {o0} %O0_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o1} %O1_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o2} %O2_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o3} %O3_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o4} %O4_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o5} %O5_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o6} %O6_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o7} %O7_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o8} %O8_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o9} %O9_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o10} %O10_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o11} %O11_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o12} %O12_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o13} %O13_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o14} %O14_2 %c0 {os}",
                "OpCooperativeMatrixStoreKHR {o15} %O15_2 %c0 {os}",
                "OpBranch %m1",
                "%m1 = OpLabel",
                last = in(reg) last, al = in(reg) unsafe { alph.index_unchecked((w * 16) as usize) },
                ks = in(reg) ks,
                os = in(reg) os,
                p0 = in(reg) pp!(0),
                p1 = in(reg) pp!(1),
                v00 = in(reg) vp!(0,0),
                v01 = in(reg) vp!(0,1),
                v10 = in(reg) vp!(1,0),
                v11 = in(reg) vp!(1,1),
                v20 = in(reg) vp!(2,0),
                v21 = in(reg) vp!(2,1),
                v30 = in(reg) vp!(3,0),
                v31 = in(reg) vp!(3,1),
                v40 = in(reg) vp!(4,0),
                v41 = in(reg) vp!(4,1),
                v50 = in(reg) vp!(5,0),
                v51 = in(reg) vp!(5,1),
                v60 = in(reg) vp!(6,0),
                v61 = in(reg) vp!(6,1),
                v70 = in(reg) vp!(7,0),
                v71 = in(reg) vp!(7,1),
                v80 = in(reg) vp!(8,0),
                v81 = in(reg) vp!(8,1),
                v90 = in(reg) vp!(9,0),
                v91 = in(reg) vp!(9,1),
                v100 = in(reg) vp!(10,0),
                v101 = in(reg) vp!(10,1),
                v110 = in(reg) vp!(11,0),
                v111 = in(reg) vp!(11,1),
                v120 = in(reg) vp!(12,0),
                v121 = in(reg) vp!(12,1),
                v130 = in(reg) vp!(13,0),
                v131 = in(reg) vp!(13,1),
                v140 = in(reg) vp!(14,0),
                v141 = in(reg) vp!(14,1),
                v150 = in(reg) vp!(15,0),
                v151 = in(reg) vp!(15,1),
                o0 = in(reg) op!(0),
                o1 = in(reg) op!(1),
                o2 = in(reg) op!(2),
                o3 = in(reg) op!(3),
                o4 = in(reg) op!(4),
                o5 = in(reg) op!(5),
                o6 = in(reg) op!(6),
                o7 = in(reg) op!(7),
                o8 = in(reg) op!(8),
                o9 = in(reg) op!(9),
                o10 = in(reg) op!(10),
                o11 = in(reg) op!(11),
                o12 = in(reg) op!(12),
                o13 = in(reg) op!(13),
                o14 = in(reg) op!(14),
                o15 = in(reg) op!(15)
            );
        }
        bar();
        b += 1;
    }
    s!(red, l, sum); bar();
    let den = sum + g!(red, l ^ 1);
    if ln & 1 == 0 { s!(ss, qrow, den); }
    { use spirv_std::memory::{Scope, Semantics};
      spirv_std::arch::control_barrier::<{ Scope::Workgroup as u32 }, { Scope::Workgroup as u32 }, { Semantics::UNIFORM_MEMORY.bits() | Semantics::WORKGROUP_MEMORY.bits() | Semantics::ACQUIRE_RELEASE.bits() }>(); }
    let rr = l >> 1; let t = qb + rr;
    if t < nt {
        let dn = g!(ss, rr); let half = (l & 1) * 128;
        let mut d = 0u32;
        while d < 128 {
            let dd = half + d;
            let o = g!(a, scr + t * os + h * 256 + dd);
            s!(a, oo + t * ots + h * 256 + dd, o / dn * sigm(g!(a, qo + t * qts + h * 512 + 256 + dd)));
            d += 1;
        }
    }
}

/// gdn_p3 with v_j, beta and the decay gate of each 16-token block staged in shared memory alongside q/k:
/// the serial token loop no longer waits on three dependent global loads + exp/log per step. Same math.
#[spirv(compute(threads(256)))]
pub fn gdn_p4(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [Vec4; 512], #[spirv(workgroup)] ks: &mut [Vec4; 512],
    #[spirv(workgroup)] vsh: &mut [f32; 512], #[spirv(workgroup)] gsh: &mut [f32; 16], #[spirv(workgroup)] bsh: &mut [f32; 16]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, l) = (wg.x, lid.x); let rq = l & 7; let j = wg.y * 32 + (l >> 3); let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd + rq * 16 * sd + j;
    let ld = |i: u32| Vec4::new(g!(a, sb + (4 * i) * sd), g!(a, sb + (4 * i + 1) * sd), g!(a, sb + (4 * i + 2) * sd), g!(a, sb + (4 * i + 3) * sd));
    let (mut s0, mut s1, mut s2, mut s3) = (ld(0), ld(1), ld(2), ld(3));
    macro_rules! all4 { (|$s:ident, $i:ident| $b:block) => {
        { let $i = 0u32; let $s = &mut s0; $b } { let $i = 1u32; let $s = &mut s1; $b } { let $i = 2u32; let $s = &mut s2; $b } { let $i = 3u32; let $s = &mut s3; $b } } }
    let red8 = |v: f32| -> f32 { let v = v + sx(v, 1); let v = v + sx(v, 2); v + sx(v, 4) };
    let dt = g!(prm, dto + h); let aa = g!(prm, ao + h);
    let mut tb = 0u32;
    while tb < ntok {
        unroll!([0, 1], |c| {
            let idx = l + 256 * c; let tk = idx >> 5; let d4 = idx & 31; let t = tb + tk;
            let cb = (co + t.min(ntok - 1) * cs) / 4;
            s!(qs, idx, g!(a4, cb + (kh * sd) / 4 + d4)); s!(ks, idx, g!(a4, cb + (qk + kh * sd) / 4 + d4));
            s!(vsh, idx, g!(a, co + t.min(ntok - 1) * cs + 2 * qk + h * sd + wg.y * 32 + d4));
        });
        if l < 16 {
            let t = (tb + l).min(ntok - 1);
            s!(bsh, l, sigm(g!(a, bo + t * bas + h)));
            let x = g!(a, alo + t * bas + h) + dt;
            s!(gsh, l, (aa * (if x > 20.0 { x } else { (1.0 + x.exp()).ln() })).exp());
        }
        bar();
        let nb = (ntok - tb).min(16);
        let mut u = 0u32;
        while u < nb {
            let t = tb + u; let cb = co + t * cs;
            let _ = cb;
            let vj = g!(vsh, u * 32 + (l >> 3));
            let beta = g!(bsh, u);
            let eg = g!(gsh, u);
            let kb = u * 32 + rq * 4;
            let mut sm = Vec4::ZERO;
            all4!(|s, i| { *s = *s * eg; sm += *s * g!(ks, kb + i); });
            let delta = (vj - red8(sm.dot(Vec4::ONE))) * beta;
            let mut o = Vec4::ZERO;
            all4!(|s, i| { *s = *s + g!(ks, kb + i) * delta; o += *s * g!(qs, kb + i); });
            let ov = red8(o.dot(Vec4::ONE));
            if rq == 0 { s!(a, oo + t * os + h * sd + j, ov * scale); }
            u += 1;
        }
        bar();
        tb += 16;
    }
    all4!(|s, i| { let v = *s; s!(a, sb + (4 * i) * sd, v.x); s!(a, sb + (4 * i + 1) * sd, v.y); s!(a, sb + (4 * i + 2) * sd, v.z); s!(a, sb + (4 * i + 3) * sd, v.w); });
}

/// gdn_p4 with the delta-rule step re-associated so the per-token critical path is one 3-wide shuffle reduction
/// (S.k, S.q, k.q) instead of two dependent ones; the state update is off the critical path. Not bit-identical.
#[spirv(compute(threads(256)))]
pub fn gdn_p5(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [Vec4; 512], #[spirv(workgroup)] ks: &mut [Vec4; 512],
    #[spirv(workgroup)] vsh: &mut [f32; 512], #[spirv(workgroup)] gsh: &mut [f32; 16], #[spirv(workgroup)] bsh: &mut [f32; 16]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, l) = (wg.x, lid.x); let rq = l & 7; let j = wg.y * 32 + (l >> 3); let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd + rq * 16 * sd + j;
    let ld = |i: u32| Vec4::new(g!(a, sb + (4 * i) * sd), g!(a, sb + (4 * i + 1) * sd), g!(a, sb + (4 * i + 2) * sd), g!(a, sb + (4 * i + 3) * sd));
    let (mut s0, mut s1, mut s2, mut s3) = (ld(0), ld(1), ld(2), ld(3));
    macro_rules! all4 { (|$s:ident, $i:ident| $b:block) => {
        { let $i = 0u32; let $s = &mut s0; $b } { let $i = 1u32; let $s = &mut s1; $b } { let $i = 2u32; let $s = &mut s2; $b } { let $i = 3u32; let $s = &mut s3; $b } } }
    let red8 = |v: f32| -> f32 { let v = v + sx(v, 1); let v = v + sx(v, 2); v + sx(v, 4) };
    let dt = g!(prm, dto + h); let aa = g!(prm, ao + h);
    let mut tb = 0u32;
    while tb < ntok {
        unroll!([0, 1], |c| {
            let idx = l + 256 * c; let tk = idx >> 5; let d4 = idx & 31; let t = tb + tk;
            let cb = (co + t.min(ntok - 1) * cs) / 4;
            s!(qs, idx, g!(a4, cb + (kh * sd) / 4 + d4)); s!(ks, idx, g!(a4, cb + (qk + kh * sd) / 4 + d4));
            s!(vsh, idx, g!(a, co + t.min(ntok - 1) * cs + 2 * qk + h * sd + wg.y * 32 + d4));
        });
        if l < 16 {
            let t = (tb + l).min(ntok - 1);
            s!(bsh, l, sigm(g!(a, bo + t * bas + h)));
            let x = g!(a, alo + t * bas + h) + dt;
            s!(gsh, l, (aa * (if x > 20.0 { x } else { (1.0 + x.exp()).ln() })).exp());
        }
        bar();
        let nb = (ntok - tb).min(16);
        let mut u = 0u32;
        while u < nb {
            let t = tb + u; let cb = co + t * cs;
            let _ = cb;
            let vj = g!(vsh, u * 32 + (l >> 3));
            let beta = g!(bsh, u);
            let eg = g!(gsh, u);
            let kb = u * 32 + rq * 4;
            // o = q.(eg*S + delta*k) = eg*(S.q) + delta*(k.q): the three dot products share one reduction round
            let (mut sk, mut sq, mut kq) = (Vec4::ZERO, Vec4::ZERO, Vec4::ZERO);
            all4!(|s, i| { let kk = g!(ks, kb + i); let qq = g!(qs, kb + i); sk += *s * kk; sq += *s * qq; kq += kk * qq; });
            let (mut r1, mut r2, mut r3) = (sk.dot(Vec4::ONE), sq.dot(Vec4::ONE), kq.dot(Vec4::ONE));
            r1 += sx(r1, 1); r2 += sx(r2, 1); r3 += sx(r3, 1);
            r1 += sx(r1, 2); r2 += sx(r2, 2); r3 += sx(r3, 2);
            r1 += sx(r1, 4); r2 += sx(r2, 4); r3 += sx(r3, 4);
            let delta = (vj - eg * r1) * beta;
            let ov = eg * r2 + delta * r3;
            all4!(|s, i| { *s = *s * eg + g!(ks, kb + i) * delta; });
            if rq == 0 { s!(a, oo + t * os + h * sd + j, ov * scale); }
            u += 1;
        }
        bar();
        tb += 16;
    }
    all4!(|s, i| { let v = *s; s!(a, sb + (4 * i) * sd, v.x); s!(a, sb + (4 * i + 1) * sd, v.y); s!(a, sb + (4 * i + 2) * sd, v.z); s!(a, sb + (4 * i + 3) * sd, v.w); });
}

/// gdn_p5 with 512 threads per workgroup: 8 state values per thread (16 row-groups, 4-step shuffle reduction),
/// halving the per-token serial work per thread. Same grid / push constants.
#[spirv(compute(threads(512)))]
pub fn gdn_p6(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [Vec4; 512], #[spirv(workgroup)] ks: &mut [Vec4; 512],
    #[spirv(workgroup)] vsh: &mut [f32; 512], #[spirv(workgroup)] gsh: &mut [f32; 16], #[spirv(workgroup)] bsh: &mut [f32; 16]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, l) = (wg.x, lid.x); let rq = l & 15; let j = wg.y * 32 + (l >> 4); let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd + rq * 8 * sd + j;
    let ld = |i: u32| Vec4::new(g!(a, sb + (4 * i) * sd), g!(a, sb + (4 * i + 1) * sd), g!(a, sb + (4 * i + 2) * sd), g!(a, sb + (4 * i + 3) * sd));
    let (mut s0, mut s1) = (ld(0), ld(1));
    macro_rules! all4 { (|$s:ident, $i:ident| $b:block) => {
        { let $i = 0u32; let $s = &mut s0; $b } { let $i = 1u32; let $s = &mut s1; $b } } }
    let red8 = |v: f32| -> f32 { let v = v + sx(v, 1); let v = v + sx(v, 2); v + sx(v, 4) };
    let dt = g!(prm, dto + h); let aa = g!(prm, ao + h);
    let mut tb = 0u32;
    while tb < ntok {
        unroll!([0], |c| {
            let idx = l + 512 * c; let tk = idx >> 5; let d4 = idx & 31; let t = tb + tk;
            let cb = (co + t.min(ntok - 1) * cs) / 4;
            s!(qs, idx, g!(a4, cb + (kh * sd) / 4 + d4)); s!(ks, idx, g!(a4, cb + (qk + kh * sd) / 4 + d4));
            s!(vsh, idx, g!(a, co + t.min(ntok - 1) * cs + 2 * qk + h * sd + wg.y * 32 + d4));
        });
        if l < 16 {
            let t = (tb + l).min(ntok - 1);
            s!(bsh, l, sigm(g!(a, bo + t * bas + h)));
            let x = g!(a, alo + t * bas + h) + dt;
            s!(gsh, l, (aa * (if x > 20.0 { x } else { (1.0 + x.exp()).ln() })).exp());
        }
        bar();
        let nb = (ntok - tb).min(16);
        let mut u = 0u32;
        while u < nb {
            let t = tb + u; let cb = co + t * cs;
            let _ = cb;
            let vj = g!(vsh, u * 32 + (l >> 4));
            let beta = g!(bsh, u);
            let eg = g!(gsh, u);
            let kb = u * 32 + rq * 2;
            // o = q.(eg*S + delta*k) = eg*(S.q) + delta*(k.q): the three dot products share one reduction round
            let (mut sk, mut sq, mut kq) = (Vec4::ZERO, Vec4::ZERO, Vec4::ZERO);
            all4!(|s, i| { let kk = g!(ks, kb + i); let qq = g!(qs, kb + i); sk += *s * kk; sq += *s * qq; kq += kk * qq; });
            let (mut r1, mut r2, mut r3) = (sk.dot(Vec4::ONE), sq.dot(Vec4::ONE), kq.dot(Vec4::ONE));
            r1 += sx(r1, 1); r2 += sx(r2, 1); r3 += sx(r3, 1);
            r1 += sx(r1, 2); r2 += sx(r2, 2); r3 += sx(r3, 2);
            r1 += sx(r1, 4); r2 += sx(r2, 4); r3 += sx(r3, 4);
            r1 += sx(r1, 8); r2 += sx(r2, 8); r3 += sx(r3, 8);
            let delta = (vj - eg * r1) * beta;
            let ov = eg * r2 + delta * r3;
            all4!(|s, i| { *s = *s * eg + g!(ks, kb + i) * delta; });
            if rq == 0 { s!(a, oo + t * os + h * sd + j, ov * scale); }
            u += 1;
        }
        bar();
        tb += 16;
    }
    all4!(|s, i| { let v = *s; s!(a, sb + (4 * i) * sd, v.x); s!(a, sb + (4 * i + 1) * sd, v.y); s!(a, sb + (4 * i + 2) * sd, v.z); s!(a, sb + (4 * i + 3) * sd, v.w); });
}

/// gdn_p6 with 1024 threads per workgroup: one warp per value column, 4 state values per thread.
#[spirv(compute(threads(1024)))]
pub fn gdn_p7(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] qs: &mut [Vec4; 512], #[spirv(workgroup)] ks: &mut [Vec4; 512],
    #[spirv(workgroup)] vsh: &mut [f32; 512], #[spirv(workgroup)] gsh: &mut [f32; 16], #[spirv(workgroup)] bsh: &mut [f32; 16]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (ntok, co, cs, bo, alo, bas, dto, ao) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (so, oo, os, nk, sd, qk) = (pu(pc, 8), pu(pc, 9), pu(pc, 10), pu(pc, 11), pu(pc, 12), pu(pc, 13));
    let (h, l) = (wg.x, lid.x); let rq = l & 31; let j = wg.y * 32 + (l >> 5); let kh = h % nk;
    let scale = 1.0 / (sd as f32).sqrt();
    let sb = so + h * sd * sd + rq * 4 * sd + j;
    let ld = |i: u32| Vec4::new(g!(a, sb + (4 * i) * sd), g!(a, sb + (4 * i + 1) * sd), g!(a, sb + (4 * i + 2) * sd), g!(a, sb + (4 * i + 3) * sd));
    let mut s0 = ld(0);
    macro_rules! all4 { (|$s:ident, $i:ident| $b:block) => {
        { let $i = 0u32; let $s = &mut s0; $b } } }
    let red8 = |v: f32| -> f32 { let v = v + sx(v, 1); let v = v + sx(v, 2); v + sx(v, 4) };
    let dt = g!(prm, dto + h); let aa = g!(prm, ao + h);
    let mut tb = 0u32;
    while tb < ntok {
        if l < 512 { unroll!([0], |c| {
            let idx = l + 512 * c; let tk = idx >> 5; let d4 = idx & 31; let t = tb + tk;
            let cb = (co + t.min(ntok - 1) * cs) / 4;
            s!(qs, idx, g!(a4, cb + (kh * sd) / 4 + d4)); s!(ks, idx, g!(a4, cb + (qk + kh * sd) / 4 + d4));
            s!(vsh, idx, g!(a, co + t.min(ntok - 1) * cs + 2 * qk + h * sd + wg.y * 32 + d4));
        }); }
        if l < 16 {
            let t = (tb + l).min(ntok - 1);
            s!(bsh, l, sigm(g!(a, bo + t * bas + h)));
            let x = g!(a, alo + t * bas + h) + dt;
            s!(gsh, l, (aa * (if x > 20.0 { x } else { (1.0 + x.exp()).ln() })).exp());
        }
        bar();
        let nb = (ntok - tb).min(16);
        let mut u = 0u32;
        while u < nb {
            let t = tb + u; let cb = co + t * cs;
            let _ = cb;
            let vj = g!(vsh, u * 32 + (l >> 5));
            let beta = g!(bsh, u);
            let eg = g!(gsh, u);
            let kb = u * 32 + rq;
            // o = q.(eg*S + delta*k) = eg*(S.q) + delta*(k.q): the three dot products share one reduction round
            let (mut sk, mut sq, mut kq) = (Vec4::ZERO, Vec4::ZERO, Vec4::ZERO);
            all4!(|s, i| { let kk = g!(ks, kb + i); let qq = g!(qs, kb + i); sk += *s * kk; sq += *s * qq; kq += kk * qq; });
            let (mut r1, mut r2, mut r3) = (sk.dot(Vec4::ONE), sq.dot(Vec4::ONE), kq.dot(Vec4::ONE));
            r1 += sx(r1, 1); r2 += sx(r2, 1); r3 += sx(r3, 1);
            r1 += sx(r1, 2); r2 += sx(r2, 2); r3 += sx(r3, 2);
            r1 += sx(r1, 4); r2 += sx(r2, 4); r3 += sx(r3, 4);
            r1 += sx(r1, 8); r2 += sx(r2, 8); r3 += sx(r3, 8);
            r1 += sx(r1, 16); r2 += sx(r2, 16); r3 += sx(r3, 16);
            let delta = (vj - eg * r1) * beta;
            let ov = eg * r2 + delta * r3;
            all4!(|s, i| { *s = *s * eg + g!(ks, kb + i) * delta; });
            if rq == 0 { s!(a, oo + t * os + h * sd + j, ov * scale); }
            u += 1;
        }
        bar();
        tb += 16;
    }
    all4!(|s, i| { let v = *s; s!(a, sb + (4 * i) * sd, v.x); s!(a, sb + (4 * i + 1) * sd, v.y); s!(a, sb + (4 * i + 2) * sd, v.z); s!(a, sb + (4 * i + 3) * sd, v.w); });
}

#[inline(always)] fn comp4(v: UVec4, i: u32) -> u32 { if i == 0 { v.x } else if i == 1 { v.y } else if i == 2 { v.z } else { v.w } }
#[inline(always)] fn codes2(word: u32, sh: u32) -> Vec4 {
    Vec4::new(((word >> sh) & 3) as f32, ((word >> (sh + 2)) & 3) as f32, ((word >> (sh + 4)) & 3) as f32, ((word >> (sh + 6)) & 3) as f32)
}
/// NVIDIA decode matvec (1 token): $R rows per workgroup of $T threads; each lane keeps the 16 x values of its
/// word in registers and reuses them for all $R rows (x:weight L2 traffic 64B : 4R B instead of 64 : 32 in
/// matvec_c8). Warp-shuffle row reductions. Same push constants / bindings / math per term as matvec_c8.
macro_rules! mv_n { ($name:ident, $T:literal, $R:literal, [$($r:literal),*]) => {
#[spirv(compute(threads($T)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (m, k, xo, yo, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 3), pu(pc, 5), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let ln = l & 31;
    let r0 = (wg.y * nwg.x + wg.x) * $R;
    let wpr = k >> 4; let bpr = k >> 7;
    let xb = xo / 4;
    let mut acc = [0.0f32; $R];
    let mut wi = l;
    while wi < wpr {
        let x0 = g!(a4, xb + wi * 4); let x1 = g!(a4, xb + wi * 4 + 1); let x2 = g!(a4, xb + wi * 4 + 2); let x3 = g!(a4, xb + wi * 4 + 3);
        let xsum = (x0 + x1 + x2 + x3).dot(Vec4::ONE);
        unroll!([$($r),*], |rr| {
            let row = (r0 + rr).min(m - 1);
            let word = g!(wq, row * wpr + wi);
            let d = g!(wsc, row * bpr + (wi >> 3));
            let tt = codes2(word, 0).dot(x0) + codes2(word, 8).dot(x1) + codes2(word, 16).dot(x2) + codes2(word, 24).dot(x3);
            s!(acc, rr, g!(acc, rr) + (tt - xsum) * d);
        });
        wi += $T;
    }
    unroll!([$($r),*], |rr| {
        let mut v = g!(acc, rr);
        v += sx(v, 16); v += sx(v, 8); v += sx(v, 4); v += sx(v, 2); v += sx(v, 1);
        if ln == 0 { s!(red, w * $R + rr, v); }
    });
    bar();
    if l < $R && r0 + l < m {
        let mut v = 0.0f32; let mut j = 0u32; while j < $T / 32 { v += g!(red, j * $R + l); j += 1; }
        let yi = yo + r0 + l; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
    }
}
}; }
mv_n!(matvec_n8a, 64, 8, [0,1,2,3,4,5,6,7]);
mv_n!(matvec_n16a, 64, 16, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
mv_n!(matvec_n16b, 128, 16, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
mv_n!(matvec_n32b, 128, 32, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]);
mv_n!(matvec_n8c, 256, 8, [0,1,2,3,4,5,6,7]);

/// matvec_n with 16-byte weight loads: each lane takes 4 consecutive words (64 weights, one scale group half) per
/// row, so every row load is a uvec4 and 4x more bytes are in flight per lane. x (16 Vec4) is held in registers.
macro_rules! mv_w { ($name:ident, $T:literal, $R:literal, [$($r:literal),*]) => {
#[spirv(compute(threads($T)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq4: &[UVec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (m, k, xo, yo, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 3), pu(pc, 5), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let ln = l & 31;
    let r0 = (wg.y * nwg.x + wg.x) * $R;
    let wpr = k >> 4; let q4 = wpr >> 2; let bpr = k >> 7;
    let xb = xo / 4;
    let mut acc = [0.0f32; $R];
    let mut qi = l;
    while qi < q4 {
        let mut xv = [Vec4::ZERO; 16]; let mut xs = [0.0f32; 4];
        unroll!([0, 1, 2, 3], |q| {
            let b = xb + (qi * 4 + q) * 4;
            let x0 = g!(a4, b); let x1 = g!(a4, b + 1); let x2 = g!(a4, b + 2); let x3 = g!(a4, b + 3);
            s!(xv, q * 4, x0); s!(xv, q * 4 + 1, x1); s!(xv, q * 4 + 2, x2); s!(xv, q * 4 + 3, x3);
            s!(xs, q, (x0 + x1 + x2 + x3).dot(Vec4::ONE));
        });
        let xsum = g!(xs, 0) + g!(xs, 1) + g!(xs, 2) + g!(xs, 3);
        unroll!([$($r),*], |rr| {
            let row = (r0 + rr).min(m - 1);
            let wv = g!(wq4, row * q4 + qi);
            let d = g!(wsc, row * bpr + (qi >> 1));
            let mut tt = 0.0f32;
            unroll!([0, 1, 2, 3], |q| {
                let word = comp4(wv, q);
                tt += codes2(word, 0).dot(g!(xv, q * 4)) + codes2(word, 8).dot(g!(xv, q * 4 + 1)) + codes2(word, 16).dot(g!(xv, q * 4 + 2)) + codes2(word, 24).dot(g!(xv, q * 4 + 3));
            });
            s!(acc, rr, g!(acc, rr) + (tt - xsum) * d);
        });
        qi += $T;
    }
    unroll!([$($r),*], |rr| {
        let mut v = g!(acc, rr);
        v += sx(v, 16); v += sx(v, 8); v += sx(v, 4); v += sx(v, 2); v += sx(v, 1);
        if ln == 0 { s!(red, w * $R + rr, v); }
    });
    bar();
    if l < $R && r0 + l < m {
        let mut v = 0.0f32; let mut j = 0u32; while j < $T / 32 { v += g!(red, j * $R + l); j += 1; }
        let yi = yo + r0 + l; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
    }
}
}; }
mv_w!(matvec_w4a, 32, 4, [0,1,2,3]);
mv_w!(matvec_w8a, 64, 8, [0,1,2,3,4,5,6,7]);
mv_w!(matvec_w4b, 64, 4, [0,1,2,3]);
mv_w!(matvec_w2a, 32, 2, [0,1]);

/// DIAGNOSTIC ONLY (wrong math): matvec_n8a load pattern with the 2-bit decode removed, to measure the memory roofline.
macro_rules! mv_z { ($name:ident, $T:literal, $R:literal, [$($r:literal),*]) => {
#[spirv(compute(threads($T)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (m, k, xo, yo, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 3), pu(pc, 5), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let ln = l & 31;
    let r0 = (wg.y * nwg.x + wg.x) * $R;
    let wpr = k >> 4; let bpr = k >> 7;
    let xb = xo / 4;
    let mut acc = [0.0f32; $R];
    let mut wi = l;
    while wi < wpr {
        let x0 = g!(a4, xb + wi * 4); let x1 = g!(a4, xb + wi * 4 + 1); let x2 = g!(a4, xb + wi * 4 + 2); let x3 = g!(a4, xb + wi * 4 + 3);
        let xsum = (x0 + x1 + x2 + x3).dot(Vec4::ONE);
        unroll!([$($r),*], |rr| {
            let row = (r0 + rr).min(m - 1);
            let word = g!(wq, row * wpr + wi);
            let d = g!(wsc, row * bpr + (wi >> 3));
            let tt = (word as f32) * x0.x;
            s!(acc, rr, g!(acc, rr) + (tt - xsum) * d);
        });
        wi += $T;
    }
    unroll!([$($r),*], |rr| {
        let mut v = g!(acc, rr);
        v += sx(v, 16); v += sx(v, 8); v += sx(v, 4); v += sx(v, 2); v += sx(v, 1);
        if ln == 0 { s!(red, w * $R + rr, v); }
    });
    bar();
    if l < $R && r0 + l < m {
        let mut v = 0.0f32; let mut j = 0u32; while j < $T / 32 { v += g!(red, j * $R + l); j += 1; }
        let yi = yo + r0 + l; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
    }
}
}; }
mv_z!(matvec_z8a, 64, 8, [0,1,2,3,4,5,6,7]);

/// 4x8-bit packed signed dot product (OpSDot, SPV_KHR_integer_dot_product): sum of the 4 byte products.
#[inline(always)] fn sdot4(a: u32, b: u32) -> i32 {
    let mut r = 0i32;
    unsafe { asm!("%i32 = OpTypeInt 32 1", "%v = OpSDot %i32 {a} {b} PackedVectorFormat4x8Bit", "OpStore {r} %v", a = in(reg) a, b = in(reg) b, r = in(reg) &mut r); }
    r
}
#[inline(always)] fn q8b(v: Vec4, inv: f32) -> u32 {
    ((rnd(v.x * inv) as i32) as u32 & 0xff) | (((rnd(v.y * inv) as i32) as u32 & 0xff) << 8) | (((rnd(v.z * inv) as i32) as u32 & 0xff) << 16) | (((rnd(v.w * inv) as i32) as u32 & 0xff) << 24)
}
/// NVIDIA single-token decode matvec with int8 dot products: each lane quantizes its 16 x values in-register
/// (q8, one scale per 128-K group = 8 adjacent lanes, same rounding as quant_x / the prefill GEMMs), then per row
/// does 4 OpSDot of the signed weight codes {-1,0,1,2}-1... ((sp|0x80808080)-0x01010101)^0x80808080 against the
/// K-transposed q words. acc += float(dot) * dw * dx. $R rows / workgroup of $T threads, shuffle reductions.
macro_rules! mv_d { ($name:ident, $T:literal, $R:literal, [$($r:literal),*]) => {
#[spirv(compute(threads($T)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (m, k, xo, yo, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 3), pu(pc, 5), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let ln = l & 31;
    let r0 = (wg.y * nwg.x + wg.x) * $R;
    let wpr = k >> 4; let bpr = k >> 7;
    let xb = xo / 4;
    let mut acc = [0.0f32; $R];
    let mut wi = l;
    while wi < wpr {
        let x0 = g!(a4, xb + wi * 4); let x1 = g!(a4, xb + wi * 4 + 1); let x2 = g!(a4, xb + wi * 4 + 2); let x3 = g!(a4, xb + wi * 4 + 3);
        let mut mx = x0.abs().max(x1.abs()).max(x2.abs()).max(x3.abs()).max_element();
        mx = mx.max(sx(mx, 1)); mx = mx.max(sx(mx, 2)); mx = mx.max(sx(mx, 4));
        let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 }; let dx = mx / 127.0;
        let q0 = q8b(Vec4::new(x0.x, x1.x, x2.x, x3.x), inv); let q1 = q8b(Vec4::new(x0.y, x1.y, x2.y, x3.y), inv);
        let q2 = q8b(Vec4::new(x0.z, x1.z, x2.z, x3.z), inv); let q3 = q8b(Vec4::new(x0.w, x1.w, x2.w, x3.w), inv);
        unroll!([$($r),*], |rr| {
            let row = (r0 + rr).min(m - 1);
            let word = g!(wq, row * wpr + wi);
            let dw = g!(wsc, row * bpr + (wi >> 3));
            let sg = |b: u32| -> u32 { let sp = (word >> (2 * b)) & 0x0303_0303; ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080 };
            let dot = sdot4(sg(0), q0) + sdot4(sg(1), q1) + sdot4(sg(2), q2) + sdot4(sg(3), q3);
            s!(acc, rr, g!(acc, rr) + (dot as f32) * dw * dx);
        });
        wi += $T;
    }
    unroll!([$($r),*], |rr| {
        let mut v = g!(acc, rr);
        v += sx(v, 16); v += sx(v, 8); v += sx(v, 4); v += sx(v, 2); v += sx(v, 1);
        if ln == 0 { s!(red, w * $R + rr, v); }
    });
    bar();
    if l < $R && r0 + l < m {
        let mut v = 0.0f32; let mut j = 0u32; while j < $T / 32 { v += g!(red, j * $R + l); j += 1; }
        let yi = yo + r0 + l; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
    }
}
}; }
mv_d!(matvec_d8a, 64, 8, [0,1,2,3,4,5,6,7]);
mv_d!(matvec_d16b, 128, 16, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]);
mv_d!(matvec_d8c, 256, 8, [0,1,2,3,4,5,6,7]);
mv_d!(matvec_d4a, 32, 4, [0,1,2,3]);

/// Decode / verify (ntok < 16) fused ssm_beta + ssm_alpha projection: one 256-thread workgroup per output row
/// (rows 0..nv beta, nv..2nv alpha), grid (2*nv, ntok). Replaces two 64-thread-per-row matvec_f32 dispatches.
/// pc: 0 nv, 1 k, 2 xo, 3 xs, 4 yb, 5 ya, 6 ys, 7 wb, 8 wa
#[spirv(compute(threads(256)))]
pub fn matvec_ab1(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 8]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (nv, k, xo, xs, yb, ya, ys, wb, wa) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7), pu(pc, 8));
    let (r, t, l) = (wg.x, wg.y, lid.x);
    let al = r >= nv; let row = if al { r - nv } else { r };
    let wo = (if al { wa } else { wb }) + row * k; let xb = xo + t * xs;
    let mut sm = 0.0f32;
    let mut i = l; while i < k { sm += g!(prm, wo + i) * g!(a, xb + i); i += 256; }
    sm += sx(sm, 16); sm += sx(sm, 8); sm += sx(sm, 4); sm += sx(sm, 2); sm += sx(sm, 1);
    if l & 31 == 0 { s!(red, l >> 5, sm); }
    bar();
    if l == 0 {
        let v = g!(red, 0) + g!(red, 1) + g!(red, 2) + g!(red, 3) + g!(red, 4) + g!(red, 5) + g!(red, 6) + g!(red, 7);
        s!(a, (if al { ya } else { yb }) + t * ys + row, v);
    }
}

/// Two-pass argmax (same result as argmax: max value, lowest index on ties). Pass 1: grid (256), each workgroup
/// scans one contiguous chunk and writes (value, index bits) to scr[2*wg], scr[2*wg+1]. pc: 0 off, 1 n, 2 scr
#[spirv(compute(threads(256)))]
pub fn argmax_p1(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(workgroup)] bv: &mut [f32; 256], #[spirv(workgroup)] bi: &mut [u32; 256]) {
    let (off, n, scr, l) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), lid.x);
    let ch = n.div_ceil(256); let s0 = wg.x * ch; let s1 = (s0 + ch).min(n);
    let mut best = -3.0e38f32; let mut idx = 0xffff_ffffu32;
    let mut i = s0 + l; while i < s1 { let v = g!(a, off + i); if v > best { best = v; idx = i; } i += 256; }
    s!(bv, l, best); s!(bi, l, idx); bar();
    unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| {
        let st = 1u32 << sh;
        if l < st { let o = g!(bv, l + st); let oi = g!(bi, l + st); if o > g!(bv, l) || (o == g!(bv, l) && oi < g!(bi, l)) { s!(bv, l, o); s!(bi, l, oi); } }
        bar();
    });
    if l == 0 { s!(a, scr + 2 * wg.x, g!(bv, 0)); s!(a, scr + 2 * wg.x + 1, f32::from_bits(g!(bi, 0))); }
}
/// Pass 2: one workgroup reduces the 256 partials; writes the index bits to dst. pc: 0 scr, 1 dst
#[spirv(compute(threads(256)))]
pub fn argmax_p2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(workgroup)] bv: &mut [f32; 256], #[spirv(workgroup)] bi: &mut [u32; 256]) {
    let (scr, dst, l) = (pu(pc, 0), pu(pc, 1), lid.x);
    s!(bv, l, g!(a, scr + 2 * l)); s!(bi, l, g!(a, scr + 2 * l + 1).to_bits()); bar();
    unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| {
        let st = 1u32 << sh;
        if l < st { let o = g!(bv, l + st); let oi = g!(bi, l + st); if o > g!(bv, l) || (o == g!(bv, l) && oi < g!(bi, l)) { s!(bv, l, o); s!(bi, l, oi); } }
        bar();
    });
    if l == 0 { s!(a, dst, f32::from_bits(g!(bi, 0))); }
}

/// nhq with ~4x fewer workgroup barriers (decode: the fused norm+Hadamard is latency-bound). Bit-identical to nhq:
/// the norm tree adds the same pairs in the same stage order (128..1; xor partners = l+st for the lanes that keep
/// the sum), and the Hadamard butterflies run in the same stage order (h = 1..16 by subgroup shuffle, 32..128 via
/// double-buffered shared memory, 256/512 in registers). Same push constants / grid / outputs as nhq.
#[spirv(compute(threads(256)))]
pub fn nhq2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] v: &mut [f32; 1024], #[spirv(workgroup)] vb: &mut [f32; 1024], #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (n, io, its, xno, oo, ots, wo, so) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (qo, sco, donorm) = (pu(pc, 8), pu(pc, 9), pu(pc, 10)); let eps = pf(pc, 11);
    let (blk, t, l) = (wg.x, wg.y, lid.x);
    let ib = io + t * its;
    let mut sc = 1.0f32;
    if donorm != 0 {
        let mut sm = 0.0f32;
        let mut i = l; while i < n { let x = g!(a, ib + i); sm += x * x; i += 256; }
        // stages 128, 64, 32 through shared memory (ping-pong halves of red), then 16..1 in-warp
        s!(red, l, sm); bar();
        sm = g!(red, l) + g!(red, l ^ 128); s!(red, 256 + l, sm); bar();
        sm = g!(red, 256 + l) + g!(red, 256 + (l ^ 64)); s!(red, l, sm); bar();
        sm = g!(red, l) + g!(red, l ^ 32);
        sm = sm + sx(sm, 16); sm = sm + sx(sm, 8); sm = sm + sx(sm, 4); sm = sm + sx(sm, 2); sm = sm + sx(sm, 1);
        bar();
        if l == 0 { s!(red, 0, sm); }
        bar();
        sc = 1.0 / (g!(red, 0) / n as f32 + eps).sqrt();
    }
    let mut x = [0.0f32; 4];
    unroll!([0, 1, 2, 3], |q| {
        let gi = blk * 1024 + l + 256 * q;
        let mut y = g!(a, ib + gi);
        if donorm != 0 { y = y * sc * g!(prm, wo + gi); s!(a, xno + t * n + gi, y); }
        if so != u32::MAX { y *= g!(prm, so + gi); }
        s!(x, q, y);
    });
    // h = 1..16: partner element e ^ h lives in lane l ^ h (same q)
    unroll!([1, 2, 4, 8, 16], |h| {
        let lo = (l & h) == 0;
        unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = sx(m, h); s!(x, q, if lo { m + o } else { o - m }); });
    });
    // h = 32, 64, 128: across warps through shared memory (ping-pong buffers, one barrier per stage)
    unroll!([0, 1, 2, 3], |q| { s!(v, l + 256 * q, g!(x, q)); }); bar();
    { let lo = (l & 32) == 0; unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = g!(v, (l ^ 32) + 256 * q); s!(x, q, if lo { m + o } else { o - m }); }); }
    unroll!([0, 1, 2, 3], |q| { s!(vb, l + 256 * q, g!(x, q)); }); bar();
    { let lo = (l & 64) == 0; unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = g!(vb, (l ^ 64) + 256 * q); s!(x, q, if lo { m + o } else { o - m }); }); }
    unroll!([0, 1, 2, 3], |q| { s!(v, l + 256 * q, g!(x, q)); }); bar();
    { let lo = (l & 128) == 0; unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = g!(v, (l ^ 128) + 256 * q); s!(x, q, if lo { m + o } else { o - m }); }); }
    // h = 256 (q bit 0), 512 (q bit 1): in registers
    { let (a0, a1, a2, a3) = (g!(x, 0), g!(x, 1), g!(x, 2), g!(x, 3));
      let (b0, b1, b2, b3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
      s!(x, 0, b0 + b2); s!(x, 2, b0 - b2); s!(x, 1, b1 + b3); s!(x, 3, b1 - b3); }
    unroll!([0, 1, 2, 3], |q| { s!(vb, l + 256 * q, g!(x, q)); }); bar();
    let b0 = l * 4;
    let x0 = g!(vb, b0) * 0.03125; let x1 = g!(vb, b0 + 1) * 0.03125; let x2 = g!(vb, b0 + 2) * 0.03125; let x3 = g!(vb, b0 + 3) * 0.03125;
    let ob = oo + t * ots + blk * 1024 + b0;
    s!(a, ob, x0); s!(a, ob + 1, x1); s!(a, ob + 2, x2); s!(a, ob + 3, x3);
    let mut mx = x0.abs().max(x1.abs()).max(x2.abs()).max(x3.abs());
    mx = mx.max(sx(mx, 1)); mx = mx.max(sx(mx, 2)); mx = mx.max(sx(mx, 4)); mx = mx.max(sx(mx, 8)); mx = mx.max(sx(mx, 16));
    let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
    let pb = (l >> 2) * 16 + (l & 3);
    let p0 = g!(vb, pb) * 0.03125; let p1 = g!(vb, pb + 4) * 0.03125; let p2 = g!(vb, pb + 8) * 0.03125; let p3 = g!(vb, pb + 12) * 0.03125;
    let q0 = (rnd(p0 * inv) as i32) as u32 & 0xff; let q1 = (rnd(p1 * inv) as i32) as u32 & 0xff;
    let q2 = (rnd(p2 * inv) as i32) as u32 & 0xff; let q3 = (rnd(p3 * inv) as i32) as u32 & 0xff;
    s!(au, qo + t * (n / 4) + blk * 256 + l, q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
    if (l & 31) == 0 { s!(a, sco + t * (n / 128) + blk * 8 + (l >> 5), mx / 127.0); }
}

/// FZ1 (NV decode, opt-in BONSAI_FUSE): nhq2 whose input is silu(a[io..]) * a[bo..] computed on load, replacing the
/// ewise(mode 1) dispatch + barrier before the down projection's Hadamard. Same expression as ewise, so the f32 inputs
/// are bit-identical; use with donorm = 0 only (the norm pass would read the raw gate).
#[spirv(compute(threads(256)))]
pub fn nhq2_sw(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] v: &mut [f32; 1024], #[spirv(workgroup)] vb: &mut [f32; 1024], #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (n, io, its, xno, oo, ots, wo, so) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (qo, sco, donorm) = (pu(pc, 8), pu(pc, 9), pu(pc, 10)); let eps = pf(pc, 11); let bo = pu(pc, 12);
    let (blk, t, l) = (wg.x, wg.y, lid.x);
    let ib = io + t * its;
    let mut sc = 1.0f32;
    if donorm != 0 {
        let mut sm = 0.0f32;
        let mut i = l; while i < n { let x = g!(a, ib + i); sm += x * x; i += 256; }
        // stages 128, 64, 32 through shared memory (ping-pong halves of red), then 16..1 in-warp
        s!(red, l, sm); bar();
        sm = g!(red, l) + g!(red, l ^ 128); s!(red, 256 + l, sm); bar();
        sm = g!(red, 256 + l) + g!(red, 256 + (l ^ 64)); s!(red, l, sm); bar();
        sm = g!(red, l) + g!(red, l ^ 32);
        sm = sm + sx(sm, 16); sm = sm + sx(sm, 8); sm = sm + sx(sm, 4); sm = sm + sx(sm, 2); sm = sm + sx(sm, 1);
        bar();
        if l == 0 { s!(red, 0, sm); }
        bar();
        sc = 1.0 / (g!(red, 0) / n as f32 + eps).sqrt();
    }
    let mut x = [0.0f32; 4];
    unroll!([0, 1, 2, 3], |q| {
        let gi = blk * 1024 + l + 256 * q;
        let mut y = { let xv = g!(a, ib + gi); (xv / (1.0 + (-xv).exp())) * g!(a, bo + gi) };
        if donorm != 0 { y = y * sc * g!(prm, wo + gi); s!(a, xno + t * n + gi, y); }
        if so != u32::MAX { y *= g!(prm, so + gi); }
        s!(x, q, y);
    });
    // h = 1..16: partner element e ^ h lives in lane l ^ h (same q)
    unroll!([1, 2, 4, 8, 16], |h| {
        let lo = (l & h) == 0;
        unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = sx(m, h); s!(x, q, if lo { m + o } else { o - m }); });
    });
    // h = 32, 64, 128: across warps through shared memory (ping-pong buffers, one barrier per stage)
    unroll!([0, 1, 2, 3], |q| { s!(v, l + 256 * q, g!(x, q)); }); bar();
    { let lo = (l & 32) == 0; unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = g!(v, (l ^ 32) + 256 * q); s!(x, q, if lo { m + o } else { o - m }); }); }
    unroll!([0, 1, 2, 3], |q| { s!(vb, l + 256 * q, g!(x, q)); }); bar();
    { let lo = (l & 64) == 0; unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = g!(vb, (l ^ 64) + 256 * q); s!(x, q, if lo { m + o } else { o - m }); }); }
    unroll!([0, 1, 2, 3], |q| { s!(v, l + 256 * q, g!(x, q)); }); bar();
    { let lo = (l & 128) == 0; unroll!([0, 1, 2, 3], |q| { let m = g!(x, q); let o = g!(v, (l ^ 128) + 256 * q); s!(x, q, if lo { m + o } else { o - m }); }); }
    // h = 256 (q bit 0), 512 (q bit 1): in registers
    { let (a0, a1, a2, a3) = (g!(x, 0), g!(x, 1), g!(x, 2), g!(x, 3));
      let (b0, b1, b2, b3) = (a0 + a1, a0 - a1, a2 + a3, a2 - a3);
      s!(x, 0, b0 + b2); s!(x, 2, b0 - b2); s!(x, 1, b1 + b3); s!(x, 3, b1 - b3); }
    unroll!([0, 1, 2, 3], |q| { s!(vb, l + 256 * q, g!(x, q)); }); bar();
    let b0 = l * 4;
    let x0 = g!(vb, b0) * 0.03125; let x1 = g!(vb, b0 + 1) * 0.03125; let x2 = g!(vb, b0 + 2) * 0.03125; let x3 = g!(vb, b0 + 3) * 0.03125;
    let ob = oo + t * ots + blk * 1024 + b0;
    s!(a, ob, x0); s!(a, ob + 1, x1); s!(a, ob + 2, x2); s!(a, ob + 3, x3);
    let mut mx = x0.abs().max(x1.abs()).max(x2.abs()).max(x3.abs());
    mx = mx.max(sx(mx, 1)); mx = mx.max(sx(mx, 2)); mx = mx.max(sx(mx, 4)); mx = mx.max(sx(mx, 8)); mx = mx.max(sx(mx, 16));
    let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
    let pb = (l >> 2) * 16 + (l & 3);
    let p0 = g!(vb, pb) * 0.03125; let p1 = g!(vb, pb + 4) * 0.03125; let p2 = g!(vb, pb + 8) * 0.03125; let p3 = g!(vb, pb + 12) * 0.03125;
    let q0 = (rnd(p0 * inv) as i32) as u32 & 0xff; let q1 = (rnd(p1 * inv) as i32) as u32 & 0xff;
    let q2 = (rnd(p2 * inv) as i32) as u32 & 0xff; let q3 = (rnd(p3 * inv) as i32) as u32 & 0xff;
    s!(au, qo + t * (n / 4) + blk * 256 + l, q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
    if (l & 31) == 0 { s!(a, sco + t * (n / 128) + blk * 8 + (l >> 5), mx / 127.0); }
}

/// FZ2 (NV decode, opt-in BONSAI_FUSE): causal conv (+ state shift) fused with the q/k l2-norm (rmsnorm mode 2).
/// Workgroups r < nrows: rmsnorm's layout (256 threads, row r = n channels at r*n); thread l < n computes conv for its
/// channel exactly as `conv`, then the same sum-of-squares tree as `rmsnorm`. Workgroups r >= nrows: conv only for the
/// remaining (v) channels, 256 per workgroup. ntok = 1. pc: cd, n, nrows, io, oo, wo, so, k, eps.
#[spirv(compute(threads(256)))]
pub fn conv_norm(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 256]) {
    let (cd, n, nrows, io, oo, wo, so, k) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let eps = pf(pc, 8);
    let (r, l) = (wg.x, lid.x);
    let qk = nrows * n;
    let ch = if r < nrows { r * n + l } else { qk + (r - nrows) * 256 + l };
    let live = if r < nrows { l < n } else { ch < cd };
    let mut y = 0.0f32;
    if live {
        let mut st = [0.0f32; 3];
        let mut i = 0; while i < 3 { s!(st, i, if i < k - 1 { g!(a, so + i * cd + ch) } else { 0.0 }); i += 1; }
        let x = g!(a, io + ch);
        let mut sm = 0.0f32;
        i = 0; while i < k - 1 { sm += g!(st, i) * g!(prm, wo + ch * k + i); i += 1; }
        sm += x * g!(prm, wo + ch * k + k - 1);
        y = sm / (1.0 + (-sm).exp());
        i = 0; while i + 1 < k - 1 { s!(st, i, g!(st, i + 1)); i += 1; }
        s!(st, k - 2, x);
        i = 0; while i < k - 1 { s!(a, so + i * cd + ch, g!(st, i)); i += 1; }
    }
    if r < nrows {
        let sm = if live { y * y } else { 0.0 };
        s!(red, l, sm); bar();
        unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
        let tot = g!(red, 0);
        let sc = 1.0 / tot.sqrt().max(eps);
        if live { s!(a, oo + ch, y * sc); }
    } else if live { s!(a, oo + ch, y); }
}

/// matvec_d8a for 2..T tokens (MTP verify): each token's x is quantized in-register exactly as in matvec_d8a and
/// every weight word / scale load is shared by the T tokens, so per-token results are bit-identical to
/// matvec_d8a (same per-lane partials, same shuffle/shared reduction order). Grid (gx, gy, ceil(ntok/T)).
macro_rules! mv_dt { ($name:ident, $T:literal, [$($t:literal),*]) => {
#[spirv(compute(threads(64)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let ln = l & 31;
    let r0 = (wg.y * nwg.x + wg.x) * 8;
    let t0 = wg.z * $T;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut acc = [0.0f32; 8 * $T];
    let mut wi = l;
    while wi < wpr {
        let mut qw = [0u32; 4 * $T]; let mut dxs = [0.0f32; $T];
        unroll!([$($t),*], |tt| {
            let tk = (t0 + tt).min(ntok - 1); let xb = (xo + tk * xs) / 4;
            let x0 = g!(a4, xb + wi * 4); let x1 = g!(a4, xb + wi * 4 + 1); let x2 = g!(a4, xb + wi * 4 + 2); let x3 = g!(a4, xb + wi * 4 + 3);
            let mut mx = x0.abs().max(x1.abs()).max(x2.abs()).max(x3.abs()).max_element();
            mx = mx.max(sx(mx, 1)); mx = mx.max(sx(mx, 2)); mx = mx.max(sx(mx, 4));
            let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 }; s!(dxs, tt, mx / 127.0);
            s!(qw, tt * 4, q8b(Vec4::new(x0.x, x1.x, x2.x, x3.x), inv)); s!(qw, tt * 4 + 1, q8b(Vec4::new(x0.y, x1.y, x2.y, x3.y), inv));
            s!(qw, tt * 4 + 2, q8b(Vec4::new(x0.z, x1.z, x2.z, x3.z), inv)); s!(qw, tt * 4 + 3, q8b(Vec4::new(x0.w, x1.w, x2.w, x3.w), inv));
        });
        unroll!([0, 1, 2, 3, 4, 5, 6, 7], |rr| {
            let row = (r0 + rr).min(m - 1);
            let word = g!(wq, row * wpr + wi);
            let dw = g!(wsc, row * bpr + (wi >> 3));
            let sg = |b: u32| -> u32 { let sp = (word >> (2 * b)) & 0x0303_0303; ((sp | 0x8080_8080) - 0x0101_0101) ^ 0x8080_8080 };
            let (s0, s1, s2, s3) = (sg(0), sg(1), sg(2), sg(3));
            unroll!([$($t),*], |tt| {
                let dot = sdot4(s0, g!(qw, tt * 4)) + sdot4(s1, g!(qw, tt * 4 + 1)) + sdot4(s2, g!(qw, tt * 4 + 2)) + sdot4(s3, g!(qw, tt * 4 + 3));
                s!(acc, rr * $T + tt, g!(acc, rr * $T + tt) + (dot as f32) * dw * g!(dxs, tt));
            });
        });
        wi += 64;
    }
    unroll!([$($t),*], |tt| {
        unroll!([0, 1, 2, 3, 4, 5, 6, 7], |rr| {
            let mut v = g!(acc, rr * $T + tt);
            v += sx(v, 16); v += sx(v, 8); v += sx(v, 4); v += sx(v, 2); v += sx(v, 1);
            if ln == 0 { s!(red, (tt * 2 + w) * 8 + rr, v); }
        });
    });
    bar();
    if l < 8 * $T {
        let tt = l >> 3; let rr = l & 7;
        if r0 + rr < m && t0 + tt < ntok {
            let v = 0.0f32 + g!(red, (tt * 2) * 8 + rr) + g!(red, (tt * 2 + 1) * 8 + rr);
            let yi = yo + (t0 + tt) * ys + r0 + rr; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
        }
    }
}
}; }
mv_dt!(matvec_d8x2, 2, [0, 1]);
mv_dt!(matvec_d8x4, 4, [0, 1, 2, 3]);

/// f32 decode matvec, 2 consecutive words (32 weights, same scale group) per lane per row: one 8-byte weight load and
/// one scale load per 2 words. Per-word term identical to matvec_c8 ((codes.x) - xsum) * d; only the summation
/// order across words differs.
macro_rules! mv_2 { ($name:ident, $T:literal, $R:literal, [$($r:literal),*]) => {
#[spirv(compute(threads($T)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq2: &[spirv_std::glam::UVec2],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 512]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (m, k, xo, yo, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 3), pu(pc, 5), pu(pc, 7));
    let l = lid.x; let w = l >> 5; let ln = l & 31;
    let r0 = (wg.y * nwg.x + wg.x) * $R;
    let wpr = k >> 4; let p2 = wpr >> 1; let bpr = k >> 7;
    let xb = xo / 4;
    let mut acc = [0.0f32; $R];
    let mut pi = l;
    while pi < p2 {
        let b = xb + pi * 8;
        let x0 = g!(a4, b); let x1 = g!(a4, b + 1); let x2 = g!(a4, b + 2); let x3 = g!(a4, b + 3);
        let y0 = g!(a4, b + 4); let y1 = g!(a4, b + 5); let y2 = g!(a4, b + 6); let y3 = g!(a4, b + 7);
        let xs0 = (x0 + x1 + x2 + x3).dot(Vec4::ONE); let xs1 = (y0 + y1 + y2 + y3).dot(Vec4::ONE);
        unroll!([$($r),*], |rr| {
            let row = (r0 + rr).min(m - 1);
            let wv = g!(wq2, row * p2 + pi);
            let d = g!(wsc, row * bpr + (pi >> 2));
            let t0 = codes2(wv.x, 0).dot(x0) + codes2(wv.x, 8).dot(x1) + codes2(wv.x, 16).dot(x2) + codes2(wv.x, 24).dot(x3);
            let t1 = codes2(wv.y, 0).dot(y0) + codes2(wv.y, 8).dot(y1) + codes2(wv.y, 16).dot(y2) + codes2(wv.y, 24).dot(y3);
            s!(acc, rr, g!(acc, rr) + (t0 - xs0) * d + (t1 - xs1) * d);
        });
        pi += $T;
    }
    unroll!([$($r),*], |rr| {
        let mut v = g!(acc, rr);
        v += sx(v, 16); v += sx(v, 8); v += sx(v, 4); v += sx(v, 2); v += sx(v, 1);
        if ln == 0 { s!(red, w * $R + rr, v); }
    });
    bar();
    if l < $R && r0 + l < m {
        let mut v = 0.0f32; let mut j = 0u32; while j < $T / 32 { v += g!(red, j * $R + l); j += 1; }
        let yi = yo + r0 + l; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
    }
}
}; }
mv_2!(matvec_p8a, 64, 8, [0,1,2,3,4,5,6,7]);
mv_2!(matvec_p8b, 128, 8, [0,1,2,3,4,5,6,7]);
mv_2!(matvec_p4a, 32, 4, [0,1,2,3]);

/// GPU token embedding, trunk table (PQ2_0 packed like every ternary matrix): out[t*k + j] = (code - 1) * scale, the
/// exact CPU `row_f32` value. pc: 0 ids (arena u32 offset), 1 out, 2 k (= n_embd). grid (k/256, ntok).
#[spirv(compute(threads(256)))]
pub fn embed_p2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32]) {
    let (ids, oo, k) = (pu(pc, 0), pu(pc, 1), pu(pc, 2)); let t = wg.y; let j = wg.x * 256 + lid.x;
    if j < k {
        let r = g!(au, ids + t);
        let c = (g!(wq, r * (k >> 4) + (j >> 4)) >> (2 * (j & 15))) & 3;
        s!(a, oo + t * k + j, (c as i32 - 1) as f32 * g!(wsc, r * (k >> 7) + (j >> 7)));
    }
}
/// GPU token embedding, MTP head table (Q4_K repacked by quant::q4k_repack): out = dm0 * q - dm1 (dm0 * q is exact,
/// so FMA contraction cannot change the result). pc as embed_p2.
#[spirv(compute(threads(256)))]
pub fn embed_q4(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32]) {
    let (ids, oo, k) = (pu(pc, 0), pu(pc, 1), pu(pc, 2)); let t = wg.y; let j = wg.x * 256 + lid.x;
    if j < k {
        let r = g!(au, ids + t);
        let n = (g!(wq, r * (k >> 3) + (j >> 3)) >> (4 * (j & 7))) & 15;
        let gi = r * (k >> 5) + (j >> 5);
        s!(a, oo + t * k + j, g!(wsc, 2 * gi) * n as f32 - g!(wsc, 2 * gi + 1));
    }
}

/// Candidate 2 (BONSAI_ABW): matvec_f32t with a wide row tile: 48 rows (all ssm_vheads rows of one matrix) x 8 tokens
/// per 384-thread workgroup, one output per thread. Activations are read by every row of the matrix at once (2x total
/// activation traffic for beta+alpha instead of 6x), and there are ntok/8 x 2 workgroups (vs matvec_ab's ntok/64).
/// Per-output summation is the same sequential K order as matvec_f32t (64-wide chunks, c = 0..63), so it is bit-identical.
/// pc as matvec_f32t (0 rows, 1 k, 2 ntok, 3 xo, 4 xs, 5 yo, 6 ys, 7 wo); k % 64 == 0. Grid (rows/48 rounded up, ntok/8 rounded up).
#[spirv(compute(threads(384)))]
pub fn matvec_f32w(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] wsh: &mut [f32; 3120], #[spirv(workgroup)] xsh: &mut [f32; 520]) {
    let (m, k, ntok, xo, xs, yo, ys, wo) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let l = lid.x; let tt = l & 7; let rr = l >> 3;
    let row0 = wg.x * 48; let tok0 = wg.y * 8;
    let mut sm = 0.0f32;
    let mut kc = 0u32;
    while kc < k {
        unroll!([0, 1, 2, 3, 4, 5, 6, 7], |q| {
            let idx = q * 384 + l; let r = idx >> 6; let c = idx & 63;
            let row = (row0 + r).min(m - 1);
            s!(wsh, r * 65 + c, g!(prm, wo + row * k + kc + c));
        });
        if l < 128 { unroll!([0, 1, 2, 3], |q| {
            let idx = q * 128 + l; let r = idx >> 6; let c = idx & 63;
            let tok = (tok0 + r).min(ntok - 1);
            s!(xsh, r * 65 + c, g!(a, xo + tok * xs + kc + c));
        }); }
        bar();
        let mut c = 0u32;
        while c < 64 { sm += g!(wsh, rr * 65 + c) * g!(xsh, tt * 65 + c); c += 1; }
        bar();
        kc += 64;
    }
    if row0 + rr < m && tok0 + tt < ntok { s!(a, yo + (tok0 + tt) * ys + row0 + rr, sm); }
}


/// Candidate 3 (BONSAI_SWQ): FFN SwiGLU fused into nhq's load. x = silu(gate) * up exactly as `ewise` mode 1
/// (silu(x) = x / (1 + exp(-x))), then the identical Hadamard + q8 path. pc as nhq with 3 = up offset (row stride its)
/// and donorm = 0; the SwiGLU result itself is not stored (only the Hadamard output and q8 feed the down projection).
#[spirv(compute(threads(256)))]
pub fn nhq_sw(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] au: &mut [u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] v: &mut [f32; 1024], #[spirv(workgroup)] red: &mut [f32; 256]) {
    use spirv_std::arch::subgroup_shuffle_xor as sx;
    let (n, io, its, xno, oo, ots, wo, so) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let (qo, sco, donorm) = (pu(pc, 8), pu(pc, 9), pu(pc, 10)); let eps = pf(pc, 11);
    let (blk, t, l) = (wg.x, wg.y, lid.x);
    let ib = io + t * its;
    let mut sc = 1.0f32;
    if donorm != 0 {
        let mut sm = 0.0f32;
        let mut i = l; while i < n { let x = g!(a, ib + i); sm += x * x; i += 256; }
        s!(red, l, sm); bar();
        unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh; if l < st { s!(red, l, g!(red, l) + g!(red, l + st)); } bar(); });
        sc = 1.0 / (g!(red, 0) / n as f32 + eps).sqrt();
    }
    unroll!([0, 1, 2, 3], |q| {
        let i = l + 256 * q; let gi = blk * 1024 + i;
        let g0 = g!(a, ib + gi); let mut x = (g0 / (1.0 + (-g0).exp())) * g!(a, xno + t * its + gi);
        if donorm != 0 { x = x * sc * g!(prm, wo + gi); }
        if so != u32::MAX { x *= g!(prm, so + gi); }
        s!(v, i, x);
    });
    bar();
    unroll!([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], |sh| {
        let h = 1u32 << sh;
        unroll!([0, 1], |q| {
            let p = l + 256 * q;
            let i = ((p >> sh) << (sh + 1)) + (p & (h - 1)); let j = i + h;
            let x = g!(v, i); let y = g!(v, j); s!(v, i, x + y); s!(v, j, x - y);
        });
        bar();
    });
    // contiguous 4 per thread: warp w owns the 128-group w of this block
    let b0 = l * 4;
    let x0 = g!(v, b0) * 0.03125; let x1 = g!(v, b0 + 1) * 0.03125; let x2 = g!(v, b0 + 2) * 0.03125; let x3 = g!(v, b0 + 3) * 0.03125;
    let ob = oo + t * ots + blk * 1024 + b0;
    s!(a, ob, x0); s!(a, ob + 1, x1); s!(a, ob + 2, x2); s!(a, ob + 3, x3);
    let mut mx = x0.abs().max(x1.abs()).max(x2.abs()).max(x3.abs());
    mx = mx.max(sx(mx, 1)); mx = mx.max(sx(mx, 2)); mx = mx.max(sx(mx, 4)); mx = mx.max(sx(mx, 8)); mx = mx.max(sx(mx, 16));
    let inv = if mx > 0.0 { 127.0 / mx } else { 0.0 };
    let pb = (l >> 2) * 16 + (l & 3); // K-transposed q8 layout (see quant_x)
    let p0 = g!(v, pb) * 0.03125; let p1 = g!(v, pb + 4) * 0.03125; let p2 = g!(v, pb + 8) * 0.03125; let p3 = g!(v, pb + 12) * 0.03125;
    let q0 = (rnd(p0 * inv) as i32) as u32 & 0xff; let q1 = (rnd(p1 * inv) as i32) as u32 & 0xff;
    let q2 = (rnd(p2 * inv) as i32) as u32 & 0xff; let q3 = (rnd(p3 * inv) as i32) as u32 & 0xff;
    s!(au, qo + t * (n / 4) + blk * 256 + l, q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
    if (l & 31) == 0 { s!(a, sco + t * (n / 128) + blk * 8 + (l >> 5), mx / 127.0); }
}

/// K1: exact f32 multi-token matvec for short prompts (3..21 tokens). Per (row, token) it is bit-identical to matvec_c8x2 /
/// matvec_c8 by construction: lane L still owns words L, L+64, ... of every row in order, each word contributes
/// `(c0.x0 + c1.x1 + c2.x2 + c3.x3 - xsum) * d` with the same expressions, and the per-row result is the same 8x8 reduction
/// tree. What changes is data movement: $G groups of 64 lanes (8 rows each) share one LDS copy of each step's x slice for
/// $T tokens (fetched from L2 once per 8*$G rows instead of once per 8 rows), and each weight word is read once per $T tokens.
/// Grid: (row blocks of 8*$G rows, y, token tiles of $T); tokens past ntok are skipped.
macro_rules! mv_s { ($name:ident, $thr:literal, $G:literal, $T:literal, $XS:literal, [$($t:literal),*]) => {
#[spirv(compute(threads($thr)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] xsh: &mut [Vec4; $XS], #[spirv(workgroup)] red: &mut [f32; 4096]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let tid = lid.x; let lane = tid & 63; let grp = tid >> 6;
    let r0 = (wg.y * nwg.x + wg.x) * (8 * $G) + grp * 8;
    let t0 = wg.z * $T; let nt = (ntok - t0).min($T);
    let wpr = k >> 4; let bpr = k >> 7;
    let mut acc = [0.0f32; 8 * $T];
    let mut base = 0u32;
    while base < wpr {
        // stage x for this step: [token][v][word] (lane-contiguous reads); global reads are contiguous per token
        let mut i = tid;
        while i < 256 * $T {
            let tt = i >> 8; let rem = i & 255; let w = rem >> 2; let v = rem & 3;
            let tk = (t0 + tt).min(ntok - 1); let wi = base + w;
            let x = if wi < wpr { g!(a4, (xo + tk * xs) / 4 + wi * 4 + v) } else { Vec4::ZERO };
            s!(xsh, tt * 256 + v * 64 + w, x);
            i += $thr;
        }
        bar();
        let wi = base + lane;
        if wi < wpr {
            let mut xv = [Vec4::ZERO; 4 * $T]; let mut xsum = [0.0f32; $T];
            unroll!([$($t),*], |tt| {
                unroll!([0,1,2,3], |v| { s!(xv, tt * 4 + v, g!(xsh, tt * 256 + v * 64 + lane)); });
                s!(xsum, tt, (g!(xv, tt * 4) + g!(xv, tt * 4 + 1) + g!(xv, tt * 4 + 2) + g!(xv, tt * 4 + 3)).dot(Vec4::ONE));
            });
            unroll!([0,1,2,3,4,5,6,7], |rr| {
                let row = (r0 + rr).min(m - 1);
                let word = g!(wq, row * wpr + wi);
                let d = g!(wsc, row * bpr + (wi >> 3));
                let c0 = codes2(word, 0); let c1 = codes2(word, 8); let c2 = codes2(word, 16); let c3 = codes2(word, 24);
                unroll!([$($t),*], |tt| {
                    let v = c0.dot(g!(xv, tt * 4)) + c1.dot(g!(xv, tt * 4 + 1)) + c2.dot(g!(xv, tt * 4 + 2)) + c3.dot(g!(xv, tt * 4 + 3));
                    s!(acc, rr * $T + tt, g!(acc, rr * $T + tt) + (v - g!(xsum, tt)) * d);
                });
            });
        }
        bar();
        base += 64;
    }
    // same 8x8 reduction tree as matvec_c8 / matvec_c8x2, per group (red slice of 512) and token
    let rb = grp * 512;
    unroll!([$($t),*], |tt| {
        unroll!([0,1,2,3,4,5,6,7], |rr| { s!(red, rb + rr * 64 + lane, g!(acc, rr * $T + tt)); });
        bar();
        let mut s = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |i| { s += g!(red, rb + lane * 8 + i); });
        bar();
        s!(red, rb + lane, s);
        bar();
        if lane < 8 && r0 + lane < m && tt < nt {
            let mut v = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |j| { v += g!(red, rb + lane * 8 + j); });
            let yi = yo + (t0 + tt) * ys + r0 + lane; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
        }
        bar();
    });
}
}; }
mv_s!(matvec_s4g4, 256, 4, 4, 1024, [0, 1, 2, 3]);
mv_s!(matvec_s4g8, 512, 8, 4, 1024, [0, 1, 2, 3]);
mv_s!(matvec_s2g8, 512, 8, 2, 512, [0, 1]);
mv_s!(matvec_s6g8, 512, 8, 6, 1536, [0, 1, 2, 3, 4, 5]);

/// 2-bit code -> f32 without an I2F conversion (16/clk/SM on GA102): 2^23 + c built by OR, then 2^23 subtracted. Exact for
/// c in 0..3, so the result equals `c as f32` bit-for-bit and every downstream product/sum is unchanged.
#[inline(always)] fn cm(word: u32, sh: u32) -> f32 { f32::from_bits(((word >> sh) & 3) | 0x4B00_0000) - 8388608.0 }
#[inline(always)] fn codes_m(word: u32, sh: u32) -> Vec4 { Vec4::new(cm(word, sh), cm(word, sh + 2), cm(word, sh + 4), cm(word, sh + 6)) }

/// matvec_c8 with the magic-number code decode (bit-identical; NV copy so the shared W9100 kernel is untouched).
#[spirv(compute(threads(64)))]
pub fn matvec_m8(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 2048]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let lane = lid.x;
    let r0 = (wg.y * nwg.x + wg.x) * 8;
    let wpr = k >> 4; let bpr = k >> 7;
    let mut t = 0u32;
    while t < ntok {
        let xb = (xo + t * xs) / 4;
        let mut acc = [0.0f32; 8];
        let mut wi = lane;
        while wi < wpr {
            let x0 = g!(a4, xb + wi * 4); let x1 = g!(a4, xb + wi * 4 + 1); let x2 = g!(a4, xb + wi * 4 + 2); let x3 = g!(a4, xb + wi * 4 + 3);
            let xsum = (x0 + x1 + x2 + x3).dot(Vec4::ONE);
            unroll!([0,1,2,3,4,5,6,7], |rr| {
                let row = (r0 + rr).min(m - 1);
                let word = g!(wq, row * wpr + wi);
                let d = g!(wsc, row * bpr + (wi >> 3));
                let tt = codes_m(word, 0).dot(x0) + codes_m(word, 8).dot(x1) + codes_m(word, 16).dot(x2) + codes_m(word, 24).dot(x3);
                s!(acc, rr, g!(acc, rr) + (tt - xsum) * d);
            });
            wi += 64;
        }
        unroll!([0,1,2,3,4,5,6,7], |rr| { s!(red, rr * 64 + lane, g!(acc, rr)); });
        bar();
        let mut s = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |i| { s += g!(red, lane * 8 + i); });
        bar();
        s!(red, lane, s);
        bar();
        if lane < 8 && r0 + lane < m {
            let mut v = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |j| { v += g!(red, lane * 8 + j); });
            let yi = yo + t * ys + r0 + lane; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
        }
        bar();
        t += 1;
    }
}

/// matvec_c8x2 / x4 with the magic-number decode (bit-identical to matvec_c8x2 per token).
macro_rules! mv_tm { ($name:ident, $T:literal, [$($t:literal),*]) => {
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
            let c0 = codes_m(word, 0); let c1 = codes_m(word, 8); let c2 = codes_m(word, 16); let c3 = codes_m(word, 24);
            unroll!([$($t),*], |tt| {
                let v = c0.dot(g!(xv, tt * 4)) + c1.dot(g!(xv, tt * 4 + 1)) + c2.dot(g!(xv, tt * 4 + 2)) + c3.dot(g!(xv, tt * 4 + 3));
                s!(acc, rr * $T + tt, g!(acc, rr * $T + tt) + (v - g!(xsum, tt)) * d);
            });
        });
        wi += 64;
    }
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
mv_tm!(matvec_m8x2, 2, [0, 1]);
mv_tm!(matvec_m8x4, 4, [0, 1, 2, 3]);

/// K1b: matvec_c8x2-equivalent with $R rows per lane (x fetched once per $R rows instead of 8) and $T tokens per pass.
/// Bit-identical per (row, token): same lane->word order, same per-word expression, and the same 8x8 reduction tree applied
/// to each 8-row block. Grid: (row blocks of $R, y, token tiles of $T).
macro_rules! mv_r { ($name:ident, $R:literal, $T:literal, [$($r:literal),*], [$($t:literal),*], [$($b:literal),*]) => {
#[spirv(compute(threads(64)))]
pub fn $name(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(num_workgroups)] nwg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a4: &[Vec4],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] wq: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] wsc: &[f32],
    #[spirv(workgroup)] red: &mut [f32; 2048]) {
    let (m, k, ntok, xo, xs, yo, ys, accum) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7));
    let lane = lid.x;
    let r0 = (wg.y * nwg.x + wg.x) * $R;
    let t0 = wg.z * $T; let nt = (ntok - t0).min($T);
    let wpr = k >> 4; let bpr = k >> 7;
    let mut acc = [0.0f32; $R * $T];
    let mut wi = lane;
    while wi < wpr {
        let mut xv = [Vec4::ZERO; 4 * $T]; let mut xsum = [0.0f32; $T];
        unroll!([$($t),*], |tt| {
            let tk = (t0 + tt).min(ntok - 1); let xb = (xo + tk * xs) / 4 + wi * 4;
            unroll!([0,1,2,3], |v| { s!(xv, tt * 4 + v, g!(a4, xb + v)); });
            s!(xsum, tt, (g!(xv, tt * 4) + g!(xv, tt * 4 + 1) + g!(xv, tt * 4 + 2) + g!(xv, tt * 4 + 3)).dot(Vec4::ONE));
        });
        unroll!([$($r),*], |rr| {
            let row = (r0 + rr).min(m - 1);
            let word = g!(wq, row * wpr + wi);
            let d = g!(wsc, row * bpr + (wi >> 3));
            let c0 = codes2(word, 0); let c1 = codes2(word, 8); let c2 = codes2(word, 16); let c3 = codes2(word, 24);
            unroll!([$($t),*], |tt| {
                let v = c0.dot(g!(xv, tt * 4)) + c1.dot(g!(xv, tt * 4 + 1)) + c2.dot(g!(xv, tt * 4 + 2)) + c3.dot(g!(xv, tt * 4 + 3));
                s!(acc, rr * $T + tt, g!(acc, rr * $T + tt) + (v - g!(xsum, tt)) * d);
            });
        });
        wi += 64;
    }
    unroll!([$($t),*], |tt| {
        unroll!([$($b),*], |bb| {
            unroll!([0,1,2,3,4,5,6,7], |rr| { s!(red, bb * 512 + rr * 64 + lane, g!(acc, (bb * 8 + rr) * $T + tt)); });
        });
        bar();
        let mut sv = [0.0f32; $R / 8];
        unroll!([$($b),*], |bb| { let mut s = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |i| { s += g!(red, bb * 512 + lane * 8 + i); }); s!(sv, bb, s); });
        bar();
        unroll!([$($b),*], |bb| { s!(red, bb * 512 + lane, g!(sv, bb)); });
        bar();
        if lane < $R && tt < nt {
            let bb = lane >> 3; let ln = lane & 7;
            if r0 + lane < m {
                let mut v = 0.0f32; unroll!([0,1,2,3,4,5,6,7], |j| { v += g!(red, bb * 512 + ln * 8 + j); });
                let yi = yo + (t0 + tt) * ys + r0 + lane; s!(a, yi, if accum != 0 { g!(a, yi) + v } else { v });
            }
        }
        bar();
    });
}
}; }
mv_r!(matvec_r16x2, 16, 2, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [0, 1], [0, 1]);
mv_r!(matvec_r32x2, 32, 2, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], [0, 1], [0, 1, 2, 3]);
mv_r!(matvec_r16x4, 16, 4, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], [0, 1, 2, 3], [0, 1]);
mv_r!(matvec_r32x1, 32, 1, [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31], [0], [0, 1, 2, 3]);

/// K2: `attn_s_h` with software-pipelined loads. Same arithmetic in the same order (per-position score chain over the 32
/// K words, max/sum trees, the sequential V accumulation over positions); only the load schedule changes: K words are
/// fetched 8 at a time ahead of their use and V rows 8 positions ahead, so each lane has several loads in flight instead of
/// one dependent load per step. pc as attn_s_h.
#[spirv(compute(threads(64)))]
pub fn attn_s_h2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
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
        let kb = (kco + (p * kvd + kh * 256) / 2) / 4;
        let mut d = Vec4::ZERO;
        let mut i0 = 0u32;
        while i0 < 32 {
            let mut wv = [UVec4::ZERO; 8];
            unroll!([0,1,2,3,4,5,6,7], |j| { s!(wv, j, g!(u4, kb + i0 + j)); });
            unroll!([0,1,2,3,4,5,6,7], |j| {
                let i = i0 + j; let w = g!(wv, j); let (x0, x1, x2, x3) = (h2(w.x), h2(w.y), h2(w.z), h2(w.w));
                d += Vec4::new(x0.x, x0.y, x1.x, x1.y) * g!(qv, 2 * i) + Vec4::new(x2.x, x2.y, x3.x, x3.y) * g!(qv, 2 * i + 1);
            });
            i0 += 8;
        }
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
    p = p0;
    while p + 8 <= p1 {
        let mut vv = [Vec4::ZERO; 8];
        unroll!([0,1,2,3,4,5,6,7], |j| {
            let vb = vco + ((p + j) * kvd + kh * 256) / 2 + 2 * l; let (y0, y1) = (h2(g!(au, vb)), h2(g!(au, vb + 1)));
            s!(vv, j, Vec4::new(y0.x, y0.y, y1.x, y1.y));
        });
        unroll!([0,1,2,3,4,5,6,7], |j| { o += g!(vv, j) * g!(sc, p + j - p0); });
        p += 8;
    }
    while p < p1 {
        let vb = vco + (p * kvd + kh * 256) / 2 + 2 * l; let (y0, y1) = (h2(g!(au, vb)), h2(g!(au, vb + 1)));
        o += Vec4::new(y0.x, y0.y, y1.x, y1.y) * g!(sc, p - p0); p += 1; }
    let pb = scr + ((t * nh + h) * smax + s) * 260;
    if l == 0 { s!(a, pb, m); s!(a, pb + 1, tot); }
    let ob = pb + 4 + l * 4; s!(a, ob, o.x); s!(a, ob + 1, o.y); s!(a, ob + 2, o.z); s!(a, ob + 3, o.w);
}

/// K2: `attn_comb` with the per-chunk partial loads issued 4 chunks ahead; same sequential merge order. pc as attn_comb.
#[spirv(compute(threads(64)))]
pub fn attn_comb2(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32]) {
    let (qo, qts, pos0, nh, scr, cn, smax, oo, ots) = (pu(pc, 0), pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6), pu(pc, 7), pu(pc, 8));
    let (h, t, l) = (wg.x, wg.y, lid.x);
    let n = pos0 + t + 1; let ns = (n + cn - 1) / cn;
    let base = scr + (t * nh + h) * smax * 260;
    let mut m = -1e30f32; let mut s = 0u32;
    while s + 4 <= ns {
        let mut mv = [0.0f32; 4]; unroll!([0,1,2,3], |j| { s!(mv, j, g!(a, base + (s + j) * 260)); });
        unroll!([0,1,2,3], |j| { m = m.max(g!(mv, j)); });
        s += 4;
    }
    while s < ns { m = m.max(g!(a, base + s * 260)); s += 1; }
    let mut den = 0.0f32; let mut o = [0.0f32; 4];
    s = 0;
    while s + 4 <= ns {
        let mut mm = [0.0f32; 4]; let mut ss = [0.0f32; 4]; let mut ov = [0.0f32; 16];
        unroll!([0,1,2,3], |j| {
            let pb = base + (s + j) * 260; s!(mm, j, g!(a, pb)); s!(ss, j, g!(a, pb + 1));
            unroll!([0,1,2,3], |q| { s!(ov, j * 4 + q, g!(a, pb + 4 + l * 4 + q)); });
        });
        unroll!([0,1,2,3], |j| {
            let w = (g!(mm, j) - m).exp(); den += g!(ss, j) * w;
            unroll!([0,1,2,3], |q| { s!(o, q, g!(o, q) + g!(ov, j * 4 + q) * w); });
        });
        s += 4;
    }
    while s < ns {
        let pb = base + s * 260; let w = (g!(a, pb) - m).exp(); den += g!(a, pb + 1) * w;
        unroll!([0, 1, 2, 3], |j| { s!(o, j, g!(o, j) + g!(a, pb + 4 + l * 4 + j) * w); });
        s += 1;
    }
    unroll!([0, 1, 2, 3], |j| {
        let d = l * 4 + j;
        s!(a, oo + t * ots + h * 256 + d, g!(o, j) / den * sigm(g!(a, qo + t * qts + h * 512 + 256 + d)));
    });
}

// The unbounded persistent-grid gsync/gbar experiment was removed during the
// NVIDIA repair. A host timeout is not GPU cancellation. No production caller
// or checked-in shader for this experiment is retained.

/// Historical FZ3 source ONLY; not emitted by the selected builder or loaded.
/// Failed strict intermediate-state parity (max_scaled 5.6e-4 > 1e-4), even
/// though sampled tokens matched. BONSAI_FUSE is rejected by production.
/// GDN output gated rmsnorm (rmsnorm mode 1, rows of hd) fused into the grouped
/// 1024-block Hadamard. Each block gathers 1024/hd whole heads; their sums of squares are recomputed here with the
/// rmsnorm reduction tree (256 lanes, lanes >= hd contribute 0), then each element is a*sc*w*silu(z) exactly as rmsnorm
/// writes it, followed by the `hadamard` kernel's sign / butterfly / scale. ntok = 1, hd = 128.
/// pc: n, io, oo, so, hd, nk, rep, wo, zo, eps.
#[spirv(compute(threads(256)))]
pub fn had_norm(#[spirv(local_invocation_id)] lid: UVec3, #[spirv(workgroup_id)] wg: UVec3, #[spirv(push_constant)] pc: &PC,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] a: &mut [f32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] prm: &[f32],
    #[spirv(workgroup)] v: &mut [f32; 1024], #[spirv(workgroup)] red: &mut [f32; 2048]) {
    let (io, oo, so, hd, nk, rep) = (pu(pc, 1), pu(pc, 2), pu(pc, 3), pu(pc, 4), pu(pc, 5), pu(pc, 6));
    let (wo, zo) = (pu(pc, 7), pu(pc, 8)); let eps = pf(pc, 9);
    let (blk, l) = (wg.x, lid.x);
    // the 8 heads of this block: slot j -> q = blk*8 + j -> source head k + nk*r
    unroll!([0, 1, 2, 3, 4, 5, 6, 7], |j| {
        let q = blk * 8 + j; let hh = (q / rep) + nk * (q % rep);
        let mut sm = 0.0f32;
        if l < hd { let x = g!(a, io + hh * hd + l); sm += x * x; }
        s!(red, j * 256 + l, sm);
    });
    bar();
    unroll!([7, 6, 5, 4, 3, 2, 1, 0], |sh| { let st = 1u32 << sh;
        if l < st { unroll!([0, 1, 2, 3, 4, 5, 6, 7], |j| { s!(red, j * 256 + l, g!(red, j * 256 + l) + g!(red, j * 256 + l + st)); }); }
        bar(); });
    unroll!([0, 1, 2, 3], |q| {
        let i = l + 256 * q;
        let gi = blk * 1024 + i;
        let i0 = gi % hd; let qh = gi / hd; let r = qh % rep; let k = qh / rep; let hh = k + nk * r;
        let tot = g!(red, (qh - blk * 8) * 256);
        let sc = 1.0 / (tot / hd as f32 + eps).sqrt();
        let mut x = g!(a, io + hh * hd + i0) * sc;
        if wo != u32::MAX { x *= g!(prm, wo + i0); }
        { let zv = g!(a, zo + hh * hd + i0); x *= zv / (1.0 + (-zv).exp()); }
        if so != u32::MAX { x *= g!(prm, so + gi); }
        s!(v, i, x);
    });
    bar();
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
        let x = g!(v, i) * 0.03125;
        s!(a, oo + gi, x);
    });
}
