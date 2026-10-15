//! Per-kernel parity: run every compute kernel from the rust-gpu set and the GLSL set on identical
//! random inputs (same device) and compare the whole output arena.
//! Metric: max |rust - glsl| / max(1, |glsl|) over every arena float; pass if <= tol (1e-4).
use super::{Buf, Ctx, NONE};
use ash::vk;

const AN: usize = 5 << 20; // arena floats
const PN: usize = 1 << 16; // params floats
const WM: u32 = 1000; const WK: u32 = 1024; // ternary test matrix
const MB: usize = 1 << 20;

struct Rng(u64);
#[path="prefill_dense_review.rs"] mod dense_review;
impl Rng { fn f(&mut self) -> f32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0 }
           fn u(&mut self) -> u32 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; (self.0 >> 32) as u32 } }

/// (name, pipeline, push constants, grid)
fn cases() -> Vec<(String, &'static str, Vec<u32>, (u32, u32, u32))> {
    let m = MB as u32; let fb = |x: f32| x.to_bits();
    let mut v: Vec<(String, &'static str, Vec<u32>, (u32, u32, u32))> = vec![
        ("matvec_t2_1 M=1000 K=1024 accum".into(), "matvec_t2_1", vec![WM, WK, 1, 0, 0, m, 0, 1], (WM.div_ceil(8), 1, 1)),
        ("matvec_t2_1 short-prompt ntok=7".into(), "matvec_t2_1", vec![WM, WK, 7, 0, WK, m, WM, 0], (7, WM.div_ceil(8), 1)),
        ("matvec_t2 ntok=13".into(), "matvec_t2", vec![WM, WK, 13, 0, WK, m, WM, 0], (WM.div_ceil(4), 1, 2)),
        ("matmul_t2 ntok=70".into(), "matmul_t2", vec![WM, WK, 70, 0, WK, m, WM, 0], (WM.div_ceil(64), 2, 1)),
        ("matmul_t2s ntok=11 accum".into(), "matmul_t2s", vec![WM, WK, 11, 0, WK, m, WM, 1], (WM.div_ceil(128), 2, 1)),
        ("matmul_t2s ntok=5".into(), "matmul_t2s", vec![WM, WK, 5, 0, WK, m, WM, 0], (WM.div_ceil(128), 1, 1)),
        ("matvec_f32".into(), "matvec_f32", vec![48, 1024, 3, 0, 1024, m, 48, 0], (48, 3, 1)),
        ("hadamard fwd signs".into(), "hadamard", vec![2048, 0, 2048, m, 2048, 0, 0, 0, 0, 0], (2, 3, 1)),
        ("hadamard fwd perm".into(), "hadamard", vec![2048, 0, 2048, m, 2048, 0, 0, 128, 4, 4], (2, 3, 1)),
        ("hadamard inv".into(), "hadamard", vec![2048, 0, 2048, m, 2048, 0, 1, 0, 0, 0], (2, 3, 1)),
        ("hadamard nosign".into(), "hadamard", vec![1024, 0, 1024, m, 1024, NONE, 0, 0, 0, 0], (1, 3, 1)),
        ("conv K=4".into(), "conv", vec![1000, 5, 0, 1000, m, 1000, 0, 2 * m, 4], (4, 1, 1)),
        ("rope imrope".into(), "rope", vec![0, 256, 1024, 4, 0, 7, 64, fb(1e7f32.powf(-2.0 / 64.0)), 11, 11, 10, 0], (2, 3, 1)),
        ("attn".into(), "attn", vec![0, 2048, m, 2 * m, 512, 5, 256, 4, 2, 3 * m, 1024, 4 * m, 64, fb(1.0 / 16.0)], (4, 3, 1)),
        ("gdn (decode kernel, 6 tok)".into(), "gdn", vec![6, 0, 1024, m, m + 64, 4, 0, 16, 2 * m, 3 * m, 512, 2, 128, 256], (4, 1, 1)),
        ("gdn_p (prefill kernel, 6 tok)".into(), "gdn_p", vec![6, 0, 1024, m, m + 64, 4, 0, 16, 2 * m, 3 * m, 512, 2, 128, 256], (4, 2, 1)),
        ("matvec_q8 M=1000 K=256".into(), "matvec_q8", vec![WM, 256, 1, 0, 0, m, 0, 0], (1, WM.div_ceil(4), 1)),
        ("matvec_q8 ntok=3 accum".into(), "matvec_q8", vec![WM, 256, 3, 0, 256, m, WM, 1], (3, WM.div_ceil(4), 1)),
        ("argmax".into(), "argmax", vec![0, 5000, m], (1, 1, 1)),
    ];
    for mode in 0..3u32 {
        v.push((format!("rmsnorm mode {mode}"), "rmsnorm", vec![1024, 0, 1024, 4096, m, 1024, 4096, if mode == 2 { NONE } else { 0 }, mode, 2 * m, 1024, 4096, fb(1e-6)], (4, 3, 1)));
        v.push((format!("ewise mode {mode}"), "ewise", vec![mode, 3000, 0, 0, 3000, m, 3000, 2 * m, 3000], (12, 3, 1)));
    }
    v
}

struct Side { c: Ctx, arena: Buf, set: ash::vk::DescriptorSet, _descriptor_buffers: [Buf; 3] }

fn side(dev: Option<&str>, rust: bool, a: &[f32], p: &[f32], w: &[u32], sc: &[f32]) -> Result<Side, String> {
    let c = Ctx::new_with(dev, 8, rust)?;
    let bytes = |x: &[f32]| x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
    let arena = c.buffer((a.len() * 4) as u64, false)?; let pb = c.buffer((p.len() * 4) as u64, false)?;
    let wb = c.buffer((w.len() * 4) as u64, false)?; let sb = c.buffer((sc.len() * 4) as u64, false)?;
    c.upload(&pb, 0, &bytes(p)); c.upload(&wb, 0, &w.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()); c.upload(&sb, 0, &bytes(sc));
    let set = c.set([(&arena, 0, arena.size), (&pb, 0, pb.size), (&wb, 0, wb.size), (&sb, 0, sb.size)]);
    Ok(Side { c, arena, set, _descriptor_buffers: [pb, wb, sb] })
}

pub fn run(dev: Option<&str>, tol: f32) -> Result<(String, bool), String> {
    let mut r = Rng(0x9e3779b97f4a7c15);
    let a: Vec<f32> = (0..AN).map(|_| r.f()).collect();
    let mut p: Vec<f32> = (0..PN).map(|_| r.f()).collect();
    for i in 16..20 { p[i] = -p[i].abs() - 0.1; } // gdn decay a < 0, like the model
    let w: Vec<u32> = (0..(WM * WK / 16) as usize).map(|_| { let mut x = 0u32; for j in 0..16 { x |= (r.u() % 3) << (2 * j); } x }).collect();
    let sc: Vec<f32> = (0..(WM * WK / 16) as usize).map(|_| r.f().abs() * 0.05 + 0.01).collect();
    let g = side(dev, false, &a, &p, &w, &sc)?; let ru = side(dev, true, &a, &p, &w, &sc)?;
    let ab: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out = String::new(); let mut all = true;
    let mut fixtures=cases();
    for nt in [1u32,2,3,21] {
        fixtures.push((format!("r012 c8 family nt={nt} row tail"),if nt==1 {"matvec_c8"} else {"matvec_c8x2"},vec![999,WK,nt,0,WK,MB as u32,1000,1],(999u32.div_ceil(8),1,nt.div_ceil(2))));
    }
    for (name, pipe, pc, grid) in fixtures {
        let mut res = vec![];
        for s in [&g, &ru] {
            s.c.upload(&s.arena, 0, &ab);
            let (actual,gr)=if !s.c.rust && pipe.starts_with("matvec_c8") {("matvec_t2_1",if pc[2]==1 {(pc[0].div_ceil(8),1,1)} else {(pc[2],pc[0].div_ceil(8),1)})} else {(pipe,grid)};
            s.c.begin(); s.c.dispatch(actual, s.set, &pc, gr); s.c.submit_wait();
            let mut b = vec![0u8; AN * 4]; s.c.download(&s.arena, 0, &mut b);
            res.push(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>());
        }
        let (mut worst, mut changed, mut bitdiff) = (0f32, 0usize, 0usize);
        for i in 0..AN {
            let (x, y) = (res[0][i], res[1][i]);
            if x.to_bits() != a[i].to_bits() { changed += 1; }
            if x.to_bits() != y.to_bits() { bitdiff += 1; }
            let d = if x.is_nan() || y.is_nan() { if x.is_nan() && y.is_nan() { 0.0 } else { f32::INFINITY } } else { (x - y).abs() / x.abs().max(1.0) };
            worst = worst.max(d);
        }
        let ok = worst <= tol && changed > 0;
        all &= ok;
        out += &format!("{:<34} {:<12} outputs written {:>8}  differing words {:>8}  max rel diff {:.3e}  {}\n", name, pipe, changed, bitdiff, worst, if ok { "PASS" } else { "FAIL" });
    }
    out += &format!("{}\n", if all { "ALL KERNELS PASS (rust-gpu vs GLSL)" } else { "KERNEL PARITY FAIL" });
    Ok((out, all))
}

/// Decode matvec microbenchmark: tiny VRAM (one K x M ternary matrix), times `iters` back-to-back dispatches
/// of each variant in one submit and checks the variants agree. GB/s counts weight + scale bytes only.
pub fn mv_bench(dev: Option<&str>, m: u32, k: u32, iters: u32, ntok: u32) -> Result<String, String> {
    let mut r = Rng(0x1234_5678_9abc_def1);
    let an = (k as usize + m as usize) * ntok as usize + 64;
    let a: Vec<f32> = (0..an).map(|_| r.f()).collect();
    let w: Vec<u32> = (0..(m as usize * k as usize / 16)).map(|_| { let mut x = 0u32; for j in 0..16 { x |= (r.u() % 3) << (2 * j); } x }).collect();
    let sc: Vec<f32> = (0..(m as usize * k as usize / 128)).map(|_| r.f().abs() * 0.05 + 0.01).collect();
    let s = side(dev, true, &a, &[0.0; 64], &w, &sc)?;
    let ab: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let yo = k * ntok; let bytes = (w.len() * 4 + sc.len() * 4) as f64;
    let mut out = String::new(); let mut res: Vec<Vec<f32>> = vec![];
    for (pipe, rpg, tok_grid) in [("matvec_t2_1", 8u32, ntok > 1), ("matvec_c8", 8, false), ("matvec_c8x2", 8, false), ("matvec_c8x4", 8, false)] {
        let groups = m.div_ceil(rpg); let gx = groups.min(65535); let gy = groups.div_ceil(gx);
        let tz = match pipe { "matvec_c8x2" => ntok.div_ceil(2), "matvec_c8x4" => ntok.div_ceil(4), _ => 1 };
        let grid = if tok_grid { (ntok, groups, 1) } else { (gx, gy, tz) };
        let pc = [m, k, ntok, 0, k, yo, m, 0];
        s.c.upload(&s.arena, 0, &ab);
        if (m as f64 * k as f64 * ntok as f64) > 64.0 * 17408.0 * 5120.0 { return Err("mv-bench: case too large for the per-submit budget".into()); }
        s.c.begin(); s.c.dispatch(pipe, s.set, &pc, grid); s.c.submit_wait();
        let mut b = vec![0u8; an * 4]; s.c.download(&s.arena, 0, &mut b);
        res.push(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).skip(yo as usize).take((m * ntok) as usize).collect());
        // r012 safety: one dispatch per submit, abort if a single submit exceeds the budget (ring-timeout guard).
        let mut tot = 0f64; let mut n = 0u32;
        for _ in 0..iters { let t = std::time::Instant::now(); s.c.begin(); s.c.dispatch(pipe, s.set, &pc, grid); s.c.submit_wait();
            let e = t.elapsed().as_secs_f64(); tot += e; n += 1; if e * 1e3 > budget_ms() { break; } }
        let dt = tot / n as f64;
        out += &format!("{pipe:14} M={m} K={k} ntok={ntok}: {:8.1} us/call  {:6.1} GB/s weights\n", dt * 1e6, bytes / dt / 1e9);
    }
    let worst = res[1..].iter().flat_map(|r| res[0].iter().zip(r)).map(|(x, y)| (x - y).abs() / x.abs().max(1.0)).fold(0f32, f32::max);
    out += &format!("max rel diff variants: {worst:.3e} {}\n", if worst < 1e-4 { "AGREE" } else { "DISAGREE" });
    Ok(out)
}

/// Prefill tile screen: one dispatch per submit, unchanged-r012 whole-output comparator.
/// Shapes are production matrices; first nt=1 dispatch already executes the fixed 64-column
/// tile and thus conservatively bounds nt<=64 work. Never run a probed variant >50ms.
pub fn prefill_bench(dev:Option<&str>, m:u32, k:u32, nt:u32, iters:u32) -> Result<String,String> {
    prefill_bench_accum(dev,m,k,nt,iters,false)
}
pub fn prefill_bench_accum(dev:Option<&str>, m:u32, k:u32, nt:u32, iters:u32,accum:bool) -> Result<String,String> {
    if std::env::var("BONSAI_SCREEN_RESUME").map_or(false,|v|v=="r6") {return dense_review::screen(dev,m,k,nt,iters,accum);}
    if !matches!(nt,1|22|43|64|128|256|512) || m==0 || m>17408 || k==0 || k>17408 || k%128!=0 || iters==0 || iters>20 {return Err("unsafe prefill-bench geometry".into());}
    let steady=std::env::var("BONSAI_SCREEN_STEADY_SECONDS").ok().map(|v|v.parse::<f64>()).transpose().map_err(|_|"invalid warm duration")?.unwrap_or(0.);
    if !steady.is_finite() || !(0.0..=5.0).contains(&steady) {return Err("unsafe warm duration".into());}
    let mut rng=Rng(0xabcdef12345678);
    let an=(k as usize+m as usize)*nt as usize+64;
    let a:Vec<f32>=(0..an).map(|_|rng.f()).collect();
    let trit=std::env::var("BONSAI_SCREEN_RESUME").map_or(false,|v|matches!(v.as_str(),"a5"|"a6"));
    let w:Vec<u32>=(0..m as usize*k as usize/16).map(|_|{let mut x=0;for j in 0..16{x|=(rng.u()%if trit {3}else{4})<<(2*j);}x}).collect();
    // Real PQ2/PTQ scales originate in f16; include all four PQ2 code values.
    let sc:Vec<f32>=(0..m as usize*k as usize/128).map(|_|half::f16::from_f32(rng.f().abs()*0.05+0.01).to_f32()).collect();
    let trit_data=if trit {Some(super::prefill_pack::trit_columns(m as usize,k as usize,&w,&sc)?)}else{None};
    let s=side(dev,true,&a,&[0.;64],&w,&sc)?;
    let ab:Vec<u8>=a.iter().flat_map(|v|v.to_le_bytes()).collect();
    let yo=k*nt;let mut reference=vec![];let mut txt=String::new();
    let (cw,cd)=super::prefill_pack::columns(m as usize,k as usize,&w,&sc)?;
    let wb=s.c.buffer((cw.len()*4) as u64,false)?;let db=s.c.buffer((cd.len()*4) as u64,false)?;
    s.c.upload(&wb,0,&cw.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>());
    s.c.upload(&db,0,&cd.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>());
    let colset=s.c.set([(&s.arena,0,s.arena.size),(&s.arena,0,s.arena.size),(&wb,0,wb.size),(&db,0,db.size)]);
    let tritbuf=if let Some((tw,_))=&trit_data {let b=s.c.buffer((tw.len()*4)as u64,false)?;s.c.upload(&b,0,&tw.iter().flat_map(|v|v.to_le_bytes()).collect::<Vec<u8>>());Some(b)}else{None};
    let tritset=tritbuf.as_ref().map(|b|s.c.set([(&s.arena,0,s.arena.size),(&s.arena,0,s.arena.size),(b,0,b.size),(&db,0,db.size)]));
    let mut pipes=vec![("matmul_t2",64,64),("matmul_pf64",64,64),("matmul_pf128",128,64),("matmul_pf128x32",128,32),("matmul_pf64x32",64,32),("matmul_pf64k64",64,64),("matmul_pf512",64,64),("matmul_pf_h16",64,64),("matmul_r64",64,64),("matmul_r128",128,128),("matmul_ru8",64,64),("matmul_rb64",64,64),("matmul_rb128",128,128),("matmul_col",64,64),("matmul_col_k16",64,64),("matmul_col_pipe",64,64),("matmul_colsoa",64,64),("matmul_colwave",64,64),("matmul_lut2",64,32),("matmul_lutv",64,32),("matmul_trit4",512,16),("matmul_trits",512,16),("matmul_col16",64,64),("matmul_col8",64,64),("matmul_col128u4",64,128),("matmul_col128u8",64,128),("matmul_col_w512",64,128),("matmul_col_r512",128,64),("matmul_col_1024",128,128),("matmul_col_r128t32",128,32),("matmul_col_r32t64",32,64),("matmul_col_r32t32",32,32),("matmul_col_u4",64,64),("matmul_col_u8",64,64),("matmul_col_u16",64,64),("matmul_col_k64",64,64),("matmul_col_fma8",64,64),("matmul_col_fma32",64,64),("matmul_col_sw4",64,64),("matmul_col_sw16",64,64),("matmul_col_dr512",128,128),("matmul_col_sq4",128,128),("matmul_col_sq",128,128),("matmul_col_tail",64,64),("matmul_col_dr16",128,64),("matmul_col_dt",64,128),("matmul_col_dr",128,64),("matmul_col128u32",64,128)];
    if std::env::var_os("BONSAI_SCREEN_REVERSE").is_some(){pipes.reverse();}
    for (pipe,rows,toks) in pipes {
        if let Ok(which)=std::env::var("BONSAI_SCREEN_RESUME") {if match which.as_str(){"c9"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_dr512"),"c8"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_sq4"),"c7"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_tail"|"matmul_col_sw4"|"matmul_col_sw16"),"c6"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_tail"|"matmul_col_dr"|"matmul_col_dr16"|"matmul_col_sq"),"c4"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_sq"),"c3"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_tail"|"matmul_col_dr16"),"c1"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_dt"|"matmul_col_dr"),"a8"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_pipe"),"a7"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_k16"),"a6"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_trits"),"a5"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_trit4"),"a4"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_colwave"),"a3"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_lutv"),"a2"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_lut2"),"a1"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_colsoa"),"r14"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col128u32"),"r13"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_fma8"|"matmul_col_fma32"),"r12"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_k64"),"r11"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_u4"|"matmul_col_u8"|"matmul_col_u16"),"r9"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_r128t32"|"matmul_col_r32t64"|"matmul_col_r32t32"),"r8"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col_w512"|"matmul_col_r512"|"matmul_col_1024"),"r7"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col128u4"|"matmul_col128u8"),"r4"=>!matches!(pipe,"matmul_t2"|"matmul_col"|"matmul_col16"|"matmul_col8"),"r3"=>!matches!(pipe,"matmul_t2"|"matmul_col"),"r2"=>!matches!(pipe,"matmul_t2"|"matmul_rb64"|"matmul_rb128"),_=>!matches!(pipe,"matmul_t2"|"matmul_r64"|"matmul_r128"|"matmul_ru8")} {continue;}}
        if pipe.starts_with("matmul_trit") && !trit {continue;} // Separate explicit ternary fixture; full-code fixtures remain unchanged.
        let set=if pipe.starts_with("matmul_trit") {tritset.unwrap()}else if pipe.starts_with("matmul_col") {colset}else{s.set};
        s.c.upload(&s.arena,0,&ab);
        let grid=(m.div_ceil(rows),1,1);
        let pc=[m,k,nt,0,k,yo,m,accum as u32];
        let t=std::time::Instant::now();s.c.begin();s.c.dispatch(pipe,set,&[m,k,1,0,k,yo,m,accum as u32],grid);s.c.submit_wait();let probe=t.elapsed().as_secs_f64();
        if probe*1e3>50. {return Err(format!("{pipe} fixed-tile probe {probe}s exceeds screen; no expanded dispatch"));}
        let mut previous=probe;let mut screened=toks.min(nt);
        while screened<nt {
            // Optional controlled warm calibration at the ALREADY screened geometry.
            // No limit/predictive margin changes; every discarded dispatch still<=50ms.
            if std::env::var_os("BONSAI_SCREEN_WARM_GEOMETRY").is_some() {
                let mut stable=std::collections::VecDeque::new();let warm=std::time::Instant::now();let mut i=0;
                while i<20 || warm.elapsed().as_secs_f64()<steady {let t=std::time::Instant::now();s.c.begin();s.c.dispatch(pipe,set,&[m,k,screened,0,k,yo,m,accum as u32],(m.div_ceil(rows),screened.div_ceil(toks),1));s.c.submit_wait();let sec=t.elapsed().as_secs_f64();if sec*1e3>50. {return Err(format!("{pipe} warm calibration exceeded screen {sec}"));}stable.push_back(sec);if stable.len()>5{stable.pop_front();}i+=1;}
                previous=stable.iter().copied().fold(0f64,f64::max);
                eprintln!("[screen-calibration] {}",serde_json::json!({"pipe":pipe,"m":m,"k":k,"nt":screened,"last_five_seconds":stable,"predictor_seconds":previous,"unchanged_limit_ms":50,"unchanged_margin":1.5}));
            }
            let next=(screened*2).min(nt);
            if previous*1e3*(next.div_ceil(screened) as f64)*1.5>50. {return Err(format!("{pipe} expanded {screened}->{next} geometry exceeds 50ms predicted screen"));}
            let t=std::time::Instant::now();s.c.begin();s.c.dispatch(pipe,set,&[m,k,next,0,k,yo,m,accum as u32],(m.div_ceil(rows),next.div_ceil(toks),1));s.c.submit_wait();previous=t.elapsed().as_secs_f64();
            if previous*1e3>50. {return Err(format!("{pipe} progressive screen exceeded 50ms at nt={next}"));}screened=next;
        }
        let grid=(m.div_ceil(rows),nt.div_ceil(toks),1);
        s.c.upload(&s.arena,0,&ab);s.c.begin();s.c.dispatch(pipe,set,&pc,grid);s.c.submit_wait();
        let mut b=vec![0u8;nt as usize*m as usize*4];s.c.download(&s.arena,yo as u64*4,&mut b);
        let result:Vec<f32>=b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
        let mut abs=0f32;let mut scaled=0f32;
        if reference.is_empty(){reference=result.clone();}else{for(&x,&y)in reference.iter().zip(&result){if !x.is_finite()||!y.is_finite(){return Err("nonfinite output".into());}abs=abs.max((x-y).abs());scaled=scaled.max((x-y).abs()/x.abs().max(1.));}}
        if scaled>1e-4{return Err(format!("{pipe}: abs={abs} scaled={scaled}; full outputs disagree"));}
        let warm_start=std::time::Instant::now();
        let mut discarded_warm_dispatches=0;
        while discarded_warm_dispatches<20 || warm_start.elapsed().as_secs_f64()<steady {let t=std::time::Instant::now();s.c.begin();s.c.dispatch(pipe,set,&pc,grid);s.c.submit_wait();if t.elapsed().as_secs_f64()*1e3>50.{return Err(format!("{pipe} discarded warm dispatch exceeded screen"));}discarded_warm_dispatches+=1;}
        let discarded_warm_seconds=warm_start.elapsed().as_secs_f64();
        let clocks=|| ["pp_dpm_sclk","pp_dpm_mclk"].iter().map(|n|((*n).to_owned(),std::fs::read_to_string(format!("/sys/class/drm/card0/device/{n}")).unwrap_or_else(|e|format!("unavailable: {e}")))).collect::<std::collections::BTreeMap<_,_>>();
        let clock_before=clocks();
        let mut times=vec![];let mut device_times=Vec::new();
        for _ in 0..iters {let t=std::time::Instant::now();s.c.begin();s.c.dispatch(pipe,set,&pc,grid);s.c.submit_wait();let sec=t.elapsed().as_secs_f64();if sec*1e3>50.{return Err(format!("{pipe} measured screen exceeded {sec}"));}times.push(sec);
            if let Some((qp,_,_))=&s.c.prof {let mut ts=[0u64;2];unsafe{s.c.device.get_query_pool_results(*qp,0,&mut ts,vk::QueryResultFlags::TYPE_64|vk::QueryResultFlags::WAIT).map_err(|e|e.to_string())?;}device_times.push(ts[1].wrapping_sub(ts[0]) as f64*s.c.props.limits.timestamp_period as f64*1e-9);}
        }
        txt+=&format!("{}\n",serde_json::json!({"kernel":pipe,"m":m,"k":k,"nt":nt,"accumulate":accum,"seconds":times,"device_seconds":device_times,"discarded_warm_dispatches":discarded_warm_dispatches,"clock_before":clock_before,"clock_after":clocks(),"discarded_warm_seconds":discarded_warm_seconds,"max_abs":abs,"max_scaled":scaled,"outputs":result.len(),"completed_dispatches":s.c.completed_dispatches.get()}));
    }
    Ok(txt)
}

/// Query-sharing attention parity/screen. f16 KV, same split-K chunk and combine ABI.
/// Small nt fixtures keep this microbenchmark below the producer 50ms dispatch screen.
pub fn prefill_attention_bench(dev:Option<&str>,ctx:u32,nt:u32,iters:u32)->Result<String,String>{
    if ctx>7000 || !matches!(nt,1|3|7|8|22|43|64) || (ctx>500 && nt>8) || iters==0 || iters>5 {return Err("unsafe prefill attention screen geometry".into());}
    prefill_attention_bench_inner(dev,ctx,nt,iters,false)
}

/// Progressively screen only already-parity-checked GQA against quad256. Original
/// canonical fixture guard remains unchanged. Stop before expanding a >50ms prediction.
pub fn prefill_gqa_bench(dev:Option<&str>)->Result<String,String>{
    let mut out=String::new();let mut previous=0f64;let mut old_work=1u32;
    for (ctx,nt) in [(7000u32,8u32),(7000,16),(7000,32),(7000,64),(8128,64)] {
        let work=nt*(ctx+nt).div_ceil(256);
        if previous*work as f64/old_work as f64*1.5>0.050 {return Err(format!("GQA geometry {ctx}/{nt} predictive screen exceeded; no dispatch"));}
        let rows=prefill_attention_bench_inner(dev,ctx,nt,5,true)?;
        previous=0.;for line in rows.lines(){let v:serde_json::Value=serde_json::from_str(line).unwrap();for t in v["seconds"].as_array().unwrap(){previous=previous.max(t.as_f64().unwrap());}}
        old_work=work;out+=&rows;
    }
    Ok(out)
}
fn prefill_attention_bench_inner(dev:Option<&str>,ctx:u32,nt:u32,iters:u32,gqa_only:bool)->Result<String,String>{
    let(nh,nkv,hd)=(24u32,4u32,256u32);let kvd=nkv*hd;let qts=2*nh*hd;let total=ctx+nt;let cn=256;let ns=total.div_ceil(cn);
    let qo=0;let kc=nt*qts;let vc=kc+total*kvd/2;let oo=vc+total*kvd/2;let scr=oo+nt*nh*hd+64;let an=(scr+nt*nh*ns*260+64)as usize;
    let mut rng=Rng(0xaabbccddee);let mut a:Vec<f32>=(0..an).map(|_|rng.f()).collect();
    for off in [kc,vc]{for i in 0..total*kvd/2{let lo=half::f16::from_f32(rng.f()).to_bits()as u32;let hi=half::f16::from_f32(rng.f()).to_bits()as u32;a[(off+i)as usize]=f32::from_bits(lo|(hi<<16));}}
    let s=side(dev,true,&a,&[0.;64],&[0;64],&[0.;64])?;let ab:Vec<u8>=a.iter().flat_map(|x|x.to_le_bytes()).collect();let mut reference=vec![];let mut txt=String::new();
    let mut attention_pipes=vec![("attn_s_h",1,1),("attn_prefill_h4",4,1),("attn_prefill_h4_256",4,1),("attn_prefill_h2_256",2,1)];
    if gqa_only {attention_pipes=vec![("attn_prefill_h4_256",4,1),("attn_gqa6",1,6),("attn_gqa3",1,3),("attn_gqa2",1,2)];}
    else if std::env::var_os("BONSAI_SCREEN_GQA").is_some(){attention_pipes.extend([("attn_gqa8",4,2),("attn_gqa6",1,6)]);}
    for (pipe,queries,heads) in attention_pipes {
        let run=||{s.c.begin();s.c.dispatch(pipe,s.set,&[qo,qts,kc,vc,kvd,ctx,hd,nh,nkv,scr,cn,ns,(1f32/16.).to_bits(),nt],(nh/heads,nt.div_ceil(queries),ns));s.c.dispatch("attn_comb",s.set,&[qo,qts,ctx,nh,scr,cn,ns,oo,nh*hd],(nh,nt,1));s.c.submit_wait();};
        s.c.upload(&s.arena,0,&ab);let t=std::time::Instant::now();run();let probe=t.elapsed().as_secs_f64();if probe*1e3>50.{return Err(format!("{pipe} bounded nt={nt} fixture exceeded screen {probe}s"));}
        let mut b=vec![0u8;nt as usize*nh as usize*hd as usize*4];s.c.download(&s.arena,oo as u64*4,&mut b);let result:Vec<f32>=b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect();
        let mut abs=0f32;let mut scaled=0f32;let mut words=0;
        if reference.is_empty(){reference=result.clone();}else{for(&x,&y)in reference.iter().zip(&result){if !x.is_finite()||!y.is_finite(){return Err("nonfinite attention output".into());}abs=abs.max((x-y).abs());scaled=scaled.max((x-y).abs()/x.abs().max(1.));if x.to_bits()!=y.to_bits(){words+=1;}}}
        if scaled>1e-4{return Err(format!("{pipe} whole output disagreement {scaled}"));}
        for _ in 0..20 {let t=std::time::Instant::now();run();if t.elapsed().as_secs_f64()*1e3>50.{return Err(format!("{pipe} discarded attention warm exceeded screen"));}}
        let mut times=vec![];for _ in 0..iters{let t=std::time::Instant::now();run();let sec=t.elapsed().as_secs_f64();if sec*1e3>50.{return Err(format!("{pipe} timing screen exceeded {sec}"));}times.push(sec);}
        txt+=&format!("{}\n",serde_json::json!({"kernel":pipe,"ctx":ctx,"nt":nt,"seconds":times,"max_abs":abs,"max_scaled":scaled,"differing_words":words,"outputs":result.len(),"completed_dispatches":s.c.completed_dispatches.get()}));
    }Ok(txt)
}

/// Hard per-submit budget for GPU microbenchmarks (ms; env BONSAI_BENCH_BUDGET_MS, default 50, max 500).
/// amdgpu's gfx ring timeout is seconds; staying ≤ 500 ms per submit keeps a wide margin on a shared card.
pub fn budget_ms() -> f64 { std::env::var("BONSAI_BENCH_BUDGET_MS").ok().and_then(|v| v.parse().ok()).unwrap_or(50.0f64).min(500.0) }

/// Attention microbenchmark + parity: old single-WG kernel vs r012 split-K (attn_s + attn_comb), 24 heads, 4 KV heads, hd 256.
/// Safety (r012 incident 23:29 CT): every dispatch is its own submit; a kernel is only run at (ctx, nt) after a probe at
/// nt=1 predicts the call (×nt×2 margin) stays under budget_ms(); anything predicted over budget is SKIPPED, not run.
pub fn attn_bench(dev: Option<&str>, ctx: u32, nt: u32, iters: u32) -> Result<String, String> {
    let (nh, nkv, hd) = (24u32, 4u32, 256u32); let kvd = nkv * hd; let qts = 2 * nh * hd; let total = ctx + nt;
    let qo = 0u32; let kco = qo + nt * qts; let vco = kco + total * kvd; let oo = vco + total * kvd; let scr = oo + nt * nh * hd + 64;
    let room = (nt * nh).max(64) * total.max(64) * 4; // also covers the old kernel's nt*nh*total scratch
    let an = (scr + room) as usize;
    let mut r = Rng(0x5eed);
    let a: Vec<f32> = (0..an).map(|_| r.f()).collect();
    let s = side(dev, true, &a, &[0.0; 64], &[0u32; 64], &[0.0; 64])?;
    let ab: Vec<u8> = a.iter().flat_map(|v| v.to_le_bytes()).collect();
    let scale = (1.0f32 / 16.0).to_bits(); let pos0 = ctx;
    let mut c = 64u32; while c < 1024 && nt * nh * total.div_ceil(c) * 260 > room { c *= 2; }
    let smax = total.div_ceil(c);
    if nt * nh * smax * 260 > room { return Err("attn-bench: scratch too small".into()); }
    let run = |new: bool, ntr: u32| {
        s.c.begin();
        if new {
            s.c.dispatch("attn_s", s.set, &[qo, qts, kco, vco, kvd, pos0, hd, nh, nkv, scr, c, smax, scale], (nh, ntr, smax));
            s.c.dispatch("attn_comb", s.set, &[qo, qts, pos0, nh, scr, c, smax, oo, nh * hd], (nh, ntr, 1));
        } else { s.c.dispatch("attn", s.set, &[qo, qts, kco, vco, kvd, pos0, hd, nh, nkv, oo, nh * hd, scr, total, scale], (nh, ntr, 1)); }
        s.c.submit_wait();
    };
    let budget = budget_ms(); let mut outs = vec![]; let mut txt = String::new();
    s.c.upload(&s.arena, 0, &ab);
    for new in [false, true] {
        let name = if new { "split-K" } else { "old" };
        let t = std::time::Instant::now(); run(new, 1); let probe = t.elapsed().as_secs_f64() * 1e3;
        let pred = probe * nt as f64 * 2.0;
        if probe > budget || pred > budget { txt += &format!("attn {name:9} ctx={ctx:5} nt={nt:2}: SKIPPED (probe {probe:.1} ms, predicted {pred:.1} ms > budget {budget} ms)\n"); outs.push(None); continue; }
        s.c.upload(&s.arena, 0, &ab); run(new, nt);
        let mut b = vec![0u8; (nt * nh * hd * 4) as usize]; s.c.download(&s.arena, oo as u64 * 4, &mut b);
        outs.push(Some(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>()));
        let mut tot = 0f64; let mut n = 0u32;
        for _ in 0..iters { let t = std::time::Instant::now(); run(new, nt); let e = t.elapsed().as_secs_f64(); tot += e; n += 1; if e * 1e3 > budget { break; } }
        txt += &format!("attn {name:9} ctx={ctx:5} nt={nt:2}: {:9.1} us/call\n", tot * 1e6 / n as f64);
    }
    match (&outs[0], &outs[1]) {
        (Some(x), Some(y)) => { let w = x.iter().zip(y).map(|(x, y)| (x - y).abs() / x.abs().max(1.0)).fold(0f32, f32::max);
            txt += &format!("max rel diff: {w:.3e} {}\n", if w < 1e-4 { "AGREE" } else { "DISAGREE" }); }
        _ => txt += "max rel diff: n/a (a side skipped)\n",
    }
    Ok(txt)
}
