//! Lossless alternate packed layout for prefill, not a quantization change.
//! Four adjacent rows form one 16-byte global load; original decode layout stays intact.
pub fn columns(rows: usize, cols: usize, words: &[u32], scales: &[f32]) -> Result<(Vec<u32>, Vec<f32>), String> {
    if rows == 0 || cols == 0 || cols % 128 != 0 { return Err("invalid column-pack geometry".into()); }
    let padded = rows.checked_add(3).ok_or("column-pack overflow")? / 4 * 4;
    let nw = padded.checked_mul(cols / 16).ok_or("column-pack overflow")?;
    let ns = padded.checked_mul(cols / 128).ok_or("column-pack overflow")?;
    if rows.checked_mul(cols / 16) != Some(words.len()) || rows.checked_mul(cols / 128) != Some(scales.len()) {
        return Err("column-pack source length mismatch".into());
    }
    let mut w = vec![0; nw]; let mut d = vec![0.; ns];
    for r in 0..rows {
        for k in 0..cols/16 { w[k*padded+r] = words[r*(cols/16)+k]; }
        for k in 0..cols/128 { d[k*padded+r] = scales[r*(cols/128)+k]; }
    }
    Ok((w,d))
}
/// Lossless four-trit radix3 bytes, column-major like columns(). Code3 is unsupported,
/// never approximated: caller must retain/use the original four-code kernel.
pub fn trit_columns(rows:usize,cols:usize,words:&[u32],scales:&[f32])->Result<(Vec<u32>,Vec<f32>),String>{
    let (mut w,d)=columns(rows,cols,words,scales)?;
    for word in &mut w {
        let old=*word;let mut packed=0;
        for g in 0..4 {let mut code=0;let mut power=1;for j in 0..4 {
            let c=(old>>(g*8+j*2))&3;if c==3 {return Err("radix3 requires code0..2; use original kernel for code3".into());}
            code+=c*power;power*=3;
        }packed|=code<<(g*8);}*word=packed;
    }Ok((w,d))
}
/// Reserve 1GiB; reject overflow/overcommit without attempting an allocation.
pub fn allocation_fits(total:u64,used:u64,additional:u64)->bool {
    used.checked_add(additional).and_then(|x|x.checked_add(1u64<<30)).map_or(false,|x|x<=total)
}
#[cfg(test)] mod tests {
    use super::*;
    #[test] fn roundtrip_bits_cpu() {
        for m in [1usize,3,4,63,64,65,1000] {for k in [128usize,1024] {
            let w:Vec<u32>=(0..m*k/16).map(|x|(x as u32).wrapping_mul(0xdeadbeef)).collect();
            let d:Vec<f32>=(0..m*k/128).map(|x|f32::from_bits((x as u32).wrapping_mul(0x1020304))).collect();
            let (a,b)=columns(m,k,&w,&d).unwrap();let p=m.div_ceil(4)*4;
            for r in 0..m {for q in 0..k/16 {assert_eq!(a[q*p+r],w[r*k/16+q]);} for q in 0..k/128 {assert_eq!(b[q*p+r].to_bits(),d[r*k/128+q].to_bits());}}
            for r in m..p {for q in 0..k/16 {assert_eq!(a[q*p+r],0);}for q in 0..k/128 {assert_eq!(b[q*p+r].to_bits(),0);}}
        }}
    }
    #[test] fn invalid_geometry_cpu() {
        for (m,k) in [(0,128),(1,0),(1,127),(usize::MAX,128),(1,usize::MAX)] {assert!(columns(m,k,&[],&[]).is_err());}
        assert!(columns(4,128,&[0;31],&[0.;4]).is_err());
    }
    #[test] fn capacity_boundary_cpu() {
        assert!(allocation_fits(16<<30,1<<30,14<<30));
        assert!(!allocation_fits(16<<30,1<<30,(14<<30)+1));
        assert!(!allocation_fits(u64::MAX,u64::MAX,1));
        assert!(!allocation_fits(0,0,0));
    }
    #[test] fn radix3_roundtrip_and_reject_cpu() {
        for code in 0..81u32 {let mut n=code;let mut byte=0;for j in 0..4 {byte|=(n%3)<<(j*2);n/=3;}
            let word=byte*0x01010101;let (w,d)=trit_columns(1,128,&[word;8],&[0.125]).unwrap();
            assert_eq!(w[0],code*0x01010101);assert_eq!(d[0],0.125);
        }
        assert!(trit_columns(1,128,&[3;8],&[1.]).is_err());
        assert!(trit_columns(0,128,&[],&[]).is_err());
    }
}
