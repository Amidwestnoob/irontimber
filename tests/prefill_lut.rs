#[test]
fn pair_lut_algebra_geometry_cpu() {
    // Exact integer inputs cover all16 code pairs including PQ2 code3=2.
    for x in -7..=7 {for y in -7..=7 {for code in 0u32..16 {
        let a=(code&3) as i32-1;let b=(code>>2) as i32-1;
        let v=match b {-1=>-y,0=>0,1=>y,_=>y+y};
        let got=match a {-1=>v-x,0=>v,1=>v+x,_=>v+(x+x)};
        assert_eq!(got,a*x+b*y);
    }}}
    let mut outputs=std::collections::BTreeSet::new();let mut lds=std::collections::BTreeSet::new();
    let mut waves=std::collections::BTreeSet::new();for l in 0..256 {assert!(waves.insert(((l&7)|((l>>3)&8),((l>>3)&7)|((l>>4)&8))));}
    assert_eq!(waves.len(),256);assert!(waves.iter().all(|&(x,y)|x<16 && y<16));
    for l in 0..256 {for r in 0..8 {assert!(outputs.insert((l/32+r*8,l%32)));}
        for code in 0..16 {assert!(lds.insert((l/32)*512+code*32+l%32));}}
    assert_eq!(outputs.len(),64*32);assert_eq!(lds.into_iter().collect::<Vec<_>>(),(0..4096).collect::<Vec<_>>());
    let mut vector_outputs=std::collections::BTreeSet::new();let mut vector_lds=std::collections::BTreeSet::new();
    for l in 0..256 {for r in 0..2 {for c in 0..4 {assert!(vector_outputs.insert((l/8+r*32,(l%8)*4+c)));}}
        for a in 0..4 {assert!(vector_lds.insert(((l/8)%8)*128+(l/64)*32+a*8+l%8));}}
    assert_eq!(vector_outputs,outputs);assert_eq!(vector_lds.into_iter().collect::<Vec<_>>(),(0..1024).collect::<Vec<_>>());
    // Non-overlap, bounded LDS16KiB and complete output coverage including masked tails.
    for m in [1,63,64,65,1001,17408] {for nt in [1,22,32,43,64,128,256] {
        let mut count=0;for gy in 0..(nt+31)/32 {for gx in 0..(m+63)/64 {for &(r,t) in &outputs {if gx*64+r<m && gy*32+t<nt {count+=1;}}}}
        assert_eq!(count,m*nt);
    }}
}
