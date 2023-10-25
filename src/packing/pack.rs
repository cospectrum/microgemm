use crate::{
    packing::block::{ColMajor, RowMajor},
    Layout, MatRef, PackSizes,
};
use core::ops::Range;
use num_traits::{One, Zero};

pub(crate) fn pack_a<T>(
    mr: usize,
    pack_sizes: &PackSizes,
    apack: &mut [T],
    a: &MatRef<T>,
    a_rows: Range<usize>,
    a_cols: Range<usize>,
) -> Layout
where
    T: One + Zero + Copy,
{
    let mc = pack_sizes.mc;
    let kc = pack_sizes.kc;
    debug_assert_eq!(a_rows.len(), mc);
    debug_assert_eq!(a_cols.len(), kc);
    debug_assert_eq!(apack.len(), mc * kc);
    debug_assert_eq!(mc % mr, 0);

    let start = a_rows.start;
    for i in 0..mc / mr {
        let buf = &mut apack[mr * kc * i..mr * kc * (i + 1)];
        let rows = start + mr * i..start + mr * (i + 1);
        ColMajor(buf).init_from(a, rows, a_cols.clone());
    }
    Layout::ColMajor
}

pub(crate) fn pack_b<T>(
    nr: usize,
    pack_sizes: &PackSizes,
    bpack: &mut [T],
    b: &MatRef<T>,
    b_rows: Range<usize>,
    b_cols: Range<usize>,
) -> Layout
where
    T: One + Zero + Copy,
{
    let kc = pack_sizes.kc;
    let nc = pack_sizes.nc;
    debug_assert_eq!(b_rows.len(), kc);
    debug_assert_eq!(b_cols.len(), nc);
    debug_assert_eq!(bpack.len(), kc * nc);
    debug_assert_eq!(nc % nr, 0);

    let start = b_cols.start;
    for i in 0..nc / nr {
        let buf = &mut bpack[kc * nr * i..kc * nr * (i + 1)];
        let cols = start + nr * i..start + nr * (i + 1);
        RowMajor(buf).init_from(b, b_rows.clone(), cols);
    }
    Layout::RowMajor
}
