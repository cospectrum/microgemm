use crate::{packing::block::RowMajor, Layout, MatRef, PackSizes};
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
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
