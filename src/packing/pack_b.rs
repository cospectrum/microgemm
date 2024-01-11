use crate::{packing::block::RowMajor, Layout, MatRef};
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn pack_b<T>(
    nr: usize,
    bpack: &mut [T],
    b: &MatRef<T>,
    b_rows: Range<usize>,
    b_cols: Range<usize>,
) -> Layout
where
    T: One + Zero + Copy,
{
    let kc = b_rows.len();
    let nc = b_cols.len();
    assert_eq!(bpack.len(), kc * nc);
    assert_eq!(nc % nr, 0);

    let start = b_cols.start;
    for i in 0..nc / nr {
        let buf = &mut bpack[kc * nr * i..kc * nr * (i + 1)];
        let cols = start + nr * i..start + nr * (i + 1);
        RowMajor(buf).init_from(b, b_rows.clone(), cols);
    }
    Layout::RowMajor
}
