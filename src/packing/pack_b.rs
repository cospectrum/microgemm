use crate::{packing::block::RowMajor, Layout, MatRef};
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn pack_b<T>(
    nr: usize,
    bpack: &mut [T],
    b: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> Layout
where
    T: One + Zero + Copy,
{
    let kc = rows.len();
    let nc = cols.len();
    assert_eq!(bpack.len(), kc * nc);
    assert_eq!(nc % nr, 0);
    assert!(kc <= b.nrows());

    let block_size = kc * nr;
    let start = cols.start;

    for i in 0..nc / nr {
        let buf = &mut bpack[block_size * i..block_size * (i + 1)];
        let cols = start + nr * i..start + nr * (i + 1);
        RowMajor(buf).init_from(b, rows.clone(), cols);
    }
    Layout::RowMajor
}
