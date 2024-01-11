use crate::{packing::block::ColMajor, Layout, MatRef};
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn pack_a<T>(
    mr: usize,
    apack: &mut [T],
    a: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> Layout
where
    T: One + Zero + Copy,
{
    let mc = rows.len();
    let kc = cols.len();
    assert_eq!(apack.len(), mc * kc);
    assert_eq!(mc % mr, 0);
    assert!(kc <= a.ncols());

    let block_size = mr * kc;
    let start = rows.start;

    for i in 0..mc / mr {
        let buf = &mut apack[block_size * i..block_size * (i + 1)];
        let rows = start + mr * i..start + mr * (i + 1);
        ColMajor(buf).init_from(a, rows, cols.clone());
    }
    Layout::ColMajor
}
