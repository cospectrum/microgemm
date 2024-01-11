use crate::{packing::block::ColMajor, Layout, MatRef};
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn pack_a<T>(
    mr: usize,
    apack: &mut [T],
    a: &MatRef<T>,
    a_rows: Range<usize>,
    a_cols: Range<usize>,
) -> Layout
where
    T: One + Zero + Copy,
{
    let mc = a_rows.len();
    let kc = a_cols.len();
    assert_eq!(apack.len(), mc * kc);
    assert_eq!(mc % mr, 0);

    let start = a_rows.start;
    for i in 0..mc / mr {
        let buf = &mut apack[mr * kc * i..mr * kc * (i + 1)];
        let rows = start + mr * i..start + mr * (i + 1);
        ColMajor(buf).init_from(a, rows, a_cols.clone());
    }
    Layout::ColMajor
}
