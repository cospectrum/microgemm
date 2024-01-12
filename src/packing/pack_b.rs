use crate::MatRef;
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn pack_b<T>(
    nr: usize,
    bpack: &mut [T],
    b: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: One + Zero + Copy,
{
    let kc = rows.len();
    let nc = cols.len();
    assert_eq!(bpack.len(), kc * nc);
    assert_eq!(nc % nr, 0);

    assert!(rows.end <= b.nrows());
    assert!(cols.start <= b.ncols());

    let start = cols.start;
    let blocks = nc / nr;

    let mut it = bpack.iter_mut();

    for i in 0..blocks - 1 {
        let cols = start + nr * i..start + nr * (i + 1);
        for row in rows.clone() {
            for col in cols.start..cols.end {
                let dst = it.next().unwrap();
                *dst = b.get(row, col);
            }
        }
    }

    let i = blocks - 1;
    let cols = start + nr * i..start + nr * (i + 1);
    for row in rows.clone() {
        for col in cols.start..cols.end {
            let dst = it.next().unwrap();
            if col < b.ncols() {
                *dst = b.get(row, col);
            } else {
                *dst = T::zero();
            }
        }
    }
}
