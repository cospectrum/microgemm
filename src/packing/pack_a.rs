use crate::MatRef;
use core::ops::Range;
use num_traits::{One, Zero};

// split submatrix to col-major blocks
#[inline]
pub(crate) fn pack_a<T>(
    mr: usize,
    apack: &mut [T],
    a: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: One + Zero + Copy,
{
    let mc = rows.len();
    let kc = cols.len();
    assert_eq!(apack.len(), mc * kc);
    assert_eq!(mc % mr, 0);

    assert!(cols.end <= a.ncols());
    assert!(rows.start < a.nrows());

    let start = rows.start;
    let blocks = mc / mr;

    let mut it = apack.iter_mut();

    for i in 0..blocks - 1 {
        let rows = start + mr * i..start + mr * (i + 1);
        for col in cols.clone() {
            for row in rows.start..rows.end {
                let dst = it.next().unwrap();
                *dst = a.get(row, col);
            }
        }
    }

    let zero = T::zero();
    let i = blocks - 1;
    let rows = start + mr * i..start + mr * (i + 1);
    let min = a.nrows().min(rows.end);

    for col in cols.clone() {
        for row in rows.start..min {
            let dst = it.next().unwrap();
            *dst = a.get(row, col);
        }
        for _ in min..rows.end {
            let dst = it.next().unwrap();
            *dst = zero;
        }
    }
}
