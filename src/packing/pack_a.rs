use crate::MatRef;
use core::ops::Range;
use num_traits::{One, Zero};

// split the submatrix into col-major blocks
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
            let idx = a.idx(rows.start, col);
            let lane = a.as_slice()[idx..]
                .iter()
                .step_by(a.row_stride())
                .take(rows.len());
            for &row in lane {
                let dst = it.next().unwrap();
                *dst = row;
            }
        }
    }

    let zero = T::zero();
    let i = blocks - 1;
    let rows = start + mr * i..start + mr * (i + 1);
    let min = a.nrows().min(rows.end);

    for col in cols.clone() {
        let range = rows.start..min;
        let idx = a.idx(range.start, col);
        let lane = a
            .as_slice()
            .iter()
            .skip(idx)
            .step_by(a.row_stride())
            .take(range.len());
        for &row in lane {
            let dst = it.next().unwrap();
            *dst = row;
        }

        for _ in min..rows.end {
            let dst = it.next().unwrap();
            *dst = zero;
        }
    }
}
