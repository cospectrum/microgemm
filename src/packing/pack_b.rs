use crate::MatRef;
use core::ops::Range;
use num_traits::{One, Zero};

// split the submatrix into row-major blocks
#[inline]
pub(crate) fn pack_b<T>(
    nr: usize,
    bpack: &mut [T],
    b: MatRef<T>,
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
    assert!(cols.start < b.ncols());

    let start = cols.start;
    let blocks = nc / nr;

    let mut it = bpack.iter_mut();

    for i in 0..blocks - 1 {
        let cols = start + nr * i..start + nr * (i + 1);
        debug_assert!(cols.start < b.ncols());
        for row in rows.clone() {
            debug_assert!(row < b.nrows());
            let idx = b.idx(row, cols.start);
            let lane = b.as_slice()[idx..]
                .iter()
                .step_by(b.col_stride())
                .take(cols.len());
            for &col in lane {
                let dst = it.next().unwrap();
                *dst = col;
            }
        }
    }

    let zero = T::zero();
    let i = blocks - 1;
    let cols = start + nr * i..start + nr * (i + 1);
    let min = b.ncols().min(cols.end);

    for row in rows.clone() {
        let range = cols.start..min;
        debug_assert!(row < b.nrows());
        debug_assert!(range.start < b.ncols());
        let idx = b.idx(row, range.start);
        let lane = b
            .as_slice()
            .iter()
            .skip(idx)
            .step_by(b.col_stride())
            .take(range.len());
        for &col in lane {
            let dst = it.next().unwrap();
            *dst = col;
        }

        for _ in min..cols.end {
            let dst = it.next().unwrap();
            *dst = zero;
        }
    }
}
