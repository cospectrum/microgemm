use crate::MatRef;
use core::ops::Range;
use num_traits::Zero;

// Pack the submatrix b[rows, cols] into (nc/nr) row-major blocks of size kc x nr.
// Values outsize of `b` will be zeroed.
#[inline]
pub(crate) fn pack_b<T>(
    nr: usize,
    bpack: &mut [T],
    b: MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: Zero + Copy,
{
    let kc = rows.len();
    let nc = cols.len();
    assert_eq!(bpack.len(), kc * nc);
    assert_eq!(nc % nr, 0);

    assert!(rows.end <= b.nrows());
    assert!(cols.start < b.ncols());

    let mut it = bpack.iter_mut();
    let cols_offset = cols.start;

    let cols_stop_at = cols.end.min(b.ncols());
    let number_of_valid_blocks = (cols_stop_at - cols.start) / nr;
    debug_assert!(number_of_valid_blocks <= nc / nr);

    for nblock in 0..number_of_valid_blocks {
        let block_cols = cols_offset + nr * nblock..cols_offset + nr * (nblock + 1);
        debug_assert!(block_cols.start < b.ncols());
        debug_assert!(block_cols.end <= b.ncols());
        debug_assert_eq!(block_cols.len(), nr);

        for row in rows.clone() {
            debug_assert!(row < b.nrows());
            let idx = b.idx(row, block_cols.start);
            let lane = b.as_slice()[idx..]
                .iter()
                .step_by(b.col_stride())
                .take(block_cols.len());
            for &val in lane {
                let dst = it.next().unwrap();
                *dst = val;
            }
        }
    }

    let remains = (cols_stop_at - cols.start) % nr;
    if remains > 0 {
        let nblock = number_of_valid_blocks;
        let block_cols = cols_offset + nr * nblock..cols_stop_at;
        debug_assert!(block_cols.start < block_cols.end);
        debug_assert_eq!(block_cols.len(), remains);

        let tail_len = nr - remains;
        debug_assert_eq!(block_cols.len() + tail_len, nr);

        for row in rows.clone() {
            debug_assert!(row < b.nrows());
            let idx = b.idx(row, block_cols.start);
            let lane = b.as_slice()[idx..]
                .iter()
                .step_by(b.col_stride())
                .copied()
                .take(block_cols.len());
            let tail = core::iter::repeat(T::zero()).take(tail_len);
            for val in lane.chain(tail) {
                let dst = it.next().unwrap();
                *dst = val;
            }
        }
    }

    for dst in it {
        *dst = T::zero();
    }
}
