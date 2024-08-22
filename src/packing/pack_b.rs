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
    assert!(nr <= nc);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::std_prelude::*;

    pub(super) fn bpack_ref<T>(
        nr: usize,
        b: MatRef<T>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<T>
    where
        T: Zero + Copy,
    {
        let mut bpack = vec![T::zero(); rows.len() * cols.len()];
        pack_b_ref(nr, bpack.as_mut(), b, rows, cols);
        bpack
    }
    pub(super) fn pack_b_ref<T>(
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
        assert!(nr <= nc);
        assert_eq!(nc % nr, 0);

        assert!(rows.end <= b.nrows());
        assert!(cols.start < b.ncols());

        let number_of_blocks = nc / nr;
        let mut it = bpack.iter_mut();

        let cols_offset = cols.start;

        for nblock in 0..number_of_blocks {
            let block_cols = cols_offset + nr * nblock..cols_offset + nr * (nblock + 1);
            for row in rows.clone() {
                for col in block_cols.clone() {
                    let dst = it.next().unwrap();
                    *dst = b.get_or_zero(row, col);
                }
            }
        }
    }
}

#[cfg(test)]
mod proptests {
    use super::{tests::bpack_ref, *};
    use crate::utils::arb_matrix;
    use proptest::{prelude::*, proptest};

    proptest! {
        #[test]
        fn proptest_pack_b(
            b in arb_matrix::<i8>(1..40, 1..40),
            nr in (1..41usize),
        ) {
            let b_ref = b.to_ref();

            let arb_rows = (0..=b.nrows())
                .prop_flat_map(|start| (start..=b.nrows()).prop_map(move |end| start..end));

            const TAKE: usize = 50;
            let arb_cols = (0..b.ncols())
                .prop_flat_map(|start| (start..start + TAKE).prop_map(move |end| (start..end)))
                .prop_filter("cols", |cols| nr <= cols.len() && cols.len() % nr == 0);

            proptest!(|(rows in arb_rows, cols in arb_cols)| {
                let mut bpack = vec![-1; rows.len() * cols.len()];
                pack_b(nr, &mut bpack, b_ref, rows.clone(), cols.clone());
                let expect = bpack_ref(nr, b_ref, rows, cols);
                prop_assert_eq!(bpack, expect);
            });
        }
    }
}
