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
    assert_eq!(bpack.len(), kc.checked_mul(nc).unwrap());

    assert!(nr <= nc);
    assert!(nr > 0);
    assert_eq!(nc % nr, 0);

    assert!(rows.end <= b.nrows());
    assert!(cols.start < b.ncols());
    let stride = b.col_stride();
    assert!(stride > 0);

    let mut it = bpack;
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

            if stride == 1 {
                let lane = &b.as_slice()[idx..idx + nr];
                it[..nr].copy_from_slice(lane);
            } else {
                let lane = b.as_slice()[idx..].iter().step_by(stride).take(nr);
                debug_assert_eq!(lane.len(), nr);
                let zip = lane.zip(&mut it[..nr]);
                #[cfg(not(kani))]
                zip.for_each(|(&src, dst)| {
                    *dst = src;
                });
            }
            it = &mut it[nr..];
        }
    }

    let remains = (cols_stop_at - cols.start) % nr;
    debug_assert!(remains < nr);
    if remains > 0 {
        let nblock = number_of_valid_blocks;
        let block_cols = cols_offset + nr * nblock..cols_stop_at;
        debug_assert!(block_cols.start < block_cols.end);
        debug_assert_eq!(block_cols.len(), remains);

        for row in rows.clone() {
            debug_assert!(row < b.nrows());
            let idx = b.idx(row, block_cols.start);

            if stride == 1 {
                let lane = &b.as_slice()[idx..idx + remains];
                it[..remains].copy_from_slice(lane);
            } else {
                let lane = b.as_slice()[idx..].iter().step_by(stride).take(remains);
                debug_assert_eq!(lane.len(), remains);
                let zip = lane.zip(&mut it[..remains]);
                #[cfg(not(kani))]
                zip.for_each(|(&src, dst)| {
                    *dst = src;
                });
            }
            #[cfg(not(kani))]
            it[remains..nr].fill(T::zero());
            it = &mut it[nr..];
        }
    }

    #[cfg(not(kani))]
    it.fill(T::zero());
}

#[cfg(test)]
mod reference {
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
        assert_eq!(bpack.len(), kc.checked_mul(nc).unwrap());

        assert!(nr <= nc);
        assert!(nr > 0);
        assert_eq!(nc % nr, 0);

        assert!(rows.end <= b.nrows());
        assert!(cols.start < b.ncols());
        assert!(b.col_stride() > 0);

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
    use super::{reference::*, *};
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

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(3)] // 1 + max(kc, number_of_valid_blocks)
    fn check_pack_b() -> Option<()> {
        const KC_LIMIT: usize = 2;
        const NUMBER_OF_VALID_BLOCKS_LIMIT: usize = 2;

        const PACK_LEN_LIMIT: usize = 11;
        const VALUES_LEN_LIMIT: usize = 13;
        const NC_LIMIT: usize = 13;

        let values = kani::vec::any_vec::<i8, VALUES_LEN_LIMIT>();
        let b = {
            let nrows = kani::any();
            let ncols = kani::any();
            let row_stride = kani::any();
            let col_stride = kani::any_where(|&col_stride| col_stride > 0);
            MatRef::from_parts(nrows, ncols, &values, row_stride, col_stride)?
        };

        let nr: usize = kani::any_where(|&nr| nr > 0);

        let rows: Range<usize> = kani::any()..kani::any();
        let cols: Range<usize> = kani::any()..kani::any();

        let kc = rows.len();
        let nc = cols.len();
        kani::assume(kc <= KC_LIMIT);
        kani::assume(nc <= NC_LIMIT);
        kani::assume(nr <= nc && nc % nr == 0);

        kani::assume(rows.end <= b.nrows());
        kani::assume(cols.start < b.ncols());

        let number_of_valid_blocks = {
            let cols_stop_at = cols.end.min(b.ncols());
            (cols_stop_at - cols.start) / nr
        };
        kani::assume(number_of_valid_blocks <= NUMBER_OF_VALID_BLOCKS_LIMIT);

        kani::assume(kc * nc <= PACK_LEN_LIMIT);
        let mut bpack = vec![0; kc * nc];
        pack_b(nr, &mut bpack, b, rows, cols);

        Some(())
    }
}
