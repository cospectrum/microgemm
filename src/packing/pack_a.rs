use crate::MatRef;
use core::ops::Range;
use num_traits::Zero;

// Pack the submatrix a[rows, cols] into (mc/mr) col-major blocks of size mr x kc.
// Values outsize of `a` will be zeroed.
#[inline]
pub(crate) fn pack_a<T>(
    mr: usize,
    apack: &mut [T],
    a: MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: Zero + Copy,
{
    let mc = rows.len();
    let kc = cols.len();
    assert_eq!(apack.len(), mc.checked_mul(kc).unwrap());

    assert!(mr <= mc);
    assert!(mr > 0);
    assert_eq!(mc % mr, 0);

    assert!(cols.end <= a.ncols());
    assert!(rows.start < a.nrows());
    assert!(a.row_stride() > 0);

    let mut it = apack;
    let rows_offset = rows.start;

    let rows_stop_at = a.nrows().min(rows.end);
    let number_of_valid_blocks = (rows_stop_at - rows.start) / mr;
    debug_assert!(number_of_valid_blocks <= mc / mr);

    for nblock in 0..number_of_valid_blocks {
        let block_rows = rows_offset + mr * nblock..rows_offset + mr * (nblock + 1);
        debug_assert!(block_rows.start < a.nrows());
        debug_assert!(block_rows.end <= a.nrows());
        debug_assert_eq!(block_rows.len(), mr);

        for col in cols.clone() {
            debug_assert!(col < a.ncols());
            let idx = a.idx(block_rows.start, col);
            let lane = a.as_slice()[idx..]
                .iter()
                .step_by(a.row_stride())
                .take(block_rows.len());
            debug_assert_eq!(lane.len(), block_rows.len());
            let n = lane.len();

            let zip = lane.zip(it[..n].as_mut());
            #[cfg(not(kani))]
            for (&src, dst) in zip {
                *dst = src;
            }
            it = it[n..].as_mut();
        }
    }

    let remains = (rows_stop_at - rows.start) % mr;
    if remains > 0 {
        let nblock = number_of_valid_blocks;
        let block_rows = rows_offset + mr * nblock..rows_stop_at;
        debug_assert!(block_rows.start < block_rows.end);
        debug_assert_eq!(block_rows.len(), remains);

        let tail_len = mr - remains;
        debug_assert_eq!(block_rows.len() + tail_len, mr);

        for col in cols.clone() {
            debug_assert!(col < a.ncols());
            let idx = a.idx(block_rows.start, col);
            let lane = a.as_slice()[idx..]
                .iter()
                .step_by(a.row_stride())
                .copied()
                .take(block_rows.len());
            debug_assert_eq!(lane.len(), block_rows.len());
            let tail = core::iter::repeat(T::zero()).take(tail_len);

            let n = lane.len() + tail_len;
            let zip = lane.chain(tail).zip(it[..n].as_mut());
            #[cfg(not(kani))]
            for (src, dst) in zip {
                *dst = src;
            }
            it = it[n..].as_mut();
        }
    }

    #[cfg(not(kani))]
    for dst in it {
        *dst = T::zero();
    }
}

#[cfg(test)]
mod reference {
    use super::*;
    use crate::std_prelude::Vec;

    pub(super) fn apack_ref<T>(
        mr: usize,
        a: MatRef<T>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<T>
    where
        T: Zero + Copy,
    {
        let mut apack = vec![T::zero(); rows.len() * cols.len()];
        pack_a_ref(mr, &mut apack, a, rows, cols);
        apack
    }

    pub(super) fn pack_a_ref<T>(
        mr: usize,
        apack: &mut [T],
        a: MatRef<T>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) where
        T: Zero + Copy,
    {
        let mc = rows.len();
        let kc = cols.len();
        assert_eq!(apack.len(), mc.checked_mul(kc).unwrap());

        assert!(mr <= mc);
        assert!(mr > 0);
        assert_eq!(mc % mr, 0);

        assert!(cols.end <= a.ncols());
        assert!(rows.start < a.nrows());
        assert!(a.row_stride() > 0);

        let number_of_blocks = mc / mr;
        let mut it = apack.iter_mut();

        let rows_offset = rows.start;

        for nblock in 0..number_of_blocks {
            let block_rows = rows_offset + mr * nblock..rows_offset + mr * (nblock + 1);
            for col in cols.clone() {
                for row in block_rows.clone() {
                    let dst = it.next().unwrap();
                    *dst = a.get_or_zero(row, col);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{reference::*, *};
    use crate::std_prelude::Vec;

    fn apack<T: Copy + Zero>(
        mr: usize,
        a: MatRef<T>,
        rows: Range<usize>,
        cols: Range<usize>,
    ) -> Vec<T> {
        let mut apack = vec![T::zero(); rows.len() * cols.len()];
        pack_a(mr, &mut apack, a, rows, cols);
        apack
    }

    #[test]
    fn test_pack_a_1() {
        #[rustfmt::skip]
        let a = [
            0,   1,  2,  3,
            4,   5,  6,  7,
            8,   9, 10, 11,
            12, 13, 14, 15
        ];
        let a = MatRef::row_major(4, 4, &a);

        let mr = 2;
        let rows = 0..4;
        let cols = 0..3;
        #[rustfmt::skip]
        assert_eq!(apack_ref(mr, a, rows.clone(), cols.clone()), [
            0, 4, 1, 5, 2, 6,
            8, 12, 9, 13, 10, 14,
        ]);
        assert_eq!(
            apack_ref(mr, a, rows.clone(), cols.clone()),
            apack(mr, a, rows.clone(), cols.clone(),)
        );

        let mr = 2;
        let rows = 1..5;
        let cols = 0..3;
        #[rustfmt::skip]
        assert_eq!(apack_ref(mr, a, rows.clone(), cols.clone()), [
            4, 8, 5, 9, 6, 10,
            12, 0, 13, 0, 14, 0,
        ]);
        assert_eq!(
            apack_ref(mr, a, rows.clone(), cols.clone()),
            apack(mr, a, rows.clone(), cols.clone(),)
        );

        let mr = 2;
        let rows = 0..4;
        let cols = 1..3;
        #[rustfmt::skip]
        assert_eq!(apack_ref(mr, a, rows.clone(), cols.clone()), [
            1, 5, 2, 6,
            9, 13, 10, 14,
        ]);
        assert_eq!(
            apack_ref(mr, a, rows.clone(), cols.clone()),
            apack(mr, a, rows.clone(), cols.clone(),)
        );
    }
}

#[cfg(test)]
mod proptests {
    use super::{reference::*, *};
    use crate::utils::arb_matrix;
    use proptest::{prelude::*, proptest};

    proptest! {
        #[test]
        fn proptest_pack_a(
            a in arb_matrix::<i8>(1..40, 1..40),
            mr in (1..41usize),
        ) {
            let a_ref = a.to_ref();

            const TAKE: usize = 50;
            let arb_rows = (0..a.nrows())
                .prop_flat_map(|start| (start..start + TAKE).prop_map(move |end| start..end))
                .prop_filter("rows", |rows| mr <= rows.len() && rows.len() % mr == 0);

            let arb_cols = (0..=a.ncols())
                .prop_flat_map(|start| (start..=a.ncols()).prop_map(move |end| start..end));

            proptest!(|(rows in arb_rows, cols in arb_cols)| {
                let mut apack = vec![-1; rows.len() * cols.len()];
                pack_a(mr, &mut apack, a_ref, rows.clone(), cols.clone());
                let expect = apack_ref(mr, a_ref, rows, cols);
                prop_assert_eq!(apack, expect);
            });
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(3)] // 1 + max(kc, number_of_valid_blocks)
    fn check_pack_a() -> Option<()> {
        const KC_LIMIT: usize = 2;
        const NUMBER_OF_VALID_BLOCKS_LIMIT: usize = 2;

        const PACK_LEN_LIMIT: usize = 11;
        const VALUES_LEN_LIMIT: usize = 13;
        const MC_LIMIT: usize = 13;

        let values = kani::vec::any_vec::<i8, VALUES_LEN_LIMIT>();
        let a = {
            let nrows = kani::any();
            let ncols = kani::any();
            let row_stride = kani::any_where(|&row_stride| row_stride > 0);
            let col_stride = kani::any();
            MatRef::from_parts(nrows, ncols, &values, row_stride, col_stride)?
        };

        let mr: usize = kani::any_where(|&mr| mr > 0);

        let rows: Range<usize> = kani::any()..kani::any();
        let cols: Range<usize> = kani::any()..kani::any();

        let kc = cols.len();
        let mc = rows.len();
        kani::assume(kc <= KC_LIMIT);
        kani::assume(mc <= MC_LIMIT);
        kani::assume(mr <= mc && mc % mr == 0);

        kani::assume(cols.end <= a.ncols());
        kani::assume(rows.start < a.nrows());

        let rows_stop_at = a.nrows().min(rows.end);
        let number_of_valid_blocks = (rows_stop_at - rows.start) / mr;
        kani::assume(number_of_valid_blocks <= NUMBER_OF_VALID_BLOCKS_LIMIT);

        kani::assume(mc.checked_mul(kc)? <= PACK_LEN_LIMIT);
        let mut apack = vec![0; mc * kc];
        pack_a(mr, &mut apack, a, rows, cols);

        Some(())
    }
}
