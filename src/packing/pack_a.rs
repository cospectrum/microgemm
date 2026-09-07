use crate::MatRef;
use core::ops::Range;
use num_traits::Zero;

// Pack the submatrix a[rows, cols] into (mc/mr) col-major blocks of size mr x kc.
// Values outsize of `a` will be zeroed.
// Specialize lane copies for the calling kernel's fixed tile width.
#[inline(always)]
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
    let stride = a.row_stride();
    assert!(stride > 0);

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

        {
            let lane_stride = a.col_stride();
            // Fast path: with `stride == 1` the block is one bounds-checked subslice of
            // `mr`-wide lanes `lane_stride` apart - no per-lane index check or re-borrow.
            let block_src = if stride == 1 && kc > 0 && lane_stride >= mr {
                let end = a.checked_idx(block_rows.start, cols.end - 1);
                a.checked_idx(block_rows.start, cols.start)
                    .zip(end.and_then(|i| i.checked_add(mr)))
                    .and_then(|(first, last)| a.as_slice().get(first..last))
            } else {
                None
            };
            if let Some(src) = block_src {
                let (block, rest) = core::mem::take(&mut it).split_at_mut(mr * kc);
                let (dst, tail_dst) = block.split_at_mut(mr * (kc - 1));
                let (lane_src, tail_src) = src.split_at(lane_stride * (kc - 1));
                let lanes = dst
                    .chunks_exact_mut(mr)
                    .zip(lane_src.chunks_exact(lane_stride));
                for (dst, lane) in lanes {
                    dst.copy_from_slice(&lane[..mr]);
                }
                tail_dst.copy_from_slice(tail_src);
                it = rest;
                continue;
            }
        }

        for col in cols.clone() {
            debug_assert!(col < a.ncols());
            let idx = a.idx(block_rows.start, col);

            if stride == 1 {
                let lane = &a.as_slice()[idx..idx + mr];
                it[..mr].copy_from_slice(lane);
            } else {
                let lane = a.as_slice()[idx..].iter().step_by(stride).take(mr);
                debug_assert_eq!(lane.len(), mr);
                let zip = lane.zip(&mut it[..mr]);
                for (&src, dst) in zip {
                    *dst = src;
                }
            }
            it = &mut it[mr..];
        }
    }

    let remains = (rows_stop_at - rows.start) % mr;
    debug_assert!(remains < mr);
    if remains > 0 {
        let nblock = number_of_valid_blocks;
        let block_rows = rows_offset + mr * nblock..rows_stop_at;
        debug_assert!(block_rows.start < block_rows.end);
        debug_assert_eq!(block_rows.len(), remains);

        for col in cols.clone() {
            debug_assert!(col < a.ncols());
            let idx = a.idx(block_rows.start, col);

            if stride == 1 {
                let lane = &a.as_slice()[idx..idx + remains];
                it[..remains].copy_from_slice(lane);
            } else {
                let lane = a.as_slice()[idx..].iter().step_by(stride).take(remains);
                debug_assert_eq!(lane.len(), remains);
                let zip = lane.zip(&mut it[..remains]);
                for (&src, dst) in zip {
                    *dst = src;
                }
            }
            it[remains..mr].fill(T::zero());
            it = &mut it[mr..];
        }
    }

    it.fill(T::zero());
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

#[cfg(not(miri))]
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

    // One arbitrary output index quantifies over the entire buffer, including
    // guards. The oracle uses layout arithmetic, independently of the packer.
    fn check<const MR: usize>(
        nrows: usize,
        stride: usize,
        lane_stride: usize,
        start: usize,
        count: usize,
        col: usize,
        kc: usize,
    ) {
        let values: [i8; 64] = kani::any();
        let mut actual: [i8; 34] = kani::any();
        let index: usize = kani::any_where(|&i| i < 34);
        let before = actual[index];
        let len = count * kc;
        let mat = MatRef::from_parts(nrows, 3, &values[..], stride, lane_stride).unwrap();
        pack_a(
            MR,
            &mut actual[1..1 + len],
            mat,
            start..start + count,
            col..col + kc,
        );
        if index == 0 || index > len {
            assert_eq!(actual[index], before);
        } else {
            let i = index - 1;
            let row = start + (i / (MR * kc)) * MR + i % MR;
            let column = col + (i / MR) % kc;
            let expected = if row < nrows {
                values[row * stride + column * lane_stride]
            } else {
                0
            };
            assert_eq!(actual[index], expected);
        }
    }

    // Constant layout cases avoid symbolic 64-bit division in iterator lengths.
    // Values and the output index remain arbitrary in every case.
    macro_rules! packing_proof {
        ($name:ident, $mr:literal, $rows:literal, $stride:literal, $ld:literal,
         $start:literal, $count:literal, $col:literal, $kc:literal) => {
            #[kani::proof]
            #[kani::unwind(33)]
            fn $name() {
                check::<$mr>($rows, $stride, $ld, $start, $count, $col, $kc);
            }
        };
    }

    packing_proof!(contiguous_leading_5, 2, 5, 1, 5, 1, 4, 1, 2);
    packing_proof!(contiguous_leading_6, 2, 5, 1, 6, 1, 4, 1, 2);
    packing_proof!(contiguous_leading_7, 2, 5, 1, 7, 1, 4, 1, 2);
    packing_proof!(overlapping_lanes_fallback, 2, 5, 1, 1, 1, 4, 1, 2);
    packing_proof!(strided_two_panels_stride_2, 2, 5, 2, 1, 1, 4, 1, 2);
    packing_proof!(strided_two_panels_stride_3, 2, 5, 3, 1, 1, 4, 1, 2);
    packing_proof!(partial_and_whole_padding_stride_1, 2, 4, 1, 12, 1, 6, 1, 2);
    packing_proof!(partial_and_whole_padding_stride_2, 2, 4, 2, 12, 1, 6, 1, 2);
    packing_proof!(partial_and_whole_padding_stride_3, 2, 4, 3, 12, 1, 6, 1, 2);
    packing_proof!(zero_depth_preserves_buffer, 2, 5, 1, 5, 1, 6, 3, 0);
    packing_proof!(neon_width_full_and_partial_panel, 8, 9, 1, 9, 0, 16, 1, 2);
}
