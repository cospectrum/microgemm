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

        {
            let lane_stride = b.row_stride();
            // Fast path: with `stride == 1` the block is one bounds-checked subslice of
            // `nr`-wide lanes `lane_stride` apart - no per-lane index check or re-borrow.
            let block_src = if stride == 1 && kc > 0 && lane_stride >= nr {
                let end = b.checked_idx(rows.end - 1, block_cols.start);
                b.checked_idx(rows.start, block_cols.start)
                    .zip(end.and_then(|i| i.checked_add(nr)))
                    .and_then(|(first, last)| b.as_slice().get(first..last))
            } else {
                None
            };
            if let Some(src) = block_src {
                let (block, rest) = core::mem::take(&mut it).split_at_mut(nr * kc);
                let (dst, tail_dst) = block.split_at_mut(nr * (kc - 1));
                let (lane_src, tail_src) = src.split_at(lane_stride * (kc - 1));
                let lanes = dst
                    .chunks_exact_mut(nr)
                    .zip(lane_src.chunks_exact(lane_stride));
                for (dst, lane) in lanes {
                    dst.copy_from_slice(&lane[..nr]);
                }
                tail_dst.copy_from_slice(tail_src);
                it = rest;
                continue;
            }
        }

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
                for (&src, dst) in zip {
                    *dst = src;
                }
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
                for (&src, dst) in zip {
                    *dst = src;
                }
            }
            it[remains..nr].fill(T::zero());
            it = &mut it[nr..];
        }
    }

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

#[cfg(not(miri))]
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
                .prop_flat_map(|start| (start..start + TAKE).prop_map(move |end| start..end))
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

    // One arbitrary output index quantifies over the entire buffer, including
    // guards. The oracle uses layout arithmetic, independently of the packer.
    // Coordinates below describe transposed B: nrows/row refer to its packed axis.
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
        let mat = MatRef::from_parts(3, nrows, &values[..], lane_stride, stride).unwrap();
        pack_b(
            MR,
            &mut actual[1..1 + len],
            mat,
            col..col + kc,
            start..start + count,
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
