use crate::{Kernel, Layout, MatMut, MatRef};
use core::ops::Range;
use num_traits::{One, Zero};

#[derive(Debug, Clone)]
pub struct PackSizes {
    pub mc: usize,
    pub kc: usize,
    pub nc: usize,
}

impl PackSizes {
    pub const fn buf_len<T, K>(&self, _: &K) -> usize
    where
        T: One + Zero + Copy,
        K: Kernel<Elem = T>,
    {
        let mr = K::MR;
        let nr = K::NR;
        self.mc * self.kc + self.kc * self.nc + mr * nr
    }
    pub(crate) fn check<T, K>(&self, _: &K)
    where
        T: One + Zero + Copy,
        K: Kernel<Elem = T>,
    {
        let mr = K::MR;
        let nr = K::NR;
        assert!(mr <= self.mc);
        assert!(nr <= self.nc);
        assert_eq!(self.mc % mr, 0);
        assert_eq!(self.nc % nr, 0);
    }
    pub(crate) fn split_buf<'buf, T>(
        &self,
        buf: &'buf mut [T],
    ) -> (&'buf mut [T], &'buf mut [T], &'buf mut [T]) {
        let (apack, tail) = buf.split_at_mut(self.mc * self.kc);
        let (bpack, dst_buf) = tail.split_at_mut(self.kc * self.nc);
        (apack, bpack, dst_buf)
    }
}

impl AsRef<PackSizes> for PackSizes {
    fn as_ref(&self) -> &PackSizes {
        self
    }
}

pub(crate) fn pack_a<T, K>(
    pack_sizes: &PackSizes,
    apack: &mut [T],
    a: &MatRef<T>,
    a_rows: Range<usize>,
    a_cols: Range<usize>,
) where
    T: One + Zero + Copy,
    K: Kernel<Elem = T>,
{
    let mc = pack_sizes.mc;
    let kc = pack_sizes.kc;
    let mr = K::MR;
    debug_assert_eq!(a_rows.len(), mc);
    debug_assert_eq!(a_cols.len(), kc);
    debug_assert_eq!(apack.len(), mc * kc);
    debug_assert_eq!(mc % mr, 0);

    let start = a_rows.start;
    for i in 0..mc / mr {
        let buf = &mut apack[mr * kc * i..mr * kc * (i + 1)];
        let rows = start + mr * i..start + mr * (i + 1);
        col_major_block(buf, a, rows, a_cols.clone());
    }
}

pub(crate) fn pack_b<T, K>(
    pack_sizes: &PackSizes,
    bpack: &mut [T],
    b: &MatRef<T>,
    b_rows: Range<usize>,
    b_cols: Range<usize>,
) where
    T: One + Zero + Copy,
    K: Kernel<Elem = T>,
{
    let kc = pack_sizes.kc;
    let nc = pack_sizes.nc;
    let nr = K::NR;
    debug_assert_eq!(b_rows.len(), kc);
    debug_assert_eq!(b_cols.len(), nc);
    debug_assert_eq!(bpack.len(), kc * nc);
    debug_assert_eq!(nc % nr, 0);

    let start = b_cols.start;
    for i in 0..nc / nr {
        let buf = &mut bpack[kc * nr * i..kc * nr * (i + 1)];
        let cols = start + nr * i..start + nr * (i + 1);
        row_major_block(buf, b, b_rows.clone(), cols);
    }
}

pub(crate) fn row_major_block<'block, T>(
    block_buf: &'block mut [T],
    from: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> MatMut<'block, T>
where
    T: Copy + Zero,
{
    let nrows = rows.len();
    let ncols = cols.len();
    assert_eq!(block_buf.len(), nrows * ncols);
    let mut block = MatMut::new(nrows, ncols, block_buf, Layout::RowMajor);

    for (i, row) in rows.enumerate() {
        for (j, col) in cols.clone().enumerate() {
            *block.get_mut(i, j) = from.get_or_zero(row, col);
        }
    }
    block
}

pub(crate) fn col_major_block<'block, T>(
    block_buf: &'block mut [T],
    from: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> MatMut<'block, T>
where
    T: Copy + Zero,
{
    let nrows = rows.len();
    let ncols = cols.len();
    assert_eq!(block_buf.len(), nrows * ncols);
    let mut block = MatMut::new(nrows, ncols, block_buf, Layout::ColumnMajor);

    for (j, col) in cols.enumerate() {
        for (i, row) in rows.clone().enumerate() {
            *block.get_mut(i, j) = from.get_or_zero(row, col);
        }
    }
    block
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Layout;

    #[rustfmt::skip]
    #[test]
    fn row_major_block_2x2() {
        let v = [
            1, 2,
            3, 4,
        ];
        let a = MatRef::new(2, 2, v.as_ref(), Layout::RowMajor);

        let block = |rows: Range<usize>, cols: Range<usize>| {
            let mut buf = vec![-1; rows.len() * cols.len()];
            row_major_block(&mut buf, &a, rows, cols);
            buf
        };

        assert_eq!(block(0..2, 0..2), a.as_slice());

        assert_eq!(block(0..1, 0..1), [1]);
        assert_eq!(block(0..1, 1..2), [2]);
        assert_eq!(block(1..2, 0..1), [3]);
        assert_eq!(block(1..2, 1..2), [4]);

        assert_eq!(block(0..1, 0..3), [1, 2, 0]);
        assert_eq!(block(0..1, 0..4), [1, 2, 0, 0]);
        assert_eq!(block(0..1, 0..5), [1, 2, 0, 0, 0]);

        assert_eq!(block(1..2, 0..3), [3, 4, 0]);
        assert_eq!(block(1..2, 0..4), [3, 4, 0, 0]);
        assert_eq!(block(1..2, 0..5), [3, 4, 0, 0, 0]);

        assert_eq!(block(0..2, 0..1), [1, 3]);
        assert_eq!(block(0..3, 0..1), [1, 3, 0]);
        assert_eq!(block(0..4, 0..1), [1, 3, 0, 0]);
        assert_eq!(block(0..5, 0..1), [1, 3, 0, 0, 0]);

        assert_eq!(block(0..2, 1..2), [2, 4]);
        assert_eq!(block(0..3, 1..2), [2, 4, 0]);
        assert_eq!(block(0..4, 1..2), [2, 4, 0, 0]);
        assert_eq!(block(0..5, 1..2), [2, 4, 0, 0, 0]);

        assert_eq!(block(0..3, 0..3), [
            1, 2, 0,
            3, 4, 0,
            0, 0, 0,
        ]);

        assert_eq!(block(0..3, 0..4), [
            1, 2, 0, 0,
            3, 4, 0, 0,
            0, 0, 0, 0,
        ]);

        assert_eq!(block(0..4, 0..4), [
            1, 2, 0, 0,
            3, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);

        assert_eq!(block(1..3, 0..3), [
            3, 4, 0,
            0, 0, 0,
        ]);

        assert_eq!(block(1..3, 1..3), [
            4, 0,
            0, 0,
        ]);

        assert_eq!(block(0..3, 1..3), [
            2, 0,
            4, 0,
            0, 0,
        ]);
    }

    #[rustfmt::skip]
    #[test]
    fn col_major_block_2x2() {
        let v = [
            1, 2,
            3, 4,
        ];
        let b = MatRef::new(2, 2, v.as_ref(), Layout::RowMajor);

        let block = |rows: Range<usize>, cols: Range<usize>| {
            let mut buf = vec![-1; rows.len() * cols.len()];
            col_major_block(&mut buf, &b, rows, cols);
            buf
        };

        assert_eq!(block(0..2, 0..2), [
            1, 3,
            2, 4,
        ]);

        assert_eq!(block(0..1, 0..1), [1]);
        assert_eq!(block(0..1, 1..2), [2]);
        assert_eq!(block(1..2, 0..1), [3]);
        assert_eq!(block(1..2, 1..2), [4]);

        assert_eq!(block(0..1, 0..3), [1, 2, 0]);
        assert_eq!(block(0..1, 0..4), [1, 2, 0, 0]);
        assert_eq!(block(0..1, 0..5), [1, 2, 0, 0, 0]);

        assert_eq!(block(1..2, 0..3), [3, 4, 0]);
        assert_eq!(block(1..2, 0..4), [3, 4, 0, 0]);
        assert_eq!(block(1..2, 0..5), [3, 4, 0, 0, 0]);

        assert_eq!(block(0..2, 0..1), [1, 3]);
        assert_eq!(block(0..3, 0..1), [1, 3, 0]);
        assert_eq!(block(0..4, 0..1), [1, 3, 0, 0]);
        assert_eq!(block(0..5, 0..1), [1, 3, 0, 0, 0]);

        assert_eq!(block(0..2, 1..2), [2, 4]);
        assert_eq!(block(0..3, 1..2), [2, 4, 0]);
        assert_eq!(block(0..4, 1..2), [2, 4, 0, 0]);
        assert_eq!(block(0..5, 1..2), [2, 4, 0, 0, 0]);

        assert_eq!(block(0..3, 0..3), [
            1, 3, 0,
            2, 4, 0,
            0, 0, 0,
        ]);

        assert_eq!(block(0..3, 0..4), [
            1, 3, 0,
            2, 4, 0,
            0, 0, 0,
            0, 0, 0,
        ]);

        assert_eq!(block(0..4, 0..4), [
            1, 3, 0, 0,
            2, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);

        assert_eq!(block(1..3, 0..3), [
            3, 0,
            4, 0,
            0, 0,
        ]);

        assert_eq!(block(1..3, 1..3), [
            4, 0,
            0, 0,
        ]);

        assert_eq!(block(0..3, 1..3), [
            2, 4, 0,
            0, 0, 0,
        ]);
    }
}
