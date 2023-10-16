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
    pub const fn buf_len<T, K>(&self) -> usize
    where
        T: One + Zero + Copy,
        K: Kernel<T>,
    {
        let mr = K::MR;
        let nr = K::NR;
        self.mc * self.kc + self.kc * self.nc + mr * nr
    }
    pub(crate) fn check<T, K>(&self, _: &K)
    where
        T: One + Zero + Copy,
        K: Kernel<T>,
    {
        let mr = K::MR;
        let nr = K::NR;
        assert!(mr <= self.mc);
        assert!(nr <= self.nc);
        assert_eq!(self.mc % mr, 0);
        assert_eq!(self.nc % nr, 0);
    }
    pub(crate) fn split_buf<'a, T>(
        &self,
        buf: &'a mut [T],
    ) -> (&'a mut [T], &'a mut [T], &'a mut [T]) {
        let (a_buf, tail) = buf.split_at_mut(self.mc * self.kc);
        let (b_buf, c_buf) = tail.split_at_mut(self.kc * self.nc);
        (a_buf, b_buf, c_buf)
    }
}

impl AsRef<PackSizes> for PackSizes {
    fn as_ref(&self) -> &PackSizes {
        self
    }
}

pub fn row_major_block<'block, T>(
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

pub fn col_major_block<'block, T>(
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
