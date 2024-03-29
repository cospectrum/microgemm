use crate::{MatMut, MatRef, Zero};
use core::ops::Range;

pub(crate) struct ColMajor<V>(pub(crate) V);

impl<T> ColMajor<&[T]>
where
    T: Copy,
{
    pub fn copy_to(&self, mat: &mut MatMut<T>, rows: Range<usize>, cols: Range<usize>) {
        let buf = self.0;
        assert_eq!(buf.len(), rows.len() * cols.len());
        let mut it = buf.iter();

        for col in cols {
            for row in rows.start..rows.end {
                let src = it.next().unwrap();
                if row < mat.nrows() && col < mat.ncols() {
                    let dst = mat.get_mut(row, col);
                    *dst = *src;
                }
            }
        }
    }
}

impl<T> ColMajor<&mut [T]>
where
    T: Copy + Zero,
{
    pub fn init_from(&mut self, mat: &MatRef<T>, rows: Range<usize>, cols: Range<usize>) {
        let buf = self.as_mut();
        assert_eq!(buf.len(), rows.len() * cols.len());
        let mut it = buf.iter_mut();

        for col in cols {
            for row in rows.start..rows.end {
                let dst = it.next().unwrap();
                *dst = mat.get_or_zero(row, col);
            }
        }
    }
}

impl<T> AsMut<[T]> for ColMajor<&mut [T]> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    #[test]
    fn col_major_block_2x2() {
        let v = [
            1, 2,
            3, 4,
        ];
        let b = MatRef::row_major(2, 2, v.as_ref());

        let block = |rows: Range<usize>, cols: Range<usize>| {
            let mut buf = vec![-1; rows.len() * cols.len()];
            ColMajor(buf.as_mut_slice()).init_from(&b, rows, cols);
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
