use crate::{Layout, MatMut, MatRef};
use core::ops::Range;
use num_traits::Zero;

pub(crate) fn pack_a<'a, T>(
    a: &MatRef<T>,
    buf: &'a mut [T],
    rows: Range<usize>,
    cols: Range<usize>,
) -> MatMut<'a, T>
where
    T: Copy + Zero,
{
    let nrows = rows.len();
    let ncols = cols.len();
    assert_eq!(buf.len(), nrows * ncols);

    let mut pack = MatMut::new(nrows, ncols, buf, Layout::RowMajor);
    let zero = T::zero();

    for (i, row) in rows.enumerate() {
        for (j, col) in cols.clone().enumerate() {
            *pack.get_mut(i, j) = a.get_or(row, col, zero);
        }
    }
    pack
}

pub(crate) fn pack_b<'b, T>(
    b: &MatRef<T>,
    buf: &'b mut [T],
    rows: Range<usize>,
    cols: Range<usize>,
) -> MatMut<'b, T>
where
    T: Copy + Zero,
{
    let nrows = rows.len();
    let ncols = cols.len();
    assert_eq!(buf.len(), nrows * ncols);

    let mut pack = MatMut::new(nrows, ncols, buf, Layout::ColumnMajor);
    let zero = T::zero();

    for (j, col) in cols.enumerate() {
        for (i, row) in rows.clone().enumerate() {
            *pack.get_mut(i, j) = b.get_or(row, col, zero);
        }
    }
    pack
}

pub(crate) fn pack_c<'a, T>(
    c: &MatRef<T>,
    buf: &'a mut [T],
    rows: Range<usize>,
    cols: Range<usize>,
) -> MatMut<'a, T>
where
    T: Copy + Zero,
{
    pack_a(c, buf, rows, cols)
}

pub(crate) fn copy_to<T>(
    to: &mut MatMut<T>,
    from: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: Copy,
{
    for (i, row) in rows.enumerate() {
        for (j, col) in cols.clone().enumerate() {
            if row < to.nrows() && col < to.ncols() {
                *to.get_mut(row, col) = from.get(i, j);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Layout;

    #[rustfmt::skip]
    #[test]
    fn apack_2x2() {
        let v = [
            1, 2,
            3, 4,
        ];
        let a = MatRef::new(2, 2, v.as_ref(), Layout::RowMajor);

        let pack = |rows: Range<usize>, cols: Range<usize>| {
            let mut buf = vec![-1; rows.len() * cols.len()];
            pack_a(&a, &mut buf, rows, cols);
            buf
        };

        assert_eq!(pack(0..2, 0..2), a.as_slice());

        assert_eq!(pack(0..1, 0..1), [1]);
        assert_eq!(pack(0..1, 1..2), [2]);
        assert_eq!(pack(1..2, 0..1), [3]);
        assert_eq!(pack(1..2, 1..2), [4]);

        assert_eq!(pack(0..1, 0..3), [1, 2, 0]);
        assert_eq!(pack(0..1, 0..4), [1, 2, 0, 0]);
        assert_eq!(pack(0..1, 0..5), [1, 2, 0, 0, 0]);

        assert_eq!(pack(1..2, 0..3), [3, 4, 0]);
        assert_eq!(pack(1..2, 0..4), [3, 4, 0, 0]);
        assert_eq!(pack(1..2, 0..5), [3, 4, 0, 0, 0]);

        assert_eq!(pack(0..2, 0..1), [1, 3]);
        assert_eq!(pack(0..3, 0..1), [1, 3, 0]);
        assert_eq!(pack(0..4, 0..1), [1, 3, 0, 0]);
        assert_eq!(pack(0..5, 0..1), [1, 3, 0, 0, 0]);

        assert_eq!(pack(0..2, 1..2), [2, 4]);
        assert_eq!(pack(0..3, 1..2), [2, 4, 0]);
        assert_eq!(pack(0..4, 1..2), [2, 4, 0, 0]);
        assert_eq!(pack(0..5, 1..2), [2, 4, 0, 0, 0]);

        assert_eq!(pack(0..3, 0..3), [
            1, 2, 0,
            3, 4, 0,
            0, 0, 0,
        ]);

        assert_eq!(pack(0..3, 0..4), [
            1, 2, 0, 0,
            3, 4, 0, 0,
            0, 0, 0, 0,
        ]);

        assert_eq!(pack(0..4, 0..4), [
            1, 2, 0, 0,
            3, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);

        assert_eq!(pack(1..3, 0..3), [
            3, 4, 0,
            0, 0, 0,
        ]);

        assert_eq!(pack(1..3, 1..3), [
            4, 0,
            0, 0,
        ]);

        assert_eq!(pack(0..3, 1..3), [
            2, 0,
            4, 0,
            0, 0,
        ]);
    }

    #[rustfmt::skip]
    #[test]
    fn bpack_2x2() {
        let v = [
            1, 2,
            3, 4,
        ];
        let b = MatRef::new(2, 2, v.as_ref(), Layout::RowMajor);

        let pack = |rows: Range<usize>, cols: Range<usize>| {
            let mut buf = vec![-1; rows.len() * cols.len()];
            pack_b(&b, &mut buf, rows, cols);
            buf
        };

        assert_eq!(pack(0..2, 0..2), [
            1, 3,
            2, 4,
        ]);

        assert_eq!(pack(0..1, 0..1), [1]);
        assert_eq!(pack(0..1, 1..2), [2]);
        assert_eq!(pack(1..2, 0..1), [3]);
        assert_eq!(pack(1..2, 1..2), [4]);

        assert_eq!(pack(0..1, 0..3), [1, 2, 0]);
        assert_eq!(pack(0..1, 0..4), [1, 2, 0, 0]);
        assert_eq!(pack(0..1, 0..5), [1, 2, 0, 0, 0]);

        assert_eq!(pack(1..2, 0..3), [3, 4, 0]);
        assert_eq!(pack(1..2, 0..4), [3, 4, 0, 0]);
        assert_eq!(pack(1..2, 0..5), [3, 4, 0, 0, 0]);

        assert_eq!(pack(0..2, 0..1), [1, 3]);
        assert_eq!(pack(0..3, 0..1), [1, 3, 0]);
        assert_eq!(pack(0..4, 0..1), [1, 3, 0, 0]);
        assert_eq!(pack(0..5, 0..1), [1, 3, 0, 0, 0]);

        assert_eq!(pack(0..2, 1..2), [2, 4]);
        assert_eq!(pack(0..3, 1..2), [2, 4, 0]);
        assert_eq!(pack(0..4, 1..2), [2, 4, 0, 0]);
        assert_eq!(pack(0..5, 1..2), [2, 4, 0, 0, 0]);

        assert_eq!(pack(0..3, 0..3), [
            1, 3, 0,
            2, 4, 0,
            0, 0, 0,
        ]);

        assert_eq!(pack(0..3, 0..4), [
            1, 3, 0,
            2, 4, 0,
            0, 0, 0,
            0, 0, 0,
        ]);

        assert_eq!(pack(0..4, 0..4), [
            1, 3, 0, 0,
            2, 4, 0, 0,
            0, 0, 0, 0,
            0, 0, 0, 0,
        ]);

        assert_eq!(pack(1..3, 0..3), [
            3, 0,
            4, 0,
            0, 0,
        ]);

        assert_eq!(pack(1..3, 1..3), [
            4, 0,
            0, 0,
        ]);

        assert_eq!(pack(0..3, 1..3), [
            2, 4, 0,
            0, 0, 0,
        ]);
    }
}
