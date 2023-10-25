use crate::{MatMut, MatRef};
use core::ops::Range;

pub(crate) fn copy_row_major_friendly<T>(
    from: &MatRef<T>,
    to: &mut MatMut<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: Copy,
{
    for (i, row) in rows.enumerate() {
        for (j, col) in cols.clone().enumerate() {
            if row < to.nrows() && col < to.ncols() {
                *to.get_mut(row, col) = from.get(i, j)
            }
        }
    }
}

pub(crate) fn copy_col_major_friendly<T>(
    from: &MatRef<T>,
    to: &mut MatMut<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: Copy,
{
    for (j, col) in cols.enumerate() {
        for (i, row) in rows.clone().enumerate() {
            if row < to.nrows() && col < to.ncols() {
                *to.get_mut(row, col) = from.get(i, j)
            }
        }
    }
}
