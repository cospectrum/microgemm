use crate::{Layout, MatMut, MatRef};
use core::ops::Range;

pub(crate) fn pack_a<'a, T>(
    a: &MatRef<T>,
    buf: &'a mut [T],
    rows: Range<usize>,
    cols: Range<usize>,
) -> MatMut<'a, T>
where
    T: Copy,
{
    let nrows = rows.len();
    let ncols = cols.len();
    assert_eq!(buf.len(), nrows * ncols);

    let mut pack = MatMut::new(nrows, ncols, buf, Layout::RowMajor);

    for (i, row) in rows.enumerate() {
        for (j, col) in cols.clone().enumerate() {
            *pack.get_mut(i, j) = a.get(row, col);
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
    T: Copy,
{
    let nrows = rows.len();
    let ncols = cols.len();
    assert_eq!(buf.len(), nrows * ncols);

    let mut pack = MatMut::new(nrows, ncols, buf, Layout::ColumnMajor);

    for (j, col) in cols.enumerate() {
        for (i, row) in rows.clone().enumerate() {
            *pack.get_mut(i, j) = b.get(row, col);
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
    T: Copy,
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
            *to.get_mut(row, col) = from.get(i, j);
        }
    }
}
