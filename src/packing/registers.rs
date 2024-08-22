use core::ops::Range;

use num_traits::Zero;

use crate::{MatMut, MatRef};

// write colmajor "registers" back to c
pub(crate) fn registers_to_c<T>(
    registers: &[T],
    c: &mut MatMut<T>,
    c_rows: Range<usize>,
    c_cols: Range<usize>,
) where
    T: Copy,
{
    assert_eq!(registers.len(), c_rows.len() * c_cols.len());
    let mut it = registers.iter();

    for col in c_cols {
        for row in c_rows.clone() {
            let src = it.next().unwrap();
            if c.in_bounds(row, col) {
                let dst = c.get_mut(row, col);
                *dst = *src;
            }
        }
    }
}

// read submatrix c[c_rows, c_cols] to "registers" with colmajor layout
pub(crate) fn registers_from_c<T>(
    registers: &mut [T],
    c: MatRef<T>,
    c_rows: Range<usize>,
    c_cols: Range<usize>,
) where
    T: Copy + Zero,
{
    assert_eq!(registers.len(), c_rows.len() * c_cols.len());
    let mut it = registers.iter_mut();

    for col in c_cols {
        for row in c_rows.clone() {
            let dst = it.next().unwrap();
            *dst = c.get_or_zero(row, col);
        }
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
        let c = MatRef::row_major(2, 2, v.as_ref());
        let block = |rows: Range<usize>, cols: Range<usize>| {
            let mut buf = vec![-1; rows.len() * cols.len()];
            registers_from_c(buf.as_mut_slice(), c, rows, cols);
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
