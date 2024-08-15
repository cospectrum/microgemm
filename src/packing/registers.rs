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
    crate::packing::block::ColMajor(registers).copy_to(c, c_rows, c_cols);
}

// read submatrix from c to colmajor "registers"
pub(crate) fn registers_from_c<T>(
    registers: &mut [T],
    c: MatRef<T>,
    c_rows: Range<usize>,
    c_cols: Range<usize>,
) where
    T: Copy + Zero,
{
    crate::packing::block::ColMajor(registers).init_from(c, c_rows, c_cols);
}
