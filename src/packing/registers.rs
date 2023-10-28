use core::ops::Range;

use num_traits::Zero;

use crate::{Layout, MatMut, MatRef};

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

pub(crate) fn registers_from_c<T>(
    registers: &mut [T],
    c: &MatRef<T>,
    c_rows: Range<usize>,
    c_cols: Range<usize>,
) -> Layout
where
    T: Copy + Zero,
{
    crate::packing::block::ColMajor(registers).init_from(c, c_rows, c_cols);
    Layout::ColMajor
}
