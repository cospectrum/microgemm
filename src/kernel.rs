use crate::{
    col_major_block, gemm_with_kernel, row_major_block, Layout, MatMut, MatRef, PackSizes,
};
use core::ops::Range;
use num_traits::{One, Zero};

pub trait Kernel<T>
where
    T: Copy + Zero + One,
    Self: Sized,
{
    const MR: usize;
    const NR: usize;

    fn microkernel(&self, alpha: T, lhs: &MatRef<T>, rhs: &MatRef<T>, beta: T, dst: &mut MatMut<T>);

    fn pack_a(
        &self,
        _: &PackSizes,
        apack: &mut [T],
        a: &MatRef<T>,
        a_rows: Range<usize>,
        a_cols: Range<usize>,
    ) -> Layout {
        row_major_block(apack, a, a_rows, a_cols);
        Layout::RowMajor
    }
    fn pack_b(
        &self,
        _: &PackSizes,
        bpack: &mut [T],
        b: &MatRef<T>,
        b_rows: Range<usize>,
        b_cols: Range<usize>,
    ) -> Layout {
        col_major_block(bpack, b, b_rows, b_cols);
        Layout::ColumnMajor
    }
    fn copy_from_c<'dst>(
        &self,
        c: &MatRef<T>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        to: &'dst mut [T],
    ) -> MatMut<'dst, T> {
        let dst = col_major_block(to, c, c_rows, c_cols);
        dst
    }
    fn copy_to_c(
        &self,
        c: &mut MatMut<T>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        from: &MatRef<T>,
    ) {
        crate::copy(from, c, c_rows, c_cols);
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        alpha: T,
        a: &MatRef<T>,
        b: &MatRef<T>,
        beta: T,
        c: &mut MatMut<T>,
        pack_sizes: &PackSizes,
        buf: &mut [T],
    ) {
        gemm_with_kernel(self, alpha, a, b, beta, c, pack_sizes, buf);
    }
}
