use crate::{gemm_with_kernel, Layout, MatMut, MatRef, PackSizes};
use core::ops::Range;
use num_traits::{One, Zero};

pub trait Kernel
where
    Self: Sized,
    Self::Elem: Copy + Zero + One,
{
    type Elem;
    const MR: usize;
    const NR: usize;

    fn microkernel(
        &self,
        alpha: Self::Elem,
        lhs: &MatRef<Self::Elem>,
        rhs: &MatRef<Self::Elem>,
        beta: Self::Elem,
        dst: &mut MatMut<Self::Elem>,
    );

    fn pack_a(
        &self,
        pack_sizes: &PackSizes,
        apack: &mut [Self::Elem],
        a: &MatRef<Self::Elem>,
        a_rows: Range<usize>,
        a_cols: Range<usize>,
    ) -> Layout {
        crate::packing::pack_a::<Self::Elem, Self>(pack_sizes, apack, a, a_rows, a_cols);
        Layout::ColumnMajor
    }
    fn pack_b(
        &self,
        pack_sizes: &PackSizes,
        bpack: &mut [Self::Elem],
        b: &MatRef<Self::Elem>,
        b_rows: Range<usize>,
        b_cols: Range<usize>,
    ) -> Layout {
        crate::packing::pack_b::<Self::Elem, Self>(pack_sizes, bpack, b, b_rows, b_cols);
        Layout::RowMajor
    }
    fn copy_from_c<'dst>(
        &self,
        c: &MatRef<Self::Elem>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        to: &'dst mut [Self::Elem],
    ) -> MatMut<'dst, Self::Elem> {
        let dst = crate::packing::col_major_block(to, c, c_rows, c_cols);
        dst
    }
    fn copy_to_c(
        &self,
        c: &mut MatMut<Self::Elem>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        from: &MatRef<Self::Elem>,
    ) {
        crate::copying::copy(from, c, c_rows, c_cols);
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        alpha: Self::Elem,
        a: &MatRef<Self::Elem>,
        b: &MatRef<Self::Elem>,
        beta: Self::Elem,
        c: &mut MatMut<Self::Elem>,
        pack_sizes: &PackSizes,
        buf: &mut [Self::Elem],
    ) {
        gemm_with_kernel(self, alpha, a, b, beta, c, pack_sizes, buf);
    }
    fn mr(&self) -> usize {
        Self::MR
    }
    fn nr(&self) -> usize {
        Self::NR
    }
}
