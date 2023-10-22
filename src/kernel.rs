use crate::{gemm_with_kernel, Layout, MatMut, MatRef, PackSizes};
use core::ops::{Mul, Range};

use generic_array::{typenum::Unsigned, ArrayLength};
use num_traits::{One, Zero};

pub trait Kernel
where
    Self: Sized,
    Self::Scalar: Copy + Zero + One,
{
    type Scalar;
    type Mr: ArrayLength + Mul<Self::Nr>;
    type Nr: ArrayLength;

    const MR: usize = Self::Mr::USIZE;
    const NR: usize = Self::Nr::USIZE;

    fn microkernel(
        &self,
        alpha: Self::Scalar,
        lhs: &MatRef<Self::Scalar>,
        rhs: &MatRef<Self::Scalar>,
        beta: Self::Scalar,
        dst: &mut MatMut<Self::Scalar>,
    );

    fn pack_a(
        &self,
        pack_sizes: &PackSizes,
        apack: &mut [Self::Scalar],
        a: &MatRef<Self::Scalar>,
        a_rows: Range<usize>,
        a_cols: Range<usize>,
    ) -> Layout {
        crate::packing::pack_a::<Self::Scalar, Self>(pack_sizes, apack, a, a_rows, a_cols);
        Layout::ColumnMajor
    }
    fn pack_b(
        &self,
        pack_sizes: &PackSizes,
        bpack: &mut [Self::Scalar],
        b: &MatRef<Self::Scalar>,
        b_rows: Range<usize>,
        b_cols: Range<usize>,
    ) -> Layout {
        crate::packing::pack_b::<Self::Scalar, Self>(pack_sizes, bpack, b, b_rows, b_cols);
        Layout::RowMajor
    }
    fn copy_from_c<'dst>(
        &self,
        c: &MatRef<Self::Scalar>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        to: &'dst mut [Self::Scalar],
    ) -> MatMut<'dst, Self::Scalar> {
        let dst = crate::packing::col_major_block(to, c, c_rows, c_cols);
        dst
    }
    fn copy_to_c(
        &self,
        c: &mut MatMut<Self::Scalar>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        from: &MatRef<Self::Scalar>,
    ) {
        crate::copying::copy(from, c, c_rows, c_cols);
    }
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        alpha: Self::Scalar,
        a: &MatRef<Self::Scalar>,
        b: &MatRef<Self::Scalar>,
        beta: Self::Scalar,
        c: &mut MatMut<Self::Scalar>,
        pack_sizes: &PackSizes,
        buf: &mut [Self::Scalar],
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
