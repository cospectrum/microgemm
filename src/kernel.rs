use crate::{gemm_with_kernel, Layout, MatMut, MatRef, PackSizes};
use core::ops::{Mul, Range};
use generic_array::{
    typenum::{Prod, Unsigned},
    ArrayLength,
};
use num_traits::{One, Zero};

pub trait Kernel
where
    Self::Scalar: Copy + Zero + One,
{
    type Scalar;
    type Mr: ArrayLength + Multiply<Self::Nr>;
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

    fn registers_from_c(
        &self,
        c: &MatRef<Self::Scalar>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        registers: &mut [Self::Scalar],
    ) -> Layout {
        crate::packing::block::ColMajor(registers).init_from(c, c_rows, c_cols);
        Layout::ColMajor
    }
    fn registers_to_c(
        &self,
        c: &mut MatMut<Self::Scalar>,
        c_rows: Range<usize>,
        c_cols: Range<usize>,
        registers: &[Self::Scalar],
    ) {
        crate::packing::block::ColMajor(registers).copy_to(c, c_rows, c_cols);
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
        packing_buf: &mut [Self::Scalar],
    ) {
        gemm_with_kernel(self, alpha, a, b, beta, c, pack_sizes, packing_buf);
    }
    fn mr(&self) -> usize {
        Self::MR
    }
    fn nr(&self) -> usize {
        Self::NR
    }
}

pub trait Multiply<Rhs> {
    type Output: ArrayLength;
}

impl<Lhs, Rhs> Multiply<Rhs> for Lhs
where
    Lhs: Mul<Rhs>,
    Prod<Lhs, Rhs>: ArrayLength,
{
    type Output = Prod<Lhs, Rhs>;
}
