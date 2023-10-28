use crate::{gemm_with_kernel, MatMut, MatRef, PackSizes};
use core::ops::Mul;
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

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn gemm(
        &self,
        alpha: Self::Scalar,
        a: &MatRef<Self::Scalar>,
        b: &MatRef<Self::Scalar>,
        beta: Self::Scalar,
        c: &mut MatMut<Self::Scalar>,
        pack_sizes: impl AsRef<PackSizes>,
        mut packing_buf: impl AsMut<[Self::Scalar]>,
    ) {
        gemm_with_kernel(
            self,
            alpha,
            a,
            b,
            beta,
            c,
            pack_sizes.as_ref(),
            packing_buf.as_mut(),
        );
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
