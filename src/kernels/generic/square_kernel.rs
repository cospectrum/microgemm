use crate::{Kernel, One, Zero};
use core::marker::PhantomData;
use core::ops::{Add, Mul};

#[derive(Debug, Clone, Copy, Default)]
pub struct GenericSquareKernel<T, const DIM: usize> {
    marker: PhantomData<T>,
}

impl<T, const DIM: usize> GenericSquareKernel<T, DIM> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

fn loop_micropanels<T, const DIM: usize>(lhs: &[T], rhs: &[T], cols: &mut [T])
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(cols.len(), DIM * DIM);
    assert_eq!(lhs.len() % DIM, 0);
    assert_eq!(lhs.len(), rhs.len());

    let left = lhs.chunks_exact(DIM);
    let right = rhs.chunks_exact(DIM);

    left.zip(right).for_each(|(a, b)| {
        let cols = cols.chunks_exact_mut(DIM);

        cols.zip(b).for_each(|(col, &scalar)| {
            col.iter_mut().zip(a).for_each(|(out, &x)| {
                *out = *out + x * scalar;
            });
        });
    });
}

fn write_cols_to_colmajor<T, const DIM: usize>(dst: &mut [T], cols: &[T], alpha: T, beta: T)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(dst.len(), DIM * DIM);
    assert_eq!(cols.len(), dst.len());
    dst.iter_mut().zip(cols).for_each(|(to, &from)| {
        *to = alpha * from + beta * *to;
    });
}

macro_rules! impl_generic_square_kernel {
    ($dim:literal) => {
        impl<T> Kernel for GenericSquareKernel<T, $dim>
        where
            T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
        {
            type Scalar = T;

            const MR: usize = $dim;
            const NR: usize = $dim;

            fn microkernel(
                &self,
                alpha: Self::Scalar,
                lhs: &crate::MatRef<Self::Scalar>,
                rhs: &crate::MatRef<Self::Scalar>,
                beta: Self::Scalar,
                dst: &mut crate::MatMut<Self::Scalar>,
            ) {
                assert_eq!(dst.nrows(), Self::MR);
                assert_eq!(dst.ncols(), Self::NR);
                assert_eq!(dst.row_stride(), 1);

                const DIM: usize = $dim;
                let mut cols = [T::zero(); DIM * DIM];
                loop_micropanels::<_, DIM>(lhs.as_slice(), rhs.as_slice(), &mut cols);
                write_cols_to_colmajor::<_, DIM>(dst.as_mut_slice(), &cols, alpha, beta);
            }
        }
    };
}

impl_generic_square_kernel!(2);
impl_generic_square_kernel!(4);
impl_generic_square_kernel!(8);
impl_generic_square_kernel!(16);
impl_generic_square_kernel!(32);
