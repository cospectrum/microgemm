use crate::{
    typenum::{U16, U2, U32, U4, U8},
    Kernel, One, Zero,
};
use core::marker::PhantomData;
use core::ops::{Add, Mul};

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
    ($struct:ident, $constant:literal, $num:ty) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $struct<T>(PhantomData<T>);

        impl<T> $struct<T> {
            pub const fn new() -> Self {
                Self(PhantomData)
            }
        }
        impl<T> Kernel for $struct<T>
        where
            T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
        {
            type Scalar = T;
            type Mr = $num;
            type Nr = $num;

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

                const DIM: usize = $constant;
                let mut cols = [T::zero(); DIM * DIM];
                loop_micropanels::<_, DIM>(lhs.as_slice(), rhs.as_slice(), &mut cols);
                write_cols_to_colmajor::<_, DIM>(dst.as_mut_slice(), &cols, alpha, beta);
            }
        }
    };
}

impl_generic_square_kernel!(Generic2x2Kernel, 2, U2);
impl_generic_square_kernel!(Generic4x4Kernel, 4, U4);
impl_generic_square_kernel!(Generic8x8Kernel, 8, U8);
impl_generic_square_kernel!(Generic16x16Kernel, 16, U16);
impl_generic_square_kernel!(Generic32x32Kernel, 32, U32);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{utils::*, Kernel};
    use rand::{thread_rng, Rng};

    #[test]
    fn test_generic_kernels_i32() {
        test_kernel_i32(Generic2x2Kernel::new());
        test_kernel_i32(Generic4x4Kernel::new());
        test_kernel_i32(Generic8x8Kernel::new());
        test_kernel_i32(Generic16x16Kernel::new());
        test_kernel_i32(Generic32x32Kernel::new());
    }

    #[test]
    fn test_generic_kernels_f32() {
        test_kernel_f32(Generic2x2Kernel::new());
        test_kernel_f32(Generic4x4Kernel::new());
        test_kernel_f32(Generic8x8Kernel::new());
        test_kernel_f32(Generic16x16Kernel::new());
        test_kernel_i32(Generic32x32Kernel::new());
    }

    fn test_kernel_f32(kernel: impl Kernel<Scalar = f32>) {
        let cmp = |expect: &[f32], got: &[f32]| {
            let eps = 75.0 * f32::EPSILON;
            assert_approx_eq(expect, got, eps);
        };
        let mut rng = thread_rng();

        for _ in 0..20 {
            let scalar = || rng.gen_range(-1.0..1.0);
            random_kernel_test(&kernel, scalar, cmp);
        }
    }
    fn test_kernel_i32(kernel: impl Kernel<Scalar = i32>) {
        for _ in 0..20 {
            test_kernel_with_random_i32(&kernel);
        }
    }
}
