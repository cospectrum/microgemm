use crate::{
    kernels::dbg_check_microkernel_inputs,
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
    ($struct:ident, $dim:literal, $dimty:ty) => {
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
            type Mr = $dimty;
            type Nr = $dimty;

            fn microkernel(
                &self,
                alpha: Self::Scalar,
                lhs: crate::MatRef<Self::Scalar>,
                rhs: crate::MatRef<Self::Scalar>,
                beta: Self::Scalar,
                dst: &mut crate::MatMut<Self::Scalar>,
            ) {
                dbg_check_microkernel_inputs(self, lhs, rhs, dst);

                const DIM: usize = $dim;
                let mut cols = [T::zero(); DIM * DIM];
                loop_micropanels::<_, DIM>(lhs.as_slice(), rhs.as_slice(), &mut cols);
                write_cols_to_colmajor::<_, DIM>(dst.as_mut_slice(), &cols, alpha, beta);
            }
        }
    };
}

impl_generic_square_kernel!(GenericKernel2x2, 2, U2);
impl_generic_square_kernel!(GenericKernel4x4, 4, U4);
impl_generic_square_kernel!(GenericKernel8x8, 8, U8);
impl_generic_square_kernel!(GenericKernel16x16, 16, U16);
impl_generic_square_kernel!(GenericKernel32x32, 32, U32);

#[cfg(test)]
mod proptests {
    use super::*;
    use crate::std_prelude::*;
    use proptest::proptest;

    use crate::{
        as_mut,
        utils::{self, arb_matrix_triple_with, arb_pack_sizes},
    };

    proptest! {
        #[test]
        fn proptest_generic_kernel_i32(
            [a, b, c] in arb_matrix_triple_with::<i32>(
                1..30, 1..30, 1..30,
                -100..100,
            ),
            alpha in -10..10,
            beta in -10..10,
        ) {
            let mut expected = c.clone();
            utils::naive_gemm(
                alpha,
                a.to_ref(),
                b.to_ref(),
                beta,
                as_mut!(expected),
            );

            let ker = GenericKernel2x2::new();
            let packs = arb_pack_sizes(&ker, 1..60, 1..60, 1..60);
            proptest!(|(pack in packs)| {
                let mut actual = c.clone();
                ker.gemm_in(
                    crate::GlobalAllocator,
                    alpha,
                    a.to_ref(),
                    b.to_ref(),
                    beta,
                    as_mut!(actual),
                    pack,
                );
                assert_eq!(actual.as_slice(), expected.as_slice());
            });
        }
    }
}
