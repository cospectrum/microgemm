use crate::{
    kernels::dbg_check_microkernel_inputs,
    typenum::{U16, U2, U32, U4, U8},
    Kernel, One, Zero,
};
use core::marker::PhantomData;
use core::ops::{Add, Mul};

fn loop_micropanels<T, const DIM: usize>(lhs: &[T], rhs: &[T], cols: &mut [T])
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
{
    assert_eq!(cols.len(), DIM * DIM);
    assert!(DIM > 0);
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

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::{
        std_prelude::*,
        utils::{is_debug_build, proptest_kernel, ProptestKernelCfg},
    };
    use proptest::prelude::*;

    fn cfg_i32() -> ProptestKernelCfg<i32> {
        let dim = if is_debug_build() { 38 } else { 83 };
        ProptestKernelCfg::default()
            .with_max_matrix_dim(dim)
            .with_max_pack_dim(2 * dim + 1)
            .with_scalar((-11..11).boxed())
    }

    #[test]
    fn proptest_generic_kernel_2x2_i32() {
        proptest_kernel(&GenericKernel2x2::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_4x4_i32() {
        proptest_kernel(&GenericKernel4x4::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_8x8_i32() {
        proptest_kernel(&GenericKernel8x8::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_16x16_i32() {
        proptest_kernel(&GenericKernel16x16::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_32x32_i32() {
        proptest_kernel(&GenericKernel32x32::new(), cfg_i32()).unwrap();
    }
}
