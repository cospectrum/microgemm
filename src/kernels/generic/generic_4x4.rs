use super::{square_microkernel, write_to_col_major};
use crate::Kernel;

use core::marker::PhantomData;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy, Default)]
pub struct Generic4x4Kernel<T> {
    marker: PhantomData<T>,
}

impl<T> Generic4x4Kernel<T> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T> Kernel for Generic4x4Kernel<T>
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    type Elem = T;

    const MR: usize = 4;
    const NR: usize = 4;

    fn microkernel(
        &self,
        alpha: T,
        lhs: &crate::MatRef<T>,
        rhs: &crate::MatRef<T>,
        beta: T,
        dst: &mut crate::MatMut<T>,
    ) {
        debug_assert_eq!(dst.nrows(), Self::MR);
        debug_assert_eq!(dst.ncols(), Self::NR);
        debug_assert_eq!(dst.row_stride(), 1);

        const DIM: usize = 4;
        let mut cols = [T::zero(); DIM * DIM];
        square_microkernel::<_, DIM>(lhs.as_slice(), rhs.as_slice(), &mut cols);
        write_to_col_major::<_, DIM>(dst.as_mut_slice(), &cols, alpha, beta);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_generic4x4_kernel_f32() {
        let kernel = &Generic4x4Kernel::<f32>::new();
        let cmp = |expect: &[f32], got: &[f32]| {
            let eps = 70.0 * f32::EPSILON;
            assert_relative_eq!(expect, got, epsilon = eps);
        };
        let mut rng = thread_rng();

        for _ in 0..20 {
            let scalar = || rng.gen_range(-1.0..1.0);
            random_kernel_test(kernel, scalar, cmp);
        }
    }

    #[test]
    fn test_generic4x4_i32() {
        let kernel = &Generic4x4Kernel::<i32>::new();
        for _ in 0..20 {
            test_kernel_with_random_i32(kernel);
        }
    }
}
