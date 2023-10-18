use core::ops::{Add, Mul};
use num_traits::{One, Zero};

use crate::Kernel;

#[derive(Debug, Clone, Copy, Default)]
pub struct Generic4x4Kernel;

impl<T> Kernel<T> for Generic4x4Kernel
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
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
        debug_assert_eq!(dst.nrows(), 4);
        debug_assert_eq!(dst.ncols(), 4);
        assert_eq!(lhs.as_slice().len(), rhs.as_slice().len());

        let mut col0 = [T::zero(); 4];
        let mut col1 = [T::zero(); 4];
        let mut col2 = [T::zero(); 4];
        let mut col3 = [T::zero(); 4];

        let update_col = |col: &mut [T; 4], v: &[T], scalar: T| {
            col.iter_mut().zip(v).for_each(|(out, &x)| {
                *out = *out + x * scalar;
            });
        };

        let left = lhs.as_slice().chunks_exact(4);
        let right = rhs.as_slice().chunks_exact(4);

        for (a, b) in left.zip(right) {
            update_col(&mut col0, a, b[0]);
            update_col(&mut col1, a, b[1]);
            update_col(&mut col2, a, b[2]);
            update_col(&mut col3, a, b[3]);
        }

        let mut write_to = |ncol: usize, from: &[T; 4]| {
            for (i, &val) in from.iter().enumerate() {
                let to = dst.get_mut(i, ncol);
                *to = alpha * val + beta * *to;
            }
        };
        write_to(0, &col0);
        write_to(1, &col1);
        write_to(2, &col2);
        write_to(3, &col3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::*;
    use rand::{thread_rng, Rng};

    const KERNEL: &Generic4x4Kernel = &Generic4x4Kernel;

    #[test]
    fn test_kernel_generic_4x4_f32() {
        let cmp = |expect: &[f32], got: &[f32]| {
            let eps = 60.0 * f32::EPSILON;
            assert_relative_eq!(expect, got, epsilon = eps);
        };
        let mut rng = thread_rng();

        for _ in 0..20 {
            let scalar = || rng.gen_range(-1.0..1.0);
            random_kernel_test(KERNEL, scalar, cmp);
        }
    }

    #[test]
    fn test_kernel_generic_4x4_i32() {
        for _ in 0..20 {
            test_kernel_with_random_i32(KERNEL);
        }
    }
}
