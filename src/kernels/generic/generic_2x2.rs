use core::{
    marker::PhantomData,
    ops::{Add, Mul},
};
use num_traits::{One, Zero};

use crate::Kernel;

pub fn generic2x2_kernel<T>() -> impl Kernel<Elem = T>
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    Generic2x2Kernel::new()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct Generic2x2Kernel<T> {
    marker: PhantomData<T>,
}

impl<T> Generic2x2Kernel<T> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T> Kernel for Generic2x2Kernel<T>
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    type Elem = T;

    const MR: usize = 2;
    const NR: usize = 2;

    fn microkernel(
        &self,
        alpha: T,
        lhs: &crate::MatRef<T>,
        rhs: &crate::MatRef<T>,
        beta: T,
        dst: &mut crate::MatMut<T>,
    ) {
        debug_assert_eq!(dst.nrows(), 2);
        debug_assert_eq!(dst.ncols(), 2);
        assert_eq!(lhs.as_slice().len(), rhs.as_slice().len());

        let mut col0 = [T::zero(); 2];
        let mut col1 = [T::zero(); 2];

        let update_col = |col: &mut [T; 2], v: &[T], scalar: T| {
            col.iter_mut().zip(v).for_each(|(out, &x)| {
                *out = *out + x * scalar;
            });
        };

        let left = lhs.as_slice().chunks_exact(2);
        let right = rhs.as_slice().chunks_exact(2);

        for (a, b) in left.zip(right) {
            update_col(&mut col0, a, b[0]);
            update_col(&mut col1, a, b[1]);
        }

        let mut write_to = |ncol: usize, from: &[T; 2]| {
            for (i, &val) in from.iter().enumerate() {
                let to = dst.get_mut(i, ncol);
                *to = alpha * val + beta * *to;
            }
        };
        write_to(0, &col0);
        write_to(1, &col1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_generic2x2_kernel_f64() {
        let kernel = generic2x2_kernel::<f64>();

        let cmp = |expect: &[f64], got: &[f64]| {
            let eps = 70.0 * f64::EPSILON;
            assert_relative_eq!(expect, got, epsilon = eps);
        };
        let mut rng = thread_rng();

        for _ in 0..20 {
            let scalar = || rng.gen_range(-1.0..1.0);
            random_kernel_test(&kernel, scalar, cmp);
        }
    }

    #[test]
    fn test_generic2x2_kernel_i32() {
        let kernel = generic2x2_kernel::<i32>();
        for _ in 0..20 {
            test_kernel_with_random_i32(&kernel);
        }
    }
}
