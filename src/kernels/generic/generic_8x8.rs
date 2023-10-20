use crate::Kernel;
use core::marker::PhantomData;
use core::ops::{Add, Mul};
use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy, Default)]
pub struct Generic8x8Kernel<T> {
    marker: PhantomData<T>,
}

impl<T> Generic8x8Kernel<T> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T> Kernel for Generic8x8Kernel<T>
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    type Elem = T;

    const MR: usize = 8;
    const NR: usize = 8;

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
        assert_eq!(lhs.as_slice().len(), rhs.as_slice().len());

        let mut col0 = [T::zero(); 8];
        let mut col1 = [T::zero(); 8];
        let mut col2 = [T::zero(); 8];
        let mut col3 = [T::zero(); 8];
        let mut col4 = [T::zero(); 8];
        let mut col5 = [T::zero(); 8];
        let mut col6 = [T::zero(); 8];
        let mut col7 = [T::zero(); 8];

        let update_col = |col: &mut [T; 8], v: &[T], scalar: T| {
            col.iter_mut().zip(v).for_each(|(out, &x)| {
                *out = *out + x * scalar;
            });
        };

        let left = lhs.as_slice().chunks_exact(8);
        let right = rhs.as_slice().chunks_exact(8);

        for (a, b) in left.zip(right) {
            update_col(&mut col0, a, b[0]);
            update_col(&mut col1, a, b[1]);
            update_col(&mut col2, a, b[2]);
            update_col(&mut col3, a, b[3]);
            update_col(&mut col4, a, b[4]);
            update_col(&mut col5, a, b[5]);
            update_col(&mut col6, a, b[6]);
            update_col(&mut col7, a, b[7]);
        }

        let mut write_to = |ncol: usize, from: &[T; 8]| {
            for (i, &val) in from.iter().enumerate() {
                let to = dst.get_mut(i, ncol);
                *to = alpha * val + beta * *to;
            }
        };
        write_to(0, &col0);
        write_to(1, &col1);
        write_to(2, &col2);
        write_to(3, &col3);
        write_to(4, &col4);
        write_to(5, &col5);
        write_to(6, &col6);
        write_to(7, &col7);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::*;
    use rand::{thread_rng, Rng};

    #[test]
    fn test_generic8x8_kernel_f32() {
        let kernel = &Generic8x8Kernel::<f32>::new();
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
    fn test_generic8x8_i32() {
        let kernel = &Generic8x8Kernel::<i32>::new();
        for _ in 0..20 {
            test_kernel_with_random_i32(kernel);
        }
    }
}
