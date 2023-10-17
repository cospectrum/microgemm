use core::{
    marker::PhantomData,
    ops::{Add, Mul},
};
use num_traits::{One, Zero};

use crate::Kernel;

#[derive(Debug, Clone, Copy, Default)]
pub struct KernelGeneric4x4<T>
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    marker: PhantomData<T>,
}

impl<T> KernelGeneric4x4<T>
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl<T> Kernel<T> for KernelGeneric4x4<T>
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
    use crate::utils::test_kernel_with_random_i32;

    use super::*;

    #[test]
    fn test_kernel_generic_4x4() {
        let kernel = KernelGeneric4x4::new();
        for _ in 0..20 {
            test_kernel_with_random_i32(&kernel);
        }
    }
}
