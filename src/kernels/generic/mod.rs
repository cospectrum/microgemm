mod square_kernel;
use square_kernel::GenericSquareKernel;

pub type Generic2x2Kernel<T> = GenericSquareKernel<T, 2>;
pub type Generic4x4Kernel<T> = GenericSquareKernel<T, 4>;
pub type Generic8x8Kernel<T> = GenericSquareKernel<T, 8>;
pub type Generic16x16Kernel<T> = GenericSquareKernel<T, 16>;
pub type Generic32x32Kernel<T> = GenericSquareKernel<T, 32>;

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
            assert_relative_eq!(expect, got, epsilon = eps);
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
