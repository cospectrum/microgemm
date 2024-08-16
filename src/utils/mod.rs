mod arbitrary;
mod naive;

mod proptest_kernel;

use approx::{AbsDiffEq, RelativeEq};

pub use arbitrary::*;
pub use naive::naive_gemm;
pub use proptest_kernel::{proptest_kernel, ProptestKernelCfg};

pub fn assert_approx_eq<T>(left: impl AsRef<[T]>, right: impl AsRef<[T]>, eps: T)
where
    T: RelativeEq<T> + Copy + core::fmt::Debug + AbsDiffEq<Epsilon = T>,
{
    let left = left.as_ref();
    let right = right.as_ref();
    assert_eq!(left.len(), right.len());

    for (&l, &r) in left.iter().zip(right) {
        assert_relative_eq!(l, r, epsilon = eps);
    }
}

#[allow(dead_code)]
pub const fn is_release_build() -> bool {
    !is_debug_build()
}

#[allow(dead_code)]
pub const fn is_debug_build() -> bool {
    let mut debug = false;
    debug_assert!({
        debug = true;
        debug
    });
    debug
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eq() {
        let eps = f32::EPSILON;
        let x = [1., 2., 3.];
        let y = [x[0] + eps, x[1] + 3. * eps, x[2] + 2. * eps];
        assert_approx_eq(x, y, 4. * eps);
    }
    #[test]
    #[should_panic]
    fn test_ne() {
        let eps = f32::EPSILON;
        let x = [1., 2., 3.];
        let y = [x[0] + eps, x[1] + 3. * eps, x[2] + 2. * eps];
        assert_approx_eq(x, y, eps);
    }
}
