mod naive;
mod random;

use approx::{AbsDiffEq, RelativeEq};

pub use naive::naive_gemm;
pub use random::*;

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
