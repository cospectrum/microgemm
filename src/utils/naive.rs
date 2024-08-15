use crate::std_prelude::*;
use crate::{MatMut, MatRef, Zero};
use core::ops::{Add, Mul};

pub fn naive_gemm<T>(alpha: T, a: MatRef<T>, b: MatRef<T>, beta: T, c: &mut MatMut<T>)
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
{
    assert_eq!(a.nrows(), c.nrows());
    assert_eq!(b.ncols(), c.ncols());
    assert_eq!(a.ncols(), b.nrows());

    let k = a.ncols();

    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let dot = (0..k)
                .map(|h| a.get(i, h) * b.get(h, j))
                .reduce(|accum, x| accum + x)
                .unwrap_or(T::zero());
            let z = c.get_mut(i, j);
            *z = alpha * dot + beta * *z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[rustfmt::skip]
    #[test]
    fn fixed_1() {
        let a = [
            1, 2, 3,
            4, 5, 6,
        ];
        let b = [
            10, 11,
            20, 21,
            30, 31,
        ];
        let a = MatRef::row_major(2, 3, &a);
        let b = MatRef::row_major(3, 2, &b);

        let mut c = [-1; 4];
        let mut c = MatMut::row_major(2, 2, c.as_mut());
        let expect = [
            140, 146,
            320, 335,
        ];
        naive_gemm(1, a, b, 0, c.as_mut());
        assert_eq!(c.as_slice(), expect);

        let mut c = [-1; 4];
        let mut c = MatMut::col_major(2, 2, c.as_mut());
        let expect = [
            140, 320,
            146, 335,
        ];
        naive_gemm(1, a, b, 0, c.as_mut());
        assert_eq!(c.as_slice(), expect);
    }

    #[rustfmt::skip]
    #[test]
    fn fixed_2() {
        let alpha = 3;
        let beta = -4;

        let a = [
            1, 2, 3,
            4, 5, 6,
        ];
        let b = [
            2, 3, 4,
            5, 6, 7,
        ];
        let mut c = [
            -4, 1,
            -5, -6,
        ];
        let expect = [
            beta * c[0] + alpha * (2 + 2 * 3 + 3 * 4),
            beta * c[1] + alpha * (5 + 2 * 6 + 3 * 7),
            beta * c[2] + alpha * (4 * 2 + 5 * 3 + 6 * 4),
            beta * c[3] + alpha * (4 * 5 + 5 * 6 + 6 * 7),
        ];

        let a = MatRef::row_major(2, 3, a.as_ref());
        let b = MatRef::col_major(3, 2, b.as_ref());
        let mut c = MatMut::row_major(2, 2, c.as_mut());

        naive_gemm(alpha, a, b, beta, c.as_mut());
        assert_eq!(c.as_slice(), expect);
    }

    #[test]
    #[rustfmt::skip]
    fn fixed_3() {
        let a = [
            1, 0, 2,
            0, -1, 3,
        ];
        let a = MatRef::row_major(2, 3, a.as_ref());
        let b = [
            2, -1,
            0, 5,
            1, 1,
        ];
        let b = MatRef::row_major(3, 2, b.as_ref());

        let mut c = [-9; 2 * 2];
        let c = &mut MatMut::row_major(2, 2, c.as_mut());
        let expect = [
            4, 1,
            3, -2,
        ];
        naive_gemm(1, a, b, 0, c);
        assert_eq!(c.as_slice(), expect);
    }
}
