use crate::{MatMut, MatRef};
use core::ops::{Add, Mul};

pub fn naive_gemm<T>(alpha: T, a: &MatRef<T>, b: &MatRef<T>, beta: T, c: &mut MatMut<T>)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
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
                .unwrap();
            let z = c.get_mut(i, j);
            *z = alpha * dot + beta * *z;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Layout;

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
        let a = MatRef::new(2, 3, &a, Layout::RowMajor);
        let b = MatRef::new(3, 2, &b, Layout::RowMajor);

        let mut c = [-1; 4];
        let mut c = MatMut::new(2, 2, c.as_mut(), Layout::RowMajor);
        let expect = [
            140, 146,
            320, 335,
        ];
        naive_gemm(1, &a, &b, 0, c.as_mut());
        assert_eq!(c.as_slice(), expect);

        let mut c = [-1; 4];
        let mut c = MatMut::new(2, 2, c.as_mut(), Layout::ColumnMajor);
        let expect = [
            140, 320,
            146, 335,
        ];
        naive_gemm(1, &a, &b, 0, c.as_mut());
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

        let a = MatRef::new(2, 3, a.as_ref(), Layout::RowMajor);
        let b = MatRef::new(3, 2, b.as_ref(), Layout::ColumnMajor);
        let mut c = MatMut::new(2, 2, c.as_mut(), Layout::RowMajor);

        naive_gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut());
        assert_eq!(c.as_slice(), expect);
    }
}
