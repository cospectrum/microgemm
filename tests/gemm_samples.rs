use microgemm::{typenum::U5, utils::naive_gemm, Kernel, Layout, MatMut, MatRef, PackSizes};

struct TestKernel;

impl Kernel for TestKernel {
    type Scalar = i32;
    type Mr = U5;
    type Nr = U5;

    fn microkernel(
        &self,
        alpha: i32,
        lhs: &MatRef<i32>,
        rhs: &MatRef<i32>,
        beta: i32,
        dst: &mut MatMut<i32>,
    ) {
        assert_eq!(lhs.nrows(), Self::MR);
        assert_eq!(rhs.ncols(), Self::NR);
        assert_eq!(dst.nrows(), Self::MR);
        assert_eq!(dst.ncols(), Self::NR);
        naive_gemm(alpha, lhs, rhs, beta, dst);
    }
}

#[rustfmt::skip]
#[test]
fn gemm_sample_one() {
    let kernel = TestKernel;

    let a = [
        28, 26, -9, -29,
        29, -8, 23, 22,
        -2, -2, 26, -21,
        -29, 2, 26, -17,
        -22, -18, -24, -23,
        -20, 14, 13, -22,
    ];
    let a = MatRef::new(6, 4, &a, Layout::RowMajor);
    let b = [
        2, -24, 20,
        -27, -1, -16,
        -12, -29, -26,
        -16, -13, -18,
    ];
    let b = MatRef::new(4, 3, &b, Layout::RowMajor);
    let mut c = vec![
        480, 417, -7102,
        2720, 13184, 2400,
        -578, 3651, 2280,
        1426, -1463, 7849,
        -8973, -12188, -7249,
        508, 627, -7298,
    ];
    let mut expect = c.clone();
    let mut c = MatMut::new(6, 3, &mut c, Layout::RowMajor);
    let mut expect = c.with_values(&mut expect);

    let alpha = 4;
    let beta = -3;

    let pack_sizes = PackSizes {mc: TestKernel::MR, kc: 2, nc: TestKernel::NR };
    let mut buf = vec![-2; pack_sizes.buf_len()];

    kernel.gemm(alpha, a.as_ref(), b.as_ref(), beta, c.as_mut(), &pack_sizes, &mut buf);
    naive_gemm(alpha, a.as_ref(), b.as_ref(), beta, expect.as_mut());
    assert_eq!(expect.as_slice(), c.as_slice());
}
