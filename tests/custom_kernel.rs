use microgemm::{utils::naive_gemm, Kernel, Layout, MatMut, MatRef, PackSizes};

struct CustomKernel;

const PACK_SIZES: PackSizes = PackSizes {
    mc: 2 * CustomKernel::MR, // must be divisible by MR
    kc: 5,
    nc: 3 * CustomKernel::NR, // must be divisible by NR
};

#[test]
fn main() {
    let kernel = CustomKernel;

    let mut buf = [0; PACK_SIZES.buf_len::<i64, CustomKernel>()];

    let m = 10;
    let k = 2;
    let n = 15;

    let a = (0..m * k).map(|x| x as i64).collect::<Vec<_>>();
    let b = (0..k * n).map(|x| x as i64).collect::<Vec<_>>();
    let mut c = (0..m * n).map(|x| x as i64).collect::<Vec<_>>();
    let mut expect = c.clone();

    let a = MatRef::new(m, k, &a, Layout::RowMajor);
    let b = MatRef::new(k, n, &b, Layout::ColumnMajor);
    let mut c = MatMut::new(m, n, &mut c, Layout::RowMajor);
    let mut expect = c.with_values(&mut expect);

    let alpha = 2;
    let beta = -3;

    // c <- alpha a b + beta c
    kernel.gemm(alpha, &a, &b, beta, &mut c, &PACK_SIZES, &mut buf);
    naive_gemm(alpha, &a, &b, beta, &mut expect);
    assert_eq!(c.as_slice(), expect.as_slice());
}

impl Kernel<i64> for CustomKernel {
    const MR: usize = 2;
    const NR: usize = 2;

    fn microkernel(
        &self,
        alpha: i64,
        lhs: &MatRef<i64>,
        rhs: &MatRef<i64>,
        beta: i64,
        dst: &mut MatMut<i64>,
    ) {
        assert_eq!(lhs.nrows(), Self::MR);
        assert_eq!(rhs.ncols(), Self::NR);
        assert_eq!(lhs.ncols(), rhs.nrows());

        assert_eq!(dst.nrows(), Self::MR);
        assert_eq!(dst.ncols(), Self::NR);

        assert_eq!(lhs.row_stride(), 1); // lhs is column-major by default
        assert_eq!(rhs.col_stride(), 1); // rhs is row-major by default

        let left = lhs.as_slice().chunks_exact(2);
        let right = rhs.as_slice().chunks_exact(2);

        let mut col0 = [0; 2];
        let mut col1 = [0; 2];

        for (a, b) in left.zip(right) {
            col0[0] += a[0] * b[0];
            col0[1] += a[1] * b[0];

            col1[0] += a[0] * b[1];
            col1[1] += a[1] * b[1];
        }
        let c00 = dst.get_mut(0, 0);
        *c00 = alpha * col0[0] + beta * *c00;

        let c01 = dst.get_mut(0, 1);
        *c01 = alpha * col1[0] + beta * *c01;

        let c10 = dst.get_mut(1, 0);
        *c10 = alpha * col0[1] + beta * *c10;

        let c11 = dst.get_mut(1, 1);
        *c11 = alpha * col1[1] + beta * *c11;
    }
}
