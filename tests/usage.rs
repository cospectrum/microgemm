use microgemm::{naive_gemm, Kernel, Layout, MatMut, MatRef, PackSizes};

struct CustomKernel;

impl Kernel<i32> for CustomKernel {
    const MR: usize = 3;
    const NR: usize = 2;

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
        assert!(dst.nrows() == Self::MR && dst.ncols() == Self::NR);

        naive_gemm(alpha, lhs, rhs, beta, dst);
    }
}

const PACK_SIZES: PackSizes = PackSizes {
    mc: 6, // must be divisible by MR
    kc: 5,
    nc: 8, // must be divisible by NR
};

#[test]
fn main() {
    let kernel = CustomKernel;
    let m = 10;
    let k = 20;
    let n = 15;

    let a = (0..m * k).map(|x| x as i32).collect::<Vec<_>>();
    let b = (0..k * n).map(|x| x as i32).collect::<Vec<_>>();
    let mut c = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();

    let a = MatRef::new(m, k, &a, Layout::RowMajor);
    let b = MatRef::new(k, n, &b, Layout::ColumnMajor);
    let mut c = MatMut::new(m, n, &mut c, Layout::RowMajor);

    let mut buf = [0; PACK_SIZES.buf_len::<i32, CustomKernel>()];

    let alpha = 2;
    let beta = -3;

    // c <- alpha a b + beta c
    kernel.gemm(alpha, &a, &b, beta, &mut c, &PACK_SIZES, &mut buf);
    println!("{:?}", c.as_slice());
}
