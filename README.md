# microgemm

Rust implementation of general matrix multiplication based on the BLIS microkernel
approach with support for custom tuning.

## Getting started

```rs
use microgemm::{gemm_with_params, naive_gemm, BlockSizes, Layout, MatMut, MatRef};

const BLOCK_SIZES: BlockSizes = BlockSizes {
    mc: 6,
    mr: 3,
    kc: 5,
    nc: 8,
    nr: 2,
};

fn main() {
    let m = 10;
    let k = 20;
    let n = 15;

    let lhs = (0..m * k).map(|x| x as i32).collect::<Vec<_>>();
    let rhs = (0..k * n).map(|x| x as i32).collect::<Vec<_>>();
    let mut dst = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();

    let lhs = MatRef::new(m, k, &lhs, Layout::RowMajor);
    let rhs = MatRef::new(k, n, &rhs, Layout::ColumnMajor);
    let mut dst = MatMut::new(m, n, &mut dst, Layout::RowMajor);

    let alpha = 2;
    let beta = -3;
    let mut buf = [0; BLOCK_SIZES.buf_len()];

    gemm_with_params(
        alpha,
        &lhs,
        &rhs,
        beta,
        &mut dst,
        microkernel,
        &BLOCK_SIZES,
        &mut buf,
    );
    println!("{:?}", dst.as_slice());
}

fn microkernel(alpha: i32, a: &MatRef<i32>, b: &MatRef<i32>, beta: i32, c: &mut MatMut<i32>) {
    assert_eq!(a.nrows(), BLOCK_SIZES.mr);
    assert_eq!(a.ncols(), BLOCK_SIZES.kc);

    assert_eq!(b.nrows(), BLOCK_SIZES.kc);
    assert_eq!(b.ncols(), BLOCK_SIZES.nr);

    assert_eq!(c.nrows(), BLOCK_SIZES.mr);
    assert_eq!(c.ncols(), BLOCK_SIZES.nr);

    naive_gemm(alpha, a, b, beta, c);
}
```
