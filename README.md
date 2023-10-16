# microgemm

General matrix multiplication with custom configuration in Rust.

## Getting started

```sh
cargo add microgemm
```

### Usage

You need to provide a microkernel, as well as block sizes and a buffer for
intermediate results.

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

    let a = (0..m * k).map(|x| x as i32).collect::<Vec<_>>();
    let b = (0..k * n).map(|x| x as i32).collect::<Vec<_>>();
    let mut c = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();

    let a = MatRef::new(m, k, &a, Layout::RowMajor);
    let b = MatRef::new(k, n, &b, Layout::ColumnMajor);
    let mut c = MatMut::new(m, n, &mut c, Layout::RowMajor);

    let alpha = 2;
    let beta = -3;
    let mut buf = [0; BLOCK_SIZES.buf_len()];

    // c <- alpha a b + beta c
    gemm_with_params(
        alpha,
        &a,
        &b,
        beta,
        &mut c,
        microkernel,
        &BLOCK_SIZES,
        &mut buf,
    );
    println!("{:?}", c.as_slice());
}

fn microkernel(alpha: i32, lhs: &MatRef<i32>, rhs: &MatRef<i32>, beta: i32, dst: &mut MatMut<i32>) {
    assert_eq!(lhs.nrows(), BLOCK_SIZES.mr);
    assert_eq!(lhs.ncols(), BLOCK_SIZES.kc);

    assert_eq!(rhs.nrows(), BLOCK_SIZES.kc);
    assert_eq!(rhs.ncols(), BLOCK_SIZES.nr);

    assert_eq!(dst.nrows(), BLOCK_SIZES.mr);
    assert_eq!(dst.ncols(), BLOCK_SIZES.nr);

    naive_gemm(alpha, lhs, rhs, beta, dst);
}
```
