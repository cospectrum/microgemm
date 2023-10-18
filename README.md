# microgemm

General matrix multiplication with custom configuration in Rust.

## Getting started

```sh
cargo add microgemm
```

```rs
use microgemm as mg;
use microgemm::Kernel as _;

#[test]
fn main() {
    let kernel = mg::select_kernel::<f32>();

    let pack_sizes = mg::PackSizes {
        mc: 4 * kernel.mr(), // MC must be divisible by MR
        kc: 8,
        nc: 2 * kernel.nr(), // NC must be divisible by NR
    };
    let buf_len = pack_sizes.buf_len(&kernel);
    let mut buf = vec![0.0; buf_len];

    let m = 10;
    let k = 16;
    let n = 15;

    let a = vec![1.0; m * k];
    let b = vec![2.0; k * n];
    let mut c = vec![3.0; m * n];

    let a = mg::MatRef::new(m, k, &a, mg::Layout::RowMajor);
    let b = mg::MatRef::new(k, n, &b, mg::Layout::RowMajor);
    let mut c = mg::MatMut::new(m, n, &mut c, mg::Layout::RowMajor);

    let alpha = 2.0;
    let beta = -3.0;

    // c <- alpha a b + beta c
    kernel.gemm(alpha, &a, &b, beta, &mut c, &pack_sizes, &mut buf);
    println!("{:?}", c.as_slice());
}
```
