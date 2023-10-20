# microgemm

General matrix multiplication with custom configuration in Rust.

## Install

```sh
cargo add microgemm
```

## Usage

The `Kernel` trait is the main abstraction of microgemm.
You can implement it yourself or use kernels that are already provided out of the box.

### Kernel Usage

```rs
use microgemm as mg;
use microgemm::Kernel as _;

fn main() {
    let kernel = mg::kernels::Generic4x4Kernel::<f32>::new();
    assert_eq!(kernel.mr(), 4);
    assert_eq!(kernel.nr(), 4);

    let pack_sizes = mg::PackSizes {
        mc: 10 * kernel.mr(), // MC must be divisible by MR
        kc: 200,
        nc: 20 * kernel.nr(), // NC must be divisible by NR
    };
    let buf_len = pack_sizes.buf_len(&kernel);
    let mut buf = vec![0.0; buf_len];

    let m = 100;
    let k = 380;
    let n = 250;

    let a = vec![2.0; m * k];
    let b = vec![3.0; k * n];
    let mut c = vec![4.0; m * n];

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

### Custom Kernel Implementation

The `Kernel` trait requires you to set the MR/NR constants and implement the `microkernel`.

```rs
use microgemm::{Kernel, MatMut, MatRef};

struct CustomKernel;

impl Kernel for CustomKernel {
    type Elem = f64;

    const MR: usize = 2;
    const NR: usize = 2;

    // dst <- alpha lhs rhs + beta dst
    fn microkernel(
        &self,
        alpha: f64,
        lhs: &MatRef<f64>,
        rhs: &MatRef<f64>,
        beta: f64,
        dst: &mut MatMut<f64>,
    ) {
        assert_eq!(lhs.row_stride(), 1); // lhs is col-major by default
        assert_eq!(rhs.col_stride(), 1); // rhs is row-major by default
        assert_eq!(lhs.nrows(), Self::MR);
        assert_eq!(rhs.ncols(), Self::NR);
        // your microkernel implementation...
    }
}
```
