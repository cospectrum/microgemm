# microgemm

General matrix multiplication with custom configuration in Rust.

## Installation

To use the latest version from git, add this to your `Cargo.toml`:
```toml
[dependencies]
microgemm = { git = "https://github.com/cospectrum/microgemm.git" }
```

The **MSRV** is `1.65.0`

## Usage

The `Kernel` trait is the main abstraction of microgemm.
You can implement it yourself or use kernels that are already provided out of the box.

### gemm

```rs
use microgemm as mg;
use microgemm::Kernel as _;

fn main() {
    let kernel = mg::kernels::Generic8x8Kernel::<f32>::new();
    assert_eq!(kernel.mr(), 8);
    assert_eq!(kernel.nr(), 8);

    let pack_sizes = mg::PackSizes {
        mc: 5 * kernel.mr(), // MC must be divisible by MR
        kc: 380,
        nc: 10 * kernel.nr(), // NC must be divisible by NR
    };
    let mut packing_buf = vec![0.0; pack_sizes.buf_len()];

    let (m, k, n) = (100, 380, 250);

    let a = vec![2.0; m * k];
    let b = vec![3.0; k * n];
    let mut c = vec![4.0; m * n];

    let a = mg::MatRef::new(m, k, &a, mg::Layout::RowMajor);
    let b = mg::MatRef::new(k, n, &b, mg::Layout::RowMajor);
    let mut c = mg::MatMut::new(m, n, &mut c, mg::Layout::RowMajor);

    let alpha = 2.0;
    let beta = -3.0;

    // c <- alpha a b + beta c
    kernel.gemm(alpha, &a, &b, beta, &mut c, &pack_sizes, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
```

### Implemented Kernels

| Name | Scalar Types | Target |
| ---- | ------------ | ------ |
| GenericNxNKernel <br> (N = 2, 4, ..., 32) | T: Copy + Zero + One + Mul<Output = T> + Add<Output = T> | Any |
| NeonKernel | f32 | AArch64 and target feature neon |

### Custom Kernel Implementation

```rs
use microgemm::{typenum::U4, Kernel, MatMut, MatRef};

struct CustomKernel;

impl Kernel for CustomKernel {
    type Scalar = f64;
    type Mr = U4;
    type Nr = U4;

    // dst <- alpha lhs rhs + beta dst
    fn microkernel(
        &self,
        alpha: f64,
        lhs: &MatRef<f64>,
        rhs: &MatRef<f64>,
        beta: f64,
        dst: &mut MatMut<f64>,
    ) {
        // lhs is col-major by default
        assert_eq!(lhs.row_stride(), 1);
        assert_eq!(lhs.nrows(), Self::MR);
        // rhs is row-major by default
        assert_eq!(rhs.col_stride(), 1);
        assert_eq!(rhs.ncols(), Self::NR);

        // your microkernel implementation...
    }
}
```

## License
Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.
