#![cfg(target_arch = "wasm32")]

use microgemm as mg;
use microgemm::Kernel as _;

fn main() {
    let kernel = if cfg!(target_feature = "simd128") {
        unsafe { mg::kernels::WasmSimd128Kernel::<f32>::new() }
    } else {
        panic!("simd128 target feature is not enabled");
    };
    assert_eq!(kernel.mr(), 4);
    assert_eq!(kernel.nr(), 4);

    let pack_sizes = mg::PackSizes {
        mc: 5 * kernel.mr(),
        kc: 380,
        nc: 10 * kernel.nr(),
    };
    let mut packing_buf = vec![0.0; pack_sizes.buf_len()];

    let alpha = 2.0;
    let beta = -3.0;
    let (m, k, n) = (100, 380, 250);

    let a = vec![2.0; m * k];
    let b = vec![3.0; k * n];
    let mut c = vec![4.0; m * n];

    let a = mg::MatRef::new(m, k, &a, mg::Layout::RowMajor);
    let b = mg::MatRef::new(k, n, &b, mg::Layout::RowMajor);
    let mut c = mg::MatMut::new(m, n, &mut c, mg::Layout::RowMajor);

    // c <- alpha a b + beta c
    kernel.gemm(alpha, &a, &b, beta, &mut c, &pack_sizes, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
