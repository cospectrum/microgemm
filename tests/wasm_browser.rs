#![cfg(target_arch = "wasm32")]
use microgemm::{kernels::Generic4x4Kernel, Kernel, Layout, MatMut, MatRef, PackSizes};
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_wasm_simd128_f32() {
    let kernel = Generic4x4Kernel::<f32>::new();
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [10., 11., 20., 21., 30., 31.];
    let mut c = vec![0f32; 2 * 2];
    let a = MatRef::new(2, 3, a.as_ref(), Layout::RowMajor);
    let b = MatRef::new(3, 2, b.as_ref(), Layout::RowMajor);
    let mut c = MatMut::new(2, 2, c.as_mut(), Layout::RowMajor);
    let pack_sizes = PackSizes {
        mc: kernel.mr(),
        kc: 2,
        nc: kernel.nr(),
    };
    let packing_buf = vec![0f32; pack_sizes.buf_len()];
    kernel.gemm(
        1f32,
        a.as_ref(),
        b.as_ref(),
        0f32,
        c.as_mut(),
        pack_sizes,
        packing_buf,
    );
    assert_eq!(c.as_slice(), [140., 146., 320., 335.]);
}
