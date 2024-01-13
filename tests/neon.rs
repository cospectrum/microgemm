#![cfg(target_arch = "aarch64")]

use microgemm::{kernels::NeonKernel, Kernel, MatMut, MatRef, PackSizes};

#[test]
fn test_neon() {
    let kernel = if cfg!(target_feature = "neon") {
        unsafe { NeonKernel::<f32>::new() }
    } else {
        println!("neon feature is not supported");
        return;
    };
    let a = [1., 2., 3., 4., 5., 6.];
    let b = [10., 11., 20., 21., 30., 31.];
    let mut c = vec![0f32; 2 * 2];
    let a = MatRef::row_major(2, 3, a.as_ref());
    let b = MatRef::row_major(3, 2, b.as_ref());
    let mut c = MatMut::row_major(2, 2, c.as_mut());
    let pack_sizes = PackSizes {
        mc: kernel.mr(),
        kc: 2,
        nc: kernel.nr(),
    };
    let mut packing_buf = vec![0f32; pack_sizes.buf_len()];
    kernel.gemm(
        1f32,
        a.as_ref(),
        b.as_ref(),
        0f32,
        c.as_mut(),
        pack_sizes,
        packing_buf.as_mut(),
    );
    assert_eq!(c.as_slice(), [140., 146., 320., 335.]);
}
