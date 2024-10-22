#![cfg(target_arch = "aarch64")]

use microgemm::{
    kernels::{NeonKernel4x4, NeonKernel8x8},
    Kernel, MatMut, MatRef, PackSizes,
};

#[test]
fn test_neon8x8() {
    let kernel = if cfg!(target_feature = "neon") {
        unsafe { NeonKernel8x8::<f32>::new() }
    } else {
        println!("neon feature is not supported");
        return;
    };
    test_kernel_f32(kernel);
}

#[test]
fn test_neon4x4() {
    let kernel = if cfg!(target_feature = "neon") {
        unsafe { NeonKernel4x4::<f32>::new() }
    } else {
        println!("neon feature is not supported");
        return;
    };
    test_kernel_f32(kernel);
}

fn test_kernel_f32(kernel: impl Kernel<Scalar = f32>) {
    let pack_sizes = PackSizes {
        mc: kernel.mr(),
        kc: 2,
        nc: kernel.nr(),
    };

    let test_cases = vec![
        TestCase {
            alpha: 1.,
            a: MatRef::row_major(2, 3, &[1., 2., 3., 4., 5., 6.]),
            b: MatRef::row_major(3, 2, &[10., 11., 20., 21., 30., 31.]),
            c: MatRef::row_major(2, 2, &[99.; 2 * 2]),
            beta: 0.,
            expect: &[140., 146., 320., 335.],
            pack_sizes,
        },
        TestCase {
            alpha: 0.,
            a: MatRef::row_major(2, 2, &[1., 2., 3., 4.]),
            b: MatRef::row_major(2, 2, &[5., 6., 3., 4.]),
            c: MatRef::row_major(2, 2, &[3.; 2 * 2]),
            beta: 1.,
            expect: &[3., 3., 3., 3.],
            pack_sizes,
        },
    ];

    for test_case in test_cases {
        test_case.test_kernel(&kernel);
    }
}

#[derive(Debug, Clone, Copy)]
struct TestCase<'a> {
    alpha: f32,
    a: MatRef<'a, f32>,
    b: MatRef<'a, f32>,
    beta: f32,
    c: MatRef<'a, f32>,
    pack_sizes: PackSizes,
    expect: &'a [f32],
}

impl<'a> TestCase<'a> {
    fn test_kernel(&self, kernel: &impl Kernel<Scalar = f32>) {
        let param = self;
        let mut actual = param.c.as_slice().to_vec();
        let mut actual = MatMut::from_parts(
            param.c.nrows(),
            param.c.ncols(),
            &mut actual,
            param.c.row_stride(),
            param.c.col_stride(),
        )
        .unwrap();
        let mut packing_buf = vec![0f32; param.pack_sizes.buf_len()];
        kernel.gemm(
            param.alpha,
            param.a,
            param.b,
            param.beta,
            &mut actual,
            param.pack_sizes,
            &mut packing_buf,
        );
        assert_eq!(actual.as_slice(), param.expect, "{:?}", self);
    }
}
