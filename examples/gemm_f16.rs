use half::f16; // with `num-traits` feature
use microgemm::{kernels::GenericKernel2x2, Kernel as _, MatMut, MatRef, PackSizes};

fn main() {
    let to_f16 = |v: &[f32]| -> Vec<f16> { v.iter().copied().map(f16::from_f32).collect() };

    let a = [1., 2., 3., 4., 5., 6.];
    let a = to_f16(&a);
    let a = MatRef::row_major(2, 3, &a);

    let b = [10., 11., 20., 21., 30., 31.];
    let b = to_f16(&b);
    let b = MatRef::row_major(3, 2, &b);

    let mut c = [f16::ZERO; 2 * 2];
    let mut c = MatMut::row_major(2, 2, c.as_mut());

    let kernel = GenericKernel2x2::<f16>::new();
    let pack_sizes = PackSizes {
        mc: 2,
        kc: 3,
        nc: 2,
    };
    let mut packing_buf = vec![f16::ZERO; pack_sizes.buf_len()];

    let (alpha, beta) = (f16::ONE, f16::ZERO);

    kernel.gemm(alpha, a, b, beta, &mut c, pack_sizes, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
