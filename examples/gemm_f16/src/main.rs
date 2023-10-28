use half::f16;
use microgemm as mg;
use microgemm::Kernel as _;

fn main() {
    let to_f16 = |v: &[f32]| -> Vec<f16> {
        v.iter().copied().map(f16::from_f32).collect()
    };

    let a = [1., 2., 3., 4., 5., 6.];
    let a = to_f16(&a);
    let a = mg::MatRef::new(2, 3, &a, mg::Layout::RowMajor);

    let b = [10., 11., 20., 21., 30., 31.];
    let b = to_f16(&b);
    let b = mg::MatRef::new(3, 2, &b, mg::Layout::RowMajor);

    let mut c = [f16::ZERO; 2 * 2];
    let mut c = mg::MatMut::new(2, 2, c.as_mut(), mg::Layout::RowMajor);

    let kernel = mg::kernels::Generic2x2Kernel::<f16>::new();
    let pack_sizes = mg::PackSizes {
        mc: 2,
        kc: 3,
        nc: 2,
    };
    let mut packing_buf = vec![f16::ZERO; pack_sizes.buf_len()];

    kernel.gemm(
        f16::ONE,
        &a,
        &b,
        f16::ZERO,
        &mut c,
        &pack_sizes,
        &mut packing_buf,
    );
    println!("{:?}", c.as_slice());
}
