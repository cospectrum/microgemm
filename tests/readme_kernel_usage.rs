use microgemm as mg;
use microgemm::Kernel as _;

#[test]
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
