use microgemm as mg;
use microgemm::Kernel as _;

#[test]
fn main() {
    let kernel = mg::generic4x4_kernel::<f32>();
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
