use microgemm::{kernels::Generic8x8Kernel, Kernel as _, MatMut, MatRef, PackSizes};

#[test]
fn test_main() {
    let kernel = Generic8x8Kernel::<f32>::new();
    assert_eq!(kernel.mr(), 8);
    assert_eq!(kernel.nr(), 8);

    let pack_sizes = PackSizes {
        mc: 5 * kernel.mr(), // MC must be divisible by MR
        kc: 190,
        nc: 9 * kernel.nr(), // NC must be divisible by NR
    };
    let mut packing_buf = vec![0.0; pack_sizes.buf_len()];

    let (alpha, beta) = (2.0, -3.0);
    let (m, k, n) = (100, 380, 250);

    let a = vec![2.0; m * k];
    let b = vec![3.0; k * n];
    let mut c = vec![4.0; m * n];

    let a = MatRef::row_major(m, k, &a);
    let b = MatRef::row_major(k, n, &b);
    let mut c = MatMut::row_major(m, n, &mut c);

    // c <- alpha a b + beta c
    kernel.gemm(alpha, &a, &b, beta, &mut c, &pack_sizes, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
