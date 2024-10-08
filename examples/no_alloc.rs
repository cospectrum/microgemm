use microgemm::{kernels::GenericKernel4x4, Kernel as _, MatMut, MatRef, PackSizes};

const M: usize = 15;
const K: usize = 16;
const N: usize = 22;

const PACK_SIZES: PackSizes = PackSizes {
    mc: M,
    kc: K,
    nc: N,
};

fn main() {
    let kernel = GenericKernel4x4::<f32>::new();

    let mut packing_buf = [0.0; PACK_SIZES.buf_len()];
    let (alpha, beta) = (2.0, -3.0);

    let a = [3.0; M * K];
    let b = [4.0; K * N];
    let mut c = [5.0; M * N];

    let a = MatRef::row_major(M, K, &a);
    let b = MatRef::row_major(K, N, &b);
    let mut c = MatMut::row_major(M, N, &mut c);

    kernel.gemm(alpha, a, b, beta, &mut c, PACK_SIZES, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
