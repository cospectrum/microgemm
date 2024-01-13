use microgemm::{kernels::Generic4x4Kernel, Kernel as _, MatMut, MatRef, PackSizes};

const M: usize = 15;
const K: usize = 16;
const N: usize = 22;

const KERNEL: Generic4x4Kernel<f32> = Generic4x4Kernel::<f32>::new();

const PACK_SIZES: &PackSizes = &PackSizes {
    mc: 2 * Generic4x4Kernel::<f32>::MR,
    kc: 16,
    nc: 3 * Generic4x4Kernel::<f32>::NR,
};

fn main() {
    let mut packing_buf = [0.0; PACK_SIZES.buf_len()];
    let (alpha, beta) = (2.0, -3.0);

    let a = [3.0; M * K];
    let b = [4.0; K * N];
    let mut c = [5.0; M * N];

    let a = MatRef::row_major(M, K, &a);
    let b = MatRef::row_major(K, N, &b);
    let mut c = MatMut::row_major(M, N, &mut c);

    KERNEL.gemm(alpha, &a, &b, beta, &mut c, PACK_SIZES, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
