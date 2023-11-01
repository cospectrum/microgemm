use microgemm as mg;
use microgemm::{kernels::Generic4x4Kernel, Kernel as _};

const M: usize = 15;
const K: usize = 16;
const N: usize = 22;

const KERNEL: Generic4x4Kernel<f32> = Generic4x4Kernel::<f32>::new();

const PACK_SIZES: &mg::PackSizes = &mg::PackSizes {
    mc: 2 * Generic4x4Kernel::<f32>::MR,
    kc: 16,
    nc: 3 * Generic4x4Kernel::<f32>::NR,
};

fn main() {
    let mut packing_buf = [0.0; PACK_SIZES.buf_len()];

    let alpha = 2.0;
    let beta = -3.0;

    let a = [3.0; M * K];
    let b = [4.0; K * N];
    let mut c = [5.0; M * N];

    let a = mg::MatRef::new(M, K, &a, mg::Layout::RowMajor);
    let b = mg::MatRef::new(K, N, &b, mg::Layout::RowMajor);
    let mut c = mg::MatMut::new(M, N, &mut c, mg::Layout::RowMajor);

    KERNEL.gemm(alpha, &a, &b, beta, &mut c, PACK_SIZES, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
