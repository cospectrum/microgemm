use microgemm as mg;
use microgemm::{kernels::Generic4x4Kernel, Kernel as _};

const M: usize = 10;
const K: usize = 15;
const N: usize = 20;

const KERNEL: Generic4x4Kernel<f32> = Generic4x4Kernel::<f32>::new();

const PACK_SIZES: &mg::PackSizes = &mg::PackSizes {
    mc: Generic4x4Kernel::<f32>::MR,
    kc: 15,
    nc: Generic4x4Kernel::<f32>::NR,
};

fn main() {
    let mut packing_buf = [0f32; PACK_SIZES.buf_len()];

    let alpha = 2f32;
    let beta = -3f32;

    let a = [3f32; M * K];
    let b = [4f32; K * N];
    let mut c = [5f32; M * N];

    let a = mg::MatRef::new(M, K, &a, mg::Layout::RowMajor);
    let b = mg::MatRef::new(K, N, &b, mg::Layout::RowMajor);
    let mut c = mg::MatMut::new(M, N, &mut c, mg::Layout::RowMajor);

    KERNEL.gemm(alpha, &a, &b, beta, &mut c, PACK_SIZES, &mut packing_buf);
    println!("{:?}", c.as_slice());
}
