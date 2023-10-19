use mg::Kernel;
use microgemm as mg;
use microgemm::kernels::Generic4x4Kernel;

const M: usize = 10;
const K: usize = 15;
const N: usize = 20;

#[test]
fn main() {
    const KERNEL: Generic4x4Kernel<f32> = Generic4x4Kernel::<f32>::new();

    const PACK_SIZES: &mg::PackSizes = &mg::PackSizes {
        mc: Generic4x4Kernel::<f32>::MR,
        kc: 15,
        nc: Generic4x4Kernel::<f32>::NR,
    };
    const BUF_LEN: usize = PACK_SIZES.buf_len(&KERNEL);
    let mut buf = [0f32; BUF_LEN];

    let alpha = 0f32;
    let beta = 0f32;

    let a = [0f32; M * K];
    let b = [0f32; K * N];
    let mut c = [0f32; M * N];

    let a = mg::MatRef::new(M, K, &a, mg::Layout::RowMajor);
    let b = mg::MatRef::new(K, N, &b, mg::Layout::RowMajor);
    let mut c = mg::MatMut::new(M, N, &mut c, mg::Layout::RowMajor);

    KERNEL.gemm(alpha, &a, &b, beta, &mut c, PACK_SIZES, &mut buf);
}
