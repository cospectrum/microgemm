#![no_std]
#![no_main]

use core::panic::PanicInfo;

use mg::Kernel;
use microgemm as mg;
use microgemm::kernels::Generic8x8Kernel;

const KERNEL: Generic8x8Kernel<f32> = Generic8x8Kernel::<f32>::new();
const M: usize = 2;
const K: usize = 3;
const N: usize = 4;

const PACK_SIZES: &mg::PackSizes = &mg::PackSizes {
    mc: Generic8x8Kernel::<f32>::MR,
    kc: K,
    nc: Generic8x8Kernel::<f32>::NR,
};

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let mut buf = [0f32; PACK_SIZES.buf_len()];
    let alpha = 0f32;
    let beta = 0f32;

    let a = [0f32; M * K];
    let b = [0f32; K * N];
    let mut c = [0f32; M * N];

    let a = mg::MatRef::new(M, K, &a, mg::Layout::RowMajor);
    let b = mg::MatRef::new(K, N, &b, mg::Layout::RowMajor);
    let mut c = mg::MatMut::new(M, N, &mut c, mg::Layout::RowMajor);
    KERNEL.gemm(alpha, &a, &b, beta, &mut c, PACK_SIZES, &mut buf);

    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
