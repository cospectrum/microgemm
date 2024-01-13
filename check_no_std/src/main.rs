#![no_std]
#![no_main]

use core::panic::PanicInfo;

use microgemm::{kernels::Generic8x8Kernel, Kernel as _, PackSizes, MatRef, MatMut};

const KERNEL: Generic8x8Kernel<f32> = Generic8x8Kernel::<f32>::new();
const M: usize = 2;
const K: usize = 3;
const N: usize = 4;

const PACK_SIZES: &PackSizes = &PackSizes {
    mc: Generic8x8Kernel::<f32>::MR,
    kc: K,
    nc: Generic8x8Kernel::<f32>::NR,
};

#[no_mangle]
pub extern "C" fn _start() -> ! {
    let mut buf = [0f32; PACK_SIZES.buf_len()];
    let (alpha, beta) = (0f32, 0f32);

    let a = [0f32; M * K];
    let b = [0f32; K * N];
    let mut c = [0f32; M * N];

    let a = MatRef::row_major(M, K, &a);
    let b = MatRef::row_major(K, N, &b);
    let mut c = MatMut::row_major(M, N, &mut c);
    KERNEL.gemm(alpha, &a, &b, beta, &mut c, PACK_SIZES, &mut buf);

    loop {}
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
