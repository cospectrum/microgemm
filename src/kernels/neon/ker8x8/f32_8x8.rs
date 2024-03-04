#![allow(unused_imports)]
use super::NeonKernel8x8;
use crate::{kernels::dbg_check_microkernel_inputs, typenum::U8, Kernel, MatMut, MatRef};
use core::arch::aarch64::{
    vaddq_f32, vfmaq_laneq_f32, vld1q_f32, vmovq_n_f32, vmulq_n_f32, vst1q_f32,
};

impl Kernel for NeonKernel8x8<f32> {
    type Scalar = f32;
    type Mr = U8;
    type Nr = U8;

    fn microkernel(
        &self,
        alpha: f32,
        lhs: &MatRef<f32>,
        rhs: &MatRef<f32>,
        beta: f32,
        dst: &mut MatMut<f32>,
    ) {
        dbg_check_microkernel_inputs(self, lhs, rhs, dst);
        let kc = lhs.ncols();
        neon_8x8_microkernel_f32(
            kc,
            alpha,
            lhs.as_slice(),
            rhs.as_slice(),
            beta,
            dst.as_mut_slice(),
        );
    }
}

fn neon_8x8_microkernel_f32(
    kc: usize,
    alpha: f32,
    lhs: &[f32],
    rhs: &[f32],
    beta: f32,
    dst_colmajor: &mut [f32],
) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), 8 * kc);

    unsafe { inner(kc, alpha, lhs.as_ptr(), rhs.as_ptr(), beta, dst_colmajor) };

    #[allow(unused_variables)]
    unsafe fn inner(
        kc: usize,
        alpha: f32,
        left: *const f32,
        right: *const f32,
        beta: f32,
        dst: &mut [f32],
    ) {
        todo!();
    }
}
