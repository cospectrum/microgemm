#![allow(unused)]
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
        todo!();
    }
}
