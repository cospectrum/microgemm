use super::NeonKernel4x4;
use crate::{kernels::dbg_check_microkernel_inputs, typenum::U4, Kernel, MatMut, MatRef};
use core::arch::aarch64::{
    vaddq_f32, vfmaq_laneq_f32, vld1q_f32, vmovq_n_f32, vmulq_n_f32, vst1q_f32,
};

impl Kernel for NeonKernel4x4<f32> {
    type Scalar = f32;
    type Mr = U4;
    type Nr = U4;

    fn microkernel(
        &self,
        alpha: f32,
        lhs: MatRef<f32>,
        rhs: MatRef<f32>,
        beta: f32,
        dst: &mut MatMut<f32>,
    ) {
        dbg_check_microkernel_inputs(self, lhs, rhs, dst);
        let kc = lhs.ncols();
        neon_4x4_microkernel_f32(
            kc,
            alpha,
            lhs.as_slice(),
            rhs.as_slice(),
            beta,
            dst.as_mut_slice(),
        );
    }
}

fn neon_4x4_microkernel_f32(
    kc: usize,
    alpha: f32,
    lhs: &[f32],
    rhs: &[f32],
    beta: f32,
    dst_colmajor: &mut [f32],
) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), 4 * kc);

    unsafe { inner(kc, alpha, lhs.as_ptr(), rhs.as_ptr(), beta, dst_colmajor) };

    unsafe fn inner(
        kc: usize,
        alpha: f32,
        mut left: *const f32,
        mut right: *const f32,
        beta: f32,
        dst: &mut [f32],
    ) {
        let mut cols0 = [vmovq_n_f32(0f32); 4];
        let mut cols1 = [vmovq_n_f32(0f32); 4];
        let mut cols2 = [vmovq_n_f32(0f32); 4];
        let mut cols3 = [vmovq_n_f32(0f32); 4];

        macro_rules! accum_lane {
            ($cols:ident, $a:ident, $b:ident) => {
                $cols[0] = vfmaq_laneq_f32::<0>($cols[0], $a, $b);
                $cols[1] = vfmaq_laneq_f32::<1>($cols[1], $a, $b);
                $cols[2] = vfmaq_laneq_f32::<2>($cols[2], $a, $b);
                $cols[3] = vfmaq_laneq_f32::<3>($cols[3], $a, $b);
            };
        }

        for _ in 0..kc / 4 {
            let a = vld1q_f32(left);
            let b = vld1q_f32(right);
            accum_lane!(cols0, a, b);

            let a = vld1q_f32(left.add(4));
            let b = vld1q_f32(right.add(4));
            accum_lane!(cols1, a, b);

            let a = vld1q_f32(left.add(8));
            let b = vld1q_f32(right.add(8));
            accum_lane!(cols2, a, b);

            let a = vld1q_f32(left.add(12));
            let b = vld1q_f32(right.add(12));
            accum_lane!(cols3, a, b);

            left = left.add(16);
            right = right.add(16);
        }

        for _ in 0..kc % 4 {
            let a = vld1q_f32(left);
            let b = vld1q_f32(right);
            accum_lane!(cols0, a, b);

            left = left.add(4);
            right = right.add(4);
        }

        for row in 0..4 {
            let sum = vaddq_f32(
                vaddq_f32(cols0[row], cols1[row]),
                vaddq_f32(cols2[row], cols3[row]),
            );
            cols0[row] = vmulq_n_f32(sum, alpha);
        }

        let it = dst.chunks_exact_mut(4).zip(cols0);
        if beta == 0f32 {
            it.for_each(|(to, from)| {
                vst1q_f32(to.as_mut_ptr(), from);
            });
        } else {
            it.for_each(|(to, from)| {
                let mut tmp = [0f32; 4];
                vst1q_f32(tmp.as_mut_ptr(), from);
                to.iter_mut().zip(tmp).for_each(|(y, x)| {
                    *y = x + beta * *y;
                });
            });
        }
    }
}
