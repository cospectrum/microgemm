use super::NeonKernel8x8;
use crate::{kernels::dbg_check_microkernel_inputs, typenum::U8, Kernel, MatMut, MatRef};

use super::simd::*;

impl Kernel for NeonKernel8x8<f32> {
    type Scalar = f32;
    type Mr = U8;
    type Nr = U8;

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
    const DIM: usize = 8;
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), DIM.checked_mul(kc).unwrap());
    assert_eq!(dst_colmajor.len(), DIM * DIM);

    unsafe {
        inner(
            kc,
            alpha,
            lhs.as_ptr(),
            rhs.as_ptr(),
            beta,
            dst_colmajor.as_mut_ptr(),
        )
    };

    unsafe fn inner(kc: usize, alpha: f32, a: *const f32, b: *const f32, beta: f32, c: *mut f32) {
        let (mut a, mut b) = (b, a);

        let mut ab11 = [vmovq_n_f32(0f32); 4];
        let mut ab12 = [vmovq_n_f32(0f32); 4];
        let mut ab21 = [vmovq_n_f32(0f32); 4];
        let mut ab22 = [vmovq_n_f32(0f32); 4];

        // Compute
        // ab_ij = a_i * b_j for all i, j
        macro_rules! prod {
            ($dest:ident, $av:expr, $bv:expr) => {
                $dest[0] = vfmaq_laneq_f32::<0>($dest[0], $bv, $av);
                $dest[1] = vfmaq_laneq_f32::<1>($dest[1], $bv, $av);
                $dest[2] = vfmaq_laneq_f32::<2>($dest[2], $bv, $av);
                $dest[3] = vfmaq_laneq_f32::<3>($dest[3], $bv, $av);
            };
        }

        for _ in 0..kc {
            let a1 = vld1q_f32(a);
            let b1 = vld1q_f32(b);
            let a2 = vld1q_f32(a.add(4));
            let b2 = vld1q_f32(b.add(4));

            prod!(ab11, a1, b1);
            prod!(ab12, a1, b2);
            prod!(ab21, a2, b1);
            prod!(ab22, a2, b2);

            a = a.add(8);
            b = b.add(8);
        }

        for i in 0..4 {
            ab11[i] = vmulq_n_f32(ab11[i], alpha);
            ab12[i] = vmulq_n_f32(ab12[i], alpha);
            ab21[i] = vmulq_n_f32(ab21[i], alpha);
            ab22[i] = vmulq_n_f32(ab22[i], alpha);
        }

        macro_rules! c {
            ($i:expr, $j:expr) => {
                c.add(8 * $i + $j)
            };
        }

        if beta != 0f32 {
            let mut c11 = [vmovq_n_f32(0f32); 4];
            let mut c12 = [vmovq_n_f32(0f32); 4];
            let mut c21 = [vmovq_n_f32(0f32); 4];
            let mut c22 = [vmovq_n_f32(0f32); 4];
            for i in 0..4 {
                c11[i] = vld1q_f32(c![i, 0]);
                c12[i] = vld1q_f32(c![i, 4]);
                c21[i] = vld1q_f32(c![i + 4, 0]);
                c22[i] = vld1q_f32(c![i + 4, 4]);
            }

            let betav = vmovq_n_f32(beta);
            for i in 0..4 {
                ab11[i] = vfmaq_f32(ab11[i], c11[i], betav);
                ab12[i] = vfmaq_f32(ab12[i], c12[i], betav);
                ab21[i] = vfmaq_f32(ab21[i], c21[i], betav);
                ab22[i] = vfmaq_f32(ab22[i], c22[i], betav);
            }
        }
        for i in 0..4 {
            vst1q_f32(c![i, 0], ab11[i]);
            vst1q_f32(c![i, 4], ab12[i]);
            vst1q_f32(c![i + 4, 0], ab21[i]);
            vst1q_f32(c![i + 4, 4], ab22[i]);
        }
    }
}
