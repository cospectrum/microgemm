use crate::{Kernel, MatMut, MatRef};
use core::arch::aarch64::{
    float32x4_t, vaddq_f32, vfmaq_laneq_f32, vld1q_f32, vmovq_n_f32, vmulq_n_f32, vst1q_f32,
};
use core::marker::PhantomData;

#[derive(Debug, Clone, Copy, Default)]
pub struct Aarch64Kernel<T> {
    marker: PhantomData<T>,
}

impl<T> Aarch64Kernel<T> {
    pub const fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl Kernel for Aarch64Kernel<f32> {
    type Elem = f32;

    const MR: usize = 4;
    const NR: usize = 4;

    fn microkernel(
        &self,
        alpha: f32,
        lhs: &MatRef<f32>,
        rhs: &MatRef<f32>,
        beta: f32,
        dst: &mut MatMut<f32>,
    ) {
        let kc = lhs.ncols();
        unsafe {
            neon_4x4_microkernel_f32(
                kc,
                alpha,
                lhs.as_slice(),
                rhs.as_slice(),
                beta,
                dst.as_mut_slice(),
            )
        };
    }
}

#[target_feature(enable = "neon")]
unsafe fn neon_4x4_microkernel_f32(
    kc: usize,
    alpha: f32,
    lhs: &[f32],
    rhs: &[f32],
    beta: f32,
    dst_colmajor: &mut [f32],
) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), 4 * kc);

    let mut cols0 = [vmovq_n_f32(0f32); 4];
    let mut cols1 = [vmovq_n_f32(0f32); 4];
    let mut cols2 = [vmovq_n_f32(0f32); 4];
    let mut cols3 = [vmovq_n_f32(0f32); 4];

    let accum_plane = |cols: &mut [float32x4_t; 4], a, b| {
        cols[0] = vfmaq_laneq_f32::<0>(cols[0], a, b);
        cols[1] = vfmaq_laneq_f32::<1>(cols[1], a, b);
        cols[2] = vfmaq_laneq_f32::<2>(cols[2], a, b);
        cols[3] = vfmaq_laneq_f32::<3>(cols[3], a, b);
    };

    let mut left = lhs.as_ptr();
    let mut right = rhs.as_ptr();

    for _ in 0..kc / 4 {
        let a = vld1q_f32(left);
        let b = vld1q_f32(right);
        accum_plane(&mut cols0, a, b);

        let a = vld1q_f32(left.add(4));
        let b = vld1q_f32(right.add(4));
        accum_plane(&mut cols1, a, b);

        let a = vld1q_f32(left.add(8));
        let b = vld1q_f32(right.add(8));
        accum_plane(&mut cols2, a, b);

        let a = vld1q_f32(left.add(12));
        let b = vld1q_f32(right.add(12));
        accum_plane(&mut cols3, a, b);

        left = left.add(4 * 4);
        right = right.add(4 * 4);
    }

    for _ in 0..kc % 4 {
        let a = vld1q_f32(left);
        let b = vld1q_f32(right);
        accum_plane(&mut cols0, a, b);

        left = left.add(4);
        right = right.add(4);
    }

    for i in 0..4 {
        cols0[i] = vaddq_f32(vaddq_f32(cols0[i], cols1[i]), vaddq_f32(cols2[i], cols3[i]));
        cols0[i] = vmulq_n_f32(cols0[i], alpha);
    }

    dst_colmajor
        .chunks_exact_mut(4)
        .zip(cols0)
        .for_each(|(dst, src)| {
            let mut array = [0f32; 4];
            vst1q_f32(array.as_mut_ptr(), src);
            dst.iter_mut().zip(array).for_each(|(y, x)| {
                *y = x + beta * *y;
            });
        });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::Generic4x4Kernel;
    use crate::utils::*;

    use rand::{thread_rng, Rng};

    #[test]
    fn test_aarch64_kernel_f32() {
        let generic_kernel = &Generic4x4Kernel::<f32>::new();
        let aarch64_kernel = &Aarch64Kernel::<f32>::new();

        let mut rng = thread_rng();
        let cmp = |expect: &[f32], got: &[f32]| {
            let eps = 65.0 * f32::EPSILON;
            assert_relative_eq!(expect, got, epsilon = eps);
        };

        for _ in 0..60 {
            let mc = rng.gen_range(1..20) * 4;
            let nc = rng.gen_range(1..20) * 4;
            let scalar = || rng.gen_range(-1.0..1.0);

            cmp_kernels_with_random_data(generic_kernel, aarch64_kernel, scalar, cmp, mc, nc);
        }
    }
}
