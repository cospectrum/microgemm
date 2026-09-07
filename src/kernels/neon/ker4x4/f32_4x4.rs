use super::NeonKernel4x4;
use crate::{kernels::dbg_check_microkernel_inputs, typenum::U4, Kernel, MatMut, MatRef};

use super::super::simd::*;

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
    const DIM: usize = 4;
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), DIM.checked_mul(kc).unwrap());
    assert_eq!(dst_colmajor.len(), DIM * DIM);

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
        for (to, from) in it {
            let mut tmp = [0f32; 4];
            vst1q_f32(tmp.as_mut_ptr(), from);
            for (y, x) in to.iter_mut().zip(tmp) {
                *y = scale_add(x, beta, *y);
            }
        }
    }
}

// Kept separate so memory-safety proofs can abstract arithmetic without
// replacing any pointer access, loop, lane selection, or bounds check.
#[inline]
fn scale_add(x: f32, beta: f32, y: f32) -> f32 {
    x + beta * y
}

#[cfg(test)]
mod tests {
    use super::neon_4x4_microkernel_f32;
    use crate::std_prelude::*;

    #[test]
    fn unrolled_and_remainder_loops_preserve_guards() {
        // Every remainder, the empty dot product, and one/two unrolled groups.
        for kc in 0..=9 {
            let lhs: Vec<f32> = (0..4 * kc).map(|i| (i % 7) as f32 - 3.0).collect();
            let rhs: Vec<f32> = (0..4 * kc).map(|i| (i % 11) as f32 - 5.0).collect();
            let mut dst = [12345.0; 18];
            dst[1..17].fill(2.0);
            neon_4x4_microkernel_f32(kc, 2.0, &lhs, &rhs, -3.0, &mut dst[1..17]);
            assert_eq!(dst[0], 12345.0);
            assert_eq!(dst[17], 12345.0);
            for col in 0..4 {
                for row in 0..4 {
                    let sum: f32 = (0..kc).map(|k| lhs[4 * k + row] * rhs[4 * k + col]).sum();
                    assert_eq!(
                        dst[1 + 4 * col + row],
                        2.0 * sum - 6.0,
                        "kc={kc}, row={row}, col={col}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn rejects_mismatched_panels() {
        neon_4x4_microkernel_f32(1, 1.0, &[0.0; 4], &[0.0; 3], 0.0, &mut [0.0; 16]);
    }

    #[test]
    #[should_panic]
    fn rejects_panels_short_for_depth() {
        neon_4x4_microkernel_f32(2, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut [0.0; 16]);
    }

    #[test]
    #[should_panic]
    fn rejects_short_destination() {
        neon_4x4_microkernel_f32(1, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut [0.0; 15]);
    }

    #[test]
    #[should_panic]
    fn rejects_panel_length_overflow() {
        neon_4x4_microkernel_f32(usize::MAX / 4 + 1, 1.0, &[], &[], 0.0, &mut [0.0; 16]);
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;
    // These imports are referenced by Kani attributes, not ordinary Rust calls.
    #[allow(unused_imports)]
    use crate::kernels::neon::simd_mock::{self, memory_arithmetic as arithmetic};

    // Each depth has its own proof: constant extents let CBMC simplify pointer
    // bounds before solving. The allocation ends exactly at the panel boundary,
    // so a read beyond a panel cannot hide in unused backing-array capacity.
    fn memory_case<const LEN: usize>() {
        let kc = (LEN - 1) / 4;
        let lhs: [f32; LEN] = kani::any();
        let rhs: [f32; LEN] = kani::any();
        let mut dst: [f32; 18] = kani::any();
        let first = dst[0].to_bits();
        let last = dst[17].to_bits();
        neon_4x4_microkernel_f32(
            kc,
            kani::any(),
            &lhs[1..],
            &rhs[1..],
            kani::any(),
            &mut dst[1..17],
        );
        assert_eq!(dst[0].to_bits(), first);
        assert_eq!(dst[17].to_bits(), last);
    }

    macro_rules! memory_proof {
        ($name:ident, $kc:literal) => {
            #[kani::proof]
            #[kani::unwind(5)]
            #[kani::stub(simd_mock::vaddq_f32, arithmetic::add)]
            #[kani::stub(simd_mock::vmulq_n_f32, arithmetic::scale)]
            #[kani::stub(simd_mock::fma_scalar, arithmetic::scalar_fma)]
            #[kani::stub(simd_mock::vfmaq_f32, arithmetic::fma)]
            #[kani::stub(super::scale_add, arbitrary_scale_add)]
            fn $name() {
                memory_case::<{ 4 * $kc + 1 }>();
            }
        };
    }

    // Zero depth, two unrolled groups, and every remainder after each group.
    memory_proof!(memory_depth_0, 0);
    memory_proof!(memory_depth_1, 1);
    memory_proof!(memory_depth_2, 2);
    memory_proof!(memory_depth_3, 3);
    memory_proof!(memory_depth_4, 4);
    memory_proof!(memory_depth_5, 5);
    memory_proof!(memory_depth_6, 6);
    memory_proof!(memory_depth_7, 7);
    memory_proof!(memory_depth_8, 8);
    memory_proof!(memory_depth_9, 9);

    /// No arithmetic stubs: all sign patterns over {-1, 1}. These operations
    /// are exact in both the scalar mock and real fused NEON arithmetic.
    #[kani::proof]
    #[kani::unwind(5)]
    fn exact_single_outer_product() {
        let left: [bool; 4] = kani::any();
        let right: [bool; 4] = kani::any();
        let mut lhs = [0.0; 4];
        let mut rhs = [0.0; 4];
        for i in 0..4 {
            lhs[i] = if left[i] { 1.0 } else { -1.0 };
            rhs[i] = if right[i] { 1.0 } else { -1.0 };
        }
        let mut dst = [1.0; 16];
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel4x4::new() };
        kernel.microkernel(
            1.0,
            MatRef::col_major(4, 1, &lhs[..]),
            MatRef::row_major(1, 4, &rhs[..]),
            -1.0,
            &mut MatMut::col_major(4, 4, &mut dst[..]),
        );
        let row: usize = kani::any_where(|&i| i < 4);
        let col: usize = kani::any_where(|&i| i < 4);
        let expected = if left[row] == right[col] { 0.0 } else { -2.0 };
        assert_eq!(dst[col * 4 + row], expected);
    }

    // No arithmetic input restrictions: any f32 result overapproximates the epilogue.
    #[allow(dead_code)]
    fn arbitrary_scale_add(_: f32, _: f32, _: f32) -> f32 {
        kani::any()
    }
}
