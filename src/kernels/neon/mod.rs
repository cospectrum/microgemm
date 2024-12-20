mod ker4x4;
mod ker8x8;

#[cfg(any(kani, miri))]
mod simd_mock;

#[cfg(target_arch = "aarch64")]
mod simd {
    #[cfg(any(kani, miri))]
    pub use super::simd_mock::{
        vaddq_f32, vfmaq_f32, vfmaq_laneq_f32, vld1q_f32, vmovq_n_f32, vmulq_n_f32, vst1q_f32,
    };
    #[cfg(not(any(kani, miri)))]
    pub use core::arch::aarch64::{
        vaddq_f32, vfmaq_f32, vfmaq_laneq_f32, vld1q_f32, vmovq_n_f32, vmulq_n_f32, vst1q_f32,
    };
}

pub use ker4x4::NeonKernel4x4;
pub use ker8x8::NeonKernel8x8;

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::utils::{assert_approx_eq, proptest_kernel, ProptestKernelCfg};
    use proptest::{strategy::Strategy, test_runner::TestCaseResult};
    use std::arch::is_aarch64_feature_detected;

    fn neon_kernel_8x8<T>() -> NeonKernel8x8<T> {
        if is_aarch64_feature_detected!("neon") {
            unsafe { NeonKernel8x8::new() }
        } else {
            panic!("neon feature is not supported");
        }
    }
    fn neon_kernel_4x4<T>() -> NeonKernel4x4<T> {
        if is_aarch64_feature_detected!("neon") {
            unsafe { NeonKernel4x4::new() }
        } else {
            panic!("neon feature is not supported");
        }
    }

    fn cfg_f32() -> ProptestKernelCfg<f32> {
        let cmp = |expect: &[f32], got: &[f32]| -> TestCaseResult {
            let eps = 75.0 * f32::EPSILON;
            assert_approx_eq(expect, got, eps);
            Ok(())
        };
        let dim = 80;
        ProptestKernelCfg::default()
            .with_cmp(cmp)
            .with_scalar((-1f32..1.0).boxed())
            .with_max_matrix_dim(dim)
            .with_max_pack_dim(2 * dim + 1)
    }

    #[test]
    fn proptest_neon_kernel_8x8_f32() {
        proptest_kernel(&neon_kernel_8x8(), cfg_f32()).unwrap();
    }
    #[test]
    fn proptest_neon_kernel_4x4_f32() {
        proptest_kernel(&neon_kernel_4x4(), cfg_f32()).unwrap();
    }
}
