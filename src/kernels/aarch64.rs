use crate::{Kernel, MatMut, MatRef};
use core::{
    arch::aarch64::{float32x4_t, vfmaq_laneq_f32, vld1q_f32, vst1q_f32},
    marker::PhantomData,
};

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
        debug_assert_eq!(dst.nrows(), 4);
        debug_assert_eq!(dst.ncols(), 4);
        assert_eq!(lhs.as_slice().len(), rhs.as_slice().len());

        let f32x4 = |v: &[f32]| -> float32x4_t {
            debug_assert_eq!(v.len(), 4);
            unsafe { vld1q_f32(v.as_ptr()) }
        };
        let load_f32x4 = |v: float32x4_t| -> [f32; 4] {
            let mut array = [0f32; 4];
            unsafe { vst1q_f32(array.as_mut_ptr(), v) };
            array
        };

        let mut col0 = f32x4(&[0f32; 4]);
        let mut col1 = f32x4(&[0f32; 4]);
        let mut col2 = f32x4(&[0f32; 4]);
        let mut col3 = f32x4(&[0f32; 4]);

        let left = lhs.as_slice().chunks_exact(4).map(f32x4);
        let right = rhs.as_slice().chunks_exact(4).map(f32x4);

        for (a, b) in left.zip(right) {
            col0 = unsafe { vfmaq_laneq_f32::<0>(col0, a, b) };
            col1 = unsafe { vfmaq_laneq_f32::<1>(col1, a, b) };
            col2 = unsafe { vfmaq_laneq_f32::<2>(col2, a, b) };
            col3 = unsafe { vfmaq_laneq_f32::<3>(col3, a, b) };
        }

        let col0 = load_f32x4(col0);
        let col1 = load_f32x4(col1);
        let col2 = load_f32x4(col2);
        let col3 = load_f32x4(col3);

        let mut write_to = |ncol: usize, from: &[f32; 4]| {
            for (i, &val) in from.iter().enumerate() {
                let to = dst.get_mut(i, ncol);
                *to = alpha * val + beta * *to;
            }
        };
        write_to(0, &col0);
        write_to(1, &col1);
        write_to(2, &col2);
        write_to(3, &col3);
    }
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
            let eps = 40.0 * f32::EPSILON;
            assert_relative_eq!(expect, got, epsilon = eps);
        };

        for _ in 0..40 {
            let mc = rng.gen_range(1..20) * 4;
            let nc = rng.gen_range(1..20) * 4;
            let scalar = || rng.gen_range(-1.0..1.0);

            cmp_kernels_with_random_data(generic_kernel, aarch64_kernel, scalar, cmp, mc, nc);
        }
    }
}
