#[cfg(target_arch = "aarch64")]
mod f32_8x8;

use super::simd;
use core::marker::PhantomData;

/// Available only for the `aarch64` target.
#[derive(Debug, Clone, Copy)]
pub struct NeonKernel8x8<T> {
    marker: PhantomData<T>,
}

impl<T> NeonKernel8x8<T> {
    /// # Safety
    ///
    /// The caller must ensure that the created kernel will only be used in an
    /// environment with `neon` support.
    ///
    /// # Examples
    ///
    /// ```
    /// use microgemm::kernels::NeonKernel8x8;
    ///
    /// let kernel = if cfg!(target_feature = "neon") {
    ///     unsafe { NeonKernel8x8::<f32>::new() }
    /// } else {
    ///     panic!("neon target feature is not enabled");
    /// };
    /// ```
    #[cfg(not(doctest))]
    pub const unsafe fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    fn _check_new_neon() {
        use crate::kernels::NeonKernel8x8;

        let _kernel = if cfg!(target_feature = "neon") {
            unsafe { NeonKernel8x8::<f32>::new() }
        } else {
            panic!("neon target feature is not enabled");
        };
    }
}
