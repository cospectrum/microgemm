#[cfg(target_arch = "aarch64")]
mod microkernel;

use core::marker::PhantomData;

/// Available only for the `aarch64` target.
#[derive(Debug, Clone, Copy)]
pub struct NeonKernel<T> {
    marker: PhantomData<T>,
}

impl<T> NeonKernel<T> {
    /// # Safety
    ///
    /// The caller must ensure that the created kernel will only be used in an
    /// environment with `neon` support.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use microgemm::kernels::NeonKernel;
    ///
    /// let kernel = if cfg!(target_feature = "neon") {
    ///     unsafe { NeonKernel::<f32>::new() }
    /// } else {
    ///     panic!("neon target feature is not enabled");
    /// };
    /// ```
    pub const unsafe fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    fn _check_new_neon() {
        use crate::kernels::NeonKernel;

        let _kernel = if cfg!(target_feature = "neon") {
            unsafe { NeonKernel::<f32>::new() }
        } else {
            panic!("neon target feature is not enabled");
        };
    }
}
