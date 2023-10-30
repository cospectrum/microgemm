#[cfg(target_arch = "wasm32")]
mod microkernel;

use core::marker::PhantomData;

/// Available only for the `wasm32` target.
#[derive(Debug, Clone, Copy)]
pub struct WasmSimd128Kernel<T> {
    marker: PhantomData<T>,
}

impl<T> WasmSimd128Kernel<T> {
    /// # Safety
    ///
    /// The caller must ensure that the created kernel will only be used in an
    /// environment with `simd128` support.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use microgemm::kernels::WasmSimd128Kernel;
    ///
    /// let kernel = if cfg!(target_feature = "simd128") {
    ///     unsafe { WasmSimd128Kernel::<f32>::new() }
    /// } else {
    ///     panic!("simd128 target feature is not enabled");
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
    fn _check_new_wasm() {
        use crate::kernels::WasmSimd128Kernel;

        let _kernel = if cfg!(target_feature = "simd128") {
            unsafe { WasmSimd128Kernel::<f32>::new() }
        } else {
            panic!("simd128 target feature is not enabled");
        };
    }
}
