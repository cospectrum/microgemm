#[cfg(target_arch = "wasm32")]
mod microkernel;

use core::marker::PhantomData;

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
    /// ```no_run
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
