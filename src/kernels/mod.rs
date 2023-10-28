mod generic;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm;

pub use generic::{
    Generic16x16Kernel, Generic2x2Kernel, Generic32x32Kernel, Generic4x4Kernel, Generic8x8Kernel,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::NeonKernel;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub use wasm::WasmSimd128Kernel;
