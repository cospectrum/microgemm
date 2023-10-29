mod generic;
#[cfg(any(target_arch = "aarch64", doc))]
mod neon;
#[cfg(any(target_arch = "wasm32", doc))]
mod wasm;

pub use generic::{
    Generic16x16Kernel, Generic2x2Kernel, Generic32x32Kernel, Generic4x4Kernel, Generic8x8Kernel,
};
#[cfg(any(target_arch = "aarch64", doc))]
pub use neon::NeonKernel;
#[cfg(any(target_arch = "wasm32", doc))]
pub use wasm::WasmSimd128Kernel;
