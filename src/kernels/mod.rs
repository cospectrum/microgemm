mod generic;
#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "wasm32")]
mod wasm;

pub use generic::{
    Generic16x16Kernel, Generic2x2Kernel, Generic32x32Kernel, Generic4x4Kernel, Generic8x8Kernel,
};
#[cfg(target_arch = "aarch64")]
pub use neon::NeonKernel;
#[cfg(target_arch = "wasm32")]
pub use wasm::WasmSimd128Kernel;
