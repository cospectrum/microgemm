mod generic;
mod neon;
mod wasm;

pub use generic::{
    Generic16x16Kernel, Generic2x2Kernel, Generic32x32Kernel, Generic4x4Kernel, Generic8x8Kernel,
};
pub use neon::NeonKernel;
pub use wasm::WasmSimd128Kernel;
