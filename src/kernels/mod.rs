mod generic;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

pub use generic::{
    Generic16x16Kernel, Generic2x2Kernel, Generic32x32Kernel, Generic4x4Kernel, Generic8x8Kernel,
};
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::NeonKernel;
