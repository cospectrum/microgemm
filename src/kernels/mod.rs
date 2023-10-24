mod generic;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;

pub use generic::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use neon::NeonKernel;
