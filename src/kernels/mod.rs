#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64;
mod generic;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub use aarch64::Aarch64Kernel;
pub use generic::*;
