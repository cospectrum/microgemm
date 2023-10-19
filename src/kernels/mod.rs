#[cfg(target_arch = "aarch64")]
mod aarch64;
mod generic;

#[cfg(target_arch = "aarch64")]
pub use aarch64::Aarch64Kernel;
pub use generic::{generic4x4_kernel, Generic4x4Kernel};