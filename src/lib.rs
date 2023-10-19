#[cfg(test)]
#[macro_use]
extern crate approx;

mod gemm;
mod kernel;
mod mat;

pub mod select;

pub(crate) mod copying;
pub(crate) mod packing;

pub mod kernels;
pub mod utils;

pub use gemm::gemm_with_kernel;
pub use kernel::Kernel;
pub use mat::*;
pub use num_traits::{One, Zero};
pub use packing::PackSizes;

pub use kernels::{generic2x2_kernel, generic4x4_kernel};
