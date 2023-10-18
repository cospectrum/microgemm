#[cfg(test)]
#[macro_use]
extern crate approx;

mod gemm;
mod kernel;
mod mat;
mod select;

pub(crate) mod copying;
pub(crate) mod packing;

pub mod kernels;
pub mod utils;

pub use gemm::gemm_with_kernel;
pub use kernel::Kernel;
pub use mat::*;
pub use packing::PackSizes;
pub use select::select_kernel;

pub use num_traits::{One, Zero};
