mod gemm;
mod kernel;
mod mat;

pub(crate) mod copying;
pub(crate) mod packing;

pub mod utils;

pub use gemm::gemm_with_kernel;
pub use kernel::Kernel;
pub use mat::*;
pub use packing::PackSizes;

pub use num_traits::{One, Zero};
