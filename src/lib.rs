#![no_std]

#[cfg(test)]
#[macro_use]
extern crate approx;

#[cfg(test)]
#[macro_use]
extern crate std;

#[cfg(test)]
mod std_prelude {
    pub use std::prelude::rust_2021::*;
}

mod gemm;
mod kernel;
mod mat;

pub(crate) mod copying;
pub(crate) mod packing;

pub mod kernels;
pub mod utils;

pub use gemm::gemm_with_kernel;
pub use kernel::Kernel;
pub use mat::*;
pub use num_traits::{One, Zero};
pub use packing::PackSizes;
