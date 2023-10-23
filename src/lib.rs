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

pub use generic_array::typenum;
pub use num_traits::{One, Zero};

pub use gemm::gemm_with_kernel;
pub use kernel::Kernel;
pub use mat::{Layout, MatMut, MatRef};
pub use packing::PackSizes;
