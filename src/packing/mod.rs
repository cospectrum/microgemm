mod pack;
mod sizes;

mod colmajor;
mod rowmajor;

pub(crate) mod block {
    pub(crate) use super::colmajor::ColMajor;
    pub(crate) use super::rowmajor::RowMajor;
}
pub(crate) use pack::{pack_a, pack_b};

pub use sizes::PackSizes;
