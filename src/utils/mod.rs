mod naive;
#[cfg(test)]
mod testing;

#[cfg(test)]
pub(crate) use testing::*;

pub use naive::naive_gemm;
