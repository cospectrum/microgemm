mod naive;
#[cfg(test)]
mod testing;

#[cfg(test)]
pub(crate) use testing::test_kernel_with_random_i32;

pub use naive::naive_gemm;
