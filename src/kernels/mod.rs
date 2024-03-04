mod generic;
#[cfg(any(target_arch = "aarch64", doc))]
mod neon;

use crate::{Kernel, MatMut, MatRef};

pub use generic::{
    Generic16x16Kernel, Generic2x2Kernel, Generic32x32Kernel, Generic4x4Kernel, Generic8x8Kernel,
};
#[cfg(any(target_arch = "aarch64", doc))]
pub use neon::NeonKernel4x4;

fn dbg_check_microkernel_inputs<T, K>(_: &K, lhs: &MatRef<T>, rhs: &MatRef<T>, dst: &mut MatMut<T>)
where
    K: Kernel<Scalar = T>,
{
    debug_assert_eq!(lhs.row_stride(), 1);
    debug_assert_eq!(lhs.nrows(), K::MR);

    debug_assert_eq!(rhs.col_stride(), 1);
    debug_assert_eq!(rhs.ncols(), K::NR);

    debug_assert_eq!(dst.row_stride(), 1);
    debug_assert_eq!(dst.nrows(), K::MR);
    debug_assert_eq!(dst.ncols(), K::NR);

    debug_assert_eq!(lhs.ncols(), rhs.nrows());
}
