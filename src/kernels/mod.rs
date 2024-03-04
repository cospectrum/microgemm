mod generic;
#[cfg(any(target_arch = "aarch64", doc))]
mod neon;

use crate::{Kernel, MatMut, MatRef};

pub use generic::{
    GenericKernel16x16, GenericKernel2x2, GenericKernel32x32, GenericKernel4x4, GenericKernel8x8,
};
#[cfg(any(target_arch = "aarch64", doc))]
pub use neon::{NeonKernel4x4, NeonKernel8x8};

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
