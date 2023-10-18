use core::ops::{Add, Mul};

use crate::{kernels::Generic4x4Kernel, Kernel, One, Zero};

pub fn select_kernel<T>() -> impl Kernel<T>
where
    T: Copy + Zero + One + Mul<Output = T> + Add<Output = T>,
{
    Generic4x4Kernel
}

#[cfg(test)]
mod tests {
    use crate::PackSizes;

    use super::*;

    #[test]
    fn test_select_kernel() {
        let ker = select_kernel::<i32>();
        let pack_sizes = PackSizes {
            mc: ker.mr(),
            kc: 4,
            nc: ker.nr(),
        };
        println!("{:?}", pack_sizes.buf_len(&ker));
        pack_sizes.check(&ker);
    }
}
