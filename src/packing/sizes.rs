use crate::Kernel;
use num_traits::{One, Zero};

#[derive(Debug, Clone, Copy)]
pub struct PackSizes {
    pub mc: usize,
    pub kc: usize,
    pub nc: usize,
}

impl PackSizes {
    pub const fn buf_len(self) -> usize {
        self.mc * self.kc + self.kc * self.nc
    }
    pub(crate) fn check<T, K>(self, _: &K)
    where
        T: One + Zero + Copy,
        K: Kernel<Scalar = T> + ?Sized,
    {
        let mr = K::MR;
        let nr = K::NR;
        assert!(mr <= self.mc);
        assert!(nr <= self.nc);
        assert_eq!(self.mc % mr, 0);
        assert_eq!(self.nc % nr, 0);
    }
    pub(crate) fn split_buf<T>(self, buf: &mut [T]) -> (&mut [T], &mut [T]) {
        let (apack, bpack) = buf.split_at_mut(self.mc * self.kc);
        (apack, bpack)
    }
}
