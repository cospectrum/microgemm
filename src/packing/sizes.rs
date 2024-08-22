use crate::Kernel;

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
    pub(crate) fn clamped<T, K>(self, _: &K) -> Self
    where
        K: Kernel<Scalar = T> + ?Sized,
    {
        let mr = K::MR;
        let nr = K::NR;
        assert!(mr <= self.mc);
        assert!(nr <= self.nc);

        let mc = self.mc - self.mc % mr;
        let kc = self.kc;
        let nc = self.nc - self.nc % nr;
        Self { mc, kc, nc }
    }
    pub(crate) fn split_buf<T>(self, buf: &mut [T]) -> (&mut [T], &mut [T]) {
        assert_eq!(self.buf_len(), buf.len());
        let (apack, bpack) = buf.split_at_mut(self.mc * self.kc);
        (apack, bpack)
    }
}
