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
    pub(crate) fn checked_buf_len(self) -> Option<usize> {
        let apack_len = self.mc.checked_mul(self.kc)?;
        let bpack_len = self.kc.checked_mul(self.nc)?;
        apack_len.checked_add(bpack_len)
    }
    pub(crate) fn clamped<T, K>(self, _: &K) -> Self
    where
        K: Kernel<Scalar = T> + ?Sized,
    {
        let mr = K::MR;
        let nr = K::NR;
        assert!(mr > 0);
        assert!(nr > 0);
        assert!(mr <= self.mc);
        assert!(nr <= self.nc);

        let mc = self.mc - self.mc % mr;
        let kc = self.kc;
        let nc = self.nc - self.nc % nr;
        Self { mc, kc, nc }
    }
    pub(crate) fn split_buf<T>(self, buf: &mut [T]) -> (&mut [T], &mut [T]) {
        assert_eq!(buf.len(), self.checked_buf_len().unwrap());
        let (apack, bpack) = buf.split_at_mut(self.mc * self.kc);
        (apack, bpack)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_sizes_bug_len() {
        let [mc, kc, nc] = [3, 2, 5];
        let pack_sizes = PackSizes { mc, kc, nc };
        assert_eq!(pack_sizes.buf_len(), pack_sizes.checked_buf_len().unwrap());
    }
}
