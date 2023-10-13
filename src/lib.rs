mod gemm;
mod mat;
mod naive;
mod packing;

pub(crate) use packing::*;

pub use gemm::*;
pub use mat::*;
pub use naive::naive_gemm;

#[derive(Debug, Clone)]
pub struct BlockSizes {
    pub mc: usize,
    pub mr: usize,
    pub kc: usize,
    pub nc: usize,
    pub nr: usize,
}

impl BlockSizes {
    pub const fn buf_len(&self) -> usize {
        self.mc * self.kc + self.kc * self.nc + self.mr * self.nr
    }
    fn check(&self) {
        assert!(self.nr <= self.nc);
        assert!(self.mr <= self.mc);
        assert_eq!(self.nc % self.nr, 0);
        assert_eq!(self.mc % self.mr, 0);
    }
    fn split_buf<'a, T>(&self, buf: &'a mut [T]) -> (&'a mut [T], &'a mut [T], &'a mut [T]) {
        assert_eq!(buf.len(), self.buf_len());
        let (a_buf, tail) = buf.split_at_mut(self.mc * self.kc);
        let (b_buf, c_buf) = tail.split_at_mut(self.kc * self.nc);
        (a_buf, b_buf, c_buf)
    }
}

impl AsRef<BlockSizes> for BlockSizes {
    fn as_ref(&self) -> &BlockSizes {
        self
    }
}
