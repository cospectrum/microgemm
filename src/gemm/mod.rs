mod kernels;
mod loops;
mod packing;

pub(crate) use packing::*;

pub use kernels::*;
pub use loops::gemm_with_params;

#[derive(Debug, Clone)]
pub struct BlockSizes {
    pub mc: usize,
    pub mr: usize,
    pub kc: usize,
    pub nc: usize,
    pub nr: usize,
}

impl BlockSizes {
    fn check(&self) {
        assert!(self.nr <= self.nc);
        assert!(self.mr <= self.mc);
        assert_eq!(self.nc % self.nr, 0);
        assert_eq!(self.mc % self.mr, 0);
    }
}

impl AsRef<BlockSizes> for BlockSizes {
    fn as_ref(&self) -> &BlockSizes {
        self
    }
}
