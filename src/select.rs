use crate::{kernels::*, Kernel};
use core::marker::PhantomData;

#[macro_export]
macro_rules! select_kernel {
    (f32) => {
        $crate::select::kernel_selector::<f32>().select()
    };
}

pub fn kernel_selector<T>() -> KernelSelector<T> {
    KernelSelector::new()
}

#[derive(Debug, Clone, Copy, Default)]
pub struct KernelSelector<T> {
    marker: PhantomData<T>,
}

impl<T> KernelSelector<T> {
    pub fn new() -> Self {
        Self {
            marker: PhantomData,
        }
    }
}

impl KernelSelector<f32> {
    #[cfg(target_arch = "aarch64")]
    pub fn select(self) -> impl Kernel<Elem = f32> {
        Aarch64Kernel::new()
    }
    #[cfg(not(target_arch = "aarch64"))]
    pub fn select(self) -> impl Kernel<Elem = f32> {
        Generic4x4Kernel::new()
    }
}
