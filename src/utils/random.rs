use crate::std_prelude::*;
use crate::{utils::naive_gemm, Kernel, Layout, MatMut, MatRef, One, PackSizes, Zero};
use core::marker::PhantomData;
use core::ops::{Add, Mul};
use rand::Rng;

pub fn test_kernel_with_random_i32<K>(kernel: &K)
where
    K: Kernel<Scalar = i32>,
{
    let rng = &mut rand::thread_rng();
    let distr = rand::distributions::Uniform::new(-30, 30);

    let mut it = rng.sample_iter(distr);
    let scalar = || it.next().unwrap();
    let cmp = |expect: &[i32], got: &[i32]| {
        assert_eq!(expect, got);
    };
    random_kernel_test(kernel, scalar, cmp);
}

pub fn random_kernel_test<T, K>(kernel: &K, scalar: impl FnMut() -> T, cmp: impl FnOnce(&[T], &[T]))
where
    K: Kernel<Scalar = T>,
    T: Copy + Zero + One,
{
    let rng = &mut rand::thread_rng();
    let mc = rng.gen_range(1..10) * K::MR;
    let nc = rng.gen_range(1..10) * K::NR;
    let test_kernel = TestKernel::<T>::new();
    cmp_kernels_with_random_data(&test_kernel, kernel, scalar, cmp, mc, nc);
}

fn toss_a_coin<T>(front: T, back: T) -> T {
    let mut rng = rand::thread_rng();
    if rng.gen_bool(0.5) {
        front
    } else {
        back
    }
}

pub fn cmp_kernels_with_random_data<T, K1, K2>(
    kernel1: &K1,
    kernel2: &K2,
    mut scalar: impl FnMut() -> T,
    cmp: impl FnOnce(&[T], &[T]),
    mc: usize,
    nc: usize,
) where
    K1: Kernel<Scalar = T>,
    K2: Kernel<Scalar = T>,
    T: Copy + Zero + One,
{
    let rng = &mut rand::thread_rng();

    let m = rng.gen_range(1..100);
    let k = rng.gen_range(1..100);
    let n = rng.gen_range(1..100);

    let random_layout = || toss_a_coin(Layout::RowMajor, Layout::ColMajor);

    let a = (0..m * k).map(|_| scalar()).collect::<Vec<T>>();
    let b = (0..k * n).map(|_| scalar()).collect::<Vec<T>>();
    let a = MatRef::new(m, k, &a, random_layout());
    let b = MatRef::new(k, n, &b, random_layout());

    let mut c1 = (0..m * n).map(|_| scalar()).collect::<Vec<T>>();
    let mut c2 = c1.clone();
    let mut c1 = MatMut::new(m, n, &mut c1, random_layout());
    let mut c2 = c1.with_values(&mut c2);

    let kc = rng.gen_range(1..k + 20);
    let pack_sizes = PackSizes { mc, kc, nc };

    let fill = scalar();
    let mut buf1 = vec![fill; pack_sizes.buf_len()];
    let mut buf2 = vec![fill; pack_sizes.buf_len()];

    let alpha = toss_a_coin(scalar(), toss_a_coin(T::zero(), T::one()));
    let beta = toss_a_coin(scalar(), toss_a_coin(T::zero(), T::one()));

    kernel1.gemm(alpha, &a, &b, beta, &mut c1, &pack_sizes, &mut buf1);
    kernel2.gemm(alpha, &a, &b, beta, &mut c2, &pack_sizes, &mut buf2);

    cmp(c1.as_slice(), c2.as_slice())
}

struct TestKernel<T> {
    marker: PhantomData<T>,
}

impl<T> TestKernel<T> {
    fn new() -> Self {
        Self {
            marker: Default::default(),
        }
    }
}

impl<T> Kernel for TestKernel<T>
where
    T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
{
    type Scalar = T;

    type Mr = crate::typenum::U0;
    type Nr = crate::typenum::U0;

    fn microkernel(&self, _: T, _: &MatRef<T>, _: &MatRef<T>, _: T, _: &mut MatMut<T>) {
        unreachable!()
    }
    fn gemm(
        &self,
        alpha: T,
        a: &MatRef<T>,
        b: &MatRef<T>,
        beta: T,
        c: &mut MatMut<T>,
        _: &PackSizes,
        _: &mut [T],
    ) {
        naive_gemm(alpha, a, b, beta, c);
    }
}
