use crate::{utils::naive_gemm, Kernel, One, Zero};
use crate::{Layout, MatMut, MatRef, PackSizes};
use rand::Rng;

use core::fmt::Debug;

pub(crate) fn test_kernel_with_random_i32<K>(kernel: &K)
where
    K: Kernel<i32>,
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

pub(crate) fn random_kernel_test<T, K>(
    kernel: &K,
    mut scalar: impl FnMut() -> T,
    cmp: impl FnOnce(&[T], &[T]),
) where
    K: Kernel<T>,
    T: Copy + Zero + One + Debug,
{
    let rng = &mut rand::thread_rng();

    let m = rng.gen_range(1..100);
    let k = rng.gen_range(1..100);
    let n = rng.gen_range(1..100);

    let mut random_layout = || {
        if rng.gen_bool(0.5) {
            Layout::RowMajor
        } else {
            Layout::ColumnMajor
        }
    };

    let a = (0..m * k).map(|_| scalar()).collect::<Vec<T>>();
    let b = (0..k * n).map(|_| scalar()).collect::<Vec<T>>();
    let a = MatRef::new(m, k, &a, random_layout());
    let b = MatRef::new(k, n, &b, random_layout());

    let mut c = (0..m * n).map(|_| scalar()).collect::<Vec<T>>();
    let mut expect = c.clone();
    let mut c = MatMut::new(m, n, &mut c, random_layout());
    let mut expect = c.with_values(&mut expect);

    let mc = rng.gen_range(1..10) * K::MR;
    let nc = rng.gen_range(1..10) * K::NR;
    let kc = rng.gen_range(1..k + 20);

    let pack_sizes = PackSizes { mc, kc, nc };
    let buf_len = pack_sizes.buf_len(kernel);
    let mut buf = vec![scalar(); buf_len];

    let alpha = scalar();
    let beta = scalar();

    kernel.gemm(alpha, &a, &b, beta, &mut c, &pack_sizes, &mut buf);
    naive_gemm(alpha, &a, &b, beta, &mut expect);

    cmp(expect.as_slice(), c.as_slice())
}
