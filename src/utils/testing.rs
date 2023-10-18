use crate::{utils::naive_gemm, Kernel};

pub(crate) fn test_kernel_with_random_i32<K>(kernel: &K)
where
    K: Kernel<i32>,
{
    use crate::{Layout, MatMut, MatRef, PackSizes};
    use rand::Rng;

    let rng = &mut rand::thread_rng();
    let distr = rand::distributions::Uniform::new(-30, 30);

    let m = rng.gen_range(1..100);
    let k = rng.gen_range(1..100);
    let n = rng.gen_range(1..100);

    let alpha = rng.gen_range(-10..10);
    let beta = rng.gen_range(-10..10);

    let a = rng.sample_iter(distr).take(m * k).collect::<Vec<i32>>();
    let b = rng.sample_iter(distr).take(k * n).collect::<Vec<i32>>();
    let mut c = rng.sample_iter(distr).take(m * n).collect::<Vec<i32>>();
    let mut expect = c.clone();

    let a = MatRef::new(m, k, &a, Layout::RowMajor);
    let b = MatRef::new(k, n, &b, Layout::RowMajor);
    let mut c = MatMut::new(m, n, &mut c, Layout::RowMajor);
    let mut expect = MatMut::new(m, n, &mut expect, Layout::RowMajor);

    let mc = rng.gen_range(1..6) * K::MR;
    let nc = rng.gen_range(1..6) * K::NR;
    let kc = rng.gen_range(1..40);
    let pack_sizes = PackSizes { mc, kc, nc };

    let buf_len = pack_sizes.buf_len(kernel);
    let fill = rng.gen_range(-10..10);
    let mut buf = vec![fill; buf_len];

    kernel.gemm(alpha, &a, &b, beta, &mut c, &pack_sizes, &mut buf);
    naive_gemm(alpha, &a, &b, beta, &mut expect);
    assert_eq!(expect.as_slice(), c.as_slice());
}
