use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};

use microgemm::kernels::GenericKernel8x8;
use microgemm::{Kernel, MatMut, MatRef, PackSizes};

fn bench_gemm(criterion: &mut Criterion) {
    let group = &mut criterion.benchmark_group("bench-gemm-f32");
    group.sample_size(10);

    const DIM: usize = 2 * 320;

    let m = DIM;
    let k = DIM;
    let n = DIM;
    let mkn = [m, k, n];

    let pack_sizes = &PackSizes {
        mc: m,
        kc: k,
        nc: n,
    };

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        use microgemm::kernels::{NeonKernel4x4, NeonKernel8x8};

        let kernel = unsafe { &NeonKernel8x8::<f32>::new() };
        bench_kernel_with(group, "neon-kernel-8x8", kernel, mkn, pack_sizes);

        let kernel = unsafe { &NeonKernel4x4::<f32>::new() };
        bench_kernel_with(group, "neon-kernel-4x4", kernel, mkn, pack_sizes);
    }

    let kernel = &GenericKernel8x8::<f32>::new();
    bench_kernel_with(group, "generic-kernel-8x8", kernel, mkn, pack_sizes);
}

fn bench_kernel_with(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    kernel: &impl Kernel<Scalar = f32>,
    mkn: [usize; 3],
    pack_sizes: &PackSizes,
) {
    let [m, k, n] = mkn;
    let a = black_box(vec![0f32; m * k]);
    let b = black_box(vec![0f32; k * n]);
    let mut c = black_box(vec![0f32; m * n]);
    let (a, b, mut c) = matrices(mkn, &a, &b, &mut c);

    let alpha = black_box(1.0);
    let beta = black_box(-1.0);
    let mut buf = black_box(vec![0f32; pack_sizes.buf_len()]);

    let mut f = || {
        kernel.gemm(alpha, &a, &b, beta, &mut c, pack_sizes, &mut buf);
    };
    group.bench_function(name, |bencher| bencher.iter(&mut f));
}

fn matrices<'a, T>(
    mkn: [usize; 3],
    a: &'a [T],
    b: &'a [T],
    c: &'a mut [T],
) -> (MatRef<'a, T>, MatRef<'a, T>, MatMut<'a, T>) {
    let [m, k, n] = mkn;
    let a = MatRef::row_major(m, k, a);
    let b = MatRef::col_major(k, n, b);
    let c = MatMut::row_major(m, n, c);
    (a, b, c)
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
