use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};
use microgemm as mg;
use microgemm::{Kernel, Layout, MatMut, MatRef, PackSizes};

fn bench_gemm(criterion: &mut Criterion) {
    let group = &mut criterion.benchmark_group("bench gemm f32");
    group.sample_size(10);

    const DIM: usize = 320;

    let m = DIM;
    let k = DIM;
    let n = DIM;
    let mkn = [m, k, n];

    let pack_sizes = &PackSizes {
        mc: m,
        kc: k,
        nc: n,
    };
    let kernel = &mg::kernels::Generic8x8Kernel::<f32>::new();
    bench_kernel_with(group, "generic kernel", kernel, mkn, pack_sizes);

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        let kernel = &mg::kernels::Aarch64Kernel::<f32>::new();
        bench_kernel_with(group, "neon kernel", kernel, mkn, pack_sizes);
    }
}

fn bench_kernel_with(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    kernel: &impl Kernel<Elem = f32>,
    mkn: [usize; 3],
    pack_sizes: &PackSizes,
) {
    let [m, k, n] = mkn;
    let a = black_box(vec![0f32; m * k]);
    let b = black_box(vec![0f32; k * n]);
    let mut c = black_box(vec![0f32; m * n]);
    let (a, b, mut c) = matrices(mkn, &a, &b, &mut c);

    let alpha = black_box(1.0);
    let beta = black_box(0.0);
    let mut buf = black_box(vec![0f32; pack_sizes.buf_len(kernel)]);

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
    let a = MatRef::new(m, k, a, Layout::RowMajor);
    let b = MatRef::new(k, n, b, Layout::ColumnMajor);
    let c = MatMut::new(m, n, c, Layout::RowMajor);
    (a, b, c)
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
