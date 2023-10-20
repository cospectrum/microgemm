use criterion::{black_box, criterion_group, criterion_main, Criterion};
use microgemm as mg;
use microgemm::{Kernel, Layout, MatMut, MatRef, PackSizes};

fn bench_gemm(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_gemm_f32");
    group.sample_size(10);

    const MC: usize = 256;
    const KC: usize = 4096;
    const NC: usize = 512;

    let m = MC;
    let k = KC;
    let n = NC;

    let pack_sizes = &PackSizes {
        mc: MC.min(m),
        kc: KC.min(k),
        nc: NC.min(n),
    };

    let a = black_box(vec![0f32; m * k]);
    let b = black_box(vec![0f32; k * n]);
    let c = black_box(vec![0f32; m * n]);
    let input_values = || (a.clone(), b.clone(), c.clone());

    let alpha = black_box(1.0);
    let beta = black_box(0.0);
    let mkn = (m, k, n);

    group.bench_function("generic8x8_kernel", |bencher| {
        let kernel = mg::kernels::Generic8x8Kernel::<f32>::new();
        let mut buf = vec![0f32; pack_sizes.buf_len(&kernel)];
        let (a, b, mut c) = input_values();
        let (a, b, mut c) = matrices(mkn, &a, &b, &mut c);
        let f = || {
            kernel.gemm(alpha, &a, &b, beta, &mut c, pack_sizes, &mut buf);
        };
        bencher.iter(f);
    });

    group.bench_function("generic4x4_kernel", |bencher| {
        let kernel = mg::kernels::Generic4x4Kernel::<f32>::new();
        let mut buf = vec![0f32; pack_sizes.buf_len(&kernel)];
        let (a, b, mut c) = input_values();
        let (a, b, mut c) = matrices(mkn, &a, &b, &mut c);
        let f = || {
            kernel.gemm(alpha, &a, &b, beta, &mut c, pack_sizes, &mut buf);
        };
        bencher.iter(f);
    });

    group.bench_function("generic2x2_kernel", |bencher| {
        let kernel = mg::kernels::Generic2x2Kernel::<f32>::new();
        let mut buf = vec![0f32; pack_sizes.buf_len(&kernel)];
        let (a, b, mut c) = input_values();
        let (a, b, mut c) = matrices(mkn, &a, &b, &mut c);
        let f = || {
            kernel.gemm(alpha, &a, &b, beta, &mut c, pack_sizes, &mut buf);
        };
        bencher.iter(f);
    });
}

fn matrices<'a, T>(
    mkn: (usize, usize, usize),
    a: &'a [T],
    b: &'a [T],
    c: &'a mut [T],
) -> (MatRef<'a, T>, MatRef<'a, T>, MatMut<'a, T>) {
    let (m, k, n) = mkn;
    let a = MatRef::new(m, k, a, Layout::RowMajor);
    let b = MatRef::new(k, n, b, Layout::ColumnMajor);
    let c = MatMut::new(m, n, c, Layout::RowMajor);
    (a, b, c)
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
