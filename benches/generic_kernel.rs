use criterion::{black_box, criterion_group, criterion_main, Criterion};
use microgemm::{
    kernels::Generic4x4Kernel, utils::naive_gemm, Kernel, Layout, MatMut, MatRef, PackSizes,
};

const MC: usize = 256;
const KC: usize = 4096;
const NC: usize = 512;

fn bench_gemm(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_generic_kernel");
    group.sample_size(20);

    let kernel = Generic4x4Kernel;
    let m = MC;
    let k = KC;
    let n = NC;

    let pack_sizes = PackSizes {
        mc: MC.min(m),
        kc: KC.min(k),
        nc: NC.min(n),
    };
    let buf_len = pack_sizes.buf_len::<f32, _>(&kernel);
    let mut buf = vec![0f32; buf_len];

    let a = black_box(vec![0f32; m * k]);
    let b = black_box(vec![0f32; k * n]);
    let mut c1 = black_box(vec![0f32; m * n]);
    let mut c2 = c1.clone();

    let a = MatRef::new(m, k, &a, Layout::RowMajor);
    let b = MatRef::new(k, n, &b, Layout::ColumnMajor);
    let mut c1 = MatMut::new(m, n, c1.as_mut(), Layout::RowMajor);
    let mut c2 = c1.with_values(c2.as_mut());

    let alpha = black_box(1.0);
    let beta = black_box(0.0);

    let mut with_kernel = || {
        kernel.gemm(
            alpha,
            a.as_ref(),
            b.as_ref(),
            beta,
            c1.as_mut(),
            &pack_sizes,
            &mut buf,
        );
    };

    let mut naive = || {
        naive_gemm(alpha, a.as_ref(), b.as_ref(), beta, c2.as_mut());
    };

    group.bench_function("generic_kernel", |bencher| {
        bencher.iter(&mut with_kernel);
    });
    group.bench_function("naive", |bencher| {
        bencher.iter(&mut naive);
    });
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
