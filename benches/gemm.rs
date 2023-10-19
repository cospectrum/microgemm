use criterion::{black_box, criterion_group, criterion_main, Criterion};
use microgemm as mg;
use microgemm::{Kernel, Layout, MatMut, MatRef, PackSizes};

const MC: usize = 256;
const KC: usize = 4096;
const NC: usize = 512;

fn bench_gemm(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("bench_kernels");
    group.sample_size(20);

    let selected_kernel = mg::select_kernel!(f32);
    let generic_kernel = mg::generic4x4_kernel::<f32>();

    let m = MC;
    let k = KC;
    let n = NC;

    let pack_sizes = PackSizes {
        mc: MC.min(m),
        kc: KC.min(k),
        nc: NC.min(n),
    };

    let a = black_box(vec![0f32; m * k]);
    let b = black_box(vec![0f32; k * n]);
    let cvals = black_box(vec![0f32; m * n]);

    let a = MatRef::new(m, k, &a, Layout::RowMajor);
    let b = MatRef::new(k, n, &b, Layout::ColumnMajor);

    let alpha = black_box(1.0);
    let beta = black_box(0.0);

    let mut values = cvals.clone();
    let mut c = MatMut::new(m, n, values.as_mut(), Layout::RowMajor);
    let mut generic_buf = vec![0f32; pack_sizes.buf_len::<f32, _>(&generic_kernel)];
    let mut with_generic_kernel = || {
        generic_kernel.gemm(
            alpha,
            a.as_ref(),
            b.as_ref(),
            beta,
            c.as_mut(),
            &pack_sizes,
            &mut generic_buf,
        );
    };

    let mut values = cvals.clone();
    let mut c = MatMut::new(m, n, values.as_mut(), Layout::RowMajor);
    let mut selected_buf = vec![0f32; pack_sizes.buf_len(&selected_kernel)];
    let mut with_selected_kernel = || {
        selected_kernel.gemm(
            alpha,
            a.as_ref(),
            b.as_ref(),
            beta,
            c.as_mut(),
            &pack_sizes,
            &mut selected_buf,
        );
    };

    group.bench_function("selected_kernel", |bencher| {
        bencher.iter(&mut with_selected_kernel);
    });

    group.bench_function("generic_kernel", |bencher| {
        bencher.iter(&mut with_generic_kernel);
    });
}

criterion_group!(benches, bench_gemm);
criterion_main!(benches);
