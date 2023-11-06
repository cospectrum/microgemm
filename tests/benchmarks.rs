use microgemm as mg;
use microgemm::{Kernel, Layout, MatMut, MatRef, PackSizes};
use std::time::{Duration, Instant};

#[ignore]
#[test]
#[cfg(target_arch = "aarch64")]
fn bench_neon_gemm_f32() {
    let neon_kernel = if cfg!(target_feature = "neon") {
        unsafe { microgemm::kernels::NeonKernel::<f32>::new() }
    } else {
        println!("neon feature is not enabled, exiting...");
        return;
    };
    let generic_4x4_kernel = mg::kernels::Generic4x4Kernel::<f32>::new();
    let generic_8x8_kernel = mg::kernels::Generic8x8Kernel::<f32>::new();
    let naive_kernel = NaiveKernel;

    const TRIES: u32 = 5;

    let sizes = (5..11).map(|x| 2usize.pow(x));
    println!(
        "{0:>4} {1:>13} {2:>13} {3:>13} {4:>13}",
        "n", "NeonKernel", "Generic4x4", "Generic8x8", "naive(rustc)"
    );
    for n in sizes {
        let t_neon = display_duration(time_with(&neon_kernel, n, TRIES));
        let t_4x4 = display_duration(time_with(&generic_4x4_kernel, n, TRIES));
        let t_8x8 = display_duration(time_with(&generic_8x8_kernel, n, TRIES));
        let t_naive = display_duration(time_with(&naive_kernel, n, TRIES));
        println!(
            "{0:>4} {1:>13} {2:>13} {3:>13} {4:>13}",
            n, t_neon, t_4x4, t_8x8, t_naive,
        );
    }
}

#[allow(dead_code)]
fn display_duration(t: Duration) -> String {
    let as_float = |s: &str, unit: &str| {
        assert!(s.contains(unit));
        let s = s.replace(unit, "");
        if s.contains('.') {
            s.parse::<f64>().unwrap()
        } else {
            s.parse::<u64>().unwrap() as f64
        }
    };
    let pretty = |s: String, unit: &str| {
        let num = as_float(&s, unit);
        if num.fract() < 0.2 {
            format!("{}{}", num.trunc(), unit)
        } else {
            let fract = (10.0 * num.fract()).trunc();
            format!("{}.{}{}", num.trunc(), fract, unit)
        }
    };
    let s = format!("{:?}", t);

    if s.contains("ms") {
        pretty(s, "ms")
    } else if s.contains("µs") {
        pretty(s, "µs")
    } else if s.contains("ns") {
        pretty(s, "ns")
    } else if s.contains('s') {
        pretty(s, "s")
    } else {
        panic!("unknown unit of time");
    }
}

#[allow(dead_code)]
fn time_with(kernel: &impl Kernel<Scalar = f32>, n: usize, tries: u32) -> Duration {
    use core::hint::black_box;

    let a = black_box(vec![0f32; n * n]);
    let b = black_box(a.clone());
    let mut c = black_box(a.clone());

    let a = &MatRef::new(n, n, a.as_ref(), Layout::RowMajor);
    let b = &MatRef::new(n, n, b.as_ref(), Layout::ColMajor);
    let c = &mut MatMut::new(n, n, c.as_mut(), Layout::RowMajor);

    let pack_sizes = &PackSizes {
        mc: n,
        kc: n,
        nc: n,
    };
    let mut packing_buf = vec![0f32; pack_sizes.buf_len()];
    let alpha = 1f32;
    let beta = 0f32;

    let mut result = Duration::from_secs(u64::MAX);
    for _ in 0..tries as usize {
        let time = Instant::now();
        kernel.gemm(alpha, a, b, beta, c, pack_sizes, &mut packing_buf);
        let time = time.elapsed();
        result = time.min(result);
    }
    result
}

struct NaiveKernel;

impl Kernel for NaiveKernel {
    type Scalar = f32;
    type Mr = mg::typenum::U1;
    type Nr = mg::typenum::U1;

    fn microkernel(
        &self,
        _: Self::Scalar,
        _: &MatRef<Self::Scalar>,
        _: &MatRef<Self::Scalar>,
        _: Self::Scalar,
        _: &mut MatMut<Self::Scalar>,
    ) {
        unreachable!();
    }

    fn gemm(
        &self,
        alpha: Self::Scalar,
        a: &MatRef<Self::Scalar>,
        b: &MatRef<Self::Scalar>,
        beta: Self::Scalar,
        c: &mut MatMut<Self::Scalar>,
        _: impl AsRef<PackSizes>,
        _: &mut [Self::Scalar],
    ) {
        assert_eq!(a.nrows(), c.nrows());
        assert_eq!(b.ncols(), c.ncols());
        assert_eq!(a.ncols(), b.nrows());

        let k = a.ncols();

        for i in 0..a.nrows() {
            for j in 0..b.ncols() {
                let dot = (0..k)
                    .map(|h| a.get(i, h) * b.get(h, j))
                    .reduce(|accum, x| accum + x)
                    .unwrap();
                let z = c.get_mut(i, j);
                *z = alpha * dot + beta * *z;
            }
        }
    }
}
