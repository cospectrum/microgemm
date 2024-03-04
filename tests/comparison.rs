use microgemm::{typenum, Kernel, MatMut, MatRef, PackSizes};
use std::time::{Duration, Instant};

#[ignore]
#[test]
#[cfg(target_arch = "aarch64")]
fn bench_aarch64_f32() {
    let neon_kernel = if cfg!(target_feature = "neon") {
        unsafe { microgemm::kernels::NeonKernel4x4::<f32>::new() }
    } else {
        println!("neon feature is not enabled, exiting...");
        return;
    };
    let mt_kernel = MatrixMultiplyKernel;
    let faer_kernel = FaerKernel;

    const TRIES: u32 = 6;

    let sizes = (7..12).map(|x| 2usize.pow(x));
    println!(
        "{0:>4} {1:>14} {2:>14} {3:>14}",
        "n", "NeonKernel4x4", "faer", "matrixmultiply",
    );
    for n in sizes {
        let t_mt = display_duration(time_with(&mt_kernel, n, TRIES));
        let t_faer = display_duration(time_with(&faer_kernel, n, TRIES));
        let t_neon = display_duration(time_with(&neon_kernel, n, TRIES));
        println!("{0:>4} {1:>14} {2:>14} {3:>14}", n, t_neon, t_faer, t_mt);
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

    let a = &MatRef::col_major(n, n, a.as_ref());
    let b = &MatRef::row_major(n, n, b.as_ref());
    let c = &mut MatMut::row_major(n, n, c.as_mut());

    let pack_sizes = &PackSizes {
        mc: n,
        kc: n,
        nc: n,
    };
    let mut packing_buf = vec![0f32; pack_sizes.buf_len()];
    let alpha = black_box(1f32);
    let beta = black_box(0f32);

    let mut result = Duration::from_secs(u64::MAX);
    for _ in 0..tries as usize {
        let time = Instant::now();
        kernel.gemm(alpha, a, b, beta, c, pack_sizes, &mut packing_buf);
        let time = time.elapsed();
        result = time.min(result);
    }
    result
}

struct FaerKernel;

impl Kernel for FaerKernel {
    type Scalar = f32;
    type Mr = typenum::U1;
    type Nr = typenum::U1;

    fn microkernel(
        &self,
        _: Self::Scalar,
        _: &MatRef<Self::Scalar>,
        _: &MatRef<Self::Scalar>,
        _: Self::Scalar,
        _: &mut MatMut<Self::Scalar>,
    ) {
        unreachable!()
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
        let lhs = unsafe {
            faer_core::mat::from_raw_parts::<f32>(
                a.as_slice().as_ptr(),
                a.nrows(),
                a.ncols(),
                a.row_stride() as isize,
                a.col_stride() as isize,
            )
        };
        let rhs = unsafe {
            faer_core::mat::from_raw_parts::<f32>(
                b.as_slice().as_ptr(),
                b.nrows(),
                b.ncols(),
                b.row_stride() as isize,
                b.col_stride() as isize,
            )
        };
        let mut acc = unsafe {
            faer_core::mat::from_raw_parts_mut::<f32>(
                c.as_mut_slice().as_mut_ptr(),
                c.nrows(),
                c.ncols(),
                c.row_stride() as isize,
                c.col_stride() as isize,
            )
        };
        faer_core::mul::matmul(
            acc.as_mut(),
            lhs.as_ref(),
            rhs.as_ref(),
            Some(alpha),
            beta,
            faer_core::Parallelism::None,
        );
    }
}

struct MatrixMultiplyKernel;

impl Kernel for MatrixMultiplyKernel {
    type Scalar = f32;
    type Mr = typenum::U1;
    type Nr = typenum::U1;

    fn microkernel(
        &self,
        _: Self::Scalar,
        _: &MatRef<Self::Scalar>,
        _: &MatRef<Self::Scalar>,
        _: Self::Scalar,
        _: &mut MatMut<Self::Scalar>,
    ) {
        unreachable!()
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
        let [m, k] = [a.nrows(), a.ncols()];
        let n = b.ncols();
        let [rsa, csa] = [a.row_stride(), a.col_stride()];
        let [rsb, csb] = [b.row_stride(), b.col_stride()];
        let [rsc, csc] = [c.row_stride(), c.col_stride()];
        let a = a.as_slice().as_ptr();
        let b = b.as_slice().as_ptr();
        let c = c.as_mut_slice().as_mut_ptr();
        unsafe {
            matrixmultiply::sgemm(
                m,
                k,
                n,
                alpha,
                a,
                rsa as isize,
                csa as isize,
                b,
                rsb as isize,
                csb as isize,
                beta,
                c,
                rsc as isize,
                csc as isize,
            );
        }
    }
}
