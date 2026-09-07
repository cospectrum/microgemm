//! Native, single-threaded comparison against matrixmultiply.

#[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
fn main() {
    eprintln!("neon_compare requires AArch64 with NEON enabled");
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
fn main() {
    native::run();
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod native {
    use criterion::black_box;
    use microgemm::{kernels::NeonKernel8x8, Kernel, MatMut, MatRef, PackSizes};
    use std::time::{Duration, Instant};

    #[derive(Clone, Copy)]
    struct Case {
        m: usize,
        k: usize,
        n: usize,
        col_c: bool,
        blocked: bool,
        beta: f32,
    }

    impl Case {
        fn pack(self) -> PackSizes {
            let round = |x| (x + 7) / 8 * 8;
            if self.blocked {
                PackSizes {
                    mc: round(self.m).min(64),
                    kc: self.k.min(256),
                    nc: round(self.n).min(1024),
                }
            } else {
                PackSizes {
                    mc: round(self.m),
                    kc: self.k,
                    nc: round(self.n),
                }
            }
        }
    }

    fn values(len: usize, mut state: u32) -> Vec<f32> {
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                ((state >> 24) as f32 - 128.0) / 256.0
            })
            .collect()
    }

    #[inline(never)]
    fn micro(case: Case, a: &[f32], b: &[f32], c: &mut [f32], packing: &mut [f32]) {
        // SAFETY: this module is compiled only when NEON is enabled.
        let kernel = unsafe { NeonKernel8x8::<f32>::new() };
        let a = MatRef::col_major(case.m, case.k, a);
        let b = MatRef::row_major(case.k, case.n, b);
        let mut c = if case.col_c {
            MatMut::col_major(case.m, case.n, c)
        } else {
            MatMut::row_major(case.m, case.n, c)
        };
        kernel.gemm(1.0, a, b, case.beta, &mut c, case.pack(), packing);
    }

    #[inline(never)]
    fn matrixmultiply(case: Case, a: &[f32], b: &[f32], c: &mut [f32]) {
        let (rs, cs) = if case.col_c { (1, case.m) } else { (case.n, 1) };
        // SAFETY: all three buffers have exactly the dimensions specified here,
        // C's strides are nonoverlapping, and its mutable borrow is exclusive.
        unsafe {
            matrixmultiply::sgemm(
                case.m,
                case.k,
                case.n,
                1.0,
                a.as_ptr(),
                1,
                case.m as isize,
                b.as_ptr(),
                case.n as isize,
                1,
                case.beta,
                c.as_mut_ptr(),
                rs as isize,
                cs as isize,
            );
        }
    }

    fn batch(reps: usize, mut operation: impl FnMut()) -> f64 {
        let start = Instant::now();
        for _ in 0..reps {
            operation();
        }
        start.elapsed().as_secs_f64() / reps as f64
    }

    fn percentile(samples: &[f64], percent: usize) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[(sorted.len() - 1) * percent / 100]
    }

    fn measure(case: Case, samples: usize) {
        let a = values(case.m * case.k, 17);
        let b = values(case.k * case.n, 31);
        let initial = values(case.m * case.n, 47);
        let mut c = initial.clone();
        let mut reference = initial.clone();
        let mut packing = vec![0.0; case.pack().buf_len()];
        micro(case, &a, &b, &mut c, &mut packing);
        matrixmultiply(case, &a, &b, &mut reference);
        for (index, (&actual, &expected)) in c.iter().zip(&reference).enumerate() {
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 2e-4 * (1.0 + expected.abs()),
                "incorrect result at {index}: {actual} vs {expected}"
            );
        }

        let mut run = |which: usize, reps: usize| {
            // Reset outside the timer, including beta != 0 cases.
            let output = if which == 0 { &mut c } else { &mut reference };
            output.copy_from_slice(&initial);
            let elapsed = batch(reps, || {
                if which == 0 {
                    micro(
                        black_box(case),
                        black_box(&a),
                        black_box(&b),
                        black_box(output),
                        black_box(&mut packing),
                    );
                } else {
                    matrixmultiply(
                        black_box(case),
                        black_box(&a),
                        black_box(&b),
                        black_box(output),
                    );
                }
            });
            black_box(&*output);
            elapsed
        };

        // Warm both paths, then calibrate batches to about 10 ms per library.
        let warmup = Instant::now();
        while warmup.elapsed() < Duration::from_millis(100) {
            run(0, 1);
            run(1, 1);
        }
        let fastest = run(0, 1).min(run(1, 1));
        let reps = (0.010 / fastest).ceil().max(1.0) as usize;
        let mut times = [Vec::new(), Vec::new()];
        let mut ratios = Vec::new();
        for sample in 0..samples {
            for which in [sample % 2, 1 - sample % 2] {
                times[which].push(run(which, reps));
            }
            ratios.push(times[0][sample] / times[1][sample]);
        }
        println!(
            "{},{},{},{},{},{},{},{},{:.3},{:.3},{:.5},{:.5},{:.5}",
            case.m,
            case.k,
            case.n,
            if case.col_c { "col" } else { "row" },
            if case.blocked { "blocked" } else { "full" },
            case.beta,
            samples,
            reps,
            percentile(&times[0], 50) * 1e6,
            percentile(&times[1], 50) * 1e6,
            percentile(&ratios, 50),
            percentile(&ratios, 10),
            percentile(&ratios, 90),
        );
        if let Ok(path) = std::env::var("MICROGEMM_RAW") {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .unwrap();
            for (sample, (micro, mm)) in times[0].iter().zip(&times[1]).enumerate() {
                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{:.9},{:.9}",
                    case.m, case.k, case.n, case.col_c, case.blocked, case.beta, sample, micro, mm
                )
                .unwrap();
            }
        }
    }

    pub fn run() {
        // `cargo test --all-targets` must not run the timing experiment.
        if !std::env::args().any(|arg| arg == "--bench") {
            return;
        }
        #[cfg(target_os = "macos")]
        {
            extern "C" {
                fn pthread_set_qos_class_self_np(class: u32, relative_priority: i32) -> i32;
            }
            // SAFETY: valid QoS class/priority, applied only to this benchmark thread.
            let status = unsafe { pthread_set_qos_class_self_np(0x19, 0) };
            assert_eq!(status, 0, "setting user-initiated QoS failed");
        }
        let samples = std::env::var("MICROGEMM_SAMPLES")
            .map(|x| x.parse::<usize>().unwrap())
            .unwrap_or(21);
        assert!(samples >= 3);
        let filter = std::env::var("MICROGEMM_CASE").unwrap_or_default();
        println!(
            "m,k,n,c_layout,packing,beta,samples,reps,micro_us,mm_us,ratio,p10_ratio,p90_ratio"
        );
        let mut cases = Vec::new();
        for n in [128, 256, 512, 1024, 2048] {
            cases.push(Case {
                m: n,
                k: n,
                n,
                col_c: false,
                blocked: false,
                beta: 0.0,
            });
        }
        for n in [256, 512, 1024, 2048] {
            cases.push(Case {
                m: n,
                k: n,
                n,
                col_c: false,
                blocked: true,
                beta: 0.0,
            });
        }
        for (m, k, n, col_c, beta) in [
            (512, 512, 512, true, 0.0),
            (512, 512, 512, false, -0.5),
            (512, 512, 512, true, -0.5),
            (128, 512, 256, false, 0.0),
            (511, 129, 257, false, 0.0),
        ] {
            cases.push(Case {
                m,
                k,
                n,
                col_c,
                blocked: true,
                beta,
            });
        }
        for case in cases {
            let name = format!(
                "{}x{}x{}-{}-{}",
                case.m,
                case.k,
                case.n,
                if case.col_c { "col" } else { "row" },
                if case.blocked { "blocked" } else { "full" }
            );
            if name.contains(&filter) {
                measure(case, samples);
            }
        }
    }
}
