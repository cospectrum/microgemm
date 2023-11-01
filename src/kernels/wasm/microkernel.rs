use super::WasmSimd128Kernel;
use crate::typenum::U4;

use core::arch::wasm32::*;

impl crate::Kernel for WasmSimd128Kernel<f32> {
    type Scalar = f32;
    type Mr = U4;
    type Nr = U4;

    fn microkernel(
        &self,
        alpha: Self::Scalar,
        lhs: &crate::MatRef<Self::Scalar>,
        rhs: &crate::MatRef<Self::Scalar>,
        beta: Self::Scalar,
        dst: &mut crate::MatMut<Self::Scalar>,
    ) {
        debug_assert_eq!(dst.nrows(), Self::MR);
        debug_assert_eq!(dst.ncols(), Self::NR);
        debug_assert_eq!(dst.row_stride(), 1);
        debug_assert_eq!(dst.col_stride(), 4);

        let kc = lhs.ncols();
        wasm_simd128_4x4_microkernel_f32(
            kc,
            alpha,
            lhs.as_slice(),
            rhs.as_slice(),
            beta,
            dst.as_mut_slice(),
        );
    }
}

fn wasm_simd128_4x4_microkernel_f32(
    kc: usize,
    alpha: f32,
    lhs: &[f32],
    rhs: &[f32],
    beta: f32,
    dst_colmajor: &mut [f32],
) {
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), 4 * kc);

    let mut cols = [f32x4_splat(0f32); 4];
    let fma = |cv, av, bv| f32x4_add(cv, f32x4_mul(av, bv));

    let left = lhs.chunks_exact(4);
    let right = rhs.chunks_exact(4);
    left.zip(right).for_each(|(a, b)| {
        let av = f32x4(a[0], a[1], a[2], a[3]);
        cols.iter_mut().zip(b).for_each(|(col, &scalar)| {
            let bv = f32x4_splat(scalar);
            *col = fma(*col, av, bv);
        });
    });

    let extract = |v: v128| -> [f32; 4] {
        [
            f32x4_extract_lane::<0>(v),
            f32x4_extract_lane::<1>(v),
            f32x4_extract_lane::<2>(v),
            f32x4_extract_lane::<3>(v),
        ]
    };
    let alpha = f32x4_splat(alpha);
    let it = dst_colmajor.chunks_exact_mut(4).zip(cols);
    it.for_each(|(dst, cv)| {
        let col = extract(f32x4_mul(cv, alpha));
        for (y, x) in dst.iter_mut().zip(col) {
            *y = x + beta * *y;
        }
    });
}

#[cfg(test)]
mod tests {
    use crate::std_prelude::*;
    use crate::{kernels::*, utils::*, *};
    use rand::{thread_rng, Rng};
    use wasm_bindgen_test::*;

    wasm_bindgen_test_configure!(run_in_browser);

    const fn wasm_simd128_kernel<T>() -> WasmSimd128Kernel<T> {
        if cfg!(target_feature = "simd128") {
            unsafe { WasmSimd128Kernel::new() }
        } else {
            panic!("simd128 target feature is not enabled");
        }
    }

    #[wasm_bindgen_test]
    fn test_wasm_simd128_kernel_f32() {
        let kernel = wasm_simd128_kernel();
        let cmp = |expect: &[f32], got: &[f32]| {
            let eps = 75.0 * f32::EPSILON;
            assert_approx_eq(expect, got, eps);
        };
        let mut rng = thread_rng();

        for _ in 0..40 {
            let scalar = || rng.gen_range(-1.0..1.0);
            random_kernel_test(&kernel, scalar, cmp);
        }
    }

    #[wasm_bindgen_test]
    fn bench_wasm_simd128_gemm_f32() {
        use core::hint::black_box;
        use instant::Instant;

        const ITER: usize = 5;
        let [m, k, n] = [512, 1024, 512];

        let alpha = black_box(1f32);
        let beta = black_box(1f32);

        let a = black_box(vec![0f32; m * k]);
        let b = black_box(vec![0f32; k * n]);
        let mut c = black_box(vec![0f32; m * n]);

        let a = &MatRef::new(m, k, &a, Layout::RowMajor);
        let b = &MatRef::new(k, n, &b, Layout::RowMajor);
        let c = &mut MatMut::new(m, n, &mut c, Layout::RowMajor);

        let wasm_kernel = &wasm_simd128_kernel();
        let generic_kernel = Generic4x4Kernel::<f32>::new();
        let pack_sizes = &PackSizes {
            mc: m,
            kc: k,
            nc: n,
        };
        let mut packing_buf = vec![0f32; pack_sizes.buf_len()];

        let mean_time = |t: Instant| t.elapsed() / ITER as u32;

        let time = Instant::now();
        for _ in 0..ITER {
            generic_kernel.gemm(alpha, a, b, beta, c, pack_sizes, &mut packing_buf);
        }
        let generic = mean_time(time);
        console_log!("generic bench: {:?}", generic);

        let time = Instant::now();
        for _ in 0..ITER {
            wasm_kernel.gemm(alpha, a, b, beta, c, pack_sizes, &mut packing_buf);
        }
        let wasm = mean_time(time);
        console_log!("wasm bench: {:?}", wasm);
    }
}
