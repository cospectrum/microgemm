use super::NeonKernel8x8;
use crate::{
    kernels::dbg_check_microkernel_inputs, typenum::U8, Kernel, MatMut, MatRef, PackSizes,
};

use super::super::simd::*;

impl Kernel for NeonKernel8x8<f32> {
    type Scalar = f32;
    type Mr = U8;
    type Nr = U8;

    fn microkernel(
        &self,
        alpha: f32,
        lhs: MatRef<f32>,
        rhs: MatRef<f32>,
        beta: f32,
        dst: &mut MatMut<f32>,
    ) {
        dbg_check_microkernel_inputs(self, lhs, rhs, dst);
        let kc = lhs.ncols();
        neon_8x8_microkernel_f32(
            kc,
            alpha,
            lhs.as_slice(),
            rhs.as_slice(),
            beta,
            dst.as_mut_slice(),
        );
    }

    fn gemm(
        &self,
        alpha: f32,
        a: MatRef<f32>,
        b: MatRef<f32>,
        beta: f32,
        c: &mut MatMut<f32>,
        pack_sizes: PackSizes,
        packing_buf: &mut [f32],
    ) {
        crate::gemm::gemm_with_tile(
            self,
            alpha,
            a,
            b,
            beta,
            c,
            pack_sizes,
            packing_buf,
            neon_8x8_tile_f32,
        );
    }
}

// Per-tile operation for the 8x8 f32 kernel: write a full in-bounds tile straight into c,
// fall back to the buffered path otherwise.
#[allow(clippy::too_many_arguments)]
fn neon_8x8_tile_f32(
    kernel: &NeonKernel8x8<f32>,
    alpha: f32,
    lhs_values: &[f32],
    rhs_values: &[f32],
    kc: usize,
    beta: f32,
    c: &mut MatMut<f32>,
    dst_rows: core::ops::Range<usize>,
    dst_cols: core::ops::Range<usize>,
    dst_buf: &mut [f32],
) {
    const DIM: usize = 8;
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    if dst_rows.len() == DIM
        && dst_cols.len() == DIM
        && dst_rows.end <= c.nrows()
        && dst_cols.end <= c.ncols()
        && ((csc == 1 && rsc >= DIM) || (rsc == 1 && csc >= DIM))
    {
        // `gemm_with_tile` hands out `mr * kc` values of apack and `kc * nr` of bpack, and
        // this op is only installed on the 8x8 kernel, so `kernel_direct` reads exactly the
        // 8 * kc values each packed slice holds.
        debug_assert_eq!(lhs_values.len(), DIM * kc);
        debug_assert_eq!(rhs_values.len(), DIM * kc);
        // Every index of the tile is <= (nrows - 1) * rsc + (ncols - 1) * csc, which
        // `MatBase::from_parts` has already checked to be inside the slice.
        let at = dst_rows.start * rsc + dst_cols.start * csc;
        let ld = if csc == 1 { rsc } else { csc };
        unsafe {
            let cptr = c.as_mut_slice().as_mut_ptr().add(at);
            prefetch_c_tile(cptr, ld);
            if csc == 1 {
                // accumulators are rows of the tile. This orientation makes the packed rhs
                // the fma's first product operand, so the NaN payload of a NaN * NaN product
                // follows b here and a in the buffered path (IEEE 754 leaves that choice
                // unspecified); every other result is bit-identical.
                let (x, y) = (lhs_values.as_ptr(), rhs_values.as_ptr());
                kernel_direct(kc, alpha, x, y, beta, cptr, ld);
            } else {
                // accumulators are columns of the tile
                let (x, y) = (rhs_values.as_ptr(), lhs_values.as_ptr());
                kernel_direct(kc, alpha, x, y, beta, cptr, ld);
            }
        }
        return;
    }
    crate::gemm::buffered_tile(
        kernel, alpha, lhs_values, rhs_values, kc, beta, c, dst_rows, dst_cols, dst_buf,
    );
}

// Hint the 8 lines of the c tile into L1 before the K-loop; the loads of c are issued
// only after ~kc fma steps, so at large n they are compulsory misses. One hint per line:
// a line is 32 bytes, so a line that straddles two cache lines gets only its first half
// hinted. The benefit is modelled, not measured (no hardware here; TCG ignores `prfm`).
#[inline]
unsafe fn prefetch_c_tile(c: *mut f32, ld: usize) {
    // `prfm` does not exist under miri/kani; evaluate the same addresses instead, so that
    // miri still bounds-checks every pointer the real code computes.
    #[cfg(any(kani, miri))]
    for v in 0..8 {
        let _ = core::ptr::read(c.add(v * ld));
    }

    #[cfg(not(any(kani, miri)))]
    for v in 0..8 {
        core::arch::asm!(
            "prfm pldl1keep, [{ptr}]",
            ptr = in(reg) c.add(v * ld),
            options(nostack, preserves_flags, readonly),
        );
    }
}

fn neon_8x8_microkernel_f32(
    kc: usize,
    alpha: f32,
    lhs: &[f32],
    rhs: &[f32],
    beta: f32,
    dst_colmajor: &mut [f32],
) {
    const DIM: usize = 8;
    assert_eq!(lhs.len(), rhs.len());
    assert_eq!(lhs.len(), DIM.checked_mul(kc).unwrap());
    assert_eq!(dst_colmajor.len(), DIM * DIM);

    unsafe {
        kernel_direct(
            kc,
            alpha,
            rhs.as_ptr(),
            lhs.as_ptr(),
            beta,
            dst_colmajor.as_mut_ptr(),
            DIM,
        )
    };
}

// acc[l] (4 contiguous elements along y's packed direction) = sum_k y_vec * x[l];
// the tile vector `l` lives at `c + l * ld + lane_base`.
#[inline]
unsafe fn kernel_direct(
    kc: usize,
    alpha: f32,
    x: *const f32,
    y: *const f32,
    beta: f32,
    c: *mut f32,
    ld: usize,
) {
    debug_assert!(ld >= 8);
    let (mut a, mut b) = (x, y);

    let mut ab11 = [vmovq_n_f32(0f32); 4];
    let mut ab12 = [vmovq_n_f32(0f32); 4];
    let mut ab21 = [vmovq_n_f32(0f32); 4];
    let mut ab22 = [vmovq_n_f32(0f32); 4];

    // Compute
    // ab_ij = a_i * b_j for all i, j
    macro_rules! prod {
        ($dest:ident, $av:expr, $bv:expr) => {
            $dest[0] = vfmaq_laneq_f32::<0>($dest[0], $bv, $av);
            $dest[1] = vfmaq_laneq_f32::<1>($dest[1], $bv, $av);
            $dest[2] = vfmaq_laneq_f32::<2>($dest[2], $bv, $av);
            $dest[3] = vfmaq_laneq_f32::<3>($dest[3], $bv, $av);
        };
    }

    for _ in 0..kc {
        let a1 = vld1q_f32(a);
        let b1 = vld1q_f32(b);
        let a2 = vld1q_f32(a.add(4));
        let b2 = vld1q_f32(b.add(4));

        prod!(ab11, a1, b1);
        prod!(ab12, a1, b2);
        prod!(ab21, a2, b1);
        prod!(ab22, a2, b2);

        a = a.add(8);
        b = b.add(8);
    }

    for i in 0..4 {
        ab11[i] = vmulq_n_f32(ab11[i], alpha);
        ab12[i] = vmulq_n_f32(ab12[i], alpha);
        ab21[i] = vmulq_n_f32(ab21[i], alpha);
        ab22[i] = vmulq_n_f32(ab22[i], alpha);
    }

    macro_rules! c {
        ($i:expr, $j:expr) => {
            c.add($i * ld + $j)
        };
    }

    let mut c11 = [vmovq_n_f32(0f32); 4];
    let mut c12 = [vmovq_n_f32(0f32); 4];
    let mut c21 = [vmovq_n_f32(0f32); 4];
    let mut c22 = [vmovq_n_f32(0f32); 4];
    for i in 0..4 {
        c11[i] = vld1q_f32(c![i, 0]);
        c12[i] = vld1q_f32(c![i, 4]);
        c21[i] = vld1q_f32(c![i + 4, 0]);
        c22[i] = vld1q_f32(c![i + 4, 4]);
    }

    let betav = vmovq_n_f32(beta);
    for i in 0..4 {
        ab11[i] = vfmaq_f32(ab11[i], c11[i], betav);
        ab12[i] = vfmaq_f32(ab12[i], c12[i], betav);
        ab21[i] = vfmaq_f32(ab21[i], c21[i], betav);
        ab22[i] = vfmaq_f32(ab22[i], c22[i], betav);
    }
    for i in 0..4 {
        vst1q_f32(c![i, 0], ab11[i]);
        vst1q_f32(c![i, 4], ab12[i]);
        vst1q_f32(c![i + 4, 0], ab21[i]);
        vst1q_f32(c![i + 4, 4], ab22[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::std_prelude::*;

    const DIM: usize = 8;

    // packed lhs: DIM x kc col-major; packed rhs: kc x DIM row-major
    fn packed(kc: usize) -> (Vec<f32>, Vec<f32>) {
        let mut lhs = vec![0f32; DIM * kc];
        let mut rhs = vec![0f32; kc * DIM];
        for k in 0..kc {
            for i in 0..DIM {
                lhs[k * DIM + i] = ((i * 3 + k * 5) % 17) as f32 - 8.0;
                rhs[k * DIM + i] = ((i * 7 + k * 11) % 13) as f32 - 6.0;
            }
        }
        (lhs, rhs)
    }

    // expect[i][j] = alpha * sum_k a[i][k] * b[k][j] + beta * c0[i][j]
    fn naive(kc: usize, alpha: f32, lhs: &[f32], rhs: &[f32], beta: f32, c0: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; DIM * DIM];
        for i in 0..DIM {
            for j in 0..DIM {
                let mut acc = 0f32;
                for k in 0..kc {
                    acc += lhs[k * DIM + i] * rhs[k * DIM + j];
                }
                out[i * DIM + j] = alpha * acc + beta * c0[i * DIM + j];
            }
        }
        out
    }

    #[test]
    fn test_kernel_direct_both_orientations() {
        let (alpha, beta) = (2f32, -3f32);
        for kc in [0usize, 1, 5] {
            for ld in [8usize, 11, 20] {
                let (lhs, rhs) = packed(kc);
                let c0: Vec<f32> = (0..DIM * DIM).map(|x| (x % 11) as f32 - 5.0).collect();
                let expect = naive(kc, alpha, &lhs, &rhs, beta, &c0);

                // row-major c: acc are rows, c[i][j] at i * ld + j
                let mut c = vec![0f32; DIM * ld];
                for i in 0..DIM {
                    for j in 0..DIM {
                        c[i * ld + j] = c0[i * DIM + j];
                    }
                }
                unsafe {
                    kernel_direct(
                        kc,
                        alpha,
                        lhs.as_ptr(),
                        rhs.as_ptr(),
                        beta,
                        c.as_mut_ptr(),
                        ld,
                    )
                };
                for i in 0..DIM {
                    for j in 0..DIM {
                        assert_eq!(
                            c[i * ld + j],
                            expect[i * DIM + j],
                            "row-major kc={kc} ld={ld} at ({i}, {j})"
                        );
                    }
                }

                // col-major c: acc are columns, c[i][j] at j * ld + i
                let mut c = vec![0f32; DIM * ld];
                for i in 0..DIM {
                    for j in 0..DIM {
                        c[j * ld + i] = c0[i * DIM + j];
                    }
                }
                unsafe {
                    kernel_direct(
                        kc,
                        alpha,
                        rhs.as_ptr(),
                        lhs.as_ptr(),
                        beta,
                        c.as_mut_ptr(),
                        ld,
                    )
                };
                for i in 0..DIM {
                    for j in 0..DIM {
                        assert_eq!(
                            c[j * ld + i],
                            expect[i * DIM + j],
                            "col-major kc={kc} ld={ld} at ({i}, {j})"
                        );
                    }
                }
            }
        }
    }

    // The direct path must actually be taken, not merely agree with the buffered one:
    // it never touches dst_buf, while the fallback fills dst_buf from c.
    #[test]
    fn test_tile_takes_direct_path() {
        let kernel = if cfg!(target_feature = "neon") {
            unsafe { NeonKernel8x8::<f32>::new() }
        } else {
            println!("neon feature is not supported");
            return;
        };
        const SENTINEL: f32 = 123.0;
        let kc = 3;
        let (lhs, rhs) = packed(kc);

        // returns true if dst_buf was left untouched, i.e. the direct path ran
        let run = |rsc: usize, csc: usize| -> bool {
            let len = (DIM - 1) * rsc + (DIM - 1) * csc + 1;
            let mut values = vec![1f32; len];
            let mut c = MatMut::from_parts(DIM, DIM, values.as_mut_slice(), rsc, csc).unwrap();
            let mut dst_buf = vec![SENTINEL; DIM * DIM];
            neon_8x8_tile_f32(
                &kernel,
                1.0,
                &lhs,
                &rhs,
                kc,
                1.0,
                &mut c,
                0..DIM,
                0..DIM,
                &mut dst_buf,
            );
            dst_buf.iter().all(|&x| x == SENTINEL)
        };

        assert!(run(DIM, 1), "row-major c must take the direct path");
        assert!(run(1, DIM), "col-major c must take the direct path");
        assert!(!run(1, 4), "aliasing strides must take the buffered path");
    }
}
