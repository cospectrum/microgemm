use super::NeonKernel8x8;

use crate::{kernels::dbg_check_microkernel_inputs, typenum::U8, Kernel, MatMut, MatRef};

use super::super::simd::*;

impl Kernel for NeonKernel8x8<f32> {
    type Scalar = f32;
    type Mr = U8;
    type Nr = U8;

    #[inline(always)]
    fn microkernel(
        &self,
        alpha: f32,
        lhs: MatRef<f32>,
        rhs: MatRef<f32>,
        beta: f32,
        dst: &mut MatMut<f32>,
    ) {
        dbg_check_microkernel_inputs(self, lhs, rhs, dst);
        const DIM: usize = 8;
        let kc = lhs.ncols();
        let (rsc, csc) = (dst.row_stride(), dst.col_stride());
        if (csc == 1 && rsc >= DIM) || (rsc == 1 && csc >= DIM) {
            kernel_direct(
                kc,
                alpha,
                lhs.as_slice(),
                rhs.as_slice(),
                beta,
                dst.as_mut_slice(),
                rsc,
                csc,
            );
        } else {
            buffered_microkernel(kc, alpha, lhs, rhs, beta, dst);
        }
    }
}

// Keep scratch storage for uncommon output layouts off the direct-write path.
#[cold]
#[inline(never)]
fn buffered_microkernel(
    kc: usize,
    alpha: f32,
    lhs: MatRef<f32>,
    rhs: MatRef<f32>,
    beta: f32,
    dst: &mut MatMut<f32>,
) {
    const DIM: usize = 8;
    // Snapshot C before computing, including when coordinates overlap.
    let mut packed = [0.0; DIM * DIM];
    crate::packing::registers_from_c(&mut packed, dst.to_ref(), 0..DIM, 0..DIM);
    kernel_direct(
        kc,
        alpha,
        lhs.as_slice(),
        rhs.as_slice(),
        beta,
        &mut packed,
        1,
        DIM,
    );
    crate::packing::registers_to_c(&packed, dst, 0..DIM, 0..DIM);
}

// acc[l] (4 contiguous elements along y's packed direction) = sum_k y_vec * x[l];
// the tile vector `l` lives at `c + l * ld + lane_base`.
/// C's row and column strides are measured in `f32` elements. One stride must
/// be 1; the other is the leading dimension (`ld`) and must be at least 8.
/// Tiles retain the parent matrix's leading dimension, including any padding.
#[inline]
#[allow(clippy::too_many_arguments)]
fn kernel_direct(
    kc: usize,
    alpha: f32,
    lhs: &[f32],
    rhs: &[f32],
    beta: f32,
    c: &mut [f32],
    rsc: usize,
    csc: usize,
) {
    // Rows of accumulators for row-major C, columns for column-major C.
    // Swapping operands can change which NaN payload survives a NaN product.
    let (x, y, ld) = if csc == 1 {
        (lhs, rhs, rsc)
    } else {
        assert_eq!(rsc, 1);
        (rhs, lhs, csc)
    };
    const DIM: usize = 8;
    let packed_len = DIM.checked_mul(kc).expect("packed panel length overflow");
    assert_eq!(x.len(), packed_len);
    assert_eq!(y.len(), packed_len);
    assert!(ld >= DIM);
    let tile_len = (DIM - 1)
        .checked_mul(ld)
        .and_then(|n| n.checked_add(DIM))
        .expect("C tile length overflow");
    assert!(c.len() >= tile_len);

    // SAFETY: the checks above cover every packed load and C load/store, including
    // pointer increments to one-past-the-end. ld >= DIM makes the tile's lines
    // disjoint. The slices keep C exclusively borrowed and separate from x/y.
    // NEON is guaranteed by the kernel's constructor.
    unsafe {
        kernel_direct_unchecked(kc, alpha, x.as_ptr(), y.as_ptr(), beta, c.as_mut_ptr(), ld);
    }
}

/// # Safety
/// NEON must be available. x and y must each point to 8 * kc readable f32 values.
/// c must point to 7 * ld + 8 readable/writable f32 values, exclusive of x/y,
/// with ld >= 8. All spans must fit their allocations without integer overflow.
#[inline]
unsafe fn kernel_direct_unchecked(
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

    #[test]
    fn public_microkernel_respects_arbitrary_output_strides() {
        if !cfg!(target_feature = "neon") {
            return;
        }
        // SAFETY: NEON is enabled for this target.
        let kernel = unsafe { NeonKernel8x8::<f32>::new() };
        const KC: usize = 3;
        let lhs: Vec<f32> = (0..DIM * KC).map(|i| (i % 5) as f32 - 2.0).collect();
        let rhs: Vec<f32> = (0..DIM * KC).map(|i| (i % 7) as f32 - 3.0).collect();
        for (rs, cs) in [
            (DIM, 1),
            (1, DIM),
            (DIM + 3, 1),
            (1, DIM + 3),
            (2, 2 * DIM + 3),
            (3, 2),
            (1, 1),
            (0, 1),
            (1, 0),
            (0, 0),
        ] {
            let span = (DIM - 1) * (rs + cs) + 1;
            let mut actual: Vec<f32> = (0..span + 4).map(|i| (i % 13) as f32 + 1.0).collect();
            let before = actual.clone();
            let mut expected = before.clone();
            // Every logical cell reads the original C; column-major scatter
            // determines which result survives when destinations overlap.
            for col in 0..DIM {
                for row in 0..DIM {
                    let at = 2 + row * rs + col * cs;
                    let sum: f32 = (0..KC)
                        .map(|k| lhs[k * DIM + row] * rhs[k * DIM + col])
                        .sum();
                    expected[at] = 2.0 * sum - 3.0 * before[at];
                }
            }
            kernel.microkernel(
                2.0,
                MatRef::col_major(DIM, KC, &lhs),
                MatRef::row_major(KC, DIM, &rhs),
                -3.0,
                &mut MatMut::from_parts(DIM, DIM, &mut actual[2..2 + span], rs, cs).unwrap(),
            );
            assert_eq!(actual, expected, "strides ({rs}, {cs})");
        }
    }

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
                kernel_direct(kc, alpha, &lhs, &rhs, beta, &mut c, ld, 1);
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
                kernel_direct(kc, alpha, &rhs, &lhs, beta, &mut c, ld, 1);
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

    #[test]
    #[should_panic]
    fn test_kernel_direct_rejects_short_panel() {
        kernel_direct(1, 1.0, &[0.0; 8], &[0.0; 7], 0.0, &mut [0.0; 64], 8, 1);
    }

    #[test]
    #[should_panic(expected = "c.len() >= tile_len")]
    fn test_kernel_direct_rejects_short_destination() {
        kernel_direct(1, 1.0, &[0.0; 8], &[0.0; 8], 0.0, &mut [0.0; 63], 8, 1);
    }

    #[test]
    #[should_panic(expected = "C tile length overflow")]
    fn test_kernel_direct_rejects_stride_overflow() {
        kernel_direct(
            1,
            1.0,
            &[0.0; 8],
            &[0.0; 8],
            0.0,
            &mut [0.0; 64],
            usize::MAX,
            1,
        );
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
            crate::gemm::direct_tile(
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

#[cfg(test)]
mod safety_tests {
    use super::kernel_direct;
    use crate::std_prelude::*;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            // Miri's isolation intentionally denies filesystem persistence.
            #[cfg(miri)]
            failure_persistence: None,
            ..ProptestConfig::default()
        })]

        #[test]
        fn strided_writes_preserve_padding_and_allocation_guards(
            kc in 0usize..=12,
            ld in 8usize..=24,
            offset in 0usize..=3,
            values in prop::collection::vec(-4i16..=4, 192),
            alpha in -3i16..=3,
            beta in -3i16..=3,
        ) {
            let x: Vec<f32> = values[..8 * kc].iter().copied().map(f32::from).collect();
            let y: Vec<f32> = values[96..96 + 8 * kc].iter().copied().map(f32::from).collect();
            let len = 7 * ld + 8;
            let mut c = vec![65536.0; offset + len + 3];
            for row in 0..8 {
                for col in 0..8 {
                    c[offset + row * ld + col] = (row + col) as f32;
                }
            }
            let before = c.clone();
            let (alpha, beta) = (f32::from(alpha), f32::from(beta));
            kernel_direct(kc, alpha, &x, &y, beta, &mut c[offset..offset + len], ld, 1);
            for i in 0..c.len() {
                let expected = if i >= offset && i < offset + len && (i - offset) % ld < 8 {
                    let (row, col) = ((i - offset) / ld, (i - offset) % ld);
                    let sum: f32 = (0..kc).map(|k| x[8 * k + row] * y[8 * k + col]).sum();
                    alpha * sum + beta * before[i]
                } else {
                    before[i]
                };
                prop_assert_eq!(c[i], expected, "at {} for kc={}, ld={}, offset={}", i, kc, ld, offset);
            }
        }
    }

    #[test]
    #[should_panic]
    fn rejects_short_first_panel() {
        kernel_direct(1, 1.0, &[0.0; 7], &[0.0; 8], 0.0, &mut [0.0; 64], 8, 1);
    }

    #[test]
    #[should_panic(expected = "ld >= DIM")]
    fn rejects_overlapping_destination_lines() {
        kernel_direct(1, 1.0, &[0.0; 8], &[0.0; 8], 0.0, &mut [0.0; 64], 7, 1);
    }

    #[test]
    #[should_panic]
    fn rejects_destination_without_contiguous_axis() {
        kernel_direct(1, 1.0, &[0.0; 8], &[0.0; 8], 0.0, &mut [0.0; 134], 2, 17);
    }

    #[test]
    #[should_panic(expected = "packed panel length overflow")]
    fn rejects_panel_length_overflow() {
        kernel_direct(usize::MAX / 8 + 1, 1.0, &[], &[], 0.0, &mut [0.0; 64], 8, 1);
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;
    // These imports are referenced by Kani attributes, not ordinary Rust calls.
    #[allow(unused_imports)]
    use crate::kernels::neon::simd_mock::{self, memory_arithmetic as arithmetic};

    // Constant panel extents and strides keep each memory proof small. Allocations
    // end exactly at each input panel and destination span; offset slices also
    // exercise pointers without vector alignment. Arithmetic alone is abstracted.
    fn memory_case<const PANEL: usize, const DEST: usize, const LD: usize>() {
        let kc = (PANEL - 1) / 8;
        let lhs: [f32; PANEL] = kani::any();
        let rhs: [f32; PANEL] = kani::any();
        let mut dst: [f32; DEST] = kani::any();
        let index: usize = kani::any_where(|&i| i < DEST);
        let before = dst[index].to_bits();
        kernel_direct(
            kc,
            kani::any(),
            &lhs[1..],
            &rhs[1..],
            kani::any(),
            &mut dst[1..],
            LD,
            1,
        );
        if index == 0 || (index - 1) % LD >= 8 {
            assert_eq!(dst[index].to_bits(), before);
        }
    }

    macro_rules! memory_proof {
        ($name:ident, $kc:literal, $ld:literal) => {
            #[kani::proof]
            #[kani::unwind(5)]
            #[kani::stub(simd_mock::vaddq_f32, arithmetic::add)]
            #[kani::stub(simd_mock::vmulq_n_f32, arithmetic::scale)]
            #[kani::stub(simd_mock::fma_scalar, arithmetic::scalar_fma)]
            #[kani::stub(simd_mock::vfmaq_f32, arithmetic::fma)]
            fn $name() {
                memory_case::<{ 8 * $kc + 1 }, { 7 * $ld + 9 }, $ld>();
            }
        };
    }

    memory_proof!(memory_depth_0_stride_8, 0, 8);
    memory_proof!(memory_depth_1_stride_8, 1, 8);
    memory_proof!(memory_depth_2_stride_8, 2, 8);
    memory_proof!(memory_depth_3_stride_8, 3, 8);
    memory_proof!(memory_depth_0_stride_11, 0, 11);
    memory_proof!(memory_depth_1_stride_11, 1, 11);
    memory_proof!(memory_depth_2_stride_11, 2, 11);
    memory_proof!(memory_depth_3_stride_11, 3, 11);
    memory_proof!(memory_depth_0_stride_12, 0, 12);
    memory_proof!(memory_depth_1_stride_12, 1, 12);
    memory_proof!(memory_depth_2_stride_12, 2, 12);
    memory_proof!(memory_depth_3_stride_12, 3, 12);

    // Compose the memory proofs above with the caller: assert the checked
    // kernel's preconditions and overapproximate values in its write footprint.
    // Raw loads/stores are verified separately, never stubbed in memory_case.
    #[allow(dead_code)]
    fn checked_kernel_footprint(
        kc: usize,
        _: f32,
        x: &[f32],
        y: &[f32],
        _: f32,
        c: &mut [f32],
        rsc: usize,
        csc: usize,
    ) {
        let packed = 8usize.checked_mul(kc).unwrap();
        assert_eq!(x.len(), packed);
        assert_eq!(y.len(), packed);
        assert!((rsc == 1 && csc >= 8) || (csc == 1 && rsc >= 8));
        let span = (7usize.checked_mul(rsc).unwrap())
            .checked_add(7usize.checked_mul(csc).unwrap())
            .unwrap()
            .checked_add(1)
            .unwrap();
        assert!(c.len() >= span);
        for row in 0..8 {
            for col in 0..8 {
                c[row * rsc + col * csc] = kani::any();
            }
        }
    }

    /// Exercise the real tile dispatcher in both orientations, at a nonzero tile
    /// origin and with padding between rows/columns. Arithmetic is abstracted;
    /// native tests check numerical orientation. Guards and scratch are checked.
    fn direct_tile_case(kernel: &NeonKernel8x8<f32>, row_major: bool) {
        let (rs, cs) = if row_major { (12, 1) } else { (1, 12) };
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rhs = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let mut storage = [4.0; 107];
        let mut c = MatMut::from_parts(9, 9, &mut storage[1..106], rs, cs).unwrap();
        let mut scratch = [123.0; 64];
        crate::gemm::direct_tile(
            kernel,
            2.0,
            &lhs,
            &rhs,
            1,
            -3.0,
            &mut c,
            1..9,
            1..9,
            &mut scratch,
        );
        let index: usize = kani::any_where(|&i| i < storage.len());
        if index == 0
            || !(1..=8).contains(&((index - 1) / 12))
            || !(1..=8).contains(&((index - 1) % 12))
        {
            assert_eq!(storage[index], 4.0);
        }
        let index = kani::any_where(|&i: &usize| i < 64);
        assert_eq!(scratch[index], 123.0);
    }

    /// Both a full tile and a ragged tile must stage C for a layout with neither
    /// stride equal to one. C packing uses the checked kernel footprint above.
    fn buffered_tile_case(kernel: &NeonKernel8x8<f32>, dim: usize) {
        let lhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let rhs = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let mut storage = [4.0; 129];
        let mut c = MatMut::from_parts(dim, dim, &mut storage[1..128], 16, 2).unwrap();
        crate::gemm::direct_tile(
            kernel,
            1.0,
            &lhs,
            &rhs,
            1,
            1.0,
            &mut c,
            0..8,
            0..8,
            &mut [123.0; 64],
        );
        let index: usize = kani::any_where(|&i| i < storage.len());
        if index == 0
            || (index - 1) / 16 >= dim
            || (index - 1) % 16 / 2 >= dim
            || (index - 1) % 2 != 0
        {
            assert_eq!(storage[index], 4.0);
        }
    }

    // Exercise the public fallback with arbitrary float bits. Its real gather
    // and scatter stay enabled; only the checked arithmetic footprint is modeled.
    fn public_fallback_case<const RS: usize, const CS: usize, const LEN: usize>() {
        let lhs: [f32; 8] = kani::any();
        let rhs: [f32; 8] = kani::any();
        let mut storage: [f32; LEN] = kani::any();
        let index: usize = kani::any_where(|&i| i < LEN);
        let before = storage[index].to_bits();
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel8x8::new() };
        kernel.microkernel(
            kani::any(),
            MatRef::col_major(8, 1, &lhs),
            MatRef::row_major(1, 8, &rhs),
            kani::any(),
            &mut MatMut::from_parts(8, 8, &mut storage[1..LEN - 1], RS, CS).unwrap(),
        );
        let mut touched = false;
        for row in 0..8 {
            for col in 0..8 {
                touched |= index == 1 + row * RS + col * CS;
            }
        }
        if !touched {
            assert_eq!(storage[index].to_bits(), before);
        }
    }

    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(super::kernel_direct, checked_kernel_footprint)]
    fn public_strided_output_preserves_guards() {
        public_fallback_case::<2, 19, 150>();
    }

    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(super::kernel_direct, checked_kernel_footprint)]
    fn public_aliased_output_preserves_guards() {
        public_fallback_case::<0, 0, 3>();
    }

    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(super::kernel_direct, checked_kernel_footprint)]
    fn direct_row_major_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel8x8::new() };
        direct_tile_case(&kernel, true);
    }

    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(super::kernel_direct, checked_kernel_footprint)]
    fn direct_col_major_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel8x8::new() };
        direct_tile_case(&kernel, false);
    }

    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(super::kernel_direct, checked_kernel_footprint)]
    fn buffered_strided_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel8x8::new() };
        buffered_tile_case(&kernel, 8);
    }

    #[kani::proof]
    #[kani::unwind(9)]
    #[kani::stub(super::kernel_direct, checked_kernel_footprint)]
    fn buffered_ragged_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel8x8::new() };
        buffered_tile_case(&kernel, 3);
    }
}
