use super::NeonKernel4x4;
use crate::{kernels::dbg_check_microkernel_inputs, typenum::U4, Kernel, MatMut, MatRef};

use super::super::simd::*;

impl Kernel for NeonKernel4x4<f32> {
    type Scalar = f32;
    type Mr = U4;
    type Nr = U4;

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
        const DIM: usize = 4;
        let kc = lhs.ncols();
        let (rsc, csc) = (dst.row_stride(), dst.col_stride());
        if (csc == 1 && rsc >= DIM) || (rsc == 1 && csc >= DIM) {
            direct_microkernel(
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
    const DIM: usize = 4;
    // Snapshot C before computing, including when coordinates overlap.
    let mut packed = [0.0; DIM * DIM];
    crate::packing::registers_from_c(&mut packed, dst.to_ref(), 0..DIM, 0..DIM);
    direct_microkernel(
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

/// C's row and column strides are measured in `f32` elements. One stride must
/// be 1; the other is the leading dimension (`ld`) and must be at least 4.
/// Tiles retain the parent matrix's leading dimension, including any padding.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn direct_microkernel(
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
    const DIM: usize = 4;
    let packed_len = DIM.checked_mul(kc).expect("packed panel length overflow");
    assert_eq!(x.len(), packed_len);
    assert_eq!(y.len(), packed_len);
    assert!(ld >= DIM);
    let tile_len = (DIM - 1)
        .checked_mul(ld)
        .and_then(|n| n.checked_add(DIM))
        .expect("C tile length overflow");
    assert!(c.len() >= tile_len);

    // SAFETY: panel lengths cover all loads and one-past-the-end increments.
    // The output span is checked, ld >= DIM keeps columns disjoint, and slices
    // provide exclusive output access. The constructor guarantees NEON support.
    unsafe { inner(kc, alpha, y.as_ptr(), x.as_ptr(), beta, c, ld) };

    unsafe fn inner(
        kc: usize,
        alpha: f32,
        mut left: *const f32,
        mut right: *const f32,
        beta: f32,
        dst: &mut [f32],
        ld: usize,
    ) {
        let mut cols0 = [vmovq_n_f32(0f32); 4];
        let mut cols1 = [vmovq_n_f32(0f32); 4];
        let mut cols2 = [vmovq_n_f32(0f32); 4];
        let mut cols3 = [vmovq_n_f32(0f32); 4];

        macro_rules! accum_lane {
            ($cols:ident, $a:ident, $b:ident) => {
                $cols[0] = vfmaq_laneq_f32::<0>($cols[0], $a, $b);
                $cols[1] = vfmaq_laneq_f32::<1>($cols[1], $a, $b);
                $cols[2] = vfmaq_laneq_f32::<2>($cols[2], $a, $b);
                $cols[3] = vfmaq_laneq_f32::<3>($cols[3], $a, $b);
            };
        }

        for _ in 0..kc / 4 {
            let a = vld1q_f32(left);
            let b = vld1q_f32(right);
            accum_lane!(cols0, a, b);

            let a = vld1q_f32(left.add(4));
            let b = vld1q_f32(right.add(4));
            accum_lane!(cols1, a, b);

            let a = vld1q_f32(left.add(8));
            let b = vld1q_f32(right.add(8));
            accum_lane!(cols2, a, b);

            let a = vld1q_f32(left.add(12));
            let b = vld1q_f32(right.add(12));
            accum_lane!(cols3, a, b);

            left = left.add(16);
            right = right.add(16);
        }

        for _ in 0..kc % 4 {
            let a = vld1q_f32(left);
            let b = vld1q_f32(right);
            accum_lane!(cols0, a, b);

            left = left.add(4);
            right = right.add(4);
        }

        for row in 0..4 {
            let sum = vaddq_f32(
                vaddq_f32(cols0[row], cols1[row]),
                vaddq_f32(cols2[row], cols3[row]),
            );
            cols0[row] = vmulq_n_f32(sum, alpha);
        }

        for (col, from) in cols0.into_iter().enumerate() {
            let to = &mut dst[col * ld..col * ld + 4];
            let mut tmp = [0f32; 4];
            vst1q_f32(tmp.as_mut_ptr(), from);
            for (y, x) in to.iter_mut().zip(tmp) {
                *y = scale_add(x, beta, *y);
            }
        }
    }
}

// Kept separate so memory-safety proofs can abstract arithmetic without
// replacing any pointer access, loop, lane selection, or bounds check.
#[inline]
fn scale_add(x: f32, beta: f32, y: f32) -> f32 {
    x + beta * y
}

#[cfg(test)]
mod tests {
    use super::*;

    const DIM: usize = 4;
    use crate::std_prelude::*;

    #[test]
    fn public_microkernel_respects_arbitrary_output_strides() {
        if !cfg!(target_feature = "neon") {
            return;
        }
        // SAFETY: NEON is enabled for this target.
        let kernel = unsafe { NeonKernel4x4::<f32>::new() };
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

    #[test]
    fn unrolled_and_remainder_loops_preserve_guards() {
        // Every remainder, the empty dot product, and one/two unrolled groups.
        for kc in 0..=9 {
            let lhs: Vec<f32> = (0..4 * kc).map(|i| (i % 7) as f32 - 3.0).collect();
            let rhs: Vec<f32> = (0..4 * kc).map(|i| (i % 11) as f32 - 5.0).collect();
            let mut dst = [12345.0; 18];
            dst[1..17].fill(2.0);
            direct_microkernel(kc, 2.0, &lhs, &rhs, -3.0, &mut dst[1..17], 1, 4);
            assert_eq!(dst[0], 12345.0);
            assert_eq!(dst[17], 12345.0);
            for col in 0..4 {
                for row in 0..4 {
                    let sum: f32 = (0..kc).map(|k| lhs[4 * k + row] * rhs[4 * k + col]).sum();
                    assert_eq!(
                        dst[1 + 4 * col + row],
                        2.0 * sum - 6.0,
                        "kc={kc}, row={row}, col={col}"
                    );
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn rejects_mismatched_panels() {
        direct_microkernel(1, 1.0, &[0.0; 4], &[0.0; 3], 0.0, &mut [0.0; 16], 1, 4);
    }

    #[test]
    #[should_panic]
    fn rejects_panels_short_for_depth() {
        direct_microkernel(2, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut [0.0; 16], 1, 4);
    }

    #[test]
    #[should_panic]
    fn rejects_short_destination() {
        direct_microkernel(1, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut [0.0; 15], 1, 4);
    }

    #[test]
    #[should_panic]
    fn rejects_panel_length_overflow() {
        direct_microkernel(usize::MAX / 4 + 1, 1.0, &[], &[], 0.0, &mut [0.0; 16], 1, 4);
    }

    // Full tiles must bypass the shared scratch buffer, even if the kernel
    // itself needs temporary storage for an unsupported output layout.
    #[test]
    fn test_tile_takes_direct_path() {
        let kernel = if cfg!(target_feature = "neon") {
            unsafe { NeonKernel4x4::<f32>::new() }
        } else {
            println!("neon feature is not supported");
            return;
        };
        const SENTINEL: f32 = 123.0;
        let kc = 3;
        let lhs = [1.0; 12];
        let rhs = [2.0; 12];

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
        assert!(run(1, 2), "full aliasing tiles bypass shared buffering");
        assert!(
            run(2, 2 * DIM + 1),
            "full nonunit tiles bypass shared buffering"
        );
        assert!(run(0, 0), "full broadcast tiles bypass shared buffering");
    }
}

#[cfg(test)]
mod safety_tests {
    use super::direct_microkernel;
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
            ld in 4usize..=12,
            offset in 0usize..=3,
            values in prop::collection::vec(-4i16..=4, 96),
            alpha in -3i16..=3,
            beta in -3i16..=3,
        ) {
            let x: Vec<f32> = values[..4 * kc].iter().copied().map(f32::from).collect();
            let y: Vec<f32> = values[48..48 + 4 * kc].iter().copied().map(f32::from).collect();
            let len = 3 * ld + 4;
            let mut c = vec![65536.0; offset + len + 3];
            for row in 0..4 {
                for col in 0..4 {
                    c[offset + row * ld + col] = (row + col) as f32;
                }
            }
            let before = c.clone();
            let (alpha, beta) = (f32::from(alpha), f32::from(beta));
            direct_microkernel(kc, alpha, &x, &y, beta, &mut c[offset..offset + len], ld, 1);
            for i in 0..c.len() {
                let expected = if i >= offset && i < offset + len && (i - offset) % ld < 4 {
                    let (row, col) = ((i - offset) / ld, (i - offset) % ld);
                    let sum: f32 = (0..kc).map(|k| x[4 * k + row] * y[4 * k + col]).sum();
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
        direct_microkernel(1, 1.0, &[0.0; 3], &[0.0; 4], 0.0, &mut [0.0; 16], 4, 1);
    }

    #[test]
    #[should_panic(expected = "ld >= DIM")]
    fn rejects_overlapping_destination_lines() {
        direct_microkernel(1, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut [0.0; 16], 3, 1);
    }

    #[test]
    #[should_panic]
    fn rejects_destination_without_contiguous_axis() {
        direct_microkernel(1, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut [0.0; 34], 2, 9);
    }

    #[test]
    #[should_panic(expected = "packed panel length overflow")]
    fn rejects_panel_length_overflow() {
        direct_microkernel(usize::MAX / 4 + 1, 1.0, &[], &[], 0.0, &mut [0.0; 16], 4, 1);
    }

    #[test]
    #[should_panic(expected = "c.len() >= tile_len")]
    fn rejects_short_destination_span() {
        direct_microkernel(1, 1.0, &[0.0; 4], &[0.0; 4], 0.0, &mut [0.0; 24], 7, 1);
    }

    #[test]
    #[should_panic(expected = "C tile length overflow")]
    fn rejects_destination_stride_overflow() {
        direct_microkernel(
            1,
            1.0,
            &[0.0; 4],
            &[0.0; 4],
            0.0,
            &mut [0.0; 16],
            usize::MAX,
            1,
        );
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;
    // These imports are referenced by Kani attributes, not ordinary Rust calls.
    #[allow(unused_imports)]
    use crate::kernels::neon::simd_mock::{self, memory_arithmetic as arithmetic};

    // Each depth has its own proof: constant extents let CBMC simplify pointer
    // bounds before solving. The allocation ends exactly at the panel boundary,
    // so a read beyond a panel cannot hide in unused backing-array capacity.
    fn memory_case<const LEN: usize, const DEST: usize, const LD: usize>() {
        let kc = (LEN - 1) / 4;
        let lhs: [f32; LEN] = kani::any();
        let rhs: [f32; LEN] = kani::any();
        let mut dst: [f32; DEST] = kani::any();
        let index: usize = kani::any_where(|&i| i < DEST);
        let before = dst[index].to_bits();
        direct_microkernel(
            kc,
            kani::any(),
            &lhs[1..],
            &rhs[1..],
            kani::any(),
            &mut dst[1..DEST - 1],
            LD,
            1,
        );
        if index == 0 || index == DEST - 1 || (index - 1) % LD >= 4 {
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
            #[kani::stub(super::scale_add, arbitrary_scale_add)]
            fn $name() {
                memory_case::<{ 4 * $kc + 1 }, { 3 * $ld + 6 }, $ld>();
            }
        };
    }

    // Zero depth, two unrolled groups, and every remainder after each group.
    memory_proof!(memory_depth_0, 0, 4);
    memory_proof!(memory_depth_1, 1, 4);
    memory_proof!(memory_depth_2, 2, 4);
    memory_proof!(memory_depth_3, 3, 4);
    memory_proof!(memory_depth_4, 4, 4);
    memory_proof!(memory_depth_5, 5, 4);
    memory_proof!(memory_depth_6, 6, 4);
    memory_proof!(memory_depth_7, 7, 4);
    memory_proof!(memory_depth_8, 8, 4);
    memory_proof!(memory_depth_9, 9, 4);

    memory_proof!(memory_depth_1_stride_7, 1, 7);
    memory_proof!(memory_depth_5_stride_7, 5, 7);
    memory_proof!(memory_depth_9_stride_8, 9, 8);

    /// No arithmetic stubs: all sign patterns over {-1, 1}. These operations
    /// are exact in both the scalar mock and real fused NEON arithmetic.
    #[kani::proof]
    #[kani::unwind(5)]
    fn exact_single_outer_product() {
        let left: [bool; 4] = kani::any();
        let right: [bool; 4] = kani::any();
        let mut lhs = [0.0; 4];
        let mut rhs = [0.0; 4];
        for i in 0..4 {
            lhs[i] = if left[i] { 1.0 } else { -1.0 };
            rhs[i] = if right[i] { 1.0 } else { -1.0 };
        }
        let mut dst = [1.0; 16];
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel4x4::new() };
        kernel.microkernel(
            1.0,
            MatRef::col_major(4, 1, &lhs[..]),
            MatRef::row_major(1, 4, &rhs[..]),
            -1.0,
            &mut MatMut::col_major(4, 4, &mut dst[..]),
        );
        let row: usize = kani::any_where(|&i| i < 4);
        let col: usize = kani::any_where(|&i| i < 4);
        let expected = if left[row] == right[col] { 0.0 } else { -2.0 };
        assert_eq!(dst[col * 4 + row], expected);
    }

    // No arithmetic input restrictions: any f32 result overapproximates the epilogue.
    #[allow(dead_code)]
    fn arbitrary_scale_add(_: f32, _: f32, _: f32) -> f32 {
        kani::any()
    }

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
        let packed = 4usize.checked_mul(kc).unwrap();
        assert_eq!(x.len(), packed);
        assert_eq!(y.len(), packed);
        assert!((rsc == 1 && csc >= 4) || (csc == 1 && rsc >= 4));
        let span = (3usize.checked_mul(rsc).unwrap())
            .checked_add(3usize.checked_mul(csc).unwrap())
            .unwrap()
            .checked_add(1)
            .unwrap();
        assert!(c.len() >= span);
        for row in 0..4 {
            for col in 0..4 {
                c[row * rsc + col * csc] = kani::any();
            }
        }
    }

    /// Exercise the real tile dispatcher in both orientations, at a nonzero tile
    /// origin and with padding between rows/columns. Arithmetic is abstracted;
    /// native tests check numerical orientation. Guards and scratch are checked.
    fn direct_tile_case(kernel: &NeonKernel4x4<f32>, row_major: bool) {
        let (rs, cs) = if row_major { (12, 1) } else { (1, 12) };
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [11.0, 12.0, 13.0, 14.0];
        let mut storage = [4.0; 55];
        let mut c = MatMut::from_parts(5, 5, &mut storage[1..54], rs, cs).unwrap();
        let mut scratch = [123.0; 16];
        crate::gemm::direct_tile(
            kernel,
            2.0,
            &lhs,
            &rhs,
            1,
            -3.0,
            &mut c,
            1..5,
            1..5,
            &mut scratch,
        );
        let index: usize = kani::any_where(|&i| i < storage.len());
        if index == 0
            || !(1..=4).contains(&((index - 1) / 12))
            || !(1..=4).contains(&((index - 1) % 12))
        {
            assert_eq!(storage[index], 4.0);
        }
        let index = kani::any_where(|&i: &usize| i < 16);
        assert_eq!(scratch[index], 123.0);
    }

    /// Both a full tile and a ragged tile must stage C for a layout with neither
    /// stride equal to one. C packing uses the checked kernel footprint above.
    fn buffered_tile_case(kernel: &NeonKernel4x4<f32>, dim: usize) {
        let lhs = [1.0, 2.0, 3.0, 4.0];
        let rhs = [11.0, 12.0, 13.0, 14.0];
        let mut storage = [4.0; 65];
        let mut c = MatMut::from_parts(dim, dim, &mut storage[1..64], 16, 2).unwrap();
        crate::gemm::direct_tile(
            kernel,
            1.0,
            &lhs,
            &rhs,
            1,
            1.0,
            &mut c,
            0..4,
            0..4,
            &mut [123.0; 16],
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
        let lhs: [f32; 4] = kani::any();
        let rhs: [f32; 4] = kani::any();
        let mut storage: [f32; LEN] = kani::any();
        let index: usize = kani::any_where(|&i| i < LEN);
        let before = storage[index].to_bits();
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel4x4::new() };
        kernel.microkernel(
            kani::any(),
            MatRef::col_major(4, 1, &lhs),
            MatRef::row_major(1, 4, &rhs),
            kani::any(),
            &mut MatMut::from_parts(4, 4, &mut storage[1..LEN - 1], RS, CS).unwrap(),
        );
        let mut touched = false;
        for row in 0..4 {
            for col in 0..4 {
                touched |= index == 1 + row * RS + col * CS;
            }
        }
        if !touched {
            assert_eq!(storage[index].to_bits(), before);
        }
    }

    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(super::direct_microkernel, checked_kernel_footprint)]
    fn public_strided_output_preserves_guards() {
        public_fallback_case::<2, 11, 42>();
    }

    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(super::direct_microkernel, checked_kernel_footprint)]
    fn public_aliased_output_preserves_guards() {
        public_fallback_case::<0, 0, 3>();
    }

    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(super::direct_microkernel, checked_kernel_footprint)]
    fn direct_row_major_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel4x4::new() };
        direct_tile_case(&kernel, true);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(super::direct_microkernel, checked_kernel_footprint)]
    fn direct_col_major_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel4x4::new() };
        direct_tile_case(&kernel, false);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(super::direct_microkernel, checked_kernel_footprint)]
    fn buffered_strided_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel4x4::new() };
        buffered_tile_case(&kernel, 4);
    }

    #[kani::proof]
    #[kani::unwind(5)]
    #[kani::stub(super::direct_microkernel, checked_kernel_footprint)]
    fn buffered_ragged_tile() {
        // SAFETY: this AArch64 harness uses Kani's scalar NEON model.
        let kernel = unsafe { NeonKernel4x4::new() };
        buffered_tile_case(&kernel, 3);
    }
}
