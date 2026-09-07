use crate::kernel::Multiply;
use crate::{Kernel, MatMut, MatRef, PackSizes};
use core::ops::Range;
use generic_array::{sequence::GenericSequence, GenericArray};
use num_traits::{One, Zero};

type Product<L, R> = <L as Multiply<R>>::Output;

// Stage edge tiles through dst_buf so the microkernel receives a complete tile.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn buffered_tile<T, K>(
    kernel: &K,
    alpha: T,
    lhs_values: &[T],
    rhs_values: &[T],
    kc: usize,
    beta: T,
    c: &mut MatMut<T>,
    dst_rows: Range<usize>,
    dst_cols: Range<usize>,
    dst_buf: &mut [T],
) where
    T: Copy + Zero + One,
    K: Kernel<Scalar = T> + ?Sized,
{
    let lhs = MatRef::col_major(K::MR, kc, lhs_values);
    let rhs = MatRef::row_major(kc, K::NR, rhs_values);
    crate::packing::registers_from_c(dst_buf, c.to_ref(), dst_rows.clone(), dst_cols.clone());
    let mut dst = MatMut::col_major(K::MR, K::NR, &mut *dst_buf);
    kernel.microkernel(alpha, lhs, rhs, beta, &mut dst);
    crate::packing::registers_to_c(&*dst_buf, c, dst_rows, dst_cols);
}

// Full tiles update C through its original strides. Edge tiles retain the
// buffered path so kernels always receive their complete MR-by-NR tile.
#[allow(clippy::too_many_arguments)]
#[inline]
pub(crate) fn direct_tile<T, K>(
    kernel: &K,
    alpha: T,
    lhs_values: &[T],
    rhs_values: &[T],
    kc: usize,
    beta: T,
    c: &mut MatMut<T>,
    dst_rows: Range<usize>,
    dst_cols: Range<usize>,
    dst_buf: &mut [T],
) where
    T: Copy + Zero + One,
    K: Kernel<Scalar = T> + ?Sized,
{
    let (rsc, csc) = (c.row_stride(), c.col_stride());
    if dst_rows.len() == K::MR
        && dst_cols.len() == K::NR
        && dst_rows.end <= c.nrows()
        && dst_cols.end <= c.ncols()
    {
        let lhs = MatRef::col_major(K::MR, kc, lhs_values);
        let rhs = MatRef::row_major(kc, K::NR, rhs_values);
        let at = c.idx(dst_rows.start, dst_cols.start);
        let mut dst = MatMut::from_parts(K::MR, K::NR, &mut c.as_mut_slice()[at..], rsc, csc)
            .expect("C tile must fit within its storage");
        kernel.microkernel(alpha, lhs, rhs, beta, &mut dst);
        return;
    }
    buffered_tile(
        kernel, alpha, lhs_values, rhs_values, kc, beta, c, dst_rows, dst_cols, dst_buf,
    );
}

#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub(crate) fn gemm_with_kernel<T, K>(
    kernel: &K,
    alpha: T,
    a: MatRef<T>,
    b: MatRef<T>,
    beta: T,
    c: &mut MatMut<T>,
    pack_sizes: PackSizes,
    packing_buf: &mut [T],
) where
    T: Copy + Zero + One,
    K: Kernel<Scalar = T> + ?Sized,
{
    assert_eq!(a.nrows(), c.nrows());
    assert_eq!(a.ncols(), b.nrows());
    assert_eq!(b.ncols(), c.ncols());
    let [m, k, n] = [a.nrows(), a.ncols(), c.ncols()];

    assert_eq!(
        packing_buf.len(),
        pack_sizes
            .checked_buf_len()
            .expect("PackSizes::buf_len should not overflow")
    );
    let pack_sizes = pack_sizes.clamped(kernel);
    let packing_buf = packing_buf[..pack_sizes.checked_buf_len().unwrap()].as_mut();
    let (apack, bpack) = pack_sizes.split_buf(packing_buf);

    let mr = K::MR;
    let nr = K::NR;
    assert!(mr > 0);
    assert!(nr > 0);

    let [mc, nc] = [pack_sizes.mc, pack_sizes.nc];
    assert!(mr <= mc);
    assert_eq!(mc % mr, 0);
    assert!(nr <= nc);
    assert_eq!(nc % nr, 0);

    let zero = Zero::zero();
    let mut dst_buf = GenericArray::<T, Product<K::Mr, K::Nr>>::generate(|_| zero);
    let dst_buf = dst_buf.as_mut_slice();

    for jc in (0..n).step_by(nc) {
        for (l4, pc) in (0..k).step_by(pack_sizes.kc).enumerate() {
            let beta = if l4 == 0 { beta } else { One::one() };

            let kc = (pc + pack_sizes.kc).min(k) - pc;
            debug_assert!(pc + kc <= k);

            let bpack = {
                let rows = pc..pc + kc;
                let cols = jc..jc + nc;
                let bpack = &mut bpack[..kc * nc];
                crate::packing::pack_b(nr, bpack, b, rows, cols);
                bpack
            };

            for ic in (0..m).step_by(mc) {
                let apack = {
                    let rows = ic..ic + mc;
                    let cols = pc..pc + kc;
                    let apack = &mut apack[..mc * kc];
                    crate::packing::pack_a(mr, apack, a, rows, cols);
                    apack
                };

                for (l2, jr) in (0..nc).step_by(nr).enumerate() {
                    let rsize = kc * nr;
                    let rhs_values = &bpack[rsize * l2..rsize * (l2 + 1)];

                    let dst_cols = jc + jr..jc + jr + nr;

                    for (l1, ir) in (0..mc).step_by(mr).enumerate() {
                        let lsize = mr * kc;
                        let lhs_values = &apack[lsize * l1..lsize * (l1 + 1)];

                        let dst_rows = ic + ir..ic + ir + mr;
                        direct_tile(
                            kernel,
                            alpha,
                            lhs_values,
                            rhs_values,
                            kc,
                            beta,
                            c,
                            dst_rows,
                            dst_cols.clone(),
                            &mut *dst_buf,
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::std_prelude::*;
    use crate::{
        typenum::{U4, U5},
        utils::naive_gemm,
    };

    struct TestKernel;

    impl Kernel for TestKernel {
        type Scalar = i32;
        type Mr = U4;
        type Nr = U5;

        fn microkernel(
            &self,
            alpha: i32,
            lhs: MatRef<i32>,
            rhs: MatRef<i32>,
            beta: i32,
            dst: &mut MatMut<i32>,
        ) {
            assert_eq!(lhs.row_stride(), 1);
            assert_eq!(lhs.nrows(), Self::MR);

            assert_eq!(rhs.col_stride(), 1);
            assert_eq!(rhs.ncols(), Self::NR);

            assert_eq!(dst.nrows(), Self::MR);
            assert_eq!(dst.ncols(), Self::NR);
            naive_gemm(alpha, lhs, rhs, beta, dst);
        }
    }

    #[rustfmt::skip]
    #[test]
    fn gemm_fixed_even() {
        let kernel = &TestKernel;

        let alpha = 2;
        let beta = -3;

        let m = 2;
        let k = 4;
        let n = 2;

        let a = [
            1, 2, 3, 4,
            5, 6, 7, 8,
        ];
        let b = [
            9, 10,
            11, 12,
            13, 14,
            15, 16,
        ];
        let a = MatRef::row_major(m, k, &a);
        let b = MatRef::row_major(k, n, &b);

        let mut c = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();
        let mut c = MatMut::row_major(m, n, c.as_mut());

        let pack_sizes = PackSizes { mc: 5 * TestKernel::MR,  kc: 2, nc: 2 * TestKernel::NR };
        let mut buf = vec![-9; pack_sizes.buf_len()];

        gemm_with_kernel(kernel, alpha, a, b, beta, &mut c, pack_sizes, &mut buf);
        assert_eq!(c.as_slice(), [260, 277, 638, 687]);
    }

    #[rustfmt::skip]
    #[test]
    fn gemm_fixed_odd() {
        let kernel = &TestKernel;

        let m = 3;
        let k = 5;
        let n = 3;

        let a = [
            1, 2, 3, 4, 5,
            5, 6, 7, 8, 9,
            -3, -4, -5, -6, -7,
        ];
        let b = [
            9, 10, -11,
            11, 12, -13,
            13, 14, -15,
            15, 16, -17,
            17, 18, -19,
        ];
        let a = MatRef::row_major(m, k, &a);
        let b = MatRef::row_major(k, n, &b);

        let mut c = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();
        let mut expect = c.clone();
        let mut c = MatMut::row_major(m, n, c.as_mut());
        let mut expect = MatMut::row_major(m, n, expect.as_mut());

        let pack_sizes = PackSizes {
            mc: 2 * TestKernel::MR,
            kc: 2,
            nc: 3 * TestKernel::NR,
        };
        let mut buf = vec![-1; pack_sizes.buf_len()];

        let alpha = 2;
        let beta = -3;

        gemm_with_kernel(kernel, alpha, a, b, beta, &mut c, pack_sizes, &mut buf);
        naive_gemm(alpha, a, b, beta, &mut expect);
        assert_eq!(c.as_slice(), expect.as_slice());
    }

    #[rustfmt::skip]
    #[test]
    fn test_gemm_sample_1() {
        let kernel = TestKernel;

        let a = [
            28, 26, -9, -29,
            29, -8, 23, 22,
            -2, -2, 26, -21,
            -29, 2, 26, -17,
            -22, -18, -24, -23,
            -20, 14, 13, -22,
        ];
        let a = MatRef::row_major(6, 4, &a);
        let b = [
            2, -24, 20,
            -27, -1, -16,
            -12, -29, -26,
            -16, -13, -18,
        ];
        let b = MatRef::row_major(4, 3, &b);
        let mut c = vec![
            480, 417, -7102,
            2720, 13184, 2400,
            -578, 3651, 2280,
            1426, -1463, 7849,
            -8973, -12188, -7249,
            508, 627, -7298,
        ];
        let mut expect = c.clone();
        let mut c = MatMut::row_major(6, 3, &mut c);
        let mut expect = MatMut::row_major(6, 3, &mut expect);

        let alpha = 4;
        let beta = -3;

        let pack_sizes = PackSizes {mc: kernel.mr(), kc: 2, nc: kernel.nr() };
        let mut buf = vec![-2; pack_sizes.buf_len()];

        kernel.gemm(alpha, a, b, beta, c.as_mut(), pack_sizes, &mut buf);
        naive_gemm(alpha, a, b, beta, expect.as_mut());
        assert_eq!(expect.as_slice(), c.as_slice());
    }

    #[test]
    fn direct_rectangular_tiles_preserve_layout_and_fallbacks() {
        use core::cell::Cell;

        struct StridedKernel(Cell<bool>);
        impl Kernel for StridedKernel {
            type Scalar = i32;
            type Mr = U4;
            type Nr = U5;

            fn microkernel(
                &self,
                alpha: i32,
                lhs: MatRef<i32>,
                rhs: MatRef<i32>,
                beta: i32,
                dst: &mut MatMut<i32>,
            ) {
                self.0
                    .set((dst.row_stride(), dst.col_stride()) != (1, Self::MR));
                assert_eq!((dst.nrows(), dst.ncols()), (Self::MR, Self::NR));
                assert_eq!(lhs.as_slice(), &[1, 2, 3, 4, -1, 2, -3, 4]);
                assert_eq!(rhs.as_slice(), &[2, 3, 5, 7, 11, -2, 3, -5, 7, -11]);
                // Honor the public contract even when coordinates overlap:
                // snapshot C, then scatter in column-major order.
                let mut values = [0; Self::MR * Self::NR];
                for j in 0..Self::NR {
                    for i in 0..Self::MR {
                        let dot = lhs.get(i, 0) * rhs.get(0, j) + lhs.get(i, 1) * rhs.get(1, j);
                        values[j * Self::MR + i] = alpha * dot + beta * dst.get(i, j);
                    }
                }
                for j in 0..Self::NR {
                    for i in 0..Self::MR {
                        *dst.get_mut(i, j) = values[j * Self::MR + i];
                    }
                }
            }
        }

        const MR: usize = StridedKernel::MR;
        const NR: usize = StridedKernel::NR;
        let lhs = [1, 2, 3, 4, -1, 2, -3, 4];
        let rhs = [2, 3, 5, 7, 11, -2, 3, -5, 7, -11];
        for (rows, rs, cs, direct) in [
            (MR + 2, 9, 1, true),
            (MR + 2, 1, 8, true),
            (MR + 2, 2, 14, true),
            (MR + 2, 14, 2, true),
            (MR + 2, 1, 2, true),
            (MR + 2, 2, 2, true),
            (MR + 2, 0, 1, true),
            (MR + 2, 1, 0, true),
            (MR + 2, 0, 0, true),
            (MR, 9, 1, false),
        ] {
            let len = (rows - 1) * rs + (NR + 1) * cs + 3;
            let mut actual: Vec<i32> = (0..len).map(|i| (i % 11) as i32 - 5).collect();
            let mut expected = actual.clone();
            let kernel = StridedKernel(Cell::new(false));
            let mut scratch = [777; MR * NR];
            buffered_tile(
                &TestKernel,
                2,
                &lhs,
                &rhs,
                2,
                -3,
                &mut MatMut::from_parts(rows, NR + 2, &mut expected[1..len - 1], rs, cs).unwrap(),
                1..1 + MR,
                1..1 + NR,
                &mut [0; MR * NR],
            );
            direct_tile(
                &kernel,
                2,
                &lhs,
                &rhs,
                2,
                -3,
                &mut MatMut::from_parts(rows, NR + 2, &mut actual[1..len - 1], rs, cs).unwrap(),
                1..1 + MR,
                1..1 + NR,
                &mut scratch,
            );
            assert_eq!(kernel.0.get(), direct);
            assert_eq!(actual, expected, "rows={rows}, strides=({rs}, {cs})");
            if direct {
                assert_eq!(scratch, [777; MR * NR]);
            }
        }
    }
}
