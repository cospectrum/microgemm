use crate::kernel::Multiply;
use crate::{Kernel, MatMut, MatRef, PackSizes};
use generic_array::{sequence::GenericSequence, GenericArray};
use num_traits::{One, Zero};

type Product<L, R> = <L as Multiply<R>>::Output;

#[allow(clippy::too_many_arguments)]
#[inline]
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
                    let rhs = MatRef::row_major(kc, nr, rhs_values);

                    let dst_cols = jc + jr..jc + jr + nr;

                    for (l1, ir) in (0..mc).step_by(mr).enumerate() {
                        let lsize = mr * kc;
                        let lhs_values = &apack[lsize * l1..lsize * (l1 + 1)];
                        let lhs = MatRef::col_major(mr, kc, lhs_values);

                        let dst_rows = ic + ir..ic + ir + mr;
                        crate::packing::registers_from_c(
                            dst_buf,
                            c.to_ref(),
                            dst_rows.clone(),
                            dst_cols.clone(),
                        );
                        let mut dst = MatMut::col_major(mr, nr, dst_buf);
                        kernel.microkernel(alpha, lhs, rhs, beta, &mut dst);
                        crate::packing::registers_to_c(dst_buf, c, dst_rows, dst_cols.clone());
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

            assert_eq!(dst.row_stride(), 1);
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
}
