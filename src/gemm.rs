use crate::{Kernel, MatMut, MatRef, PackSizes};
use num_traits::{One, Zero};

#[allow(clippy::too_many_arguments)]
pub fn gemm_with_kernel<T, K>(
    kernel: &K,
    alpha: T,
    a: &MatRef<T>,
    b: &MatRef<T>,
    beta: T,
    c: &mut MatMut<T>,
    pack_sizes: &PackSizes,
    packing_buf: &mut [T],
) where
    T: Copy + Zero + One,
    K: Kernel<T>,
{
    pack_sizes.check(kernel);
    assert_eq!(pack_sizes.buf_len(kernel), packing_buf.len());
    let (apack, bpack, dst_buf) = pack_sizes.split_buf(packing_buf);

    assert_eq!(a.nrows(), c.nrows());
    assert_eq!(a.ncols(), b.nrows());
    assert_eq!(b.ncols(), c.ncols());
    let (m, k, n) = (a.nrows(), a.ncols(), c.ncols());

    let mc = pack_sizes.mc;
    let nc = pack_sizes.nc;
    let kc = pack_sizes.kc;

    let mr = K::MR;
    let nr = K::NR;

    for jc in (0..n).step_by(nc) {
        for (l4, pc) in (0..k).step_by(kc).enumerate() {
            let beta = if l4 == 0 { beta } else { One::one() };
            let rhs_layout = kernel.pack_b(pack_sizes, bpack, b, pc..pc + kc, jc..jc + nc);

            for ic in (0..m).step_by(mc) {
                let lhs_layout = kernel.pack_a(pack_sizes, apack, a, ic..ic + mc, pc..pc + kc);

                for (l2, jr) in (0..nc).step_by(nr).enumerate() {
                    let rhs_values = &bpack[kc * nr * l2..kc * nr * (l2 + 1)];
                    let rhs = MatRef::new(kc, nr, rhs_values, rhs_layout);

                    let dst_cols = jc + jr..jc + jr + nr;

                    for (l1, ir) in (0..mc).step_by(mr).enumerate() {
                        let lhs_values = &apack[kc * mr * l1..kc * mr * (l1 + 1)];
                        let lhs = MatRef::new(mr, kc, lhs_values, lhs_layout);

                        let dst_rows = ic + ir..ic + ir + mr;

                        let mut dst = kernel.copy_from_c(
                            &c.to_ref(),
                            dst_rows.clone(),
                            dst_cols.clone(),
                            dst_buf,
                        );
                        kernel.microkernel(alpha, &lhs, &rhs, beta, &mut dst);
                        kernel.copy_to_c(c, dst_rows, dst_cols.clone(), &dst.to_ref());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{utils::naive_gemm, Layout};

    struct TestKernel;

    impl Kernel<i32> for TestKernel {
        const MR: usize = 5;
        const NR: usize = 5;

        fn microkernel(
            &self,
            alpha: i32,
            lhs: &MatRef<i32>,
            rhs: &MatRef<i32>,
            beta: i32,
            dst: &mut MatMut<i32>,
        ) {
            assert_eq!(lhs.nrows(), Self::MR);
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
        let a = MatRef::new(m, k, &a, Layout::RowMajor);
        let b = MatRef::new(k, n, &b, Layout::RowMajor);

        let mut c = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();
        let mut c = MatMut::new(m, n, c.as_mut(), Layout::RowMajor);

        let pack_sizes = PackSizes { mc: 5 * TestKernel::MR,  kc: 2, nc: 2 * TestKernel::NR };
        let mut buf = vec![-9; pack_sizes.buf_len::<i32, TestKernel>(kernel)];

        gemm_with_kernel(kernel, alpha, &a, &b, beta, &mut c, &pack_sizes, &mut buf);
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
        let a = MatRef::new(m, k, &a, Layout::RowMajor);
        let b = MatRef::new(k, n, &b, Layout::RowMajor);

        let mut c = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();
        let mut expect = c.clone();
        let mut c = MatMut::new(m, n, c.as_mut(), Layout::RowMajor);
        let mut expect = MatMut::new(m, n, expect.as_mut(), Layout::RowMajor);

        let pack_sizes = PackSizes {
            mc: 2 * TestKernel::MR,
            kc: 2,
            nc: 3 * TestKernel::NR,
        };
        let mut buf = vec![-1; pack_sizes.buf_len::<i32, _>(kernel)];

        let alpha = 2;
        let beta = -3;

        gemm_with_kernel(kernel, alpha, &a, &b, beta, &mut c, &pack_sizes, &mut buf);
        naive_gemm(alpha, &a, &b, beta, &mut expect);
        assert_eq!(c.as_slice(), expect.as_slice());
    }

    #[test]
    fn test_random_gemm() {
        use crate::utils::test_kernel_with_random_i32;

        let kernel = TestKernel;
        for _ in 0..20 {
            test_kernel_with_random_i32(&kernel);
        }
    }
}