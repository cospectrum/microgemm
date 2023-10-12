use crate::{copy_to, pack_a, pack_b, pack_c, BlockSizes, MatMut, MatRef};
use num_traits::{One, Zero};

pub fn gemm_with_params<T>(
    alpha: T,
    a: &MatRef<T>,
    b: &MatRef<T>,
    beta: T,
    c: &mut MatMut<T>,
    mut microkernel: impl FnMut(T, &MatRef<T>, &MatRef<T>, T, &mut MatMut<T>),
    block_sizes: &BlockSizes,
) where
    T: Copy + Zero + One,
{
    block_sizes.check();
    assert_eq!(a.nrows(), c.nrows());
    assert_eq!(a.ncols(), b.nrows());
    assert_eq!(b.ncols(), c.ncols());

    let (m, k, n) = (a.nrows(), a.ncols(), c.ncols());
    let mc = block_sizes.mc;
    let nc = block_sizes.nc;
    let kc = block_sizes.kc;

    let mr = block_sizes.mr;
    let nr = block_sizes.nr;

    let mut b_buf = vec![T::zero(); kc * nc];
    let mut a_buf = vec![T::zero(); mc * kc];
    let mut c_buf = vec![T::zero(); mr * nr];

    let lhs = MatRef::from_parts(mr, kc, &[], kc, 1);
    let rhs = MatRef::from_parts(kc, nr, &[], 1, kc);

    for jc in (0..n).step_by(nc) {
        for (l4, pc) in (0..k).step_by(kc).enumerate() {
            let beta = if l4 == 0 { beta } else { One::one() };
            let bp = pack_b(b, b_buf.as_mut(), pc..pc + kc, jc..jc + nc);
            let bp = bp.to_ref();

            for ic in (0..m).step_by(mc) {
                let ap = pack_a(a, a_buf.as_mut(), ic..ic + mc, pc..pc + kc);
                let ap = ap.to_ref();

                for (l2, jr) in (0..nc).step_by(nr).enumerate() {
                    let rhs_vals = &bp.as_slice()[kc * nr * l2..kc * nr * (l2 + 1)];
                    let rhs = rhs.with_values(rhs_vals);
                    let dst_cols = jc + jr..jc + jr + nr;

                    for (l1, ir) in (0..mc).step_by(mr).enumerate() {
                        let lhs_vals = &ap.as_slice()[kc * mr * l1..kc * mr * (l1 + 1)];
                        let lhs = lhs.with_values(lhs_vals);
                        let dst_rows = ic + ir..ic + ir + mr;

                        let mut dst =
                            pack_c(&c.to_ref(), &mut c_buf, dst_rows.clone(), dst_cols.clone());
                        microkernel(alpha, &lhs, &rhs, beta, &mut dst);
                        copy_to(c, &dst.to_ref(), dst_rows, dst_cols.clone());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{naive_gemm, Layout};

    use super::*;

    #[rustfmt::skip]
    #[test]
    fn fixed_even() {
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

        let block_sizes = BlockSizes { mc: 2, mr: 1, kc: 2, nc: 2, nr: 1 };
        let ker = naive_gemm;
        gemm_with_params(alpha, &a, &b, beta, &mut c, ker, &block_sizes);
        assert_eq!(c.as_slice(), [260, 277, 638, 687]);
    }

    #[test]
    fn fixed_odd() {
        let alpha = 2;
        let beta = -3;

        let m = 3;
        let k = 5;
        let n = 3;

        let a = [1, 2, 3, 4, 5, 5, 6, 7, 8, 9, -3, -4, -5, -6, -7];
        let b = [
            9, 10, -11, 11, 12, -13, 13, 14, -15, 15, 16, -17, 17, 18, -19,
        ];
        let a = MatRef::new(m, k, &a, Layout::RowMajor);
        let b = MatRef::new(k, n, &b, Layout::RowMajor);

        let mut c = (0..m * n).map(|x| x as i32).collect::<Vec<_>>();
        let mut expect = c.clone();
        let mut c = MatMut::new(m, n, c.as_mut(), Layout::RowMajor);
        let mut expect = MatMut::new(m, n, expect.as_mut(), Layout::RowMajor);

        let block_sizes = BlockSizes {
            mc: 2,
            mr: 1,
            kc: 2,
            nc: 2,
            nr: 1,
        };
        let ker = naive_gemm;

        gemm_with_params(alpha, &a, &b, beta, &mut c, ker, &block_sizes);
        naive_gemm(alpha, &a, &b, beta, &mut expect);
        assert_eq!(c.as_slice(), expect.as_slice());
    }

    #[test]
    fn test_random() {
        use rand::Rng;

        let rng = &mut rand::thread_rng();
        let distr = rand::distributions::Uniform::new(-30, 30);

        let m = rng.gen_range(1..100);
        let k = rng.gen_range(1..100);
        let n = rng.gen_range(1..100);

        let alpha = rng.gen_range(-10..10);
        let beta = rng.gen_range(-10..10);

        let a = rng.sample_iter(distr).take(m * k).collect::<Vec<i32>>();
        let b = rng.sample_iter(distr).take(k * n).collect::<Vec<i32>>();
        let mut c = rng.sample_iter(distr).take(m * n).collect::<Vec<i32>>();
        let mut expect = c.clone();

        let a = MatRef::new(m, k, &a, Layout::RowMajor);
        let b = MatRef::new(k, n, &b, Layout::RowMajor);
        let mut c = MatMut::new(m, n, &mut c, Layout::RowMajor);
        let mut expect = MatMut::new(m, n, &mut expect, Layout::RowMajor);

        let block_sizes = BlockSizes {
            mc: 2,
            mr: 1,
            kc: 2,
            nc: 2,
            nr: 1,
        };
        let ker = naive_gemm;

        gemm_with_params(alpha, &a, &b, beta, &mut c, ker, &block_sizes);
        naive_gemm(alpha, &a, &b, beta, &mut expect);
        assert_eq!(expect.as_slice(), c.as_slice());
    }
}
