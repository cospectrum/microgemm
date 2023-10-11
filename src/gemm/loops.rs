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

    assert_eq!(m % mc, 0);
    assert_eq!(k % kc, 0);
    assert_eq!(n % nc, 0);

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
    fn fixed() {
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
}
