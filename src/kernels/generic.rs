use crate::{
    kernels::dbg_check_microkernel_inputs,
    typenum::{U16, U2, U32, U4, U8},
    Kernel, MatMut, One, Zero,
};
use core::marker::PhantomData;
use core::ops::{Add, Mul};

fn loop_micropanels<T, const DIM: usize>(lhs: &[T], rhs: &[T], cols: &mut [T])
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Zero,
{
    assert_eq!(cols.len(), DIM * DIM);
    assert!(DIM > 0);
    assert_eq!(lhs.len() % DIM, 0);
    assert_eq!(lhs.len(), rhs.len());

    let left = lhs.chunks_exact(DIM);
    let right = rhs.chunks_exact(DIM);

    left.zip(right).for_each(|(a, b)| {
        let cols = cols.chunks_exact_mut(DIM);

        cols.zip(b).for_each(|(col, &scalar)| {
            col.iter_mut().zip(a).for_each(|(out, &x)| {
                *out = *out + x * scalar;
            });
        });
    });
}

// Only final stores depend on C's orientation. Keep scalar multiplication in
// its original order: generic scalar multiplication need not commute.
#[inline]
fn write_cols<T, const DIM: usize>(dst: &mut MatMut<T>, cols: &mut [T], alpha: T, beta: T)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!((dst.nrows(), dst.ncols()), (DIM, DIM));
    assert_eq!(cols.len(), DIM * DIM);
    let (rs, cs) = (dst.row_stride(), dst.col_stride());
    if (rs == 1 && cs >= DIM) || (cs == 1 && rs >= DIM) {
        for (j, col) in cols.chunks_exact(DIM).enumerate() {
            for (i, &from) in col.iter().enumerate() {
                let to = dst.get_mut(i, j);
                *to = alpha * from + beta * *to;
            }
        }
    } else {
        // Snapshot every old C value before scattering: aliased coordinates
        // must each use the original value, then the last column-major store wins.
        for (j, col) in cols.chunks_exact_mut(DIM).enumerate() {
            for (i, from) in col.iter_mut().enumerate() {
                *from = alpha * *from + beta * dst.get(i, j);
            }
        }
        for (j, col) in cols.chunks_exact(DIM).enumerate() {
            for (i, &from) in col.iter().enumerate() {
                *dst.get_mut(i, j) = from;
            }
        }
    }
}

macro_rules! impl_generic_square_kernel {
    ($struct:ident, $dim:literal, $dimty:ty) => {
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $struct<T>(PhantomData<T>);

        impl<T> $struct<T> {
            pub const fn new() -> Self {
                Self(PhantomData)
            }
        }
        impl<T> Kernel for $struct<T>
        where
            T: Copy + Zero + One + Add<Output = T> + Mul<Output = T>,
        {
            type Scalar = T;
            type Mr = $dimty;
            type Nr = $dimty;

            #[inline]
            fn microkernel(
                &self,
                alpha: Self::Scalar,
                lhs: crate::MatRef<Self::Scalar>,
                rhs: crate::MatRef<Self::Scalar>,
                beta: Self::Scalar,
                dst: &mut crate::MatMut<Self::Scalar>,
            ) {
                dbg_check_microkernel_inputs(self, lhs, rhs, dst);

                const DIM: usize = $dim;
                let mut cols = [T::zero(); DIM * DIM];
                loop_micropanels::<_, DIM>(lhs.as_slice(), rhs.as_slice(), &mut cols);
                write_cols::<_, DIM>(dst, &mut cols, alpha, beta);
            }
        }
    };
}

impl_generic_square_kernel!(GenericKernel2x2, 2, U2);
impl_generic_square_kernel!(GenericKernel4x4, 4, U4);
impl_generic_square_kernel!(GenericKernel8x8, 8, U8);
impl_generic_square_kernel!(GenericKernel16x16, 16, U16);
impl_generic_square_kernel!(GenericKernel32x32, 32, U32);

#[cfg(not(miri))]
#[cfg(test)]
mod proptests {
    use super::*;
    use crate::{
        std_prelude::*,
        utils::{is_debug_build, proptest_kernel, ProptestKernelCfg},
    };
    use proptest::prelude::*;

    fn cfg_i32() -> ProptestKernelCfg<i32> {
        let dim = if is_debug_build() { 38 } else { 83 };
        ProptestKernelCfg::default()
            .with_max_matrix_dim(dim)
            .with_max_pack_dim(2 * dim + 1)
            .with_scalar((-11..11).boxed())
    }

    #[test]
    fn proptest_generic_kernel_2x2_i32() {
        proptest_kernel(&GenericKernel2x2::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_4x4_i32() {
        proptest_kernel(&GenericKernel4x4::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_8x8_i32() {
        proptest_kernel(&GenericKernel8x8::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_16x16_i32() {
        proptest_kernel(&GenericKernel16x16::new(), cfg_i32()).unwrap();
    }
    #[test]
    fn proptest_generic_kernel_32x32_i32() {
        proptest_kernel(&GenericKernel32x32::new(), cfg_i32()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{std_prelude::*, MatRef};

    fn compare_strided_microkernel<K: Kernel<Scalar = i32>>(kernel: K) {
        let dim = K::MR;
        let depth = 3;
        let a: Vec<_> = (0..dim * depth).map(|i| i as i32 % 7 - 3).collect();
        let b: Vec<_> = (0..dim * depth).map(|i| i as i32 % 11 - 5).collect();
        for (rs, cs) in [
            (1, dim + 2),
            (dim + 3, 1),
            (2, 2 * dim + 3),
            (2 * dim + 3, 2),
            (1, 1),
            (2, 2),
            (0, 1),
            (1, 0),
            (0, 0),
        ] {
            let len = (dim - 1) * (rs + cs) + 3;
            let old: Vec<_> = (0..len).map(|i| i as i32 % 13 - 6).collect();
            let mut actual = old.clone();
            let mut expected = old.clone();
            // Independent snapshot oracle, including last-write-wins aliases
            // and untouched prefix, suffix and interior padding.
            for j in 0..dim {
                for i in 0..dim {
                    let mut sum = 0;
                    for k in 0..depth {
                        sum += a[k * dim + i] * b[k * dim + j];
                    }
                    let at = 1 + i * rs + j * cs;
                    expected[at] = 2 * sum - 3 * old[at];
                }
            }
            kernel.microkernel(
                2,
                MatRef::col_major(dim, depth, &a),
                MatRef::row_major(depth, dim, &b),
                -3,
                &mut MatMut::from_parts(dim, dim, &mut actual[1..len - 1], rs, cs).unwrap(),
            );
            assert_eq!(actual, expected, "dimension {dim}, strides ({rs}, {cs})");
        }
    }

    #[test]
    fn arbitrary_microkernel_strides_preserve_snapshot_semantics() {
        compare_strided_microkernel(GenericKernel2x2::new());
        compare_strided_microkernel(GenericKernel4x4::new());
        compare_strided_microkernel(GenericKernel8x8::new());
        compare_strided_microkernel(GenericKernel16x16::new());
        compare_strided_microkernel(GenericKernel32x32::new());
    }

    // Integer 2x2 matrices form a scalar ring whose multiplication does not
    // commute. A row-major destination must not transpose A and B's product.
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Matrix([i32; 4]);

    impl Add for Matrix {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            let [a, b, c, d] = self.0;
            let [e, f, g, h] = rhs.0;
            Self([a + e, b + f, c + g, d + h])
        }
    }
    impl Mul for Matrix {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            let [a, b, c, d] = self.0;
            let [e, f, g, h] = rhs.0;
            Self([a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h])
        }
    }
    impl Zero for Matrix {
        fn zero() -> Self {
            Self([0; 4])
        }
        fn is_zero(&self) -> bool {
            *self == Self::zero()
        }
    }
    impl One for Matrix {
        fn one() -> Self {
            Self([1, 0, 0, 1])
        }
    }

    #[test]
    fn row_major_preserves_noncommutative_scalar_order() {
        let kernel = GenericKernel2x2::<Matrix>::new();
        let a = Matrix([1, 2, 0, 1]);
        let b = Matrix([1, 0, 3, 1]);
        assert_ne!(a * b, b * a);
        let alpha = Matrix([2, 1, 0, 1]);
        let beta = Matrix([1, 0, 1, 2]);
        let initial = Matrix([3, 1, 2, 0]);
        let expected = alpha * (a * b) + beta * initial;
        let mut c = [initial; 4];
        kernel.microkernel(
            alpha,
            MatRef::col_major(2, 1, &[a; 2]),
            MatRef::row_major(1, 2, &[b; 2]),
            beta,
            &mut MatMut::row_major(2, 2, &mut c),
        );
        assert_eq!(c, [expected; 4]);
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(9)]
    fn direct_2x2_preserves_output_and_padding() {
        let a: [i8; 2] = kani::any();
        let b: [i8; 2] = kani::any();
        let old: [i8; 7] = kani::any();
        let a = a.map(i32::from);
        let b = b.map(i32::from);
        let mut c = old.map(i32::from);
        let row_major: bool = kani::any();
        let (rs, cs) = if row_major { (3, 1) } else { (1, 3) };
        GenericKernel2x2::new().microkernel(
            2,
            crate::MatRef::col_major(2, 1, &a),
            crate::MatRef::row_major(1, 2, &b),
            -3,
            &mut MatMut::from_parts(2, 2, &mut c[1..6], rs, cs).unwrap(),
        );
        for j in 0..2 {
            for i in 0..2 {
                let at = 1 + i * rs + j * cs;
                assert_eq!(c[at], 2 * (a[i] * b[j]) - 3 * i32::from(old[at]));
            }
        }
        assert_eq!(c[0], i32::from(old[0]));
        assert_eq!(c[3], i32::from(old[3]));
        assert_eq!(c[6], i32::from(old[6]));
    }

    #[kani::proof]
    #[kani::unwind(11)]
    fn arbitrary_2x2_strides_preserve_snapshot_semantics() {
        let a: [i8; 2] = kani::any();
        let b: [i8; 2] = kani::any();
        let old: [i8; 9] = kani::any();
        let a = a.map(i32::from);
        let b = b.map(i32::from);
        let mut actual = old.map(i32::from);
        let mut expected = actual;
        let rs: usize = kani::any();
        let cs: usize = kani::any();
        kani::assume(rs <= 3 && cs <= 3);
        GenericKernel2x2::new().microkernel(
            2,
            crate::MatRef::col_major(2, 1, &a),
            crate::MatRef::row_major(1, 2, &b),
            -3,
            &mut MatMut::from_parts(2, 2, &mut actual[1..8], rs, cs).unwrap(),
        );
        for j in 0..2 {
            for i in 0..2 {
                let at = 1 + i * rs + j * cs;
                expected[at] = 2 * (a[i] * b[j]) - 3 * i32::from(old[at]);
            }
        }
        for i in 0..9 {
            assert_eq!(actual[i], expected[i]);
        }
    }
}
