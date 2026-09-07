#![cfg(target_arch = "aarch64")]

use microgemm::{
    kernels::{NeonKernel4x4, NeonKernel8x8},
    Kernel, MatMut, MatRef, PackSizes,
};

#[test]
fn test_neon8x8() {
    let kernel = if cfg!(target_feature = "neon") {
        unsafe { NeonKernel8x8::<f32>::new() }
    } else {
        println!("neon feature is not supported");
        return;
    };
    test_kernel_f32(kernel);
}

#[test]
fn test_neon4x4() {
    let kernel = if cfg!(target_feature = "neon") {
        unsafe { NeonKernel4x4::<f32>::new() }
    } else {
        println!("neon feature is not supported");
        return;
    };
    test_kernel_f32(kernel);
}

fn test_kernel_f32(kernel: impl Kernel<Scalar = f32>) {
    let pack_sizes = PackSizes {
        mc: kernel.mr(),
        kc: 2,
        nc: kernel.nr(),
    };

    let test_cases = vec![
        TestCase {
            alpha: 1.,
            a: MatRef::row_major(2, 3, &[1., 2., 3., 4., 5., 6.]),
            b: MatRef::row_major(3, 2, &[10., 11., 20., 21., 30., 31.]),
            c: MatRef::row_major(2, 2, &[99.; 2 * 2]),
            beta: 0.,
            expect: &[140., 146., 320., 335.],
            pack_sizes,
        },
        TestCase {
            alpha: 0.,
            a: MatRef::row_major(2, 2, &[1., 2., 3., 4.]),
            b: MatRef::row_major(2, 2, &[5., 6., 3., 4.]),
            c: MatRef::row_major(2, 2, &[3.; 2 * 2]),
            beta: 1.,
            expect: &[3., 3., 3., 3.],
            pack_sizes,
        },
    ];

    for test_case in test_cases {
        test_case.test_kernel(&kernel);
    }
}

#[derive(Debug, Clone, Copy)]
struct TestCase<'a> {
    alpha: f32,
    a: MatRef<'a, f32>,
    b: MatRef<'a, f32>,
    beta: f32,
    c: MatRef<'a, f32>,
    pack_sizes: PackSizes,
    expect: &'a [f32],
}

impl<'a> TestCase<'a> {
    fn test_kernel(&self, kernel: &impl Kernel<Scalar = f32>) {
        let param = self;
        let mut actual = param.c.as_slice().to_vec();
        let mut actual = MatMut::from_parts(
            param.c.nrows(),
            param.c.ncols(),
            &mut actual,
            param.c.row_stride(),
            param.c.col_stride(),
        )
        .unwrap();
        let mut packing_buf = vec![0f32; param.pack_sizes.buf_len()];
        kernel.gemm(
            param.alpha,
            param.a,
            param.b,
            param.beta,
            &mut actual,
            param.pack_sizes,
            &mut packing_buf,
        );
        assert_eq!(actual.as_slice(), param.expect, "{self:?}");
    }
}

// Force packed output to compare direct GEMM against copy-in/out.
struct BufferedKernel<K>(K);

impl<K: Kernel<Scalar = f32>> Kernel for BufferedKernel<K> {
    type Scalar = f32;
    type Mr = K::Mr;
    type Nr = K::Nr;

    fn microkernel(
        &self,
        alpha: f32,
        lhs: MatRef<f32>,
        rhs: MatRef<f32>,
        beta: f32,
        dst: &mut MatMut<f32>,
    ) {
        let mut values = vec![0.0; Self::MR * Self::NR];
        for j in 0..Self::NR {
            for i in 0..Self::MR {
                values[j * Self::MR + i] = dst.get(i, j);
            }
        }
        self.0.microkernel(
            alpha,
            lhs,
            rhs,
            beta,
            &mut MatMut::col_major(Self::MR, Self::NR, &mut values),
        );
        for j in 0..Self::NR {
            for i in 0..Self::MR {
                *dst.get_mut(i, j) = values[j * Self::MR + i];
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CLayout {
    /// rsc = n, csc = 1: direct path when n >= 8
    Row,
    /// rsc = 1, csc = m: direct path when m >= 8
    Col,
    /// rsc = 1, csc = 13: direct path, ld > 8
    ColStrided,
    /// rsc = 11, csc = 1: direct path, ld > 8
    RowStrided,
    /// rsc = 1, csc = 2: overlapping columns, must fall back to the buffered path
    Aliased,
    /// Neither axis is contiguous: buffered gather/scatter.
    Strided,
    /// Every element refers to the same location: buffered traversal order matters.
    Broadcast,
}

const C_LAYOUTS: [CLayout; 7] = [
    CLayout::Row,
    CLayout::Col,
    CLayout::ColStrided,
    CLayout::RowStrided,
    CLayout::Aliased,
    CLayout::Strided,
    CLayout::Broadcast,
];

impl CLayout {
    fn strides(self, m: usize, n: usize) -> (usize, usize) {
        match self {
            CLayout::Row => (n, 1),
            CLayout::Col => (1, m),
            CLayout::ColStrided => (1, 13),
            CLayout::RowStrided => (11, 1),
            CLayout::Aliased => (1, 2),
            CLayout::Strided => (2 * n + 3, 2),
            CLayout::Broadcast => (0, 0),
        }
    }
    fn len(self, m: usize, n: usize) -> usize {
        let (rsc, csc) = self.strides(m, n);
        (m - 1) * rsc + (n - 1) * csc + 1
    }
}

const ALPHAS: [f32; 4] = [0., 1., -1., 2.5];
const BETAS: [f32; 4] = [0., 1., -1., 2.5];

/// Deterministic, never exactly zero, so bit-exactness is meaningful.
fn lcg_fill(values: &mut [f32], seed: u32) {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    for x in values.iter_mut() {
        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
        *x = (((s >> 16) & 0x3ff) as f32) / 1024. - 0.5 + 0.001;
    }
}

fn round_up_8(x: usize) -> usize {
    ((x + 7) / 8).max(1) * 8
}

#[allow(clippy::too_many_arguments)]
fn check_direct_matches_buffered<K: Kernel<Scalar = f32>>(
    direct: &K,
    buffered: &BufferedKernel<K>,
    m: usize,
    k: usize,
    n: usize,
    layout: CLayout,
    alpha: f32,
    beta: f32,
    big_pack: bool,
) {
    let (rsc, csc) = layout.strides(m, n);
    let clen = layout.len(m, n);

    let mut a_values = vec![0f32; m * k];
    let mut b_values = vec![0f32; k * n];
    let mut c_values = vec![0f32; clen];
    lcg_fill(&mut a_values, 1 + m as u32 + 31 * k as u32);
    lcg_fill(&mut b_values, 2 + n as u32 + 31 * k as u32);
    lcg_fill(&mut c_values, 3 + clen as u32);

    let a = if big_pack {
        MatRef::col_major(m, k, a_values.as_slice())
    } else {
        MatRef::row_major(m, k, a_values.as_slice())
    };
    let b = MatRef::row_major(k, n, b_values.as_slice());

    let pack_sizes = if big_pack {
        PackSizes {
            mc: round_up_8(m),
            kc: k,
            nc: round_up_8(n),
        }
    } else {
        PackSizes {
            mc: 8,
            kc: if k > 1 { 2 } else { 1 },
            nc: 8,
        }
    };

    let mut got = c_values.clone();
    let mut expect = c_values.clone();
    let mut got_mat = MatMut::from_parts(m, n, got.as_mut_slice(), rsc, csc).unwrap();
    let mut expect_mat = MatMut::from_parts(m, n, expect.as_mut_slice(), rsc, csc).unwrap();

    let mut buf = vec![0f32; pack_sizes.buf_len()];
    direct.gemm(alpha, a, b, beta, &mut got_mat, pack_sizes, &mut buf);
    let mut buf = vec![0f32; pack_sizes.buf_len()];
    buffered.gemm(alpha, a, b, beta, &mut expect_mat, pack_sizes, &mut buf);

    for i in 0..clen {
        assert_eq!(
            got[i].to_bits(),
            expect[i].to_bits(),
            "m={m} k={k} n={n} layout={layout:?} alpha={alpha} beta={beta} \
             big_pack={big_pack} at {i}: {} vs {}",
            got[i],
            expect[i]
        );
    }
}

#[test]
fn test_neon8x8_direct_matches_buffered() {
    let direct = if cfg!(target_feature = "neon") {
        unsafe { NeonKernel8x8::<f32>::new() }
    } else {
        println!("neon feature is not supported");
        return;
    };
    direct_matches_buffered(direct);
}

#[test]
fn test_neon4x4_direct_matches_buffered() {
    if !cfg!(target_feature = "neon") {
        return;
    }
    // SAFETY: NEON is enabled for this target.
    direct_matches_buffered(unsafe { NeonKernel4x4::<f32>::new() });
}

fn direct_matches_buffered<K: Kernel<Scalar = f32> + Copy>(direct: K) {
    let buffered = BufferedKernel(direct);
    let d = K::MR;

    // `MatRef::row_major` rejects a zero dimension, so k = 0 is not representable.
    let (dims, ks): (&[usize], &[usize]) = if cfg!(miri) {
        (&[d, d + 1], &[1, 7])
    } else {
        (
            &[1, d - 1, d, d + 1, 2 * d, 2 * d + 1, 3 * d],
            &[1, 2, 7, 64],
        )
    };
    let layouts: &[CLayout] = if cfg!(miri) {
        &[CLayout::Row, CLayout::Col, CLayout::ColStrided]
    } else {
        &C_LAYOUTS
    };

    // sweep the shapes, cycling through the alpha/beta combinations
    let mut i = 0usize;
    for &m in dims {
        for &n in dims {
            for &k in ks {
                for &layout in layouts {
                    for big_pack in [false, true] {
                        let alpha = ALPHAS[i % ALPHAS.len()];
                        let beta = BETAS[(i / ALPHAS.len()) % BETAS.len()];
                        i += 1;
                        check_direct_matches_buffered(
                            &direct, &buffered, m, k, n, layout, alpha, beta, big_pack,
                        );
                    }
                }
            }
        }
    }

    // and sweep every alpha/beta combination on a few shapes
    let shapes: &[(usize, usize, usize)] = if cfg!(miri) {
        &[(d, 7, d + 1)]
    } else {
        &[
            (d, 7, d),
            (2 * d, 7, 2 * d + 1),
            (3 * d, 2, d + 1),
            (2 * d + 1, 64, 3 * d),
        ]
    };
    let ab_layouts: &[CLayout] = if cfg!(miri) {
        &[CLayout::Row, CLayout::ColStrided]
    } else {
        layouts
    };
    let ab_packs: &[bool] = if cfg!(miri) { &[true] } else { &[false, true] };
    for &(m, k, n) in shapes {
        for &layout in ab_layouts {
            for &big_pack in ab_packs {
                for &alpha in ALPHAS.iter() {
                    for &beta in BETAS.iter() {
                        check_direct_matches_buffered(
                            &direct, &buffered, m, k, n, layout, alpha, beta, big_pack,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn test_neon8x8_zero_beta_preserves_nonfinite_c() {
    if !cfg!(target_feature = "neon") {
        return;
    }
    // SAFETY: NEON is enabled for this target.
    let kernel = unsafe { NeonKernel8x8::<f32>::new() };
    // One full tile plus edges exercises direct writes and buffered writes together.
    let a = MatRef::col_major(9, 1, &[1.0; 9]);
    let b = MatRef::row_major(1, 9, &[2.0; 9]);
    let pack_sizes = PackSizes {
        mc: 16,
        kc: 1,
        nc: 16,
    };
    let mut packing = vec![0.0; pack_sizes.buf_len()];
    for layout in [CLayout::Row, CLayout::Col] {
        let (rs, cs) = layout.strides(9, 9);
        for beta in [0.0, -0.0] {
            let mut values: Vec<_> = (0..81)
                .map(|i| [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0][i % 4])
                .collect();
            let mut c = MatMut::from_parts(9, 9, &mut values, rs, cs).unwrap();
            kernel.gemm(1.0, a, b, beta, &mut c, pack_sizes, &mut packing);
            for (i, value) in values.iter().enumerate() {
                if i % 4 == 3 {
                    assert_eq!(*value, 2.0);
                } else {
                    assert!(value.is_nan(), "beta={beta} layout={layout:?} at {i}");
                }
            }
        }
    }
}
