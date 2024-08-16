use core::fmt;
use core::ops::Mul;

use crate::utils::{arb_matrix_triple_with, arb_pack_sizes, naive_gemm};
use crate::Kernel;
use crate::{as_mut, std_prelude::*};
use proptest::sample::size_range;
use proptest::test_runner::TestCaseResult;
use proptest::{prelude::*, sample::SizeRange};

type AssertEq<T> = dyn Fn(&[T], &[T]) -> TestCaseResult;

pub struct ProptestKernelCfg<T> {
    pub mkn: [SizeRange; 3],
    pub mc: SizeRange,
    pub kc: SizeRange,
    pub nc: SizeRange,
    pub scalar: BoxedStrategy<T>,
    pub cmp: Option<Box<AssertEq<T>>>,
}

impl<T> Default for ProptestKernelCfg<T>
where
    T: Arbitrary,
    T::Strategy: 'static,
{
    fn default() -> Self {
        let dim = 20;
        let mat_dim = size_range(1..=dim);
        let pack_dim = size_range(1..=2 * dim + 1);
        Self {
            scalar: T::arbitrary().boxed(),
            mkn: [mat_dim.clone(), mat_dim.clone(), mat_dim.clone()],
            mc: pack_dim.clone(),
            kc: pack_dim.clone(),
            nc: pack_dim.clone(),
            cmp: None,
        }
    }
}

impl<T> ProptestKernelCfg<T> {
    #[allow(dead_code)]
    pub fn with_cmp<F>(mut self, cmp: F) -> Self
    where
        F: 'static + Fn(&[T], &[T]) -> TestCaseResult,
    {
        self.cmp = Some(Box::new(cmp));
        self
    }
    #[allow(dead_code)]
    pub fn with_scalar(mut self, scalar: BoxedStrategy<T>) -> Self {
        self.scalar = scalar;
        self
    }
    #[allow(dead_code)]
    pub fn with_max_matrix_dim(self, dim: usize) -> Self {
        let range = size_range(1..=dim);
        let mkn = [range.clone(), range.clone(), range.clone()];
        Self { mkn, ..self }
    }
    #[allow(dead_code)]
    pub fn with_max_pack_dim(self, dim: usize) -> Self {
        let range = size_range(1..=dim);
        Self {
            mc: range.clone(),
            kc: range.clone(),
            nc: range.clone(),
            ..self
        }
    }
}

pub fn proptest_kernel<T, K>(kernel: &K, cfg: ProptestKernelCfg<T>) -> TestCaseResult
where
    K: Kernel<Scalar = T>,
    T: fmt::Debug + Copy + PartialEq + 'static + Mul<Output = T> + crate::Zero,
{
    let cmp = match cfg.cmp {
        Some(f) => f,
        None => {
            let cmp = |a: &[T], b: &[T]| -> TestCaseResult {
                prop_assert_eq!(a.len(), b.len());
                for (&left, &right) in a.iter().zip(b) {
                    prop_assert_eq!(left, right);
                }
                Ok(())
            };
            Box::new(cmp)
        }
    };

    let arb_pack_sizes = arb_pack_sizes(kernel, cfg.mc, cfg.kc, cfg.nc);

    let [m, k, n] = cfg.mkn;
    let triples = arb_matrix_triple_with(m, k, n, cfg.scalar.clone());
    let alphas = cfg.scalar.clone();
    let betas = cfg.scalar.clone();

    proptest!(|(
        [a, b, c] in triples,
        alpha in alphas,
        beta in betas,
    )| {
        let [a, b] = [a.to_ref(), b.to_ref()];
        let mut expect = c.clone();
        naive_gemm(alpha, a, b, beta, as_mut!(expect));

        proptest!(|(pack_sizes in arb_pack_sizes.clone())| {
            let mut actual = c.clone();
            kernel.gemm_in(
                crate::GlobalAllocator,
                alpha,
                a,
                b,
                beta,
                as_mut!(actual),
                pack_sizes,
            );
            cmp(expect.as_slice(), actual.as_slice())?;
        });
    });

    Ok(())
}
