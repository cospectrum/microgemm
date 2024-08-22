use crate::mat::base::MatBase;
use crate::{Kernel, MatRef, PackSizes};
use proptest::prelude::*;
use std::prelude::rust_2021::*;
use std::{fmt, ops::RangeInclusive};

#[allow(unused_imports)]
use proptest::prelude::*;
use proptest::sample::SizeRange;

type Mat<T> = MatBase<Vec<T>, T>;

impl<T> Mat<T> {
    pub fn to_ref(&self) -> MatRef<T> {
        MatRef::from_parts(
            self.nrows(),
            self.ncols(),
            self.as_slice(),
            self.row_stride(),
            self.col_stride(),
        )
    }
}

#[macro_export]
macro_rules! as_mut {
    ($mat:ident) => {{
        let [nrows, ncols] = [$mat.nrows(), $mat.ncols()];
        let [row_stride, col_stride] = [$mat.row_stride(), $mat.col_stride()];
        &mut $crate::MatMut::from_parts(nrows, ncols, $mat.as_mut_slice(), row_stride, col_stride)
    }};
}

pub fn arb_pack_sizes<T, K>(
    _: &K,
    mc: impl Into<SizeRange>,
    kc: impl Into<SizeRange>,
    nc: impl Into<SizeRange>,
) -> BoxedStrategy<PackSizes>
where
    K: Kernel<Scalar = T>,
{
    let mc = to_range(mc).prop_filter("mr <= mc", |&mc| K::MR <= mc);
    let kc = to_range(kc);
    let nc = to_range(nc).prop_filter("nr <= nc", |&nc| K::NR <= nc);

    mc.prop_flat_map(move |mc| {
        let nc = nc.clone();
        kc.clone()
            .prop_flat_map(move |kc| nc.clone().prop_map(move |nc| PackSizes { mc, kc, nc }))
    })
    .boxed()
}

#[allow(dead_code)]
pub fn arb_matrix_triple<T>(
    m: impl Into<SizeRange>,
    k: impl Into<SizeRange>,
    n: impl Into<SizeRange>,
) -> BoxedStrategy<[Mat<T>; 3]>
where
    T: Arbitrary + fmt::Debug + Clone + 'static,
    T::Strategy: Clone + 'static,
{
    arb_matrix_triple_with(m, k, n, any::<T>())
}

pub fn arb_matrix_triple_with<T>(
    m: impl Into<SizeRange>,
    k: impl Into<SizeRange>,
    n: impl Into<SizeRange>,
    scalars: impl Strategy<Value = T> + Clone + 'static,
) -> BoxedStrategy<[Mat<T>; 3]>
where
    T: fmt::Debug + Clone + 'static,
{
    let [m, k, n] = [m.into(), k.into(), n.into()];
    arb_matrix_with(m.clone(), k.clone(), scalars.clone())
        .prop_flat_map(move |a| {
            let a = a.clone();
            let scalars = scalars.clone();
            arb_matrix_with(a.ncols(), n.clone(), scalars.clone()).prop_flat_map(move |b| {
                let a = a.clone();
                let b = b.clone();
                let scalars = scalars.clone();
                arb_matrix_with(a.nrows(), b.ncols(), scalars)
                    .prop_map(move |c| [a.clone(), b.clone(), c])
            })
        })
        .boxed()
}

pub fn arb_matrix<T>(
    nrows: impl Into<SizeRange>,
    ncols: impl Into<SizeRange>,
) -> BoxedStrategy<Mat<T>>
where
    T: Arbitrary + fmt::Debug + Clone,
    T::Strategy: Clone + 'static,
{
    arb_matrix_with(nrows, ncols, any::<T>())
}

#[derive(Debug, Clone, Copy)]
enum Layout {
    Rowmajor,
    Colmajor,
}

/// Create a strategy to generate matrices with a given scalar strategy and
/// dimensions selected within the specified ranges.
pub(crate) fn arb_matrix_with<T>(
    nrows: impl Into<SizeRange>,
    ncols: impl Into<SizeRange>,
    scalars: impl Strategy<Value = T> + Clone + 'static,
) -> BoxedStrategy<Mat<T>>
where
    T: fmt::Debug + Clone,
{
    layout_dims(nrows.into(), ncols.into())
        .prop_flat_map(move |((r, c), layout)| fixed_matrix(r, c, scalars.clone(), layout))
        .boxed()
}

prop_compose! {
    fn layout_dims(nrows_range: SizeRange, ncols_range: SizeRange)(
        dim in dims(nrows_range, ncols_range),
        layout in arb_layout(),
    ) -> ((usize, usize), Layout) {
        (dim, layout)
    }
}

fn arb_layout() -> impl Strategy<Value = Layout> {
    prop_oneof![Just(Layout::Rowmajor), Just(Layout::Colmajor)]
}

fn fixed_matrix<T>(
    nrows: usize,
    ncols: usize,
    strategy: impl Strategy<Value = T> + 'static,
    layout: Layout,
) -> BoxedStrategy<Mat<T>>
where
    T: Clone + fmt::Debug,
{
    let size = nrows * ncols;
    let vecs = proptest::collection::vec(strategy, size);

    vecs.prop_map(move |v| match layout {
        Layout::Rowmajor => Mat::row_major(nrows, ncols, v),
        Layout::Colmajor => Mat::col_major(nrows, ncols, v),
    })
    .boxed()
}

fn dims(nrows: impl Into<SizeRange>, ncols: impl Into<SizeRange>) -> BoxedStrategy<(usize, usize)> {
    let nrows = to_range(nrows);
    let ncols = to_range(ncols);
    nrows
        .prop_flat_map(move |r| ncols.clone().prop_map(move |c| (r, c)))
        .boxed()
}

fn to_range(range: impl Into<SizeRange>) -> RangeInclusive<usize> {
    let range = range.into();
    range.start()..=range.end_incl()
}
