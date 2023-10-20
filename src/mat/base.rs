use crate::Layout;
use core::marker::PhantomData;
use num_traits::Zero;

pub type MatRef<'a, T> = MatBase<&'a [T], T>;
pub type MatMut<'a, T> = MatBase<&'a mut [T], T>;

#[derive(Debug, Clone)]
pub struct MatBase<V, T> {
    nrows: usize,
    ncols: usize,
    values: V,
    row_stride: usize,
    col_stride: usize,
    marker: PhantomData<T>,
}

impl<V, T> MatBase<V, T> {
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        values: V,
        row_stride: usize,
        col_stride: usize,
    ) -> Self {
        Self {
            values,
            nrows,
            ncols,
            row_stride,
            col_stride,
            marker: PhantomData,
        }
    }
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn row_stride(&self) -> usize {
        self.row_stride
    }
    pub fn col_stride(&self) -> usize {
        self.col_stride
    }
    pub fn with_values(&self, values: V) -> Self {
        Self { values, ..*self }
    }
    pub(crate) fn idx(&self, row: usize, col: usize) -> usize {
        row * self.row_stride + col * self.col_stride
    }
    pub(crate) fn is_row_major(&self) -> bool {
        !self.is_col_major()
    }
    pub(crate) fn is_col_major(&self) -> bool {
        self.row_stride < self.col_stride
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsRef<[T]>,
{
    pub fn new(nrows: usize, ncols: usize, values: V, layout: Layout) -> Self {
        assert_eq!(values.as_ref().len(), nrows * ncols);
        Self::new_unchecked(nrows, ncols, values, layout)
    }
    pub(crate) fn new_unchecked(nrows: usize, ncols: usize, values: V, layout: Layout) -> Self {
        let (row_stride, col_stride) = match layout {
            Layout::RowMajor => (ncols, 1),
            Layout::ColumnMajor => (1, nrows),
        };
        Self::from_parts(nrows, ncols, values, row_stride, col_stride)
    }
    pub fn as_slice(&self) -> &[T] {
        self.values.as_ref()
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsRef<[T]>,
    T: Copy,
{
    pub fn get(&self, row: usize, col: usize) -> T {
        self.values.as_ref()[self.idx(row, col)]
    }
    pub fn get_or(&self, row: usize, col: usize, default: T) -> T {
        if row < self.nrows && col < self.ncols {
            self.get(row, col)
        } else {
            default
        }
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsRef<[T]>,
    T: Copy + Zero,
{
    pub fn get_or_zero(&self, row: usize, col: usize) -> T {
        if row < self.nrows && col < self.ncols {
            self.get(row, col)
        } else {
            T::zero()
        }
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsMut<[T]>,
{
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.values.as_mut()
    }
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let idx = self.idx(row, col);
        &mut self.values.as_mut()[idx]
    }
}

impl<'a, T> MatMut<'a, T> {
    pub fn to_ref(&'a self) -> MatRef<'a, T> {
        MatRef::from_parts(
            self.nrows,
            self.ncols,
            self.values,
            self.row_stride,
            self.col_stride,
        )
    }
}

impl<'a, T> AsRef<MatRef<'a, T>> for MatRef<'a, T> {
    fn as_ref(&self) -> &MatRef<'a, T> {
        self
    }
}

impl<'a, T> AsMut<MatMut<'a, T>> for MatMut<'a, T> {
    fn as_mut(&mut self) -> &mut MatMut<'a, T> {
        self
    }
}
