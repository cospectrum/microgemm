use crate::Layout;
use core::marker::PhantomData;
use num_traits::Zero;

#[derive(Debug, Clone)]
pub struct MatBase<V, T> {
    pub(super) nrows: usize,
    pub(super) ncols: usize,
    pub(super) values: V,
    pub(super) row_stride: usize,
    pub(super) col_stride: usize,
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
            Layout::ColMajor => (1, nrows),
        };
        Self::from_parts(nrows, ncols, values, row_stride, col_stride)
    }
    /// Extracts a slice containing the matrix values.
    ///
    /// # Examples
    ///
    /// ```
    /// use microgemm::{MatRef, Layout};
    ///
    /// let values = [1, 2, 3, 4];
    /// let mat = MatRef::new(2, 2, &values, Layout::RowMajor);
    /// assert_eq!(mat.as_slice(), &values);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.values.as_ref()
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsRef<[T]>,
    T: Copy,
{
    /// Returns an element at (row, col)
    ///
    /// # Panics
    ///
    /// Panics if `row * mat.row_stride() + col * mat.col_stride() >= mat.as_slice().len()`
    ///
    /// # Examples
    ///
    /// ```
    /// use microgemm::{MatRef, Layout};
    ///
    /// let values = [1, 2, 3, 4];
    /// let mat = MatRef::new(2, 2, &values, Layout::RowMajor);
    /// assert_eq!(mat.get(1, 0), 3);
    /// ```
    pub fn get(&self, row: usize, col: usize) -> T {
        self.values.as_ref()[self.idx(row, col)]
    }
    pub(crate) fn get_or(&self, row: usize, col: usize, default: T) -> T {
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
    pub(crate) fn get_or_zero(&self, row: usize, col: usize) -> T {
        self.get_or(row, col, T::zero())
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsMut<[T]>,
{
    /// Extracts a mutable slice containing the matrix values.
    ///
    /// # Examples
    ///
    /// ```
    /// use microgemm::{MatMut, Layout};
    ///
    /// let mut values = [1, 2, 3, 4];
    /// let mut mat = MatMut::new(2, 2, &mut values, Layout::RowMajor);
    /// assert_eq!(mat.as_mut_slice(), &mut [1, 2, 3, 4]);
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.values.as_mut()
    }
    /// Returns a mutable reference to an element at (row, col)
    ///
    /// # Panics
    ///
    /// Panics if `row * mat.row_stride() + col * mat.col_stride() >= mat.as_mut_slice().len()`
    ///
    /// # Examples
    ///
    /// ```
    /// use microgemm::{MatMut, Layout};
    ///
    /// let mut values = [1, 2, 3, 4];
    /// let mut mat = MatMut::new(2, 2, &mut values, Layout::RowMajor);
    /// let x = mat.get_mut(1, 0);
    /// assert_eq!(*x, 3);
    /// *x = 0;
    /// assert_eq!(values, [1, 2, 0, 4]);
    /// ```
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let idx = self.idx(row, col);
        &mut self.values.as_mut()[idx]
    }
}
