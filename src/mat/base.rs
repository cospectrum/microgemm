use crate::Zero;
use core::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct MatBase<V, T> {
    pub(super) nrows: usize,
    pub(super) ncols: usize,
    pub(super) values: V,
    pub(super) row_stride: usize,
    pub(super) col_stride: usize,
    marker: PhantomData<T>,
}

impl<V, T> MatBase<V, T> {
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
    pub(crate) fn checked_idx(&self, row: usize, col: usize) -> Option<usize> {
        if !self.in_bounds(row, col) {
            return None;
        }
        let row_at = row.checked_mul(self.row_stride)?;
        let col_at = col.checked_mul(self.col_stride)?;
        row_at.checked_add(col_at)
    }
    pub(crate) fn idx(&self, row: usize, col: usize) -> usize {
        debug_assert!(row < self.nrows());
        debug_assert!(col < self.ncols());
        row * self.row_stride + col * self.col_stride
    }
    #[inline]
    pub(crate) fn in_bounds(&self, row: usize, col: usize) -> bool {
        row < self.nrows() && col < self.ncols()
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsRef<[T]>,
{
    /// Creates a matrix from a given number of rows/columns, values and strides.
    /// Returns `None` if the last index overflows.
    pub fn from_parts(
        nrows: usize,
        ncols: usize,
        values: V,
        row_stride: usize,
        col_stride: usize,
    ) -> Option<Self> {
        let mat = Self {
            values,
            nrows,
            ncols,
            row_stride,
            col_stride,
            marker: PhantomData,
        };
        let last_row = nrows.checked_sub(1)?;
        let last_col = ncols.checked_sub(1)?;
        let last_idx = mat.checked_idx(last_row, last_col)?;
        if last_idx < mat.as_slice().len() {
            Some(mat)
        } else {
            None
        }
    }
    /// Creates a matrix with row-major layout.
    ///
    /// # Panics
    /// 1. If `values.as_ref().len() != nrows * ncols`.
    /// 2. If the last index overflows.
    pub fn row_major(nrows: usize, ncols: usize, values: V) -> Self {
        assert_eq!(values.as_ref().len(), nrows.checked_mul(ncols).unwrap());
        let (row_stride, col_stride) = (ncols, 1);
        Self::from_parts(nrows, ncols, values, row_stride, col_stride).unwrap()
    }
    /// Creates a matrix with col-major layout.
    ///
    /// # Panics
    /// 1. If `values.as_ref().len() != nrows * ncols`.
    /// 2. If the last index overflows.
    pub fn col_major(nrows: usize, ncols: usize, values: V) -> Self {
        assert_eq!(values.as_ref().len(), nrows.checked_mul(ncols).unwrap());
        let (row_stride, col_stride) = (1, nrows);
        Self::from_parts(nrows, ncols, values, row_stride, col_stride).unwrap()
    }
    /// Extracts a slice containing the matrix values.
    ///
    /// # Examples
    ///
    /// ```
    /// use microgemm::MatRef;
    ///
    /// let values = [1, 2, 3, 4];
    /// let mat = MatRef::row_major(2, 2, &values);
    /// assert_eq!(mat.as_slice(), &values);
    /// ```
    pub fn as_slice(&self) -> &[T] {
        self.values.as_ref()
    }
    pub fn as_ptr(&self) -> *const T {
        self.as_slice().as_ptr()
    }
}

impl<V, T> MatBase<V, T>
where
    V: AsRef<[T]>,
    T: Copy,
{
    /// Returns an element at (row, col).
    ///
    /// # Panics
    /// Panics if `row * mat.row_stride() + col * mat.col_stride() >= mat.as_slice().len()`
    ///
    /// # Examples
    ///
    /// ```
    /// use microgemm::MatRef;
    ///
    /// let values = [1, 2, 3, 4];
    /// let mat = MatRef::row_major(2, 2, &values);
    /// assert_eq!(mat.get(1, 0), 3);
    /// ```
    pub fn get(&self, row: usize, col: usize) -> T {
        self.as_slice()[self.idx(row, col)]
    }
    pub(crate) fn get_or(&self, row: usize, col: usize, default: T) -> T {
        if self.in_bounds(row, col) {
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
    /// use microgemm::MatMut;
    ///
    /// let mut values = [1, 2, 3, 4];
    /// let mut mat = MatMut::row_major(2, 2, &mut values);
    /// assert_eq!(mat.as_mut_slice(), &mut [1, 2, 3, 4]);
    /// ```
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.values.as_mut()
    }
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.as_mut_slice().as_mut_ptr()
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
    /// use microgemm::MatMut;
    ///
    /// let mut values = [1, 2, 3, 4];
    /// let mut mat = MatMut::row_major(2, 2, &mut values);
    /// let x = mat.get_mut(1, 0);
    /// assert_eq!(*x, 3);
    /// *x = 0;
    /// assert_eq!(values, [1, 2, 0, 4]);
    /// ```
    pub fn get_mut(&mut self, row: usize, col: usize) -> &mut T {
        let idx = self.idx(row, col);
        &mut self.as_mut_slice()[idx]
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[rustfmt::skip]
    #[test]
    fn test_mat_ref_row_major() {
        let values = [
            1, 2,
            3, 4,
        ];
        let mat = MatRef::row_major(2, 2, &values);
        let unpack = [
            mat.get(0, 0), mat.get(0, 1),
            mat.get(1, 0), mat.get(1, 1),
        ];
        assert_eq!(unpack, values);
    }
    #[rustfmt::skip]
    #[test]
    fn test_mat_ref_col_major() {
        let values = [
            1, 3,
            2, 4,
        ];
        let values_t = [
            1, 2,
            3, 4,
        ];
        let mat = MatRef::col_major(2, 2, &values);
        let unpack = [
            mat.get(0, 0), mat.get(0, 1),
            mat.get(1, 0), mat.get(1, 1),
        ];
        assert_eq!(unpack, values_t);
    }
    #[rustfmt::skip]
    #[test]
    fn test_mat_mut() {
        let mut values = [
            1, 3,
            2, 4,
        ];
        let mut mat = MatMut::col_major(2, 2, &mut values);

        let unpack = [
            *mat.get_mut(0, 0), *mat.get_mut(0, 1),
            *mat.get_mut(1, 0), *mat.get_mut(1, 1),
        ];
        assert_eq!(unpack, [1, 2, 3, 4]);

        *mat.get_mut(0, 1) = -2;
        let expect = [
            1, 3,
            -2, 4,
        ];
        assert_eq!(mat.as_slice(), expect);
    }
}
