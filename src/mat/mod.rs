mod layout;

pub mod base;

pub use layout::Layout;

pub type MatRef<'a, T> = base::MatBase<&'a [T], T>;
pub type MatMut<'a, T> = base::MatBase<&'a mut [T], T>;

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
