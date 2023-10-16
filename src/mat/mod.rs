mod base;

pub use base::{MatMut, MatRef};

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Layout {
    RowMajor,
    ColumnMajor,
}

impl AsRef<Layout> for Layout {
    fn as_ref(&self) -> &Layout {
        self
    }
}
