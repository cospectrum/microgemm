mod base;

pub use base::{MatMut, MatRef};

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Layout {
    RowMajor,
    ColumnMajor,
    General {
        row_stride: usize,
        col_stride: usize,
    },
}

impl AsRef<Layout> for Layout {
    fn as_ref(&self) -> &Layout {
        self
    }
}
