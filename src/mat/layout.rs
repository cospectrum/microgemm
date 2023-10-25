#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Layout {
    RowMajor,
    ColMajor,
}

impl AsRef<Layout> for Layout {
    fn as_ref(&self) -> &Layout {
        self
    }
}
