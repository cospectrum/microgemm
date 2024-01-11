use crate::{Layout, MatRef};
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn pack_b<T>(
    nr: usize,
    bpack: &mut [T],
    b: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) -> Layout
where
    T: One + Zero + Copy,
{
    if b.ncols() < cols.end {
        pack_with_padding(nr, bpack, b, rows, cols);
        return Layout::RowMajor;
    };

    let kc = rows.len();
    let nc = cols.len();
    assert_eq!(bpack.len(), kc * nc);
    assert_eq!(nc % nr, 0);

    assert!(rows.end <= b.nrows());
    assert!(cols.end <= b.ncols());

    let block_size = kc * nr;
    let start = cols.start;

    for i in 0..nc / nr {
        let buf = &mut bpack[block_size * i..block_size * (i + 1)];
        let cols = start + nr * i..start + nr * (i + 1);

        let mut it = buf.iter_mut();

        for row in rows.clone() {
            for col in cols.start..cols.end {
                let dst = it.next().unwrap();
                *dst = b.get(row, col);
            }
        }
    }
    Layout::RowMajor
}

fn pack_with_padding<T>(
    nr: usize,
    bpack: &mut [T],
    b: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: One + Zero + Copy,
{
    let kc = rows.len();
    let nc = cols.len();
    assert_eq!(bpack.len(), kc * nc);
    assert_eq!(nc % nr, 0);

    assert!(rows.end <= b.nrows());
    assert!(cols.start < b.ncols());

    let block_size = kc * nr;
    let start = cols.start;

    for i in 0..nc / nr {
        let buf = &mut bpack[block_size * i..block_size * (i + 1)];
        let cols = start + nr * i..start + nr * (i + 1);

        let mut it = buf.iter_mut();

        for row in rows.clone() {
            for col in cols.start..cols.end {
                let dst = it.next().unwrap();
                if col < b.ncols() {
                    *dst = b.get(row, col);
                } else {
                    *dst = T::zero();
                }
            }
        }
    }
}
