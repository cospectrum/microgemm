use crate::MatRef;
use core::ops::Range;
use num_traits::{One, Zero};

#[inline]
pub(crate) fn pack_a<T>(
    mr: usize,
    apack: &mut [T],
    a: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: One + Zero + Copy,
{
    if a.nrows() < rows.end {
        pack_with_padding(mr, apack, a, rows, cols);
        return;
    };

    let mc = rows.len();
    let kc = cols.len();
    assert_eq!(apack.len(), mc * kc);
    assert_eq!(mc % mr, 0);

    assert!(cols.end <= a.ncols());
    assert!(rows.end <= a.nrows());

    let block_size = mr * kc;
    let start = rows.start;
    let blocks = mc / mr;

    for i in 0..blocks {
        let buf = &mut apack[block_size * i..block_size * (i + 1)];
        let rows = start + mr * i..start + mr * (i + 1);

        let mut it = buf.iter_mut();

        for col in cols.clone() {
            for row in rows.start..rows.end {
                let dst = it.next().unwrap();
                *dst = a.get(row, col);
            }
        }
    }
}

fn pack_with_padding<T>(
    mr: usize,
    apack: &mut [T],
    a: &MatRef<T>,
    rows: Range<usize>,
    cols: Range<usize>,
) where
    T: One + Zero + Copy,
{
    let mc = rows.len();
    let kc = cols.len();
    assert_eq!(apack.len(), mc * kc);
    assert_eq!(mc % mr, 0);

    assert!(cols.end <= a.ncols());
    assert!(rows.start < a.nrows());

    let block_size = mr * kc;
    let start = rows.start;

    for i in 0..mc / mr {
        let buf = &mut apack[block_size * i..block_size * (i + 1)];
        let rows = start + mr * i..start + mr * (i + 1);

        let mut it = buf.iter_mut();

        for col in cols.clone() {
            for row in rows.start..rows.end {
                let dst = it.next().unwrap();
                if row < a.nrows() {
                    *dst = a.get(row, col);
                } else {
                    *dst = T::zero();
                }
            }
        }
    }
}
