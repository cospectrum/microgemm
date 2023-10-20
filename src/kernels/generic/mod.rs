mod generic_2x2;
mod generic_4x4;
mod generic_8x8;

use core::ops::{Add, Mul};

pub use generic_2x2::*;
pub use generic_4x4::*;
pub use generic_8x8::*;

fn square_microkernel<T, const DIM: usize>(lhs: &[T], rhs: &[T], cols: &mut [T])
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(cols.len(), DIM * DIM);
    assert_eq!(lhs.len() % DIM, 0);
    assert_eq!(lhs.len(), rhs.len());

    let left = lhs.chunks_exact(DIM);
    let right = rhs.chunks_exact(DIM);

    left.zip(right).for_each(|(a, b)| {
        let cols = cols.chunks_exact_mut(DIM);

        cols.zip(b).for_each(|(col, &scalar)| {
            col.iter_mut().zip(a).for_each(|(out, &x)| {
                *out = *out + x * scalar;
            });
        });
    });
}

fn write_to_col_major<T, const DIM: usize>(dst: &mut [T], cols: &[T], alpha: T, beta: T)
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    assert_eq!(dst.len(), DIM * DIM);
    assert_eq!(cols.len(), dst.len());
    dst.iter_mut().zip(cols).for_each(|(to, &from)| {
        *to = alpha * from + beta * *to;
    });
}
