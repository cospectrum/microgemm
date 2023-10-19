use microgemm::{Kernel, MatMut, MatRef};

struct CustomKernel;

impl Kernel<f64> for CustomKernel {
    const MR: usize = 2;
    const NR: usize = 2;

    // dst <- alpha lhs rhs + beta dst
    #[allow(unused_variables)]
    fn microkernel(
        &self,
        alpha: f64,
        lhs: &MatRef<f64>,
        rhs: &MatRef<f64>,
        beta: f64,
        dst: &mut MatMut<f64>,
    ) {
        assert_eq!(lhs.nrows(), Self::MR);
        assert_eq!(rhs.ncols(), Self::NR);
        // your implementation...
    }
}
