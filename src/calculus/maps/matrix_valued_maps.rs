use std::marker::PhantomData;

use crate::calculus::dual::dual_matrix::DualM;
use crate::calculus::dual::dual_vector::DualV;
use crate::calculus::types::matrix::IsMatrix;
use crate::calculus::types::M;
use crate::calculus::types::V;
use crate::tensor::element::SMat;
use crate::tensor::mut_tensor::MutTensorDDRC;
use crate::tensor::mut_tensor::MutTensorDRC;
use crate::tensor::mut_view::IsMutTensorLike;

// Matrix-valued map on a vector space.
//
// This is a function which takes a vector and returns a matrix:
//
//  f: ℝᵐ -> ℝʳ x ℝᶜ
//
pub struct MatrixValuedMapFromVector;

impl MatrixValuedMapFromVector {
    // Finite difference quotient of the matrix-valued map.
    //
    // The derivative is a rank-3 tensor with shape (Rₒ x Cₒ x Rᵢ).
    //
    // For efficiency reasons, we return Rᵢ x [Rₒ x Cₒ]
    pub fn sym_diff_quotient<TFn, const OUTROWS: usize, const OUTCOLS: usize, const INROWS: usize>(
        matrix_valued: TFn,
        a: V<INROWS>,
        eps: f64,
    ) -> MutTensorDRC<f64, OUTROWS, OUTCOLS>
    where
        TFn: Fn(V<INROWS>) -> M<OUTROWS, OUTCOLS>,
    {
        let mut out = MutTensorDRC::<f64, OUTROWS, OUTCOLS>::from_shape([INROWS]);
        for i1 in 0..INROWS {
            let mut a_plus = a;

            a_plus[i1] += eps;

            let mut a_minus = a;
            a_minus[i1] -= eps;

            let val = (matrix_valued(a_plus) - matrix_valued(a_minus)).scaled(1.0 / (2.0 * eps));

            *out.mut_view().get_mut([i1]) = val;
        }
        out
    }

    pub fn fw_autodiff<TFn, const OUTROWS: usize, const OUTCOLS: usize, const INROWS: usize>(
        matrix_valued: TFn,
        a: V<INROWS>,
    ) -> MutTensorDRC<f64, OUTROWS, OUTCOLS>
    where
        TFn: Fn(DualV<INROWS>) -> DualM<OUTROWS, OUTCOLS>,
    {
        MutTensorDRC {
            mut_array: matrix_valued(DualV::v(a))
                .dij_val
                .unwrap()
                .mut_array
                .into_shape([INROWS])
                .unwrap(),
            phantom: PhantomData,
        }
    }
}

// Matrix-valued map on a product space (=matrices).
//
// This is a function which takes a matrix and returns a matrix:
//
//  f: ℝᵐ x ℝⁿ -> ℝʳ x ℝᶜ
//
pub struct MatrixValuedMapFromMatrix;

impl MatrixValuedMapFromMatrix {
    // Finite difference quotient of the matrix-valued map.
    //
    // The derivative is a rank-4 tensor with shape (Rₒ x Cₒ x Rᵢ x Cᵢ).
    //
    // For efficiency reasons, we return Rᵢ x Cᵢ x [Rₒ x Cₒ]
    pub fn sym_diff_quotient<
        TFn,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        vector_field: TFn,
        a: M<INROWS, INCOLS>,
        eps: f64,
    ) -> MutTensorDDRC<f64, OUTROWS, OUTCOLS>
    where
        TFn: Fn(M<INROWS, INCOLS>) -> M<OUTROWS, OUTCOLS>,
    {
        let mut out = MutTensorDDRC::<f64, OUTROWS, OUTCOLS>::from_shape_and_val(
            [INROWS, INCOLS],
            SMat::<f64, OUTROWS, OUTCOLS>::zeros(),
        );
        for i1 in 0..INROWS {
            for i0 in 0..INCOLS {
                let mut a_plus = a;

                a_plus[(i1, i0)] += eps;

                let mut a_minus = a;
                a_minus[(i1, i0)] -= eps;

                let val = (vector_field(a_plus) - vector_field(a_minus)) / (2.0 * eps);

                *out.mut_view().get_mut([i1, i0]) = val;
            }
        }
        out
    }

    pub fn fw_autodiff<
        TFn,
        const OUTROWS: usize,
        const OUTCOLS: usize,
        const INROWS: usize,
        const INCOLS: usize,
    >(
        matrix_valued: TFn,
        a: M<INROWS, INCOLS>,
    ) -> MutTensorDDRC<f64, OUTROWS, OUTCOLS>
    where
        TFn: Fn(DualM<INROWS, INCOLS>) -> DualM<OUTROWS, OUTCOLS>,
    {
        matrix_valued(DualM::v(a)).dij_val.unwrap()
    }
}

mod test {

    #[test]
    fn test_batched_matrix_valued_map_from_vector() {
        use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromVector;
        use crate::calculus::types::matrix::IsMatrix;
        use crate::calculus::types::scalar::IsScalar;
        use crate::calculus::types::vector::IsVector;
        use crate::calculus::types::V;
        use crate::tensor::view::IsTensorLike;

        //      [[ i ]]
        //      [[   ]]
        //      [[ j ]]      [[                      ]]
        //      [[   ]]      [[   0   -k    j    x   ]]
        //      [[ k ]]      [[                      ]]
        //  hat [[   ]]   =  [[   k    0   -i    y   ]]
        //      [[ x ]]      [[                      ]]
        //      [[   ]]      [[  -j    i    0    z   ]]
        //      [[ y ]]      [[                      ]]
        //      [[   ]]
        //      [[ z ]]
        fn hat_fn<S: IsScalar, M34: IsMatrix<S, 3, 4>, V6: IsVector<S, 6>>(v: V6) -> M34 {
            let i = v.get(0);
            let j = v.get(1);
            let k = v.get(2);
            let ni = -i.clone();
            let nj = -j.clone();
            let nk = -k.clone();
            let x = v.get(3);
            let y = v.get(4);
            let z = v.get(5);

            let ret: M34 = M34::from_array2([
                [S::c(0.0), nk, j, x],
                [k, S::c(0.0), ni, y],
                [nj, i, S::c(0.0), z],
            ]);
            ret
        }

        let a = V::<6>::new(0.1, 0.2, 0.4, 0.7, 0.8, 0.9);

        let finite_diff = MatrixValuedMapFromVector::sym_diff_quotient(hat_fn, a, 1e-6);
        let auto_grad = MatrixValuedMapFromVector::fw_autodiff(hat_fn, a);
        approx::assert_abs_diff_eq!(
            finite_diff.view().elem_view(),
            auto_grad.view().elem_view(),
            epsilon = 0.0001
        );
    }

    #[test]
    fn test_batched_matrix_valued_map_from_matrix() {
        use crate::calculus::maps::matrix_valued_maps::MatrixValuedMapFromMatrix;
        use crate::calculus::types::matrix::IsMatrix;
        use crate::calculus::types::scalar::IsScalar;
        use crate::calculus::types::M;
        use crate::tensor::view::IsTensorLike;

        //      [[ a   b ]]       1    [[  d  -b ]]
        //  inv [[       ]] =  ------- [[        ]]
        //      [[ c   d ]]    ad - bc [[ -c   a ]]

        fn f<S: IsScalar, M22: IsMatrix<S, 2, 2>>(m: M22) -> M22 {
            let a = m.get((0, 0));
            let b = m.get((0, 1));

            let c = m.get((1, 0));
            let d = m.get((1, 1));

            let det = S::c(1.0) / (a.clone() * d.clone() - (b.clone() * c.clone()));

            M22::from_array2([
                [det.clone() * d, -det.clone() * b],
                [-det.clone() * c, det * a],
            ])
        }
        let a = M::<2, 2>::new(0.1, 0.2, 0.4, 0.7);

        let finite_diff = MatrixValuedMapFromMatrix::sym_diff_quotient(f, a, 1e-6);
        let auto_grad = MatrixValuedMapFromMatrix::fw_autodiff(f, a);

        approx::assert_abs_diff_eq!(
            finite_diff.view().elem_view(),
            auto_grad.view().elem_view(),
            epsilon = 2.0
        );
    }
}
