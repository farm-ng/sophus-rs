use crate::traits::IsCameraDistortionImpl;

use sophus_calculus::types::matrix::IsMatrix;
use sophus_calculus::types::params::ParamsImpl;
use sophus_calculus::types::scalar::IsScalar;
use sophus_calculus::types::vector::IsVector;
use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;

use std::marker::PhantomData;

/// Affine "distortion" implementation
///
/// This is not a distortion in the traditional sense, but rather a simple affine transformation.
#[derive(Debug, Clone, Copy)]
pub struct AffineDistortionImpl<S: IsScalar<1>> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<1>> ParamsImpl<S, 4, 1> for AffineDistortionImpl<S> {
    fn are_params_valid(params: &S::Vector<4>) -> bool {
        params.real()[0] != 0.0 && params.real()[1] != 0.0
    }

    fn params_examples() -> Vec<S::Vector<4>> {
        vec![S::Vector::<4>::from_c_array([1.0, 1.0, 0.0, 0.0])]
    }

    fn invalid_params_examples() -> Vec<S::Vector<4>> {
        vec![
            S::Vector::<4>::from_c_array([0.0, 1.0, 0.0, 0.0]),
            S::Vector::<4>::from_c_array([1.0, 0.0, 0.0, 0.0]),
        ]
    }
}

impl<S: IsScalar<1>> IsCameraDistortionImpl<S, 0, 4> for AffineDistortionImpl<S> {
    fn distort(
        params: &S::Vector<4>,
        proj_point_in_camera_z1_plane: &S::Vector<2>,
    ) -> S::Vector<2> {
        S::Vector::<2>::from_array([
            proj_point_in_camera_z1_plane.get(0) * params.get(0) + params.get(2),
            proj_point_in_camera_z1_plane.get(1) * params.get(1) + params.get(3),
        ])
    }

    fn undistort(params: &S::Vector<4>, distorted_point: &S::Vector<2>) -> S::Vector<2> {
        S::Vector::<2>::from_array([
            (distorted_point.get(0) - params.get(2)) / params.get(0),
            (distorted_point.get(1) - params.get(3)) / params.get(1),
        ])
    }

    fn dx_distort_x(
        params: &VecF64<4>,
        _proj_point_in_camera_z1_plane: &VecF64<2>,
    ) -> MatF64<2, 2> {
        MatF64::<2, 2>::from_array2([[params[0], 0.0], [0.0, params[1]]])
    }
}
