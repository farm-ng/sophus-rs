use std::marker::PhantomData;

use sophus_calculus::types::scalar::IsScalar;
use sophus_calculus::types::vector::IsVector;
use sophus_calculus::types::MatF64;
use sophus_calculus::types::VecF64;

use super::affine::AffineDistortionImpl;
use super::generic_camera::Camera;
use super::traits::IsProjection;

/// Orthographic projection implementation
#[derive(Debug, Clone)]
pub struct ProjectionOrtho<S: IsScalar<1>> {
    phantom: PhantomData<S>,
}

impl<S: IsScalar<1>> IsProjection<S> for ProjectionOrtho<S> {
    fn proj(point_in_camera: &S::Vector<3>) -> S::Vector<2> {
        point_in_camera.get_fixed_rows::<2>(0)
    }

    fn unproj(point_in_camera: &S::Vector<2>, extension: S) -> S::Vector<3> {
        S::Vector::<3>::from_array([point_in_camera.get(0), point_in_camera.get(1), extension])
    }

    fn dx_proj_x(_point_in_camera: &VecF64<3>) -> MatF64<2, 3> {
        unimplemented!("dx_proj_x not implemented for ProjectionOrtho")
    }
}

/// Orthographic camera
pub type OrthoCamera<S> = Camera<S, 0, 4, AffineDistortionImpl<S>, ProjectionOrtho<S>>;
