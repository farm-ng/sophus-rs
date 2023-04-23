use nalgebra::{SMatrix, SVector};
use notan::{log::{debug, warn, trace}};

use crate::calculus;

use super::{affine::AffineDistortionImpl, traits::CameraDistortionImpl};
type V<const N: usize> = SVector<f64, N>;
type M<const N: usize, const O: usize> = SMatrix<f64, N, O>;

#[derive(Debug, Clone)]
pub struct KannalaBrandtDistortionImpl;

impl calculus::traits::ParamsImpl<8> for KannalaBrandtDistortionImpl {
    fn are_params_valid(params: &V<8>) -> bool {
        params[0] != 0.0 && params[1] != 0.0
    }

    fn params_examples() -> Vec<V<8>> {
        vec![V::<8>::from_vec(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    }

    fn invalid_params_examples() -> Vec<V<8>> {
        vec![
            V::<8>::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            V::<8>::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
    }
}

impl CameraDistortionImpl<4, 8> for KannalaBrandtDistortionImpl {
    fn distort(params: &V<8>, proj_point_in_camera_z1_plane: &V<2>) -> V<2> {
        let k0 = params[4];
        let k1 = params[5];
        let k2 = params[6];
        let k3 = params[7];

        let radius_sq = proj_point_in_camera_z1_plane[0] * proj_point_in_camera_z1_plane[0]
            + proj_point_in_camera_z1_plane[1] * proj_point_in_camera_z1_plane[1];

        if radius_sq > 1e-8 {
            let radius = radius_sq.sqrt();
            let radius_inverse = 1.0 / radius;
            let theta = radius.atan2(1.0);
            let theta2 = theta * theta;
            let theta4 = theta2 * theta2;
            let theta6 = theta2 * theta4;
            let theta8 = theta4 * theta4;

            let r_distorted = theta * (1.0 + k0 * theta2 + k1 * theta4 + k2 * theta6 + k3 * theta8);
            let scaling = r_distorted * radius_inverse;
            return V::<2>::new(
                scaling * proj_point_in_camera_z1_plane[0] * params[0] + params[2],
                scaling * proj_point_in_camera_z1_plane[1] * params[1] + params[3],
            );
        }
        let pinhole_params = params.fixed_rows::<4>(0).into_owned();
        AffineDistortionImpl::distort(&pinhole_params, proj_point_in_camera_z1_plane)
    }

    fn undistort(params: &V<8>, distorted_point: &V<2>) -> V<2> {
        let fu = params[0];
        let fv = params[1];
        let u0 = params[2];
        let v0 = params[3];

        let k0 = params[4];
        let k1 = params[5];
        let k2 = params[6];
        let k3 = params[7];

        let un = (distorted_point[0] - u0) / fu;
        let vn = (distorted_point[1] - v0) / fv;
        let rth2 = un * un + vn * vn;

        if rth2 < 1e-8 {
            return V::<2>::new(un, vn);
        }

        let rth = rth2.sqrt();

        let mut th = rth.sqrt();

        let mut iters = 0;
        loop {
            let th2 = th * th;
            let th4 = th2 * th2;
            let th6 = th2 * th4;
            let th8 = th4 * th4;

            let thd = th * (1.0 + k0 * th2 + k1 * th4 + k2 * th6 + k3 * th8);
            let d_thd_wtr_th =
                1.0 + 3.0 * k0 * th2 + 5.0 * k1 * th4 + 7.0 * k2 * th6 + 9.0 * k3 * th8;

            let step = (thd - rth) / d_thd_wtr_th;
            th -= step;

            if step.abs() < 1e-8 {
                break;
            }

            iters += 1;

            const MAX_ITERS: usize = 20;
            const HIGH_ITERS: usize = MAX_ITERS / 2;

            if iters == HIGH_ITERS {
                debug!(
                    "undistort: did not converge in {} iterations, step: {}",
                    iters, step
                );
            }

            if iters > HIGH_ITERS {
                trace!(
                    "undistort: did not converge in {} iterations, step: {}",
                    iters, step
                );
            }

            if iters >= 20 {
                warn!("undistort: max iters ({}) reached, step: {}", iters, step);
                break;
            }
        }

        let radius_undistorted = th.tan();

        if radius_undistorted < 0.0 {
            V::<2>::new(
                -radius_undistorted * un / rth,
                -radius_undistorted * vn / rth,
            )
        } else {
            V::<2>::new(radius_undistorted * un / rth, radius_undistorted * vn / rth)
        }
    }

    fn dx_distort_x(_params: &V<8>, proj_point_in_camera_z1_plane: &V<2>) -> M<2, 2> {
        unimplemented!("dx_distort_x")
    }
}