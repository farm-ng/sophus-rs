use crate::calculus::types::vector::IsVector;
use crate::calculus::types::MatF64;
use crate::calculus::types::VecF64;
use crate::lie::rotation2::Isometry2;
use crate::lie::rotation3::Isometry3;
use crate::opt::cost_fn::CostFn;
use crate::opt::cost_fn::CostSignature;
use crate::opt::example_problems::cost_fn::isometry2_prior::Isometry2PriorCostFn;
use crate::opt::example_problems::cost_fn::isometry2_prior::Isometry2PriorTermSignature;
use crate::opt::example_problems::cost_fn::isometry3_prior::Isometry3PriorCostFn;
use crate::opt::example_problems::cost_fn::isometry3_prior::Isometry3PriorTermSignature;
use crate::opt::nlls::optimize;
use crate::opt::nlls::OptParams;
use crate::opt::variables::VarFamily;
use crate::opt::variables::VarKind;
use crate::opt::variables::VarPoolBuilder;

/// Simple 2D isometry prior problem
pub struct SimpleIso2PriorProblem {
    /// True world from robot isometry
    pub true_world_from_robot: Isometry2<f64>,
    /// Estimated world from robot isometry
    pub est_world_from_robot: Isometry2<f64>,
}

impl Default for SimpleIso2PriorProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleIso2PriorProblem {
    fn new() -> Self {
        let p = VecF64::<3>::from_c_array([0.2, 0.0, 1.0]);
        let true_world_from_robot = Isometry2::<f64>::exp(&p);
        Self {
            true_world_from_robot,
            est_world_from_robot: Isometry2::<f64>::identity(),
        }
    }

    /// Test the simple 2D isometry prior problem
    pub fn test(&self) {
        let cost_signature = vec![Isometry2PriorTermSignature {
            isometry_prior_mean: self.true_world_from_robot,
            entity_indices: [0],
        }];

        let obs_pose_a_from_pose_b_poses =
            CostSignature::<1, Isometry2<f64>, Isometry2PriorTermSignature> {
                family_names: ["poses".into()],
                terms: cost_signature,
            };

        let family: VarFamily<Isometry2<f64>> =
            VarFamily::new(VarKind::Free, vec![self.est_world_from_robot]);

        let families = VarPoolBuilder::new().add_family("poses", family).build();

        approx::assert_abs_diff_ne!(
            self.true_world_from_robot.compact(),
            self.est_world_from_robot.compact(),
            epsilon = 1e-6
        );

        let up_families = optimize(
            families,
            vec![CostFn::new(
                obs_pose_a_from_pose_b_poses.clone(),
                Isometry2PriorCostFn {},
            )],
            OptParams {
                num_iter: 1,         // should converge in single iteration
                initial_lm_nu: 1e-6, // if lm prior param is tiny
            },
        );
        let refined_world_from_robot = up_families.get_members::<Isometry2<f64>>("poses".into());

        approx::assert_abs_diff_eq!(
            self.true_world_from_robot.compact(),
            refined_world_from_robot[0].compact(),
            epsilon = 1e-6
        );
    }
}

/// Simple 3D isometry prior problem
pub struct SimpleIso3PriorProblem {
    /// True world from robot isometry
    pub true_world_from_robot: Isometry3<f64>,
    /// Estimated world from robot isometry
    pub est_world_from_robot: Isometry3<f64>,
}

impl Default for SimpleIso3PriorProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl SimpleIso3PriorProblem {
    fn new() -> Self {
        let p = VecF64::<6>::from_c_array([0.2, 0.0, 1.0, 0.2, 0.0, 1.0]);
        let true_world_from_robot = Isometry3::<f64>::exp(&p);
        Self {
            true_world_from_robot,
            est_world_from_robot: Isometry3::<f64>::identity(),
        }
    }

    /// Test the simple 3D isometry prior problem
    pub fn test(&self) {
        let cost_signature = vec![Isometry3PriorTermSignature {
            isometry_prior: (self.true_world_from_robot, MatF64::<6, 6>::identity()),
            entity_indices: [0],
        }];

        let obs_pose_a_from_pose_b_poses =
            CostSignature::<1, (Isometry3<f64>, MatF64<6, 6>), Isometry3PriorTermSignature> {
                family_names: ["poses".into()],
                terms: cost_signature,
            };

        let family: VarFamily<Isometry3<f64>> =
            VarFamily::new(VarKind::Free, vec![self.est_world_from_robot]);

        let families = VarPoolBuilder::new().add_family("poses", family).build();

        approx::assert_abs_diff_ne!(
            self.true_world_from_robot.compact(),
            self.est_world_from_robot.compact(),
            epsilon = 1e-6
        );

        let up_families = optimize(
            families,
            vec![CostFn::new(
                obs_pose_a_from_pose_b_poses.clone(),
                Isometry3PriorCostFn {},
            )],
            OptParams {
                num_iter: 1,         // should converge in single iteration
                initial_lm_nu: 1e-6, // if lm prior param is tiny
            },
        );
        let refined_world_from_robot = up_families.get_members::<Isometry3<f64>>("poses".into());

        approx::assert_abs_diff_eq!(
            self.true_world_from_robot.compact(),
            refined_world_from_robot[0].compact(),
            epsilon = 1e-6
        );
    }
}

mod tests {

    #[test]
    fn simple_prior_opt_tests() {
        use super::SimpleIso2PriorProblem;
        use super::SimpleIso3PriorProblem;

        SimpleIso2PriorProblem::new().test();
        SimpleIso3PriorProblem::new().test();
    }
}
