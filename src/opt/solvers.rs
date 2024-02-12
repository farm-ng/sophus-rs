use std::ops::AddAssign;
use std::ops::Deref;

use sprs::DenseVector;

use crate::opt::cost_args::CompareIdx;
use crate::opt::nlls::c_from_var_kind;
use crate::opt::nlls::VarKind;

use super::nlls::EvaluatedCost;
use super::nlls::IsVarTuple;
use super::nlls::VarPool;

pub struct SparseNormalEquation {
    sparse_hessian: sprs::CsMat<f64>,
    neg_gradient: nalgebra::DVector<f64>,
}

impl SparseNormalEquation {
    fn from_families_and_cost<const NUM: usize, const NUM_ARGS: usize>(
        variables: &VarPool,
        costs: Vec<EvaluatedCost<NUM, NUM_ARGS>>,
        nu: f64,
    ) -> SparseNormalEquation {
        assert!(variables.num_of_kind(VarKind::Marginalized) == 0);
        assert!(variables.num_of_kind(VarKind::Free) >= 1);

        // Note let's first focus on these special cases, before attempting a
        // general version covering all cases holistically. Also, it might not be trivial
        // to implement VarKind::Marginalized > 1.
        //  -  Example, the the arrow-head sparsity uses a recursive application of the Schur-Complement.
        let num_var_params = variables.num_free_params();
        println!("num_var_params: {:?}", num_var_params);
        let mut hessian_triplet = sprs::TriMat::new((num_var_params, num_var_params));
        let mut neg_grad = nalgebra::DVector::<f64>::zeros(num_var_params);

        let mut start_indices_per_arg = Vec::new();

        assert_eq!(costs.len(), 1);
        for evaluated_cost in costs.iter() {
            let num_args = evaluated_cost.family_names.len();

            let mut dof_per_arg = Vec::new();
            for name in evaluated_cost.family_names.iter() {
                let family = variables.families.get(name).unwrap();
                start_indices_per_arg.push(family.get_start_indices());
                dof_per_arg.push(family.free_or_marg_dof());
            }

            for evaluated_term in evaluated_cost.terms.iter() {
                assert_eq!(evaluated_term.idx.len(), 1);
                let idx = evaluated_term.idx[0].clone();
                assert_eq!(idx.len(), num_args);

                println!("{:?}", idx);

                for arg_id_alpha in 0..num_args {
                    println!("arg_id {}", arg_id_alpha);

                    let dof_alpha = dof_per_arg[arg_id_alpha];
                    if dof_alpha == 0 {
                        continue;
                    }

                    let var_idx_alpha = idx[arg_id_alpha];
                    let start_idx_alpha = start_indices_per_arg[arg_id_alpha][var_idx_alpha];
                    println!("start_idx_alpha: {}", start_idx_alpha);

                    if start_idx_alpha == -1 {
                        continue;
                    }

                    let grad_block = evaluated_term.gradient.block(arg_id_alpha);
                    let start_idx_alpha = start_idx_alpha as usize;
                    assert_eq!(dof_alpha, grad_block.nrows());

                    neg_grad
                        .rows_mut(start_idx_alpha, dof_alpha)
                        .add_assign(-grad_block);

                    let hessian_block = evaluated_term.hessian.block(arg_id_alpha, arg_id_alpha);
                    assert_eq!(dof_alpha, hessian_block.nrows());
                    assert_eq!(dof_alpha, hessian_block.ncols());

                    // block diagonal
                    for r in 0..dof_alpha {
                        for c in 0..dof_alpha {
                            let mut d = 0.0;
                            if r == c {
                                d = nu;
                            }
                            hessian_triplet.add_triplet(
                                start_idx_alpha + r,
                                start_idx_alpha + c,
                                hessian_block[(r, c)] + d,
                            );
                        }
                    }

                    // off diagonal hessian
                    for arg_id_beta in 0..num_args {
                        // skip diagonal blocks
                        if arg_id_alpha == arg_id_beta {
                            continue;
                        }
                        let dof_beta = dof_per_arg[arg_id_beta];
                        if dof_beta == 0 {
                            continue;
                        }

                        let var_idx_beta = idx[arg_id_beta];
                        let start_idx_beta = start_indices_per_arg[arg_id_beta][var_idx_beta];
                        if start_idx_beta == -1 {
                            continue;
                        }
                        let start_idx_beta = start_idx_beta as usize;

                        let hessian_block_alpha_beta =
                            evaluated_term.hessian.block(arg_id_alpha, arg_id_beta);
                        let hessian_block_beta_alpha =
                            evaluated_term.hessian.block(arg_id_beta, arg_id_alpha);

                        assert_eq!(dof_alpha, hessian_block_alpha_beta.nrows());
                        assert_eq!(dof_beta, hessian_block_alpha_beta.ncols());
                        assert_eq!(dof_beta, hessian_block_beta_alpha.nrows());
                        assert_eq!(dof_alpha, hessian_block_beta_alpha.ncols());

                        // alpha-beta off-diagonal
                        for r in 0..dof_alpha {
                            for c in 0..dof_beta {
                                hessian_triplet.add_triplet(
                                    start_idx_alpha + r,
                                    start_idx_beta + c,
                                    hessian_block_alpha_beta[(r, c)],
                                );
                            }
                        }
                    }
                }
            }
        }

        Self {
            sparse_hessian: hessian_triplet.to_csr(),
            neg_gradient: neg_grad,
        }
    }

    pub fn is_symmetric<N, I, Iptr, IpStorage, IStorage, DStorage>(
        mat: &sprs::CsMatBase<N, I, IpStorage, IStorage, DStorage, Iptr>,
    ) -> bool
    where
        N: PartialEq + std::fmt::Display,
        I: sprs::SpIndex,
        Iptr: sprs::SpIndex,
        IpStorage: Deref<Target = [Iptr]>,
        IStorage: Deref<Target = [I]>,
        DStorage: Deref<Target = [N]>,
    {
        if mat.rows() != mat.cols() {
            return false;
        }
        for (outer_ind, vec) in mat.outer_iterator().enumerate() {
            for (inner_ind, value) in vec.iter() {
                match mat.get_outer_inner(inner_ind, outer_ind) {
                    None => return false,
                    Some(transposed_val) => {
                        if transposed_val != value {
                            println!("{} != {}", transposed_val, value);
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    fn solve(&mut self) -> Vec<f64> {
        Self::is_symmetric(&self.sparse_hessian.view());

        let ldl = sprs_ldl::Ldl::new().check_symmetry(sprs::SymmetryCheck::DontCheckSymmetry);
        let ldl_num = ldl.numeric(self.sparse_hessian.view()).unwrap();

        ldl_num.solve(
            self.neg_gradient
                .iter()
                .map(|x: &f64| *x)
                .collect::<Vec<f64>>(),
        )
    }
}

pub fn solve<const NUM: usize, const NUM_ARGS: usize, VarTuple: IsVarTuple<NUM_ARGS> + 'static>(
    variables: &VarPool,
    costs: Vec<EvaluatedCost<NUM, NUM_ARGS>>,
    nu: f64,
) -> VarPool {
    assert!(variables.num_of_kind(VarKind::Marginalized) <= 1);
    assert!(variables.num_of_kind(VarKind::Free) >= 1);

    if variables.num_of_kind(VarKind::Marginalized) == 0 {
        let mut sne = SparseNormalEquation::from_families_and_cost(variables, costs, nu);
        sne.solve();
        let delta = sne.solve();
        variables.update(delta.into())
    } else {
        todo!()
    }
}
