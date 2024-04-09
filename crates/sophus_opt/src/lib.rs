#![feature(portable_simd)]
#![deny(missing_docs)]

//! # Non-linear least squares optimization module

/// Block vector and matrix operations
pub mod block;
/// Evaluated costs
pub mod cost;
/// Cost function arguments
pub mod cost_args;
/// Cost functions
pub mod cost_fn;
/// Example problems
pub mod example_problems;
/// LDLt Cholesky factorization
pub mod ldlt;
/// Entry point for the non-linear least squares optimization
pub mod nlls;
/// Robust kernel functions
pub mod robust_kernel;
/// Linear solvers
pub mod solvers;
/// Evaluated cost terms
pub mod term;
/// Decision variables
pub mod variables;
