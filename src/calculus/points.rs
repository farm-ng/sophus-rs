use super::types::scalar::IsScalar;
use super::types::vector::IsVector;
use super::types::vector::IsVectorLike;
use super::types::V;

pub fn example_points<S: IsScalar, const POINT: usize>(
) -> Vec<S::Vector<POINT>> {
    let points4 = vec![
        V::<4>::from_array([0.1, 0.0, 0.0, 0.0]),
        V::<4>::from_array([1.0, 0.0, 1.0, 0.5]),
        V::<4>::from_array([0.7, 5.0, 0.1, (-5.0)]),
        V::<4>::from_array([2.0, (-3.0), 1.0, 0.5]),
    ];

    let mut out: Vec<S::Vector<POINT>> = vec![];
    for p4 in points4 {
        let mut v = S::Vector::<POINT>::zero();
        for i in 0..POINT.min(4) {
            let val = p4[i];
            v.set_c(i, val);
        }
        out.push(v)
    }
    out
}
