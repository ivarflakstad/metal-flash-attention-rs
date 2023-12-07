use crate::datatype::DataType;
use std::ops::{AddAssign, Mul};

// Naive matrix multiplication for testing
pub(crate) fn matrix_mul<T: DataType>(
    a: Vec<T>,
    b: Vec<T>,
    m: usize,
    n: usize,
    k: usize,
) -> Vec<T>
where
    T: AddAssign + Mul<Output = T> + Copy,
{
    let size = m * n;

    let mut c = Vec::with_capacity(size);

    for idx in 0..size {
        let i = idx / m;
        let j = idx % n;

        let mut sum = T::from_f64(0.0);
        for di in 0..k {
            sum += a[(i * k) + di] * b[(di * n) + j];
        }
        c.push(sum);
    }

    c
}

pub(crate) fn euclidean_distance<T>(a: Vec<T>, b: Vec<T>) -> f32
where
    T: Into<f32> + Clone + Copy,
{
    assert_eq!(a.len(), b.len(), "Lengths not equal");

    let mut sum = 0.0;

    for i in 0..a.len() {
        sum += (a[i].into() - b[i].into()).powi(2);
    }

    sum.sqrt()
}

pub(crate) fn vertices_approx_eq<T>(a: Vec<T>, b: Vec<T>, tolerance: f32)
where
    T: Into<f32> + Clone + Copy,
{
    assert_eq!(a.len(), b.len(), "Lengths not equal");

    let distance = euclidean_distance(a, b);
    assert!(
        distance < tolerance,
        "Distance not less than tolerance: {} < {} ",
        distance,
        tolerance
    );
}

#[cfg(test)]
mod tests {
    use crate::test_utils::matrix_mul;

    #[test]
    fn naive_matrix_mul_correctness() {
        let m = 3;
        let n = 3;
        let k = 2;
        let a = vec![1., 2., 6., 24., 120., 720.];
        let b = vec![1., 2., 6., 24., 120., 720.];
        let result = matrix_mul::<f32>(a, b, m, n, k);
        assert_eq!(
            result,
            &[49., 242., 1446., 582., 2892., 17316., 17400., 86640., 519120.]
        );
    }
}
