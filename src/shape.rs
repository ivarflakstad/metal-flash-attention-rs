#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

impl Shape {
    pub fn new(shape: Vec<usize>) -> Self {
        Self(shape)
    }

    pub fn rank(&self) -> usize {
        self.0.len()
    }

    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }

    pub fn stride_contiguous(&self) -> Vec<usize> {
        let mut stride = vec![0; self.0.len()];
        let mut acc = 1;
        for (i, dim) in self.0.iter().enumerate().rev() {
            stride[i] = acc;
            acc *= dim;
        }
        stride
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(v: &[usize; N]) -> Self {
        Self(v.to_vec())
    }
}
impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Self {
        Self(v.to_vec())
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Self(v.to_vec())
    }
}
