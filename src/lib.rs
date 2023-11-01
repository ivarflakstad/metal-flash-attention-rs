pub mod attention;
pub mod datatype;
pub mod gemm;
pub mod pipeline;
pub mod shape;
pub mod tensor;
mod utils;

#[cfg(test)]
mod test_utils;

type Result<T> = std::result::Result<T, String>;
