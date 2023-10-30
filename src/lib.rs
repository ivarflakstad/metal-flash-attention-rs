pub mod datatype;
pub mod gemm;
pub mod shape;
pub mod tensor;
mod utils;

type Result<T> = std::result::Result<T, String>;
