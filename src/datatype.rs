use std::fmt::Debug;
use std::hash::Hash;

use half::{bf16, f16};

pub trait DataType: Clone + Copy + PartialEq + Eq + Debug + Hash {
    type Type: Default + Clone + Copy + PartialEq + Debug + Sized;
    const TYPE_ID: u32;

    const FROM_F64_FN: fn(f64) -> Self::Type;
    const TO_F64_FN: fn(Self::Type) -> f64;

    #[inline]
    fn from_f64(v: f64) -> Self::Type {
        Self::FROM_F64_FN(v)
    }

    #[inline]
    fn to_f64(v: Self::Type) -> f64 {
        Self::TO_F64_FN(v)
    }
}

macro_rules! datatype {
    ($dt:ident, $dt_ty:ty, $type_id:expr, $from_f64:expr, $to_f64:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $dt;

        impl DataType for $dt {
            type Type = $dt_ty;
            const TYPE_ID: u32 = $type_id;
            const FROM_F64_FN: fn(f64) -> Self::Type = $from_f64;
            const TO_F64_FN: fn(Self::Type) -> f64 = $to_f64;
        }
    };
}

datatype!(Float, f32, 3, |v: f64| v as f32, |v: f32| v as f64);

#[cfg(target_arch = "aarch64")]
datatype!(Half, f16, 16, |v: f64| f16::from_f64(v), |v: f16| v
    .to_f64());

datatype!(Bool, bool, 53, |v: f64| v != 0.0, |v: bool| if v {
    1.0
} else {
    0.0
});
datatype!(UInt8, u8, 49, |v: f64| v as u8, |v: u8| v as f64);
datatype!(UInt16, u16, 41, |v: f64| v as u16, |v: u16| v as f64);
datatype!(UInt32, u32, 33, |v: f64| v as u32, |v: u32| v as f64);
datatype!(BFloat, bf16, 121, |v: f64| bf16::from_f64(v), |v: bf16| v
    .to_f64());

pub trait TensorElement: DataType {}

impl TensorElement for UInt8 {}

impl TensorElement for Float {}

#[cfg(target_arch = "aarch64")]
impl TensorElement for Half {}

pub trait TensorFloatingPoint: TensorElement {
    const FN_NAME: &'static str;
    const SIZE: u16;
}

impl TensorFloatingPoint for Float {
    const FN_NAME: &'static str = "sgemm";
    const SIZE: u16 = 4;
}

#[cfg(target_arch = "aarch64")]
impl TensorFloatingPoint for Half {
    const FN_NAME: &'static str = "hgemm";
    const SIZE: u16 = 2;
}
