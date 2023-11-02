use metal::{
    ComputeCommandEncoderRef, DeviceRef, FunctionConstantValues, MTLDataType, MTLResourceUsage,
    MTLSize, NSUInteger,
};
use num_traits::Float;
use std::cmp::max;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

use crate::datatype::TensorFloatingPoint;
use crate::pipeline::Pipeline;
use crate::tensor::Tensor;
use crate::utils;
use crate::{assert_eq_result, assert_result, Result};

pub fn encode_gemm<T: TensorFloatingPoint>(
    device: &DeviceRef,
    encoder: &ComputeCommandEncoderRef,
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &mut Tensor<T>,
    d: &Option<Tensor<T>>,
    transpose_a: bool,
    transpose_b: bool,
    transpose_d: bool,
    alpha: f32,
    beta: f32,
    fused_bias: bool,
) -> Result<()> {
    assert_eq_result!(alpha, 1.0, "Alpha must be 1.0");
    assert_eq_result!(beta, 0.0, "Beta must be 0.0");

    let a_shape = a.shape();
    let b_shape = b.shape();
    let c_shape = c.shape();
    assert_result!(
        a.shape().len() >= 2 && b.shape().len() >= 2 && c.shape().len() >= 2,
        "All shapes must have at least 2 dimensions"
    );

    let la = a_shape.len() - 1;
    let lb = b_shape.len() - 1;

    let (m, k) = if transpose_a {
        (a_shape[la], a_shape[la - 1])
    } else {
        (a_shape[la - 1], a_shape[la])
    };

    let (n, b_k) = if transpose_b {
        (b_shape[lb - 1], b_shape[lb])
    } else {
        (b_shape[lb], b_shape[lb - 1])
    };

    assert_eq_result!(
        k,
        b_k,
        "K must be equal to B_K, but got K={} and B_K={}",
        k,
        b_k
    );
    assert_result!(m > 0, "M must be greater than 0");
    assert_result!(n > 0, "N must be greater than 0");
    assert_result!(k > 0, "K must be greater than 0");
    assert_eq_result!(k, n, "K must be equal to N, but got K={} and N={}", k, n);
    assert_result!(
        m * k >= m * n,
        "A matrix must be larger or equal to result rows * interior columns"
    );
    assert_result!(
        k * n >= m * n,
        "B matrix must be larger or equal to result columns * interior columns"
    );

    let mut batched = false;
    if a_shape.len() > 2 || c_shape.len() > 2 {
        assert_result!(
            c_shape.len() > 2,
            "Misshapen matrices. If A #dims > 2 then C #dims must be so as well"
        );
        assert_result!(
            a_shape.len() > 2,
            "Misshapen matrices. If C #dims > 2 then A #dims must be so as well"
        );
        batched = true;
    }

    if !fused_bias {
        assert_result!(d.is_none(), "Bias tensor provided without fused_bias flag");
    } else {
        let d_t = d
            .as_ref()
            .ok_or("Fused bias provided without bias tensor")?;
        let d_shape = d_t.shape();
        assert_result!(!d_shape.is_empty(), "Bias tensor must not be empty");
        if transpose_d {
            assert_eq_result!(
                d_shape[d_shape.len() - 1],
                m,
                "Bias tensor must have M rows"
            );
        } else {
            assert_eq_result!(
                d_shape[d_shape.len() - 1],
                n,
                "Bias tensor must have N columns"
            );
        }
    }

    let parameters = GemmParameters {
        m: m as NSUInteger,
        n: n as NSUInteger,
        k: k as NSUInteger,
        a_trans: transpose_a,
        b_trans: transpose_b,
        d_trans: transpose_d,
        alpha,
        beta,
        batched,
        fused_activation: false,
        fused_bias,
        fn_name: T::FN_NAME,
    };

    let pipeline = create_pipeline::<T>(device, parameters).expect("Failed to create pipeline");
    encode(encoder, a, b, c, d, pipeline);

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GemmParameters {
    m: NSUInteger,
    n: NSUInteger,
    k: NSUInteger,
    a_trans: bool,
    b_trans: bool,
    d_trans: bool,
    alpha: f32,
    beta: f32,
    batched: bool,
    fused_activation: bool,
    fused_bias: bool,
    fn_name: &'static str,
}

impl Hash for GemmParameters {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.m.hash(state);
        self.n.hash(state);
        self.k.hash(state);
        self.a_trans.hash(state);
        self.b_trans.hash(state);
        Float::integer_decode(self.alpha).hash(state);
        Float::integer_decode(self.beta).hash(state);
        self.batched.hash(state);
        self.fused_activation.hash(state);
        self.fused_bias.hash(state);
        self.fn_name.hash(state);
    }
}

impl Eq for GemmParameters {}

pub fn encode<T: TensorFloatingPoint>(
    encoder: &ComputeCommandEncoderRef,
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &Tensor<T>,
    d: &Option<Tensor<T>>,
    pipeline: Pipeline,
) {
    encoder.set_compute_pipeline_state(&pipeline.pipeline(0));
    encoder
        .set_threadgroup_memory_length(0, pipeline.thread_group_memory_lengths()[0] as NSUInteger);

    encoder.set_buffers(
        0,
        &[Some(&a.buffer()), Some(&b.buffer()), Some(&c.buffer())],
        &[0; 3],
    );
    if let Some(d) = d {
        encoder.set_buffer(3, Some(d.buffer()), 0);
    }

    let mut grid_z = 1;
    if pipeline.flags() & 0x1 > 0 {
        panic!("Batched gemm not implemented yet");
        //   let batch_dimensions_a = tensors.a.shape.dropLast(2);
        //   let batch_dimensions_b = tensors.b.shape.dropLast(2);
        //   let batch_dimensions_c = tensors.c.shape.dropLast(2);
        //   assert!(batch_dimensions_a.iter().product() > 0);
        //   assert!(
        //     batch_dimensions_b.iter().product() == 1 ||
        //     batch_dimensions_b == batch_dimensions_a);
        //   assert!(batch_dimensions_a == batch_dimensions_c);
        //   grid_z = batch_dimensions_a.iter().product();
        //
        //   if let Some(batch_dimensions_d) = tensors.d { .shape.dropLast(1)
        //     assert!(
        //       batch_dimensions_d.reduce(1, *) == 1 ||
        //       batch_dimensions_d == batch_dimensions_a);
        //   }
        //
        //   // Mixed precision will cause undefined behavior.
        //   let element_size = mem::size_of::<T>();
        //   let byte_stride = |shape: Vec<u64>| -> u32 {
        //       let rank = shape.len();
        //       let mut output = element_size * shape[rank - 2] * shape[rank - 1];
        //       if shape.dropLast(2).product() == 1 {
        //           output = 0
        //       }
        //       output
        //   } as u32;
        //   let byte_stride_a = byte_stride(tensors.a.shape);
        //   let byte_stride_b = byte_stride(tensors.b.shape);
        //   let byte_stride_c = byte_stride(tensors.c.shape);
        //
        //   var byteStrideD = 0
        //   if let shapeD = tensors.d?.shape {
        //     let rank = shapeD.count
        //     byteStrideD = element_size * shapeD[rank - 1]
        //     if shapeD.dropLast(1).reduce(1, *) == 1 {
        //       byteStrideD = 0
        //     }
        //   }
        //   withUnsafeTemporaryAllocation(
        //     of: SIMD4<UInt64>.self, capacity: gridZ
        //   ) { buffer in
        //     for i in 0..<buffer.count {
        //       buffer[i] = SIMD4(
        //           UInt64(truncatingIfNeeded: i * byte_stride_a),
        //           UInt64(truncatingIfNeeded: i * byte_stride_b),
        //           UInt64(truncatingIfNeeded: i * byte_stride_c),
        //           UInt64(truncatingIfNeeded: i * byteStrideD))
        //     }
        //
        //     let bufferLength = buffer.count * MemoryLayout<SIMD3<UInt64>>.stride
        //     assert(MemoryLayout<SIMD3<UInt64>>.stride == 8 * 4)
        //     encoder.setBytes(buffer.baseAddress!, length: bufferLength, index: 10)
        //   }
    } else {
        assert_eq!(a.shape().len(), 2);
        assert_eq!(b.shape().len(), 2);
        assert_eq!(c.shape().len(), 2);
        if let Some(t) = d {
            assert_eq!(t.shape().len(), 1);
        }
        grid_z = 1;
    }

    let mut thread_groups_count = pipeline.grid_sizes()[0];
    thread_groups_count.depth = grid_z;
    let threads_per_threadgroup = pipeline.group_sizes()[0];

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
}

fn create_pipeline<T: TensorFloatingPoint>(
    device: &DeviceRef,
    p: GemmParameters,
) -> Result<Pipeline> {
    if let Some(pipeline) = utils::get_cached_pipeline(p) {
        return Ok(pipeline);
    }

    assert_eq_result!(p.alpha, 1.0, "Alpha must be 1.0");
    assert_eq_result!(p.beta, 0.0, "Beta must be 0.0");
    assert_result!(!p.fused_activation, "Fused activation must be false");

    let lib = utils::load_mfa_lib(device)?;

    let config = gemm_config(&p);
    let m_group = config.m_group;
    let n_group = config.n_group;
    let k_simd = config.k_simd.value;
    let m_splits = config.m_splits.value;
    let n_splits = config.n_splits.value;

    let a_block_bytes = m_group * k_simd * T::SIZE;
    let b_block_bytes = k_simd * n_group * T::SIZE;
    let c_block_bytes = m_group * n_group * T::SIZE;
    let mut thread_group_memory_length = a_block_bytes + b_block_bytes;

    if p.m % 8 > 0 && p.n % 8 > 0 {
        thread_group_memory_length = max(thread_group_memory_length, c_block_bytes);
    }
    if p.fused_bias {
        let d_block_bytes = if p.d_trans {
            m_group * T::SIZE
        } else {
            n_group * T::SIZE
        };
        thread_group_memory_length = max(thread_group_memory_length, d_block_bytes);
    }

    let grid_size = MTLSize::new(
        utils::ceil_divide(p.n, n_group)?,
        utils::ceil_divide(p.m, m_group)?,
        1,
    );

    let group_size = MTLSize::new((32 * m_splits * n_splits) as NSUInteger, 1, 1);

    let mut flags = 0;
    if p.batched {
        flags |= 0x1;
    }
    if p.fused_activation {
        flags |= 0x2;
    }
    if p.fused_bias {
        flags |= 0x4;
    }

    let constant_values = config.create_function_constant_values();
    let function = lib.get_function(T::FN_NAME, Some(constant_values)).unwrap();

    let pipeline = Pipeline::new(
        device,
        vec![function],
        flags,
        vec![0],
        vec![thread_group_memory_length],
        vec![grid_size],
        vec![group_size],
    );
    utils::cache_pipeline(p, pipeline.clone())?;
    Ok(pipeline)
}

trait ConstantValueType: Debug {
    const MTL_DATA_TYPE: MTLDataType;
}
impl ConstantValueType for NSUInteger {
    const MTL_DATA_TYPE: MTLDataType = MTLDataType::UInt;
}
impl ConstantValueType for bool {
    const MTL_DATA_TYPE: MTLDataType = MTLDataType::Bool;
}
impl ConstantValueType for f32 {
    const MTL_DATA_TYPE: MTLDataType = MTLDataType::Float;
}
impl ConstantValueType for u16 {
    const MTL_DATA_TYPE: MTLDataType = MTLDataType::UShort;
}

#[derive(Debug)]
struct ConstantValue<T: ConstantValueType> {
    value: T,
    index: NSUInteger,
}

impl<T: ConstantValueType> ConstantValue<T> {
    fn new(value: T, index: NSUInteger) -> Self {
        Self { value, index }
    }
}

trait CreateFunctionConstantValues {
    fn add_constant_value<T: ConstantValueType>(
        constant_values: &mut FunctionConstantValues,
        t: &ConstantValue<T>,
    ) {
        constant_values.set_constant_value_at_index(
            utils::void_ptr(&t.value),
            T::MTL_DATA_TYPE,
            t.index,
        );
    }
    fn create_function_constant_values(&self) -> FunctionConstantValues;
}

#[derive(Debug)]
pub struct GemmConfig {
    m: ConstantValue<NSUInteger>,
    n: ConstantValue<NSUInteger>,
    k: ConstantValue<NSUInteger>,
    a_trans: ConstantValue<bool>,
    b_trans: ConstantValue<bool>,
    d_trans: ConstantValue<bool>,
    alpha: ConstantValue<f32>,
    beta: ConstantValue<f32>,
    batched: ConstantValue<bool>,
    fused_activation: ConstantValue<bool>,
    fused_bias: ConstantValue<bool>,
    m_simd: ConstantValue<u16>,
    n_simd: ConstantValue<u16>,
    k_simd: ConstantValue<u16>,
    m_splits: ConstantValue<u16>,
    n_splits: ConstantValue<u16>,
    m_group: u16,
    n_group: u16,
}

impl CreateFunctionConstantValues for GemmConfig {
    fn create_function_constant_values(&self) -> FunctionConstantValues {
        let mut constants = FunctionConstantValues::new();

        // Dimensions of each matrix.
        Self::add_constant_value(&mut constants, &self.m);
        Self::add_constant_value(&mut constants, &self.n);
        Self::add_constant_value(&mut constants, &self.k);

        // Whether each matrix is transposed.
        Self::add_constant_value(&mut constants, &self.a_trans);
        Self::add_constant_value(&mut constants, &self.b_trans);
        Self::add_constant_value(&mut constants, &self.d_trans);

        // Alpha and beta constants from BLAS.
        Self::add_constant_value(&mut constants, &self.alpha);
        Self::add_constant_value(&mut constants, &self.beta);

        // Whether the operation is batched, fused, or fused with bias.
        Self::add_constant_value(&mut constants, &self.batched);
        Self::add_constant_value(&mut constants, &self.fused_activation);
        Self::add_constant_value(&mut constants, &self.fused_bias);

        // Metal API validation. Setting garbage values for unused constants. May not be necessary.
        let garbage = false;
        constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 102);
        constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 103);
        constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 113);
        constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 50000);

        // SIMD
        Self::add_constant_value(&mut constants, &self.m_simd);
        Self::add_constant_value(&mut constants, &self.n_simd);
        Self::add_constant_value(&mut constants, &self.k_simd);

        // Splits
        Self::add_constant_value(&mut constants, &self.m_splits);
        Self::add_constant_value(&mut constants, &self.n_splits);

        constants
    }
}

pub fn gemm_config(p: &GemmParameters) -> GemmConfig {
    let mut c_elements = p.m * p.n;
    if p.batched {
        c_elements *= 2;
    }

    let is_half = p.fn_name == "hgemm";
    let is_float = p.fn_name == "sgemm";

    let mut m_group = 32;
    let mut n_group = 32;
    let mut k_simd = 32;
    if c_elements > 10 ^ 6 {
        m_group = 48;
        n_group = 48;
    }
    // If K_simd is perfectly equal to matrix K, the compiler can elide a large
    // amount of logic in the kernel.
    if p.k >= 33 && p.k <= 40 {
        k_simd = 40;
    } else if is_half && p.k >= 73 && p.k >= 80 {
        k_simd = 80;
    } else if c_elements > 10 ^ 6 {
        if p.k <= 16 {
            k_simd = 16;
        } else if p.k <= 24 {
            k_simd = 24;
        } else if p.k <= 32 {
            k_simd = 32;
        } else if p.k <= 48 {
            k_simd = 24;
        } else if p.k <= 64 {
            k_simd = 32;
        } else if is_float {
            k_simd = 24;
        }
    }

    let m_splits = 2;
    let n_splits = 2;
    let m_simd = m_group / m_splits;
    let n_simd = n_group / n_splits;

    GemmConfig {
        m: ConstantValue::new(p.m, 0),
        n: ConstantValue::new(p.n, 1),
        k: ConstantValue::new(p.k, 2),
        a_trans: ConstantValue::new(p.a_trans, 10),
        b_trans: ConstantValue::new(p.b_trans, 11),
        d_trans: ConstantValue::new(p.d_trans, 13),
        alpha: ConstantValue::new(p.alpha, 20),
        beta: ConstantValue::new(p.beta, 21),
        batched: ConstantValue::new(p.batched, 100),
        fused_activation: ConstantValue::new(p.fused_activation, 101),
        fused_bias: ConstantValue::new(p.fused_bias, 50001),
        m_simd: ConstantValue::new(m_simd, 200),
        n_simd: ConstantValue::new(n_simd, 201),
        k_simd: ConstantValue::new(k_simd, 202),
        m_splits: ConstantValue::new(m_splits, 210),
        n_splits: ConstantValue::new(n_splits, 211),
        m_group,
        n_group,
    }
}

#[cfg(test)]
mod tests {
    use crate::datatype::{Float, TensorFloatingPoint};
    use crate::gemm::encode_gemm;
    use crate::tensor::Tensor;
    use crate::test_utils::{matrix_mul, vertices_approx_eq};
    use metal::Device;

    #[test]
    fn correctness() {
        const M: usize = 100;
        const N: usize = 100;
        const K: usize = 100;

        type T = Float;

        const ITERATIONS: usize = 20;

        let device = Device::system_default().expect("No device found");
        let command_queue = device.new_command_queue();

        let avg_magnitude = 0.5 * K as f32;
        let mag_multiplier = 2. / 10f32.powi(1 + (T::SIZE as i32 / 2));
        let avg_deviation = (K as f32).sqrt();
        let tolerance = (avg_magnitude * mag_multiplier).max(avg_magnitude * 3e-7);

        for i in 0..ITERATIONS {
            let a = Tensor::<T>::random(&device, vec![M, K], 0.0..1.0);
            let b = Tensor::<T>::random(&device, vec![K, N], 0.0..1.0);
            let mut c = Tensor::<T>::new(&device, vec![K, N]);

            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            encode_gemm(
                &device, &encoder, &a, &b, &mut c, &None, false, false, false, 1.0, 0.0, false,
            )
            .expect("Encoding failed");
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            let expected = matrix_mul::<T>(a.contents(), b.contents(), M, K, N);
            vertices_approx_eq(c.contents(), expected, tolerance);
        }
    }
}
