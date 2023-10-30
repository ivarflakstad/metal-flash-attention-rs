use std::cmp::max;
use std::hash::{Hash, Hasher};
use std::mem;

use metal::{
    ComputeCommandEncoderRef, ComputePipelineState, DeviceRef, Function, FunctionConstantValues,
    MTLDataType, MTLSize, NSUInteger,
};
use num_traits::Float;

use crate::{assert_eq_result, assert_result, Result};
use crate::datatype::TensorFloatingPoint;
use crate::tensor::Tensor;
use crate::utils;

const M_SIMD: u16 = 16;
const N_SIMD: u16 = 16;
const K_SIMD: u16 = 32;
const M_SPLITS: u16 = 2;
const N_SPLITS: u16 = 2;

const M_GROUP: u16 = M_SIMD * M_SPLITS;
const N_GROUP: u16 = N_SIMD * N_SPLITS;

const A_BLOCK_LENGTH: u16 = M_GROUP * K_SIMD;
const B_BLOCK_LENGTH: u16 = K_SIMD * N_GROUP;

pub fn encode_gemm<T: TensorFloatingPoint>(
    device: &DeviceRef,
    encoder: &ComputeCommandEncoderRef,
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &Tensor<T>,
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
        // let batch_dimensions_d = d_shape.split_last().map(|(_, rest)| rest.to_vec()).unwrap();
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

// struct GEMMTensors<T: TensorElement> {
//     a: Tensor<T>,
//     b: Tensor<T>,
//     c: Tensor<T>,
//     d: Option<Tensor<T>>,
// }
//
// impl<T: TensorElement> GEMMTensors<T> {
//     fn new(a: Tensor<T>, b: Tensor<T>, c: Tensor<T>, d: Option<Tensor<T>>) -> Self {
//         Self { a, b, c, d }
//     }
// }
/*
sgemm(device float *A [[buffer(0)]],
      device float *B [[buffer(1)]],
      device float *C [[buffer(2)]],
      device void *D [[buffer(3), function_constant(use_activation)]],

      threadgroup float *threadgroup_block [[threadgroup(0)]],
      constant ulong4 *matrix_offsets [[buffer(10), function_constant(batched)]],
      typename activation_functor<float>::function_table table [[buffer(11), function_constant(use_activation_function)]],
      constant uint *activation_function_offsets [[buffer(12), function_constant(batched_activation_function)]],

      uint3 gid [[threadgroup_position_in_grid]],
      ushort sidx [[simdgroup_index_in_threadgroup]],
      ushort lane_id [[thread_index_in_simdgroup]])
{
  _gemm_impl<float>(A, B, C, D, threadgroup_block, matrix_offsets, table, activation_function_offsets, gid, sidx, lane_id);
}
 */
pub fn encode<T: TensorFloatingPoint>(
    encoder: &ComputeCommandEncoderRef,
    a: &Tensor<T>,
    b: &Tensor<T>,
    c: &Tensor<T>,
    d: &Option<Tensor<T>>,
    pipeline: Pipeline,
) {
    encoder.set_compute_pipeline_state(&pipeline.pipelines[0]);
    encoder.set_threadgroup_memory_length(0, pipeline.thread_group_memory_lengths[0] as NSUInteger);

    encoder.set_buffers(
        0,
        &[Some(&a.buffer()), Some(&b.buffer()), Some(&c.buffer())],
        &[0; 3],
    );
    if let Some(d) = d {
        encoder.set_buffer(3, Some(d.buffer()), 0);
    }

    let mut grid_z = 1;
    if pipeline.flags & 0x1 > 0 {
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

    // let pipeline_state = &pipeline.pipelines[0];
    // let n = a.shape()[1] as NSUInteger;
    // let w = pipeline_state.thread_execution_width();
    // let h = pipeline_state.max_total_threads_per_threadgroup() / w;
    // let thread_groups_count = MTLSize::new(n, n, 1);
    // let threads_per_threadgroup = MTLSize::new(w, h, 1);

    let mut thread_groups_count = pipeline.grid_sizes[0];
    thread_groups_count.depth = grid_z;
    let threads_per_threadgroup = pipeline.group_sizes[0];

    println!("thread_groups_count: {:?}", thread_groups_count);
    println!("threads_per_threadgroup: {:?}", threads_per_threadgroup);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);

    println!("encoder: {:?}", encoder);
    println!("c: {:?}", c.contents());
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    pipelines: Vec<ComputePipelineState>,
    flags: u32,
    device_memory_lengths: Vec<u64>,
    thread_group_memory_lengths: Vec<u16>,
    grid_sizes: Vec<MTLSize>,
    group_sizes: Vec<MTLSize>,
}

impl Pipeline {
    fn new(
        device: &DeviceRef,
        functions: Vec<Function>,
        flags: u32,
        device_memory_lengths: Vec<u64>,
        thread_group_memory_lengths: Vec<u16>,
        grid_sizes: Vec<MTLSize>,
        group_sizes: Vec<MTLSize>,
    ) -> Pipeline {
        let mut pipelines = Vec::with_capacity(functions.len());
        for f in functions.iter() {
            let pipeline = device.new_compute_pipeline_state_with_function(f).unwrap();
            pipelines.push(pipeline);
        }
        Pipeline {
            pipelines,
            flags,
            device_memory_lengths,
            thread_group_memory_lengths,
            grid_sizes,
            group_sizes,
        }
    }
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

    let constant_values = function_constant_values(&p);
    println!("constant_values: {:?}", constant_values);
    let lib = utils::load_mfa_lib(device).unwrap();

    let function = lib.get_function(T::FN_NAME, Some(constant_values)).unwrap();
    println!("function: {:?}", function);

    let mut block_elements = A_BLOCK_LENGTH + B_BLOCK_LENGTH;
    if (p.m % 8 != 0) && (p.n % 8 != 0) {
        let c_block_length = M_GROUP * N_GROUP;
        block_elements = max(c_block_length, block_elements)
    }
    if p.fused_bias {
        if p.d_trans {
            block_elements = max(block_elements, M_GROUP)
        } else {
            block_elements = max(block_elements, N_GROUP)
        }
    }
    let block_bytes = block_elements * mem::size_of::<T>() as u16;

    let grid_size = MTLSize::new(
        utils::ceil_divide(p.n, N_GROUP)?,
        utils::ceil_divide(p.m, M_GROUP)?,
        1,
    );

    let group_size = MTLSize::new((32 * M_SPLITS * N_SPLITS) as NSUInteger, 1, 1);

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
    let pipeline = Pipeline::new(
        device,
        vec![function],
        flags,
        vec![0],
        vec![block_bytes],
        vec![grid_size],
        vec![group_size],
    );
    utils::cache_pipeline(p, pipeline.clone())?;
    Ok(pipeline)
}

pub fn function_constant_values(p: &GemmParameters) -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();

    // Dimensions of each matrix.
    constants.set_constant_value_at_index(utils::void_ptr(&p.m), MTLDataType::UInt, 0);
    constants.set_constant_value_at_index(utils::void_ptr(&p.n), MTLDataType::UInt, 1);
    constants.set_constant_value_at_index(utils::void_ptr(&p.k), MTLDataType::UInt, 2);

    // Whether each matrix is transposed.
    constants.set_constant_value_at_index(utils::void_ptr(&p.a_trans), MTLDataType::Bool, 10);
    constants.set_constant_value_at_index(utils::void_ptr(&p.b_trans), MTLDataType::Bool, 11);
    constants.set_constant_value_at_index(utils::void_ptr(&p.d_trans), MTLDataType::Bool, 13);

    // Alpha and beta constants from BLAS.
    constants.set_constant_value_at_index(utils::void_ptr(&p.alpha), MTLDataType::Float, 20);
    constants.set_constant_value_at_index(utils::void_ptr(&p.beta), MTLDataType::Float, 21);

    // Whether the operation is batched, fused, or fused with bias.
    constants.set_constant_value_at_index(utils::void_ptr(&p.batched), MTLDataType::Bool, 100);
    constants.set_constant_value_at_index(
        utils::void_ptr(&p.fused_activation),
        MTLDataType::Bool,
        101,
    );
    constants.set_constant_value_at_index(utils::void_ptr(&p.fused_bias), MTLDataType::Bool, 50001);

    // SIMD
    constants.set_constant_value_at_index(utils::void_ptr(&M_SIMD), MTLDataType::UShort, 200);
    constants.set_constant_value_at_index(utils::void_ptr(&N_SIMD), MTLDataType::UShort, 201);
    constants.set_constant_value_at_index(utils::void_ptr(&K_SIMD), MTLDataType::UShort, 202);

    // Splits
    constants.set_constant_value_at_index(utils::void_ptr(&M_SPLITS), MTLDataType::UShort, 210);
    constants.set_constant_value_at_index(utils::void_ptr(&N_SPLITS), MTLDataType::UShort, 211);

    // Metal API validation. Setting garbage values for unused constants. May not be necessary.
    // let garbage = false;
    // constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 102);
    // constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 103);
    // constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 113);
    // constants.set_constant_value_at_index(utils::void_ptr(&garbage), MTLDataType::Bool, 50000);

    constants
}
