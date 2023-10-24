use metal::{
    CommandQueueRef, ComputePipelineStateRef, Device, DeviceRef, FunctionConstantValues,
    MTLDataType, MTLResourceOptions, MTLSize, NSUInteger,
};
use std::ffi::c_void;
const LIB_DATA: &[u8] = include_bytes!("libMetalFlashAttention.metallib");

const M: usize = 1024;
const N: usize = 1024;
const K: usize = 1024;

const A_TRANS: bool = false;
const B_TRANS: bool = false;
const D_TRANS: bool = false;

const ALPHA: f32 = 1.0;
const BETA: f32 = 0.0;

const BATCHED: bool = false;
const FUSED_ACTIVATION: bool = false;
const FUSED_BIAS: bool = false;

const M_SIMD: u16 = 16;
const N_SIMD: u16 = 16;
const K_SIMD: u16 = 32;
const M_SPLITS: u16 = 2;
const N_SPLITS: u16 = 2;

const M_GROUP: u16 = M_SIMD * M_SPLITS;
const N_GROUP: u16 = N_SIMD * N_SPLITS;

const fn ceil_divide(target: u64, granularity: u16) -> u64 {
    (target + granularity as u64 - 1) / granularity as u64
}

const GRID_SIZE_WIDTH: u64 = ceil_divide(N as u64, N_GROUP);
const GRID_SIZE_HEIGHT: u64 = ceil_divide(M as u64, M_GROUP);
const GRID_SIZE_DEPTH: u64 = 1;
const GROUP_SIZE_WIDTH: u64 = (32 * M_SPLITS * N_SPLITS) as u64;
const GROUP_SIZE_HEIGHT: u64 = 1;
const GROUP_SIZE_DEPTH: u64 = 1;
fn main() {
    let device = Device::system_default().expect("No device found");
    let command_queue = device.new_command_queue();
    let lib = device.new_library_with_data(LIB_DATA).unwrap();
    println!("functions: {:?}", lib.function_names());

    let vals = function_constant_values();
    let function = lib.get_function("sgemm", Some(vals)).unwrap();
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .unwrap();

    let left = [1.0f32; N];
    let right = [2.0f32; N];
    let result = gemm(&device, &command_queue, &pipeline, &left, &right) as *const [f32; N];
    println!("{:?}", unsafe { *result });
}

fn gemm(
    device: &DeviceRef,
    queue: &CommandQueueRef,
    pipeline: &ComputePipelineStateRef,
    left: &[f32],
    right: &[f32],
) -> *mut c_void {
    assert_eq!(left.len(), right.len());
    let size = left.len() * std::mem::size_of::<f32>();
    println!("sizeof: {}", std::mem::size_of::<f64>());
    println!("size: {}", size);

    let buffer_a = device.new_buffer_with_data(
        void_ptr(&left),
        size as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );

    let buffer_b = device.new_buffer_with_data(
        void_ptr(&right),
        size as NSUInteger,
        MTLResourceOptions::StorageModeShared,
    );

    let buffer_result = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

    let command_buffer = queue.new_command_buffer();
    let compute_encoder = command_buffer.new_compute_command_encoder();
    compute_encoder.set_compute_pipeline_state(&pipeline);
    compute_encoder.set_buffers(
        0,
        &[Some(&buffer_a), Some(&buffer_b), Some(&buffer_result)],
        &[0; 3],
    );

    // let n = 9u64;
    // let w = pipeline.thread_execution_width();
    // let h = pipeline.max_total_threads_per_threadgroup() / w;
    let grid_size = metal::MTLSize::new(GRID_SIZE_WIDTH, GRID_SIZE_HEIGHT, GRID_SIZE_DEPTH);
    let group_size = metal::MTLSize::new(GROUP_SIZE_WIDTH, GROUP_SIZE_HEIGHT, GROUP_SIZE_DEPTH);
    println!("grid_size: {:?}", grid_size);
    println!("group_size: {:?}", group_size);
    compute_encoder.dispatch_threads(grid_size, group_size);

    // end encoding and execute commands
    compute_encoder.end_encoding();
    command_buffer.commit();

    command_buffer.wait_until_completed();

    buffer_result.contents()
}

struct AsyncPipeline {
    pipeline: ComputePipelineStateRef,
    queue: CommandQueueRef,
    device: DeviceRef,
    flags: u32,
    device_memory_lengths: Vec<u64>,
    thread_group_memory_lengths: Vec<u64>,
    grid_sizes: Vec<MTLSize>,
    group_sizes: Vec<MTLSize>,
}

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

fn function_constant_values() -> FunctionConstantValues {
    let constants = FunctionConstantValues::new();
    constants.set_constant_value_at_index(void_ptr(&M), MTLDataType::UInt, 0);
    constants.set_constant_value_at_index(void_ptr(&N), MTLDataType::UInt, 1);
    constants.set_constant_value_at_index(void_ptr(&K), MTLDataType::UInt, 2);
    constants.set_constant_value_at_index(void_ptr(&A_TRANS), MTLDataType::Bool, 10);
    constants.set_constant_value_at_index(void_ptr(&B_TRANS), MTLDataType::Bool, 11);
    constants.set_constant_value_at_index(void_ptr(&D_TRANS), MTLDataType::Bool, 13);
    constants.set_constant_value_at_index(void_ptr(&ALPHA), MTLDataType::Float, 20);
    constants.set_constant_value_at_index(void_ptr(&BETA), MTLDataType::Float, 21);
    constants.set_constant_value_at_index(void_ptr(&BATCHED), MTLDataType::Bool, 100);
    constants.set_constant_value_at_index(void_ptr(&FUSED_ACTIVATION), MTLDataType::Bool, 101);
    constants.set_constant_value_at_index(void_ptr(&FUSED_BIAS), MTLDataType::Bool, 50001);

    constants.set_constant_value_at_index(void_ptr(&M_SIMD), MTLDataType::UShort, 200);
    constants.set_constant_value_at_index(void_ptr(&N_SIMD), MTLDataType::UShort, 201);
    constants.set_constant_value_at_index(void_ptr(&K_SIMD), MTLDataType::UShort, 202);
    constants.set_constant_value_at_index(void_ptr(&M_SPLITS), MTLDataType::UShort, 210);
    constants.set_constant_value_at_index(void_ptr(&N_SPLITS), MTLDataType::UShort, 211);

    constants
}

pub fn void_ptr<T>(v: &T) -> *const core::ffi::c_void {
    v as *const T as *const core::ffi::c_void
}
pub fn deref_void_ptr<T>(ptr: *const core::ffi::c_void) -> *const T {
    ptr as *const T
}
