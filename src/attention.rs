use std::arch::aarch64::vrsqrtes_f32;
use std::cmp::max;
use std::hash::Hash;

use metal::{
    ComputeCommandEncoderRef, DeviceRef, FunctionConstantValues, MTLDataType, MTLSize, NSUInteger,
};

use crate::datatype::{DataType, Half, TensorFloatingPoint};
use crate::pipeline::Pipeline;
use crate::tensor::Tensor;
use crate::utils::{ceil_divide, void_ptr};
use crate::Result;
use crate::{assert_eq_result, assert_result, utils};

pub fn encode_attention<T: TensorFloatingPoint>(
    device: &DeviceRef,
    encoder: &ComputeCommandEncoderRef,
    o: &Tensor<T>,
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
    mask: Option<&Tensor<T>>,
    transpose_q: bool,
    transpose_k: bool,
    transpose_v: bool,
    transpose_o: bool,
    block_sparse: bool,
) -> Result<()> {
    let q_shape = q.shape();
    let k_shape = k.shape();
    let v_shape = v.shape();
    let o_shape = o.shape();
    assert!(q.count() >= 3);
    assert_eq!(q.count(), k.count());
    assert_eq!(q.count(), v.count());
    assert_eq!(q.count(), o.count());

    let batch_dim_index = q_shape.len() - 4;
    if q_shape.len() > 3 {
        assert_eq!(q_shape[..batch_dim_index], k_shape[..batch_dim_index]);
        assert_eq!(q_shape[..batch_dim_index], v_shape[..batch_dim_index]);
        assert_eq!(q_shape[..batch_dim_index], o_shape[..batch_dim_index]);
    }

    let mut q_r = 0;
    let mut o_r = 0;
    let mut k_c = 0;
    let mut v_c = 0;

    let mut q_h = 0;
    let mut k_h = 0;
    let mut v_h = 0;
    let mut o_h = 0;

    let mut q_d = 0;
    let mut k_d = 0;
    let mut v_d = 0;
    let mut o_d = 0;

    let leading_dim_index = q_shape.len() - 1;
    if transpose_q {
        q_h = q_shape[leading_dim_index - 2];
        q_d = q_shape[leading_dim_index - 1];
        q_r = q_shape[leading_dim_index];
    } else {
        q_r = q_shape[leading_dim_index - 2];
        q_h = q_shape[leading_dim_index - 1];
        q_d = q_shape[leading_dim_index];
    }
    if transpose_k {
        k_c = k_shape[leading_dim_index - 2];
        k_h = k_shape[leading_dim_index - 1];
        k_d = k_shape[leading_dim_index];
    } else {
        k_h = k_shape[leading_dim_index - 2];
        k_d = k_shape[leading_dim_index - 1];
        k_c = k_shape[leading_dim_index];
    }
    if transpose_v {
        v_h = v_shape[leading_dim_index - 2];
        v_d = v_shape[leading_dim_index - 1];
        v_c = v_shape[leading_dim_index];
    } else {
        v_c = v_shape[leading_dim_index - 2];
        v_h = v_shape[leading_dim_index - 1];
        v_d = v_shape[leading_dim_index];
    }
    if transpose_o {
        o_h = o_shape[leading_dim_index - 2];
        o_d = o_shape[leading_dim_index - 1];
        o_r = o_shape[leading_dim_index];
    } else {
        o_r = o_shape[leading_dim_index - 2];
        o_h = o_shape[leading_dim_index - 1];
        o_d = o_shape[leading_dim_index];
    }

    assert_eq_result!(q_r, o_r, "R does not match");

    assert_eq_result!(k_c, v_c, "C does not match");

    assert_eq_result!(q_h, k_h, "H does not match");
    assert_eq_result!(q_h, v_h, "H does not match");
    assert_eq_result!(q_h, o_h, "H does not match");

    assert_eq_result!(q_d, k_d, "D does not match");
    assert_eq_result!(q_d, v_d, "D does not match");
    assert_eq_result!(q_d, o_d, "D does not match");

    let batched = q_shape.len() > 3;

    let parameters = AttentionParameters {
        data_type: T::TYPE_ID,
        r: q_r,
        c: k_c,
        h: q_h,
        d: q_d,
        q_trans: transpose_q,
        k_trans: transpose_k,
        v_trans: transpose_v,
        o_trans: transpose_o,
        batched,
        masked: mask.is_some(),
        block_sparse,
    };

    let pipeline = create_pipeline(device, parameters)?;
    encode(encoder, &pipeline, q, k, v, o, mask)?;

    Ok(())
}

pub fn encode<T: TensorFloatingPoint>(
    encoder: &ComputeCommandEncoderRef,
    pipeline: &Pipeline,
    q: &Tensor<T>,
    k: &Tensor<T>,
    v: &Tensor<T>,
    o: &Tensor<T>,
    mask: Option<&Tensor<T>>,
) -> Result<()> {
    encoder.set_buffer(0, Some(q.buffer()), 0);
    encoder.set_buffer(1, Some(k.buffer()), 0);
    encoder.set_buffer(2, Some(v.buffer()), 0);
    encoder.set_buffer(3, Some(o.buffer()), 0);

    let mut grid_z: NSUInteger = 0;
    let mut scratch_buffer_size = 0;
    let mut partials_buffer_size = 0;
    let mut locks_buffer_size = 0;

    if pipeline.flags() & 0x4 > 0 {
        let grid_size = pipeline.grid_sizes()[1];
        scratch_buffer_size = grid_size.height * grid_size.width
    }
    if pipeline.flags() & 0x8 > 0 {
        let grid_size = pipeline.grid_sizes()[0];
        partials_buffer_size = pipeline.device_memory_lengths()[0];
        locks_buffer_size = grid_size.height * grid_size.width;
    }
    if pipeline.flags() & 0x1 > 0 {
        grid_z = q.count() as NSUInteger;
        partials_buffer_size *= grid_z;
        locks_buffer_size *= grid_z;

        // TODO: batch + mask
    } else {
        assert_eq_result!(q.count(), 3, "q must have 3 dimensions");
        assert_eq_result!(k.count(), 3, "k must have 3 dimensions");
        assert_eq_result!(v.count(), 3, "v must have 3 dimensions");
        assert_eq_result!(o.count(), 3, "o must have 3 dimensions");
        if let Some(mask_tensor) = mask {
            assert_eq_result!(mask_tensor.count(), 3, "mask tensor must have 3 dimensions");
        }
        grid_z = 1;
    }

    if scratch_buffer_size > 0 {
        // TODO: Get scratch buffer from cache if it exists and is large enough, otherwise
        //  create a new one.
    }
    if locks_buffer_size > 0 {
        // TODO: Get locks buffer from cache if it exists and is large enough, otherwise
        //  create a new one.
    }
    if partials_buffer_size > 0 {
        // TODO: Get partials buffer from cache if it exists and is large enough, otherwise
        //  create a new one.
    }

    if pipeline.flags() & 0x2 > 0 {
        let mask_tensor =
            mask.ok_or("Mask tensor must be provided if masked attention is enabled")?;
        let mask_shape = mask_tensor.shape();
        assert_eq_result!(
            mask_shape[mask_tensor.shape().len() - 3],
            1,
            "Mask must have shape (..., 1, ...)"
        );
        encoder.set_buffer(12, Some(mask_tensor.buffer()), 0);
    }
    if pipeline.flags() & 0x4 > 0 {
        scratch_buffer_size *= grid_z;
        assert_result!(
            pipeline.flags() & 0x2 > 0,
            "Block sparse attention requires masked attention"
        );
        assert_result!(
            scratch_buffer_size > 0,
            "Scratch buffer size must be greater than 0"
        );

        encoder.set_compute_pipeline_state(pipeline.pipeline(1));
        encoder.set_threadgroup_memory_length(
            0,
            pipeline.thread_group_memory_lengths()[1] as NSUInteger,
        );

        let mut grid_size = pipeline.grid_sizes()[1];
        grid_size.depth = grid_z as NSUInteger;
        encoder.dispatch_thread_groups(grid_size, pipeline.group_sizes()[1]);
    }

    encoder.set_compute_pipeline_state(pipeline.pipeline(0));
    encoder
        .set_threadgroup_memory_length(0, pipeline.thread_group_memory_lengths()[0] as NSUInteger);

    let mut grid_size = pipeline.grid_sizes()[0];
    grid_size.depth = grid_z as NSUInteger;
    encoder.dispatch_thread_groups(grid_size, pipeline.group_sizes()[0]);

    Ok(())
}

enum AttentionMask {
    UpperTriangular,
    BlockSparse(usize, f32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionParameters {
    data_type: u32,
    r: usize,
    c: usize,
    h: usize,
    d: usize,
    q_trans: bool,
    k_trans: bool,
    v_trans: bool,
    o_trans: bool,
    batched: bool,
    masked: bool,
    block_sparse: bool,
}

fn create_pipeline(device: &DeviceRef, p: AttentionParameters) -> Result<Pipeline> {
    if let Some(pipeline) = utils::get_cached_pipeline(p) {
        return Ok(pipeline);
    }

    let lib = utils::load_mfa_lib(device)?;

    let constants = FunctionConstantValues::new();

    // Dimensions
    constants.set_constant_value_at_index(void_ptr(&p.r), MTLDataType::UInt, 0);
    constants.set_constant_value_at_index(void_ptr(&p.c), MTLDataType::UInt, 1);
    constants.set_constant_value_at_index(void_ptr(&p.h), MTLDataType::UInt, 2);
    constants.set_constant_value_at_index(void_ptr(&p.d), MTLDataType::UInt, 2);

    // Transpositions
    constants.set_constant_value_at_index(void_ptr(&p.q_trans), MTLDataType::Bool, 10);
    constants.set_constant_value_at_index(void_ptr(&p.k_trans), MTLDataType::Bool, 11);
    constants.set_constant_value_at_index(void_ptr(&p.v_trans), MTLDataType::Bool, 12);
    constants.set_constant_value_at_index(void_ptr(&p.o_trans), MTLDataType::Bool, 13);

    // TODO: Generic rsqrt
    let alpha = unsafe { vrsqrtes_f32(p.d as f32) };
    constants.set_constant_value_at_index(void_ptr(&alpha), MTLDataType::Float, 20);

    constants.set_constant_value_at_index(void_ptr(&p.data_type), MTLDataType::UInt, 30);

    constants.set_constant_value_at_index(void_ptr(&p.batched), MTLDataType::Bool, 100);
    constants.set_constant_value_at_index(void_ptr(&p.masked), MTLDataType::Bool, 50000);
    constants.set_constant_value_at_index(void_ptr(&p.block_sparse), MTLDataType::Bool, 102);

    let triangular = false;
    constants.set_constant_value_at_index(void_ptr(&triangular), MTLDataType::Bool, 103);

    let forward = true;
    let backward = false;
    let generate_block_mask = false;
    let grouped_query = false;
    constants.set_constant_value_at_index(void_ptr(&forward), MTLDataType::Bool, 110);
    constants.set_constant_value_at_index(void_ptr(&backward), MTLDataType::Bool, 111);
    constants.set_constant_value_at_index(void_ptr(&generate_block_mask), MTLDataType::Bool, 112);
    constants.set_constant_value_at_index(void_ptr(&grouped_query), MTLDataType::Bool, 113);

    let mut r_simd: u16 = 8;
    let mut c_simd: u16 = 32;
    let mut r_splits: u16 = 4;
    let mut fuse_async_loads = false;

    if p.data_type == Half::TYPE_ID {
        let d = p.d;
        if p.masked {
            if d <= 16 {
                r_simd = 16;
                c_simd = 64;
                r_splits = 4;
            } else if d <= 24 {
                r_simd = 8;
                c_simd = 64;
                r_splits = 8;
            } else if d <= 80 {
                r_simd = 8;
                c_simd = 64;
                r_splits = 4;
            } else {
                r_simd = 8;
                c_simd = 32;
                r_splits = 4;
            }
        } else {
            r_simd = 8;
            r_splits = 8;

            if d <= 8 {
                r_simd = 16;
                c_simd = 64;
            } else if d <= 16 {
                c_simd = 72;
                fuse_async_loads = true;
            } else if d <= 24 {
                c_simd = 56;
                fuse_async_loads = true;
            } else if d <= 56 {
                c_simd = 64;
            } else if d <= 64 {
                c_simd = 40;
                fuse_async_loads = true;
            } else if d <= 96 {
                c_simd = 64;
            } else if d <= 304 {
                c_simd = 32;
                r_splits = 4;
            } else {
                c_simd = 40;
                r_splits = 8;
            }
        }
    }

    constants.set_constant_value_at_index(void_ptr(&r_simd), MTLDataType::UShort, 200);
    constants.set_constant_value_at_index(void_ptr(&c_simd), MTLDataType::UShort, 201);
    constants.set_constant_value_at_index(void_ptr(&r_splits), MTLDataType::UShort, 210);
    if fuse_async_loads {
        constants.set_constant_value_at_index(void_ptr(&fuse_async_loads), MTLDataType::Bool, 213);
    }

    let mut functions = vec![];
    let attention = lib.get_function("attention", Some(constants.clone()))?;
    functions.push(attention);

    if p.block_sparse {
        let generate_block_mask = true;
        constants.set_constant_value_at_index(
            void_ptr(&generate_block_mask),
            MTLDataType::Bool,
            112,
        );
        let block_masked_attention = lib.get_function("attention", Some(constants.clone()))?;
        functions.push(block_masked_attention.clone());
    }

    let d_simd = (p.d + 7) as u16 / 64;
    let r_group = r_simd * r_splits;

    let mut r_block_dim = r_group;
    let mut c_block_dim = c_simd;
    let mut d_block_dim = d_simd;

    let set_bank_offset = |dim: &mut u16, index: NSUInteger| -> Result<()> {
        assert_eq_result!(*dim % 8, 0, "Original dimension must be divisible by 8.");
        let dim_bytes = *dim * p.data_type as u16;

        // How the heuristic works:
        //
        // FP16:
        // Pad 8 -> 8       (16B -> 16B)
        // Pad 16 -> 24     (32B -> 48B)
        // Pad 24 -> 24     (48B -> 48B)
        // Pad 32 -> 36, 40 (64B -> 72B, 80B)
        // Pad 40 -> 40     (80B -> 80B)
        // Pad 48 -> 52, 56 (96B -> 104B, 112B)
        // Pad 56 -> 56     (112B -> 112B)
        // Pad 64 -> 72     (128B -> 144B)
        // Pad 80 -> 88     (160B -> 176B)
        // Pad 96 -> 104    (192B -> 208B)

        let dim_bytes_mod = dim_bytes % 64;
        if dim_bytes_mod == 16 || dim_bytes_mod == 48 {
            return Ok(());
        } else if dim_bytes_mod == 0 || dim_bytes_mod == 32 {
            let bank_offset_bytes: u16 = 16;
            let bank_offset = bank_offset_bytes / p.data_type as u16;
            *dim += bank_offset;
            constants.set_constant_value_at_index(
                void_ptr(&bank_offset),
                MTLDataType::UShort,
                index,
            );
        }

        Ok(())
    };

    set_bank_offset(&mut r_block_dim, 220)?;
    set_bank_offset(&mut c_block_dim, 221)?;
    set_bank_offset(&mut d_block_dim, 222)?;

    let mut k_block_length = 0u16;
    let mut v_block_length = 0u16;
    let mut q_block_length = 0u16;
    let mut o_block_length = 0u16;

    if p.q_trans {
        q_block_length = d_simd * r_block_dim;
    } else {
        q_block_length = r_group * d_block_dim
    }
    if p.k_trans {
        k_block_length = c_simd * d_block_dim;
    } else {
        k_block_length = d_simd * c_block_dim;
    }
    if p.v_trans {
        v_block_length = d_simd * c_block_dim;
    } else {
        v_block_length = c_simd * d_block_dim;
    }
    if p.o_trans {
        o_block_length = d_simd * r_block_dim;
    } else {
        o_block_length = r_group * d_block_dim;
    }

    let mut device_elements = 0;

    if triangular {
        let floats_per_cache_line = 128 / 4;
        let mut num_lm_elements = 2 * r_group;
        num_lm_elements += floats_per_cache_line - 1;
        num_lm_elements /= floats_per_cache_line;
        num_lm_elements *= floats_per_cache_line;

        let mut num_0_elements = r_group * d_simd;
        num_0_elements += floats_per_cache_line - 1;
        num_0_elements /= floats_per_cache_line;
        num_0_elements *= floats_per_cache_line;

        let head_0_blocks = ceil_divide(p.r as NSUInteger, r_group)?;
        device_elements = (num_lm_elements + num_0_elements) as NSUInteger;
        device_elements *= p.h as NSUInteger * head_0_blocks;
    } else {
        device_elements = 0;
    }
    let mut device_bytes = vec![device_elements * 4];

    let mut block_elements = 0u16;
    if fuse_async_loads {
        block_elements = k_block_length + v_block_length;
    } else {
        block_elements = max(k_block_length, v_block_length);
    }
    block_elements = max(block_elements, q_block_length);
    block_elements = max(block_elements, o_block_length);
    let mut block_bytes = vec![block_elements * p.data_type as u16];

    let mut grid_x = ceil_divide(p.r as NSUInteger, r_group)?;
    if triangular {
        let complete_blocks = p.r as NSUInteger / r_group as NSUInteger;
        let upper_blocks = complete_blocks / 2;
        let lower_blocks = complete_blocks - upper_blocks;
        let edge_blocks = grid_x - complete_blocks as NSUInteger;
        assert_result!(
            lower_blocks >= upper_blocks,
            "Lower blocks must be greater than or equal to upper blocks"
        );

        grid_x = ((lower_blocks + edge_blocks) * 2) as NSUInteger;
    }

    let mut grid_sizes = vec![MTLSize::new(grid_x, p.h as NSUInteger, 1)];
    let mut group_sizes = vec![MTLSize::new(32 * r_group as NSUInteger, 1, 1)];

    if p.block_sparse {
        block_bytes.push(4 * r_splits);
        device_bytes.push(0);
        grid_sizes.push(MTLSize::new(
            ceil_divide(p.r as NSUInteger, r_group)?,
            ceil_divide(p.c as NSUInteger, c_simd)?,
            1,
        ));
        group_sizes.push(MTLSize::new(32 * r_splits as NSUInteger, 1, 1));

        // TODO: Scratch buffer
    }

    let mut flags = 0u32;
    if p.batched {
        flags |= 0x1;
    }
    if p.masked {
        flags |= 0x2;
    }
    if p.block_sparse {
        flags |= 0x4;
    }
    if triangular {
        flags |= 0x8;
    }

    let pipeline = Pipeline::new(
        device,
        functions,
        flags,
        device_bytes,
        block_bytes,
        grid_sizes,
        group_sizes,
    )?;
    utils::cache_pipeline(p, pipeline.clone())?;
    Ok(pipeline)
}
