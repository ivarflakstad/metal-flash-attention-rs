use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::{mem, slice};

use metal::{Buffer, BufferRef, DeviceRef, MTLResourceOptions};
use rand::Rng;

use crate::datatype::TensorElement;

// TODO: Attention
// enum AttentionMask {
//   UpperTriangular,
//   BlockSparse(u32, f32)
// }
// impl<T: TensorFloatingPoint> TensorBuffer<T> {
//      fn from_rand(device: &DeviceRef, shape: Vec<u64>, mask: AttentionMask) -> Self {
// }
#[derive(Debug, Clone)]
pub struct Tensor<T: TensorElement> {
    buffer: Buffer,
    shape: Vec<usize>,
    count: usize,
    allocated_size: usize,
    _marker: PhantomData<T>,
}

impl<T: TensorElement> Tensor<T> {
    pub fn new(device: &DeviceRef, shape: Vec<usize>) -> Self {
        let count: usize = shape.iter().product();
        let allocated_size = count * mem::size_of::<T::Type>();
        let buffer =
            device.new_buffer(allocated_size as u64, MTLResourceOptions::StorageModeShared);
        Tensor {
            buffer,
            shape,
            count,
            allocated_size,
            _marker: PhantomData,
        }
    }

    pub fn zeros(device: &DeviceRef, shape: Vec<usize>) -> Self {
        let count = shape.iter().product();
        let allocated_size = count * mem::size_of::<T::Type>();
        let buffer = device.new_buffer_with_data(
            vec![T::from_f64(0.0); count].as_ptr().cast(),
            allocated_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Tensor {
            buffer,
            shape,
            count,
            allocated_size,
            _marker: PhantomData,
        }
    }

    pub fn random(device: &DeviceRef, shape: Vec<usize>, range: Range<f64>) -> Self {
        let count = shape.iter().product();
        let allocated_size = count * mem::size_of::<T::Type>();

        let mut rng = rand::thread_rng();
        let entries: Vec<T::Type> = (0..count)
            .map(|_| T::from_f64(rng.gen_range(range.clone())))
            .collect();

        let buffer = device.new_buffer_with_data(
            entries.as_ptr().cast(),
            allocated_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Tensor {
            buffer,
            shape,
            count,
            allocated_size,
            _marker: PhantomData,
        }
    }

    pub fn linear(device: &DeviceRef, shape: Vec<usize>, range: Range<f64>) -> Self {
        let count = shape.iter().product();
        let allocated_size = count * mem::size_of::<T::Type>();

        let entries: Vec<T::Type> = (0..count)
            .map(|i| T::from_f64(i as f64 / count as f64 * (range.end - range.start)))
            .collect();

        let buffer = device.new_buffer_with_data(
            entries.as_ptr().cast(),
            allocated_size as u64,
            MTLResourceOptions::StorageModeShared,
        );
        Tensor {
            buffer,
            shape,
            count,
            allocated_size,
            _marker: PhantomData,
        }
    }

    pub fn copy(tensor: &Tensor<T>) -> Self {
        tensor.clone()
    }

    pub fn buffer(&self) -> &BufferRef {
        &self.buffer
    }

    pub fn shape(&self) -> &Vec<usize> {
        &self.shape
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn allocated_size(&self) -> usize {
        self.allocated_size
    }

    #[inline]
    pub fn contents(&self) -> Vec<T::Type> {
        let contents =
            unsafe { slice::from_raw_parts(self.buffer.contents() as *const T::Type, self.count) };
        contents.to_vec()
    }
}
