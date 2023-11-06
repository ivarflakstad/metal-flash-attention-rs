use std::hash::Hash;

use metal::{ComputePipelineState, DeviceRef, Function, MTLSize, NSUInteger};

use crate::attention::AttentionParameters;
use crate::gemm::GemmParameters;
use crate::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Parameters {
    Gemm(GemmParameters),
    Attention(AttentionParameters),
}

impl From<GemmParameters> for Parameters {
    fn from(p: GemmParameters) -> Self {
        Parameters::Gemm(p)
    }
}

impl From<AttentionParameters> for Parameters {
    fn from(p: AttentionParameters) -> Self {
        Parameters::Attention(p)
    }
}

#[derive(Debug, Clone)]
pub struct Pipeline {
    pipelines: Vec<ComputePipelineState>,
    flags: u32,
    device_memory_lengths: Vec<NSUInteger>,
    thread_group_memory_lengths: Vec<u16>,
    grid_sizes: Vec<MTLSize>,
    group_sizes: Vec<MTLSize>,
}

impl Pipeline {
    pub(crate) fn new(
        device: &DeviceRef,
        functions: Vec<Function>,
        flags: u32,
        device_memory_lengths: Vec<u64>,
        thread_group_memory_lengths: Vec<u16>,
        grid_sizes: Vec<MTLSize>,
        group_sizes: Vec<MTLSize>,
    ) -> Result<Pipeline> {
        let mut pipelines = Vec::with_capacity(functions.len());
        for f in functions.iter() {
            let pipeline = device.new_compute_pipeline_state_with_function(f)?;
            pipelines.push(pipeline);
        }
        Ok(Pipeline {
            pipelines,
            flags,
            device_memory_lengths,
            thread_group_memory_lengths,
            grid_sizes,
            group_sizes,
        })
    }

    pub fn pipeline(&self, index: usize) -> &ComputePipelineState {
        &self.pipelines[index]
    }

    pub fn flags(&self) -> u32 {
        self.flags
    }

    pub fn device_memory_lengths(&self) -> &[NSUInteger] {
        &self.device_memory_lengths
    }

    pub fn thread_group_memory_lengths(&self) -> &[u16] {
        &self.thread_group_memory_lengths
    }

    pub fn grid_sizes(&self) -> &[MTLSize] {
        &self.grid_sizes
    }

    pub fn group_sizes(&self) -> &[MTLSize] {
        &self.group_sizes
    }
}
