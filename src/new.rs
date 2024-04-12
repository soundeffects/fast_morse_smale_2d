// ░█▀█░█▀▀░█░█░░░░█▀▄░█▀▀
// ░█░█░█▀▀░█▄█░░░░█▀▄░▀▀█
// ░▀░▀░▀▀▀░▀░▀░▀░░▀░▀░▀▀▀
//
// This file defines the `new` method for the MorseSmaleSolver class. This method creates a new instance
// of a solver, and it creates the GPU context necessary to send tasks to the GPU when other methods
// require such.

use crate::MorseSmaleSolver;
use std::{borrow::Cow, mem::size_of};
use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferAddress, BufferBindingType,
    BufferSize, ComputePipelineDescriptor, DeviceDescriptor, Features, Instance, Limits,
    PipelineLayoutDescriptor, RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
    ShaderStages,
};

impl MorseSmaleSolver {
    ///
    /// MorseSmaleSolver::new
    ///
    /// Creates a new solver instance, which creates and stores its own GPU context. This context is used
    /// to send tasks to the GPU on demand.
    ///
    pub async fn new() -> Result<Self, String> {
        let adapter = Instance::default()
            .request_adapter(&RequestAdapterOptions::default())
            .await;

        if adapter.is_none() {
            return Err("Failed to instantiate adapter. Cannot access the GPU.".to_string());
        }

        let device_result = adapter
            .unwrap()
            .request_device(
                &DeviceDescriptor {
                    label: None,
                    required_features: Features::empty(),
                    required_limits: Limits::downlevel_defaults(),
                },
                None,
            )
            .await;

        if device_result.is_err() {
            return Err("Failed to instantiate device. Cannot access the GPU.".to_string());
        }

        let (device, queue) = device_result.unwrap();

        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Morse-Smale Complex Compute: Shader Module"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        // This can be though of as the function signature for our CPU-GPU function.
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Morse-Smale Complex Compute: Bind Group Layout"),
            entries: &[
                // Invocation parameters binding
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: BufferSize::new(size_of::<u32>() as BufferAddress),
                    },
                    count: None,
                },
                // Morse function buffer binding
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Gradient field buffer binding
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::COMPUTE,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Morse-Smale Complex Compute: Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Morse-Smale Complex Compute: Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: "compute_gradient",
        });

        Ok(Self {
            device,
            queue,
            bind_group_layout,
            _pipeline_layout: pipeline_layout,
            pipeline,
        })
    }
}
