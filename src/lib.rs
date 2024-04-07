use bytemuck::cast_slice;
use flume::bounded;
use std::borrow::Cow;
use std::mem::size_of_val;
use wgpu::util::{BufferInitDescriptor, DeviceExt};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferAddress, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Features, Instance, Limits, Maintain, MapMode, Queue,
    RequestAdapterOptions, ShaderModuleDescriptor, ShaderSource,
};

pub struct MorseSmaleSolver {
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
}

impl MorseSmaleSolver {
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
            label: Some("Compute Shader Module"),
            source: ShaderSource::Wgsl(Cow::Borrowed(include_str!("shader.wgsl"))),
        });

        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute Shader Pipeline"),
            layout: None,
            module: &module,
            entry_point: "main",
        });

        Ok(Self {
            device,
            queue,
            pipeline,
        })
    }

    pub async fn run(&self, input: Vec<u32>) -> Result<Vec<u32>, String> {
        let slice = input.as_slice();

        // Instantiate the buffer on the CPU which will receive the final results.
        let cpu_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Uninitialized CPU Buffer"),
            size: size_of_val(slice) as BufferAddress,
            // Buffer can be read outside of shaders, buffer can be a copy destination
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Instantiate the buffer on the GPU which will be intialized with our input.
        let gpu_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Unitialized GPU Buffer"),
            contents: cast_slice(slice),
            // Buffer is used in shaders, and can be the destination or source of a copy
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        });

        // Create a bind group which describes how the shader can access buffers assigned to it.
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Shader Bind Group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: gpu_buffer.as_entire_binding(),
            }],
        });

        // Create an encoder which formats instructions the GPU can read.
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Command Encoder"),
            });

        // Write the command to perform the compute shader pass.
        {
            let mut shader_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Compute Shader Pass"),
                timestamp_writes: None,
            });

            // Provide all the descriptions of execution that we have created to the shader pass.
            shader_pass.set_pipeline(&self.pipeline);
            shader_pass.set_bind_group(0, &bind_group, &[]);
            shader_pass.insert_debug_marker("compute collatz iterations");

            // Describe the dispatch of the pass.
            // Takes 3D bounds, and spawns a compute process for each cell in the bounds.
            shader_pass.dispatch_workgroups(input.len() as u32, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = cpu_buffer.slice(..);
        // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.

        let (sender, receiver) = bounded(1);
        buffer_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = receiver.recv_async().await {
            // Gets contents of buffer
            let data = buffer_slice.get_mapped_range();
            // Since contents are in bytes, this converts these bytes back to u32
            let result: Vec<u32> = cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are dropped before we unmap the buffer.
            // Unmaps buffer from memory
            // If you are familiar with C++ these 2 lines can be thought of similarly to:
            //   delete myPointer;
            //   myPointer = NULL;
            // It effectively frees the memory
            drop(data);
            cpu_buffer.unmap();

            Ok(result)
        } else {
            Err("Could not successfully read the result".to_string())
        }
    }
}
