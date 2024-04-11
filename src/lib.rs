use bytemuck::cast_slice;
use flume::bounded;
use image::GrayImage;
use std::borrow::Cow;
use std::mem::size_of;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BufferAddress, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor,
    Device, DeviceDescriptor, Extent3d, Features, ImageCopyTexture, ImageDataLayout, Instance,
    Limits, Maintain, MapMode, Origin3d, Queue, RequestAdapterOptions, ShaderModuleDescriptor,
    ShaderSource, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
};

#[cfg(test)]
mod tests;

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

    pub async fn with_image(&self, image: GrayImage) -> Result<Vec<u8>, String> {
        let dimensions = image.dimensions();

        let extent = Extent3d {
            width: dimensions.0,
            height: dimensions.1,
            // All textures are stored as 3D, we represent our 2D texture
            // by setting depth to 1.
            depth_or_array_layers: 1,
        };

        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("Morse Function As Texture"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::R8Unorm,
            // TEXTURE_BINDING tells wgpu that we want to use this texture in shaders
            // COPY_DST means that we want to copy data to this texture
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        self.queue.write_texture(
            ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &image,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(dimensions.0),
                rows_per_image: Some(dimensions.1),
            },
            extent,
        );

        let gradient_elements = dimensions.0 * dimensions.1 * 4;

        let gradient_size = (size_of::<u8>() * gradient_elements as usize) as BufferAddress;

        // Instantiate the buffer written to for stage 1 and read in stage 2, for gradients.
        let gradient = self.device.create_buffer(&BufferDescriptor {
            label: Some("Discrete Gradient Field"),
            size: gradient_size,
            // Buffer is used in shaders, and can be the destination or source of a copy
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Instantiate buffer on CPU to get data
        let receptacle = self.device.create_buffer(&BufferDescriptor {
            label: Some("Receptacle for CPU"),
            size: gradient_size,
            // Buffer can be read outside of shaders, buffer can be a copy destination
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create a bind group which describes how the shader can access buffers assigned to it.
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Compute Shader Bind Group"),
            layout: &bind_group_layout,
            entries: &[BindGroupEntry {
                binding: 0,
                resource: gradient.as_entire_binding(),
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
            shader_pass.insert_debug_marker("construct morse-smale complexes");

            // Describe the dispatch of the pass.
            // Takes 3D bounds, and spawns a compute process for each cell in the bounds.
            shader_pass.dispatch_workgroups(dimensions.0 - 1, dimensions.1 - 1, 1);
        }

        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(&gradient, 0, &receptacle, 0, gradient_size);

        self.queue.submit(Some(encoder.finish()));

        // Note that we're not calling `.await` here.
        let buffer_slice = receptacle.slice(..);
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
            let result: Vec<u8> = cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are dropped before we unmap the buffer.
            // Unmaps buffer from memory
            // If you are familiar with C++ these 2 lines can be thought of similarly to:
            //   delete myPointer;
            //   myPointer = NULL;
            // It effectively frees the memory
            drop(data);
            receptacle.unmap();

            Ok(result)
        } else {
            Err("Could not successfully read the result".to_string())
        }
    }
}
