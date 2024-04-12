// ░█░█░▀█▀░▀█▀░█░█░░░░░█▀▀░█▀█░█▄█░█▀█░█░░░█▀▀░█▀▀░░░░█▀▄░█▀▀
// ░█▄█░░█░░░█░░█▀█░░░░░▀▀█░█▀█░█░█░█▀▀░█░░░█▀▀░▀▀█░░░░█▀▄░▀▀█
// ░▀░▀░▀▀▀░░▀░░▀░▀░▀▀▀░▀▀▀░▀░▀░▀░▀░▀░░░▀▀▀░▀▀▀░▀▀▀░▀░░▀░▀░▀▀▀
//
// This file defines the `with_samples` method for the MorseSmaleSolver struct. This method sends a new
// compute shader pass to the GPU in order to construct the Morse-Smale complexes corresponding to the
// data provided to this method.

use crate::MorseSmaleSolver;
use bytemuck::cast_slice;
use flume::bounded;
use std::mem::size_of;
use wgpu::{
    util::{BufferInitDescriptor, DeviceExt},
    BindGroupDescriptor, BindGroupEntry, BufferAddress, BufferDescriptor, BufferUsages,
    CommandEncoderDescriptor, ComputePassDescriptor, Maintain, MapMode,
};

impl MorseSmaleSolver {
    ///
    /// MorseSmaleSolver::with_samples
    ///
    /// Sends a new task to the GPU with the given set of samples of a Morse-Smale function as input. The
    /// Samples must represent a rectangular, regularly sampled region. The domain_with provided should
    /// evenly divide the length of the sample list--it should represent the length of the x-dimension of
    /// the rectangular region that the samples represent.
    ///
    /// Note that, due to the naive way this method currently passes data to the GPU, you cannot exceed
    /// a sample list length of approximately six million, otherwise this method will panic.
    ///
    pub async fn with_samples(
        &self,
        samples: Vec<f32>,
        domain_width: u32,
    ) -> Result<Vec<u32>, String> {
        let invocation_parameters_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Morse-Smale Complex Compute: Invocation Parameters Buffer"),
            contents: cast_slice(&[domain_width]),
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let morse_function_samples_buffer = self.device.create_buffer_init(&BufferInitDescriptor {
            label: Some("Morse-Smale Complex Compute: Morse Function Samples Buffer"),
            contents: cast_slice(samples.as_slice()),
            usage: BufferUsages::STORAGE,
        });

        // Instantiate the buffer written to for stage 1 and read in stage 2, for gradients.
        let discrete_gradient_field_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Morse-Smale Complex Compute: Discrete Gradient Field Buffer"),
            size: (size_of::<f32>() * samples.len() * 4) as BufferAddress,
            // Buffer is used in shaders, and can be the destination or source of a copy
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // For portability reasons, WebGPU draws a distinction between memory that is
        // accessible by the CPU and memory that is accessible by the GPU. Only
        // buffers accessible by the CPU can be mapped and accessed by the CPU and
        // only buffers visible to the GPU can be used in shaders. In order to get
        // data from the GPU, we need to use CommandEncoder::copy_buffer_to_buffer
        // (which we will later) to copy the buffer modified by the GPU into a
        // mappable, CPU-accessible buffer which we'll create here.
        let receptacle_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Morse-Smale Complex Compute: Receptacle Buffer"),
            size: (size_of::<f32>() * samples.len() * 4) as BufferAddress,
            // Buffer can be read outside of shaders, buffer can be a copy destination
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create a bind group which describes how the shader can access buffers assigned to it.
        // This ties actual resources stored in the GPU to our metaphorical function described
        // by the bind group layout.
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("Morse-Smale Complex Compute: Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: invocation_parameters_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: morse_function_samples_buffer.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: discrete_gradient_field_buffer.as_entire_binding(),
                },
            ],
        });

        // Create an encoder which formats instructions the GPU can read.
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Morse-Smale Complex Compute: Command Encoder"),
            });

        // Write the command to perform the compute shader pass.
        {
            let mut shader_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("Morse-Smale Complex Compute: Shader Pass"),
                timestamp_writes: None,
            });

            // Provide all the descriptions of execution that we have created to the shader pass.
            shader_pass.set_pipeline(&self.pipeline);
            shader_pass.set_bind_group(0, &bind_group, &[]);
            shader_pass.insert_debug_marker("Morse-Smale Complex Compute");

            // Describe the dispatch of the pass.
            // Takes 3D bounds, and spawns a compute process for each cell in the bounds.
            shader_pass.dispatch_workgroups(
                domain_width - 1,
                (samples.len() / domain_width as usize) as u32 - 1,
                1,
            );
        }
        // The pass is finished by dropping it.

        // Sets adds copy operation to command encoder.
        // Will copy data from storage buffer on GPU to staging buffer on CPU.
        encoder.copy_buffer_to_buffer(
            &discrete_gradient_field_buffer,
            0,
            &receptacle_buffer,
            0,
            (size_of::<f32>() * samples.len() * 4) as BufferAddress,
        );

        // Finalize the command encoder, add the contained commands to the queue and flush.
        self.queue.submit(Some(encoder.finish()));

        // Finally time to get our results.
        // First we get a buffer slice which represents a chunk of the buffer (which we
        // can't access yet).
        // We want the whole thing so use unbounded range.
        // Note that we're not calling `.await` here.
        let receptacle_slice = receptacle_buffer.slice(..);

        // Now things get complicated. WebGPU, for safety reasons, only allows either the GPU
        // or CPU to access a buffer's contents at a time. We need to "map" the buffer which means
        // flipping ownership of the buffer over to the CPU and making access legal. We do this
        // with `BufferSlice::map_async`.
        //
        // The problem is that map_async is not an async function so we can't await it. What
        // we need to do instead is pass in a closure that will be executed when the slice is
        // either mapped or the mapping has failed.
        //
        // The problem with this is that we don't have a reliable way to wait in the main
        // code for the buffer to be mapped and even worse, calling get_mapped_range or
        // get_mapped_range_mut prematurely will cause a panic, not return an error.
        //
        // Using channels solves this as awaiting the receiving of a message from
        // the passed closure will force the outside code to wait. It also doesn't hurt
        // if the closure finishes before the outside code catches up as the message is
        // buffered and receiving will just pick that up.
        //
        // It may also be worth noting that although on native, the usage of asynchronous
        // channels is wholly unnecessary, for the sake of portability to WASM (std channels
        // don't work on WASM,) we'll use async channels that work on both native and WASM.
        let (sender, receiver) = bounded(1);
        receptacle_slice.map_async(MapMode::Read, move |v| sender.send(v).unwrap());

        // In order for the mapping to be completed, one of three things must happen.
        // One of those can be calling `Device::poll`. This isn't necessary on the web as devices
        // are polled automatically but natively, we need to make sure this happens manually.
        // `Maintain::Wait` will cause the thread to wait on native but not on WebGpu.
        //
        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        self.device.poll(Maintain::wait()).panic_on_timeout();

        // Awaits until `buffer_future` can be read from
        if let Ok(Ok(())) = receiver.recv_async().await {
            // Gets contents of buffer
            let data = receptacle_slice.get_mapped_range();
            // Since contents are in bytes, this converts these bytes back to u32
            let result: Vec<u32> = cast_slice(&data).to_vec();

            // With the current interface, we have to make sure all mapped views are dropped before we unmap the buffer.
            // Unmaps buffer from memory
            // If you are familiar with C++ these 2 lines can be thought of similarly to:
            //   delete myPointer;
            //   myPointer = NULL;
            // It effectively frees the memory
            drop(data);
            receptacle_buffer.unmap();

            Ok(result)
        } else {
            Err("Could not successfully read the result".to_string())
        }
    }
}
