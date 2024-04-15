// ░█░░░▀█▀░█▀▄░░░░█▀▄░█▀▀
// ░█░░░░█░░█▀▄░░░░█▀▄░▀▀█
// ░▀▀▀░▀▀▀░▀▀░░▀░░▀░▀░▀▀▀
//
// This is the main file of the library. The library is structured such that the main interface of this
// library, the MorseSmaleSolver struct, is defined in this file, and other files each implement a method
// of this class.

use wgpu::{BindGroupLayout, ComputePipeline, Device, PipelineLayout, Queue};

///
/// MorseSmaleSolver
///
/// The MorseSmaleSolver struct is the interface for this library. It contains context and provides
/// methods which will construct the Morse-Smale complexes when given a regularly sampled range of a
/// Morse function.
///
/// The member fields of this struct represent the context that this struct stores. These member fields
/// are private--the user is expected to interface with the methods the struct provides instead. Use the
/// MorseSmaleSolver::new method to create a new solver instance with its own context, and use the
/// MorseSmaleSolver::with_samples method on that instance to start the Morse-Smale complex construction.
///
pub struct MorseSmaleSolver {
    pub(crate) device: Device,
    pub(crate) queue: Queue,
    pub(crate) bind_group_layout: BindGroupLayout,
    pub(crate) _pipeline_layout: PipelineLayout,
    pub(crate) pipeline: ComputePipeline,
}

// The following modules each implement one method of the MorseSmaleSolver struct.
pub mod new;
pub mod with_samples;
