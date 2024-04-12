// ░█▀▀░█░█░█▀█░█▀▄░█▀▀░█▀▄░░░░█░█░█▀▀░█▀▀░█░░
// ░▀▀█░█▀█░█▀█░█░█░█▀▀░█▀▄░░░░█▄█░█░█░▀▀█░█░░
// ░▀▀▀░▀░▀░▀░▀░▀▀░░▀▀▀░▀░▀░▀░░▀░▀░▀▀▀░▀▀▀░▀▀▀
//
// This file writes shader code, which is what the GPU will run. The shader pass uses the data provided
// to it and computes the Morse-Smale complex from the data provided by running a bunch of invocations o
// this code in parallel.

struct InvocationParameters {
 domain_width: u32,
     };

@group(0) @binding(0) var<uniform> invocation_parameters: InvocationParameters;
@group(0) @binding(1) var<storage, read> morse_function: array<f32>;
@group(0) @binding(2) var<storage, read_write> gradient: array<u32>;

fn linearized(x: u32, y: u32) -> u32 {
  return invocation_parameters.domain_width * y + x;
}

@compute
@workgroup_size(1)
fn compute_gradient(@builtin(global_invocation_id) id: vec3<u32>) {
  var top_left_corner: f32 = morse_function[linearized(id.x, id.y)];
  var top_right_corner: f32 = morse_function[linearized(id.x + 1, id.y)];
  var bottom_left_corner: f32 = morse_function[linearized(id.x, id.y + 1)];
  var bottom_right_corner: f32 = morse_function[linearized(id.x + 1, id.y + 1)];

  var gradient_position: u32 = linearized(id.x, id.y) * 4;
  gradient[gradient_position] = u32(round(top_left_corner));
  gradient[gradient_position + 1] = u32(round(top_right_corner));
  gradient[gradient_position + 2] = u32(round(bottom_left_corner));
  gradient[gradient_position + 3] = u32(round(bottom_right_corner));
}
