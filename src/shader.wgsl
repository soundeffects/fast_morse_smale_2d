@group(0)
@binding(0)
var<storage, read_write> gradient: array<u8>; // both input and output buffer, for convenience

fn solve_gradient(n_base: u32) -> u32 {
  var n = n_base;
  var i = 0u;
  loop {
    if (n <= 1u) {
      break;
    }
    
    if (n % 2u == 0u) {
      n = n / 2u;
    } else {
      n = 3u * n + 1u;
    }
    
    i = i + 1u;
  }
  return i;
}

@compute
@workgroup_size(1)
fn main(@builtin(local_invocation_id) global_id: vec3<u32>) {
  v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
}
